# TTS Endpoint (`/v1/audio/speech`) — Design

**Issue:** #367
**Date:** 2026-06-08
**Status:** Approved (brainstorming)

## Summary

Add an OpenAI-compatible text-to-speech endpoint, `POST /v1/audio/speech`, to
olmlx. It mirrors the existing Whisper transcription path (#366): TTS becomes a
first-class `ModelManager` kind, an engine function (`generate_speech`) drives
inference under the serialized lock, and a route is added to the shared
`routers/audio.py`. The backend is the `mlx-audio` package; the v1 model is
Kokoro-82M. Both streaming and non-streaming responses are supported, across all
six OpenAI audio formats.

This gives olmlx full OpenAI audio parity (transcription + speech) so the OpenAI
SDK's `client.audio.speech.create()` works without changes.

## Scope

- `POST /v1/audio/speech` in `routers/audio.py` (shared with the Whisper route).
- `mlx-audio` dependency (pinned).
- Kokoro-82M (`prince-canuma/Kokoro-82M`) as the v1 model — 24 kHz mono,
  multi-voice. The path is built generically so other mlx-audio TTS models
  (Bark, Dia, CSM) can be added later, but only Kokoro is targeted for v1.
- Voice selection: OpenAI voice names mapped to Kokoro voices, with native
  Kokoro voice ids passing through.
- All six OpenAI output formats: `mp3` (default), `opus`, `aac`, `flac`, `wav`,
  `pcm`. `wav`/`pcm` natively; the compressed formats via the already-required
  ffmpeg.
- Streaming (chunked audio, per Kokoro segment) **and** buffered responses.
- TTS model class loadable via `olmlx models pull`.

## Out of scope

- **Parakeet.** The issue lists "Parakeet TTS," but Parakeet is NVIDIA's *STT*
  (ASR) model — it is not a TTS model and is not in mlx-audio's TTS suite.
  Dropped from acceptance; Kokoro is the v1 model.
- The newer OpenAI `stream_format="sse"` semantic-event stream (only
  `stream_format="audio"` chunked binary is supported).
- Voice cloning / custom voice training.
- Distributed TTS, KV-quant / prompt-cache for TTS (not applicable).

## Architecture

### 1. Dependency

Add `mlx-audio` (pinned) to `pyproject.toml`, alongside `mlx-whisper` /
`mlx-vlm`. Same MLX community lineage.

### 2. Model manager integration (mirrors Whisper, #366)

- `_detect_model_kind` returns `"tts"` for Kokoro-family configs. Discriminator
  determined during TDD (config `model_type` / Kokoro-specific markers); robust
  to the absence of a standard `model_type` key, the same way the Whisper
  detector keys off dims.
- New load branch in `_load_model`: `kind == "tts"` loads via
  `mlx_audio.tts.utils.load_model(load_path)` and returns the model with no
  tokenizer/caps/speculative (the speech path drives mlx-audio directly).
- `LoadedModel` gains `is_tts: bool = False`. `is_tts` guards the LLM-only paths
  (prompt cache, KV-quant, speculative, keep-alive bookkeeping) exactly where
  `is_whisper` does today.
- `olmlx models pull prince-canuma/Kokoro-82M` works via the existing
  auto-register-on-pull behavior; a friendly alias is added to `models.json`
  defaults.

### 3. Engine path — `engine/inference.py::generate_speech`

- Async; acquires the inference lock; `ensure_loaded`; rejects a non-TTS model
  with a clear HTTP 400 (the #366 pattern — without it a text/VLM model name
  would load and fail cryptically inside mlx-audio).
- Empty-input guard up front.
- Runs `model.generate(text, voice=…, speed=…)` in a worker thread, yielding
  float audio segments (24 kHz) as they are produced. Per-segment yield is what
  enables streaming first-chunk latency.
- Exposed as an async generator of float audio chunks; the router decides
  whether to stream or buffer.

### 4. Audio encoding — `utils/audio.py` (new)

Converts the float audio segment stream into the requested container:

- `pcm`: float → int16 little-endian, raw (24 kHz, mono, signed 16-bit).
- `wav`: streaming WAV (RIFF header followed by PCM).
- `mp3` / `opus` / `aac` / `flac`: pipe PCM into an ffmpeg subprocess
  (`ffmpeg -f s16le -ar 24000 -ac 1 -i pipe:0 -f <fmt> pipe:1`) and stream
  stdout. ffmpeg is already required on PATH for Whisper.
- The ffmpeg process is terminated on client disconnect / generation end —
  guaranteed cleanup (the bench orphaned-process lesson). PCM is fed to stdin as
  segments arrive; stdout chunks are read concurrently.

### 5. Router — `routers/audio.py` (shared)

- `POST /v1/audio/speech`, JSON body via a new `SpeechRequest` schema:
  `model` (str), `input` (str), `voice` (str), `response_format`
  (default `"mp3"`), `speed` (float, default 1.0, OpenAI range 0.25–4.0).
- Returns a `StreamingResponse` with the format's media type. This works for
  both SDK call styles: `.create()` buffers the stream, `.with_streaming_response`
  iterates chunks.
- **Voice handling:** an `OPENAI_VOICE_MAP` maps the six OpenAI voice names
  (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) to chosen Kokoro
  voices; any other string passes through as a native Kokoro voice id; an
  unknown id yields HTTP 422.
- Input length bounded by a new `OLMLX_TTS_MAX_INPUT_CHARS` setting (config.py).

### 6. Data flow

```
client → POST /v1/audio/speech (JSON)
  → router validates SpeechRequest, maps voice, picks encoder
  → generate_speech(manager, model, input, voice, speed)   [inference lock]
      → ensure_loaded (kind=tts) → model.generate(...) in worker thread
      → yields float audio segments (24 kHz)
  → encoder (utils/audio.py): pcm/wav native, else ffmpeg pipe
  → StreamingResponse streams encoded bytes to client
```

### 7. Error handling

| Condition | Response |
|---|---|
| Model is not a TTS model | 400, actionable message (#366 pattern) |
| Unknown voice id | 422 |
| Unsupported `response_format` | 422 |
| Input exceeds `OLMLX_TTS_MAX_INPUT_CHARS` | 413 |
| Empty input | 422 |
| Client disconnect mid-stream | cancel generation, kill ffmpeg, release lock |

### 8. Testing (TDD)

- **Unit** (mlx-audio mocked, following the autouse MLX-mock pattern):
  - voice mapping (OpenAI names → Kokoro; passthrough; unknown → 422),
  - format dispatch + wav-header correctness,
  - ffmpeg invocation arguments,
  - non-TTS model rejection (400),
  - oversized / empty input.
- **Live** (`tests/live/test_tts_speech.py`, `real_model` marker, **outside**
  `tests/integration/` to dodge its autouse MLX mock):
  - real Kokoro via the OpenAI SDK `client.audio.speech.create()` for each
    format,
  - streaming first-chunk latency (< 1s target on Apple Silicon for short
    utterances),
  - audible / non-empty output sanity checks.

## Acceptance criteria

- OpenAI SDK `client.audio.speech.create()` works without changes (default
  `voice="alloy"`, `response_format="mp3"`).
- Kokoro loads and produces audible output across all six formats.
- Streaming mode delivers the first chunk quickly (< 1s for short utterances).
- Non-TTS model names are rejected with a clear 400.

## Caveats resolved during implementation (do not change the design shape)

- Exact mlx-audio `load_model` / `generate` API surface.
- Kokoro's sample rate (assumed 24 kHz) and the canonical voice-id list.
- The precise `_detect_model_kind` discriminator for Kokoro configs.

These are verified against the installed `mlx-audio` package during TDD rather
than assumed from documentation.
