# Whisper transcription endpoint (`/v1/audio/transcriptions`) — design

Issue: #366

## Goal

Add an OpenAI-compatible speech-to-text route so `client.audio.transcriptions.create(file=..., model=...)`
works against olmlx. `mlx-whisper` does the heavy lifting; whisper models are loaded and managed
through the existing `ModelManager` alongside LLMs.

## Acceptance criteria (from the issue)

- OpenAI SDK `client.audio.transcriptions.create(file=..., model=...)` works against olmlx.
- Format coverage: `json`, `verbose_json`, `srt` (also `text`, `vtt`).
- File-size cap configurable via `OLMLX_AUDIO_MAX_BYTES`.

## Out of scope

- Streaming transcription.
- Diarization.
- Timestamp granularity beyond what `mlx-whisper` provides natively (we expose `word`/`segment`
  granularity, both native to mlx-whisper, but nothing finer).

## Dependency & runtime

- Add `mlx-whisper` to `pyproject.toml` (installed: `0.4.3`).
- **ffmpeg is a runtime requirement** — `mlx_whisper.audio.load_audio` shells out to ffmpeg to
  decode arbitrary audio containers. We write the upload to a temp file and pass the path to
  `transcribe()`. If ffmpeg is missing, surface a clear `ValueError` (→ HTTP 400) instructing the
  operator to install ffmpeg, rather than leaking a raw `FileNotFoundError`/subprocess error.

## Architecture

### `engine/model_manager.py` — whisper model kind

- `_detect_model_kind` gains a `"whisper"` result. Discriminator, checked **early** (right after
  `config` is loaded and `model_type` computed, before the VLM-keys block):
  - `model_type == "whisper"`, **or**
  - whisper dimension keys present: `"n_mels" in config and "n_audio_state" in config`.
  - The dims check is the robust one: mlx-community whisper repos ship a non-HF `config.json`
    containing `mlx_whisper.whisper.ModelDimensions` fields (`n_mels`, `n_audio_ctx`,
    `n_audio_state`, …) and `load_model` pops `model_type`, so `model_type` may be absent.
- `_load_model` gains a whisper branch (placed before the VLM/text dispatch):
  `model = mlx_whisper.load_models.load_model(load_path, dtype=mx.float16)`, returns
  `(model, None, False, TemplateCaps(), None)` — no tokenizer, no caps, no speculative decoder.
  `float16` matches `transcribe()`'s default `fp16=True`.
- `LoadedModel` gains `is_whisper: bool = False`.
- `_probe_cache_capabilities` early-returns when `lm.is_whisper` (one-shot STT has no prompt cache).
- The kv-quant / spectral-calibration resolution in `ensure_loaded` is skipped for whisper (guard on
  the detected kind / a whisper flag) — `make_prompt_cache` and quant factories assume an LLM layer
  stack and don't apply to a `Whisper` module.
- Keep-alive, LRU eviction, memory check, and `active_refs` all work unchanged — whisper models are
  registered in `self._loaded` like any other model and appear in `/api/ps`.

### `engine/inference.py` — `generate_transcription`

```
async def generate_transcription(
    manager, model_name, audio_path, *,
    language=None, prompt=None, temperature=0.0,
    response_format="json", word_timestamps=False, keep_alive=None,
) -> dict
```

- `lm = await manager.ensure_loaded(model_name, keep_alive)`.
- `async with _inference_locked(lm.inference_queue_timeout, sync_mode=lm.sync_mode):` then
  `with _inference_ref(lm):` — same guard pattern as `generate_embeddings`.
- Inside the lock, inject the managed model into `mlx_whisper.transcribe.ModelHolder` by setting
  `ModelHolder.model = lm.model` and `ModelHolder.model_path = <load_path>` so `transcribe()` reuses
  it instead of loading its own copy. The injection is race-free because `_inference_locked`
  serializes inference.
- Run `mlx_whisper.transcribe(audio_path, path_or_hf_repo=<load_path>, language=..., initial_prompt=prompt,
  temperature=..., word_timestamps=...)` via `asyncio.to_thread` — it is synchronous and can be slow
  on long audio, so it must not block the event loop.
- Return the raw result dict (`text`, `segments`, `language`).

Note: the load path used as `path_or_hf_repo` must match what the model was loaded from so the
`ModelHolder.model_path` guard (`model_path != cls.model_path`) sees a hit. We resolve it from the
store the same way `_load_model` does (or store it on `LoadedModel`); simplest is to reuse
`manager`'s store to resolve `lm.hf_path` to the local dir.

### `routers/audio.py` (new)

- `POST /v1/audio/transcriptions`, multipart via FastAPI `UploadFile` + `Form(...)`.
- Form fields:
  - `file: UploadFile` (required)
  - `model: str = Form(...)` (required)
  - `language: str | None = Form(None)`
  - `prompt: str | None = Form(None)` → `initial_prompt`
  - `response_format: str = Form("json")`
  - `temperature: float = Form(0.0)`
  - `timestamp_granularities[]` → accept the bracketed OpenAI field name; enable `word_timestamps`
    when `word` is present (segment granularity is the default and always available).
- Stream the upload to a `tempfile` enforcing `settings.audio_max_bytes`; on overflow delete the temp
  file and return HTTP 413. Clean up the temp file in a `finally`.
- Validate `response_format` ∈ {`json`, `verbose_json`, `text`, `srt`, `vtt`}; otherwise `ValueError`.
- Call `generate_transcription(...)`, then serialize per format:
  - `json` → `TranscriptionResponse {text}` (JSON)
  - `verbose_json` → `VerboseTranscriptionResponse {task:"transcribe", language, duration, text, segments[]}`
  - `text` → `PlainTextResponse(result["text"])`
  - `srt` → `PlainTextResponse(srt_from_segments(result["segments"]))`
  - `vtt` → `PlainTextResponse(vtt_from_segments(result["segments"]))`
- Errors raise `ValueError` (→ 400 via the app-level handler), consistent with other routers.

### `schemas/audio.py` (new)

- `TranscriptionResponse { text: str }`
- `TranscriptionSegment { id, seek, start, end, text, tokens, temperature, avg_logprob,
  compression_ratio, no_speech_prob }` (mirrors mlx-whisper segment dict; tolerant of extras).
- `VerboseTranscriptionResponse { task: "transcribe", language: str, duration: float, text: str,
  segments: list[TranscriptionSegment] }`
- Request shape is handled by `Form(...)` params in the router (a multipart upload can't cleanly be a
  single Pydantic body alongside the file), so no request model here.

### `config.py`

- `audio_max_bytes: int = Field(100 * 1024 * 1024, gt=0)` → `OLMLX_AUDIO_MAX_BYTES`. Default 100 MB.

### `app.py`

- `app.include_router(audio.router)` alongside `openai.router`.

### Registry aliases

- Seed convenience names in `cli.py`'s `DEFAULT_MODELS` dict (the existing mechanism `ensure_config()`
  uses to write the initial `~/.olmlx/models.json`), so a fresh install resolves `model="whisper-turbo"`
  without manual registration:
  - `whisper-turbo:latest` → `mlx-community/whisper-large-v3-turbo`
  - `whisper-large:latest` → `mlx-community/whisper-large-v3-mlx`
- These are plain name→HF-path mappings, identical in form to the existing `DEFAULT_MODELS` LLM
  entries; they only land on first run when no `models.json` exists yet (matching current behaviour
  for the seeded LLMs).
- Direct HF paths (e.g. `mlx-community/whisper-large-v3-turbo`) continue to auto-register on first use
  regardless, so existing installs with a populated `models.json` still work by passing the HF path.

## SRT/VTT formatting

Small pure helpers converting `segments` (each has `start`, `end` in seconds and `text`) to SubRip /
WebVTT. Timestamp format: SRT `HH:MM:SS,mmm`, VTT `HH:MM:SS.mmm`, VTT prefixed with a `WEBVTT` header.
These are pure functions, unit-tested against a fixed segment list.

## Error handling

- Unknown `response_format` → `ValueError` → 400.
- Upload exceeds `audio_max_bytes` → 413.
- ffmpeg missing / decode failure → `ValueError` with a clear "install ffmpeg" message → 400.
- Unknown/unloadable model → existing `ModelManager` errors (400/503/504) bubble through unchanged.

## Testing (TDD)

1. `_detect_model_kind` returns `"whisper"` for (a) `model_type:"whisper"` config and (b) a
   dims-only mlx config; returns non-whisper for an LLM config.
2. `srt_from_segments` / `vtt_from_segments` produce correct timestamps and framing from a fixed
   segment list.
3. Router rejects an over-cap upload with 413.
4. Router happy path for `json`, `verbose_json`, `text`, `srt` with `generate_transcription` mocked
   to return a fixed result dict — assert body shape / content-type per format.
5. Unknown `response_format` → 400.
6. ffmpeg-missing path surfaces a clear 400 (simulate by making the decode raise).

Model-loading and real transcription are not unit-tested against real weights (no audio fixture in
CI); the engine path is covered by the kind-detection test plus a mocked router test.

## Docs

- New CLAUDE.md "Audio transcription" bullet under Key Design Decisions describing the route, the
  ModelHolder-injection integration, the ffmpeg requirement, and `OLMLX_AUDIO_MAX_BYTES`.
