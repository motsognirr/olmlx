# Native audio input for multimodal LLMs (#426)

**Status:** Design approved — pending implementation plan
**Date:** 2026-06-08
**Issue:** [#426](https://github.com/motsognirr/olmlx/issues/426)
**Related:** #427 (video/multi-frame, sibling), #367 (TTS / audio *output*), #428/#429 (VLM image plumbing this mirrors)

## Goal

Let an encoder-free multimodal LLM (Gemma 4) take an audio clip *into* a chat request as a
first-class modality — the way it already takes images — and reason over it. This is distinct
from the existing Whisper STT endpoint (`/v1/audio/transcriptions`), which is a separate model
producing a transcript, and from #367 (TTS, audio out).

## Background: how images flow today (the path audio mirrors)

```
routers/*  →  msg["images"]: list[str]        (normalize_image_block, utils/images.py)
           →  _extract_images(messages)        (engine/inference.py)
           →  _apply_chat_template_vlm(images=) (+ _inject_image_markers for the tools branch)
           →  mlx_vlm.stream_generate(..., image=images)
```

Model kind is detected by `_detect_model_kind` against `_VLM_CONFIG_KEYS` in
`engine/model_manager.py`; `LoadedModel.is_vlm` selects the VLM path.

## Critical data-flow finding (the one place audio ≠ images)

`mlx_vlm.generate`/`stream_generate` (mlx-vlm 0.4.4) **already accept `audio: str | list[str]`** and
`apply_chat_template` accepts `num_audios`. Gemma 4 ships full audio support
(`models/gemma4/audio.py`, `audio_feature_extractor.py`; config `audio_config`,
`audio_token_id=258881`, `audio_soft_tokens_per_image=750`, `audio_ms_per_token=40`).

Internally `generate(audio=[...])` calls `load_audio(item, sr=...)` per item. `load_audio` accepts:
- a numpy array,
- a local file path,
- an `http(s)://` URL,

and decodes wav/mp3/flac via `miniaudio`, and m4a/aac/ogg/opus via **ffmpeg** (already a hard olmlx
dependency for Whisper/TTS). It does **NOT** accept base64 or `data:` URIs.

OpenAI's `input_audio` block carries **base64** (`{data, format}`). Images get base64 for free
(`load_image` decodes data-URIs); audio does not. **Therefore olmlx must materialize base64 → a
temp file before handing the path to mlx_vlm.** This is the only structural divergence from the
image path.

## Decision: base64 handling — temp-file materialization (Approach A)

Decode base64 → a `NamedTemporaryFile` with the suffix from `format` (`.wav`/`.mp3`/…), pass the
path to mlx_vlm, delete after generation. URLs and local paths pass straight through untouched.

Rejected alternatives:
- **B — in-process decode to ndarray**: re-implements mlx_vlm's format dispatch + the ffmpeg
  shell-out for m4a/opus (`read_audio`); more code, more format-bug surface.
- **C — patch mlx_vlm to accept data-URIs**: a fork/upstream PR + version pin; out of step with
  treating mlx_vlm as a black box (as the image path does).

A is least code and leans on mlx_vlm's existing decode + the ffmpeg we already require; the
temp-file lifecycle is a small, contained concern.

## Design

### 1. Internal representation

After router normalization, audio rides on the message like images:

```python
msg["audio"]: list[str]   # each: a load_audio-acceptable string — local path, http(s) URL,
                          # or a temp-file path materialized from base64
```

No change to the internal message-dict format.

### 2. Source normalization — `normalize_audio_block(block) -> str`

New helper paralleling `utils/images.py:normalize_image_block`. `utils/audio.py` already exists
(TTS *output* helpers); add the input-normalizer there (or a sibling module) without touching the
existing TTS functions. Accepts the three source forms:

- **OpenAI** `{"type": "input_audio", "input_audio": {"data": "<b64>", "format": "wav"}}`
  → decode base64 → temp file `.<format>` → return path.
- **data-URI** `data:audio/...;base64,...` → decode → temp file.
- **URL / local path** (`http(s)://…` or a filesystem path) → return the string unchanged
  (mlx_vlm's `load_audio` loads it).
- Malformed/missing data, or an unsupported `format` → `ValueError` (router maps to **422**).

Materialized temp-file paths are tracked so the engine deletes them after generation (see §5).

### 3. Router wiring (all three surfaces)

- **`routers/openai.py`** — extend `_normalize_multimodal_messages` to split `input_audio` parts
  into `msg["audio"]` (it already splits `image_url` into `msg["images"]`).
- **`routers/chat.py`** (Ollama `/api/chat`) — accept an `audio` field on the message paralleling
  the native `images` field; pass through.
- **`routers/anthropic.py`** — extend `_convert_messages` to recognize an audio block. No Anthropic
  standard exists, so define one mirroring their image block (`type: "image"` → `type: "audio"`),
  documented as an **olmlx extension**:
  `{"type": "audio", "source": {"type": "base64"|"url", "media_type": "audio/wav", "data"|"url": …}}`.

### 4. Schemas

- **`schemas/openai.py`** — content parts are already `list[dict[str, Any]]`; `input_audio` is
  accepted without a model change. Document the shape.
- **`schemas/anthropic.py`** — `AnthropicContentBlock` already carries `type` + freeform fields; the
  audio block validates as-is. Document.
- No restrictive Pydantic models for the block bodies (consistent with image parts today).

### 5. Engine threading — `engine/inference.py`

- Add `_extract_audio(messages) -> list[str]` paralleling `_extract_images`, called in
  `generate_chat` next to `images = _extract_images(...)`.
- `_apply_chat_template_vlm` gains an `audio` param: pass `num_audios=len(audio)` to
  `mlx_vlm.apply_chat_template` (non-tools path). The Gemma 4 processor expands `<audio>`
  placeholders itself (`full_audio_sequence`, 750 soft-tokens/clip) — **no hand-injected markers**
  (unlike the tools+images branch).
- **Tools + audio is out of scope for v1**: the raw-tokenizer tools branch would need an
  audio-marker injector mirroring `_inject_image_markers`. When `tools` are present alongside audio,
  reject (422) or ignore-and-log — pick reject for a clear contract.
- Thread `audio` through `_stream_completion` / `_full_completion_inner` → `async_mlx_stream` →
  `mlx_vlm.stream_generate(..., audio=audio)`, alongside the existing `image=images`.
- **Temp-file cleanup**: wrap the VLM generate call so materialized temp paths are deleted in a
  `finally`, covering stream, non-stream, and the cancellation/disconnect path (same discipline as
  the prefill-cancellation cleanup).

### 6. Capability detection & errors

- A model accepts audio iff it is a VLM whose config has `audio_config` / `audio_token_id`
  (Gemma 4: `258881`) **and** the loaded processor exposes a `feature_extractor`. Add a cheap
  `is_audio_capable`-style probe read from the already-loaded processor/config — **not** a new
  `ModelManager` kind (audio models are VLMs, caught by the existing `_VLM_CONFIG_KEYS` path).
- Error matrix (all **422** with a clear message):
  - audio → non-audio VLM: "model X cannot accept audio input."
  - audio → non-VLM (text/whisper/tts/reranker): rejected.
  - malformed block / unsupported format: rejected at normalization.
  - tools + audio: rejected (v1 scope).
  - ffmpeg missing for m4a/aac/ogg/opus: clear error naming ffmpeg (wav/mp3/flac don't need it).

### 7. Composition with existing features

Inherits the VLM path's current behavior:
- grammar + prompt-cache **enabled** (#429);
- speculative decoding **skipped** for audio requests (drafts are text-only — same rule as VLM
  images, #429);
- distributed VLM **out of scope**.

Text-only and image-only requests are byte-for-byte unaffected — audio threading is additive and
gated on `msg["audio"]` being present.

### 8. Testing

- **Unit** (under the integration MLX mock):
  - `normalize_audio_block`: base64→temp, data-URI→temp, URL passthrough, local-path passthrough,
    malformed→ValueError, unsupported format→ValueError.
  - `_extract_audio`.
  - Router splitting for all three surfaces (OpenAI `input_audio`, Ollama `audio` field, Anthropic
    block).
  - 422 paths: non-audio model, non-VLM model, tools+audio rejection.
  - Temp-file cleanup runs on success and on error.
- **Live** (`tests/live/test_vlm_audio.py`, `real_model`, **outside** `tests/integration/` to dodge
  the autouse MLX mock): drive `mlx-community/gemma-4-e2b-it-4bit` (smallest audio-capable Gemma 4,
  already pulled, ~2 GB, fits 64 GB easily) with a short WAV + text prompt across
  `/v1/chat/completions`, `/api/chat`, and `/v1/messages`; assert a coherent grounded answer. Reuse
  the live-SDK pattern from the existing VLM tests (`tests/live/test_vlm_tools_images.py`).

### 9. Dependencies & docs

- Bump `mlx-vlm>=0.4.4` (audio arg + Gemma 4 audio live there).
- `miniaudio` (transitive via mlx-vlm) covers wav/mp3/flac; **ffmpeg** (already required) covers
  m4a/aac/ogg/opus.
- Document the model-pull command (`olmlx models pull mlx-community/gemma-4-e2b-it-4bit`), the
  supported input formats, and the Anthropic-extension audio block. Add a CLAUDE.md design note
  paralleling the existing VLM entries.

## Out of scope (v1)

Tools + audio; streaming audio input; diarization; audio *output* (#367); speculative/distributed
audio; audio in the non-VLM text path; non-Gemma-4 audio VLMs (the path is generic — gemma3n,
qwen3_omni_moe, phi4mm, minicpmo also expose audio in mlx_vlm — but only Gemma 4 is wired/tested in
v1).

## Acceptance criteria (from #426)

- A `/v1/chat/completions` (and `/api/chat`, plus the olmlx-extension `/v1/messages`) request
  carrying an audio clip + text prompt returns a coherent answer grounded in the audio on a
  Gemma 4 checkpoint.
- Text-only and image requests are unaffected.
- Clear (422) error when the loaded model / installed mlx_vlm can't accept audio.
