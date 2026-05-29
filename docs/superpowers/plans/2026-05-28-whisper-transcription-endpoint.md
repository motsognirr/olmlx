# Whisper Transcription Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an OpenAI-compatible `POST /v1/audio/transcriptions` route backed by `mlx-whisper`, with whisper models managed through the existing `ModelManager`.

**Architecture:** A new `routers/audio.py` accepts a multipart upload, streams it to a temp file under a size cap, and calls a new `generate_transcription()` in `engine/inference.py`. That function loads the whisper model via `ModelManager` (new `"whisper"` model kind in `model_manager.py`), injects it into `mlx_whisper`'s module-level `ModelHolder`, and runs `mlx_whisper.transcribe()` in a worker thread under the inference lock. Results are serialized to `json`/`verbose_json`/`text`/`srt`/`vtt`.

**Tech Stack:** Python, FastAPI, mlx-whisper (0.4.3), pydantic-settings, pytest. ffmpeg is a runtime requirement.

---

## File Structure

- **Create** `olmlx/routers/audio.py` — the endpoint, multipart handling, size cap, format serialization, srt/vtt helpers.
- **Create** `olmlx/schemas/audio.py` — `TranscriptionResponse`, `TranscriptionSegment`, `VerboseTranscriptionResponse`.
- **Modify** `olmlx/config.py` — add `audio_max_bytes` setting.
- **Modify** `olmlx/engine/model_manager.py` — `"whisper"` kind detection, whisper load branch, `is_whisper` flag, probe/kv-quant guards.
- **Modify** `olmlx/engine/inference.py` — add `generate_transcription()`.
- **Modify** `olmlx/app.py` — register `audio.router`.
- **Modify** `olmlx/cli.py` — seed whisper names in `DEFAULT_MODELS`.
- **Modify** `CLAUDE.md` — audio-transcription design bullet.
- **Create** `tests/test_routers_audio.py`, `tests/test_audio_formats.py`; **extend** `tests/test_model_manager.py`, `tests/test_config.py`, `tests/test_inference.py` (or a new `tests/test_inference_transcription.py`), `tests/test_cli.py`.

Note: `pyproject.toml`/`uv.lock` already have `mlx-whisper` (added during spec phase). Run `uv sync --no-editable` if the dep is not yet installed in your environment.

---

## Task 1: Config — `audio_max_bytes`

**Files:**
- Modify: `olmlx/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_audio_max_bytes_default():
    from olmlx.config import Settings

    s = Settings()
    assert s.audio_max_bytes == 100 * 1024 * 1024


def test_audio_max_bytes_env_override(monkeypatch):
    monkeypatch.setenv("OLMLX_AUDIO_MAX_BYTES", "1048576")
    from olmlx.config import Settings

    s = Settings()
    assert s.audio_max_bytes == 1048576
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_audio_max_bytes_default -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'audio_max_bytes'`

- [ ] **Step 3: Implement**

In `olmlx/config.py`, inside the `Settings` class, add a field near the other numeric limits (e.g. after `prompt_cache_*` fields). Use the `Field` import already present in the file:

```python
    audio_max_bytes: int = Field(
        100 * 1024 * 1024,
        gt=0,
        description="Max upload size for /v1/audio/transcriptions (OLMLX_AUDIO_MAX_BYTES).",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -k audio_max_bytes -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add olmlx/config.py tests/test_config.py
git commit -m "feat(config): add OLMLX_AUDIO_MAX_BYTES setting (#366)"
```

---

## Task 2: Whisper model-kind detection

**Files:**
- Modify: `olmlx/engine/model_manager.py:1654` (inside `_detect_model_kind`)
- Test: `tests/test_model_manager.py` (class `TestDetectModelKind`)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_model_manager.py` inside `class TestDetectModelKind`:

```python
    def test_whisper_by_model_type(self, tmp_path, registry, mock_store):
        config_path = self._make_config(tmp_path, {"model_type": "whisper"})
        manager = self._make_manager(registry, mock_store)
        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/whisper")
        assert kind == "whisper"

    def test_whisper_by_dims_without_model_type(self, tmp_path, registry, mock_store):
        # mlx-community whisper repos ship a non-HF config.json with dims and
        # no usable model_type (load_model pops it).
        config_path = self._make_config(
            tmp_path,
            {
                "n_mels": 80,
                "n_audio_ctx": 1500,
                "n_audio_state": 384,
                "n_audio_head": 6,
                "n_audio_layer": 4,
                "n_vocab": 51865,
                "n_text_ctx": 448,
                "n_text_state": 384,
                "n_text_head": 6,
                "n_text_layer": 4,
            },
        )
        manager = self._make_manager(registry, mock_store)
        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/whisper-mlx")
        assert kind == "whisper"

    def test_llama_not_whisper(self, tmp_path, registry, mock_store):
        config_path = self._make_config(tmp_path, {"model_type": "llama"})
        manager = self._make_manager(registry, mock_store)
        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/model")
        assert kind == "text"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_model_manager.py::TestDetectModelKind -k whisper -v`
Expected: FAIL — the dims test returns `"unknown"`, the model_type test returns `"unknown"`.

- [ ] **Step 3: Implement**

In `olmlx/engine/model_manager.py`, locate (currently line 1654):

```python
        model_type = config.get("model_type", "").lower()
        if not model_type:
            return "unknown"
```

Replace those three lines with:

```python
        model_type = config.get("model_type", "").lower()

        # Whisper STT (issue #366). Check before the empty-model_type return:
        # mlx-community whisper repos ship a non-HF config.json carrying
        # mlx_whisper.whisper.ModelDimensions fields, and load_model pops
        # "model_type", so model_type is often absent. The dims keys are the
        # robust discriminator; the model_type check covers HF-style configs.
        if model_type == "whisper" or (
            "n_mels" in config and "n_audio_state" in config
        ):
            return "whisper"

        if not model_type:
            return "unknown"
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_model_manager.py::TestDetectModelKind -v`
Expected: PASS (all, including the new three and the pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/model_manager.py tests/test_model_manager.py
git commit -m "feat(engine): detect whisper model kind (#366)"
```

---

## Task 3: Load whisper models through ModelManager

**Files:**
- Modify: `olmlx/engine/model_manager.py` — `LoadedModel.is_whisper` field (~line 740), `_load_model` whisper branch (~line 3098), `_probe_cache_capabilities` guard (~line 2008), kv-quant/spectral guard in `ensure_loaded` (~line 1203 / ~1457).
- Test: `tests/test_model_manager.py`

- [ ] **Step 1: Write the failing tests**

Add a new class to `tests/test_model_manager.py`:

```python
class TestWhisperLoad:
    def test_load_model_whisper_branch(self, tmp_path, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        fake_whisper = MagicMock()
        with (
            patch.object(
                manager, "_detect_model_kind", return_value="whisper"
            ),
            patch(
                "mlx_whisper.load_models.load_model", return_value=fake_whisper
            ) as mock_load,
        ):
            model, tok, is_vlm, caps, spec = manager._load_model("test/whisper")
        assert model is fake_whisper
        assert tok is None
        assert is_vlm is False
        assert spec is None
        mock_load.assert_called_once()

    def test_probe_cache_skipped_for_whisper(self, registry, mock_store):
        from olmlx.engine.model_manager import LoadedModel

        manager = ModelManager(registry, mock_store)
        lm = LoadedModel(
            name="whisper:latest",
            hf_path="test/whisper",
            model=MagicMock(),
            tokenizer=None,
            is_whisper=True,
        )
        # Defaults that the probe would normally overwrite.
        lm.supports_cache_trim = True
        lm.supports_cache_persistence = False
        with patch(
            "mlx_lm.models.cache.make_prompt_cache"
        ) as mock_make:
            manager._probe_cache_capabilities(lm)
        mock_make.assert_not_called()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_model_manager.py::TestWhisperLoad -v`
Expected: FAIL — `LoadedModel` has no `is_whisper`; `_load_model` has no whisper branch.

- [ ] **Step 3a: Add the `is_whisper` field**

In `olmlx/engine/model_manager.py`, in the `LoadedModel` dataclass, add after `is_flash_moe: bool = False` (near line 740):

```python
    is_whisper: bool = False
```

- [ ] **Step 3b: Add the whisper load branch**

In `_load_model`, locate (currently line 3098):

```python
        kind = self._detect_model_kind(hf_path)
        logger.info("Detected model kind for %s: %s", hf_path, kind)
```

Immediately after the `logger.info(...)` line, insert:

```python
        if kind == "whisper":
            # Whisper STT (issue #366). Load via mlx-whisper's loader and
            # return no tokenizer/caps/speculative — the transcription path
            # drives mlx_whisper.transcribe() directly.
            import mlx.core as mx
            import mlx_whisper.load_models as whisper_loader

            model = whisper_loader.load_model(load_path, dtype=mx.float16)
            return model, None, False, TemplateCaps(), None
```

(`TemplateCaps` is already imported in this module — confirm via `grep -n "TemplateCaps" olmlx/engine/model_manager.py`; it is used by other branches.)

- [ ] **Step 3c: Guard `_probe_cache_capabilities`**

In `_probe_cache_capabilities` (line ~2008), add at the very top of the method body, before the `try: from mlx_lm.models.cache import make_prompt_cache`:

```python
        if lm.is_whisper:
            # Whisper has no LLM-style prompt cache; nothing to probe.
            return
```

- [ ] **Step 3d: Guard kv-quant/spectral resolution for whisper**

In `ensure_loaded`, the kv-quant value is resolved at line ~1203:

```python
                kv_cache_quant = model_config.resolved_kv_cache_quant()
```

This value flows into `_find_spectral_dir` (line ~1457) and the `LoadedModel(kv_cache_quant=...)` kwarg. Whisper models must not trigger spectral calibration. The model kind is known via `_detect_model_kind`. Add a guard immediately after the `kv_cache_quant = ...` line:

```python
                # Whisper models (issue #366) have no LLM KV cache — never
                # apply KV-cache quantization / spectral calibration to them.
                if self._detect_model_kind(hf_path) == "whisper":
                    kv_cache_quant = None
```

Then set `is_whisper=True` on the constructed `LoadedModel`. Find the `LoadedModel(` constructor call (line ~1462) and add a kwarg alongside `is_flash_moe=is_flash_moe,`:

```python
                        is_whisper=(self._detect_model_kind(hf_path) == "whisper"),
```

Note: `_detect_model_kind` reads a cached local `config.json` after download, so the extra calls here are cheap local file reads. If you prefer to avoid the double call, capture `_kind = self._detect_model_kind(hf_path)` once near the kv-quant guard and reuse it for both the guard and the `is_whisper` kwarg.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_model_manager.py::TestWhisperLoad -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/model_manager.py tests/test_model_manager.py
git commit -m "feat(engine): load whisper models via ModelManager (#366)"
```

---

## Task 4: SRT/VTT/timestamp formatters

**Files:**
- Create: `olmlx/routers/audio.py` (formatter helpers first; the route is added in Task 7)
- Test: `tests/test_audio_formats.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audio_formats.py`:

```python
"""Tests for srt/vtt formatting helpers in olmlx.routers.audio."""

from olmlx.routers.audio import _format_timestamp, srt_from_segments, vtt_from_segments

SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": " Hello world"},
    {"start": 2.5, "end": 3661.0, "text": " Second line"},
]


def test_format_timestamp_srt():
    assert _format_timestamp(0.0, decimal=",") == "00:00:00,000"
    assert _format_timestamp(3661.5, decimal=",") == "01:01:01,500"


def test_format_timestamp_vtt():
    assert _format_timestamp(3661.5, decimal=".") == "01:01:01.500"


def test_srt_from_segments():
    out = srt_from_segments(SEGMENTS)
    lines = out.splitlines()
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:02,500"
    assert lines[2] == "Hello world"
    assert lines[3] == ""
    assert lines[4] == "2"
    assert lines[5] == "00:00:02,500 --> 01:01:01,000"
    assert lines[6] == "Second line"


def test_vtt_from_segments():
    out = vtt_from_segments(SEGMENTS)
    lines = out.splitlines()
    assert lines[0] == "WEBVTT"
    assert lines[1] == ""
    assert lines[2] == "00:00:00.000 --> 00:00:02.500"
    assert lines[3] == "Hello world"


def test_empty_segments():
    assert srt_from_segments([]) == ""
    assert vtt_from_segments([]) == "WEBVTT\n"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_audio_formats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'olmlx.routers.audio'`

- [ ] **Step 3: Implement the helpers**

Create `olmlx/routers/audio.py` with (route added later in Task 7):

```python
"""OpenAI-compatible audio transcription route (/v1/audio/transcriptions).

Backed by mlx-whisper. Whisper models are managed through ModelManager;
see engine/inference.py::generate_transcription and CLAUDE.md.
"""

from __future__ import annotations


def _format_timestamp(seconds: float, *, decimal: str) -> str:
    """Format seconds as HH:MM:SS<decimal>mmm (decimal "," for SRT, "." for VTT)."""
    if seconds < 0:
        seconds = 0.0
    millis = round(seconds * 1000.0)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


def srt_from_segments(segments: list[dict]) -> str:
    """Render whisper segments as SubRip (.srt) text."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp(float(seg["start"]), decimal=",")
        end = _format_timestamp(float(seg["end"]), decimal=",")
        text = str(seg.get("text", "")).strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def vtt_from_segments(segments: list[dict]) -> str:
    """Render whisper segments as WebVTT (.vtt) text."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp(float(seg["start"]), decimal=".")
        end = _format_timestamp(float(seg["end"]), decimal=".")
        text = str(seg.get("text", "")).strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)
```

Note on the empty-VTT assertion: `"\n".join(["WEBVTT", ""])` == `"WEBVTT\n"`, matching the test. For non-empty SRT, `"\n".join` of blocks each ending in `\n` yields a blank line between blocks; the `test_srt_from_segments` assertions account for this.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_audio_formats.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add olmlx/routers/audio.py tests/test_audio_formats.py
git commit -m "feat(audio): srt/vtt segment formatters (#366)"
```

---

## Task 5: Schemas

**Files:**
- Create: `olmlx/schemas/audio.py`
- Test: covered by Task 7 router tests (no standalone schema test needed — these are plain response models)

- [ ] **Step 1: Implement the schemas**

Create `olmlx/schemas/audio.py`:

```python
"""Response schemas for /v1/audio/transcriptions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TranscriptionResponse(BaseModel):
    """Default `response_format=json` body."""

    text: str


class TranscriptionSegment(BaseModel):
    """One whisper segment. Tolerant of extra keys mlx-whisper may add."""

    model_config = ConfigDict(extra="ignore")

    id: int = 0
    seek: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class VerboseTranscriptionResponse(BaseModel):
    """`response_format=verbose_json` body (OpenAI-compatible shape)."""

    task: str = "transcribe"
    language: str = ""
    duration: float = 0.0
    text: str = ""
    segments: list[TranscriptionSegment] = []
```

- [ ] **Step 2: Verify it imports**

Run: `uv run python -c "from olmlx.schemas.audio import VerboseTranscriptionResponse, TranscriptionResponse, TranscriptionSegment; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add olmlx/schemas/audio.py
git commit -m "feat(audio): transcription response schemas (#366)"
```

---

## Task 6: `generate_transcription` engine function

**Files:**
- Modify: `olmlx/engine/inference.py` (add function near `generate_embeddings`, ~line 3181)
- Test: Create `tests/test_inference_transcription.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_transcription.py`:

```python
"""Tests for generate_transcription."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmlx.engine.inference import generate_transcription


@pytest.mark.asyncio
async def test_generate_transcription_injects_and_calls(mock_manager):
    lm = mock_manager._loaded["qwen3:latest"]
    lm.is_whisper = True
    mock_manager.ensure_loaded = AsyncMock(return_value=lm)

    result = {"text": "hello", "segments": [], "language": "en"}

    import mlx_whisper.transcribe as wt

    # Reset holder so we can assert injection happened.
    wt.ModelHolder.model = None
    wt.ModelHolder.model_path = None

    with patch("mlx_whisper.transcribe.transcribe", return_value=result) as mock_tx:
        out = await generate_transcription(
            mock_manager,
            "whisper-turbo",
            "/tmp/clip.wav",
            language="en",
            prompt="hi",
            temperature=0.0,
            word_timestamps=False,
        )

    assert out == result
    # Our managed model was injected into the holder.
    assert wt.ModelHolder.model is lm.model
    mock_tx.assert_called_once()
    _, kwargs = mock_tx.call_args
    assert kwargs["initial_prompt"] == "hi"
    assert kwargs["language"] == "en"


@pytest.mark.asyncio
async def test_generate_transcription_ffmpeg_missing(mock_manager):
    lm = mock_manager._loaded["qwen3:latest"]
    lm.is_whisper = True
    mock_manager.ensure_loaded = AsyncMock(return_value=lm)

    with patch(
        "mlx_whisper.transcribe.transcribe",
        side_effect=FileNotFoundError("ffmpeg"),
    ):
        with pytest.raises(ValueError, match="ffmpeg"):
            await generate_transcription(
                mock_manager, "whisper-turbo", "/tmp/clip.wav"
            )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_inference_transcription.py -v`
Expected: FAIL — `ImportError: cannot import name 'generate_transcription'`

- [ ] **Step 3: Implement**

In `olmlx/engine/inference.py`, add after `generate_embeddings` (after its `return embeddings`, ~line 3253):

```python
async def generate_transcription(
    manager: ModelManager,
    model_name: str,
    audio_path: str,
    *,
    language: str | None = None,
    prompt: str | None = None,
    temperature: float = 0.0,
    word_timestamps: bool = False,
    keep_alive: str | None = None,
) -> dict:
    """Transcribe an audio file with a whisper model managed by ModelManager.

    Loads (or reuses) the whisper model via ``ensure_loaded``, injects it into
    mlx_whisper's module-level ``ModelHolder`` so ``transcribe()`` reuses the
    managed model instead of loading its own, and runs the (synchronous)
    transcription in a worker thread. Returns the raw mlx-whisper result dict
    (``text``, ``segments``, ``language``). Issue #366.
    """
    import mlx_whisper.transcribe as whisper_transcribe

    lm = await manager.ensure_loaded(model_name, keep_alive)

    # Resolve the on-disk path used as path_or_hf_repo so the ModelHolder
    # cache key matches what we inject.
    if manager.store is not None:
        load_path = str(manager.store.local_path(lm.hf_path))
    else:
        load_path = lm.hf_path

    async with _inference_locked(lm.inference_queue_timeout, sync_mode=lm.sync_mode):
        with _inference_ref(lm):
            # Inject the managed model so transcribe() reuses it. Safe because
            # _inference_locked serializes inference (no ModelHolder race).
            whisper_transcribe.ModelHolder.model = lm.model
            whisper_transcribe.ModelHolder.model_path = load_path

            def _run() -> dict:
                try:
                    return whisper_transcribe.transcribe(
                        audio_path,
                        path_or_hf_repo=load_path,
                        language=language,
                        initial_prompt=prompt,
                        temperature=temperature,
                        word_timestamps=word_timestamps,
                    )
                except FileNotFoundError as exc:
                    raise ValueError(
                        "Audio decoding failed: ffmpeg was not found on PATH. "
                        "Install ffmpeg (e.g. `brew install ffmpeg`) to use "
                        "/v1/audio/transcriptions."
                    ) from exc
                except RuntimeError as exc:
                    # mlx_whisper.audio.load_audio raises RuntimeError on a
                    # failed ffmpeg decode (bad/corrupt/unsupported file).
                    raise ValueError(f"Audio decoding failed: {exc}") from exc

            return await asyncio.to_thread(_run)
```

Confirm `asyncio` is already imported at the top of `inference.py` (it is — `_inference_lock = asyncio.Lock()` exists). Confirm `_inference_locked` and `_inference_ref` names: `grep -n "_inference_locked\|def _inference_ref\|_inference_ref(" olmlx/engine/inference.py` (used by `generate_embeddings`).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_inference_transcription.py -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/inference.py tests/test_inference_transcription.py
git commit -m "feat(engine): generate_transcription via mlx-whisper (#366)"
```

---

## Task 7: The route + app registration

**Files:**
- Modify: `olmlx/routers/audio.py` (add router + endpoint to the formatter module from Task 4)
- Modify: `olmlx/app.py:21` (import) and `olmlx/app.py:311` (include)
- Test: Create `tests/test_routers_audio.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_routers_audio.py`:

```python
"""Tests for olmlx.routers.audio (/v1/audio/transcriptions)."""

from unittest.mock import AsyncMock, patch

import pytest

RESULT = {
    "text": "Hello world",
    "language": "en",
    "segments": [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 2.5,
            "text": " Hello world",
            "tokens": [1, 2],
            "temperature": 0.0,
            "avg_logprob": -0.1,
            "compression_ratio": 1.0,
            "no_speech_prob": 0.01,
        }
    ],
}


def _file():
    return {"file": ("clip.wav", b"RIFFfakewavdata", "audio/wav")}


@pytest.mark.asyncio
async def test_transcription_json(app_client):
    with patch(
        "olmlx.routers.audio.generate_transcription",
        new_callable=AsyncMock,
        return_value=RESULT,
    ):
        resp = await app_client.post(
            "/v1/audio/transcriptions",
            files=_file(),
            data={"model": "whisper-turbo"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"text": "Hello world"}


@pytest.mark.asyncio
async def test_transcription_verbose_json(app_client):
    with patch(
        "olmlx.routers.audio.generate_transcription",
        new_callable=AsyncMock,
        return_value=RESULT,
    ):
        resp = await app_client.post(
            "/v1/audio/transcriptions",
            files=_file(),
            data={"model": "whisper-turbo", "response_format": "verbose_json"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    assert body["text"] == "Hello world"
    assert body["duration"] == 2.5
    assert len(body["segments"]) == 1
    assert body["segments"][0]["text"] == " Hello world"


@pytest.mark.asyncio
async def test_transcription_text(app_client):
    with patch(
        "olmlx.routers.audio.generate_transcription",
        new_callable=AsyncMock,
        return_value=RESULT,
    ):
        resp = await app_client.post(
            "/v1/audio/transcriptions",
            files=_file(),
            data={"model": "whisper-turbo", "response_format": "text"},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "Hello world"


@pytest.mark.asyncio
async def test_transcription_srt(app_client):
    with patch(
        "olmlx.routers.audio.generate_transcription",
        new_callable=AsyncMock,
        return_value=RESULT,
    ):
        resp = await app_client.post(
            "/v1/audio/transcriptions",
            files=_file(),
            data={"model": "whisper-turbo", "response_format": "srt"},
        )
    assert resp.status_code == 200
    assert "00:00:00,000 --> 00:00:02,500" in resp.text
    assert "Hello world" in resp.text


@pytest.mark.asyncio
async def test_transcription_unknown_format(app_client):
    resp = await app_client.post(
        "/v1/audio/transcriptions",
        files=_file(),
        data={"model": "whisper-turbo", "response_format": "bogus"},
    )
    assert resp.status_code == 400
    assert "response_format" in resp.text


@pytest.mark.asyncio
async def test_transcription_too_large(app_client, monkeypatch):
    monkeypatch.setattr("olmlx.routers.audio.settings.audio_max_bytes", 4)
    resp = await app_client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"way too many bytes", "audio/wav")},
        data={"model": "whisper-turbo"},
    )
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_transcription_word_timestamps(app_client):
    with patch(
        "olmlx.routers.audio.generate_transcription",
        new_callable=AsyncMock,
        return_value=RESULT,
    ) as mock_tx:
        resp = await app_client.post(
            "/v1/audio/transcriptions",
            files=_file(),
            data={
                "model": "whisper-turbo",
                "timestamp_granularities[]": "word",
            },
        )
    assert resp.status_code == 200
    _, kwargs = mock_tx.call_args
    assert kwargs["word_timestamps"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_routers_audio.py -v`
Expected: FAIL — route not registered (404) / import errors.

- [ ] **Step 3a: Add the route to `olmlx/routers/audio.py`**

Append to `olmlx/routers/audio.py` (which currently holds only the formatter helpers and the module docstring/imports). Add these imports at the top of the file (below the existing `from __future__ import annotations`):

```python
import os
import tempfile

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import PlainTextResponse

from olmlx.config import settings
from olmlx.engine.inference import generate_transcription
from olmlx.schemas.audio import (
    TranscriptionResponse,
    TranscriptionSegment,
    VerboseTranscriptionResponse,
)

router = APIRouter()

_VALID_FORMATS = {"json", "verbose_json", "text", "srt", "vtt"}
```

Then append the endpoint at the end of the file:

```python
@router.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: list[str] | None = Form(
        None, alias="timestamp_granularities[]"
    ),
):
    if response_format not in _VALID_FORMATS:
        raise ValueError(
            f"Unsupported response_format '{response_format}'. "
            f"Must be one of: {', '.join(sorted(_VALID_FORMATS))}."
        )

    word_timestamps = bool(
        timestamp_granularities and "word" in timestamp_granularities
    )

    manager = request.app.state.model_manager
    max_bytes = settings.audio_max_bytes

    # Stream the upload to a temp file under the size cap.
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    received = 0
    try:
        with os.fdopen(tmp_fd, "wb") as tmp:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > max_bytes:
                    return PlainTextResponse(
                        f"Audio file exceeds the limit of {max_bytes} bytes "
                        f"(OLMLX_AUDIO_MAX_BYTES).",
                        status_code=413,
                    )
                tmp.write(chunk)

        result = await generate_transcription(
            manager,
            model,
            tmp_path,
            language=language,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    text = result.get("text", "")
    segments = result.get("segments", []) or []

    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "srt":
        return PlainTextResponse(srt_from_segments(segments))
    if response_format == "vtt":
        return PlainTextResponse(vtt_from_segments(segments))
    if response_format == "verbose_json":
        duration = float(segments[-1]["end"]) if segments else 0.0
        return VerboseTranscriptionResponse(
            language=result.get("language", "") or "",
            duration=duration,
            text=text,
            segments=[TranscriptionSegment(**s) for s in segments],
        )
    # default: json
    return TranscriptionResponse(text=text)
```

Note: returning a 413 `PlainTextResponse` from inside the handler short-circuits cleanly because the `finally` still unlinks the temp file. The `srt_from_segments`/`vtt_from_segments` helpers are defined earlier in this same module (Task 4).

- [ ] **Step 3b: Register the router in `olmlx/app.py`**

Edit the import block at `olmlx/app.py:21`:

```python
from olmlx.routers import (
    anthropic,
    audio,
    blobs,
    chat,
    embed,
    generate,
    manage,
    models,
    openai,
    status,
)
```

And add the include after `app.include_router(openai.router)` (line 311):

```python
    app.include_router(audio.router)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_routers_audio.py -v`
Expected: PASS (all 7)

- [ ] **Step 5: Commit**

```bash
git add olmlx/routers/audio.py olmlx/app.py tests/test_routers_audio.py
git commit -m "feat(audio): /v1/audio/transcriptions route (#366)"
```

---

## Task 8: Default whisper model names

**Files:**
- Modify: `olmlx/cli.py:30` (`DEFAULT_MODELS`)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_default_models_include_whisper():
    from olmlx.cli import DEFAULT_MODELS

    assert DEFAULT_MODELS["whisper-turbo:latest"] == (
        "mlx-community/whisper-large-v3-turbo"
    )
    assert "whisper-large:latest" in DEFAULT_MODELS
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py::test_default_models_include_whisper -v`
Expected: FAIL — KeyError.

- [ ] **Step 3: Implement**

In `olmlx/cli.py`, extend `DEFAULT_MODELS` (line 30):

```python
DEFAULT_MODELS = {
    "llama3.2:latest": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistral:7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "qwen2.5:3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "gemma2:2b": "mlx-community/gemma-2-2b-it-4bit",
    "whisper-turbo:latest": "mlx-community/whisper-large-v3-turbo",
    "whisper-large:latest": "mlx-community/whisper-large-v3-mlx",
}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli.py::test_default_models_include_whisper -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/cli.py tests/test_cli.py
git commit -m "feat(cli): seed default whisper model names (#366)"
```

---

## Task 9: Docs

**Files:**
- Modify: `CLAUDE.md` (add a Key Design Decisions bullet; add `routers/audio.py` and `schemas/audio.py` to the structure tree)

- [ ] **Step 1: Add the design bullet**

In `CLAUDE.md`, under "## Key Design Decisions", add a new bullet:

```markdown
- **Audio transcription** (`routers/audio.py`): OpenAI-compatible `POST /v1/audio/transcriptions` backed by `mlx-whisper`. Multipart upload (`file` + `model`, optional `language`/`prompt`/`response_format`/`temperature`/`timestamp_granularities[]`) is streamed to a temp file under `OLMLX_AUDIO_MAX_BYTES` (default 100 MB; 413 on overflow). Whisper models are a first-class `ModelManager` kind: `_detect_model_kind` returns `"whisper"` (via `model_type == "whisper"` or the presence of mlx whisper dims `n_mels`/`n_audio_state` — mlx-community repos ship a non-HF `config.json`), `_load_model` loads them with `mlx_whisper.load_models.load_model`, and `LoadedModel.is_whisper` guards the LLM-only prompt-cache probe and KV-quant/spectral paths. `engine/inference.py::generate_transcription` acquires the inference lock + active-ref, injects the managed model into `mlx_whisper.transcribe.ModelHolder` (the module-level singleton — race-free under the serialized lock) so `transcribe()` reuses it instead of loading its own, and runs the synchronous transcribe in a worker thread. Response formats: `json` (`{text}`), `verbose_json` (`{task, language, duration, text, segments[]}`), `text`, `srt`, `vtt` (srt/vtt rendered from segments by helpers in `routers/audio.py`). **Requires ffmpeg on PATH** (mlx-whisper decodes via ffmpeg); a missing/failed decode surfaces as HTTP 400. Convenience names `whisper-turbo`/`whisper-large` are seeded into `DEFAULT_MODELS`; direct HF paths auto-register as usual. Streaming transcription and diarization are out of scope.
```

Also add to the structure tree under `routers/` and `schemas/`:

```markdown
│   ├── audio.py        # /v1/audio/transcriptions — OpenAI Whisper STT (mlx-whisper)
```
```markdown
│   ├── audio.py        # Audio transcription response models
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document /v1/audio/transcriptions (#366)"
```

---

## Task 10: Full verification

- [ ] **Step 1: Run the full new-test set**

Run:
```bash
uv run pytest tests/test_config.py tests/test_model_manager.py \
  tests/test_audio_formats.py tests/test_inference_transcription.py \
  tests/test_routers_audio.py tests/test_cli.py -v
```
Expected: all PASS.

- [ ] **Step 2: Run the whole suite to check for regressions**

Run: `uv run pytest -q`
Expected: no new failures vs. baseline (pre-existing skips/xfails unchanged).

- [ ] **Step 3: Lint & format (required before push — user preference)**

Run:
```bash
uv run ruff check olmlx tests
uv run ruff format olmlx tests
```
Expected: clean (or only formatting changes applied). Re-run `ruff check` after format. Commit any formatting changes:

```bash
git add -A && git commit -m "style: ruff format (#366)" || echo "nothing to format"
```

- [ ] **Step 4: Smoke-import the app**

Run: `uv run python -c "from olmlx.app import create_app; create_app(); print('app ok')"`
Expected: prints `app ok` (route registration imports cleanly).

---

## Self-Review Notes

- **Spec coverage:** route (T7), multipart upload (T7), formats json/verbose_json/text/srt/vtt (T4+T7), model selection via ModelManager (T2+T3), whisper engine path in model_manager.py (T2+T3), configurable max bytes (T1+T7), schemas (T5), engine inference (T6), aliases (T8), ffmpeg requirement + error (T6), docs (T9). All spec sections map to a task.
- **Type consistency:** `generate_transcription` signature matches between T6 definition and T7 call site and T6 tests (`language`, `prompt`, `temperature`, `word_timestamps`, `keep_alive`). `srt_from_segments`/`vtt_from_segments`/`_format_timestamp` names match between T4 and T7. `is_whisper` field name consistent across T3 and T6.
- **Placeholders:** none — every code step shows full code.
