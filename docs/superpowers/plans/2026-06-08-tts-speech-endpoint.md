# TTS Endpoint (`/v1/audio/speech`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an OpenAI-compatible `POST /v1/audio/speech` text-to-speech endpoint backed by mlx-audio (Kokoro-82M), supporting all six OpenAI audio formats with streaming.

**Architecture:** Mirror the Whisper transcription path (#366). TTS becomes a first-class `ModelManager` kind (`is_tts`), an engine async-generator (`generate_speech`) drives mlx-audio under the serialized inference lock and bridges its synchronous segment generator to async via a worker thread + queue, and a route in the shared `routers/audio.py` encodes the float-audio stream into the requested container (raw PCM / pure-Python WAV / ffmpeg-piped compressed formats) via a new `utils/audio.py`.

**Tech Stack:** Python, FastAPI/Starlette, mlx-audio (Kokoro), ffmpeg (already required), pytest.

---

## Reference facts (verified against installed `mlx-audio==0.4.4`)

- **Model API:** `mlx_audio.tts.utils.load_model(Path) -> nn.Module`. The model exposes `model.generate(text: str, voice: str, speed: float, lang_code="a") -> Iterator[GenerationResult]`. Each `GenerationResult` has `.audio` (1-D `mx.array` of float samples), `.sample_rate` (24000 for Kokoro), `.samples`, `.segment_idx`.
- **Detection:** Kokoro's `config.json` has **no** `model_type` key. Its distinctive keys are `istftnet` and `plbert` (StyleTTS/Kokoro signature). It also contains `n_mels` but **not** `n_audio_state`, so the existing Whisper detector does not misfire.
- **Path resolution:** mlx-audio resolves the Kokoro class from the on-disk directory name parts (olmlx stores under `<owner>--<repo>`, which contains "Kokoro"). No `model_type` needed.
- **Voices (Kokoro-82M, 54 total):** include `af_alloy`, `am_echo`, `bm_fable`, `am_onyx`, `af_nova`, `af_sky`, `af_heart`, etc. Full list embedded in Task 4.
- **No model-alias seed list exists** in the registry — models auto-register on pull. We document `olmlx models pull prince-canuma/Kokoro-82M` rather than seeding an alias.

## File Structure

- **Modify** `pyproject.toml` — add `mlx-audio` dependency.
- **Modify** `olmlx/config.py` — add `tts_max_input_chars` setting.
- **Modify** `olmlx/engine/model_manager.py` — `is_tts` field, `_detect_model_kind` → "tts", `kind == "tts"` load branch, three guard sites, `LoadedModel` construction.
- **Create** `olmlx/engine/tts.py` — `OPENAI_VOICE_MAP`, `KOKORO_VOICES`, `resolve_voice()`. (Pure, dependency-free; lives in engine so both router and tests import it without pulling FastAPI.)
- **Create** `olmlx/utils/audio.py` — float→PCM conversion, WAV header, ffmpeg streaming encoder, format/media-type tables.
- **Modify** `olmlx/schemas/audio.py` — add `SpeechRequest`.
- **Modify** `olmlx/engine/inference.py` — add `generate_speech` async generator.
- **Modify** `olmlx/routers/audio.py` — add `POST /v1/audio/speech`.
- **Create** `tests/test_tts_voice.py`, `tests/test_audio_encoding.py`, `tests/test_inference_speech.py`, `tests/test_routers_speech.py` — unit tests.
- **Create** `tests/live/test_tts_speech.py` — real-Kokoro SDK test (`real_model`).
- **Modify** `CLAUDE.md` — document the endpoint.

---

## Task 1: Add mlx-audio dependency and config setting

**Files:**
- Modify: `pyproject.toml`
- Modify: `olmlx/config.py:124-130` (next to `audio_max_bytes`)

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, in the `dependencies` array, add the line after `"mlx-whisper>=0.4.3",`:

```toml
    "mlx-audio>=0.4.4",
```

- [ ] **Step 2: Add the config setting**

In `olmlx/config.py`, immediately after the `audio_max_bytes` field block (ends at `] = 100 * 1024 * 1024`), add:

```python
    tts_max_input_chars: Annotated[
        int,
        Field(
            gt=0,
            description="Max input length for /v1/audio/speech (OLMLX_TTS_MAX_INPUT_CHARS).",
        ),
    ] = 8192
```

- [ ] **Step 3: Sync and verify import**

Run: `uv sync --no-editable && uv run python -c "import mlx_audio.tts.utils; from olmlx.config import settings; print(settings.tts_max_input_chars)"`
Expected: prints `8192` with no import error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock olmlx/config.py
git commit -m "feat(tts): add mlx-audio dependency and tts_max_input_chars setting (#367)"
```

---

## Task 2: Detect TTS models in `_detect_model_kind`

**Files:**
- Modify: `olmlx/engine/model_manager.py:1713-1719` (after the Whisper branch)
- Test: `tests/test_model_kind_tts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_kind_tts.py`:

```python
"""TTS model-kind detection (#367)."""

from unittest.mock import patch

from olmlx.engine.model_manager import ModelManager


def _detect(config: dict) -> str:
    mgr = ModelManager.__new__(ModelManager)  # bypass __init__
    mgr.store = None
    with patch("huggingface_hub.hf_hub_download") as dl:
        import json
        import tempfile

        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(config, f)
        f.flush()
        dl.return_value = f.name
        return mgr._detect_model_kind("some/repo")


def test_kokoro_config_detected_as_tts():
    # Kokoro config signature: istftnet + plbert, no model_type.
    cfg = {"istftnet": {}, "plbert": {}, "n_mels": 24, "n_token": 178}
    assert _detect(cfg) == "tts"


def test_whisper_not_misdetected_as_tts():
    cfg = {"n_mels": 80, "n_audio_state": 768}
    assert _detect(cfg) == "whisper"


def test_plain_text_model_not_tts():
    cfg = {"model_type": "qwen3"}
    assert _detect(cfg) != "tts"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_kind_tts.py -v`
Expected: `test_kokoro_config_detected_as_tts` FAILS (returns "unknown", not "tts").

- [ ] **Step 3: Add the detection branch**

In `olmlx/engine/model_manager.py`, immediately **after** the Whisper detection block (the `if model_type == "whisper" or (...): return "whisper"` ending at line ~1716) and **before** `if not model_type:`, insert:

```python
        # TTS (issue #367). Kokoro/StyleTTS configs carry no ``model_type``
        # but ship a distinctive ``istftnet`` + ``plbert`` signature. Check
        # before the empty-model_type return (Kokoro has no model_type) and
        # after Whisper (Kokoro also has ``n_mels`` but not ``n_audio_state``,
        # so the Whisper branch above already declined it).
        if "istftnet" in config and "plbert" in config:
            return "tts"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_kind_tts.py -v`
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/model_manager.py tests/test_model_kind_tts.py
git commit -m "feat(tts): detect Kokoro/TTS models in _detect_model_kind (#367)"
```

---

## Task 3: `LoadedModel.is_tts`, load branch, and guards

**Files:**
- Modify: `olmlx/engine/model_manager.py` — field (~479), load branch (~3510), construction (~1456), guards (~809, ~1101, ~2100)
- Test: `tests/test_load_tts_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_load_tts_model.py`:

```python
"""Loading a TTS model sets is_tts and skips LLM-only paths (#367)."""

from unittest.mock import MagicMock, patch

from olmlx.engine.model_manager import LoadedModel


def test_loadedmodel_has_is_tts_default_false():
    lm = LoadedModel(name="x", hf_path="x", model=object(), tokenizer=None)
    assert lm.is_tts is False


def test_load_branch_uses_mlx_audio(tmp_path):
    # The kind=="tts" branch must call mlx_audio.tts.utils.load_model and
    # return (model, None, False, TemplateCaps(), None).
    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.template_caps import TemplateCaps

    mgr = ModelManager.__new__(ModelManager)
    fake_model = MagicMock()
    with (
        patch.object(ModelManager, "_detect_model_kind", return_value="tts"),
        patch("mlx_audio.tts.utils.load_model", return_value=fake_model) as load,
    ):
        result = ModelManager._load_model_tts(mgr, "owner/Kokoro-82M", str(tmp_path))
    model, tok, is_vlm, caps, dec = result
    assert model is fake_model
    assert tok is None and is_vlm is False and dec is None
    assert isinstance(caps, TemplateCaps)
    load.assert_called_once()
```

> Note: Task 3 introduces a small `_load_model_tts` helper so the branch is unit-testable without driving the full `_load_model` dispatch.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_load_tts_model.py -v`
Expected: `test_loadedmodel_has_is_tts_default_false` FAILS (unexpected keyword / attribute), `test_load_branch_uses_mlx_audio` FAILS (`_load_model_tts` undefined).

- [ ] **Step 3a: Add the `is_tts` field**

In `olmlx/engine/model_manager.py`, in the `LoadedModel` dataclass, after `is_whisper: bool = False` (line ~479), add:

```python
    is_tts: bool = False
```

- [ ] **Step 3b: Add the `_load_model_tts` helper and call it from the load dispatch**

In `olmlx/engine/model_manager.py`, just after the `if kind == "whisper":` block (which ends with `return model, None, False, TemplateCaps(), None` at line ~3510), add:

```python
        if kind == "tts":
            return self._load_model_tts(hf_path, load_path)
```

Then add the helper method (place it near `_load_model`, e.g. directly below the `_load_model` method body). Use the exact signature the test calls:

```python
    def _load_model_tts(self, hf_path: str, load_path: str):
        """Load a TTS model (Kokoro) via mlx-audio (issue #367).

        Returns the 5-tuple shape ``_load_model`` uses
        ``(model, tokenizer, is_vlm, caps, speculative_decoder)``. TTS has no
        tokenizer / chat template / speculative decoder — the speech path
        drives ``model.generate`` directly.
        """
        from pathlib import Path

        import mlx_audio.tts.utils as tts_utils

        model = tts_utils.load_model(Path(load_path))
        return model, None, False, TemplateCaps(), None
```

- [ ] **Step 3c: Set `is_tts` on construction**

In `olmlx/engine/model_manager.py`, in the `LoadedModel(...)` construction (line ~1456), after `is_whisper=(_model_kind == "whisper"),` add:

```python
                        is_tts=(_model_kind == "tts"),
```

- [ ] **Step 3d: Skip KV-quant for TTS**

In `olmlx/engine/model_manager.py` at the `_model_kind` block (line ~1100), change:

```python
                _model_kind = self._detect_model_kind(hf_path)
                if _model_kind == "whisper":
                    kv_cache_quant = None
```

to:

```python
                _model_kind = self._detect_model_kind(hf_path)
                if _model_kind in ("whisper", "tts"):
                    kv_cache_quant = None
```

- [ ] **Step 3e: Guard the grammar-cache drop (line ~809)**

Change `if not lm.is_whisper:` (the grammar `drop_for_tokenizer` guard) to:

```python
        if not (lm.is_whisper or lm.is_tts):
```

- [ ] **Step 3f: Guard the prompt-cache probe (line ~2100)**

Change `if lm.is_whisper:` (the `# Whisper has no LLM-style prompt cache; nothing to probe.` guard) to:

```python
        if lm.is_whisper or lm.is_tts:
            # Whisper/TTS have no LLM-style prompt cache; nothing to probe.
            return
```

(Keep the existing comment line below replaced by the one above, or leave the original comment — just ensure the condition includes `lm.is_tts`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_load_tts_model.py -v`
Expected: both PASS.

- [ ] **Step 5: Run the broader model_manager tests for regressions**

Run: `uv run pytest tests/ -k "model_manager or load_tts or model_kind" -q`
Expected: PASS (no regressions).

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/model_manager.py tests/test_load_tts_model.py
git commit -m "feat(tts): is_tts LoadedModel kind + mlx-audio load branch + guards (#367)"
```

---

## Task 4: Voice resolution (`engine/tts.py`)

**Files:**
- Create: `olmlx/engine/tts.py`
- Test: `tests/test_tts_voice.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tts_voice.py`:

```python
"""OpenAI -> Kokoro voice resolution (#367)."""

import pytest

from olmlx.engine.tts import KOKORO_VOICES, UnknownVoiceError, resolve_voice


def test_openai_names_map_to_kokoro():
    assert resolve_voice("alloy") == "af_alloy"
    assert resolve_voice("echo") == "am_echo"
    assert resolve_voice("fable") == "bm_fable"
    assert resolve_voice("onyx") == "am_onyx"
    assert resolve_voice("nova") == "af_nova"
    assert resolve_voice("shimmer") == "af_sky"


def test_native_kokoro_voice_passes_through():
    assert resolve_voice("af_heart") == "af_heart"
    assert resolve_voice("bm_george") == "bm_george"


def test_unknown_voice_raises():
    with pytest.raises(UnknownVoiceError):
        resolve_voice("definitely_not_a_voice")


def test_all_map_targets_are_real_voices():
    from olmlx.engine.tts import OPENAI_VOICE_MAP

    for target in OPENAI_VOICE_MAP.values():
        assert target in KOKORO_VOICES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tts_voice.py -v`
Expected: FAIL with "No module named 'olmlx.engine.tts'".

- [ ] **Step 3: Implement `engine/tts.py`**

Create `olmlx/engine/tts.py`:

```python
"""Voice resolution for /v1/audio/speech (Kokoro). Issue #367.

Maps the six OpenAI voice names to Kokoro voice ids and validates native
Kokoro ids passed through directly. Pure / dependency-free so routers and
tests import it without pulling FastAPI or mlx.
"""

from __future__ import annotations


class UnknownVoiceError(ValueError):
    """Raised when a requested voice is neither an OpenAI alias nor a known
    Kokoro voice id. Surfaced as HTTP 422 by the router."""


# Kokoro-82M ships 54 voices (prince-canuma/Kokoro-82M, voices/*.pt).
KOKORO_VOICES: frozenset[str] = frozenset(
    {
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
        "am_onyx", "am_puck", "am_santa",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        "ef_dora", "em_alex", "em_santa", "ff_siwis",
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        "if_sara", "im_nicola",
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        "pf_dora", "pm_alex", "pm_santa",
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    }
)

# OpenAI voice name -> Kokoro voice id. Names chosen to match where Kokoro
# ships an identically-named voice; shimmer has no Kokoro twin (-> af_sky).
OPENAI_VOICE_MAP: dict[str, str] = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    "shimmer": "af_sky",
}


def resolve_voice(voice: str) -> str:
    """Map an OpenAI voice name to a Kokoro voice id, or pass a native id
    through. Raise :class:`UnknownVoiceError` for anything unrecognized."""
    if voice in OPENAI_VOICE_MAP:
        return OPENAI_VOICE_MAP[voice]
    if voice in KOKORO_VOICES:
        return voice
    raise UnknownVoiceError(
        f"Unknown voice '{voice}'. Use an OpenAI voice "
        f"({', '.join(sorted(OPENAI_VOICE_MAP))}) or a Kokoro voice id."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tts_voice.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/tts.py tests/test_tts_voice.py
git commit -m "feat(tts): OpenAI->Kokoro voice resolution (#367)"
```

---

## Task 5: `SpeechRequest` schema

**Files:**
- Modify: `olmlx/schemas/audio.py`
- Test: `tests/test_schema_speech.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_schema_speech.py`:

```python
"""SpeechRequest schema (#367)."""

import pytest
from pydantic import ValidationError

from olmlx.schemas.audio import SpeechRequest


def test_defaults():
    r = SpeechRequest(model="kokoro", input="hello", voice="alloy")
    assert r.response_format == "mp3"
    assert r.speed == 1.0


def test_speed_bounds():
    with pytest.raises(ValidationError):
        SpeechRequest(model="m", input="x", voice="alloy", speed=5.0)
    with pytest.raises(ValidationError):
        SpeechRequest(model="m", input="x", voice="alloy", speed=0.1)


def test_required_fields():
    with pytest.raises(ValidationError):
        SpeechRequest(model="m", voice="alloy")  # missing input
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schema_speech.py -v`
Expected: FAIL with ImportError on `SpeechRequest`.

- [ ] **Step 3: Add the schema**

In `olmlx/schemas/audio.py`, add the import of `Field` and the new model. Change the import line `from pydantic import BaseModel, ConfigDict` to:

```python
from pydantic import BaseModel, ConfigDict, Field
```

Then append:

```python
class SpeechRequest(BaseModel):
    """`POST /v1/audio/speech` request body (OpenAI-compatible)."""

    model: str
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_schema_speech.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add olmlx/schemas/audio.py tests/test_schema_speech.py
git commit -m "feat(tts): SpeechRequest schema (#367)"
```

---

## Task 6: Audio encoding (`utils/audio.py`)

**Files:**
- Create: `olmlx/utils/audio.py`
- Test: `tests/test_audio_encoding.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_encoding.py`:

```python
"""Audio encoding helpers for /v1/audio/speech (#367)."""

import struct

import numpy as np
import pytest

from olmlx.utils.audio import (
    FORMAT_MEDIA_TYPES,
    SUPPORTED_FORMATS,
    UnsupportedFormatError,
    float_to_pcm16,
    validate_format,
    wav_bytes,
)


def test_float_to_pcm16_roundtrip():
    samples = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    raw = float_to_pcm16(samples)
    assert isinstance(raw, bytes)
    decoded = np.frombuffer(raw, dtype="<i2")
    assert decoded[0] == 0
    assert decoded[1] == 32767
    assert decoded[2] == -32767
    assert abs(int(decoded[3]) - 16383) <= 1


def test_float_to_pcm16_clips():
    samples = np.array([2.0, -2.0], dtype=np.float32)
    decoded = np.frombuffer(float_to_pcm16(samples), dtype="<i2")
    assert decoded[0] == 32767
    assert decoded[1] == -32767


def test_wav_bytes_has_valid_header():
    pcm = float_to_pcm16(np.zeros(100, dtype=np.float32))
    out = wav_bytes(pcm, sample_rate=24000)
    assert out[:4] == b"RIFF"
    assert out[8:12] == b"WAVE"
    # data chunk size == len(pcm)
    riff_size = struct.unpack("<I", out[4:8])[0]
    assert riff_size == 36 + len(pcm)


def test_validate_format_accepts_supported():
    for fmt in SUPPORTED_FORMATS:
        validate_format(fmt)  # no raise
        assert fmt in FORMAT_MEDIA_TYPES


def test_validate_format_rejects_unknown():
    with pytest.raises(UnsupportedFormatError):
        validate_format("ogg2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_audio_encoding.py -v`
Expected: FAIL with "No module named 'olmlx.utils.audio'".

- [ ] **Step 3: Implement `utils/audio.py`**

Create `olmlx/utils/audio.py`:

```python
"""Audio encoding for /v1/audio/speech (#367).

Converts the float-audio segment stream from mlx-audio into the requested
OpenAI container:

- ``pcm``  : raw signed 16-bit little-endian PCM (no header), streamed.
- ``wav``  : pure-Python RIFF/WAVE (buffered so the header sizes are exact).
- ``mp3``/``opus``/``aac``/``flac`` : piped through ffmpeg, streamed.

ffmpeg is already a hard requirement for the Whisper path.
"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator

import numpy as np

SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {"mp3", "opus", "aac", "flac", "wav", "pcm"}
)

FORMAT_MEDIA_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

# ffmpeg output muxer per format (formats not listed are native-encoded).
_FFMPEG_MUXER: dict[str, str] = {
    "mp3": "mp3",
    "opus": "opus",
    "aac": "adts",
    "flac": "flac",
}


class UnsupportedFormatError(ValueError):
    """Raised for an unknown ``response_format``. Surfaced as HTTP 422."""


def validate_format(fmt: str) -> None:
    if fmt not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported response_format '{fmt}'. "
            f"Must be one of: {', '.join(sorted(SUPPORTED_FORMATS))}."
        )


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert a float waveform in [-1, 1] to signed 16-bit LE PCM bytes."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def wav_bytes(pcm: bytes, *, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw 16-bit PCM in a complete RIFF/WAVE header."""
    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    header = b"RIFF"
    header += struct.pack("<I", 36 + len(pcm))
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
    header += b"data"
    header += struct.pack("<I", len(pcm))
    return header + pcm


async def ffmpeg_encode(
    pcm_chunks: AsyncIterator[bytes], *, fmt: str, sample_rate: int, channels: int = 1
) -> AsyncIterator[bytes]:
    """Stream PCM chunks through ffmpeg, yielding encoded container bytes.

    The ffmpeg subprocess is always reaped — on normal completion, on a feed
    error, and on generator close (client disconnect) — so a dropped stream
    cannot orphan an ffmpeg process.
    """
    muxer = _FFMPEG_MUXER[fmt]
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-loglevel", "error",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels),
        "-i", "pipe:0", "-f", muxer, "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def _feed() -> None:
        try:
            async for chunk in pcm_chunks:
                proc.stdin.write(chunk)
                await proc.stdin.drain()
        finally:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()

    feeder = asyncio.create_task(_feed())
    try:
        while True:
            data = await proc.stdout.read(65536)
            if not data:
                break
            yield data
        await feeder
        await proc.wait()
        if proc.returncode not in (0, None):
            err = (await proc.stderr.read()).decode(errors="replace")
            raise RuntimeError(f"ffmpeg encode failed ({proc.returncode}): {err}")
    finally:
        feeder.cancel()
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_audio_encoding.py -v`
Expected: all PASS.

- [ ] **Step 5: Add an ffmpeg integration test (round-trips real encoding)**

Append to `tests/test_audio_encoding.py`:

```python
@pytest.mark.asyncio
async def test_ffmpeg_encode_mp3_produces_bytes():
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    pcm = float_to_pcm16((0.1 * np.sin(np.arange(24000) / 10.0)).astype(np.float32))

    async def _chunks():
        yield pcm

    from olmlx.utils.audio import ffmpeg_encode

    out = b""
    async for b in ffmpeg_encode(_chunks(), fmt="mp3", sample_rate=24000):
        out += b
    assert len(out) > 0
    # MP3 frame sync (0xFF 0xEx) or an ID3 header.
    assert out[:3] == b"ID3" or out[0] == 0xFF
```

- [ ] **Step 6: Run it**

Run: `uv run pytest tests/test_audio_encoding.py -v`
Expected: PASS (or `test_ffmpeg_encode_mp3_produces_bytes` skipped if no ffmpeg).

- [ ] **Step 7: Commit**

```bash
git add olmlx/utils/audio.py tests/test_audio_encoding.py
git commit -m "feat(tts): audio encoding helpers (pcm/wav/ffmpeg) (#367)"
```

---

## Task 7: Engine `generate_speech`

**Files:**
- Modify: `olmlx/engine/inference.py` (add `generate_speech` near `generate_transcription`, ~line 4965)
- Test: `tests/test_inference_speech.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_speech.py`:

```python
"""generate_speech engine path (#367)."""

import types
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from olmlx.engine.inference import generate_speech


class _Result:
    def __init__(self, audio):
        self.audio = audio
        self.sample_rate = 24000


def _fake_lm(is_tts=True):
    lm = MagicMock()
    lm.is_tts = is_tts
    lm.inference_queue_timeout = 5.0
    lm.sync_mode = "thread"
    lm.name = "kokoro"

    def _gen(text, voice=None, speed=1.0, **kw):
        yield _Result(np.array([0.1, 0.2], dtype=np.float32))
        yield _Result(np.array([0.3], dtype=np.float32))

    lm.model = types.SimpleNamespace(generate=_gen)
    return lm


@pytest.mark.asyncio
async def test_generate_speech_yields_float_segments(monkeypatch):
    lm = _fake_lm()
    manager = MagicMock()
    manager.ensure_loaded = AsyncMock(return_value=lm)
    manager.store = None

    chunks = []
    async for seg in generate_speech(
        manager, "kokoro", "hello world", voice="af_heart", speed=1.0
    ):
        chunks.append(seg)

    assert len(chunks) == 2
    assert np.allclose(chunks[0], [0.1, 0.2])
    assert np.allclose(chunks[1], [0.3])
    lm.release_ref.assert_called_once()


@pytest.mark.asyncio
async def test_generate_speech_rejects_non_tts():
    lm = _fake_lm(is_tts=False)
    manager = MagicMock()
    manager.ensure_loaded = AsyncMock(return_value=lm)
    manager.store = None

    with pytest.raises(ValueError, match="not a TTS model"):
        async for _ in generate_speech(manager, "qwen3", "hi", voice="af_heart"):
            pass
    lm.release_ref.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inference_speech.py -v`
Expected: FAIL with ImportError on `generate_speech`.

- [ ] **Step 3: Implement `generate_speech`**

In `olmlx/engine/inference.py`, after the `generate_transcription` function (ends ~line 4965), add. (Imports `asyncio`, `threading`, `numpy as np`, `_inference_locked`, `_inference_ref`, `surface_var`, `_tracing` are already module-level in `inference.py` — verify and reuse; only add `import numpy as np` / `import threading` at module top if missing.)

```python
async def generate_speech(
    manager: ModelManager,
    model_name: str,
    text: str,
    *,
    voice: str,
    speed: float = 1.0,
    keep_alive: str | None = None,
):
    """Stream TTS audio segments for a managed mlx-audio model (issue #367).

    Async generator yielding 1-D float32 numpy arrays (24 kHz mono), one per
    mlx-audio segment, under the serialized inference lock. mlx-audio's
    ``model.generate`` is a *synchronous* generator; we run the whole loop in
    one worker thread (keeping all MLX work on a single thread, per the #284
    stream hazards) and bridge segments to the event loop via a queue.

    Raises ``ValueError`` (-> HTTP 400) if the model is not a TTS model.
    """
    lm = await manager.ensure_loaded(model_name, keep_alive, pin=True)
    try:
        if not lm.is_tts:
            raise ValueError(
                f"Model '{model_name}' is not a TTS model. "
                "/v1/audio/speech requires a TTS model (e.g. a Kokoro repo "
                "such as prince-canuma/Kokoro-82M)."
            )

        async with _inference_locked(
            lm.inference_queue_timeout, sync_mode=lm.sync_mode
        ):
            with (
                _tracing.span(
                    "inference",
                    model=lm.name,
                    surface=surface_var.get(),
                    strategy="none",
                ),
                _inference_ref(lm, adopt=True),
            ):
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()
                stop = threading.Event()
                sentinel = object()

                def _worker() -> None:
                    try:
                        for result in lm.model.generate(
                            text, voice=voice, speed=speed
                        ):
                            if stop.is_set():
                                break
                            # np.asarray forces eval on THIS thread, keeping
                            # the MLX graph materialization off the loop.
                            audio = np.asarray(result.audio, dtype=np.float32)
                            loop.call_soon_threadsafe(queue.put_nowait, audio)
                        loop.call_soon_threadsafe(queue.put_nowait, sentinel)
                    except Exception as exc:  # noqa: BLE001 - re-raised on loop
                        loop.call_soon_threadsafe(queue.put_nowait, exc)

                worker = asyncio.create_task(asyncio.to_thread(_worker))
                try:
                    while True:
                        item = await queue.get()
                        if item is sentinel:
                            break
                        if isinstance(item, Exception):
                            raise item
                        yield item
                finally:
                    stop.set()
                    await worker
    finally:
        lm.release_ref()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_inference_speech.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/inference.py tests/test_inference_speech.py
git commit -m "feat(tts): generate_speech engine path (#367)"
```

---

## Task 8: Router `POST /v1/audio/speech`

**Files:**
- Modify: `olmlx/routers/audio.py`
- Test: `tests/test_routers_speech.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_routers_speech.py`:

```python
"""Router tests for /v1/audio/speech (#367)."""

from unittest.mock import patch

import numpy as np
import pytest


def _floats():
    async def _gen(*args, **kwargs):
        yield np.array([0.0, 0.1, -0.1], dtype=np.float32)
        yield np.array([0.2], dtype=np.float32)

    return _gen


@pytest.mark.asyncio
async def test_speech_pcm(app_client):
    with patch("olmlx.routers.audio.generate_speech", _floats()):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={"model": "kokoro", "input": "hi", "voice": "alloy",
                  "response_format": "pcm"},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/pcm")
    # 4 samples * 2 bytes
    assert len(resp.content) == 8


@pytest.mark.asyncio
async def test_speech_wav_has_riff(app_client):
    with patch("olmlx.routers.audio.generate_speech", _floats()):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={"model": "kokoro", "input": "hi", "voice": "alloy",
                  "response_format": "wav"},
        )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_speech_unknown_voice_422(app_client):
    resp = await app_client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "hi", "voice": "nope"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_speech_bad_format_422(app_client):
    resp = await app_client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "hi", "voice": "alloy",
              "response_format": "ogg2"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_speech_empty_input_422(app_client):
    resp = await app_client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "", "voice": "alloy"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_speech_non_tts_model_400(app_client):
    async def _raise(*a, **k):
        raise ValueError("Model 'qwen3' is not a TTS model.")
        yield  # pragma: no cover - makes this an async generator

    with patch("olmlx.routers.audio.generate_speech", _raise):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={"model": "qwen3", "input": "hi", "voice": "alloy"},
        )
    assert resp.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_routers_speech.py -v`
Expected: FAIL (404 — route not defined).

- [ ] **Step 3: Implement the route**

In `olmlx/routers/audio.py`, add imports at the top (after the existing imports):

```python
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from olmlx.engine.inference import generate_speech
from olmlx.engine.tts import UnknownVoiceError, resolve_voice
from olmlx.schemas.audio import SpeechRequest
from olmlx.utils import audio as audio_utils
```

Then append the route:

```python
_TTS_SAMPLE_RATE = 24000


@router.post("/v1/audio/speech")
async def create_speech(request: Request, body: SpeechRequest):
    # --- cheap validation before touching the model -----------------------
    if not body.input.strip():
        raise HTTPException(status_code=422, detail="input must not be empty.")
    if len(body.input) > settings.tts_max_input_chars:
        raise HTTPException(
            status_code=413,
            detail=(
                f"input exceeds {settings.tts_max_input_chars} characters "
                "(OLMLX_TTS_MAX_INPUT_CHARS)."
            ),
        )
    try:
        audio_utils.validate_format(body.response_format)
    except audio_utils.UnsupportedFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        kokoro_voice = resolve_voice(body.voice)
    except UnknownVoiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    manager = request.app.state.model_manager
    fmt = body.response_format

    # Float segments -> PCM byte chunks.
    async def _pcm_chunks():
        async for seg in generate_speech(
            manager, body.model, body.input,
            voice=kokoro_voice, speed=body.speed,
        ):
            yield audio_utils.float_to_pcm16(seg)

    media_type = audio_utils.FORMAT_MEDIA_TYPES[fmt]

    # WAV is buffered so the header sizes are exact (small utterances).
    if fmt == "wav":
        pcm = b""
        try:
            async for chunk in _pcm_chunks():
                pcm += chunk
        except ValueError as exc:  # non-TTS model etc.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(
            audio_utils.wav_bytes(pcm, sample_rate=_TTS_SAMPLE_RATE),
            media_type=media_type,
        )

    # PCM streams raw; compressed formats stream through ffmpeg.
    if fmt == "pcm":
        byte_stream = _pcm_chunks()
    else:
        byte_stream = audio_utils.ffmpeg_encode(
            _pcm_chunks(), fmt=fmt, sample_rate=_TTS_SAMPLE_RATE
        )

    # Prime the first chunk so model-load / non-TTS errors (ValueError) become
    # a clean HTTP 400 *before* the 200 streaming response starts.
    agen = byte_stream.__aiter__()
    try:
        first = await agen.__anext__()
    except StopAsyncIteration:
        first = b""
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _body():
        if first:
            yield first
        async for chunk in agen:
            yield chunk

    return StreamingResponse(_body(), media_type=media_type)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_routers_speech.py -v`
Expected: all PASS (the mp3 path in `test_speech_*` uses pcm/wav to avoid requiring ffmpeg; ffmpeg formats are covered in Task 6 and the live test).

- [ ] **Step 5: Verify ForceJSONMiddleware doesn't interfere**

`/v1/audio/speech` takes a JSON body (not multipart). Confirm the existing `ForceJSONMiddleware` leaves JSON requests alone.

Run: `uv run pytest tests/test_routers_speech.py tests/test_routers_audio.py -q`
Expected: PASS (transcription multipart + speech JSON both work).

- [ ] **Step 6: Commit**

```bash
git add olmlx/routers/audio.py tests/test_routers_speech.py
git commit -m "feat(tts): POST /v1/audio/speech route (#367)"
```

---

## Task 9: Live test with real Kokoro

**Files:**
- Create: `tests/live/test_tts_speech.py`

- [ ] **Step 1: Write the live test**

Create `tests/live/test_tts_speech.py`:

```python
"""Live TTS test against real Kokoro via the OpenAI SDK (#367).

real_model; skipped in CI (`-m "not real_model"`) and when the model isn't
present. Drives the OpenAI SDK over the in-process ASGI app (see the live SDK
test pattern). Requires ffmpeg on PATH for the compressed formats.
"""

from __future__ import annotations

import shutil

import pytest

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required"),
]

KOKORO = "prince-canuma/Kokoro-82M"


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["mp3", "wav", "pcm", "opus", "aac", "flac"])
async def test_speech_create_all_formats(openai_client, fmt):
    resp = await openai_client.audio.speech.create(
        model=KOKORO,
        voice="alloy",
        input="Hello from olmlx.",
        response_format=fmt,
    )
    data = resp.content if hasattr(resp, "content") else resp.read()
    assert len(data) > 0


@pytest.mark.asyncio
async def test_speech_streaming_first_chunk(openai_client):
    async with openai_client.audio.speech.with_streaming_response.create(
        model=KOKORO,
        voice="af_heart",
        input="Streaming first chunk latency check.",
        response_format="pcm",
    ) as resp:
        total = 0
        async for chunk in resp.iter_bytes():
            total += len(chunk)
            if total > 0:
                break
        assert total > 0
```

> **Note:** reuse the existing live-test fixtures (`openai_client` / in-process ASGI). If the live suite uses a different fixture name (check `tests/live/test_responses_sdk.py`), match it. If no `openai_client` fixture exists, add one mirroring that file's client setup.

- [ ] **Step 2: Run it (only if Kokoro is available locally)**

Run: `uv run pytest tests/live/test_tts_speech.py -v -m real_model`
Expected: PASS if Kokoro is downloaded and ffmpeg present; otherwise skipped/downloads on first run.

- [ ] **Step 3: Confirm it's excluded from the default suite**

Run: `uv run pytest tests/live/test_tts_speech.py -m "not real_model" -q`
Expected: deselected (0 run).

- [ ] **Step 4: Commit**

```bash
git add tests/live/test_tts_speech.py
git commit -m "test(tts): live Kokoro speech test via OpenAI SDK (#367)"
```

---

## Task 10: Docs + final verification

**Files:**
- Modify: `CLAUDE.md` (Project Structure + a Key Design Decision bullet)

- [ ] **Step 1: Update Project Structure**

In `CLAUDE.md`, under `routers/`, change the `audio.py` line to:

```
│   ├── audio.py        # /v1/audio/transcriptions (Whisper STT) + /v1/audio/speech (Kokoro TTS)
```

- [ ] **Step 2: Add a Key Design Decision bullet**

In `CLAUDE.md`, after the **Audio transcription** bullet, add:

```markdown
- **Text-to-speech** (#367): OpenAI-compatible `POST /v1/audio/speech` via mlx-audio (Kokoro-82M). TTS is a first-class `ModelManager` kind: `_detect_model_kind` matches the Kokoro config signature (`istftnet` + `plbert`; no `model_type`), and `LoadedModel.is_tts` guards the LLM-only prompt-cache / KV-quant paths (mirrors `is_whisper`). `generate_speech` (`engine/inference.py`) drives mlx-audio's synchronous `model.generate` in one worker thread under the serialized lock, bridging float-audio segments (24 kHz mono) to async via a queue (single-thread MLX work, per #284). Voices: the six OpenAI names map to Kokoro ids (`engine/tts.py`), native Kokoro ids pass through, unknown → 422. Output (`utils/audio.py`): `pcm` raw-streamed, `wav` pure-Python buffered (exact header), `mp3`/`opus`/`aac`/`flac` piped through ffmpeg (already required) with guaranteed subprocess reaping. The router primes the first chunk so a non-TTS model surfaces as 400 before the 200 stream begins. Model load: `olmlx models pull prince-canuma/Kokoro-82M`. **Out of scope:** voice cloning/training, the SSE `stream_format` event API, distributed TTS, non-Kokoro TTS models (architecture is generic but only Kokoro is wired/tested in v1).
```

- [ ] **Step 3: Run ruff (per repo convention)**

Run: `uv run ruff check olmlx/ tests/ && uv run ruff format --check olmlx/ tests/`
Expected: clean (fix anything it flags with `uv run ruff format olmlx/ tests/` then re-check).

- [ ] **Step 4: Run the full non-live test suite**

Run: `uv run pytest -m "not real_model" -q`
Expected: PASS (no regressions).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(tts): document /v1/audio/speech endpoint (#367)"
```

- [ ] **Step 6: Push and open the PR**

```bash
git push -u origin feat/tts-speech-endpoint
gh pr create --fill --base main
```

---

## Self-Review notes

- **Spec coverage:** endpoint (Task 8), mlx-audio dep (Task 1), Kokoro model + load + detection (Tasks 2-3), voice map+passthrough+422 (Task 4/8), all six formats via ffmpeg (Task 6), streaming + non-streaming (Tasks 7-8), TTS model-class loadable via pull (Task 3 + docs Task 10), unit + live tests (every task + Task 9). Parakeet explicitly dropped (spec). SSE `stream_format` out of scope (spec).
- **Type/name consistency:** `generate_speech` (engine) ↔ patched in router tests; `resolve_voice`/`UnknownVoiceError`/`KOKORO_VOICES`/`OPENAI_VOICE_MAP` (engine/tts.py) consistent across Tasks 4/8; `float_to_pcm16`/`wav_bytes`/`ffmpeg_encode`/`validate_format`/`FORMAT_MEDIA_TYPES`/`SUPPORTED_FORMATS`/`UnsupportedFormatError` (utils/audio.py) consistent across Tasks 6/8; `is_tts`/`_load_model_tts` consistent across Task 3; `SpeechRequest` fields consistent across Tasks 5/8.
- **Open risk to confirm during execution:** the live-test fixture name (`openai_client`) — match whatever `tests/live/test_responses_sdk.py` uses (Task 9 note). The exact line numbers in `model_manager.py`/`inference.py` are approximate — anchor on the quoted surrounding code, not the numbers.
```
