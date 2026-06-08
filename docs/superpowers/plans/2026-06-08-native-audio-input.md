# Native Audio Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let multimodal LLMs (Gemma 4) accept an audio clip as a first-class chat modality, mirroring the existing VLM image plumbing, across the OpenAI, Ollama, and Anthropic surfaces.

**Architecture:** Audio rides on each message as `msg["audio"]: list[str]` (load-ready strings: URL, path, or `data:` URI), exactly like `msg["images"]`. Routers normalize their content blocks into that list; `engine/inference.py` extracts it, passes `num_audios` to the VLM chat template, materializes any `data:` URIs to temp files (because `mlx_vlm.load_audio` rejects base64/data-URIs), threads the resulting paths to `mlx_vlm.stream_generate(audio=...)`, and deletes the temp files in a `finally`.

**Tech Stack:** Python 3.11, FastAPI, pydantic v2, mlx-vlm ≥0.4.4 (already exposes `audio=` on `generate`/`stream_generate` and ships Gemma 4 audio), pytest. ffmpeg (already required) handles m4a/aac/ogg/opus; miniaudio (transitive via mlx-vlm) handles wav/mp3/flac.

**Spec:** `docs/superpowers/specs/2026-06-08-native-audio-input-design.md`

---

## File Structure

- **Create** `olmlx/utils/audio_input.py` — `normalize_audio_block`, `materialize_audio`, `cleanup_temp_audio`. (Kept separate from `olmlx/utils/audio.py`, which holds TTS *output* helpers — different responsibility.)
- **Create** `tests/test_audio_input_util.py` — unit tests for the three helpers.
- **Modify** `olmlx/engine/inference.py` — add `_extract_audio`, `_audio_capable`; thread `audio` through `_apply_chat_template_vlm`, `generate_chat`, `_stream_completion`, `_full_completion`, `_full_completion_inner`.
- **Modify** `olmlx/utils/streaming.py` — thread `audio` through `async_mlx_stream`.
- **Modify** `olmlx/routers/openai.py` — split `input_audio` parts into `msg["audio"]`.
- **Modify** `olmlx/routers/anthropic.py` — convert `audio` blocks into `msg["audio"]`.
- **Modify** `olmlx/schemas/chat.py` — add `audio` field to the Ollama message model.
- **Modify** `tests/test_routers_openai.py`, `tests/test_inference.py` — router + engine unit tests.
- **Create** `tests/live/test_vlm_audio.py` — real-model end-to-end test.
- **Modify** `pyproject.toml` — bump `mlx-vlm>=0.4.4`.
- **Modify** `CLAUDE.md` — design note.

Branch: `feat/audio-input-426` (already created; the spec is committed there).

---

## Task 1: `normalize_audio_block` — content block → load-ready string

**Files:**
- Create: `olmlx/utils/audio_input.py`
- Test: `tests/test_audio_input_util.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audio_input_util.py`:

```python
import pytest

from olmlx.utils.audio_input import normalize_audio_block


def test_openai_input_audio_builds_data_uri():
    block = {"type": "input_audio", "input_audio": {"data": "QQ==", "format": "wav"}}
    assert normalize_audio_block(block) == "data:audio/wav;base64,QQ=="


def test_openai_input_audio_mp3():
    block = {"type": "input_audio", "input_audio": {"data": "QQ==", "format": "mp3"}}
    assert normalize_audio_block(block) == "data:audio/mp3;base64,QQ=="


def test_openai_input_audio_missing_data_raises():
    with pytest.raises(ValueError, match="input_audio"):
        normalize_audio_block({"type": "input_audio", "input_audio": {"format": "wav"}})


def test_anthropic_audio_base64_builds_data_uri():
    block = {
        "type": "audio",
        "source": {"type": "base64", "media_type": "audio/mpeg", "data": "QQ=="},
    }
    assert normalize_audio_block(block) == "data:audio/mpeg;base64,QQ=="


def test_anthropic_audio_base64_defaults_media_type():
    block = {"type": "audio", "source": {"type": "base64", "data": "QQ=="}}
    assert normalize_audio_block(block) == "data:audio/wav;base64,QQ=="


def test_anthropic_audio_url_source():
    block = {"type": "audio", "source": {"type": "url", "url": "http://x/a.wav"}}
    assert normalize_audio_block(block) == "http://x/a.wav"


def test_anthropic_audio_unsupported_source_raises():
    block = {"type": "audio", "source": {"type": "file", "id": "abc"}}
    with pytest.raises(ValueError, match="unsupported audio source"):
        normalize_audio_block(block)


def test_not_an_audio_block_raises():
    with pytest.raises(ValueError, match="not an audio block"):
        normalize_audio_block({"type": "text", "text": "hi"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_audio_input_util.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.utils.audio_input'`

- [ ] **Step 3: Implement `normalize_audio_block`**

Create `olmlx/utils/audio_input.py`:

```python
"""Normalize and materialize audio references from API surfaces (#426).

``mlx_vlm.utils.load_audio`` accepts file paths, http(s) URLs, and numpy
arrays — but NOT base64 or ``data:`` URIs.  This module converts the OpenAI
(``input_audio``) and Anthropic (``audio`` + ``source``) content-block shapes
into one load-ready string (URL / path / ``data:`` URI), and separately
materializes ``data:`` URIs into temp files just before generation.
"""

from __future__ import annotations

from typing import Any


def normalize_audio_block(block: dict[str, Any]) -> str:
    """Convert an OpenAI ``input_audio`` or Anthropic ``audio`` content block to
    a string the engine can load (URL, path, or ``data:`` URI).

    Raises ``ValueError`` for missing fields, unsupported source types, or a
    non-audio block.
    """
    btype = block.get("type")

    # OpenAI: {"type": "input_audio", "input_audio": {"data": "<b64>", "format": "wav"}}
    if btype == "input_audio":
        spec = block.get("input_audio") or {}
        data = spec.get("data")
        if not isinstance(data, str) or not data:
            raise ValueError("input_audio block missing input_audio.data")
        fmt = spec.get("format") or "wav"
        return f"data:audio/{fmt};base64,{data}"

    # Anthropic / internal: {"type": "audio", "source": {...}}  (olmlx extension)
    if btype == "audio":
        source = block.get("source") or {}
        stype = source.get("type")
        if stype == "url":
            url = source.get("url")
            if not isinstance(url, str) or not url:
                raise ValueError("audio source type=url missing 'url'")
            return url
        if stype == "base64":
            data = source.get("data")
            if not isinstance(data, str) or not data:
                raise ValueError("audio source type=base64 missing 'data'")
            media_type = source.get("media_type") or "audio/wav"
            return f"data:{media_type};base64,{data}"
        raise ValueError(f"unsupported audio source type: {stype!r}")

    raise ValueError(f"not an audio block: type={btype!r}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_audio_input_util.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/utils/audio_input.py tests/test_audio_input_util.py
git commit -m "feat(audio): normalize_audio_block for OpenAI/Anthropic audio blocks (#426)"
```

---

## Task 2: `materialize_audio` + `cleanup_temp_audio` — data-URI → temp file

**Files:**
- Modify: `olmlx/utils/audio_input.py`
- Test: `tests/test_audio_input_util.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_audio_input_util.py`:

```python
import base64
import os

from olmlx.utils.audio_input import cleanup_temp_audio, materialize_audio


def test_materialize_passthrough_url_and_path():
    paths, temps = materialize_audio(["http://x/a.wav", "/tmp/local.mp3"])
    assert paths == ["http://x/a.wav", "/tmp/local.mp3"]
    assert temps == []


def test_materialize_data_uri_writes_temp_file_with_suffix():
    raw = b"RIFFblah"
    uri = "data:audio/wav;base64," + base64.b64encode(raw).decode()
    paths, temps = materialize_audio([uri])
    try:
        assert len(paths) == 1 and len(temps) == 1
        assert paths == temps
        assert paths[0].endswith(".wav")
        with open(paths[0], "rb") as fh:
            assert fh.read() == raw
    finally:
        cleanup_temp_audio(temps)
    assert not os.path.exists(temps[0])


def test_materialize_mpeg_maps_to_mp3_suffix():
    uri = "data:audio/mpeg;base64," + base64.b64encode(b"x").decode()
    paths, temps = materialize_audio([uri])
    try:
        assert paths[0].endswith(".mp3")
    finally:
        cleanup_temp_audio(temps)


def test_cleanup_is_idempotent_and_swallows_missing():
    cleanup_temp_audio(["/nonexistent/abc.wav"])  # must not raise


def test_materialize_none_returns_empty():
    paths, temps = materialize_audio(None)
    assert paths == [] and temps == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_audio_input_util.py -k materialize -v`
Expected: FAIL with `ImportError: cannot import name 'materialize_audio'`

- [ ] **Step 3: Implement materialization helpers**

Append to `olmlx/utils/audio_input.py`:

```python
import base64
import binascii
import contextlib
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# Map a ``data:`` URI subtype to a file suffix.  Extension matters: mlx_vlm's
# read_audio routes m4a/aac/ogg/opus to ffmpeg by suffix, and others to
# miniaudio.  Unmapped subtypes fall back to ``.<subtype>``.
_SUFFIX_BY_SUBTYPE = {
    "mpeg": ".mp3",
    "mp3": ".mp3",
    "wav": ".wav",
    "x-wav": ".wav",
    "mp4": ".m4a",
    "x-m4a": ".m4a",
    "m4a": ".m4a",
    "aac": ".aac",
    "ogg": ".ogg",
    "opus": ".opus",
    "flac": ".flac",
    "webm": ".webm",
}


def _suffix_for_media_type(media_type: str) -> str:
    subtype = media_type.split("/", 1)[-1].lower() if "/" in media_type else "wav"
    return _SUFFIX_BY_SUBTYPE.get(subtype, f".{subtype}")


def materialize_audio(
    audio: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve audio references to loadable paths.

    ``data:`` URIs are base64-decoded into temp files (mlx_vlm's ``load_audio``
    cannot read data URIs); URLs and filesystem paths pass through unchanged.

    Returns ``(paths, temp_paths)`` where ``paths`` is what to hand to
    ``mlx_vlm`` and ``temp_paths`` ⊆ ``paths`` must be deleted by the caller
    via ``cleanup_temp_audio`` once generation is done.
    """
    paths: list[str] = []
    temp_paths: list[str] = []
    for item in audio or []:
        if isinstance(item, str) and item.startswith("data:"):
            header, _, b64 = item.partition(",")
            media_type = header[len("data:") :].split(";", 1)[0] or "audio/wav"
            try:
                raw = base64.b64decode(b64, validate=False)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"invalid base64 audio data: {exc}") from exc
            fd, tmp = tempfile.mkstemp(
                suffix=_suffix_for_media_type(media_type), prefix="olmlx-audio-"
            )
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            paths.append(tmp)
            temp_paths.append(tmp)
        else:
            paths.append(item)
    return paths, temp_paths


def cleanup_temp_audio(temp_paths: list[str] | None) -> None:
    """Delete materialized temp audio files; never raises."""
    for path in temp_paths or []:
        with contextlib.suppress(OSError):
            os.unlink(path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_audio_input_util.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/utils/audio_input.py tests/test_audio_input_util.py
git commit -m "feat(audio): materialize data-URI audio to temp files (#426)"
```

---

## Task 3: `_extract_audio` in the engine

**Files:**
- Modify: `olmlx/engine/inference.py` (add after `_extract_images`, currently ending at line 2309)
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_inference.py`:

```python
def test_extract_audio_collects_from_messages():
    from olmlx.engine.inference import _extract_audio

    msgs = [
        {"role": "user", "content": "hi", "audio": ["a.wav"]},
        {"role": "user", "content": "x", "audio": ["b.mp3", "c.flac"]},
        {"role": "user", "content": "no audio"},
    ]
    assert _extract_audio(msgs) == ["a.wav", "b.mp3", "c.flac"]


def test_extract_audio_returns_none_when_absent():
    from olmlx.engine.inference import _extract_audio

    assert _extract_audio([{"role": "user", "content": "hi"}]) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_inference.py -k extract_audio -v`
Expected: FAIL with `ImportError: cannot import name '_extract_audio'`

- [ ] **Step 3: Implement `_extract_audio`**

In `olmlx/engine/inference.py`, immediately after `_extract_images` (after line 2309), add:

```python
def _extract_audio(messages: list[dict]) -> list[str] | None:
    """Extract audio refs (URLs/paths/data-URIs) from message content (#426)."""
    audio: list[str] = []
    for msg in messages:
        if msg.get("audio"):
            audio.extend(msg["audio"])
    return audio if audio else None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_inference.py -k extract_audio -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/inference.py tests/test_inference.py
git commit -m "feat(audio): _extract_audio engine helper (#426)"
```

---

## Task 4: `_audio_capable` capability probe

**Files:**
- Modify: `olmlx/engine/inference.py` (add near `_extract_audio`)
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_inference.py`:

```python
def test_audio_capable_true_for_vlm_with_feature_extractor():
    from types import SimpleNamespace

    from olmlx.engine.inference import _audio_capable

    proc = SimpleNamespace(feature_extractor=object())
    lm = SimpleNamespace(is_vlm=True, tokenizer=proc)
    assert _audio_capable(lm) is True


def test_audio_capable_false_for_vlm_without_feature_extractor():
    from types import SimpleNamespace

    from olmlx.engine.inference import _audio_capable

    proc = SimpleNamespace(feature_extractor=None)
    lm = SimpleNamespace(is_vlm=True, tokenizer=proc)
    assert _audio_capable(lm) is False


def test_audio_capable_false_for_text_model():
    from types import SimpleNamespace

    from olmlx.engine.inference import _audio_capable

    lm = SimpleNamespace(is_vlm=False, tokenizer=object())
    assert _audio_capable(lm) is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_inference.py -k audio_capable -v`
Expected: FAIL with `ImportError: cannot import name '_audio_capable'`

- [ ] **Step 3: Implement `_audio_capable`**

In `olmlx/engine/inference.py`, immediately after `_extract_audio`, add:

```python
def _audio_capable(lm: Any) -> bool:
    """True when the loaded model can accept audio input.

    Audio models are VLMs whose mlx-vlm processor wires a ``feature_extractor``
    (Gemma 4, gemma3n, qwen3_omni_moe, phi4mm, minicpmo).  Read from the
    already-loaded processor — no new ModelManager kind needed.
    """
    if not getattr(lm, "is_vlm", False):
        return False
    return getattr(lm.tokenizer, "feature_extractor", None) is not None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_inference.py -k audio_capable -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/inference.py tests/test_inference.py
git commit -m "feat(audio): _audio_capable probe (#426)"
```

---

## Task 5: `_apply_chat_template_vlm` accepts `audio` (num_audios) + rejects tools+audio

**Files:**
- Modify: `olmlx/engine/inference.py:2173-2249` (`_apply_chat_template_vlm`)
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_inference.py`:

```python
def test_apply_chat_template_vlm_passes_num_audios(monkeypatch):
    import mlx_vlm

    from olmlx.engine import inference as inf

    captured = {}

    def fake_apply(processor, config, messages, num_images=0, num_audios=0, **kw):
        captured["num_images"] = num_images
        captured["num_audios"] = num_audios
        return "PROMPT"

    monkeypatch.setattr(mlx_vlm, "apply_chat_template", fake_apply)

    class _M:
        config = {}

    out = inf._apply_chat_template_vlm(
        processor=object(),
        model=_M(),
        messages=[{"role": "user", "content": "hi"}],
        images=None,
        audio=["a.wav", "b.wav"],
    )
    assert out == "PROMPT"
    assert captured == {"num_images": 0, "num_audios": 2}


def test_apply_chat_template_vlm_rejects_tools_with_audio():
    from olmlx.engine import inference as inf

    with pytest.raises(ValueError, match="tools.*audio|audio.*tools"):
        inf._apply_chat_template_vlm(
            processor=object(),
            model=object(),
            messages=[{"role": "user", "content": "hi"}],
            images=None,
            audio=["a.wav"],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_inference.py -k apply_chat_template_vlm -v`
Expected: FAIL — `_apply_chat_template_vlm() got an unexpected keyword argument 'audio'`

- [ ] **Step 3: Add the `audio` parameter and wiring**

In `olmlx/engine/inference.py`, change the signature of `_apply_chat_template_vlm` (line 2173-2180) to add `audio`:

```python
def _apply_chat_template_vlm(
    processor: Any,
    model: Any,
    messages: list[dict],
    images: list[str] | None = None,
    tools: list[dict] | None = None,
    enable_thinking: bool | None = None,
    audio: list[str] | None = None,
) -> str:
```

Immediately after the docstring (before `if tools:` at line 2188), add the rejection guard:

```python
    if audio and tools:
        raise ValueError(
            "tools + audio is not supported in this version: combining native "
            "tool calling with audio input is out of scope (#426). Send the "
            "audio without tools, or the tools without audio."
        )
```

In the no-tools branch, change the `num_images` computation (line 2232) and the `mlx_vlm.apply_chat_template` call (line 2241-2243) to pass `num_audios`:

```python
    config = model.config if hasattr(model, "config") else {}
    num_images = len(images) if images else 0
    num_audios = len(audio) if audio else 0
    # mlx_vlm.apply_chat_template forwards **kwargs to the tokenizer's
    # apply_chat_template, so enable_thinking reaches the Jinja template
    # (templates that don't declare the variable ignore it).  Only forward
    # when explicitly set so the template's own default is preserved otherwise.
    extra_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        extra_kwargs["enable_thinking"] = enable_thinking
    # Pass the full message list so the model gets proper conversation context
    result = mlx_vlm.apply_chat_template(
        processor,
        config,
        messages,
        num_images=num_images,
        num_audios=num_audios,
        **extra_kwargs,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_inference.py -k apply_chat_template_vlm -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/inference.py tests/test_inference.py
git commit -m "feat(audio): thread num_audios through VLM chat template; reject tools+audio (#426)"
```

---

## Task 6: Thread `audio` through `async_mlx_stream`

**Files:**
- Modify: `olmlx/utils/streaming.py:660-708` (`async_mlx_stream`)
- Test: `tests/test_streaming.py` (create if absent)

- [ ] **Step 1: Write the failing test**

Create or append to `tests/test_streaming.py`:

```python
def test_async_mlx_stream_forwards_audio_to_mlx_vlm(monkeypatch):
    import mlx_vlm

    from olmlx.utils import streaming

    captured = {}

    def fake_stream_generate(model, tokenizer, **kwargs):
        captured.update(kwargs)
        return iter(())

    monkeypatch.setattr(mlx_vlm, "stream_generate", fake_stream_generate)

    stream = streaming.async_mlx_stream(
        model=object(),
        tokenizer=object(),
        prompt="hi",
        max_tokens=4,
        is_vlm=True,
        images=["i.png"],
        audio=["a.wav"],
    )
    # Drain the worker so gen_factory runs.
    stream.close()
    assert captured.get("audio") == ["a.wav"]
    assert captured.get("image") == ["i.png"]
```

> Note: if `stream.close()` is not sufficient to force `gen_factory` execution in this harness, drive one iteration instead (see how existing `tests/test_streaming.py` cases start/iterate a `CancellableStream`); the assertion on `captured` is the point.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_streaming.py -k forwards_audio -v`
Expected: FAIL — `async_mlx_stream() got an unexpected keyword argument 'audio'`

- [ ] **Step 3: Add `audio` to `async_mlx_stream`**

In `olmlx/utils/streaming.py`, add `audio` to the signature (after `images` at line 666):

```python
def async_mlx_stream(
    model: Any,
    tokenizer: Any,
    prompt: str | list[int],
    max_tokens: int = 512,
    is_vlm: bool = False,
    images: list[str] | None = None,
    audio: list[str] | None = None,
    memory_limit: int = 0,
    trace_context: Any = None,
    **kwargs: Any,
) -> CancellableStream:
```

In the `is_vlm` branch of `gen_factory` (line 701-708), pass `audio`:

```python
        if is_vlm:
            import mlx_vlm

            return mlx_vlm.stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                image=images,
                audio=audio,
                max_tokens=max_tokens,
                **kwargs,
            )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_streaming.py -k forwards_audio -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/utils/streaming.py tests/test_streaming.py
git commit -m "feat(audio): thread audio through async_mlx_stream (#426)"
```

---

## Task 7: Thread `audio` through `_stream_completion` (materialize + cleanup)

**Files:**
- Modify: `olmlx/engine/inference.py:3471-3488` (signature), `:3710-3720` (the `async_mlx_stream` call)
- Test: covered indirectly by the live test (Task 14); this is mechanical plumbing.

- [ ] **Step 1: Add `audio` to the `_stream_completion` signature**

In `olmlx/engine/inference.py`, add `audio` after `images` (line 3477):

```python
async def _stream_completion(
    lm: LoadedModel,
    prompt: str | list[int],
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
    audio: list[str] | None = None,
    *,
    use_prompt_cache: bool = False,
    prompt_tokens: list[int] | None = None,
    cache_id: str = "",
    keep_alive: str | None = None,
    grammar_active: bool = False,
    adopt_pin: bool = False,
    messages: list[dict] | None = None,
    tokenizer: Any = None,
    template_kwargs: dict | None = None,
) -> AsyncGenerator[dict, None]:
```

- [ ] **Step 2: Materialize audio and add a cleanup `finally` around the stream**

Find the function's main `try:` (the explicit acquire/release block beginning around line 3489). At the top of that `try`, before streaming starts, materialize:

```python
        # Materialize data-URI audio to temp files just before generation;
        # mlx_vlm.load_audio cannot read data URIs (#426).  Cleaned up in the
        # finally that releases the inference lock, after the stream drains.
        from olmlx.utils.audio_input import cleanup_temp_audio, materialize_audio

        audio_paths, _audio_temps = materialize_audio(audio)
```

In the existing `finally` block of `_stream_completion` (the one that releases the inference lock), add as the first statement:

```python
            cleanup_temp_audio(_audio_temps)
```

Then change the `async_mlx_stream(...)` call (line 3710-3720) to pass `audio=audio_paths`:

```python
                stream = async_mlx_stream(
                    lm.model,
                    lm.tokenizer,
                    prompt,
                    max_tokens=max_tokens,
                    is_vlm=lm.is_vlm,
                    images=images,
                    audio=audio_paths,
                    memory_limit=memory_limit,
                    trace_context=_tracing.current_context(),
                    **gen_kwargs,
                )
```

> If `_stream_completion` has no single function-level `finally`, wrap the streaming body in `try: ... finally: cleanup_temp_audio(_audio_temps)`. The temp files must outlive the fully-drained stream, so the cleanup must be in the generator's terminal `finally`, not inline after `async_mlx_stream(...)`.

- [ ] **Step 3: Run the regression suite for streaming/inference**

Run: `uv run pytest tests/test_inference.py tests/test_streaming.py -q`
Expected: PASS (no regressions; existing image/text tests still green)

- [ ] **Step 4: Commit**

```bash
git add olmlx/engine/inference.py
git commit -m "feat(audio): materialize + thread audio through _stream_completion (#426)"
```

---

## Task 8: Thread `audio` through `_full_completion` / `_full_completion_inner`

**Files:**
- Modify: `olmlx/engine/inference.py:3976-3982` (`_full_completion` signature), `:4143-4149` (`_full_completion_inner` signature), `:4181-4258` (the two `mlx_vlm.stream_generate` calls), `:4069` (`_full_completion` → `_full_completion_inner` call)

- [ ] **Step 1: Add `audio` to both signatures**

`_full_completion` (line 3976-3982) — add `audio` after `images`:

```python
async def _full_completion(
    lm: LoadedModel,
    prompt: str | list[int],
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
    audio: list[str] | None = None,
    *,
```

`_full_completion_inner` (line 4143-4149) — add `audio` after `images`:

```python
async def _full_completion_inner(
    lm: LoadedModel,
    prompt: str | list[int],
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
    audio: list[str] | None = None,
    *,
```

- [ ] **Step 2: Forward `audio` from `_full_completion` to `_full_completion_inner`**

At the `_full_completion_inner(...)` call inside `_full_completion` (around line 4069, where `images` is passed positionally), pass `audio` in the same position. Confirm argument order matches the new signature (images then audio). Example:

```python
                    result = await _full_completion_inner(
                        lm,
                        prompt,
                        max_tokens,
                        gen_kwargs,
                        stats,
                        images,
                        audio,
                        ...
                    )
```

- [ ] **Step 3: Materialize audio and pass to both stream_generate calls**

At the top of `_full_completion_inner` (before the `if lm.is_vlm and images:` branch at line 4181), materialize:

```python
        from olmlx.utils.audio_input import cleanup_temp_audio, materialize_audio

        audio_paths, _audio_temps = materialize_audio(audio)
```

Change the VLM branch condition and both `mlx_vlm.stream_generate` calls so audio also triggers the VLM path and is forwarded. Update line 4181:

```python
        if lm.is_vlm and (images or audio_paths):
```

Update the first `stream_generate` call (line 4195-4202) to add `audio=audio_paths`:

```python
            for response in mlx_vlm.stream_generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                image=images,
                audio=audio_paths,
                max_tokens=max_tokens,
                **gen_kwargs,
            ):
```

Update the second VLM `stream_generate` call (the `elif lm.is_vlm:` branch, line 4247-4253) likewise:

```python
            for response in mlx_vlm.stream_generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                image=images,
                audio=audio_paths,
                max_tokens=max_tokens,
                **gen_kwargs,
            ):
```

Wrap the generation body of `_full_completion_inner` so temp files are cleaned up after the call returns or raises. The cleanest is to wrap the whole body in `try: ... finally: cleanup_temp_audio(_audio_temps)`. If the function already has a top-level `try/finally` (for the inference lock), add `cleanup_temp_audio(_audio_temps)` to that `finally`.

> Also update the speculative-skip guard at line 4171 so audio requests don't attempt speculative decoding: change `not (lm.is_vlm and images)` to `not (lm.is_vlm and (images or audio))`. Use the descriptor list `audio` here (not `audio_paths`) — this guard sits *above* the materialize line, so `audio_paths` is not yet defined; presence is identical either way. (Audio rides the VLM path, which is exempt from speculative — same rule as images.)

- [ ] **Step 4: Run the regression suite**

Run: `uv run pytest tests/test_inference.py -q`
Expected: PASS (no regressions)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/inference.py
git commit -m "feat(audio): thread audio through _full_completion path (#426)"
```

---

## Task 9: Wire `generate_chat` — extract, capability-check, template, downstream calls

**Files:**
- Modify: `olmlx/engine/inference.py:4483` (extract), `:4512-4552` (`_apply_chat_template_vlm` calls), `:4665-4681` and `:4702-4709` (downstream calls)
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing test (capability rejection)**

Append to `tests/test_inference.py`:

```python
@pytest.mark.asyncio
async def test_generate_chat_rejects_audio_on_non_audio_model(monkeypatch):
    """Audio sent to a model that can't accept it -> ValueError (router -> 422)."""
    from types import SimpleNamespace

    from olmlx.engine import inference as inf

    # Text model: not a VLM, no feature_extractor.
    lm = SimpleNamespace(is_vlm=False, tokenizer=object(), name="text-model")

    class _Manager:
        async def ensure_loaded(self, *a, **k):
            return lm

    msgs = [{"role": "user", "content": "what is this?", "audio": ["a.wav"]}]
    with pytest.raises(ValueError, match="cannot accept audio"):
        await inf.generate_chat(_Manager(), "text-model", msgs, {})
```

> If `generate_chat`'s `ensure_loaded` signature or manager interface differs, mirror the existing `generate_chat` tests in `tests/test_inference.py` for the exact fake-manager shape. The assertion — a `ValueError` containing "cannot accept audio" — is the contract.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_inference.py -k rejects_audio_on_non_audio -v`
Expected: FAIL (no rejection raised; or AttributeError)

- [ ] **Step 3: Extract audio + capability check in `generate_chat`**

In `generate_chat`, right after `images = _extract_images(messages)` (line 4483), add:

```python
        audio = _extract_audio(messages)
        if audio and not _audio_capable(lm):
            raise ValueError(
                f"Model {lm.name!r} cannot accept audio input: it is not an "
                "audio-capable multimodal model. Load a model with an audio "
                "tower (e.g. a Gemma 4 checkpoint)."
            )
```

> Place this after `lm` is bound (the loaded model). If `_extract_images` runs before `lm` exists in this function, move only the capability check to just after `lm = await ...ensure_loaded(...)`; keep `audio = _extract_audio(messages)` next to the image extraction. Verify `lm` is in scope at the chosen line.

- [ ] **Step 4: Pass `audio` to the three `_apply_chat_template_vlm` calls**

In the `if lm.is_vlm:` block (lines 4521-4552), add `audio=audio` to each `_apply_chat_template_vlm(...)` call. The tools branches will raise via Task 5's guard if audio is present (correct — tools+audio is out of scope). For the no-tools branch (line 4546-4552):

```python
            else:
                prompt = _apply_chat_template_vlm(
                    lm.tokenizer,
                    lm.model,
                    messages,
                    images,
                    enable_thinking=vlm_thinking,
                    audio=audio,
                )
```

And add `audio=audio` to the two tools branches (lines 4522-4529 and 4535-4541) as well, so a tools+audio request raises the clear error rather than silently dropping audio.

- [ ] **Step 5: Pass `audio` to `_stream_completion` and `_full_completion`**

At the `_stream_completion(...)` call (line 4665-4681), add `audio` after the positional `images` (line 4671):

```python
            gen = _stream_completion(
                lm,
                prompt,
                mt,
                gen_kwargs,
                stats,
                images,
                audio,
                use_prompt_cache=use_prompt_cache,
                ...
            )
```

At the `_full_completion(...)` call (line 4702-4709), add `audio` after `images` (line 4708):

```python
                    result = await _full_completion(
                        lm,
                        prompt,
                        mt,
                        gen_kwargs,
                        stats,
                        images,
                        audio,
                        has_tools=bool(tools),
                        ...
                    )
```

- [ ] **Step 6: Run the test + full inference suite**

Run: `uv run pytest tests/test_inference.py -q`
Expected: PASS (rejection test green, no regressions)

- [ ] **Step 7: Commit**

```bash
git add olmlx/engine/inference.py tests/test_inference.py
git commit -m "feat(audio): wire audio extraction, capability check, and threading into generate_chat (#426)"
```

---

## Task 10: OpenAI router — split `input_audio` parts into `msg["audio"]`

**Files:**
- Modify: `olmlx/routers/openai.py:296-326` (`_normalize_multimodal_messages`), and the import at line 23
- Test: `tests/test_routers_openai.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_routers_openai.py`:

```python
def test_normalize_splits_input_audio_into_audio_list():
    from olmlx.routers.openai import _normalize_multimodal_messages

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is said?"},
                {"type": "input_audio", "input_audio": {"data": "QQ==", "format": "wav"}},
            ],
        }
    ]
    out = _normalize_multimodal_messages(messages)
    assert out[0]["content"] == "what is said?"
    assert out[0]["audio"] == ["data:audio/wav;base64,QQ=="]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_routers_openai.py -k input_audio -v`
Expected: FAIL — `KeyError: 'audio'` (audio not split out)

- [ ] **Step 3: Handle `input_audio` in the normalizer**

In `olmlx/routers/openai.py`, update the import at line 23:

```python
from olmlx.utils.audio_input import normalize_audio_block
from olmlx.utils.images import normalize_image_block
```

In `_normalize_multimodal_messages` (lines 309-325), add an `audio` accumulator and an `input_audio` branch:

```python
        texts: list[str] = []
        images: list[str] = []
        audio: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            # "input_text"/"input_image" are the Responses-API / newer-SDK
            # spellings of the Chat-Completions "text"/"image_url" parts.
            if ptype in ("text", "input_text"):
                text = part.get("text") or ""
                if text:
                    texts.append(text)
            elif ptype == "image_url":
                images.append(normalize_image_block(part))
            elif ptype == "input_audio":
                audio.append(normalize_audio_block(part))
        m["content"] = " ".join(texts)
        if images:
            m["images"] = (m.get("images") or []) + images
        if audio:
            m["audio"] = (m.get("audio") or []) + audio
    return messages
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_routers_openai.py -k input_audio -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/routers/openai.py tests/test_routers_openai.py
git commit -m "feat(audio): OpenAI input_audio content parts -> msg audio (#426)"
```

---

## Task 11: Anthropic router — convert `audio` blocks into `msg["audio"]`

**Files:**
- Modify: `olmlx/routers/anthropic.py:20` (import), `:220-252` (`_convert_messages` user branch)
- Test: `tests/test_routers_anthropic.py` (create if absent)

- [ ] **Step 1: Write the failing test**

Create or append to `tests/test_routers_anthropic.py`:

```python
def test_convert_messages_collects_audio_block():
    from olmlx.routers.anthropic import _convert_messages
    from olmlx.schemas.anthropic import AnthropicMessagesRequest

    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=16,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe"},
                    {
                        "type": "audio",
                        "source": {
                            "type": "base64",
                            "media_type": "audio/wav",
                            "data": "QQ==",
                        },
                    },
                ],
            }
        ],
    )
    msgs = _convert_messages(req)
    user = [m for m in msgs if m["role"] == "user"][0]
    assert user["audio"] == ["data:audio/wav;base64,QQ=="]
```

> If `AnthropicContentBlock` rejects `type: "audio"` at validation, relax its `type` field to accept the string (it is already `type: str`); confirm by running the test. No new Pydantic model is needed.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_routers_anthropic.py -k audio_block -v`
Expected: FAIL — `KeyError: 'audio'`

- [ ] **Step 3: Handle the `audio` block in `_convert_messages`**

In `olmlx/routers/anthropic.py`, update the import at line 20:

```python
from olmlx.utils.audio_input import normalize_audio_block
from olmlx.utils.images import normalize_image_block
```

In the user branch (lines 220-252), add an `audio` accumulator and block handler, and attach it to the user message:

```python
        elif msg.role == "user":
            text_parts = []
            user_images: list[str] = []
            user_audio: list[str] = []
            for block in msg.content:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
                elif block.type == "image":
                    user_images.append(
                        normalize_image_block(
                            {"type": "image", "source": block.source or {}}
                        )
                    )
                elif block.type == "audio":
                    user_audio.append(
                        normalize_audio_block(
                            {"type": "audio", "source": block.source or {}}
                        )
                    )
                elif block.type == "tool_result":
                    result_content = ""
                    if isinstance(block.content, str):
                        result_content = block.content
                    elif isinstance(block.content, list):
                        result_content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in block.content
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id or "",
                            "content": result_content,
                        }
                    )
            if text_parts or user_images or user_audio:
                user_msg: dict = {"role": "user", "content": " ".join(text_parts)}
                if user_images:
                    user_msg["images"] = user_images
                if user_audio:
                    user_msg["audio"] = user_audio
                messages.append(user_msg)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_routers_anthropic.py -k audio_block -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/routers/anthropic.py tests/test_routers_anthropic.py
git commit -m "feat(audio): Anthropic audio content block -> msg audio (olmlx extension) (#426)"
```

---

## Task 12: Ollama `/api/chat` — add `audio` field to the message schema

**Files:**
- Modify: `olmlx/schemas/chat.py:21` (add `audio` next to `images`)
- Test: `tests/test_routers_chat.py` (create if absent)

- [ ] **Step 1: Write the failing test**

Create or append to `tests/test_routers_chat.py`:

```python
def test_chat_message_accepts_audio_field_and_dumps_it():
    from olmlx.schemas.chat import ChatMessage

    m = ChatMessage(role="user", content="hi", audio=["a.wav"])
    dumped = m.model_dump(exclude_none=True)
    assert dumped["audio"] == ["a.wav"]
```

> Use the actual message class name in `olmlx/schemas/chat.py` (the one carrying `images: list[str] | None` at line 21). If it is not `ChatMessage`, adjust the import.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_routers_chat.py -k audio_field -v`
Expected: FAIL — `ValidationError` / unexpected field `audio`

- [ ] **Step 3: Add the `audio` field**

In `olmlx/schemas/chat.py`, directly below the `images` field (line 21), add:

```python
    images: list[str] | None = None
    audio: list[str] | None = None
```

(No router code change needed: `chat.py` already does `m.model_dump(exclude_none=True)`, so an `audio` field flows into the message dict that `_extract_audio` reads.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_routers_chat.py -k audio_field -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/schemas/chat.py tests/test_routers_chat.py
git commit -m "feat(audio): Ollama /api/chat message accepts audio field (#426)"
```

---

## Task 13: Dependency floor + docs

**Files:**
- Modify: `pyproject.toml` (mlx-vlm floor), `CLAUDE.md` (design note)

- [ ] **Step 1: Bump the mlx-vlm floor**

In `pyproject.toml`, change `"mlx-vlm>=0.4.3",` to:

```toml
    "mlx-vlm>=0.4.4",
```

- [ ] **Step 2: Re-lock and verify the environment still resolves**

Run: `uv sync --no-editable`
Expected: resolves with `mlx-vlm==0.4.4` (already installed); no errors.

- [ ] **Step 3: Add a CLAUDE.md design note**

In `CLAUDE.md`, after the VLM-related bullets (near the `**VLM tools + images (#428)**` / `**VLM prompt caching (#429)**` entries), add:

```markdown
- **Native audio input** (#426): Encoder-free multimodal LLMs (Gemma 4) accept an audio clip as a first-class chat modality, mirroring the image path. Audio rides on each message as `msg["audio"]: list[str]` (URL/path/data-URI), normalized by `utils/audio_input.py:normalize_audio_block` from OpenAI `input_audio`, the Ollama `audio` message field, and an Anthropic `audio` content block (an **olmlx extension** — Anthropic's API has no audio-input block). `engine/inference.py:_extract_audio` collects it; `_apply_chat_template_vlm` passes `num_audios` so the Gemma 4 processor expands its own `<audio>` placeholders (no hand-injected markers); audio threads through to `mlx_vlm.stream_generate(audio=…)`. **Base64 ≠ images:** `mlx_vlm.load_audio` rejects base64/data-URIs, so `materialize_audio` decodes `data:` URIs to temp files (suffix by media-type — extension routes m4a/aac/ogg/opus to ffmpeg, others to miniaudio) just before generation, deleted in the stream/full-completion `finally`. Capability gate: `_audio_capable(lm)` = VLM whose processor exposes a `feature_extractor`; audio to a non-audio model → `ValueError` → 422. **Out of scope (v1):** tools + audio (rejected at template time), streaming audio input, diarization, speculative/distributed audio, non-VLM text path. Other audio VLMs (gemma3n, qwen3_omni_moe, phi4mm, minicpmo) ride the same generic path but only Gemma 4 is wired/tested. Live coverage: `tests/live/test_vlm_audio.py` (real_model). **Requires ffmpeg on PATH** for m4a/aac/ogg/opus.
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock CLAUDE.md
git commit -m "docs(audio): bump mlx-vlm floor to 0.4.4; CLAUDE.md note (#426)"
```

---

## Task 14: Live end-to-end test (real model)

**Files:**
- Create: `tests/live/test_vlm_audio.py`

This test runs only on a machine with `mlx-community/gemma-4-e2b-it-4bit` downloaded; it is skipped in CI (`-m "not real_model"`) and skipped when the model is absent. It mirrors `tests/live/test_vlm_tools_images.py`.

- [ ] **Step 1: Generate a tiny WAV fixture in-test**

We synthesize a short spoken-word-free tone WAV in-process (no asset checked in). The assertion is that the model produces a coherent answer that references audio at all (not that it transcribes speech — a pure tone has none). For a content-grounded assertion, replace the synthesized tone with a real speech clip path via the `OLMLX_TEST_AUDIO` env var when available.

- [ ] **Step 2: Write the test**

Create `tests/live/test_vlm_audio.py`:

```python
"""Live VLM audio-input test (#426).

Loads a REAL Gemma 4 audio-capable VLM and verifies that a chat request
carrying an audio clip + text returns a coherent answer across all three
surfaces (OpenAI, Ollama, Anthropic).

Lives OUTSIDE tests/integration/ on purpose: that package's autouse
mock_mlx_primitives fixture would run this against mocks. Skipped in CI via
-m "not real_model", and skipped when the model is not present locally so it
never triggers a multi-GB download.
"""

import base64
import io
import os
import struct
import wave

import pytest

from olmlx.config import settings

AUDIO_MODEL = "mlx-community/gemma-4-e2b-it-4bit"


def _model_present() -> bool:
    from olmlx.models.store import _safe_dir_name

    return (settings.models_dir / _safe_dir_name(AUDIO_MODEL) / "config.json").exists()


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(not _model_present(), reason=f"{AUDIO_MODEL} not downloaded"),
]


def _wav_b64(seconds: float = 1.0, sr: int = 16000, freq: float = 440.0) -> str:
    """A short mono sine-tone WAV, base64-encoded. Real speech via OLMLX_TEST_AUDIO."""
    override = os.environ.get("OLMLX_TEST_AUDIO")
    if override:
        with open(override, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    import math

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(int(seconds * sr)):
            val = int(32767 * 0.2 * math.sin(2 * math.pi * freq * i / sr))
            w.writeframes(struct.pack("<h", val))
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.asyncio
async def test_openai_chat_accepts_audio():
    from httpx import ASGITransport, AsyncClient

    from olmlx.app import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": AUDIO_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this audio briefly."},
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": _wav_b64(), "format": "wav"},
                                },
                            ],
                        }
                    ],
                    "max_tokens": 64,
                    "stream": False,
                },
                timeout=600,
            )
    assert resp.status_code == 200, resp.text
    text = resp.json()["choices"][0]["message"]["content"]
    assert isinstance(text, str) and text.strip()


@pytest.mark.asyncio
async def test_audio_on_non_audio_model_returns_422():
    """A text-only model must reject audio with 422 (capability gate)."""
    from httpx import ASGITransport, AsyncClient

    from olmlx.app import create_app

    # Any small text model present locally; skip if none configured.
    text_model = os.environ.get("OLMLX_TEST_TEXT_MODEL")
    if not text_model:
        pytest.skip("set OLMLX_TEST_TEXT_MODEL to a local text model")

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": text_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "what is this?"},
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": _wav_b64(), "format": "wav"},
                                },
                            ],
                        }
                    ],
                    "max_tokens": 16,
                },
                timeout=600,
            )
    assert resp.status_code == 422
    assert "audio" in resp.text.lower()
```

> Mirror the exact ASGI-driving / lifespan pattern used in `tests/live/test_vlm_tools_images.py` if it differs from the above (e.g. a shared `app`/client fixture). Add Ollama (`/api/chat`) and Anthropic (`/v1/messages`) variants once the OpenAI path is green, reusing `_wav_b64()`.

- [ ] **Step 3: Run the live test (on a machine with the model)**

Run: `uv run pytest tests/live/test_vlm_audio.py -v -m real_model`
Expected: PASS for `test_openai_chat_accepts_audio` (200 + non-empty text); the 422 test runs only when `OLMLX_TEST_TEXT_MODEL` is set.

- [ ] **Step 4: Commit**

```bash
git add tests/live/test_vlm_audio.py
git commit -m "test(audio): live Gemma 4 audio-input coverage across surfaces (#426)"
```

---

## Final verification

- [ ] **Run the full unit suite (mock path)**

Run: `uv run pytest -m "not real_model" -q`
Expected: PASS (all new unit tests green; no regressions to image/text/tool paths)

- [ ] **Run ruff (required before push — see memory)**

Run: `uv run ruff check olmlx tests && uv run ruff format --check olmlx tests`
Expected: clean. Fix and re-run if needed.

- [ ] **Manual smoke (optional, on a machine with the audio model)**

Start the server (`uv run olmlx`) and POST an `input_audio` request to `/v1/chat/completions`; confirm a grounded reply and that no `olmlx-audio-*` temp files linger in `$TMPDIR` after the request.

---

## Notes for the implementer

- **Temp-file lifetime is the subtle part.** Materialize *inside* `_stream_completion` / `_full_completion_inner` (not in the routers), and delete in the generator's terminal `finally`, so the file outlives the fully-drained stream. Deleting right after `async_mlx_stream(...)` returns would race the worker thread.
- **`num_audios` only needs the count**, taken from the descriptor list — it is computed before materialization and is independent of temp-file paths.
- **Tools + audio is rejected, not silently dropped** (Task 5 guard). This is intentional v1 scope.
- **Argument order matters**: `images` then `audio`, positional, at every call site — keep them consistent so the plumbing reads cleanly against the existing image path.
- Run each task's tests before moving on; the threading tasks (7–9) lean on the live test for true end-to-end validation since they require a real VLM.
