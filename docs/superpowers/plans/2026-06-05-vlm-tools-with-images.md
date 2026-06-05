# VLM Native Tool Calling With Images (#428) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop dropping images when a chat request carries both `tools` and an image, and accept image inputs on the OpenAI and Anthropic surfaces so all three APIs can serve vision + function calling in one request.

**Architecture:** A surgical engine fix injects image content-parts into the last user message before the raw-tokenizer `apply_chat_template` call (Approach B), so the chat template emits `<|image|>` placeholders while `tools=` still renders natively. The image *data* already flows to `mlx_vlm.generate(image=images)` unchanged. Two router intake paths (OpenAI, Anthropic) convert their multimodal content shapes into the engine's `msg["images"]` convention via one shared normalization helper. Ollama `/api/chat` already works once the engine fix lands and is left untouched.

**Tech Stack:** Python, FastAPI, Pydantic v2, mlx-vlm 0.4.4, pytest.

**Spec:** `docs/superpowers/specs/2026-06-05-vlm-tools-with-images-design.md`

---

## File Structure

- **Create:** `olmlx/utils/images.py` — `normalize_image_block(block) -> str`, mapping OpenAI `image_url` / Anthropic `image` blocks to a `load_image`-acceptable string (URL, path, or data URI).
- **Modify:** `olmlx/engine/inference.py` — `_apply_chat_template_vlm` tools branch: inject image markers + count guard; add `_inject_image_markers` and `_vlm_image_token` helpers.
- **Modify:** `olmlx/schemas/openai.py` — widen `OpenAIChatMessage.content` to accept a content-parts list.
- **Modify:** `olmlx/routers/openai.py` — add `_normalize_multimodal_messages` and call it after `model_dump`.
- **Modify:** `olmlx/schemas/anthropic.py` — add `source` field to `AnthropicContentBlock`.
- **Modify:** `olmlx/routers/anthropic.py` — handle `image` blocks in `_convert_messages`.
- **Create/Modify tests:** `tests/test_images_util.py`, `tests/test_inference.py`, `tests/test_routers_openai.py`, `tests/test_routers_anthropic.py`, `tests/integration/test_vlm_tools_images.py`.
- **Modify:** `CLAUDE.md` — update the VLM-gating design note.

---

## Task 1: Shared image-normalization helper

**Files:**
- Create: `olmlx/utils/images.py`
- Test: `tests/test_images_util.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_images_util.py
import pytest

from olmlx.utils.images import normalize_image_block


def test_openai_image_url_passthrough():
    block = {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
    assert normalize_image_block(block) == "http://x/y.png"


def test_openai_image_url_data_uri_passthrough():
    uri = "data:image/png;base64,AAAA"
    block = {"type": "image_url", "image_url": {"url": uri}}
    assert normalize_image_block(block) == uri


def test_anthropic_base64_builds_data_uri():
    block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "QQ=="},
    }
    assert normalize_image_block(block) == "data:image/jpeg;base64,QQ=="


def test_anthropic_base64_defaults_media_type():
    block = {"type": "image", "source": {"type": "base64", "data": "QQ=="}}
    assert normalize_image_block(block) == "data:image/png;base64,QQ=="


def test_anthropic_url_source():
    block = {"type": "image", "source": {"type": "url", "url": "http://x/z.png"}}
    assert normalize_image_block(block) == "http://x/z.png"


def test_missing_url_raises():
    with pytest.raises(ValueError, match="image_url"):
        normalize_image_block({"type": "image_url", "image_url": {}})


def test_unsupported_source_type_raises():
    block = {"type": "image", "source": {"type": "file", "id": "abc"}}
    with pytest.raises(ValueError, match="unsupported image source"):
        normalize_image_block(block)


def test_not_an_image_block_raises():
    with pytest.raises(ValueError, match="not an image block"):
        normalize_image_block({"type": "text", "text": "hi"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_images_util.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.utils.images'`

- [ ] **Step 3: Write the implementation**

```python
# olmlx/utils/images.py
"""Normalize image references from API surfaces into mlx_vlm-loadable strings.

``mlx_vlm.utils.load_image`` accepts file paths, http(s) URLs, and
``data:image/...;base64,...`` data URIs (PIL sniffs the real format, so the
declared media type in a data URI is cosmetic).  This module converts the
OpenAI (``image_url``) and Anthropic (``image`` + ``source``) content-block
shapes into one of those forms (issue #428).
"""

from __future__ import annotations

from typing import Any


def normalize_image_block(block: dict[str, Any]) -> str:
    """Convert an OpenAI ``image_url`` or Anthropic ``image`` content block to a
    string ``load_image`` accepts (URL, path, or data URI).

    Raises ``ValueError`` for missing fields, unsupported source types, or a
    non-image block.
    """
    btype = block.get("type")

    # OpenAI: {"type": "image_url", "image_url": {"url": "..."}}
    if btype == "image_url":
        url = (block.get("image_url") or {}).get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("image_url block missing image_url.url")
        return url

    # Anthropic: {"type": "image", "source": {...}}
    if btype == "image":
        source = block.get("source") or {}
        stype = source.get("type")
        if stype == "url":
            url = source.get("url")
            if not isinstance(url, str) or not url:
                raise ValueError("image source type=url missing 'url'")
            return url
        if stype == "base64":
            data = source.get("data")
            if not isinstance(data, str) or not data:
                raise ValueError("image source type=base64 missing 'data'")
            media_type = source.get("media_type") or "image/png"
            return f"data:{media_type};base64,{data}"
        raise ValueError(f"unsupported image source type: {stype!r}")

    raise ValueError(f"not an image block: type={btype!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_images_util.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/utils/images.py tests/test_images_util.py
git commit -m "feat(images): shared image-block normalization helper (#428)"
```

---

## Task 2: Engine fix — inject image markers in the VLM native-tools path

This is the core change. The current `_apply_chat_template_vlm` tools branch (`olmlx/engine/inference.py`, around lines 2153-2176) logs a warning and ignores images. Replace it so images become content-parts on the last user message and add a guard that the rendered prompt has the expected placeholder count.

**Files:**
- Modify: `olmlx/engine/inference.py` (`_apply_chat_template_vlm`, ~2138-2176; add two module-level helpers just above it)
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inference.py  (add near the other apply_chat_template tests)
from unittest.mock import MagicMock

import pytest

from olmlx.engine.inference import _apply_chat_template_vlm


def _fake_gemma_processor():
    """Processor whose tokenizer renders one <|image|> per image content-part."""

    def fake_apply(messages, **kwargs):
        n = sum(
            1
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), list)
            for p in m["content"]
            if isinstance(p, dict) and p.get("type") == "image"
        )
        tool_text = "<|tool>record_metric<tool|>" if kwargs.get("tools") else ""
        return tool_text + ("<|image|>" * n) + "describe"

    tok = MagicMock()
    tok.apply_chat_template = MagicMock(side_effect=fake_apply)
    tok.image_token = "<|image|>"
    processor = MagicMock()
    processor.tokenizer = tok
    return processor, tok


def test_vlm_tools_with_images_injects_markers(caplog):
    processor, tok = _fake_gemma_processor()
    messages = [{"role": "user", "content": "describe"}]
    tools = [{"type": "function", "function": {"name": "record_metric"}}]

    prompt = _apply_chat_template_vlm(
        processor, MagicMock(), messages, ["a.jpg"], tools=tools
    )

    sent = tok.apply_chat_template.call_args[0][0]
    user = [m for m in sent if m["role"] == "user"][-1]
    assert isinstance(user["content"], list)
    assert {"type": "image"} in user["content"]
    assert {"type": "text", "text": "describe"} in user["content"]
    assert prompt.count("<|image|>") == 1
    assert "will be ignored" not in caplog.text


def test_vlm_tools_two_images_inject_two_markers():
    processor, tok = _fake_gemma_processor()
    messages = [{"role": "user", "content": "compare"}]
    tools = [{"type": "function", "function": {"name": "record_metric"}}]

    prompt = _apply_chat_template_vlm(
        processor, MagicMock(), messages, ["a.jpg", "b.jpg"], tools=tools
    )
    assert prompt.count("<|image|>") == 2


def test_vlm_tools_no_images_unchanged():
    processor, tok = _fake_gemma_processor()
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "record_metric"}}]

    _apply_chat_template_vlm(processor, MagicMock(), messages, None, tools=tools)
    sent = tok.apply_chat_template.call_args[0][0]
    user = [m for m in sent if m["role"] == "user"][-1]
    assert user["content"] == "hi"  # still a plain string


def test_vlm_tools_image_count_mismatch_raises():
    """Template that ignores image parts -> placeholder count mismatch -> error."""

    def fake_apply(messages, **kwargs):
        return "<|tool>record_metric<tool|>describe"  # no <|image|> emitted

    tok = MagicMock()
    tok.apply_chat_template = MagicMock(side_effect=fake_apply)
    tok.image_token = "<|image|>"
    processor = MagicMock()
    processor.tokenizer = tok
    messages = [{"role": "user", "content": "describe"}]
    tools = [{"type": "function", "function": {"name": "record_metric"}}]

    with pytest.raises(ValueError, match="image placeholder"):
        _apply_chat_template_vlm(processor, MagicMock(), messages, ["a.jpg"], tools=tools)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_inference.py -k "vlm_tools" -v`
Expected: FAIL — `test_vlm_tools_with_images_injects_markers` fails (content not rewritten / warning logged) and `test_vlm_tools_image_count_mismatch_raises` fails (no error raised).

- [ ] **Step 3: Add the two helpers above `_apply_chat_template_vlm`**

Insert immediately before `def _apply_chat_template_vlm(` (currently ~line 2138):

```python
def _vlm_image_token(processor: Any) -> str:
    """Best-effort image placeholder token for a VLM processor/tokenizer."""
    for obj in (processor, getattr(processor, "tokenizer", None)):
        token = getattr(obj, "image_token", None)
        if isinstance(token, str) and token:
            return token
    return "<image>"


def _inject_image_markers(messages: list[dict], num_images: int) -> list[dict]:
    """Rewrite the last user message's content into a parts list with
    ``num_images`` image markers so the chat template emits image placeholders
    while ``tools=`` still renders natively (issue #428).

    Single-turn scope: all images attach to the last user turn.
    """
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            text = m.get("content")
            parts: list[dict[str, Any]] = [{"type": "image"} for _ in range(num_images)]
            if isinstance(text, list):
                parts.extend(text)
            elif isinstance(text, str) and text:
                parts.append({"type": "text", "text": text})
            m["content"] = parts
            return out
    logger.warning(
        "VLM tools+images: no user message to attach %d image(s) to; "
        "rendering text-only",
        num_images,
    )
    return out
```

- [ ] **Step 4: Replace the tools branch body**

In `_apply_chat_template_vlm`, replace the current `if tools:` block (the warning + `tok = …` + `return tok.apply_chat_template(...)`) with:

```python
    if tools:
        # Use the tokenizer directly to get clean native tool formatting.
        # mlx_vlm.apply_chat_template wraps text content in dicts that some
        # Jinja templates render as Python list repr — garbling the prompt.
        tok = (
            processor.tokenizer
            if hasattr(processor, "tokenizer")
            and hasattr(processor.tokenizer, "apply_chat_template")
            else processor
        )
        # Images are carried separately (msg["images"]); inject content-part
        # markers so the template emits image placeholders alongside the native
        # tool tags (#428).  The image data is threaded to mlx_vlm.generate.
        if images:
            messages = _inject_image_markers(messages, len(images))
        kwargs: dict = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        prompt = tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        if images:
            image_token = _vlm_image_token(processor)
            found = prompt.count(image_token)
            if found != len(images):
                raise ValueError(
                    f"VLM tools+images: expected {len(images)} image placeholder(s) "
                    f"({image_token!r}) in the rendered prompt but found {found}; "
                    "the model's chat template may not support image content parts."
                )
        return prompt
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_inference.py -k "vlm_tools" -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Run the broader inference suite to check for regressions**

Run: `uv run pytest tests/test_inference.py -q`
Expected: PASS (no regressions in existing VLM/template tests)

- [ ] **Step 7: Commit**

```bash
git add olmlx/engine/inference.py tests/test_inference.py
git commit -m "feat(vlm): thread images through native-tools path (#428)"
```

---

## Task 3: OpenAI surface — accept multimodal content

**Files:**
- Modify: `olmlx/schemas/openai.py` (`OpenAIChatMessage.content`, line 18)
- Modify: `olmlx/routers/openai.py` (add `_normalize_multimodal_messages`; call after `model_dump`, ~line 311)
- Test: `tests/test_routers_openai.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_routers_openai.py  (add)
from olmlx.routers.openai import _normalize_multimodal_messages
from olmlx.schemas.openai import OpenAIChatRequest


def test_openai_request_accepts_multimodal_content():
    req = OpenAIChatRequest(
        model="m",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                ],
            }
        ],
    )
    assert isinstance(req.messages[0].content, list)


def test_normalize_multimodal_splits_text_and_images():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    out = _normalize_multimodal_messages(msgs)
    assert out[0]["content"] == "what is this?"
    assert out[0]["images"] == ["data:image/png;base64,AAAA"]


def test_normalize_multimodal_leaves_string_content_untouched():
    msgs = [{"role": "user", "content": "plain"}]
    out = _normalize_multimodal_messages(msgs)
    assert out[0]["content"] == "plain"
    assert "images" not in out[0]


def test_normalize_multimodal_text_only_list():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "just text"}]}]
    out = _normalize_multimodal_messages(msgs)
    assert out[0]["content"] == "just text"
    assert "images" not in out[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_routers_openai.py -k "multimodal" -v`
Expected: FAIL — `OpenAIChatRequest` rejects list content (validation error) and `_normalize_multimodal_messages` does not exist (ImportError).

- [ ] **Step 3: Widen the schema content type**

In `olmlx/schemas/openai.py`, change line 18 inside `OpenAIChatMessage`:

```python
    content: str | list[dict[str, Any]] | None = None
```

(`Any` is already imported at the top of the module.)

- [ ] **Step 4: Add the normalization helper to the OpenAI router**

In `olmlx/routers/openai.py`, add the import near the other engine imports:

```python
from olmlx.utils.images import normalize_image_block
```

Add this function at module scope (e.g. just above `openai_chat`):

```python
def _normalize_multimodal_messages(messages: list[dict]) -> list[dict]:
    """Split OpenAI multimodal content lists into a text ``content`` string plus
    a separate ``images`` list (the engine's Ollama-style convention, #428).

    OpenAI carries images inline as content parts:
        content: [{"type": "text", "text": ...},
                  {"type": "image_url", "image_url": {"url": ...}}]
    String content is left untouched.
    """
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        texts: list[str] = []
        images: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in ("text", "input_text"):
                text = part.get("text") or ""
                if text:
                    texts.append(text)
            elif ptype == "image_url":
                images.append(normalize_image_block(part))
        m["content"] = " ".join(texts)
        if images:
            m["images"] = (m.get("images") or []) + images
    return messages
```

- [ ] **Step 5: Call the helper after `model_dump`**

In `openai_chat` (around line 311), immediately after:

```python
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
```

add:

```python
    messages = _normalize_multimodal_messages(messages)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_routers_openai.py -k "multimodal" -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Run the OpenAI router suite for regressions**

Run: `uv run pytest tests/test_routers_openai.py -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add olmlx/schemas/openai.py olmlx/routers/openai.py tests/test_routers_openai.py
git commit -m "feat(openai): accept multimodal image content (#428)"
```

---

## Task 4: Anthropic surface — convert image blocks

**Files:**
- Modify: `olmlx/schemas/anthropic.py` (`AnthropicContentBlock`, add `source`, ~line 34)
- Modify: `olmlx/routers/anthropic.py` (`_convert_messages` user-block loop, ~lines 219-241)
- Test: `tests/test_routers_anthropic.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_routers_anthropic.py  (add)
from olmlx.routers.anthropic import _convert_messages
from olmlx.schemas.anthropic import AnthropicMessagesRequest


def test_anthropic_converts_base64_image_block():
    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=16,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "QQ==",
                        },
                    },
                ],
            }
        ],
    )
    msgs = _convert_messages(req)
    user = [m for m in msgs if m["role"] == "user"][0]
    assert user["content"] == "describe"
    assert user["images"] == ["data:image/jpeg;base64,QQ=="]


def test_anthropic_image_only_block_creates_user_message():
    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=16,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "data": "QQ=="},
                    }
                ],
            }
        ],
    )
    msgs = _convert_messages(req)
    user = [m for m in msgs if m["role"] == "user"][0]
    assert user["content"] == ""
    assert user["images"] == ["data:image/png;base64,QQ=="]


def test_anthropic_text_only_block_has_no_images_key():
    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=16,
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    )
    msgs = _convert_messages(req)
    user = [m for m in msgs if m["role"] == "user"][0]
    assert user["content"] == "hello"
    assert "images" not in user
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_routers_anthropic.py -k "image" -v`
Expected: FAIL — image blocks are dropped, so `user["images"]` is missing / the image-only message is not created.

- [ ] **Step 3: Add the `source` field to the content-block schema**

In `olmlx/schemas/anthropic.py`, inside `AnthropicContentBlock` (after the `tool_result fields`, ~line 35), add:

```python
    # image fields
    source: dict | None = None
```

- [ ] **Step 4: Handle image blocks in `_convert_messages`**

Add the import near the top of `olmlx/routers/anthropic.py`:

```python
from olmlx.utils.images import normalize_image_block
```

In the `elif msg.role == "user":` branch (currently ~lines 219-241), replace the loop and the trailing append with:

```python
        elif msg.role == "user":
            text_parts = []
            user_images: list[str] = []
            for block in msg.content:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
                elif block.type == "image":
                    user_images.append(
                        normalize_image_block(
                            {"type": "image", "source": block.source or {}}
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
            if text_parts or user_images:
                user_msg: dict = {"role": "user", "content": " ".join(text_parts)}
                if user_images:
                    user_msg["images"] = user_images
                messages.append(user_msg)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_routers_anthropic.py -k "image" -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Run the Anthropic router suite for regressions**

Run: `uv run pytest tests/test_routers_anthropic.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add olmlx/schemas/anthropic.py olmlx/routers/anthropic.py tests/test_routers_anthropic.py
git commit -m "feat(anthropic): convert image content blocks to images (#428)"
```

---

## Task 5: Gated live integration test (Gemma 4)

A `real_model`-marked test (excluded from CI via `-m "not real_model"`) that loads Gemma 4 26B-A4B locally and verifies a grounded tool call with an image on `/api/chat` and `/v1/chat/completions`. Reuses the no-mocks integration fixture pattern from `tests/integration/test_real_model.py`.

**Files:**
- Create: `tests/integration/test_vlm_tools_images.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_vlm_tools_images.py
"""Live VLM tools+images test (#428). Skipped in CI via -m "not real_model"."""

import base64
import io
import json

import pytest

pytestmark = pytest.mark.real_model

VLM_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_number",
            "description": "Record the number shown in the image.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        },
    }
]


def _png_data_uri_with_number(text: str = "42") -> str:
    """Render a small white PNG with large black text and return a data URI."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (160, 96), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


@pytest.fixture
async def real_ctx(tmp_path, monkeypatch):
    models_config = tmp_path / "models.json"
    models_config.write_text(json.dumps({}))
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text("{}")
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    monkeypatch.setattr("olmlx.config.settings.models_dir", models_dir)
    monkeypatch.setattr("olmlx.config.settings.models_config", models_config)
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_config)
    monkeypatch.setattr("olmlx.models.store.settings.models_dir", models_dir)

    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.registry import ModelRegistry
    from olmlx.models.store import ModelStore

    registry = ModelRegistry()
    registry._aliases_path = aliases_path
    registry.load()
    store = ModelStore(registry)
    manager = ModelManager(registry, store)
    manager.start_expiry_checker()

    from olmlx.app import create_app

    app = create_app()
    app.state.registry = registry
    app.state.model_manager = manager
    app.state.model_store = store

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await manager.stop()


async def test_openai_vlm_tools_with_image_produces_tool_call(real_ctx):
    client = real_ctx
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the number and record it."},
                        {"type": "image_url", "image_url": {"url": _png_data_uri_with_number("42")}},
                    ],
                }
            ],
            "tools": TOOLS,
            "max_tokens": 128,
        },
        timeout=600,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    tool_calls = body["choices"][0]["message"].get("tool_calls")
    assert tool_calls, f"expected a tool call, got: {body}"
    assert tool_calls[0]["function"]["name"] == "record_number"


async def test_ollama_vlm_tools_with_image_produces_tool_call(real_ctx):
    client = real_ctx
    resp = await client.post(
        "/api/chat",
        json={
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Read the number and record it.",
                    "images": [_png_data_uri_with_number("42")],
                }
            ],
            "tools": TOOLS,
            "stream": False,
        },
        timeout=600,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["message"].get("tool_calls"), f"expected a tool call, got: {body}"
```

- [ ] **Step 2: Verify CI exclusion still works (fast, no model download)**

Run: `uv run pytest tests/integration/test_vlm_tools_images.py -m "not real_model" -q`
Expected: `2 deselected` (or collected/deselected) — the tests are skipped, no model loads.

- [ ] **Step 3: Run the live test locally (loads Gemma 4, slow)**

Run: `uv run pytest tests/integration/test_vlm_tools_images.py -m real_model -v`
Expected: PASS — both endpoints return a `record_number` tool call grounded in the image. If the model reads a wrong number, the test still passes (asserts the call exists, not the exact value); if it fails to call the tool at all, investigate the rendered prompt.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_vlm_tools_images.py
git commit -m "test(vlm): gated live tools+images integration test (#428)"
```

---

## Task 6: Update CLAUDE.md VLM-gating note

**Files:**
- Modify: `CLAUDE.md` (the design note that currently says the VLM native-tools path drops images / VLM gating)

- [ ] **Step 1: Find the relevant note**

Run: `grep -n "native-tools path does not support images\|VLM native-tools\|images.*ignored\|VLM (image) path\|mlx_vlm.generate.*doesn't accept" CLAUDE.md`
Expected: locate the VLM-related design bullet(s).

- [ ] **Step 2: Update the note**

Edit the VLM design note so it reads (adjust to the exact surrounding bullet wording):

```
- **VLM tools + images** (#428): The native-tools path threads images by injecting
  `{type:"image"}` content-parts into the last user message before the raw-tokenizer
  `apply_chat_template` call (Approach B), so the chat template emits `<|image|>`
  placeholders alongside native tool tags; a post-render guard asserts the placeholder
  count matches `len(images)`. Image *data* already flowed to `mlx_vlm.generate(image=…)`.
  Image intake: Ollama `/api/chat` (`images` field, native), OpenAI `/v1/chat/completions`
  (`image_url` content parts, split into `images` by `_normalize_multimodal_messages`),
  and Anthropic `/v1/messages` (`image` blocks via `normalize_image_block` in
  `_convert_messages`). Scope is single-turn (images attach to the last user turn).
  Grammar / speculative / prompt-cache on the image path remain gated (#429).
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude-md): VLM tools+images intake across surfaces (#428)"
```

---

## Final verification

- [ ] **Run the full fast suite (CI-equivalent)**

Run: `uv run pytest -m "not real_model" -q`
Expected: PASS (no regressions).

- [ ] **Lint and format (required before pushing)**

Run: `uv run ruff check olmlx tests && uv run ruff format olmlx tests`
Expected: clean; commit any formatting changes.

- [ ] **Manual live confirmation** (if not already done in Task 5): `uv run pytest tests/integration/test_vlm_tools_images.py -m real_model -v` passes on Gemma 4.

---

## Self-Review Notes

- **Spec coverage:** engine fix (Task 2), OpenAI intake (Task 3), Anthropic intake (Task 4), shared helper (Task 1), Ollama left untouched (covered by Task 2 — already works), fast + gated live tests (Tasks 1-5), CLAUDE.md (Task 6), error handling (count guard in Task 2, helper `ValueError`s in Task 1). All spec sections map to a task.
- **Type consistency:** `normalize_image_block(block: dict) -> str` defined in Task 1, used identically in Tasks 3-4. `_inject_image_markers(messages, num_images)` and `_vlm_image_token(processor)` defined and used only in Task 2. `_normalize_multimodal_messages(messages)` defined and used in Task 3.
- **Scope:** single feature across three surfaces; one plan is appropriate (shared helper + engine seam couple the work).
