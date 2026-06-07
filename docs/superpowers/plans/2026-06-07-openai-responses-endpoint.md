# OpenAI `/v1/responses` Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenAI's `/v1/responses` API to olmlx as a drop-in surface (text, streaming with full semantic events, function tool-calls, `previous_response_id` continuation, reasoning items, image input, structured outputs), implemented as a pure translation layer over the existing `generate_chat` engine.

**Architecture:** A new `routers/responses.py` translates a Responses request into the engine's message-dict format, calls `generate_chat` unchanged, then translates the output back into Responses output-items / SSE events. A bounded-LRU `engine/responses_state.py` holds responses for `previous_response_id` continuation. Mirrors the proven `routers/anthropic.py` buffer-and-split structure. Design doc: `docs/superpowers/specs/2026-06-07-openai-responses-endpoint-design.md`.

**Tech Stack:** FastAPI, Pydantic v2, pytest + httpx `AsyncClient` (autouse MLX mock in `tests/`), the OpenAI Python SDK (for the live test only).

---

## Conventions used in this plan

- All test commands assume the repo root and `uv`. Run a single test with:
  `uv run pytest tests/test_routers_responses.py::TestClass::test_name -v`
- Router-level tests patch the engine: `patch("olmlx.routers.responses.generate_chat", ...)`.
- The `app_client` fixture (in `tests/conftest.py`) provides an httpx `AsyncClient`
  bound to a real `create_app()` with a mock manager. Reuse it.
- `TimingStats(prompt_eval_count=N, eval_count=M)` builds usage in tests.

---

## Task 1: Config setting + Pydantic schemas

**Files:**
- Modify: `olmlx/config.py` (add one setting near the other feature flags, ~line 114)
- Create: `olmlx/schemas/responses.py`
- Test: `tests/test_routers_responses.py` (new file — schema validation tests)

- [ ] **Step 1: Add the store-size setting to `config.py`**

In `olmlx/config.py`, inside `class Settings`, near the `prompt_cache_*` block (~line 114), add:

```python
    # Max number of stored Responses-API responses for previous_response_id
    # continuation (in-memory LRU; lost on restart).
    responses_store_max: Annotated[int, Field(gt=0)] = 256
```

(`Annotated` and `Field` are already imported in `config.py`.)

- [ ] **Step 2: Write the failing schema test**

Create `tests/test_routers_responses.py`:

```python
"""Tests for olmlx.routers.responses and its schemas."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from olmlx.schemas.responses import ResponsesRequest
from olmlx.utils.timing import TimingStats


class TestResponsesRequestSchema:
    def test_string_input_accepted(self):
        req = ResponsesRequest(model="qwen3", input="hello")
        assert req.input == "hello"
        assert req.store is True
        assert req.stream is False

    def test_list_input_accepted(self):
        req = ResponsesRequest(
            model="qwen3",
            input=[{"role": "user", "content": "hi"}],
        )
        assert isinstance(req.input, list)

    def test_defaults(self):
        req = ResponsesRequest(model="qwen3", input="x")
        assert req.previous_response_id is None
        assert req.tools is None
        assert req.max_output_tokens is None
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_routers_responses.py::TestResponsesRequestSchema -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.schemas.responses'`

- [ ] **Step 4: Implement the schemas**

Create `olmlx/schemas/responses.py`:

```python
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from olmlx.schemas.common import ModelName


class ResponsesRequest(BaseModel):
    model: ModelName
    # `input` is either a plain string (one user turn) or a list of input
    # items (messages / function_call / function_call_output). Items are kept
    # as freeform dicts and validated during translation (router raises
    # ValueError -> 400/422 for unknown shapes).
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    stream: bool = False
    max_output_tokens: int | None = Field(None, ge=1)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    previous_response_id: str | None = None
    store: bool = True
    reasoning: dict[str, Any] | None = None  # {"effort": "low"|"medium"|"high"}
    text: dict[str, Any] | None = None       # {"format": {"type": ...}}
    metadata: dict[str, Any] | None = None
    seed: int | None = None

    @field_validator("max_output_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int | None) -> int | None:
        if v is None:
            return v
        from olmlx.schemas.common import validate_token_limit

        return validate_token_limit(v, "max_output_tokens")


class ResponsesResponse(BaseModel):
    """Top-level Responses object. Output items and usage are kept as dicts so
    the router builds them directly; response_model_exclude_none trims absent
    fields the SDK treats as optional."""

    id: str
    object: str = "response"
    created_at: int
    status: str
    model: str
    output: list[dict[str, Any]]
    usage: dict[str, Any] | None = None
    previous_response_id: str | None = None
    incomplete_details: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, Any] | None = None
    parallel_tool_calls: bool = True
    temperature: float | None = None
    top_p: float | None = None
    tool_choice: str | dict[str, Any] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
```

- [ ] **Step 5: Run the schema test to verify it passes**

Run: `uv run pytest tests/test_routers_responses.py::TestResponsesRequestSchema -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add olmlx/config.py olmlx/schemas/responses.py tests/test_routers_responses.py
git commit -m "feat(responses): request/response schemas + store-size setting (#368)"
```

---

## Task 2: Bounded-LRU state store

**Files:**
- Create: `olmlx/engine/responses_state.py`
- Test: `tests/test_responses_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_responses_state.py`:

```python
"""Tests for the Responses-API state store."""

from olmlx.engine.responses_state import ResponsesStore


def test_put_get_roundtrip():
    store = ResponsesStore(max_entries=4)
    store.put("resp_1", {"output_items": [{"a": 1}]})
    assert store.get("resp_1") == {"output_items": [{"a": 1}]}


def test_missing_returns_none():
    store = ResponsesStore(max_entries=4)
    assert store.get("nope") is None


def test_delete():
    store = ResponsesStore(max_entries=4)
    store.put("resp_1", {"x": 1})
    assert store.delete("resp_1") is True
    assert store.get("resp_1") is None
    assert store.delete("resp_1") is False


def test_lru_eviction():
    store = ResponsesStore(max_entries=2)
    store.put("a", {"v": 1})
    store.put("b", {"v": 2})
    store.put("c", {"v": 3})  # evicts "a"
    assert store.get("a") is None
    assert store.get("b") == {"v": 2}
    assert store.get("c") == {"v": 3}


def test_get_refreshes_lru():
    store = ResponsesStore(max_entries=2)
    store.put("a", {"v": 1})
    store.put("b", {"v": 2})
    store.get("a")          # "a" now most-recently used
    store.put("c", {"v": 3})  # evicts "b", not "a"
    assert store.get("a") == {"v": 1}
    assert store.get("b") is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_responses_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.engine.responses_state'`

- [ ] **Step 3: Implement the store**

Create `olmlx/engine/responses_state.py`:

```python
"""In-memory LRU store for Responses-API state (previous_response_id).

Single-user, localhost-only: a bounded OrderedDict keyed by response_id.
Not persisted; lost on restart. Bounded by OLMLX_RESPONSES_STORE_MAX so it is
not counted against the memory limit (text-only entries).
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any

from olmlx.config import settings


class ResponsesStore:
    def __init__(self, max_entries: int | None = None) -> None:
        self._max = (
            max_entries if max_entries is not None else settings.responses_store_max
        )
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = Lock()

    def put(self, response_id: str, entry: dict[str, Any]) -> None:
        with self._lock:
            self._store[response_id] = entry
            self._store.move_to_end(response_id)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def get(self, response_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._store.get(response_id)
            if entry is not None:
                self._store.move_to_end(response_id)
            return entry

    def delete(self, response_id: str) -> bool:
        with self._lock:
            return self._store.pop(response_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_store = ResponsesStore()


def get_store() -> ResponsesStore:
    """Return the process-wide Responses store singleton."""
    return _store
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_responses_state.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/responses_state.py tests/test_responses_state.py
git commit -m "feat(responses): bounded-LRU state store for previous_response_id (#368)"
```

---

## Task 3: Translation helpers (tools, input items, options, grammar, reasoning)

**Files:**
- Create: `olmlx/routers/responses.py` (helpers only this task; endpoint added in Task 4)
- Test: `tests/test_routers_responses.py` (append `TestTranslation`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_routers_responses.py`:

```python
from olmlx.routers.responses import (
    _build_input_messages,
    _convert_tools,
    _grammar_from_text_format,
    _resolve_reasoning,
)


class TestTranslation:
    def test_string_input(self):
        msgs = _build_input_messages("hello")
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_message_item_string_content(self):
        msgs = _build_input_messages([{"role": "user", "content": "hi"}])
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_message_item_text_parts(self):
        msgs = _build_input_messages(
            [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
        )
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_message_item_image_part(self):
        msgs = _build_input_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {"type": "input_image", "image_url": "http://x/y.png"},
                    ],
                }
            ]
        )
        assert msgs[0]["content"] == "describe"
        assert msgs[0]["images"] == ["http://x/y.png"]

    def test_function_call_output_item(self):
        msgs = _build_input_messages(
            [{"type": "function_call_output", "call_id": "call_1", "output": "42"}]
        )
        assert msgs == [
            {"role": "tool", "tool_call_id": "call_1", "content": "42"}
        ]

    def test_function_call_item(self):
        msgs = _build_input_messages(
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "SF"}',
                }
            ]
        )
        assert msgs[0]["role"] == "assistant"
        tc = msgs[0]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["id"] == "call_1"

    def test_unknown_item_type_raises(self):
        with pytest.raises(ValueError):
            _build_input_messages([{"type": "mystery"}])

    def test_convert_function_tool(self):
        tools = _convert_tools(
            [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "d",
                    "parameters": {"type": "object"},
                }
            ]
        )
        assert tools[0]["function"]["name"] == "get_weather"
        assert tools[0]["type"] == "function"

    def test_builtin_tool_rejected(self):
        with pytest.raises(ValueError):
            _convert_tools([{"type": "web_search"}])

    def test_grammar_json_object(self):
        spec = _grammar_from_text_format({"format": {"type": "json_object"}})
        assert spec is not None

    def test_grammar_none_for_text(self):
        assert _grammar_from_text_format({"format": {"type": "text"}}) is None
        assert _grammar_from_text_format(None) is None

    def test_resolve_reasoning_presence(self):
        assert _resolve_reasoning({"effort": "high"}) is True
        assert _resolve_reasoning({"effort": "none"}) is False
        assert _resolve_reasoning(None) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_routers_responses.py::TestTranslation -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.routers.responses'`

- [ ] **Step 3: Implement the helpers**

Create `olmlx/routers/responses.py`:

```python
import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from olmlx.engine.grammar import parse_response_format
from olmlx.engine.inference import generate_chat
from olmlx.engine.responses_state import get_store
from olmlx.engine.tool_parser import (
    fill_missing_required_args,
    parse_model_output,
    resolve_tool_names,
)
from olmlx.routers.common import build_inference_options, resolve_openai_think
from olmlx.schemas.responses import ResponsesRequest, ResponsesResponse
from olmlx.utils.images import normalize_image_block

logger = logging.getLogger(__name__)

router = APIRouter()

_BUILTIN_TOOL_TYPES = {"web_search", "code_interpreter", "computer_use", "file_search"}


def _make_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def _make_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _make_reasoning_id() -> str:
    return f"rs_{uuid.uuid4().hex[:24]}"


def _make_fc_id() -> str:
    return f"fc_{uuid.uuid4().hex[:24]}"


def _make_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _message_item_to_engine(item: dict) -> dict:
    """Convert a Responses `message` input item to an engine message dict."""
    role = item["role"]
    content = item.get("content")
    if isinstance(content, str):
        return {"role": role, "content": content}
    texts: list[str] = []
    images: list[str] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("input_text", "text", "output_text"):
            txt = part.get("text") or ""
            if txt:
                texts.append(txt)
        elif ptype in ("input_image", "image_url"):
            raw = part.get("image_url")
            if isinstance(raw, str):
                block = {"type": "image_url", "image_url": {"url": raw}}
            else:
                block = {"type": "image_url", "image_url": raw or {}}
            images.append(normalize_image_block(block))
        else:
            raise ValueError(f"unsupported content part type: {ptype!r}")
    msg: dict = {"role": role, "content": " ".join(texts)}
    if images:
        msg["images"] = images
    return msg


def _build_input_messages(input_data: str | list[dict]) -> list[dict]:
    """Translate a Responses `input` field into engine message dicts."""
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    messages: list[dict] = []
    for item in input_data:
        itype = item.get("type")
        role = item.get("role")
        if itype in (None, "message") and role is not None:
            messages.append(_message_item_to_engine(item))
        elif itype == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id"),
                            "type": "function",
                            "function": {
                                "name": item["name"],
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
        elif itype == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": item.get("output", ""),
                }
            )
        else:
            raise ValueError(f"unsupported input item type: {itype!r}")
    return messages


def _convert_tools(tools: list[dict] | None) -> list[dict] | None:
    """Convert Responses function tools to engine OpenAI-nested format."""
    if not tools:
        return None
    converted: list[dict] = []
    for t in tools:
        ttype = t.get("type")
        if ttype != "function":
            if ttype in _BUILTIN_TOOL_TYPES:
                raise ValueError(
                    f"built-in tool {ttype!r} is not supported; only custom "
                    "'function' tools are available"
                )
            raise ValueError(f"unsupported tool type: {ttype!r}")
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                },
            }
        )
    return converted


def _grammar_from_text_format(text_cfg: dict | None):
    """Map a Responses `text.format` config to a GrammarSpec (or None)."""
    fmt = (text_cfg or {}).get("format")
    if not fmt:
        return None
    ftype = fmt.get("type")
    if ftype in (None, "text"):
        return None
    if ftype == "json_object":
        return parse_response_format({"type": "json_object"})
    if ftype == "json_schema":
        return parse_response_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.get("name", "schema"),
                    "schema": fmt.get("schema", {}),
                },
            }
        )
    raise ValueError(f"unsupported text.format type: {ftype!r}")


def _resolve_reasoning(reasoning: dict | None) -> bool | None:
    """Map Responses `reasoning.effort` to the engine enable_thinking flag."""
    effort = (reasoning or {}).get("effort")
    return resolve_openai_think(effort, None)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_routers_responses.py::TestTranslation -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add olmlx/routers/responses.py tests/test_routers_responses.py
git commit -m "feat(responses): request translation helpers (#368)"
```

---

## Task 4: Non-streaming POST endpoint (text) + router registration

**Files:**
- Modify: `olmlx/routers/responses.py` (add output builder + POST handler)
- Modify: `olmlx/app.py` (register router)
- Test: `tests/test_routers_responses.py` (append `TestNonStreamingText`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_routers_responses.py`:

```python
class TestNonStreamingText:
    @pytest.mark.asyncio
    async def test_text_response_shape(self, app_client):
        stats = TimingStats(prompt_eval_count=5, eval_count=3)
        mock_result = {"text": "Hello there.", "done": True, "stats": stats}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "stream": False},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert data["id"].startswith("resp_")
        # one message item with one output_text part
        msg = next(it for it in data["output"] if it["type"] == "message")
        assert msg["role"] == "assistant"
        assert msg["content"][0]["type"] == "output_text"
        assert msg["content"][0]["text"] == "Hello there."
        assert data["usage"]["input_tokens"] == 5
        assert data["usage"]["output_tokens"] == 3
        assert data["usage"]["total_tokens"] == 8

    @pytest.mark.asyncio
    async def test_timeout_marks_incomplete(self, app_client):
        mock_result = {
            "text": "partial",
            "done": True,
            "done_reason": "timeout",
            "stats": TimingStats(),
        }
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi"},
            )
        data = resp.json()
        assert data["status"] == "incomplete"
        assert data["incomplete_details"]["reason"] == "max_output_tokens"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_routers_responses.py::TestNonStreamingText -v`
Expected: FAIL with 404 (route not registered) — assertion error on status_code.

- [ ] **Step 3: Build the output builder + POST handler**

Append to `olmlx/routers/responses.py`:

```python
def _build_output_items(
    thinking: str, visible_text: str, tool_uses: list[dict]
) -> list[dict]:
    """Assemble Responses output items in canonical order."""
    items: list[dict] = []
    if thinking:
        items.append(
            {
                "type": "reasoning",
                "id": _make_reasoning_id(),
                "summary": [],
                "content": [{"type": "reasoning_text", "text": thinking}],
            }
        )
    if visible_text:
        items.append(
            {
                "type": "message",
                "id": _make_message_id(),
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": visible_text, "annotations": []}
                ],
            }
        )
    for tu in tool_uses:
        items.append(
            {
                "type": "function_call",
                "id": _make_fc_id(),
                "call_id": _make_call_id(),
                "name": tu["name"],
                "arguments": json.dumps(tu.get("input", {})),
                "status": "completed",
            }
        )
    return items


def _usage_dict(stats) -> dict:
    if stats is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    prompt = stats.prompt_eval_count
    completion = stats.eval_count
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _build_response_object(
    req: ResponsesRequest,
    response_id: str,
    created: int,
    output_items: list[dict],
    stats,
    done_reason: str | None,
) -> ResponsesResponse:
    incomplete = None
    status = "completed"
    if done_reason == "timeout":
        status = "incomplete"
        incomplete = {"reason": "max_output_tokens"}
    return ResponsesResponse(
        id=response_id,
        created_at=created,
        status=status,
        model=req.model,
        output=output_items,
        usage=_usage_dict(stats),
        previous_response_id=req.previous_response_id,
        incomplete_details=incomplete,
        instructions=req.instructions,
        max_output_tokens=req.max_output_tokens,
        metadata=req.metadata,
        temperature=req.temperature,
        top_p=req.top_p,
        tool_choice=req.tool_choice,
        tools=req.tools or [],
    )


@router.post(
    "/v1/responses",
    response_model=ResponsesResponse,
    response_model_exclude_none=True,
)
async def create_response(req: ResponsesRequest, request: Request):
    manager = request.app.state.model_manager
    logger.info(
        "Responses request: model=%s stream=%s tools=%d prev=%s",
        req.model,
        req.stream,
        len(req.tools or []),
        req.previous_response_id,
    )

    # --- translate request -> engine inputs (client errors -> 400/422) ---
    try:
        messages = _build_input_messages(req.input)
        tools = _convert_tools(req.tools)
        grammar_spec = _grammar_from_text_format(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if req.instructions:
        messages.insert(0, {"role": "system", "content": req.instructions})

    options = build_inference_options(
        temperature=req.temperature, top_p=req.top_p, seed=req.seed
    )
    max_tokens = req.max_output_tokens or 512
    enable_thinking = _resolve_reasoning(req.reasoning)
    response_id = _make_response_id()
    created = int(time.time())
    cache_id = (req.previous_response_id or response_id)[:256]

    result = await generate_chat(
        manager,
        req.model,
        messages,
        options,
        tools=tools,
        stream=False,
        max_tokens=max_tokens,
        cache_id=cache_id,
        enable_thinking=enable_thinking,
        grammar_spec=grammar_spec,
    )

    parse_text = result.get("raw_text") or result.get("text", "")
    thinking, visible_text, tool_uses = parse_model_output(
        parse_text,
        bool(tools),
        thinking_expected=bool(result.get("thinking_expected")),
    )
    resolve_tool_names(tool_uses, req.tools)
    fill_missing_required_args(tool_uses, req.tools)

    output_items = _build_output_items(thinking, visible_text, tool_uses)
    response = _build_response_object(
        req, response_id, created, output_items, result.get("stats"),
        result.get("done_reason"),
    )
    return response
```

- [ ] **Step 4: Register the router in `app.py`**

In `olmlx/app.py`, add the import alongside the other router imports and register it
after the openai router (~line 400):

```python
    app.include_router(responses.router)
```

Add `responses` to the routers import block at the top of `app.py` (find the line
importing `openai` from `.routers` and add `responses` to it; match the existing
import style).

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_routers_responses.py::TestNonStreamingText -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add olmlx/routers/responses.py olmlx/app.py tests/test_routers_responses.py
git commit -m "feat(responses): non-streaming text endpoint + router registration (#368)"
```

---

## Task 5: Tool calls, reasoning, structured outputs, images (non-streaming)

**Files:**
- Test: `tests/test_routers_responses.py` (append `TestNonStreamingFeatures`)
- (No new implementation — these are exercised by Task 3/4 code; this task is the
  coverage that proves the wiring. If a test fails, fix the helper it exercises.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_routers_responses.py`:

```python
class TestNonStreamingFeatures:
    @pytest.mark.asyncio
    async def test_function_call_item_emitted(self, app_client):
        raw = '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>'
        mock_result = {"text": raw, "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "weather in SF?",
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ],
                },
            )
        data = resp.json()
        fc = next(it for it in data["output"] if it["type"] == "function_call")
        assert fc["name"] == "get_weather"
        assert json.loads(fc["arguments"]) == {"city": "SF"}
        assert fc["call_id"].startswith("call_")

    @pytest.mark.asyncio
    async def test_reasoning_item_from_think(self, app_client):
        mock_result = {
            "text": "<think>step one</think>The answer is 42.",
            "done": True,
            "stats": TimingStats(),
            "thinking_expected": True,
        }
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "q", "reasoning": {"effort": "high"}},
            )
        data = resp.json()
        reasoning = next(it for it in data["output"] if it["type"] == "reasoning")
        assert "step one" in reasoning["content"][0]["text"]
        msg = next(it for it in data["output"] if it["type"] == "message")
        assert msg["content"][0]["text"] == "The answer is 42."
        assert mock_gen.call_args.kwargs["enable_thinking"] is True

    @pytest.mark.asyncio
    async def test_structured_output_threads_grammar(self, app_client):
        mock_result = {"text": "{}", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "q",
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "thing",
                            "schema": {"type": "object"},
                        }
                    },
                },
            )
        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs["grammar_spec"] is not None

    @pytest.mark.asyncio
    async def test_image_input_threads_images(self, app_client):
        mock_result = {"text": "a cat", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "what is this?"},
                                {
                                    "type": "input_image",
                                    "image_url": "http://x/cat.png",
                                },
                            ],
                        }
                    ],
                },
            )
        assert resp.status_code == 200
        sent_messages = mock_gen.call_args.args[2]
        assert sent_messages[0]["images"] == ["http://x/cat.png"]

    @pytest.mark.asyncio
    async def test_builtin_tool_rejected_422(self, app_client):
        resp = await app_client.post(
            "/v1/responses",
            json={"model": "qwen3", "input": "q", "tools": [{"type": "web_search"}]},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_unknown_input_item_422(self, app_client):
        resp = await app_client.post(
            "/v1/responses",
            json={"model": "qwen3", "input": [{"type": "mystery"}]},
        )
        assert resp.status_code == 422
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_routers_responses.py::TestNonStreamingFeatures -v`
Expected: PASS (6 tests). If any fail, fix the corresponding helper in Task 3/4.

- [ ] **Step 3: Commit**

```bash
git add tests/test_routers_responses.py
git commit -m "test(responses): non-streaming tool/reasoning/grammar/image coverage (#368)"
```

---

## Task 6: `previous_response_id` continuation + store write + `store: false`

**Files:**
- Modify: `olmlx/routers/responses.py` (read prior history, store result, honor `store`)
- Test: `tests/test_routers_responses.py` (append `TestStateContinuation`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_routers_responses.py`:

```python
from olmlx.engine.responses_state import get_store


class TestStateContinuation:
    @pytest.fixture(autouse=True)
    def _clear_store(self):
        get_store().clear()
        yield
        get_store().clear()

    @pytest.mark.asyncio
    async def test_response_is_stored(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses", json={"model": "qwen3", "input": "hi"}
            )
        rid = resp.json()["id"]
        assert get_store().get(rid) is not None

    @pytest.mark.asyncio
    async def test_store_false_skips(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "store": False},
            )
        rid = resp.json()["id"]
        assert get_store().get(rid) is None

    @pytest.mark.asyncio
    async def test_continuation_prepends_history_and_threads_cache_id(self, app_client):
        # First turn.
        first = {"text": "Blue.", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = first
            r1 = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "favorite color?"},
            )
        rid = r1.json()["id"]

        # Second turn references the first.
        second = {"text": "Because.", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = second
            await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "why?",
                    "previous_response_id": rid,
                },
            )
            sent_messages = mock_gen.call_args.args[2]
            # history (user + assistant) precedes the new user turn
            roles = [m["role"] for m in sent_messages]
            assert roles == ["user", "assistant", "user"]
            assert sent_messages[1]["content"] == "Blue."
            assert sent_messages[-1]["content"] == "why?"
            # cache_id is the previous response id (prompt-cache reuse)
            assert mock_gen.call_args.kwargs["cache_id"] == rid[:256]

    @pytest.mark.asyncio
    async def test_unknown_previous_id_404(self, app_client):
        resp = await app_client.post(
            "/v1/responses",
            json={"model": "qwen3", "input": "x", "previous_response_id": "resp_nope"},
        )
        assert resp.status_code == 404
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_routers_responses.py::TestStateContinuation -v`
Expected: FAIL — continuation/store not implemented (history not prepended; unknown id returns 200; store empty).

- [ ] **Step 3: Implement continuation + storage**

In `olmlx/routers/responses.py`, add a history-reconstruction helper near the other
helpers:

```python
def _history_messages_from_store(previous_response_id: str) -> list[dict]:
    """Rebuild engine messages from a stored response (input + output items)."""
    entry = get_store().get(previous_response_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"previous_response_id not found: {previous_response_id!r}",
        )
    messages = list(entry["input_messages"])
    for item in entry["output_items"]:
        itype = item.get("type")
        if itype == "message":
            text = "".join(
                part.get("text", "")
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
            if text:
                messages.append({"role": "assistant", "content": text})
        elif itype == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id"),
                            "type": "function",
                            "function": {
                                "name": item["name"],
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
        # reasoning items are not replayed into prompt history
    return messages
```

Then modify `create_response` so the message list includes prior history, and the
result is stored. Replace the block from `messages = _build_input_messages(req.input)`
through the `if req.instructions:` insert with:

```python
    try:
        new_messages = _build_input_messages(req.input)
        tools = _convert_tools(req.tools)
        grammar_spec = _grammar_from_text_format(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if req.previous_response_id:
        messages = _history_messages_from_store(req.previous_response_id)
        messages.extend(new_messages)
    else:
        messages = new_messages

    if req.instructions:
        messages.insert(0, {"role": "system", "content": req.instructions})
```

(`_history_messages_from_store` raises `HTTPException(404)` directly, which must NOT
be wrapped by the `try/except ValueError` above — keep the call outside that block,
as shown.)

After building `response` (just before `return response`), add storage:

```python
    if req.store:
        get_store().put(
            response_id,
            {
                "input_messages": messages,
                "output_items": output_items,
                "model": req.model,
                "previous_response_id": req.previous_response_id,
                "response": response.model_dump(exclude_none=True),
            },
        )
    return response
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_routers_responses.py::TestStateContinuation -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add olmlx/routers/responses.py tests/test_routers_responses.py
git commit -m "feat(responses): previous_response_id continuation + state storage (#368)"
```

---

## Task 7: Streaming with full semantic events

**Files:**
- Modify: `olmlx/routers/responses.py` (streaming generator + branch in handler)
- Test: `tests/test_routers_responses.py` (append `TestStreaming`)

The streaming path buffers `generate_chat` (tool parsing needs the full output, same
as the chat/anthropic routers), parses once, then replays the assembled items as the
full semantic event sequence with a monotonic `sequence_number`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_routers_responses.py`:

```python
def _parse_sse(body: str) -> list[dict]:
    """Parse an SSE body into a list of {event, data} dicts."""
    events = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        events.append({"event": event, "data": data})
    return events


class TestStreaming:
    @pytest.mark.asyncio
    async def test_text_stream_event_sequence(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Hel", "done": False}
                yield {"text": "lo", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=2)}

            return gen()

        with patch(
            "olmlx.routers.responses.generate_chat", side_effect=mock_stream
        ):
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "stream": True},
            )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        types = [e["event"] for e in events]
        assert types[0] == "response.created"
        assert "response.output_text.delta" in types
        assert types[-1] == "response.completed"
        # sequence numbers are present and monotonically increasing
        seqs = [e["data"]["sequence_number"] for e in events]
        assert seqs == sorted(seqs)
        # the deltas reconstruct the full text
        text = "".join(
            e["data"]["delta"]
            for e in events
            if e["event"] == "response.output_text.delta"
        )
        assert text == "Hello"
        # final response object is completed with the message item
        final = events[-1]["data"]["response"]
        assert final["status"] == "completed"

    @pytest.mark.asyncio
    async def test_tool_call_stream(self, app_client):
        raw = '<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": raw, "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch(
            "olmlx.routers.responses.generate_chat", side_effect=mock_stream
        ):
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "go",
                    "stream": True,
                    "tools": [
                        {"type": "function", "name": "f", "parameters": {"type": "object"}}
                    ],
                },
            )
        events = _parse_sse(resp.text)
        types = [e["event"] for e in events]
        assert "response.function_call_arguments.delta" in types
        assert "response.function_call_arguments.done" in types
        final = events[-1]["data"]["response"]
        fc = next(it for it in final["output"] if it["type"] == "function_call")
        assert fc["name"] == "f"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_routers_responses.py::TestStreaming -v`
Expected: FAIL — streaming branch not implemented (handler ignores `stream`).

- [ ] **Step 3: Implement streaming**

Add to `olmlx/routers/responses.py`. First a small SSE helper near `_make_response_id`:

```python
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
```

Then the streaming generator (place after `_build_response_object`):

```python
async def _stream_response(
    result,
    req: ResponsesRequest,
    response_id: str,
    created: int,
    tools: list[dict] | None,
    messages: list[dict],
):
    """Buffer the engine output, parse once, replay full semantic events."""
    seq = 0

    def ev(event: str, data: dict) -> str:
        nonlocal seq
        payload = {"type": event, "sequence_number": seq, **data}
        seq += 1
        return _sse(event, payload)

    def base_response(status: str, output_items: list[dict], stats, done_reason):
        return _build_response_object(
            req, response_id, created, output_items, stats, done_reason
        ).model_dump(exclude_none=True)

    try:
        # 1. Buffer the engine stream.
        full_text = ""
        raw_text = ""
        done_reason = None
        thinking_expected = False
        stats = None
        async for chunk in result:
            if chunk.get("cache_info"):
                continue
            if "thinking_expected" in chunk:
                thinking_expected = bool(chunk["thinking_expected"])
                continue
            if chunk.get("done"):
                raw_text = chunk.get("raw_text", raw_text)
                done_reason = chunk.get("done_reason")
                stats = chunk.get("stats")
                break
            full_text += chunk.get("text", "")

        parse_text = raw_text or full_text
        thinking, visible_text, tool_uses = parse_model_output(
            parse_text, bool(tools), thinking_expected=thinking_expected
        )
        resolve_tool_names(tool_uses, req.tools)
        fill_missing_required_args(tool_uses, req.tools)
        output_items = _build_output_items(thinking, visible_text, tool_uses)

        # 2. created / in_progress
        in_progress_resp = base_response("in_progress", [], stats, None)
        in_progress_resp["status"] = "in_progress"
        yield ev("response.created", {"response": in_progress_resp})
        yield ev("response.in_progress", {"response": in_progress_resp})

        # 3. Replay each output item through the event envelope.
        for out_index, item in enumerate(output_items):
            yield ev(
                "response.output_item.added",
                {"output_index": out_index, "item": item},
            )
            if item["type"] == "message":
                part = item["content"][0]
                yield ev(
                    "response.content_part.added",
                    {
                        "item_id": item["id"],
                        "output_index": out_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                )
                yield ev(
                    "response.output_text.delta",
                    {
                        "item_id": item["id"],
                        "output_index": out_index,
                        "content_index": 0,
                        "delta": part["text"],
                    },
                )
                yield ev(
                    "response.output_text.done",
                    {
                        "item_id": item["id"],
                        "output_index": out_index,
                        "content_index": 0,
                        "text": part["text"],
                    },
                )
                yield ev(
                    "response.content_part.done",
                    {
                        "item_id": item["id"],
                        "output_index": out_index,
                        "content_index": 0,
                        "part": part,
                    },
                )
            elif item["type"] == "function_call":
                yield ev(
                    "response.function_call_arguments.delta",
                    {
                        "item_id": item["id"],
                        "output_index": out_index,
                        "delta": item["arguments"],
                    },
                )
                yield ev(
                    "response.function_call_arguments.done",
                    {
                        "item_id": item["id"],
                        "output_index": out_index,
                        "arguments": item["arguments"],
                    },
                )
            # reasoning items: added/done only (no text deltas in v1)
            yield ev(
                "response.output_item.done",
                {"output_index": out_index, "item": item},
            )

        # 4. completed
        final = base_response("completed", output_items, stats, done_reason)
        yield ev("response.completed", {"response": final})

        # 5. store (mirrors the non-streaming path)
        if req.store:
            get_store().put(
                response_id,
                {
                    "input_messages": messages,
                    "output_items": output_items,
                    "model": req.model,
                    "previous_response_id": req.previous_response_id,
                    "response": final,
                },
            )
    except Exception as exc:
        logger.error("Error during Responses streaming: %s", exc, exc_info=True)
        yield ev(
            "response.failed",
            {
                "response": {
                    "id": response_id,
                    "status": "failed",
                    "error": {
                        "code": "server_error",
                        "message": "An internal server error occurred during streaming.",
                    },
                }
            },
        )
    finally:
        await result.aclose()
```

Now branch in `create_response`. Replace the single `result = await generate_chat(... stream=False ...)` call and everything after it with a stream/non-stream split:

```python
    if req.stream:
        result = await generate_chat(
            manager,
            req.model,
            messages,
            options,
            tools=tools,
            stream=True,
            max_tokens=max_tokens,
            cache_id=cache_id,
            enable_thinking=enable_thinking,
            grammar_spec=grammar_spec,
        )
        return StreamingResponse(
            _stream_response(result, req, response_id, created, tools, messages),
            media_type="text/event-stream",
        )

    result = await generate_chat(
        manager,
        req.model,
        messages,
        options,
        tools=tools,
        stream=False,
        max_tokens=max_tokens,
        cache_id=cache_id,
        enable_thinking=enable_thinking,
        grammar_spec=grammar_spec,
    )
    # ... existing non-streaming parse/build/store/return ...
```

Keep the existing non-streaming parse → `_build_output_items` → `_build_response_object`
→ store → `return response` code below the `result = await generate_chat(... stream=False ...)`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_routers_responses.py::TestStreaming -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the whole file to confirm no regressions**

Run: `uv run pytest tests/test_routers_responses.py -v`
Expected: PASS (all classes)

- [ ] **Step 6: Commit**

```bash
git add olmlx/routers/responses.py tests/test_routers_responses.py
git commit -m "feat(responses): streaming with full semantic SSE events (#368)"
```

---

## Task 8: `GET` / `DELETE /v1/responses/{id}`

**Files:**
- Modify: `olmlx/routers/responses.py` (two endpoints)
- Test: `tests/test_routers_responses.py` (append `TestRetrieveDelete`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_routers_responses.py`:

```python
class TestRetrieveDelete:
    @pytest.fixture(autouse=True)
    def _clear_store(self):
        get_store().clear()
        yield
        get_store().clear()

    @pytest.mark.asyncio
    async def test_get_then_delete_lifecycle(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            created = await app_client.post(
                "/v1/responses", json={"model": "qwen3", "input": "hi"}
            )
        rid = created.json()["id"]

        got = await app_client.get(f"/v1/responses/{rid}")
        assert got.status_code == 200
        assert got.json()["id"] == rid

        deleted = await app_client.delete(f"/v1/responses/{rid}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True

        gone = await app_client.get(f"/v1/responses/{rid}")
        assert gone.status_code == 404

    @pytest.mark.asyncio
    async def test_get_unknown_404(self, app_client):
        resp = await app_client.get("/v1/responses/resp_unknown")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_unknown_404(self, app_client):
        resp = await app_client.delete("/v1/responses/resp_unknown")
        assert resp.status_code == 404
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_routers_responses.py::TestRetrieveDelete -v`
Expected: FAIL with 405/404 (routes not defined).

- [ ] **Step 3: Implement the endpoints**

Append to `olmlx/routers/responses.py`:

```python
@router.get("/v1/responses/{response_id}")
async def get_response(response_id: str):
    entry = get_store().get(response_id)
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"response not found: {response_id!r}"
        )
    return entry["response"]


@router.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str):
    if not get_store().delete(response_id):
        raise HTTPException(
            status_code=404, detail=f"response not found: {response_id!r}"
        )
    return {"id": response_id, "object": "response.deleted", "deleted": True}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_routers_responses.py::TestRetrieveDelete -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add olmlx/routers/responses.py tests/test_routers_responses.py
git commit -m "feat(responses): GET/DELETE /v1/responses/{id} (#368)"
```

---

## Task 9: Live SDK acceptance test

**Files:**
- Create: `tests/live/test_responses_sdk.py`
- Possibly modify: `pyproject.toml` (add `openai` to the test/dev deps if absent)

This is a `real_model` test that drives the actual OpenAI Python SDK against a running
olmlx, satisfying issue #368's acceptance criteria. It lives in `tests/live/`
(outside `tests/integration/`) to dodge that directory's autouse MLX mock — the same
placement convention as `tests/live/test_vlm_tools_images.py`.

- [ ] **Step 1: Confirm the `openai` package is available for tests**

Run: `uv run python -c "import openai; print(openai.__version__)"`
If it errors, add `openai` to the dev dependency group in `pyproject.toml` and run
`uv sync --no-editable`. (Check the existing live tests first — they may already
depend on it.)

- [ ] **Step 2: Write the live test**

Create `tests/live/test_responses_sdk.py`:

```python
"""Live acceptance test: OpenAI SDK against olmlx /v1/responses (#368).

Marked real_model — requires a real model and a running server fixture. Skipped
in the default unit run. Mirrors tests/live/test_vlm_tools_images.py placement so
the integration autouse MLX mock does not apply.
"""

import json

import pytest

pytestmark = pytest.mark.real_model

# NOTE: `live_server_url` and `live_model` are the existing live fixtures used by
# tests/live/test_vlm_tools_images.py. If they are named differently in
# tests/live/conftest.py, use those names instead.


def _client(base_url):
    from openai import OpenAI

    return OpenAI(base_url=f"{base_url}/v1", api_key="not-needed")


def test_text(live_server_url, live_model):
    client = _client(live_server_url)
    resp = client.responses.create(model=live_model, input="Say 'pong'.")
    assert resp.status == "completed"
    assert resp.output_text  # SDK convenience accessor


def test_streaming_text(live_server_url, live_model):
    client = _client(live_server_url)
    chunks = []
    with client.responses.stream(model=live_model, input="Count: 1 2 3") as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
        final = stream.get_final_response()
    assert "".join(chunks)
    assert final.status == "completed"


def test_tool_use(live_server_url, live_model):
    client = _client(live_server_url)
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    resp = client.responses.create(
        model=live_model,
        input="What's the weather in Paris? Use the tool.",
        tools=tools,
    )
    calls = [it for it in resp.output if it.type == "function_call"]
    assert calls, "expected a function_call output item"
    assert calls[0].name == "get_weather"
    json.loads(calls[0].arguments)  # arguments parse as JSON
```

- [ ] **Step 3: Run the live test (manual / real-model env)**

Run: `uv run pytest tests/live/test_responses_sdk.py -v -m real_model`
Expected: PASS against a real model. (Skipped in CI default run; run locally with a
small model loaded.)

If the live fixtures are named differently, inspect `tests/live/conftest.py` and
adjust the fixture names — do not invent fixtures.

- [ ] **Step 4: Commit**

```bash
git add tests/live/test_responses_sdk.py pyproject.toml
git commit -m "test(responses): live OpenAI SDK acceptance test (#368)"
```

---

## Task 10: Final verification

- [ ] **Step 1: Run the full responses + state test set**

Run: `uv run pytest tests/test_routers_responses.py tests/test_responses_state.py -v`
Expected: PASS (all)

- [ ] **Step 2: Run the broader router + app suite for regressions**

Run: `uv run pytest tests/test_routers_openai.py tests/test_routers_anthropic.py tests/test_app.py -q`
Expected: PASS (no regressions from the `app.py` router registration).

- [ ] **Step 3: Lint + format (project requirement before push)**

Run: `uv run ruff check olmlx tests && uv run ruff format olmlx tests`
Expected: no errors; formatting clean.

- [ ] **Step 4: Update docs**

Add a short `/v1/responses` entry to the user manual / README where the other OpenAI
endpoints (`/v1/chat/completions`, `/v1/embeddings`) are documented, noting: supported
(text, streaming, function tools, `previous_response_id`, reasoning items, image input,
structured outputs via `text.format`); not supported (built-in tools). Add a one-line
note to `CLAUDE.md`'s router list under `routers/`.

- [ ] **Step 5: Commit docs**

```bash
git add -A
git commit -m "docs(responses): document /v1/responses endpoint (#368)"
```

---

## Self-review notes (for the implementer)

- **Spec coverage:** text non-stream (T4), streaming full events (T7), function tools
  (T5, T7), `previous_response_id` + cache_id (T6), reasoning items (T5), image input
  (T5), structured outputs (T5), state store + `store:false` (T6), GET/DELETE (T8),
  error paths 400/422/404 (T5, T6, T8), live SDK acceptance (T9). All spec sections map
  to a task.
- **404 vs 422 boundary:** unknown `previous_response_id` is a 404 (resource lookup);
  malformed input items / bad image / built-in tool / bad schema are 422 (request
  shape). `_history_messages_from_store` raises `HTTPException(404)` and must stay
  outside the `try/except ValueError` that maps translation errors to 422.
- **Type consistency:** helper names are stable across tasks — `_build_input_messages`,
  `_convert_tools`, `_grammar_from_text_format`, `_resolve_reasoning`,
  `_build_output_items`, `_build_response_object`, `_history_messages_from_store`,
  `_stream_response`. Store entries always carry `input_messages`, `output_items`,
  `model`, `previous_response_id`, `response`.
- **Engine untouched:** every task calls `generate_chat` with the existing keyword
  signature; no engine edits.
```
