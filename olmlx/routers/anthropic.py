import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from olmlx.config import settings
from olmlx.engine.inference import (
    INIT_ORPHAN_DETECT_LIMIT,
    _inference_ref,
    count_chat_tokens,
    generate_chat,
)
from olmlx.routers.common import build_inference_options
from olmlx.engine.tool_parser import (
    _make_tool_use_id,
    parse_model_output,
    resolve_tool_names,
)
from olmlx.schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicTokenCountResponse,
    AnthropicUsage,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_THINKING_TYPE_MAP: dict[str, bool] = {
    "enabled": True,
    "disabled": False,
    "adaptive": True,
}


def _build_anthropic_model_map() -> list[tuple[str, str]]:
    """Pre-sort anthropic_models by key length descending, filtering invalid entries.

    Keys with dashes/colons are rejected by the Settings validator; this only
    filters empty/whitespace entries that slip through.
    """
    entries = [
        (family.strip().lower(), local_model.strip())
        for family, local_model in settings.anthropic_models.items()
        if family.strip() and local_model.strip()
    ]
    return sorted(entries, key=lambda x: (-len(x[0]), x[0]))


_anthropic_model_map = _build_anthropic_model_map()


def _resolve_anthropic_model(model: str) -> str:
    """Resolve Claude model names to local models via config mapping.

    Matches family keywords against whole segments (split on - and :) in the
    model name. Longer keys take priority.
    """
    if not _anthropic_model_map:
        return model
    segments = model.lower().replace(":", "-").split("-")
    for family, local_model in _anthropic_model_map:
        if family in segments:
            logger.info(
                "Resolved Anthropic model %s → %s (family: %s)",
                model,
                local_model,
                family,
            )
            return local_model
    return model


THINKING_CHUNK_SIZE = 1000
TEXT_CHUNK_SIZE = 100
KEEPALIVE_PING_INTERVAL = 5.0


def _make_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


# --- Tool/message conversion ---


def _convert_tools(req: AnthropicMessagesRequest) -> list[dict] | None:
    """Convert Anthropic tool definitions to OpenAI-style for chat templates."""
    if not req.tools:
        return None
    tools = []
    for tool in req.tools:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema.model_dump(exclude_none=True),
                },
            }
        )
    return tools


BILLING_HEADER_PREFIX = "x-anthropic-billing-header"


def _strip_billing_headers(
    system: str | list[AnthropicContentBlock] | None,
) -> str | list[AnthropicContentBlock] | None:
    """Strip Claude Code billing header blocks/lines from system prompt.

    These headers change every request and break KV cache prefix matching.
    """
    if system is None:
        return None

    if isinstance(system, str):
        lines = system.split("\n")
        filtered = [
            line for line in lines if not line.startswith(BILLING_HEADER_PREFIX)
        ]
        if len(filtered) < len(lines):
            logger.info("Stripped billing header from string system prompt")
        result = "\n".join(filtered)
        if len(filtered) < len(lines):
            result = result.strip()
        return result if result else None

    # list[AnthropicContentBlock]
    filtered = [
        b for b in system if not (b.text and b.text.startswith(BILLING_HEADER_PREFIX))
    ]
    if len(filtered) < len(system):
        logger.info(
            "Stripped %d billing header block(s) from system prompt",
            len(system) - len(filtered),
        )
    return filtered if filtered else None


def _convert_messages(req: AnthropicMessagesRequest) -> list[dict]:
    """Convert Anthropic message format to internal chat format.

    Handles text, tool_use, and tool_result content blocks.
    """
    messages = []

    # System message (strip billing headers to preserve KV cache)
    system = _strip_billing_headers(req.system)
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        else:
            text = " ".join(b.text for b in system if b.text)
            messages.append({"role": "system", "content": text})

    for msg in req.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
            continue

        if msg.role == "assistant":
            text_parts = []
            tool_calls = []
            for block in msg.content:
                if block.type == "thinking":
                    # Skip thinking blocks in history — model regenerates its own
                    continue
                elif block.type == "text" and block.text:
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.id or _make_tool_use_id(),
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": block.input or {},
                            },
                        }
                    )

            entry = {
                "role": "assistant",
                "content": " ".join(text_parts) if text_parts else "",
            }
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)

        elif msg.role == "user":
            text_parts = []
            for block in msg.content:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
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
            if text_parts:
                messages.append({"role": "user", "content": " ".join(text_parts)})

    return messages


def _build_options(req: AnthropicMessagesRequest) -> dict:
    return build_inference_options(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=req.stop_sequences,
    )


# --- SSE helpers ---


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _signature_delta_sse(block_idx: int, signature: str = "") -> str:
    """Emit a `signature_delta` event for a thinking block.

    The Anthropic SDK populates `ThinkingBlock.signature` from this delta; we
    have nothing to sign for non-Claude models, so emit an empty string before
    `content_block_stop`.
    """
    return _sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {"type": "signature_delta", "signature": signature},
        },
    )


_PING_SENTINEL = object()


async def _with_keepalive_pings(
    aiter: AsyncIterator[Any],
    interval: float = KEEPALIVE_PING_INTERVAL,
) -> AsyncIterator[Any]:
    """Yield items from aiter; yield _PING_SENTINEL if no item arrives within interval seconds."""
    ait = aiter.__aiter__()
    next_item_task = None
    try:
        while True:
            if next_item_task is None:
                next_item_task = asyncio.ensure_future(ait.__anext__())
            done, _ = await asyncio.wait({next_item_task}, timeout=interval)
            if done:
                try:
                    item = next_item_task.result()
                except StopAsyncIteration:
                    return
                next_item_task = None
                yield item
            else:
                yield _PING_SENTINEL
    finally:
        if next_item_task is not None and not next_item_task.done():
            next_item_task.cancel()
            try:
                await next_item_task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass


def _emit_content_block(
    block_idx: int,
    block_type: str,
    delta_type: str,
    content_key: str,
    content: str,
    chunk_size: int,
    signature: str | None = None,
) -> list[str]:
    """Emit SSE strings for a complete content block (start + deltas + stop).

    When `signature` is provided (thinking blocks), a `signature_delta` is
    emitted before `content_block_stop` so the Anthropic SDK can populate
    `ThinkingBlock.signature`.
    """
    events = []
    events.append(
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {"type": block_type, content_key: ""},
            },
        )
    )
    if content:
        for i in range(0, len(content), chunk_size):
            events.append(
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {
                            "type": delta_type,
                            content_key: content[i : i + chunk_size],
                        },
                    },
                )
            )
    if signature is not None:
        events.append(_signature_delta_sse(block_idx, signature))
    events.append(
        _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
    )
    return events


async def _stream_buffered_with_tools(
    result: AsyncIterator[dict[str, Any]],
    declared_tools: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str | dict[str, Any]]:
    """Buffer full output, parse tools, yield SSE strings. Yields a final dict with metadata.

    Buffering is load-bearing for correctness when tools are present and cannot
    be replaced with incremental parsing — see ``parse_model_output`` in
    ``engine/tool_parser.py``:

    1. The parser tries 9 formats in priority order (Qwen, Mistral, Llama,
       DeepSeek, MiniMax, gemma4, standalone <function=...>, bare JSON,
       gpt-oss channels). The first parser to match wins, so an incremental
       parser would commit to a guess before higher-priority formats had a
       chance to appear.
    2. ``_try_bare_json`` requires brace-balanced JSON across the whole output.
       A ``{`` token could be visible text or the start of a tool call; once
       streamed as text it can't be retracted.
    3. ``_try_xml_func`` matches ``<function=...>`` anywhere in the output,
       including inside prose — indefinite lookahead would be required on
       every ``<`` token.
    4. ``_parse_gpt_oss_channels`` classifies blocks only at their terminating
       marker (``<|end|>``/``<|call|>``/``<|return|>``).
    5. Anthropic SSE requires ``content_block_start`` to carry the tool's
       ``id`` and ``name`` upfront, so the JSON body must be parsed before
       any tool events can be emitted.

    Keepalive ping events (see ``_with_keepalive_pings``) prevent connection
    timeouts during the buffering window.
    """
    text_chunks: list[str] = []
    raw_text = ""
    output_tokens = 0
    # Engine meta chunk arrives before any text — capture it so the orphan
    # `</think>` heuristic in `parse_model_output` is gated symmetrically
    # with the non-tools paths (issue #307 review round 5).
    thinking_expected = False

    async for chunk in _with_keepalive_pings(result, interval=KEEPALIVE_PING_INTERVAL):
        if chunk is _PING_SENTINEL:
            yield _sse("ping", {"type": "ping"})
            continue
        if isinstance(chunk, dict) and chunk.get("cache_info"):
            yield chunk  # Forward to stream_sse for message_start
            continue
        if isinstance(chunk, dict) and "thinking_expected" in chunk:
            thinking_expected = bool(chunk["thinking_expected"])
            continue
        if chunk.get("done"):
            stats = chunk.get("stats")
            if stats:
                output_tokens = stats.eval_count
            # For gpt-oss channel format, raw_text is in the done chunk
            raw_text = chunk.get("raw_text", "")
            done_reason = chunk.get("done_reason")
            break
        text_chunks.append(chunk.get("text", ""))
    else:
        done_reason = None

    full_text = "".join(text_chunks)

    if raw_text:
        logger.info(
            "Raw model output (%d chars before filter, %d chars visible): %s",
            len(raw_text),
            len(full_text),
            raw_text[:1000],
        )
        text_for_parsing = raw_text
    else:
        logger.info("Model output (%d chars): %s", len(full_text), full_text[:1000])
        text_for_parsing = full_text

    thinking, visible_text, tool_uses = parse_model_output(
        text_for_parsing,
        True,
        thinking_expected=thinking_expected,
    )

    resolve_tool_names(tool_uses, declared_tools)

    if tool_uses:
        logger.info(
            "Parsed %d tool call(s): %s",
            len(tool_uses),
            [tu["name"] for tu in tool_uses],
        )

    block_idx = 0

    if thinking:
        for event in _emit_content_block(
            block_idx,
            "thinking",
            "thinking_delta",
            "thinking",
            thinking,
            THINKING_CHUNK_SIZE,
            signature="",
        ):
            yield event
        block_idx += 1

    for event in _emit_content_block(
        block_idx, "text", "text_delta", "text", visible_text, TEXT_CHUNK_SIZE
    ):
        yield event
    block_idx += 1

    for tool_use in tool_uses:
        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_use["id"],
                    "name": tool_use["name"],
                    "input": {},
                },
            },
        )
        yield _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(tool_use["input"]),
                },
            },
        )
        yield _sse(
            "content_block_stop", {"type": "content_block_stop", "index": block_idx}
        )
        block_idx += 1

    if tool_uses:
        stop_reason = "tool_use"
    elif done_reason == "timeout":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    yield {
        "stop_reason": stop_reason,
        "output_tokens": output_tokens,
    }


def _flush_thinking_buffer(
    state: str, buffer: str, block_idx: int, text_block_started: bool
) -> tuple[list[str], int]:
    """Flush remaining buffer at end of stream. Returns (sse_events, updated_block_idx)."""
    events: list[str] = []
    if state == "thinking":
        if buffer:
            events.append(
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "thinking_delta", "thinking": buffer},
                    },
                )
            )
        events.append(_signature_delta_sse(block_idx))
        events.append(
            _sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": block_idx},
            )
        )
        block_idx += 1
        events.append(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        events.append(
            _sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": block_idx},
            )
        )
    elif text_block_started:
        events.append(
            _sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": block_idx},
            )
        )
    elif state == "text" and not text_block_started:
        events.append(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        if buffer:
            events.append(
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "text_delta", "text": buffer},
                    },
                )
            )
        events.append(
            _sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": block_idx},
            )
        )
    else:
        # state == "init" — the stream ended before we could classify the
        # leading buffer.  This happens when `thinking_expected` held the
        # init state waiting for an orphan `</think>` that never arrived
        # (issue #307 — short non-thinking output on a thinking-capable
        # model).  Emit any held content as a text block so the response
        # isn't dropped.
        events.append(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        if buffer:
            events.append(
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "text_delta", "text": buffer},
                    },
                )
            )
        events.append(
            _sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": block_idx},
            )
        )
    return events, block_idx


async def _stream_thinking_state_machine(result):
    """Stream incrementally with thinking state machine. Yields a final dict with metadata."""
    block_idx = 0
    output_tokens = 0
    buffer = ""
    state = "init"  # "init", "thinking", "text"
    text_block_started = False
    done_reason = None
    # Engine forwards this via a meta chunk; when set, we wait for an orphan
    # `</think>` to classify the leading buffer as thinking instead of text
    # (issue #307 — Qwen3.5/3.6 emit thinking without the `<think>` opener).
    thinking_expected = False

    async for chunk in _with_keepalive_pings(result, interval=KEEPALIVE_PING_INTERVAL):
        if chunk is _PING_SENTINEL:
            yield _sse("ping", {"type": "ping"})
            continue
        if isinstance(chunk, dict) and chunk.get("cache_info"):
            yield chunk  # Forward to stream_sse for message_start
            continue
        if isinstance(chunk, dict) and "thinking_expected" in chunk:
            thinking_expected = bool(chunk["thinking_expected"])
            continue
        if chunk.get("done"):
            stats = chunk.get("stats")
            if stats:
                output_tokens = stats.eval_count
            done_reason = chunk.get("done_reason")
            break

        token_text = chunk.get("text", "")
        buffer += token_text

        while buffer:
            if state == "init":
                open_idx = buffer.find("<think>")
                close_idx = buffer.find("</think>")
                if buffer.startswith("<think>"):
                    state = "thinking"
                    buffer = buffer[7:]
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {"type": "thinking", "thinking": ""},
                        },
                    )
                elif (
                    close_idx >= 0
                    and thinking_expected
                    and (open_idx == -1 or close_idx < open_idx)
                ):
                    # Orphaned `</think>` — Qwen3.5/3.6 chat templates miss
                    # the `<think>` opener but still emit the closer (#307).
                    # Everything buffered before it is the thinking block.
                    # Gated on `thinking_expected` so a non-thinking model
                    # that legitimately mentions the literal `</think>`
                    # token isn't silently reclassified as thinking, and on
                    # the open/close order so that a buffer assembled with
                    # `preamble<think>...</think>answer` (text-before-tag,
                    # uncommon but real for slow first tokens) routes through
                    # the standard `<think>` branch instead of swallowing the
                    # `<think>` opener as orphan content.
                    orphan_thinking = buffer[:close_idx]
                    # Skip the whole content block when `</think>` is the
                    # very first token — emitting an empty thinking block
                    # would diverge from the non-streaming path which
                    # produces no block in that case (issue #307 review
                    # round 10).
                    if orphan_thinking:
                        yield _sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": {"type": "thinking", "thinking": ""},
                            },
                        )
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block_idx,
                                "delta": {
                                    "type": "thinking_delta",
                                    "thinking": orphan_thinking,
                                },
                            },
                        )
                        yield _signature_delta_sse(block_idx)
                        yield _sse(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": block_idx},
                        )
                        block_idx += 1
                    buffer = buffer[close_idx + len("</think>") :].lstrip("\n")
                    state = "text"
                elif len(buffer) < 7 and "<think>".startswith(buffer):
                    break
                elif (
                    thinking_expected
                    and close_idx == -1
                    and len(buffer) <= INIT_ORPHAN_DETECT_LIMIT
                ):
                    # Keep waiting for a (possibly orphaned) `</think>`.
                    # Gated on `close_idx == -1`: once we've seen `</think>`
                    # we have enough information to make a decision (either
                    # the orphan branch above already fired, or the text
                    # state will emit the buffered content) — no point
                    # waiting longer.
                    # NOTE: this can delay the first `text_delta` event by
                    # up to `INIT_ORPHAN_DETECT_LIMIT` characters (currently
                    # 1024) for a thinking-capable model that produces a
                    # short direct answer with no `</think>`; the keep-alive
                    # ping loop covers the wait, and the buffered content
                    # is emitted at stream end via `_flush_thinking_buffer`.
                    break
                else:
                    state = "text"

            elif state == "thinking":
                end_idx = buffer.find("</think>")
                if end_idx >= 0:
                    thinking_chunk = buffer[:end_idx]
                    if thinking_chunk:
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block_idx,
                                "delta": {
                                    "type": "thinking_delta",
                                    "thinking": thinking_chunk,
                                },
                            },
                        )
                    yield _signature_delta_sse(block_idx)
                    yield _sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": block_idx},
                    )
                    block_idx += 1
                    buffer = buffer[end_idx + len("</think>") :]
                    state = "text"
                elif len(buffer) > len("</think>"):
                    safe = buffer[: -len("</think>")]
                    buffer = buffer[-len("</think>") :]
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "thinking_delta", "thinking": safe},
                        },
                    )
                    break
                else:
                    break

            elif state == "text":
                if not text_block_started:
                    yield _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    text_block_started = True
                if buffer:
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "text_delta", "text": buffer},
                        },
                    )
                    buffer = ""
                break

    # Flush remaining buffer
    flush_events, block_idx = _flush_thinking_buffer(
        state, buffer, block_idx, text_block_started
    )
    for event in flush_events:
        yield event

    yield {
        "stop_reason": "max_tokens" if done_reason == "timeout" else "end_turn",
        "output_tokens": output_tokens,
    }


@router.post(
    "/v1/messages/count_tokens",
    response_model=AnthropicTokenCountResponse,
    response_model_exclude_none=True,
)
async def anthropic_count_tokens(req: AnthropicMessagesRequest, request: Request):
    logger.info(
        "Anthropic count_tokens: model=%s messages=%d tools=%d",
        req.model,
        len(req.messages),
        len(req.tools or []),
    )
    resolved_model = _resolve_anthropic_model(req.model)
    manager = request.app.state.model_manager
    lm = await manager.ensure_loaded(resolved_model, pin=True)

    try:
        messages = _convert_messages(req)
        tools = _convert_tools(req)

        enable_thinking: bool | None = None
        if req.thinking is not None:
            enable_thinking = _THINKING_TYPE_MAP.get(req.thinking.type)

        with _inference_ref(lm, adopt=True):
            loop = asyncio.get_running_loop()
            token_count = await loop.run_in_executor(
                None,
                partial(
                    count_chat_tokens,
                    lm.text_tokenizer,
                    messages,
                    tools,
                    lm.template_caps,
                    enable_thinking=enable_thinking,
                ),
            )
        return AnthropicTokenCountResponse(input_tokens=token_count)
    finally:
        lm.release_ref()


@router.post(
    "/v1/messages",
    response_model=AnthropicMessagesResponse,
    response_model_exclude_none=True,
)
async def anthropic_messages(req: AnthropicMessagesRequest, request: Request):
    logger.info(
        "Anthropic request: model=%s stream=%s tools=%d messages=%d max_tokens=%d",
        req.model,
        req.stream,
        len(req.tools or []),
        len(req.messages),
        req.max_tokens,
    )
    resolved_model = _resolve_anthropic_model(req.model)
    manager = request.app.state.model_manager
    messages = _convert_messages(req)
    options = _build_options(req)
    tools = _convert_tools(req)
    has_tools = bool(tools)
    msg_id = _make_msg_id()
    logger.debug("Converted %d messages, %d tools", len(messages), len(tools or []))

    cache_id = request.headers.get("x-cache-id", "")[:256]

    enable_thinking: bool | None = None
    if req.thinking is not None:
        enable_thinking = _THINKING_TYPE_MAP.get(req.thinking.type)
        logger.debug(
            "enable_thinking=%s (thinking.type=%s)", enable_thinking, req.thinking.type
        )
        if req.thinking.budget_tokens is not None:
            logger.info(
                "budget_tokens=%d received but not supported (thinking is on/off only)",
                req.thinking.budget_tokens,
            )

    if req.stream:
        result = await generate_chat(
            manager,
            resolved_model,
            messages,
            options,
            tools=tools,
            stream=True,
            max_tokens=req.max_tokens,
            cache_id=cache_id,
            enable_thinking=enable_thinking,
        )

        async def stream_sse():
            path = None
            try:
                path = (
                    _stream_buffered_with_tools(result, declared_tools=tools)
                    if has_tools
                    else _stream_thinking_state_machine(result)
                )

                # Emit message_start as early as possible so the client
                # receives keep-alive pings during prefill instead of
                # buffering them until the first content token arrives.
                #
                # Phase 1: Look for cache_info.  Any pings that arrive
                #          before it are buffered (this window is <1 ms).
                # Phase 2: After cache_info (or first ping if no cache),
                #          emit message_start and yield pings directly.
                cache_read = 0
                cache_creation = 0
                message_started = False
                pending_pings: list[str] = []

                def _emit_message_start():
                    nonlocal message_started
                    message_started = True
                    msg_data: dict[str, Any] = {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": req.model,
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cache_creation_input_tokens": cache_creation,
                            "cache_read_input_tokens": cache_read,
                        },
                    }
                    return _sse(
                        "message_start",
                        {"type": "message_start", "message": msg_data},
                    )

                meta = {}
                async for event in path:
                    if isinstance(event, dict) and event.get("cache_info"):
                        cache_read = event.get("cache_read_tokens") or 0
                        cache_creation = event.get("cache_creation_tokens") or 0
                        logger.debug(
                            "Cache stats for message_start: read=%d creation=%d",
                            cache_read,
                            cache_creation,
                        )
                        # Emit message_start immediately with cache stats
                        if not message_started:
                            yield _emit_message_start()
                        else:
                            logger.warning(
                                "Duplicate cache_info received after message_start "
                                "(read=%d, creation=%d) — stats dropped",
                                cache_read,
                                cache_creation,
                            )
                        # Replay any pings that arrived before cache_info
                        for ping in pending_pings:
                            yield ping
                        pending_pings.clear()
                    elif isinstance(event, str) and event.startswith("event: ping"):
                        if message_started:
                            # After message_start: yield pings directly to client
                            yield event
                        else:
                            # Before cache_info or content: buffer pings.
                            # cache_info arrives within ms (before first 5s
                            # ping), so this buffer is normally empty.
                            pending_pings.append(event)
                    elif isinstance(event, dict):
                        meta = event
                    else:
                        if not message_started:
                            # Content arrived before cache_info — no-cache case
                            yield _emit_message_start()
                            for ping in pending_pings:
                                yield ping
                            pending_pings.clear()
                        yield event

                if not message_started:
                    yield _emit_message_start()
                    for ping in pending_pings:
                        yield ping

                yield _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": meta.get("stop_reason", "end_turn"),
                        },
                        "usage": {
                            "output_tokens": meta.get("output_tokens", 0),
                        },
                    },
                )
                yield _sse("message_stop", {"type": "message_stop"})
            except Exception as exc:
                logger.error("Error during Anthropic streaming: %s", exc, exc_info=True)
                yield _sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": "An internal server error occurred during streaming.",
                        },
                    },
                )
            finally:
                try:
                    if path is not None:
                        await path.aclose()
                finally:
                    await result.aclose()

        return StreamingResponse(stream_sse(), media_type="text/event-stream")
    else:
        result = await generate_chat(
            manager,
            resolved_model,
            messages,
            options,
            tools=tools,
            stream=False,
            max_tokens=req.max_tokens,
            cache_id=cache_id,
            enable_thinking=enable_thinking,
        )
        text = result.get("text", "")
        stats = result.get("stats")
        thinking = result.get("thinking", "")

        logger.debug("Raw model output (%d chars): %s", len(text), text[:500])

        # For gpt-oss, tool_uses may already be pre-parsed in generate_chat
        pre_parsed_tool_uses = result.get("tool_uses")

        if pre_parsed_tool_uses is not None:
            # Already parsed by _parse_gpt_oss_channels in generate_chat
            tool_uses = pre_parsed_tool_uses
            visible_text = text
            resolve_tool_names(tool_uses, tools)
        else:
            # Pass `thinking_expected` so the orphan-`</think>` heuristic
            # only fires when the engine actually requested thinking
            # (issue #307 review — symmetric with the Ollama fix).
            thinking_parsed, visible_text, tool_uses = parse_model_output(
                text,
                has_tools,
                thinking_expected=bool(result.get("thinking_expected")),
            )
            thinking = thinking or thinking_parsed
            resolve_tool_names(tool_uses, tools)

        content_blocks = []

        if thinking:
            content_blocks.append(
                AnthropicContentBlock(type="thinking", thinking=thinking, signature="")
            )

        if visible_text:
            content_blocks.append(AnthropicContentBlock(type="text", text=visible_text))

        for tu in tool_uses:
            content_blocks.append(
                AnthropicContentBlock(
                    type="tool_use",
                    id=tu["id"],
                    name=tu["name"],
                    input=tu["input"],
                )
            )

        if not content_blocks:
            content_blocks.append(AnthropicContentBlock(type="text", text=""))

        done_reason = result.get("done_reason")
        if tool_uses:
            stop_reason = "tool_use"
        elif done_reason == "timeout":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"
        usage = AnthropicUsage(
            input_tokens=stats.prompt_eval_count if stats else 0,
            output_tokens=stats.eval_count if stats else 0,
            cache_creation_input_tokens=result.get("cache_creation_tokens") or 0,
            cache_read_input_tokens=result.get("cache_read_tokens") or 0,
        )
        return AnthropicMessagesResponse(
            id=msg_id,
            content=content_blocks,
            model=req.model,
            stop_reason=stop_reason,
            usage=usage,
        )
