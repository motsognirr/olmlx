import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from olmlx.engine.inference import count_chat_tokens, generate_chat
from olmlx.engine.tool_parser import _make_tool_use_id, parse_model_output
from olmlx.schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicTokenCountResponse,
    AnthropicUsage,
)

router = APIRouter()
logger = logging.getLogger(__name__)

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


def _convert_messages(req: AnthropicMessagesRequest) -> list[dict]:
    """Convert Anthropic message format to internal chat format.

    Handles text, tool_use, and tool_result content blocks.
    """
    messages = []

    # System message
    if req.system:
        if isinstance(req.system, str):
            messages.append({"role": "system", "content": req.system})
        else:
            text = " ".join(b.text for b in req.system if b.text)
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
    opts = {}
    if req.temperature is not None:
        opts["temperature"] = req.temperature
    if req.top_p is not None:
        opts["top_p"] = req.top_p
    if req.top_k is not None:
        opts["top_k"] = req.top_k
    if req.stop_sequences:
        opts["stop"] = req.stop_sequences
    return opts


# --- SSE helpers ---


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


_PING_SENTINEL = object()


async def _with_keepalive_pings(aiter, interval=KEEPALIVE_PING_INTERVAL):
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
) -> list[str]:
    """Emit SSE strings for a complete content block (start + deltas + stop)."""
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
    events.append(
        _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
    )
    return events


async def _stream_buffered_with_tools(result, tool_names):
    """Buffer full output, parse tools, yield SSE strings. Yields a final dict with metadata."""
    full_text = ""
    output_tokens = 0

    async for chunk in _with_keepalive_pings(result, interval=KEEPALIVE_PING_INTERVAL):
        if chunk is _PING_SENTINEL:
            yield _sse("ping", {"type": "ping"})
            continue
        if isinstance(chunk, dict) and chunk.get("cache_info"):
            yield chunk  # Forward to stream_sse for message_start
            continue
        if chunk.get("done"):
            stats = chunk.get("stats")
            if stats:
                output_tokens = stats.eval_count
            break
        full_text += chunk.get("text", "")

    await result.aclose()

    logger.info("Raw model output (%d chars): %s", len(full_text), full_text[:1000])

    thinking, visible_text, tool_uses = parse_model_output(
        full_text,
        True,
        tool_names=tool_names,
    )

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

    yield {
        "stop_reason": "tool_use" if tool_uses else "end_turn",
        "output_tokens": output_tokens,
    }


async def _stream_thinking_state_machine(result):
    """Stream incrementally with thinking state machine. Yields a final dict with metadata."""
    block_idx = 0
    output_tokens = 0
    buffer = ""
    state = "init"  # "init", "thinking", "text"
    text_block_started = False

    async for chunk in _with_keepalive_pings(result, interval=KEEPALIVE_PING_INTERVAL):
        if chunk is _PING_SENTINEL:
            yield _sse("ping", {"type": "ping"})
            continue
        if isinstance(chunk, dict) and chunk.get("cache_info"):
            yield chunk  # Forward to stream_sse for message_start
            continue
        if chunk.get("done"):
            stats = chunk.get("stats")
            if stats:
                output_tokens = stats.eval_count
            break

        token_text = chunk.get("text", "")
        buffer += token_text

        while buffer:
            if state == "init":
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
                elif len(buffer) < 7 and "<think>".startswith(buffer):
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
                    yield _sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": block_idx},
                    )
                    block_idx += 1
                    buffer = buffer[end_idx + 8 :]
                    state = "text"
                elif len(buffer) > 8:
                    safe = buffer[:-8]
                    buffer = buffer[-8:]
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

    await result.aclose()

    # Flush remaining buffer
    if state == "thinking":
        if buffer:
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "thinking_delta", "thinking": buffer},
                },
            )
        yield _sse(
            "content_block_stop", {"type": "content_block_stop", "index": block_idx}
        )
        block_idx += 1
        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {"type": "text", "text": ""},
            },
        )
        yield _sse(
            "content_block_stop", {"type": "content_block_stop", "index": block_idx}
        )
    elif text_block_started:
        yield _sse(
            "content_block_stop", {"type": "content_block_stop", "index": block_idx}
        )
    elif state == "text" and not text_block_started:
        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {"type": "text", "text": ""},
            },
        )
        if buffer:
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "text_delta", "text": buffer},
                },
            )
        yield _sse(
            "content_block_stop", {"type": "content_block_stop", "index": block_idx}
        )
    else:
        # state == "init" — no output at all, emit empty text block
        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {"type": "text", "text": ""},
            },
        )
        yield _sse(
            "content_block_stop", {"type": "content_block_stop", "index": block_idx}
        )

    yield {
        "stop_reason": "end_turn",
        "output_tokens": output_tokens,
    }


@router.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(req: AnthropicMessagesRequest, request: Request):
    logger.info(
        "Anthropic count_tokens: model=%s messages=%d tools=%d",
        req.model,
        len(req.messages),
        len(req.tools or []),
    )
    manager = request.app.state.model_manager
    lm = await manager.ensure_loaded(req.model)

    messages = _convert_messages(req)
    tools = _convert_tools(req)

    lm.active_refs += 1
    try:
        loop = asyncio.get_running_loop()
        token_count = await loop.run_in_executor(
            None,
            count_chat_tokens,
            lm.text_tokenizer,
            messages,
            tools,
            lm.template_caps,
        )
    finally:
        lm.active_refs -= 1
    return AnthropicTokenCountResponse(input_tokens=token_count)


@router.post("/v1/messages")
async def anthropic_messages(req: AnthropicMessagesRequest, request: Request):
    logger.info(
        "Anthropic request: model=%s stream=%s tools=%d messages=%d max_tokens=%d",
        req.model,
        req.stream,
        len(req.tools or []),
        len(req.messages),
        req.max_tokens,
    )
    manager = request.app.state.model_manager
    messages = _convert_messages(req)
    options = _build_options(req)
    tools = _convert_tools(req)
    has_tools = bool(tools)
    tool_names = {t["function"]["name"] for t in tools} if tools else None
    msg_id = _make_msg_id()
    logger.debug("Converted %d messages, %d tools", len(messages), len(tools or []))

    if req.stream:
        result = await generate_chat(
            manager,
            req.model,
            messages,
            options,
            tools=tools,
            stream=True,
            max_tokens=req.max_tokens,
        )

        async def stream_sse():
            try:
                path = (
                    _stream_buffered_with_tools(result, tool_names)
                    if has_tools
                    else _stream_thinking_state_machine(result)
                )

                # Consume cache_info if present (forwarded by helpers),
                # then emit message_start with accurate cache stats.
                # Keepalive pings may arrive before cache_info (during lock
                # wait), so skip them here and replay after message_start.
                cache_read = 0
                cache_creation = 0
                first_event = None
                pending_pings: list[str] = []
                async for event in path:
                    if isinstance(event, dict) and event.get("cache_info"):
                        cache_read = event.get("cache_read_tokens", 0)
                        cache_creation = event.get("cache_creation_tokens", 0)
                        logger.debug(
                            "Cache stats for message_start: read=%d creation=%d",
                            cache_read,
                            cache_creation,
                        )
                    elif isinstance(event, str) and event.startswith("event: ping"):
                        pending_pings.append(event)
                    else:
                        first_event = event
                        break

                yield _sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": req.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "cache_creation_input_tokens": cache_creation,
                                "cache_read_input_tokens": cache_read,
                            },
                        },
                    },
                )

                # Replay any pings that arrived before message_start
                for ping in pending_pings:
                    yield ping

                meta = {}
                if first_event is not None:
                    if isinstance(first_event, dict):
                        meta = first_event
                    else:
                        yield first_event

                async for event in path:
                    if isinstance(event, dict):
                        meta = event
                    else:
                        yield event

                yield _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": meta.get("stop_reason", "end_turn"),
                            "stop_sequence": None,
                        },
                        "usage": {
                            "output_tokens": meta.get("output_tokens", 0),
                        },
                    },
                )
                yield _sse("message_stop", {"type": "message_stop"})
            finally:
                await result.aclose()

        return StreamingResponse(stream_sse(), media_type="text/event-stream")
    else:
        result = await generate_chat(
            manager,
            req.model,
            messages,
            options,
            tools=tools,
            stream=False,
            max_tokens=req.max_tokens,
        )
        text = result.get("text", "")
        stats = result.get("stats")

        logger.debug("Raw model output (%d chars): %s", len(text), text[:500])

        thinking, visible_text, tool_uses = parse_model_output(
            text,
            has_tools,
            tool_names=tool_names,
        )

        content_blocks = []

        if thinking:
            content_blocks.append(AnthropicContentBlock(type="thinking", text=thinking))

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

        stop_reason = "tool_use" if tool_uses else "end_turn"
        usage = AnthropicUsage(
            input_tokens=stats.prompt_eval_count if stats else 0,
            output_tokens=stats.eval_count if stats else 0,
        )
        return AnthropicMessagesResponse(
            id=msg_id,
            content=content_blocks,
            model=req.model,
            stop_reason=stop_reason,
            usage=usage,
        )
