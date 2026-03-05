import json
import logging
import re
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from mlx_ollama.engine.inference import generate_chat
from mlx_ollama.schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicUsage,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _make_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _make_tool_use_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


# --- Tag parsing for model output ---

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Fallback: bare JSON tool calls (Llama-style)
_TOOL_CALL_JSON_RE = re.compile(
    r'\{"name"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|parameters)"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)
# Mistral-style [TOOL_CALLS]
_MISTRAL_TOOL_RE = re.compile(
    r"\[TOOL_CALLS\]\s*(\[.*?\])",
    re.DOTALL,
)


def _parse_model_output(text: str, has_tools: bool) -> tuple[str, str, list[dict]]:
    """Parse raw model output into (thinking_text, visible_text, tool_use_blocks).

    Handles:
    - <think>...</think> blocks (Qwen3/3.5 thinking)
    - <tool_call>...</tool_call> blocks (Qwen3/3.5 tool calling)
    - Bare JSON tool calls (Llama-style)
    - [TOOL_CALLS] prefix (Mistral-style)
    """
    thinking = ""
    tool_uses = []

    # Extract thinking blocks
    think_matches = _THINK_RE.findall(text)
    if think_matches:
        thinking = "\n".join(m.strip() for m in think_matches)
        text = _THINK_RE.sub("", text)

    # Extract tool calls (only if tools were provided)
    if has_tools:
        # Try <tool_call> tags first (Qwen format)
        for match in _TOOL_CALL_RE.finditer(text):
            try:
                call = json.loads(match.group(1))
                name = call.get("name", "")
                arguments = call.get("arguments") or call.get("parameters") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                tool_uses.append({
                    "type": "tool_use",
                    "id": _make_tool_use_id(),
                    "name": name,
                    "input": arguments,
                })
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning("Failed to parse <tool_call> block: %s", e)

        if tool_uses:
            text = _TOOL_CALL_RE.sub("", text)
        else:
            # Try Mistral [TOOL_CALLS] format
            mistral_match = _MISTRAL_TOOL_RE.search(text)
            if mistral_match:
                try:
                    calls = json.loads(mistral_match.group(1))
                    for call in calls:
                        name = call.get("name", "")
                        arguments = call.get("arguments") or call.get("parameters") or {}
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        tool_uses.append({
                            "type": "tool_use",
                            "id": _make_tool_use_id(),
                            "name": name,
                            "input": arguments,
                        })
                    text = _MISTRAL_TOOL_RE.sub("", text)
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning("Failed to parse [TOOL_CALLS] block: %s", e)

            if not tool_uses:
                # Try bare JSON tool calls
                for match in _TOOL_CALL_JSON_RE.finditer(text):
                    try:
                        name = match.group(1)
                        arguments = json.loads(match.group(2))
                        tool_uses.append({
                            "type": "tool_use",
                            "id": _make_tool_use_id(),
                            "name": name,
                            "input": arguments,
                        })
                    except (json.JSONDecodeError, AttributeError):
                        continue
                if tool_uses:
                    text = _TOOL_CALL_JSON_RE.sub("", text)

    visible_text = text.strip()
    return thinking, visible_text, tool_uses


# --- Tool/message conversion ---

def _convert_tools(req: AnthropicMessagesRequest) -> list[dict] | None:
    """Convert Anthropic tool definitions to OpenAI-style for chat templates."""
    if not req.tools:
        return None
    tools = []
    for tool in req.tools:
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema.model_dump(exclude_none=True),
            },
        })
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
                    tool_calls.append({
                        "id": block.id or _make_tool_use_id(),
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input or {}),
                        },
                    })

            entry = {"role": "assistant", "content": " ".join(text_parts) if text_parts else ""}
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
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.tool_use_id or "",
                        "content": result_content,
                    })
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


@router.post("/v1/messages")
async def anthropic_messages(req: AnthropicMessagesRequest, request: Request):
    manager = request.app.state.model_manager
    messages = _convert_messages(req)
    options = _build_options(req)
    tools = _convert_tools(req)
    has_tools = bool(tools)
    msg_id = _make_msg_id()

    if req.stream:
        result = await generate_chat(
            manager, req.model, messages, options,
            tools=tools, stream=True, max_tokens=req.max_tokens,
        )

        async def stream_sse():
            # message_start
            yield _sse("message_start", {
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
                        "input_tokens": 0, "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
            })

            # Accumulate the full model output
            full_text = ""
            output_tokens = 0
            async for chunk in result:
                if chunk.get("done"):
                    stats = chunk.get("stats")
                    if stats:
                        output_tokens = stats.eval_count
                    break
                full_text += chunk.get("text", "")
                output_tokens += 1  # approximate if stats unavailable

            logger.debug("Raw model output (%d chars): %s", len(full_text), full_text[:500])

            # Parse into thinking, text, and tool calls
            thinking, visible_text, tool_uses = _parse_model_output(full_text, has_tools)

            if tool_uses:
                logger.info("Parsed %d tool call(s): %s",
                            len(tool_uses),
                            [tu["name"] for tu in tool_uses])

            block_idx = 0

            # Emit thinking block if present
            if thinking:
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {"type": "thinking", "thinking": ""},
                })
                # Send thinking in chunks to avoid huge single events
                chunk_size = 1000
                for i in range(0, len(thinking), chunk_size):
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": thinking[i:i + chunk_size],
                        },
                    })
                yield _sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": block_idx,
                })
                block_idx += 1

            # Emit text block
            yield _sse("content_block_start", {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {"type": "text", "text": ""},
            })
            if visible_text:
                # Send text in chunks for a smoother feel
                chunk_size = 100
                for i in range(0, len(visible_text), chunk_size):
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {
                            "type": "text_delta",
                            "text": visible_text[i:i + chunk_size],
                        },
                    })
            yield _sse("content_block_stop", {
                "type": "content_block_stop",
                "index": block_idx,
            })
            block_idx += 1

            # Emit tool_use blocks
            for tool_use in tool_uses:
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_use["id"],
                        "name": tool_use["name"],
                        "input": {},
                    },
                })
                input_json = json.dumps(tool_use["input"])
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": input_json,
                    },
                })
                yield _sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": block_idx,
                })
                block_idx += 1

            # message_delta
            stop_reason = "tool_use" if tool_uses else "end_turn"
            yield _sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            })

            # message_stop
            yield _sse("message_stop", {"type": "message_stop"})

        return StreamingResponse(stream_sse(), media_type="text/event-stream")
    else:
        result = await generate_chat(
            manager, req.model, messages, options,
            tools=tools, stream=False, max_tokens=req.max_tokens,
        )
        text = result.get("text", "")
        stats = result.get("stats")

        logger.debug("Raw model output (%d chars): %s", len(text), text[:500])

        thinking, visible_text, tool_uses = _parse_model_output(text, has_tools)

        content_blocks = []

        if thinking:
            content_blocks.append(
                AnthropicContentBlock(type="thinking", text=thinking)
            )

        if visible_text:
            content_blocks.append(
                AnthropicContentBlock(type="text", text=visible_text)
            )

        for tu in tool_uses:
            content_blocks.append(AnthropicContentBlock(
                type="tool_use",
                id=tu["id"],
                name=tu["name"],
                input=tu["input"],
            ))

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
