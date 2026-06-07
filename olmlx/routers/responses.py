import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from olmlx.engine.grammar import GrammarSpec, parse_response_format
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
            name = item.get("name")
            if not name:
                raise ValueError("function_call item missing 'name'")
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id"),
                            "type": "function",
                            "function": {
                                "name": name,
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
        name = t.get("name")
        if not name:
            raise ValueError("function tool missing 'name'")
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                },
            }
        )
    return converted


def _grammar_from_text_format(text_cfg: dict | None) -> GrammarSpec | None:
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
                                # `name` is always present: set by _build_output_items invariant.
                                "name": item["name"],
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
        # reasoning items are not replayed into prompt history
    return messages


def _resolve_reasoning(reasoning: dict | None) -> bool | None:
    """Map Responses `reasoning.effort` to the engine enable_thinking flag."""
    effort = (reasoning or {}).get("effort")
    return resolve_openai_think(effort, None)


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
        parallel_tool_calls=req.parallel_tool_calls,
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

    try:
        new_messages = _build_input_messages(req.input)
        tools = _convert_tools(req.tools)
        grammar_spec = _grammar_from_text_format(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if req.previous_response_id:
        conversation = _history_messages_from_store(req.previous_response_id)
        conversation.extend(new_messages)
    else:
        conversation = new_messages

    # Per OpenAI Responses semantics, `instructions` are NOT carried across
    # previous_response_id — each turn supplies its own. So the system message
    # is applied only to the engine input and is NOT part of the stored
    # conversation (which is replayed on the next continuation turn).
    engine_messages = conversation
    if req.instructions:
        engine_messages = [
            {"role": "system", "content": req.instructions},
            *conversation,
        ]

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
        engine_messages,
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
    resolve_tool_names(tool_uses, tools)
    fill_missing_required_args(tool_uses, tools)

    output_items = _build_output_items(thinking, visible_text, tool_uses)
    response = _build_response_object(
        req,
        response_id,
        created,
        output_items,
        result.get("stats"),
        result.get("done_reason"),
    )
    if req.store:
        get_store().put(
            response_id,
            {
                "input_messages": conversation,
                "output_items": output_items,
                "model": req.model,
                "previous_response_id": req.previous_response_id,
                "response": response.model_dump(exclude_none=True),
            },
        )
    return response
