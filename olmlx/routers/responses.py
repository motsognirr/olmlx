import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from olmlx.config import settings
from olmlx.engine.grammar import GrammarSpec, parse_response_format
from olmlx.engine.inference import generate_chat
from olmlx.engine.responses_state import get_store
from olmlx.routers.streaming_common import (
    BufferedModelOutput,
    collect_stream,
    parse_buffered_output,
)
from olmlx.routers.common import build_inference_options, resolve_openai_think
from olmlx.routers.thinking_split import flush_split_thinking, split_thinking_parts
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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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
                                # Default to "{}" (valid JSON), never "" — an
                                # empty string breaks downstream JSON parsing of
                                # tool-call arguments.
                                "arguments": item.get("arguments") or "{}",
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
                                "arguments": item.get("arguments") or "{}",
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
    if done_reason in ("timeout", "length"):
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
        # The OpenAI SDK requires tool_choice to be non-null; default to "auto".
        tool_choice=req.tool_choice if req.tool_choice is not None else "auto",
        tools=req.tools or [],
    )


def _store_response(
    req: ResponsesRequest,
    response_id: str,
    conversation: list[dict],
    output_items: list[dict],
    response_dict: dict,
) -> None:
    """Persist a completed response for previous_response_id continuation.

    Shared by the streaming and non-streaming paths so the stored entry shape
    can't drift between them. No-op when ``store`` is false.
    """
    if not req.store:
        return
    get_store().put(
        response_id,
        {
            "input_messages": conversation,
            "output_items": output_items,
            "model": req.model,
            "previous_response_id": req.previous_response_id,
            "response": response_dict,
        },
    )


async def _stream_response(
    result,
    req: ResponsesRequest,
    response_id: str,
    created: int,
    tools: list[dict] | None,
    conversation: list[dict],
):
    """Emit Responses SSE events from the engine stream.

    Routes by whether tools are requested.  The **text path**
    (``_stream_text_response``) emits ``response.created`` immediately — before
    consuming any token — and one ``response.output_text.delta`` per engine
    chunk, so clients see real time-to-first-token (issue #547).  The **tools
    path** (``_stream_tools_response``) must buffer the whole generation:
    ``parse_model_output`` needs the full, brace-balanced output to detect tool
    calls and assign upfront tool IDs, so streaming is impossible there.
    """
    seq = 0

    def ev(event: str, data: dict) -> str:
        nonlocal seq
        payload = {"type": event, "sequence_number": seq, **data}
        seq += 1
        return _sse(event, payload)

    def base_response(status: str, output_items: list[dict], stats, done_reason):
        obj = _build_response_object(
            req, response_id, created, output_items, stats, done_reason
        ).model_dump(exclude_none=True)
        obj["status"] = status
        return obj

    async def _stream_tools_response():
        """Buffer the engine output, parse once, replay full semantic events."""
        out: BufferedModelOutput = await collect_stream(result)

        thinking, visible_text, tool_uses = parse_buffered_output(
            out, tools, fill_missing_args=True
        )
        output_items = _build_output_items(thinking, visible_text, tool_uses)
        done_reason = out.done_reason
        stats = out.stats

        in_progress_resp = base_response("in_progress", [], None, None)
        yield ev("response.created", {"response": in_progress_resp})
        yield ev("response.in_progress", {"response": in_progress_resp})

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
                        "logprobs": [],
                    },
                )
                yield ev(
                    "response.output_text.done",
                    {
                        "item_id": item["id"],
                        "output_index": out_index,
                        "content_index": 0,
                        "text": part["text"],
                        "logprobs": [],
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
                        "name": item["name"],
                    },
                )
            yield ev(
                "response.output_item.done",
                {"output_index": out_index, "item": item},
            )

        final_status = (
            "incomplete" if done_reason in ("timeout", "length") else "completed"
        )
        final = base_response(final_status, output_items, stats, done_reason)
        yield ev("response.completed", {"response": final})

        _store_response(req, response_id, conversation, output_items, final)

    async def _stream_text_response():
        """Real-time path (no tools): emit events as tokens arrive.

        ``response.created``/``response.in_progress`` are emitted up front with
        zero usage, then thinking and visible fragments — split by the shared
        ``split_thinking_parts`` state machine — are routed to a reasoning item
        and per-fragment ``response.output_text.delta`` events respectively.
        """
        reasoning_id = _make_reasoning_id()
        message_id = _make_message_id()
        split_state: dict = {}
        stats = None
        done_reason = None
        thinking_text = ""
        visible_text = ""
        reasoning_open = False
        reasoning_closed = False
        message_open = False
        reasoning_index: int | None = None
        message_index: int | None = None

        def reasoning_item() -> dict:
            content = (
                [{"type": "reasoning_text", "text": thinking_text}]
                if thinking_text
                else []
            )
            return {
                "type": "reasoning",
                "id": reasoning_id,
                "summary": [],
                "content": content,
            }

        def message_item(status: str) -> dict:
            return {
                "type": "message",
                "id": message_id,
                "status": status,
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": visible_text, "annotations": []}
                ],
            }

        def close_reasoning() -> list[str]:
            """Close an open reasoning item: reasoning_text.done then item.done."""
            nonlocal reasoning_closed
            if not reasoning_open or reasoning_closed:
                return []
            reasoning_closed = True
            return [
                ev(
                    "response.reasoning_text.done",
                    {
                        "item_id": reasoning_id,
                        "output_index": reasoning_index,
                        "content_index": 0,
                        "text": thinking_text,
                    },
                ),
                ev(
                    "response.output_item.done",
                    {"output_index": reasoning_index, "item": reasoning_item()},
                ),
            ]

        def route(channel: str, fragment: str) -> list[str]:
            """SSE events for *fragment*, opening/closing items as needed."""
            nonlocal thinking_text, visible_text
            nonlocal reasoning_open, message_open
            nonlocal reasoning_index, message_index
            if not fragment:
                return []
            events: list[str] = []
            if channel == "thinking" and not message_open:
                if not reasoning_open:
                    reasoning_open = True
                    reasoning_index = 0
                    events.append(
                        ev(
                            "response.output_item.added",
                            {"output_index": reasoning_index, "item": reasoning_item()},
                        )
                    )
                thinking_text += fragment
                events.append(
                    ev(
                        "response.reasoning_text.delta",
                        {
                            "item_id": reasoning_id,
                            "output_index": reasoning_index,
                            "content_index": 0,
                            "delta": fragment,
                        },
                    )
                )
                return events
            # Visible content (or stray thinking after the message opened): close
            # the reasoning item, then stream the fragment as an output_text delta.
            events.extend(close_reasoning())
            if not message_open:
                message_open = True
                message_index = 1 if reasoning_index is not None else 0
                events.append(
                    ev(
                        "response.output_item.added",
                        {
                            "output_index": message_index,
                            "item": message_item("in_progress"),
                        },
                    )
                )
                events.append(
                    ev(
                        "response.content_part.added",
                        {
                            "item_id": message_id,
                            "output_index": message_index,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            },
                        },
                    )
                )
            visible_text += fragment
            events.append(
                ev(
                    "response.output_text.delta",
                    {
                        "item_id": message_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "delta": fragment,
                        "logprobs": [],
                    },
                )
            )
            return events

        initial = base_response("in_progress", [], None, None)
        yield ev("response.created", {"response": initial})
        yield ev("response.in_progress", {"response": initial})

        async for chunk in result:
            if chunk.get("cache_info"):
                continue
            # `done` is checked before `thinking_expected` so a terminal chunk
            # that also carries the meta key can't have its stats dropped.
            if chunk.get("done"):
                done_reason = chunk.get("done_reason")
                stats = chunk.get("stats")
                break
            if "thinking_expected" in chunk:
                split_state["thinking_expected"] = bool(chunk["thinking_expected"])
                continue
            for channel, fragment in split_thinking_parts(
                chunk.get("text", ""), split_state
            ):
                for event in route(channel, fragment):
                    yield event

        thinking_tail, content_tail = flush_split_thinking(split_state)
        for channel, fragment in (
            ("thinking", thinking_tail),
            ("content", content_tail),
        ):
            for event in route(channel, fragment):
                yield event

        # Close any item left open: a pure/truncated-thinking response never
        # opened a message, so its reasoning item must be closed here.
        for event in close_reasoning():
            yield event
        if message_open:
            yield ev(
                "response.output_text.done",
                {
                    "item_id": message_id,
                    "output_index": message_index,
                    "content_index": 0,
                    "text": visible_text,
                    "logprobs": [],
                },
            )
            yield ev(
                "response.content_part.done",
                {
                    "item_id": message_id,
                    "output_index": message_index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": visible_text,
                        "annotations": [],
                    },
                },
            )
            yield ev(
                "response.output_item.done",
                {"output_index": message_index, "item": message_item("completed")},
            )

        output_items: list[dict] = []
        if thinking_text:
            output_items.append(reasoning_item())
        if visible_text:
            output_items.append(message_item("completed"))

        final_status = (
            "incomplete" if done_reason in ("timeout", "length") else "completed"
        )
        final = base_response(final_status, output_items, stats, done_reason)
        yield ev("response.completed", {"response": final})

        _store_response(req, response_id, conversation, output_items, final)

    gen = _stream_tools_response() if tools else _stream_text_response()
    try:
        async for event in gen:
            yield event
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
        # Guard cleanup: a raising aclose() (e.g. already-closed generator)
        # must not propagate out of the finally and truncate the SSE stream.
        try:
            await result.aclose()
        except Exception as exc:
            logger.warning("Responses stream aclose failed: %s", exc)


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
    max_tokens = req.max_output_tokens or settings.default_max_tokens
    enable_thinking = _resolve_reasoning(req.reasoning)
    response_id = _make_response_id()
    created = int(time.time())
    cache_id = (req.previous_response_id or response_id)[:256]

    if req.stream:
        result = await generate_chat(
            manager,
            req.model,
            engine_messages,
            options,
            tools=tools,
            stream=True,
            max_tokens=max_tokens,
            cache_id=cache_id,
            enable_thinking=enable_thinking,
            grammar_spec=grammar_spec,
        )
        return StreamingResponse(
            _stream_response(result, req, response_id, created, tools, conversation),
            media_type="text/event-stream",
        )

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

    # Wrap non-streaming result in BufferedModelOutput for shared parse path
    out = BufferedModelOutput(
        full_text=result.get("text", ""),
        raw_text=result.get("raw_text", ""),
        done_reason=result.get("done_reason"),
        stats=result.get("stats"),
        thinking_expected=bool(result.get("thinking_expected")),
    )
    thinking, visible_text, tool_uses = parse_buffered_output(
        out, tools, fill_missing_args=True
    )

    output_items = _build_output_items(thinking, visible_text, tool_uses)
    response = _build_response_object(
        req,
        response_id,
        created,
        output_items,
        result.get("stats"),
        result.get("done_reason"),
    )
    _store_response(
        req,
        response_id,
        conversation,
        output_items,
        response.model_dump(exclude_none=True),
    )
    return response


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
