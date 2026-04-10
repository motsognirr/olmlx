import json
import logging
import re
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from olmlx.engine.inference import (
    generate_chat,
    generate_completion,
    generate_embeddings,
)
from olmlx.engine.tool_parser import parse_model_output, resolve_tool_names
from olmlx.schemas.openai import (
    OpenAIChatMessage,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAICompletionChoice,
    OpenAICompletionRequest,
    OpenAICompletionResponse,
    OpenAIEmbeddingData,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    OpenAIModel,
    OpenAIModelList,
    OpenAIUsage,
)

logger = logging.getLogger(__name__)

router = APIRouter()

JSON_MODE_SYSTEM_MSG = (
    "Respond with valid JSON only. Do not include any text outside the JSON object."
)


def _make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


def _fill_missing_required_args(
    tool_uses: list[dict],
    declared_tools: list[dict] | None,
) -> None:
    """Fill missing required string arguments in tool calls from the client's schema.

    Models sometimes omit fields the client marks as required (e.g. opencode's
    ``description`` on bash).  Walk the declared tool schemas and inject an
    empty string for any required string parameter the model left out.
    Mutates *tool_uses* in place.
    """
    if not declared_tools:
        return

    # Build lookup: lowercase tool name -> {param: type} for required params
    schema_by_tool: dict[str, dict[str, str]] = {}
    for tool in declared_tools:
        func = tool.get("function") or {}
        name = (func.get("name") or "").lower()
        params = func.get("parameters") or {}
        required = set(params.get("required", []))
        properties = params.get("properties") or {}
        schema_by_tool[name] = {
            k: v.get("type", "") for k, v in properties.items() if k in required
        }

    for tu in tool_uses:
        required_params = schema_by_tool.get((tu.get("name") or "").lower())
        if not required_params:
            continue
        inp = tu.get("input") or {}
        changed = False
        for param, param_type in required_params.items():
            if param not in inp or inp[param] is None:
                if param_type == "string":
                    logger.warning(
                        "Tool '%s' missing required string param '%s', injecting empty string",
                        tu.get("name"),
                        param,
                    )
                    inp[param] = ""
                    changed = True
                else:
                    logger.warning(
                        "Tool '%s' missing required param '%s' (type %r), cannot inject default",
                        tu.get("name"),
                        param,
                        param_type,
                    )
        if changed:
            tu["input"] = inp


def _to_openai_tool_calls(tool_uses: list[dict]) -> list[dict]:
    """Convert parsed tool_use blocks to OpenAI tool_calls format."""
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": tu["name"],
                "arguments": json.dumps(tu.get("input", {})),
            },
        }
        for tu in tool_uses
    ]


def _strip_thinking_streaming(text: str, state: dict) -> str:
    """Strip ``<think>...</think>`` blocks from streaming text chunks.

    Uses *state* dict to track position across calls.  Keys:

    - ``phase``: one of ``"detect"``, ``"in_think"``, ``"passthrough"``
    - ``buffer``: accumulated text waiting to be resolved

    **Phases:**

    ``detect`` (initial) — The chat template may have opened ``<think>``
    inside the prompt, so the generated text could start mid-think with
    only a closing ``</think>``.  We buffer all content until we can
    determine which case we're in:

    * ``</think>`` seen first → discard buffer (orphaned thinking),
      switch to ``passthrough``.
    * ``<think>`` seen first → emit buffer, switch to ``in_think``.
    * Neither tag after the stream ends → emit buffer (no thinking).

    ``in_think`` — Inside a ``<think>`` block; discard until ``</think>``.

    ``passthrough`` — Emit everything, but still strip any new
    ``<think>...</think>`` blocks that appear later.
    """
    buf = state.get("buffer", "") + text
    out_parts: list[str] = []
    phase = state.get("phase", "detect")

    while buf:
        if phase == "detect":
            open_idx = buf.find("<think>")
            close_idx = buf.find("</think>")

            if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
                # Orphaned </think> — discard everything before it.
                buf = buf[close_idx + len("</think>") :]
                phase = "passthrough"
            elif open_idx != -1:
                # Normal <think> — emit text before it, enter in_think.
                out_parts.append(buf[:open_idx])
                buf = buf[open_idx + len("<think>") :]
                phase = "in_think"
            else:
                # Neither tag yet.  Keep buffering to detect a potential
                # orphaned </think> at the start of the stream.  Once the
                # buffer grows large enough that an orphaned tag is very
                # unlikely, emit the safe prefix and transition to
                # passthrough so non-thinking models stream progressively.
                # The threshold is generous to catch real orphaned tags
                # (thinking content before </think>) while avoiding
                # unbounded buffering for non-thinking models.
                _DETECT_LIMIT = 200
                if len(buf) > _DETECT_LIMIT:
                    out_parts.append(buf)
                    buf = ""
                    phase = "passthrough"
                break

        elif phase == "in_think":
            end = buf.find("</think>")
            if end == -1:
                buf = ""
            else:
                buf = buf[end + len("</think>") :]
                phase = "passthrough"

        else:  # passthrough
            open_idx = buf.find("<think>")
            if open_idx == -1:
                longest_partial = 0
                for i in range(1, min(len("<think>"), len(buf) + 1)):
                    if "<think>".startswith(buf[-i:]):
                        longest_partial = i
                        break
                if longest_partial:
                    out_parts.append(buf[:-longest_partial])
                    buf = buf[-longest_partial:]
                else:
                    out_parts.append(buf)
                    buf = ""
            else:
                out_parts.append(buf[:open_idx])
                buf = buf[open_idx + len("<think>") :]
                phase = "in_think"

    state["buffer"] = buf
    state["phase"] = phase
    return "".join(out_parts)


def _flush_thinking_buffer(state: dict) -> str:
    """Flush any remaining buffer when the stream ends.

    If we're still in ``detect`` phase (never saw any think tag),
    the buffered content is real output — emit it.
    """
    buf = state.get("buffer", "")
    phase = state.get("phase", "detect")
    state["buffer"] = ""
    if phase == "detect":
        return buf
    return ""


async def _stream_openai_sse(
    result,
    response_id,
    model,
    created,
    object_type,
    format_content,
    format_done,
    strip_thinking=False,
):
    """Shared SSE streaming for OpenAI-compatible endpoints.

    format_content(text) -> choices[0] dict for content chunks
    format_done() -> choices[0] dict for the final chunk
    """
    think_state: dict = {}
    try:
        async for chunk in result:
            if chunk.get("cache_info"):
                continue
            if chunk.get("done"):
                # Flush any buffered content from thinking detection.
                if strip_thinking:
                    flushed = _flush_thinking_buffer(think_state)
                    if flushed:
                        data = {
                            "id": response_id,
                            "object": object_type,
                            "created": created,
                            "model": model,
                            "choices": [format_content(flushed)],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                data = {
                    "id": response_id,
                    "object": object_type,
                    "created": created,
                    "model": model,
                    "choices": [format_done()],
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                text = chunk.get("text", "")
                if strip_thinking:
                    text = _strip_thinking_streaming(text, think_state)
                if not text:
                    continue
                data = {
                    "id": response_id,
                    "object": object_type,
                    "created": created,
                    "model": model,
                    "choices": [format_content(text)],
                }
                yield f"data: {json.dumps(data)}\n\n"
    except Exception as exc:
        logger.error("Error during OpenAI streaming: %s", exc, exc_info=True)
        error_payload = json.dumps(
            {
                "error": {
                    "message": "An internal server error occurred during streaming.",
                    "type": "server_error",
                    "code": "internal_error",
                }
            }
        )
        yield f"data: {error_payload}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await result.aclose()


async def _stream_openai_sse_with_tools(
    result, response_id, model, created, declared_tools=None
):
    """Buffer full output, parse tool calls, then emit OpenAI-compliant SSE.

    Matches the exact chunk sequence that OpenAI produces and that the
    Vercel AI SDK (used by Opencode) expects:

    1. Role chunk:  delta={role: "assistant", content: null}
    2. (Optional) Content chunks if visible text precedes tool calls
    3. Per tool call:
       a. Intro:  delta={tool_calls: [{index, id, type, function: {name, arguments: ""}}]}
       b. Args:   delta={tool_calls: [{index, function: {arguments: "<json>"}}]}
    4. Done:  delta={}, finish_reason="tool_calls"
    """
    full_text = ""
    raw_text = ""
    try:
        async for chunk in result:
            if chunk.get("cache_info"):
                continue
            if chunk.get("done"):
                # Read raw_text from done chunk for gpt-oss tool call parsing
                raw_text = chunk.get("raw_text", raw_text)
                break
            full_text += chunk.get("text", "")

        # Use raw_text for parsing so channel-format tool calls aren't lost;
        # fall back to full_text for non-gpt-oss models
        text_for_parsing = raw_text if raw_text else full_text
        _thinking, visible_text, tool_uses = parse_model_output(text_for_parsing, True)
        resolve_tool_names(tool_uses, declared_tools)
        _fill_missing_required_args(tool_uses, declared_tools)
        logger.debug(
            "Buffered tool stream (%d chars): thinking=%d visible=%d tool_uses=%d raw=%s",
            len(full_text),
            len(_thinking),
            len(visible_text),
            len(tool_uses),
            full_text[:2000],
        )

        def _chunk(choices_0):
            return json.dumps(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, **choices_0}],
                }
            )

        if tool_uses:
            tool_calls = _to_openai_tool_calls(tool_uses)

            # 1. Role announcement
            yield f"data: {_chunk({'delta': {'role': 'assistant', 'content': None}, 'finish_reason': None})}\n\n"

            # 2. Content chunks if visible text present
            if visible_text:
                yield f"data: {_chunk({'delta': {'content': visible_text}, 'finish_reason': None})}\n\n"

            # 3. Per-tool-call: intro (name + empty args) then args
            for i, tc in enumerate(tool_calls):
                # Intro chunk: id, type, name, empty arguments
                intro = {
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["function"]["name"], "arguments": ""},
                }
                yield f"data: {_chunk({'delta': {'tool_calls': [intro]}, 'finish_reason': None})}\n\n"

                # Arguments chunk
                args_chunk = {
                    "index": i,
                    "function": {"arguments": tc["function"]["arguments"]},
                }
                yield f"data: {_chunk({'delta': {'tool_calls': [args_chunk]}, 'finish_reason': None})}\n\n"

            # 4. Done
            yield f"data: {_chunk({'delta': {}, 'finish_reason': 'tool_calls'})}\n\n"
        else:
            # No tool calls — emit as normal content
            yield f"data: {_chunk({'delta': {'role': 'assistant', 'content': None}, 'finish_reason': None})}\n\n"
            if visible_text:
                yield f"data: {_chunk({'delta': {'content': visible_text}, 'finish_reason': None})}\n\n"
            yield f"data: {_chunk({'delta': {}, 'finish_reason': 'stop'})}\n\n"

        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.error("Error during OpenAI streaming: %s", exc, exc_info=True)
        error_payload = json.dumps(
            {
                "error": {
                    "message": "An internal server error occurred during streaming.",
                    "type": "server_error",
                    "code": "internal_error",
                }
            }
        )
        yield f"data: {error_payload}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await result.aclose()


def _build_options(req) -> dict:
    opts = {}
    if req.temperature is not None:
        opts["temperature"] = req.temperature
    if req.top_p is not None:
        opts["top_p"] = req.top_p
    if req.seed is not None:
        opts["seed"] = req.seed
    if req.stop is not None:
        opts["stop"] = req.stop if isinstance(req.stop, list) else [req.stop]
    if req.frequency_penalty:
        opts["frequency_penalty"] = req.frequency_penalty
    if req.presence_penalty:
        opts["presence_penalty"] = req.presence_penalty
    return opts


@router.post("/v1/chat/completions")
async def openai_chat(req: OpenAIChatRequest, request: Request):
    logger.info(
        "OpenAI chat request: model=%s stream=%s tools=%d messages=%d max_tokens=%s max_completion_tokens=%s",
        req.model,
        req.stream,
        len(req.tools or []),
        len(req.messages),
        req.max_tokens,
        req.max_completion_tokens,
    )
    manager = request.app.state.model_manager
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    if req.response_format and req.response_format.type in (
        "json_object",
        "json_schema",
    ):
        if req.response_format.type == "json_schema":
            raw_name = req.response_format.json_schema["name"]
            schema_name = re.sub(r"[^A-Za-z0-9_\-]", "", raw_name)[:64]
            logger.info(
                "response_format type 'json_schema' is not enforced; "
                "output may not conform to the provided schema",
            )
            json_prompt = (
                f"Respond with valid JSON only, conforming to the '{schema_name}' schema. "
                "Do not include any text outside the JSON object."
                if schema_name
                else JSON_MODE_SYSTEM_MSG
            )
        else:
            json_prompt = JSON_MODE_SYSTEM_MSG
        if messages and messages[0].get("role") == "system":
            existing = messages[0].get("content") or ""
            if json_prompt not in existing:
                sep = "\n\n" if existing else ""
                messages[0]["content"] = existing + sep + json_prompt
        else:
            messages.insert(0, {"role": "system", "content": json_prompt})
    options = _build_options(req)
    max_tokens = req.max_completion_tokens or req.max_tokens or 512
    chat_id = _make_id()
    created = int(time.time())
    cache_id = request.headers.get("x-cache-id", "")[:256]

    if req.stream:
        result = await generate_chat(
            manager,
            req.model,
            messages,
            options,
            tools=req.tools,
            stream=True,
            max_tokens=max_tokens,
            cache_id=cache_id,
        )

        if req.tools:
            return StreamingResponse(
                _stream_openai_sse_with_tools(
                    result,
                    chat_id,
                    req.model,
                    created,
                    declared_tools=req.tools,
                ),
                media_type="text/event-stream",
            )

        return StreamingResponse(
            _stream_openai_sse(
                result,
                chat_id,
                req.model,
                created,
                "chat.completion.chunk",
                lambda text: {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                },
                lambda: {"index": 0, "delta": {}, "finish_reason": "stop"},
                strip_thinking=True,
            ),
            media_type="text/event-stream",
        )
    else:
        result = await generate_chat(
            manager,
            req.model,
            messages,
            options,
            tools=req.tools,
            stream=False,
            max_tokens=max_tokens,
            cache_id=cache_id,
        )
        text = result.get("text", "")
        # Use raw_text for tool parsing (preserves gpt-oss channel tokens)
        parse_text = result.get("raw_text", text)
        logger.debug(
            "Raw model output (%d chars): %s", len(parse_text), parse_text[:1000]
        )
        usage = OpenAIUsage.from_stats(result.get("stats"))

        has_tools = bool(req.tools)
        _thinking, visible_text, tool_uses = parse_model_output(parse_text, has_tools)
        resolve_tool_names(tool_uses, req.tools)
        _fill_missing_required_args(tool_uses, req.tools)

        tool_calls = _to_openai_tool_calls(tool_uses) if tool_uses else None
        finish_reason = "tool_calls" if tool_uses else "stop"
        content = visible_text
        if not content:
            content = None

        return OpenAIChatResponse(
            id=chat_id,
            created=created,
            model=req.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIChatMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )


@router.post("/v1/completions")
async def openai_completions(req: OpenAICompletionRequest, request: Request):
    manager = request.app.state.model_manager
    options = _build_options(req)
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    max_tokens = req.max_tokens or 512
    comp_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if req.stream:
        result = await generate_completion(
            manager,
            req.model,
            prompt,
            options,
            stream=True,
            max_tokens=max_tokens,
        )

        return StreamingResponse(
            _stream_openai_sse(
                result,
                comp_id,
                req.model,
                created,
                "text_completion",
                lambda text: {"index": 0, "text": text, "finish_reason": None},
                lambda: {"index": 0, "text": "", "finish_reason": "stop"},
            ),
            media_type="text/event-stream",
        )
    else:
        result = await generate_completion(
            manager,
            req.model,
            prompt,
            options,
            stream=False,
            max_tokens=max_tokens,
        )
        usage = OpenAIUsage.from_stats(result.get("stats"))
        return OpenAICompletionResponse(
            id=comp_id,
            created=created,
            model=req.model,
            choices=[
                OpenAICompletionChoice(
                    index=0,
                    text=result.get("text", ""),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )


@router.get("/v1/models")
async def openai_list_models(request: Request):
    registry = request.app.state.registry
    models = registry.list_models()
    data = [OpenAIModel(id=name, created=int(time.time())) for name in models]
    return OpenAIModelList(data=data)


@router.post("/v1/embeddings")
async def openai_embeddings(req: OpenAIEmbeddingRequest, request: Request):
    manager = request.app.state.model_manager
    texts = req.input if isinstance(req.input, list) else [req.input]
    embeddings = await generate_embeddings(manager, req.model, texts)
    data = [
        OpenAIEmbeddingData(index=i, embedding=emb) for i, emb in enumerate(embeddings)
    ]
    return OpenAIEmbeddingResponse(
        data=data,
        model=req.model,
    )
