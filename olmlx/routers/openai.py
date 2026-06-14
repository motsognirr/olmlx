import base64
import json
import logging
import re
import struct
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from olmlx.config import settings
from olmlx.engine.grammar import parse_response_format
from olmlx.engine.inference import (
    INIT_ORPHAN_DETECT_LIMIT,
    generate_chat,
    generate_completion,
    generate_embeddings,
)
from olmlx.engine.panel import panel_generate_chat
from olmlx.engine.tool_parser import (
    fill_missing_required_args,
    parse_model_output,
    resolve_tool_names,
)
from olmlx.routers.common import (
    build_inference_options,
    collect_content_parts,
    resolve_openai_think,
)
from olmlx.routers.streaming_common import collect_stream, parse_buffered_output
from olmlx.routers.thinking_split import (
    flush_thinking_buffer,
    strip_thinking_streaming,
)
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

    def _compact_choice(choice: dict) -> dict:
        return {k: v for k, v in choice.items() if v is not None}

    think_state: dict = {}
    try:
        async for chunk in result:
            if chunk.get("cache_info"):
                continue
            if "thinking_expected" in chunk:
                # Engine-emitted meta: tells the thinking stripper whether
                # to wait for an orphan </think> (issue #307).  Without
                # `thinking_expected`, a model that legitimately mentions
                # the literal `</think>` token would have its prefix
                # silently reclassified as thinking.
                if chunk["thinking_expected"]:
                    think_state["thinking_expected"] = True
                    think_state["detect_limit"] = INIT_ORPHAN_DETECT_LIMIT
                continue
            if chunk.get("done"):
                # Flush any buffered content from thinking detection.
                if strip_thinking:
                    flushed = flush_thinking_buffer(think_state)
                    if flushed:
                        data = {
                            "id": response_id,
                            "object": object_type,
                            "created": created,
                            "model": model,
                            "choices": [_compact_choice(format_content(flushed))],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                done_reason = chunk.get("done_reason")
                finish_reason = (
                    "length"
                    if done_reason == "timeout"
                    else format_done().get("finish_reason", "stop")
                )
                done_choice = format_done()
                done_choice["finish_reason"] = finish_reason
                data = {
                    "id": response_id,
                    "object": object_type,
                    "created": created,
                    "model": model,
                    "choices": [_compact_choice(done_choice)],
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                text = chunk.get("text", "")
                if strip_thinking:
                    text = strip_thinking_streaming(text, think_state)
                if not text:
                    continue
                data = {
                    "id": response_id,
                    "object": object_type,
                    "created": created,
                    "model": model,
                    "choices": [_compact_choice(format_content(text))],
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
    try:
        out = await collect_stream(result)
        done_reason = out.done_reason
        _thinking, visible_text, tool_uses = parse_buffered_output(out, declared_tools)
        logger.debug(
            "Buffered tool stream (%d chars): thinking=%d visible=%d tool_uses=%d raw=%s",
            len(out.full_text),
            len(_thinking),
            len(visible_text),
            len(tool_uses),
            out.full_text[:2000],
        )

        def _chunk(choices_0):
            choices_0_clean = {k: v for k, v in choices_0.items() if v is not None}
            return json.dumps(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, **choices_0_clean}],
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
            fr = "length" if done_reason == "timeout" else "stop"
            yield f"data: {_chunk({'delta': {}, 'finish_reason': fr})}\n\n"

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
    return build_inference_options(
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
        stop=req.stop,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
    )


def _normalize_multimodal_messages(messages: list[dict]) -> list[dict]:
    """Split OpenAI multimodal content lists into a text ``content`` string plus
    separate ``images``/``audio`` lists (the engine's Ollama-style convention,
    #428).

    OpenAI carries images inline as content parts:
        content: [{"type": "text", "text": ...},
                  {"type": "image_url", "image_url": {"url": ...}}]
    String content is left untouched.  Part-type recognition lives in the
    shared ``collect_content_parts`` (issue #471).
    """
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        texts, images, audio = collect_content_parts(content)
        m["content"] = " ".join(texts)
        if images:
            m["images"] = (m.get("images") or []) + images
        if audio:
            m["audio"] = (m.get("audio") or []) + audio
    return messages


@router.post(
    "/v1/chat/completions",
    response_model=OpenAIChatResponse,
    response_model_exclude_none=True,
)
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
    # A malformed image content part (e.g. image_url with no url) is a client
    # error — surface as 422 rather than an uncaught 500.
    try:
        messages = _normalize_multimodal_messages(messages)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    grammar_spec = None
    if req.response_format and req.response_format.type in (
        "json_object",
        "json_schema",
    ):
        # Schema-shape problems are client errors — surface as 422 so the
        # caller sees a meaningful message instead of FastAPI's default
        # 500 for uncaught exceptions.
        try:
            grammar_spec = parse_response_format(
                req.response_format.model_dump(exclude_none=True)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        # Belt-and-braces: also prepend the system-message hint so the model
        # is told what it's being constrained to. The grammar enforces shape;
        # the hint helps the model produce *meaningful* JSON (correct field
        # names, sensible values) rather than the shortest grammar-valid
        # output it can emit.
        if req.response_format.type == "json_schema":
            raw_name = req.response_format.json_schema["name"]
            schema_name = re.sub(r"[^A-Za-z0-9_\-]", "", raw_name)[:64]
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
    max_tokens = (
        req.max_completion_tokens or req.max_tokens or settings.default_max_tokens
    )
    chat_id = _make_id()
    created = int(time.time())
    cache_id = request.headers.get("x-cache-id", "")[:256]
    enable_thinking = resolve_openai_think(
        req.reasoning_effort, req.chat_template_kwargs
    )
    registry = request.app.state.registry
    dispatch = panel_generate_chat if registry.is_panel(req.model) else generate_chat

    if req.stream:
        result = await dispatch(
            manager,
            req.model,
            messages,
            options,
            tools=req.tools,
            stream=True,
            max_tokens=max_tokens,
            cache_id=cache_id,
            enable_thinking=enable_thinking,
            grammar_spec=grammar_spec,
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
        result = await dispatch(
            manager,
            req.model,
            messages,
            options,
            tools=req.tools,
            stream=False,
            max_tokens=max_tokens,
            cache_id=cache_id,
            enable_thinking=enable_thinking,
            grammar_spec=grammar_spec,
        )
        text = result.get("text", "")
        # Use raw_text for tool parsing (preserves gpt-oss channel tokens);
        # ``or text`` (not ``.get(key, text)``) so an empty-string ``raw_text``
        # from generate_chat also falls back to the cleaned text, matching
        # the streaming path above and the Ollama chat router.
        parse_text = result.get("raw_text") or text
        logger.debug(
            "Raw model output (%d chars): %s", len(parse_text), parse_text[:1000]
        )
        usage = OpenAIUsage.from_stats(result.get("stats"))

        has_tools = bool(req.tools)
        # Pass `thinking_expected` so the orphan-`</think>` heuristic only
        # fires when the engine actually requested thinking (issue #307
        # review): a non-thinking model that mentions the literal token
        # would otherwise have its prefix silently dropped from content.
        _thinking, visible_text, tool_uses = parse_model_output(
            parse_text,
            has_tools,
            thinking_expected=bool(result.get("thinking_expected")),
        )
        resolve_tool_names(tool_uses, req.tools)
        fill_missing_required_args(tool_uses, req.tools)

        tool_calls = _to_openai_tool_calls(tool_uses) if tool_uses else None
        done_reason = result.get("done_reason")
        if tool_uses:
            finish_reason = "tool_calls"
        elif done_reason == "timeout":
            finish_reason = "length"
        else:
            finish_reason = "stop"
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


@router.post(
    "/v1/completions",
    response_model=OpenAICompletionResponse,
    response_model_exclude_none=True,
)
async def openai_completions(req: OpenAICompletionRequest, request: Request):
    manager = request.app.state.model_manager
    options = _build_options(req)
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    max_tokens = req.max_tokens or settings.default_max_tokens
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
                    finish_reason="length"
                    if result.get("done_reason") == "timeout"
                    else "stop",
                )
            ],
            usage=usage,
        )


@router.get(
    "/v1/models",
    response_model=OpenAIModelList,
    response_model_exclude_none=True,
)
async def openai_list_models(request: Request):
    registry = request.app.state.registry
    created = int(time.time())
    models = registry.list_models()
    data = [OpenAIModel(id=name, created=created) for name in models]
    # Synthetic panel models have no weights but are selectable by name, so
    # clients (opencode, etc.) need to see them in the model list.
    data += [OpenAIModel(id=name, created=created) for name in registry.list_panels()]
    return OpenAIModelList(data=data)


@router.post(
    "/v1/embeddings",
    response_model=OpenAIEmbeddingResponse,
    response_model_exclude_none=True,
)
async def openai_embeddings(req: OpenAIEmbeddingRequest, request: Request):
    manager = request.app.state.model_manager
    texts = req.input if isinstance(req.input, list) else [req.input]
    embeddings = await generate_embeddings(manager, req.model, texts)
    if req.encoding_format == "base64":
        # OpenAI's base64 format packs each embedding as little-endian
        # float32 bytes, then base64-encodes the result into a string.
        data = [
            OpenAIEmbeddingData(
                index=i,
                embedding=base64.b64encode(struct.pack(f"<{len(emb)}f", *emb)).decode(
                    "ascii"
                ),
            )
            for i, emb in enumerate(embeddings)
        ]
    else:
        data = [
            OpenAIEmbeddingData(index=i, embedding=emb)
            for i, emb in enumerate(embeddings)
        ]
    return OpenAIEmbeddingResponse(
        data=data,
        model=req.model,
    )
