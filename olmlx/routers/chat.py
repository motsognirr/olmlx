import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from olmlx.config import settings
from olmlx.engine.grammar import parse_response_format
from olmlx.engine.inference import generate_chat
from olmlx.engine.panel import panel_generate_chat
from olmlx.routers.common import format_error, resolve_think_flag
from olmlx.routers.streaming_common import collect_stream, parse_model_output_post
from olmlx.routers.thinking_split import (
    flush_split_thinking,
    split_thinking_streaming,
)
from olmlx.schemas.chat import ChatRequest, Message, ToolCall, ToolCallFunction
from olmlx.utils.streaming import safe_ndjson_stream

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_tool_calls(
    raw_text: str,
    tools: list[dict] | None,
    thinking_expected: bool = False,
) -> tuple[str, str, list[ToolCall] | None]:
    """Parse *raw_text* into ``(thinking, visible_text, tool_calls)``.

    Returns ``(thinking, visible_text, None)`` when no tool calls were
    parsed.  When ``tools`` is ``None``, the tool-call branches inside
    ``parse_model_output`` are skipped so tool-call markup is left in
    ``visible_text`` verbatim; thinking blocks (``<think>...</think>``
    and the Gemma 4 channel format) are still stripped unconditionally
    because that handling runs above the ``has_tools`` gate.
    """
    thinking, visible_text, tool_uses = parse_model_output_post(
        raw_text, has_tools=bool(tools), declared_tools=tools, thinking_expected=thinking_expected
    )
    if not tool_uses:
        return thinking, visible_text, None
    # ``fill_missing_required_args`` normalizes ``tu["input"]`` to a dict
    # for every tool present in the declared list, so the ``or {}`` below
    # is only load-bearing for *unknown* tool names (not in ``tools``) —
    # those are left untouched on purpose, and their ``input`` may still
    # be ``None`` (e.g. gpt-oss ``<|message|>null`` round-trips through
    # ``json.loads`` as ``None``).  Keep the guard so
    # ``ToolCallFunction.arguments`` (``dict[str, Any]``) never sees a
    # bare ``None`` on those paths and the request doesn't 500 on
    # Pydantic validation.
    return (
        thinking,
        visible_text,
        [
            ToolCall(
                function=ToolCallFunction(name=tu["name"], arguments=tu["input"] or {})
            )
            for tu in tool_uses
        ],
    )


async def _stream_chat_with_tools(
    result: AsyncGenerator[dict, None],
    model_name: str,
    tools: list[dict],
) -> AsyncGenerator[str, None]:
    """Buffer the full stream, parse tool calls, then emit two NDJSON lines.

    The Ollama NDJSON protocol does not permit interleaving tool-call
    deltas with content deltas the way OpenAI's SSE does, so when tools
    are declared we trade progressive streaming for a correctly
    structured result: one non-done message chunk carrying ``content``
    (+ optional ``thinking``) + ``tool_calls``, followed by a done
    chunk.  Mirrors ``_stream_openai_sse_with_tools``.
    """
    try:
        try:
            out = await collect_stream(result)
        except Exception as exc:
            logger.error("Error during chat streaming (tools): %s", exc, exc_info=True)
            yield format_error(model_name)
            return
    finally:
        # ``result`` is the async generator from ``generate_chat``;
        # ``collect_stream`` stops at the done chunk, leaving the
        # underlying ``CancellableStream`` suspended.  Calling ``aclose``
        # drains it and releases the inference lock — required on both
        # the clean path and the inner-exception path.
        #
        # On the inner-exception path the inference lock stays held
        # until the consumer pulls one more time after the error chunk:
        # ``finally`` runs when the next ``__anext__`` executes
        # ``return``, not when the caller awaits the ``yield
        # format_error``.  Starlette iterates promptly so for this
        # localhost-only server the window is negligible.
        #
        # We intentionally do NOT wrap this in a try/except.
        # ``drain_and_join`` is the mechanism that releases the
        # inference lock, so swallowing its exception risks a silently-
        # leaked lock that wedges every subsequent request to this
        # worker.  Matches the pattern in ``safe_ndjson_stream``.
        await result.aclose()

    try:
        if out.raw_text and out.full_text:
            logger.debug(
                "raw_text supersedes %d-char accumulated full_text for "
                "tool-call parsing (gpt-oss-style channel payload)",
                len(out.full_text),
            )
        thinking, visible_text, tool_calls = _build_tool_calls(
            out.parse_text, tools, thinking_expected=out.thinking_expected
        )
        now = datetime.now(timezone.utc).isoformat()
        # Skip the non-done chunk when the model emitted only thinking
        # with no visible text and no tool call: clients would otherwise
        # see a spurious ``{"content": ""}`` line before the done chunk.
        if visible_text or thinking or tool_calls:
            yield (
                json.dumps(
                    {
                        "model": model_name,
                        "created_at": now,
                        "message": Message(
                            role="assistant",
                            content=visible_text,
                            thinking=thinking or None,
                            tool_calls=tool_calls,
                        ).model_dump(exclude_none=True),
                        "done": False,
                    }
                )
                + "\n"
            )
        final = {
            "model": model_name,
            "created_at": now,
            "message": Message(role="assistant", content="", thinking=None).model_dump(
                exclude_none=True
            ),
            "done": True,
            "done_reason": out.done_reason or "stop",
        }
        if out.stats:
            final.update(out.stats.to_dict())
        yield json.dumps(final) + "\n"
    except Exception as exc:
        logger.error("Error building chat tool response: %s", exc, exc_info=True)
        yield format_error(model_name)


@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    manager = request.app.state.model_manager
    options = req.options.model_dump(exclude_none=True) if req.options else {}
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    tools = [t.model_dump(exclude_none=True) for t in req.tools] if req.tools else None
    max_tokens = options.pop("num_predict", settings.default_max_tokens)
    cache_id = request.headers.get("x-cache-id", "")[:256]
    enable_thinking = resolve_think_flag(req.think)
    try:
        grammar_spec = parse_response_format(req.format)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    dispatch = (
        panel_generate_chat
        if request.app.state.registry.is_panel(req.model)
        else generate_chat
    )

    if req.stream:
        result = await dispatch(
            manager,
            req.model,
            messages,
            options,
            tools=tools,
            stream=True,
            keep_alive=req.keep_alive,
            max_tokens=max_tokens,
            cache_id=cache_id,
            enable_thinking=enable_thinking,
            grammar_spec=grammar_spec,
        )

        if tools:
            # Buffer the full stream so we can emit structured
            # ``tool_calls``: the NDJSON protocol can't interleave
            # tool-call deltas with content deltas the way OpenAI's
            # SSE can, so progressive streaming is traded for correct
            # structure here.
            return StreamingResponse(
                _stream_chat_with_tools(result, req.model, tools),
                media_type="application/x-ndjson",
            )

        think_state: dict = {}

        def format_chunk(chunk):
            if chunk.get("cache_info"):
                return None
            if "thinking_expected" in chunk:
                think_state["thinking_expected"] = bool(chunk["thinking_expected"])
                return None
            now = datetime.now(timezone.utc).isoformat()
            if chunk.get("done"):
                thinking_tail, content_tail = flush_split_thinking(think_state)
                stats = chunk.get("stats")
                # Emit any held content as a regular non-done chunk
                # *before* the done marker.  Ollama clients accumulate
                # ``message.content`` from non-done chunks and the
                # standard done frame has ``content=""``; putting
                # accumulated text in the done frame would silently drop
                # it on those clients.
                pre_done = ""
                if thinking_tail or content_tail:
                    pre_done = (
                        json.dumps(
                            {
                                "model": req.model,
                                "created_at": now,
                                "message": Message(
                                    role="assistant",
                                    content=content_tail,
                                    thinking=thinking_tail or None,
                                ).model_dump(exclude_none=True),
                                "done": False,
                            }
                        )
                        + "\n"
                    )
                final = {
                    "model": req.model,
                    "created_at": now,
                    "message": Message(
                        role="assistant", content="", thinking=None
                    ).model_dump(exclude_none=True),
                    "done": True,
                    "done_reason": chunk.get("done_reason", "stop"),
                }
                if stats:
                    final.update(stats.to_dict())
                return pre_done + json.dumps(final) + "\n"
            text = chunk.get("text", "")
            # Always run the splitter, even when ``thinking_expected`` is
            # False: Gemma 4 emits its channel-format thinking
            # (``<|channel>thought\n...<channel|>``) regardless of the
            # engine's thinking flag, and we still need to keep those
            # tokens out of ``message.content`` (issue #306).  The
            # splitter's detect buffer is bounded at 200 chars when
            # ``thinking_expected`` is False so the latency impact stays
            # within one chunk for non-thinking responses.
            #
            # All emitter branches below use ``exclude_none=True`` so
            # null ``thinking`` / ``images`` / ``tool_calls`` fields are
            # suppressed from the wire payload.
            thinking_chunk, content_chunk = split_thinking_streaming(text, think_state)
            if not thinking_chunk and not content_chunk:
                return None
            return (
                json.dumps(
                    {
                        "model": req.model,
                        "created_at": now,
                        "message": Message(
                            role="assistant",
                            content=content_chunk,
                            thinking=thinking_chunk or None,
                        ).model_dump(exclude_none=True),
                        "done": False,
                    }
                )
                + "\n"
            )

        return StreamingResponse(
            safe_ndjson_stream(
                result,
                format_chunk,
                lambda exc: format_error(req.model),
                log=logger,
                log_prefix="chat streaming",
            ),
            media_type="application/x-ndjson",
        )
    else:
        result = await dispatch(
            manager,
            req.model,
            messages,
            options,
            tools=tools,
            stream=False,
            keep_alive=req.keep_alive,
            max_tokens=max_tokens,
            cache_id=cache_id,
            enable_thinking=enable_thinking,
            grammar_spec=grammar_spec,
        )
        now = datetime.now(timezone.utc).isoformat()
        stats = result.get("stats")
        # ``_full_completion`` only populates ``raw_text`` for gpt-oss
        # channel format (where ``text`` has the channel tokens
        # stripped and ``parse_model_output`` needs the un-stripped form
        # to find tool calls / thinking).  For every other model the
        # ``or`` falls back to ``text``, which is already the unstripped
        # output.
        text = result.get("text", "")
        parse_text = result.get("raw_text") or text
        # Pass ``thinking_expected`` so the orphan-``</think>``
        # heuristic only fires when the engine actually requested
        # thinking — keeps streaming and non-streaming behaviour
        # symmetric and prevents a non-thinking model that mentions
        # the literal token from having its prefix silently routed to
        # thinking.
        #
        # Unlike the streaming path, ``_build_tool_calls`` is left
        # unguarded: the response hasn't started, so a parse failure
        # surfaces as FastAPI's standard 500 — same convention as the
        # OpenAI non-streaming path in ``routers/openai.py``.
        thinking, visible_text, tool_calls = _build_tool_calls(
            parse_text,
            tools,
            thinking_expected=bool(result.get("thinking_expected")),
        )
        response = {
            "model": req.model,
            "created_at": now,
            "message": Message(
                role="assistant",
                content=visible_text,
                thinking=thinking or None,
                tool_calls=tool_calls,
            ).model_dump(exclude_none=True),
            "done": True,
            "done_reason": result.get("done_reason", "stop"),
        }
        if stats:
            response.update(stats.to_dict())
        return response
