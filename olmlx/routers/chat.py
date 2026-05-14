import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from olmlx.engine.inference import INIT_ORPHAN_DETECT_LIMIT, generate_chat
from olmlx.engine.tool_parser import (
    fill_missing_required_args,
    parse_model_output,
    resolve_tool_names,
)
from olmlx.routers.common import format_error
from olmlx.schemas.chat import ChatRequest, Message, ToolCall, ToolCallFunction
from olmlx.utils.streaming import safe_ndjson_stream

logger = logging.getLogger(__name__)

router = APIRouter()


# Tag pairs the streaming splitter recognizes.  Adding a third entry
# requires only updating this table — the state machine below is
# tag-pair-aware.
_THINKING_PAIRS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<|channel>thought\n", "<channel|>"),
)


def _find_earliest_open(buf: str) -> tuple[int, str, str]:
    """Earliest open-tag occurrence in *buf*.

    Returns ``(idx, open_tag, paired_close)`` or ``(-1, "", "")`` if none.
    """
    best_idx = -1
    best_open = ""
    best_close = ""
    for open_tag, close_tag in _THINKING_PAIRS:
        idx = buf.find(open_tag)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx, best_open, best_close = idx, open_tag, close_tag
    return best_idx, best_open, best_close


def _find_earliest_close(buf: str) -> tuple[int, str]:
    """Earliest close-tag occurrence (any pair) in *buf*.

    Returns ``(idx, close_tag)`` or ``(-1, "")`` if none.
    """
    best_idx = -1
    best_close = ""
    for _, close_tag in _THINKING_PAIRS:
        idx = buf.find(close_tag)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx, best_close = idx, close_tag
    return best_idx, best_close


def _longest_open_tag_suffix(buf: str) -> int:
    """Largest ``k`` such that ``buf[-k:]`` is a prefix of some open tag.

    Used in passthrough phase to hold back bytes that might be the start
    of a tag straddling a chunk boundary.
    """
    longest = 0
    for open_tag, _ in _THINKING_PAIRS:
        for i in range(min(len(open_tag), len(buf)), longest, -1):
            if open_tag.startswith(buf[-i:]):
                longest = i
                break
    return longest


def _split_thinking_streaming(text: str, state: dict) -> tuple[str, str]:
    """Split a streaming token into ``(thinking_chunk, content_chunk)``.

    Routes thinking text into a separate output channel so the Ollama API
    can populate ``message.thinking`` (issue #307).  Recognizes both the
    Qwen-style ``<think>...</think>`` and Gemma 4's
    ``<|channel>thought\\n...<channel|>`` formats (issue #306).

    State keys:

    - ``phase`` – ``"detect"``, ``"in_think"``, ``"passthrough"``.
    - ``buffer`` – accumulated text waiting to be resolved.
    - ``expected_close`` – when in ``in_think``, the close tag paired
      with the open tag we entered through; ensures cross-format
      mentions inside thinking content can't end the block early.
    - ``thinking_expected`` – when True, the detect phase tolerates a
      longer orphan-thinking preamble before giving up
      (``INIT_ORPHAN_DETECT_LIMIT``) and the orphan-close heuristic
      fires.  Off-by-default keeps non-thinking models from
      misclassifying a literal ``</think>``-in-prose as thinking.
    """
    buf = state.get("buffer", "") + text
    thinking_parts: list[str] = []
    content_parts: list[str] = []
    phase = state.get("phase", "detect")
    expected_close = state.get("expected_close", "")
    thinking_expected = bool(state.get("thinking_expected"))
    detect_limit = INIT_ORPHAN_DETECT_LIMIT if thinking_expected else 200

    while buf:
        if phase == "detect":
            open_idx, open_tag, paired_close = _find_earliest_open(buf)
            close_idx, close_tag = _find_earliest_close(buf)

            if (
                close_idx != -1
                and (open_idx == -1 or close_idx < open_idx)
                and thinking_expected
            ):
                # Orphan close: prefix is thinking.  Gated on
                # ``thinking_expected`` so a non-thinking model that
                # legitimately mentions the literal token isn't routed
                # to ``message.thinking``.
                thinking_parts.append(buf[:close_idx])
                buf = buf[close_idx + len(close_tag) :].lstrip("\n")
                phase = "passthrough"
            elif open_idx != -1:
                content_parts.append(buf[:open_idx])
                buf = buf[open_idx + len(open_tag) :]
                expected_close = paired_close
                phase = "in_think"
            else:
                if len(buf) > detect_limit:
                    content_parts.append(buf)
                    buf = ""
                    phase = "passthrough"
                break

        elif phase == "in_think":
            end = buf.find(expected_close)
            if end == -1:
                # Hold back up to ``len(expected_close)`` trailing chars so
                # a close tag straddling a chunk boundary still matches on
                # the next chunk; emit the rest as thinking.
                hold = len(expected_close)
                if len(buf) > hold:
                    thinking_parts.append(buf[:-hold])
                    buf = buf[-hold:]
                break
            thinking_parts.append(buf[:end])
            buf = buf[end + len(expected_close) :].lstrip("\n")
            expected_close = ""
            phase = "passthrough"

        else:  # passthrough
            open_idx, open_tag, paired_close = _find_earliest_open(buf)
            if open_idx == -1:
                longest_partial = _longest_open_tag_suffix(buf)
                if longest_partial:
                    content_parts.append(buf[:-longest_partial])
                    buf = buf[-longest_partial:]
                else:
                    content_parts.append(buf)
                    buf = ""
                break
            content_parts.append(buf[:open_idx])
            buf = buf[open_idx + len(open_tag) :]
            expected_close = paired_close
            phase = "in_think"

    state["buffer"] = buf
    state["phase"] = phase
    state["expected_close"] = expected_close
    return "".join(thinking_parts), "".join(content_parts)


def _flush_split_thinking(state: dict) -> tuple[str, str]:
    """Flush remaining buffer at stream end.

    If still in ``detect`` (no tag ever seen), treat as content.  In
    ``in_think`` (open tag without close), treat as thinking so the
    response isn't truncated.
    """
    buf = state.get("buffer", "")
    phase = state.get("phase", "detect")
    state["buffer"] = ""
    state["phase"] = "passthrough"
    state["expected_close"] = ""
    if not buf:
        return "", ""
    if phase == "in_think":
        return buf, ""
    return "", buf


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
    thinking, visible_text, tool_uses = parse_model_output(
        raw_text, has_tools=bool(tools), thinking_expected=thinking_expected
    )
    resolve_tool_names(tool_uses, tools)
    fill_missing_required_args(tool_uses, tools)
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
    full_text = ""
    raw_text = ""
    thinking_expected = False
    done_reason = "stop"
    stats = None
    try:
        try:
            async for chunk in result:
                if chunk.get("cache_info"):
                    continue
                if "thinking_expected" in chunk:
                    thinking_expected = bool(chunk["thinking_expected"])
                    continue
                if chunk.get("done"):
                    # ``raw_text`` here is set by gpt-oss models so the
                    # re-parse below sees the full channel-formatted
                    # output (the visible-only ``text`` would lose
                    # tool-call markup).  For Gemma 4 / Qwen ``raw_text``
                    # is absent and we fall back to the accumulated
                    # ``full_text``.  Invariant: ``generate_chat`` only
                    # emits ``raw_text`` on the done chunk.
                    raw_text = chunk.get("raw_text", raw_text)
                    done_reason = chunk.get("done_reason", "stop")
                    stats = chunk.get("stats")
                    break
                full_text += chunk.get("text", "")
        except Exception as exc:
            logger.error("Error during chat streaming (tools): %s", exc, exc_info=True)
            yield format_error(model_name)
            return
    finally:
        # ``result`` is the async generator from ``generate_chat``; we
        # exit the inner loop via ``break`` on the done chunk, leaving
        # the underlying ``CancellableStream`` suspended.  Calling
        # ``aclose`` drains it and releases the inference lock — required
        # on both the clean-break path and the inner-exception path.
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
        if raw_text and full_text:
            logger.debug(
                "raw_text supersedes %d-char accumulated full_text for "
                "tool-call parsing (gpt-oss-style channel payload)",
                len(full_text),
            )
        parse_text = raw_text or full_text
        thinking, visible_text, tool_calls = _build_tool_calls(
            parse_text, tools, thinking_expected=thinking_expected
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
            "message": Message(role="assistant", content="").model_dump(
                exclude_none=True
            ),
            "done": True,
            "done_reason": done_reason,
        }
        if stats:
            final.update(stats.to_dict())
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
    max_tokens = options.pop("num_predict", 512)
    cache_id = request.headers.get("x-cache-id", "")[:256]

    if req.stream:
        result = await generate_chat(
            manager,
            req.model,
            messages,
            options,
            tools=tools,
            stream=True,
            keep_alive=req.keep_alive,
            max_tokens=max_tokens,
            cache_id=cache_id,
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
                thinking_tail, content_tail = _flush_split_thinking(think_state)
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
            thinking_chunk, content_chunk = _split_thinking_streaming(text, think_state)
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
        result = await generate_chat(
            manager,
            req.model,
            messages,
            options,
            tools=tools,
            stream=False,
            keep_alive=req.keep_alive,
            max_tokens=max_tokens,
            cache_id=cache_id,
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
