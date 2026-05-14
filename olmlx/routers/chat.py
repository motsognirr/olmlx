import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from olmlx.engine.inference import INIT_ORPHAN_DETECT_LIMIT, generate_chat
from olmlx.engine.tool_parser import parse_model_output
from olmlx.routers.common import format_error
from olmlx.schemas.chat import ChatRequest, Message
from olmlx.utils.streaming import safe_ndjson_stream

logger = logging.getLogger(__name__)

router = APIRouter()


def _split_thinking_streaming(text: str, state: dict) -> tuple[str, str]:
    """Split a streaming token into ``(thinking_chunk, content_chunk)``.

    Mirrors ``_strip_thinking_streaming`` in the OpenAI router but routes the
    thinking text into a separate output channel instead of discarding it,
    so the Ollama API can populate ``message.thinking`` (issue #307).

    State keys:
    - ``phase``: ``"detect"``, ``"in_think"``, ``"passthrough"``
    - ``buffer``: accumulated text waiting to be resolved
    - ``thinking_expected``: when True, the detect phase tolerates a
      longer orphan-thinking preamble before giving up.
    """
    buf = state.get("buffer", "") + text
    thinking_parts: list[str] = []
    content_parts: list[str] = []
    phase = state.get("phase", "detect")
    thinking_expected = bool(state.get("thinking_expected"))
    detect_limit = INIT_ORPHAN_DETECT_LIMIT if thinking_expected else 200

    while buf:
        if phase == "detect":
            open_idx = buf.find("<think>")
            close_idx = buf.find("</think>")

            if (
                close_idx != -1
                and (open_idx == -1 or close_idx < open_idx)
                and thinking_expected
            ):
                # Orphan `</think>`: prefix is thinking content.
                # Gated on `thinking_expected` so a non-thinking model that
                # legitimately mentions the literal `</think>` token isn't
                # silently routed into `message.thinking`.
                thinking_parts.append(buf[:close_idx])
                buf = buf[close_idx + len("</think>") :].lstrip("\n")
                phase = "passthrough"
            elif open_idx != -1:
                # Pre-think text is content; tag itself is consumed.
                content_parts.append(buf[:open_idx])
                buf = buf[open_idx + len("<think>") :]
                phase = "in_think"
            else:
                # No tag yet.  Buffer until we can decide, then flush as
                # content if no tag ever appears.
                if len(buf) > detect_limit:
                    content_parts.append(buf)
                    buf = ""
                    phase = "passthrough"
                break

        elif phase == "in_think":
            end = buf.find("</think>")
            if end == -1:
                # Hold back up to 8 trailing chars (the length of `</think>`)
                # in case the tag is split across two chunks; emit the rest
                # as thinking.
                if len(buf) > 8:
                    thinking_parts.append(buf[:-8])
                    buf = buf[-8:]
                break
            thinking_parts.append(buf[:end])
            buf = buf[end + len("</think>") :].lstrip("\n")
            phase = "passthrough"

        else:  # passthrough
            open_idx = buf.find("<think>")
            if open_idx == -1:
                # Hold back any trailing prefix of `<think>` so a follow-up
                # chunk can complete the tag (avoids loop-spin on bare "<").
                longest_partial = 0
                for i in range(1, min(len("<think>"), len(buf) + 1)):
                    if "<think>".startswith(buf[-i:]):
                        longest_partial = i
                        break
                if longest_partial:
                    content_parts.append(buf[:-longest_partial])
                    buf = buf[-longest_partial:]
                else:
                    content_parts.append(buf)
                    buf = ""
                break
            content_parts.append(buf[:open_idx])
            buf = buf[open_idx + len("<think>") :]
            phase = "in_think"

    state["buffer"] = buf
    state["phase"] = phase
    return "".join(thinking_parts), "".join(content_parts)


def _flush_split_thinking(state: dict) -> tuple[str, str]:
    """Flush remaining buffer at stream end.

    If the buffer is still in ``detect`` (no tag ever seen), treat it as
    content.  In ``in_think`` (open tag without close), treat as thinking so
    the response isn't truncated.
    """
    buf = state.get("buffer", "")
    phase = state.get("phase", "detect")
    state["buffer"] = ""
    if not buf:
        return "", ""
    if phase == "detect":
        return "", buf
    if phase == "in_think":
        return buf, ""
    return "", buf


@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    manager = request.app.state.model_manager
    options = req.options.model_dump(exclude_none=True) if req.options else {}
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    tools = [t.model_dump() for t in req.tools] if req.tools else None
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
                final = {
                    "model": req.model,
                    "created_at": now,
                    "message": Message(
                        role="assistant",
                        content=content_tail,
                        thinking=thinking_tail or None,
                    ).model_dump(exclude_none=True),
                    "done": True,
                    "done_reason": chunk.get("done_reason", "stop"),
                }
                if stats:
                    final.update(stats.to_dict())
                return json.dumps(final) + "\n"
            text = chunk.get("text", "")
            # Fast path: when thinking is not expected, pass every token
            # through immediately as content.  The detect buffer in
            # `_split_thinking_streaming` would otherwise hold the first
            # 200 chars of every Ollama response (issue #307 review).
            if not think_state.get("thinking_expected"):
                if not text:
                    return None
                return (
                    json.dumps(
                        {
                            "model": req.model,
                            "created_at": now,
                            "message": Message(
                                role="assistant", content=text
                            ).model_dump(exclude_none=True),
                            "done": False,
                        }
                    )
                    + "\n"
                )
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
        # `_full_completion` only populates `raw_text` for gpt-oss channel
        # format (where `text` has the channel tokens stripped and
        # `parse_model_output` needs the un-stripped form to find tool
        # calls / thinking).  For every other model the `or` falls back
        # to `text`, which is already the unstripped output.
        raw = result.get("raw_text") or result.get("text", "")
        thinking, visible, _tool_uses = parse_model_output(raw, has_tools=bool(tools))
        response = {
            "model": req.model,
            "created_at": now,
            "message": Message(
                role="assistant",
                content=visible,
                thinking=thinking or None,
            ).model_dump(exclude_none=True),
            "done": True,
            "done_reason": result.get("done_reason", "stop"),
        }
        if stats:
            response.update(stats.to_dict())
        return response
