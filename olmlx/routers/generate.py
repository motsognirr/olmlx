import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from olmlx.engine.grammar import parse_response_format
from olmlx.engine.inference import generate_completion
from olmlx.engine.tool_parser import parse_model_output
from olmlx.routers.common import format_error, resolve_think_flag
from olmlx.routers.thinking_split import flush_split_thinking, split_thinking_streaming
from olmlx.schemas.generate import GenerateRequest
from olmlx.utils.streaming import safe_ndjson_stream

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/generate")
async def generate(req: GenerateRequest, request: Request):
    manager = request.app.state.model_manager
    options = req.options.model_dump(exclude_none=True) if req.options else {}

    prompt = req.prompt
    max_tokens = options.pop("num_predict", 512)
    enable_thinking = resolve_think_flag(req.think)
    try:
        grammar_spec = parse_response_format(req.format)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if req.stream:
        result = await generate_completion(
            manager,
            req.model,
            prompt,
            options,
            stream=True,
            keep_alive=req.keep_alive,
            max_tokens=max_tokens,
            images=req.images,
            apply_chat_template=not req.raw,
            system=req.system if not req.raw else None,
            enable_thinking=enable_thinking,
            grammar_spec=grammar_spec,
        )

        think_state: dict = {}

        def _frame(now: str, response: str, thinking: str) -> dict:
            frame = {
                "model": req.model,
                "created_at": now,
                "response": response,
                "done": False,
            }
            if thinking:
                frame["thinking"] = thinking
            return frame

        def format_chunk(chunk):
            if "thinking_expected" in chunk:
                think_state["thinking_expected"] = bool(chunk["thinking_expected"])
                return None
            now = datetime.now(timezone.utc).isoformat()
            if chunk.get("done"):
                thinking_tail, content_tail = flush_split_thinking(think_state)
                stats = chunk.get("stats")
                lines = []
                if thinking_tail or content_tail:
                    lines.append(
                        json.dumps(_frame(now, content_tail, thinking_tail)) + "\n"
                    )
                final = {
                    "model": req.model,
                    "created_at": now,
                    "response": "",
                    "done": True,
                    "done_reason": chunk.get("done_reason", "stop"),
                }
                if stats:
                    final.update(stats.to_dict())
                lines.append(json.dumps(final) + "\n")
                return lines
            thinking_chunk, content_chunk = split_thinking_streaming(
                chunk.get("text", ""), think_state
            )
            if not thinking_chunk and not content_chunk:
                return None
            return json.dumps(_frame(now, content_chunk, thinking_chunk)) + "\n"

        return StreamingResponse(
            safe_ndjson_stream(
                result,
                format_chunk,
                lambda exc: format_error(req.model),
                log=logger,
                log_prefix="generate streaming",
            ),
            media_type="application/x-ndjson",
        )
    else:
        result = await generate_completion(
            manager,
            req.model,
            prompt,
            options,
            stream=False,
            keep_alive=req.keep_alive,
            max_tokens=max_tokens,
            images=req.images,
            apply_chat_template=not req.raw,
            system=req.system if not req.raw else None,
            enable_thinking=enable_thinking,
            grammar_spec=grammar_spec,
        )
        now = datetime.now(timezone.utc).isoformat()
        stats = result.get("stats")
        # NOTE: the non-streaming path extracts thinking via parse_model_output
        # (engine/tool_parser.py); the streaming path above uses
        # split_thinking_streaming (routers/thinking_split.py, _THINKING_PAIRS).
        # These do NOT have parity, and closing the gap is out of scope here
        # (it's a pre-existing characteristic shared with /api/chat's streaming
        # splitter):
        #   * parse_model_output also handles the gpt-oss channel format and
        #     prefix-less Gemma 4 `thought\n...` blocks; the streaming splitter
        #     handles neither (a bare `thought\n` open tag can't be matched
        #     incrementally without false-positiving on ordinary prose).
        #   * parse_model_output does not apply the streaming-only
        #     INIT_ORPHAN_DETECT_LIMIT.
        # Practical impact: a gpt-oss model with think=true + stream=true leaks
        # its channel-format thinking into `response`. When extending the tag
        # set, update both sites where the format is safe to match in a stream.
        thinking, visible_text, _ = parse_model_output(
            result.get("text", ""),
            has_tools=False,
            thinking_expected=bool(result.get("thinking_expected")),
        )
        response = {
            "model": req.model,
            "created_at": now,
            "response": visible_text,
            "done": True,
            "done_reason": result.get("done_reason", "stop"),
        }
        if thinking:
            response["thinking"] = thinking
        if stats:
            response.update(stats.to_dict())
        return response
