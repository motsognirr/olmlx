import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from olmlx.engine.inference import generate_chat
from olmlx.schemas.chat import ChatRequest, Message
from olmlx.utils.streaming import safe_ndjson_stream

logger = logging.getLogger(__name__)

router = APIRouter()


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

        def format_chunk(chunk):
            if chunk.get("cache_info"):
                return None
            now = datetime.now(timezone.utc).isoformat()
            if chunk.get("done"):
                stats = chunk.get("stats")
                final = {
                    "model": req.model,
                    "created_at": now,
                    "message": Message(role="assistant", content="").model_dump(),
                    "done": True,
                    "done_reason": "stop",
                }
                if stats:
                    final.update(stats.to_dict())
                return json.dumps(final) + "\n"
            text = chunk.get("text", "")
            return (
                json.dumps(
                    {
                        "model": req.model,
                        "created_at": now,
                        "message": Message(role="assistant", content=text).model_dump(),
                        "done": False,
                    }
                )
                + "\n"
            )

        def format_error(exc):
            return (
                json.dumps(
                    {
                        "model": req.model,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "error": "An internal server error occurred during streaming.",
                        "done": True,
                        "done_reason": "error",
                    }
                )
                + "\n"
            )

        return StreamingResponse(
            safe_ndjson_stream(
                result,
                format_chunk,
                format_error,
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
        response = {
            "model": req.model,
            "created_at": now,
            "message": Message(
                role="assistant", content=result.get("text", "")
            ).model_dump(),
            "done": True,
            "done_reason": "stop",
        }
        if stats:
            response.update(stats.to_dict())
        return response
