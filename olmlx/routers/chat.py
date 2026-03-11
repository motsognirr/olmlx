import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from olmlx.engine.inference import generate_chat
from olmlx.schemas.chat import ChatRequest, Message

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    manager = request.app.state.model_manager
    options = req.options.model_dump(exclude_none=True) if req.options else {}
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    tools = [t.model_dump() for t in req.tools] if req.tools else None
    max_tokens = options.pop("num_predict", 512)

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
        )

        async def stream_response():
            try:
                full_text = ""
                async for chunk in result:
                    if chunk.get("cache_info"):
                        continue
                    now = datetime.now(timezone.utc).isoformat()
                    if chunk.get("done"):
                        stats = chunk.get("stats")
                        final = {
                            "model": req.model,
                            "created_at": now,
                            "message": Message(
                                role="assistant", content=""
                            ).model_dump(),
                            "done": True,
                            "done_reason": "stop",
                        }
                        if stats:
                            final.update(stats.to_dict())
                        yield json.dumps(final) + "\n"
                    else:
                        text = chunk.get("text", "")
                        full_text += text
                        yield (
                            json.dumps(
                                {
                                    "model": req.model,
                                    "created_at": now,
                                    "message": Message(
                                        role="assistant", content=text
                                    ).model_dump(),
                                    "done": False,
                                }
                            )
                            + "\n"
                        )
            except Exception as exc:
                logger.error("Error during chat streaming: %s", exc, exc_info=True)
                yield (
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
            finally:
                await result.aclose()

        return StreamingResponse(stream_response(), media_type="application/x-ndjson")
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
