import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from olmlx.engine.inference import generate_completion
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
        )

        def format_chunk(chunk):
            now = datetime.now(timezone.utc).isoformat()
            if chunk.get("done"):
                stats = chunk.get("stats")
                final = {
                    "model": req.model,
                    "created_at": now,
                    "response": "",
                    "done": True,
                    "done_reason": "stop",
                }
                if stats:
                    final.update(stats.to_dict())
                return json.dumps(final) + "\n"
            return (
                json.dumps(
                    {
                        "model": req.model,
                        "created_at": now,
                        "response": chunk.get("text", ""),
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
        )
        now = datetime.now(timezone.utc).isoformat()
        stats = result.get("stats")
        response = {
            "model": req.model,
            "created_at": now,
            "response": result.get("text", ""),
            "done": True,
            "done_reason": "stop",
        }
        if stats:
            response.update(stats.to_dict())
        return response
