import json
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from mlx_ollama.engine.inference import generate_completion
from mlx_ollama.schemas.generate import GenerateRequest

router = APIRouter()


@router.post("/api/generate")
async def generate(req: GenerateRequest, request: Request):
    manager = request.app.state.model_manager
    options = req.options.model_dump(exclude_none=True) if req.options else {}

    prompt = req.prompt
    if req.system and not req.raw:
        prompt = f"{req.system}\n\n{prompt}"

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
        )

        async def stream_response():
            try:
                async for chunk in result:
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
                        yield json.dumps(final) + "\n"
                    else:
                        yield (
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
            finally:
                await result.aclose()

        return StreamingResponse(stream_response(), media_type="application/x-ndjson")
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
