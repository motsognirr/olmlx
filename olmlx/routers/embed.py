from fastapi import APIRouter, Request

from olmlx.engine.inference import generate_embeddings
from olmlx.schemas.embed import (
    EmbedRequest,
    EmbedResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
)
from olmlx.utils.timing import Timer

router = APIRouter()


@router.post("/api/embed")
async def embed(req: EmbedRequest, request: Request):
    manager = request.app.state.model_manager
    texts = req.input if isinstance(req.input, list) else [req.input]

    with Timer() as t:
        embeddings, _ = await generate_embeddings(
            manager, req.model, texts, keep_alive=req.keep_alive
        )

    return EmbedResponse(
        model=req.model,
        embeddings=embeddings,
        total_duration=t.duration_ns,
    )


@router.post("/api/embeddings")
async def embeddings(req: EmbeddingsRequest, request: Request):
    manager = request.app.state.model_manager

    result, _ = await generate_embeddings(
        manager, req.model, [req.prompt], keep_alive=req.keep_alive
    )

    return EmbeddingsResponse(embedding=result[0] if result else [])
