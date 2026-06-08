import uuid

from fastapi import APIRouter, Request

from olmlx.engine.inference import generate_rerank
from olmlx.schemas.rerank import RerankRequest, RerankResponse, RerankResult

router = APIRouter()


async def _rerank(req: RerankRequest, request: Request) -> RerankResponse:
    manager = request.app.state.model_manager
    out = await generate_rerank(
        manager,
        req.model,
        req.query,
        req.documents,
        top_n=req.top_n,
        max_tokens_per_doc=req.max_tokens_per_doc,
        return_documents=req.return_documents,
        keep_alive=req.keep_alive,
    )
    return RerankResponse(
        id=f"rerank-{uuid.uuid4().hex}",
        results=[RerankResult(**r) for r in out["results"]],
        meta={"api_version": {"version": "2"}},
    )


@router.post("/v1/rerank", response_model=RerankResponse)
async def rerank_v1(req: RerankRequest, request: Request) -> RerankResponse:
    return await _rerank(req, request)


@router.post("/rerank", response_model=RerankResponse)
async def rerank_alias(req: RerankRequest, request: Request) -> RerankResponse:
    return await _rerank(req, request)
