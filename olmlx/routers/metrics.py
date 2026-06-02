from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from olmlx.utils.metrics import REGISTRY

router = APIRouter()


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
