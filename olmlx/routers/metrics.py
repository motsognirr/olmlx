from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from olmlx.utils.metrics import REGISTRY

router = APIRouter()


@router.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint.

    Defined as a sync endpoint so FastAPI runs it in a threadpool: the lazy
    collector walks all loaded models and `generate_latest` serializes the whole
    registry, neither of which should block the event loop during a scrape.
    """
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
