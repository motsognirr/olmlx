from datetime import datetime, timezone
import time

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from olmlx import __version__
from olmlx.schemas.status import PsResponse, RunningModel, VersionResponse

router = APIRouter()


# Metrics tracking
class Metrics:
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_generation_time: float = 0.0
    model_load_count: int = 0
    model_unload_count: int = 0
    start_time: float = time.time()


metrics = Metrics()


@router.get("/")
async def root():
    return PlainTextResponse("Ollama is running")


@router.head("/")
async def root_head():
    return PlainTextResponse("Ollama is running")


@router.get("/api/version")
async def version():
    return VersionResponse(version=__version__)


@router.get("/api/ps")
async def ps(request: Request):
    manager = request.app.state.model_manager
    loaded = manager.get_loaded()
    models = []
    for lm in loaded:
        expires = ""
        if lm.expires_at is not None:
            expires = datetime.fromtimestamp(lm.expires_at, tz=timezone.utc).isoformat()
        models.append(
            RunningModel(
                name=lm.name,
                model=lm.hf_path,
                size=lm.size_bytes,
                expires_at=expires,
                size_vram=lm.size_bytes,
                active_refs=lm.active_refs,
            )
        )
    return PsResponse(models=models)


@router.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics endpoint."""
    uptime = time.time() - metrics.start_time
    avg_gen_time = (
        metrics.total_generation_time / metrics.total_requests
        if metrics.total_requests > 0
        else 0
    )
    tokens_per_sec = (
        metrics.total_tokens_generated / uptime if uptime > 0 else 0
    )

    metric_lines = [
        f'olmlx_requests_total {metrics.total_requests}',
        f'olmlx_tokens_generated_total {metrics.total_tokens_generated}',
        f'olmlx_generation_time_seconds_sum {metrics.total_generation_time}',
        f'olmlx_generation_time_seconds_count {metrics.total_requests}',
        f'olmlx_model_loads_total {metrics.model_load_count}',
        f'olmlx_model_unloads_total {metrics.model_unload_count}',
        f'olmlx_uptime_seconds {uptime:.2f}',
        f'olmlx_avg_generation_time_seconds {avg_gen_time:.4f}',
        f'olmlx_tokens_per_second {tokens_per_sec:.2f}',
    ]

    return PlainTextResponse("\n".join(metric_lines) + "\n")
