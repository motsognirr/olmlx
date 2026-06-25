import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, PlainTextResponse

from olmlx import __version__
from olmlx.schemas.models import ModelDetails
from olmlx.schemas.status import PsResponse, RunningModel, VersionResponse

router = APIRouter()

_UI_INDEX = Path(__file__).resolve().parent.parent / "ui" / "index.html"


def _wants_html(request: Request) -> bool:
    """Browsers send ``Accept: text/html``; API/heartbeat clients (curl, the
    Ollama Go client) send ``*/*`` or a JSON type. Only browsers get the UI so
    existing ``/`` heartbeat integrations keep their plain-text response."""
    return "text/html" in request.headers.get("accept", "") and _UI_INDEX.exists()


@router.get("/")
async def root(request: Request):
    if _wants_html(request):
        return FileResponse(_UI_INDEX, media_type="text/html")
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
    store = request.app.state.model_store
    loaded = manager.get_loaded()
    models = []
    for lm in loaded:
        expires = ""
        if lm.expires_at is not None:
            expires = datetime.fromtimestamp(lm.expires_at, tz=timezone.utc).isoformat()

        # Read metadata from the manifest (backfilled at load time) to avoid
        # expensive _dir_size / _extract_metadata calls on the event loop.
        size = lm.size_bytes
        meta = {"family": "", "parameter_size": "", "quantization_level": ""}
        digest = ""
        if store is not None:
            local_dir = store.local_path(lm.hf_path)
            manifest_path = local_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    m = json.loads(manifest_path.read_text())
                    if size == 0:
                        size = m.get("size", 0)
                    digest = m.get("digest", "")
                    meta["family"] = m.get("family", "")
                    meta["parameter_size"] = m.get("parameter_size", "")
                    meta["quantization_level"] = m.get("quantization_level", "")
                except Exception:
                    pass

        # Override with HQQ weight quantization if applied at load time.
        if lm.weight_quant:
            q_str = lm.weight_quant
            parts = q_str.split(":")
            bits = parts[1] if len(parts) > 1 else ""
            meta["quantization_level"] = f"HQQ-{bits}bit"

        cache_metrics: dict[str, int] = {}
        cache_store = getattr(lm, "prompt_cache_store", None)
        if cache_store is not None and hasattr(cache_store, "metrics"):
            cache_metrics = cache_store.metrics.to_dict()
        vlm_store = getattr(lm, "vlm_prompt_cache_store", None)
        if vlm_store is not None:
            cache_metrics = {**cache_metrics, **vlm_store.metrics()}

        # Continuous-batching occupancy (batching plan §8).
        batch_metrics: dict[str, int] = {}
        scheduler = getattr(lm, "batch_scheduler", None)
        if scheduler is not None and hasattr(scheduler, "stats"):
            batch_metrics = scheduler.stats()

        models.append(
            RunningModel(
                name=lm.name,
                model=lm.hf_path,
                size=size,
                digest=digest,
                details=ModelDetails(
                    format="mlx",
                    family=meta["family"],
                    parameter_size=meta["parameter_size"],
                    quantization_level=meta["quantization_level"],
                ),
                expires_at=expires,
                size_vram=size,
                active_refs=lm.active_refs,
                cache_metrics=cache_metrics,
                batch_metrics=batch_metrics,
                adapter_base=getattr(lm, "adapter_base", None),
            )
        )
    return PsResponse(models=models)
