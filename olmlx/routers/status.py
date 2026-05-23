import json
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from olmlx import __version__
from olmlx.models.store import _dir_size, _extract_metadata
from olmlx.schemas.models import ModelDetails
from olmlx.schemas.status import PsResponse, RunningModel, VersionResponse

router = APIRouter()


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
    store = request.app.state.model_store
    loaded = manager.get_loaded()
    models = []
    for lm in loaded:
        expires = ""
        if lm.expires_at is not None:
            expires = datetime.fromtimestamp(lm.expires_at, tz=timezone.utc).isoformat()

        # Resolve on-disk model metadata for details and size.
        size = lm.size_bytes
        meta = {"family": "", "parameter_size": "", "quantization_level": ""}
        digest = ""
        if store is not None:
            local_dir = store.local_path(lm.hf_path)
            if local_dir.exists():
                if size == 0:
                    size = _dir_size(local_dir)
                meta = _extract_metadata(local_dir)
                manifest_path = local_dir / "manifest.json"
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text())
                        digest = manifest.get("digest", "")
                    except Exception:
                        pass

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
            )
        )
    return PsResponse(models=models)
