from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from olmlx import __version__
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
        # Pull disk-side metadata (family, quantization, digest, on-disk size)
        # from the store — derived from manifest.json when present, or from
        # config.json + dir contents otherwise (issue #340).
        manifest = store.show(lm.name)
        if manifest is not None:
            details = ModelDetails(
                format=manifest.format,
                family=manifest.family,
                parameter_size=manifest.parameter_size,
                quantization_level=manifest.quantization_level,
            )
            digest = manifest.digest
            disk_size = manifest.size or lm.size_bytes
        else:
            details = ModelDetails()
            digest = ""
            disk_size = lm.size_bytes
        models.append(
            RunningModel(
                name=lm.name,
                model=lm.hf_path,
                size=disk_size,
                digest=digest,
                details=details,
                expires_at=expires,
                size_vram=lm.size_bytes,
                active_refs=lm.active_refs,
            )
        )
    return PsResponse(models=models)
