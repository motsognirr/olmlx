from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from olmlx.schemas.models import (
    ModelDetails,
    ModelInfo,
    ShowRequest,
    ShowResponse,
    TagsResponse,
)

router = APIRouter()


@router.get("/api/tags")
async def list_models(request: Request):
    store = request.app.state.model_store
    registry = request.app.state.registry
    local_models = store.list_local()

    # Also include configured models not yet downloaded
    configured = registry.list_models()
    local_names = {m.name for m in local_models}

    models = []
    for m in local_models:
        models.append(
            ModelInfo(
                name=m.name,
                model=m.hf_path,
                modified_at=m.modified_at,
                size=m.size,
                digest=m.digest,
                details=ModelDetails(
                    format=m.format,
                    family=m.family,
                    parameter_size=m.parameter_size,
                    quantization_level=m.quantization_level,
                ),
            )
        )
    # Add configured but not-yet-pulled models
    for name, model_config in configured.items():
        normalized = registry.normalize_name(name)
        if normalized not in local_names:
            models.append(
                ModelInfo(
                    name=normalized,
                    model=model_config.hf_path,
                )
            )

    return TagsResponse(models=models)


@router.post("/api/show")
async def show_model(req: ShowRequest, request: Request):
    store = request.app.state.model_store
    manifest = store.show(req.model)
    if manifest is None:
        return JSONResponse(
            {"error": f"model '{req.model}' not found"}, status_code=404
        )
    return ShowResponse(
        details=ModelDetails(
            format=manifest.format,
            family=manifest.family,
            parameter_size=manifest.parameter_size,
            quantization_level=manifest.quantization_level,
        ),
        modified_at=manifest.modified_at,
        model_info={"hf_path": manifest.hf_path},
    )
