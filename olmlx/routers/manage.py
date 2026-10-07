import json
import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from olmlx.schemas.manage import (
    AbortRequest,
    CopyRequest,
    CreateRequest,
    DeleteRequest,
    UnloadRequest,
    WarmupRequest,
)
from olmlx.schemas.pull import PullRequest
from olmlx.utils.streaming import safe_ndjson_stream

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/pull")
async def pull_model(req: PullRequest, request: Request):
    store = request.app.state.model_store

    if req.stream:
        return StreamingResponse(
            safe_ndjson_stream(
                store.pull(req.model),
                format_chunk=lambda e: json.dumps(e) + "\n",
                format_error=lambda e: (
                    json.dumps({"status": "error", "error": str(e)}) + "\n"
                ),
                log=logger,
                log_prefix="pull streaming",
            ),
            media_type="application/x-ndjson",
        )
    else:
        events = []
        try:
            async for event in store.pull(req.model):
                events.append(event)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        return events[-1] if events else {"status": "success"}


@router.post("/api/copy")
async def copy_model(req: CopyRequest, request: Request):
    registry = request.app.state.registry
    try:
        registry.add_alias(req.destination, req.source)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return Response(status_code=200)


def _shares_weights(registry, name: str) -> bool:
    """True when another registry entry serves the same weights as *name*
    (an alias from ``/api/copy`` or a derived ``/api/create`` model), so
    deleting *name* must leave the files in place for the others."""
    from olmlx.models.store import _strip_ollama_tag

    normalized = registry.normalize_name(name)
    target = registry.resolve(name)
    if target is None:
        return False
    hf = _strip_ollama_tag(target.hf_path)
    return any(
        other != normalized and _strip_ollama_tag(mc.hf_path) == hf
        for other, mc in registry.list_models().items()
    )


@router.delete("/api/delete")
async def delete_model(req: DeleteRequest, request: Request):
    store = request.app.state.model_store
    registry = request.app.state.registry
    in_registry = (
        registry.is_adapter(req.model)
        or registry.normalize_name(req.model) in registry.list_models()
    )
    # Like Ollama, removing one name never deletes weights another name
    # still uses (#760).
    if in_registry and not registry.is_adapter(req.model):
        shared = _shares_weights(registry, req.model)
    else:
        shared = False
    deleted = False if shared else store.delete(req.model)
    if not deleted and not in_registry:
        # Decide the 404 before touching the registry (#760).
        return JSONResponse(
            {"error": f"model '{req.model}' not found"}, status_code=404
        )
    if in_registry:
        if shared:
            # Aliases of this entry still use the kept weights: make them
            # standalone entries first, or they'd point at nothing (#760).
            registry.promote_aliases_of(req.model)
        registry.remove(req.model)
    return Response(status_code=200)


@router.post("/api/create")
async def create_model(req: CreateRequest, request: Request):
    """Create a model from a base: an alias, or — with ``SYSTEM`` and/or
    ``PARAMETER`` — a registry entry carrying them as per-model defaults.

    Accepts both a Modelfile and Ollama's structured request shape (#760).
    """
    from dataclasses import replace

    from olmlx.utils.modelfile import (
        Modelfile,
        coerce_parameters,
        parse_modelfile,
    )

    registry = request.app.state.registry

    try:
        mf = parse_modelfile(req.modelfile) if req.modelfile else Modelfile()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    unsupported = list(mf.unsupported)
    for field_name in ("template", "files", "adapters", "messages"):
        if getattr(req, field_name):
            unsupported.append(field_name)
    if unsupported:
        return JSONResponse(
            {
                "error": "unsupported by olmlx /api/create: "
                + ", ".join(dict.fromkeys(unsupported))
                + " (only FROM, SYSTEM and PARAMETER are supported)"
            },
            status_code=400,
        )

    from_model = req.from_ or mf.from_model
    if not from_model:
        error = (
            "FROM is required in Modelfile"
            if req.modelfile
            else "'from' or a modelfile with FROM is required"
        )
        return JSONResponse({"error": error}, status_code=400)
    system = req.system if req.system is not None else mf.system
    try:
        params = coerce_parameters({**mf.parameters, **(req.parameters or {})})
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Resolve the base model
    try:
        resolved = registry.resolve(from_model)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if resolved is None:
        return JSONResponse(
            {"error": f"base model '{from_model}' not found"}, status_code=404
        )

    normalized = registry.normalize_name(req.model)
    same_name = normalized == registry.normalize_name(from_model)
    if not same_name and normalized in registry.alias_chain(from_model):
        # Replacing *normalized* would cut the base out from under itself.
        return JSONResponse(
            {
                "error": f"cannot create '{req.model}' from '{from_model}': "
                f"'{from_model}' is an alias of '{req.model}'"
            },
            status_code=400,
        )
    mc = None
    if system is not None or params:
        # Build (and so validate) the new entry before touching the registry:
        # a rejected value must leave an existing model of this name intact.
        try:
            mc = replace(
                resolved,
                options={**resolved.options, **params},
                system=system if system is not None else resolved.system,
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
    try:
        # Re-creating a name replaces it: drop the earlier alias/entry first,
        # since resolve() prefers an alias and would shadow a new mapping.
        # Skipped when the new name *is* the base (already resolved above).
        if not same_name:
            # Aliases of the old entry keep what they pointed at (Ollama's
            # copy semantics) instead of dangling or following the new one.
            registry.promote_aliases_of(normalized)
            registry.remove(normalized)
        if mc is None:
            registry.add_alias(normalized, from_model)
        else:
            registry.add_mapping(normalized, mc.hf_path, model_config=mc)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if req.stream:

        async def stream():
            yield json.dumps({"status": "reading model metadata"}) + "\n"
            yield json.dumps({"status": "creating model layer"}) + "\n"
            yield json.dumps({"status": "success"}) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")
    return {"status": "success"}


@router.post("/api/push")
async def push_model():
    return JSONResponse(
        {"error": "push is not supported; models are stored on HuggingFace"},
        status_code=501,
    )


@router.post("/api/warmup")
async def warmup_model(req: WarmupRequest, request: Request):
    """Preload a model into VRAM to reduce first-request latency."""
    manager = request.app.state.model_manager
    from olmlx.engine.model_manager import ModelNotFoundError

    try:
        await manager.ensure_loaded(req.model, keep_alive=req.keep_alive)
    except ModelNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except (ValueError, RuntimeError) as e:
        return JSONResponse(
            {"error": f"warmup failed: {e}"},
            status_code=400,
        )
    return {"status": "loaded"}


@router.post("/api/abort")
async def abort_generation(req: AbortRequest, request: Request):
    """Cancel an in-progress generation.

    Note: This is a no-op in the current implementation since we buffer
    output for tool parsing. The generation will complete but the client
    can simply disconnect to stop receiving chunks.
    """
    logger.info(
        "Abort requested for model %s (no-op, client should disconnect)", req.model
    )
    return {
        "status": "no-op",
        "message": "client should disconnect to cancel generation",
    }


@router.post("/api/unload")
async def unload_model(req: UnloadRequest, request: Request):
    """Manually unload a model from VRAM."""
    from olmlx.engine.model_manager import ActiveRequestsError

    manager = request.app.state.model_manager
    try:
        unloaded = await manager.unload(req.model)
    except ActiveRequestsError as e:
        # 409 is narrow on purpose: only "model has active requests".
        # Resource-close failures inside ``_close_loaded_model`` are
        # absorbed by ``ModelManager.unload`` itself (the model is gone
        # from ``_loaded`` regardless), so the router never sees them —
        # they show up only in the per-resource log lines inside the
        # helper.
        return JSONResponse({"error": str(e)}, status_code=409)
    if not unloaded:
        return JSONResponse(
            {"error": f"model '{req.model}' is not loaded"}, status_code=404
        )
    return {"status": "unloaded"}
