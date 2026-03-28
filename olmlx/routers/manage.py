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


@router.delete("/api/delete")
async def delete_model(req: DeleteRequest, request: Request):
    store = request.app.state.model_store
    registry = request.app.state.registry
    deleted = store.delete(req.model)
    registry.remove(req.model)
    if not deleted:
        return JSONResponse(
            {"error": f"model '{req.model}' not found"}, status_code=404
        )
    return Response(status_code=200)


@router.post("/api/create")
async def create_model(req: CreateRequest, request: Request):
    """Parse a basic Modelfile and create a new model entry."""
    registry = request.app.state.registry

    if not req.modelfile:
        return JSONResponse({"error": "modelfile is required"}, status_code=400)

    # Parse Modelfile
    from_model = None
    parameters = {}

    for line in req.modelfile.splitlines():
        line = line.strip()
        if line.upper().startswith("FROM "):
            from_model = line[5:].strip()
        elif line.upper().startswith("SYSTEM "):
            system_prompt = line[7:].strip().strip('"')
            parameters["system"] = system_prompt
        elif line.upper().startswith("PARAMETER "):
            parts = line[10:].strip().split(None, 1)
            if len(parts) == 2:
                parameters[parts[0]] = parts[1]

    if not from_model:
        return JSONResponse({"error": "FROM is required in Modelfile"}, status_code=400)

    # Resolve the base model
    hf_path = registry.resolve(from_model)
    if hf_path is None:
        return JSONResponse(
            {"error": f"base model '{from_model}' not found"}, status_code=404
        )

    # Create alias for the new model
    normalized = registry.normalize_name(req.model)
    registry.add_alias(normalized, from_model)

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
    try:
        await manager.ensure_loaded(req.model, keep_alive=req.keep_alive)
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
    manager = request.app.state.model_manager
    try:
        unloaded = manager.unload(req.model)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    if not unloaded:
        return JSONResponse(
            {"error": f"model '{req.model}' is not loaded"}, status_code=404
        )
    return {"status": "unloaded"}
