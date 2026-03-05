import json
import logging
import re

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_ollama.schemas.manage import CopyRequest, CreateRequest, DeleteRequest
from mlx_ollama.schemas.pull import PullRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/pull")
async def pull_model(req: PullRequest, request: Request):
    store = request.app.state.model_store

    if req.stream:
        async def stream_progress():
            try:
                async for event in store.pull(req.model):
                    yield json.dumps(event) + "\n"
            except Exception as e:
                yield json.dumps({"status": "error", "error": str(e)}) + "\n"

        return StreamingResponse(
            stream_progress(), media_type="application/x-ndjson"
        )
    else:
        events = []
        try:
            async for event in store.pull(req.model):
                events.append(event)
        except Exception as e:
            return JSONResponse(
                {"error": str(e)}, status_code=500
            )
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
        return JSONResponse({"error": f"model '{req.model}' not found"}, status_code=404)
    return Response(status_code=200)


@router.post("/api/create")
async def create_model(req: CreateRequest, request: Request):
    """Parse a basic Modelfile and create a new model entry."""
    registry = request.app.state.registry
    store = request.app.state.model_store

    if not req.modelfile:
        return JSONResponse({"error": "modelfile is required"}, status_code=400)

    # Parse Modelfile
    from_model = None
    system_prompt = None
    parameters = {}

    for line in req.modelfile.splitlines():
        line = line.strip()
        if line.upper().startswith("FROM "):
            from_model = line[5:].strip()
        elif line.upper().startswith("SYSTEM "):
            system_prompt = line[7:].strip().strip('"')
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
