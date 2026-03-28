import hashlib
import os
import tempfile

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

router = APIRouter()

# Maximum blob upload size: 10 GB
MAX_BLOB_SIZE = 10 * 1024 * 1024 * 1024


@router.head("/api/blobs/{digest}")
async def check_blob(digest: str, request: Request):
    store = request.app.state.model_store
    if store.has_blob(digest):
        return Response(status_code=200)
    return Response(status_code=404)


@router.post("/api/blobs/{digest}")
async def upload_blob(digest: str, request: Request):
    store = request.app.state.model_store

    # Fast-path: reject via Content-Length header before reading the body
    content_length = request.headers.get("content-length")
    try:
        cl = int(content_length) if content_length is not None else 0
    except ValueError:
        cl = 0
    if cl > MAX_BLOB_SIZE:
        return JSONResponse(
            {"error": f"blob too large (limit: {MAX_BLOB_SIZE} bytes)"},
            status_code=413,
        )

    # Stream to a temp file with O(chunk) memory and incremental digest.
    blobs_dir = store.models_dir / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=blobs_dir)
    try:
        hasher = hashlib.sha256()
        received = 0
        with os.fdopen(tmp_fd, "wb") as tmp:
            async for chunk in request.stream():
                received += len(chunk)
                if received > MAX_BLOB_SIZE:
                    os.unlink(tmp_path)
                    return JSONResponse(
                        {"error": f"blob too large (limit: {MAX_BLOB_SIZE} bytes)"},
                        status_code=413,
                    )
                tmp.write(chunk)
                hasher.update(chunk)

        computed = "sha256:" + hasher.hexdigest()
        if digest != computed:
            os.unlink(tmp_path)
            return JSONResponse({"error": "digest mismatch"}, status_code=400)

        os.replace(tmp_path, blobs_dir / digest)
        return Response(status_code=201)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
