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

    # Check Content-Length header first for early rejection
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

    body = await request.body()

    if len(body) > MAX_BLOB_SIZE:
        return JSONResponse(
            {"error": f"blob too large (limit: {MAX_BLOB_SIZE} bytes)"},
            status_code=413,
        )

    # Verify digest
    import hashlib

    computed = "sha256:" + hashlib.sha256(body).hexdigest()
    if digest != computed:
        return JSONResponse({"error": "digest mismatch"}, status_code=400)

    await store.save_blob(digest, body)
    return Response(status_code=201)
