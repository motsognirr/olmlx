from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

router = APIRouter()


@router.head("/api/blobs/{digest}")
async def check_blob(digest: str, request: Request):
    store = request.app.state.model_store
    if store.has_blob(digest):
        return Response(status_code=200)
    return Response(status_code=404)


@router.post("/api/blobs/{digest}")
async def upload_blob(digest: str, request: Request):
    store = request.app.state.model_store
    body = await request.body()

    # Verify digest
    import hashlib

    computed = "sha256:" + hashlib.sha256(body).hexdigest()
    if digest != computed:
        return JSONResponse({"error": "digest mismatch"}, status_code=400)

    await store.save_blob(digest, body)
    return Response(status_code=201)
