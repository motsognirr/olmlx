"""OpenAI-compatible text-to-image endpoint (#723)."""

import asyncio
import base64
import contextlib
import logging
import threading
import time

from fastapi import APIRouter, HTTPException, Request, Response

from olmlx.config import settings
from olmlx.engine.image_gen import ImageGenerationCancelled, encode_image
from olmlx.engine.inference import generate_image
from olmlx.schemas.images import (
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def _watch_disconnect(request: Request, cancel: threading.Event) -> None:
    """Set *cancel* when the client goes away.

    Non-streaming handlers are not cancelled by the server on disconnect, so
    without this a closed client would still pay for a full multi-minute
    generation under the inference lock. This blocks on ``request.receive()``
    rather than polling ``request.is_disconnected()``: the latter receives
    under a pre-cancelled scope, and behind the app's ``BaseHTTPMiddleware``s
    that drops the ``http.disconnect`` message, so it never reports one. The
    request body has already been read by FastAPI, so the next ASGI message is
    the disconnect.
    """
    while not cancel.is_set():
        message = await request.receive()
        if message["type"] == "http.disconnect":
            logger.info("Client disconnected; cancelling image generation")
            cancel.set()
            return


@router.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def images_generations(req: ImageGenerationRequest, request: Request):
    if len(req.prompt) > settings.image_max_prompt_chars:
        raise HTTPException(
            status_code=413,
            detail=(
                f"prompt exceeds {settings.image_max_prompt_chars} characters "
                "(OLMLX_IMAGE_MAX_PROMPT_CHARS)."
            ),
        )
    width, height = req.dimensions
    limit = settings.image_max_dimension
    if width > limit or height > limit:
        raise ValueError(
            f"size {req.size} exceeds the {limit}px limit (OLMLX_IMAGE_MAX_DIMENSION)"
        )

    manager = request.app.state.model_manager
    cancel = threading.Event()
    watcher = asyncio.create_task(_watch_disconnect(request, cancel))
    try:
        out = await generate_image(
            manager,
            req.model,
            req.prompt,
            width=width,
            height=height,
            seed=req.seed,
            steps=req.steps,
            guidance=req.guidance,
            negative_prompt=req.negative_prompt,
            keep_alive=req.keep_alive,
            cancel_event=cancel,
        )
    except ImageGenerationCancelled:
        # The client is gone; nobody reads this. 499 = client closed request.
        return Response(status_code=499)
    finally:
        watcher.cancel()
        # The watcher's outcome is irrelevant here; never let its failure
        # (e.g. a receive() error once the request completes) replace the
        # real response or exception.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watcher

    # PIL encoding is CPU-bound (hundreds of ms for a 1024px PNG); keep it off
    # the event loop.
    data = await asyncio.to_thread(encode_image, out["image"], req.output_format)
    return ImageGenerationResponse(
        created=int(time.time()),
        data=[
            ImageData(b64_json=base64.b64encode(data).decode("ascii"), seed=out["seed"])
        ],
        output_format=req.output_format,
        size=f"{width}x{height}",
    )
