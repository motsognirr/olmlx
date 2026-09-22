"""OpenAI-compatible text-to-image endpoint (#723)."""

import asyncio
import base64
import contextlib
import logging
import threading
import time

from fastapi import APIRouter, Request, Response

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

# How often to poll for a client disconnect while a generation runs. The
# worker checks the cancel event once per diffusion step (~seconds each), so a
# sub-second poll adds no meaningful latency to the abort.
_DISCONNECT_POLL_S = 0.5


async def _watch_disconnect(request: Request, cancel: threading.Event) -> None:
    """Set *cancel* when the client goes away (non-streaming handlers are not
    cancelled by the server on disconnect, so without this a closed client
    would still pay for a full multi-minute generation under the lock)."""
    while not cancel.is_set():
        if await request.is_disconnected():
            logger.info("Client disconnected; cancelling image generation")
            cancel.set()
            return
        await asyncio.sleep(_DISCONNECT_POLL_S)


@router.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def images_generations(req: ImageGenerationRequest, request: Request):
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
        cancel.set()  # stops the watcher loop
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
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
