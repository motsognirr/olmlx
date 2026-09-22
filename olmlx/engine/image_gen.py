"""mflux text-to-image adapter (#723).

The thin layer between olmlx and mflux: which mflux variants we serve, how a
declared ``hf_path`` maps to one (exact match only), how a model is built, and
how one image is generated with per-step cancellation. Everything here is
synchronous and runs on a worker thread; the inference lock, model lifecycle
and HTTP surface live in ``inference.generate_image`` / ``routers/images.py``.

mflux is an optional dependency (the ``[image]`` extra), so it is imported
lazily inside each function — importing this module never requires it.
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImageVariant:
    """One servable mflux model family."""

    #: ``mflux.models.common.config.model_config.AVAILABLE_MODELS`` key.
    key: str
    module: str
    class_name: str


# The two variants are not constructor-compatible (QwenImage takes LoRA args,
# QwenImage21 does not), but both accept the same ``generate_image`` keywords
# olmlx exposes — seed, prompt, steps, size, guidance, negative_prompt — and
# carry different *defaults* (4 steps / guidance 4.0 vs 40 / 1.0). Omitting an
# unset keyword lets each variant apply its own default.
_VARIANTS: tuple[ImageVariant, ...] = (
    ImageVariant(
        key="qwen-image",  # Qwen/Qwen-Image-2512 (20B MMDiT)
        module="mflux.models.qwen.variants.txt2img.qwen_image",
        class_name="QwenImage",
    ),
    ImageVariant(
        key="qwen-image-2.1",  # Qwen/Qwen-Image-2.1 (7.1B DiT + Qwen3-VL encoder)
        module="mflux.models.qwen21.variants.txt2img.qwen_image_21",
        class_name="QwenImage21",
    ),
)


class ImageGenerationCancelled(Exception):
    """Raised from the per-step mflux callback when the request is cancelled."""


def resolve_image_variant(hf_path: str) -> tuple[ImageVariant, Any]:
    """Map a declared ``hf_path`` to ``(variant, mflux ModelConfig)``.

    Exact string match against each supported variant's ``model_name`` and
    ``aliases`` — never ``ModelConfig.from_name``, which is a loose substring
    matcher that resolves ``Qwen/Qwen3-32B-4bit`` (a text LLM) to a Qwen-Image
    base. A miss raises ``ValueError`` so a typo fails at load with a clear
    message before the store downloads tens of GB of the wrong thing.

    Raises ``ImportError`` if mflux is not installed.
    """
    from mflux.models.common.config.model_config import (  # type: ignore[import-not-found]
        AVAILABLE_MODELS,
    )

    supported: list[str] = []
    for variant in _VARIANTS:
        try:
            cfg = AVAILABLE_MODELS[variant.key]
        except KeyError as exc:
            # mflux internals drifted (the extra is upper-bounded for this).
            # ImportError so the loader reports "incompatible mflux".
            raise ImportError(
                f"mflux has no '{variant.key}' model config (AVAILABLE_MODELS)"
            ) from exc
        try:
            model_name = cfg.model_name
            names = {model_name, *cfg.aliases}
        except (AttributeError, TypeError) as exc:
            raise ImportError(
                f"mflux '{variant.key}' model config has no model_name/aliases"
            ) from exc
        if hf_path in names:
            return variant, cfg
        supported.append(model_name)
    raise ValueError(
        f"'{hf_path}' is not a supported image model. Declare one of "
        f'{sorted(supported)} as the hf_path of a "type": "image" entry '
        "in models.json (exact repo id)."
    )


def load_image_model(hf_path: str, quantize: int | None, model_path: str) -> Any:
    """Build the mflux model for *hf_path* from the local directory *model_path*.

    *model_path* is the olmlx ModelStore directory the repo was downloaded
    into (``OLMLX_MODELS_DIR``). It must always be passed: with
    ``model_path=None`` mflux resolves and downloads the repo itself, into the
    Hugging Face cache, bypassing olmlx's model storage. Raises
    ``ImportError`` if mflux is missing and ``ValueError`` for an unsupported
    ``hf_path``.
    """
    variant, mflux_config = resolve_image_variant(hf_path)
    module = importlib.import_module(variant.module)
    try:
        cls = getattr(module, variant.class_name)
    except AttributeError as exc:
        raise ImportError(
            f"mflux module {variant.module} has no {variant.class_name}"
        ) from exc
    return cls(quantize=quantize, model_path=model_path, model_config=mflux_config)


class _CancelCallback:
    """mflux in-loop subscriber that aborts the denoise loop on cancel.

    ``GenerationContext.in_loop`` calls every subscriber after each diffusion
    step; the loop's ``try`` only catches ``KeyboardInterrupt``, so raising
    here propagates cleanly out of ``generate_image``.
    """

    def __init__(self, cancel_event: threading.Event):
        self._cancel_event = cancel_event

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps) -> None:  # noqa: ARG002
        if self._cancel_event.is_set():
            # mflux calls in-loop callbacks BEFORE its per-step
            # ``mx.eval(latents)``. Materialize this step's graph first: it
            # feeds persistent model state (Qwen21Transformer._geometry_cache
            # rope/mask arrays built on the first step), which would otherwise
            # stay lazy and bound to this worker thread and crash the next
            # request's worker with "There is no Stream(gpu, N)".
            if latents is not None:
                import mlx.core as mx

                mx.eval(latents)
            raise ImageGenerationCancelled(f"image generation cancelled at step {t}")


def generate_image(
    model: Any,
    prompt: str,
    *,
    seed: int,
    width: int,
    height: int,
    steps: int | None = None,
    guidance: float | None = None,
    negative_prompt: str | None = None,
    cancel_event: threading.Event | None = None,
) -> Any:
    """Generate one image; return the ``PIL.Image.Image``.

    Must run on a single worker thread under the inference lock (the model's
    callback registry is mutated for the duration of the call).
    """
    kwargs: dict[str, Any] = {
        "seed": seed,
        "prompt": prompt,
        "width": width,
        "height": height,
    }
    if steps is not None:
        kwargs["num_inference_steps"] = steps
    if guidance is not None:
        kwargs["guidance"] = guidance
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    callback = None
    if cancel_event is not None:
        if cancel_event.is_set():
            raise ImageGenerationCancelled("image generation cancelled before start")
        callback = _CancelCallback(cancel_event)
        model.callbacks.register(callback)
    try:
        return model.generate_image(**kwargs).image
    finally:
        if callback is not None:
            in_loop = model.callbacks.in_loop
            if callback in in_loop:
                in_loop.remove(callback)
        # mflux memoizes text-encoder outputs per prompt in ``prompt_cache`` with
        # no bound; on a long-lived server that grows without limit. Drop it
        # after every request (prefix caching is out of scope for v1).
        prompt_cache = getattr(model, "prompt_cache", None)
        if isinstance(prompt_cache, dict):
            prompt_cache.clear()


#: ``output_format`` -> PIL save format.
IMAGE_FORMATS: dict[str, str] = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}


def encode_image(image: Any, output_format: str) -> bytes:
    """Encode a PIL image to bytes (``png`` / ``jpeg`` / ``webp``)."""
    import io

    pil_format = IMAGE_FORMATS[output_format]
    if pil_format == "JPEG" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")  # JPEG has no alpha channel
    buf = io.BytesIO()
    image.save(buf, format=pil_format)
    return buf.getvalue()
