"""Workaround for mlx-vlm 0.6.x scrambling gemma4 audio conv weights on
pre-converted MLX checkpoints.

mlx-vlm 0.5.0's ``load_model`` skipped ``sanitize_weights`` entirely for
checkpoints whose safetensors carry the ``format: mlx`` metadata (the
``is_mlx_format`` guard). mlx-vlm 0.6.0 removed that guard, so gemma4's
``Model.sanitize`` conv transposes — PyTorch ``[out, in, kH, kW]`` ->
MLX ``[out, kH, kW, in]`` — now also run on mlx-community checkpoints whose
audio-tower conv weights are ALREADY in MLX layout, scrambling them:

    Expected shape (128, 3, 3, 1) but received shape (128, 3, 1, 3) for
    parameter audio_tower.subsample_conv_projection.layer0.conv.weight

This breaks loading every gemma4 checkpoint that ships an audio tower
(e2b/e4b/12B/26B/31b families). Upstream main has since added shape-based
guards to the transposes (``expected_in``), but that fix is unreleased as
of 0.6.4.

The patch wraps ``Model.sanitize``: conv weights that are already in MLX
layout (detected with upstream main's exact discriminant — last dim equals
the layer's input channels) are pre-inverted to PyTorch layout so the
wrapped sanitize's transpose restores them. This stays correct if the
installed sanitize is the fixed (guarded) upstream version too: the
pre-inverted PyTorch-layout weight fails the guard's already-MLX check and
gets transposed back, so the patch never double-applies.

Call ``ensure_gemma4_sanitize_patch()`` after ``import mlx_vlm`` and before
``mlx_vlm.load(...)``.

Remove once the pinned mlx-vlm release carries the upstream guard —
``tests/test_gemma4_sanitize_fix.py::test_removal_gate_upstream_still_broken``
fails loudly when that time comes.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("olmlx.engine.gemma4_sanitize_fix")

_PATCH_ATTR = "_olmlx_gemma4_sanitize_patch"
_original: Callable[..., dict[str, Any]] | None = None


def _pre_invert_mlx_layout(model_self: Any, weights: dict[str, Any]) -> dict[str, Any]:
    """Return ``weights`` with already-MLX-layout audio conv weights
    transposed to PyTorch layout, so the following unconditional sanitize
    transpose lands them back in MLX layout."""
    fixed: dict[str, Any] = {}
    for k, v in weights.items():
        if (
            "subsample_conv_projection" in k
            and "conv.weight" in k
            and getattr(v, "ndim", 0) == 4
        ):
            # Upstream main's discriminant: MLX layout iff the last dim is
            # the layer's input-channel count (1 for layer0, the previous
            # layer's out-channels for layer1). kH/kW are small odd kernel
            # sizes, so they can't collide with 128-ish channel counts.
            expected_in = None
            audio_config = getattr(model_self.config, "audio_config", None)
            if audio_config is not None:
                if ".layer0." in k:
                    expected_in = 1
                elif ".layer1." in k:
                    expected_in = audio_config.subsampling_conv_channels[0]
            if expected_in is not None and v.shape[-1] == expected_in:
                # Inverse of sanitize's transpose(0, 2, 3, 1).
                v = v.transpose(0, 3, 1, 2)
        elif "depthwise_conv1d.weight" in k and getattr(v, "ndim", 0) == 3:
            if v.shape[-1] == 1:
                # Depthwise conv1d has in=1: MLX layout is [out, kW, 1].
                # transpose(0, 2, 1) is its own inverse.
                v = v.transpose(0, 2, 1)
        fixed[k] = v
    return fixed


def ensure_gemma4_sanitize_patch() -> None:
    """Idempotently patch ``mlx_vlm.models.gemma4.gemma4.Model.sanitize``.

    Safe to call unconditionally before any ``mlx_vlm.load``; a no-op if
    already applied or if the module layout has changed (the removal-gate
    test catches the day this stops being needed).
    """
    global _original
    try:
        from mlx_vlm.models.gemma4.gemma4 import Model
    except ImportError:  # pragma: no cover - future mlx-vlm restructure
        logger.debug("mlx_vlm gemma4 module not found; sanitize patch skipped")
        return

    current = Model.sanitize
    if getattr(current, _PATCH_ATTR, False):
        return
    _original = current

    def sanitize(self: Any, weights: dict[str, Any]) -> dict[str, Any]:
        return current(self, _pre_invert_mlx_layout(self, weights))

    setattr(sanitize, _PATCH_ATTR, True)
    Model.sanitize = sanitize
    logger.debug("Patched mlx_vlm gemma4 Model.sanitize (MLX-layout conv guard)")


def _original_sanitize() -> Callable[..., dict[str, Any]]:
    """The unpatched upstream sanitize (for the removal-gate test)."""
    if _original is not None:
        return _original
    from mlx_vlm.models.gemma4.gemma4 import Model

    return Model.sanitize
