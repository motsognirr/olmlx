"""Workaround for mlx-vlm 0.6.4 scrambling gemma4 audio conv weights on
pre-converted MLX checkpoints.

Through 0.6.3, mlx-vlm's ``load_model`` skipped ``sanitize_weights`` entirely
for checkpoints whose safetensors carry the ``format: mlx`` metadata (the
``is_mlx_format`` guard). mlx-vlm 0.6.4 removed that guard, so gemma4's
``Model.sanitize`` conv transposes — PyTorch ``[out, in, kH, kW]`` ->
MLX ``[out, kH, kW, in]`` — now also run on mlx-community checkpoints whose
audio-tower conv weights are ALREADY in MLX layout, scrambling them:

    Expected shape (128, 3, 3, 1) but received shape (128, 3, 1, 3) for
    parameter audio_tower.subsample_conv_projection.layer0.conv.weight

This breaks loading every gemma4 checkpoint that ships an audio tower
(e2b/e4b/12B/26B/31b families). Upstream main has since added shape-based
guards to the gemma4 transposes (``expected_in``), but that fix is unreleased
as of 0.6.4. Pinning ``<0.6.4`` would avoid the monkeypatch entirely; 0.6.4
is taken anyway for its server/TTS/STT endpoints, gemma4_unified module, and
prefill performance work, so this shim carries the difference.

Scope: **gemma4 only** — the removed guard was global, and other families
also ship unguarded layout-changing sanitizes in 0.6.4 (``qwen3_omni_moe``
audio conv transposes, ``phi4mm`` every-4-dim-weight transposes,
``falcon_perception``'s shape-preserving ``.w13.`` interleave,
rfdetr/rt_detr_v2/sam3 conv transposes). Pre-converted MLX checkpoints of
those families will fail (or silently corrupt) the same way. They are left
unpatched because no olmlx deployment loads them today and each needs its own
family-specific discriminant; extend this module on the same pattern if one
shows up.

The patch wraps ``Model.sanitize``: conv weights that are already in MLX
layout (detected with upstream main's exact discriminant — last dim equals
the layer's input channels) are pre-inverted to PyTorch layout so the
wrapped sanitize's transpose restores them. This stays correct if the
installed sanitize is the fixed (guarded) upstream version too: the
pre-inverted PyTorch-layout weight fails the guard's already-MLX check and
gets transposed back, so the patch never double-applies.

Applied via :func:`olmlx.engine.vlm_load.load_vlm`, the single chokepoint
every ``mlx_vlm.load`` call site uses.

Remove once the pinned mlx-vlm release carries the upstream guard —
``tests/test_gemma4_sanitize_fix.py::test_removal_gate_upstream_still_broken``
fails loudly when that time comes.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("olmlx.engine.gemma4_sanitize_fix")

# Set on the wrapper function; carries the wrapped (original) sanitize so the
# original is always recoverable from ``Model.sanitize`` itself — module
# reloads can't strand it in a stale global.
_PATCH_ATTR = "_olmlx_gemma4_sanitize_original"


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

    Safe to call unconditionally before any ``mlx_vlm.load``. Degrades to a
    no-op (WARNING log) if the gemma4 module import or attribute lookup
    fails — a broken gemma4 import must not take down loads of unrelated
    VLM families, and the removal-gate test catches the day this module
    stops being needed.
    """
    try:
        from mlx_vlm.models.gemma4.gemma4 import Model

        current = Model.sanitize
    except Exception:
        logger.warning(
            "gemma4 sanitize workaround could not be applied "
            "(mlx_vlm.models.gemma4.gemma4 import or attribute lookup "
            "failed); pre-converted gemma4 audio checkpoints may fail to "
            "load — see engine/gemma4_sanitize_fix.py",
            exc_info=True,
        )
        return

    if getattr(current, _PATCH_ATTR, None) is not None:
        return

    def sanitize(self: Any, weights: dict[str, Any]) -> dict[str, Any]:
        return current(self, _pre_invert_mlx_layout(self, weights))

    setattr(sanitize, _PATCH_ATTR, current)
    Model.sanitize = sanitize
    logger.debug("Patched mlx_vlm gemma4 Model.sanitize (MLX-layout conv guard)")


def _original_sanitize() -> Callable[..., dict[str, Any]]:
    """The unpatched upstream sanitize (for the removal-gate test).

    Read off the wrapper itself rather than module state, so it stays
    correct across module reloads."""
    from mlx_vlm.models.gemma4.gemma4 import Model

    current = Model.sanitize
    return getattr(current, _PATCH_ATTR, None) or current
