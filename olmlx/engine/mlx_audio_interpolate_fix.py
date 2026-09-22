"""Workaround for mlx-audio computing scaled interpolation sizes in float.

``mlx_audio.tts.models.interpolate.interpolate`` derives the output width from
a ``scale_factor`` like this::

    curr_size = max(
        1, int(math.ceil(float(input.shape[i + 2]) * float(scale_factor[i])))
    )

``1/300`` has no exact binary representation and the nearest double sits just
*above* the true value, so ``70200 * (1 / 300) == 234.00000000000003`` and the
ceil returns 235 where 234 is meant. The error is silent everywhere the result
is only used as a length — until something has to line up with it.

Kokoro's vocoder is exactly that case. ``SineGen._f02sine``
(``mlx_audio/tts/models/kokoro/istftnet.py``) round-trips the upsampled F0
signal through ``scale_factor=1/upsample_scale`` and then back through
``scale_factor=upsample_scale``, so a single spurious frame returns as 300
extra samples and the ``sine_waves * uv`` product fails to broadcast::

    [broadcast_shapes] Shapes (1,70200,1) and (1,70500,9) cannot be broadcast.

This is deterministic per utterance and hits ~18% of frame counts (the ones
whose ``frames * 300 * (1 / 300)`` product lands above the integer), which is
why /v1/audio/speech failed for some ordinary sentences and voices but not
others — the predicted duration, not the text or the voice, decides it. See
issue #703.

The patch converts ``scale_factor`` to an explicit ``size`` before delegating
to the upstream function, snapping products that are integral up to float
error. Genuinely fractional products keep upstream's ceil semantics, so the
only behaviour that changes is the crash. All interpolation math stays
upstream.

Scope: the whole ``interpolate`` helper, not just Kokoro — ``kitten_tts`` and
``soprano`` import the same function and round-trip scale factors the same
way. Upstream ``main`` still carries the float ceil as of mlx-audio 0.5.5, so
there is no released version to upgrade to;
``test_removal_gate_upstream_still_broken`` fails the day there is.

Applied via :meth:`olmlx.engine.model_manager.ModelManager._load_model_tts`,
the single ``mlx_audio`` chokepoint, before ``load_model`` imports the model
modules that bind ``interpolate`` by name.
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Any, Callable, Sequence

logger = logging.getLogger("olmlx")

_PATCH_ATTR = "_olmlx_interpolate_original"

# Relative slack for treating a scaled size as integral. The float error from
# a reciprocal scale factor is a handful of ulps (~1e-16 relative at these
# magnitudes); 1e-9 is far above that and far below any real fractional size.
_INTEGRAL_TOL = 1e-9


def _scaled_sizes(shape: Sequence[int], scale_factor: Any) -> list[int]:
    """Output spatial sizes for ``scale_factor``, immune to reciprocal error.

    Mirrors upstream's ceil, except a product that is an integer up to float
    error is snapped to that integer first.
    """
    spatial_dims = len(shape) - 2
    if not isinstance(scale_factor, (list, tuple)):
        scale_factor = [scale_factor] * spatial_dims

    sizes: list[int] = []
    for i in range(spatial_dims):
        raw = float(shape[i + 2]) * float(scale_factor[i])
        nearest = round(raw)
        if nearest >= 1 and math.isclose(
            raw, nearest, rel_tol=_INTEGRAL_TOL, abs_tol=_INTEGRAL_TOL
        ):
            raw = float(nearest)
        sizes.append(max(1, int(math.ceil(raw))))
    return sizes


def _rebind_import_callers(
    original: Callable[..., Any], patched: Callable[..., Any]
) -> None:
    """Point modules that did ``from ..interpolate import interpolate`` at the patch.

    A ``from X import Y`` binding made before the patch holds the original
    function object, so patching the defining module alone would leave
    already-imported callers (kokoro/kitten_tts/soprano) broken. Modules
    imported *after* the patch pick it up through the defining module.
    """
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            if getattr(module, "interpolate", None) is original:
                module.interpolate = patched  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - exotic module __getattr__
            continue


def ensure_interpolate_scale_patch() -> None:
    """Idempotently patch ``mlx_audio.tts.models.interpolate.interpolate``.

    Safe to call unconditionally before any ``mlx_audio`` model load. Degrades
    to a no-op (WARNING log) if the import or attribute lookup fails — an
    mlx-audio layout change must not take down TTS loading outright, and the
    removal-gate test catches the day this module stops being needed.
    """
    try:
        from mlx_audio.tts.models import interpolate as interp_mod

        current = interp_mod.interpolate
    except Exception:
        logger.warning(
            "mlx-audio interpolate workaround could not be applied "
            "(mlx_audio.tts.models.interpolate import or attribute lookup "
            "failed); Kokoro speech may fail with a [broadcast_shapes] error "
            "— see engine/mlx_audio_interpolate_fix.py",
            exc_info=True,
        )
        return

    if getattr(current, _PATCH_ATTR, None) is not None:
        return

    def interpolate(
        input: Any,
        size: Any = None,
        scale_factor: Any = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> Any:
        # Only the scale_factor path is miscomputed. Leave every other call
        # shape — explicit size, both given, neither given, <3D input — to
        # upstream so its ValueErrors are raised unchanged.
        if size is None and scale_factor is not None and getattr(input, "ndim", 0) >= 3:
            size = _scaled_sizes(input.shape, scale_factor)
            scale_factor = None
        return current(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )

    setattr(interpolate, _PATCH_ATTR, current)
    interp_mod.interpolate = interpolate
    _rebind_import_callers(current, interpolate)
    logger.debug("Patched mlx_audio interpolate (integral scale_factor snap)")


def _original_interpolate() -> Callable[..., Any]:
    """The unpatched upstream function (for the removal-gate test).

    Read off the wrapper itself rather than module state, so it stays correct
    across module reloads."""
    from mlx_audio.tts.models import interpolate as interp_mod

    current = interp_mod.interpolate
    return getattr(current, _PATCH_ATTR, None) or current
