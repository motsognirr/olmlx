"""Tests for the mlx-audio ``interpolate(scale_factor=...)`` rounding fix (#703).

``mlx_audio.tts.models.interpolate.interpolate`` derives the output width from
a float product and ceils it::

    curr_size = max(1, int(math.ceil(float(input.shape[i + 2]) * float(scale_factor[i]))))

``1/300`` is not representable in binary, and the nearest double is slightly
*above* the true value, so ``70200 * (1/300) == 234.00000000000003`` and the
ceil yields 235 instead of 234. Kokoro's ``SineGen._f02sine`` round-trips the
F0 signal through ``scale_factor=1/upsample_scale`` then ``scale_factor=
upsample_scale``, so the off-by-one frame comes back as an off-by-300-sample
waveform and the ``sine_waves * uv`` broadcast blows up::

    [broadcast_shapes] Shapes (1,70200,1) and (1,70500,9) cannot be broadcast.

``olmlx/engine/mlx_audio_interpolate_fix.py`` carries the workaround until
upstream fixes the size computation.
"""

import importlib
import math

import mlx.core as mx
import pytest

interp_mod = pytest.importorskip("mlx_audio.tts.models.interpolate")

import olmlx.engine.mlx_audio_interpolate_fix as interpolate_fix  # noqa: E402
from olmlx.engine.mlx_audio_interpolate_fix import (  # noqa: E402
    _original_interpolate,
    ensure_interpolate_scale_patch,
)

# 234 frames * 300 samples/frame — the exact width from the issue report.
BROKEN_WIDTH = 70200
UPSAMPLE_SCALE = 300


def test_removal_gate_upstream_still_broken():
    # When this fails, mlx-audio has fixed the size computation and
    # olmlx/engine/mlx_audio_interpolate_fix.py can be deleted.
    raw = _original_interpolate()
    x = mx.zeros((1, 1, BROKEN_WIDTH))
    out = raw(x, scale_factor=1 / UPSAMPLE_SCALE, mode="linear")
    assert out.shape[2] == 235, (
        "upstream interpolate no longer miscomputes the scaled size; "
        "drop engine/mlx_audio_interpolate_fix.py"
    )


def test_patched_scale_factor_snaps_to_exact_integer():
    ensure_interpolate_scale_patch()
    x = mx.zeros((1, 1, BROKEN_WIDTH))
    out = interp_mod.interpolate(x, scale_factor=1 / UPSAMPLE_SCALE, mode="linear")
    assert out.shape[2] == BROKEN_WIDTH // UPSAMPLE_SCALE


def test_patched_round_trip_preserves_width():
    # down by 1/scale then up by scale must return the original width — this
    # is exactly what SineGen._f02sine does to the F0 signal.
    ensure_interpolate_scale_patch()
    for frames in (1, 7, 14, 116, 117, 124, 125, 234, 235, 240):
        width = frames * UPSAMPLE_SCALE
        x = mx.zeros((1, 1, width))
        down = interp_mod.interpolate(x, scale_factor=1 / UPSAMPLE_SCALE, mode="linear")
        up = interp_mod.interpolate(down, scale_factor=UPSAMPLE_SCALE, mode="linear")
        assert down.shape[2] == frames, f"{frames} frames: {down.shape}"
        assert up.shape[2] == width, f"{frames} frames: {up.shape}"


def test_patched_fractional_scale_still_ceils():
    # Genuinely fractional products keep upstream's ceil semantics — the fix
    # only snaps products that are an integer up to float error.
    ensure_interpolate_scale_patch()
    x = mx.zeros((1, 1, 7))
    assert interp_mod.interpolate(x, scale_factor=1.5, mode="linear").shape[2] == 11
    assert math.ceil(7 * 1.5) == 11
    assert interp_mod.interpolate(x, scale_factor=1 / 3, mode="linear").shape[2] == 3


def test_patched_explicit_size_is_untouched():
    ensure_interpolate_scale_patch()
    x = mx.zeros((1, 1, 7))
    assert interp_mod.interpolate(x, size=13, mode="linear").shape[2] == 13


def test_patched_argument_validation_preserved():
    ensure_interpolate_scale_patch()
    x = mx.zeros((1, 1, 7))
    with pytest.raises(ValueError):
        interp_mod.interpolate(x, size=4, scale_factor=2.0)
    with pytest.raises(ValueError):
        interp_mod.interpolate(x)
    with pytest.raises(ValueError):
        interp_mod.interpolate(mx.zeros((1, 7)), scale_factor=2.0)


def test_patch_is_idempotent():
    ensure_interpolate_scale_patch()
    first = interp_mod.interpolate
    ensure_interpolate_scale_patch()
    assert interp_mod.interpolate is first
    assert _original_interpolate() is not first


def test_patch_rebinds_from_import_callers():
    # istftnet does ``from ..interpolate import interpolate`` at import time,
    # so patching the defining module alone leaves an already-imported caller
    # holding the broken function.
    istftnet = pytest.importorskip("mlx_audio.tts.models.kokoro.istftnet")
    importlib.reload(istftnet)  # re-bind the pre-patch original
    ensure_interpolate_scale_patch()
    assert istftnet.interpolate is interp_mod.interpolate


def test_original_interpolate_survives_module_reload():
    # Reloading the fix module must not make it forget the unpatched upstream
    # function, or the removal gate above fires spuriously.
    ensure_interpolate_scale_patch()
    reloaded = importlib.reload(interpolate_fix)
    raw = reloaded._original_interpolate()
    assert raw is not interp_mod.interpolate
    assert not hasattr(raw, reloaded._PATCH_ATTR)


def test_sinegen_no_longer_raises_broadcast_error():
    # End-to-end reproduction of the issue: Kokoro's SineGen crashed for any
    # frame count whose ``frames * 300 * (1/300)`` product rounds up.
    istftnet = pytest.importorskip("mlx_audio.tts.models.kokoro.istftnet")
    ensure_interpolate_scale_patch()
    gen = istftnet.SineGen(24000, upsample_scale=UPSAMPLE_SCALE, harmonic_num=8)
    f0 = mx.full((1, BROKEN_WIDTH, 1), 200.0)
    sine_waves, uv, noise = gen(f0)
    assert sine_waves.shape == (1, BROKEN_WIDTH, 9)
    assert uv.shape == (1, BROKEN_WIDTH, 1)
    assert not mx.any(mx.isnan(sine_waves)).item()
