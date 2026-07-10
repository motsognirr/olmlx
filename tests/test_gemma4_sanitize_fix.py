"""Tests for the mlx-vlm 0.6.x gemma4 audio-tower sanitize workaround.

mlx-vlm 0.6.0 removed the ``is_mlx_format`` guard around ``sanitize_weights``
in ``load_model``, so the gemma4 ``Model.sanitize`` conv transposes
(PyTorch [out, in, kH, kW] -> MLX [out, kH, kW, in]) now also run on
pre-converted mlx-community checkpoints whose conv weights are ALREADY in
MLX layout — scrambling them and failing the load with e.g.::

    Expected shape (128, 3, 3, 1) but received shape (128, 3, 1, 3) for
    parameter audio_tower.subsample_conv_projection.layer0.conv.weight

Upstream main has since added shape-based guards (``expected_in``) to the
transposes, but the fix is not in any released version (<= 0.6.4).
``olmlx/engine/gemma4_sanitize_fix.py`` carries the workaround until then.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

mlx_vlm_gemma4 = pytest.importorskip("mlx_vlm.models.gemma4.gemma4")

from olmlx.engine.gemma4_sanitize_fix import (  # noqa: E402
    _original_sanitize,
    ensure_gemma4_sanitize_patch,
)

L0_KEY = "audio_tower.subsample_conv_projection.layer0.conv.weight"
L1_KEY = "audio_tower.subsample_conv_projection.layer1.conv.weight"
DW_KEY = "audio_tower.conformer.0.lconv1d.depthwise_conv1d.weight"

# gemma-4-e4b: layer0 in=1 out=128 k=3x3, layer1 in=128 out=32 k=3x3,
# depthwise conv1d out=1536 kW=5 in=1 (values don't matter, shapes do).
SUBSAMPLING_CONV_CHANNELS = [128, 32]


def _fake_model_self() -> SimpleNamespace:
    """Minimal stand-in for a gemma4 Model: sanitize only touches
    ``self.config`` and ``self.audio_tower``."""
    return SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(use_clipped_linears=False),
            audio_config=SimpleNamespace(
                subsampling_conv_channels=SUBSAMPLING_CONV_CHANNELS
            ),
        ),
        audio_tower=object(),
    )


def _mlx_layout_weights() -> dict[str, mx.array]:
    return {
        L0_KEY: mx.zeros((128, 3, 3, 1)),
        L1_KEY: mx.zeros((32, 3, 3, 128)),
        DW_KEY: mx.zeros((1536, 5, 1)),
    }


def _pytorch_layout_weights() -> dict[str, mx.array]:
    return {
        L0_KEY: mx.zeros((128, 1, 3, 3)),
        L1_KEY: mx.zeros((32, 128, 3, 3)),
        DW_KEY: mx.zeros((1536, 1, 5)),
    }


def test_patched_sanitize_preserves_mlx_layout():
    # The workaround: already-MLX-layout conv weights must come out of
    # sanitize with their shapes intact.
    ensure_gemma4_sanitize_patch()
    out = mlx_vlm_gemma4.Model.sanitize(_fake_model_self(), _mlx_layout_weights())
    assert out[L0_KEY].shape == (128, 3, 3, 1)
    assert out[L1_KEY].shape == (32, 3, 3, 128)
    assert out[DW_KEY].shape == (1536, 5, 1)


def test_patched_sanitize_still_converts_pytorch_layout():
    # Unconverted (PyTorch-layout) weights must still be transposed to MLX
    # layout, exactly as upstream sanitize does.
    ensure_gemma4_sanitize_patch()
    out = mlx_vlm_gemma4.Model.sanitize(_fake_model_self(), _pytorch_layout_weights())
    assert out[L0_KEY].shape == (128, 3, 3, 1)
    assert out[L1_KEY].shape == (32, 3, 3, 128)
    assert out[DW_KEY].shape == (1536, 5, 1)


def test_patch_is_idempotent():
    ensure_gemma4_sanitize_patch()
    first = mlx_vlm_gemma4.Model.sanitize
    ensure_gemma4_sanitize_patch()
    assert mlx_vlm_gemma4.Model.sanitize is first


def test_removal_gate_upstream_still_broken():
    # REMOVAL GATE: when the installed mlx-vlm's own sanitize stops
    # scrambling already-MLX-layout conv weights (upstream main's
    # ``expected_in`` guards, unreleased as of 0.6.4), this test fails —
    # delete olmlx/engine/gemma4_sanitize_fix.py, its call sites, and this
    # test file, and drop the pyproject comment referencing it.
    ensure_gemma4_sanitize_patch()
    raw = _original_sanitize()
    out = raw(_fake_model_self(), _mlx_layout_weights())
    assert out[L0_KEY].shape != (128, 3, 3, 1), (
        "mlx-vlm's gemma4 sanitize is now layout-aware: the workaround in "
        "olmlx/engine/gemma4_sanitize_fix.py is obsolete — remove it."
    )
