"""Tests for the mlx-vlm 0.6.4 gemma4 audio-tower sanitize workaround.

mlx-vlm 0.6.4 removed the ``is_mlx_format`` guard around ``sanitize_weights``
in ``load_model`` (present through 0.6.3), so the gemma4 ``Model.sanitize``
conv transposes (PyTorch [out, in, kH, kW] -> MLX [out, kH, kW, in]) now also
run on pre-converted mlx-community checkpoints whose conv weights are ALREADY
in MLX layout — scrambling them and failing the load with e.g.::

    Expected shape (128, 3, 3, 1) but received shape (128, 3, 1, 3) for
    parameter audio_tower.subsample_conv_projection.layer0.conv.weight

Upstream main has since added shape-based guards (``expected_in``) to the
gemma4 transposes, but the fix is not in any released version (<= 0.6.4).
``olmlx/engine/gemma4_sanitize_fix.py`` carries the workaround until then.
"""

import importlib
from types import SimpleNamespace

import mlx.core as mx
import pytest

mlx_vlm_gemma4 = pytest.importorskip("mlx_vlm.models.gemma4.gemma4")

import olmlx.engine.gemma4_sanitize_fix as gemma4_sanitize_fix  # noqa: E402
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

MLX_SHAPES = {
    L0_KEY: (128, 3, 3, 1),
    L1_KEY: (32, 3, 3, 128),
    DW_KEY: (1536, 5, 1),
}

PYTORCH_SHAPES = {
    L0_KEY: (128, 1, 3, 3),
    L1_KEY: (32, 128, 3, 3),
    DW_KEY: (1536, 1, 5),
}


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


@pytest.mark.parametrize(
    "shapes",
    [MLX_SHAPES, PYTORCH_SHAPES],
    ids=["already-mlx-layout-preserved", "pytorch-layout-converted"],
)
def test_patched_sanitize_lands_in_mlx_layout(shapes):
    # The workaround: already-MLX-layout conv weights must come out of
    # sanitize with their shapes intact, while unconverted (PyTorch-layout)
    # weights must still be transposed to MLX layout as upstream does.
    ensure_gemma4_sanitize_patch()
    weights = {k: mx.zeros(shape) for k, shape in shapes.items()}
    out = mlx_vlm_gemma4.Model.sanitize(_fake_model_self(), weights)
    for k, expected in MLX_SHAPES.items():
        assert out[k].shape == expected


def test_patch_is_idempotent():
    ensure_gemma4_sanitize_patch()
    first = mlx_vlm_gemma4.Model.sanitize
    ensure_gemma4_sanitize_patch()
    assert mlx_vlm_gemma4.Model.sanitize is first


def test_original_sanitize_survives_module_reload():
    # The original must be recoverable from the wrapper itself, not module
    # state: after a reload of the fix module (fresh globals) with the patch
    # already applied, _original_sanitize() must still return the unpatched
    # upstream function — otherwise the removal gate below fires spuriously.
    ensure_gemma4_sanitize_patch()
    reloaded = importlib.reload(gemma4_sanitize_fix)
    reloaded.ensure_gemma4_sanitize_patch()
    raw = reloaded._original_sanitize()
    assert getattr(raw, reloaded._PATCH_ATTR, None) is None


def test_removal_gate_upstream_still_broken():
    # REMOVAL GATE: when the installed mlx-vlm's own sanitize stops
    # scrambling already-MLX-layout conv weights (upstream main's
    # ``expected_in`` guards, unreleased as of 0.6.4), this test fails.
    # Then: delete olmlx/engine/gemma4_sanitize_fix.py, the ensure call in
    # olmlx/engine/vlm_load.py, this test file, and the pyproject comment
    # referencing the workaround. NOTE before assuming all is well: the
    # 0.6.4 regression (removed is_mlx_format guard) also affects other
    # families (qwen3_omni_moe, phi4mm, falcon_perception, rfdetr,
    # rt_detr_v2, sam3) — check whether the release that fixes gemma4 also
    # guards those before loading their pre-converted checkpoints.
    ensure_gemma4_sanitize_patch()
    raw = _original_sanitize()
    weights = {k: mx.zeros(shape) for k, shape in MLX_SHAPES.items()}
    out = raw(_fake_model_self(), weights)
    assert out[L0_KEY].shape != MLX_SHAPES[L0_KEY], (
        "mlx-vlm's gemma4 sanitize is now layout-aware: the workaround in "
        "olmlx/engine/gemma4_sanitize_fix.py is obsolete — remove it (see "
        "this test's comment for the checklist)."
    )
