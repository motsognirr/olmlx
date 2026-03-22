"""Tests for pipeline parallelism module."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class TestLayerAssignment:
    """Test layer index calculation for pipeline parallelism."""

    def test_even_split_2_ranks(self):
        from olmlx.engine.pipeline import _compute_layer_range

        # 64 layers, 2 ranks, even split [32, 32]
        # rank 0 = last layers, rank 1 = first layers (DeepSeek convention)
        start, end = _compute_layer_range(rank=0, layer_counts=[32, 32])
        assert start == 32
        assert end == 64

        start, end = _compute_layer_range(rank=1, layer_counts=[32, 32])
        assert start == 0
        assert end == 32

    def test_uneven_split_2_ranks(self):
        from olmlx.engine.pipeline import _compute_layer_range

        # 64 layers, rank 0 gets 44, rank 1 gets 20
        start, end = _compute_layer_range(rank=0, layer_counts=[44, 20])
        assert start == 20
        assert end == 64

        start, end = _compute_layer_range(rank=1, layer_counts=[44, 20])
        assert start == 0
        assert end == 20

    def test_even_split_3_ranks(self):
        from olmlx.engine.pipeline import _compute_layer_range

        # 63 layers, 3 ranks, [21, 21, 21]
        start, end = _compute_layer_range(rank=0, layer_counts=[21, 21, 21])
        assert start == 42
        assert end == 63

        start, end = _compute_layer_range(rank=1, layer_counts=[21, 21, 21])
        assert start == 21
        assert end == 42

        start, end = _compute_layer_range(rank=2, layer_counts=[21, 21, 21])
        assert start == 0
        assert end == 21

    def test_uneven_split_3_ranks(self):
        from olmlx.engine.pipeline import _compute_layer_range

        # 64 layers: [30, 20, 14]
        start, end = _compute_layer_range(rank=0, layer_counts=[30, 20, 14])
        assert start == 34  # 20 + 14
        assert end == 64

        start, end = _compute_layer_range(rank=1, layer_counts=[30, 20, 14])
        assert start == 14
        assert end == 34

        start, end = _compute_layer_range(rank=2, layer_counts=[30, 20, 14])
        assert start == 0
        assert end == 14

    def test_default_even_split(self):
        from olmlx.engine.pipeline import _compute_layer_counts

        # 64 layers, 2 ranks -> [32, 32]
        counts = _compute_layer_counts(64, 2)
        assert counts == [32, 32]

    def test_default_uneven_total(self):
        from olmlx.engine.pipeline import _compute_layer_counts

        # 65 layers, 2 ranks -> [33, 32] (extra goes to rank 0)
        counts = _compute_layer_counts(65, 2)
        assert sum(counts) == 65
        assert len(counts) == 2


class TestLayerCountValidation:
    """Test validation of layer_counts parameter."""

    def test_layer_counts_wrong_sum(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=64)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        with pytest.raises(ValueError, match="must sum to"):
            apply_pipeline(model, group, layer_counts=[30, 20])

    def test_layer_counts_wrong_length(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=64)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        with pytest.raises(ValueError, match="must have.*entries"):
            apply_pipeline(model, group, layer_counts=[32, 16, 16])


class TestDoubleApply:
    """Test that apply_pipeline rejects double application."""

    def test_double_apply_raises(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=8)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        inner2 = _make_mock_inner_model(num_layers=8)
        model2 = _make_mock_outer_model(inner2)
        model2.model = inner  # reuse already-patched inner

        with pytest.raises(RuntimeError, match="already applied"):
            apply_pipeline(model2, group, layer_counts=[4, 4])


class TestUnsupportedModel:
    """Test error for models without standard structure."""

    def test_no_inner_model(self):
        from olmlx.engine.pipeline import apply_pipeline

        model = SimpleNamespace()  # no .model attribute
        group = _make_mock_group(rank=0, size=2)

        with pytest.raises(ValueError, match="does not have a standard"):
            apply_pipeline(model, group)

    def test_inner_missing_layers(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = SimpleNamespace(embed_tokens=None, norm=None)  # no .layers
        model = SimpleNamespace(model=inner)
        group = _make_mock_group(rank=0, size=2)

        with pytest.raises(ValueError, match="does not have a standard"):
            apply_pipeline(model, group)


class TestMonkeyPatch:
    """Test that apply_pipeline correctly patches the model."""

    def test_pipeline_state_set(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=8)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        assert inner.pipeline_rank == 0
        assert inner.pipeline_size == 2
        assert inner.start_idx == 4
        assert inner.end_idx == 8
        assert inner.num_layers == 4

    def test_non_owned_layers_nullified(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=8)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        # Rank 0 owns layers 4-7, layers 0-3 should be None
        assert all(layer is None for layer in inner.layers[:4])
        assert all(layer is not None for layer in inner.layers[4:8])
        # Truncated to end_idx
        assert len(inner.layers) == 8

    def test_rank1_layers(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=8)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=1, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        # Rank 1 (highest) owns first layers 0-3
        assert inner.start_idx == 0
        assert inner.end_idx == 4
        assert len(inner.layers) == 4
        assert all(layer is not None for layer in inner.layers)

    def test_outer_layers_property_patched(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=8)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        # Outer model.layers should return only owned layers
        owned = model.layers
        assert len(owned) == 4
        assert all(layer is not None for layer in owned)

    def test_inner_call_replaced(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=8)
        original_call = inner.__call__
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        # __call__ should be replaced
        assert inner.__call__ != original_call


class TestPreShardedApplyPipeline:
    """Test apply_pipeline with pre_sharded=True (reduced model)."""

    def test_sets_start_idx_zero(self):
        from olmlx.engine.pipeline import apply_pipeline

        # Simulate a pre-sharded rank 0 model with 4 layers (from original 8)
        inner = _make_mock_inner_model(num_layers=4)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4], pre_sharded=True)

        assert inner.start_idx == 0
        assert inner.end_idx == 4
        assert inner.num_layers == 4
        assert inner.pipeline_rank == 0
        assert inner.pipeline_size == 2

    def test_skips_nullification(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=4)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4], pre_sharded=True)

        # All layers should remain (no nullification)
        assert all(layer is not None for layer in inner.layers)
        assert len(inner.layers) == 4

    def test_patches_call(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=4)
        original_call = inner.__call__
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4], pre_sharded=True)

        assert inner.__call__ != original_call

    def test_outer_layers_patched(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=4)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4], pre_sharded=True)

        assert len(model.layers) == 4

    def test_gpt_oss_pre_sharded(self):
        from olmlx.engine.pipeline import apply_pipeline

        # Pre-sharded gpt_oss rank 1 with 4 layers (from original 8)
        inner = _make_mock_gpt_oss_inner(num_layers=4)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=1, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4], pre_sharded=True)

        assert inner.start_idx == 0
        assert inner.end_idx == 4
        assert hasattr(inner, "_owned_layer_types")
        assert len(inner._owned_layer_types) == 4

    def test_layer_count_mismatch_raises(self):
        from olmlx.engine.pipeline import apply_pipeline

        # Model has 3 layers but layer_counts says rank 0 should have 4
        inner = _make_mock_inner_model(num_layers=3)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        with pytest.raises(ValueError, match="shard may be stale"):
            apply_pipeline(model, group, layer_counts=[4, 4], pre_sharded=True)

    def test_requires_layer_counts(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_inner_model(num_layers=4)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        with pytest.raises(ValueError, match="layer_counts is required"):
            apply_pipeline(model, group, pre_sharded=True)


class TestHostfileParsing:
    """Test backward-compatible hostfile parsing."""

    def test_no_strategy_defaults_to_tensor(self):
        hostfile = {
            "hosts": ["10.0.1.1", "10.0.1.2"],
            "model": "mlx-community/Qwen3-8B-4bit",
        }
        strategy = hostfile.get("strategy", "tensor")
        layers = hostfile.get("layers")
        assert strategy == "tensor"
        assert layers is None

    def test_pipeline_strategy_with_layers(self):
        hostfile = {
            "hosts": ["10.0.1.1", "10.0.1.2"],
            "model": "mlx-community/Qwen3-32B-4bit",
            "strategy": "pipeline",
            "layers": [44, 20],
        }
        strategy = hostfile.get("strategy", "tensor")
        layers = hostfile.get("layers")
        assert strategy == "pipeline"
        assert layers == [44, 20]


class TestGptOssDetection:
    """Test that gpt_oss models are detected."""

    def test_detects_gpt_oss(self):
        from olmlx.engine.pipeline import _is_gpt_oss

        inner = _make_mock_inner_model(num_layers=4)
        inner.layer_types = ["sliding_attention", "full_attention"] * 2
        inner.window_size = 128
        inner.ga_idx = 1
        inner.swa_idx = 0
        assert _is_gpt_oss(inner) is True

    def test_not_gpt_oss_missing_layer_types(self):
        from olmlx.engine.pipeline import _is_gpt_oss

        inner = _make_mock_inner_model(num_layers=4)
        assert _is_gpt_oss(inner) is False

    def test_gpt_oss_not_detected_as_llama(self):
        from olmlx.engine.pipeline import _is_llama_sliding_window

        inner = _make_mock_inner_model(num_layers=4)
        inner.layer_types = ["sliding_attention", "full_attention"] * 2
        inner.window_size = 128
        inner.ga_idx = 1
        inner.swa_idx = 0
        # gpt_oss has window_size, not sliding_window — should NOT match Llama
        assert _is_llama_sliding_window(inner) is False


class TestGptOssMonkeyPatch:
    """Test pipeline patching for gpt_oss models."""

    def test_gpt_oss_pipeline_state_set(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_gpt_oss_inner(num_layers=8)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        assert inner.pipeline_rank == 0
        assert inner.pipeline_size == 2
        assert inner.start_idx == 4
        assert inner.end_idx == 8
        assert inner.num_layers == 4

    def test_gpt_oss_layer_types_preserved(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_gpt_oss_inner(num_layers=8)
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        # _owned_layer_types should be set to the owned range
        assert hasattr(inner, "_owned_layer_types")
        assert len(inner._owned_layer_types) == 4

    def test_gpt_oss_call_replaced(self):
        from olmlx.engine.pipeline import apply_pipeline

        inner = _make_mock_gpt_oss_inner(num_layers=8)
        original_call = inner.__call__
        model = _make_mock_outer_model(inner)
        group = _make_mock_group(rank=0, size=2)

        apply_pipeline(model, group, layer_counts=[4, 4])

        assert inner.__call__ != original_call


class TestLlamaDetection:
    """Test that Llama sliding window models are detected."""

    def test_detects_sliding_window(self):
        from olmlx.engine.pipeline import _is_llama_sliding_window

        inner = _make_mock_inner_model(num_layers=4)
        inner.sliding_window = 4096
        inner.swa_idx = 0
        assert _is_llama_sliding_window(inner) is True

    def test_no_sliding_window(self):
        from olmlx.engine.pipeline import _is_llama_sliding_window

        inner = _make_mock_inner_model(num_layers=4)
        assert _is_llama_sliding_window(inner) is False

    def test_sliding_window_none(self):
        from olmlx.engine.pipeline import _is_llama_sliding_window

        inner = _make_mock_inner_model(num_layers=4)
        inner.sliding_window = None
        assert _is_llama_sliding_window(inner) is False


# -- Helpers --


def _make_mock_group(rank: int, size: int):
    group = MagicMock()
    group.rank.return_value = rank
    group.size.return_value = size
    return group


class _MockLayer:
    """Minimal mock for a transformer layer."""

    def __init__(self, idx):
        self.idx = idx
        self.use_sliding = False

    def __call__(self, h, mask, cache=None):
        return h


def _make_mock_inner_model(num_layers: int):
    inner = SimpleNamespace(
        embed_tokens=MagicMock(),
        layers=list(_MockLayer(i) for i in range(num_layers)),
        norm=MagicMock(),
    )
    # Add a callable __call__ so we can check it gets replaced
    inner.__call__ = lambda self, *a, **kw: None
    return inner


def _make_mock_gpt_oss_inner(num_layers: int):
    layer_types = ["sliding_attention", "full_attention"] * (num_layers // 2)
    if num_layers % 2:
        layer_types.append("sliding_attention")
    inner = SimpleNamespace(
        embed_tokens=MagicMock(),
        layers=list(_MockLayer(i) for i in range(num_layers)),
        norm=MagicMock(),
        layer_types=layer_types,
        window_size=128,
        ga_idx=layer_types.index("full_attention"),
        swa_idx=layer_types.index("sliding_attention"),
    )
    inner.__call__ = lambda self, *a, **kw: None
    return inner


def _make_mock_outer_model(inner):
    model = SimpleNamespace(model=inner)
    # Outer model exposes layers as property-like access
    model.layers = inner.layers
    return model
