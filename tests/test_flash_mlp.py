"""Tests for olmlx.engine.flash.flash_mlp and flash_model."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from olmlx.engine.flash.flash_mlp import FlashMLP, WindowManager
from olmlx.engine.flash.predictor import SparsityPredictor
from olmlx.engine.flash.weight_store import FlashWeightStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bundled_model(tmp_path, hidden=16, inter=8, num_layers=2):
    """Create synthetic safetensors and bundle them for flash inference."""
    from safetensors.numpy import save_file

    from olmlx.engine.flash.bundler import bundle_ffn_weights

    tensors = {}
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}.mlp"
        tensors[f"{prefix}.gate_proj.weight"] = np.random.randn(inter, hidden).astype(
            np.float16
        )
        tensors[f"{prefix}.up_proj.weight"] = np.random.randn(inter, hidden).astype(
            np.float16
        )
        tensors[f"{prefix}.down_proj.weight"] = np.random.randn(hidden, inter).astype(
            np.float16
        )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file(tensors, str(model_dir / "model.safetensors"))

    flash_dir = tmp_path / "flash"
    bundle_ffn_weights(model_dir, flash_dir)
    return flash_dir, model_dir, tensors


class DenseMLP(nn.Module):
    """Standard dense MLP for comparison."""

    def __init__(self, gate_w, up_w, down_w):
        super().__init__()
        # gate_w: (inter, hidden), up_w: (inter, hidden), down_w: (hidden, inter)
        inter, hidden = gate_w.shape
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.gate_proj.weight = mx.array(gate_w)
        self.up_proj.weight = mx.array(up_w)
        self.down_proj.weight = mx.array(down_w)

    def __call__(self, x):
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        # SwiGLU: silu(gate) * up
        act = mx.sigmoid(gate_out) * gate_out * up_out
        return self.down_proj(act)


# ---------------------------------------------------------------------------
# WindowManager tests
# ---------------------------------------------------------------------------


class TestWindowManager:
    def test_empty_window(self):
        wm = WindowManager(num_layers=2, window_size=3)
        w = wm.get_window(0)
        assert len(w) == 0

    def test_update_and_get(self):
        wm = WindowManager(num_layers=2, window_size=3)
        wm.update(0, mx.array([1, 3, 5]))
        w = wm.get_window(0)
        assert w == {1, 3, 5}

    def test_window_union(self):
        wm = WindowManager(num_layers=1, window_size=3)
        wm.update(0, mx.array([1, 2]))
        wm.update(0, mx.array([3, 4]))
        w = wm.get_window(0)
        assert w == {1, 2, 3, 4}

    def test_sliding_window_drops_old(self):
        wm = WindowManager(num_layers=1, window_size=2)
        wm.update(0, mx.array([1, 2]))
        wm.update(0, mx.array([3, 4]))
        wm.update(0, mx.array([5, 6]))
        w = wm.get_window(0)
        # Window size 2 = last 2 updates: [3,4] and [5,6]
        assert 1 not in w and 2 not in w
        assert w == {3, 4, 5, 6}

    def test_reset_clears_all(self):
        wm = WindowManager(num_layers=2, window_size=3)
        wm.update(0, mx.array([1, 2, 3]))
        wm.update(1, mx.array([4, 5, 6]))
        wm.reset()
        assert len(wm.get_window(0)) == 0
        assert len(wm.get_window(1)) == 0

    def test_layers_independent(self):
        wm = WindowManager(num_layers=2, window_size=3)
        wm.update(0, mx.array([1, 2]))
        wm.update(1, mx.array([10, 20]))
        assert wm.get_window(0) == {1, 2}
        assert wm.get_window(1) == {10, 20}


class TestDynamicWindowManager:
    def test_shrinks_when_over_budget(self):
        """Window should shrink when neuron count exceeds budget."""
        wm = WindowManager(
            num_layers=1,
            window_size=10,
            memory_budget_fraction=0.5,
            intermediate_size=10,
        )
        # Budget = 10 * 0.5 = 5 neurons
        # Add tokens with many unique neurons
        wm.update(0, mx.array([0, 1, 2]))
        wm.update(0, mx.array([3, 4, 5]))
        wm.update(0, mx.array([6, 7, 8]))

        # Window should have been trimmed to stay near budget of 5
        w = wm.get_window(0)
        assert len(w) <= 6  # Some tolerance

    def test_disabled_by_default(self):
        """Without memory_budget_fraction, window behaves as fixed."""
        wm = WindowManager(num_layers=1, window_size=2)
        wm.update(0, mx.array([1, 2]))
        wm.update(0, mx.array([3, 4]))
        wm.update(0, mx.array([5, 6]))
        w = wm.get_window(0)
        # Fixed window_size=2: only last 2 updates
        assert w == {3, 4, 5, 6}

    def test_oldest_entries_dropped_first(self):
        """When shrinking, oldest token entries are dropped."""
        wm = WindowManager(
            num_layers=1,
            window_size=10,
            memory_budget_fraction=0.3,
            intermediate_size=10,
        )
        # Budget = 3 neurons
        wm.update(0, mx.array([0, 1]))  # oldest
        wm.update(0, mx.array([2, 3]))  # middle
        wm.update(0, mx.array([4, 5]))  # newest

        w = wm.get_window(0)
        # Newest neurons should be present
        assert 4 in w
        assert 5 in w

    def test_layers_independent_budgets(self):
        """Each layer's window adjusts independently."""
        wm = WindowManager(
            num_layers=2,
            window_size=10,
            memory_budget_fraction=0.5,
            intermediate_size=10,
        )
        # Layer 0: many neurons
        for i in range(5):
            wm.update(0, mx.array([i * 2, i * 2 + 1]))
        # Layer 1: few neurons
        wm.update(1, mx.array([0, 1]))

        wm.get_window(0)  # exercise layer 0
        w1 = wm.get_window(1)
        # Layer 1 should have all its neurons (under budget)
        assert w1 == {0, 1}

    def test_warns_when_single_token_exceeds_budget(self, caplog):
        """Log warning when a single token's neurons exceed the budget."""
        import logging

        wm = WindowManager(
            num_layers=1,
            window_size=10,
            memory_budget_fraction=0.2,
            intermediate_size=10,
        )
        # Budget = 2 neurons, but one token activates 5
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.flash.flash_mlp"):
            wm.update(0, mx.array([0, 1, 2, 3, 4]))
        assert "exceeds budget" in caplog.text


# ---------------------------------------------------------------------------
# FlashMLP tests
# ---------------------------------------------------------------------------


class TestFlashMLP:
    @pytest.fixture()
    def flash_mlp_setup(self, tmp_path):
        """Create a FlashMLP with matching dense MLP for comparison."""
        hidden, inter, num_layers = 16, 8, 1
        flash_dir, model_dir, tensors = _make_bundled_model(
            tmp_path, hidden, inter, num_layers
        )

        store = FlashWeightStore(flash_dir, num_io_threads=2, cache_budget_neurons=64)
        pred = SparsityPredictor(hidden, inter, rank=4)
        wm = WindowManager(num_layers=1, window_size=3)

        gate_w = tensors["model.layers.0.mlp.gate_proj.weight"]
        up_w = tensors["model.layers.0.mlp.up_proj.weight"]
        down_w = tensors["model.layers.0.mlp.down_proj.weight"]

        flash_mlp = FlashMLP(
            layer_idx=0,
            hidden_size=hidden,
            intermediate_size=inter,
            predictor=pred,
            weight_store=store,
            window_manager=wm,
            sparsity_threshold=0.5,
            min_active_neurons=inter,  # all neurons for exact match test
        )
        dense_mlp = DenseMLP(gate_w, up_w, down_w)
        return flash_mlp, dense_mlp, hidden, inter

    def test_output_shape(self, flash_mlp_setup):
        flash_mlp, _, hidden, _ = flash_mlp_setup
        x = mx.random.normal((1, 1, hidden)).astype(mx.float16)
        out = flash_mlp(x)
        mx.eval(out)
        assert out.shape == (1, 1, hidden)

    def test_all_neurons_matches_dense(self, flash_mlp_setup):
        """Loading all neurons should match the dense MLP output."""
        flash_mlp, dense_mlp, hidden, _ = flash_mlp_setup
        x = mx.random.normal((1, 1, hidden)).astype(mx.float16)

        flash_out = flash_mlp(x)
        dense_out = dense_mlp(x)
        mx.eval(flash_out, dense_out)

        assert mx.allclose(flash_out, dense_out, atol=1e-2)

    def test_sparse_output_shape(self, tmp_path):
        """With fewer neurons, output should still have correct shape."""
        hidden, inter, num_layers = 16, 8, 1
        flash_dir, _, _ = _make_bundled_model(tmp_path, hidden, inter, num_layers)

        store = FlashWeightStore(flash_dir, num_io_threads=2, cache_budget_neurons=64)
        pred = SparsityPredictor(hidden, inter, rank=4)
        wm = WindowManager(num_layers=1, window_size=3)

        flash_mlp = FlashMLP(
            layer_idx=0,
            hidden_size=hidden,
            intermediate_size=inter,
            predictor=pred,
            weight_store=store,
            window_manager=wm,
            sparsity_threshold=0.5,
            min_active_neurons=2,  # sparse: only 2 neurons
        )

        x = mx.random.normal((1, 1, hidden)).astype(mx.float16)
        out = flash_mlp(x)
        mx.eval(out)
        assert out.shape == (1, 1, hidden)

    def test_window_updates_after_call(self, flash_mlp_setup):
        flash_mlp, _, hidden, _ = flash_mlp_setup
        x = mx.random.normal((1, 1, hidden)).astype(mx.float16)
        flash_mlp(x)
        w = flash_mlp.window_manager.get_window(0)
        assert len(w) > 0  # Window should have entries after inference


# ---------------------------------------------------------------------------
# FlashModelWrapper.shard() tests
# ---------------------------------------------------------------------------


class _FakeAttention(nn.Module):
    """Minimal attention module with projections for shard tests."""

    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads


class _FakeLayer(nn.Module):
    """Minimal transformer layer with attention + MLP."""

    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.self_attn = _FakeAttention(dim, n_heads, n_kv_heads)
        self.mlp = nn.Linear(dim, dim)  # placeholder, will be replaced


class _FakeModel(nn.Module):
    """Minimal model with layers list for shard tests."""

    def __init__(self, dim, n_heads, n_kv_heads, num_layers=2):
        super().__init__()
        self.layers = [_FakeLayer(dim, n_heads, n_kv_heads) for _ in range(num_layers)]

    def __call__(self, x, cache=None, **kwargs):
        return x


class TestFlashModelWrapperShard:
    @pytest.fixture()
    def wrapper_setup(self, tmp_path):
        """Create a FlashModelWrapper around a fake model."""
        from olmlx.engine.flash.flash_model import FlashConfig, FlashModelWrapper

        dim, n_heads, n_kv_heads, num_layers = 64, 8, 4, 2
        inter = 32
        model = _FakeModel(dim, n_heads, n_kv_heads, num_layers)

        flash_dir, _, _ = _make_bundled_model(
            tmp_path, hidden=dim, inter=inter, num_layers=num_layers
        )
        store = FlashWeightStore(flash_dir, num_io_threads=2, cache_budget_neurons=64)
        from types import SimpleNamespace

        predictor_bank = SimpleNamespace(
            predictors=[
                SparsityPredictor(dim, inter, rank=4) for _ in range(num_layers)
            ],
        )
        config = FlashConfig(
            hidden_size=dim,
            intermediate_size=inter,
            num_layers=num_layers,
        )

        wrapper = FlashModelWrapper(model, predictor_bank, store, config)
        return wrapper, model, dim, n_heads, n_kv_heads, num_layers

    def test_shard_divides_attention_heads(self, wrapper_setup):
        """shard() should divide n_heads and n_kv_heads by world size."""
        from olmlx.engine.pre_shard import FakeGroup

        wrapper, _, _, n_heads, n_kv_heads, num_layers = wrapper_setup
        group = FakeGroup(rank=0, size=2)

        wrapper.shard(group)

        for layer in wrapper.layers:
            assert layer.self_attn.n_heads == n_heads // 2
            assert layer.self_attn.n_kv_heads == n_kv_heads // 2

    def test_shard_leaves_flash_mlp_untouched(self, wrapper_setup):
        """shard() should not modify FlashMLP instances."""
        from olmlx.engine.pre_shard import FakeGroup

        wrapper, model, *_ = wrapper_setup

        # Capture FlashMLP instances before sharding
        mlps_before = [layer.mlp for layer in model.layers]
        assert all(isinstance(m, FlashMLP) for m in mlps_before)

        wrapper.shard(FakeGroup(rank=0, size=2))

        # FlashMLP instances should be the same objects, untouched
        for layer, mlp_before in zip(model.layers, mlps_before):
            assert layer.mlp is mlp_before

    def test_shard_does_not_proxy_to_inner_model(self, wrapper_setup):
        """shard() on wrapper should NOT call inner model's shard()."""
        from unittest.mock import MagicMock

        from olmlx.engine.pre_shard import FakeGroup

        wrapper, model, *_ = wrapper_setup
        model.shard = MagicMock()

        wrapper.shard(FakeGroup(rank=0, size=2))

        model.shard.assert_not_called()

    def test_shard_wraps_attention_projections(self, wrapper_setup):
        """After shard(), attention proj layers should be distributed wrappers."""
        from olmlx.engine.pre_shard import FakeGroup

        wrapper, model, *_ = wrapper_setup

        # Before: plain nn.Linear
        assert isinstance(model.layers[0].self_attn.q_proj, nn.Linear)

        wrapper.shard(FakeGroup(rank=0, size=2))

        # After: should NOT be plain nn.Linear (wrapped by shard_linear)
        attn = model.layers[0].self_attn
        assert type(attn.q_proj) is not nn.Linear
        assert type(attn.k_proj) is not nn.Linear
        assert type(attn.v_proj) is not nn.Linear
        assert type(attn.o_proj) is not nn.Linear

    def test_shard_parameters_materializable(self, wrapper_setup):
        """mx.eval(wrapper.parameters()) should work after sharding."""
        from olmlx.engine.pre_shard import FakeGroup

        wrapper, *_ = wrapper_setup
        wrapper.shard(FakeGroup(rank=0, size=2))

        # Should not raise
        mx.eval(wrapper.parameters())
