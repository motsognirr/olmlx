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
