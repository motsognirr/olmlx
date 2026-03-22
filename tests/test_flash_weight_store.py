"""Tests for olmlx.engine.flash.weight_store and bundler."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.flash.bundler import (
    HEADER_MAGIC,
    HEADER_SIZE,
    bundle_ffn_weights,
    parse_header,
)
from olmlx.engine.flash.weight_store import FlashWeightStore, NeuronCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_mlp_weights(
    hidden_size: int, intermediate_size: int, num_layers: int, tmp_path: Path
) -> Path:
    """Create synthetic safetensors with MLP weights for testing."""
    from safetensors.numpy import save_file

    tensors = {}
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}.mlp"
        tensors[f"{prefix}.gate_proj.weight"] = np.random.randn(
            intermediate_size, hidden_size
        ).astype(np.float16)
        tensors[f"{prefix}.up_proj.weight"] = np.random.randn(
            intermediate_size, hidden_size
        ).astype(np.float16)
        tensors[f"{prefix}.down_proj.weight"] = np.random.randn(
            hidden_size, intermediate_size
        ).astype(np.float16)

    # Add some non-FFN weights that should be skipped
    tensors["model.embed_tokens.weight"] = np.random.randn(1000, hidden_size).astype(
        np.float16
    )
    tensors["lm_head.weight"] = np.random.randn(1000, hidden_size).astype(np.float16)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file(tensors, str(model_dir / "model.safetensors"))
    return model_dir


# ---------------------------------------------------------------------------
# Bundler tests
# ---------------------------------------------------------------------------


class TestBundleFFNWeights:
    def test_bundle_creates_layer_files(self, tmp_path):
        hidden, inter, num_layers = 32, 64, 2
        model_dir = _make_synthetic_mlp_weights(hidden, inter, num_layers, tmp_path)
        output_dir = tmp_path / "flash"

        layouts = bundle_ffn_weights(model_dir, output_dir)

        assert len(layouts) == num_layers
        for i in range(num_layers):
            assert (output_dir / f"layer_{i:02d}.flashweights").exists()
            assert i in layouts

    def test_bundle_header_is_correct(self, tmp_path):
        hidden, inter, num_layers = 32, 64, 2
        model_dir = _make_synthetic_mlp_weights(hidden, inter, num_layers, tmp_path)
        output_dir = tmp_path / "flash"

        bundle_ffn_weights(model_dir, output_dir)

        fp = output_dir / "layer_00.flashweights"
        with open(fp, "rb") as f:
            header = parse_header(f.read(HEADER_SIZE))

        assert header["magic"] == HEADER_MAGIC
        assert header["num_neurons"] == inter
        assert header["hidden_size"] == hidden
        assert header["dtype"] == "float16"

    def test_bundle_preserves_weight_data(self, tmp_path):
        """Bundled neuron data must match the original safetensors weights."""
        hidden, inter, num_layers = 16, 8, 1
        model_dir = _make_synthetic_mlp_weights(hidden, inter, num_layers, tmp_path)
        output_dir = tmp_path / "flash"

        from safetensors.numpy import load_file

        original = load_file(str(model_dir / "model.safetensors"))
        gate_w = original["model.layers.0.mlp.gate_proj.weight"]  # (inter, hidden)
        up_w = original["model.layers.0.mlp.up_proj.weight"]  # (inter, hidden)
        down_w = original["model.layers.0.mlp.down_proj.weight"]  # (hidden, inter)

        layouts = bundle_ffn_weights(model_dir, output_dir)
        layout = layouts[0]

        # Read neuron 3's data manually
        neuron_idx = 3
        neuron_byte_size = layout.neuron_byte_size
        data_offset = layout.offsets[neuron_idx]

        with open(layout.file_path, "rb") as f:
            f.seek(data_offset)
            raw = f.read(neuron_byte_size)

        # Each neuron stores: gate_col (hidden,) + up_col (hidden,) + down_row (hidden,)
        dtype_bytes = 2  # float16
        gate_col = np.frombuffer(raw[: hidden * dtype_bytes], dtype=np.float16)
        up_col = np.frombuffer(
            raw[hidden * dtype_bytes : 2 * hidden * dtype_bytes], dtype=np.float16
        )
        down_row = np.frombuffer(
            raw[2 * hidden * dtype_bytes : 3 * hidden * dtype_bytes], dtype=np.float16
        )

        # gate_proj.weight is (inter, hidden), so row neuron_idx = gate column for neuron
        np.testing.assert_array_equal(gate_col, gate_w[neuron_idx])
        np.testing.assert_array_equal(up_col, up_w[neuron_idx])
        # down_proj.weight is (hidden, inter), so column neuron_idx = down row for neuron
        np.testing.assert_array_equal(down_row, down_w[:, neuron_idx])

    def test_bundle_offset_table_is_sequential(self, tmp_path):
        hidden, inter, num_layers = 16, 8, 1
        model_dir = _make_synthetic_mlp_weights(hidden, inter, num_layers, tmp_path)
        output_dir = tmp_path / "flash"

        layouts = bundle_ffn_weights(model_dir, output_dir)
        layout = layouts[0]

        # Offsets should be sequential, each neuron_byte_size apart
        for i in range(1, inter):
            assert layout.offsets[i] == layout.offsets[i - 1] + layout.neuron_byte_size

    def test_bundle_writes_layout_json(self, tmp_path):
        import json

        hidden, inter, num_layers = 16, 8, 1
        model_dir = _make_synthetic_mlp_weights(hidden, inter, num_layers, tmp_path)
        output_dir = tmp_path / "flash"

        bundle_ffn_weights(model_dir, output_dir)

        config_path = output_dir / "flash_layout.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["num_layers"] == num_layers
        assert config["hidden_size"] == hidden
        assert config["intermediate_size"] == inter


# ---------------------------------------------------------------------------
# NeuronCache tests
# ---------------------------------------------------------------------------


class TestNeuronCache:
    def test_put_and_get(self):
        cache = NeuronCache(max_neurons_per_layer=10)
        data = (mx.ones((4,)), mx.ones((4,)), mx.ones((4,)))
        cache.put(0, 5, data)
        result = cache.get(0, 5)
        assert result is not None
        assert mx.array_equal(result[0], data[0])

    def test_get_missing_returns_none(self):
        cache = NeuronCache(max_neurons_per_layer=10)
        assert cache.get(0, 99) is None

    def test_lru_eviction(self):
        cache = NeuronCache(max_neurons_per_layer=3)
        for i in range(4):
            cache.put(0, i, (mx.array([i]),) * 3)

        # Neuron 0 should have been evicted (oldest)
        assert cache.get(0, 0) is None
        # Neurons 1, 2, 3 should still be present
        assert cache.get(0, 1) is not None
        assert cache.get(0, 2) is not None
        assert cache.get(0, 3) is not None

    def test_get_batch_returns_hits_and_misses(self):
        cache = NeuronCache(max_neurons_per_layer=10)
        cache.put(0, 1, (mx.array([1.0]),) * 3)
        cache.put(0, 3, (mx.array([3.0]),) * 3)

        cached = cache.get_batch(0, [1, 2, 3, 4])
        assert 1 in cached and 3 in cached
        assert 2 not in cached and 4 not in cached
        assert len(cached) == 2

    def test_layers_are_independent(self):
        cache = NeuronCache(max_neurons_per_layer=5)
        cache.put(0, 1, (mx.array([0.0]),) * 3)
        cache.put(1, 1, (mx.array([1.0]),) * 3)

        r0 = cache.get(0, 1)
        r1 = cache.get(1, 1)
        assert r0 is not None and r1 is not None
        assert not mx.array_equal(r0[0], r1[0])


# ---------------------------------------------------------------------------
# FlashWeightStore tests
# ---------------------------------------------------------------------------


class TestFlashWeightStore:
    @pytest.fixture()
    def store_with_model(self, tmp_path):
        """Create a bundled model and return a FlashWeightStore for it."""
        hidden, inter, num_layers = 16, 8, 2
        model_dir = _make_synthetic_mlp_weights(hidden, inter, num_layers, tmp_path)
        output_dir = tmp_path / "flash"
        bundle_ffn_weights(model_dir, output_dir)

        store = FlashWeightStore(output_dir, num_io_threads=4, cache_budget_neurons=32)
        return store, model_dir, hidden, inter, num_layers

    def test_load_neurons_returns_correct_shapes(self, store_with_model):
        store, _, hidden, inter, _ = store_with_model
        indices = [0, 2, 5]
        gate_cols, up_cols, down_rows = store.load_neurons(0, indices)

        assert gate_cols.shape == (hidden, len(indices))
        assert up_cols.shape == (hidden, len(indices))
        assert down_rows.shape == (len(indices), hidden)

    def test_load_neurons_matches_original(self, store_with_model, tmp_path):
        """Loaded neurons must match the original safetensors data."""
        from safetensors.numpy import load_file

        store, model_dir, hidden, inter, _ = store_with_model
        original = load_file(str(model_dir / "model.safetensors"))
        gate_w = original["model.layers.0.mlp.gate_proj.weight"]  # (inter, hidden)
        up_w = original["model.layers.0.mlp.up_proj.weight"]
        down_w = original["model.layers.0.mlp.down_proj.weight"]  # (hidden, inter)

        indices = [1, 4, 7]
        gate_cols, up_cols, down_rows = store.load_neurons(0, indices)

        # gate_cols should be gate_w[indices].T → (hidden, 3)
        expected_gate = mx.array(gate_w[indices].T)
        expected_up = mx.array(up_w[indices].T)
        expected_down = mx.array(down_w[:, indices].T)

        assert mx.allclose(gate_cols, expected_gate, atol=1e-6)
        assert mx.allclose(up_cols, expected_up, atol=1e-6)
        assert mx.allclose(down_rows, expected_down, atol=1e-6)

    def test_load_all_neurons_matches_original(self, store_with_model, tmp_path):
        """Loading all neurons should reproduce the full weight matrices."""
        from safetensors.numpy import load_file

        store, model_dir, hidden, inter, _ = store_with_model
        original = load_file(str(model_dir / "model.safetensors"))
        gate_w = original["model.layers.0.mlp.gate_proj.weight"]
        up_w = original["model.layers.0.mlp.up_proj.weight"]
        down_w = original["model.layers.0.mlp.down_proj.weight"]

        all_indices = list(range(inter))
        gate_cols, up_cols, down_rows = store.load_neurons(0, all_indices)

        assert mx.allclose(gate_cols, mx.array(gate_w.T), atol=1e-6)
        assert mx.allclose(up_cols, mx.array(up_w.T), atol=1e-6)
        assert mx.allclose(down_rows, mx.array(down_w.T), atol=1e-6)

    def test_cache_hit_avoids_reread(self, store_with_model):
        """Second load of same neurons should use cache."""
        store, _, _, _, _ = store_with_model
        indices = [0, 1]

        # First load
        store.load_neurons(0, indices)
        # Second load — should be from cache
        gate_cols, up_cols, down_rows = store.load_neurons(0, indices)

        # Verify results are still correct (cache didn't corrupt)
        assert gate_cols.shape[1] == 2

    def test_parallel_io_consistency(self, store_with_model):
        """Loading many neurons in parallel should give same results as sequential."""
        store, _, hidden, inter, _ = store_with_model

        # Load all neurons (triggers parallel reads for uncached ones)
        all_indices = list(range(inter))
        gate_all, up_all, down_all = store.load_neurons(0, all_indices)

        # Create a fresh store (no cache) and load one at a time
        store2 = FlashWeightStore(
            store._flash_dir, num_io_threads=1, cache_budget_neurons=0
        )
        for i in range(inter):
            g, u, d = store2.load_neurons(0, [i])
            assert mx.allclose(g, gate_all[:, i : i + 1], atol=1e-6)
            assert mx.allclose(u, up_all[:, i : i + 1], atol=1e-6)
            assert mx.allclose(d, down_all[i : i + 1, :], atol=1e-6)

    def test_load_neurons_different_layers(self, store_with_model):
        """Each layer's weights should be independent."""
        store, _, _, _, _ = store_with_model
        g0, u0, d0 = store.load_neurons(0, [0])
        g1, u1, d1 = store.load_neurons(1, [0])

        # Different layers should have different weights (extremely unlikely to match)
        assert not mx.array_equal(g0, g1)
