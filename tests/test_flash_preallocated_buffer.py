"""Tests for PreallocatedNeuronBuffer (Paper §3.3)."""

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.flash.bundler import bundle_ffn_weights
from olmlx.engine.flash.weight_store import (
    FlashWeightStore,
    PreallocatedNeuronBuffer,
)
from tests.test_flash_weight_store import _make_synthetic_mlp_weights


class TestPreallocatedNeuronBuffer:
    def test_insert_and_retrieve(self):
        buf = PreallocatedNeuronBuffer(max_neurons=10, hidden_size=4)
        gate = np.array([1, 2, 3, 4], dtype=np.float16)
        up = np.array([5, 6, 7, 8], dtype=np.float16)
        down = np.array([9, 10, 11, 12], dtype=np.float16)

        buf.insert(0, gate, up, down)

        gate_cols, up_cols, down_rows = buf.get_matrices([0])
        mx.eval(gate_cols, up_cols, down_rows)

        assert gate_cols.shape == (4, 1)
        assert up_cols.shape == (4, 1)
        assert down_rows.shape == (1, 4)
        np.testing.assert_array_almost_equal(np.array(gate_cols[:, 0]), gate, decimal=2)

    def test_multiple_neurons(self):
        buf = PreallocatedNeuronBuffer(max_neurons=10, hidden_size=4)
        for i in range(5):
            buf.insert(
                i,
                np.full(4, i, dtype=np.float16),
                np.full(4, i + 10, dtype=np.float16),
                np.full(4, i + 20, dtype=np.float16),
            )

        gate_cols, up_cols, down_rows = buf.get_matrices([0, 2, 4])
        mx.eval(gate_cols, up_cols, down_rows)

        assert gate_cols.shape == (4, 3)
        assert down_rows.shape == (3, 4)
        # Neuron 0 gate should be all 0s
        np.testing.assert_array_almost_equal(
            np.array(gate_cols[:, 0]), np.zeros(4), decimal=2
        )
        # Neuron 4 gate should be all 4s
        np.testing.assert_array_almost_equal(
            np.array(gate_cols[:, 2]), np.full(4, 4), decimal=2
        )

    def test_eviction_when_full(self):
        buf = PreallocatedNeuronBuffer(max_neurons=3, hidden_size=4)
        for i in range(4):
            buf.insert(
                i,
                np.full(4, i, dtype=np.float16),
                np.full(4, i, dtype=np.float16),
                np.full(4, i, dtype=np.float16),
            )

        # Neuron 0 should be evicted (oldest)
        assert not buf.contains(0)
        assert buf.contains(1)
        assert buf.contains(2)
        assert buf.contains(3)

    def test_eviction_preserves_remaining_data(self):
        buf = PreallocatedNeuronBuffer(max_neurons=3, hidden_size=4)
        for i in range(4):
            buf.insert(
                i,
                np.full(4, float(i), dtype=np.float16),
                np.full(4, float(i), dtype=np.float16),
                np.full(4, float(i), dtype=np.float16),
            )

        # Neurons 1, 2, 3 should still have correct data
        gate_cols, _, _ = buf.get_matrices([1, 2, 3])
        mx.eval(gate_cols)
        np.testing.assert_array_almost_equal(
            np.array(gate_cols[:, 0]), np.full(4, 1), decimal=2
        )
        np.testing.assert_array_almost_equal(
            np.array(gate_cols[:, 2]), np.full(4, 3), decimal=2
        )

    def test_lru_access_updates_order(self):
        """Accessing a neuron should make it least-recently-used last."""
        buf = PreallocatedNeuronBuffer(max_neurons=3, hidden_size=4)
        for i in range(3):
            buf.insert(
                i,
                np.full(4, float(i), dtype=np.float16),
                np.full(4, float(i), dtype=np.float16),
                np.full(4, float(i), dtype=np.float16),
            )

        # Access neuron 0 (making it most recently used)
        buf.get_matrices([0])

        # Insert neuron 3 — should evict neuron 1 (oldest non-accessed)
        buf.insert(
            3,
            np.full(4, 3.0, dtype=np.float16),
            np.full(4, 3.0, dtype=np.float16),
            np.full(4, 3.0, dtype=np.float16),
        )

        assert buf.contains(0)  # accessed recently
        assert not buf.contains(1)  # evicted
        assert buf.contains(2)
        assert buf.contains(3)

    def test_num_used_tracking(self):
        buf = PreallocatedNeuronBuffer(max_neurons=5, hidden_size=4)
        assert buf.num_used == 0
        buf.insert(
            0,
            np.zeros(4, dtype=np.float16),
            np.zeros(4, dtype=np.float16),
            np.zeros(4, dtype=np.float16),
        )
        assert buf.num_used == 1
        buf.insert(
            1,
            np.zeros(4, dtype=np.float16),
            np.zeros(4, dtype=np.float16),
            np.zeros(4, dtype=np.float16),
        )
        assert buf.num_used == 2

    def test_reinsertion_of_existing_neuron(self):
        """Reinserting an already-present neuron should update its data."""
        buf = PreallocatedNeuronBuffer(max_neurons=5, hidden_size=4)
        buf.insert(
            0,
            np.full(4, 1.0, dtype=np.float16),
            np.zeros(4, dtype=np.float16),
            np.zeros(4, dtype=np.float16),
        )

        # Reinsert with new data
        buf.insert(
            0,
            np.full(4, 9.0, dtype=np.float16),
            np.zeros(4, dtype=np.float16),
            np.zeros(4, dtype=np.float16),
        )

        gate_cols, _, _ = buf.get_matrices([0])
        mx.eval(gate_cols)
        np.testing.assert_array_almost_equal(
            np.array(gate_cols[:, 0]), np.full(4, 9.0), decimal=2
        )


class TestFlashWeightStorePreallocated:
    def test_load_neurons_matches_original_store(self, tmp_path):
        """Preallocated buffer should produce same results as NeuronCache."""
        from safetensors.numpy import load_file

        hidden, inter, num_layers = 16, 8, 1
        model_dir = _make_synthetic_mlp_weights(hidden, inter, num_layers, tmp_path)
        output_dir = tmp_path / "flash"
        bundle_ffn_weights(model_dir, output_dir)

        original = load_file(str(model_dir / "model.safetensors"))
        gate_w = original["model.layers.0.mlp.gate_proj.weight"]

        store = FlashWeightStore(
            output_dir, use_preallocated_buffer=True, cache_budget_neurons=32
        )
        indices = [0, 3, 5]
        gate_cols, up_cols, down_rows = store.load_neurons(0, indices)

        expected = mx.array(gate_w[indices].T)
        assert mx.allclose(gate_cols, expected, atol=1e-6)
        store.close()

    def test_requested_cached_neurons_survive_missing_inserts(self, tmp_path):
        """A cached-but-requested neuron must not be evicted by inserting the
        missing batch for the *same* forward. ``get_cached_indices`` is
        membership-only (no LRU refresh), so without touching the requested
        neurons an insert could evict one, then ``get_matrices`` KeyErrors
        mid-forward (#624). Buffer holds [0,1,2]; requesting [0,1,3] inserts
        3 and must evict 2 (unrequested), keeping 0 and 1.
        """
        from safetensors.numpy import load_file

        hidden, inter, num_layers = 16, 8, 1
        model_dir = _make_synthetic_mlp_weights(hidden, inter, num_layers, tmp_path)
        output_dir = tmp_path / "flash"
        bundle_ffn_weights(model_dir, output_dir)

        gate_w = load_file(str(model_dir / "model.safetensors"))[
            "model.layers.0.mlp.gate_proj.weight"
        ]

        store = FlashWeightStore(
            output_dir, use_preallocated_buffer=True, cache_budget_neurons=3
        )
        try:
            store.load_neurons(0, [0, 1, 2])  # fill the buffer
            gate_cols, _, _ = store.load_neurons(0, [0, 1, 3])
            expected = mx.array(gate_w[[0, 1, 3]].T)
            assert mx.allclose(gate_cols, expected, atol=1e-6)
        finally:
            store.close()

    def test_more_active_neurons_than_capacity_raises_clear_error(self, tmp_path):
        """Requesting more distinct neurons than the buffer can hold is
        physically impossible to satisfy; it must raise a clear error naming
        the capacity, not a cryptic KeyError mid-forward (#624)."""
        hidden, inter, num_layers = 16, 8, 1
        model_dir = _make_synthetic_mlp_weights(hidden, inter, num_layers, tmp_path)
        output_dir = tmp_path / "flash"
        bundle_ffn_weights(model_dir, output_dir)

        store = FlashWeightStore(
            output_dir, use_preallocated_buffer=True, cache_budget_neurons=2
        )
        try:
            with pytest.raises(RuntimeError, match="active neurons"):
                store.load_neurons(0, [0, 1, 2])
        finally:
            store.close()

    def test_zero_cache_budget_preallocated_rejected(self, tmp_path):
        """A preallocated buffer with capacity 0 can hold nothing; construction
        must reject it clearly rather than crashing later on
        ``_access_order.popitem`` on an empty dict (#624)."""
        hidden, inter, num_layers = 16, 8, 1
        model_dir = _make_synthetic_mlp_weights(hidden, inter, num_layers, tmp_path)
        output_dir = tmp_path / "flash"
        bundle_ffn_weights(model_dir, output_dir)

        with pytest.raises(ValueError, match="cache_budget_neurons"):
            FlashWeightStore(
                output_dir, use_preallocated_buffer=True, cache_budget_neurons=0
            )

    def test_backward_compat_neuron_cache(self, tmp_path):
        """use_preallocated_buffer=False still uses NeuronCache."""
        hidden, inter, num_layers = 16, 8, 1
        model_dir = _make_synthetic_mlp_weights(hidden, inter, num_layers, tmp_path)
        output_dir = tmp_path / "flash"
        bundle_ffn_weights(model_dir, output_dir)

        store = FlashWeightStore(
            output_dir, use_preallocated_buffer=False, cache_budget_neurons=32
        )
        indices = [0, 1, 2]
        gate_cols, _, _ = store.load_neurons(0, indices)
        assert gate_cols.shape == (hidden, 3)
        store.close()
