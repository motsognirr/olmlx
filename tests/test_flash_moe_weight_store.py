"""Tests for olmlx.engine.flash.moe_weight_store — runtime expert loading."""

import mlx.core as mx
import numpy as np
import pytest

from tests.test_flash_moe_bundler import (
    _make_synthetic_moe_weights,
    _make_synthetic_nemotron_moe_weights,
)


# ---------------------------------------------------------------------------
# FlashMoeWeightStore tests
# ---------------------------------------------------------------------------


class TestFlashMoeWeightStore:
    @pytest.fixture()
    def store_with_model(self, tmp_path):
        """Create bundled MoE model and return FlashMoeWeightStore."""
        hidden, inter, experts = 64, 32, 8
        num_moe, num_dense = 2, 1
        model_dir = _make_synthetic_moe_weights(
            hidden, inter, experts, num_moe, num_dense, tmp_path
        )
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        bundle_moe_experts(model_dir, output_dir)

        from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

        store = FlashMoeWeightStore(
            output_dir, num_io_threads=4, cache_budget_experts=16
        )
        return store, model_dir, hidden, inter, experts

    def test_load_experts_correct_shapes(self, store_with_model):
        """Loaded expert weights should have correct dimensions."""
        store, _, hidden, inter, _ = store_with_model
        expert_indices = [0, 2, 5]

        loaded = store.load_experts(1, expert_indices)  # layer 1 is first MoE layer

        assert loaded.gate_weight.shape == (len(expert_indices), inter, hidden)
        assert loaded.up_weight.shape == (len(expert_indices), inter, hidden)
        assert loaded.down_weight.shape == (len(expert_indices), hidden, inter)

    def test_load_experts_matches_original(self, store_with_model):
        """Loaded expert weights must match original safetensors data."""
        from safetensors.numpy import load_file

        store, model_dir, hidden, inter, _ = store_with_model
        original = load_file(str(model_dir / "model.safetensors"))

        # Layer 1 is first MoE layer
        gate_w = original["model.layers.1.mlp.switch_mlp.gate_proj.weight"]
        up_w = original["model.layers.1.mlp.switch_mlp.up_proj.weight"]
        down_w = original["model.layers.1.mlp.switch_mlp.down_proj.weight"]

        expert_indices = [1, 4, 7]
        loaded = store.load_experts(1, expert_indices)

        for i, eidx in enumerate(expert_indices):
            assert mx.allclose(
                loaded.gate_weight[i],
                mx.array(gate_w[eidx]),
                atol=1e-6,
            )
            assert mx.allclose(
                loaded.up_weight[i],
                mx.array(up_w[eidx]),
                atol=1e-6,
            )
            assert mx.allclose(
                loaded.down_weight[i],
                mx.array(down_w[eidx]),
                atol=1e-6,
            )

    def test_cache_hit_avoids_reread(self, store_with_model):
        """Second load of same experts should use cache."""
        store, _, _, _, _ = store_with_model
        expert_indices = [0, 1]

        # First load
        loaded1 = store.load_experts(1, expert_indices)
        # Second load — should be from cache
        loaded2 = store.load_experts(1, expert_indices)

        assert mx.array_equal(loaded1.gate_weight, loaded2.gate_weight)

    def test_parallel_io_consistency(self, store_with_model):
        """Loading many experts in parallel should give same results as sequential."""
        from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

        store, _, hidden, inter, experts = store_with_model

        # Load all experts (triggers parallel reads)
        all_indices = list(range(experts))
        loaded_all = store.load_experts(1, all_indices)

        # Create a fresh store (no cache) and load one at a time
        store2 = FlashMoeWeightStore(
            store._flash_dir, num_io_threads=1, cache_budget_experts=0
        )
        for i in range(experts):
            loaded_one = store2.load_experts(1, [i])
            assert mx.allclose(
                loaded_one.gate_weight[0], loaded_all.gate_weight[i], atol=1e-6
            )

    def test_load_experts_different_layers(self, store_with_model):
        """Each layer's expert weights should be independent."""
        store, _, _, _, _ = store_with_model
        loaded_1 = store.load_experts(1, [0])  # MoE layer 1
        loaded_2 = store.load_experts(2, [0])  # MoE layer 2

        assert not mx.array_equal(loaded_1.gate_weight, loaded_2.gate_weight)

    def test_load_experts_empty_list_raises(self, store_with_model):
        """load_experts with empty expert_indices should raise ValueError, not mx.stack crash."""
        store, _, _, _, _ = store_with_model
        with pytest.raises(ValueError, match="expert_indices must not be empty"):
            store.load_experts(1, [])

    def test_expert_index_map(self, store_with_model):
        """LoadedExperts should provide a mapping from global to local indices."""
        store, _, _, _, _ = store_with_model
        expert_indices = [3, 7, 1]
        loaded = store.load_experts(1, expert_indices)

        assert loaded.expert_index_map == {3: 0, 7: 1, 1: 2}

    def test_load_experts_provides_remap_lut(self, store_with_model):
        """LoadedExperts.remap_lut maps global expert indices to local stack positions."""
        store, _, _, _, num_experts = store_with_model
        expert_indices = [5, 2, 7]
        loaded = store.load_experts(1, expert_indices)

        assert loaded.remap_lut is not None
        assert loaded.remap_lut.shape == (num_experts,)
        assert loaded.remap_lut.dtype == mx.uint32

        lut = loaded.remap_lut.tolist()
        assert lut[5] == 0  # first requested global maps to stack pos 0
        assert lut[2] == 1
        assert lut[7] == 2
        # Unrequested entries carry the sentinel
        requested = set(expert_indices)
        for i in range(num_experts):
            if i not in requested:
                assert lut[i] == 0xFFFFFFFF

    def test_load_experts_empty_indices_raises(self, store_with_model):
        """load_experts with empty indices should raise ValueError, not mx.stack crash."""
        store, _, _, _, _ = store_with_model
        with pytest.raises(ValueError, match="expert_indices"):
            store.load_experts(1, [])

    def test_load_experts_tolerates_out_of_order_completion(
        self, store_with_model, monkeypatch
    ):
        """load_experts must return correct stacked weights even if futures complete
        in a different order than submission."""
        import time

        store, _, _, _, _ = store_with_model

        # Baseline with warm cache (deterministic order).
        baseline = store.load_experts(1, [5, 2, 7, 1])
        mx.eval(baseline.up_weight, baseline.down_weight)

        # Build a fresh cold store so the next call actually issues I/O.
        from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

        cold = FlashMoeWeightStore(
            store._flash_dir, num_io_threads=4, cache_budget_experts=0
        )
        try:
            original = cold._read_expert
            delay_target = 5

            def delayed(layer_idx, expert_idx):
                if expert_idx == delay_target:
                    time.sleep(0.05)
                return original(layer_idx, expert_idx)

            monkeypatch.setattr(cold, "_read_expert", delayed)
            reordered = cold.load_experts(1, [5, 2, 7, 1])
            mx.eval(reordered.up_weight, reordered.down_weight)

            # Stacked tensors must be identical in input order regardless of completion order.
            assert mx.allclose(reordered.up_weight, baseline.up_weight, atol=0, rtol=0)
            assert mx.allclose(
                reordered.down_weight, baseline.down_weight, atol=0, rtol=0
            )
            assert baseline.expert_index_map == reordered.expert_index_map
        finally:
            cold.close()


class TestFlashMoeWeightStoreQuantized:
    @pytest.fixture()
    def quant_store(self, tmp_path):
        """Create bundled quantized MoE model and return store."""
        hidden, inter, experts = 64, 32, 4
        model_dir = _make_synthetic_moe_weights(
            hidden, inter, experts, 1, 0, tmp_path, quantized=True
        )
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        bundle_moe_experts(model_dir, output_dir)

        from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

        store = FlashMoeWeightStore(
            output_dir, num_io_threads=4, cache_budget_experts=16
        )
        return store, model_dir, hidden, inter, experts

    def test_load_quantized_experts(self, quant_store):
        """Quantized experts should load with scales and biases."""
        store, _, hidden, inter, _ = quant_store
        loaded = store.load_experts(0, [0, 2])

        assert loaded.is_quantized
        assert loaded.gate_scales is not None
        assert loaded.gate_biases is not None
        assert loaded.up_scales is not None
        assert loaded.down_scales is not None

    def test_quantized_data_matches_original(self, quant_store):
        """Quantized packed weights should match original safetensors."""
        from safetensors.numpy import load_file

        store, model_dir, _, _, _ = quant_store
        original = load_file(str(model_dir / "model.safetensors"))

        gate_w_orig = original["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
        gate_s_orig = original["model.layers.0.mlp.switch_mlp.gate_proj.scales"]

        loaded = store.load_experts(0, [1])

        # Packed weight should match
        np.testing.assert_array_equal(
            np.array(loaded.gate_weight[0]),
            gate_w_orig[1],
        )
        np.testing.assert_array_equal(
            np.array(loaded.gate_scales[0]),
            gate_s_orig[1],
        )


class TestFlashMoeWeightStoreNemotron:
    """Test weight store with Nemotron-style fc1/fc2 projections."""

    @pytest.fixture()
    def nemotron_store(self, tmp_path):
        """Create bundled Nemotron MoE model and return store."""
        hidden, inter, experts = 64, 32, 8
        pattern = "ME"  # layer 1 is MoE
        model_dir = _make_synthetic_nemotron_moe_weights(
            hidden, inter, experts, 2, pattern, tmp_path
        )
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        bundle_moe_experts(model_dir, output_dir)

        from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

        store = FlashMoeWeightStore(
            output_dir, num_io_threads=4, cache_budget_experts=16
        )
        return store, model_dir, hidden, inter, experts

    def test_load_fc1_fc2_experts(self, nemotron_store):
        """fc1/fc2 experts should load into up/down fields (no gate)."""
        store, _, hidden, inter, _ = nemotron_store
        loaded = store.load_experts(1, [0, 2, 5])

        # fc1 maps to up_weight, fc2 maps to down_weight
        assert loaded.up_weight.shape == (3, inter, hidden)
        assert loaded.down_weight.shape == (3, hidden, inter)
        # No gate projection for fc1/fc2 style
        assert loaded.gate_weight is None

    def test_fc1_fc2_data_matches_original(self, nemotron_store):
        """Loaded fc1/fc2 weights must match original safetensors."""
        from safetensors.numpy import load_file

        store, model_dir, _, _, _ = nemotron_store
        original = load_file(str(model_dir / "model.safetensors"))

        fc1_w = original["backbone.layers.1.mixer.switch_mlp.fc1.weight"]
        fc2_w = original["backbone.layers.1.mixer.switch_mlp.fc2.weight"]

        loaded = store.load_experts(1, [3])

        np.testing.assert_array_equal(np.array(loaded.up_weight[0]), fc1_w[3])
        np.testing.assert_array_equal(np.array(loaded.down_weight[0]), fc2_w[3])
