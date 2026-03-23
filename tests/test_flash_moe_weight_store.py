"""Tests for olmlx.engine.flash.moe_weight_store — runtime expert loading."""

import mlx.core as mx
import numpy as np
import pytest

from tests.test_flash_moe_bundler import _make_synthetic_moe_weights


# ---------------------------------------------------------------------------
# ExpertCache tests
# ---------------------------------------------------------------------------


class TestExpertCache:
    def test_put_and_get(self):
        from olmlx.engine.flash.moe_weight_store import ExpertCache

        cache = ExpertCache(max_experts_per_layer=10)
        data = {"gate_proj": mx.ones((4, 8)), "up_proj": mx.ones((4, 8))}
        cache.put(0, 5, data)
        result = cache.get(0, 5)
        assert result is not None
        assert mx.array_equal(result["gate_proj"], data["gate_proj"])

    def test_get_missing_returns_none(self):
        from olmlx.engine.flash.moe_weight_store import ExpertCache

        cache = ExpertCache(max_experts_per_layer=10)
        assert cache.get(0, 99) is None

    def test_lru_eviction(self):
        from olmlx.engine.flash.moe_weight_store import ExpertCache

        cache = ExpertCache(max_experts_per_layer=3)
        for i in range(4):
            cache.put(0, i, {"w": mx.array([float(i)])})

        # Expert 0 should have been evicted (oldest)
        assert cache.get(0, 0) is None
        assert cache.get(0, 1) is not None
        assert cache.get(0, 2) is not None
        assert cache.get(0, 3) is not None

    def test_get_batch(self):
        from olmlx.engine.flash.moe_weight_store import ExpertCache

        cache = ExpertCache(max_experts_per_layer=10)
        cache.put(0, 1, {"w": mx.array([1.0])})
        cache.put(0, 3, {"w": mx.array([3.0])})

        cached = cache.get_batch(0, [1, 2, 3, 4])
        assert 1 in cached and 3 in cached
        assert 2 not in cached and 4 not in cached

    def test_layers_are_independent(self):
        from olmlx.engine.flash.moe_weight_store import ExpertCache

        cache = ExpertCache(max_experts_per_layer=5)
        cache.put(0, 1, {"w": mx.array([0.0])})
        cache.put(1, 1, {"w": mx.array([1.0])})

        r0 = cache.get(0, 1)
        r1 = cache.get(1, 1)
        assert r0 is not None and r1 is not None
        assert not mx.array_equal(r0["w"], r1["w"])


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

    def test_expert_index_map(self, store_with_model):
        """LoadedExperts should provide a mapping from global to local indices."""
        store, _, _, _, _ = store_with_model
        expert_indices = [3, 7, 1]
        loaded = store.load_experts(1, expert_indices)

        assert loaded.expert_index_map == {3: 0, 7: 1, 1: 2}


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
