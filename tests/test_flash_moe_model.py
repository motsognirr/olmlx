"""Tests for olmlx.engine.flash.flash_moe_model — Flash-MoE model wrapper."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from tests.test_flash_moe_bundler import _make_synthetic_moe_weights


# ---------------------------------------------------------------------------
# Synthetic DeepSeek-V3-like model for testing
# ---------------------------------------------------------------------------


class _MockMLP(nn.Module):
    """Standard dense MLP."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _MockSwitchGLU(nn.Module):
    """Mock SwitchGLU with stacked expert weights."""

    def __init__(self, hidden_size, intermediate_size, num_experts):
        super().__init__()
        self.gate_proj = _MockSwitchLinear(hidden_size, intermediate_size, num_experts)
        self.up_proj = _MockSwitchLinear(hidden_size, intermediate_size, num_experts)
        self.down_proj = _MockSwitchLinear(intermediate_size, hidden_size, num_experts)


class _MockSwitchLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts):
        super().__init__()
        self.weight = mx.random.normal((num_experts, out_dim, in_dim))


class _MockMoEGate(nn.Module):
    """Mock router that returns fixed indices and uniform scores."""

    def __init__(self, num_experts, num_experts_per_tok):
        super().__init__()
        self.weight = mx.random.normal((num_experts, 8))  # dummy
        self.num_experts_per_tok = num_experts_per_tok

    def __call__(self, x):
        B, L, _ = x.shape
        K = self.num_experts_per_tok
        inds = mx.zeros((B, L, K), dtype=mx.int32)
        scores = mx.ones((B, L, K)) / K
        return inds, scores


class _MockMoE(nn.Module):
    """Mock DeepseekV3MoE."""

    def __init__(
        self, hidden_size, intermediate_size, num_experts, num_experts_per_tok
    ):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "n_shared_experts": 1,
                "moe_intermediate_size": intermediate_size,
                "n_routed_experts": num_experts,
                "num_experts_per_tok": num_experts_per_tok,
            },
        )()
        self.gate = _MockMoEGate(num_experts, num_experts_per_tok)
        self.switch_mlp = _MockSwitchGLU(hidden_size, intermediate_size, num_experts)
        self.shared_experts = _MockMLP(hidden_size, intermediate_size)
        self.sharding_group = None
        # Tag for detection
        self._is_moe = True

    def __call__(self, x):
        inds, scores = self.gate(x)
        return x  # Simplified — just return x for testing


class _MockDecoderLayer(nn.Module):
    def __init__(
        self, hidden_size, intermediate_size, num_experts, num_experts_per_tok, is_moe
    ):
        super().__init__()
        if is_moe:
            self.mlp = _MockMoE(
                hidden_size, intermediate_size, num_experts, num_experts_per_tok
            )
        else:
            self.mlp = _MockMLP(hidden_size, intermediate_size)


class _MockModel(nn.Module):
    """Mock model with mix of dense and MoE layers."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_experts,
        num_experts_per_tok,
        num_dense,
        num_moe,
    ):
        super().__init__()
        self.args = type(
            "Args",
            (),
            {
                "hidden_size": hidden_size,
                "moe_intermediate_size": intermediate_size,
                "n_routed_experts": num_experts,
                "num_experts_per_tok": num_experts_per_tok,
            },
        )()
        layers = []
        for i in range(num_dense):
            layers.append(
                _MockDecoderLayer(
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    num_experts_per_tok,
                    is_moe=False,
                )
            )
        for i in range(num_moe):
            layers.append(
                _MockDecoderLayer(
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    num_experts_per_tok,
                    is_moe=True,
                )
            )
        self.layers = layers

    def __call__(self, x, cache=None):
        for layer in self.layers:
            x = layer.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlashMoeModelWrapper:
    @pytest.fixture()
    def model_and_store(self, tmp_path):
        """Create a mock model and matching FlashMoeWeightStore."""
        hidden, inter, experts = 64, 32, 8
        num_dense, num_moe = 1, 2
        num_experts_per_tok = 2

        # Create bundled weights for the MoE layers (layers 1 and 2)
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

        model = _MockModel(
            hidden, inter, experts, num_experts_per_tok, num_dense, num_moe
        )
        return model, store, hidden, inter, experts, num_experts_per_tok

    def test_replaces_moe_layers(self, model_and_store):
        """MoE layers should be replaced with FlashMoE, dense layers preserved."""
        model, store, hidden, inter, experts, num_experts_per_tok = model_and_store

        from olmlx.engine.flash.flash_moe import FlashMoE
        from olmlx.engine.flash.flash_moe_model import FlashMoeModelWrapper

        moe_layer_indices = [1, 2]
        wrapped = FlashMoeModelWrapper(
            model, store, moe_layer_indices, hidden, inter, experts, num_experts_per_tok
        )

        # Layer 0 (dense) should be unchanged
        assert isinstance(wrapped.layers[0].mlp, _MockMLP)
        # Layers 1, 2 (MoE) should be replaced entirely
        from olmlx.engine.flash.flash_moe_model import _FlashMoEDeepSeek

        for i in [1, 2]:
            mlp = wrapped.layers[i].mlp
            assert isinstance(mlp, _FlashMoEDeepSeek)
            assert isinstance(mlp._flash_moe, FlashMoE)

    def test_preserves_gate_and_shared(self, model_and_store):
        """Gate (router) and shared experts should remain in the MoE layer."""
        model, store, hidden, inter, experts, num_experts_per_tok = model_and_store

        from olmlx.engine.flash.flash_moe_model import FlashMoeModelWrapper

        moe_layer_indices = [1, 2]
        wrapped = FlashMoeModelWrapper(
            model, store, moe_layer_indices, hidden, inter, experts, num_experts_per_tok
        )

        for i in [1, 2]:
            mlp = wrapped.layers[i].mlp
            assert hasattr(mlp, "gate")
            assert hasattr(mlp, "shared_experts")

    def test_frees_switch_mlp_weights(self, model_and_store):
        """Original SwitchGLU weights should be deleted after wrapping."""
        model, store, hidden, inter, experts, num_experts_per_tok = model_and_store

        from olmlx.engine.flash.flash_moe_model import FlashMoeModelWrapper

        moe_layer_indices = [1, 2]
        wrapped = FlashMoeModelWrapper(
            model, store, moe_layer_indices, hidden, inter, experts, num_experts_per_tok
        )

        for i in [1, 2]:
            mlp = wrapped.layers[i].mlp
            # The replacement should not have switch_mlp or experts
            assert not hasattr(mlp, "switch_mlp")
            assert not hasattr(mlp, "experts")

    def test_proxies_attributes(self, model_and_store):
        """Wrapper should proxy model attributes like layers and args."""
        model, store, hidden, inter, experts, num_experts_per_tok = model_and_store

        from olmlx.engine.flash.flash_moe_model import FlashMoeModelWrapper

        moe_layer_indices = [1, 2]
        wrapped = FlashMoeModelWrapper(
            model, store, moe_layer_indices, hidden, inter, experts, num_experts_per_tok
        )

        assert wrapped.layers is model.layers
        assert wrapped.args is model.args
