"""Tests for olmlx.engine.flash.flash_moe — runtime FlashMoE module."""

import mlx.core as mx
import mlx.nn as nn

from tests.test_flash_moe_bundler import _make_synthetic_moe_weights


def _setup_flash_moe(tmp_path, hidden=64, inter=32, experts=8, num_experts_per_tok=2):
    """Create bundled MoE weights and return a FlashMoE instance + reference weights."""
    model_dir = _make_synthetic_moe_weights(hidden, inter, experts, 1, 0, tmp_path)
    output_dir = tmp_path / "flash_moe"

    from olmlx.engine.flash.moe_bundler import bundle_moe_experts

    bundle_moe_experts(model_dir, output_dir)

    from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

    store = FlashMoeWeightStore(output_dir, num_io_threads=4, cache_budget_experts=16)

    from olmlx.engine.flash.flash_moe import FlashMoE

    flash_moe = FlashMoE(
        layer_idx=0,
        hidden_size=hidden,
        intermediate_size=inter,
        num_experts=experts,
        num_experts_per_tok=num_experts_per_tok,
        weight_store=store,
    )

    return flash_moe, store, model_dir


class TestFlashMoE:
    def test_output_shape(self, tmp_path):
        """FlashMoE output should match input shape."""
        hidden, inter, experts = 64, 32, 8
        flash_moe, store, _ = _setup_flash_moe(tmp_path, hidden, inter, experts)

        x = mx.random.normal((2, 5, hidden))
        # Simulate router indices and scores
        inds = mx.array(
            [
                [[0, 3], [1, 5], [2, 7], [4, 6], [0, 1]],
                [[3, 5], [2, 4], [6, 7], [1, 0], [5, 3]],
            ]
        )
        scores = mx.ones(inds.shape, dtype=mx.float32) * 0.5

        output = flash_moe(x, inds, scores)
        assert output.shape == x.shape

    def test_matches_manual_computation(self, tmp_path):
        """FlashMoE output should match manually computed expert dispatch."""
        hidden, inter, experts = 32, 16, 4
        num_experts_per_tok = 2
        flash_moe, store, model_dir = _setup_flash_moe(
            tmp_path, hidden, inter, experts, num_experts_per_tok
        )

        from safetensors.numpy import load_file

        original = load_file(str(model_dir / "model.safetensors"))
        gate_w = mx.array(original["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
        up_w = mx.array(original["model.layers.0.mlp.switch_mlp.up_proj.weight"])
        down_w = mx.array(original["model.layers.0.mlp.switch_mlp.down_proj.weight"])

        # Single token, two experts
        x = mx.random.normal((1, 1, hidden))
        inds = mx.array([[[0, 2]]])  # select experts 0 and 2
        scores = mx.array([[[0.6, 0.4]]])

        output = flash_moe(x, inds, scores)
        mx.eval(output)

        # Manual computation: for each expert, compute SwiGLU and weight by score
        x_flat = x.reshape(1, hidden)
        manual_output = mx.zeros((1, hidden))
        for i, (eidx, score) in enumerate([(0, 0.6), (2, 0.4)]):
            g = x_flat @ gate_w[eidx].T  # (1, inter)
            u = x_flat @ up_w[eidx].T  # (1, inter)
            activated = nn.silu(g) * u
            expert_out = activated @ down_w[eidx].T  # (1, hidden)
            manual_output = manual_output + expert_out * score

        assert mx.allclose(output.reshape(1, hidden), manual_output, atol=1e-3)

    def test_different_experts_give_different_outputs(self, tmp_path):
        """Routing to different experts should produce different results."""
        hidden, inter, experts = 64, 32, 8
        flash_moe, store, _ = _setup_flash_moe(tmp_path, hidden, inter, experts)

        x = mx.random.normal((1, 1, hidden))

        inds1 = mx.array([[[0, 1]]])
        inds2 = mx.array([[[6, 7]]])
        scores = mx.array([[[0.5, 0.5]]])

        out1 = flash_moe(x, inds1, scores)
        out2 = flash_moe(x, inds2, scores)

        assert not mx.allclose(out1, out2, atol=1e-6)

    def test_scores_weighting(self, tmp_path):
        """Score of 0 for one expert should produce output from only the other."""
        hidden, inter, experts = 32, 16, 4
        flash_moe, store, _ = _setup_flash_moe(tmp_path, hidden, inter, experts)

        x = mx.random.normal((1, 1, hidden))

        # Only expert 0 contributes (score for expert 1 is 0)
        inds = mx.array([[[0, 1]]])
        scores_a = mx.array([[[1.0, 0.0]]])
        scores_b = mx.array([[[0.0, 1.0]]])

        out_a = flash_moe(x, inds, scores_a)
        out_b = flash_moe(x, inds, scores_b)

        # out_a should differ from out_b since they weight different experts
        assert not mx.allclose(out_a, out_b, atol=1e-6)

    def test_batch_processing(self, tmp_path):
        """FlashMoE should handle batch dimension correctly."""
        hidden, inter, experts = 64, 32, 8
        flash_moe, store, _ = _setup_flash_moe(tmp_path, hidden, inter, experts)

        batch_size = 3
        seq_len = 4
        x = mx.random.normal((batch_size, seq_len, hidden))
        inds = mx.random.randint(0, experts, (batch_size, seq_len, 2))
        scores = mx.softmax(mx.random.normal((batch_size, seq_len, 2)), axis=-1)

        output = flash_moe(x, inds, scores)
        assert output.shape == (batch_size, seq_len, hidden)
