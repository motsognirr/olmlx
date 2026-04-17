"""Tests for olmlx.engine.flash.flash_moe — runtime FlashMoE module."""

import mlx.core as mx
import mlx.nn as nn

from tests.test_flash_moe_bundler import (
    _make_synthetic_moe_weights,
    _make_synthetic_nemotron_moe_weights,
)


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

    def test_output_matches_python_remap_reference(self, tmp_path):
        """mx.take-based remap must produce bit-identical output to a Python-list remap."""
        hidden, inter, experts = 32, 16, 8
        flash_moe, store, _ = _setup_flash_moe(
            tmp_path, hidden, inter, experts, num_experts_per_tok=2
        )

        x = mx.random.normal((2, 4, hidden))
        # Mix of repeated and distinct experts to exercise the remap fully
        inds = mx.array(
            [
                [[0, 3], [1, 5], [0, 3], [2, 7]],
                [[4, 6], [1, 5], [2, 7], [0, 3]],
            ]
        )
        scores = mx.softmax(mx.random.normal(inds.shape).astype(mx.float32), axis=-1)

        out_new = flash_moe(x, inds, scores)
        mx.eval(out_new)

        # Reference: reconstruct output using a pure-Python remap on the same cached
        # weights. Pull them back via store.load_experts (deterministic for the same set).
        B, L, K = inds.shape
        flat = inds.reshape(-1).tolist()
        unique = sorted(set(flat))
        loaded = store.load_experts(layer_idx=0, expert_indices=unique)
        idx_map = loaded.expert_index_map
        remap_py = mx.array(
            [idx_map[int(i)] for i in flat], dtype=mx.uint32
        ).reshape(B, L, K)

        x_expanded = mx.expand_dims(x, (-2, -3))
        if loaded.is_quantized:
            qkw = dict(
                transpose=True,
                group_size=loaded.group_size,
                bits=loaded.bits,
                mode=loaded.quant_mode,
            )
            g = mx.gather_qmm(
                x_expanded, loaded.gate_weight, loaded.gate_scales, loaded.gate_biases,
                rhs_indices=remap_py, **qkw,
            )
            u = mx.gather_qmm(
                x_expanded, loaded.up_weight, loaded.up_scales, loaded.up_biases,
                rhs_indices=remap_py, **qkw,
            )
            act = nn.silu(g) * u
            e = mx.gather_qmm(
                act, loaded.down_weight, loaded.down_scales, loaded.down_biases,
                rhs_indices=remap_py, **qkw,
            )
        else:
            g = mx.gather_mm(  # pyright: ignore[reportCallIssue]
                x_expanded, loaded.gate_weight.swapaxes(-1, -2), rhs_indices=remap_py
            )
            u = mx.gather_mm(  # pyright: ignore[reportCallIssue]
                x_expanded, loaded.up_weight.swapaxes(-1, -2), rhs_indices=remap_py
            )
            act = nn.silu(g) * u
            e = mx.gather_mm(  # pyright: ignore[reportCallIssue]
                act, loaded.down_weight.swapaxes(-1, -2), rhs_indices=remap_py
            )
        e = e.squeeze(-2)
        out_ref = (e * scores[..., None]).sum(axis=-2).astype(x.dtype)

        assert mx.allclose(out_new, out_ref, atol=0, rtol=0)


def _setup_nemotron_flash_moe(
    tmp_path, hidden=64, inter=32, experts=8, num_experts_per_tok=2
):
    """Create bundled Nemotron fc1/fc2 MoE weights and return a FlashMoE instance."""
    model_dir = _make_synthetic_nemotron_moe_weights(
        hidden, inter, experts, 2, "ME", tmp_path
    )
    output_dir = tmp_path / "flash_moe"

    from olmlx.engine.flash.moe_bundler import bundle_moe_experts

    bundle_moe_experts(model_dir, output_dir)

    from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

    store = FlashMoeWeightStore(output_dir, num_io_threads=4, cache_budget_experts=16)

    from olmlx.engine.flash.flash_moe import FlashMoE

    # Nemotron uses relu2 (ReLU²) activation — no gating
    def relu2(x):
        return mx.square(nn.relu(x))

    flash_moe = FlashMoE(
        layer_idx=1,
        hidden_size=hidden,
        intermediate_size=inter,
        num_experts=experts,
        num_experts_per_tok=num_experts_per_tok,
        weight_store=store,
        activation=relu2,
    )

    return flash_moe, store, model_dir


class TestFlashMoENemotron:
    """Test FlashMoE with non-gated fc1/fc2 expert style (Nemotron-H)."""

    def test_output_shape(self, tmp_path):
        """FlashMoE with fc1/fc2 experts should produce correct output shape."""
        hidden, inter, experts = 64, 32, 8
        flash_moe, _, _ = _setup_nemotron_flash_moe(tmp_path, hidden, inter, experts)

        x = mx.random.normal((2, 3, hidden))
        inds = mx.array([[[0, 3], [1, 5], [2, 7]], [[3, 5], [2, 4], [6, 7]]])
        scores = mx.ones(inds.shape, dtype=mx.float32) * 0.5

        output = flash_moe(x, inds, scores)
        assert output.shape == x.shape

    def test_matches_manual_computation(self, tmp_path):
        """FlashMoE fc1/fc2 output should match manual: relu2(x @ fc1.T) @ fc2.T."""
        hidden, inter, experts = 32, 16, 4
        flash_moe, _, model_dir = _setup_nemotron_flash_moe(
            tmp_path, hidden, inter, experts, num_experts_per_tok=2
        )

        from safetensors.numpy import load_file

        original = load_file(str(model_dir / "model.safetensors"))
        fc1_w = mx.array(original["backbone.layers.1.mixer.switch_mlp.fc1.weight"])
        fc2_w = mx.array(original["backbone.layers.1.mixer.switch_mlp.fc2.weight"])

        x = mx.random.normal((1, 1, hidden))
        inds = mx.array([[[0, 2]]])
        scores = mx.array([[[0.6, 0.4]]])

        output = flash_moe(x, inds, scores)
        mx.eval(output)

        # Manual: relu2(x @ fc1.T) @ fc2.T
        x_flat = x.reshape(1, hidden)
        manual_output = mx.zeros((1, hidden))
        for eidx, score in [(0, 0.6), (2, 0.4)]:
            h = x_flat @ fc1_w[eidx].T
            activated = mx.square(nn.relu(h))
            expert_out = activated @ fc2_w[eidx].T
            manual_output = manual_output + expert_out * score

        assert mx.allclose(output.reshape(1, hidden), manual_output, atol=1e-3)
