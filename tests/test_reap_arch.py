from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.reap.arch import (
    STYLE_DEEPSEEK,
    STYLE_GPT_OSS,
    STYLE_MINIMAX,
    STYLE_QWEN3,
    STYLE_QWEN3_NEXT,
    applied_scores,
    detect_router_style,
    find_moe_module,
)
from tests.reap_factories import (
    make_tiny_deepseek_v3,
    make_tiny_gpt_oss,
    make_tiny_qwen3_moe,
)


class TestDetectRouterStyle:
    def test_qwen3_moe(self):
        model, _ = make_tiny_qwen3_moe()
        assert detect_router_style(model.model.layers[0].mlp) == STYLE_QWEN3

    def test_gpt_oss(self):
        model, _ = make_tiny_gpt_oss()
        assert detect_router_style(model.model.layers[0].mlp) == STYLE_GPT_OSS

    def test_deepseek_v3_moe_layer(self):
        model, args = make_tiny_deepseek_v3()
        # first_k_dense_replace=1 -> layer 0 dense, layer 1 MoE
        assert detect_router_style(model.model.layers[1].mlp) == STYLE_DEEPSEEK

    def test_qwen3_next_style_via_fake(self):
        class Fake(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(8, 4, bias=False)
                self.shared_expert_gate = nn.Linear(8, 1, bias=False)

        assert detect_router_style(Fake()) == STYLE_QWEN3_NEXT

    def test_minimax_style_via_fake(self):
        class Fake(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(8, 4, bias=False)
                self.e_score_correction_bias = mx.zeros((4,))

        assert detect_router_style(Fake()) == STYLE_MINIMAX


class TestFindMoeModule:
    def test_qwen3_moe(self):
        model, args = make_tiny_qwen3_moe(num_experts=8, top_k=2)
        info = find_moe_module(model.model.layers[0])
        assert info is not None
        assert (info.attr_name, info.container_name, info.gate_attr) == (
            "mlp",
            "switch_mlp",
            "gate",
        )
        assert info.num_experts == 8 and info.top_k == 2

    def test_gpt_oss_container_is_experts(self):
        model, _ = make_tiny_gpt_oss()
        info = find_moe_module(model.model.layers[0])
        assert (info.container_name, info.gate_attr) == ("experts", "router")

    def test_dense_layer_returns_none(self):
        model, _ = make_tiny_deepseek_v3()
        assert (
            find_moe_module(model.model.layers[0]) is None
        )  # dense (first_k_dense_replace=1)
        assert find_moe_module(model.model.layers[1]) is not None


class TestAppliedScores:
    def test_qwen3_matches_block_math(self):
        model, _ = make_tiny_qwen3_moe(num_experts=8, top_k=2)
        moe = model.model.layers[0].mlp
        x = mx.random.normal((3, 5, 64))
        logits = moe.gate(x)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        inds = mx.stop_gradient(mx.argpartition(-probs, kth=1, axis=-1)[..., :2])
        scores = applied_scores(STYLE_QWEN3, moe, logits, inds)
        expected = mx.take_along_axis(probs, inds, axis=-1)
        expected = expected / mx.sum(
            expected, axis=-1, keepdims=True
        )  # norm_topk_prob=True
        assert mx.allclose(scores, expected, atol=1e-6)

    def test_scores_sum_to_one_when_renormed(self):
        model, _ = make_tiny_qwen3_moe()
        moe = model.model.layers[0].mlp
        x = mx.random.normal((2, 4, 64))
        logits = moe.gate(x)
        inds = mx.argpartition(-logits, kth=1, axis=-1)[..., :2]
        s = applied_scores(STYLE_QWEN3, moe, logits, inds)
        assert mx.allclose(mx.sum(s, axis=-1), mx.ones(s.shape[:-1]), atol=1e-5)

    def test_gpt_oss_softmax_over_selected(self):
        model, _ = make_tiny_gpt_oss(num_experts=8, top_k=2)
        moe = model.model.layers[0].mlp
        x = mx.random.normal((2, 4, 64))
        logits = moe.router(x)
        inds = mx.argpartition(-logits, kth=1, axis=-1)[..., :2]
        s = applied_scores(STYLE_GPT_OSS, moe, logits, inds)
        assert mx.allclose(
            s, mx.softmax(mx.take_along_axis(logits, inds, axis=-1), axis=-1), atol=1e-6
        )

    def test_deepseek_passthrough(self):
        model, _ = make_tiny_deepseek_v3()
        moe = model.model.layers[1].mlp
        x = mx.random.normal((2, 4, 64))
        gate_out = moe.gate(x)  # (inds, scores)
        s = applied_scores(STYLE_DEEPSEEK, moe, gate_out, gate_out[0])
        assert mx.array_equal(s, gate_out[1])
