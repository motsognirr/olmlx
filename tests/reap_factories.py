"""Tiny in-process MoE models for REAP tests. Never touches mlx_lm.load."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten


def _build(module, args):
    model = module.Model(args)
    mx.eval(model.parameters())
    return model, args


def make_tiny_qwen3_moe(*, num_experts=8, top_k=2, num_layers=2, hidden=64, vocab=128):
    from mlx_lm.models import qwen3_moe

    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe",
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        intermediate_size=hidden * 2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        rms_norm_eps=1e-5,
        vocab_size=vocab,
        rope_theta=10000.0,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=hidden // 2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
    )
    return _build(qwen3_moe, args)


def make_tiny_gpt_oss(
    *,
    num_experts=8,
    top_k=2,
    num_layers=2,
    hidden=64,
    vocab=128,
    sliding_window=4096,
):
    from mlx_lm.models import gpt_oss

    # GptOssMoeModel.__init__ requires both "sliding_attention" and
    # "full_attention" to appear in layer_types (it indexes .index() on
    # each), so an all-"full_attention" list (as sketched in the brief)
    # raises ValueError. Alternate the two so both are always present.
    layer_types = (["sliding_attention", "full_attention"] * ((num_layers + 1) // 2))[
        :num_layers
    ]

    # sliding_window defaults to 4096, which dwarfs the tests' 16-token
    # sequences (sliding-window semantics can't diverge from full causal at
    # that length). Callers that need real windowed-attention behavior (e.g.
    # a streaming-vs-hooked regression test) pass a small value explicitly.
    args = gpt_oss.ModelArgs(
        model_type="gpt_oss",
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        intermediate_size=hidden // 2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        rms_norm_eps=1e-5,
        vocab_size=vocab,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        sliding_window=sliding_window,
        rope_theta=10000.0,
        layer_types=layer_types,
    )
    return _build(gpt_oss, args)


def make_tiny_deepseek_v3(
    *, num_experts=8, top_k=2, num_layers=2, hidden=64, vocab=128
):
    from mlx_lm.models import deepseek_v3

    args = deepseek_v3.ModelArgs(
        model_type="deepseek_v3",
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        intermediate_size=hidden * 2,
        moe_intermediate_size=hidden // 2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vocab,
        n_routed_experts=num_experts,
        n_shared_experts=1,
        num_experts_per_tok=top_k,
        routed_scaling_factor=1.0,
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        kv_lora_rank=16,
        # DeepseekV3Attention branches on `q_lora_rank is None` to decide
        # whether to build the low-rank query projection at all; 0 (as
        # sketched in the brief) falls into the "build it" branch and
        # nn.Linear(hidden_size, 0, ...) divides by zero in its scale
        # calculation. None means "no q-LoRA", matching real dense-query
        # configs.
        q_lora_rank=None,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
    model, args = _build(deepseek_v3, args)
    # MoEGate.weight is a bare `mx.zeros(...)` attribute in mlx-lm's
    # deepseek_v3.py (not an nn.Linear, so it's never touched by mlx's
    # default random-init policy) — real usage always overwrites it via
    # checkpoint load. Left at zero, `x @ weight.T` is identically 0 for
    # every token, so top-k selection degenerates into a pure tie broken by
    # argpartition's *unstable* tie-break, which is sensitive to incidental
    # array memory layout (e.g. contiguous-after-eval vs a lazy gather) and
    # produced flaky/order-dependent equivalence & rotation-detection
    # results. Give it a small random draw so routing actually depends on
    # the input, matching every other (nn.Linear-backed) router style.
    for layer in model.model.layers:
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if gate is not None and hasattr(gate, "weight"):
            gate.weight = mx.random.normal(gate.weight.shape) * 0.02
    mx.eval(model.parameters())
    return model, args


def tiny_config_dict(args, model_type: str) -> dict:
    cfg = dataclasses.asdict(args)
    cfg["model_type"] = model_type
    return cfg


def save_tiny_model(model, config: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    mx.eval(*weights.values())
    mx.save_safetensors(str(out_dir / "model.safetensors"), weights)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    return out_dir
