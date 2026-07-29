"""Per-architecture MoE introspection for REAP.

Mirrors the router-style dispatch chain in flash/flash_moe_model.py
_replace_moe_layers (detection order is load-bearing there and here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

STYLE_QWEN3 = "qwen3"  # plain linear gate, softmax before top-k
STYLE_QWEN3_NEXT = "qwen3_next"  # linear gate + shared_expert_gate
STYLE_MINIMAX = "minimax"  # linear gate + block-level e_score_correction_bias
STYLE_DEEPSEEK = "deepseek"  # custom gate module returning (inds, scores)
STYLE_GPT_OSS = "gpt_oss"  # `router` linear + `experts` container

_MOE_ATTRS = ("mlp", "block_sparse_moe", "mixer")
_CONTAINER_ATTRS = ("switch_mlp", "experts")
_PROJ_PROBES = ("gate_proj", "fc1")


@dataclass
class MoeModuleInfo:
    attr_name: str
    module: Any
    container_name: str
    gate_attr: str
    style: str
    num_experts: int
    top_k: int


def _gate_is_linear(gate: Any) -> bool:
    linear_types: tuple[type, ...] = (nn.Linear,)
    if hasattr(nn, "QuantizedLinear"):
        linear_types = (nn.Linear, nn.QuantizedLinear)
    return isinstance(gate, linear_types)


def detect_router_style(moe_module: Any) -> str:
    gate = getattr(moe_module, "gate", None)
    if _gate_is_linear(gate) and hasattr(moe_module, "shared_expert_gate"):
        return STYLE_QWEN3_NEXT
    if _gate_is_linear(gate) and hasattr(moe_module, "e_score_correction_bias"):
        return STYLE_MINIMAX
    if _gate_is_linear(gate):
        return STYLE_QWEN3
    if gate is not None:
        return STYLE_DEEPSEEK
    if getattr(moe_module, "router", None) is not None:
        return STYLE_GPT_OSS
    raise ValueError("unrecognized MoE router layout")


def _find_container(moe_module: Any) -> tuple[str, Any] | None:
    for name in _CONTAINER_ATTRS:
        cont = getattr(moe_module, name, None)
        if cont is None:
            continue
        for probe in _PROJ_PROBES:
            proj = getattr(cont, probe, None)
            if proj is not None and getattr(proj, "weight", None) is not None:
                return name, cont
    return None


def _top_k_of(moe_module: Any) -> int | None:
    for owner in (moe_module, getattr(moe_module, "gate", None)):
        if owner is None:
            continue
        for attr in ("top_k", "num_experts_per_tok", "num_local_experts_per_tok"):
            val = getattr(owner, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    config = getattr(getattr(moe_module, "gate", None), "config", None)
    val = getattr(config, "num_experts_per_tok", None)
    return val if isinstance(val, int) else None


def find_moe_module(
    layer: Any, *, top_k_hint: int | None = None
) -> MoeModuleInfo | None:
    for attr in _MOE_ATTRS:
        moe = getattr(layer, attr, None)
        if moe is None:
            continue
        found = _find_container(moe)
        if found is None:
            continue
        container_name, container = found
        try:
            style = detect_router_style(moe)
        except ValueError:
            continue
        # gpt-oss keeps experts under the same attr the container probe found;
        # in that layout `moe` IS the block and `container` its SwitchGLU.
        proj = getattr(container, "gate_proj", None) or getattr(container, "fc1", None)
        num_experts = int(proj.weight.shape[0])
        top_k = _top_k_of(moe) or top_k_hint
        if top_k is None:
            raise ValueError(f"cannot determine top_k for MoE module {attr!r}")
        gate_attr = "router" if style == STYLE_GPT_OSS else "gate"
        return MoeModuleInfo(
            attr_name=attr,
            module=moe,
            container_name=container_name,
            gate_attr=gate_attr,
            style=style,
            num_experts=num_experts,
            top_k=int(top_k),
        )
    return None


def applied_scores(
    style: str, moe_module: Any, gate_out: Any, inds: mx.array
) -> mx.array:
    """Routing weight actually applied to each selected expert's output.

    gate_out is the raw gate/router output: logits for linear styles, the
    (inds, scores) tuple for DeepSeek-style custom gates.

    Score math mirrors the installed mlx-lm model sources exactly:
    - qwen3_moe.py / qwen3_next.py: softmax(precise=True) over the full
      logits *before* top-k selection, then renormalize the selected
      scores if norm_topk_prob is set.
    - gpt_oss.py: softmax(precise=True) over only the *selected* logits
      (no renorm — softmax over top-k already sums to one).
    - minimax.py: sigmoid over the full (unbiased) logits, select, then
      renormalize with a 1e-20 epsilon (top-k selection itself uses a
      bias-corrected score, but the values taken and renormalized are the
      unbiased sigmoid scores).
    - deepseek_v3.py: MoEGate already returns (inds, scores) with all
      selection/renorm/scaling baked in — pure passthrough.
    """
    if style == STYLE_DEEPSEEK:
        return gate_out[1]
    if style in (STYLE_QWEN3, STYLE_QWEN3_NEXT):
        probs = mx.softmax(gate_out.astype(mx.float32), axis=-1, precise=True)
        scores = mx.take_along_axis(probs, inds, axis=-1)
        if getattr(moe_module, "norm_topk_prob", False):
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        return scores
    if style == STYLE_MINIMAX:
        probs = mx.sigmoid(gate_out.astype(mx.float32))
        scores = mx.take_along_axis(probs, inds, axis=-1)
        return scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
    if style == STYLE_GPT_OSS:
        selected = mx.take_along_axis(gate_out, inds, axis=-1)
        return mx.softmax(selected, axis=-1, precise=True)
    raise ValueError(f"unknown router style {style!r}")
