"""Equivalence checking for REAP pruning: mask dropped experts on the full model
and compare against the pruned artifact.

Also the runtime backbone of `reap report`'s sanity check. The rotation test in
tests/test_reap_equivalence.py proves this check can detect misrouting; keep it
sensitive (max-abs on logits, no mean-pooling)."""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.reap.arch import STYLE_DEEPSEEK, find_moe_module


class _MaskedGate(nn.Module):
    """Additive -inf on dropped experts' logits; exact removal from softmax mass."""

    def __init__(self, inner: Any, mask_vec: mx.array) -> None:
        super().__init__()
        self.inner = inner
        self._mask_vec = mask_vec

    def __call__(self, x):
        return self.inner(x) + self._mask_vec


def _dropped_mask(num_experts: int, kept: list[int]) -> mx.array:
    """-inf (additive) at dropped expert positions, 0.0 at kept ones.

    mlx arrays don't support item assignment, so build this via mx.where over
    a boolean membership array rather than in-place indexed writes.
    """
    kept_set = set(kept)
    is_dropped = mx.array([e not in kept_set for e in range(num_experts)])
    mask = mx.where(is_dropped, mx.array(-mx.inf), mx.array(0.0))
    return mask


def mask_dropped_experts(model, keep: dict[int, list[int]]) -> Callable[[], None]:
    """Forces dropped experts unroutable IN PLACE on the full model; returns restore()."""
    inner = getattr(model, "model", model)
    layers = inner.layers
    restores: list[Callable[[], None]] = []
    for layer_idx, kept in keep.items():
        info = find_moe_module(layers[layer_idx])
        if info is None:
            raise ValueError(f"layer {layer_idx} has no MoE module")
        dropped = [e for e in range(info.num_experts) if e not in set(kept)]
        if not dropped:
            continue
        neg = _dropped_mask(info.num_experts, kept)

        if info.style == STYLE_DEEPSEEK:
            gate = info.module.gate
            orig_bias = gate.e_score_correction_bias
            gate.e_score_correction_bias = orig_bias + neg

            def _restore(g=gate, b=orig_bias):
                g.e_score_correction_bias = b

            restores.append(_restore)
        else:
            gate = getattr(info.module, info.gate_attr)
            setattr(info.module, info.gate_attr, _MaskedGate(gate, neg))

            def _restore(m=info.module, a=info.gate_attr, g=gate):
                setattr(m, a, g)

            restores.append(_restore)

    def restore_all() -> None:
        for r in reversed(restores):
            r()

    return restore_all


def max_logit_divergence(model_a, model_b, batches: list[mx.array]) -> float:
    """max |logits_a - logits_b| over the batches (float32 compare)."""
    worst = 0.0
    for batch in batches:
        la = model_a(batch).astype(mx.float32)
        lb = model_b(batch).astype(mx.float32)
        worst = max(worst, float(mx.max(mx.abs(la - lb))))
    return worst
