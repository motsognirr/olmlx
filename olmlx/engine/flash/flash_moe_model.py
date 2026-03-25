"""Flash MoE model wrapper.

Wraps an mlx-lm model, replacing MoE layers' SwitchGLU dispatch with
FlashMoE instances that load expert weights on demand from SSD.
Compatible with mlx_lm.stream_generate().
"""

from __future__ import annotations

import gc
import logging

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.flash.flash_moe import FlashMoE
from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

logger = logging.getLogger(__name__)


class _FlashMoEDeepSeek(nn.Module):
    """Replacement MoE layer for DeepSeek-V3 / Kimi-K2.5 style models.

    Keeps gate and shared_experts in RAM, uses FlashMoE for expert dispatch.
    """

    def __init__(self, original_moe, flash_moe: FlashMoE):
        super().__init__()
        if getattr(original_moe, "sharding_group", None) is not None:
            raise NotImplementedError(
                "Flash-MoE does not support distributed tensor parallelism. "
                "Each rank loads all needed experts, so all_sum would produce "
                "incorrect results. Disable distributed or Flash-MoE."
            )
        self.gate = original_moe.gate
        self._flash_moe = flash_moe
        if hasattr(original_moe, "shared_experts"):
            self.shared_experts = original_moe.shared_experts

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self._flash_moe(x, inds, scores)
        y = y.astype(x.dtype)
        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts(x)
        return y


class _FlashMoEGptOss(nn.Module):
    """Replacement MoE layer for gpt-oss style models.

    Keeps router in RAM, uses FlashMoE for expert dispatch.
    """

    def __init__(self, original_mlp, flash_moe: FlashMoE):
        super().__init__()
        if getattr(original_mlp, "sharding_group", None) is not None:
            raise NotImplementedError(
                "Flash-MoE does not support distributed tensor parallelism. "
                "Each rank loads all needed experts, so all_sum would produce "
                "incorrect results. Disable distributed or Flash-MoE."
            )
        self.router = original_mlp.router
        self.num_experts_per_tok = original_mlp.num_experts_per_tok
        self._flash_moe = flash_moe

    def __call__(self, x):
        g = self.router(x)
        k = self.num_experts_per_tok
        # topk
        part_inds = mx.argpartition(g, kth=-k, axis=-1)
        inds = part_inds[..., -k:]
        scores = mx.take_along_axis(g, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
        y = self._flash_moe(x, inds, scores)
        return y.astype(x.dtype)


class _FlashMoEQwen3Next(nn.Module):
    """Replacement MoE layer for Qwen3-Next style models.

    Gate is a plain nn.Linear (returns logits, not (inds, scores)).
    Keeps gate, shared_expert, and shared_expert_gate in RAM.
    """

    def __init__(self, original_moe, flash_moe: FlashMoE):
        super().__init__()
        if getattr(original_moe, "sharding_group", None) is not None:
            raise NotImplementedError(
                "Flash-MoE does not support distributed tensor parallelism. "
                "Each rank loads all needed experts, so all_sum would produce "
                "incorrect results. Disable distributed or Flash-MoE."
            )
        self.gate = original_moe.gate
        self.top_k = original_moe.top_k
        self.norm_topk_prob = original_moe.norm_topk_prob
        self._flash_moe = flash_moe
        self.shared_expert = original_moe.shared_expert
        self.shared_expert_gate = original_moe.shared_expert_gate

    def __call__(self, x):
        scores = mx.softmax(self.gate(x).astype(mx.float32), axis=-1)
        k = self.top_k
        inds = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self._flash_moe(x, inds, scores)
        y = y.astype(x.dtype)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        return y + shared_y


class FlashMoeModelWrapper(nn.Module):
    """Wraps an mlx-lm model for Flash-MoE inference.

    - Router (gate), shared experts, attention, embeddings stay in RAM
    - SwitchGLU expert weights are replaced with FlashMoE (SSD-loaded)
    - Compatible with mlx_lm.stream_generate() without changes
    """

    def __init__(
        self,
        model: nn.Module,
        weight_store: FlashMoeWeightStore,
        moe_layer_indices: list[int],
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        self._model = model
        self._weight_store = weight_store
        _replace_moe_layers(
            model,
            weight_store,
            moe_layer_indices,
            hidden_size,
            intermediate_size,
            num_experts,
            num_experts_per_tok,
        )

    def __call__(self, inputs, cache=None, **kwargs):
        """Forward pass — delegates to the wrapped model."""
        return self._model(inputs, cache=cache, **kwargs)

    def __getattr__(self, name):
        """Proxy non-private attributes to the wrapped model."""
        if name.startswith("_") or name in ("training",):
            return super().__getattr__(name)
        return getattr(self._model, name)

    @property
    def layers(self):
        return self._model.layers

    @property
    def args(self):
        return self._model.args


def _replace_moe_layers(
    model: nn.Module,
    weight_store: FlashMoeWeightStore,
    moe_layer_indices: list[int],
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_experts_per_tok: int,
) -> None:
    """Replace MoE layers with FlashMoE variants."""
    layers = model.layers
    replaced = 0

    for layer_idx in moe_layer_indices:
        layer = layers[layer_idx]
        moe_module = layer.mlp

        # Extract activation function from the original SwitchGLU before we delete it
        activation = None
        for attr in ("switch_mlp", "experts"):
            switch = getattr(moe_module, attr, None)
            if switch is not None and hasattr(switch, "activation"):
                activation = switch.activation
                break

        # Create FlashMoE for this layer
        flash_moe = FlashMoE(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            weight_store=weight_store,
            activation=activation,
        )

        # Detect router style and create appropriate replacement.
        # Qwen3-Next: gate is a plain nn.Linear (returns logits); we apply softmax.
        # DeepSeek-V3: gate is a custom module returning (inds, scores) directly.
        gate = getattr(moe_module, "gate", None)
        if isinstance(gate, nn.Linear) and hasattr(moe_module, "shared_expert_gate"):
            # Qwen3-Next style: plain nn.Linear gate + shared_expert + shared_expert_gate
            replacement = _FlashMoEQwen3Next(moe_module, flash_moe)
        elif gate is not None:
            # DeepSeek-V3 / Kimi-K2.5 style: gate returns (inds, scores)
            replacement = _FlashMoEDeepSeek(moe_module, flash_moe)
        else:
            # gpt-oss style (has router + experts)
            replacement = _FlashMoEGptOss(moe_module, flash_moe)

        # Delete original SwitchGLU weights before replacing
        for attr in ("switch_mlp", "experts"):
            if hasattr(moe_module, attr):
                switch = getattr(moe_module, attr)
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    if hasattr(switch, proj):
                        delattr(switch, proj)
                delattr(moe_module, attr)
                break

        # Replace the entire mlp module
        layer.mlp = replacement
        replaced += 1

    gc.collect()
    mx.clear_cache()
    logger.info("Replaced %d MoE layers with FlashMoE", replaced)
