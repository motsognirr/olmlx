"""Model-specific adapters for dflash speculative decoding.

Each adapter knows how to:
- Extract intermediate hidden states from the target model
- Snapshot and restore KV caches for rollback
- Trim caches after rejected tokens
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


class TargetAdapter(ABC):
    """Abstract interface for model-specific target integration."""

    @abstractmethod
    def forward_with_hidden(
        self,
        model: Any,
        tokens: mx.array,
        cache: Any,
        target_layer_ids: list[int],
    ) -> tuple[mx.array, dict[int, mx.array], Any]:
        """Forward pass that also captures intermediate hidden states.

        Args:
            model: The target model (or its inner transformer).
            tokens: (1, seq_len) input token IDs.
            cache: KV cache (or None for first call).
            target_layer_ids: Which layer indices to capture.

        Returns:
            (logits, hidden_states, cache) where hidden_states maps
            layer_idx -> (1, seq_len, hidden_size).
        """

    @abstractmethod
    def trim_cache(self, cache: list, num_tokens: int) -> None:
        """Remove the last num_tokens from each layer's cache."""


class Qwen3Adapter(TargetAdapter):
    """Adapter for Qwen3 models (standard multi-head attention)."""

    def forward_with_hidden(
        self,
        model: Any,
        tokens: mx.array,
        cache: Any,
        target_layer_ids: list[int],
    ) -> tuple[mx.array, dict[int, mx.array], Any]:
        inner = getattr(model, "model", model)
        layers = inner.layers
        norm = getattr(inner, "norm", None)
        lm_head = getattr(model, "lm_head", None)

        embed = getattr(inner, "embed_tokens", None) or getattr(inner, "embed", None)
        h = embed(tokens)

        captured: dict[int, mx.array] = {}
        target_set = set(target_layer_ids)

        for i, layer in enumerate(layers):
            h = layer(h, cache=cache[i] if cache is not None else None)
            if i in target_set:
                captured[i] = h

        if norm is not None:
            h = norm(h)

        logits = lm_head(h) if lm_head is not None else h
        return logits, captured, cache

    def trim_cache(self, cache: list, num_tokens: int) -> None:
        """Trim last num_tokens from cache using mlx-lm's trim_prompt_cache."""
        try:
            from mlx_lm.models.cache import trim_prompt_cache

            trim_prompt_cache(cache, num_tokens)
        except ImportError:
            for layer_cache in cache:
                layer_cache.offset = max(0, layer_cache.offset - num_tokens)


# Registry
ADAPTERS: dict[str, type[TargetAdapter]] = {
    "qwen3": Qwen3Adapter,
    "qwen2": Qwen3Adapter,  # same attention structure
}


def get_adapter(model_type: str) -> TargetAdapter:
    """Look up and instantiate an adapter by model type.

    Raises KeyError if no adapter is registered for the given type.
    """
    if model_type not in ADAPTERS:
        raise KeyError(
            f"No dflash adapter for model_type={model_type!r}. "
            f"Available: {list(ADAPTERS.keys())}"
        )
    return ADAPTERS[model_type]()
