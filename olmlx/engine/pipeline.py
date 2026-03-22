"""Pipeline parallelism for distributed inference across Apple Silicon machines.

Assigns different layers to different ranks, allowing uneven memory splits
(e.g., 44 layers on 64GB machine, 20 layers on 24GB machine).

Following the DeepSeek V3.2 convention: rank 0 = last layers (closest to
output), highest rank = first layers (closest to input).
"""

from __future__ import annotations

import logging
import types
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _compute_layer_counts(total_layers: int, world_size: int) -> list[int]:
    """Compute default even layer split across ranks.

    Extra layers go to rank 0 (the output-side rank).
    """
    base = total_layers // world_size
    extra = total_layers % world_size
    counts = [base] * world_size
    counts[0] += extra
    return counts


def _compute_layer_range(rank: int, layer_counts: list[int]) -> tuple[int, int]:
    """Compute (start_idx, end_idx) for a given rank.

    Rank 0 gets the last layers, highest rank gets the first layers.
    """
    # Sum of layers for all ranks with higher rank number (= earlier layers)
    start_idx = sum(layer_counts[r] for r in range(rank + 1, len(layer_counts)))
    end_idx = start_idx + layer_counts[rank]
    return start_idx, end_idx


def _is_gpt_oss(inner: Any) -> bool:
    """Detect if the inner model uses gpt_oss-style layer_types attention."""
    return (
        hasattr(inner, "layer_types")
        and hasattr(inner, "window_size")
        and hasattr(inner, "ga_idx")
        and hasattr(inner, "swa_idx")
    )


def _is_llama_sliding_window(inner: Any) -> bool:
    """Detect if the inner model uses Llama-style sliding window attention."""
    return (
        hasattr(inner, "sliding_window")
        and inner.sliding_window is not None
        and hasattr(inner, "swa_idx")
    )


def _validate_inner_model(model: Any) -> Any:
    """Validate and return the inner model with standard transformer structure.

    Raises ValueError if the model doesn't have embed_tokens, layers, norm.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise ValueError(
            f"Model {type(model).__name__} does not have a standard inner model "
            f"(.model attribute). Pipeline parallelism requires models with "
            f"embed_tokens -> layers -> norm structure."
        )
    for attr in ("embed_tokens", "layers", "norm"):
        if not hasattr(inner, attr):
            raise ValueError(
                f"Model {type(model).__name__} does not have a standard "
                f"transformer structure (missing inner.{attr}). "
                f"Pipeline parallelism requires embed_tokens -> layers -> norm."
            )
    return inner


def apply_pipeline(
    model: Any,
    group: Any,
    layer_counts: list[int] | None = None,
    pre_sharded: bool = False,
) -> None:
    """Apply pipeline parallelism to a model loaded by mlx_lm.load().

    Monkey-patches the inner model's __call__ with pipeline-aware send/recv/
    all_gather logic. Nullifies non-owned layers to save memory.

    Args:
        model: The outer Model returned by mlx_lm.load().
        group: MLX distributed group.
        layer_counts: Number of layers per rank. If None, splits evenly.
            Must sum to total layer count. Length must equal world_size.
        pre_sharded: If True, the model already contains only owned layers
            (renumbered 0-based). Skips validation against total layer count,
            skips nullification, and uses start_idx=0.
    """
    inner = _validate_inner_model(model)
    if hasattr(inner, "pipeline_rank"):
        raise RuntimeError(
            f"apply_pipeline() already applied to {type(model).__name__}"
        )
    rank = group.rank()
    world_size = group.size()

    if pre_sharded:
        # Model already has only owned layers, renumbered 0-based
        owned_count = len(inner.layers)

        if layer_counts is None:
            raise ValueError("layer_counts is required when pre_sharded=True")
        if len(layer_counts) != world_size:
            raise ValueError(
                f"layer_counts must have {world_size} entries (one per rank), "
                f"got {len(layer_counts)}"
            )
        expected = layer_counts[rank]
        if owned_count != expected:
            raise ValueError(
                f"Pre-sharded model has {owned_count} layers but "
                f"layer_counts[{rank}]={expected}; shard may be stale or incorrect"
            )

        inner.pipeline_rank = rank
        inner.pipeline_size = world_size
        inner.start_idx = 0
        inner.end_idx = owned_count
        inner.num_layers = owned_count
    else:
        total_layers = len(inner.layers)

        if layer_counts is None:
            layer_counts = _compute_layer_counts(total_layers, world_size)

        if len(layer_counts) != world_size:
            raise ValueError(
                f"layer_counts must have {world_size} entries (one per rank), "
                f"got {len(layer_counts)}"
            )
        if sum(layer_counts) != total_layers:
            raise ValueError(
                f"layer_counts must sum to {total_layers} (total layers), "
                f"got {sum(layer_counts)}"
            )

        start_idx, end_idx = _compute_layer_range(rank, layer_counts)

        # Set pipeline state on inner model
        inner.pipeline_rank = rank
        inner.pipeline_size = world_size
        inner.start_idx = start_idx
        inner.end_idx = end_idx
        inner.num_layers = end_idx - start_idx

        # Nullify non-owned layers and truncate.
        # Note: embed_tokens, norm, and lm_head (on outer model) remain on all ranks.
        # norm is needed because all_gather broadcasts the final hidden state to all
        # ranks, each of which must independently produce valid output for the compute
        # graph. This matches the DeepSeek V3.2 reference implementation.
        inner.layers = inner.layers[:end_idx]
        inner.layers[:start_idx] = [None] * start_idx

    # Replace inner model's __call__ with pipeline-aware version
    if _is_gpt_oss(inner):
        _patch_gpt_oss_call(inner)
    elif _is_llama_sliding_window(inner):
        _patch_llama_call(inner)
    else:
        _patch_standard_call(inner)

    # Patch outer model's layers property to return only owned layers
    # (needed for correct KV cache creation via make_prompt_cache)
    _patch_outer_layers(model, inner)

    logger.info(
        "Pipeline parallelism applied: rank %d/%d, layers %d-%d (%d owned%s)",
        rank,
        world_size,
        inner.start_idx,
        inner.end_idx - 1,
        inner.num_layers,
        ", pre-sharded" if pre_sharded else "",
    )


def _patch_standard_call(inner: Any) -> None:
    """Replace __call__ for standard models (Qwen3, Mistral, etc.)."""
    from mlx_lm.models.base import create_attention_mask

    def pipeline_call(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        elif self.pipeline_rank == self.pipeline_size - 1:
            # Input rank: compute embeddings
            h = self.embed_tokens(inputs)
        else:
            # Non-input ranks: create zero shape template for recv_like
            # (avoids wasteful embed_tokens forward pass)
            h = mx.zeros(
                (inputs.shape[0], inputs.shape[1], self.embed_tokens.weight.shape[-1]),
                dtype=self.embed_tokens.weight.dtype,
            )

        if cache is None:
            cache = [None] * self.num_layers

        mask = create_attention_mask(h, cache[0])

        # Receive from the previous process in the pipeline (higher rank = earlier layers)
        if self.pipeline_rank < self.pipeline_size - 1:
            h = mx.distributed.recv_like(h, self.pipeline_rank + 1)

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i])

        # Send to the next process in the pipeline (lower rank = later layers)
        if self.pipeline_rank != 0:
            h = mx.distributed.send(h, (self.pipeline_rank - 1) % self.pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast result to all ranks so each can independently compute
        # norm + lm_head (needed to keep the compute graph valid on all ranks).
        # Relies on all_gather returning tensors in rank order — this matches
        # the DeepSeek V3.2 reference implementation (deepseek_v32.py:474-476)
        # and is standard MPI semantics for all_gather.
        if self.pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)

    inner.__call__ = types.MethodType(pipeline_call, inner)


def _patch_gpt_oss_call(inner: Any) -> None:
    """Replace __call__ for gpt_oss models with layer_types-based dual masks."""
    from mlx_lm.models.base import create_attention_mask

    # Store the owned slice of layer_types for the iteration loop
    owned_layer_types = inner.layer_types[inner.start_idx : inner.end_idx]
    inner._owned_layer_types = owned_layer_types

    # Precompute FA/SWA cache indices from owned layer_types
    fa_cache_idx = None
    swa_cache_idx = None
    for i, lt in enumerate(owned_layer_types):
        if lt == "full_attention":
            if fa_cache_idx is None:
                fa_cache_idx = i
        else:
            if swa_cache_idx is None:
                swa_cache_idx = i

    inner._fa_cache_idx = fa_cache_idx
    inner._swa_cache_idx = swa_cache_idx

    def pipeline_call(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        elif self.pipeline_rank == self.pipeline_size - 1:
            h = self.embed_tokens(inputs)
        else:
            h = mx.zeros(
                (inputs.shape[0], inputs.shape[1], self.embed_tokens.weight.shape[-1]),
                dtype=self.embed_tokens.weight.dtype,
            )

        if cache is None:
            cache = [None] * self.num_layers

        fa_mask = create_attention_mask(
            h, cache[self._fa_cache_idx] if self._fa_cache_idx is not None else None
        )
        if self._swa_cache_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self._swa_cache_idx], window_size=self.window_size
            )
        else:
            swa_mask = fa_mask

        if self.pipeline_rank < self.pipeline_size - 1:
            h = mx.distributed.recv_like(h, self.pipeline_rank + 1)

        for i in range(self.num_layers):
            layer = self.layers[self.start_idx + i]
            mask = (
                fa_mask if self._owned_layer_types[i] == "full_attention" else swa_mask
            )
            h = layer(h, mask, cache[i])

        if self.pipeline_rank != 0:
            h = mx.distributed.send(h, (self.pipeline_rank - 1) % self.pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        if self.pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)

    inner.__call__ = types.MethodType(pipeline_call, inner)


def _patch_llama_call(inner: Any) -> None:
    """Replace __call__ for Llama models with sliding window attention."""
    from mlx_lm.models.base import create_attention_mask

    # Precompute FA/SWA cache indices once to avoid per-token linear scan
    fa_cache_idx = None
    swa_cache_idx = None
    for i in range(inner.num_layers):
        layer = inner.layers[inner.start_idx + i]
        if layer.use_sliding:
            if swa_cache_idx is None:
                swa_cache_idx = i
        else:
            if fa_cache_idx is None:
                fa_cache_idx = i

    inner._fa_cache_idx = fa_cache_idx
    inner._swa_cache_idx = swa_cache_idx

    def pipeline_call(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        elif self.pipeline_rank == self.pipeline_size - 1:
            h = self.embed_tokens(inputs)
        else:
            h = mx.zeros(
                (inputs.shape[0], inputs.shape[1], self.embed_tokens.weight.shape[-1]),
                dtype=self.embed_tokens.weight.dtype,
            )

        if cache is None:
            cache = [None] * self.num_layers

        fa_mask = create_attention_mask(
            h, cache[self._fa_cache_idx] if self._fa_cache_idx is not None else None
        )
        if self._swa_cache_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self._swa_cache_idx], window_size=self.sliding_window
            )
        else:
            swa_mask = fa_mask

        if self.pipeline_rank < self.pipeline_size - 1:
            h = mx.distributed.recv_like(h, self.pipeline_rank + 1)

        for i in range(self.num_layers):
            layer = self.layers[self.start_idx + i]
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, cache=cache[i])

        if self.pipeline_rank != 0:
            h = mx.distributed.send(h, (self.pipeline_rank - 1) % self.pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        if self.pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)

    inner.__call__ = types.MethodType(pipeline_call, inner)


_pipeline_model_cache: dict[type, type] = {}


def _patch_outer_layers(model: Any, inner: Any) -> None:
    """Patch the outer model so model.layers returns only owned layers.

    This is needed for make_prompt_cache to create the correct number of
    KV cache entries.
    """
    owned_layers = inner.layers[inner.start_idx : inner.end_idx]

    if isinstance(model, nn.Module):
        model_cls = model.__class__
        if model_cls not in _pipeline_model_cache:

            class PipelineModel(model_cls):
                @property
                def layers(self):
                    return self.model.layers[self.model.start_idx : self.model.end_idx]

                @layers.setter
                def layers(self, value):
                    raise AttributeError(
                        "model.layers is managed by pipeline parallelism; "
                        "assign to model.model.layers instead"
                    )

            _pipeline_model_cache[model_cls] = PipelineModel

        model.__class__ = _pipeline_model_cache[model_cls]
    else:
        # For non-nn.Module (e.g. tests), just set the attribute directly
        model.layers = owned_layers
