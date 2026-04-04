"""TurboQuant KV cache: drop-in replacement for mlx-lm's KVCache.

Stores bit-packed quantized key/value vectors using TurboQuant_mse and
dequantizes on fetch, providing transparent memory compression.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from mlx_lm.models.cache import KVCache, _BaseCache, create_attention_mask

from olmlx.engine.turboquant import (
    TurboQuantRotation,
    turboquant_dequantize,
    turboquant_quantize,
)

logger = logging.getLogger(__name__)


class TurboQuantKVCache(_BaseCache):
    """KV cache with TurboQuant compression.

    Stores only bit-packed indices and float32 norms. Dequantizes the
    full cache on each ``update_and_fetch`` call — O(n · head_dim²) per
    step due to the rotation matrix multiply.
    """

    step = 256

    def __init__(
        self,
        bits: int,
        rotation_key: TurboQuantRotation,
        rotation_value: TurboQuantRotation,
    ):
        self._bits = bits
        self.rotation_key = rotation_key
        self.rotation_value = rotation_value
        self._key_indices: mx.array | None = None
        self._key_norms: mx.array | None = None
        self._value_indices: mx.array | None = None
        self._value_norms: mx.array | None = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Quantize new K/V, append to store, dequantize all and return."""
        B, n_heads, num_steps, head_dim = keys.shape
        input_dtype = keys.dtype
        prev = self.offset

        # Quantize incoming tokens (returns bit-packed indices)
        k_idx, k_nrm = turboquant_quantize(keys, self.rotation_key, self._bits)
        v_idx, v_nrm = turboquant_quantize(values, self.rotation_value, self._bits)

        packed_dim = k_idx.shape[-1]  # head_dim // (8 // bits)

        # Allocate or expand buffers
        if self._key_indices is None or (prev + num_steps) > self._key_indices.shape[2]:
            new_steps = (num_steps + self.step - 1) // self.step * self.step
            idx_shape = (B, n_heads, new_steps, packed_dim)
            nrm_shape = (B, n_heads, new_steps, 1)

            if self._key_indices is not None:
                if prev % self.step != 0:
                    self._key_indices = self._key_indices[..., :prev, :]
                    self._key_norms = self._key_norms[..., :prev, :]
                    self._value_indices = self._value_indices[..., :prev, :]
                    self._value_norms = self._value_norms[..., :prev, :]
                self._key_indices = mx.concatenate(
                    [self._key_indices, mx.zeros(idx_shape, dtype=mx.uint8)], axis=2
                )
                self._key_norms = mx.concatenate(
                    [self._key_norms, mx.zeros(nrm_shape, dtype=mx.float32)], axis=2
                )
                self._value_indices = mx.concatenate(
                    [self._value_indices, mx.zeros(idx_shape, dtype=mx.uint8)], axis=2
                )
                self._value_norms = mx.concatenate(
                    [self._value_norms, mx.zeros(nrm_shape, dtype=mx.float32)], axis=2
                )
            else:
                self._key_indices = mx.zeros(idx_shape, dtype=mx.uint8)
                self._key_norms = mx.zeros(nrm_shape, dtype=mx.float32)
                self._value_indices = mx.zeros(idx_shape, dtype=mx.uint8)
                self._value_norms = mx.zeros(nrm_shape, dtype=mx.float32)

        # Store quantized data
        self.offset += num_steps
        self._key_indices[..., prev : self.offset, :] = k_idx
        self._key_norms[..., prev : self.offset, :] = k_nrm
        self._value_indices[..., prev : self.offset, :] = v_idx
        self._value_norms[..., prev : self.offset, :] = v_nrm

        # Dequantize full cache and return in the original dtype
        k_out = turboquant_dequantize(
            self._key_indices[..., : self.offset, :],
            self._key_norms[..., : self.offset, :],
            self.rotation_key,
            self._bits,
            dtype=input_dtype,
        )
        v_out = turboquant_dequantize(
            self._value_indices[..., : self.offset, :],
            self._value_norms[..., : self.offset, :],
            self.rotation_value,
            self._bits,
            dtype=input_dtype,
        )
        return k_out, v_out

    @property
    def state(self):
        if self._key_indices is None:
            return []
        return [
            self._key_indices[..., : self.offset, :],
            self._key_norms[..., : self.offset, :],
            self._value_indices[..., : self.offset, :],
            self._value_norms[..., : self.offset, :],
        ]

    @state.setter
    def state(self, v):
        # Should be unreachable: _is_serializable_cache() guards all save paths.
        # Raised here as a hard stop in case that guard is bypassed.
        raise NotImplementedError(
            "TurboQuantKVCache does not support state restoration. "
            "Disable disk cache offload when using TurboQuant."
        )

    def is_trimmable(self):
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        if self.offset == 0:
            self._key_indices = None
            self._key_norms = None
            self._value_indices = None
            self._value_norms = None
        return n

    def make_mask(self, *args, **kwargs):
        kwargs["offset"] = self.offset
        return create_attention_mask(*args, **kwargs)

    def empty(self):
        return self._key_indices is None or self.offset == 0


def _detect_head_dim(model: Any) -> int:
    """Detect head_dim from model args/config or K projection layer shape.

    Handles models where head_dim != hidden_size // num_attention_heads
    (e.g. Gemma 3, Phi-3/4).  Falls back to model.config when model.args
    is missing (e.g. mlx-vlm gemma4 LanguageModel).
    """
    # Prefer explicit head_dim from model args or config
    model_cfg = getattr(model, "args", None) or getattr(model, "config", None)
    if model_cfg is None:
        raise RuntimeError("TurboQuant: model has no 'args' or 'config' attribute")

    head_dim = getattr(model_cfg, "head_dim", None)
    if head_dim is not None:
        return head_dim

    # Derive from K projection weight shape: k_proj.weight is (n_kv_heads * head_dim, hidden_size)
    try:
        layer = model.layers[0]
        k_proj = layer.self_attn.k_proj
        weight = k_proj.weight
        if isinstance(weight, mx.array):
            kv_out_dim = weight.shape[0]
            n_kv_heads = getattr(model_cfg, "num_key_value_heads", None)
            if n_kv_heads:
                return kv_out_dim // n_kv_heads
    except (AttributeError, IndexError):
        pass

    # Last resort: hidden_size // num_attention_heads
    try:
        return model_cfg.hidden_size // model_cfg.num_attention_heads
    except AttributeError as e:
        raise RuntimeError(
            f"TurboQuant: cannot detect head_dim for this model architecture. "
            f"Missing attribute: {e}"
        ) from e


def make_turboquant_cache(model: Any, bits: int) -> list:
    """Create a cache list with TurboQuantKVCache for attention layers.

    For hybrid models (e.g. Nemotron-H with SSM + attention layers), only
    attention-layer caches (KVCache) are replaced with TurboQuantKVCache.
    Non-attention caches (e.g. ArraysCache for SSM/Mamba layers) are preserved.
    """
    num_layers = len(model.layers)
    head_dim = _detect_head_dim(model)

    packing_factor = 8 // bits
    if head_dim % packing_factor != 0:
        raise ValueError(
            f"TurboQuant {bits}-bit requires head_dim divisible by {packing_factor}, "
            f"got head_dim={head_dim}"
        )

    # Get default cache layout from model if available (hybrid models
    # return different cache types per layer, e.g. ArraysCache for SSM)
    if hasattr(model, "make_cache"):
        default_caches = model.make_cache()
        if not isinstance(default_caches, list):
            default_caches = [None] * num_layers
    else:
        default_caches = [None] * num_layers

    caches = []
    tq_count = 0
    for i, default in enumerate(default_caches):
        if default is None or isinstance(default, KVCache):
            # Detect per-layer head dim from K projection weight shape.
            # Models like gemma4 have different head dims for full vs sliding
            # attention layers (global_head_dim=512 vs head_dim=256).
            layer_head_dim = head_dim
            try:
                attn = model.layers[i].self_attn
                k_weight = attn.k_proj.weight
                if isinstance(k_weight, mx.array):
                    # Prefer the layer's own n_kv_heads (handles gemma4 where
                    # full attention uses num_global_key_value_heads ≠ num_key_value_heads)
                    n_kv = getattr(attn, "n_kv_heads", None)
                    if n_kv is None:
                        model_cfg = getattr(model, "args", None) or getattr(
                            model, "config", None
                        )
                        n_kv = getattr(model_cfg, "num_key_value_heads", None)
                    if n_kv:
                        layer_head_dim = k_weight.shape[0] // n_kv
            except (AttributeError, IndexError):
                pass

            if layer_head_dim % (8 // bits) != 0:
                # Not compatible with TurboQuant packing — keep default cache
                caches.append(default if default is not None else KVCache())
                continue

            rot_k = TurboQuantRotation(head_dim=layer_head_dim, seed=i * 2)
            rot_v = TurboQuantRotation(head_dim=layer_head_dim, seed=i * 2 + 1)
            caches.append(
                TurboQuantKVCache(bits=bits, rotation_key=rot_k, rotation_value=rot_v)
            )
            tq_count += 1
        else:
            caches.append(default)

    logger.info(
        "Created TurboQuant KV cache: %d/%d cache entries quantized, %d-bit, head_dim=%d",
        tq_count,
        len(caches),
        bits,
        head_dim,
    )
    return caches
