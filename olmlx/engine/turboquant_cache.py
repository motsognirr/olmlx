"""TurboQuant KV cache: drop-in replacement for mlx-lm's KVCache.

Stores quantized key/value vectors using TurboQuant_mse and dequantizes
on fetch, providing transparent memory compression.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from mlx_lm.models.cache import _BaseCache, create_attention_mask

from olmlx.engine.turboquant import (
    TurboQuantRotation,
    turboquant_dequantize,
    turboquant_quantize,
)

logger = logging.getLogger(__name__)


class TurboQuantKVCache(_BaseCache):
    """KV cache with TurboQuant compression.

    Quantizes K/V vectors on store and dequantizes on fetch.
    Implements the same interface as mlx-lm's KVCache.
    """

    step = 256

    def __init__(
        self,
        bits: int,
        rotation_key: TurboQuantRotation,
        rotation_value: TurboQuantRotation,
    ):
        self.bits = bits
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
        prev = self.offset

        # Quantize incoming tokens
        k_idx, k_nrm = turboquant_quantize(keys, self.rotation_key, self.bits)
        v_idx, v_nrm = turboquant_quantize(values, self.rotation_value, self.bits)

        # Allocate or expand buffers
        if self._key_indices is None or (prev + num_steps) > self._key_indices.shape[2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            idx_shape = (B, n_heads, new_steps, head_dim)
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
                    [self._key_norms, mx.zeros(nrm_shape, dtype=mx.float16)], axis=2
                )
                self._value_indices = mx.concatenate(
                    [self._value_indices, mx.zeros(idx_shape, dtype=mx.uint8)], axis=2
                )
                self._value_norms = mx.concatenate(
                    [self._value_norms, mx.zeros(nrm_shape, dtype=mx.float16)], axis=2
                )
            else:
                self._key_indices = mx.zeros(idx_shape, dtype=mx.uint8)
                self._key_norms = mx.zeros(nrm_shape, dtype=mx.float16)
                self._value_indices = mx.zeros(idx_shape, dtype=mx.uint8)
                self._value_norms = mx.zeros(nrm_shape, dtype=mx.float16)

        # Store quantized data
        self.offset += num_steps
        self._key_indices[..., prev : self.offset, :] = k_idx
        self._key_norms[..., prev : self.offset, :] = k_nrm
        self._value_indices[..., prev : self.offset, :] = v_idx
        self._value_norms[..., prev : self.offset, :] = v_nrm

        # Dequantize full cache and return
        k_out = turboquant_dequantize(
            self._key_indices[..., : self.offset, :],
            self._key_norms[..., : self.offset, :],
            self.rotation_key,
            self.bits,
        )
        v_out = turboquant_dequantize(
            self._value_indices[..., : self.offset, :],
            self._value_norms[..., : self.offset, :],
            self.rotation_value,
            self.bits,
        )
        return k_out, v_out

    def is_trimmable(self):
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self._key_indices is None


def make_turboquant_cache(model: Any, bits: int) -> list[TurboQuantKVCache]:
    """Create a list of TurboQuantKVCache objects, one per model layer."""
    num_layers = len(model.layers)

    try:
        head_dim = model.args.head_dim
    except AttributeError:
        head_dim = model.args.hidden_size // model.args.num_attention_heads

    caches = []
    for i in range(num_layers):
        rot_k = TurboQuantRotation(head_dim=head_dim, seed=i * 2)
        rot_v = TurboQuantRotation(head_dim=head_dim, seed=i * 2 + 1)
        caches.append(TurboQuantKVCache(bits=bits, rotation_key=rot_k, rotation_value=rot_v))

    logger.info(
        "Created TurboQuant KV cache: %d layers, %d-bit, head_dim=%d",
        num_layers,
        bits,
        head_dim,
    )
    return caches
