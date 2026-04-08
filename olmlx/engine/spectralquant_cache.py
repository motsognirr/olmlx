"""SpectralQuant KV cache: drop-in replacement for mlx-lm's KVCache.

Stores bit-packed quantized key/value vectors using spectral rotation
and non-uniform quantization. Dequantizes on fetch for transparent
memory compression.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_lm.models.cache import KVCache, _BaseCache, create_attention_mask

from olmlx.engine.spectralquant import (
    SpectralRotation,
    spectral_dequantize,
    spectral_quantize,
)

logger = logging.getLogger(__name__)


class SpectralQuantKVCache(_BaseCache):
    """KV cache with SpectralQuant compression.

    Stores bit-packed indices for semantic and tail regimes plus float32
    norms. Dequantizes the full cache on each ``update_and_fetch`` call.
    """

    step = 256

    def __init__(
        self,
        rotation_key: SpectralRotation,
        rotation_value: SpectralRotation,
        codebook_sem_key: mx.array,
        codebook_tail_key: mx.array,
        codebook_sem_value: mx.array,
        codebook_tail_value: mx.array,
        d_eff: int,
        bits_high: int,
        bits_low: int,
    ):
        self.rotation_key = rotation_key
        self.rotation_value = rotation_value
        self.codebook_sem_key = codebook_sem_key
        self.codebook_tail_key = codebook_tail_key
        self.codebook_sem_value = codebook_sem_value
        self.codebook_tail_value = codebook_tail_value
        self.d_eff = d_eff
        self.bits_high = bits_high
        self.bits_low = bits_low

        # Buffers for keys
        self._k_sem: mx.array | None = None
        self._k_tail: mx.array | None = None
        self._k_norms: mx.array | None = None
        # Buffers for values
        self._v_sem: mx.array | None = None
        self._v_tail: mx.array | None = None
        self._v_norms: mx.array | None = None
        self.offset = 0

    def _packed_dims(self, head_dim: int) -> tuple[int, int]:
        """Compute packed dimensions for semantic and tail regimes."""
        d_tail = head_dim - self.d_eff

        def _packed(dim, bits):
            if bits == 8:
                return dim
            if bits == 1:
                return (dim + 7) // 8
            factor = 8 // bits
            return dim // factor

        return _packed(self.d_eff, self.bits_high), _packed(d_tail, self.bits_low)

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Quantize new K/V, append to store, dequantize all and return."""
        B, n_heads, num_steps, head_dim = keys.shape
        input_dtype = keys.dtype
        prev = self.offset

        # Quantize incoming tokens
        k_sem, k_tail, k_nrm = spectral_quantize(
            keys,
            self.rotation_key,
            self.codebook_sem_key,
            self.codebook_tail_key,
            self.d_eff,
            self.bits_high,
            self.bits_low,
        )
        v_sem, v_tail, v_nrm = spectral_quantize(
            values,
            self.rotation_value,
            self.codebook_sem_value,
            self.codebook_tail_value,
            self.d_eff,
            self.bits_high,
            self.bits_low,
        )

        sem_packed_dim, tail_packed_dim = self._packed_dims(head_dim)

        # Allocate or expand buffers
        if self._k_sem is None or (prev + num_steps) > self._k_sem.shape[2]:
            new_steps = (num_steps + self.step - 1) // self.step * self.step
            sem_shape = (B, n_heads, new_steps, sem_packed_dim)
            tail_shape = (B, n_heads, new_steps, tail_packed_dim)
            nrm_shape = (B, n_heads, new_steps, 1)

            if self._k_sem is not None:
                if prev % self.step != 0:
                    self._k_sem = self._k_sem[..., :prev, :]
                    self._k_tail = self._k_tail[..., :prev, :]
                    self._k_norms = self._k_norms[..., :prev, :]
                    self._v_sem = self._v_sem[..., :prev, :]
                    self._v_tail = self._v_tail[..., :prev, :]
                    self._v_norms = self._v_norms[..., :prev, :]
                self._k_sem = mx.concatenate(
                    [self._k_sem, mx.zeros(sem_shape, dtype=mx.uint8)], axis=2
                )
                self._k_tail = mx.concatenate(
                    [self._k_tail, mx.zeros(tail_shape, dtype=mx.uint8)], axis=2
                )
                self._k_norms = mx.concatenate(
                    [self._k_norms, mx.zeros(nrm_shape, dtype=mx.float32)], axis=2
                )
                self._v_sem = mx.concatenate(
                    [self._v_sem, mx.zeros(sem_shape, dtype=mx.uint8)], axis=2
                )
                self._v_tail = mx.concatenate(
                    [self._v_tail, mx.zeros(tail_shape, dtype=mx.uint8)], axis=2
                )
                self._v_norms = mx.concatenate(
                    [self._v_norms, mx.zeros(nrm_shape, dtype=mx.float32)], axis=2
                )
            else:
                self._k_sem = mx.zeros(sem_shape, dtype=mx.uint8)
                self._k_tail = mx.zeros(tail_shape, dtype=mx.uint8)
                self._k_norms = mx.zeros(nrm_shape, dtype=mx.float32)
                self._v_sem = mx.zeros(sem_shape, dtype=mx.uint8)
                self._v_tail = mx.zeros(tail_shape, dtype=mx.uint8)
                self._v_norms = mx.zeros(nrm_shape, dtype=mx.float32)

        # Store quantized data
        self.offset += num_steps
        self._k_sem[..., prev : self.offset, :] = k_sem
        self._k_tail[..., prev : self.offset, :] = k_tail
        self._k_norms[..., prev : self.offset, :] = k_nrm
        self._v_sem[..., prev : self.offset, :] = v_sem
        self._v_tail[..., prev : self.offset, :] = v_tail
        self._v_norms[..., prev : self.offset, :] = v_nrm

        # Dequantize full cache
        k_out = spectral_dequantize(
            self._k_sem[..., : self.offset, :],
            self._k_tail[..., : self.offset, :],
            self._k_norms[..., : self.offset, :],
            self.rotation_key,
            self.codebook_sem_key,
            self.codebook_tail_key,
            self.d_eff,
            self.bits_high,
            self.bits_low,
            dtype=input_dtype,
        )
        v_out = spectral_dequantize(
            self._v_sem[..., : self.offset, :],
            self._v_tail[..., : self.offset, :],
            self._v_norms[..., : self.offset, :],
            self.rotation_value,
            self.codebook_sem_value,
            self.codebook_tail_value,
            self.d_eff,
            self.bits_high,
            self.bits_low,
            dtype=input_dtype,
        )
        return k_out, v_out

    @property
    def state(self):
        if self._k_sem is None:
            return []
        return [
            self._k_sem[..., : self.offset, :],
            self._k_tail[..., : self.offset, :],
            self._k_norms[..., : self.offset, :],
            self._v_sem[..., : self.offset, :],
            self._v_tail[..., : self.offset, :],
            self._v_norms[..., : self.offset, :],
        ]

    @state.setter
    def state(self, v):
        raise NotImplementedError(
            "SpectralQuantKVCache does not support state restoration. "
            "Disable disk cache offload when using SpectralQuant."
        )

    def is_trimmable(self):
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        if self.offset == 0:
            self._k_sem = None
            self._k_tail = None
            self._k_norms = None
            self._v_sem = None
            self._v_tail = None
            self._v_norms = None
        return n

    def make_mask(self, *args, **kwargs):
        kwargs["offset"] = self.offset
        return create_attention_mask(*args, **kwargs)

    def empty(self):
        return self._k_sem is None or self.offset == 0


def make_spectral_cache(
    model: Any,
    calibration_dir: Path,
    avg_bits: int = 4,
) -> list:
    """Create a cache list with SpectralQuantKVCache for attention layers.

    Loads calibration data from disk and creates per-layer caches.

    Args:
        model: The loaded MLX model.
        calibration_dir: Path to spectral calibration directory.
        avg_bits: Target average bits (used to select calibration data).

    Returns:
        List of cache objects (SpectralQuantKVCache for attention layers,
        preserved for non-attention layers).
    """
    from olmlx.engine.spectralquant_calibrate import load_calibration
    from olmlx.engine.turboquant_cache import _detect_head_dim

    calibration_dir = Path(calibration_dir)
    calibration = load_calibration(calibration_dir)

    num_layers = len(model.layers)
    head_dim = _detect_head_dim(model)

    # Determine n_kv_heads
    model_cfg = getattr(model, "args", None) or getattr(model, "config", None)
    n_kv_heads = getattr(model_cfg, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = getattr(model_cfg, "num_attention_heads", 1)

    # Get default cache layout (hybrid model support)
    if hasattr(model, "make_cache"):
        default_caches = model.make_cache()
        if not isinstance(default_caches, list):
            default_caches = [None] * num_layers
    else:
        default_caches = [None] * num_layers

    caches = []
    sq_count = 0

    for i, default in enumerate(default_caches):
        if default is not None and not isinstance(default, KVCache):
            # Non-attention cache (e.g. ArraysCache for SSM layers)
            caches.append(default)
            continue

        # Check if calibration data exists for this layer
        # Use head 0 to check — all heads should be calibrated together
        key_cal = calibration.get((i, 0, "key"))
        val_cal = calibration.get((i, 0, "value"))
        if key_cal is None or val_cal is None:
            logger.warning(
                "No spectral calibration for layer %d, using default cache", i
            )
            caches.append(default if default is not None else KVCache())
            continue

        # For simplicity, use head 0's calibration for the layer cache
        # (the cache operates on all heads simultaneously, so we use
        # per-layer calibration rather than per-head)
        rotation_k = SpectralRotation(key_cal["eigenvectors"])
        rotation_v = SpectralRotation(val_cal["eigenvectors"])

        cache = SpectralQuantKVCache(
            rotation_key=rotation_k,
            rotation_value=rotation_v,
            codebook_sem_key=key_cal["codebook_sem"],
            codebook_tail_key=key_cal["codebook_tail"],
            codebook_sem_value=val_cal["codebook_sem"],
            codebook_tail_value=val_cal["codebook_tail"],
            d_eff=key_cal["d_eff"],
            bits_high=key_cal["bits_high"],
            bits_low=key_cal["bits_low"],
        )
        caches.append(cache)
        sq_count += 1

    logger.info(
        "Created SpectralQuant KV cache: %d/%d layers quantized, head_dim=%d",
        sq_count,
        len(caches),
        head_dim,
    )
    return caches
