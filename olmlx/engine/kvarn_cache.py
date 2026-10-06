"""KVarN KV cache (#748): ``TurboQuantKVCache`` with the KVarN codec.

Subclassing (rather than a parallel implementation) is deliberate: the
step-aligned buffer management, dequant side buffer, ``trim``,
``__deepcopy__`` (``mx.Dtype`` sharing + eager side-buffer eval),
``release_dequant_buffers`` (shed-on-store), ``ensure_state_materialized`` /
``_pin_state_to_offset`` (cross-thread packed-buffer materialization) are the
surface that needed #653/#654/#655/#657 to survive mlx thread-local streams,
and CI runs on CPU so a re-implementation could regress them invisibly. Only
the codec and the per-K/V bit widths differ.

The ``_key_norms`` / ``_value_norms`` buffers hold KVarN's per-tile
``[means..., scales...]`` (last dim ``2 * n_tiles``) instead of a norm.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from olmlx.engine.kvarn import (
    KVarNRotation,
    choose_tile,
    kvarn_dequantize,
    kvarn_encode,
    kvarn_quantize,
)
from olmlx.engine.turboquant_cache import (
    TurboQuantKVCache,
    _detect_head_dim,
    build_kv_quant_caches,
    memoized_rotation,
)

logger = logging.getLogger(__name__)


class KVarNKVCache(TurboQuantKVCache):
    """Variance-normalized, Hadamard-rotated KV cache with asymmetric K/V
    bit widths. See ``olmlx.engine.kvarn`` for the codec."""

    def __init__(
        self,
        key_bits: int,
        value_bits: int,
        rotation_key: KVarNRotation,
        rotation_value: KVarNRotation,
    ):
        # Same side-effect-free constructor invariant as the base (see
        # ``TurboQuantKVCache.__init__``): ``__deepcopy__`` bypasses it.
        super().__init__(
            bits=key_bits, rotation_key=rotation_key, rotation_value=rotation_value
        )
        # Override the base's symmetric widths.
        self._key_bits = key_bits
        self._value_bits = value_bits

    def _quantize(self, x: mx.array, rotation: Any, bits: int):
        return kvarn_quantize(x, rotation, bits)

    def _encode(
        self, x: mx.array, rotation: Any, bits: int, dtype: mx.Dtype
    ) -> tuple[mx.array, mx.array, mx.array]:
        # The quantize kernel already builds the reconstruction to pick the
        # norm-preserving scale; emit it instead of re-deriving it.
        return kvarn_encode(x, rotation, bits, dtype)

    def _dequantize(
        self,
        packed: mx.array,
        side: mx.array,
        rotation: Any,
        bits: int,
        dtype: mx.Dtype | None = None,
    ) -> mx.array:
        return kvarn_dequantize(packed, side, rotation, bits, dtype=dtype)


def make_kvarn_cache(
    model: Any, key_bits: int, value_bits: int, rotation_memo: dict | None = None
) -> list:
    """Create a cache list with ``KVarNKVCache`` for attention layers.

    Hybrid layouts keep their non-KVCache layers; an attention layer whose
    head dim has no power-of-two tile ≥ 16 keeps a plain ``KVCache``.
    """
    head_dim = _detect_head_dim(model)

    def _rotation(layer_head_dim: int, seed: int) -> KVarNRotation:
        return memoized_rotation(
            rotation_memo,
            ("kvarn", layer_head_dim, seed),
            lambda: KVarNRotation(head_dim=layer_head_dim, seed=seed),
        )

    def _make_layer(i: int, layer_head_dim: int) -> KVarNKVCache | None:
        if choose_tile(layer_head_dim) is None:
            return None
        return KVarNKVCache(
            key_bits=key_bits,
            value_bits=value_bits,
            rotation_key=_rotation(layer_head_dim, i * 2),
            rotation_value=_rotation(layer_head_dim, i * 2 + 1),
        )

    caches, n_quantized = build_kv_quant_caches(model, _make_layer, head_dim=head_dim)
    if n_quantized == 0:
        logger.warning(
            "KVarN: no attention layer has a compatible head_dim (%d); "
            "KV cache left unquantized",
            head_dim,
        )
    logger.info(
        "Created KVarN KV cache: %d/%d cache entries quantized, k%dv%d, head_dim=%d",
        n_quantized,
        len(caches),
        key_bits,
        value_bits,
        head_dim,
    )
    return caches
