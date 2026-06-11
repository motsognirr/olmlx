"""Shard KV cache: drop-in replacement for mlx-lm's KVCache (#377 Tier 1).

Asymmetric K/V compression with an FP16 sink + FP16 sliding window and a
quantized middle region, dequantized transiently on every fetch:

    [ sink (exact) | middle (compressed) | window (exact) ]

K middle is stored *de-roped* (RoPE inverted at compress time) in a per-head
PCA basis with rank truncation; RoPE is re-applied at fetch using the
absolute positions sink_len..sink_len+mid_len.  V middle is rotation +
product-VQ.  No persistent dequantized side buffer — resident memory is
quantized + sink/window only (the Tier-1 memory contract from #377).

Deepcopy/snapshot safety: no ``mx.Dtype`` attributes (default deepcopy walk
works — the SpectralQuant pattern), and every mutable array is exposed via
``state`` so ``snapshot_cache_for_persistence``'s flatten + ``mx.eval``
materializes in-place-write graphs before any cross-thread reuse (#284).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_lm.models.cache import KVCache, _BaseCache, create_attention_mask

from olmlx.engine.shardquant import (
    RopeSpec,
    rope_transform,
    shard_compress_keys,
    shard_compress_values,
    shard_decompress_keys,
    shard_decompress_values,
)

logger = logging.getLogger(__name__)

#: Default sink/window sizes from the Shard reference design.
SINK_TOKENS = 4
WINDOW_TOKENS = 64


class ShardKVCache(_BaseCache):
    """KV cache with Shard-style asymmetric K/V compression."""

    step = 256

    def __init__(
        self,
        rope_spec: RopeSpec | None,
        k_basis: mx.array,
        k_rank: int,
        k_codebook: mx.array,
        k_bits: int,
        v_rotation: mx.array,
        v_codebooks: mx.array,
        sink_size: int = SINK_TOKENS,
        window_size: int = WINDOW_TOKENS,
        k_mean: mx.array | None = None,
    ):
        self.rope_spec = rope_spec
        self.k_basis = k_basis
        self.k_rank = k_rank
        self.k_codebook = k_codebook
        self.k_bits = k_bits
        # (H, D) per-head mean of unit-normalized calibration keys; None
        # for Tier-1 (un-centered) calibration artifacts.
        self.k_mean = k_mean
        self.v_rotation = v_rotation
        self.v_codebooks = v_codebooks
        self.sink_size = sink_size
        self.window_size = window_size

        # Exact regions (input dtype). Window buffers are small (<= window
        # + one prefill batch transiently); plain concat/slice is fine.
        self._k_sink: mx.array | None = None
        self._v_sink: mx.array | None = None
        self._k_win: mx.array | None = None
        self._v_win: mx.array | None = None
        # Compressed middle, grown in `step` increments like the other
        # quant caches. Only [..., :_mid_len, :] is valid.
        self._k_mid: mx.array | None = None
        self._k_mid_norms: mx.array | None = None
        self._v_mid: mx.array | None = None
        self._v_mid_norms: mx.array | None = None
        self._mid_len = 0
        self.offset = 0

    # -- internal helpers ---------------------------------------------------

    def _sink_len(self) -> int:
        return 0 if self._k_sink is None else self._k_sink.shape[2]

    def _win_len(self) -> int:
        return 0 if self._k_win is None else self._k_win.shape[2]

    def _append_middle(self, k_packed, k_norms, v_idx, v_norms) -> None:
        n = k_packed.shape[2]
        need = self._mid_len + n
        if self._k_mid is None or need > self._k_mid.shape[2]:
            new_steps = (need + self.step - 1) // self.step * self.step

            def _grow(buf: mx.array | None, last_dim: int, dtype) -> mx.array:
                shape = (*k_packed.shape[:2], new_steps, last_dim)
                fresh = mx.zeros(shape, dtype=dtype)
                if buf is not None and self._mid_len > 0:
                    fresh[..., : self._mid_len, :] = buf[..., : self._mid_len, :]
                return fresh

            self._k_mid = _grow(self._k_mid, k_packed.shape[-1], mx.uint8)
            self._k_mid_norms = _grow(self._k_mid_norms, 1, mx.float32)
            self._v_mid = _grow(self._v_mid, v_idx.shape[-1], mx.uint8)
            self._v_mid_norms = _grow(self._v_mid_norms, 1, mx.float32)
        assert (
            self._k_mid is not None
            and self._k_mid_norms is not None
            and self._v_mid is not None
            and self._v_mid_norms is not None
        )
        self._k_mid[..., self._mid_len : need, :] = k_packed
        self._k_mid_norms[..., self._mid_len : need, :] = k_norms
        self._v_mid[..., self._mid_len : need, :] = v_idx
        self._v_mid_norms[..., self._mid_len : need, :] = v_norms
        self._mid_len = need

    def _compress_into_middle(self, k: mx.array, v: mx.array) -> None:
        """Compress a window-front slice. Its absolute start position is
        sink_len + mid_len (the next middle slot)."""
        start = self._sink_len() + self._mid_len
        if self.rope_spec is not None:
            k = rope_transform(k, self.rope_spec, start, inverse=True)
        k_packed, k_norms = shard_compress_keys(
            k,
            self.k_basis,
            self.k_rank,
            self.k_codebook,
            self.k_bits,
            mean=self.k_mean,
        )
        v_idx, v_norms = shard_compress_values(v, self.v_rotation, self.v_codebooks)
        self._append_middle(k_packed, k_norms, v_idx, v_norms)

    def _decompress_middle(self, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        assert (
            self._k_mid is not None
            and self._k_mid_norms is not None
            and self._v_mid is not None
            and self._v_mid_norms is not None
        )
        m = self._mid_len
        k = shard_decompress_keys(
            self._k_mid[..., :m, :],
            self._k_mid_norms[..., :m, :],
            self.k_basis,
            self.k_rank,
            self.k_codebook,
            self.k_bits,
            mean=self.k_mean,
        )
        if self.rope_spec is not None:
            k = rope_transform(k, self.rope_spec, self._sink_len())
        v = shard_decompress_values(
            self._v_mid[..., :m, :],
            self._v_mid_norms[..., :m, :],
            self.v_rotation,
            self.v_codebooks,
        )
        return k.astype(dtype), v.astype(dtype)

    # -- _BaseCache interface -----------------------------------------------

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        num_steps = keys.shape[2]
        dtype = keys.dtype

        # 1. Fill the sink from the front of the stream.
        if self._sink_len() < self.sink_size:
            take = min(self.sink_size - self._sink_len(), num_steps)
            k_head, v_head = keys[..., :take, :], values[..., :take, :]
            if self._k_sink is None:
                self._k_sink, self._v_sink = k_head, v_head
            else:
                self._k_sink = mx.concatenate([self._k_sink, k_head], axis=2)
                self._v_sink = mx.concatenate([self._v_sink, v_head], axis=2)
            keys, values = keys[..., take:, :], values[..., take:, :]

        # 2. Append the rest to the window.
        if keys.shape[2] > 0:
            if self._k_win is None:
                self._k_win, self._v_win = keys, values
            else:
                self._k_win = mx.concatenate([self._k_win, keys], axis=2)
                self._v_win = mx.concatenate([self._v_win, values], axis=2)

        # 3. Spill window overflow into the compressed middle.
        overflow = self._win_len() - self.window_size
        if overflow > 0:
            assert self._k_win is not None and self._v_win is not None
            self._compress_into_middle(
                self._k_win[..., :overflow, :], self._v_win[..., :overflow, :]
            )
            self._k_win = self._k_win[..., overflow:, :]
            self._v_win = self._v_win[..., overflow:, :]

        self.offset += num_steps

        # 4. Assemble the full view (transient middle dequant).
        parts_k, parts_v = [], []
        if self._k_sink is not None:
            parts_k.append(self._k_sink)
            parts_v.append(self._v_sink)
        if self._mid_len > 0:
            mk, mv = self._decompress_middle(dtype)
            parts_k.append(mk)
            parts_v.append(mv)
        if self._k_win is not None and self._win_len() > 0:
            parts_k.append(self._k_win)
            parts_v.append(self._v_win)
        if len(parts_k) == 1:
            return parts_k[0], parts_v[0]
        return mx.concatenate(parts_k, axis=2), mx.concatenate(parts_v, axis=2)

    @property
    def state(self):
        if self.offset == 0:
            return []
        arrays = []
        for k_arr, v_arr in ((self._k_sink, self._v_sink), (self._k_win, self._v_win)):
            if k_arr is not None and k_arr.shape[2] > 0:
                arrays.append(k_arr)
                arrays.append(v_arr)
        if self._mid_len > 0:
            assert (
                self._k_mid is not None
                and self._k_mid_norms is not None
                and self._v_mid is not None
                and self._v_mid_norms is not None
            )
            arrays.extend(
                [
                    self._k_mid[..., : self._mid_len, :],
                    self._k_mid_norms[..., : self._mid_len, :],
                    self._v_mid[..., : self._mid_len, :],
                    self._v_mid_norms[..., : self._mid_len, :],
                ]
            )
        return arrays

    @state.setter
    def state(self, v):
        raise NotImplementedError(
            "ShardKVCache does not support state restoration. "
            "Disable disk cache offload when using shard quantization."
        )

    def is_trimmable(self):
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        remaining = n
        # Window first (newest tokens)
        take = min(remaining, self._win_len())
        if take > 0:
            assert self._k_win is not None and self._v_win is not None
            keep = self._win_len() - take
            self._k_win = self._k_win[..., :keep, :] if keep else None
            self._v_win = self._v_win[..., :keep, :] if keep else None
            remaining -= take
        # Then middle (per-token storage: trim = shrink the valid length)
        take = min(remaining, self._mid_len)
        if take > 0:
            self._mid_len -= take
            remaining -= take
        # Then sink
        take = min(remaining, self._sink_len())
        if take > 0:
            assert self._k_sink is not None and self._v_sink is not None
            keep = self._sink_len() - take
            self._k_sink = self._k_sink[..., :keep, :] if keep else None
            self._v_sink = self._v_sink[..., :keep, :] if keep else None
            remaining -= take
        self.offset -= n
        if self.offset == 0:
            self._k_mid = None
            self._k_mid_norms = None
            self._v_mid = None
            self._v_mid_norms = None
            self._mid_len = 0
        return n

    def make_mask(self, *args, **kwargs):
        kwargs["offset"] = self.offset
        return create_attention_mask(*args, **kwargs)

    def empty(self):
        return self.offset == 0


def make_shard_cache(model: Any, calibration_dir: Path, bits: int) -> list:
    """Create a cache list with ShardKVCache for attention layers.

    Mirrors ``make_spectral_cache``: non-attention caches (ArraysCache for
    SSM layers) are preserved; layers without calibration data fall back to
    the default cache with a warning.
    """
    from olmlx.engine.shardquant_calibrate import load_shard_calibration

    if calibration_dir is None:
        # Reachable when ModelManager has no store (_find_shard_dir returns
        # None instead of raising); fail with the remedy rather than an
        # opaque Path(None) TypeError downstream.
        raise ValueError(
            "Shard KV quant configured but no calibration directory was "
            "resolved. Run 'olmlx shard prepare <model>' and retry."
        )

    calibration, meta = load_shard_calibration(Path(calibration_dir))

    # _find_shard_dir raises on mismatch for the server path; warn here for
    # direct callers (mirrors make_spectral_cache).
    cal_bits = meta.get("bits")
    if cal_bits is not None and cal_bits != bits:
        logger.warning(
            "Shard calibration was run with bits=%d but shard:%d was "
            "configured; runtime uses the configured %d-bit packing against "
            "%d-bit codebooks. Re-run 'olmlx shard prepare' with --bits %d.",
            cal_bits,
            bits,
            bits,
            cal_bits,
            bits,
        )

    num_layers = len(model.layers)
    if hasattr(model, "make_cache"):
        default_caches = model.make_cache()
        if not isinstance(default_caches, list):
            default_caches = [None] * num_layers
    else:
        default_caches = [None] * num_layers

    caches = []
    quantized = 0
    for i, default in enumerate(default_caches):
        if default is not None and not isinstance(default, KVCache):
            caches.append(default)
            continue
        entry = calibration.get(i)
        if entry is None:
            logger.warning("No shard calibration for layer %d, using default", i)
            caches.append(default if default is not None else KVCache())
            continue
        rope_spec = None
        if entry.get("rope_freqs") is not None:
            rope_spec = RopeSpec(
                dims=entry["rope_dims"],
                freqs=entry["rope_freqs"],
                traditional=entry["rope_traditional"],
            )
        caches.append(
            ShardKVCache(
                rope_spec=rope_spec,
                k_basis=entry["k_basis"],
                k_rank=entry["k_rank"],
                k_codebook=entry["k_codebook"],
                k_bits=bits,
                v_rotation=entry["v_rotation"],
                v_codebooks=entry["v_codebooks"],
                k_mean=entry.get("k_mean"),
            )
        )
        quantized += 1

    logger.info(
        "Created Shard KV cache: %d/%d layers quantized, %d-bit",
        quantized,
        len(caches),
        bits,
    )
    return caches
