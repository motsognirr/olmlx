"""KVarN: variance-normalized KV cache quantization (#748).

Modeled on KVarN (Huawei CSL, 2026, Apache 2.0; originally a vLLM backend).
Its analysis finds that aggressive KV quantization hurts long reasoning mainly
because errors *accumulate* across decode steps, and that most end-to-end
degradation comes from a small fraction of tokens whose **magnitudes** get
crushed — not from directional error.

Per token and per tile of ``tile`` coordinates:

1. Hadamard rotation (block-diagonal, with seeded random sign flips) — spreads
   outlier channels across the tile so no single coordinate dominates.
2. Variance normalization — subtract the tile mean, divide by the tile std, so
   every tile hits the Gaussian Lloyd-Max codebook at unit scale regardless of
   the token's magnitude.
3. Scalar quantization against the N(0, 1) Lloyd-Max codebook.
4. A per-tile scale stored alongside the mean. It is chosen so the
   reconstructed tile has *exactly* the original centered norm (the
   reconstruction is re-centered first), so each token's L2 norm survives
   quantization. MSE-optimal codebooks are biased toward shrinkage — the
   TurboQuant reconstruction of a unit vector has norm < 1 — and that
   per-step shrinkage is the magnitude error KVarN targets.

Keys and values take independent bit widths (``kvarn:k4v2``): keys feed the
softmax and are far more sensitive than values.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

from olmlx.config import parse_kvarn_bits  # noqa: F401 — re-export
from olmlx.engine.shardquant import make_v_rotation
from olmlx.engine.turboquant import get_codebook, pack_indices, unpack_indices

#: Largest tile; also the preferred one when head_dim allows it.
_MAX_TILE = 128
#: Smallest tile worth a (mean, scale) pair; below this the per-tile stats
#: overhead dominates, so the layer stays unquantized.
_MIN_TILE = 16


def choose_tile(head_dim: int) -> int | None:
    """Largest power-of-two tile ≤ 128 dividing ``head_dim``, or ``None``
    when that tile would be smaller than 16 (layer left unquantized)."""
    tile = _MAX_TILE
    while tile >= _MIN_TILE:
        if head_dim % tile == 0:
            return tile
        tile //= 2
    return None


class KVarNRotation:
    """Per-layer block-diagonal randomized Hadamard rotation.

    Mathematically ``R = blockdiag(H_tile, ...) @ diag(signs)`` (orthogonal),
    applied as ``y = x @ R.T``. It is never materialized densely: the kernels
    flip signs, reshape to ``(..., n_tiles, tile)`` and multiply by the single
    ``tile x tile`` Hadamard, so a ``head_dim`` > ``tile`` layer (Gemma 4's
    512-wide global layers) doesn't pay for the zero blocks. The Sylvester
    Hadamard is symmetric, so the same matrix serves both directions.
    ``TurboQuantKVCache.__deepcopy__`` shares objects stored under
    ``rotation_key`` / ``rotation_value`` by reference.
    """

    def __init__(self, head_dim: int, seed: int):
        tile = choose_tile(head_dim)
        if tile is None:
            raise ValueError(
                f"KVarN: head_dim={head_dim} has no power-of-two tile ≥ "
                f"{_MIN_TILE} dividing it"
            )
        self.head_dim = head_dim
        self.tile = tile
        rng = np.random.RandomState(seed)
        self.signs: mx.array = mx.array(
            rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=head_dim)
        )
        self.hadamard: mx.array = make_v_rotation(tile)
        # Materialize eagerly: the prompt cache (and this rotation) is built on
        # the event-loop thread, prefill/decode run on a generation worker.
        # Under mlx thread-local streams (#499) a lazy op would stay bound to
        # the constructing thread — same fix as TurboQuantRotation.
        mx.eval(self.signs, self.hadamard)

    def dense_matrix(self) -> np.ndarray:
        """The full ``head_dim x head_dim`` ``R`` (for tests/inspection only)."""
        h = np.array(self.hadamard)
        block = np.zeros((self.head_dim, self.head_dim), dtype=np.float32)
        for start in range(0, self.head_dim, self.tile):
            block[start : start + self.tile, start : start + self.tile] = h
        return block * np.array(self.signs)[None, :]


def _unit_codebook(bits: int) -> mx.array:
    """The N(0, 1) Lloyd-Max codebook (materialized, cached by turboquant)."""
    return get_codebook(bits, 1)


@lru_cache(maxsize=128)
def _compiled_quantize_core(
    bits: int,
    tile: int,
    x_shape: tuple,
    x_dtype: mx.Dtype,
    out_dtype: mx.Dtype | None,
):
    """Compiled (rotate + normalize + quantize + scale) kernel, cached per
    full input shape for the same reason as turboquant's (mlx's shapeless
    compile bakes non-last dims into the trace). ``x_dtype`` keys the cache
    only. With ``out_dtype`` set it also returns the reconstruction (what
    ``kvarn_dequantize`` would produce), saving the cache a second
    unpack/gather pass per decode step.

    Bit-packing happens *inside* the kernel so the packed indices are a
    sibling output of the same primitive as the reconstruction. In the cache
    the packed indices feed only the never-per-token-evaluated
    ``_key_indices`` slice_update chain; packed outside the kernel they would
    stay a lazy subgraph per token per layer, pinning the kernel's output
    buffers until end-of-generation — exhausting Metal's buffer limit
    (``metal::malloc Resource limit``) within a few thousand decode tokens.
    """
    n_levels = 1 << bits
    head_dim = x_shape[-1]
    n_tiles = head_dim // tile
    lead = tuple(x_shape[:-1])
    tiled = lead + (n_tiles, tile)

    @mx.compile
    def _fn(
        x: mx.array, signs: mx.array, hadamard: mx.array, codebook: mx.array
    ) -> tuple[mx.array, ...]:
        yt = (x.astype(mx.float32) * signs).reshape(tiled) @ hadamard
        mu = mx.mean(yt, axis=-1, keepdims=True)
        c = yt - mu
        c_norm = mx.sqrt(mx.sum(c * c, axis=-1, keepdims=True))
        sigma = c_norm / mx.sqrt(mx.array(float(tile), dtype=mx.float32))
        eps = mx.array(1e-30, dtype=mx.float32)
        z = c / mx.maximum(sigma, eps)

        best_dist = mx.abs(z - codebook[0])
        best_idx = mx.zeros(z.shape, dtype=mx.uint8)
        for ci in range(1, n_levels):
            d = mx.abs(z - codebook[ci])
            better = d < best_dist
            best_idx = mx.where(better, ci, best_idx).astype(mx.uint8)
            best_dist = mx.where(better, d, best_dist)

        # Norm-preserving scale: the re-centered reconstruction gets exactly
        # the original centered norm. A tile whose indices are all equal has a
        # zero centered reconstruction → scale 0 → reconstructs to its mean.
        zh = codebook[best_idx.astype(mx.uint32)]
        zc = zh - mx.mean(zh, axis=-1, keepdims=True)
        zc_norm = mx.sqrt(mx.sum(zc * zc, axis=-1, keepdims=True))
        scale = mx.where(zc_norm > eps, c_norm / mx.maximum(zc_norm, eps), 0.0)

        stats = mx.concatenate([mu[..., 0], scale[..., 0]], axis=-1)
        packed = pack_indices(best_idx.reshape(lead + (head_dim,)), bits)
        if out_dtype is None:
            return (packed, stats)
        recon = ((mu + scale * zc) @ hadamard).reshape(lead + (head_dim,)) * signs
        return (packed, stats, recon.astype(out_dtype))

    return _fn


def kvarn_quantize(
    x: mx.array, rotation: KVarNRotation, bits: int
) -> tuple[mx.array, mx.array]:
    """Quantize ``x`` of shape ``(..., head_dim)``.

    Returns ``(packed, stats)``: bit-packed uint8 indices of shape
    ``(..., head_dim // (8 // bits))`` and float32 per-tile stats of shape
    ``(..., 2 * n_tiles)`` laid out as ``[means..., scales...]``.
    """
    fn = _compiled_quantize_core(bits, rotation.tile, tuple(x.shape), x.dtype, None)
    out = fn(x, rotation.signs, rotation.hadamard, _unit_codebook(bits))
    return out[0], out[1]


def kvarn_encode(
    x: mx.array, rotation: KVarNRotation, bits: int, dtype: mx.Dtype
) -> tuple[mx.array, mx.array, mx.array]:
    """``kvarn_quantize`` plus the reconstruction in ``dtype``, in one pass."""
    fn = _compiled_quantize_core(bits, rotation.tile, tuple(x.shape), x.dtype, dtype)
    out = fn(x, rotation.signs, rotation.hadamard, _unit_codebook(bits))
    return out[0], out[1], out[2]


@lru_cache(maxsize=128)
def _compiled_dequant_core(n_levels: int, tile: int, indices_shape: tuple):
    """Compiled (gather + re-center + rescale + inverse rotate) kernel.
    ``n_levels`` keys the cache so K and V at different widths never alias."""
    head_dim = indices_shape[-1]
    n_tiles = head_dim // tile
    lead = tuple(indices_shape[:-1])

    @mx.compile
    def _fn(
        indices: mx.array,
        stats: mx.array,
        signs: mx.array,
        hadamard: mx.array,
        codebook: mx.array,
    ) -> mx.array:
        zh = codebook[indices.astype(mx.uint32)].reshape(lead + (n_tiles, tile))
        zc = zh - mx.mean(zh, axis=-1, keepdims=True)
        mu = stats[..., :n_tiles][..., None]
        scale = stats[..., n_tiles:][..., None]
        yt = mu + scale * zc
        return (yt @ hadamard).reshape(lead + (head_dim,)) * signs

    return _fn


def kvarn_dequantize(
    packed: mx.array,
    stats: mx.array,
    rotation: KVarNRotation,
    bits: int,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Reconstruct ``(..., head_dim)`` vectors from ``kvarn_quantize`` output."""
    indices = unpack_indices(packed, bits, rotation.head_dim)
    fn = _compiled_dequant_core(1 << bits, rotation.tile, tuple(indices.shape))
    out = fn(indices, stats, rotation.signs, rotation.hadamard, _unit_codebook(bits))
    return out.astype(dtype) if dtype is not None else out
