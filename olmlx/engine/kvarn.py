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
from olmlx.engine.turboquant import get_codebook, pack_indices, unpack_indices

#: Bit widths supported for each of K and V (the packers handle 2 and 4).
KVARN_SUPPORTED_BITS: tuple[int, ...] = (2, 4)

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


def _hadamard(n: int) -> np.ndarray:
    """Orthonormal Sylvester Hadamard matrix of size ``n`` (a power of two)."""
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(np.float32(n))


class KVarNRotation:
    """Per-layer block-diagonal randomized Hadamard rotation.

    ``matrix = blockdiag(H_tile, ...) @ diag(signs)`` — orthogonal, so the
    inverse is the transpose. Exposes the same ``matrix`` / ``matrix_T``
    attribute names as ``TurboQuantRotation``; ``TurboQuantKVCache`` shares
    objects stored under ``rotation_key`` / ``rotation_value`` by reference
    in ``__deepcopy__``.
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
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=head_dim)
        h = _hadamard(tile)
        block = np.zeros((head_dim, head_dim), dtype=np.float32)
        for start in range(0, head_dim, tile):
            block[start : start + tile, start : start + tile] = h
        self.matrix: mx.array = mx.array(block * signs[None, :])
        self.matrix_T: mx.array = self.matrix.T
        # Materialize eagerly: the prompt cache (and this rotation) is built on
        # the event-loop thread, prefill/decode run on a generation worker.
        # Under mlx thread-local streams (#499) the lazy ``.T`` would stay bound
        # to the constructing thread — same fix as TurboQuantRotation.
        mx.eval(self.matrix, self.matrix_T)


def _unit_codebook(bits: int) -> mx.array:
    """The N(0, 1) Lloyd-Max codebook (materialized, cached by turboquant)."""
    return get_codebook(bits, 1)


@lru_cache(maxsize=128)
def _compiled_quantize_core(
    n_levels: int, tile: int, x_shape: tuple, x_dtype: mx.Dtype
):
    """Compiled (rotate + normalize + quantize + scale) kernel, cached per
    full input shape for the same reason as turboquant's (mlx's shapeless
    compile bakes non-last dims into the trace)."""
    head_dim = x_shape[-1]
    n_tiles = head_dim // tile
    lead = tuple(x_shape[:-1])

    @mx.compile
    def _fn(
        x: mx.array, rotation_T: mx.array, codebook: mx.array
    ) -> tuple[mx.array, mx.array]:
        y = x.astype(mx.float32) @ rotation_T
        yt = y.reshape(lead + (n_tiles, tile))
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
        return best_idx.reshape(lead + (head_dim,)), stats

    return _fn


def kvarn_quantize(
    x: mx.array, rotation: KVarNRotation, bits: int
) -> tuple[mx.array, mx.array]:
    """Quantize ``x`` of shape ``(..., head_dim)``.

    Returns ``(packed, stats)``: bit-packed uint8 indices of shape
    ``(..., head_dim // (8 // bits))`` and float32 per-tile stats of shape
    ``(..., 2 * n_tiles)`` laid out as ``[means..., scales...]``.
    """
    codebook = _unit_codebook(bits)
    fn = _compiled_quantize_core(1 << bits, rotation.tile, tuple(x.shape), x.dtype)
    idx, stats = fn(x, rotation.matrix_T, codebook)
    return pack_indices(idx, bits), stats


@lru_cache(maxsize=128)
def _compiled_dequant_core(n_levels: int, tile: int, indices_shape: tuple):
    """Compiled (gather + re-center + rescale + inverse rotate) kernel.
    ``n_levels`` keys the cache so K and V at different widths never alias."""
    head_dim = indices_shape[-1]
    n_tiles = head_dim // tile
    lead = tuple(indices_shape[:-1])

    @mx.compile
    def _fn(
        indices: mx.array, stats: mx.array, rotation: mx.array, codebook: mx.array
    ) -> mx.array:
        zh = codebook[indices.astype(mx.uint32)].reshape(lead + (n_tiles, tile))
        zc = zh - mx.mean(zh, axis=-1, keepdims=True)
        mu = stats[..., :n_tiles][..., None]
        scale = stats[..., n_tiles:][..., None]
        y = (mu + scale * zc).reshape(lead + (head_dim,))
        return y @ rotation

    return _fn


def kvarn_dequantize(
    packed: mx.array,
    stats: mx.array,
    rotation: KVarNRotation,
    bits: int,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Reconstruct ``(..., head_dim)`` vectors from ``kvarn_quantize`` output."""
    codebook = _unit_codebook(bits)
    indices = unpack_indices(packed, bits, rotation.head_dim)
    fn = _compiled_dequant_core(1 << bits, rotation.tile, tuple(indices.shape))
    out = fn(indices, stats, rotation.matrix, codebook)
    return out.astype(dtype) if dtype is not None else out
