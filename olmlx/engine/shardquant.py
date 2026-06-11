"""ShardQuant primitives: asymmetric K/V KV-cache compression (#377 Tier 1).

Reference design: https://github.com/krish1905/shard
- K: undo RoPE -> per-head PCA basis -> rank truncation -> scalar Lloyd-Max.
- V: orthogonal rotation (Hadamard) -> per-position product VQ (256 centroids).
Both sides normalize to the unit sphere and store float32 norms, matching
the TurboQuant/SpectralQuant convention in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from olmlx.engine.spectralquant import pack_indices, unpack_indices

__all__ = [
    "RopeSpec",
    "detect_rope_spec",
    "rope_transform",
    "make_v_rotation",
    "fit_vq_codebooks",
    "vq_assign",
    "vq_gather",
    "scalar_assign",
    "shard_compress_keys",
    "shard_decompress_keys",
    "shard_compress_values",
    "shard_decompress_values",
    "pack_indices",
    "unpack_indices",
]

# Number of centroids per product-VQ codebook. uint8 indices, one byte per
# subvector; subvector size g = 8 // bits gives exactly `bits` bits/dim.
VQ_CENTROIDS = 256

# Sequence-chunk size for the broadcast argmin in vq_assign / scalar_assign.
# Bounds the transient (..., chunk, P, 256) distance tensor (~256 MB at
# H=8, D/g=64 — same budget rationale as spectralquant's broadcast cap).
_ASSIGN_SEQ_CHUNK = 512


@dataclass
class RopeSpec:
    """Angular-frequency description of a layer's rotary embedding.

    ``freqs`` (dims//2,) are *angular* frequencies: the rotation angle for
    pair ``i`` at position ``p`` is ``p * freqs[i]``.  Stored in the shard
    calibration artifacts so calibration-time de-rope and runtime re-rope
    are guaranteed to use identical transforms.
    """

    dims: int
    freqs: mx.array
    traditional: bool


def detect_rope_spec(attn: Any) -> RopeSpec | None:
    """Extract a RopeSpec from an attention module, or None if unsupported.

    Handles ``mlx.nn.RoPE`` (base/scale parameterization) and mlx-lm's
    custom rope modules (Llama3RoPE, ...) that precompute wavelength-like
    ``_freqs`` and call ``mx.fast.rope(..., base=None, freqs=self._freqs)``
    — for those the angular frequency is ``1 / _freqs``.  Unknown rope
    types return None; the caller falls back to identity (no de-rope),
    which stays correct (undo/redo of identity) at reduced rank collapse.
    """
    rope = getattr(attn, "rope", None)
    if rope is None:
        return None
    import mlx.nn as nn

    if isinstance(rope, nn.RoPE):
        dims = rope.dims
        base = float(rope.base)
        scale = float(getattr(rope, "scale", 1.0))
        freqs = scale * mx.power(
            mx.array(base, dtype=mx.float32),
            -mx.arange(0, dims, 2, dtype=mx.float32) / dims,
        )
        return RopeSpec(dims=dims, freqs=freqs, traditional=bool(rope.traditional))

    inv_freqs = getattr(rope, "_freqs", None)
    dims = getattr(rope, "dims", None)
    if isinstance(inv_freqs, mx.array) and isinstance(dims, int):
        if inv_freqs.ndim == 1 and inv_freqs.shape[0] == dims // 2:
            return RopeSpec(
                dims=dims,
                freqs=(1.0 / inv_freqs.astype(mx.float32)),
                traditional=bool(getattr(rope, "traditional", False)),
            )
    return None


def rope_transform(
    x: mx.array, spec: RopeSpec, offset: int, *, inverse: bool = False
) -> mx.array:
    """Apply (or invert) rotary embedding for contiguous positions.

    Args:
        x: (..., S, D) tensor; the last ``D - spec.dims`` dims pass through.
        spec: RopeSpec with angular freqs.
        offset: absolute position of x[..., 0, :].
        inverse: apply the inverse rotation (de-rope).
    """
    S = x.shape[-2]
    pos = mx.arange(offset, offset + S, dtype=mx.float32)
    angles = pos[:, None] * spec.freqs[None, :]  # (S, dims//2)
    cos = mx.cos(angles)
    sin = mx.sin(angles)
    if inverse:
        sin = -sin

    half = spec.dims // 2
    x32 = x.astype(mx.float32)
    head = x32[..., : spec.dims]
    if spec.traditional:
        xr = head.reshape(*head.shape[:-1], half, 2)
        x1, x2 = xr[..., 0], xr[..., 1]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        rotated = mx.stack([y1, y2], axis=-1).reshape(*head.shape)
    else:
        x1, x2 = head[..., :half], head[..., half:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        rotated = mx.concatenate([y1, y2], axis=-1)

    if spec.dims < x.shape[-1]:
        rotated = mx.concatenate([rotated, x32[..., spec.dims :]], axis=-1)
    return rotated.astype(x.dtype)


def make_v_rotation(dim: int, seed: int = 0) -> mx.array:
    """Orthonormal rotation for the V path.

    Sylvester-Hadamard (scaled 1/sqrt(dim)) when ``dim`` is a power of two,
    otherwise a seeded random orthogonal matrix via QR.  The matrix is
    persisted in the calibration artifacts, so the runtime never needs to
    re-derive it — one uniform code path either way.
    """
    if dim > 0 and (dim & (dim - 1)) == 0:
        h = np.array([[1.0]], dtype=np.float64)
        while h.shape[0] < dim:
            h = np.block([[h, h], [h, -h]])
        return mx.array((h / np.sqrt(dim)).astype(np.float32))
    rng = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rng.randn(dim, dim).astype(np.float32))
    return mx.array(q.astype(np.float32))


def fit_vq_codebooks(
    data: np.ndarray,
    group_size: int,
    n_centroids: int = VQ_CENTROIDS,
    max_iter: int = 25,
    seed: int = 0,
) -> mx.array:
    """Fit per-position product-VQ codebooks via plain numpy k-means.

    Args:
        data: (N, D) rotated value vectors; D % group_size == 0.
        group_size: subvector length g.

    Returns:
        (D // group_size, n_centroids, group_size) float32 codebooks.
    """
    n, d = data.shape
    if d % group_size != 0:
        raise ValueError(f"dim {d} not divisible by group_size {group_size}")
    p = d // group_size
    sub = data.reshape(n, p, group_size).astype(np.float32)
    rng = np.random.RandomState(seed)
    codebooks = np.empty((p, n_centroids, group_size), dtype=np.float32)

    for j in range(p):
        pts = sub[:, j, :]  # (N, g)
        init_idx = rng.randint(0, n, size=n_centroids)
        cent = pts[init_idx].copy()
        for _ in range(max_iter):
            # (N, K) squared distances via expansion; K*g is small.
            d2 = (
                (pts**2).sum(-1, keepdims=True)
                - 2.0 * pts @ cent.T
                + (cent**2).sum(-1)[None, :]
            )
            assign = d2.argmin(axis=1)
            new_cent = cent.copy()
            moved = 0.0
            for k in range(n_centroids):
                mask = assign == k
                if mask.any():
                    nc = pts[mask].mean(axis=0)
                    moved = max(moved, float(np.abs(nc - new_cent[k]).max()))
                    new_cent[k] = nc
                else:
                    # Re-seed empty clusters to a random point.
                    new_cent[k] = pts[rng.randint(0, n)]
            cent = new_cent
            if moved < 1e-6:
                break
        codebooks[j] = cent
    return mx.array(codebooks)


def vq_assign(x: mx.array, codebooks: mx.array) -> mx.array:
    """Assign each subvector to its nearest centroid.

    Args:
        x: (..., S, P, g) subvectors.
        codebooks: (P, K, g).

    Returns:
        (..., S, P) uint8 indices.
    """
    c_sq = mx.sum(codebooks * codebooks, axis=-1)  # (P, K)
    chunks = []
    S = x.shape[-3]
    for s0 in range(0, S, _ASSIGN_SEQ_CHUNK):
        xc = x[..., s0 : s0 + _ASSIGN_SEQ_CHUNK, :, :].astype(mx.float32)
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 ; ||x||^2 constant in argmin.
        scores = c_sq - 2.0 * mx.einsum("...pg,pkg->...pk", xc, codebooks)
        chunks.append(mx.argmin(scores, axis=-1).astype(mx.uint8))
    return mx.concatenate(chunks, axis=-2) if len(chunks) > 1 else chunks[0]


def vq_gather(idx: mx.array, codebooks: mx.array) -> mx.array:
    """Inverse of vq_assign: (..., S, P) uint8 -> (..., S, P, g) float32."""
    p, k, g = codebooks.shape
    flat = codebooks.reshape(p * k, g)
    flat_idx = idx.astype(mx.uint32) + mx.arange(p, dtype=mx.uint32) * k
    return flat[flat_idx]


def scalar_assign(y: mx.array, codebook: mx.array) -> mx.array:
    """Nearest-centroid index per element, chunked over the seq axis.

    Args:
        y: (..., S, R) coefficients; with a per-head codebook the head axis
           must be axis -3, i.e. (B, H, S, R).
        codebook: (K,) shared centroids, or (H, K) per-head centroids.

    Returns:
        (..., S, R) uint8 indices.
    """
    if codebook.ndim == 2:
        # (H, K) -> (H, 1, 1, K): broadcasts against (..., H, S, R, 1).
        codebook = codebook[:, None, None, :]
    chunks = []
    S = y.shape[-2]
    for s0 in range(0, S, _ASSIGN_SEQ_CHUNK):
        yc = y[..., s0 : s0 + _ASSIGN_SEQ_CHUNK, :].astype(mx.float32)
        dists = mx.abs(yc[..., None] - codebook)
        chunks.append(mx.argmin(dists, axis=-1).astype(mx.uint8))
    return mx.concatenate(chunks, axis=-2) if len(chunks) > 1 else chunks[0]


def _codebook_gather(idx: mx.array, codebook: mx.array) -> mx.array:
    """Centroid lookup for (K,) shared or (H, K) per-head codebooks.

    idx: (B, H, S, R) uint8 indices -> (B, H, S, R) float32 values.
    """
    if codebook.ndim == 1:
        return codebook[idx.astype(mx.uint32)]
    h, k = codebook.shape
    flat = codebook.reshape(h * k)
    offsets = (mx.arange(h, dtype=mx.uint32) * k)[None, :, None, None]
    return flat[idx.astype(mx.uint32) + offsets]


_NORM_EPS = 1e-8


def _normalize(x: mx.array) -> tuple[mx.array, mx.array]:
    """Unit-sphere normalize; returns (normalized float32, float32 norms)."""
    x32 = x.astype(mx.float32)
    norms = mx.sqrt(mx.sum(x32 * x32, axis=-1, keepdims=True))
    eps = mx.array(_NORM_EPS, dtype=mx.float32)
    return x32 / mx.maximum(norms, eps), norms


def shard_compress_keys(
    keys: mx.array,
    basis: mx.array,
    rank: int,
    codebook: mx.array,
    bits: int,
    mean: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Compress (already de-roped) keys: normalize -> subtract per-head
    mean -> per-head project -> rank-truncate -> scalar quantize -> pack.

    Mean-centering is load-bearing for rank truncation: the basis comes
    from a *centered* covariance, so truncating un-centered projections
    drops the mean direction — the dominant component of every key.

    Args:
        keys: (B, H, S, D) de-roped keys.
        basis: (H, D, D), rows = eigenvectors.
        rank: kept leading coefficients.
        codebook: (2**bits,) shared or (H, 2**bits) per-head Lloyd-Max
            centroids for the kept coefficients.
        mean: (H, D) per-head mean of the unit-normalized calibration keys,
            or None for legacy (un-centered) artifacts.

    Returns:
        (packed, norms): packed uint8 indices (B, H, S, packed_rank) +
        float32 norms (B, H, S, 1).
    """
    xn, norms = _normalize(keys)
    if mean is not None:
        xn = xn - mean[None, :, None, :].astype(mx.float32)
    y = mx.matmul(xn, mx.swapaxes(basis, -1, -2))  # (B,H,S,D) @ (H,D,D)^T
    idx = scalar_assign(y[..., :rank], codebook)
    return pack_indices(idx, bits), norms


def shard_decompress_keys(
    packed: mx.array,
    norms: mx.array,
    basis: mx.array,
    rank: int,
    codebook: mx.array,
    bits: int,
    mean: mx.array | None = None,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Inverse of shard_compress_keys (still in the no-RoPE basis)."""
    idx = unpack_indices(packed, bits, rank)
    y = _codebook_gather(idx, codebook)
    d = basis.shape[-1]
    if rank < d:
        pad = mx.zeros((*y.shape[:-1], d - rank), dtype=y.dtype)
        y = mx.concatenate([y, pad], axis=-1)
    x = mx.matmul(y, basis)
    if mean is not None:
        x = x + mean[None, :, None, :].astype(x.dtype)
    x = x * norms
    return x.astype(dtype) if dtype is not None else x


def shard_compress_values(
    values: mx.array,
    rotation: mx.array,
    codebooks: mx.array,
) -> tuple[mx.array, mx.array]:
    """Compress values: normalize -> rotate -> product VQ.

    Args:
        values: (B, H, S, D).
        rotation: (D, D) orthonormal (rows = basis).
        codebooks: (P, K, g) with P * g == D.

    Returns:
        (idx, norms): uint8 indices (B, H, S, P) + float32 norms (B, H, S, 1).
    """
    p, _, g = codebooks.shape
    xn, norms = _normalize(values)
    rotated = xn @ rotation.T
    sub = rotated.reshape(*rotated.shape[:-1], p, g)
    return vq_assign(sub, codebooks), norms


def shard_decompress_values(
    idx: mx.array,
    norms: mx.array,
    rotation: mx.array,
    codebooks: mx.array,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Inverse of shard_compress_values."""
    sub = vq_gather(idx, codebooks)
    d = rotation.shape[0]
    rotated = sub.reshape(*sub.shape[:-2], d)
    x = (rotated @ rotation) * norms
    return x.astype(dtype) if dtype is not None else x
