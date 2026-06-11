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
        y: (..., S, R) coefficients.
        codebook: (K,) sorted centroids.

    Returns:
        (..., S, R) uint8 indices.
    """
    chunks = []
    S = y.shape[-2]
    for s0 in range(0, S, _ASSIGN_SEQ_CHUNK):
        yc = y[..., s0 : s0 + _ASSIGN_SEQ_CHUNK, :].astype(mx.float32)
        dists = mx.abs(yc[..., None] - codebook)
        chunks.append(mx.argmin(dists, axis=-1).astype(mx.uint8))
    return mx.concatenate(chunks, axis=-2) if len(chunks) > 1 else chunks[0]
