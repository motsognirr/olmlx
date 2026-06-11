"""ShardQuant primitives: asymmetric K/V KV-cache compression (#377 Tier 1).

Reference design: https://github.com/krish1905/shard
- K: undo RoPE -> per-head PCA basis -> rank truncation -> scalar Lloyd-Max.
- V: orthogonal rotation (Hadamard) -> per-position product VQ (256 centroids).
Both sides normalize to the unit sphere and store float32 norms, matching
the TurboQuant/SpectralQuant convention in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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

# Sequence-chunk size for the broadcast argmin in the K compress path.
# Bounds the transient (..., chunk, rank, 2**bits) distance tensor
# (~20 MB at H=8, rank~80, 4-bit — same budget rationale as
# spectralquant's broadcast cap).
_ASSIGN_SEQ_CHUNK = 512

# The V path's score tensor is (..., chunk, P, 256) — at H=8, P=64 that is
# 16 KB *per token*, 13x the K path's footprint — so it gets a smaller
# chunk (~270 MB transient).  A 2 GB transient per layer per prefill chunk
# (the 512 value) thrashed Metal and dominated shard prefill time.
_VQ_SEQ_CHUNK = 64


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
    fn = _compiled_rope_core(
        tuple(x.shape), x.dtype, spec.dims, spec.traditional, inverse
    )
    return fn(x, spec.freqs, pos)


@lru_cache(maxsize=128)
def _compiled_rope_core(
    x_shape: tuple,
    x_dtype: mx.Dtype,
    dims: int,
    traditional: bool,
    inverse: bool,
):
    """Compiled rotary apply/invert.

    Positions arrive as an *array* input so the trace is keyed only by
    shape — a varying integer offset (the de-rope offset advances every
    decode step) would otherwise force a re-trace per step.  Cached per
    full input shape because mlx's compile bakes leading dims into the
    trace (same rationale as spectralquant's compiled kernels).
    """
    half = dims // 2

    @mx.compile
    def _fn(x: mx.array, freqs: mx.array, pos: mx.array) -> mx.array:
        angles = pos[:, None] * freqs[None, :]  # (S, dims//2)
        cos = mx.cos(angles)
        sin = mx.sin(angles)
        if inverse:
            sin = -sin
        x32 = x.astype(mx.float32)
        head = x32[..., :dims]
        if traditional:
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
        if dims < x_shape[-1]:
            rotated = mx.concatenate([rotated, x32[..., dims:]], axis=-1)
        return rotated.astype(x_dtype)

    return _fn


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
    for s0 in range(0, S, _VQ_SEQ_CHUNK):
        xc = x[..., s0 : s0 + _VQ_SEQ_CHUNK, :, :].astype(mx.float32)
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
    mids = _sorted_codebook_midpoints(codebook)
    chunks = []
    S = y.shape[-2]
    for s0 in range(0, S, _ASSIGN_SEQ_CHUNK):
        yc = y[..., s0 : s0 + _ASSIGN_SEQ_CHUNK, :].astype(mx.float32)
        chunks.append(mx.sum(yc[..., None] >= mids, axis=-1).astype(mx.uint8))
    return mx.concatenate(chunks, axis=-2) if len(chunks) > 1 else chunks[0]


def _sorted_codebook_midpoints(codebook: mx.array) -> mx.array:
    """Decision boundaries for a *sorted* scalar codebook.

    Nearest-centroid lookup in a sorted codebook reduces to counting
    midpoint thresholds crossed: ``idx = sum(y >= midpoints)``.  This
    avoids materializing the (..., R, K) float distance tensor of the
    abs-argmin formulation — a 4x memory-traffic cut that dominates
    prefill compress time.  ``fit_codebook`` sorts its output, so the
    precondition holds for every calibration artifact.
    """
    mids = (codebook[..., 1:] + codebook[..., :-1]) / 2.0
    if codebook.ndim == 2:
        # (H, K-1) -> (H, 1, 1, K-1): broadcasts against (..., H, S, R, 1).
        return mids[:, None, None, :]
    return mids


_NORM_EPS = 1e-8


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
    h, d = basis.shape[0], basis.shape[-1]
    mean_arr = (
        mean.astype(mx.float32)
        if mean is not None
        else mx.zeros((h, d), dtype=mx.float32)
    )
    idx_chunks, norm_chunks = [], []
    S = keys.shape[2]
    for s0 in range(0, S, _ASSIGN_SEQ_CHUNK):
        kc = keys[..., s0 : s0 + _ASSIGN_SEQ_CHUNK, :]
        fn = _compiled_k_compress_core(
            tuple(kc.shape),
            kc.dtype,
            rank,
            int(codebook.shape[-1]),
            codebook.ndim == 2,
        )
        idx, norms = fn(kc, basis, codebook, mean_arr)
        idx_chunks.append(idx)
        norm_chunks.append(norms)
    idx = mx.concatenate(idx_chunks, axis=2) if len(idx_chunks) > 1 else idx_chunks[0]
    norms = (
        mx.concatenate(norm_chunks, axis=2) if len(norm_chunks) > 1 else norm_chunks[0]
    )
    return pack_indices(idx, bits), norms


@lru_cache(maxsize=128)
def _compiled_k_compress_core(
    x_shape: tuple,
    x_dtype: mx.Dtype,
    rank: int,
    n_levels: int,
    per_head: bool,
):
    """Compiled (normalize + center + project + truncate + assign) kernel.

    The mean is always an input (callers pass zeros for the legacy
    un-centered path) so the trace count stays low.  ``n_levels`` is in
    the key because the index shape alone cannot distinguish bit widths
    (same rationale as spectralquant's compiled cores)."""

    @mx.compile
    def _fn(
        x: mx.array, basis: mx.array, codebook: mx.array, mean: mx.array
    ) -> tuple[mx.array, mx.array]:
        x32 = x.astype(mx.float32)
        norms = mx.sqrt(mx.sum(x32 * x32, axis=-1, keepdims=True))
        eps = mx.array(_NORM_EPS, dtype=mx.float32)
        xn = x32 / mx.maximum(norms, eps)
        xn = xn - mean[None, :, None, :]
        y = mx.matmul(xn, mx.swapaxes(basis, -1, -2))[..., :rank]
        # Sorted-codebook threshold count (see _sorted_codebook_midpoints);
        # `per_head` is the (H, K) layout.
        mids = (codebook[..., 1:] + codebook[..., :-1]) / 2.0
        if per_head:
            mids = mids[:, None, None, :]
        idx = mx.sum(y[..., None] >= mids, axis=-1).astype(mx.uint8)
        return idx, norms

    return _fn


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
    h, d = basis.shape[0], basis.shape[-1]
    mean_arr = (
        mean.astype(mx.float32)
        if mean is not None
        else mx.zeros((h, d), dtype=mx.float32)
    )
    fn = _compiled_k_decompress_core(
        tuple(packed.shape),
        norms.dtype,
        rank,
        bits,
        int(codebook.shape[-1]),
        codebook.ndim == 2,
        d,
        dtype if dtype is not None else mx.float32,
    )
    return fn(packed, norms, basis, codebook, mean_arr)


@lru_cache(maxsize=128)
def _compiled_k_decompress_core(
    packed_shape: tuple,
    norms_dtype: mx.Dtype,
    rank: int,
    bits: int,
    n_levels: int,
    per_head: bool,
    head_dim: int,
    out_dtype: mx.Dtype,
):
    """Compiled (unpack + gather + pad + back-project + un-center + rescale).

    The unpack branches on ``bits`` at trace time (static), so the whole
    inverse path fuses into one graph."""

    @mx.compile
    def _fn(
        packed: mx.array,
        norms: mx.array,
        basis: mx.array,
        codebook: mx.array,
        mean: mx.array,
    ) -> mx.array:
        idx = unpack_indices(packed, bits, rank)
        if per_head:
            flat = codebook.reshape(-1)
            offs = (mx.arange(codebook.shape[0], dtype=mx.uint32) * n_levels)[
                None, :, None, None
            ]
            y = flat[idx.astype(mx.uint32) + offs]
        else:
            y = codebook[idx.astype(mx.uint32)]
        if rank < head_dim:
            pad = mx.zeros((*y.shape[:-1], head_dim - rank), dtype=y.dtype)
            y = mx.concatenate([y, pad], axis=-1)
        x = mx.matmul(y, basis)
        x = x + mean[None, :, None, :].astype(x.dtype)
        x = x * norms.astype(x.dtype)
        return x.astype(out_dtype)

    return _fn


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
    idx_chunks, norm_chunks = [], []
    S = values.shape[2]
    for s0 in range(0, S, _VQ_SEQ_CHUNK):
        vc = values[..., s0 : s0 + _VQ_SEQ_CHUNK, :]
        fn = _compiled_v_compress_core(
            tuple(vc.shape), vc.dtype, tuple(codebooks.shape)
        )
        idx, norms = fn(vc, rotation, codebooks)
        idx_chunks.append(idx)
        norm_chunks.append(norms)
    idx = mx.concatenate(idx_chunks, axis=2) if len(idx_chunks) > 1 else idx_chunks[0]
    norms = (
        mx.concatenate(norm_chunks, axis=2) if len(norm_chunks) > 1 else norm_chunks[0]
    )
    return idx, norms


@lru_cache(maxsize=128)
def _compiled_v_compress_core(x_shape: tuple, x_dtype: mx.Dtype, cb_shape: tuple):
    """Compiled (normalize + rotate + reshape + VQ assign) kernel."""
    p, _, g = cb_shape

    @mx.compile
    def _fn(
        x: mx.array, rotation: mx.array, codebooks: mx.array
    ) -> tuple[mx.array, mx.array]:
        x32 = x.astype(mx.float32)
        norms = mx.sqrt(mx.sum(x32 * x32, axis=-1, keepdims=True))
        eps = mx.array(_NORM_EPS, dtype=mx.float32)
        xn = x32 / mx.maximum(norms, eps)
        rotated = xn @ rotation.T
        sub = rotated.reshape(*rotated.shape[:-1], p, g)
        c_sq = mx.sum(codebooks * codebooks, axis=-1)  # (P, K)
        scores = c_sq - 2.0 * mx.einsum("...pg,pkg->...pk", sub, codebooks)
        idx = mx.argmin(scores, axis=-1).astype(mx.uint8)
        return idx, norms

    return _fn


def shard_decompress_values(
    idx: mx.array,
    norms: mx.array,
    rotation: mx.array,
    codebooks: mx.array,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Inverse of shard_compress_values."""
    fn = _compiled_v_decompress_core(
        tuple(idx.shape),
        norms.dtype,
        tuple(codebooks.shape),
        dtype if dtype is not None else mx.float32,
    )
    return fn(idx, norms, rotation, codebooks)


@lru_cache(maxsize=128)
def _compiled_v_decompress_core(
    idx_shape: tuple,
    norms_dtype: mx.Dtype,
    cb_shape: tuple,
    out_dtype: mx.Dtype,
):
    """Compiled (gather + reshape + un-rotate + rescale) kernel."""
    p, k, g = cb_shape

    @mx.compile
    def _fn(
        idx: mx.array, norms: mx.array, rotation: mx.array, codebooks: mx.array
    ) -> mx.array:
        flat = codebooks.reshape(p * k, g)
        flat_idx = idx.astype(mx.uint32) + mx.arange(p, dtype=mx.uint32) * k
        sub = flat[flat_idx]
        rotated = sub.reshape(*sub.shape[:-2], p * g)
        x = (rotated @ rotation) * norms.astype(mx.float32)
        return x.astype(out_dtype)

    return _fn
