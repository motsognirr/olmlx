"""Metal kernels for the fused shard decode path (#377 Tier 2).

Two slim kernels keep DRAM traffic at the packed size:

- ``shard_k_scores``: one simdgroup per middle token, ``_K_TGT`` tokens per
  threadgroup (the per-token ``basis`` re-reads then hit L2 — the basis
  working set is Hk·D·D·4B — instead of DRAM).  Reconstructs each key in
  registers (unpack → codebook → Σρ y·basis + mean → ×norm → re-rope at
  sink_len + j) and dots it with each grouped query.
- ``shard_v_accum``: thread per (dim, q-head, head·chunk); accumulates
  weight·norm·centroid in *rotated* space over a token chunk; partials are
  summed and un-rotated outside the kernel (the orthonormal rotation folds
  out of the weighted sum).

Both read the full capacity-aligned cache buffers and loop to ``mid_len``
(runtime param) — kernels are JIT-compiled once per bit width and reused
across calls/shapes.  Lane-ownership in the K kernel requires
``head_dim % 32 == 0`` and, with rope, ``(dims // 2) % 32 == 0`` and the
non-traditional pair layout; ``kernels_supported`` gates eligibility and
callers fall back to the reference path (identical math, Tier-1 cost).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from olmlx.engine.shardquant_cache import ShardKVCache

__all__ = [
    "kernels_supported",
    "shard_middle_scores_kernel",
    "shard_middle_weighted_v_kernel",
]

# Tokens per threadgroup in the K-score kernel (simdgroups per group).
_K_TGT = 16
# Middle tokens per V-accumulate chunk (partial-sum parallelism).
_V_JCHUNK = 2048
# Hard bounds baked into the kernels' register/threadgroup layout.
_MAX_D = 256
_MAX_RANK = 128

_K_SCORES_SRC = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;
    uint j = thread_position_in_grid.y;
    uint h = thread_position_in_grid.z;
    int mid = params[0];
    int sink = params[1];
    int rank = params[2];
    int rdims = params[3];
    int G = params[4];
    int D = params[5];
    int cap = params[6];
    int PB = params[7];
    int K = params[8];

    threadgroup float ts_y[TGT][MAX_RANK];
    bool live = (int)j < mid;

    if (live) {
        const device uint8_t* crow = kcodes + ((size_t)h * cap + j) * PB;
        for (int r = (int)lane; r < rank; r += 32) {
            uint code;
            if (BITS == 8)      code = crow[r];
            else if (BITS == 4) code = (crow[r >> 1] >> ((r & 1) * 4)) & 0xF;
            else                code = (crow[r >> 2] >> ((r & 3) * 2)) & 0x3;
            ts_y[ty][r] = cb[h * K + code];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) return;

    // Reconstruct this token's key; lane owns dims d = lane + 32k.
    float kd[MAX_D / 32];
    int nloc = D / 32;
    float norm = knorms[(size_t)h * cap + j];
    const device float* brow = basis + (size_t)h * D * D;
    for (int k = 0; k < nloc; ++k) {
        int d = (int)lane + 32 * k;
        float acc = kmean[h * D + d];
        for (int r = 0; r < rank; ++r) acc += ts_y[ty][r] * brow[r * D + d];
        kd[k] = acc * norm;
    }
    // Re-rope at absolute position sink + j (non-traditional pairs
    // (d, d + rhalf); rhalf % 32 == 0 keeps both pair halves in this lane.
    // `half` is the Metal fp16 type, hence the name).
    if (rdims > 0) {
        int rhalf = rdims / 2;
        float pos = (float)(sink + (int)j);
        for (int k = 0; k < nloc; ++k) {
            int d = (int)lane + 32 * k;
            if (d < rhalf) {
                int k2 = k + rhalf / 32;
                float angle = pos * freqs[d];
                float c = metal::precise::cos(angle);
                float s = metal::precise::sin(angle);
                float a = kd[k];
                float b = kd[k2];
                kd[k] = a * c - b * s;
                kd[k2] = a * s + b * c;
            }
        }
    }
    // Dot with each grouped query head.
    for (int g = 0; g < G; ++g) {
        const device float* qrow = q + ((size_t)h * G + g) * D;
        float acc = 0.0f;
        for (int k = 0; k < nloc; ++k) acc += kd[k] * qrow[(int)lane + 32 * k];
        float total = simd_sum(acc);
        if (lane == 0) scores[((size_t)h * G + g) * mid + j] = total;
    }
"""

_V_ACCUM_SRC = r"""
    int d = (int)thread_position_in_grid.x;
    int g = (int)thread_position_in_grid.y;
    int z = (int)thread_position_in_grid.z;
    int mid = params[0];
    int G = params[1];
    int D = params[2];
    int cap = params[3];
    int P = params[4];
    int GS = params[5];
    int jchunk = params[6];
    int nch = params[7];
    int HK = params[8];
    int K = params[9];
    int woff = params[10];
    int wlen = params[11];
    if (d >= D) return;
    int h = z / nch;
    int ch = z % nch;
    int p = d / GS;
    int sub = d % GS;
    int j0 = ch * jchunk;
    int j1 = metal::min(j0 + jchunk, mid);
    const device uint8_t* codes = vcodes + (size_t)h * cap * P;
    const device float* wrow = w + ((size_t)h * G + g) * wlen + woff;
    float acc = 0.0f;
    for (int j = j0; j < j1; ++j) {
        uint code = codes[(size_t)j * P + p];
        acc += wrow[j] * vnorms[(size_t)h * cap + j]
             * vcb[((size_t)p * K + code) * GS + sub];
    }
    partials[((((size_t)ch * HK + h) * G + g) * (size_t)D) + d] = acc;
"""


@lru_cache(maxsize=8)
def _k_scores_kernel(bits: int):
    return mx.fast.metal_kernel(
        name=f"olmlx_shard_k_scores_b{bits}",
        input_names=["q", "kcodes", "knorms", "basis", "cb", "kmean", "freqs", "params"],
        output_names=["scores"],
        source=_K_SCORES_SRC,
    )


@lru_cache(maxsize=1)
def _v_accum_kernel():
    return mx.fast.metal_kernel(
        name="olmlx_shard_v_accum",
        input_names=["w", "vcodes", "vnorms", "vcb", "params"],
        output_names=["partials"],
        source=_V_ACCUM_SRC,
    )


def kernels_supported(cache: ShardKVCache, head_dim: int) -> bool:
    """Eligibility gate for the Metal kernels; callers fall back to the
    reference path when False (identical math, Tier-1 cost)."""
    if not mx.metal.is_available():
        return False
    if mx.default_device().type != mx.DeviceType.gpu:
        return False
    if head_dim % 32 != 0 or head_dim > _MAX_D:
        return False
    if cache.k_rank > _MAX_RANK:
        return False
    if cache.k_bits not in (2, 4, 8):
        return False
    spec = cache.rope_spec
    if spec is not None:
        if spec.traditional:
            return False
        if spec.dims > head_dim or (spec.dims // 2) % 32 != 0:
            return False
    return True


def _k_codebook_2d(cache: ShardKVCache) -> mx.array:
    """Per-head (H, K) float32 codebook (legacy shared 1-D broadcast),
    materialized once per cache and reused across decode steps."""
    cb = getattr(cache, "_fused_k_cb2d", None)
    if cb is None:
        cb = cache.k_codebook.astype(mx.float32)
        if cb.ndim == 1:
            h = cache.k_basis.shape[0]
            cb = mx.broadcast_to(cb[None, :], (h, cb.shape[0]))
        cb = cb + mx.zeros_like(cb)  # force a contiguous materialization
        cache._fused_k_cb2d = cb
    return cb


def _k_mean_2d(cache: ShardKVCache) -> mx.array:
    mean = getattr(cache, "_fused_k_mean", None)
    if mean is None:
        h, d = cache.k_basis.shape[0], cache.k_basis.shape[-1]
        mean = (
            cache.k_mean.astype(mx.float32)
            if cache.k_mean is not None
            else mx.zeros((h, d), dtype=mx.float32)
        )
        cache._fused_k_mean = mean
    return mean


def shard_middle_scores_kernel(qg: mx.array, cache: ShardKVCache) -> mx.array:
    """Kernel-backed equivalent of ``shard_middle_scores_ref``.

    Args:
        qg: (1, H_kv, G, D) roped queries (any float dtype).
        cache: ShardKVCache with mid_len > 0.

    Returns:
        (1, H_kv, G, mid_len) float32 unscaled scores.
    """
    _, hk, g, d = qg.shape
    mid = cache._mid_len
    cap = cache._k_mid.shape[2]
    pb = cache._k_mid.shape[-1]
    spec = cache.rope_spec
    rdims = spec.dims if spec is not None else 0
    freqs = (
        spec.freqs.astype(mx.float32)
        if spec is not None
        else mx.zeros((1,), dtype=mx.float32)
    )
    cb = _k_codebook_2d(cache)
    params = mx.array(
        [mid, cache._sink_len(), cache.k_rank, rdims, g, d, cap, pb, cb.shape[-1]],
        dtype=mx.int32,
    )
    mid_pad = ((mid + _K_TGT - 1) // _K_TGT) * _K_TGT
    out = _k_scores_kernel(cache.k_bits)(
        inputs=[
            qg.astype(mx.float32).reshape(hk, g, d),
            cache._k_mid[0],
            cache._k_mid_norms[0, :, :, 0],
            cache.k_basis.astype(mx.float32),
            cb,
            _k_mean_2d(cache),
            freqs,
            params,
        ],
        template=[
            ("BITS", cache.k_bits),
            ("TGT", _K_TGT),
            ("MAX_RANK", _MAX_RANK),
            ("MAX_D", _MAX_D),
        ],
        grid=(32, mid_pad, hk),
        threadgroup=(32, _K_TGT, 1),
        output_shapes=[(hk, g, mid)],
        output_dtypes=[mx.float32],
    )[0]
    return out[None]


def shard_middle_weighted_v_kernel(
    w_full: mx.array, woff: int, cache: ShardKVCache
) -> mx.array:
    """Kernel-backed equivalent of ``shard_middle_weighted_v_ref``.

    Takes the *full* softmax row plus the middle-region offset (instead of
    a sliced view) so no per-step contiguous copy of the weights is needed.

    Args:
        w_full: (1, H_kv, G, S_e + mid_len) float32 softmax weights.
        woff: column where the middle region starts (== S_e).
        cache: ShardKVCache with mid_len > 0.

    Returns:
        (1, H_kv, G, D) float32 middle V contribution (un-rotated).
    """
    _, hk, g, wlen = w_full.shape
    mid = cache._mid_len
    cap = cache._v_mid.shape[2]
    p, k, gs = cache.v_codebooks.shape
    d = p * gs
    nch = (mid + _V_JCHUNK - 1) // _V_JCHUNK
    params = mx.array(
        [mid, g, d, cap, p, gs, _V_JCHUNK, nch, hk, k, woff, wlen],
        dtype=mx.int32,
    )
    partials = _v_accum_kernel()(
        inputs=[
            w_full.reshape(hk, g, wlen),
            cache._v_mid[0],
            cache._v_mid_norms[0, :, :, 0],
            cache.v_codebooks.astype(mx.float32),
            params,
        ],
        grid=(d, g, hk * nch),
        threadgroup=(min(d, 256), 1, 1),
        output_shapes=[(nch, hk, g, d)],
        output_dtypes=[mx.float32],
    )[0]
    out_rot = partials.sum(axis=0)  # (Hk, G, D), rotated space
    return (out_rot @ cache.v_rotation)[None]
