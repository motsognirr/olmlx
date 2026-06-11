"""Fused decode-path attention for ShardKVCache (#377 Tier 2).

Computes middle-region Q·K and the V-weighted sum directly from the packed
form so decode never materializes the FP16 middle.  This module holds the
plain-MLX reference math and the decode assembly; the Metal kernels live in
``shardquant_kernels.py``.  The reference path is also the production
fallback for unsupported configurations — identical math, Tier-1 cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from olmlx.engine.shardquant_cache import ShardFusedKV, ShardKVCache

__all__ = [
    "fused_sdpa_decode",
    "shard_middle_scores_ref",
    "shard_middle_weighted_v_ref",
]


def shard_middle_scores_ref(qg: mx.array, cache: ShardKVCache) -> mx.array:
    """Middle-region attention scores, reference backend.

    Args:
        qg: (B, H_kv, G, D) roped queries (grouped GQA layout).
        cache: a ShardKVCache with mid_len > 0.

    Returns:
        (B, H_kv, G, mid_len) float32 unscaled scores.
    """
    k_mid, _ = cache._decompress_middle(mx.float32)
    return qg.astype(mx.float32) @ k_mid.swapaxes(-1, -2)


def shard_middle_weighted_v_ref(w: mx.array, cache: ShardKVCache) -> mx.array:
    """Middle-region V contribution with the un-rotation folded out.

    ``Σⱼ wⱼ·(ṽⱼ@R)·nⱼ = ((w ⊙ n) @ ṽ) @ R`` — validates the fold the
    V-accumulate kernel relies on.

    Args:
        w: (B, H_kv, G, mid_len) float32 softmax weights.
        cache: a ShardKVCache with mid_len > 0.

    Returns:
        (B, H_kv, G, D) float32.
    """
    from olmlx.engine.shardquant import vq_gather

    m = cache._mid_len
    codes = cache._v_mid[..., :m, :]
    norms = cache._v_mid_norms[..., :m, :]  # (B, Hk, m, 1)
    sub = vq_gather(codes, cache.v_codebooks)  # (B, Hk, m, P, g)
    vec = sub.reshape(*sub.shape[:-2], -1)  # (B, Hk, m, D) rotated space
    wn = w * norms.swapaxes(-1, -2)  # (B, Hk, G, m)
    return (wn @ vec) @ cache.v_rotation


def _middle_scores(qg: mx.array, cache: ShardKVCache, backend: str) -> mx.array:
    if backend in ("ref", "auto"):
        return shard_middle_scores_ref(qg, cache)
    raise ValueError(f"unknown backend {backend!r}")


def _middle_weighted_v(w: mx.array, cache: ShardKVCache, backend: str) -> mx.array:
    if backend in ("ref", "auto"):
        return shard_middle_weighted_v_ref(w, cache)
    raise ValueError(f"unknown backend {backend!r}")


def fused_sdpa_decode(
    queries: mx.array,
    handle: ShardFusedKV,
    scale: float,
    *,
    backend: str = "auto",
) -> mx.array:
    """Decode-step attention over [sink | middle | window] from a handle.

    Args:
        queries: (1, n_q, 1, D) roped queries (model already applied RoPE
            at the absolute position).
        handle: ShardFusedKV from ``update_and_fetch`` (implies B == 1,
            q_len == 1, mid_len > 0, exact regions non-empty).
        scale: softmax scale (same value the model passes to sdpa).
        backend: "auto" (kernels when supported, else ref) or "ref".

    Returns:
        (1, n_q, 1, D) in the query dtype.
    """
    cache = handle.cache
    B, nq, L, D = queries.shape
    Hk = handle.k_exact.shape[1]
    grp = nq // Hk
    qg = queries.astype(mx.float32).reshape(B, Hk, grp, D)

    k_e = handle.k_exact.astype(mx.float32)
    v_e = handle.v_exact.astype(mx.float32)
    s_exact = qg @ k_e.swapaxes(-1, -2)  # (B, Hk, grp, S_e)
    s_mid = _middle_scores(qg, cache, backend)  # (B, Hk, grp, m)

    s = mx.concatenate([s_exact, s_mid], axis=-1) * scale
    w = mx.softmax(s, axis=-1)
    se = s_exact.shape[-1]
    out = w[..., :se] @ v_e + _middle_weighted_v(w[..., se:], cache, backend)
    return out.reshape(B, nq, L, D).astype(queries.dtype)
