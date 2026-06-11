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

from olmlx.engine.spectralquant import pack_indices, unpack_indices

__all__ = [
    "RopeSpec",
    "detect_rope_spec",
    "rope_transform",
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
