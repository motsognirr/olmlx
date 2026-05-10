"""SpectralQuant: data-driven KV cache compression via spectral analysis.

Improves on TurboQuant by using eigenvector rotations (from calibration)
instead of random rotations, enabling non-uniform bit allocation across
semantic (high-variance) and tail (low-variance) coordinate regimes.
"""

from functools import lru_cache, partial

import mlx.core as mx
import numpy as np

from olmlx.engine.turboquant import (
    pack_indices as _tq_pack,
    unpack_indices as _tq_unpack,
)


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack quantized indices into bytes, supporting 1-8 bit widths.

    Extends turboquant's pack_indices with 1-bit and 8-bit support, and
    pads odd tail dimensions (which arise when SpectralQuant allocates an
    odd ``head_dim - d_eff``) so 4-bit and 2-bit packing don't trip over
    the even/multiple-of-4 assumption baked into turboquant's packer.
    """
    if bits == 8:
        return indices.astype(mx.uint8)
    if bits == 1:
        # Pad to next multiple of 8 so every i::8 stride produces the
        # same length; otherwise the bitwise accumulation broadcasts wrongly.
        dim = indices.shape[-1]
        n_bytes = (dim + 7) // 8
        pad = n_bytes * 8 - dim
        if pad > 0:
            indices = mx.concatenate(
                [indices, mx.zeros(indices.shape[:-1] + (pad,), dtype=indices.dtype)],
                axis=-1,
            )
        result = mx.zeros(indices.shape[:-1] + (n_bytes,), dtype=mx.uint8)
        for i in range(8):
            result = result | ((indices[..., i::8] & 0x1) << i).astype(mx.uint8)
        return result
    # 2- and 4-bit delegate to turboquant's packer, but pad to the factor
    # (4 for 2-bit, 2 for 4-bit) so odd tail dimensions work.
    factor = 8 // bits
    dim = indices.shape[-1]
    pad = (-dim) % factor
    if pad:
        indices = mx.concatenate(
            [indices, mx.zeros(indices.shape[:-1] + (pad,), dtype=indices.dtype)],
            axis=-1,
        )
    return _tq_pack(indices, bits)


def unpack_indices(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Unpack bit-packed indices, supporting 1-8 bit widths."""
    if bits == 8:
        return packed.astype(mx.uint8)
    if bits == 1:
        parts = []
        for i in range(8):
            parts.append((packed >> i) & 0x1)
        interleaved = mx.stack(parts, axis=-1)
        flat_dim = packed.shape[-1] * 8
        flat = interleaved.reshape(packed.shape[:-1] + (flat_dim,))
        return flat[..., :dim].astype(mx.uint8)
    # Match the padding applied in pack_indices — unpack the padded dim then
    # slice off the trailing padding so callers see the original dim.
    factor = 8 // bits
    padded_dim = ((dim + factor - 1) // factor) * factor
    unpadded = _tq_unpack(packed, bits, padded_dim)
    if padded_dim != dim:
        unpadded = unpadded[..., :dim]
    return unpadded


class SpectralRotation:
    """Eigenvector-based rotation for spectral quantization.

    Unlike TurboQuantRotation (random QR), this uses data-driven
    eigenvectors from calibration to concentrate variance in leading
    coordinates.
    """

    def __init__(self, V: mx.array):
        """Initialize with eigenvector matrix V (head_dim, head_dim)."""
        self.V: mx.array = V
        self.V_T: mx.array = V.T

    def rotate(self, x: mx.array) -> mx.array:
        """Project into spectral basis: x @ V^T."""
        return x @ self.V_T

    def unrotate(self, x: mx.array) -> mx.array:
        """Reconstruct from spectral basis: x @ V."""
        return x @ self.V


def allocate_bits(d_eff: int, head_dim: int, avg_bits: int) -> tuple[int, int]:
    """Find (b_high, b_low) minimizing budget slack.

    Solves: d_eff * b_high + (head_dim - d_eff) * b_low ≈ head_dim * avg_bits
    Subject to: b_high >= b_low >= 1, both in [1, 8].

    Returns:
        (b_high, b_low): bits for semantic and tail regimes.
    """
    budget = head_dim * avg_bits
    d_tail = head_dim - d_eff

    best_pair = (avg_bits, avg_bits)  # fallback: uniform
    best_slack = float("inf")

    for b_high in range(8, 0, -1):
        for b_low in range(min(b_high, 8), 0, -1):
            total = d_eff * b_high + d_tail * b_low
            slack = abs(total - budget)
            if slack < best_slack:
                best_slack = slack
                best_pair = (b_high, b_low)

    return best_pair


def fit_codebook(
    data: mx.array,
    bits: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> mx.array:
    """Fit a Lloyd-Max codebook to 1D data.

    Args:
        data: Flattened 1D data array.
        bits: Number of bits (codebook will have 2^bits centroids).
        max_iter: Maximum iterations.
        tol: Convergence tolerance for centroid movement.

    Returns:
        Sorted codebook of shape (2^bits,).
    """
    n_levels = 1 << bits
    data = data.astype(mx.float32)
    data_np = np.array(data)

    # Initialize centroids uniformly across data range
    lo, hi = float(data_np.min()), float(data_np.max())
    centroids = np.linspace(lo, hi, n_levels).astype(np.float32)

    for _ in range(max_iter):
        # Assignment: find nearest centroid for each data point
        # Do this in numpy to avoid large MLX materialization
        dists = np.abs(data_np[:, None] - centroids[None, :])
        assignments = dists.argmin(axis=1)

        # Update: recompute each centroid as conditional mean
        new_centroids = centroids.copy()
        for k in range(n_levels):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = data_np[mask].mean()

        # Check convergence
        max_change = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if max_change < tol:
            break

    centroids.sort()
    return mx.array(centroids)


# Above this many centroids, the unrolled where-loop becomes slower than
# building the (..., dim, n_centroids) distance tensor and reducing with
# ``argmin`` — measured on Apple Silicon for small (decode) shapes. For larger
# (prefill) shapes the broadcast tensor is too big and the loop wins again,
# so the dispatch in ``_compiled_spectral_quantize_full`` also weighs shape.
_LOOP_ARGMIN_CUTOFF = 32

# Per-element cap on the (..., dim, n_centroids) broadcast tensor. Above this,
# fall back to the loop even when ``n_levels`` is large. Tuned to keep
# intermediate float32 allocations under ~256 MB on Apple Silicon.
_ARGMIN_BROADCAST_CAP = 64 * 1024 * 1024


def _make_codebook_argmin(n_levels: int, dim_outer: int):
    """Pick loop-unroll vs broadcast-argmin once per compiled trace.

    Returns a closure that takes ``(y, codebook)`` and returns ``best_idx``.
    The branch fires at trace time (Python-side); the produced graph is
    already specialized for the chosen strategy.
    """
    use_loop = (
        n_levels <= _LOOP_ARGMIN_CUTOFF
        or dim_outer * n_levels > _ARGMIN_BROADCAST_CAP
    )
    if use_loop:
        def _loop(y: mx.array, codebook: mx.array) -> mx.array:
            best_dist = mx.abs(y - codebook[0])
            best_idx = mx.array(0, dtype=mx.uint8)
            for ci in range(1, n_levels):
                d = mx.abs(y - codebook[ci])
                better = d < best_dist
                best_idx = mx.where(better, ci, best_idx).astype(mx.uint8)
                best_dist = mx.where(better, d, best_dist)
            return best_idx
        return _loop

    def _vec(y: mx.array, codebook: mx.array) -> mx.array:
        dists = mx.abs(y[..., None] - codebook)
        return mx.argmin(dists, axis=-1).astype(mx.uint8)
    return _vec


@lru_cache(maxsize=64)
def _compiled_spectral_quantize_full(
    n_levels_sem: int,
    n_levels_tail: int,
    x_shape: tuple,
    x_dtype: mx.Dtype,
    d_eff: int,
):
    """Compiled (norm + rotate + split + per-regime argmin) kernel.

    Cached per full input signature because mlx 0.31's compile bakes leading
    tensor dims into the trace.
    """
    head_dim = x_shape[-1]
    d_tail = head_dim - d_eff
    # ``outer`` = product of leading dims times the regime's slice width;
    # used to estimate the size of the broadcast distance tensor and pick
    # loop vs argmin per regime.
    outer = 1
    for d in x_shape[:-1]:
        outer *= d
    sem_argmin = _make_codebook_argmin(n_levels_sem, outer * d_eff)
    tail_argmin = _make_codebook_argmin(n_levels_tail, outer * d_tail)

    @mx.compile
    def _fn(
        x: mx.array,
        rotation_T: mx.array,
        codebook_sem: mx.array,
        codebook_tail: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        x32 = x.astype(mx.float32)
        norms = mx.sqrt(mx.sum(x32 * x32, axis=-1, keepdims=True))
        x_norm = (x32 / mx.maximum(norms, 1e-8)).astype(x.dtype)
        y = x_norm @ rotation_T

        y_sem = y[..., :d_eff]
        y_tail = y[..., d_eff:]
        idx_sem = sem_argmin(y_sem, codebook_sem)
        idx_tail = tail_argmin(y_tail, codebook_tail)
        return idx_sem, idx_tail, norms

    return _fn


def spectral_quantize(
    x: mx.array,
    rotation: SpectralRotation,
    codebook_sem: mx.array,
    codebook_tail: mx.array,
    d_eff: int,
    bits_high: int,
    bits_low: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize vectors using spectral rotation + non-uniform bit allocation.

    Args:
        x: Input tensor (B, n_heads, seq_len, head_dim).
        rotation: Spectral rotation (from calibration eigenvectors).
        codebook_sem: Codebook for semantic regime (2^bits_high centroids).
        codebook_tail: Codebook for tail regime (2^bits_low centroids).
        d_eff: Effective dimensionality (semantic/tail split point).
        bits_high: Bits for semantic regime.
        bits_low: Bits for tail regime.

    Returns:
        (packed_sem, packed_tail, norms): bit-packed indices + float32 norms.
    """
    fn = _compiled_spectral_quantize_full(
        1 << bits_high, 1 << bits_low, x.shape, x.dtype, d_eff
    )
    idx_sem, idx_tail, norms = fn(x, rotation.V_T, codebook_sem, codebook_tail)
    packed_sem = pack_indices(idx_sem, bits_high)
    packed_tail = pack_indices(idx_tail, bits_low)
    return packed_sem, packed_tail, norms


def spectral_dequantize(
    packed_sem: mx.array,
    packed_tail: mx.array,
    norms: mx.array,
    rotation: SpectralRotation,
    codebook_sem: mx.array,
    codebook_tail: mx.array,
    d_eff: int,
    bits_high: int,
    bits_low: int,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Dequantize spectral-quantized vectors.

    Args:
        packed_sem: Bit-packed semantic indices.
        packed_tail: Bit-packed tail indices.
        norms: Vector norms (B, n_heads, seq_len, 1).
        rotation: Spectral rotation used during quantization.
        codebook_sem: Semantic codebook.
        codebook_tail: Tail codebook.
        d_eff: Effective dimensionality.
        bits_high: Bits for semantic regime.
        bits_low: Bits for tail regime.
        dtype: Output dtype (default: float32).

    Returns:
        Reconstructed tensor of same shape as original input.
    """
    head_dim = rotation.V.shape[0]
    d_tail = head_dim - d_eff

    # Unpacking branches on bits and includes Python control flow; keep it
    # outside the compiled inner kernel.
    idx_sem = unpack_indices(packed_sem, bits_high, d_eff)
    idx_tail = unpack_indices(packed_tail, bits_low, d_tail)

    fn = _compiled_spectral_dequant_core(
        idx_sem.shape, idx_tail.shape, norms.dtype
    )
    result = fn(idx_sem, idx_tail, norms, rotation.V, codebook_sem, codebook_tail)
    return result.astype(dtype) if dtype is not None else result


@lru_cache(maxsize=64)
def _compiled_spectral_dequant_core(
    sem_shape: tuple, tail_shape: tuple, norms_dtype: mx.Dtype
):
    """Compiled (gather both regimes + concat + inverse rotate + rescale).

    Cached per shape signature for the same reason as the quantize kernel.
    """

    @mx.compile
    def _fn(
        idx_sem: mx.array,
        idx_tail: mx.array,
        norms: mx.array,
        rotation_V: mx.array,
        codebook_sem: mx.array,
        codebook_tail: mx.array,
    ) -> mx.array:
        y_sem = codebook_sem[idx_sem.astype(mx.uint32)]
        y_tail = codebook_tail[idx_tail.astype(mx.uint32)]
        y_hat = mx.concatenate([y_sem, y_tail], axis=-1)
        x_hat = y_hat @ rotation_V
        return x_hat * norms.astype(x_hat.dtype)

    return _fn
