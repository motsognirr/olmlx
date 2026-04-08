"""SpectralQuant: data-driven KV cache compression via spectral analysis.

Improves on TurboQuant by using eigenvector rotations (from calibration)
instead of random rotations, enabling non-uniform bit allocation across
semantic (high-variance) and tail (low-variance) coordinate regimes.
Includes selective QJL (Johnson-Lindenstrauss) for attention score correction.
"""

import mlx.core as mx
import numpy as np

from olmlx.engine.turboquant import (
    pack_indices as _tq_pack,
    unpack_indices as _tq_unpack,
)


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack quantized indices into bytes, supporting 1-8 bit widths.

    Extends turboquant's pack_indices with 1-bit and 8-bit support.
    """
    if bits == 8:
        return indices.astype(mx.uint8)
    if bits == 1:
        # Pack 8 indices per byte — pad to next multiple of 8 first so
        # indices[..., i::8] always has shape (..., n_bytes) for every i.
        dim = indices.shape[-1]
        n_bytes = (dim + 7) // 8
        pad = n_bytes * 8 - dim
        if pad > 0:
            indices = mx.concatenate(
                [indices, mx.zeros(indices.shape[:-1] + (pad,), dtype=mx.uint8)],
                axis=-1,
            )
        result = mx.zeros(indices.shape[:-1] + (n_bytes,), dtype=mx.uint8)
        for i in range(8):
            result = result | ((indices[..., i::8] & 0x1) << i).astype(mx.uint8)
        return result
    return _tq_pack(indices, bits)


def unpack_indices(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Unpack bit-packed indices, supporting 1-8 bit widths."""
    if bits == 8:
        return packed.astype(mx.uint8)
    if bits == 1:
        parts = []
        for i in range(8):
            parts.append((packed >> i) & 0x1)
        # Stack gives (packed_dim, 8), flatten to (packed_dim * 8), truncate to dim
        interleaved = mx.stack(parts, axis=-1)
        flat_dim = packed.shape[-1] * 8
        flat = interleaved.reshape(packed.shape[:-1] + (flat_dim,))
        return flat[..., :dim].astype(mx.uint8)
    return _tq_unpack(packed, bits, dim)


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
    head_dim = x.shape[-1]
    d_tail = head_dim - d_eff

    # Compute norms in float32 to avoid overflow
    norms = mx.sqrt(mx.sum(x.astype(mx.float32) ** 2, axis=-1, keepdims=True))
    # Normalize in float32 — 1e-8 underflows to 0.0 in float16
    x_norm = (
        x.astype(mx.float32) / mx.maximum(norms, mx.array(1e-8, dtype=mx.float32))
    ).astype(x.dtype)

    # Rotate into spectral basis
    y = rotation.rotate(x_norm)

    # Split into semantic and tail regimes
    y_sem = y[..., :d_eff]
    y_tail = y[..., d_eff:]

    # Quantize each regime: find nearest centroid via vectorized broadcast
    def _quantize_regime(data, codebook, bits, dim):
        # (..., dim, 1) vs (n_centroids,) → (..., dim, n_centroids)
        dists = mx.abs(data[..., None] - codebook)
        best_idx = mx.argmin(dists, axis=-1).astype(mx.uint8)
        return pack_indices(best_idx, bits)

    packed_sem = _quantize_regime(y_sem, codebook_sem, bits_high, d_eff)
    packed_tail = _quantize_regime(y_tail, codebook_tail, bits_low, d_tail)

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

    # Unpack and lookup centroids for each regime
    idx_sem = unpack_indices(packed_sem, bits_high, d_eff)
    y_sem = codebook_sem[idx_sem.astype(mx.uint32)]

    idx_tail = unpack_indices(packed_tail, bits_low, d_tail)
    y_tail = codebook_tail[idx_tail.astype(mx.uint32)]

    # Concatenate and inverse rotate
    y_hat = mx.concatenate([y_sem, y_tail], axis=-1)
    x_hat = rotation.unrotate(y_hat)

    # Rescale by original norms
    result = x_hat * norms.astype(x_hat.dtype)
    return result.astype(dtype) if dtype is not None else result


# TODO: Wire QJLSketcher into SpectralQuantKVCache for attention score
# correction.  Currently unused — scores are computed on raw dequantized keys.


class QJLSketcher:
    """Selective Johnson-Lindenstrauss sketcher for attention score correction.

    Maintains a random sign matrix S for projecting semantic-regime residuals.
    Used to improve attention score accuracy for compressed keys.
    """

    def __init__(self, d_eff: int, n_projections: int, seed: int):
        """Initialize with random sign matrix.

        Args:
            d_eff: Effective dimensionality (semantic regime size).
            n_projections: Number of random projections.
            seed: Random seed for reproducibility.
        """
        self.d_eff = d_eff
        self.n_projections = n_projections
        rng = np.random.RandomState(seed)
        # Rademacher ±1 random matrix
        signs = rng.choice([-1.0, 1.0], size=(n_projections, d_eff)).astype(np.float32)
        self.S: mx.array = mx.array(signs)

    def compute_signs(self, residual: mx.array) -> mx.array:
        """Compute packed sign bits from semantic residuals.

        Args:
            residual: (..., d_eff) semantic-regime residuals.

        Returns:
            Packed uint8 signs of shape (..., ceil(n_projections/8)).
        """
        # Project: (..., d_eff) @ (d_eff, n_proj) → (..., n_proj)
        projected = residual @ self.S.T
        # Convert to binary: positive → 1, negative/zero → 0
        bits = (projected > 0).astype(mx.uint8)

        # Pack 8 bits per byte
        n_bytes = (self.n_projections + 7) // 8
        batch_shape = bits.shape[:-1]
        packed = mx.zeros(batch_shape + (n_bytes,), dtype=mx.uint8)
        for i in range(self.n_projections):
            byte_idx = i // 8
            bit_idx = i % 8
            packed = packed.at[..., byte_idx].add(bits[..., i] << bit_idx)
        return packed

    def correct_scores(self, q_semantic: mx.array, packed_signs: mx.array) -> mx.array:
        """Compute attention score correction from QJL signs.

        Args:
            q_semantic: (..., n_queries, d_eff) query semantic coordinates.
            packed_signs: (..., seq_len, ceil(n_proj/8)) packed key signs.

        Returns:
            Correction of shape (..., n_queries, seq_len).
        """
        # Unpack signs to ±1 floats
        sign_parts = []
        for i in range(self.n_projections):
            byte_idx = i // 8
            bit_idx = i % 8
            bit_val = (packed_signs[..., byte_idx] >> bit_idx) & 1
            sign_parts.append(bit_val.astype(mx.float32) * 2.0 - 1.0)
        # (..., seq_len, n_projections)
        signs_float = mx.stack(sign_parts, axis=-1)

        # Sketch queries: (..., n_queries, d_eff) @ (d_eff, n_proj)
        q_sketch = q_semantic @ self.S.T  # (..., n_queries, n_proj)

        # Score correction: (..., n_queries, n_proj) @ (..., n_proj, seq_len)
        scale = float(self.d_eff) / float(self.n_projections)
        correction = scale * (q_sketch @ mx.swapaxes(signs_float, -2, -1))
        return correction
