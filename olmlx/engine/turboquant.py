"""TurboQuant: data-oblivious vector quantization for KV cache compression.

Implements TurboQuant_mse from https://arxiv.org/abs/2504.19874.
Algorithm: random rotation → scalar quantization per coordinate → inverse rotation.
"""

import mlx.core as mx
import numpy as np

# Standard Gaussian N(0,1) Lloyd-Max centroids.
# For dimension d, scale by 1/sqrt(d) to get the codebook for the
# Beta distribution that arises from random rotation on the unit sphere.
GAUSSIAN_CODEBOOKS: dict[int, list[float]] = {
    2: [
        -1.5104176085,
        -0.4527800346,
        0.4527800346,
        1.5104176085,
    ],
    4: [
        -2.732589571,
        -2.0690172265,
        -1.618046386,
        -1.2562311973,
        -0.9423404565,
        -0.6567591185,
        -0.3880482995,
        -0.1283950299,
        0.1283950299,
        0.3880482995,
        0.6567591185,
        0.9423404565,
        1.2562311973,
        1.618046386,
        2.0690172265,
        2.732589571,
    ],
}

# Cache scaled codebooks keyed by (bits, dim)
_codebook_cache: dict[tuple[int, int], mx.array] = {}


def get_codebook(bits: int, dim: int) -> mx.array:
    """Return Lloyd-Max codebook scaled for the given dimension."""
    if bits not in GAUSSIAN_CODEBOOKS:
        raise ValueError(
            f"TurboQuant supports {sorted(GAUSSIAN_CODEBOOKS.keys())}-bit, got {bits}"
        )
    key = (bits, dim)
    if key not in _codebook_cache:
        centroids = mx.array(GAUSSIAN_CODEBOOKS[bits], dtype=mx.float32)
        _codebook_cache[key] = centroids / mx.sqrt(mx.array(float(dim)))
    return _codebook_cache[key]


class TurboQuantRotation:
    """Per-layer orthogonal rotation matrix via QR decomposition."""

    def __init__(self, head_dim: int, seed: int):
        rng = np.random.RandomState(seed)
        random_matrix = rng.randn(head_dim, head_dim).astype(np.float32)
        q, _ = np.linalg.qr(random_matrix)
        self.matrix: mx.array = mx.array(q)


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack quantized indices into bytes.

    4-bit: 2 indices per byte (low/high nibble).
    2-bit: 4 indices per byte.
    """
    if bits == 4:
        # Pack pairs: low nibble = even indices, high nibble = odd indices
        low = indices[..., 0::2] & 0xF
        high = (indices[..., 1::2] & 0xF) << 4
        return (low | high).astype(mx.uint8)
    elif bits == 2:
        # Pack quads: bits 0-1 = idx[0], bits 2-3 = idx[1], etc.
        i0 = indices[..., 0::4] & 0x3
        i1 = (indices[..., 1::4] & 0x3) << 2
        i2 = (indices[..., 2::4] & 0x3) << 4
        i3 = (indices[..., 3::4] & 0x3) << 6
        return (i0 | i1 | i2 | i3).astype(mx.uint8)
    raise ValueError(f"Unsupported bits={bits}")


def unpack_indices(packed: mx.array, bits: int, head_dim: int) -> mx.array:
    """Unpack bit-packed indices back to one-per-element uint8."""
    if bits == 4:
        low = packed & 0xF
        high = (packed >> 4) & 0xF
        # Interleave via stack + reshape: [low0, high0, low1, high1, ...]
        parts = mx.stack([low, high], axis=-1)  # (..., packed_dim, 2)
        return parts.reshape(packed.shape[:-1] + (head_dim,)).astype(mx.uint8)
    elif bits == 2:
        i0 = packed & 0x3
        i1 = (packed >> 2) & 0x3
        i2 = (packed >> 4) & 0x3
        i3 = (packed >> 6) & 0x3
        parts = mx.stack([i0, i1, i2, i3], axis=-1)  # (..., packed_dim, 4)
        return parts.reshape(packed.shape[:-1] + (head_dim,)).astype(mx.uint8)
    raise ValueError(f"Unsupported bits={bits}")


def turboquant_quantize(
    x: mx.array,
    rotation: TurboQuantRotation,
    bits: int,
) -> tuple[mx.array, mx.array]:
    """Quantize vectors using TurboQuant_mse.

    Args:
        x: Input tensor of shape (B, n_heads, seq_len, head_dim).
        rotation: Rotation matrix for this layer.
        bits: Quantization bit-width (2 or 4).

    Returns:
        (packed_indices, norms): bit-packed indices as uint8, norms as float32
            (float32 to avoid overflow for vectors with L2-norm > 65504).
    """
    head_dim = x.shape[-1]
    codebook = get_codebook(bits, head_dim)

    # Compute norms in float32 to avoid overflow (float16 max ~65504)
    norms = mx.sqrt(mx.sum(x.astype(mx.float32) ** 2, axis=-1, keepdims=True))
    # Normalize to unit sphere (avoid division by zero)
    x_norm = x / mx.maximum(norms.astype(x.dtype), mx.array(1e-8, dtype=x.dtype))

    # Rotate: y = x_norm @ Πᵀ  (equivalent to Π @ x_norm per vector)
    y = x_norm @ rotation.matrix.T

    # Scalar quantize: find nearest centroid per coordinate.
    # Iterate over centroids to avoid materializing the full distances tensor
    # which would be (B, heads, seq, head_dim, n_centroids) — OOM on long prefills.
    best_idx = mx.zeros(y.shape, dtype=mx.uint8)
    best_dist = mx.full(y.shape, float("inf"))
    for ci in range(len(codebook)):
        d = mx.abs(y - codebook[ci])
        better = d < best_dist
        best_idx = mx.where(better, mx.array(ci, dtype=mx.uint8), best_idx)
        best_dist = mx.minimum(best_dist, d)

    return pack_indices(best_idx, bits), norms


def turboquant_dequantize(
    packed_indices: mx.array,
    norms: mx.array,
    rotation: TurboQuantRotation,
    bits: int,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Dequantize TurboQuant_mse encoded vectors.

    Args:
        packed_indices: Bit-packed quantized indices.
        norms: Vector norms of shape (B, n_heads, seq_len, 1).
        rotation: Rotation matrix used during quantization.
        bits: Quantization bit-width (2 or 4).

    Returns:
        Reconstructed tensor of shape (B, n_heads, seq_len, head_dim).
    """
    head_dim = rotation.matrix.shape[0]
    codebook = get_codebook(bits, head_dim)

    # Unpack indices and lookup centroids
    indices = unpack_indices(packed_indices, bits, head_dim)
    y_hat = codebook[indices.astype(mx.uint32)]

    # Inverse rotate: x_hat = y_hat @ Π
    x_hat = y_hat @ rotation.matrix

    # Rescale by original norms
    result = x_hat * norms.astype(x_hat.dtype)
    return result.astype(dtype) if dtype is not None else result
