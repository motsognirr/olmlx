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
        (indices, norms): indices as uint8 (B, n_heads, seq_len, head_dim),
                          norms as float16 (B, n_heads, seq_len, 1).
    """
    head_dim = x.shape[-1]
    codebook = get_codebook(bits, head_dim)

    # Compute and store vector norms
    norms = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
    # Normalize to unit sphere (avoid division by zero)
    x_norm = x / mx.maximum(norms, mx.array(1e-8))

    # Rotate: y = x_norm @ Πᵀ  (equivalent to Π @ x_norm per vector)
    y = x_norm @ rotation.matrix.T

    # Scalar quantize: find nearest centroid per coordinate
    # codebook shape: (n_centroids,) → broadcast against y
    # distances shape: (..., head_dim, n_centroids)
    distances = mx.abs(mx.expand_dims(y, -1) - codebook)
    indices = mx.argmin(distances, axis=-1).astype(mx.uint8)

    return indices, norms.astype(mx.float16)


def turboquant_dequantize(
    indices: mx.array,
    norms: mx.array,
    rotation: TurboQuantRotation,
    bits: int,
) -> mx.array:
    """Dequantize TurboQuant_mse encoded vectors.

    Args:
        indices: Quantized indices of shape (B, n_heads, seq_len, head_dim).
        norms: Vector norms of shape (B, n_heads, seq_len, 1).
        rotation: Rotation matrix used during quantization.
        bits: Quantization bit-width (2 or 4).

    Returns:
        Reconstructed tensor of shape (B, n_heads, seq_len, head_dim).
    """
    head_dim = indices.shape[-1]
    codebook = get_codebook(bits, head_dim)

    # Lookup centroids
    y_hat = codebook[indices.astype(mx.uint32)]

    # Inverse rotate: x_hat = y_hat @ Π
    x_hat = y_hat @ rotation.matrix

    # Rescale by original norms
    return x_hat * norms.astype(x_hat.dtype)
