"""Exact repacking of ggml block quantizations as MLX affine quantization.

``Q8_0`` is ``d * q`` with int8 ``q`` — ``d * u - 128 d`` for ``u = q + 128``
— and ``Q4_0`` is ``d * (n - 8)``: both are MLX's affine layout at group size
32, so these weights are repacked rather than re-quantized. The one place
precision is lost is where a consumer needs the per-block scale in bfloat16
(the model's dtype); :func:`q8_0_to_bf16_affine` owns the rule for that so it
exists once.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from olmlx.engine.edge0 import ggml_quants


def bf16(values: np.ndarray) -> mx.array:
    """Float values as a bfloat16 MLX array."""
    return mx.array(np.ascontiguousarray(values, dtype=np.float32)).astype(mx.bfloat16)


def blocks_to_affine(raw: np.ndarray, ggml_type: int, rows: int):
    """Split ``Q8_0`` / ``Q4_0`` rows into MLX affine (codes, scales, biases).

    Returned arrays keep a ``[rows, blocks, ...]`` layout (codes as bytes,
    scales/biases float32) so callers can regroup rows or blocks before
    flattening to MLX's packed ``uint32`` words.
    """
    if ggml_type == ggml_quants.GGML_Q8_0:
        blocks = raw.reshape(rows, -1, 34)
        codes = blocks[..., 2:] ^ np.uint8(0x80)
        offset = -128.0
    elif ggml_type == ggml_quants.GGML_Q4_0:
        blocks = raw.reshape(rows, -1, 18)
        qs = blocks[..., 2:]
        # ggml: byte j = element j | element j+16 << 4. MLX: nibble i = element i.
        low, high = qs & 0x0F, qs >> 4
        codes = np.concatenate(
            [
                low[..., 0::2] | (low[..., 1::2] << 4),
                high[..., 0::2] | (high[..., 1::2] << 4),
            ],
            axis=-1,
        )
        offset = -8.0
    else:
        raise ValueError(f"unsupported ggml type {ggml_type} for affine repacking")
    scales = (
        np.ascontiguousarray(blocks[..., :2]).view("<f2")[..., 0].astype(np.float32)
    )
    return np.ascontiguousarray(codes), scales, scales * np.float32(offset)


def packed_q8_0_to_bf16_affine(
    codes: np.ndarray, scales: np.ndarray, prefix: str
) -> dict[str, mx.array]:
    """MLX 8-bit parameters ``{prefix}.weight/.scales/.biases`` in bfloat16.

    *codes* / *scales* are :func:`blocks_to_affine` output (possibly
    reordered). The model runs in bfloat16, so the block scales are rounded
    to it — and the bias is derived from the *rounded* scale (x128 is exact),
    which keeps every weight ``scale * (u - 128)``: a relative scale error
    only, never a constant offset on all 32 weights of a block.
    """
    rows = codes.shape[0]
    scales_bf16 = bf16(scales)
    return {
        f"{prefix}.weight": mx.array(codes.reshape(rows, -1).view("<u4")),
        f"{prefix}.scales": scales_bf16,
        f"{prefix}.biases": scales_bf16 * -128.0,
    }
