"""Vectorized numpy decoders for the ggml block formats an Edge0 pack uses.

An Edge0 ("strata") pack stores its weights in llama.cpp's ggml block
quantizations rather than MLX's affine layout: routed experts are ``Q3_K``
(gate/up) and ``IQ4_NL`` (down), the dense trunk is ``Q8_0`` plus plain
floats. MLX has no kernel for any of them, so olmlx decodes them itself.

Two forms are offered for the expert formats:

* ``dequantize_*`` returns dense float32 weights.
* ``*_codes_and_scales`` returns the integer codes and the per-sub-block
  scales separately. ``weights == codes * scales`` exactly (one multiply), and
  the split form is a quarter the size of float32, which is what the runtime
  keeps resident.

Every function takes a flat ``uint8`` buffer holding whole blocks and returns
values in storage order (row-major over the tensor).
"""

from __future__ import annotations

import numpy as np

GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q8_0 = 8
GGML_Q3_K = 11
GGML_IQ4_NL = 20
GGML_I8 = 24
GGML_BF16 = 30

#: ggml type -> (values per block, bytes per block).
_BLOCK_GEOMETRY: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_BF16: (1, 2),
    GGML_I8: (1, 1),
    GGML_Q4_0: (32, 18),
    GGML_Q8_0: (32, 34),
    GGML_Q3_K: (256, 110),
    GGML_IQ4_NL: (32, 18),
}

#: ggml's non-linear 4-bit codebook (``kvalues_iq4nl``). The one definition:
#: the MLX-op decoder and the Metal kernels build theirs from it.
KVALUES_IQ4NL = (
    -127,
    -104,
    -83,
    -65,
    -49,
    -35,
    -22,
    -10,
    1,
    13,
    25,
    38,
    53,
    69,
    89,
    113,
)
_KVALUES_IQ4NL = np.array(KVALUES_IQ4NL, dtype=np.int8)

_Q3K_SHIFTS = np.arange(4, dtype=np.uint8) * 2
_Q3K_HIGH_BITS = np.arange(8, dtype=np.uint8)


def block_geometry(ggml_type: int) -> tuple[int, int]:
    """Return ``(values per block, bytes per block)`` for a ggml type."""
    try:
        return _BLOCK_GEOMETRY[ggml_type]
    except KeyError:
        raise ValueError(f"unsupported ggml type {ggml_type}") from None


def _blocks(raw, block_bytes: int) -> np.ndarray:
    buf = np.frombuffer(raw, dtype=np.uint8)
    if buf.size % block_bytes:
        raise ValueError(
            f"buffer of {buf.size} bytes is not a whole number of "
            f"{block_bytes}-byte blocks"
        )
    return buf.reshape(-1, block_bytes)


def _f16_column(blocks: np.ndarray, offset: int) -> np.ndarray:
    """Read the little-endian float16 at *offset* of every block as float32."""
    return (
        np.ascontiguousarray(blocks[:, offset : offset + 2])
        .view("<f2")
        .reshape(-1)
        .astype(np.float32)
    )


def q3_k_codes_and_scales(raw) -> tuple[np.ndarray, np.ndarray]:
    """Split ``Q3_K`` blocks into codes and sub-block scales.

    Returns ``(codes, scales)``: ``codes`` is int8 in ``[-4, 3]``, 256 per
    block; ``scales`` is float32, 16 per block (one per 16 consecutive
    values).

    Block layout (110 bytes / 256 values): ``hmask[32]`` (bit *b* of byte *l*
    is the high bit of value ``32*b + l``), ``qs[64]`` (two 32-byte halves,
    each holding four 2-bit planes), ``scales[12]`` (sixteen 6-bit scales),
    ``d`` (float16 super-block scale).
    """
    blocks = _blocks(raw, 110)
    n = blocks.shape[0]
    hmask = blocks[:, 0:32]
    qs = blocks[:, 32:96].reshape(n, 2, 1, 32)
    low = (qs >> _Q3K_SHIFTS[None, None, :, None]) & 3
    high = (hmask[:, None, :] >> _Q3K_HIGH_BITS[None, :, None]) & 1
    codes = low.astype(np.int8) + 4 * high.reshape(n, 2, 4, 32).astype(np.int8) - 4

    aux = np.ascontiguousarray(blocks[:, 96:108]).view("<u4")
    a0, a1, tmp = aux[:, 0], aux[:, 1], aux[:, 2]
    k1, k2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
    six_bit = np.stack(
        [
            (a0 & k2) | ((tmp & k1) << 4),
            (a1 & k2) | (((tmp >> 2) & k1) << 4),
            ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4),
            ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4),
        ],
        axis=1,
    ).astype("<u4")
    sub_scales = six_bit.view(np.uint8).reshape(n, 16).astype(np.float32) - 32.0
    scales = sub_scales * _f16_column(blocks, 108)[:, None]
    return codes.reshape(-1), scales.reshape(-1)


def dequantize_q3_k(raw) -> np.ndarray:
    codes, scales = q3_k_codes_and_scales(raw)
    return (codes.reshape(-1, 16).astype(np.float32) * scales[:, None]).reshape(-1)


def iq4_nl_codes_and_scales(raw) -> tuple[np.ndarray, np.ndarray]:
    """Split ``IQ4_NL`` blocks into codebook values and block scales.

    Returns ``(codes, scales)``: ``codes`` is the int8 codebook value of each
    weight, 32 per block; ``scales`` is float32, one per block.

    Block layout (18 bytes / 32 values): float16 ``d`` then ``qs[16]``; the
    low nibble of byte *j* is value *j*, the high nibble value ``16 + j``.
    """
    blocks = _blocks(raw, 18)
    qs = blocks[:, 2:18]
    nibbles = np.concatenate([qs & 0x0F, qs >> 4], axis=1)
    return _KVALUES_IQ4NL[nibbles].reshape(-1), _f16_column(blocks, 0)


def dequantize_iq4_nl(raw) -> np.ndarray:
    codes, scales = iq4_nl_codes_and_scales(raw)
    return (codes.reshape(-1, 32).astype(np.float32) * scales[:, None]).reshape(-1)


def dequantize_q4_0(raw) -> np.ndarray:
    """``Q4_0``: float16 ``d`` + 16 bytes; value = ``(nibble - 8) * d``.

    Same split-half nibble order as ``IQ4_NL``.
    """
    blocks = _blocks(raw, 18)
    qs = blocks[:, 2:18]
    nibbles = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.float32) - 8.0
    return (nibbles * _f16_column(blocks, 0)[:, None]).reshape(-1)


def dequantize_q8_0(raw) -> np.ndarray:
    """``Q8_0``: 34-byte blocks of float16 ``d`` + 32 int8 values."""
    blocks = _blocks(raw, 34)
    values = blocks[:, 2:34].view(np.int8).astype(np.float32)
    return (values * _f16_column(blocks, 0)[:, None]).reshape(-1)


def _dequantize_bf16(raw) -> np.ndarray:
    bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return bits.view(np.float32)


_DEQUANTIZERS = {
    GGML_F32: lambda raw: np.frombuffer(raw, dtype="<f4"),
    GGML_F16: lambda raw: np.frombuffer(raw, dtype="<f2").astype(np.float32),
    GGML_BF16: _dequantize_bf16,
    GGML_I8: lambda raw: np.frombuffer(raw, dtype=np.int8).astype(np.float32),
    GGML_Q4_0: dequantize_q4_0,
    GGML_Q8_0: dequantize_q8_0,
    GGML_Q3_K: dequantize_q3_k,
    GGML_IQ4_NL: dequantize_iq4_nl,
}


def dequantize(raw, ggml_type: int) -> np.ndarray:
    """Decode a flat buffer of *ggml_type* blocks to float32."""
    try:
        decode = _DEQUANTIZERS[ggml_type]
    except KeyError:
        raise ValueError(f"unsupported ggml type {ggml_type}") from None
    return decode(np.frombuffer(raw, dtype=np.uint8))
