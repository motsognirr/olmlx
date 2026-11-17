"""Tests for olmlx.engine.edge0.ggml_quants: vectorized ggml block decoders.

Each decoder is checked against a scalar transliteration of ggml's reference
``dequantize_row_*`` on random blocks (every bit pattern is a valid block).
"""

import numpy as np
import pytest

from olmlx.engine.edge0 import ggml_quants as gq

_KVALUES_IQ4NL = (
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


def _random_blocks(rng, n_blocks, block_bytes, d_offset):
    raw = rng.integers(0, 256, size=(n_blocks, block_bytes), dtype=np.uint8)
    d = rng.uniform(-0.05, 0.05, size=n_blocks).astype(np.float16)
    raw[:, d_offset : d_offset + 2] = d.view(np.uint8).reshape(n_blocks, 2)
    return raw.reshape(-1)


def _ref_q3_k(raw):
    """Scalar port of ggml ``dequantize_row_q3_K``."""
    out = []
    kmask1, kmask2 = 0x03030303, 0x0F0F0F0F
    for block in np.asarray(raw).reshape(-1, 110):
        hm = block[0:32]
        qs = block[32:96]
        d_all = float(block[108:110].view(np.float16)[0])
        aux = [int(v) for v in block[96:108].view("<u4")] + [0]
        tmp = aux[2]
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
        scales = np.array(aux, dtype="<u4").view(np.int8)
        idx, m, q = 0, 1, 0
        for _chunk in range(2):
            shift = 0
            for _j in range(4):
                for half in (0, 16):
                    dl = d_all * (int(scales[idx]) - 32)
                    idx += 1
                    for low in range(16):
                        code = (int(qs[q + low + half]) >> shift) & 3
                        high = 0 if int(hm[low + half]) & m else 4
                        out.append(dl * (code - high))
                shift += 2
                m <<= 1
            q += 32
    return np.array(out, dtype=np.float32)


def _ref_iq4_nl(raw):
    out = []
    for block in np.asarray(raw).reshape(-1, 18):
        d = float(block[0:2].view(np.float16)[0])
        qs = block[2:18]
        y = [0.0] * 32
        for j in range(16):
            y[j] = d * _KVALUES_IQ4NL[int(qs[j]) & 0xF]
            y[j + 16] = d * _KVALUES_IQ4NL[int(qs[j]) >> 4]
        out.extend(y)
    return np.array(out, dtype=np.float32)


def _ref_q8_0(raw):
    out = []
    for block in np.asarray(raw).reshape(-1, 34):
        d = float(block[0:2].view(np.float16)[0])
        out.extend(d * float(q) for q in block[2:34].view(np.int8))
    return np.array(out, dtype=np.float32)


class TestQ3K:
    def test_matches_scalar_reference(self):
        raw = _random_blocks(np.random.default_rng(0), 7, 110, 108)
        got = gq.dequantize_q3_k(raw)
        assert got.dtype == np.float32
        np.testing.assert_allclose(got, _ref_q3_k(raw), rtol=1e-6, atol=0)

    def test_codes_and_scales_reconstruct_the_weights(self):
        # The split form (integer codes + per-16 sub-block scales) is what the
        # runtime keeps on the GPU; it must multiply back to the dense decode.
        raw = _random_blocks(np.random.default_rng(1), 5, 110, 108)
        codes, scales = gq.q3_k_codes_and_scales(raw)
        assert codes.dtype == np.int8 and codes.shape == (5 * 256,)
        assert codes.min() >= -4 and codes.max() <= 3
        assert scales.dtype == np.float32 and scales.shape == (5 * 16,)
        dense = (codes.reshape(-1, 16) * scales[:, None]).reshape(-1)
        np.testing.assert_allclose(dense, _ref_q3_k(raw), rtol=1e-6, atol=0)

    def test_rejects_partial_blocks(self):
        with pytest.raises(ValueError, match="110"):
            gq.dequantize_q3_k(np.zeros(109, dtype=np.uint8))


class TestIQ4NL:
    def test_matches_scalar_reference(self):
        raw = _random_blocks(np.random.default_rng(2), 9, 18, 0)
        got = gq.dequantize_iq4_nl(raw)
        assert got.dtype == np.float32
        np.testing.assert_allclose(got, _ref_iq4_nl(raw), rtol=1e-6, atol=0)

    def test_codes_and_scales_reconstruct_the_weights(self):
        raw = _random_blocks(np.random.default_rng(3), 4, 18, 0)
        codes, scales = gq.iq4_nl_codes_and_scales(raw)
        assert codes.dtype == np.int8 and codes.shape == (4 * 32,)
        assert scales.shape == (4,)
        dense = (codes.reshape(-1, 32) * scales[:, None]).reshape(-1)
        np.testing.assert_allclose(dense, _ref_iq4_nl(raw), rtol=1e-6, atol=0)

    def test_rejects_partial_blocks(self):
        with pytest.raises(ValueError, match="18"):
            gq.dequantize_iq4_nl(np.zeros(19, dtype=np.uint8))


def _ref_q4_0(raw):
    out = []
    for block in np.asarray(raw).reshape(-1, 18):
        d = float(block[0:2].view(np.float16)[0])
        y = [0.0] * 32
        for j in range(16):
            y[j] = d * ((int(block[2 + j]) & 0xF) - 8)
            y[j + 16] = d * ((int(block[2 + j]) >> 4) - 8)
        out.extend(y)
    return np.array(out, dtype=np.float32)


class TestQ40:
    def test_matches_scalar_reference(self):
        raw = _random_blocks(np.random.default_rng(6), 6, 18, 0)
        np.testing.assert_allclose(
            gq.dequantize_q4_0(raw), _ref_q4_0(raw), rtol=1e-6, atol=0
        )


class TestQ80:
    def test_matches_scalar_reference(self):
        raw = _random_blocks(np.random.default_rng(4), 6, 34, 0)
        np.testing.assert_allclose(
            gq.dequantize_q8_0(raw), _ref_q8_0(raw), rtol=1e-6, atol=0
        )


class TestDispatch:
    def test_block_geometry(self):
        assert gq.block_geometry(gq.GGML_Q3_K) == (256, 110)
        assert gq.block_geometry(gq.GGML_IQ4_NL) == (32, 18)
        assert gq.block_geometry(gq.GGML_Q8_0) == (32, 34)
        assert gq.block_geometry(gq.GGML_Q4_0) == (32, 18)
        assert gq.block_geometry(gq.GGML_F32) == (1, 4)
        assert gq.block_geometry(gq.GGML_BF16) == (1, 2)

    def test_dequantize_by_type(self):
        raw = _random_blocks(np.random.default_rng(5), 2, 110, 108)
        np.testing.assert_array_equal(
            gq.dequantize(raw, gq.GGML_Q3_K), gq.dequantize_q3_k(raw)
        )

    def test_float_types_pass_through(self):
        values = np.array([1.5, -2.25, 0.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(
            gq.dequantize(values.view(np.uint8), gq.GGML_F32), values
        )
        np.testing.assert_array_equal(
            gq.dequantize(values.astype(np.float16).view(np.uint8), gq.GGML_F16),
            values,
        )
        # bfloat16 is the top half of the float32 bit pattern.
        bf16 = (values.view(np.uint32) >> 16).astype(np.uint16)
        np.testing.assert_array_equal(
            gq.dequantize(bf16.view(np.uint8), gq.GGML_BF16), values
        )

    def test_unknown_type_is_rejected(self):
        with pytest.raises(ValueError, match="ggml type 999"):
            gq.dequantize(np.zeros(4, dtype=np.uint8), 999)
