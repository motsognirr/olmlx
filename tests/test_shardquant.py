"""Tests for Shard KV-cache quantization primitives (#377 Tier 1)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


class TestRopeSpec:
    def _spec(self, dims=64, base=10000.0, traditional=False, scale=1.0):
        from olmlx.engine.shardquant import RopeSpec

        freqs = scale * mx.power(
            mx.array(base, dtype=mx.float32),
            -mx.arange(0, dims, 2, dtype=mx.float32) / dims,
        )
        return RopeSpec(dims=dims, freqs=freqs, traditional=traditional)

    def test_roundtrip_inverse(self):
        """inverse(apply(x)) reconstructs x for any offset."""
        from olmlx.engine.shardquant import rope_transform

        spec = self._spec()
        mx.random.seed(0)
        x = mx.random.normal((1, 4, 10, 64)).astype(mx.float16)
        for offset in (0, 7, 1000):
            y = rope_transform(x, spec, offset)
            back = rope_transform(y, spec, offset, inverse=True)
            np.testing.assert_allclose(
                np.array(back, dtype=np.float32),
                np.array(x, dtype=np.float32),
                atol=2e-3,
            )

    def test_matches_nn_rope(self):
        """Forward application matches mlx.nn.RoPE (non-traditional)."""
        from olmlx.engine.shardquant import rope_transform

        dims = 32
        rope = nn.RoPE(dims, traditional=False, base=10000.0)
        spec = self._spec(dims=dims)
        mx.random.seed(1)
        x = mx.random.normal((1, 2, 6, dims))
        for offset in (0, 5):
            expected = rope(x, offset=offset)
            got = rope_transform(x, spec, offset)
            np.testing.assert_allclose(np.array(got), np.array(expected), atol=1e-4)

    def test_matches_nn_rope_traditional(self):
        from olmlx.engine.shardquant import rope_transform

        dims = 32
        rope = nn.RoPE(dims, traditional=True, base=10000.0)
        spec = self._spec(dims=dims, traditional=True)
        mx.random.seed(2)
        x = mx.random.normal((1, 2, 6, dims))
        expected = rope(x, offset=3)
        got = rope_transform(x, spec, 3)
        np.testing.assert_allclose(np.array(got), np.array(expected), atol=1e-4)

    def test_partial_dims_passthrough(self):
        """Dims beyond spec.dims are untouched (partial rotary models)."""
        from olmlx.engine.shardquant import rope_transform

        spec = self._spec(dims=16)
        mx.random.seed(3)
        x = mx.random.normal((1, 1, 4, 64))
        y = rope_transform(x, spec, 11)
        np.testing.assert_allclose(
            np.array(y[..., 16:]), np.array(x[..., 16:]), atol=1e-6
        )

    def test_rotation_preserves_norms(self):
        from olmlx.engine.shardquant import rope_transform

        spec = self._spec()
        mx.random.seed(4)
        x = mx.random.normal((1, 2, 8, 64))
        y = rope_transform(x, spec, 42)
        np.testing.assert_allclose(
            np.array(mx.linalg.norm(y, axis=-1)),
            np.array(mx.linalg.norm(x, axis=-1)),
            rtol=1e-4,
        )


class TestDetectRopeSpec:
    def test_detects_nn_rope(self):
        from olmlx.engine.shardquant import detect_rope_spec

        class Attn:
            rope = nn.RoPE(64, traditional=False, base=500000.0, scale=1.0)

        spec = detect_rope_spec(Attn())
        assert spec is not None
        assert spec.dims == 64
        assert spec.traditional is False
        # freqs[0] = scale * base^0 = 1.0
        assert abs(float(spec.freqs[0]) - 1.0) < 1e-6
        # freqs decay with base
        assert float(spec.freqs[-1]) < float(spec.freqs[0])

    def test_detects_freqs_carrying_rope(self):
        """mlx-lm custom ropes (Llama3RoPE etc.) carry wavelength-like _freqs
        and call mx.fast.rope(..., base=None, freqs=self._freqs); the spec's
        angular freqs are their reciprocal."""
        from olmlx.engine.shardquant import detect_rope_spec

        class FakeLlama3RoPE:
            dims = 32
            traditional = False
            _freqs = mx.power(
                mx.array(10000.0), mx.arange(0, 32, 2, dtype=mx.float32) / 32
            )

        class Attn:
            rope = FakeLlama3RoPE()

        spec = detect_rope_spec(Attn())
        assert spec is not None
        assert spec.dims == 32
        np.testing.assert_allclose(
            np.array(spec.freqs),
            1.0 / np.array(FakeLlama3RoPE._freqs),
            rtol=1e-5,
        )

    def test_no_rope_returns_none(self):
        from olmlx.engine.shardquant import detect_rope_spec

        class Attn:
            pass

        assert detect_rope_spec(Attn()) is None

    def test_unknown_rope_returns_none(self):
        from olmlx.engine.shardquant import detect_rope_spec

        class WeirdRope:
            pass  # no dims, no _freqs

        class Attn:
            rope = WeirdRope()

        assert detect_rope_spec(Attn()) is None


class TestVRotation:
    def test_hadamard_for_power_of_two(self):
        from olmlx.engine.shardquant import make_v_rotation

        R = make_v_rotation(64)
        Rn = np.array(R)
        # Orthonormal
        np.testing.assert_allclose(Rn @ Rn.T, np.eye(64), atol=1e-5)
        # Hadamard: all entries ±1/sqrt(64)
        np.testing.assert_allclose(np.abs(Rn), 1.0 / 8.0, atol=1e-6)

    def test_qr_fallback_for_non_power_of_two(self):
        from olmlx.engine.shardquant import make_v_rotation

        R = make_v_rotation(80, seed=3)
        Rn = np.array(R)
        np.testing.assert_allclose(Rn @ Rn.T, np.eye(80), atol=1e-4)
        # Deterministic per seed
        np.testing.assert_array_equal(np.array(make_v_rotation(80, seed=3)), Rn)


class TestProductVQ:
    def test_fit_assign_gather_roundtrip_exact_on_clustered_data(self):
        """Data drawn exactly from K distinct subvectors reconstructs exactly."""
        from olmlx.engine.shardquant import fit_vq_codebooks, vq_assign, vq_gather

        rng = np.random.RandomState(0)
        P, g, K = 4, 2, 16
        true_centroids = rng.randn(P, K, g).astype(np.float32) * 5
        picks = rng.randint(0, K, size=(1000, P))
        data = np.stack(
            [true_centroids[p, picks[:, p]] for p in range(P)], axis=1
        )  # (N, P, g)
        cbs = fit_vq_codebooks(data.reshape(1000, P * g), group_size=g, seed=0)
        assert cbs.shape == (P, 256, g)
        x = mx.array(data.reshape(1, 1, 1000, P * g))
        idx = vq_assign(x.reshape(1, 1, 1000, P, g), cbs)
        assert idx.shape == (1, 1, 1000, P)
        assert idx.dtype == mx.uint8
        recon = vq_gather(idx, cbs)
        np.testing.assert_allclose(np.array(recon).reshape(1000, P, g), data, atol=1e-4)

    def test_quality_on_gaussian_data(self):
        """256 centroids on unit-sphere-ish 2-dim subvectors: high cosine sim."""
        from olmlx.engine.shardquant import fit_vq_codebooks, vq_assign, vq_gather

        rng = np.random.RandomState(1)
        N, D, g = 4000, 16, 2
        data = rng.randn(N, D).astype(np.float32)
        data /= np.linalg.norm(data, axis=-1, keepdims=True)
        cbs = fit_vq_codebooks(data, group_size=g, seed=1)
        x = mx.array(data.reshape(1, 1, N, D // g, g))
        recon = np.array(vq_gather(vq_assign(x, cbs), cbs)).reshape(N, D)
        cos = np.sum(recon * data, axis=-1) / (
            np.linalg.norm(recon, axis=-1) * np.linalg.norm(data, axis=-1)
        )
        assert cos.mean() > 0.95

    def test_fewer_points_than_centroids(self):
        from olmlx.engine.shardquant import fit_vq_codebooks

        rng = np.random.RandomState(2)
        cbs = fit_vq_codebooks(rng.randn(50, 8).astype(np.float32), group_size=2)
        assert cbs.shape == (4, 256, 2)
        assert np.isfinite(np.array(cbs)).all()


class TestConfigValidation:
    def test_shard_4_accepted(self):
        from olmlx.config import validate_kv_cache_quant_format

        assert validate_kv_cache_quant_format("shard:4") == "shard:4"

    def test_shard_2_accepted(self):
        from olmlx.config import validate_kv_cache_quant_format

        assert validate_kv_cache_quant_format("shard:2") == "shard:2"

    def test_shard_8_accepted(self):
        from olmlx.config import validate_kv_cache_quant_format

        assert validate_kv_cache_quant_format("shard:8") == "shard:8"

    def test_shard_3_rejected(self):
        from olmlx.config import validate_kv_cache_quant_format

        with pytest.raises(ValueError):
            validate_kv_cache_quant_format("shard:3")

    def test_spectral_8_still_rejected(self):
        """8-bit is shard-only; spectral/turboquant stay {2,4}."""
        from olmlx.config import validate_kv_cache_quant_format

        with pytest.raises(ValueError):
            validate_kv_cache_quant_format("spectral:8")
        with pytest.raises(ValueError):
            validate_kv_cache_quant_format("turboquant:8")


def _random_orthogonal(dim, seed):
    rng = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rng.randn(dim, dim).astype(np.float32))
    return q


class TestKeyCompress:
    def test_roundtrip_high_quality_full_rank_8bit(self):
        """Full rank + 8-bit: reconstruction within scalar-quant error."""
        from olmlx.engine.shardquant import (
            shard_compress_keys,
            shard_decompress_keys,
        )
        from olmlx.engine.spectralquant import fit_codebook

        B, H, S, D = 1, 2, 50, 32
        basis = mx.array(
            np.stack([_random_orthogonal(D, h) for h in range(H)])
        )  # (H, D, D)
        mx.random.seed(5)
        x = mx.random.normal((B, H, S, D))
        # Codebook fit on the actual rotated coefficients.
        x32 = x.astype(mx.float32)
        xn = x32 / mx.sqrt(mx.sum(x32 * x32, axis=-1, keepdims=True))
        y = mx.matmul(xn, mx.swapaxes(basis, -1, -2))
        cb = fit_codebook(y.reshape(-1), bits=8)

        packed, norms = shard_compress_keys(x, basis, rank=D, codebook=cb, bits=8)
        assert norms.shape == (B, H, S, 1)
        recon = shard_decompress_keys(
            packed, norms, basis, rank=D, codebook=cb, bits=8, dtype=x.dtype
        )
        assert recon.shape == x.shape
        cos = np.sum(np.array(recon) * np.array(x), -1) / (
            np.linalg.norm(np.array(recon), axis=-1)
            * np.linalg.norm(np.array(x), axis=-1)
        )
        assert cos.mean() > 0.99

    def test_rank_truncation_on_low_rank_data(self):
        """Data living in an r-dim subspace survives rank-r truncation."""
        from olmlx.engine.shardquant import (
            shard_compress_keys,
            shard_decompress_keys,
        )
        from olmlx.engine.spectralquant import fit_codebook

        B, H, S, D, R = 1, 1, 200, 16, 4
        rng = np.random.RandomState(7)
        Q = _random_orthogonal(D, 9)
        coeffs = rng.randn(S, R).astype(np.float32)
        data = coeffs @ Q[:R, :]  # rows of Q span the subspace
        x = mx.array(data.reshape(B, H, S, D))
        basis = mx.array(Q.reshape(1, D, D))  # rows = basis vectors
        xn = np.array(x) / np.linalg.norm(np.array(x), axis=-1, keepdims=True)
        y = mx.array(xn) @ mx.array(Q.T)
        cb = fit_codebook(mx.array(np.array(y)[..., :R].reshape(-1)), bits=4)

        packed, norms = shard_compress_keys(x, basis, rank=R, codebook=cb, bits=4)
        # Packed coefficient payload is rank-sized, not head_dim-sized.
        assert packed.shape[-1] == R // 2  # 4-bit pack: 2 per byte
        recon = shard_decompress_keys(
            packed, norms, basis, rank=R, codebook=cb, bits=4, dtype=x.dtype
        )
        cos = np.sum(np.array(recon) * np.array(x), -1) / (
            np.linalg.norm(np.array(recon), axis=-1)
            * np.linalg.norm(np.array(x), axis=-1)
        )
        assert cos.mean() > 0.9

    def test_per_head_bases_are_independent(self):
        """Head h must be projected with basis[h], not basis[0]."""
        from olmlx.engine.shardquant import (
            shard_compress_keys,
            shard_decompress_keys,
        )
        from olmlx.engine.spectralquant import fit_codebook

        D = 8
        basis = mx.array(
            np.stack([np.eye(D, dtype=np.float32), _random_orthogonal(D, 1)])
        )
        mx.random.seed(8)
        x = mx.random.normal((1, 2, 30, D))
        cb = fit_codebook(mx.random.normal((4000,)) * 0.35, bits=8)
        packed, norms = shard_compress_keys(x, basis, rank=D, codebook=cb, bits=8)
        recon = shard_decompress_keys(
            packed, norms, basis, rank=D, codebook=cb, bits=8, dtype=x.dtype
        )
        for h in range(2):
            cos = np.sum(np.array(recon)[0, h] * np.array(x)[0, h], -1) / (
                np.linalg.norm(np.array(recon)[0, h], axis=-1)
                * np.linalg.norm(np.array(x)[0, h], axis=-1)
            )
            assert cos.mean() > 0.95, f"head {h} mis-projected"

    def test_odd_rank_roundtrip(self):
        """Rank not a multiple of the pack width: the spectral packers pad
        the tail and unpack slices it back, so an odd rank must round-trip
        without corrupting the coefficient count."""
        from olmlx.engine.shardquant import (
            shard_compress_keys,
            shard_decompress_keys,
        )
        from olmlx.engine.spectralquant import fit_codebook

        B, H, S, D, R = 1, 2, 40, 16, 5  # 5 is odd; 4-bit packs 2/byte
        basis = mx.array(np.stack([_random_orthogonal(D, 20 + h) for h in range(H)]))
        rng = np.random.RandomState(11)
        coeffs = rng.randn(H, S, R).astype(np.float32)
        data = np.einsum("hsr,hrd->hsd", coeffs, np.array(basis)[:, :R, :])
        x = mx.array(data.reshape(B, H, S, D))
        xn = data / np.linalg.norm(data, axis=-1, keepdims=True)
        y = np.einsum("hsd,hrd->hsr", xn, np.array(basis)[:, :R, :])
        cb = fit_codebook(mx.array(y.reshape(-1)), bits=4)

        packed, norms = shard_compress_keys(x, basis, rank=R, codebook=cb, bits=4)
        assert packed.shape[-1] == (R + 1) // 2  # padded to 3 bytes
        recon = shard_decompress_keys(
            packed, norms, basis, rank=R, codebook=cb, bits=4, dtype=x.dtype
        )
        assert recon.shape == x.shape
        cos = np.sum(np.array(recon) * data.reshape(B, H, S, D), -1) / (
            np.linalg.norm(np.array(recon), axis=-1)
            * np.linalg.norm(data.reshape(B, H, S, D), axis=-1)
        )
        assert cos.mean() > 0.9


class TestValueCompress:
    def test_roundtrip_quality(self):
        from olmlx.engine.shardquant import (
            fit_vq_codebooks,
            make_v_rotation,
            shard_compress_values,
            shard_decompress_values,
        )

        B, H, S, D, g = 1, 2, 300, 16, 2
        R = make_v_rotation(D)
        mx.random.seed(9)
        x = mx.random.normal((B, H, S, D)).astype(mx.float16)
        x32 = np.array(x, dtype=np.float32).reshape(-1, D)
        xn = x32 / np.linalg.norm(x32, axis=-1, keepdims=True)
        rotated = xn @ np.array(R).T
        cbs = fit_vq_codebooks(rotated, group_size=g, seed=0)

        idx, norms = shard_compress_values(x, R, cbs)
        assert idx.shape == (B, H, S, D // g)
        assert idx.dtype == mx.uint8
        recon = shard_decompress_values(idx, norms, R, cbs, dtype=x.dtype)
        assert recon.shape == x.shape
        assert recon.dtype == x.dtype
        cos = np.sum(
            np.array(recon, dtype=np.float32) * x32.reshape(B, H, S, D), -1
        ) / (
            np.linalg.norm(np.array(recon, dtype=np.float32), axis=-1)
            * np.linalg.norm(x32.reshape(B, H, S, D), axis=-1)
        )
        assert cos.mean() > 0.9


class TestScalarAssign:
    def test_matches_naive_argmin(self):
        from olmlx.engine.shardquant import scalar_assign

        rng = np.random.RandomState(3)
        codebook = mx.array(np.sort(rng.randn(16)).astype(np.float32))
        y = mx.array(rng.randn(1, 2, 700, 5).astype(np.float32))
        idx = scalar_assign(y, codebook)
        assert idx.dtype == mx.uint8
        expected = np.argmin(
            np.abs(np.array(y)[..., None] - np.array(codebook)), axis=-1
        )
        np.testing.assert_array_equal(np.array(idx), expected)
