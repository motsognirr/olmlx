"""Tests for SpectralQuant KV cache compression."""

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# SpectralRotation tests
# ---------------------------------------------------------------------------


class TestSpectralRotation:
    """Tests for eigenvector-based rotation."""

    def test_roundtrip_preserves_vectors(self):
        """unrotate(rotate(x)) should reconstruct x."""
        from olmlx.engine.spectralquant import SpectralRotation

        # Create a valid orthogonal matrix (eigenvectors are orthonormal)
        rng = np.random.RandomState(42)
        mat = rng.randn(64, 64).astype(np.float32)
        q, _ = np.linalg.qr(mat)
        V = mx.array(q)

        rot = SpectralRotation(V)
        x = mx.random.normal((2, 4, 10, 64))  # (B, heads, seq, head_dim)
        reconstructed = rot.unrotate(rot.rotate(x))
        np.testing.assert_allclose(np.array(reconstructed), np.array(x), atol=1e-5)

    def test_preserves_inner_products(self):
        """Rotation should preserve dot products (orthogonal transform)."""
        from olmlx.engine.spectralquant import SpectralRotation

        rng = np.random.RandomState(7)
        q, _ = np.linalg.qr(rng.randn(32, 32).astype(np.float32))
        rot = SpectralRotation(mx.array(q))

        a = mx.random.normal((5, 32))
        b = mx.random.normal((5, 32))
        original_dots = mx.sum(a * b, axis=-1)
        rotated_dots = mx.sum(rot.rotate(a) * rot.rotate(b), axis=-1)
        np.testing.assert_allclose(
            np.array(original_dots), np.array(rotated_dots), atol=1e-5
        )

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        from olmlx.engine.spectralquant import SpectralRotation

        q, _ = np.linalg.qr(np.random.randn(128, 128).astype(np.float32))
        rot = SpectralRotation(mx.array(q))
        x = mx.random.normal((1, 8, 20, 128))
        assert rot.rotate(x).shape == x.shape
        assert rot.unrotate(x).shape == x.shape

    def test_stores_eigenvectors(self):
        """Should store V and V_T as attributes."""
        from olmlx.engine.spectralquant import SpectralRotation

        q, _ = np.linalg.qr(np.random.randn(64, 64).astype(np.float32))
        V = mx.array(q)
        rot = SpectralRotation(V)
        np.testing.assert_array_equal(np.array(rot.V), np.array(V))
        np.testing.assert_allclose(np.array(rot.V_T), np.array(V.T), atol=1e-7)


# ---------------------------------------------------------------------------
# pack_indices / unpack_indices tests
# ---------------------------------------------------------------------------


class TestPackUnpack:
    """Round-trip tests for the variable-bit index packers."""

    @pytest.mark.parametrize("bits", [1, 2, 4, 8])
    @pytest.mark.parametrize("dim", [8, 15, 16, 31, 32, 127, 128])
    @pytest.mark.parametrize("dtype", [np.uint8, np.int32])
    def test_roundtrip(self, bits, dim, dtype):
        """unpack(pack(x)) should reproduce x for any dim — including
        odd/non-multiple-of-factor tail sizes that SpectralQuant produces
        when d_eff lands on an odd number. The int32 dtype is defensive
        coverage: today's only caller (`_quantize_regime`) casts to uint8
        before calling `pack_indices`, but this keeps the packer honest."""
        from olmlx.engine.spectralquant import pack_indices, unpack_indices

        max_val = (1 << bits) - 1
        rng = np.random.RandomState(bits * 100 + dim)
        indices = mx.array(rng.randint(0, max_val + 1, size=(2, 3, dim)).astype(dtype))
        packed = pack_indices(indices, bits)
        restored = unpack_indices(packed, bits, dim)
        assert restored.shape == indices.shape
        np.testing.assert_array_equal(np.array(restored), np.array(indices))


# ---------------------------------------------------------------------------
# Bit allocation tests
# ---------------------------------------------------------------------------


class TestBitAllocation:
    """Tests for non-uniform bit allocation."""

    def test_budget_constraint(self):
        """Allocated bits should not exceed total budget."""
        from olmlx.engine.spectralquant import allocate_bits

        d_eff = 5
        head_dim = 128
        avg_bits = 4
        b_high, b_low = allocate_bits(d_eff, head_dim, avg_bits)
        total = d_eff * b_high + (head_dim - d_eff) * b_low
        budget = head_dim * avg_bits
        # Should be within one b_high of the budget (rounding)
        assert total <= budget + d_eff

    def test_high_ge_low(self):
        """Semantic regime should get >= bits as tail regime."""
        from olmlx.engine.spectralquant import allocate_bits

        b_high, b_low = allocate_bits(d_eff=4, head_dim=128, avg_bits=4)
        assert b_high >= b_low

    def test_both_at_least_1(self):
        """Both regimes should get at least 1 bit."""
        from olmlx.engine.spectralquant import allocate_bits

        b_high, b_low = allocate_bits(d_eff=10, head_dim=128, avg_bits=2)
        assert b_high >= 1
        assert b_low >= 1

    def test_avg_bits_2(self):
        """Should work with 2-bit average budget."""
        from olmlx.engine.spectralquant import allocate_bits

        b_high, b_low = allocate_bits(d_eff=5, head_dim=128, avg_bits=2)
        total = 5 * b_high + 123 * b_low
        budget = 128 * 2
        assert total <= budget + 5

    def test_high_d_eff_falls_back_to_uniform(self):
        """When d_eff is large, should approximate uniform allocation."""
        from olmlx.engine.spectralquant import allocate_bits

        b_high, b_low = allocate_bits(d_eff=120, head_dim=128, avg_bits=4)
        # With most dimensions being semantic, difference should be small
        assert b_high - b_low <= 2


# ---------------------------------------------------------------------------
# Lloyd-Max codebook fitting tests
# ---------------------------------------------------------------------------


class TestLloydMax:
    """Tests for Lloyd-Max codebook fitting."""

    def test_codebook_size(self):
        """Fitted codebook should have 2^bits centroids."""
        from olmlx.engine.spectralquant import fit_codebook

        data = mx.random.normal((1000,))
        codebook = fit_codebook(data, bits=4)
        assert codebook.shape == (16,)

    def test_codebook_sorted(self):
        """Centroids should be sorted ascending."""
        from olmlx.engine.spectralquant import fit_codebook

        data = mx.random.normal((1000,))
        codebook = fit_codebook(data, bits=2)
        cb_np = np.array(codebook)
        assert all(cb_np[i] < cb_np[i + 1] for i in range(len(cb_np) - 1))

    def test_codebook_spans_data_range(self):
        """Centroids should cover the data range."""
        from olmlx.engine.spectralquant import fit_codebook

        data = mx.random.normal((2000,))
        codebook = fit_codebook(data, bits=2)
        assert float(codebook[0]) < 0  # Should have negative centroids
        assert float(codebook[-1]) > 0  # And positive ones

    def test_mse_decreases_with_more_bits(self):
        """4-bit should have lower MSE than 2-bit."""
        from olmlx.engine.spectralquant import fit_codebook

        data = mx.random.normal((2000,))
        cb2 = fit_codebook(data, bits=2)
        cb4 = fit_codebook(data, bits=4)

        # Compute MSE for each
        def mse(data, cb):
            best_dist = mx.full(data.shape, float("inf"))
            for c in range(cb.shape[0]):
                d = (data - cb[c]) ** 2
                best_dist = mx.minimum(best_dist, d)
            return float(mx.mean(best_dist))

        assert mse(data, cb4) < mse(data, cb2)

    def test_2bit_codebook(self):
        """2-bit codebook should have 4 centroids."""
        from olmlx.engine.spectralquant import fit_codebook

        data = mx.random.normal((500,))
        codebook = fit_codebook(data, bits=2)
        assert codebook.shape == (4,)


# ---------------------------------------------------------------------------
# Spectral quantize / dequantize tests
# ---------------------------------------------------------------------------


class TestSpectralQuantizeDequantize:
    """Tests for full spectral quantize/dequantize pipeline."""

    @pytest.fixture()
    def setup(self):
        """Create rotation, codebooks, and test data."""
        from olmlx.engine.spectralquant import (
            SpectralRotation,
            fit_codebook,
        )

        head_dim = 64
        d_eff = 4
        rng = np.random.RandomState(42)
        q, _ = np.linalg.qr(rng.randn(head_dim, head_dim).astype(np.float32))
        rotation = SpectralRotation(mx.array(q))

        # Generate some data and fit codebooks on rotated data
        x = mx.random.normal((100, head_dim))
        norms = mx.linalg.norm(x, axis=-1, keepdims=True)
        x_norm = x / mx.maximum(norms, mx.array(1e-8))
        rotated = rotation.rotate(x_norm)
        sem_data = rotated[..., :d_eff].reshape(-1)
        tail_data = rotated[..., d_eff:].reshape(-1)

        return {
            "rotation": rotation,
            "codebook_sem": fit_codebook(sem_data, bits=8),
            "codebook_tail": fit_codebook(tail_data, bits=4),
            "d_eff": d_eff,
            "bits_high": 8,
            "bits_low": 4,
            "head_dim": head_dim,
        }

    def test_roundtrip_cosine_similarity(self, setup):
        """Quantize→dequantize should achieve reasonable cosine similarity."""
        from olmlx.engine.spectralquant import (
            spectral_dequantize,
            spectral_quantize,
        )

        x = mx.random.normal((2, 4, 10, setup["head_dim"]))
        packed_sem, packed_tail, norms = spectral_quantize(
            x,
            setup["rotation"],
            setup["codebook_sem"],
            setup["codebook_tail"],
            setup["d_eff"],
            setup["bits_high"],
            setup["bits_low"],
        )
        reconstructed = spectral_dequantize(
            packed_sem,
            packed_tail,
            norms,
            setup["rotation"],
            setup["codebook_sem"],
            setup["codebook_tail"],
            setup["d_eff"],
            setup["bits_high"],
            setup["bits_low"],
        )
        # Cosine similarity per vector
        x_flat = x.reshape(-1, setup["head_dim"])
        r_flat = reconstructed.reshape(-1, setup["head_dim"])
        cos = mx.sum(x_flat * r_flat, axis=-1) / (
            mx.linalg.norm(x_flat, axis=-1) * mx.linalg.norm(r_flat, axis=-1)
        )
        mean_cos = float(mx.mean(cos))
        assert mean_cos > 0.85, f"Mean cosine similarity {mean_cos} too low"

    def test_output_shape_matches_input(self, setup):
        """Dequantized output should have same shape as input."""
        from olmlx.engine.spectralquant import (
            spectral_dequantize,
            spectral_quantize,
        )

        x = mx.random.normal((1, 2, 5, setup["head_dim"]))
        packed_sem, packed_tail, norms = spectral_quantize(
            x,
            setup["rotation"],
            setup["codebook_sem"],
            setup["codebook_tail"],
            setup["d_eff"],
            setup["bits_high"],
            setup["bits_low"],
        )
        reconstructed = spectral_dequantize(
            packed_sem,
            packed_tail,
            norms,
            setup["rotation"],
            setup["codebook_sem"],
            setup["codebook_tail"],
            setup["d_eff"],
            setup["bits_high"],
            setup["bits_low"],
        )
        assert reconstructed.shape == x.shape

    def test_norms_preserved(self, setup):
        """L2 norms should be approximately preserved after roundtrip."""
        from olmlx.engine.spectralquant import (
            spectral_dequantize,
            spectral_quantize,
        )

        x = mx.random.normal((1, 2, 8, setup["head_dim"]))
        packed_sem, packed_tail, norms = spectral_quantize(
            x,
            setup["rotation"],
            setup["codebook_sem"],
            setup["codebook_tail"],
            setup["d_eff"],
            setup["bits_high"],
            setup["bits_low"],
        )
        reconstructed = spectral_dequantize(
            packed_sem,
            packed_tail,
            norms,
            setup["rotation"],
            setup["codebook_sem"],
            setup["codebook_tail"],
            setup["d_eff"],
            setup["bits_high"],
            setup["bits_low"],
        )
        orig_norms = mx.linalg.norm(x, axis=-1)
        recon_norms = mx.linalg.norm(reconstructed, axis=-1)
        np.testing.assert_allclose(
            np.array(orig_norms), np.array(recon_norms), rtol=0.15
        )

    def test_dtype_passthrough(self, setup):
        """Should respect dtype parameter."""
        from olmlx.engine.spectralquant import (
            spectral_dequantize,
            spectral_quantize,
        )

        x = mx.random.normal((1, 1, 4, setup["head_dim"]))
        packed_sem, packed_tail, norms = spectral_quantize(
            x,
            setup["rotation"],
            setup["codebook_sem"],
            setup["codebook_tail"],
            setup["d_eff"],
            setup["bits_high"],
            setup["bits_low"],
        )
        result = spectral_dequantize(
            packed_sem,
            packed_tail,
            norms,
            setup["rotation"],
            setup["codebook_sem"],
            setup["codebook_tail"],
            setup["d_eff"],
            setup["bits_high"],
            setup["bits_low"],
            dtype=mx.float16,
        )
        assert result.dtype == mx.float16

    def test_roundtrip_d_eff_1_tail_127(self):
        """Regression: Qwen3-4B layer 0 calibrates to d_eff=1, so tail_dim=127
        (odd) and bits_low=2 — the exact path that broke before the odd-dim
        packing fix."""
        from olmlx.engine.spectralquant import (
            SpectralRotation,
            fit_codebook,
            spectral_dequantize,
            spectral_quantize,
        )

        head_dim = 128
        d_eff = 1
        bits_high = 4
        bits_low = 2
        rng = np.random.RandomState(0)
        q, _ = np.linalg.qr(rng.randn(head_dim, head_dim).astype(np.float32))
        rotation = SpectralRotation(mx.array(q))

        x = mx.random.normal((200, head_dim))
        norms = mx.linalg.norm(x, axis=-1, keepdims=True)
        rotated = rotation.rotate(x / mx.maximum(norms, mx.array(1e-8)))
        codebook_sem = fit_codebook(rotated[..., :d_eff].reshape(-1), bits=bits_high)
        codebook_tail = fit_codebook(rotated[..., d_eff:].reshape(-1), bits=bits_low)

        y = mx.random.normal((2, 3, 5, head_dim))
        packed_sem, packed_tail, y_norms = spectral_quantize(
            y, rotation, codebook_sem, codebook_tail, d_eff, bits_high, bits_low
        )
        reconstructed = spectral_dequantize(
            packed_sem,
            packed_tail,
            y_norms,
            rotation,
            codebook_sem,
            codebook_tail,
            d_eff,
            bits_high,
            bits_low,
        )
        assert reconstructed.shape == y.shape


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestCalibration:
    """Tests for eigenspectral calibration pipeline."""

    def test_compute_covariance(self):
        """Covariance should be symmetric positive semi-definite."""
        from olmlx.engine.spectralquant_calibrate import compute_covariance

        data = mx.random.normal((500, 32))
        cov = compute_covariance(data)
        assert cov.shape == (32, 32)
        # Symmetric
        np.testing.assert_allclose(np.array(cov), np.array(cov.T), atol=1e-5)

    def test_eigendecompose_returns_sorted_desc(self):
        """Eigenvalues should be sorted in descending order."""
        from olmlx.engine.spectralquant_calibrate import eigendecompose

        # Create a matrix with known eigenstructure
        rng = np.random.RandomState(42)
        data = rng.randn(500, 16).astype(np.float32)
        cov = mx.array(data.T @ data / len(data))
        eigenvalues, eigenvectors = eigendecompose(cov)

        assert eigenvalues.shape == (16,)
        assert eigenvectors.shape == (16, 16)
        # Descending order
        ev = np.array(eigenvalues)
        assert all(ev[i] >= ev[i + 1] for i in range(len(ev) - 1))

    def test_eigendecompose_nonnegative(self):
        """Eigenvalues should be non-negative (clamped)."""
        from olmlx.engine.spectralquant_calibrate import eigendecompose

        rng = np.random.RandomState(7)
        data = rng.randn(100, 8).astype(np.float32)
        cov = mx.array(data.T @ data / len(data))
        eigenvalues, _ = eigendecompose(cov)
        assert all(float(v) >= 0 for v in eigenvalues)

    def test_eigenvectors_orthonormal(self):
        """Eigenvectors should form an orthonormal basis."""
        from olmlx.engine.spectralquant_calibrate import eigendecompose

        rng = np.random.RandomState(42)
        data = rng.randn(500, 16).astype(np.float32)
        cov = mx.array(data.T @ data / len(data))
        _, V = eigendecompose(cov)
        product = V @ V.T
        np.testing.assert_allclose(np.array(product), np.eye(16), atol=1e-4)

    def test_compute_d_eff(self):
        """d_eff should be between 1 and head_dim."""
        from olmlx.engine.spectralquant_calibrate import compute_d_eff

        # Eigenvalues with clear spectral gap: 3 large, rest small
        eigenvalues = mx.array([10.0, 5.0, 2.0, 0.01, 0.01, 0.01, 0.01, 0.01])
        d_eff = compute_d_eff(eigenvalues)
        assert 1 <= d_eff <= 8
        # Should be small — most energy in first few dimensions
        assert d_eff <= 5

    def test_compute_d_eff_uniform_eigenvalues(self):
        """Uniform eigenvalues → d_eff close to full dimension."""
        from olmlx.engine.spectralquant_calibrate import compute_d_eff

        eigenvalues = mx.ones(32)
        d_eff = compute_d_eff(eigenvalues)
        assert d_eff == 32  # All dimensions equally important

    def test_calibrate_head_end_to_end(self):
        """Full calibration of one head should produce valid rotation + codebooks."""
        from olmlx.engine.spectralquant_calibrate import calibrate_head

        # Simulate KV vectors with low-rank structure
        rng = np.random.RandomState(42)
        # Most variance in first 3 dimensions
        data = np.zeros((500, 32), dtype=np.float32)
        data[:, :3] = rng.randn(500, 3) * 10.0
        data[:, 3:] = rng.randn(500, 29) * 0.1
        kv_data = mx.array(data)

        result = calibrate_head(kv_data, avg_bits=4)
        assert "eigenvectors" in result
        assert "d_eff" in result
        assert "codebook_sem" in result
        assert "codebook_tail" in result
        assert "bits_high" in result
        assert "bits_low" in result
        assert result["eigenvectors"].shape == (32, 32)
        assert 1 <= result["d_eff"] <= 32

    def test_save_and_load_calibration(self, tmp_path):
        """Calibration data should survive save/load roundtrip."""
        from olmlx.engine.spectralquant_calibrate import (
            load_calibration,
            save_calibration,
        )

        # Create mock calibration data for 2 layers, 2 heads each
        calibration = {}
        rng = np.random.RandomState(42)
        for layer in range(2):
            for head in range(2):
                for kind in ("key", "value"):
                    q, _ = np.linalg.qr(rng.randn(16, 16).astype(np.float32))
                    key = (layer, head, kind)
                    calibration[key] = {
                        "eigenvectors": mx.array(q),
                        "d_eff": 3,
                        "codebook_sem": mx.array([0.1, 0.2, 0.3, 0.4]),
                        "codebook_tail": mx.array([-1.0, 0.0, 1.0, 2.0]),
                        "bits_high": 4,
                        "bits_low": 2,
                    }

        save_calibration(calibration, tmp_path)
        loaded = load_calibration(tmp_path)

        for key in calibration:
            assert key in loaded
            np.testing.assert_allclose(
                np.array(loaded[key]["eigenvectors"]),
                np.array(calibration[key]["eigenvectors"]),
                atol=1e-5,
            )
            assert loaded[key]["d_eff"] == calibration[key]["d_eff"]
            assert loaded[key]["bits_high"] == calibration[key]["bits_high"]


# ---------------------------------------------------------------------------
# SpectralQuantKVCache tests
# ---------------------------------------------------------------------------


class TestSpectralQuantKVCache:
    """Tests for SpectralQuantKVCache drop-in replacement."""

    @pytest.fixture()
    def cache_setup(self):
        """Create calibration data and cache for testing."""
        from olmlx.engine.spectralquant import (
            SpectralRotation,
            fit_codebook,
        )
        from olmlx.engine.spectralquant_cache import SpectralQuantKVCache

        head_dim = 32
        d_eff = 4
        bits_high = 4
        bits_low = 2

        rng = np.random.RandomState(42)
        q, _ = np.linalg.qr(rng.randn(head_dim, head_dim).astype(np.float32))
        rotation_k = SpectralRotation(mx.array(q))
        q2, _ = np.linalg.qr(rng.randn(head_dim, head_dim).astype(np.float32))
        rotation_v = SpectralRotation(mx.array(q2))

        # Fit codebooks on synthetic data (normalized, matching spectral_quantize)
        data = mx.random.normal((500, head_dim))
        norms = mx.linalg.norm(data, axis=-1, keepdims=True)
        data_n = data / mx.maximum(norms, mx.array(1e-8))
        rotated = rotation_k.rotate(data_n)
        cb_sem = fit_codebook(rotated[..., :d_eff].reshape(-1), bits=bits_high)
        cb_tail = fit_codebook(rotated[..., d_eff:].reshape(-1), bits=bits_low)

        cache = SpectralQuantKVCache(
            rotation_key=rotation_k,
            rotation_value=rotation_v,
            codebook_sem_key=cb_sem,
            codebook_tail_key=cb_tail,
            codebook_sem_value=cb_sem,
            codebook_tail_value=cb_tail,
            d_eff=d_eff,
            bits_high=bits_high,
            bits_low=bits_low,
        )
        return cache, head_dim

    def test_update_and_fetch_shape(self, cache_setup):
        """Output shape should match input shape."""
        cache, hd = cache_setup
        keys = mx.random.normal((1, 2, 5, hd))
        values = mx.random.normal((1, 2, 5, hd))
        k_out, v_out = cache.update_and_fetch(keys, values)
        assert k_out.shape == (1, 2, 5, hd)
        assert v_out.shape == (1, 2, 5, hd)

    def test_sequential_updates_accumulate(self, cache_setup):
        """Multiple updates should accumulate correctly."""
        cache, hd = cache_setup
        k1 = mx.random.normal((1, 2, 3, hd))
        v1 = mx.random.normal((1, 2, 3, hd))
        cache.update_and_fetch(k1, v1)

        k2 = mx.random.normal((1, 2, 4, hd))
        v2 = mx.random.normal((1, 2, 4, hd))
        k_out, v_out = cache.update_and_fetch(k2, v2)
        assert k_out.shape == (1, 2, 7, hd)  # 3 + 4 = 7
        assert v_out.shape == (1, 2, 7, hd)

    def test_offset_tracks_position(self, cache_setup):
        """Offset should track total tokens stored."""
        cache, hd = cache_setup
        assert cache.offset == 0
        keys = mx.random.normal((1, 2, 5, hd))
        values = mx.random.normal((1, 2, 5, hd))
        cache.update_and_fetch(keys, values)
        assert cache.offset == 5

    def test_trim(self, cache_setup):
        """Trim should reduce offset."""
        cache, hd = cache_setup
        keys = mx.random.normal((1, 2, 10, hd))
        values = mx.random.normal((1, 2, 10, hd))
        cache.update_and_fetch(keys, values)
        trimmed = cache.trim(3)
        assert trimmed == 3
        assert cache.offset == 7

    def test_trim_to_empty(self, cache_setup):
        """Trimming all tokens should clear buffers."""
        cache, hd = cache_setup
        keys = mx.random.normal((1, 2, 5, hd))
        values = mx.random.normal((1, 2, 5, hd))
        cache.update_and_fetch(keys, values)
        cache.trim(5)
        assert cache.offset == 0
        assert cache.empty()

    def test_is_trimmable(self, cache_setup):
        cache, _ = cache_setup
        assert cache.is_trimmable()

    def test_state_setter_raises(self, cache_setup):
        """State setter should raise NotImplementedError."""
        cache, hd = cache_setup
        keys = mx.random.normal((1, 2, 3, hd))
        values = mx.random.normal((1, 2, 3, hd))
        cache.update_and_fetch(keys, values)
        with pytest.raises(NotImplementedError):
            cache.state = cache.state

    def test_reconstruction_quality(self, cache_setup):
        """Reconstructed vectors should have reasonable cosine similarity."""
        cache, hd = cache_setup
        keys = mx.random.normal((1, 2, 20, hd))
        values = mx.random.normal((1, 2, 20, hd))
        k_out, v_out = cache.update_and_fetch(keys, values)

        # Check cosine similarity
        k_flat = keys.reshape(-1, hd)
        kr_flat = k_out.reshape(-1, hd)
        cos = mx.sum(k_flat * kr_flat, axis=-1) / (
            mx.linalg.norm(k_flat, axis=-1) * mx.linalg.norm(kr_flat, axis=-1)
        )
        assert float(mx.mean(cos)) > 0.7  # Compressed, so lower bar


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestSpectralConfig:
    """Tests for spectral quant config validation."""

    def test_spectral_4_accepted(self):
        from olmlx.config import ExperimentalSettings

        s = ExperimentalSettings(kv_cache_quant="spectral:4")
        assert s.kv_cache_quant == "spectral:4"

    def test_spectral_2_accepted(self):
        from olmlx.config import ExperimentalSettings

        s = ExperimentalSettings(kv_cache_quant="spectral:2")
        assert s.kv_cache_quant == "spectral:2"

    def test_spectral_3_rejected(self):
        from olmlx.config import ExperimentalSettings

        with pytest.raises(Exception):
            ExperimentalSettings(kv_cache_quant="spectral:3")

    def test_turboquant_still_accepted(self):
        from olmlx.config import ExperimentalSettings

        s = ExperimentalSettings(kv_cache_quant="turboquant:4")
        assert s.kv_cache_quant == "turboquant:4"

    def test_resolve_config_holder_prefers_inner_when_it_has_args(self):
        """When backbone (inner) has .args, use it — matches pre-existing behavior."""
        from unittest.mock import MagicMock

        from olmlx.engine.spectralquant_calibrate import _resolve_config_holder

        inner = MagicMock(spec=[])
        inner.args = MagicMock(spec=[])
        model = MagicMock(spec=[])
        model.args = MagicMock(spec=[])

        assert _resolve_config_holder(inner, model) is inner

    def test_resolve_cache_owner_prefers_model_with_make_cache(self):
        """Qwen3Next: top-level has hybrid make_cache(); backbone only has layers."""
        from unittest.mock import MagicMock

        from olmlx.engine.spectralquant_calibrate import _resolve_cache_owner

        inner = MagicMock(spec=["layers"])
        model = MagicMock(spec=["layers", "make_cache"])
        assert _resolve_cache_owner(inner, model) is model

    def test_is_attention_cache_state_4d(self):
        """Only 4D (B, n_kv, seq, head_dim) states are attention caches we calibrate."""
        from olmlx.engine.spectralquant_calibrate import _is_attention_cache_state

        ok_keys = mx.zeros((1, 2, 16, 256))
        ok_vals = mx.zeros((1, 2, 16, 256))
        assert _is_attention_cache_state([ok_keys, ok_vals]) is True

    def test_is_attention_cache_state_ssm_3d(self):
        """Qwen3Next SSM cache: conv state (index 0) is 3D, so the entry is skipped."""
        from olmlx.engine.spectralquant_calibrate import _is_attention_cache_state

        ssm_conv = mx.zeros((1, 3, 8192))  # conv state
        ssm_hid = mx.zeros((1, 32, 128, 128))
        assert _is_attention_cache_state([ssm_conv, ssm_hid]) is False

    def test_resolve_cache_owner_ignores_layers_without_make_cache(self):
        """Without make_cache on the top-level model, route to backbone regardless of .layers."""
        from unittest.mock import MagicMock

        from olmlx.engine.spectralquant_calibrate import _resolve_cache_owner

        inner = MagicMock(spec=["layers"])
        model = MagicMock(spec=["layers"])
        assert _resolve_cache_owner(inner, model) is inner

    def test_resolve_config_holder_falls_back_to_model(self):
        """Qwen3Next regression: backbone has no .args, so calibrate must use model.args."""
        from unittest.mock import MagicMock

        from olmlx.engine.spectralquant_calibrate import _resolve_config_holder

        inner = MagicMock(spec=["layers"])  # no .args
        model = MagicMock(spec=[])
        model.args = MagicMock(spec=[])
        model.args.head_dim = 256

        holder = _resolve_config_holder(inner, model)
        assert holder is model
        assert holder.args.head_dim == 256

    def test_resolve_config_holder_raises_when_neither_has_args(self):
        """Unsupported architecture: surface a clear error instead of a cryptic AttributeError."""
        from unittest.mock import MagicMock

        from olmlx.engine.spectralquant_calibrate import _resolve_config_holder

        inner = MagicMock(spec=["layers"])
        model = MagicMock(spec=[])
        with pytest.raises(RuntimeError, match="Unsupported architecture"):
            _resolve_config_holder(inner, model)

    def test_disk_cache_guard(self):
        """SpectralQuantKVCache should block disk serialization."""
        from olmlx.engine.model_manager import _is_serializable_cache
        from olmlx.engine.spectralquant_cache import SpectralQuantKVCache

        rng = np.random.RandomState(0)
        q, _ = np.linalg.qr(rng.randn(32, 32).astype(np.float32))
        from olmlx.engine.spectralquant import (
            SpectralRotation,
            fit_codebook,
        )

        rot = SpectralRotation(mx.array(q))
        cb = fit_codebook(mx.random.normal((100,)), bits=4)
        cache = SpectralQuantKVCache(
            rotation_key=rot,
            rotation_value=rot,
            codebook_sem_key=cb,
            codebook_tail_key=cb,
            codebook_sem_value=cb,
            codebook_tail_value=cb,
            d_eff=4,
            bits_high=4,
            bits_low=2,
        )
        assert not _is_serializable_cache([cache])
