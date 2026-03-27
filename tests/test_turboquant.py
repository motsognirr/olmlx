"""Tests for TurboQuant KV cache quantization."""

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Core algorithm tests
# ---------------------------------------------------------------------------


class TestCodebook:
    """Tests for precomputed Lloyd-Max codebooks."""

    def test_codebook_has_2_and_4_bit(self):
        from olmlx.engine.turboquant import GAUSSIAN_CODEBOOKS

        assert 2 in GAUSSIAN_CODEBOOKS
        assert 4 in GAUSSIAN_CODEBOOKS

    def test_codebook_count(self):
        from olmlx.engine.turboquant import GAUSSIAN_CODEBOOKS

        assert len(GAUSSIAN_CODEBOOKS[2]) == 4
        assert len(GAUSSIAN_CODEBOOKS[4]) == 16

    def test_codebook_symmetry(self):
        """Centroids should be symmetric around 0."""
        from olmlx.engine.turboquant import GAUSSIAN_CODEBOOKS

        for bits in [2, 4]:
            c = np.array(GAUSSIAN_CODEBOOKS[bits])
            np.testing.assert_allclose(c, -c[::-1], atol=1e-6)

    def test_codebook_sorted(self):
        from olmlx.engine.turboquant import GAUSSIAN_CODEBOOKS

        for bits in [2, 4]:
            c = GAUSSIAN_CODEBOOKS[bits]
            assert all(c[i] < c[i + 1] for i in range(len(c) - 1))

    def test_scaled_codebook(self):
        """get_codebook should return centroids scaled by 1/sqrt(dim)."""
        from olmlx.engine.turboquant import GAUSSIAN_CODEBOOKS, get_codebook

        cb = get_codebook(bits=2, dim=128)
        expected = mx.array(GAUSSIAN_CODEBOOKS[2]) / mx.sqrt(mx.array(128.0))
        np.testing.assert_allclose(
            np.array(cb), np.array(expected), atol=1e-6
        )


class TestRotation:
    """Tests for per-layer rotation matrices."""

    def test_orthogonality(self):
        """Π @ Πᵀ should be identity."""
        from olmlx.engine.turboquant import TurboQuantRotation

        rot = TurboQuantRotation(head_dim=128, seed=42)
        product = rot.matrix @ rot.matrix.T
        identity = mx.eye(128)
        np.testing.assert_allclose(
            np.array(product), np.array(identity), atol=1e-5
        )

    def test_deterministic(self):
        """Same seed should produce same rotation."""
        from olmlx.engine.turboquant import TurboQuantRotation

        r1 = TurboQuantRotation(head_dim=64, seed=7)
        r2 = TurboQuantRotation(head_dim=64, seed=7)
        np.testing.assert_array_equal(np.array(r1.matrix), np.array(r2.matrix))

    def test_per_layer_different(self):
        """Different seeds should produce different rotations."""
        from olmlx.engine.turboquant import TurboQuantRotation

        r1 = TurboQuantRotation(head_dim=64, seed=0)
        r2 = TurboQuantRotation(head_dim=64, seed=1)
        assert not np.allclose(np.array(r1.matrix), np.array(r2.matrix))

    def test_shape(self):
        from olmlx.engine.turboquant import TurboQuantRotation

        rot = TurboQuantRotation(head_dim=128, seed=0)
        assert rot.matrix.shape == (128, 128)


class TestQuantizeDequantize:
    """Tests for quantize/dequantize roundtrip."""

    def _random_vectors(self, shape, seed=0):
        """Generate random vectors with varying norms (like real KV vectors)."""
        np.random.seed(seed)
        x = np.random.randn(*shape).astype(np.float32)
        return mx.array(x)

    def test_roundtrip_mse_4bit(self):
        """4-bit roundtrip should have low MSE."""
        from olmlx.engine.turboquant import (
            TurboQuantRotation,
            turboquant_dequantize,
            turboquant_quantize,
        )

        rot = TurboQuantRotation(head_dim=128, seed=0)
        x = self._random_vectors((1, 1, 16, 128))

        indices, norms = turboquant_quantize(x, rot, bits=4)
        x_hat = turboquant_dequantize(indices, norms, rot, bits=4)

        # Relative MSE should be small for 4-bit
        mse = float(mx.mean((x - x_hat) ** 2))
        var = float(mx.mean(x**2))
        relative_mse = mse / var
        assert relative_mse < 0.05, f"4-bit relative MSE {relative_mse} too high"

    def test_roundtrip_mse_2bit(self):
        """2-bit roundtrip has higher MSE but still bounded."""
        from olmlx.engine.turboquant import (
            TurboQuantRotation,
            turboquant_dequantize,
            turboquant_quantize,
        )

        rot = TurboQuantRotation(head_dim=128, seed=0)
        x = self._random_vectors((1, 1, 16, 128))

        indices, norms = turboquant_quantize(x, rot, bits=2)
        x_hat = turboquant_dequantize(indices, norms, rot, bits=2)

        mse = float(mx.mean((x - x_hat) ** 2))
        var = float(mx.mean(x**2))
        relative_mse = mse / var
        assert relative_mse < 0.25, f"2-bit relative MSE {relative_mse} too high"

    def test_output_shapes(self):
        from olmlx.engine.turboquant import (
            TurboQuantRotation,
            turboquant_dequantize,
            turboquant_quantize,
        )

        rot = TurboQuantRotation(head_dim=64, seed=0)
        x = self._random_vectors((1, 4, 32, 64))

        indices, norms = turboquant_quantize(x, rot, bits=4)
        assert indices.shape == (1, 4, 32, 64)
        assert indices.dtype == mx.uint8
        assert norms.shape == (1, 4, 32, 1)

        x_hat = turboquant_dequantize(indices, norms, rot, bits=4)
        assert x_hat.shape == x.shape

    def test_batch_dimensions(self):
        """Should handle batch>1 and multiple heads."""
        from olmlx.engine.turboquant import (
            TurboQuantRotation,
            turboquant_dequantize,
            turboquant_quantize,
        )

        rot = TurboQuantRotation(head_dim=64, seed=0)
        x = self._random_vectors((2, 8, 10, 64))

        indices, norms = turboquant_quantize(x, rot, bits=2)
        x_hat = turboquant_dequantize(indices, norms, rot, bits=2)

        assert x_hat.shape == x.shape

    def test_single_token(self):
        """Should work with seq_len=1 (decoding step)."""
        from olmlx.engine.turboquant import (
            TurboQuantRotation,
            turboquant_dequantize,
            turboquant_quantize,
        )

        rot = TurboQuantRotation(head_dim=128, seed=0)
        x = self._random_vectors((1, 8, 1, 128))

        indices, norms = turboquant_quantize(x, rot, bits=4)
        x_hat = turboquant_dequantize(indices, norms, rot, bits=4)

        assert x_hat.shape == x.shape

    def test_norm_preservation(self):
        """Dequantized vectors should approximately preserve input norms."""
        from olmlx.engine.turboquant import (
            TurboQuantRotation,
            turboquant_dequantize,
            turboquant_quantize,
        )

        rot = TurboQuantRotation(head_dim=128, seed=0)
        x = self._random_vectors((1, 1, 100, 128))

        indices, norms = turboquant_quantize(x, rot, bits=4)
        x_hat = turboquant_dequantize(indices, norms, rot, bits=4)

        orig_norms = mx.sqrt(mx.sum(x**2, axis=-1))
        recon_norms = mx.sqrt(mx.sum(x_hat**2, axis=-1))

        # Norms should be close (within 20%)
        ratio = np.array(recon_norms / orig_norms)
        assert np.all(ratio > 0.7) and np.all(ratio < 1.3)


# ---------------------------------------------------------------------------
# TurboQuantKVCache tests
# ---------------------------------------------------------------------------


class TestTurboQuantKVCache:
    """Tests for the KV cache wrapper."""

    def _make_cache(self, bits=4, head_dim=64):
        from olmlx.engine.turboquant import TurboQuantRotation
        from olmlx.engine.turboquant_cache import TurboQuantKVCache

        rot_k = TurboQuantRotation(head_dim=head_dim, seed=0)
        rot_v = TurboQuantRotation(head_dim=head_dim, seed=1)
        return TurboQuantKVCache(bits=bits, rotation_key=rot_k, rotation_value=rot_v)

    def test_update_and_fetch_basic(self):
        """First update should store and return data."""
        cache = self._make_cache(bits=4, head_dim=64)
        keys = mx.random.normal((1, 4, 8, 64))
        values = mx.random.normal((1, 4, 8, 64))

        k_out, v_out = cache.update_and_fetch(keys, values)

        assert k_out.shape == (1, 4, 8, 64)
        assert v_out.shape == (1, 4, 8, 64)
        assert cache.offset == 8

    def test_sequential_updates(self):
        """Multiple updates should accumulate tokens."""
        cache = self._make_cache(bits=4, head_dim=64)

        # Prefill: 16 tokens
        k1 = mx.random.normal((1, 4, 16, 64))
        v1 = mx.random.normal((1, 4, 16, 64))
        k_out, v_out = cache.update_and_fetch(k1, v1)
        assert k_out.shape == (1, 4, 16, 64)
        assert cache.offset == 16

        # Decode: 1 token at a time
        k2 = mx.random.normal((1, 4, 1, 64))
        v2 = mx.random.normal((1, 4, 1, 64))
        k_out, v_out = cache.update_and_fetch(k2, v2)
        assert k_out.shape == (1, 4, 17, 64)
        assert v_out.shape == (1, 4, 17, 64)
        assert cache.offset == 17

    def test_trim(self):
        """Trim should reduce offset."""
        cache = self._make_cache(bits=4, head_dim=64)
        keys = mx.random.normal((1, 4, 32, 64))
        values = mx.random.normal((1, 4, 32, 64))
        cache.update_and_fetch(keys, values)

        trimmed = cache.trim(10)
        assert trimmed == 10
        assert cache.offset == 22

    def test_trim_clamps_to_offset(self):
        """Trim more than offset should clamp."""
        cache = self._make_cache(bits=4, head_dim=64)
        keys = mx.random.normal((1, 4, 5, 64))
        values = mx.random.normal((1, 4, 5, 64))
        cache.update_and_fetch(keys, values)

        trimmed = cache.trim(100)
        assert trimmed == 5
        assert cache.offset == 0

    def test_is_trimmable(self):
        cache = self._make_cache()
        assert cache.is_trimmable()

    def test_empty(self):
        cache = self._make_cache()
        assert cache.empty()

        keys = mx.random.normal((1, 4, 1, 64))
        values = mx.random.normal((1, 4, 1, 64))
        cache.update_and_fetch(keys, values)
        assert not cache.empty()

    def test_reconstruction_quality(self):
        """Dequantized cache should be close to original (4-bit)."""
        cache = self._make_cache(bits=4, head_dim=128)
        keys = mx.random.normal((1, 8, 32, 128))
        values = mx.random.normal((1, 8, 32, 128))

        k_out, v_out = cache.update_and_fetch(keys, values)

        # Check relative MSE
        k_mse = float(mx.mean((keys - k_out) ** 2))
        k_var = float(mx.mean(keys**2))
        assert k_mse / k_var < 0.05, f"Key 4-bit relative MSE too high: {k_mse/k_var}"

        v_mse = float(mx.mean((values - v_out) ** 2))
        v_var = float(mx.mean(values**2))
        assert v_mse / v_var < 0.05, f"Value 4-bit relative MSE too high: {v_mse/v_var}"

    def test_make_mask(self):
        """make_mask should delegate to create_attention_mask."""
        cache = self._make_cache()
        keys = mx.random.normal((1, 4, 10, 64))
        values = mx.random.normal((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        # Should not raise — just verify it's callable
        mask = cache.make_mask(N=1, return_array=True, window_size=None)
        # For N=1, mask should be None (single token attention)
        assert mask is None

    def test_fetch_after_trim_returns_correct_length(self):
        """After trim, next fetch should return trimmed + new tokens."""
        cache = self._make_cache(bits=4, head_dim=64)

        # Add 20 tokens
        k1 = mx.random.normal((1, 4, 20, 64))
        v1 = mx.random.normal((1, 4, 20, 64))
        cache.update_and_fetch(k1, v1)

        # Trim 5
        cache.trim(5)
        assert cache.offset == 15

        # Add 3 more
        k2 = mx.random.normal((1, 4, 3, 64))
        v2 = mx.random.normal((1, 4, 3, 64))
        k_out, v_out = cache.update_and_fetch(k2, v2)

        assert k_out.shape == (1, 4, 18, 64)
        assert cache.offset == 18


# ---------------------------------------------------------------------------
# make_turboquant_cache tests
# ---------------------------------------------------------------------------


class TestMakeTurboQuantCache:
    """Tests for the factory function that creates per-layer caches."""

    def test_creates_correct_number_of_layers(self):
        from unittest.mock import MagicMock

        from olmlx.engine.turboquant_cache import make_turboquant_cache

        model = MagicMock()
        model.layers = [MagicMock() for _ in range(4)]
        # head_dim from model.args
        model.args.head_dim = 64

        cache = make_turboquant_cache(model, bits=4)
        assert len(cache) == 4

    def test_each_layer_is_turboquant_cache(self):
        from unittest.mock import MagicMock

        from olmlx.engine.turboquant_cache import (
            TurboQuantKVCache,
            make_turboquant_cache,
        )

        model = MagicMock()
        model.layers = [MagicMock() for _ in range(2)]
        model.args.head_dim = 64

        cache = make_turboquant_cache(model, bits=4)
        for c in cache:
            assert isinstance(c, TurboQuantKVCache)

    def test_per_layer_different_rotations(self):
        from unittest.mock import MagicMock

        from olmlx.engine.turboquant_cache import make_turboquant_cache

        model = MagicMock()
        model.layers = [MagicMock() for _ in range(2)]
        model.args.head_dim = 64

        cache = make_turboquant_cache(model, bits=4)
        # Different layers should have different rotation matrices
        r0 = np.array(cache[0].rotation_key.matrix)
        r1 = np.array(cache[1].rotation_key.matrix)
        assert not np.allclose(r0, r1)

    def test_head_dim_fallback(self):
        """When model.args.head_dim is missing, derive from hidden_size / num_attention_heads."""
        from unittest.mock import MagicMock

        from olmlx.engine.turboquant_cache import make_turboquant_cache

        class FakeArgs:
            hidden_size = 512
            num_attention_heads = 8

        model = MagicMock()
        model.layers = [MagicMock() for _ in range(1)]
        model.args = FakeArgs()

        cache = make_turboquant_cache(model, bits=4)
        assert cache[0].rotation_key.matrix.shape == (64, 64)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestKvCacheQuantConfig:
    """Tests for the kv_cache_quant experimental setting."""

    def test_default_is_none(self):
        from olmlx.config import ExperimentalSettings

        s = ExperimentalSettings(
            _env_file=None,
            **{k: v.default for k, v in ExperimentalSettings.model_fields.items() if v.default is not None and k != "kv_cache_quant"},
        )
        assert s.kv_cache_quant is None

    def test_turboquant_4(self, monkeypatch):
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", "turboquant:4")
        from olmlx.config import ExperimentalSettings

        s = ExperimentalSettings(_env_file=None)
        assert s.kv_cache_quant == "turboquant:4"

    def test_turboquant_2(self, monkeypatch):
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", "turboquant:2")
        from olmlx.config import ExperimentalSettings

        s = ExperimentalSettings(_env_file=None)
        assert s.kv_cache_quant == "turboquant:2"


# ---------------------------------------------------------------------------
# Integration: _make_turboquant_prompt_cache
# ---------------------------------------------------------------------------


class TestInferenceIntegration:
    """Tests for TurboQuant integration in inference.py."""

    def test_make_turboquant_prompt_cache_creates_turboquant(self):
        from unittest.mock import MagicMock

        from olmlx.engine.inference import _make_turboquant_prompt_cache
        from olmlx.engine.turboquant_cache import TurboQuantKVCache

        model = MagicMock()
        model.layers = [MagicMock() for _ in range(2)]
        model.args.head_dim = 64

        cache = _make_turboquant_prompt_cache(model, bits=4)
        assert len(cache) == 2
        assert all(isinstance(c, TurboQuantKVCache) for c in cache)

    def test_make_turboquant_prompt_cache_bits(self):
        from unittest.mock import MagicMock

        from olmlx.engine.inference import _make_turboquant_prompt_cache

        model = MagicMock()
        model.layers = [MagicMock() for _ in range(1)]
        model.args.head_dim = 64

        cache_4 = _make_turboquant_prompt_cache(model, bits=4)
        assert cache_4[0].bits == 4

        cache_2 = _make_turboquant_prompt_cache(model, bits=2)
        assert cache_2[0].bits == 2


class TestBitsValidation:
    """Validate that unsupported bit-widths raise clear errors."""

    def test_invalid_bits_raises(self):
        from olmlx.engine.turboquant import get_codebook

        with pytest.raises(ValueError, match="TurboQuant supports"):
            get_codebook(bits=3, dim=128)

    def test_valid_bits_no_error(self):
        from olmlx.engine.turboquant import get_codebook

        get_codebook(bits=2, dim=128)
        get_codebook(bits=4, dim=128)


class TestDiskCacheGuard:
    """Verify TurboQuant caches are not saved to disk."""

    def test_turboquant_cache_not_serializable(self):
        from olmlx.engine.model_manager import _is_serializable_cache
        from olmlx.engine.turboquant import TurboQuantRotation
        from olmlx.engine.turboquant_cache import TurboQuantKVCache

        rot = TurboQuantRotation(head_dim=64, seed=0)
        cache = [TurboQuantKVCache(bits=4, rotation_key=rot, rotation_value=rot)]
        assert not _is_serializable_cache(cache)

    def test_standard_cache_serializable(self):
        from mlx_lm.models.cache import KVCache

        from olmlx.engine.model_manager import _is_serializable_cache

        cache = [KVCache()]
        assert _is_serializable_cache(cache)
