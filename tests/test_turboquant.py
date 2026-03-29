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
        np.testing.assert_allclose(np.array(cb), np.array(expected), atol=1e-6)


class TestRotation:
    """Tests for per-layer rotation matrices."""

    def test_orthogonality(self):
        """Π @ Πᵀ should be identity."""
        from olmlx.engine.turboquant import TurboQuantRotation

        rot = TurboQuantRotation(head_dim=128, seed=42)
        product = rot.matrix @ rot.matrix.T
        identity = mx.eye(128)
        np.testing.assert_allclose(np.array(product), np.array(identity), atol=1e-5)

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

    def test_pack_unpack_4bit(self):
        """4-bit pack/unpack should roundtrip perfectly."""
        from olmlx.engine.turboquant import pack_indices, unpack_indices

        indices = mx.array([0, 15, 3, 12, 7, 8, 1, 14], dtype=mx.uint8).reshape(
            1, 1, 1, 8
        )
        packed = pack_indices(indices, bits=4)
        assert packed.shape == (1, 1, 1, 4)
        unpacked = unpack_indices(packed, bits=4, head_dim=8)
        np.testing.assert_array_equal(np.array(unpacked), np.array(indices))

    def test_pack_unpack_2bit(self):
        """2-bit pack/unpack should roundtrip perfectly."""
        from olmlx.engine.turboquant import pack_indices, unpack_indices

        indices = mx.array([0, 3, 1, 2, 3, 0, 2, 1], dtype=mx.uint8).reshape(1, 1, 1, 8)
        packed = pack_indices(indices, bits=2)
        assert packed.shape == (1, 1, 1, 2)
        unpacked = unpack_indices(packed, bits=2, head_dim=8)
        np.testing.assert_array_equal(np.array(unpacked), np.array(indices))

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
        # 4-bit: 2 indices per byte → head_dim // 2
        assert indices.shape == (1, 4, 32, 32)
        assert indices.dtype == mx.uint8
        assert norms.shape == (1, 4, 32, 1)

        x_hat = turboquant_dequantize(indices, norms, rot, bits=4)
        assert x_hat.shape == x.shape

    def test_output_shapes_2bit(self):
        from olmlx.engine.turboquant import (
            TurboQuantRotation,
            turboquant_quantize,
        )

        rot = TurboQuantRotation(head_dim=64, seed=0)
        x = self._random_vectors((1, 4, 32, 64))

        indices, norms = turboquant_quantize(x, rot, bits=2)
        # 2-bit: 4 indices per byte → head_dim // 4
        assert indices.shape == (1, 4, 32, 16)
        assert indices.dtype == mx.uint8

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
        assert k_mse / k_var < 0.05, f"Key 4-bit relative MSE too high: {k_mse / k_var}"

        v_mse = float(mx.mean((values - v_out) ** 2))
        v_var = float(mx.mean(values**2))
        assert v_mse / v_var < 0.05, (
            f"Value 4-bit relative MSE too high: {v_mse / v_var}"
        )

    def test_no_public_bits_attribute(self):
        """TurboQuantKVCache must not expose a public 'bits' attribute.

        mlx-lm's scaled_dot_product_attention uses ``hasattr(cache, 'bits')``
        to decide whether to use quantized SDPA. TurboQuantKVCache returns
        dequantized tensors, so it must not trigger that path.
        """
        cache = self._make_cache(bits=4, head_dim=64)
        assert not hasattr(cache, "bits"), (
            "TurboQuantKVCache must not have a public 'bits' attribute — "
            "mlx-lm uses hasattr(cache, 'bits') to dispatch to quantized SDPA"
        )

    def test_state_returns_arrays_for_eval(self):
        """state property must return arrays so mx.eval(c.state) works.

        mlx-lm's generate_step calls mx.eval([c.state for c in prompt_cache])
        to force lazy evaluation. This must not raise.
        """
        cache = self._make_cache(bits=4, head_dim=64)
        keys = mx.random.normal((1, 4, 8, 64))
        values = mx.random.normal((1, 4, 8, 64))
        cache.update_and_fetch(keys, values)

        state = cache.state
        # Should be evaluable without error
        mx.eval(state)

    def test_state_empty_cache(self):
        """state on an empty cache should return an empty list (like _BaseCache)."""
        cache = self._make_cache(bits=4, head_dim=64)
        state = cache.state
        assert state == []

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

    def test_head_dim_fallback_from_k_proj(self):
        """When model.args.head_dim is missing, derive from k_proj weight shape."""
        from unittest.mock import MagicMock

        from olmlx.engine.turboquant_cache import make_turboquant_cache

        class FakeArgs:
            # Gemma 3 style: head_dim not in args, and hidden_size // num_heads is wrong
            num_key_value_heads = 4

        model = MagicMock()
        layer = MagicMock()
        # k_proj output dim = num_kv_heads * head_dim = 4 * 64 = 256
        layer.self_attn.k_proj.weight = mx.zeros((256, 512))
        model.layers = [layer]
        model.args = FakeArgs()

        cache = make_turboquant_cache(model, bits=4)
        assert cache[0].rotation_key.matrix.shape == (64, 64)

    def test_head_dim_from_args(self):
        """When model.args.head_dim is present, use it directly."""
        from unittest.mock import MagicMock

        from olmlx.engine.turboquant_cache import make_turboquant_cache

        model = MagicMock()
        model.layers = [MagicMock() for _ in range(1)]
        model.args.head_dim = 128

        cache = make_turboquant_cache(model, bits=4)
        assert cache[0].rotation_key.matrix.shape == (128, 128)

    def test_hybrid_model_preserves_non_kv_caches(self):
        """Hybrid models (e.g. Nemotron-H) have SSM layers needing ArraysCache."""
        from unittest.mock import MagicMock

        from mlx_lm.models.cache import ArraysCache, KVCache

        from olmlx.engine.turboquant_cache import (
            TurboQuantKVCache,
            make_turboquant_cache,
        )

        model = MagicMock()
        model.layers = [MagicMock() for _ in range(4)]
        model.args.head_dim = 64
        # Simulate Nemotron-H: SSM, attention, SSM, attention
        model.make_cache.return_value = [
            ArraysCache(size=2),
            KVCache(),
            ArraysCache(size=2),
            KVCache(),
        ]

        cache = make_turboquant_cache(model, bits=4)
        assert len(cache) == 4
        assert isinstance(cache[0], ArraysCache)
        assert isinstance(cache[1], TurboQuantKVCache)
        assert isinstance(cache[2], ArraysCache)
        assert isinstance(cache[3], TurboQuantKVCache)

    def test_hybrid_model_sparse_cache(self):
        """Nemotron-H: make_cache() returns fewer entries than model.layers.

        Nemotron-H has block types M, *, -, E but only M and * get cache entries.
        make_cache() returns a list shorter than len(model.layers). The factory
        must iterate over default_caches as-is, not pad to num_layers.
        """
        from unittest.mock import MagicMock

        from mlx_lm.models.cache import ArraysCache, KVCache

        from olmlx.engine.turboquant_cache import (
            TurboQuantKVCache,
            make_turboquant_cache,
        )

        model = MagicMock()
        # 6 layers total: M, *, -, E, M, *
        model.layers = [MagicMock() for _ in range(6)]
        model.args.head_dim = 64
        # make_cache returns only 4 entries (for M and * blocks)
        model.make_cache.return_value = [
            ArraysCache(size=2),  # M
            KVCache(),  # *
            ArraysCache(size=2),  # M
            KVCache(),  # *
        ]

        cache = make_turboquant_cache(model, bits=4)
        # Result should have 4 entries matching make_cache output, not 6
        assert len(cache) == 4
        assert isinstance(cache[0], ArraysCache)
        assert isinstance(cache[1], TurboQuantKVCache)
        assert isinstance(cache[2], ArraysCache)
        assert isinstance(cache[3], TurboQuantKVCache)

    def test_model_without_make_cache_all_turboquant(self):
        """Models without make_cache() get TurboQuantKVCache for all layers."""
        from unittest.mock import MagicMock

        from olmlx.engine.turboquant_cache import (
            TurboQuantKVCache,
            make_turboquant_cache,
        )

        model = MagicMock(spec=[])  # no make_cache attribute
        model.layers = [MagicMock() for _ in range(3)]
        model.args = MagicMock()
        model.args.head_dim = 64

        cache = make_turboquant_cache(model, bits=4)
        assert len(cache) == 3
        assert all(isinstance(c, TurboQuantKVCache) for c in cache)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestKvCacheQuantConfig:
    """Tests for the kv_cache_quant experimental setting."""

    def test_default_is_none(self):
        from olmlx.config import ExperimentalSettings

        s = ExperimentalSettings(
            _env_file=None,
            **{
                k: v.default
                for k, v in ExperimentalSettings.model_fields.items()
                if v.default is not None and k != "kv_cache_quant"
            },
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

    def test_invalid_bits_rejected(self, monkeypatch):
        """Unsupported bit-width should fail at config validation time."""
        from pydantic import ValidationError

        from olmlx.config import ExperimentalSettings

        monkeypatch.setenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", "turboquant:3")
        with pytest.raises(ValidationError):
            ExperimentalSettings(_env_file=None)

    def test_missing_bits_rejected(self, monkeypatch):
        """Malformed value without bits should fail at config validation."""
        from pydantic import ValidationError

        from olmlx.config import ExperimentalSettings

        monkeypatch.setenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", "turboquant:")
        with pytest.raises(ValidationError):
            ExperimentalSettings(_env_file=None)

    def test_unknown_method_rejected(self, monkeypatch):
        from pydantic import ValidationError

        from olmlx.config import ExperimentalSettings

        monkeypatch.setenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", "foo:4")
        with pytest.raises(ValidationError):
            ExperimentalSettings(_env_file=None)

    def test_bare_string_rejected(self, monkeypatch):
        from pydantic import ValidationError

        from olmlx.config import ExperimentalSettings

        monkeypatch.setenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", "turboquant")
        with pytest.raises(ValidationError):
            ExperimentalSettings(_env_file=None)


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
        assert cache_4[0]._bits == 4

        cache_2 = _make_turboquant_prompt_cache(model, bits=2)
        assert cache_2[0]._bits == 2


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
