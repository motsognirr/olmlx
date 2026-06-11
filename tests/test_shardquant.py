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
            np.testing.assert_allclose(
                np.array(got), np.array(expected), atol=1e-4
            )

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
