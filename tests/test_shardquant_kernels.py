"""Metal-kernel parity tests for the fused shard decode path (#377 Tier 2).

Each kernel must match the plain-MLX reference math on the same cache.
GPU-only: skipped when Metal is unavailable.
"""

import mlx.core as mx
import pytest

from tests.test_shardquant_cache import _feed, _make_cache

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _decode_step(cache, D, H, seed=11):
    mx.random.seed(seed)
    k1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
    v1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
    cache.update_and_fetch(k1, v1)


class TestKScoreKernel:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("rope", [True, False])
    @pytest.mark.parametrize("rank", [None, 48])  # None = full rank
    def test_matches_reference(self, bits, rope, rank):
        from olmlx.engine.shardquant_fused import shard_middle_scores_ref
        from olmlx.engine.shardquant_kernels import shard_middle_scores_kernel

        D, H, G = 64, 2, 3
        cache = _make_cache(D=D, H=H, bits=bits, rank=rank, rope=rope)
        _feed(cache, 80, D=D, H=H, step=7)
        _decode_step(cache, D, H)
        assert cache._mid_len > 0
        mx.random.seed(5)
        qg = mx.random.normal((1, H, G, D))

        want = shard_middle_scores_ref(qg, cache)
        got = shard_middle_scores_kernel(qg, cache)
        assert got.shape == want.shape
        assert mx.allclose(got, want, atol=2e-3, rtol=1e-3), float(
            mx.abs(got - want).max()
        )

    def test_long_middle_capacity_padding(self):
        # mid_len far past one `step` growth, exercising the capacity-vs-
        # valid-length split and multiple threadgroup tiles.
        from olmlx.engine.shardquant_fused import shard_middle_scores_ref
        from olmlx.engine.shardquant_kernels import shard_middle_scores_kernel

        D, H = 64, 2
        cache = _make_cache(D=D, H=H, bits=4)
        _feed(cache, 700, D=D, H=H, step=64)
        mx.random.seed(6)
        qg = mx.random.normal((1, H, 4, D))
        want = shard_middle_scores_ref(qg, cache)
        got = shard_middle_scores_kernel(qg, cache)
        assert mx.allclose(got, want, atol=2e-3, rtol=1e-3)


class TestVAccumKernel:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_matches_reference(self, bits):
        from olmlx.engine.shardquant_fused import shard_middle_weighted_v_ref
        from olmlx.engine.shardquant_kernels import shard_middle_weighted_v_kernel

        D, H, G = 64, 2, 3
        cache = _make_cache(D=D, H=H, bits=bits)
        _feed(cache, 80, D=D, H=H, step=7)
        m = cache._mid_len
        se = 10
        mx.random.seed(9)
        w_full = mx.softmax(mx.random.normal((1, H, G, se + m)), axis=-1)

        want = shard_middle_weighted_v_ref(w_full[..., se:], cache)
        got = shard_middle_weighted_v_kernel(w_full, se, cache)
        assert got.shape == want.shape
        assert mx.allclose(got, want, atol=2e-4, rtol=1e-3), float(
            mx.abs(got - want).max()
        )

    def test_multi_chunk_partials(self, monkeypatch):
        # Force several j-chunks so the partial-sum reduction is exercised.
        from olmlx.engine import shardquant_kernels as sk
        from olmlx.engine.shardquant_fused import shard_middle_weighted_v_ref

        monkeypatch.setattr(sk, "_V_JCHUNK", 16)
        D, H = 64, 2
        cache = _make_cache(D=D, H=H, bits=4)
        _feed(cache, 120, D=D, H=H, step=8)
        m = cache._mid_len
        mx.random.seed(10)
        w_full = mx.softmax(mx.random.normal((1, H, 2, 5 + m)), axis=-1)
        want = shard_middle_weighted_v_ref(w_full[..., 5:], cache)
        got = sk.shard_middle_weighted_v_kernel(w_full, 5, cache)
        assert mx.allclose(got, want, atol=2e-4, rtol=1e-3)


class TestKernelGate:
    def test_supported_predicate(self):
        from olmlx.engine.shardquant_kernels import kernels_supported

        D, H = 64, 2
        ok = _make_cache(D=D, H=H, bits=4)
        assert kernels_supported(ok, D)
        # Traditional rope layout → unsupported (ref fallback).
        trad = _make_cache(D=D, H=H, bits=4)
        trad.rope_spec.traditional = True
        assert not kernels_supported(trad, D)
        # head_dim not a multiple of 32 → unsupported.
        odd = _make_cache(D=48, H=H, bits=4)
        assert not kernels_supported(odd, 48)
