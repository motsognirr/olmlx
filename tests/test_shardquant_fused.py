"""Tests for the fused shard decode path (#377 Tier 2) — reference backend."""

import copy

import mlx.core as mx
import pytest

from tests.test_shardquant_cache import _feed, _make_cache


def _grouped_q(nq, D, seed=7):
    mx.random.seed(seed)
    return mx.random.normal((1, nq, 1, D)).astype(mx.float16)


def _manual_sdpa_f32(q, k, v, scale):
    """Ground-truth attention in fp32: q (1,nq,1,D), k/v (1,Hk,S,D)."""
    B, nq, L, D = q.shape
    Hk = k.shape[1]
    g = nq // Hk
    qg = q.astype(mx.float32).reshape(B, Hk, g, D)
    s = (qg @ k.astype(mx.float32).swapaxes(-1, -2)) * scale
    w = mx.softmax(s, axis=-1)
    out = w @ v.astype(mx.float32)
    return out.reshape(B, nq, L, D)


class TestMiddleRefOps:
    def test_weighted_v_ref_matches_literal_decompress(self):
        from olmlx.engine.shardquant import shard_decompress_values
        from olmlx.engine.shardquant_fused import shard_middle_weighted_v_ref

        D, H = 64, 2
        cache = _make_cache(D=D, H=H, bits=4, sink=4, window=8)
        _feed(cache, 40, D=D, H=H, step=5)
        m = cache._mid_len
        assert m > 0
        mx.random.seed(3)
        w = mx.softmax(mx.random.normal((1, H, 2, m)), axis=-1)

        got = shard_middle_weighted_v_ref(w, cache)

        v_lit = shard_decompress_values(
            cache._v_mid,
            cache._v_mid_norms,
            cache.v_rotation,
            cache.v_codebooks,
            dtype=mx.float32,
        )[..., :m, :]
        want = w @ v_lit
        assert mx.allclose(got, want, atol=1e-4), float(mx.abs(got - want).max())

    def test_scores_ref_matches_decompressed_keys(self):
        from olmlx.engine.shardquant_fused import shard_middle_scores_ref

        D, H = 64, 2
        cache = _make_cache(D=D, H=H, bits=4, sink=4, window=8)
        _feed(cache, 40, D=D, H=H, step=5)
        m = cache._mid_len
        mx.random.seed(4)
        qg = mx.random.normal((1, H, 2, D))

        got = shard_middle_scores_ref(qg, cache)
        k_mid, _ = cache._decompress_middle(mx.float32)
        want = qg.astype(mx.float32) @ k_mid.swapaxes(-1, -2)
        assert got.shape == (1, H, 2, m)
        assert mx.allclose(got, want, atol=1e-5)


class TestFusedSdpaDecodeRef:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("rope", [True, False])
    def test_matches_tier1_attention(self, bits, rope):
        from olmlx.engine.shardquant_cache import ShardFusedKV
        from olmlx.engine.shardquant_fused import fused_sdpa_decode

        D, H, nq = 64, 2, 4
        cache = _make_cache(D=D, H=H, bits=bits, rank=48, rope=rope)
        _feed(cache, 50, D=D, H=H, step=5)
        # Tier-1 ground truth from an identical fork of the cache state.
        ref_cache = copy.deepcopy(cache)

        mx.random.seed(11)
        k_new = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        v_new = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        q = _grouped_q(nq, D)
        scale = D**-0.5

        k_full, v_full = ref_cache.update_and_fetch(k_new, v_new)
        want = _manual_sdpa_f32(q, k_full, v_full, scale)

        cache.update_and_fetch(k_new, v_new)  # same state advance
        exact_k = mx.concatenate([cache._k_sink, cache._k_win], axis=2)
        exact_v = mx.concatenate([cache._v_sink, cache._v_win], axis=2)
        handle = ShardFusedKV(cache=cache, k_exact=exact_k, v_exact=exact_v)
        got = fused_sdpa_decode(q, handle, scale, backend="ref")

        assert got.dtype == q.dtype
        assert mx.allclose(got.astype(mx.float32), want, atol=2e-3), float(
            mx.abs(got.astype(mx.float32) - want).max()
        )


class TestFusedCacheMode:
    def _fused_cache(self, **kw):
        cache = _make_cache(**kw)
        cache.fused = True
        return cache

    def test_handle_returned_only_for_single_token_with_middle(self):
        from olmlx.engine.shardquant_cache import ShardFusedKV

        D, H = 64, 2
        cache = self._fused_cache(D=D, H=H)
        # Prefill (multi-token): arrays, never a handle.
        k, v = _feed(cache, 30, D=D, H=H, step=10)[2]
        assert isinstance(k, mx.array) and isinstance(v, mx.array)
        # Decode with mid_len > 0: handle.
        assert cache._mid_len > 0
        k1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        out = cache.update_and_fetch(k1, k1)
        assert isinstance(out[0], ShardFusedKV) and out[0] is out[1]
        assert out[0].k_exact.shape[2] == cache._sink_len() + cache._win_len()

    def test_no_handle_before_middle_exists(self):
        D, H = 64, 2
        cache = self._fused_cache(D=D, H=H, sink=4, window=16)
        k1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        for _ in range(10):  # 10 tokens < sink + window: middle stays empty
            k, v = cache.update_and_fetch(k1, k1)
        assert isinstance(k, mx.array)

    def test_handle_exact_equals_materialized_exact_regions(self):
        D, H = 64, 2
        cache = self._fused_cache(D=D, H=H)
        _feed(cache, 30, D=D, H=H, step=10)
        ref = copy.deepcopy(cache)
        ref.fused = False
        k1 = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        h, _ = cache.update_and_fetch(k1, k1)
        k_full, _ = ref.update_and_fetch(k1, k1)
        m = cache._mid_len
        sink = cache._sink_len()
        assert mx.allclose(h.k_exact[..., :sink, :], k_full[..., :sink, :])
        assert mx.allclose(h.k_exact[..., sink:, :], k_full[..., sink + m :, :])

    def test_trim_and_state_unaffected_by_fused_flag(self):
        D, H = 64, 2
        cache = self._fused_cache(D=D, H=H)
        _feed(cache, 30, D=D, H=H, step=10)
        n = cache.trim(5)
        assert n == 5 and cache.offset == 25
        assert all(isinstance(a, mx.array) for a in cache.state)

    def test_make_shard_cache_fused_parameter(self):
        import inspect

        from olmlx.engine.shardquant_cache import make_shard_cache

        sig = inspect.signature(make_shard_cache)
        assert "fused" in sig.parameters
        assert sig.parameters["fused"].default is False
