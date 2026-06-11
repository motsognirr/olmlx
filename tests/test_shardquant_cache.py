"""Tests for ShardKVCache (#377 Tier 1)."""

import copy

import mlx.core as mx
import numpy as np
import pytest


def _random_orthogonal(dim, seed):
    rng = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rng.randn(dim, dim).astype(np.float32))
    return q


def _make_cache(D=16, H=2, bits=4, rank=None, sink=4, window=8, rope=True):
    """Build a ShardKVCache with synthetic (but valid) calibration tensors."""
    from olmlx.engine.shardquant import RopeSpec, fit_vq_codebooks, make_v_rotation
    from olmlx.engine.shardquant_cache import ShardKVCache
    from olmlx.engine.spectralquant import fit_codebook

    rank = rank if rank is not None else D
    basis = mx.array(np.stack([_random_orthogonal(D, 10 + h) for h in range(H)]))
    rng = np.random.RandomState(0)
    k_codebook = fit_codebook(
        mx.array(rng.randn(4096).astype(np.float32) * 0.3), bits=bits
    )
    v_rot = make_v_rotation(D)
    g = 8 // bits
    sample = rng.randn(4096, D).astype(np.float32)
    sample /= np.linalg.norm(sample, axis=-1, keepdims=True)
    v_codebooks = fit_vq_codebooks(sample @ np.array(v_rot).T, group_size=g)
    spec = None
    if rope:
        freqs = mx.power(
            mx.array(10000.0, dtype=mx.float32),
            -mx.arange(0, D, 2, dtype=mx.float32) / D,
        )
        spec = RopeSpec(dims=D, freqs=freqs, traditional=False)
    return ShardKVCache(
        rope_spec=spec,
        k_basis=basis,
        k_rank=rank,
        k_codebook=k_codebook,
        k_bits=bits,
        v_rotation=v_rot,
        v_codebooks=v_codebooks,
        sink_size=sink,
        window_size=window,
    )


def _feed(cache, total, D=16, H=2, step=1, seed=0):
    """Feed `total` tokens, returning the exact K/V that were fed."""
    mx.random.seed(seed)
    ks = mx.random.normal((1, H, total, D)).astype(mx.float16)
    vs = mx.random.normal((1, H, total, D)).astype(mx.float16)
    out = None
    for s0 in range(0, total, step):
        out = cache.update_and_fetch(
            ks[..., s0 : s0 + step, :], vs[..., s0 : s0 + step, :]
        )
    return ks, vs, out


class TestShardKVCacheBasics:
    def test_output_shape_and_dtype(self):
        cache = _make_cache()
        ks, vs, (k, v) = _feed(cache, 20)
        assert k.shape == (1, 2, 20, 16)
        assert v.shape == (1, 2, 20, 16)
        assert k.dtype == mx.float16
        assert cache.offset == 20

    def test_small_prompt_stays_exact(self):
        """Until sink+window overflows, everything is FP16-exact."""
        cache = _make_cache(sink=4, window=8)
        ks, vs, (k, v) = _feed(cache, 10)  # 10 <= 4 + 8
        np.testing.assert_array_equal(np.array(k), np.array(ks))
        np.testing.assert_array_equal(np.array(v), np.array(vs))

    def test_sink_and_window_exact_after_overflow(self):
        cache = _make_cache(sink=4, window=8)
        ks, vs, (k, v) = _feed(cache, 40)
        # First sink tokens exact
        np.testing.assert_array_equal(np.array(k[..., :4, :]), np.array(ks[..., :4, :]))
        np.testing.assert_array_equal(np.array(v[..., :4, :]), np.array(vs[..., :4, :]))
        # Last window tokens exact
        np.testing.assert_array_equal(
            np.array(k[..., -8:, :]), np.array(ks[..., -8:, :])
        )
        np.testing.assert_array_equal(
            np.array(v[..., -8:, :]), np.array(vs[..., -8:, :])
        )

    def test_middle_reconstruction_quality(self):
        cache = _make_cache(sink=4, window=8, bits=8)
        ks, vs, (k, v) = _feed(cache, 64)
        mid_k = np.array(k[..., 4:-8, :], dtype=np.float32)
        ref_k = np.array(ks[..., 4:-8, :], dtype=np.float32)
        cos = np.sum(mid_k * ref_k, -1) / (
            np.linalg.norm(mid_k, axis=-1) * np.linalg.norm(ref_k, axis=-1) + 1e-9
        )
        assert cos.mean() > 0.9, f"K middle cosine {cos.mean()}"
        mid_v = np.array(v[..., 4:-8, :], dtype=np.float32)
        ref_v = np.array(vs[..., 4:-8, :], dtype=np.float32)
        cos_v = np.sum(mid_v * ref_v, -1) / (
            np.linalg.norm(mid_v, axis=-1) * np.linalg.norm(ref_v, axis=-1) + 1e-9
        )
        assert cos_v.mean() > 0.85, f"V middle cosine {cos_v.mean()}"

    def test_prefill_then_decode_matches_pure_decode(self):
        """One 30-token prefill + decode steps ≈ 1-token-at-a-time feed."""
        c1 = _make_cache()
        c2 = _make_cache()
        mx.random.seed(3)
        ks = mx.random.normal((1, 2, 34, 16)).astype(mx.float16)
        vs = mx.random.normal((1, 2, 34, 16)).astype(mx.float16)
        c1.update_and_fetch(ks[..., :30, :], vs[..., :30, :])
        k1 = v1 = None
        for i in range(30, 34):
            k1, v1 = c1.update_and_fetch(ks[..., i : i + 1, :], vs[..., i : i + 1, :])
        out = None
        for i in range(34):
            out = c2.update_and_fetch(ks[..., i : i + 1, :], vs[..., i : i + 1, :])
        k2, v2 = out
        np.testing.assert_allclose(
            np.array(k1, dtype=np.float32),
            np.array(k2, dtype=np.float32),
            atol=5e-2,
        )

    def test_no_rope_spec_still_works(self):
        cache = _make_cache(rope=False)
        ks, vs, (k, v) = _feed(cache, 30)
        assert k.shape == (1, 2, 30, 16)

    def test_empty(self):
        cache = _make_cache()
        assert cache.empty()
        _feed(cache, 3)
        assert not cache.empty()


class TestShardKVCacheTrim:
    def test_is_trimmable(self):
        assert _make_cache().is_trimmable()

    def test_trim_within_window(self):
        cache = _make_cache(sink=4, window=8)
        ks, vs, _ = _feed(cache, 10)
        assert cache.trim(3) == 3
        assert cache.offset == 7
        k, v = cache.update_and_fetch(ks[..., 7:8, :], vs[..., 7:8, :])
        assert k.shape[2] == 8
        np.testing.assert_array_equal(np.array(k[..., :8, :]), np.array(ks[..., :8, :]))

    def test_trim_into_middle(self):
        cache = _make_cache(sink=4, window=8)
        ks, vs, _ = _feed(cache, 40)
        # 40 = 4 sink + 28 middle + 8 window; trim 20 reaches the middle.
        assert cache.trim(20) == 20
        assert cache.offset == 20
        k, v = cache.update_and_fetch(ks[..., 20:21, :], vs[..., 20:21, :])
        assert k.shape[2] == 21
        # Sink still exact
        np.testing.assert_array_equal(np.array(k[..., :4, :]), np.array(ks[..., :4, :]))

    def test_trim_to_empty(self):
        cache = _make_cache()
        ks, vs, _ = _feed(cache, 30)
        assert cache.trim(30) == 30
        assert cache.offset == 0
        assert cache.empty()
        # Reusable after full trim
        _feed(cache, 5)
        assert cache.offset == 5

    def test_trim_clamps_to_offset(self):
        cache = _make_cache()
        _feed(cache, 10)
        assert cache.trim(99) == 10
        assert cache.offset == 0


class TestShardKVCacheStateAndCopy:
    def test_state_exposes_arrays_and_setter_raises(self):
        cache = _make_cache()
        _feed(cache, 40)
        st = cache.state
        assert len(st) > 0
        assert all(isinstance(a, mx.array) for a in st)
        with pytest.raises(NotImplementedError):
            cache.state = st

    def test_state_empty_when_fresh(self):
        assert _make_cache().state == []

    def test_deepcopy_then_diverge(self):
        """Snapshot path: deepcopy, then mutating the original must not
        affect the copy (and vice versa)."""
        cache = _make_cache()
        ks, vs, _ = _feed(cache, 40)
        snap = copy.deepcopy(cache)
        mx.eval(list(snap.state))
        snap.update_and_fetch(ks[..., :1, :] * 0, vs[..., :1, :] * 0)
        # Continue the original with different tokens
        mx.random.seed(99)
        more_k = mx.random.normal((1, 2, 5, 16)).astype(mx.float16)
        cache.update_and_fetch(more_k, more_k)
        assert cache.offset == 45
        assert snap.offset == 41

    def test_no_dtype_attribute(self):
        """No mx.Dtype attrs — keeps the default deepcopy walk safe
        (the TurboQuant pickle failure mode)."""
        cache = _make_cache()
        _feed(cache, 40)
        assert not any(isinstance(v, mx.Dtype) for v in cache.__dict__.values())


class TestMakeShardCache:
    def _calib_entry(self, D=16, H=2, bits=4):
        from olmlx.engine.shardquant import fit_vq_codebooks, make_v_rotation
        from olmlx.engine.spectralquant import fit_codebook

        rng = np.random.RandomState(0)
        sample = rng.randn(1024, D).astype(np.float32)
        sample /= np.linalg.norm(sample, axis=-1, keepdims=True)
        v_rot = make_v_rotation(D)
        return {
            "k_basis": mx.array(np.stack([_random_orthogonal(D, h) for h in range(H)])),
            "k_rank": D // 2,
            "k_codebook": fit_codebook(
                mx.array(rng.randn(2048).astype(np.float32) * 0.3), bits=bits
            ),
            "v_rotation": v_rot,
            "v_codebooks": fit_vq_codebooks(
                sample @ np.array(v_rot).T, group_size=8 // bits
            ),
            "rope_freqs": None,
            "rope_dims": None,
            "rope_traditional": False,
        }

    def test_make_shard_cache_quantizes_attention_layers(self, tmp_path, monkeypatch):
        from mlx_lm.models.cache import KVCache

        from olmlx.engine.shardquant_cache import ShardKVCache, make_shard_cache

        calibration = {0: self._calib_entry(), 1: self._calib_entry()}
        monkeypatch.setattr(
            "olmlx.engine.shardquant_calibrate.load_shard_calibration",
            lambda d: (calibration, {"bits": 4, "head_dim": 16, "n_kv_heads": 2}),
        )

        class Model:
            layers = [object(), object()]

            def make_cache(self):
                return [KVCache(), KVCache()]

        caches = make_shard_cache(Model(), tmp_path, bits=4)
        assert len(caches) == 2
        assert all(isinstance(c, ShardKVCache) for c in caches)

    def test_make_shard_cache_preserves_non_attention(self, tmp_path, monkeypatch):
        from mlx_lm.models.cache import ArraysCache, KVCache

        from olmlx.engine.shardquant_cache import ShardKVCache, make_shard_cache

        calibration = {0: self._calib_entry(), 1: self._calib_entry()}
        monkeypatch.setattr(
            "olmlx.engine.shardquant_calibrate.load_shard_calibration",
            lambda d: (calibration, {"bits": 4, "head_dim": 16, "n_kv_heads": 2}),
        )

        ssm = ArraysCache(size=2)

        class Model:
            layers = [object(), object()]

            def make_cache(self):
                return [ssm, KVCache()]

        caches = make_shard_cache(Model(), tmp_path, bits=4)
        assert caches[0] is ssm
        assert isinstance(caches[1], ShardKVCache)

    def test_missing_layer_calibration_falls_back(self, tmp_path, monkeypatch):
        from mlx_lm.models.cache import KVCache

        from olmlx.engine.shardquant_cache import ShardKVCache, make_shard_cache

        calibration = {0: self._calib_entry()}  # layer 1 missing
        monkeypatch.setattr(
            "olmlx.engine.shardquant_calibrate.load_shard_calibration",
            lambda d: (calibration, {"bits": 4, "head_dim": 16, "n_kv_heads": 2}),
        )

        class Model:
            layers = [object(), object()]

            def make_cache(self):
                return [KVCache(), KVCache()]

        caches = make_shard_cache(Model(), tmp_path, bits=4)
        assert isinstance(caches[0], ShardKVCache)
        assert isinstance(caches[1], KVCache)
        assert not isinstance(caches[1], ShardKVCache)
