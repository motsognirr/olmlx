"""Tests for the KVarN variance-normalized KV cache quantizer (#748)."""

import copy
import threading
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest


def _run_in_thread(fn):
    """Run ``fn()`` on a fresh thread; return {'value': ...} or {'error': exc}."""
    result: dict = {}

    def _target():
        try:
            result["value"] = fn()
        except Exception as exc:  # noqa: BLE001 — the exception IS the result
            result["error"] = exc

    t = threading.Thread(target=_target)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "worker thread hung"
    return result


def _make_cache(key_bits=4, value_bits=2, head_dim=128, layer=0):
    from olmlx.engine.kvarn import KVarNRotation
    from olmlx.engine.kvarn_cache import KVarNKVCache

    return KVarNKVCache(
        key_bits=key_bits,
        value_bits=value_bits,
        rotation_key=KVarNRotation(head_dim=head_dim, seed=layer * 2),
        rotation_value=KVarNRotation(head_dim=head_dim, seed=layer * 2 + 1),
    )


# ---------------------------------------------------------------------------
# Spec parsing / config validation
# ---------------------------------------------------------------------------


class TestParseKvarnBits:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("k4v2", (4, 2)),
            ("k2v4", (2, 4)),
            ("k4v4", (4, 4)),
            ("k2v2", (2, 2)),
            ("4", (4, 4)),
            ("2", (2, 2)),
        ],
    )
    def test_valid(self, spec, expected):
        from olmlx.engine.kvarn import parse_kvarn_bits

        assert parse_kvarn_bits(spec) == expected

    @pytest.mark.parametrize(
        "spec",
        [
            "",
            "k4",
            "v2",
            "k3v2",
            "k4v3",
            "8",
            "k4v2x",
            "K4V2",
            "4v2",
            "k 4v2",
            "k4v2\n",
        ],
    )
    def test_invalid(self, spec):
        from olmlx.engine.kvarn import parse_kvarn_bits

        with pytest.raises(ValueError):
            parse_kvarn_bits(spec)


class TestKvarnConfigValidation:
    @pytest.mark.parametrize("v", ["kvarn:k4v2", "kvarn:k2v4", "kvarn:4", "kvarn:2"])
    def test_accepted(self, v):
        from olmlx.config import validate_kv_cache_quant_format

        assert validate_kv_cache_quant_format(v) == v

    @pytest.mark.parametrize("v", ["kvarn:k4v3", "kvarn:8", "kvarn:", "kvarn"])
    def test_rejected(self, v):
        from olmlx.config import validate_kv_cache_quant_format

        with pytest.raises(ValueError):
            validate_kv_cache_quant_format(v)

    def test_settings_env(self, monkeypatch):
        from olmlx.config import Settings

        monkeypatch.setenv("OLMLX_KV_CACHE_QUANT", "kvarn:k4v2")
        assert Settings(_env_file=None).kv_cache_quant == "kvarn:k4v2"

    def test_existing_methods_still_valid(self):
        from olmlx.config import validate_kv_cache_quant_format

        for v in ("turboquant:2", "spectral:4", "shard:8"):
            assert validate_kv_cache_quant_format(v) == v

    def test_parse_kv_cache_quant_kv(self):
        from olmlx.engine.kv_budget import _parse_kv_cache_quant_kv

        assert _parse_kv_cache_quant_kv("kvarn:k4v2") == ("kvarn", 4, 2)
        assert _parse_kv_cache_quant_kv("kvarn:4") == ("kvarn", 4, 4)
        assert _parse_kv_cache_quant_kv("turboquant:2") == ("turboquant", 2, 2)


# ---------------------------------------------------------------------------
# Hadamard rotation
# ---------------------------------------------------------------------------


class TestChooseTile:
    @pytest.mark.parametrize(
        "head_dim,tile",
        [(128, 128), (64, 64), (256, 128), (512, 128), (96, 32), (80, 16), (72, None)],
    )
    def test_choose_tile(self, head_dim, tile):
        from olmlx.engine.kvarn import choose_tile

        assert choose_tile(head_dim) == tile


class TestKVarNRotation:
    def test_orthogonal(self):
        from olmlx.engine.kvarn import KVarNRotation

        rot = KVarNRotation(head_dim=128, seed=0)
        m = rot.dense_matrix()
        np.testing.assert_allclose(m @ m.T, np.eye(128), atol=1e-5)

    def test_block_diagonal_tiles(self):
        """head_dim=256 → two independent 128-wide Hadamard tiles."""
        from olmlx.engine.kvarn import KVarNRotation

        rot = KVarNRotation(head_dim=256, seed=0)
        assert rot.tile == 128
        m = rot.dense_matrix()
        assert np.all(m[:128, 128:] == 0)
        assert np.all(m[128:, :128] == 0)
        # Hadamard entries all have magnitude 1/sqrt(tile)
        block = np.abs(m[:128, :128])
        np.testing.assert_allclose(block, 1 / np.sqrt(128), rtol=1e-6)

    def test_seed_deterministic_and_distinct(self):
        from olmlx.engine.kvarn import KVarNRotation

        a = KVarNRotation(head_dim=64, seed=3).dense_matrix()
        b = KVarNRotation(head_dim=64, seed=3).dense_matrix()
        c = KVarNRotation(head_dim=64, seed=4).dense_matrix()
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_unsupported_head_dim_raises(self):
        from olmlx.engine.kvarn import KVarNRotation

        with pytest.raises(ValueError, match="head_dim"):
            KVarNRotation(head_dim=72, seed=0)

    def test_eval_safe_from_other_thread(self):
        """Built on the event-loop thread, used on the generation worker
        (mlx thread-local streams, #499): must be materialized leaves."""
        from olmlx.engine.kvarn import KVarNRotation, kvarn_quantize

        built = _run_in_thread(lambda: KVarNRotation(head_dim=64, seed=1))
        rot = built["value"]

        def use():
            x = mx.random.normal((1, 2, 4, 64))
            packed, stats = kvarn_quantize(x, rot, bits=4)
            mx.eval(packed, stats, rot.signs, rot.hadamard)
            return True

        res = _run_in_thread(use)
        assert "error" not in res, res.get("error")


# ---------------------------------------------------------------------------
# Quantize / dequantize
# ---------------------------------------------------------------------------


def _norms(x):
    return np.linalg.norm(np.array(x.astype(mx.float32)), axis=-1)


class TestQuantizeDequantize:
    @pytest.mark.parametrize("bits", [2, 4])
    def test_shapes(self, bits):
        from olmlx.engine.kvarn import KVarNRotation, kvarn_dequantize, kvarn_quantize

        rot = KVarNRotation(head_dim=128, seed=0)
        x = mx.random.normal((1, 4, 7, 128))
        packed, stats = kvarn_quantize(x, rot, bits=bits)
        assert packed.dtype == mx.uint8
        assert packed.shape == (1, 4, 7, 128 // (8 // bits))
        assert stats.dtype == mx.float32
        assert stats.shape == (1, 4, 7, 2)  # (mean, scale) x 1 tile
        out = kvarn_dequantize(packed, stats, rot, bits=bits, dtype=mx.float16)
        assert out.shape == x.shape
        assert out.dtype == mx.float16

    def test_stats_per_tile(self):
        from olmlx.engine.kvarn import KVarNRotation, kvarn_quantize

        rot = KVarNRotation(head_dim=96, seed=0)  # tile 32 → 3 tiles
        _, stats = kvarn_quantize(mx.random.normal((1, 1, 3, 96)), rot, bits=2)
        assert stats.shape == (1, 1, 3, 6)

    @pytest.mark.parametrize("bits", [2, 4])
    def test_norm_preserved_per_token(self, bits):
        """The headline property: per-token magnitude survives quantization
        (variance-normalized scale), even for wildly different token scales."""
        from olmlx.engine.kvarn import KVarNRotation, kvarn_dequantize, kvarn_quantize

        mx.random.seed(0)
        rot = KVarNRotation(head_dim=128, seed=0)
        scales = mx.array([0.01, 1.0, 50.0, 3000.0]).reshape(1, 1, 4, 1)
        x = mx.random.normal((1, 2, 4, 128)) * scales
        packed, stats = kvarn_quantize(x, rot, bits=bits)
        out = kvarn_dequantize(packed, stats, rot, bits=bits)
        np.testing.assert_allclose(_norms(out), _norms(x), rtol=1e-4)

    def test_turboquant_shrinks_where_kvarn_does_not(self):
        """Documents the motivation: Lloyd-Max reconstruction is biased toward
        smaller magnitudes; KVarN's scale undoes that bias."""
        from olmlx.engine.kvarn import KVarNRotation, kvarn_dequantize, kvarn_quantize
        from olmlx.engine.turboquant import (
            TurboQuantRotation,
            turboquant_dequantize,
            turboquant_quantize,
        )

        mx.random.seed(1)
        x = mx.random.normal((1, 4, 64, 128))
        trot = TurboQuantRotation(head_dim=128, seed=0)
        ti, tn = turboquant_quantize(x, trot, bits=2)
        tq_ratio = float(
            np.mean(_norms(turboquant_dequantize(ti, tn, trot, 2)) / _norms(x))
        )

        krot = KVarNRotation(head_dim=128, seed=0)
        kp, ks = kvarn_quantize(x, krot, bits=2)
        kv_ratio = float(np.mean(_norms(kvarn_dequantize(kp, ks, krot, 2)) / _norms(x)))

        assert tq_ratio < 0.97
        assert kv_ratio == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("bits,min_cos", [(2, 0.9), (4, 0.99)])
    def test_reconstruction_quality(self, bits, min_cos):
        from olmlx.engine.kvarn import KVarNRotation, kvarn_dequantize, kvarn_quantize

        mx.random.seed(2)
        rot = KVarNRotation(head_dim=128, seed=0)
        x = mx.random.normal((1, 4, 32, 128))
        packed, stats = kvarn_quantize(x, rot, bits=bits)
        out = np.array(kvarn_dequantize(packed, stats, rot, bits=bits))
        xn = np.array(x)
        cos = np.sum(out * xn, -1) / (
            np.linalg.norm(out, axis=-1) * np.linalg.norm(xn, axis=-1)
        )
        assert float(np.mean(cos)) > min_cos

    def test_4bit_better_than_2bit(self):
        from olmlx.engine.kvarn import KVarNRotation, kvarn_dequantize, kvarn_quantize

        mx.random.seed(3)
        rot = KVarNRotation(head_dim=128, seed=0)
        x = mx.random.normal((1, 2, 16, 128))
        errs = {}
        for bits in (2, 4):
            p, s = kvarn_quantize(x, rot, bits=bits)
            errs[bits] = float(mx.mean((kvarn_dequantize(p, s, rot, bits) - x) ** 2))
        assert errs[4] < errs[2] / 4

    def test_outlier_channel_handled(self):
        """A massive outlier channel (typical of keys) is spread by the
        Hadamard rotation instead of crushing the other coordinates."""
        from olmlx.engine.kvarn import KVarNRotation, kvarn_dequantize, kvarn_quantize

        mx.random.seed(4)
        rot = KVarNRotation(head_dim=128, seed=0)
        base = mx.random.normal((1, 1, 16, 128))
        outlier = mx.zeros((128,)).at[5].add(200.0)
        x = base + outlier
        p, s = kvarn_quantize(x, rot, bits=4)
        out = kvarn_dequantize(p, s, rot, bits=4)
        rel = float(mx.sqrt(mx.sum((out - x) ** 2)) / mx.sqrt(mx.sum(x**2)))
        assert rel < 0.15

    @pytest.mark.parametrize("head_dim", [128, 256, 96])
    def test_tiled_rotation_matches_dense(self, head_dim):
        """The tile-wise kernels implement exactly x @ R.T / y @ R."""
        from olmlx.engine.kvarn import KVarNRotation, kvarn_dequantize, kvarn_quantize

        mx.random.seed(5)
        rot = KVarNRotation(head_dim=head_dim, seed=7)
        r = rot.dense_matrix()
        x = mx.random.normal((1, 2, 3, head_dim))
        p, s = kvarn_quantize(x, rot, bits=4)
        out = np.array(kvarn_dequantize(p, s, rot, bits=4))
        # The dequantized vector rotated forward must be the per-tile
        # reconstruction mu + scale * centered(codebook) — check via norms of
        # the rotated residual instead of re-deriving: R orthogonal ⇒ equal.
        xr = np.array(x) @ r.T
        outr = out @ r.T
        np.testing.assert_allclose(
            np.linalg.norm(outr - xr, axis=-1),
            np.linalg.norm(out - np.array(x), axis=-1),
            rtol=1e-4,
        )
        # Per-tile means of the reconstruction match the stored means.
        n_tiles = head_dim // rot.tile
        means = outr.reshape(1, 2, 3, n_tiles, rot.tile).mean(-1)
        np.testing.assert_allclose(means, np.array(s)[..., :n_tiles], atol=1e-4)

    @pytest.mark.parametrize("bits", [2, 4])
    def test_encode_recon_matches_dequantize(self, bits):
        from olmlx.engine.kvarn import (
            KVarNRotation,
            kvarn_dequantize,
            kvarn_encode,
            kvarn_quantize,
        )

        rot = KVarNRotation(head_dim=256, seed=0)
        x = mx.random.normal((1, 2, 5, 256)).astype(mx.float16)
        p, s, recon = kvarn_encode(x, rot, bits, mx.float16)
        p2, s2 = kvarn_quantize(x, rot, bits)
        np.testing.assert_array_equal(np.array(p), np.array(p2))
        assert recon.dtype == mx.float16
        np.testing.assert_allclose(
            np.array(recon.astype(mx.float32)),
            np.array(kvarn_dequantize(p, s, rot, bits, dtype=mx.float32)),
            atol=2e-2,
        )

    def test_zero_vector_roundtrips_to_zero(self):
        from olmlx.engine.kvarn import KVarNRotation, kvarn_dequantize, kvarn_quantize

        rot = KVarNRotation(head_dim=64, seed=0)
        x = mx.zeros((1, 1, 2, 64))
        p, s = kvarn_quantize(x, rot, bits=2)
        out = kvarn_dequantize(p, s, rot, bits=2)
        assert not bool(mx.any(mx.isnan(out)))
        np.testing.assert_allclose(np.array(out), 0.0, atol=1e-6)

    def test_bfloat16_input(self):
        from olmlx.engine.kvarn import KVarNRotation, kvarn_dequantize, kvarn_quantize

        rot = KVarNRotation(head_dim=128, seed=0)
        x = mx.random.normal((1, 2, 3, 128)).astype(mx.bfloat16)
        p, s = kvarn_quantize(x, rot, bits=4)
        out = kvarn_dequantize(p, s, rot, bits=4, dtype=mx.bfloat16)
        assert out.dtype == mx.bfloat16
        assert not bool(mx.any(mx.isnan(out.astype(mx.float32))))


# ---------------------------------------------------------------------------
# KVarNKVCache
# ---------------------------------------------------------------------------


class TestKVarNKVCache:
    def test_inherits_turboquant_cache(self):
        """Subclassing inherits every cross-thread Metal-stream fix
        (deepcopy, dequant shed, packed-buffer materialization)."""
        from olmlx.engine.turboquant_cache import TurboQuantKVCache

        assert isinstance(_make_cache(), TurboQuantKVCache)

    def test_update_and_fetch_shapes_asymmetric(self):
        c = _make_cache(key_bits=4, value_bits=2)
        k = mx.random.normal((1, 2, 5, 128)).astype(mx.float16)
        v = mx.random.normal((1, 2, 5, 128)).astype(mx.float16)
        kk, vv = c.update_and_fetch(k, v)
        assert kk.shape == (1, 2, 5, 128) and vv.shape == (1, 2, 5, 128)
        assert kk.dtype == mx.float16
        assert c.offset == 5
        ki, kn, vi, vn = c.state
        assert ki.shape == (1, 2, 5, 64)  # 4-bit keys
        assert vi.shape == (1, 2, 5, 32)  # 2-bit values
        assert kn.shape == (1, 2, 5, 2) and vn.shape == (1, 2, 5, 2)

    def test_matches_standalone_dequantize(self):
        from olmlx.engine.kvarn import kvarn_dequantize, kvarn_quantize

        c = _make_cache(key_bits=4, value_bits=2)
        k = mx.random.normal((1, 2, 3, 128))
        v = mx.random.normal((1, 2, 3, 128))
        kk, vv = c.update_and_fetch(k, v)
        p, s = kvarn_quantize(k, c.rotation_key, bits=4)
        np.testing.assert_allclose(
            np.array(kk),
            np.array(kvarn_dequantize(p, s, c.rotation_key, 4)),
            atol=1e-5,
        )
        p, s = kvarn_quantize(v, c.rotation_value, bits=2)
        np.testing.assert_allclose(
            np.array(vv),
            np.array(kvarn_dequantize(p, s, c.rotation_value, 2)),
            atol=1e-5,
        )

    def test_incremental_decode_and_grow(self):
        c = _make_cache()
        k = mx.random.normal((1, 2, 250, 128))
        c.update_and_fetch(k, k)
        for _ in range(10):  # crosses the 256 step boundary
            step = mx.random.normal((1, 2, 1, 128))
            kk, vv = c.update_and_fetch(step, step)
        assert c.offset == 260
        assert kk.shape == (1, 2, 260, 128)
        assert c._key_indices.shape[2] >= 260

    def test_trim(self):
        c = _make_cache()
        x = mx.random.normal((1, 2, 10, 128))
        c.update_and_fetch(x, x)
        assert c.is_trimmable()
        assert c.trim(4) == 4
        assert c.offset == 6
        kk, _ = c.update_and_fetch(x[..., :2, :], x[..., :2, :])
        assert kk.shape[2] == 8
        c.trim(100)
        assert c.empty()

    def test_release_and_rebuild_matches(self):
        c = _make_cache()
        x = mx.random.normal((1, 2, 6, 128))
        kk_before, vv_before = c.update_and_fetch(x, x)
        kk_before = np.array(kk_before)
        vv_before = np.array(vv_before)
        c.release_dequant_buffers()
        assert c._key_dequant is None
        step = mx.random.normal((1, 2, 1, 128))
        kk, vv = c.update_and_fetch(step, step)
        np.testing.assert_allclose(np.array(kk[..., :6, :]), kk_before, atol=1e-5)
        np.testing.assert_allclose(np.array(vv[..., :6, :]), vv_before, atol=1e-5)

    def test_deepcopy_shares_rotations_and_is_independent(self):
        c = _make_cache()
        x = mx.random.normal((1, 2, 4, 128)).astype(mx.float16)
        c.update_and_fetch(x, x)
        d = copy.deepcopy(c)
        assert d.rotation_key is c.rotation_key
        assert d._key_bits == 4 and d._value_bits == 2
        d.update_and_fetch(x, x)
        assert c.offset == 4 and d.offset == 8

    def test_state_setter_rejected(self):
        c = _make_cache()
        with pytest.raises(NotImplementedError):
            c.state = []

    def test_packed_state_materialized_cross_thread(self):
        """Generate on one worker thread, materialize there, then resume on
        another — the flat-path reuse pattern that crashed TurboQuant."""
        c = _make_cache()

        def gen():
            x = mx.random.normal((1, 2, 8, 128))
            kk, vv = c.update_and_fetch(x, x)
            mx.eval(kk, vv)
            c.ensure_state_materialized()

        assert "error" not in _run_in_thread(gen)
        c.release_dequant_buffers()

        def resume():
            step = mx.random.normal((1, 2, 1, 128))
            kk, vv = c.update_and_fetch(step, step)
            mx.eval(kk, vv)
            return kk.shape

        res = _run_in_thread(resume)
        assert "error" not in res, res.get("error")
        assert res["value"] == (1, 2, 9, 128)


# ---------------------------------------------------------------------------
# Factory + wiring
# ---------------------------------------------------------------------------


def _mock_model(n_layers=2, head_dim=128):
    model = MagicMock()
    model.layers = [MagicMock() for _ in range(n_layers)]
    model.args.head_dim = head_dim
    del model.make_cache
    return model


class TestMakeKvarnCache:
    def test_creates_kvarn_layers(self):
        from olmlx.engine.kvarn_cache import KVarNKVCache, make_kvarn_cache

        cache = make_kvarn_cache(_mock_model(3), key_bits=4, value_bits=2)
        assert len(cache) == 3
        assert all(isinstance(c, KVarNKVCache) for c in cache)
        assert cache[0]._key_bits == 4 and cache[0]._value_bits == 2
        # Distinct per-layer, per-K/V rotations
        assert not np.array_equal(
            cache[0].rotation_key.dense_matrix(),
            cache[0].rotation_value.dense_matrix(),
        )
        assert not np.array_equal(
            cache[0].rotation_key.dense_matrix(),
            cache[1].rotation_key.dense_matrix(),
        )

    def test_hybrid_preserves_non_kv_layers(self):
        from mlx_lm.models.cache import ArraysCache, KVCache

        from olmlx.engine.kvarn_cache import KVarNKVCache, make_kvarn_cache

        model = MagicMock()
        model.layers = [MagicMock() for _ in range(2)]
        model.args.head_dim = 128
        arrays = ArraysCache(size=2)
        model.make_cache.return_value = [arrays, KVCache()]
        cache = make_kvarn_cache(model, key_bits=4, value_bits=2)
        assert cache[0] is arrays
        assert isinstance(cache[1], KVarNKVCache)

    def test_mlx_vlm_plain_kvcache_is_quantized(self):
        """VLMs (Gemma 4 via mlx-vlm) build their layout from mlx-vlm's own
        cache classes, which don't subclass mlx-lm's — a plain mlx-vlm
        ``KVCache`` must still be quantized, or ``kv_cache_quant`` is a no-op
        on every VLM. Its windowed and model-specific subclasses (Qwen3.8's
        ``QSAKVCache`` carries sparse-indexer state) must be left alone."""
        from mlx_vlm.models.cache import KVCache as VlmKVCache
        from mlx_vlm.models.cache import RotatingKVCache as VlmRotating

        from olmlx.engine.kvarn_cache import KVarNKVCache, make_kvarn_cache

        class _ModelSpecificKVCache(VlmKVCache):
            pass

        rotating = VlmRotating(max_size=8)
        special = _ModelSpecificKVCache()
        model = MagicMock()
        model.layers = [MagicMock() for _ in range(3)]
        model.args.head_dim = 128
        model.make_cache.return_value = [rotating, VlmKVCache(), special]
        cache = make_kvarn_cache(model, key_bits=4, value_bits=2)
        assert cache[0] is rotating
        assert isinstance(cache[1], KVarNKVCache)
        assert cache[2] is special

    def test_unsupported_head_dim_falls_back(self):
        from mlx_lm.models.cache import KVCache

        from olmlx.engine.kvarn_cache import KVarNKVCache, make_kvarn_cache

        cache = make_kvarn_cache(_mock_model(1, head_dim=72), key_bits=4, value_bits=2)
        assert isinstance(cache[0], KVCache)
        assert not isinstance(cache[0], KVarNKVCache)

    def test_prompt_cache_for_lm_dispatch(self):
        from olmlx.engine.inference import _make_prompt_cache_for_lm
        from olmlx.engine.kvarn_cache import KVarNKVCache

        lm = MagicMock()
        lm.kv_cache_quant = "kvarn:k4v2"
        lm.is_vlm = False
        lm.model = _mock_model(2)
        cache = _make_prompt_cache_for_lm(lm)
        assert all(isinstance(c, KVarNKVCache) for c in cache)
        assert cache[0]._key_bits == 4 and cache[0]._value_bits == 2

    def test_lazy_state_flagged(self):
        from olmlx.engine.inference import _cache_list_contains_lazy_state

        assert _cache_list_contains_lazy_state([_make_cache()]) is True

    def test_not_disk_serializable(self):
        from olmlx.engine.cache_capabilities import _is_serializable_cache

        assert _is_serializable_cache([_make_cache()]) is False

    def test_bench_scenarios(self):
        from olmlx.bench.scenarios import SCENARIOS

        by_name = {s.name: s for s in SCENARIOS}
        assert by_name["kvarn-k4v2"].env_overrides == {
            "OLMLX_KV_CACHE_QUANT": "kvarn:k4v2"
        }
        assert by_name["kvarn-2"].env_overrides == {"OLMLX_KV_CACHE_QUANT": "kvarn:2"}


class TestKvarnBudgetEstimate:
    """The estimate gates live generation: KVarN (like TurboQuant) holds the
    packed state plus a full-precision dequant side buffer until the cache is
    stored, and layers the codec can't take stay plain fp16 KVCache."""

    def _model(self):
        args = MagicMock(spec=[])
        args.num_hidden_layers = 4
        args.num_attention_heads = 8
        args.num_key_value_heads = 2
        args.head_dim = 128
        args.hidden_size = 1024
        model = MagicMock(spec=["args"])
        model.args = args
        return model

    def test_k4v2_ratio(self):
        from olmlx.engine.kv_budget import estimate_kv_cache_bytes

        model = self._model()
        fp16 = estimate_kv_cache_bytes(model, 10000)
        kvarn = estimate_kv_cache_bytes(model, 10000, kv_cache_quant="kvarn:k4v2")
        # fp16 K+V: 2*256 bytes. kvarn: K 64+8+256, V 32+8+256 → 624 bytes.
        assert fp16 / kvarn == pytest.approx(512 / 624, rel=0.01)

    def test_symmetric_2(self):
        from olmlx.engine.kv_budget import estimate_kv_cache_bytes

        model = self._model()
        fp16 = estimate_kv_cache_bytes(model, 10000)
        kvarn = estimate_kv_cache_bytes(model, 10000, kv_cache_quant="kvarn:2")
        assert fp16 / kvarn == pytest.approx(512 / 592, rel=0.01)

    @staticmethod
    def _layered_model(head_dims):
        from types import SimpleNamespace

        args = SimpleNamespace(
            num_hidden_layers=len(head_dims),
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=128,
            hidden_size=1024,
        )
        layers = [
            SimpleNamespace(self_attn=SimpleNamespace(n_kv_heads=2, head_dim=hd))
            for hd in head_dims
        ]
        return SimpleNamespace(args=args, model=SimpleNamespace(layers=layers))

    @pytest.mark.parametrize(
        "spec,quant_ratio,bad_head_dim",
        [
            ("kvarn:k4v2", 624 / 512, 72),  # 72: no power-of-two tile ≥ 16
            ("turboquant:2", 292 / 256, 66),  # 66: not divisible by 4
        ],
    )
    def test_incompatible_layer_charged_at_fp16(self, spec, quant_ratio, bad_head_dim):
        from olmlx.engine.kv_budget import MEMORY_SAFETY_FACTOR, estimate_kv_cache_bytes

        model = self._layered_model([128, bad_head_dim])
        n = 1000
        got = estimate_kv_cache_bytes(model, n, kv_cache_quant=spec)
        quantized = 2 * 2 * 128 * n * 2 * quant_ratio
        plain = 2 * 2 * bad_head_dim * n * 2
        assert got == pytest.approx(
            (quantized + plain) * MEMORY_SAFETY_FACTOR, rel=1e-6
        )


class TestPackedBufferMaterialization:
    """The packed buffers never feed the returned K/V (those come from the
    side buffer), so without help they grow a lazy slice_update chain every
    decode token. Each pending write pins a buffer: a 36-layer model hit
    ``metal::malloc Resource limit (499000)`` at ~3.4k generated tokens
    (TurboQuant on main too). The cache must fold the chain into the token
    graph at each ``step`` boundary."""

    @pytest.mark.parametrize("make", ["kvarn", "turboquant"])
    def test_packed_buffers_materialized_at_step_boundary(self, make):
        from olmlx.engine.turboquant import TurboQuantRotation
        from olmlx.engine.turboquant_cache import TurboQuantKVCache

        if make == "kvarn":
            c = _make_cache(head_dim=64)
        else:
            c = TurboQuantKVCache(
                bits=4,
                rotation_key=TurboQuantRotation(head_dim=64, seed=0),
                rotation_value=TurboQuantRotation(head_dim=64, seed=1),
            )
        step = c.step

        def decode_to_boundary():
            x = mx.random.normal((1, 1, 10, 64))
            mx.eval(c.update_and_fetch(x, x))
            for _ in range(step - 10):  # last write lands exactly on `step`
                t = mx.random.normal((1, 1, 1, 64))
                # Mirrors mlx-lm: only the returned K/V reach the eval.
                mx.eval(c.update_and_fetch(t, t))
            assert c.offset == step

        assert "error" not in _run_in_thread(decode_to_boundary)

        # No ensure_state_materialized: the boundary step must already have
        # materialized the packed chain. Reading it from another thread
        # raises "There is no Stream" on Metal if it is still lazy.
        def read_packed():
            mx.eval(c._key_indices, c._key_norms, c._value_indices, c._value_norms)

        res = _run_in_thread(read_packed)
        assert "error" not in res, res.get("error")
