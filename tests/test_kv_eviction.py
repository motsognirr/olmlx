"""Tests for StreamingLLM sink+window KV eviction (#505).

The feature is an opt-in KV layout: for pure full-attention models, each layer's
cache becomes ``RotatingKVCache(max_size=sink+window, keep=sink)`` — mlx-lm's
own windowed cache, which keeps the first ``sink`` (attention-sink) tokens plus
a rotating recent ``window`` and drops the middle. Correctness of the masking /
rope is mlx-lm's; these tests cover olmlx's config plumbing, the cache-selection
logic (only pure full-attention models are converted), and the eviction policy
at the cache level (no-eviction exactness vs a plain KVCache + size bounding).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_lm.models.cache import KVCache, RotatingKVCache


class TestKvEvictionConfigValidation:
    def test_valid_values(self):
        from olmlx.config import validate_kv_eviction_format

        assert validate_kv_eviction_format(None) is None
        assert validate_kv_eviction_format("4:512") == "4:512"
        assert validate_kv_eviction_format("0:64") == "0:64"

    @pytest.mark.parametrize(
        "bad", ["512", "4:0", "-1:64", "a:b", "4:512:9", "4:", ":512"]
    )
    def test_invalid_values_rejected(self, bad):
        from olmlx.config import validate_kv_eviction_format

        with pytest.raises(ValueError, match="kv_eviction"):
            validate_kv_eviction_format(bad)

    def test_settings_env(self, monkeypatch):
        from olmlx.config import Settings

        monkeypatch.setenv("OLMLX_KV_EVICTION", "8:256")
        assert Settings().kv_eviction == "8:256"

    def test_settings_rejects_bad(self):
        from pydantic import ValidationError

        from olmlx.config import Settings

        with pytest.raises(ValidationError):
            Settings(kv_eviction="nope", _env_file=None)


class TestResolvedKvEviction:
    def test_per_model_overrides_global(self, monkeypatch):
        from olmlx.engine.registry import ModelConfig

        monkeypatch.setattr("olmlx.config.settings.kv_eviction", "4:512")
        mc = ModelConfig(hf_path="org/m", kv_eviction="0:64")
        assert mc.resolved_kv_eviction() == "0:64"

    def test_falls_back_to_global(self, monkeypatch):
        from olmlx.engine.registry import ModelConfig

        monkeypatch.setattr("olmlx.config.settings.kv_eviction", "4:512")
        assert ModelConfig(hf_path="org/m").resolved_kv_eviction() == "4:512"

    def test_none_when_unset(self, monkeypatch):
        from olmlx.engine.registry import ModelConfig

        monkeypatch.setattr("olmlx.config.settings.kv_eviction", None)
        assert ModelConfig(hf_path="org/m").resolved_kv_eviction() is None

    def test_from_entry_validates(self):
        from olmlx.engine.registry import ModelConfig

        mc = ModelConfig.from_entry({"hf_path": "org/m", "kv_eviction": "4:512"})
        assert mc.kv_eviction == "4:512"
        with pytest.raises(ValueError, match="kv_eviction"):
            ModelConfig.from_entry({"hf_path": "org/m", "kv_eviction": "bad"})


class TestParse:
    def test_parse(self):
        from olmlx.engine.inference import _parse_kv_eviction

        assert _parse_kv_eviction("4:512") == (4, 512)


class TestEvictionCacheBuilder:
    def test_pure_full_attention_converted(self, monkeypatch):
        from olmlx.engine import inference

        monkeypatch.setattr(
            inference, "make_prompt_cache", lambda model: [KVCache(), KVCache()]
        )
        cache = inference._make_eviction_prompt_cache(object(), sink=4, window=64)
        assert len(cache) == 2
        assert all(isinstance(c, RotatingKVCache) for c in cache)
        assert all(c.max_size == 68 and c.keep == 4 for c in cache)

    def test_hybrid_model_left_untouched(self, monkeypatch):
        """A model whose default cache mixes RotatingKVCache (SWA) must not be
        converted — one per-forward mask is built from cache[0]."""
        from olmlx.engine import inference

        default = [KVCache(), RotatingKVCache(max_size=128, keep=0)]
        monkeypatch.setattr(inference, "make_prompt_cache", lambda model: default)
        cache = inference._make_eviction_prompt_cache(object(), sink=4, window=64)
        assert cache is default  # unchanged

    def test_selected_when_kv_eviction_set(self, monkeypatch):
        """_make_prompt_cache_for_lm routes through the eviction builder."""
        from types import SimpleNamespace

        from olmlx.engine import inference

        monkeypatch.setattr(inference, "make_prompt_cache", lambda model: [KVCache()])
        monkeypatch.setattr(inference, "_get_model_for_cache", lambda m, v: m)
        lm = SimpleNamespace(
            model=object(), is_vlm=False, kv_cache_quant=None, kv_eviction="4:64"
        )
        cache = inference._make_prompt_cache_for_lm(lm)
        assert isinstance(cache[0], RotatingKVCache)

    def test_quant_takes_precedence_over_eviction(self, monkeypatch):
        """When both are set on the LoadedModel, the quant branch wins."""
        from types import SimpleNamespace

        from olmlx.engine import inference

        sentinel = ["quant-cache"]
        monkeypatch.setattr(
            inference, "_make_turboquant_prompt_cache", lambda *a, **k: sentinel
        )
        lm = SimpleNamespace(
            model=object(),
            is_vlm=False,
            kv_cache_quant="turboquant:4",
            kv_eviction="4:64",
        )
        assert inference._make_prompt_cache_for_lm(lm) is sentinel


class TestEvictionPolicyAtCacheLevel:
    """The sink+window behavior is mlx-lm's RotatingKVCache; pin the two
    properties olmlx relies on."""

    @staticmethod
    def _kv(n, d=4, h=2):
        k = mx.random.normal((1, h, n, d)).astype(mx.float16)
        v = mx.random.normal((1, h, n, d)).astype(mx.float16)
        return k, v

    def test_no_eviction_matches_plain_cache(self):
        """max_size >= total tokens → identical retained K/V to a plain KVCache,
        so model output is unchanged when nothing is evicted."""
        mx.random.seed(0)
        plain, rot = KVCache(), RotatingKVCache(max_size=10_000, keep=4)
        # prefill chunk + a few decode steps
        for n in (6, 1, 1, 1):
            k, v = self._kv(n)
            pk, pv = plain.update_and_fetch(k, v)
            rk, rv = rot.update_and_fetch(k, v)
            assert mx.array_equal(pk, rk)
            assert mx.array_equal(pv, rv)
        assert plain.offset == rot.offset == 9

    def test_eviction_bounds_kv_count(self):
        mx.random.seed(1)
        sink, window = 4, 8
        rot = RotatingKVCache(max_size=sink + window, keep=sink)
        # Prefill past capacity, then decode many steps.
        k, v = self._kv(20)
        rot.update_and_fetch(k, v)
        for _ in range(30):
            k, v = self._kv(1)
            rk, _ = rot.update_and_fetch(k, v)
            assert rk.shape[2] <= sink + window
        # offset tracks the true token count even though KV is bounded.
        assert rot.offset == 50
        assert rot.size() == sink + window
