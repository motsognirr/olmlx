"""Regression coverage for olmlx.engine.model_manager.

Focuses on pure logic with heavily mocked LoadedModel/mlx objects:
- cache-classification discriminators (persistence / trim / checkpoint),
- LoadedModel ref counting + property unwrapping,
- LRU eviction ordering and active_refs protection,
- keep-alive / expiry math,
- error/edge paths in config sanitation, chat-template loading, draft
  attention version resolution, vocab matching, flash-dir detection,
  and prompt-cache invalidation.

These complement (not duplicate) tests/test_model_manager.py.
"""

import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmlx.engine.model_manager import (
    LoadedModel,
    ModelManager,
    _cache_supports_checkpoint_persistence,
    _cache_supports_persistence,
    _cache_supports_trim,
    _ensure_tokenizer_eos_in_stops,
    _is_serializable_cache,
    _kv_quant_blocks_snapshot,
    _layer_layout_is_mixed_excluded,
    _resolve_attention_causal,
    _sanitize_model_config_in_place,
)


def _fake_layer(class_name: str):
    """Return an object whose ``type(obj).__name__`` is ``class_name``."""

    cls = type(class_name, (), {})
    return cls()


# --------------------------------------------------------------------------
# Cache-classification discriminators (pure, allowlist-based).
# --------------------------------------------------------------------------


class TestCacheClassification:
    def test_trim_allowlist_accepts_all_known(self):
        cache = [_fake_layer("KVCache"), _fake_layer("QuantizedKVCache")]
        assert _cache_supports_trim(cache) is True

    def test_trim_rejects_rotating(self):
        cache = [_fake_layer("KVCache"), _fake_layer("RotatingKVCache")]
        assert _cache_supports_trim(cache) is False

    def test_persistence_empty_list_is_false(self):
        # Empty == no evidence of safety; a false-positive here crashes the
        # next request, so the helper must refuse to claim safety.
        assert _cache_supports_persistence([]) is False

    def test_persistence_rejects_arrays_cache(self):
        # ArraysCache (gated-delta SSM state, issue #284) is the motivating
        # exclusion from the flat persistence allowlist.
        cache = [_fake_layer("KVCache"), _fake_layer("ArraysCache")]
        assert _cache_supports_persistence(cache) is False

    def test_persistence_no_mro_walk_subclass_rejected(self):
        # Exact class-name match (no MRO walk): a subclass of an allowlisted
        # class must NOT pass — that is the documented safety guarantee.
        class KVCache:  # base in allowlist by name
            pass

        class BadSSMCache(KVCache):
            pass

        # type(obj).__name__ == "BadSSMCache", not in allowlist.
        assert _cache_supports_persistence([BadSSMCache()]) is False

    def test_checkpoint_accepts_arrays_cache(self):
        # ArraysCache IS in the checkpoint allowlist (snapshot helper evals
        # the lazy graph), unlike the flat path.
        cache = [_fake_layer("ArraysCache"), _fake_layer("KVCache")]
        assert _cache_supports_checkpoint_persistence(cache) is True

    def test_checkpoint_empty_is_false(self):
        assert _cache_supports_checkpoint_persistence([]) is False

    def test_checkpoint_rejects_unknown_layer(self):
        cache = [_fake_layer("KVCache"), _fake_layer("MysterySSMCache")]
        assert _cache_supports_checkpoint_persistence(cache) is False

    def test_mixed_excluded_empty_set_never_fires(self):
        # _EXCLUDED_MIXED_LAYER_PAIRS is currently empty (#396 lifted), so no
        # layout is mixed-excluded.
        cache = [_fake_layer("RotatingKVCache"), _fake_layer("ArraysCache")]
        assert _layer_layout_is_mixed_excluded(cache) is False

    def test_mixed_excluded_empty_cache_false(self):
        assert _layer_layout_is_mixed_excluded([]) is False


class TestKvQuantSnapshotGate:
    def test_none_does_not_block(self):
        assert _kv_quant_blocks_snapshot(None) is False

    def test_turboquant_does_not_block(self):
        # _KV_QUANT_PREFIXES_BLOCKING_SNAPSHOT is empty: turboquant handles
        # its own deepcopy, so it must not be reported as blocking.
        assert _kv_quant_blocks_snapshot("turboquant:4") is False

    def test_spectral_does_not_block(self):
        assert _kv_quant_blocks_snapshot("spectral:2") is False


class TestIsSerializableCache:
    def test_plain_kvcache_is_serializable(self):
        # No TurboQuant/Spectral wrappers -> safetensors-serializable.
        assert _is_serializable_cache([_fake_layer("KVCache")]) is True

    def test_turboquant_cache_not_serializable(self):
        from olmlx.engine.turboquant_cache import TurboQuantKVCache

        tq = TurboQuantKVCache.__new__(TurboQuantKVCache)
        assert _is_serializable_cache([_fake_layer("KVCache"), tq]) is False


# --------------------------------------------------------------------------
# LoadedModel: ref-counting, properties.
# --------------------------------------------------------------------------


class TestLoadedModelRefs:
    def _lm(self, **kw):
        kw.setdefault("model", MagicMock())
        kw.setdefault("tokenizer", MagicMock())
        return LoadedModel(
            name="m:latest",
            hf_path="org/model",
            **kw,
        )

    def test_acquire_release_ref_round_trips(self):
        lm = self._lm()
        assert lm.active_refs == 0
        lm.acquire_ref()
        lm.acquire_ref()
        assert lm.active_refs == 2
        lm.release_ref()
        assert lm.active_refs == 1

    def test_is_speculative_reflects_decoder(self):
        lm = self._lm()
        assert lm.is_speculative is False
        lm.speculative_decoder = MagicMock()
        assert lm.is_speculative is True

    def test_text_tokenizer_unwraps_vlm_processor(self):
        inner = MagicMock(name="inner_tok")
        processor = MagicMock()
        processor.tokenizer = inner
        lm = self._lm(is_vlm=True, tokenizer=processor)
        assert lm.text_tokenizer is inner

    def test_text_tokenizer_returns_tokenizer_for_text_model(self):
        tok = MagicMock()
        # Even if a text tokenizer happens to expose .tokenizer, the
        # non-VLM path must return the tokenizer itself.
        tok.tokenizer = MagicMock()
        lm = self._lm(is_vlm=False, tokenizer=tok)
        assert lm.text_tokenizer is tok


# --------------------------------------------------------------------------
# _resolve_attention_causal: draft-checkpoint version math.
# --------------------------------------------------------------------------


class TestResolveAttentionCausal:
    def test_missing_version_defaults_bidirectional(self):
        # Absent key -> default 2 -> bidirectional (causal=False).
        assert _resolve_attention_causal({}) is False

    def test_version_2_is_bidirectional(self):
        assert _resolve_attention_causal({"dflash_attention_version": 2}) is False

    def test_version_1_is_causal_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            assert _resolve_attention_causal({"dflash_attention_version": 1}) is True
        assert "causal attention" in caplog.text

    def test_string_version_parsed(self):
        # Hand-edited config storing "2" must parse as int 2.
        assert _resolve_attention_causal({"dflash_attention_version": "2"}) is False

    def test_fractional_treated_as_v1(self):
        # 1.5 -> int(float()) == 1 -> causal (fractional values are misconfigs).
        assert _resolve_attention_causal({"dflash_attention_version": 1.5}) is True

    def test_unparseable_version_defaults_bidirectional(self):
        assert (
            _resolve_attention_causal({"dflash_attention_version": "nonsense"}) is False
        )


# --------------------------------------------------------------------------
# _sanitize_model_config_in_place: layer_types truncation.
# --------------------------------------------------------------------------


class TestSanitizeModelConfig:
    def test_truncates_excess_layer_types(self, tmp_path):
        cfg = {
            "num_hidden_layers": 2,
            "layer_types": ["a", "b", "c", "d"],
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        _sanitize_model_config_in_place(tmp_path)
        out = json.loads((tmp_path / "config.json").read_text())
        assert out["layer_types"] == ["a", "b"]

    def test_leaves_matching_layer_types_untouched(self, tmp_path):
        cfg = {"num_hidden_layers": 2, "layer_types": ["a", "b"]}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        _sanitize_model_config_in_place(tmp_path)
        out = json.loads((tmp_path / "config.json").read_text())
        assert out["layer_types"] == ["a", "b"]

    def test_missing_config_is_noop(self, tmp_path):
        # No config.json -> returns without error.
        _sanitize_model_config_in_place(tmp_path)
        assert not (tmp_path / "config.json").exists()

    def test_invalid_json_is_noop(self, tmp_path):
        (tmp_path / "config.json").write_text("{not valid json")
        # Must swallow JSONDecodeError and leave the file untouched.
        _sanitize_model_config_in_place(tmp_path)
        assert (tmp_path / "config.json").read_text() == "{not valid json"


# --------------------------------------------------------------------------
# _ensure_tokenizer_eos_in_stops: #308 workaround branch coverage.
# --------------------------------------------------------------------------


class TestEnsureEosInStops:
    def test_adds_inner_eos_to_stop_set(self):
        tok = MagicMock()
        tok.add_eos_token = MagicMock()  # callable marker
        tok.eos_token_ids = {7}
        tok._tokenizer = MagicMock()
        tok._tokenizer.eos_token_id = 42
        _ensure_tokenizer_eos_in_stops(tok)
        assert 42 in tok.eos_token_ids

    def test_non_set_stops_skips(self):
        tok = MagicMock()
        tok.add_eos_token = MagicMock()
        tok.eos_token_ids = [7]  # not a set -> skip
        tok._tokenizer = MagicMock()
        tok._tokenizer.eos_token_id = 42
        # Must not raise; list left unmodified.
        _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids == [7]

    def test_no_add_eos_token_marker_is_noop(self):
        tok = object()  # no add_eos_token attribute
        _ensure_tokenizer_eos_in_stops(tok)  # must not raise

    def test_list_inner_eos_filters_ints(self):
        tok = MagicMock()
        tok.add_eos_token = MagicMock()
        tok.eos_token_ids = set()
        tok._tokenizer = MagicMock()
        tok._tokenizer.eos_token_id = [1, "x", 2]
        _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids == {1, 2}


# --------------------------------------------------------------------------
# Flash-dir detection helpers.
# --------------------------------------------------------------------------


class TestFlashDirDetection:
    def test_flash_dir_none_when_no_layout(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        d = mock_store.local_path("org/m")
        d.joinpath("flash").mkdir(parents=True)
        # Directory exists but no flash_layout.json -> None.
        assert manager._flash_dir("org/m") is None

    def test_flash_dir_returns_path_when_layout_present(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        flash = mock_store.local_path("org/m") / "flash"
        flash.mkdir(parents=True)
        (flash / "flash_layout.json").write_text("{}")
        assert manager._flash_dir("org/m") == flash

    def test_flash_dir_none_without_store(self, registry):
        manager = ModelManager(registry, store=None)
        assert manager._flash_dir("org/m") is None

    def test_flash_moe_dir_returns_path_when_layout_present(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        moe = mock_store.local_path("org/m") / "flash_moe"
        moe.mkdir(parents=True)
        (moe / "flash_moe_layout.json").write_text("{}")
        assert manager._flash_moe_dir("org/m") == moe

    def test_flash_moe_dir_none_without_store(self, registry):
        manager = ModelManager(registry, store=None)
        assert manager._flash_moe_dir("org/m") is None


# --------------------------------------------------------------------------
# _check_vocab_match.
# --------------------------------------------------------------------------


class TestCheckVocabMatch:
    def _model(self, vocab):
        m = MagicMock()
        m.args = MagicMock()
        m.args.vocab_size = vocab
        return m

    def test_matching_vocab_ok(self):
        ModelManager._check_vocab_match(self._model(1000), self._model(1000))

    def test_mismatched_vocab_raises(self):
        with pytest.raises(ValueError, match="vocab_size"):
            ModelManager._check_vocab_match(self._model(1000), self._model(2000))

    def test_missing_vocab_warns_and_returns(self, caplog):
        # target.args has no vocab_size attribute path -> None -> warn+return.
        target = MagicMock()
        target.args = None
        draft = self._model(1000)
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            ModelManager._check_vocab_match(target, draft)
        assert "Could not verify vocab" in caplog.text


# --------------------------------------------------------------------------
# invalidate_prompt_cache.
# --------------------------------------------------------------------------


class TestInvalidatePromptCache:
    def test_removes_entry_for_loaded_model(self, mock_manager):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.prompt_cache_store = MagicMock()
        mock_manager.invalidate_prompt_cache("qwen3", "cache-123")
        lm.prompt_cache_store.remove.assert_called_once_with("cache-123")

    def test_noop_for_unloaded_model(self, mock_manager):
        # Must not raise when the model isn't loaded.
        mock_manager.invalidate_prompt_cache("not-loaded", "cache-123")


# --------------------------------------------------------------------------
# _load_chat_template: file-based loading branches.
# --------------------------------------------------------------------------


class TestLoadChatTemplate:
    def test_keeps_existing_template(self, tmp_path):
        tok = MagicMock()
        tok.chat_template = "existing"
        ModelManager._load_chat_template(tok, str(tmp_path))
        assert tok.chat_template == "existing"

    def test_loads_from_jinja(self, tmp_path):
        tok = MagicMock()
        tok.chat_template = None
        (tmp_path / "chat_template.jinja").write_text("JINJA-TEMPLATE")
        ModelManager._load_chat_template(tok, str(tmp_path))
        assert tok.chat_template == "JINJA-TEMPLATE"

    def test_loads_from_json(self, tmp_path):
        tok = MagicMock()
        tok.chat_template = None
        (tmp_path / "chat_template.json").write_text(
            json.dumps({"chat_template": "JSON-TEMPLATE"})
        )
        ModelManager._load_chat_template(tok, str(tmp_path))
        assert tok.chat_template == "JSON-TEMPLATE"

    def test_malformed_json_leaves_template_none(self, tmp_path):
        tok = MagicMock()
        tok.chat_template = None
        (tmp_path / "chat_template.json").write_text("{not json")
        # JSONDecodeError is swallowed; no hf_path so no hub fallback.
        ModelManager._load_chat_template(tok, str(tmp_path))
        assert tok.chat_template is None


# --------------------------------------------------------------------------
# LRU eviction ordering + active_refs protection.
# --------------------------------------------------------------------------


class TestLruEvictionOrdering:
    def _add(self, manager, name, loaded_at, active_refs=0):
        lm = LoadedModel(
            name=name,
            hf_path=f"org/{name}",
            model=MagicMock(),
            tokenizer=MagicMock(),
            loaded_at=loaded_at,
            active_refs=active_refs,
        )
        manager._loaded[name] = lm
        return lm

    def test_pop_lru_evicts_oldest_first(self, registry, mock_store, monkeypatch):
        # limit==loaded==3: the loop pops while len >= limit, so it removes
        # exactly one (the LRU/oldest) to drop below the limit.
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 3)
        manager = ModelManager(registry, mock_store)
        now = time.time()
        self._add(manager, "old", now - 100)
        self._add(manager, "mid", now - 50)
        self._add(manager, "new", now - 1)
        evictees = manager._pop_lru_evictees()
        assert [e.name for e in evictees] == ["old"]
        assert "old" not in manager._loaded
        assert set(manager._loaded) == {"mid", "new"}

    def test_pop_lru_skips_active_and_evicts_oldest_idle(
        self, registry, mock_store, monkeypatch
    ):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 3)
        manager = ModelManager(registry, mock_store)
        now = time.time()
        # Oldest model is pinned (active) -> must be protected; the next
        # oldest idle model is evicted instead.
        self._add(manager, "old_active", now - 100, active_refs=1)
        self._add(manager, "mid_idle", now - 50)
        self._add(manager, "new_idle", now - 1)
        evictees = manager._pop_lru_evictees()
        assert [e.name for e in evictees] == ["mid_idle"]
        assert "old_active" in manager._loaded

    def test_pop_lru_raises_when_all_active(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)
        self._add(manager, "a", time.time(), active_refs=1)
        self._add(manager, "b", time.time(), active_refs=1)
        with pytest.raises(RuntimeError, match="in use"):
            manager._pop_lru_evictees()

    def test_pop_one_idle_lru_picks_oldest(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        now = time.time()
        self._add(manager, "old", now - 100)
        self._add(manager, "new", now - 1)
        popped = manager._pop_one_idle_lru()
        assert popped.name == "old"
        assert "old" not in manager._loaded

    def test_pop_one_idle_lru_none_when_all_active(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        self._add(manager, "a", time.time(), active_refs=2)
        assert manager._pop_one_idle_lru() is None
        assert "a" in manager._loaded


# --------------------------------------------------------------------------
# keep-alive / expiry math.
# --------------------------------------------------------------------------


class TestExpiryMath:
    def _add(self, manager, name, expires_at, active_refs=0):
        lm = LoadedModel(
            name=name,
            hf_path=f"org/{name}",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=expires_at,
            active_refs=active_refs,
        )
        manager._loaded[name] = lm
        return lm

    def test_resolve_keep_alive_uses_default_when_none(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        with patch("olmlx.engine.model_manager.settings.default_keep_alive", "5m"):
            assert manager._resolve_keep_alive(None) == 300.0

    def test_resolve_keep_alive_explicit_overrides_default(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        assert manager._resolve_keep_alive("30s") == 30.0

    async def test_expire_stale_removes_only_expired_idle(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        manager._close_loaded_model = MagicMock()
        now = time.time()
        self._add(manager, "expired", now - 10)  # past -> expire
        self._add(manager, "fresh", now + 1000)  # future -> keep
        self._add(manager, "never", None)  # never-expire -> keep
        with patch.object(manager, "_flush_metal", new_callable=AsyncMock):
            await manager._expire_stale()
        assert "expired" not in manager._loaded
        assert "fresh" in manager._loaded
        assert "never" in manager._loaded

    async def test_expire_stale_skips_active_even_if_expired(
        self, registry, mock_store
    ):
        manager = ModelManager(registry, mock_store)
        manager._close_loaded_model = MagicMock()
        now = time.time()
        # Expired in the keep-alive sense but serving a request -> protected.
        self._add(manager, "busy", now - 10, active_refs=1)
        with patch.object(manager, "_flush_metal", new_callable=AsyncMock):
            await manager._expire_stale()
        assert "busy" in manager._loaded
        manager._close_loaded_model.assert_not_called()

    async def test_expire_stale_no_flush_when_nothing_expired(
        self, registry, mock_store
    ):
        manager = ModelManager(registry, mock_store)
        manager._close_loaded_model = MagicMock()
        self._add(manager, "fresh", time.time() + 1000)
        with patch.object(
            manager, "_flush_metal", new_callable=AsyncMock
        ) as mock_flush:
            await manager._expire_stale()
        # Nothing popped -> flush must be skipped.
        mock_flush.assert_not_awaited()
