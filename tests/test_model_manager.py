"""Tests for olmlx.engine.model_manager."""

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmlx.config import FlashMoeConfig
from olmlx.engine.model_manager import (
    LoadedModel,
    ModelLoadTimeoutError,
    ModelManager,
    SpectralCalibrationMissingError,
    _ensure_tokenizer_eos_in_stops,
    parse_keep_alive,
)
from olmlx.engine.registry import ModelConfig, SpeculativeConfig
from olmlx.engine.template_caps import TemplateCaps


class TestParseKeepAlive:
    def test_seconds(self):
        assert parse_keep_alive("30s") == 30.0

    def test_minutes(self):
        assert parse_keep_alive("5m") == 300.0

    def test_hours(self):
        assert parse_keep_alive("2h") == 7200.0

    def test_zero(self):
        assert parse_keep_alive("0") == 0.0

    def test_negative_one(self):
        assert parse_keep_alive("-1") is None

    def test_integer(self):
        assert parse_keep_alive(60) == 60.0

    def test_negative_integer(self):
        assert parse_keep_alive(-1) is None

    def test_float(self):
        assert parse_keep_alive(30.5) == 30.5

    @pytest.mark.parametrize("value", ["invalid", "1d", "abc123", ""])
    def test_invalid_format_warns_and_defaults(self, value, caplog):
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            assert parse_keep_alive(value) == 300.0  # default
        assert "Invalid keep_alive format" in caplog.text

    def test_zero_integer(self):
        assert parse_keep_alive(0) == 0.0

    def test_bare_integer_string(self):
        """Bare integer string '1800' should be treated as seconds."""
        assert parse_keep_alive("1800") == 1800.0

    def test_bare_integer_string_zero(self):
        assert parse_keep_alive("0") == 0.0


class TestLoadedModel:
    def test_defaults(self):
        lm = LoadedModel(
            name="test:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        assert lm.is_vlm is False
        assert lm.size_bytes == 0
        assert lm.expires_at is None
        assert isinstance(lm.template_caps, TemplateCaps)
        assert lm.loaded_at > 0


class TestCloseLoadedModelGrammarDrop:
    """The xgrammar compile cache is keyed on ``id(tokenizer)``; entries
    must be dropped before the tokenizer can be GC'd or a future tokenizer
    landing on the recycled address would be served a stale grammar
    (issue #464). ``_close_loaded_model`` therefore must reach
    ``drop_for_tokenizer`` no matter which earlier close step fails.
    """

    @staticmethod
    def _make_lm(**overrides):
        from types import SimpleNamespace

        defaults = dict(
            name="test:latest",
            hf_path="test/model",
            # SimpleNamespace: no ``prefetcher`` attribute, so the flash
            # close branch is skipped.
            model=SimpleNamespace(),
            tokenizer=MagicMock(),
            prompt_cache_store=MagicMock(),
        )
        defaults.update(overrides)
        return LoadedModel(**defaults)

    def test_grammar_drop_runs_when_vlm_cache_clear_raises(self, monkeypatch):
        """A failing ``vlm_prompt_cache_store.clear()`` must not skip the
        grammar-cache drop — pre-#464 this step was the only unguarded one
        in ``_close_loaded_model``, so its raise skipped every later step
        and escaped as a bare exception instead of the documented
        ExceptionGroup.
        """
        drop_calls: list[Any] = []
        monkeypatch.setattr(
            "olmlx.engine.grammar.drop_for_tokenizer", drop_calls.append
        )
        vlm_store = MagicMock()
        vlm_store.clear.side_effect = RuntimeError("vlm clear boom")
        lm = self._make_lm(vlm_prompt_cache_store=vlm_store)

        with pytest.raises(ExceptionGroup) as excinfo:
            ModelManager._close_loaded_model(lm)

        # The failure is folded into the documented ExceptionGroup contract
        # (unload()'s ``except ExceptionGroup`` absorbs it) ...
        assert any(
            isinstance(e, RuntimeError) and "vlm clear boom" in str(e)
            for e in excinfo.value.exceptions
        )
        # ... and the grammar drop still ran, with the tokenizer the cache
        # was keyed on.
        assert drop_calls == [lm.text_tokenizer]

    def test_grammar_drop_runs_when_every_other_close_step_raises(self, monkeypatch):
        """Defense in depth: even if all guarded steps fail, the drop runs."""
        drop_calls: list[Any] = []
        monkeypatch.setattr(
            "olmlx.engine.grammar.drop_for_tokenizer", drop_calls.append
        )
        weight_store = MagicMock()
        weight_store.close.side_effect = RuntimeError("ws boom")
        spec_decoder = MagicMock()
        spec_decoder.close.side_effect = RuntimeError("spec boom")
        prompt_store = MagicMock()
        prompt_store.clear.side_effect = RuntimeError("pc boom")
        vlm_store = MagicMock()
        vlm_store.clear.side_effect = RuntimeError("vlm boom")
        lm = self._make_lm(
            weight_store=weight_store,
            speculative_decoder=spec_decoder,
            prompt_cache_store=prompt_store,
            vlm_prompt_cache_store=vlm_store,
        )

        with pytest.raises(ExceptionGroup) as excinfo:
            ModelManager._close_loaded_model(lm)

        assert len(excinfo.value.exceptions) == 4
        assert drop_calls == [lm.text_tokenizer]


class TestModelManager:
    def test_init(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        assert manager._loaded == {}
        assert manager.store is mock_store

    def test_get_loaded_empty(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        assert manager.get_loaded() == []

    def test_get_loaded(self, mock_manager):
        loaded = mock_manager.get_loaded()
        assert len(loaded) == 1
        assert loaded[0].name == "qwen3:latest"

    @pytest.mark.asyncio
    async def test_unload(self, mock_manager):
        await mock_manager.unload("qwen3")
        assert mock_manager.get_loaded() == []

    @pytest.mark.asyncio
    async def test_unload_not_loaded(self, mock_manager):
        assert await mock_manager.unload("nonexistent") is False

    @pytest.mark.asyncio
    async def test_unload_active_refs_raises(self, mock_manager):
        from olmlx.engine.model_manager import ActiveRequestsError

        lm = mock_manager._loaded["qwen3:latest"]
        lm.active_refs = 1
        # ActiveRequestsError is the narrow type the unload HTTP handler
        # catches for 409. It also subclasses RuntimeError so legacy
        # callers using ``except RuntimeError:`` continue to work.
        with pytest.raises(ActiveRequestsError, match="active"):
            await mock_manager.unload("qwen3")
        assert issubclass(ActiveRequestsError, RuntimeError)
        assert len(mock_manager.get_loaded()) == 1  # still loaded
        lm.active_refs = 0

    @pytest.mark.asyncio
    async def test_unload_absorbs_close_failure(self, mock_manager):
        """unload() returns True even when _close_loaded_model raises.

        The model is already popped from ``_loaded`` before close is
        attempted, so the user-visible semantics are satisfied: the
        model is gone. Surfacing the ExceptionGroup as a 500 would
        leave the HTTP client unable to distinguish "close failed,
        model is gone" from an unrelated 500.
        """
        from unittest.mock import MagicMock

        # Replace _close_loaded_model with one that raises like the
        # real helper does when a resource close fails.
        mock_manager._close_loaded_model = MagicMock(
            side_effect=ExceptionGroup("simulated", [RuntimeError("prefetcher boom")])
        )
        result = await mock_manager.unload("qwen3")
        assert result is True
        assert "qwen3:latest" not in mock_manager._loaded

    @pytest.mark.asyncio
    async def test_unload_offloads_close_to_thread(self, mock_manager, monkeypatch):
        """unload() must run _close_loaded_model off the event loop.

        Same problem as #315 on a different code path: the sync HTTP
        ``/api/unload`` handler in routers/manage.py calls ``unload``,
        which calls ``_close_loaded_model``, which joins 48 threads
        synchronously. Doing that on the event loop stalls every
        concurrent coroutine until the pools drain.
        """
        original_to_thread = asyncio.to_thread
        to_thread_calls: list[Any] = []

        async def _tracking_to_thread(fn, *args, **kwargs):
            to_thread_calls.append(fn)
            return await original_to_thread(fn, *args, **kwargs)

        monkeypatch.setattr(
            "olmlx.engine.model_manager.asyncio.to_thread", _tracking_to_thread
        )

        result = await mock_manager.unload("qwen3")

        assert result is True
        assert mock_manager._close_loaded_model in to_thread_calls

    @pytest.mark.asyncio
    async def test_unload_calls_flush_metal(self, mock_manager):
        """unload() must call _flush_metal() after closing the model."""
        mock_manager._close_loaded_model = MagicMock()  # type: ignore[method-assign]
        with patch.object(
            mock_manager, "_flush_metal", new_callable=AsyncMock
        ) as mock_flush:
            result = await mock_manager.unload("qwen3")
        assert result is True
        mock_flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unload_skips_flush_when_cleanup_pending(self, mock_manager):
        """If any deferred cleanup is in flight, _flush_metal is skipped.
        The guard is global because mx.clear_cache() flushes the entire
        Metal allocator — any background thread allocating Metal memory,
        even for a different model, makes the concurrent clear unsafe."""
        import asyncio

        mock_manager._close_loaded_model = MagicMock()  # type: ignore[method-assign]
        # Set a deferred cleanup for a DIFFERENT model — the guard is
        # global, so this should still suppress the flush.
        mock_manager._pending_cleanups["other:latest"] = asyncio.create_task(
            asyncio.sleep(0)
        )
        try:
            with patch.object(
                mock_manager, "_flush_metal", new_callable=AsyncMock
            ) as mock_flush:
                result = await mock_manager.unload("qwen3")
            assert result is True
            mock_flush.assert_not_awaited()
        finally:
            task = mock_manager._pending_cleanups.pop("other:latest", None)
            if task is not None:
                task.cancel()

    @pytest.mark.asyncio
    async def test_unload_returns_true_even_when_flush_raises(self, mock_manager):
        """If _flush_metal() raises, unload() still returns True — the model
        is already popped from _loaded and resources are closed.
        Metal memory will be reclaimed by the next flush."""
        mock_manager._close_loaded_model = MagicMock()  # type: ignore[method-assign]
        with patch.object(
            mock_manager,
            "_flush_metal",
            new_callable=AsyncMock,
            side_effect=RuntimeError("flush boom"),
        ):
            result = await mock_manager.unload("qwen3")
        assert result is True
        assert "qwen3:latest" not in mock_manager._loaded
        mock_manager._close_loaded_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_loaded_cached(self, mock_manager):
        lm = await mock_manager.ensure_loaded("qwen3")
        assert lm.name == "qwen3:latest"

    @pytest.mark.asyncio
    async def test_concurrent_ensure_loaded_sees_registered_model_during_probe(
        self, registry, mock_store, monkeypatch
    ):
        """A second ensure_loaded during the probe's async flush must return
        the already-registered model with fully-configured cache flags."""
        import asyncio

        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 10)
        monkeypatch.setattr(
            "olmlx.engine.model_manager.mx.metal.is_available", lambda: True
        )
        manager = ModelManager(registry, mock_store)

        probe_in_flight = asyncio.Event()
        probe_done = asyncio.Event()

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.chat_template = None

        async def _stalling_probe(lm):
            # Set flags synchronously (matching the real implementation's
            # contract), then block until the test signals completion.
            lm.supports_cache_trim = True
            lm.supports_cache_persistence = True
            probe_in_flight.set()
            await probe_done.wait()

        manager._probe_cache_capabilities = _stalling_probe  # type: ignore[method-assign]

        total_ram = 64 * 1024**3

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(model, tokenizer, False, TemplateCaps(), None),
            ),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * 1024**3,
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
        ):
            task1 = asyncio.ensure_future(manager.ensure_loaded("qwen3"))

            # Wait for the probe to set flags and signal readiness
            await asyncio.wait_for(probe_in_flight.wait(), timeout=5.0)

            # Second caller for the same model — must get the cached LM
            task2 = asyncio.ensure_future(manager.ensure_loaded("qwen3"))

            # Let probe complete
            probe_done.set()

            lm1 = await asyncio.wait_for(task1, timeout=5.0)
            lm2 = await asyncio.wait_for(task2, timeout=5.0)

        assert lm1 is lm2
        # Flags were set synchronously before the probe awaited, so the
        # second caller sees fully-configured state.
        assert lm1.supports_cache_trim is True
        assert lm2.supports_cache_trim is True
        assert lm1.supports_cache_persistence is True
        assert lm2.supports_cache_persistence is True
        assert "qwen3:latest" in manager._loaded

    @pytest.mark.asyncio
    async def test_ensure_loaded_pin_survives_unload_during_probe(
        self, registry, mock_store, monkeypatch
    ):
        """Regression for PR #394 review: unload() runs without the manager
        lock, so during a freshly-loaded model's _probe_cache_capabilities
        await it can pop the model out from under a caller that asked for
        a pin. The pin must be acquired BEFORE the probe await so the
        active_refs == 0 check in unload() rejects the close."""
        import asyncio

        from olmlx.engine.model_manager import ActiveRequestsError

        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 10)
        monkeypatch.setattr(
            "olmlx.engine.model_manager.mx.metal.is_available", lambda: True
        )
        manager = ModelManager(registry, mock_store)

        probe_in_flight = asyncio.Event()
        probe_done = asyncio.Event()

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.chat_template = None

        async def _stalling_probe(lm):
            lm.supports_cache_trim = True
            lm.supports_cache_persistence = True
            probe_in_flight.set()
            await probe_done.wait()

        manager._probe_cache_capabilities = _stalling_probe  # type: ignore[method-assign]

        total_ram = 64 * 1024**3

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(model, tokenizer, False, TemplateCaps(), None),
            ),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock),
            patch.object(manager, "_close_loaded_model"),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * 1024**3,
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
        ):
            # Caller asks for a pin.
            load_task = asyncio.ensure_future(manager.ensure_loaded("qwen3", pin=True))

            # Wait until the probe has started (model is registered in _loaded).
            await asyncio.wait_for(probe_in_flight.wait(), timeout=5.0)

            # Concurrent unload while the probe is still awaiting. With the
            # bug, active_refs is still 0 here and unload pops the model.
            # With the fix, the pin was already acquired and unload raises.
            unload_raised = False
            try:
                await manager.unload("qwen3")
            except ActiveRequestsError:
                unload_raised = True

            # Let the probe finish.
            probe_done.set()
            lm = await asyncio.wait_for(load_task, timeout=5.0)

        assert unload_raised, (
            "unload() should have raised ActiveRequestsError; the pin was "
            "not acquired before the probe await."
        )
        assert "qwen3:latest" in manager._loaded
        assert manager._loaded["qwen3:latest"] is lm
        assert lm.active_refs == 1  # caller still holds the pin

    @pytest.mark.asyncio
    async def test_ensure_loaded_refreshes_expiry(self, mock_manager):
        lm = await mock_manager.ensure_loaded("qwen3", keep_alive="10m")
        assert lm.expires_at is not None
        assert lm.expires_at > time.time()

    @pytest.mark.asyncio
    async def test_ensure_loaded_never_expire(self, mock_manager):
        lm = await mock_manager.ensure_loaded("qwen3", keep_alive="-1")
        assert lm.expires_at is None

    @pytest.mark.asyncio
    async def test_ensure_loaded_unknown_model(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        with pytest.raises(ValueError, match="not found"):
            await manager.ensure_loaded("unknown_model")

    @pytest.mark.asyncio
    async def test_ensure_loaded_evicts_lru(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)

        # Pre-load a model
        old_lm = LoadedModel(
            name="llama3:8b",
            hf_path="mlx-community/Llama-3-8B-Instruct",
            model=MagicMock(),
            tokenizer=MagicMock(),
            loaded_at=time.time() - 100,
        )
        manager._loaded["llama3:8b"] = old_lm

        # Mock _load_model
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        with patch.object(
            manager,
            "_load_model",
            return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
        ):
            await manager.ensure_loaded("qwen3")

        assert "llama3:8b" not in manager._loaded
        assert "qwen3:latest" in manager._loaded

    @pytest.mark.asyncio
    async def test_stop(self, mock_manager):
        # Should not raise even without expiry task
        await mock_manager.stop()
        assert mock_manager._loaded == {}

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_cleanups(self, mock_manager):
        """stop() cancels and clears pending cleanup tasks."""

        async def dummy():
            await asyncio.sleep(100)

        task = asyncio.create_task(dummy())
        mock_manager._pending_cleanups["test:latest"] = task
        await mock_manager.stop()
        assert mock_manager._pending_cleanups == {}
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_cancels_expiry_task(self, mock_manager):
        # Create a dummy task
        async def dummy():
            await asyncio.sleep(100)

        mock_manager._expiry_task = asyncio.create_task(dummy())
        await mock_manager.stop()
        assert mock_manager._expiry_task.cancelled()


class TestResolveDraftPath:
    """``_resolve_draft_path`` accepts both HF repo ids and local paths.

    Operators training drafts via ``olmlx dflash prepare`` end up with a
    directory under ``~/.olmlx/models/<target>/dflash/`` and configure
    ``--speculative-draft-model /abs/path/to/dflash``. Without
    short-circuiting local paths, the resolver passes the absolute path
    into ``store.ensure_downloaded`` → ``huggingface_hub.HfApi`` →
    ``HFValidationError`` ("Repo id must be in the form 'repo_name' or
    'namespace/repo_name'") and the request fails with a confusing 400.
    """

    def test_local_directory_returns_as_is(self, tmp_path, registry, mock_store):
        local = tmp_path / "my-draft"
        local.mkdir()
        manager = ModelManager(registry, mock_store)
        # Should not call ``store.ensure_downloaded`` — the path exists.
        with patch.object(
            mock_store, "ensure_downloaded", side_effect=AssertionError("called")
        ):
            resolved = manager._resolve_draft_path(str(local))
        assert resolved == str(local)

    def test_hf_repo_id_goes_through_store(self, tmp_path, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        expected = tmp_path / "downloaded"
        expected.mkdir()
        with patch.object(
            mock_store, "ensure_downloaded", return_value=expected
        ) as mock_dl:
            resolved = manager._resolve_draft_path("namespace/repo_name")
        mock_dl.assert_called_once_with("namespace/repo_name")
        assert resolved == str(expected)

    def test_absolute_missing_path_raises_file_not_found(
        self, tmp_path, registry, mock_store
    ):
        """Absolute paths are unambiguous local references; they
        cannot also be valid HF repo ids. Falling through to
        ``ensure_downloaded`` for a missing absolute path would raise
        ``HFValidationError`` ("Repo id must be in the form
        'repo_name' or 'namespace/repo_name'") which is actively
        misleading. Raise ``FileNotFoundError`` with the actual path
        so a typo or a path pointing at a not-yet-trained draft
        produces a clear, actionable error.
        """
        manager = ModelManager(registry, mock_store)
        missing = tmp_path / "definitely-not-here"
        # Make sure ``ensure_downloaded`` is NOT consulted for a
        # missing absolute path.
        with patch.object(
            mock_store, "ensure_downloaded", side_effect=AssertionError("called")
        ):
            with pytest.raises(FileNotFoundError, match="definitely-not-here"):
                manager._resolve_draft_path(str(missing))

    def test_relative_path_collision_does_not_short_circuit(
        self, tmp_path, registry, mock_store, monkeypatch
    ):
        """Relative paths must NOT short-circuit even if a directory by
        that name happens to exist relative to CWD. Otherwise a valid HF
        repo id like ``"my-org/dflash-draft"`` is silently swapped for
        whatever the current working directory contains under that path.
        Only absolute paths are unambiguous local references.
        """
        # Set CWD to tmp_path and create a directory matching a plausible
        # HF repo id within it.
        monkeypatch.chdir(tmp_path)
        collision = tmp_path / "namespace" / "repo_name"
        collision.mkdir(parents=True)

        manager = ModelManager(registry, mock_store)
        downloaded = tmp_path / "downloaded"
        downloaded.mkdir()
        with patch.object(
            mock_store, "ensure_downloaded", return_value=downloaded
        ) as mock_dl:
            resolved = manager._resolve_draft_path("namespace/repo_name")
        # Must have gone through ``ensure_downloaded`` rather than
        # picking up the colliding local directory.
        mock_dl.assert_called_once_with("namespace/repo_name")
        assert resolved == str(downloaded)


class TestDetectModelKind:
    def _make_config(self, tmp_path, config_data):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))
        return str(config_path)

    def _make_manager(self, registry, mock_store):
        return ModelManager(registry, mock_store)

    def test_text_model(self, tmp_path, registry, mock_store):
        config_path = self._make_config(tmp_path, {"model_type": "llama"})
        manager = self._make_manager(registry, mock_store)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/model")
        assert kind == "text"

    def test_vlm_with_vision_keys(self, tmp_path, registry, mock_store):
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "qwen2_vl",
                "vision_config": {"hidden_size": 1024},
            },
        )
        manager = self._make_manager(registry, mock_store)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/vlm")
        assert kind == "vlm"

    def test_config_download_fails(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        with patch(
            "huggingface_hub.hf_hub_download", side_effect=Exception("not found")
        ):
            kind = manager._detect_model_kind("nonexistent/model")
        assert kind == "unknown"

    def test_no_model_type(self, tmp_path, registry, mock_store):
        config_path = self._make_config(tmp_path, {"hidden_size": 1024})
        manager = self._make_manager(registry, mock_store)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/model")
        assert kind == "unknown"

    def test_vision_keys_unsupported_by_vlm_but_supported_by_lm(
        self, tmp_path, registry, mock_store
    ):
        """Model has vision keys but mlx-vlm doesn't support it; mlx-lm does → text."""
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "llama",  # known to mlx-lm
                "vision_config": {},
                "image_token_id": 42,
            },
        )
        manager = self._make_manager(registry, mock_store)

        import importlib.util

        real_find_spec = importlib.util.find_spec

        def no_vlm_yes_lm(name, *args, **kwargs):
            if name.startswith("mlx_vlm.models."):
                return None
            return real_find_spec(name, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            with patch("importlib.util.find_spec", side_effect=no_vlm_yes_lm):
                kind = manager._detect_model_kind("test/model")
        assert kind == "text"

    def test_vision_keys_unsupported_by_both(self, tmp_path, registry, mock_store):
        """Model has vision keys but neither library supports it → unknown."""
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "custom_vlm",
                "vision_config": {},
            },
        )
        manager = self._make_manager(registry, mock_store)

        import importlib.util

        real_find_spec = importlib.util.find_spec

        def none_for_models(name, *args, **kwargs):
            if name.startswith(("mlx_lm.models.", "mlx_vlm.models.")):
                return None
            return real_find_spec(name, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            with patch("importlib.util.find_spec", side_effect=none_for_models):
                kind = manager._detect_model_kind("test/vlm")
        # Neither library supports it — return unknown to try both
        assert kind == "unknown"

    def test_text_model_with_real_imports(self, tmp_path, registry, mock_store):
        """Test _detect_model_kind with a model_type that exists in mlx-lm."""
        config_path = self._make_config(tmp_path, {"model_type": "llama"})
        manager = self._make_manager(registry, mock_store)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/model")
        # llama is a known text model type
        assert kind == "text"

    def test_uses_local_config_first(self, tmp_path, registry, mock_store):
        """When config.json exists locally, skip HF hub download."""
        # Write config.json in the store's local path
        local_dir = mock_store.local_path("test/model")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))

        manager = self._make_manager(registry, mock_store)
        # Should NOT call hf_hub_download
        with patch("huggingface_hub.hf_hub_download") as mock_dl:
            kind = manager._detect_model_kind("test/model")
        assert kind == "text"
        mock_dl.assert_not_called()

    def test_hybrid_linear_attention_vlm_routes_to_text(
        self, tmp_path, registry, mock_store
    ):
        """Issue #284: VLMs with hybrid SSM+attention layers (Qwen3.5,
        Qwen3_5_moe) must route through mlx-lm's text path.  The mlx-vlm
        path crashes on stream synchronization for these models even on
        text-only requests; mlx-lm has dedicated text-only modules
        (qwen3_5.py, qwen3_5_moe.py) that work correctly.

        Discriminator: ``text_config.layer_types`` containing
        ``"linear_attention"`` signals the hybrid architecture.  Standard
        VLMs (Gemma 4, Qwen2-VL) lack this field and continue to route
        through mlx-vlm.
        """
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "qwen3_5",  # mlx-lm has qwen3_5.py
                "vision_config": {"hidden_size": 1024},
                "image_token_id": 248056,
                "text_config": {
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ],
                },
            },
        )
        manager = self._make_manager(registry, mock_store)

        # Mock find_spec so the test doesn't depend on the installed mlx-lm
        # version actually shipping mlx_lm.models.qwen3_5 — without the
        # mock, an older install would silently route through mlx-vlm and
        # the assertion would mask a regression.
        import importlib.util

        real_find_spec = importlib.util.find_spec

        def find_spec_with_qwen3_5(name, *args, **kwargs):
            if name == "mlx_lm.models.qwen3_5":
                return object()  # truthy sentinel
            return real_find_spec(name, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            with patch("importlib.util.find_spec", side_effect=find_spec_with_qwen3_5):
                kind = manager._detect_model_kind("test/qwen3_5")
        assert kind == "text"

    def test_hybrid_linear_attention_vlm_raises_when_no_mlx_lm_module(
        self, tmp_path, registry, mock_store
    ):
        """Issue #284: when the discriminator fires (linear_attention layers
        present) but mlx-lm has no module for the model_type, raise a
        clear ValueError at detection time rather than falling through to
        the mlx-vlm path that we know crashes."""
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "future_hybrid_vlm",
                "vision_config": {"hidden_size": 1024},
                "text_config": {"layer_types": ["linear_attention", "full_attention"]},
            },
        )
        manager = self._make_manager(registry, mock_store)

        import importlib.util

        real_find_spec = importlib.util.find_spec

        def no_mlx_lm_module(name, *args, **kwargs):
            if name.startswith("mlx_lm.models."):
                return None
            return real_find_spec(name, *args, **kwargs)

        with (
            patch("huggingface_hub.hf_hub_download", return_value=config_path),
            patch("importlib.util.find_spec", side_effect=no_mlx_lm_module),
            pytest.raises(ValueError, match="hybrid linear-attention"),
        ):
            manager._detect_model_kind("test/future_hybrid_vlm")

    def test_hybrid_linear_attention_vlm_uses_text_config_model_type(
        self, tmp_path, registry, mock_store
    ):
        """Issue #284: when the top-level model_type is VLM-specific (e.g.
        a hypothetical ``qwen3_5_vl``) but ``text_config.model_type`` names
        the architecture mlx-lm actually has a module for (``qwen3_5``),
        the lookup should prefer the text_config key.  Otherwise the
        routing falls through to ``unknown`` and the model fails to load.
        """
        config_path = self._make_config(
            tmp_path,
            {
                # Top-level: VLM-specific name with no mlx-lm module.
                "model_type": "qwen3_5_vl",
                "vision_config": {"hidden_size": 1024},
                "text_config": {
                    # Inner: the architecture name mlx-lm has a module for.
                    "model_type": "qwen3_5",
                    "layer_types": ["linear_attention", "full_attention"],
                },
            },
        )
        manager = self._make_manager(registry, mock_store)

        import importlib.util

        real_find_spec = importlib.util.find_spec

        def find_spec_with_qwen3_5_only(name, *args, **kwargs):
            if name == "mlx_lm.models.qwen3_5":
                return object()
            if name == "mlx_lm.models.qwen3_5_vl":
                return None  # mlx-lm has no qwen3_5_vl module
            return real_find_spec(name, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            with patch(
                "importlib.util.find_spec", side_effect=find_spec_with_qwen3_5_only
            ):
                kind = manager._detect_model_kind("test/qwen3_5_vl")
        assert kind == "text"

    def test_hybrid_linear_attention_vlm_falls_back_to_top_level_model_type(
        self, tmp_path, registry, mock_store
    ):
        """Inverse of the previous case: ``text_config.model_type`` names a
        module mlx-lm doesn't ship (e.g. Qwen3.6's ``qwen3_5_moe_text``),
        but the top-level ``model_type`` does (``qwen3_5_moe``). The
        discriminator should fall back to the top-level type instead of
        raising — otherwise olmlx serve refuses to load any
        ``_text``-suffixed hybrid VLM, even though ``mlx_lm.load()`` (which
        the rest of the engine actually uses) handles them via the
        top-level type without issue.
        """
        config_path = self._make_config(
            tmp_path,
            {
                # Top-level: the name mlx-lm has a module for.
                "model_type": "qwen3_5_moe",
                "vision_config": {"hidden_size": 1024},
                "text_config": {
                    # Inner: the new ``_text``-suffixed convention with no
                    # matching mlx-lm module.
                    "model_type": "qwen3_5_moe_text",
                    "layer_types": ["linear_attention", "full_attention"],
                },
            },
        )
        manager = self._make_manager(registry, mock_store)

        import importlib.util

        real_find_spec = importlib.util.find_spec

        def find_spec_with_top_level_only(name, *args, **kwargs):
            if name == "mlx_lm.models.qwen3_5_moe":
                return object()
            if name == "mlx_lm.models.qwen3_5_moe_text":
                return None  # mlx-lm has no _text-suffixed module
            return real_find_spec(name, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            with patch(
                "importlib.util.find_spec", side_effect=find_spec_with_top_level_only
            ):
                kind = manager._detect_model_kind("test/qwen3_5_moe")
        assert kind == "text"

    def test_hybrid_linear_attention_vlm_raises_when_mlx_lm_import_fails(
        self, tmp_path, registry, mock_store
    ):
        """Issue #284: when ``from mlx_lm.utils import MODEL_REMAPPING``
        raises ImportError (older mlx-lm without that export, or mlx-lm
        absent), the discriminator has already fired — raise a clear
        ValueError at detection time rather than fall through to the
        mlx-vlm path that we know crashes."""
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "qwen3_5",
                "vision_config": {"hidden_size": 1024},
                "text_config": {"layer_types": ["linear_attention", "full_attention"]},
            },
        )
        manager = self._make_manager(registry, mock_store)

        # Setting sys.modules["mlx_lm.utils"] = None makes Python raise
        # ImportError on subsequent ``from mlx_lm.utils import ...``
        # statements, regardless of whether mlx_lm.utils was already
        # imported in the test environment.
        with (
            patch("huggingface_hub.hf_hub_download", return_value=config_path),
            patch.dict("sys.modules", {"mlx_lm.utils": None}),
            pytest.raises(ValueError, match="hybrid linear-attention"),
        ):
            manager._detect_model_kind("test/qwen3_5")

    def test_standard_vlm_without_linear_attention_stays_vlm(
        self, tmp_path, registry, mock_store
    ):
        """Regression fence: standard VLMs (no linear_attention) must
        continue to load through mlx-vlm.  Only the hybrid SSM bug warrants
        the mlx-lm detour."""
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "qwen2_vl",  # known mlx-vlm model
                "vision_config": {"hidden_size": 1024},
                "image_token_id": 151655,
                # No layer_types — standard transformer.
            },
        )
        manager = self._make_manager(registry, mock_store)

        # Mock find_spec for the mlx-vlm verification block so the test
        # doesn't depend on the installed mlx-vlm version actually shipping
        # mlx_vlm.models.qwen2_vl.
        import importlib.util

        real_find_spec = importlib.util.find_spec

        def find_spec_with_qwen2_vl(name, *args, **kwargs):
            if name == "mlx_vlm.models.qwen2_vl":
                return object()  # truthy sentinel
            return real_find_spec(name, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            with patch("importlib.util.find_spec", side_effect=find_spec_with_qwen2_vl):
                kind = manager._detect_model_kind("test/qwen2_vl")
        assert kind == "vlm"

    def test_whisper_by_model_type(self, tmp_path, registry, mock_store):
        config_path = self._make_config(tmp_path, {"model_type": "whisper"})
        manager = self._make_manager(registry, mock_store)
        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/whisper")
        assert kind == "whisper"

    def test_whisper_by_dims_without_model_type(self, tmp_path, registry, mock_store):
        # mlx-community whisper repos ship a non-HF config.json with dims and
        # no usable model_type (load_model pops it).
        config_path = self._make_config(
            tmp_path,
            {
                "n_mels": 80,
                "n_audio_ctx": 1500,
                "n_audio_state": 384,
                "n_audio_head": 6,
                "n_audio_layer": 4,
                "n_vocab": 51865,
                "n_text_ctx": 448,
                "n_text_state": 384,
                "n_text_head": 6,
                "n_text_layer": 4,
            },
        )
        manager = self._make_manager(registry, mock_store)
        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/whisper-mlx")
        assert kind == "whisper"

    def test_llama_not_whisper(self, tmp_path, registry, mock_store):
        config_path = self._make_config(tmp_path, {"model_type": "llama"})
        manager = self._make_manager(registry, mock_store)
        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/model")
        assert kind == "text"

    def test_gemma4_unified_text_routes_to_text(self, registry, mock_store):
        """mlx-community 'gemma4_unified' text checkpoints (e.g.
        gemma-4-12B-it-4bit) route to the mlx-lm text path (language tower
        only); their multimodal weight layout (vision_embedder.*) loads in
        neither mlx-lm's gemma4_text nor mlx-vlm 0.4.4's gemma4.  Detection must
        NOT rewrite config.json on disk — the earlier approach corrupted the
        shared store copy.
        """
        local_dir = mock_store.local_path("mlx-community/gemma-4-12B-it-4bit")
        local_dir.mkdir(parents=True)
        config = {
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "text_config": {"model_type": "gemma4_unified_text"},
            "vision_config": {"model_type": "gemma4_unified_vision"},
            "audio_config": {"model_type": "gemma4_unified_audio"},
        }
        cfg_path = local_dir / "config.json"
        original = json.dumps(config)
        cfg_path.write_text(original)

        manager = self._make_manager(registry, mock_store)
        kind = manager._detect_model_kind("mlx-community/gemma-4-12B-it-4bit")

        assert kind == "text"
        # config.json must be byte-for-byte untouched on disk.
        assert cfg_path.read_text() == original
        assert json.loads(cfg_path.read_text())["model_type"] == "gemma4_unified"

    def test_standard_gemma4_vlm_unaffected(self, tmp_path, registry, mock_store):
        """Standard 'gemma4' checkpoints (e4b/31b) keep their VLM routing."""
        config_path = self._make_config(
            tmp_path,
            {"model_type": "gemma4", "vision_config": {"hidden_size": 1024}},
        )
        manager = self._make_manager(registry, mock_store)
        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/gemma4")
        assert kind == "vlm"


class TestGemma4UnifiedTextLoader:
    """Dispatch logic for loading the language tower of 'gemma4_unified'
    checkpoints (the heavy weight load itself is covered by the live test).
    """

    def _write_config(self, d, cfg):
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(cfg))

    def test_dispatch_missing_config(self, tmp_path):
        from olmlx.engine.model_manager import _maybe_load_gemma4_unified_text

        assert _maybe_load_gemma4_unified_text(str(tmp_path)) is None

    def test_dispatch_skips_non_unified(self, tmp_path):
        from olmlx.engine.model_manager import _maybe_load_gemma4_unified_text

        self._write_config(tmp_path, {"model_type": "gemma4", "vision_config": {}})
        assert _maybe_load_gemma4_unified_text(str(tmp_path)) is None

    def test_dispatch_skips_unified_non_text(self, tmp_path):
        from olmlx.engine.model_manager import _maybe_load_gemma4_unified_text

        self._write_config(
            tmp_path,
            {
                "model_type": "gemma4_unified",
                "text_config": {"model_type": "something_else"},
            },
        )
        assert _maybe_load_gemma4_unified_text(str(tmp_path)) is None

    def test_dispatch_invokes_loader_for_unified_text(self, tmp_path, monkeypatch):
        import olmlx.engine.model_manager as mm

        self._write_config(
            tmp_path,
            {
                "model_type": "gemma4_unified",
                "text_config": {"model_type": "gemma4_unified_text"},
            },
        )
        sentinel = (object(), object())
        called = {}

        def fake_loader(load_path, cfg):
            called["load_path"] = load_path
            called["cfg"] = cfg
            return sentinel

        monkeypatch.setattr(mm, "_load_gemma4_unified_text", fake_loader)
        result = mm._maybe_load_gemma4_unified_text(str(tmp_path))

        assert result is sentinel
        assert called["load_path"] == str(tmp_path)
        assert called["cfg"]["model_type"] == "gemma4_unified"


class TestQuantizeLanguageTower:
    """The rebuilt language tower must honor the checkpoint's *per-layer*
    quantization overrides.  mlx-community 'gemma4_unified' checkpoints carry
    mixed precision (a global ``bits``/``group_size`` plus per-module overrides,
    e.g. MLP projections at 8-bit) keyed by the full ``language_model.*`` path.
    A blanket ``nn.quantize`` at the global bits builds QuantizedLinear params
    whose packed shapes mismatch the stored 8-bit weights, so the strict load
    raises ``Expected shape ... but received shape ...`` — the original failure.
    """

    def test_honors_per_layer_bit_overrides(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        from olmlx.engine.model_manager import _quantize_language_tower

        class Tower(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(128, 256, bias=False) for _ in range(2)]

        # Reference checkpoint: layer 0 at 8-bit, layer 1 at the global 4-bit.
        ref = Tower()

        def ref_predicate(path, module):
            return {"group_size": 64, "bits": 8 if path == "layers.0" else 4}

        nn.quantize(ref, group_size=64, bits=4, class_predicate=ref_predicate)
        mx.eval(ref.parameters())
        lang_weights = dict(tree_flatten(ref.parameters()))

        # 4-bit packs in_features/8 columns, 8-bit packs in_features/4 — so the
        # mixed precision yields differently shaped weights that only load if
        # the target is quantized with matching per-layer bits.
        assert lang_weights["layers.0.weight"].shape == (256, 32)  # 8-bit
        assert lang_weights["layers.1.weight"].shape == (256, 16)  # 4-bit

        quant = {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "language_model.layers.0": {"group_size": 64, "bits": 8},
        }
        target = Tower()
        _quantize_language_tower(target, quant, lang_weights)

        # Strict load proves every packed param shape matches the checkpoint.
        target.load_weights(list(lang_weights.items()), strict=True)
        assert target.layers[0].bits == 8
        assert target.layers[1].bits == 4

    def test_skips_unquantized_modules(self):
        """A module absent from the quant dict and not stored as quantized
        (no ``.scales``) must be left in full precision."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        from olmlx.engine.model_manager import _quantize_language_tower

        class Tower(nn.Module):
            def __init__(self):
                super().__init__()
                self.quantized = nn.Linear(128, 256, bias=False)
                self.plain = nn.Linear(128, 256, bias=False)

        ref = Tower()
        nn.quantize(
            ref,
            group_size=64,
            bits=4,
            class_predicate=lambda p, m: p == "quantized",
        )
        mx.eval(ref.parameters())
        lang_weights = dict(tree_flatten(ref.parameters()))

        target = Tower()
        _quantize_language_tower(target, {"group_size": 64, "bits": 4}, lang_weights)
        target.load_weights(list(lang_weights.items()), strict=True)

        assert hasattr(target.quantized, "bits")  # quantized
        assert not hasattr(target.plain, "bits")  # left full precision


class TestProbeCacheCapabilities:
    """Exercise _probe_cache_capabilities, including the probe-failure path
    promoted to WARNING + non-persistable default in issue #284."""

    def _make_lm(self):
        from olmlx.engine.model_manager import LoadedModel

        lm = LoadedModel(
            name="probe-test:latest",
            hf_path="test/probe",
            model=MagicMock(),
            tokenizer=MagicMock(),
            template_caps=TemplateCaps(),
        )
        # Default for the dataclass is False (issue #284 safety default);
        # set True here so the test can verify the failure path explicitly
        # flips it back to False.
        lm.supports_cache_persistence = True
        lm.supports_cache_trim = False
        return lm

    @pytest.mark.asyncio
    async def test_probe_empty_cache_list_disables_persistence(
        self, registry, mock_store
    ):
        """If ``make_prompt_cache`` returns an empty list (a degenerate model
        with no cache layers), ``_cache_supports_persistence`` returns
        False — there's no evidence the cache layout is safe — and the
        probe leaves persistence disabled.  Trim's vacuous-True for the
        same input is fine because trim has a graceful fallback;
        persistence does not."""
        manager = ModelManager(registry, mock_store)
        lm = self._make_lm()

        with (
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock) as mock_flush,
        ):
            await manager._probe_cache_capabilities(lm)

        assert lm.supports_cache_trim is True
        assert lm.supports_cache_persistence is False
        # [] is non-None, so flush should be called
        mock_flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_trimmable_layout_uses_checkpoint_persistence(
        self, registry, mock_store
    ):
        """Issue #343 / Task 5.2-5.3: a layout containing a ``RotatingKVCache``
        layer makes the cache non-trimmable, but the checkpoint path can still
        reuse it safely (#343 — trim is avoided entirely).  The probe must set
        ``uses_checkpoint_persistence=True`` and ``supports_cache_persistence=True``
        for these layouts."""

        class _FakeRotating:
            pass

        _FakeRotating.__name__ = "RotatingKVCache"

        manager = ModelManager(registry, mock_store)
        lm = self._make_lm()

        probe_cache = [_FakeRotating(), _FakeRotating()]
        with (
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=probe_cache),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock) as mock_flush,
        ):
            await manager._probe_cache_capabilities(lm)

        assert lm.supports_cache_trim is False
        assert lm.uses_checkpoint_persistence is True
        assert lm.supports_cache_persistence is True
        mock_flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_chunked_kv_cache_layout_uses_checkpoint_persistence(
        self, registry, mock_store
    ):
        """``ChunkedKVCache`` is the other non-trimmable layout cited by
        #343 and CLAUDE.md (mlx-lm's chunk-based cache; affects newer
        Apple-published checkpoints).  Like ``RotatingKVCache`` it sits
        in the checkpoint-persist allowlist — so after Tasks 5.2/5.3 the
        probe must set ``uses_checkpoint_persistence=True`` and keep
        ``supports_cache_persistence=True``."""

        class _FakeChunked:
            pass

        _FakeChunked.__name__ = "ChunkedKVCache"

        manager = ModelManager(registry, mock_store)
        lm = self._make_lm()

        probe_cache = [_FakeChunked(), _FakeChunked()]
        with (
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=probe_cache),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock) as mock_flush,
        ):
            await manager._probe_cache_capabilities(lm)

        assert lm.supports_cache_trim is False
        assert lm.uses_checkpoint_persistence is True
        assert lm.supports_cache_persistence is True
        mock_flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mixed_layout_kv_plus_rotating_uses_checkpoint_persistence(
        self, registry, mock_store
    ):
        """Real Gemma 4 layout: full-attention layers produce ``KVCache``
        (trimmable + persistable), sliding-window layers produce
        ``RotatingKVCache`` (non-trimmable, layout-persistable).
        After Tasks 5.2/5.3: the checkpoint path supports this layout
        (KVCache+RotatingKVCache is not in the excluded-pairs set), so
        the probe must set ``uses_checkpoint_persistence=True`` and
        ``supports_cache_persistence=True``."""

        class _FakeKV:
            pass

        class _FakeRotating:
            pass

        _FakeKV.__name__ = "KVCache"
        _FakeRotating.__name__ = "RotatingKVCache"

        manager = ModelManager(registry, mock_store)
        lm = self._make_lm()

        probe_cache = [_FakeKV(), _FakeRotating(), _FakeKV(), _FakeRotating()]
        with (
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=probe_cache),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock) as mock_flush,
        ):
            await manager._probe_cache_capabilities(lm)

        assert lm.supports_cache_trim is False
        assert lm.uses_checkpoint_persistence is True
        assert lm.supports_cache_persistence is True
        mock_flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_trimmable_layout_keeps_persistence(self, registry, mock_store):
        """Guard the inverse: a fully trimmable layout (plain ``KVCache``
        layers) must still report persistence True after the #343 fix.
        Otherwise the fix would silently disable cache reuse for the
        primary supported model family (Qwen3, Llama-3, etc.)."""

        class _FakeKV:
            pass

        _FakeKV.__name__ = "KVCache"

        manager = ModelManager(registry, mock_store)
        lm = self._make_lm()

        probe_cache = [_FakeKV(), _FakeKV()]
        with (
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=probe_cache),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock) as mock_flush,
        ):
            await manager._probe_cache_capabilities(lm)

        assert lm.supports_cache_trim is True
        assert lm.supports_cache_persistence is True
        mock_flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_trimmable_persistable_layout_no_disabled_log(
        self, registry, mock_store, caplog
    ):
        """After Tasks 5.2/5.3: a pure RotatingKVCache layout now sets
        ``uses_checkpoint_persistence=True`` and ``supports_cache_persistence=True``,
        so the "cross-request reuse disabled" log gate
        (``not lm.supports_cache_persistence``) is False and neither the
        #343 nor the #284 attribution is emitted."""
        import logging

        class _FakeRotating:
            pass

        _FakeRotating.__name__ = "RotatingKVCache"

        manager = ModelManager(registry, mock_store)
        lm = self._make_lm()

        with (
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=[_FakeRotating(), _FakeRotating()],
            ),
            caplog.at_level(logging.INFO, logger="olmlx.engine.model_manager"),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock) as mock_flush,
        ):
            await manager._probe_cache_capabilities(lm)

        assert lm.supports_cache_persistence is True
        assert lm.uses_checkpoint_persistence is True
        # The "disabled" log branches are gated on not supports_cache_persistence;
        # since persistence is now True for this layout, neither fires.
        assert "issue #343" not in caplog.text
        assert "issue #284" not in caplog.text
        mock_flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_arrays_cache_layout_uses_checkpoint_persistence(
        self, registry, mock_store, caplog
    ):
        """After Tasks 5.2/5.3: a pure ArraysCache layout (hybrid SSM,
        e.g. Qwen3.5) now sets ``uses_checkpoint_persistence=True`` and
        ``supports_cache_persistence=True`` — the checkpoint path can
        reuse the cache safely via mx.eval before snapshot (#284).
        The "cross-request reuse disabled" logs must NOT fire because
        persistence is enabled."""
        import logging

        class _FakeArrays:
            pass

        _FakeArrays.__name__ = "ArraysCache"

        manager = ModelManager(registry, mock_store)
        lm = self._make_lm()

        with (
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=[_FakeArrays(), _FakeArrays()],
            ),
            caplog.at_level(logging.INFO, logger="olmlx.engine.model_manager"),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock) as mock_flush,
        ):
            await manager._probe_cache_capabilities(lm)

        assert lm.supports_cache_trim is False
        assert lm.uses_checkpoint_persistence is True
        assert lm.supports_cache_persistence is True
        # The "disabled" log branches are gated on not supports_cache_persistence;
        # since persistence is now True for this layout, neither fires.
        assert "issue #284" not in caplog.text
        assert "issue #343" not in caplog.text
        mock_flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_turboquant_kv_quant_does_not_block_checkpoint_persistence(
        self, registry, mock_store
    ):
        """With ``TurboQuantKVCache.__deepcopy__`` overriding the
        ``mx.Dtype`` pickle hazard, ``turboquant:4`` no longer blocks the
        checkpoint path. Verifies the gate at ``_probe_cache_capabilities``
        wires through ``lm.kv_cache_quant`` correctly: an ArraysCache
        layout with TurboQuant configured must still set
        ``uses_checkpoint_persistence=True`` (otherwise hybrid SSM models
        on TurboQuant get zero cross-request reuse — the bug this PR
        targets)."""

        class _FakeArrays:
            pass

        _FakeArrays.__name__ = "ArraysCache"

        manager = ModelManager(registry, mock_store)
        lm = self._make_lm()
        lm.kv_cache_quant = "turboquant:4"

        with (
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=[_FakeArrays(), _FakeArrays()],
            ),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock),
        ):
            await manager._probe_cache_capabilities(lm)

        assert lm.uses_checkpoint_persistence is True
        assert lm.supports_cache_persistence is True

    @pytest.mark.asyncio
    async def test_spectralquant_kv_quant_does_not_block_checkpoint_persistence(
        self, registry, mock_store
    ):
        """``SpectralQuantKVCache`` deepcopies cleanly already; the
        defensive block by analogy with the disk-save path is dropped."""

        class _FakeArrays:
            pass

        _FakeArrays.__name__ = "ArraysCache"

        manager = ModelManager(registry, mock_store)
        lm = self._make_lm()
        lm.kv_cache_quant = "spectral:4"

        with (
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=[_FakeArrays(), _FakeArrays()],
            ),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock),
        ):
            await manager._probe_cache_capabilities(lm)

        assert lm.uses_checkpoint_persistence is True
        assert lm.supports_cache_persistence is True

    @pytest.mark.asyncio
    async def test_probe_failure_warns_and_disables_persistence(
        self, registry, mock_store, caplog
    ):
        """When ``make_prompt_cache`` raises, the probe must log at WARNING,
        force ``supports_cache_persistence = False`` (no graceful fallback
        for cache reuse), and force ``supports_cache_trim = True`` (the
        request path's existing partial-trim fallback handles it)."""
        import logging

        manager = ModelManager(registry, mock_store)
        lm = self._make_lm()

        with (
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                side_effect=RuntimeError("simulated probe failure"),
            ),
            caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock) as mock_flush,
        ):
            await manager._probe_cache_capabilities(lm)

        assert lm.supports_cache_trim is True
        assert lm.supports_cache_persistence is False
        assert "Cache probe raised an exception" in caplog.text
        # probe_cache was never created, so flush should NOT be called
        mock_flush.assert_not_awaited()


class TestLoadModel:
    def _make_manager(self, registry, mock_store):
        return ModelManager(registry, mock_store)

    def _pre_download(self, mock_store, hf_path):
        """Simulate a downloaded model by creating config.json in the store."""
        local_dir = mock_store.local_path(hf_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

    def test_load_text_model(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/path")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                model, tokenizer, is_vlm, caps, _ = manager._load_model("test/path")

        assert is_vlm is False
        assert model is mock_model

    def test_load_vlm_model(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/vlm")
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                model, tokenizer, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True

    def test_load_vlm_loads_chat_template_from_jinja_file(self, registry, mock_store):
        """When VLM tokenizer has no chat_template, load from chat_template.jinja."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        template = (
            "{% if tools %}tools{% endif %}{% if enable_thinking %}<think>{% endif %}"
        )
        (local_dir / "chat_template.jinja").write_text(template)

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                _, _, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True
        assert mock_tok.chat_template == template
        assert caps.supports_tools is True
        assert caps.supports_enable_thinking is True

    def test_load_vlm_loads_chat_template_from_json_file(self, registry, mock_store):
        """When VLM tokenizer has no chat_template, load from chat_template.json."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        template = "{% if tools %}tools{% endif %}"
        (local_dir / "chat_template.json").write_text(
            json.dumps({"chat_template": template})
        )

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                _, _, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True
        assert mock_tok.chat_template == template
        assert caps.supports_tools is True

    def test_load_vlm_falls_back_on_oserror(self, registry, mock_store):
        """When mlx_vlm.load fails with OSError, fall back to _try_lm_then_vlm."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/vlm")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.side_effect = OSError("preprocessor_config.json")
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
            with patch.dict(
                "sys.modules", {"mlx_vlm": mock_mlx_vlm, "mlx_lm": mock_mlx_lm}
            ):
                model, tokenizer, is_vlm, caps, _ = manager._load_model("test/vlm")

        # Should have fallen back to mlx-lm (not VLM)
        assert is_vlm is False
        assert model is mock_model

    def test_load_vlm_skips_chat_template_when_already_set(self, registry, mock_store):
        """When VLM tokenizer already has chat_template, don't overwrite."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        (local_dir / "chat_template.jinja").write_text("file template")

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = "existing template"
        mock_processor.tokenizer = mock_tok

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                _, _, _, _, _ = manager._load_model("test/vlm")

        assert mock_tok.chat_template == "existing template"

    def test_load_fallback_loads_chat_template_from_jinja_file(
        self, registry, mock_store
    ):
        """When fallback to VLM and tokenizer has no chat_template, load from file."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/path")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        template = "{% if tools %}tools{% endif %}"
        (local_dir / "chat_template.jinja").write_text(template)

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        with patch.object(manager, "_detect_model_kind", return_value="unknown"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.side_effect = ValueError("fail")
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
            ):
                _, _, is_vlm, caps, _ = manager._load_model("test/path")

        assert is_vlm is True
        assert mock_tok.chat_template == template
        assert caps.supports_tools is True

    def test_load_vlm_downloads_chat_template_from_hub(self, registry, mock_store):
        """When chat_template.jinja not local, try downloading from HF hub."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        # No chat_template files locally

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        template = "{% if tools %}tools{% endif %}"
        # Write the downloaded file to a temp location
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write(template)
            downloaded_path = f.name

        mock_hf_mod = MagicMock()
        mock_hf_mod.hf_hub_download.return_value = downloaded_path

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules",
                {"mlx_vlm": mock_mlx_vlm, "huggingface_hub": mock_hf_mod},
            ):
                _, _, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True
        assert mock_tok.chat_template == template
        assert caps.supports_tools is True
        mock_hf_mod.hf_hub_download.assert_called_once_with(
            "test/vlm", "chat_template.jinja"
        )

        Path(downloaded_path).unlink(missing_ok=True)

    def test_load_vlm_falls_back_to_base_model_for_chat_template(
        self, registry, mock_store
    ):
        """When primary repo lacks chat_template, try base_model from HF card."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        template = "{% if tools %}tools{% endif %}"
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write(template)
            downloaded_path = f.name

        # test/vlm fails, google/base-model fails, google/base-model-it succeeds
        mock_hf_mod = MagicMock()
        mock_hf_mod.hf_hub_download.side_effect = [
            Exception("404"),  # test/vlm
            Exception("404"),  # google/base-model
            downloaded_path,  # google/base-model-it
        ]
        mock_card_data = MagicMock()
        mock_card_data.base_model = "google/base-model"
        mock_info = MagicMock()
        mock_info.card_data = mock_card_data
        mock_hf_mod.model_info.return_value = mock_info

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules",
                {"mlx_vlm": mock_mlx_vlm, "huggingface_hub": mock_hf_mod},
            ):
                _, _, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True
        assert mock_tok.chat_template == template
        assert caps.supports_tools is True
        # Should have tried base_model-it
        mock_hf_mod.hf_hub_download.assert_any_call(
            "google/base-model-it", "chat_template.jinja"
        )

        Path(downloaded_path).unlink(missing_ok=True)

    def test_load_vlm_no_double_it_suffix_for_instruct_base(self, registry, mock_store):
        """When base_model already ends with -it, don't try base-model-it-it."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        template = "{% if tools %}tools{% endif %}"
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write(template)
            downloaded_path = f.name

        # test/vlm fails, google/gemma-4-27b-it fails → must NOT try -it-it
        mock_hf_mod = MagicMock()
        mock_hf_mod.hf_hub_download.side_effect = [
            Exception("404"),  # test/vlm
            Exception("404"),  # google/gemma-4-27b-it (base)
        ]
        mock_card_data = MagicMock()
        mock_card_data.base_model = "google/gemma-4-27b-it"
        mock_info = MagicMock()
        mock_info.card_data = mock_card_data
        mock_hf_mod.model_info.return_value = mock_info

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules",
                {"mlx_vlm": mock_mlx_vlm, "huggingface_hub": mock_hf_mod},
            ):
                _, _, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True
        # Must NOT have tried google/gemma-4-27b-it-it
        all_repos = [c[0][0] for c in mock_hf_mod.hf_hub_download.call_args_list]
        assert "google/gemma-4-27b-it-it" not in all_repos, (
            f"Should not try double -it suffix, but tried: {all_repos}"
        )

        Path(downloaded_path).unlink(missing_ok=True)

    def test_load_vlm_hub_download_fails_gracefully(self, registry, mock_store):
        """When all HF hub attempts fail, caps remain empty but loading succeeds."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        mock_hf_mod = MagicMock()
        mock_hf_mod.hf_hub_download.side_effect = Exception("network error")
        mock_hf_mod.model_info.side_effect = Exception("network error")

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules",
                {"mlx_vlm": mock_mlx_vlm, "huggingface_hub": mock_hf_mod},
            ):
                _, _, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True
        assert caps.supports_tools is False

    def test_load_text_fallback_to_vlm(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/path")
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.side_effect = ValueError("unsupported")
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
            ):
                model, tokenizer, is_vlm, caps, _ = manager._load_model("test/path")

        assert is_vlm is True

    def test_load_unknown_tries_mlx_lm_first(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/path")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="unknown"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                model, tokenizer, is_vlm, caps, _ = manager._load_model("test/path")

        assert is_vlm is False

    def test_load_unknown_fallback_to_vlm(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/path")
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="unknown"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.side_effect = ValueError("fail")
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
            ):
                model, tokenizer, is_vlm, caps, _ = manager._load_model("test/path")

        assert is_vlm is True

    def test_load_uses_local_path(self, registry, mock_store):
        """When model is already downloaded, load from local path, not HF repo ID."""
        manager = self._make_manager(registry, mock_store)
        # Create a fake downloaded model
        local_dir = mock_store.local_path("test/path")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                manager._load_model("test/path")

        # Should have been called with local path, not HF repo ID
        call_arg = mock_mlx_lm.load.call_args[0][0]
        assert call_arg == str(local_dir)

    def test_load_downloads_when_not_cached(self, registry, mock_store):
        """When model is not downloaded, download it first."""
        manager = self._make_manager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            with patch("huggingface_hub.snapshot_download") as mock_dl:
                with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                    manager._load_model("test/path")

        mock_dl.assert_called_once()
        assert mock_dl.call_args[1]["repo_id"] == "test/path"

    def test_load_keeps_partial_dir_on_download_failure(self, registry, mock_store):
        """If snapshot_download fails in _load_model, partial dir is kept for resume."""
        manager = self._make_manager(registry, mock_store)

        with patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("download failed"),
        ):
            with pytest.raises(Exception, match="download failed"):
                manager._load_model("test/path")

        # Dir kept for resume, marker stays so is_downloaded() returns False
        local_dir = mock_store.local_path("test/path")
        assert local_dir.exists()
        assert (local_dir / ".downloading").exists()
        assert not mock_store.is_downloaded("test/path")

    def test_load_removes_downloading_marker_on_success(self, registry, mock_store):
        """After successful download in _load_model, .downloading marker is gone."""
        manager = self._make_manager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            with patch("huggingface_hub.snapshot_download"):
                with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                    manager._load_model("test/path")

        local_dir = mock_store.local_path("test/path")
        assert not (local_dir / ".downloading").exists()

    def test_load_succeeds_when_marker_unlink_raises_oserror(
        self, registry, mock_store
    ):
        """If marker.unlink() raises a non-ENOENT OSError, _load_model still succeeds."""
        manager = self._make_manager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        original_unlink = Path.unlink

        def unlink_that_fails_on_downloading(self_path, **kwargs):
            if self_path.name == ".downloading":
                raise OSError("permission denied")
            return original_unlink(self_path, **kwargs)

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            with patch("huggingface_hub.snapshot_download"):
                with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                    with patch.object(Path, "unlink", unlink_that_fails_on_downloading):
                        model, tok, is_vlm, caps, _ = manager._load_model("test/path")

        assert model is mock_model

    def test_load_text_model_with_weight_quant(self, registry, mock_store):
        """When weight_quant is configured, quantize_model is called."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/path")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                with patch.object(manager, "_maybe_quantize_model") as mock_quant:
                    model, tokenizer, is_vlm, caps, _ = manager._load_model(
                        "test/path", weight_quant_str="hqq:4"
                    )

        mock_quant.assert_called_once_with(mock_model, False, "hqq:4", "test/path")

    def test_load_text_model_without_weight_quant(self, registry, mock_store):
        """When weight_quant is None, quantize_model is not called."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/path")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                with patch.object(manager, "_maybe_quantize_model") as mock_quant:
                    model, tokenizer, is_vlm, caps, _ = manager._load_model(
                        "test/path", weight_quant_str=None
                    )

        mock_quant.assert_called_once_with(mock_model, False, None, "test/path")


class TestFlashMoeVlmFallback:
    """Flash-MoE loading should fall back to mlx-vlm for unsupported model types."""

    def _make_manager(self, registry, mock_store):
        return ModelManager(registry, mock_store)

    def _pre_download(self, mock_store, hf_path):
        local_dir = mock_store.local_path(hf_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

    def _make_flash_moe_dir(self, mock_store, hf_path):
        flash_moe_dir = mock_store.local_path(hf_path) / "flash_moe"
        flash_moe_dir.mkdir(parents=True, exist_ok=True)
        moe_config = {
            "moe_layer_indices": [0, 1],
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        }
        (flash_moe_dir / "flash_moe_config.json").write_text(json.dumps(moe_config))
        (flash_moe_dir / "flash_moe_layout.json").write_text("{}")
        return flash_moe_dir

    def _mock_model_exp(self):
        exp = MagicMock()
        exp.kv_cache_quant = None
        return exp

    @staticmethod
    def _flash_moe_config():
        return FlashMoeConfig(enabled=True, cache_budget_experts=16, io_threads=4)

    def test_flash_moe_falls_back_to_vlm_on_unsupported_model_type(
        self, registry, mock_store
    ):
        """When mlx-lm can't load the model (e.g. gemma4), fall back to mlx-vlm."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/moe-vlm")
        flash_moe_dir = self._make_flash_moe_dir(mock_store, "test/moe-vlm")
        fm_config = self._flash_moe_config()

        mock_vlm_model = MagicMock()
        mock_vlm_model.language_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        mock_wrapped = MagicMock()

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = ValueError("Model type gemma4 not supported.")

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.load.return_value = (mock_vlm_model, mock_processor)

        with patch.dict(
            "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
        ):
            with patch(
                "olmlx.engine.model_manager._load_with_model_type_fallback",
                side_effect=ValueError("Model type gemma4 not supported."),
            ):
                with patch(
                    "olmlx.engine.flash.flash_moe_model.FlashMoeModelWrapper",
                    return_value=mock_wrapped,
                ):
                    with patch(
                        "olmlx.engine.flash.moe_weight_store.FlashMoeWeightStore"
                    ):
                        model, tokenizer, is_vlm, caps = manager._load_flash_moe_model(
                            "test/moe-vlm",
                            str(mock_store.local_path("test/moe-vlm")),
                            flash_moe_dir,
                            flash_moe_config=fm_config,
                        )

        assert is_vlm is True
        mock_mlx_vlm.load.assert_called_once()

    def test_flash_moe_uses_language_model_from_vlm(self, registry, mock_store):
        """VLM fallback should extract language_model for the MoE wrapper."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/moe-vlm2")
        flash_moe_dir = self._make_flash_moe_dir(mock_store, "test/moe-vlm2")
        fm_config = self._flash_moe_config()

        mock_language_model = MagicMock()
        mock_vlm_model = MagicMock()
        mock_vlm_model.language_model = mock_language_model
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.load.return_value = (mock_vlm_model, mock_processor)

        captured_model = {}

        def capture_wrapper(model, store, **kwargs):
            captured_model["model"] = model
            return MagicMock()

        with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
            with patch(
                "olmlx.engine.model_manager._load_with_model_type_fallback",
                side_effect=ValueError("Model type gemma4 not supported."),
            ):
                with patch(
                    "olmlx.engine.flash.flash_moe_model.FlashMoeModelWrapper",
                    side_effect=capture_wrapper,
                ):
                    with patch(
                        "olmlx.engine.flash.moe_weight_store.FlashMoeWeightStore"
                    ):
                        manager._load_flash_moe_model(
                            "test/moe-vlm2",
                            str(mock_store.local_path("test/moe-vlm2")),
                            flash_moe_dir,
                            flash_moe_config=fm_config,
                        )

        assert captured_model["model"] is mock_language_model

    def test_flash_moe_still_works_with_mlx_lm(self, registry, mock_store):
        """When mlx-lm succeeds, it should NOT fall back to mlx-vlm."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/moe-text")
        flash_moe_dir = self._make_flash_moe_dir(mock_store, "test/moe-text")
        fm_config = self._flash_moe_config()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch(
            "olmlx.engine.model_manager._load_with_model_type_fallback",
            return_value=(mock_model, mock_tokenizer),
        ):
            with patch(
                "olmlx.engine.flash.flash_moe_model.FlashMoeModelWrapper",
                return_value=MagicMock(),
            ):
                with patch("olmlx.engine.flash.moe_weight_store.FlashMoeWeightStore"):
                    model, tokenizer, is_vlm, caps = manager._load_flash_moe_model(
                        "test/moe-text",
                        str(mock_store.local_path("test/moe-text")),
                        flash_moe_dir,
                        flash_moe_config=fm_config,
                    )

        assert is_vlm is False


class TestModelLoadTimeout:
    """Test configurable timeout for model loading."""

    GB = 1024 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_timeout_fires_on_slow_load(self, registry, mock_store, monkeypatch):
        """When _load_model takes longer than the timeout, raise TimeoutError."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        def slow_load(hf_path, **kwargs):
            time.sleep(0.4)
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        with (
            patch.object(manager, "_load_model", side_effect=slow_load),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            with pytest.raises(ModelLoadTimeoutError, match="OLMLX_MODEL_LOAD_TIMEOUT"):
                await manager.ensure_loaded("qwen3")

    @pytest.mark.asyncio
    async def test_no_timeout_by_default(self, registry, mock_store, monkeypatch):
        """With default None timeout, fast loads succeed normally."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", None
        )
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, int(total_ram * 0.50)],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"

    @pytest.mark.asyncio
    async def test_timeout_allows_fast_loads(self, registry, mock_store, monkeypatch):
        """With a generous timeout, fast loads succeed."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 10.0
        )
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, int(total_ram * 0.50)],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"

    @pytest.mark.asyncio
    async def test_cleanup_on_timeout(self, registry, mock_store, monkeypatch):
        """On timeout, gc.collect and mx.clear_cache are called, model not in _loaded."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        def slow_load(hf_path, **kwargs):
            time.sleep(0.4)
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        with (
            patch.object(manager, "_load_model", side_effect=slow_load),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
            patch("olmlx.engine.model_manager.mx.synchronize") as mock_sync,
        ):
            with pytest.raises(TimeoutError):
                await manager.ensure_loaded("qwen3")

        # Only pre-load flush — the BaseException handler skips gc/clear
        # when a deferred cleanup is pending (background thread still running).
        assert mock_gc.call_count == 1
        assert mock_sync.call_count == 1
        assert mock_clear.call_count == 1
        assert "qwen3:latest" not in manager._loaded

    @pytest.mark.asyncio
    async def test_deferred_cleanup_after_timeout(
        self, registry, mock_store, monkeypatch
    ):
        """After timeout, a deferred task cleans up GPU memory when the thread finishes."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        def slow_load(hf_path, **kwargs):
            time.sleep(0.3)  # Short enough to finish during the test
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        with (
            patch.object(manager, "_load_model", side_effect=slow_load),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
            patch("olmlx.engine.model_manager.mx.synchronize") as mock_sync,
        ):
            with pytest.raises(TimeoutError):
                await manager.ensure_loaded("qwen3")

            # Only pre-load flush (BaseException handler skips when deferred)
            assert mock_gc.call_count == 1
            assert mock_sync.call_count == 1
            assert mock_clear.call_count == 1

            # Await the cleanup task directly (deterministic, no sleep needed)
            cleanup_task = manager._pending_cleanups.get("qwen3:latest")
            assert cleanup_task is not None
            await cleanup_task

            # Deferred cleanup adds one more call each
            assert mock_gc.call_count == 2
            assert mock_sync.call_count == 2
            assert mock_clear.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_after_timeout_waits_for_cleanup(
        self, registry, mock_store, monkeypatch
    ):
        """Retrying after timeout waits for deferred cleanup before starting new load."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        call_count = 0

        def slow_then_fast(hf_path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                time.sleep(0.3)  # First call triggers timeout
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        total_ram = 64 * self.GB

        with (
            patch.object(manager, "_load_model", side_effect=slow_then_fast),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, 1 * self.GB, int(total_ram * 0.50)],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            # First call times out
            with pytest.raises(TimeoutError):
                await manager.ensure_loaded("qwen3")

            # Pending cleanup should exist
            assert "qwen3:latest" in manager._pending_cleanups

            # Retry — should wait for cleanup to finish, then load fresh
            monkeypatch.setattr(
                "olmlx.engine.model_manager.settings.model_load_timeout", None
            )
            lm = await manager.ensure_loaded("qwen3")
            assert lm.name == "qwen3:latest"

            # Cleanup should be complete
            assert "qwen3:latest" not in manager._pending_cleanups

    @pytest.mark.asyncio
    async def test_cleanup_wait_does_not_block_other_models(
        self, registry, mock_store, monkeypatch
    ):
        """Retrying a timed-out model while another model loads concurrently.

        The cleanup wait for qwen3 must not hold the lock and block a
        concurrent ensure_loaded for llama3:8b.  Verified structurally
        by checking completion order (no timing dependency).
        """
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 2)
        manager = ModelManager(registry, mock_store)

        call_count = 0

        def slow_then_fast(hf_path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                time.sleep(0.5)  # First call: triggers timeout, cleanup takes 0.5s
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.50)

        results = []

        with (
            patch.object(manager, "_load_model", side_effect=slow_then_fast),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_before, mem_after, mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            # Timeout on qwen3 — orphaned thread runs for ~0.5s
            with pytest.raises(TimeoutError):
                await manager.ensure_loaded("qwen3")

            monkeypatch.setattr(
                "olmlx.engine.model_manager.settings.model_load_timeout", None
            )

            # Launch qwen3 retry (waits for cleanup) AND llama3 load
            # concurrently.  Track completion order.
            async def load_and_track(name):
                lm = await manager.ensure_loaded(name)
                results.append(lm.name)
                return lm

            qwen_task = asyncio.create_task(load_and_track("qwen3"))
            llama_task = asyncio.create_task(load_and_track("llama3:8b"))

            await asyncio.gather(qwen_task, llama_task)

            # llama3 should finish before qwen3 (which waits for cleanup).
            # If the lock were held during cleanup, qwen3 would block
            # llama3 and finish first.
            assert results.index("llama3:8b") < results.index("qwen3:latest"), (
                f"Expected llama3:8b to complete before qwen3:latest, got: {results}"
            )

    @pytest.mark.asyncio
    async def test_stale_cleanup_entry_on_gc_failure(
        self, registry, mock_store, monkeypatch
    ):
        """If gc.collect raises inside _cleanup, _pending_cleanups is still cleared."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        def slow_load(hf_path, **kwargs):
            time.sleep(0.3)
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        gc_call_count = 0

        def gc_collect_that_fails_second_time():
            nonlocal gc_call_count
            gc_call_count += 1
            if gc_call_count == 2:
                # Second call is inside _cleanup — simulate failure
                raise RuntimeError("gc failure")

        with (
            patch.object(manager, "_load_model", side_effect=slow_load),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch(
                "olmlx.engine.model_manager.gc.collect",
                side_effect=gc_collect_that_fails_second_time,
            ),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            with pytest.raises(ModelLoadTimeoutError):
                await manager.ensure_loaded("qwen3")

            # Await the cleanup task — it should fail but still clear the entry
            cleanup_task = manager._pending_cleanups.get("qwen3:latest")
            assert cleanup_task is not None
            # The task raises RuntimeError from gc.collect (which runs
            # first in the outer try), but the inner finally still pops
            # the entry regardless.
            with pytest.raises(RuntimeError, match="gc failure"):
                await cleanup_task

            # Key assertion: entry is cleared despite gc failure
            assert "qwen3:latest" not in manager._pending_cleanups

    @pytest.mark.asyncio
    async def test_stop_cancels_load_task(self, registry, mock_store, monkeypatch):
        """stop() cancels the underlying load_task when cancelling cleanup tasks."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        def very_slow_load(hf_path, **kwargs):
            time.sleep(10)  # Would run forever without cancellation
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        with (
            patch.object(manager, "_load_model", side_effect=very_slow_load),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            with pytest.raises(ModelLoadTimeoutError):
                await manager.ensure_loaded("qwen3")

            assert "qwen3:latest" in manager._pending_cleanups
            cleanup_task = manager._pending_cleanups["qwen3:latest"]

            # stop() should cancel cleanup without "exception never retrieved"
            await manager.stop()
            assert cleanup_task.cancelled()
            assert manager._pending_cleanups == {}

    @pytest.mark.asyncio
    async def test_raises_model_load_timeout_error(
        self, registry, mock_store, monkeypatch
    ):
        """Timeout raises ModelLoadTimeoutError (not plain TimeoutError)."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        def slow_load(hf_path, **kwargs):
            time.sleep(0.4)
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        with (
            patch.object(manager, "_load_model", side_effect=slow_load),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            with pytest.raises(ModelLoadTimeoutError):
                await manager.ensure_loaded("qwen3")

    @pytest.mark.asyncio
    async def test_memory_error_with_timeout_frees_load_task(
        self, registry, mock_store, monkeypatch
    ):
        """MemoryError after successful load with timeout frees model weights.

        When timeout is set, load_task holds the result tuple.  The except
        handler must del load_task before gc.collect so the Metal buffers
        are actually reclaimable.
        """
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 10.0
        )
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        # mem_after exceeds limit to trigger MemoryError
        mem_after = int(total_ram * 0.90)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
            patch("olmlx.engine.model_manager.mx.synchronize") as mock_sync,
        ):
            with pytest.raises(MemoryError):
                await manager.ensure_loaded("qwen3")

            # gc/clear should have been called for cleanup
            assert mock_gc.call_count >= 1
            assert mock_sync.call_count >= 1
            assert mock_clear.call_count >= 1
            assert "qwen3:latest" not in manager._loaded


class TestTryLmThenVlmFallback:
    """Test that _try_lm_then_vlm only falls back on expected exceptions."""

    def _make_manager(self, registry, mock_store):
        return ModelManager(registry, mock_store)

    def _pre_download(self, mock_store, hf_path):
        local_dir = mock_store.local_path(hf_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

    def test_fallback_on_value_error(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/path")
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = ValueError("unsupported model type")
        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.load.return_value = (MagicMock(), mock_processor)

        with patch.dict(
            "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
        ):
            _, _, is_vlm, _ = manager._try_lm_then_vlm("test/path", "test")
        assert is_vlm is True

    def test_fallback_on_key_error(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = KeyError("missing key")
        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.load.return_value = (MagicMock(), mock_processor)

        with patch.dict(
            "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
        ):
            _, _, is_vlm, _ = manager._try_lm_then_vlm("test/path", "test")
        assert is_vlm is True

    def test_no_fallback_on_import_error(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = ImportError("no module")

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            with pytest.raises(ImportError):
                manager._try_lm_then_vlm("test/path", "test")

    def test_no_fallback_on_runtime_error(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = RuntimeError("GPU error")

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            with pytest.raises(RuntimeError):
                manager._try_lm_then_vlm("test/path", "test")

    def test_no_fallback_on_memory_error(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = MemoryError("out of memory")

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            with pytest.raises(MemoryError):
                manager._try_lm_then_vlm("test/path", "test")

    def test_both_fail_with_file_not_found_raises_value_error(
        self, registry, mock_store
    ):
        manager = self._make_manager(registry, mock_store)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = ValueError("config.json not found")
        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.load.side_effect = FileNotFoundError(
            "Config not found at nonexistent-org/nonexistent-model"
        )

        with patch.dict(
            "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
        ):
            with pytest.raises(ValueError, match="not found"):
                manager._try_lm_then_vlm(
                    "nonexistent-org/nonexistent-model",
                    "nonexistent-org/nonexistent-model",
                )


class _FakeTokenizerWrapper:
    """Minimal stand-in for mlx-lm's TokenizerWrapper for stop-token tests."""

    def __init__(
        self,
        inner_eos: int | list[int] | None,
        stops: set[int] | None,
    ):
        inner = MagicMock()
        inner.eos_token_id = inner_eos
        self._tokenizer = inner
        self.eos_token_ids: set[int] | None = stops

    def add_eos_token(self, token: str) -> None:
        assert self.eos_token_ids is not None
        self.eos_token_ids.add(int(token))


class TestEnsureTokenizerEosInStops:
    """Issue #308: <|im_end|> leaks when config.json eos_token_id != template EOT."""

    def test_adds_inner_eos_when_missing(self):
        # Repro: Qwen2.5-Coder-1.5B has config.eos_token_id=151643 (<|endoftext|>)
        # but tokenizer_config eos_token=<|im_end|> (151645). The chat template
        # ends turns with 151645, so it must be in the stop set.
        tok = _FakeTokenizerWrapper(inner_eos=151645, stops={151643})
        _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids == {151643, 151645}

    def test_noop_when_already_present(self):
        tok = _FakeTokenizerWrapper(inner_eos=151645, stops={151645})
        _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids == {151645}

    def test_noop_when_inner_eos_missing(self, caplog):
        # None is legitimate (HF tokenizers without an EOS); must not warn.
        # Scope caplog to our logger to avoid spurious failures from unrelated
        # WARNING-level emissions (e.g. deprecation notices from pytest plugins).
        tok = _FakeTokenizerWrapper(inner_eos=None, stops={151643})
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids == {151643}
        assert not caplog.records, f"unexpected warnings: {caplog.records}"

    def test_adds_list_inner_eos(self):
        # Defensive: HF stock tokenizers expose eos_token_id as a single int,
        # but custom trust_remote_code=True tokenizers may surface list[int].
        tok = _FakeTokenizerWrapper(inner_eos=[151645, 151643], stops={151643})
        _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids == {151643, 151645}

    def test_noop_when_stops_not_a_set(self, caplog):
        # A wrapper with add_eos_token but eos_token_ids=None must not crash;
        # the guard returns early so the stop set is left unchanged. Should
        # also DEBUG-log so a future mlx-lm change of the stop-set type is
        # discoverable rather than silently disabling the workaround.
        tok = _FakeTokenizerWrapper(inner_eos=151645, stops=None)
        with caplog.at_level(logging.DEBUG, logger="olmlx.engine.model_manager"):
            _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids is None
        assert any("not set" in r.message for r in caplog.records)

    def test_debug_logs_when_tokenizer_attr_missing(self, caplog):
        # Past the add_eos_token gate but no _tokenizer attribute — mlx-lm
        # likely renamed the field. Must log at DEBUG so the regression is
        # discoverable without spamming WARNING for deliberate variants.
        tok = _FakeTokenizerWrapper(inner_eos=151645, stops={151643})
        del tok._tokenizer
        with caplog.at_level(logging.DEBUG, logger="olmlx.engine.model_manager"):
            _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids == {151643}
        assert any("not accessible" in r.message for r in caplog.records)

    def test_warns_on_unexpected_type(self, caplog):
        # Defends against a refactor that accidentally skips the warning
        # branch or promotes/demotes its log level.
        tok = _FakeTokenizerWrapper(inner_eos=None, stops={151643})
        tok._tokenizer.eos_token_id = 3.14  # float: not int, list, or None
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids == {151643}  # unchanged
        assert any("unexpected type" in r.message for r in caplog.records)

    def test_noop_on_non_wrapper(self):
        # mlx-vlm processors / plain HF tokenizers don't expose add_eos_token.
        processor = MagicMock(spec=["tokenizer", "eos_token_id"])
        # Calling on a non-wrapper must not raise.
        _ensure_tokenizer_eos_in_stops(processor)


class TestLoadWithModelTypeFallbackEosFix:
    """Issue #308: _load_with_model_type_fallback must augment stop tokens."""

    def test_main_path_augments_stops(self, tmp_path):
        from olmlx.engine.model_manager import _load_with_model_type_fallback

        (tmp_path / "config.json").write_text("{}")

        tok = _FakeTokenizerWrapper(inner_eos=151645, stops={151643})
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (MagicMock(), tok)

        _, returned = _load_with_model_type_fallback(mock_mlx_lm, str(tmp_path))
        assert returned is tok
        assert 151645 in tok.eos_token_ids
        assert 151643 in tok.eos_token_ids

    def test_flash_strict_fallback_path_augments_stops(self, tmp_path):
        # When mlx_lm.load raises "parameters not in model", flash/prepare's
        # strict-fallback branch reloads via load_model + load_tokenizer; the
        # EOS augmentation must still run after that branch.
        from olmlx.engine.flash.prepare import (
            _STRICT_LOAD_ERROR,
            load_model_with_strict_fallback,
        )

        tok = _FakeTokenizerWrapper(inner_eos=151645, stops={151643})
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = ValueError(_STRICT_LOAD_ERROR)
        mock_mlx_lm.utils.load_model.return_value = (
            MagicMock(),
            {"eos_token_id": 151643},
        )
        mock_mlx_lm.utils.load_tokenizer.return_value = tok

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            _, returned = load_model_with_strict_fallback(str(tmp_path), lazy=False)
        assert returned is tok
        assert 151645 in tok.eos_token_ids

    def test_flash_load_path_augments_stops(self, tmp_path):
        # Flash mode bypasses _load_with_model_type_fallback and calls
        # mlx_lm.load() directly via load_model_with_strict_fallback. That
        # path must also apply the EOS workaround — otherwise Flash-mode
        # Qwen2.5-Coder-1.5B-Instruct would still leak <|im_end|>.
        from olmlx.engine.flash.prepare import load_model_with_strict_fallback

        tok = _FakeTokenizerWrapper(inner_eos=151645, stops={151643})
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (MagicMock(), tok)

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            _, returned = load_model_with_strict_fallback(str(tmp_path), lazy=False)
        assert returned is tok
        assert 151645 in tok.eos_token_ids
        assert 151643 in tok.eos_token_ids

    def test_fallback_path_augments_stops(self, tmp_path):
        # Exercises the model_type-remapping branch: mlx_lm.load raises, then
        # load_model + load_tokenizer are called with the stripped model_type.
        # Same EOS mismatch scenario as the main path must still be repaired.
        # We patch transformers' CONFIG_MAPPING so the test owns its
        # precondition — without the patch a future transformers release
        # dropping the chosen model_type would silently re-raise instead of
        # exercising the fallback.
        import transformers.models.auto.configuration_auto as auto_cfg

        from olmlx.engine.model_manager import _load_with_model_type_fallback

        original_cfg = {"model_type": "fakemodel2", "eos_token_id": 151643}
        (tmp_path / "config.json").write_text(json.dumps(original_cfg))

        tok = _FakeTokenizerWrapper(inner_eos=151645, stops={151643})
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = ValueError("unsupported model_type")
        mock_mlx_lm.utils.load_model.return_value = (
            MagicMock(),
            {"eos_token_id": 151643},
        )
        mock_mlx_lm.utils.load_tokenizer.return_value = tok

        # "fakemodel2" → strips last digit → "fakemodel" via the regex
        # ``re.sub(r"\d+$", lambda m: m.group()[:-1], ...)`` in
        # _load_with_model_type_fallback. CONFIG_MAPPING must contain
        # "fakemodel" for the fallback branch to proceed.
        with patch.object(auto_cfg, "CONFIG_MAPPING", {"fakemodel": object()}):
            _, returned = _load_with_model_type_fallback(mock_mlx_lm, str(tmp_path))
        assert returned is tok
        assert 151645 in tok.eos_token_ids
        assert 151643 in tok.eos_token_ids
        # config.json must be restored after the temporary remap.
        assert json.loads((tmp_path / "config.json").read_text()) == original_cfg


class TestFlashLoadFailureCleanup:
    """A flash load that fails after the FlashWeightStore is constructed must
    close it (and the wrapper's prefetcher) — otherwise every failed attempt
    leaks per-layer fds + the store/prefetcher thread pools (#624)."""

    def test_wrapper_build_failure_closes_weight_store(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        import olmlx.engine.flash.flash_model as fm
        import olmlx.engine.flash.predictor as fp
        import olmlx.engine.flash.prepare as fprep
        import olmlx.engine.flash.weight_store as ws
        import olmlx.engine.model_manager as mm

        closed = {"store": 0}

        class _SpyStore:
            def __init__(self, *args, **kwargs):
                pass

            def close(self):
                closed["store"] += 1

        monkeypatch.setattr(ws, "FlashWeightStore", _SpyStore)
        monkeypatch.setattr(
            fp.PredictorBank, "load", classmethod(lambda cls, path: object())
        )
        monkeypatch.setattr(
            fprep,
            "load_model_with_strict_fallback",
            lambda path, *, lazy: (object(), object()),
        )
        monkeypatch.setattr(mm, "detect_caps", lambda tok: TemplateCaps())

        def _boom(*args, **kwargs):
            raise RuntimeError("wrapper build failed")

        monkeypatch.setattr(fm, "FlashModelWrapper", _boom)

        flash_dir = tmp_path / "flash"
        flash_dir.mkdir()
        (flash_dir / "flash_layout.json").write_text(
            json.dumps(
                {
                    "hidden_size": 16,
                    "intermediate_size": 8,
                    "num_layers": 1,
                    "layers": {},
                }
            )
        )

        from olmlx.config import ExperimentalSettings

        model_exp = ExperimentalSettings(_env_file=None)
        flash_config = SimpleNamespace(
            sparsity_threshold=0.5,
            min_active_neurons=0,
            max_active_neurons=0,
            memory_budget_fraction=0.9,
            prefetch=False,
            flash_speculative=False,
            flash_speculative_draft_model=None,
            flash_speculative_tokens=4,
        )

        manager = ModelManager(MagicMock(), MagicMock())
        with pytest.raises(RuntimeError, match="wrapper build failed"):
            manager._load_flash_model(
                "hf/model",
                str(tmp_path),
                flash_dir,
                model_exp=model_exp,
                flash_config=flash_config,
            )
        assert closed["store"] == 1, "weight store must be closed on failed load"

    def test_speculative_vocab_mismatch_releases_draft_and_store(
        self, tmp_path, monkeypatch
    ):
        """A speculative-flash failure *after* the draft model loads (vocab
        mismatch) must close the store and drop the draft model's GPU weights
        back to the pool — otherwise the draft leaks until GC (#624)."""
        from types import SimpleNamespace

        import olmlx.engine.flash.flash_model as fm
        import olmlx.engine.flash.predictor as fp
        import olmlx.engine.flash.prepare as fprep
        import olmlx.engine.flash.weight_store as ws
        import olmlx.engine.model_manager as mm

        closed = {"store": 0, "clear_cache": 0}

        class _SpyStore:
            def __init__(self, *args, **kwargs):
                pass

            def close(self):
                closed["store"] += 1

        monkeypatch.setattr(ws, "FlashWeightStore", _SpyStore)
        monkeypatch.setattr(
            fp.PredictorBank, "load", classmethod(lambda cls, path: object())
        )

        def _fake_load(path, *, lazy):
            if path == "draft/x":
                return SimpleNamespace(args=SimpleNamespace(vocab_size=200)), object()
            return object(), object()

        monkeypatch.setattr(fprep, "load_model_with_strict_fallback", _fake_load)
        monkeypatch.setattr(mm, "detect_caps", lambda tok: TemplateCaps())
        # Wrapper succeeds with a vocab that mismatches the draft.
        monkeypatch.setattr(
            fm,
            "FlashModelWrapper",
            lambda *a, **k: SimpleNamespace(
                args=SimpleNamespace(vocab_size=100), prefetcher=None
            ),
        )
        monkeypatch.setattr(
            mm.mx,
            "clear_cache",
            lambda: closed.__setitem__("clear_cache", closed["clear_cache"] + 1),
        )

        flash_dir = tmp_path / "flash"
        flash_dir.mkdir()
        (flash_dir / "flash_layout.json").write_text(
            json.dumps(
                {
                    "hidden_size": 16,
                    "intermediate_size": 8,
                    "num_layers": 1,
                    "layers": {},
                }
            )
        )

        from olmlx.config import ExperimentalSettings

        model_exp = ExperimentalSettings(_env_file=None)
        flash_config = SimpleNamespace(
            sparsity_threshold=0.5,
            min_active_neurons=0,
            max_active_neurons=0,
            memory_budget_fraction=0.9,
            prefetch=False,
            flash_speculative=True,
            flash_speculative_draft_model="draft/x",
            flash_speculative_tokens=4,
        )

        manager = ModelManager(MagicMock(), MagicMock())
        with pytest.raises(ValueError, match="vocab_size"):
            manager._load_flash_model(
                "hf/model",
                str(tmp_path),
                flash_dir,
                model_exp=model_exp,
                flash_config=flash_config,
            )
        assert closed["store"] == 1, "weight store must be closed on failed load"
        assert closed["clear_cache"] >= 1, "draft weights must be released to the pool"


class TestExpiryChecker:
    @pytest.mark.asyncio
    async def test_expired_models_removed(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        lm = LoadedModel(
            name="expired:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=time.time() - 10,  # already expired
        )
        manager._loaded["expired:latest"] = lm

        await manager._expire_stale()

        assert "expired:latest" not in manager._loaded

    @pytest.mark.asyncio
    async def test_non_expired_models_kept(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        lm = LoadedModel(
            name="active:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=time.time() + 1000,
        )
        manager._loaded["active:latest"] = lm

        await manager._expire_stale()

        assert "active:latest" in manager._loaded

    @pytest.mark.asyncio
    async def test_expiry_skips_model_with_active_refs(self, registry, mock_store):
        """Models with active_refs > 0 must not be expired."""
        manager = ModelManager(registry, mock_store)
        lm = LoadedModel(
            name="busy:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=time.time() - 10,  # expired
            active_refs=1,  # but actively in use
        )
        manager._loaded["busy:latest"] = lm

        await manager._expire_stale()

        assert "busy:latest" in manager._loaded

    @pytest.mark.asyncio
    async def test_expire_stale_isolates_per_model_failures(
        self, registry, mock_store, caplog
    ):
        """A failing close() on model A must not skip models B and C.

        Without per-model isolation, a single broken prefetcher would block
        every other expired model in the same cycle from being cleaned up.
        """
        manager = ModelManager(registry, mock_store)

        def _flash_lm(name: str, *, raises: bool = False):
            prefetcher = MagicMock()
            if raises:
                prefetcher.close.side_effect = RuntimeError(f"{name} boom")
            weight_store = MagicMock()
            model = MagicMock()
            model.prefetcher = prefetcher
            lm = LoadedModel(
                name=name,
                hf_path=f"x/{name}",
                model=model,
                tokenizer=MagicMock(),
                weight_store=weight_store,
                is_flash=True,
                expires_at=time.time() - 10,
            )
            return lm, prefetcher, weight_store

        a, _, ws_a = _flash_lm("a", raises=True)
        b, _, ws_b = _flash_lm("b")
        c, _, ws_c = _flash_lm("c")
        manager._loaded["a"] = a
        manager._loaded["b"] = b
        manager._loaded["c"] = c

        with caplog.at_level(logging.ERROR, logger="olmlx.engine.model_manager"):
            await manager._expire_stale()  # must not raise

        assert "a" not in manager._loaded
        assert "b" not in manager._loaded
        assert "c" not in manager._loaded
        # Sibling weight stores must have been closed despite A's failure.
        ws_a.close.assert_called_once()
        ws_b.close.assert_called_once()
        ws_c.close.assert_called_once()
        # _close_loaded_model logs per-resource; A's prefetcher failure
        # surfaces as "Error closing prefetcher for a".
        assert any(
            "Error closing prefetcher for a" in r.message for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_expire_stale_drops_refs_before_gc(
        self, registry, mock_store, monkeypatch
    ):
        """The expired-models list must be dropped before gc.collect().

        Otherwise gc.collect() can't reclaim the Metal buffers referenced
        by the LoadedModel objects, and the mx.clear_cache() that was
        specifically added to flush expired-model memory is effectively
        a no-op. Mirrors the ``del evicted`` pattern in
        _evict_lru_if_needed.

        Uses a weakref to assert the LoadedModel is unreachable at the
        moment gc.collect() runs — proving expired_lms was dropped.

        Assumes CPython refcount semantics: an object with refcount 0 is
        deallocated immediately, so the weakref resolves to None as soon
        as the last strong reference goes away. On a non-refcounting
        runtime (PyPy, Jython) a back-reference cycle introduced by
        MagicMock could keep the LM alive — but we also monkeypatch
        gc.collect here, so the cycle collector would not run to clean
        it up. The codebase is CPython-only (uv-managed cpython-3.11),
        so this is fine.
        """
        import weakref

        manager = ModelManager(registry, mock_store)
        weakref_alive_at_gc: list[bool] = []
        ref_holder: dict[str, Any] = {}

        def _fake_gc():
            weakref_alive_at_gc.append(ref_holder["wr"]() is not None)

        monkeypatch.setattr("olmlx.engine.model_manager.gc.collect", _fake_gc)
        monkeypatch.setattr("olmlx.engine.model_manager.mx.clear_cache", lambda: None)

        lm = LoadedModel(
            name="expired:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=time.time() - 10,
        )
        manager._loaded["expired:latest"] = lm
        ref_holder["wr"] = weakref.ref(lm)
        del lm  # only manager._loaded holds it now

        await manager._expire_stale()

        # If expired_lms was still alive at gc time, the weakref would
        # resolve to a live object. The fix asserts it's dead.
        assert weakref_alive_at_gc == [False]

    @pytest.mark.asyncio
    async def test_expire_stale_offloads_close_to_thread(
        self, registry, mock_store, monkeypatch
    ):
        """Close runs off the event loop.

        ``executor.shutdown(wait=True)`` is synchronous. Running it on
        the event loop thread would stall every concurrent coroutine
        until the pools drained, even with the lock released. The fix
        is ``await asyncio.to_thread(self._close_loaded_model, lm)``.
        This test asserts the call went through ``asyncio.to_thread``.
        """
        manager = ModelManager(registry, mock_store)
        original_to_thread = asyncio.to_thread
        to_thread_calls: list[Any] = []

        async def _tracking_to_thread(fn, *args, **kwargs):
            to_thread_calls.append(fn)
            return await original_to_thread(fn, *args, **kwargs)

        monkeypatch.setattr(
            "olmlx.engine.model_manager.asyncio.to_thread", _tracking_to_thread
        )

        lm = LoadedModel(
            name="expired:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=time.time() - 10,
        )
        manager._loaded["expired:latest"] = lm

        await manager._expire_stale()

        # The close was routed through to_thread (off-event-loop).
        assert manager._close_loaded_model in to_thread_calls

    @pytest.mark.asyncio
    async def test_expire_stale_releases_lock_before_closing(
        self, registry, mock_store
    ):
        """_close_loaded_model must run outside self._lock.

        ``executor.shutdown(wait=True)`` is synchronous and can take long
        enough to be noticeable. Holding ``self._lock`` during that would
        stall every concurrent ``ensure_loaded()`` caller until the pool
        drained — a latency spike on a server doing real inference when
        a keep-alive happens to expire.
        """
        manager = ModelManager(registry, mock_store)
        lock_held_during_close: list[bool] = []

        def _record_lock_state(_lm):
            lock_held_during_close.append(manager._lock.locked())

        manager._close_loaded_model = _record_lock_state  # type: ignore[assignment]
        lm = LoadedModel(
            name="expired:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=time.time() - 10,
        )
        manager._loaded["expired:latest"] = lm

        await manager._expire_stale()

        assert lock_held_during_close == [False]

    @pytest.mark.asyncio
    async def test_check_expiry_loop_survives_unhandled_error(
        self, registry, mock_store, caplog, monkeypatch
    ):
        """The background expiry task must survive a raising _expire_stale.

        If _expire_stale ever propagates, the unguarded `while True` in
        _check_expiry_loop exits permanently — no log, no restart, models
        accumulate forever. Defense in depth on top of per-model isolation.
        """
        manager = ModelManager(registry, mock_store)
        sleep_calls = {"n": 0}

        async def _fake_sleep(_seconds):
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 2:
                raise asyncio.CancelledError()

        monkeypatch.setattr("olmlx.engine.model_manager.asyncio.sleep", _fake_sleep)
        call_count = {"n": 0}

        async def _raising_expire():
            call_count["n"] += 1
            raise RuntimeError("simulated failure")

        manager._expire_stale = _raising_expire  # type: ignore[method-assign]

        with caplog.at_level(logging.ERROR, logger="olmlx.engine.model_manager"):
            with pytest.raises(asyncio.CancelledError):
                await manager._check_expiry_loop()

        # First iteration raised → loop continued → second iteration cancelled.
        assert call_count["n"] == 1
        assert any("Expiry check failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_expire_stale_closes_flash_resources(self, registry, mock_store):
        """_expire_stale must close prefetcher + weight_store on a Flash model.

        Otherwise the keep-alive timer leaks ThreadPoolExecutor workers and
        per-layer file descriptors for every expired Flash model (issue #178).
        """
        manager = ModelManager(registry, mock_store)
        # Wire both prefetcher and weight_store through the same ``parent``
        # MagicMock so their .close() calls are recorded in a single ordered
        # mock_calls list. _close_loaded_model accesses prefetcher via
        # ``lm.model.prefetcher`` and weight_store via ``lm.weight_store``;
        # both end up resolving to attributes on ``parent`` here, which is
        # what makes the cross-resource ordering assertion work.
        parent = MagicMock()
        prefetcher = parent.prefetcher
        weight_store = parent.weight_store
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        lm = LoadedModel(
            name="expired:latest",
            hf_path="test/model",
            model=flash_model,
            tokenizer=MagicMock(),
            weight_store=weight_store,
            is_flash=True,
            expires_at=time.time() - 10,
        )
        manager._loaded["expired:latest"] = lm

        await manager._expire_stale()

        assert "expired:latest" not in manager._loaded
        prefetcher.close.assert_called_once()
        weight_store.close.assert_called_once()
        call_names = [c[0] for c in parent.mock_calls]
        assert call_names.index("prefetcher.close") < call_names.index(
            "weight_store.close"
        )

    @pytest.mark.asyncio
    async def test_expire_stale_skips_active(self, registry, mock_store):
        """Models with active_refs > 0 must not be expired or closed."""
        manager = ModelManager(registry, mock_store)
        prefetcher = MagicMock()
        weight_store = MagicMock()
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        lm = LoadedModel(
            name="busy:latest",
            hf_path="test/model",
            model=flash_model,
            tokenizer=MagicMock(),
            weight_store=weight_store,
            is_flash=True,
            expires_at=time.time() - 10,
            active_refs=1,
        )
        manager._loaded["busy:latest"] = lm

        await manager._expire_stale()

        assert "busy:latest" in manager._loaded
        prefetcher.close.assert_not_called()
        weight_store.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_expired_model_object_still_usable(self, registry, mock_store):
        """Even if a model is removed from _loaded, the object remains usable."""
        manager = ModelManager(registry, mock_store)
        model_mock = MagicMock()
        tokenizer_mock = MagicMock()
        lm = LoadedModel(
            name="removed:latest",
            hf_path="test/model",
            model=model_mock,
            tokenizer=tokenizer_mock,
            expires_at=time.time() - 10,
        )
        manager._loaded["removed:latest"] = lm

        # Hold a reference, then expire it
        held_ref = lm
        now = time.time()
        expired = [
            name
            for name, m in manager._loaded.items()
            if m.expires_at is not None and m.expires_at <= now and m.active_refs == 0
        ]
        for name in expired:
            del manager._loaded[name]

        # Model object still accessible via held reference
        assert "removed:latest" not in manager._loaded
        assert held_ref.model is model_mock
        assert held_ref.tokenizer is tokenizer_mock


class TestMemoryCheck:
    """Test that models exceeding the memory limit are rejected on load."""

    GB = 1024 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_model_exceeding_memory_limit_raises(
        self, registry, mock_store, monkeypatch
    ):
        """When Metal memory after loading exceeds the limit, raise MemoryError."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        # Before load: 1 GB baseline, after load: 80% of RAM (exceeds 75% limit)
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.80)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            with pytest.raises(MemoryError, match="memory limit"):
                await manager.ensure_loaded("qwen3")

        # Model should NOT be in _loaded
        assert "qwen3:latest" not in manager._loaded

    @pytest.mark.asyncio
    async def test_model_within_memory_limit_loads(
        self, registry, mock_store, monkeypatch
    ):
        """When Metal memory is within the limit, the model loads normally."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.50)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"
        assert "qwen3:latest" in manager._loaded

    @pytest.mark.asyncio
    async def test_custom_memory_limit_fraction(
        self, registry, mock_store, monkeypatch
    ):
        """Configurable memory_limit_fraction is respected."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.memory_limit_fraction", 0.90
        )
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        # 80% usage after load — below the 90% custom limit, should load fine
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.80)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"

    @pytest.mark.asyncio
    async def test_inference_headroom_fraction_tightens_admission(
        self, registry, mock_store, monkeypatch
    ):
        """inference_headroom_fraction reserves room below memory_limit_fraction.

        A model whose weights land at 50% of RAM loads fine with no headroom
        (50% < 75% limit), but must be rejected when 30% is reserved for the
        KV cache / activations (effective limit 45% < 50%).  Issue #223: a
        model that passes the static weights-only check can still swap during
        decode because the KV cache allocates on top of the weights.
        """
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.memory_limit_fraction", 0.75
        )
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.inference_headroom_fraction",
            0.30,
            raising=False,
        )
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.50)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            with pytest.raises(MemoryError, match="memory limit"):
                await manager.ensure_loaded("qwen3")

        assert "qwen3:latest" not in manager._loaded

    @pytest.mark.asyncio
    async def test_ensure_loaded_pin_increments_active_refs(self, registry, mock_store):
        """ensure_loaded(pin=True) returns the model already pinned (refs==1)."""
        manager = ModelManager(registry, mock_store)
        total_ram = 64 * self.GB
        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(MagicMock(), MagicMock(), False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, int(total_ram * 0.40)],
            ),
            patch("olmlx.utils.memory.get_system_memory_bytes", return_value=total_ram),
            patch("olmlx.utils.memory.is_memory_pressure_high", return_value=False),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            lm = await manager.ensure_loaded("qwen3", pin=True)
        assert lm.active_refs == 1
        # Re-ensuring the already-loaded model with pin=True pins again.
        with patch("olmlx.utils.memory.is_memory_pressure_high", return_value=False):
            lm2 = await manager.ensure_loaded("qwen3", pin=True)
        assert lm2 is lm
        assert lm.active_refs == 2

    @pytest.mark.asyncio
    async def test_ensure_loaded_default_does_not_pin(self, registry, mock_store):
        """Default ensure_loaded (pin=False) leaves active_refs at 0."""
        manager = ModelManager(registry, mock_store)
        total_ram = 64 * self.GB
        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(MagicMock(), MagicMock(), False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, int(total_ram * 0.40)],
            ),
            patch("olmlx.utils.memory.get_system_memory_bytes", return_value=total_ram),
            patch("olmlx.utils.memory.is_memory_pressure_high", return_value=False),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            lm = await manager.ensure_loaded("qwen3")
        assert lm.active_refs == 0

    @pytest.mark.asyncio
    async def test_inference_headroom_default_does_not_reject(
        self, registry, mock_store, monkeypatch
    ):
        """With the default headroom (0.0), the admission check is unchanged.

        A model at 50% of RAM loads fine under the 75% limit — the headroom
        knob is opt-in and must not narrow the limit when left at its default.
        """
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.memory_limit_fraction", 0.75
        )
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.50)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"

    @pytest.mark.asyncio
    async def test_memory_error_message_includes_guidance(self, registry, mock_store):
        """The error message should include actionable guidance."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.80)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            with pytest.raises(MemoryError) as exc_info:
                await manager.ensure_loaded("qwen3")

        msg = str(exc_info.value)
        assert "OLMLX_MEMORY_LIMIT_FRACTION" in msg
        assert "smaller" in msg or "quantized" in msg

    @pytest.mark.asyncio
    async def test_cleanup_called_on_rejection(self, registry, mock_store):
        """_flush_metal is called twice when a model is rejected:
        once pre-load and once post-rejection cleanup."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.80)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch.object(manager, "_flush_metal", new_callable=AsyncMock) as mock_flush,
        ):
            with pytest.raises(MemoryError):
                await manager.ensure_loaded("qwen3")

        # Called twice: once for pre-load cache flush, once for post-rejection cleanup
        assert mock_flush.await_count == 2

    @pytest.mark.asyncio
    async def test_cache_flushed_after_eviction(
        self, registry, mock_store, monkeypatch
    ):
        """After LRU eviction, Metal cache is flushed before measuring mem_before."""
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        # Pre-load a model
        existing = LoadedModel(
            name="old:latest",
            hf_path="org/old",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        manager._loaded["old:latest"] = existing

        total_ram = 64 * self.GB
        # After cache flush + load, 50% usage — well within 75% limit
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.50)

        call_order = []

        def track_gc():
            call_order.append("gc.collect")

        def track_clear():
            call_order.append("mx.clear_cache")

        def track_sync():
            call_order.append("mx.synchronize")

        def track_get_metal(*args):
            call_order.append("get_metal")
            return (
                mem_before
                if len([c for c in call_order if c == "get_metal"]) == 1
                else mem_after
            )

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=track_get_metal,
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect", side_effect=track_gc),
            patch("olmlx.engine.model_manager.mx.clear_cache", side_effect=track_clear),
            patch("olmlx.engine.model_manager.mx.synchronize", side_effect=track_sync),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"
        # Cache flush must happen before the first memory measurement
        gc_idx = call_order.index("gc.collect")
        sync_idx = call_order.index("mx.synchronize")
        clear_idx = call_order.index("mx.clear_cache")
        first_metal_idx = call_order.index("get_metal")
        assert gc_idx < first_metal_idx
        assert sync_idx < first_metal_idx
        assert clear_idx < first_metal_idx
        # Internal ordering: gc → synchronize → clear
        assert gc_idx < sync_idx < clear_idx

    @pytest.mark.asyncio
    async def test_model_mb_not_negative_in_error(self, registry, mock_store):
        """When MLX reuses cached buffers, model_mb should not be negative."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        # Simulate cache reuse: mem_after < mem_before but total still over limit
        mem_before = int(total_ram * 0.70)
        mem_after = int(total_ram * 0.80)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            with pytest.raises(MemoryError) as exc_info:
                await manager.ensure_loaded("qwen3")

        msg = str(exc_info.value)
        # Extract the MB number from "requires ~X MB"
        import re

        match = re.search(r"requires ~(\d+) MB", msg)
        assert match is not None
        assert int(match.group(1)) >= 0

    @pytest.mark.asyncio
    async def test_cleanup_on_unexpected_exception_after_load(
        self, registry, mock_store
    ):
        """If get_metal_memory raises after load, GPU cleanup must still run."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, OSError("Metal query failed")],
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
            patch("olmlx.engine.model_manager.mx.synchronize") as mock_sync,
        ):
            with pytest.raises(OSError, match="Metal query failed"):
                await manager.ensure_loaded("qwen3")

        # Cleanup must have been called: once pre-load flush + once post-failure
        assert mock_gc.call_count == 2
        assert mock_clear.call_count == 2
        assert mock_sync.call_count == 2
        # Model must NOT be in _loaded
        assert "qwen3:latest" not in manager._loaded

    @pytest.mark.asyncio
    async def test_load_succeeds_when_system_memory_unknown(self, registry, mock_store):
        """If get_system_memory_bytes returns 0, memory check is skipped and load succeeds."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, 2 * self.GB],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=0,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
        ):
            await manager.ensure_loaded("qwen3")

        assert "qwen3:latest" in manager._loaded

    @pytest.mark.asyncio
    async def test_cleanup_when_load_model_itself_fails(self, registry, mock_store):
        """If _load_model raises (e.g. partial GPU alloc then OOM), GPU cache must be flushed."""
        manager = ModelManager(registry, mock_store)

        with (
            patch.object(
                manager,
                "_load_model",
                side_effect=RuntimeError("Metal OOM during mlx_lm.load"),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=False,
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
            patch("olmlx.engine.model_manager.mx.synchronize") as mock_sync,
        ):
            with pytest.raises(RuntimeError, match="Metal OOM"):
                await manager.ensure_loaded("qwen3")

        # Pre-load flush + post-failure flush = 2 calls each
        assert mock_gc.call_count == 2
        assert mock_clear.call_count == 2
        assert mock_sync.call_count == 2
        assert "qwen3:latest" not in manager._loaded


class TestPreloadIdleModelEviction:
    """Issue #223: under sustained memory pressure, the pre-load hygiene
    pass must evict idle (active_refs == 0) resident models — not just their
    prompt caches — so loading a new model on top of resident weights does
    not push Metal into swap.  Only relevant when max_loaded_models > 1.
    """

    GB = 1024 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_evicts_idle_model_when_pressure_persists(
        self, registry, mock_store, monkeypatch
    ):
        """Prompt-cache flush isn't enough → evict the idle resident model."""
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 2)
        manager = ModelManager(registry, mock_store)

        # An idle model already resident, loaded earlier (clear LRU victim).
        resident = LoadedModel(
            name="llama3:8b",
            hf_path="mlx-community/Llama-3-8B-Instruct",
            model=MagicMock(),
            tokenizer=MagicMock(),
            loaded_at=time.time() - 100,
        )
        manager._loaded["llama3:8b"] = resident

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.50)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(MagicMock(), MagicMock(), False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            # Pressure persists through the prompt-cache flush, then clears
            # once the idle model is evicted: outer guard, loop entry, then
            # loop re-check / final check both report cleared.
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                side_effect=[True, True, False, False],
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"
        assert "qwen3:latest" in manager._loaded
        # The idle resident model must have been evicted to free its weights.
        assert "llama3:8b" not in manager._loaded

    @pytest.mark.asyncio
    async def test_does_not_evict_active_model_under_pressure(
        self, registry, mock_store, monkeypatch
    ):
        """A model serving requests (active_refs > 0) must not be evicted."""
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 2)
        manager = ModelManager(registry, mock_store)

        resident = LoadedModel(
            name="llama3:8b",
            hf_path="mlx-community/Llama-3-8B-Instruct",
            model=MagicMock(),
            tokenizer=MagicMock(),
            loaded_at=time.time() - 100,
        )
        resident.active_refs = 1  # in-flight request
        manager._loaded["llama3:8b"] = resident

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.50)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(MagicMock(), MagicMock(), False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            # Pressure stays high throughout — but the only resident model is
            # active, so eviction can't help; the load proceeds anyway.
            patch(
                "olmlx.utils.memory.is_memory_pressure_high",
                return_value=True,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"
        assert "qwen3:latest" in manager._loaded
        # Active model survived.
        assert "llama3:8b" in manager._loaded
        resident.active_refs = 0

    @pytest.mark.asyncio
    async def test_pressure_check_uses_effective_budget_fraction(
        self, registry, mock_store, monkeypatch
    ):
        """The pre-load pressure/eviction trigger must use the same effective
        budget (limit - headroom) as the admission check, not the raw limit.

        Otherwise, with headroom configured, an idle model can leave Metal
        below the raw limit but above the effective budget — the hygiene pass
        skips eviction and the load is then rejected even though evicting the
        idle model first would have made it fit.
        """
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.memory_limit_fraction", 0.75
        )
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.inference_headroom_fraction", 0.30
        )
        manager = ModelManager(registry, mock_store)

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.40)  # under effective 0.45 budget

        mock_pressure = MagicMock(return_value=False)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(MagicMock(), MagicMock(), False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.utils.memory.is_memory_pressure_high", mock_pressure),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
            patch("olmlx.engine.model_manager.mx.synchronize"),
        ):
            await manager.ensure_loaded("qwen3")

        assert mock_pressure.call_count >= 1
        # Every pre-load pressure check uses the effective budget (0.45),
        # not the raw memory_limit_fraction (0.75).
        for call in mock_pressure.call_args_list:
            assert call.args[0] == pytest.approx(0.45)


class TestEnsureLoadedNotFoundSuggestions:
    @pytest.mark.asyncio
    async def test_ensure_loaded_not_found_suggests_similar(self, registry, mock_store):
        """When model not found, error should include 'Did you mean' with suggestions."""
        manager = ModelManager(registry, mock_store)
        with pytest.raises(ValueError, match="Did you mean") as exc_info:
            await manager.ensure_loaded("qwem3")  # typo for qwen3
        # Should mention the similar model
        assert "qwen3:latest" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_not_found_does_not_evict_loaded_model(
        self, registry, mock_store, monkeypatch
    ):
        """Requesting a non-existent model must not evict already-loaded models."""
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)

        existing = LoadedModel(
            name="qwen3:latest",
            hf_path="Qwen/Qwen3-8B",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        manager._loaded["qwen3:latest"] = existing

        with pytest.raises(ValueError, match="not found"):
            await manager.ensure_loaded("claude-haiku-4-5-20251001")

        # The existing model must still be loaded
        assert "qwen3:latest" in manager._loaded


class TestPerModelConfig:
    @pytest.mark.asyncio
    async def test_kv_cache_quant_stored_on_loaded_model(self, mock_manager):
        """LoadedModel should have kv_cache_quant from per-model config."""
        lm = LoadedModel(
            name="test:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            kv_cache_quant="turboquant:4",
        )
        assert lm.kv_cache_quant == "turboquant:4"

    @pytest.mark.asyncio
    async def test_default_options_stored_on_loaded_model(self):
        """LoadedModel should have default_options from per-model config."""
        lm = LoadedModel(
            name="test:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            default_options={"temperature": 0.5, "num_predict": 1024},
        )
        assert lm.default_options == {"temperature": 0.5, "num_predict": 1024}

    @pytest.mark.asyncio
    async def test_default_options_empty_by_default(self):
        """LoadedModel default_options should be empty dict by default."""
        lm = LoadedModel(
            name="test:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        assert lm.default_options == {}
        assert lm.kv_cache_quant is None

    @pytest.mark.asyncio
    async def test_ensure_loaded_uses_model_config_keep_alive(
        self, tmp_path, monkeypatch
    ):
        """Per-model keep_alive is used when request doesn't specify one."""
        from olmlx.engine.registry import ModelRegistry
        from olmlx.models.store import ModelStore

        config = {
            "qwen3:latest": {
                "hf_path": "Qwen/Qwen3-8B-MLX",
                "keep_alive": "30m",
            }
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        monkeypatch.setattr(
            "olmlx.models.store.settings.models_dir", tmp_path / "models"
        )

        reg = ModelRegistry()
        reg.load()
        store = ModelStore(reg)
        manager = ModelManager(reg, store)

        # Pre-load a mock model to avoid actual loading
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(
            manager,
            "_load_model",
            return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
        ):
            lm = await manager.ensure_loaded("qwen3")  # no keep_alive specified

        # Should use per-model keep_alive of 30m = 1800s
        assert lm.expires_at is not None
        assert lm.expires_at >= time.time() + 1790  # ~30 minutes

    @pytest.mark.asyncio
    async def test_ensure_loaded_request_keep_alive_wins(self, tmp_path, monkeypatch):
        """Request keep_alive takes priority over per-model keep_alive."""
        from olmlx.engine.registry import ModelRegistry
        from olmlx.models.store import ModelStore

        config = {
            "qwen3:latest": {
                "hf_path": "Qwen/Qwen3-8B-MLX",
                "keep_alive": "30m",
            }
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        monkeypatch.setattr(
            "olmlx.models.store.settings.models_dir", tmp_path / "models"
        )

        reg = ModelRegistry()
        reg.load()
        store = ModelStore(reg)
        manager = ModelManager(reg, store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(
            manager,
            "_load_model",
            return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
        ):
            lm = await manager.ensure_loaded("qwen3", keep_alive="1m")

        # Should use request keep_alive of 1m = 60s, not model's 30m
        assert lm.expires_at is not None
        assert lm.expires_at < time.time() + 70  # ~1 minute

    def test_find_spectral_dir_returns_path_when_calibration_exists(self, tmp_path):
        from olmlx.engine.model_manager import ModelManager

        manager = ModelManager(MagicMock(), MagicMock())
        spectral_path = tmp_path / "spectral"
        spectral_path.mkdir()
        (spectral_path / "spectral_config.json").write_text("{}")
        mock_store = MagicMock()
        mock_store.local_path.return_value = tmp_path
        manager.store = mock_store

        result = manager._find_spectral_dir("test/model", "spectral:4")
        assert result == spectral_path

    def test_find_spectral_dir_returns_path_when_avg_bits_match(self, tmp_path):
        """Returns path when calibration exists with matching avg_bits."""
        import json as _json

        from olmlx.engine.model_manager import ModelManager

        manager = ModelManager(MagicMock(), MagicMock())
        spectral_path = tmp_path / "spectral"
        spectral_path.mkdir()
        (spectral_path / "spectral_config.json").write_text(
            _json.dumps({"meta": {"avg_bits": 4}})
        )
        mock_store = MagicMock()
        mock_store.local_path.return_value = tmp_path
        manager.store = mock_store

        result = manager._find_spectral_dir("test/model", "spectral:4")
        assert result == spectral_path

    def test_find_spectral_dir_returns_none_for_turboquant(self):
        from olmlx.engine.model_manager import ModelManager

        manager = ModelManager(MagicMock(), MagicMock())
        manager.store = MagicMock()

        result = manager._find_spectral_dir("test/model", "turboquant:4")
        assert result is None

    def test_find_spectral_dir_returns_none_when_none(self):
        from olmlx.engine.model_manager import ModelManager

        manager = ModelManager(MagicMock(), MagicMock())
        manager.store = MagicMock()

        result = manager._find_spectral_dir("test/model", None)
        assert result is None

    def test_find_spectral_dir_raises_when_calibration_missing(
        self, tmp_path, monkeypatch
    ):
        from olmlx.engine.model_manager import (
            ModelManager,
            SpectralCalibrationMissingError,
        )

        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.kv_cache_auto_calibrate", False
        )
        manager = ModelManager(MagicMock(), MagicMock())
        spectral_path = tmp_path / "spectral"
        spectral_path.mkdir()
        mock_store = MagicMock()
        mock_store.local_path.return_value = tmp_path
        manager.store = mock_store

        with pytest.raises(SpectralCalibrationMissingError) as exc_info:
            manager._find_spectral_dir("test/model", "spectral:4")
        msg = str(exc_info.value)
        assert "olmlx spectral prepare" in msg
        assert "test/model" in msg
        assert "C4" in msg
        assert "256" in msg
        assert "8192" in msg
        assert "--avg-bits 2" in msg
        assert "olmlx spectral prepare test/model" in msg
        assert str(spectral_path) in msg

    def test_find_spectral_dir_raises_when_dir_missing(self, tmp_path, monkeypatch):
        from olmlx.engine.model_manager import (
            ModelManager,
            SpectralCalibrationMissingError,
        )

        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.kv_cache_auto_calibrate", False
        )
        manager = ModelManager(MagicMock(), MagicMock())
        mock_store = MagicMock()
        mock_store.local_path.return_value = tmp_path
        manager.store = mock_store

        with pytest.raises(SpectralCalibrationMissingError) as exc_info:
            manager._find_spectral_dir("test/model", "spectral:2")
        msg = str(exc_info.value)
        assert "olmlx spectral prepare" in msg
        assert "test/model" in msg
        assert "--avg-bits 2" in msg
        assert "--avg-bits 4" in msg

    def test_find_spectral_dir_raises_valueerror_for_malformed_config(self):
        from olmlx.engine.model_manager import ModelManager

        manager = ModelManager(MagicMock(), MagicMock())
        manager.store = MagicMock()

        with pytest.raises(ValueError, match="spectral:abc"):
            manager._find_spectral_dir("test/model", "spectral:abc")

    def test_find_spectral_dir_raises_valueerror_for_missing_colon(self):
        from olmlx.engine.model_manager import ModelManager

        manager = ModelManager(MagicMock(), MagicMock())
        manager.store = MagicMock()

        with pytest.raises(ValueError, match="spectral:"):
            manager._find_spectral_dir("test/model", "spectral:")

    def test_find_spectral_dir_raises_valueerror_for_bad_bit_width(self):
        from olmlx.engine.model_manager import ModelManager

        manager = ModelManager(MagicMock(), MagicMock())
        manager.store = MagicMock()

        with pytest.raises(ValueError, match="spectral:3"):
            manager._find_spectral_dir("test/model", "spectral:3")

    def test_find_spectral_dir_validates_bits_before_early_return(self, tmp_path):
        """Bit-width validation fires even when calibration data exists."""
        from olmlx.engine.model_manager import ModelManager

        manager = ModelManager(MagicMock(), MagicMock())
        spectral_path = tmp_path / "spectral"
        spectral_path.mkdir()
        (spectral_path / "spectral_config.json").write_text("{}")
        mock_store = MagicMock()
        mock_store.local_path.return_value = tmp_path
        manager.store = mock_store

        with pytest.raises(ValueError, match="spectral:3"):
            manager._find_spectral_dir("test/model", "spectral:3")

    def test_find_spectral_dir_raises_on_bit_width_mismatch(self, tmp_path):
        """Raises when calibration exists but at a different bit width."""
        import json as _json

        from olmlx.engine.model_manager import (
            ModelManager,
            SpectralCalibrationMissingError,
        )

        manager = ModelManager(MagicMock(), MagicMock())
        spectral_path = tmp_path / "spectral"
        spectral_path.mkdir()
        (spectral_path / "spectral_config.json").write_text(
            _json.dumps({"meta": {"avg_bits": 4}})
        )
        mock_store = MagicMock()
        mock_store.local_path.return_value = tmp_path
        manager.store = mock_store

        with pytest.raises(SpectralCalibrationMissingError) as exc_info:
            manager._find_spectral_dir("test/model", "spectral:2")
        msg = str(exc_info.value)
        assert "--avg-bits 2" in msg
        assert "generated with --avg-bits 4" in msg
        assert "OLMLX_KV_CACHE_QUANT=spectral:4" in msg

    def test_find_spectral_dir_raises_for_corrupt_config(self, tmp_path):
        """Raises SpectralCalibrationMissingError when config file is not valid JSON."""
        from olmlx.engine.model_manager import (
            ModelManager,
            SpectralCalibrationMissingError,
        )

        manager = ModelManager(MagicMock(), MagicMock())
        spectral_path = tmp_path / "spectral"
        spectral_path.mkdir()
        (spectral_path / "spectral_config.json").write_text("not valid json {{")
        mock_store = MagicMock()
        mock_store.local_path.return_value = tmp_path
        manager.store = mock_store

        with pytest.raises(SpectralCalibrationMissingError) as exc_info:
            manager._find_spectral_dir("test/model", "spectral:4")
        msg = str(exc_info.value)
        assert "unreadable" in msg
        assert "olmlx spectral prepare test/model" in msg


class TestEvictLruIfNeeded:
    """Tests for ModelManager._pop_lru_evictees + _close_evictees.

    Eviction is split: ``_pop_lru_evictees`` pops under the lock, then
    ``_close_evictees`` runs the close off the event loop with the
    lock released so concurrent ``ensure_loaded`` callers — including
    ones requesting an already-loaded model — aren't stalled by the
    48-thread executor.shutdown join. Issue #315.
    """

    def test_no_eviction_below_capacity(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 3)
        manager = ModelManager(registry, mock_store)
        lm = LoadedModel(
            name="a", hf_path="a/a", model=MagicMock(), tokenizer=MagicMock()
        )
        manager._loaded["a"] = lm
        assert manager._pop_lru_evictees() == []
        assert "a" in manager._loaded

    def test_evicts_oldest_inactive(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=MagicMock(),
            tokenizer=MagicMock(),
            loaded_at=time.time() - 100,
        )
        manager._loaded["old"] = old
        evictees = manager._pop_lru_evictees()
        assert [e.name for e in evictees] == ["old"]
        assert "old" not in manager._loaded

    def test_raises_when_all_active(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)
        active = LoadedModel(
            name="active",
            hf_path="a/a",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        active.active_refs = 1
        manager._loaded["active"] = active
        with pytest.raises(RuntimeError, match="All loaded models are in use"):
            manager._pop_lru_evictees()

    def test_pop_one_idle_lru_excludes_named_model(self, registry, mock_store):
        """Pressure eviction must never pop the model we're about to load.

        If a concurrent caller loads ``normalized`` during the lock-release
        window it is idle (active_refs == 0) and could be the LRU victim;
        evicting it would close a model the other caller just received and
        cause this caller to load a duplicate.  ``exclude`` guards against it.
        """
        manager = ModelManager(registry, mock_store)
        older = LoadedModel(
            name="older",
            hf_path="o/o",
            model=MagicMock(),
            tokenizer=MagicMock(),
            loaded_at=time.time() - 100,
        )
        newer = LoadedModel(
            name="newer",
            hf_path="n/n",
            model=MagicMock(),
            tokenizer=MagicMock(),
            loaded_at=time.time(),
        )
        manager._loaded["older"] = older
        manager._loaded["newer"] = newer
        # "older" is the LRU victim, but excluded → next idle model returned.
        popped = manager._pop_one_idle_lru(exclude="older")
        assert popped is newer
        assert "newer" not in manager._loaded
        assert "older" in manager._loaded

    def test_pop_one_idle_lru_returns_none_when_only_excluded(
        self, registry, mock_store
    ):
        manager = ModelManager(registry, mock_store)
        only = LoadedModel(
            name="only", hf_path="o/o", model=MagicMock(), tokenizer=MagicMock()
        )
        manager._loaded["only"] = only
        assert manager._pop_one_idle_lru(exclude="only") is None
        assert "only" in manager._loaded

    def test_pinned_model_not_evicted(self, registry, mock_store):
        """A pinned model (acquire_ref) is not idle, so eviction skips it.

        This is the protection ensure_loaded(pin=True) buys: a model pinned
        under the lock cannot be popped by the pressure-eviction loop in the
        handoff window before the caller's _inference_ref adopts the pin.
        """
        manager = ModelManager(registry, mock_store)
        lm = LoadedModel(
            name="a", hf_path="a/a", model=MagicMock(), tokenizer=MagicMock()
        )
        manager._loaded["a"] = lm
        lm.acquire_ref()
        assert lm.active_refs == 1
        assert manager._pop_one_idle_lru() is None  # pinned → protected
        lm.release_ref()
        assert lm.active_refs == 0
        assert manager._pop_one_idle_lru() is lm  # released → evictable

    @pytest.mark.asyncio
    async def test_close_evictees_does_not_touch_gc_or_metal(
        self, registry, mock_store
    ):
        """``_close_evictees`` MUST NOT flush gc / Metal itself.

        ``ensure_loaded`` (and the matching exception handlers) own that
        flush so it can run unconditionally before the post-eviction
        memory baseline and be suppressed when a deferred cleanup is in
        flight. Doing it inside ``_close_evictees`` would either double-
        flush during a real load or skip when a non-eviction caller
        depends on it.
        """
        manager = ModelManager(registry, mock_store)
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        with (
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
            patch("olmlx.engine.model_manager.mx.synchronize") as mock_sync,
        ):
            await manager._close_evictees([old])
            mock_gc.assert_not_called()
            mock_sync.assert_not_called()
            mock_clear.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_evictees_nulls_speculative_decoder(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        decoder = MagicMock()
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=MagicMock(),
            tokenizer=MagicMock(),
            speculative_decoder=decoder,
        )
        await manager._close_evictees([old])
        decoder.close.assert_called_once()

    def test_close_loaded_model_continues_on_failure(self, registry, mock_store):
        """A raising prefetcher.close() must not skip weight_store/decoder cleanup.

        Without try/finally chaining, a single resource failure during eviction
        or expiry would leak the weight store's file descriptors and leave the
        speculative decoder's GDN monkey-patch installed indefinitely.

        _close_loaded_model always raises ExceptionGroup on any error
        (single or multiple) so callers see a stable exception type.
        """
        manager = ModelManager(registry, mock_store)
        prefetcher = MagicMock()
        prefetcher.close.side_effect = RuntimeError("boom")
        weight_store = MagicMock()
        decoder = MagicMock()
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        lm = LoadedModel(
            name="x",
            hf_path="x/x",
            model=flash_model,
            tokenizer=MagicMock(),
            weight_store=weight_store,
            speculative_decoder=decoder,
            is_flash=True,
        )

        with pytest.raises(ExceptionGroup) as exc_info:
            manager._close_loaded_model(lm)

        # Single-failure case is still wrapped in ExceptionGroup for a
        # stable contract — see docstring on _close_loaded_model.
        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], RuntimeError)
        assert "boom" in str(exc_info.value.exceptions[0])
        # Both subsequent resources must still be released.
        weight_store.close.assert_called_once()
        decoder.close.assert_called_once()
        # LoadedModel-owned references are nulled so later code can't
        # accidentally observe a closed resource. ``lm.model.prefetcher``
        # is intentionally left alone — see helper docstring.
        assert lm.weight_store is None
        assert lm.speculative_decoder is None

    def test_close_loaded_model_surfaces_multiple_failures(self, registry, mock_store):
        """When two close() calls raise, both errors must surface.

        Python's nested-try/finally silently replaces an earlier exception
        with a later one. The per-resource try/except pattern collects all
        failures and raises an ExceptionGroup so neither failure is hidden.
        """
        manager = ModelManager(registry, mock_store)
        prefetcher = MagicMock()
        prefetcher.close.side_effect = RuntimeError("prefetcher-boom")
        weight_store = MagicMock()
        weight_store.close.side_effect = RuntimeError("weight-store-boom")
        decoder = MagicMock()
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        lm = LoadedModel(
            name="x",
            hf_path="x/x",
            model=flash_model,
            tokenizer=MagicMock(),
            weight_store=weight_store,
            speculative_decoder=decoder,
            is_flash=True,
        )

        with pytest.raises(ExceptionGroup) as exc_info:
            manager._close_loaded_model(lm)

        messages = [str(e) for e in exc_info.value.exceptions]
        assert any("prefetcher-boom" in m for m in messages)
        assert any("weight-store-boom" in m for m in messages)
        # Decoder still closed despite both prior failures.
        decoder.close.assert_called_once()
        # Failed close() leaves the reference alive — preserves the
        # partially-closed object for inspection / retry. Successful
        # decoder close still nulls its field.
        assert lm.weight_store is weight_store
        assert lm.speculative_decoder is None

    def test_close_loaded_model_clears_prompt_cache(self, registry, mock_store):
        """_close_loaded_model must clear prompt caches and null the store on eviction.

        CachedPromptState objects hold per-layer GPU KV cache buffers. If
        we don't clear them during eviction, the GPU memory remains
        allocated even after ``gc.collect()`` — the Metal buffers survive
        because the PromptCacheStore references keep them alive. On success,
        the store reference is also nulled (consistent with weight_store /
        speculative_decoder) so re-entry skips a redundant clear().
        """
        manager = ModelManager(registry, mock_store)
        cache_store = MagicMock()
        lm = LoadedModel(
            name="x",
            hf_path="x/x",
            model=MagicMock(),
            tokenizer=MagicMock(),
            prompt_cache_store=cache_store,
        )
        manager._close_loaded_model(lm)
        cache_store.clear.assert_called_once()
        assert lm.prompt_cache_store is None

    def test_close_loaded_model_preserves_model_for_caller_nulling(
        self, registry, mock_store
    ):
        """_close_loaded_model must NOT null model/tokenizer — caller does it.

        Model/tokenizer nulling moved from _close_loaded_model (worker
        thread) to _close_evictees (event loop) to prevent a race: between
        ensure_loaded() returning and the caller accessing lm.model, the
        worker thread could null it and crash the caller. The caller
        (_close_evictees) sets model=None after the thread joins, with no
        await between null and del.
        """
        manager = ModelManager(registry, mock_store)
        mock_model = MagicMock()
        mock_tok = MagicMock()
        lm = LoadedModel(
            name="x",
            hf_path="x/x",
            model=mock_model,
            tokenizer=mock_tok,
        )
        manager._close_loaded_model(lm)
        # _close_loaded_model preserves fields; caller (_close_evictees)
        # nulls them on the event loop.
        assert lm.model is mock_model
        assert lm.tokenizer is mock_tok

    def test_close_loaded_model_preserves_model_on_prior_failure(
        self, registry, mock_store
    ):
        """_close_loaded_model never nulls model/tokenizer (caller does it).

        The ``_close_evictees`` finally-drain path may call
        ``_close_loaded_model`` a second time after a first call raised.
        Model/tokenizer nulling is handled by _close_evictees on the
        event loop (not in the worker thread) to prevent a race between
        ensure_loaded() return and the worker thread nulling the fields.
        This test verifies the model/tokenizer survive a failed close so
        re-entry can inspect lm.model.
        """
        manager = ModelManager(registry, mock_store)
        weight_store = MagicMock()
        weight_store.close.side_effect = RuntimeError("stuck-weight-store")
        mock_model = MagicMock()
        mock_tok = MagicMock()
        lm = LoadedModel(
            name="x",
            hf_path="x/x",
            model=mock_model,
            tokenizer=mock_tok,
            weight_store=weight_store,
        )

        with pytest.raises(ExceptionGroup):
            manager._close_loaded_model(lm)
        # Model and tokenizer are always preserved by _close_loaded_model
        # (nulling happens in _close_evictees, on the event loop).
        assert lm.model is mock_model
        assert lm.tokenizer is mock_tok

    def test_prefetcher_close_call_count_on_partial_failure(self, registry, mock_store):
        """Prefetcher.close() must be called exactly once on a partial-failure path.

        On re-entry from the _close_evictees drain, weight_store and
        speculative_decoder are already nulled (they're nulled on success),
        so re-entry skips them.  The prefetcher lives on FlashModelWrapper
        and cannot be nulled the same way — but must still not be
        double-closed on re-entry.  Currently the prefetcher IS re-attempted
        on re-entry (a known limitation tracked by this test); the test
        documents the contract and prevents a future change from silently
        making it worse.
        """
        manager = ModelManager(registry, mock_store)
        prefetcher = MagicMock()
        weight_store = MagicMock()
        weight_store.close.side_effect = RuntimeError("stuck")
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        lm = LoadedModel(
            name="x",
            hf_path="x/x",
            model=flash_model,
            tokenizer=MagicMock(),
            weight_store=weight_store,
            is_flash=True,
        )

        with pytest.raises(ExceptionGroup):
            manager._close_loaded_model(lm)

        # weight_store failed → on re-entry, prefetcher should not have
        # been nulled (it can't be — it lives on FlashModelWrapper).
        assert lm.model is flash_model

        # Re-entry: call _close_loaded_model again (simulating the drain).
        # The prefetcher IS re-closed here (known limitation).
        # Verify it was called once across BOTH passes, not twice on the
        # re-entry — the call counts document the current behaviour.
        with pytest.raises(ExceptionGroup):
            manager._close_loaded_model(lm)
        assert prefetcher.close.call_count == 2  # known limitation; must not regress

    @pytest.mark.asyncio
    async def test_close_evictees_nulls_model_on_event_loop(self, registry, mock_store):
        """_close_evictees nulls model/tokenizer after the worker thread joins.

        Nulling must happen on the event loop (not in the worker thread) to
        prevent a race: between ensure_loaded() returning and the caller
        accessing lm.model, the worker thread could null it and crash the
        caller. Since _close_evictees runs on the event loop and has no
        await between null and del, no other coroutine can observe the
        nulled field.
        """
        manager = ModelManager(registry, mock_store)
        mock_model = MagicMock()
        mock_tok = MagicMock()
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=mock_model,
            tokenizer=mock_tok,
        )
        await manager._close_evictees([old])
        # After _close_evictees completes, model/tokenizer must be None
        # so the caller's gc.collect() can reclaim Metal buffers.
        assert old.model is None
        assert old.tokenizer is None

    @pytest.mark.asyncio
    async def test_close_evictees_absorbs_close_failure(
        self, registry, mock_store, caplog
    ):
        """Close failures must not propagate.

        A stuck prefetcher would otherwise permanently block all future
        model loads. The close site logs the error and continues.
        """
        manager = ModelManager(registry, mock_store)
        prefetcher = MagicMock()
        prefetcher.close.side_effect = RuntimeError("stuck-prefetcher")
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=flash_model,
            tokenizer=MagicMock(),
            is_flash=True,
        )

        with caplog.at_level(logging.ERROR, logger="olmlx.engine.model_manager"):
            await manager._close_evictees([old])  # must not raise

        # _close_loaded_model logs per-resource; close site absorbs silently.
        assert any(
            "Error closing prefetcher for old" in r.message for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_close_evictees_closes_flash_resources(self, registry, mock_store):
        """Closing a Flash evictee must close prefetcher + weight_store.

        Otherwise ThreadPoolExecutor workers and per-layer file descriptors
        leak for every evicted Flash model (issue #178).
        """
        manager = ModelManager(registry, mock_store)
        # Wire both resources through the same ``parent`` MagicMock so the
        # cross-resource ordering assertion below has a single ordered call
        # log to inspect. See the matching test in TestExpiryChecker.
        parent = MagicMock()
        prefetcher = parent.prefetcher
        weight_store = parent.weight_store
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=flash_model,
            tokenizer=MagicMock(),
            weight_store=weight_store,
            is_flash=True,
        )
        await manager._close_evictees([old])
        prefetcher.close.assert_called_once()
        weight_store.close.assert_called_once()
        # Order matters: prefetcher tasks submit into the weight store's pool,
        # so the prefetcher must shut down before the weight store.
        call_names = [c[0] for c in parent.mock_calls]
        assert call_names.index("prefetcher.close") < call_names.index(
            "weight_store.close"
        )

    @pytest.mark.asyncio
    async def test_close_evictees_offloads_close_to_thread(
        self, registry, mock_store, monkeypatch
    ):
        """The close must run off the event loop.

        ``executor.shutdown(wait=True)`` is synchronous and joins 48 threads
        (16 prefetch + 32 weight store). Running it on the event loop thread
        stalls every concurrent coroutine — even ones that don't touch the
        model manager — until the pools drain. Issue #315.
        """
        manager = ModelManager(registry, mock_store)
        original_to_thread = asyncio.to_thread
        to_thread_calls: list[Any] = []

        async def _tracking_to_thread(fn, *args, **kwargs):
            to_thread_calls.append(fn)
            return await original_to_thread(fn, *args, **kwargs)

        monkeypatch.setattr(
            "olmlx.engine.model_manager.asyncio.to_thread", _tracking_to_thread
        )

        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        await manager._close_evictees([old])

        assert manager._close_loaded_model in to_thread_calls

    @pytest.mark.asyncio
    async def test_close_evictees_runs_without_lock(self, registry, mock_store):
        """``_close_evictees`` must NOT require ``self._lock``.

        The whole point of splitting pop and close is so the close runs
        with the lock released — that's what lets concurrent
        ``ensure_loaded`` callers (e.g. for an already-loaded model)
        return immediately while a Flash close drains its 48-thread
        pools. We verify by acquiring the lock around the call and
        watching the close still complete.
        """
        manager = ModelManager(registry, mock_store)
        close_completed = threading.Event()

        def _close(_lm):
            close_completed.set()

        manager._close_loaded_model = _close  # type: ignore[method-assign]
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        # If _close_evictees tried to take the lock, this would deadlock.
        async with manager._lock:
            await manager._close_evictees([old])
        assert close_completed.is_set()

    @pytest.mark.asyncio
    async def test_close_evictees_cancellation_drains_remaining(
        self, registry, mock_store
    ):
        """A CancelledError mid-loop must NOT leak remaining evictees.

        ``asyncio.to_thread`` propagates ``asyncio.CancelledError`` when
        the awaiting task is cancelled (e.g. client disconnect during
        eviction). The popped evictees that haven't been closed yet
        would otherwise be dropped on the floor — they're already
        gone from ``_loaded``, so nothing else will close their
        prefetcher / weight store pools (48 threads each). The fix
        drains the remainder synchronously in a finally before
        re-raising so the cleanup contract holds even on the abnormal
        path.
        """
        manager = ModelManager(registry, mock_store)
        closed_names: list[str] = []

        def _close(lm):
            closed_names.append(lm.name)

        manager._close_loaded_model = _close  # type: ignore[method-assign]

        # Patch ``asyncio.to_thread`` to record the close call and raise
        # ``CancelledError`` on the second invocation — simulating a
        # client disconnect while ``_close_evictees`` is mid-loop.
        original_to_thread = asyncio.to_thread

        async def _cancelling_to_thread(fn, *args, **kwargs):
            if len(closed_names) == 1:
                raise asyncio.CancelledError()
            return await original_to_thread(fn, *args, **kwargs)

        evictees = [
            LoadedModel(
                name=f"e{i}",
                hf_path=f"e{i}/r",
                model=MagicMock(),
                tokenizer=MagicMock(),
            )
            for i in range(3)
        ]
        with patch(
            "olmlx.engine.model_manager.asyncio.to_thread", _cancelling_to_thread
        ):
            with pytest.raises(asyncio.CancelledError):
                await manager._close_evictees(evictees)

        # All three evictees must be closed despite cancellation —
        # the first via the normal async path, the remaining two via
        # the sync drain in the cancellation handler.
        assert sorted(closed_names) == ["e0", "e1", "e2"]

    @pytest.mark.asyncio
    async def test_close_evictees_unblocks_event_loop(self, registry, mock_store):
        """A slow close must not block the event loop.

        Use a threading.Event rendezvous so the assertion does not
        depend on wall-clock timing. The slow-close worker thread
        blocks on the event; while it blocks, the event loop services
        the sibling coroutine, which signals the event so the close
        can finish.
        """
        manager = ModelManager(registry, mock_store)
        block_close = threading.Event()
        sibling_ran = False

        def _slow_close(_lm):
            # Block until the sibling coroutine signals — proves the
            # event loop kept turning while this worker thread waited.
            assert block_close.wait(timeout=5.0)

        manager._close_loaded_model = _slow_close  # type: ignore[method-assign]

        async def _sibling():
            nonlocal sibling_ran
            sibling_ran = True
            block_close.set()

        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        sibling_task = asyncio.create_task(_sibling())
        await manager._close_evictees([old])
        await sibling_task
        assert sibling_ran

    @pytest.mark.asyncio
    async def test_ensure_loaded_releases_lock_during_close(
        self, registry, mock_store, monkeypatch
    ):
        """``ensure_loaded`` must release the lock while closing evictees.

        Otherwise an ``ensure_loaded`` call for an already-loaded
        model is stalled for the duration of an unrelated Flash close
        — the regression Claude raised on the first round of #315.
        """
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)
        lock_held_during_close: list[bool] = []

        async def _spy_close(_evictees):
            lock_held_during_close.append(manager._lock.locked())

        manager._close_evictees = _spy_close  # type: ignore[method-assign]

        # Pre-load a model so eviction triggers when we ask for another.
        existing = LoadedModel(
            name="existing:latest",
            hf_path="existing/repo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            loaded_at=time.time() - 100,
        )
        manager._loaded["existing:latest"] = existing

        # Stub the registry + load path so ensure_loaded reaches the close.
        manager.registry.resolve = MagicMock(  # type: ignore[method-assign]
            return_value=ModelConfig(hf_path="new/repo")
        )
        manager.registry.normalize_name = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda n: f"{n}:latest"
        )

        # ``_detect_model_kind`` is offloaded via ``asyncio.to_thread`` before
        # the load (#614); let that call through (stub it fast, no network) and
        # abort only the load itself.
        manager._detect_model_kind = lambda hf_path: "text"  # type: ignore[method-assign]
        real_to_thread = asyncio.to_thread

        async def _abort_load(func, *a, **kw):
            if func is manager._detect_model_kind:
                return await real_to_thread(func, *a, **kw)
            raise RuntimeError("stop before load")

        monkeypatch.setattr("olmlx.engine.model_manager.asyncio.to_thread", _abort_load)

        with pytest.raises(RuntimeError, match="stop before load"):
            await manager.ensure_loaded("new")

        assert lock_held_during_close == [False]

    @pytest.mark.asyncio
    async def test_ensure_loaded_preload_memory_hygiene(
        self, registry, mock_store, monkeypatch
    ):
        """ensure_loaded must flush prompt caches from remaining models before load.

        After closing evictees, Metal memory can still be under pressure
        if the previous model's prompt caches or residual Metal allocations
        weren't fully reclaimed. The pre-load memory hygiene path must run
        OUTSIDE _lock (Bug 1) and use async_evict_all_to_disk() to offload
        disk I/O to a worker thread.
        """
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 2)
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", None
        )
        monkeypatch.setattr(
            "olmlx.utils.memory.is_memory_pressure_high",
            lambda _fraction, threshold=0.9: True,
        )
        manager = ModelManager(registry, mock_store)

        cache_store = MagicMock()
        cache_store.async_evict_all_to_disk = AsyncMock()
        existing = LoadedModel(
            name="existing:latest",
            hf_path="existing/repo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            prompt_cache_store=cache_store,
            loaded_at=time.time() - 100,
        )
        manager._loaded["existing:latest"] = existing

        manager.registry.resolve = MagicMock(  # type: ignore[method-assign]
            return_value=ModelConfig(hf_path="new/repo")
        )
        manager.registry.normalize_name = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda n: f"{n}:latest"
        )

        def _shard(*args, **kwargs):
            raise RuntimeError("stop before load")

        monkeypatch.setattr(manager, "_load_model_and_shard", _shard)

        with pytest.raises(RuntimeError, match="stop before load"):
            await manager.ensure_loaded("new")

        # Pre-load memory hygiene must have flushed the remaining model's
        # prompt caches (async path — offloaded to thread).
        cache_store.async_evict_all_to_disk.assert_called()

    @pytest.mark.asyncio
    async def test_ensure_loaded_preload_memory_hygiene_happy_path(
        self, registry, mock_store, monkeypatch
    ):
        """The hygiene flush succeeds and the subsequent load completes normally.

        Verifies the full happy path: memory pressure triggers the flush,
        the flush completes, and then the normal loading flow proceeds
        without errors.  Without this, a hygiene-pass bug that derails the
        load path (e.g. accidentally clearing load state) would go undetected
        until a real bench sweep.
        """
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 2)
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", None
        )
        # First call (pre-load hygiene check): pressure high → flush.
        # Second call (post-hygiene check): pressure resolved → proceed.
        # Use MagicMock side_effect so a third call raises StopIteration
        # — making unexpected extra pressure checks immediately visible
        # rather than silently swallowed by itertools.repeat.
        mock_pressure = MagicMock(side_effect=[True, False])
        monkeypatch.setattr("olmlx.utils.memory.is_memory_pressure_high", mock_pressure)
        manager = ModelManager(registry, mock_store)

        cache_store = MagicMock()
        cache_store.async_evict_all_to_disk = AsyncMock()
        existing = LoadedModel(
            name="existing:latest",
            hf_path="existing/repo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            prompt_cache_store=cache_store,
            loaded_at=time.time() - 100,
        )
        manager._loaded["existing:latest"] = existing

        manager.registry.resolve = MagicMock(  # type: ignore[method-assign]
            return_value=ModelConfig(hf_path="new/repo")
        )
        manager.registry.normalize_name = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda n: f"{n}:latest"
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        def _shard(*args, **kwargs):
            return (mock_model, mock_tokenizer, False, TemplateCaps(), False, None)

        monkeypatch.setattr(manager, "_load_model_and_shard", _shard)

        lm = await manager.ensure_loaded("new")
        assert lm.name == "new:latest"
        cache_store.async_evict_all_to_disk.assert_called()


class TestSpeculativeLoading:
    """Tests for standalone speculative decoder loading in _load_model."""

    def test_load_model_creates_speculative_decoder(self, monkeypatch):
        """When speculative is enabled, _load_model should return a SpeculativeDecoder."""
        from olmlx.config import ExperimentalSettings
        from olmlx.engine.speculative import SpeculativeDecoder

        target_model = MagicMock()
        target_model.args.vocab_size = 32000
        target_tokenizer = MagicMock()
        caps = TemplateCaps()

        draft_model = MagicMock()
        draft_model.args.vocab_size = 32000

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-draft")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(
            manager,
            "_try_lm_then_vlm",
            lambda *a, **kw: (target_model, target_tokenizer, False, caps),
        )
        monkeypatch.setattr(manager, "_detect_model_kind", lambda *a: "text")
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (draft_model, MagicMock())
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm", mock_mlx_lm)

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, "test/draft-model", 5)

        model, tok, is_vlm, caps_out, decoder = manager._load_model(
            "test/target-model", model_exp=model_exp, spec_config=spec_config
        )

        assert isinstance(decoder, SpeculativeDecoder)
        assert decoder._lambda == 5
        assert model is target_model

    def test_load_model_rejects_vocab_mismatch(self, monkeypatch):
        """Should raise ValueError when draft/target vocab sizes differ."""
        from olmlx.config import ExperimentalSettings

        target_model = MagicMock()
        target_model.args.vocab_size = 32000

        draft_model = MagicMock()
        draft_model.args.vocab_size = 64000

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-draft")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(
            manager,
            "_try_lm_then_vlm",
            lambda *a, **kw: (target_model, MagicMock(), False, TemplateCaps()),
        )
        monkeypatch.setattr(manager, "_detect_model_kind", lambda *a: "text")
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (draft_model, MagicMock())
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm", mock_mlx_lm)

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, "test/draft-model", 4)

        with pytest.raises(ValueError, match="vocab_size"):
            manager._load_model(
                "test/target-model", model_exp=model_exp, spec_config=spec_config
            )

    def test_load_speculative_decoder_rejects_disabled_config(self, monkeypatch):
        """``_load_speculative_decoder`` keeps the invariant that callers gate
        on ``spec_config.enabled`` before invoking it. Direct invocation with
        ``enabled=False`` must raise — assert is elided under ``python -O``,
        so the guard is a real ``RuntimeError``."""
        registry = MagicMock()
        store = MagicMock()
        manager = ModelManager(registry, store)

        spec_config = SpeculativeConfig(False, "test/draft", 4)
        with pytest.raises(RuntimeError, match="spec_config.enabled=False"):
            manager._load_speculative_decoder(MagicMock(), "test/target", spec_config)

    def test_load_model_requires_draft_model_path(self, monkeypatch):
        """Should raise ValueError when speculative is enabled but no draft model."""
        from olmlx.config import ExperimentalSettings

        registry = MagicMock()
        store = MagicMock()

        manager = ModelManager(registry, store)
        monkeypatch.setattr(
            manager,
            "_try_lm_then_vlm",
            lambda *a, **kw: (MagicMock(), MagicMock(), False, TemplateCaps()),
        )
        monkeypatch.setattr(manager, "_detect_model_kind", lambda *a: "text")
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, None, 4)

        with pytest.raises(ValueError, match="speculative_draft_model"):
            manager._load_model(
                "test/target-model", model_exp=model_exp, spec_config=spec_config
            )

    def test_flash_path_warns_when_standalone_speculative_set(
        self, monkeypatch, caplog
    ):
        """A Flash model combined with the standalone speculative flag must
        log a warning so the user notices the redirect to flash_speculative."""
        import logging

        from olmlx.config import ExperimentalSettings

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-flash")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: True)
        monkeypatch.setattr(
            manager, "_flash_dir", lambda hf_path: Path("/tmp/test-flash/flash")
        )
        sentinel = (object(), object(), False, TemplateCaps(), object())
        monkeypatch.setattr(
            manager,
            "_load_flash_model",
            lambda hf_path, load_path, flash_dir, *, model_exp, flash_config: sentinel,
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, "test/draft", 4)

        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            result = manager._load_model(
                "test/flash-model", model_exp=model_exp, spec_config=spec_config
            )
        # Flash path forwards its tuple element-for-element (PLD support
        # added a conditional decoder swap, so identity is no longer
        # preserved — value equality is the meaningful contract here).
        assert result == sentinel
        assert "OLMLX_SPECULATIVE" in caplog.text
        assert "Flash" in caplog.text

    def test_flash_path_swaps_in_pld_decoder(self, monkeypatch, caplog):
        """Flash (non-MoE) + PLD should swap the flash-returned decoder
        for a PLD decoder — PLD doesn't conflict with the flash forward
        wrapper, so we honour OLMLX_SPECULATIVE_STRATEGY=pld instead of
        warning + dropping it (which is what classic/dflash/eagle hit)."""
        import logging

        from olmlx.config import ExperimentalSettings
        from olmlx.engine.registry import ResolvedFlashConfig

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-flash")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: True)
        monkeypatch.setattr(
            manager, "_flash_dir", lambda hf_path: Path("/tmp/test-flash/flash")
        )
        flash_model = object()
        flash_tok = object()
        flash_caps = TemplateCaps()
        # The flash loader returns ``decoder=None`` when flash_speculative
        # is off; the PLD swap happens in ``_load_model`` after the call.
        monkeypatch.setattr(
            manager,
            "_load_flash_model",
            lambda *a, **kw: (flash_model, flash_tok, False, flash_caps, None),
        )
        sentinel_pld = object()
        monkeypatch.setattr(
            manager,
            "_load_pld_decoder",
            lambda model, spec_config, *, is_vlm=False: sentinel_pld,
        )

        flash_config = ResolvedFlashConfig(
            enabled=True,
            sparsity_threshold=0.5,
            min_active_neurons=128,
            max_active_neurons=None,
            memory_budget_fraction=None,
            prefetch=False,
            flash_speculative=False,
        )
        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, None, 10, strategy="pld")

        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            model, tok, is_vlm, caps, decoder = manager._load_model(
                "test/flash-model",
                model_exp=model_exp,
                spec_config=spec_config,
                flash_config=flash_config,
            )
        assert model is flash_model
        assert tok is flash_tok
        assert decoder is sentinel_pld
        # PLD is honoured, so the warn-and-ignore message must NOT fire.
        assert "OLMLX_SPECULATIVE" not in caplog.text

    def test_flash_path_warns_when_classic_spec_set_with_flash_speculative(
        self, monkeypatch, caplog
    ):
        """Flash + flash_speculative=True + OLMLX_SPECULATIVE=true (classic)
        must still log the warn-and-ignore message. The user's classic
        spec setting is ignored either way — flash_speculative being on
        doesn't change that, so suppressing the warning would hide a
        misconfiguration."""
        import logging

        from olmlx.config import ExperimentalSettings
        from olmlx.engine.registry import ResolvedFlashConfig

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-flash")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: True)
        monkeypatch.setattr(
            manager, "_flash_dir", lambda hf_path: Path("/tmp/test-flash/flash")
        )
        sentinel = (object(), object(), False, TemplateCaps(), object())
        monkeypatch.setattr(
            manager,
            "_load_flash_model",
            lambda *a, **kw: sentinel,
        )

        flash_config = ResolvedFlashConfig(
            enabled=True,
            sparsity_threshold=0.5,
            min_active_neurons=128,
            max_active_neurons=None,
            memory_budget_fraction=None,
            prefetch=False,
            flash_speculative=True,
            flash_speculative_draft_model="test/draft",
        )
        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, "test/draft", 4, strategy="classic")

        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            manager._load_model(
                "test/flash-model",
                model_exp=model_exp,
                spec_config=spec_config,
                flash_config=flash_config,
            )
        # The user's OLMLX_SPECULATIVE setting is ignored — warning fires.
        assert "OLMLX_SPECULATIVE" in caplog.text

    def test_flash_path_rejects_pld_with_flash_speculative(self, monkeypatch):
        """Flash + flash_speculative + PLD must raise — two speculative
        decoders fighting over the same target is a configuration error,
        not a silent override."""
        from olmlx.config import ExperimentalSettings
        from olmlx.engine.registry import ResolvedFlashConfig

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-flash")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: True)
        monkeypatch.setattr(
            manager, "_flash_dir", lambda hf_path: Path("/tmp/test-flash/flash")
        )

        flash_config = ResolvedFlashConfig(
            enabled=True,
            sparsity_threshold=0.5,
            min_active_neurons=128,
            max_active_neurons=None,
            memory_budget_fraction=None,
            prefetch=False,
            flash_speculative=True,
            flash_speculative_draft_model="test/draft",
        )
        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, None, 10, strategy="pld")

        with pytest.raises(ValueError, match="flash_speculative.*pld"):
            manager._load_model(
                "test/flash-model",
                model_exp=model_exp,
                spec_config=spec_config,
                flash_config=flash_config,
            )

    def test_flash_moe_path_supports_classic_speculative(self, monkeypatch, caplog):
        """Flash-MoE + classic speculative should load the decoder (not drop it)."""
        import logging

        from olmlx.config import ExperimentalSettings

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-moe")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: True)
        monkeypatch.setattr(
            manager, "_flash_moe_dir", lambda hf_path: Path("/tmp/test-moe/flash_moe")
        )
        sentinel_load = (object(), object(), False, TemplateCaps())
        monkeypatch.setattr(
            manager,
            "_load_flash_moe_model",
            lambda hf_path, load_path, flash_moe_dir, *, flash_moe_config: (
                sentinel_load
            ),
        )
        sentinel_decoder = object()
        monkeypatch.setattr(
            manager,
            "_load_speculative_decoder",
            lambda model, hf_path, spec_config, *, is_vlm=False: sentinel_decoder,
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, "test/draft", 4)
        fm_config = FlashMoeConfig(enabled=True, cache_budget_experts=48, io_threads=32)

        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            model, tokenizer, is_vlm, caps, decoder = manager._load_model(
                "test/moe-model",
                model_exp=model_exp,
                spec_config=spec_config,
                flash_moe_config=fm_config,
            )
        # Flash-MoE now supports classic speculative; decoder is loaded.
        assert (model, tokenizer, is_vlm, caps) == sentinel_load
        assert decoder is sentinel_decoder
        assert "OLMLX_SPECULATIVE" not in caplog.text

    def test_flash_moe_path_supports_pld(self, monkeypatch, caplog):
        """Flash-MoE + PLD should load the PLD decoder — PLD is the only
        speculative strategy that composes with Flash-MoE because it
        doesn't need a draft model and doesn't hook target hidden states."""
        import logging

        from olmlx.config import ExperimentalSettings

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-moe")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: True)
        monkeypatch.setattr(
            manager, "_flash_moe_dir", lambda hf_path: Path("/tmp/test-moe/flash_moe")
        )
        sentinel_load = (object(), object(), False, TemplateCaps())
        monkeypatch.setattr(
            manager,
            "_load_flash_moe_model",
            lambda hf_path, load_path, flash_moe_dir, *, flash_moe_config: (
                sentinel_load
            ),
        )
        sentinel_decoder = object()
        # The classic loader must NOT be called — failing here means PLD
        # silently fell back to classic.
        monkeypatch.setattr(
            manager,
            "_load_speculative_decoder",
            lambda *a, **kw: pytest.fail(
                "PLD strategy should not invoke _load_speculative_decoder"
            ),
        )
        monkeypatch.setattr(
            manager,
            "_load_pld_decoder",
            lambda model, spec_config, *, is_vlm=False: sentinel_decoder,
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, None, 10, strategy="pld")
        fm_config = FlashMoeConfig(enabled=True, cache_budget_experts=48, io_threads=32)

        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            model, tokenizer, is_vlm, caps, decoder = manager._load_model(
                "test/moe-model",
                model_exp=model_exp,
                spec_config=spec_config,
                flash_moe_config=fm_config,
            )
        assert (model, tokenizer, is_vlm, caps) == sentinel_load
        assert decoder is sentinel_decoder

    def test_load_pld_decoder_unwraps_vlm_language_model(self, monkeypatch):
        """When ``is_vlm=True``, _load_pld_decoder must construct the
        PromptLookupDecoder against ``target.language_model``, not the
        VLM wrapper. Mirrors the unwrap that
        ``_load_speculative_decoder`` does so the decoder operates on
        the text decoder's KV cache directly.
        """
        from olmlx.engine.registry import SpeculativeConfig

        registry = MagicMock()
        store = MagicMock()
        manager = ModelManager(registry, store)

        text_decoder = MagicMock(name="language_model")
        vlm_target = MagicMock(name="vlm")
        vlm_target.language_model = text_decoder
        # PromptLookupDecoder probes target_model for find_gdn_class
        # (named_modules walk) — return empty to skip GDN setup. The
        # decoder's __init__ doesn't call into the model otherwise.
        text_decoder.named_modules.return_value = iter([])

        captured = {}

        def fake_decoder_ctor(*args, **kw):
            captured["target_model"] = kw.get("target_model")
            return MagicMock()

        monkeypatch.setattr(
            "olmlx.engine.speculative.PromptLookupDecoder", fake_decoder_ctor
        )
        spec_config = SpeculativeConfig(
            True,
            None,
            10,
            strategy="pld",
            pld_max_ngram=3,
            pld_min_ngram=1,
            pld_lookup_window=8192,
        )
        manager._load_pld_decoder(vlm_target, spec_config, is_vlm=True)
        # The decoder must be built against the inner text decoder,
        # not the VLM wrapper.
        assert captured["target_model"] is text_decoder

    def test_load_pld_decoder_rejects_vlm_without_language_model(self, monkeypatch):
        """A VLM target missing ``.language_model`` must surface as
        ValueError with a message pointing at the missing attribute,
        not as a confusing AttributeError deep inside the decoder."""
        from olmlx.engine.registry import SpeculativeConfig

        registry = MagicMock()
        store = MagicMock()
        manager = ModelManager(registry, store)

        # A bare object() has no .language_model attribute.
        broken_vlm = object()
        spec_config = SpeculativeConfig(
            True,
            None,
            10,
            strategy="pld",
            pld_max_ngram=3,
            pld_min_ngram=1,
            pld_lookup_window=8192,
        )
        with pytest.raises(ValueError, match="language_model"):
            manager._load_pld_decoder(broken_vlm, spec_config, is_vlm=True)

    def test_text_path_dispatches_pld(self, monkeypatch):
        """speculative_strategy='pld' on a plain text target should route
        to _load_pld_decoder rather than the classic loader."""
        from olmlx.config import ExperimentalSettings
        from olmlx.engine.registry import SpeculativeConfig

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-text")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_detect_model_kind", lambda *a: "text")
        monkeypatch.setattr(
            manager,
            "_try_lm_then_vlm",
            lambda *a, **kw: (object(), object(), False, TemplateCaps()),
        )
        monkeypatch.setattr(
            manager,
            "_load_speculative_decoder",
            lambda *a, **kw: pytest.fail(
                "PLD strategy should not invoke _load_speculative_decoder"
            ),
        )
        sentinel = object()
        monkeypatch.setattr(
            manager,
            "_load_pld_decoder",
            lambda model, spec_config, *, is_vlm=False: sentinel,
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, None, None, strategy="pld")
        _model, _tok, _is_vlm, _caps, decoder = manager._load_model(
            "test/text-model", model_exp=model_exp, spec_config=spec_config
        )
        assert decoder is sentinel

    def test_flash_moe_path_rejects_dflash(self, monkeypatch):
        """Flash-MoE + dflash should raise ValueError."""
        from olmlx.config import ExperimentalSettings

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-moe")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: True)
        monkeypatch.setattr(
            manager, "_flash_moe_dir", lambda hf_path: Path("/tmp/test-moe/flash_moe")
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, "test/draft", 4, strategy="dflash")
        fm_config = FlashMoeConfig(enabled=True, cache_budget_experts=48, io_threads=32)

        with pytest.raises(ValueError, match="dflash.*not supported on Flash-MoE"):
            manager._load_model(
                "test/moe-model",
                model_exp=model_exp,
                spec_config=spec_config,
                flash_moe_config=fm_config,
            )

    def test_flash_moe_path_rejects_flash_speculative(self, monkeypatch):
        """Flash-MoE + flash_speculative should raise ValueError."""
        from olmlx.config import ExperimentalSettings
        from olmlx.engine.registry import ResolvedFlashConfig

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-moe")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: True)
        monkeypatch.setattr(
            manager, "_flash_moe_dir", lambda hf_path: Path("/tmp/test-moe/flash_moe")
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(False, None, 4)
        fm_config = FlashMoeConfig(enabled=True, cache_budget_experts=48, io_threads=32)
        flash_config = ResolvedFlashConfig(
            enabled=False,
            sparsity_threshold=0.0,
            min_active_neurons=0,
            max_active_neurons=None,
            memory_budget_fraction=None,
            flash_speculative=True,
        )

        with pytest.raises(
            ValueError, match="flash_speculative.*not supported on Flash-MoE"
        ):
            manager._load_model(
                "test/moe-model",
                model_exp=model_exp,
                spec_config=spec_config,
                flash_moe_config=fm_config,
                flash_config=flash_config,
            )


class TestDFlashLoading:
    """Tests for dflash decoder loading in _load_model."""

    def test_load_model_requires_dflash_draft_model(self, monkeypatch):
        """speculative_strategy='dflash' without a draft model should raise."""
        from olmlx.config import ExperimentalSettings
        from olmlx.engine.registry import SpeculativeConfig

        registry = MagicMock()
        store = MagicMock()

        manager = ModelManager(registry, store)
        monkeypatch.setattr(
            manager,
            "_try_lm_then_vlm",
            lambda *a, **kw: (MagicMock(), MagicMock(), False, TemplateCaps()),
        )
        monkeypatch.setattr(manager, "_detect_model_kind", lambda *a: "text")
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(
            enabled=True,
            draft_model=None,
            num_tokens=4,
            strategy="dflash",
        )

        with pytest.raises(ValueError, match="speculative_draft_model"):
            manager._load_model(
                "test/target-model",
                model_exp=model_exp,
                spec_config=spec_config,
            )

    def test_load_model_creates_dflash_decoder(self, monkeypatch):
        """speculative_strategy='dflash' should route through _load_dflash_decoder."""
        from olmlx.config import ExperimentalSettings
        from olmlx.engine.dflash.decoder import DFlashDecoder
        from olmlx.engine.registry import SpeculativeConfig

        target_model = MagicMock()
        target_model.args.vocab_size = 32000

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-dflash-draft")
        store.local_path.return_value = Path("/tmp/test-target")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(
            manager,
            "_try_lm_then_vlm",
            lambda *a, **kw: (target_model, MagicMock(), False, TemplateCaps()),
        )
        monkeypatch.setattr(manager, "_detect_model_kind", lambda *a: "text")
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)

        # Mock _load_dflash_decoder to verify it's called
        mock_decoder = MagicMock(spec=DFlashDecoder)
        monkeypatch.setattr(
            manager,
            "_load_dflash_decoder",
            lambda *a, **kw: mock_decoder,
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(
            enabled=True,
            draft_model="test/dflash-draft",
            num_tokens=4,
            strategy="dflash",
        )

        model, tok, is_vlm, caps_out, decoder = manager._load_model(
            "test/target-model",
            model_exp=model_exp,
            spec_config=spec_config,
        )

        assert decoder is mock_decoder


class TestEagleLoading:
    """Tests for ``_load_eagle_decoder`` schema validation and the
    decoder-construction path.

    Strategy mirrors ``TestDFlashLoading``: mock the heavy lifting
    (``_try_lm_then_vlm``, kind detection, flash gates) and stub
    ``_load_eagle_decoder`` itself when we want to verify routing.
    For schema-validation paths we drive ``_load_eagle_decoder``
    directly with a synthetic draft directory.
    """

    def _make_target_with(self, vocab_size: int, hidden_size: int) -> Any:
        """Build a fake target whose ``.args`` exposes both fields the
        loader walks for cross-checks. Uses a plain object rather than
        MagicMock so ``getattr(args, ...)`` returns None for absent
        fields instead of an auto-generated child mock."""
        target = MagicMock()

        class _Args:
            pass

        target.args = _Args()
        target.args.vocab_size = vocab_size
        target.args.hidden_size = hidden_size
        # Strip the chained-attr search paths the loader walks
        # (``.model``, ``.language_model``) so the first match wins.
        target.model = None
        target.language_model = None
        return target

    def _write_eagle_draft_dir(
        self,
        tmp_path: Path,
        *,
        vocab_size: int = 64,
        hidden_size: int = 16,
        block_size: int = 4,
        target_layer_id: int | None = 2,
        omit_eagle_config: bool = False,
    ) -> Path:
        """Write a minimal EAGLE draft directory (config + 1 weight
        shard) that ``_load_eagle_decoder`` can parse. Weight tensors
        are zero-filled — we only test the loader's metadata path,
        not real inference."""
        import mlx.core as mx

        draft_dir = tmp_path / "eagle_draft"
        draft_dir.mkdir()
        cfg: dict[str, Any] = {
            "hidden_size": hidden_size,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": hidden_size // 2,
            "intermediate_size": hidden_size * 2,
            "vocab_size": vocab_size,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 512,
        }
        if not omit_eagle_config:
            eagle_block: dict[str, Any] = {"block_size": block_size}
            if target_layer_id is not None:
                eagle_block["target_layer_id"] = target_layer_id
            cfg["eagle_config"] = eagle_block
        (draft_dir / "config.json").write_text(json.dumps(cfg))

        from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

        m = EagleDraftModel(
            EagleConfig(
                hidden_size=hidden_size,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=hidden_size // 2,
                intermediate_size=hidden_size * 2,
                vocab_size=vocab_size,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                max_position_embeddings=512,
                block_size=block_size,
            )
        )
        import mlx.utils as mx_utils

        weights = dict(mx_utils.tree_flatten(m.parameters()))
        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith("embed_tokens.") and not k.startswith("lm_head.")
        }
        mx.save_safetensors(
            str(draft_dir / "model-00001-of-00001.safetensors"), weights
        )
        return draft_dir

    def test_rejects_missing_eagle_config_block(self, tmp_path, registry, mock_store):
        from olmlx.engine.registry import SpeculativeConfig

        draft_dir = self._write_eagle_draft_dir(tmp_path, omit_eagle_config=True)
        manager = ModelManager(registry, mock_store)
        target = self._make_target_with(vocab_size=64, hidden_size=16)
        spec = SpeculativeConfig(
            enabled=True, draft_model=str(draft_dir), num_tokens=4, strategy="eagle"
        )
        with pytest.raises(ValueError, match="eagle_config"):
            manager._load_eagle_decoder(target, spec)

    def test_rejects_vocab_mismatch(self, tmp_path, registry, mock_store):
        from olmlx.engine.registry import SpeculativeConfig

        draft_dir = self._write_eagle_draft_dir(tmp_path, vocab_size=64)
        manager = ModelManager(registry, mock_store)
        # Target's vocab is 128, draft's is 64 — mismatch.
        target = self._make_target_with(vocab_size=128, hidden_size=16)
        spec = SpeculativeConfig(
            enabled=True, draft_model=str(draft_dir), num_tokens=4, strategy="eagle"
        )
        with pytest.raises(ValueError, match="vocab_size"):
            manager._load_eagle_decoder(target, spec)

    def test_rejects_hidden_size_mismatch(self, tmp_path, registry, mock_store):
        """Cross-target ``hidden_size`` mismatch must be caught at load
        time, not at the first prefill — the latter surfaces as a
        cryptic shape error inside ``input_proj``."""
        from olmlx.engine.registry import SpeculativeConfig

        draft_dir = self._write_eagle_draft_dir(tmp_path, hidden_size=16)
        manager = ModelManager(registry, mock_store)
        target = self._make_target_with(vocab_size=64, hidden_size=32)
        spec = SpeculativeConfig(
            enabled=True, draft_model=str(draft_dir), num_tokens=4, strategy="eagle"
        )
        with pytest.raises(ValueError, match="hidden_size"):
            manager._load_eagle_decoder(target, spec)

    def test_constructs_decoder_with_block_size_override(
        self, tmp_path, registry, mock_store
    ):
        """``spec_config.num_tokens`` overrides the saved ``block_size``.
        This is how operators tune block_size at the CLI without
        retraining."""
        from olmlx.engine.eagle.decoder import EagleDecoder
        from olmlx.engine.registry import SpeculativeConfig

        draft_dir = self._write_eagle_draft_dir(
            tmp_path, vocab_size=64, hidden_size=16, block_size=4
        )
        manager = ModelManager(registry, mock_store)
        # Build a real synthetic target so EagleDecoder's
        # ``_get_layers(target_model)`` works.
        from tests.test_dflash import _Target

        target = _Target(vocab_size=64, hidden_size=16, num_layers=4)
        spec = SpeculativeConfig(
            enabled=True,
            draft_model=str(draft_dir),
            num_tokens=2,  # override the saved 4
            strategy="eagle",
        )
        decoder = manager._load_eagle_decoder(target, spec)
        assert isinstance(decoder, EagleDecoder)
        assert decoder._block_size == 2
        # target_layer_id from saved config (2) should have been threaded.
        assert decoder._target_layer_id == 2

    def test_warns_on_missing_target_layer_id(
        self, tmp_path, registry, mock_store, caplog
    ):
        """Pre-fix checkpoints have no ``target_layer_id``. The loader
        must emit a ``logger.warning`` so the operator gets nudged to
        retrain rather than silently shipping a degraded draft."""
        import logging

        from olmlx.engine.registry import SpeculativeConfig

        draft_dir = self._write_eagle_draft_dir(
            tmp_path, vocab_size=64, hidden_size=16, target_layer_id=None
        )
        manager = ModelManager(registry, mock_store)
        from tests.test_dflash import _Target

        target = _Target(vocab_size=64, hidden_size=16, num_layers=4)
        spec = SpeculativeConfig(
            enabled=True, draft_model=str(draft_dir), num_tokens=2, strategy="eagle"
        )
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            decoder = manager._load_eagle_decoder(target, spec)
        assert any("target_layer_id" in r.message for r in caplog.records)
        # Falls back to last layer.
        assert decoder._target_layer_id == 3  # last index of 4 layers


class TestSpectralAutoCalibrate:
    """Tests for ``_find_spectral_dir`` and ``_auto_calibrate_spectral``."""

    def test_find_spectral_dir_returns_none_when_quant_is_none(
        self, registry, mock_store
    ):
        manager = ModelManager(registry, mock_store)
        result = manager._find_spectral_dir("test/model", None)
        assert result is None

    def test_find_spectral_dir_returns_none_for_non_spectral(
        self, registry, mock_store
    ):
        manager = ModelManager(registry, mock_store)
        result = manager._find_spectral_dir("test/model", "turboquant:4")
        assert result is None

    def test_find_spectral_dir_returns_none_when_store_is_none(self, registry):
        manager = ModelManager(registry, None)
        result = manager._find_spectral_dir("test/model", "spectral:4")
        assert result is None

    def test_find_spectral_dir_returns_path_when_calibration_exists(
        self, tmp_path, registry, mock_store
    ):
        local_dir = mock_store.local_path("test/model")
        spectral_dir = local_dir / "spectral"
        spectral_dir.mkdir(parents=True, exist_ok=True)
        (spectral_dir / "spectral_config.json").write_text("{}")
        manager = ModelManager(registry, mock_store)
        result = manager._find_spectral_dir("test/model", "spectral:4")
        assert result == spectral_dir

    def test_find_spectral_dir_raises_when_no_data_and_auto_calibrate_off(
        self, registry, mock_store, monkeypatch
    ):
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.kv_cache_auto_calibrate", False
        )
        manager = ModelManager(registry, mock_store)
        with pytest.raises(SpectralCalibrationMissingError, match="Run 'olmlx"):
            manager._find_spectral_dir("test/model", "spectral:4")

    def test_auto_calibrate_spectral_on_success(
        self, tmp_path, registry, mock_store, monkeypatch
    ):
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "kv_cache_quant", "spectral:4")
        monkeypatch.setattr(_settings, "kv_cache_auto_calibrate", True)
        # Set up a spectral dir where calibrate_model would write output
        local_dir = mock_store.local_path("test/model")
        expected_output = local_dir / "spectral"
        expected_output.mkdir(parents=True, exist_ok=True)
        (expected_output / "spectral_config.json").write_text("{}")
        with patch(
            "olmlx.engine.spectralquant_calibrate.calibrate_model",
            return_value=str(expected_output),
        ):
            manager = ModelManager(registry, mock_store)
            result = manager._auto_calibrate_spectral("test/model", "spectral:4")
            assert result == expected_output

    def test_auto_calibrate_spectral_raises_on_calibration_failure(
        self, tmp_path, registry, mock_store, monkeypatch
    ):
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "kv_cache_quant", "spectral:4")
        monkeypatch.setattr(_settings, "kv_cache_auto_calibrate", True)
        with patch(
            "olmlx.engine.spectralquant_calibrate.calibrate_model",
            side_effect=RuntimeError("GPU out of memory"),
        ):
            manager = ModelManager(registry, mock_store)
            with pytest.raises(
                SpectralCalibrationMissingError, match="Auto-calibration failed"
            ):
                manager._auto_calibrate_spectral("test/model", "spectral:4")

    def test_auto_calibrate_spectral_raises_when_output_missing(
        self, tmp_path, registry, mock_store, monkeypatch
    ):
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "kv_cache_quant", "spectral:4")
        monkeypatch.setattr(_settings, "kv_cache_auto_calibrate", True)
        # calibrate_model returns a path that doesn't contain spectral_config.json
        fake_output = tmp_path / "spectral"
        fake_output.mkdir(exist_ok=True)  # dir exists but no config file
        with patch(
            "olmlx.engine.spectralquant_calibrate.calibrate_model",
            return_value=str(fake_output),
        ):
            manager = ModelManager(registry, mock_store)
            with pytest.raises(
                SpectralCalibrationMissingError,
                match="spectral data not found",
            ):
                manager._auto_calibrate_spectral("test/model", "spectral:4")

    def test_find_spectral_dir_triggers_auto_calibrate(
        self, tmp_path, registry, mock_store, monkeypatch
    ):
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "kv_cache_quant", "spectral:4")
        monkeypatch.setattr(_settings, "kv_cache_auto_calibrate", True)
        local_dir = mock_store.local_path("test/model")
        expected_output = local_dir / "spectral"
        expected_output.mkdir(parents=True, exist_ok=True)
        (expected_output / "spectral_config.json").write_text("{}")
        with patch(
            "olmlx.engine.spectralquant_calibrate.calibrate_model",
            return_value=str(expected_output),
        ):
            manager = ModelManager(registry, mock_store)
            result = manager._auto_calibrate_spectral("test/model", "spectral:4")
            assert result == expected_output

    def test_auto_calibrate_spectral_asserts_method_is_spectral(
        self, registry, mock_store
    ):
        manager = ModelManager(registry, mock_store)
        with pytest.raises(AssertionError):
            manager._auto_calibrate_spectral("test/model", "turboquant:4")


class TestWhisperLoad:
    def test_load_model_whisper_branch(self, tmp_path, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        fake_whisper = MagicMock()
        # Pre-create the local model dir so ensure_downloaded() takes the
        # is_downloaded() short-circuit. Without it this test reached the
        # real ``snapshot_download("test/whisper")`` over the network
        # (caught by the real-model guard, #470).
        local_dir = mock_store.local_path("test/whisper")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}")
        with (
            patch.object(manager, "_detect_model_kind", return_value="whisper"),
            patch(
                "mlx_whisper.load_models.load_model", return_value=fake_whisper
            ) as mock_load,
        ):
            model, tok, is_vlm, caps, spec = manager._load_model("test/whisper")
        assert model is fake_whisper
        assert tok is None
        assert is_vlm is False
        assert spec is None
        mock_load.assert_called_once()

    def test_probe_cache_skipped_for_whisper(self, registry, mock_store):
        from olmlx.engine.model_manager import LoadedModel

        manager = ModelManager(registry, mock_store)
        lm = LoadedModel(
            name="whisper:latest",
            hf_path="test/whisper",
            model=MagicMock(),
            tokenizer=None,
            is_whisper=True,
        )
        lm.supports_cache_trim = True
        lm.supports_cache_persistence = False
        with patch("mlx_lm.models.cache.make_prompt_cache") as mock_make:
            manager._probe_cache_capabilities(lm)
        mock_make.assert_not_called()

    def test_close_clears_whisper_model_holder(self, registry, mock_store):
        import importlib

        from olmlx.engine.model_manager import LoadedModel, ModelManager

        whisper_transcribe = importlib.import_module("mlx_whisper.transcribe")
        manager = ModelManager(registry, mock_store)
        sentinel = MagicMock()
        lm = LoadedModel(
            name="whisper:latest",
            hf_path="test/whisper",
            model=sentinel,
            tokenizer=None,
            is_whisper=True,
        )
        whisper_transcribe.ModelHolder.model = sentinel
        whisper_transcribe.ModelHolder.model_path = "test/whisper"

        manager._close_loaded_model(lm)

        assert whisper_transcribe.ModelHolder.model is None
        assert whisper_transcribe.ModelHolder.model_path is None

    def test_close_preserves_other_whisper_in_holder(self, registry, mock_store):
        import importlib

        from olmlx.engine.model_manager import LoadedModel, ModelManager

        whisper_transcribe = importlib.import_module("mlx_whisper.transcribe")
        manager = ModelManager(registry, mock_store)
        other = MagicMock()
        lm = LoadedModel(
            name="whisper:latest",
            hf_path="test/whisper",
            model=MagicMock(),
            tokenizer=None,
            is_whisper=True,
        )
        # Holder references a DIFFERENT model than the one being closed.
        whisper_transcribe.ModelHolder.model = other
        whisper_transcribe.ModelHolder.model_path = "other/whisper"

        manager._close_loaded_model(lm)

        assert whisper_transcribe.ModelHolder.model is other


class TestBuildSpeculativeDecoder:
    """Unit tests for the consolidated ``_build_speculative_decoder`` dispatch.

    This helper replaces three near-identical ``spec_config.strategy``
    if/elif chains in ``_load_model`` (flash-MoE, VLM, and text paths). Each
    case pins that a given strategy routes to the right ``_load_*_decoder``
    method with the right arguments, including the per-path variations:
    ``is_vlm`` (pld / classic) and ``unwrap_language_model`` (self-speculative
    target unwrap on the VLM path).
    """

    def _manager(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        # Replace every concrete loader with a recording stub returning a
        # unique sentinel so we can assert both dispatch target and that the
        # helper propagates the loader's return value unchanged.
        for name in (
            "_load_dflash_decoder",
            "_load_eagle_decoder",
            "_load_mtp_decoder",
            "_load_pld_decoder",
            "_load_self_speculative_decoder",
            "_load_proxy_tuning_decoder",
            "_load_speculative_decoder",
        ):
            setattr(manager, name, MagicMock(return_value=f"{name}-result"))
        return manager

    def test_dflash(self, registry, mock_store):
        manager = self._manager(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="dflash"
        )
        result = manager._build_speculative_decoder(model, "hf/path", cfg)
        manager._load_dflash_decoder.assert_called_once_with(model, cfg)
        assert result == "_load_dflash_decoder-result"

    def test_eagle(self, registry, mock_store):
        manager = self._manager(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="eagle"
        )
        result = manager._build_speculative_decoder(model, "hf/path", cfg)
        manager._load_eagle_decoder.assert_called_once_with(model, cfg)
        assert result == "_load_eagle_decoder-result"

    def test_mtp(self, registry, mock_store):
        manager = self._manager(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="mtp"
        )
        result = manager._build_speculative_decoder(model, "hf/path", cfg)
        manager._load_mtp_decoder.assert_called_once_with(model, cfg)
        assert result == "_load_mtp_decoder-result"

    def test_pld_text(self, registry, mock_store):
        manager = self._manager(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="pld"
        )
        manager._build_speculative_decoder(model, "hf/path", cfg)
        manager._load_pld_decoder.assert_called_once_with(model, cfg, is_vlm=False)

    def test_pld_vlm(self, registry, mock_store):
        manager = self._manager(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="pld"
        )
        manager._build_speculative_decoder(model, "hf/path", cfg, is_vlm=True)
        manager._load_pld_decoder.assert_called_once_with(model, cfg, is_vlm=True)

    def test_self_speculative_no_unwrap(self, registry, mock_store):
        manager = self._manager(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="self_speculative"
        )
        manager._build_speculative_decoder(model, "hf/path", cfg)
        manager._load_self_speculative_decoder.assert_called_once_with(model, cfg)

    def test_self_speculative_unwrap_language_model(self, registry, mock_store):
        manager = self._manager(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="self_speculative"
        )
        manager._build_speculative_decoder(
            model, "hf/path", cfg, is_vlm=True, unwrap_language_model=True
        )
        manager._load_self_speculative_decoder.assert_called_once_with(
            model.language_model, cfg
        )

    def test_proxy_tuning(self, registry, mock_store):
        manager = self._manager(registry, mock_store)
        model = MagicMock()
        tok = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="proxy_tuning"
        )
        manager._build_speculative_decoder(model, "hf/path", cfg, tokenizer=tok)
        manager._load_proxy_tuning_decoder.assert_called_once_with(model, tok, cfg)

    def test_classic_text(self, registry, mock_store):
        manager = self._manager(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="classic"
        )
        manager._build_speculative_decoder(model, "hf/path", cfg)
        manager._load_speculative_decoder.assert_called_once_with(
            model, "hf/path", cfg, is_vlm=False
        )

    def test_classic_vlm(self, registry, mock_store):
        manager = self._manager(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="classic"
        )
        manager._build_speculative_decoder(model, "hf/path", cfg, is_vlm=True)
        manager._load_speculative_decoder.assert_called_once_with(
            model, "hf/path", cfg, is_vlm=True
        )


class TestProbeBundledDraft:
    """Unit tests for ``_probe_bundled_draft_dir`` pure helper."""

    def test_probe_returns_path_when_both_present(self, tmp_path):
        from olmlx.engine.speculative_loaders import _probe_bundled_draft_dir

        d = tmp_path / "dflash"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"")
        assert _probe_bundled_draft_dir(tmp_path, "dflash") == d

    def test_probe_returns_path_for_eagle(self, tmp_path):
        from olmlx.engine.speculative_loaders import _probe_bundled_draft_dir

        d = tmp_path / "eagle"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "weights.safetensors").write_bytes(b"")
        assert _probe_bundled_draft_dir(tmp_path, "eagle") == d

    def test_probe_returns_none_when_dir_missing(self, tmp_path):
        from olmlx.engine.speculative_loaders import _probe_bundled_draft_dir

        assert _probe_bundled_draft_dir(tmp_path, "dflash") is None

    def test_probe_returns_none_when_config_missing(self, tmp_path):
        from olmlx.engine.speculative_loaders import _probe_bundled_draft_dir

        d = tmp_path / "dflash"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"")
        assert _probe_bundled_draft_dir(tmp_path, "dflash") is None

    def test_probe_returns_none_when_weights_missing(self, tmp_path):
        from olmlx.engine.speculative_loaders import _probe_bundled_draft_dir

        d = tmp_path / "dflash"
        d.mkdir()
        (d / "config.json").write_text("{}")
        assert _probe_bundled_draft_dir(tmp_path, "dflash") is None


class TestBuildSpeculativeDecoderBundledProbe:
    """Integration tests: ``_build_speculative_decoder`` auto-detects bundled draft."""

    def _manager_with_mocks(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        for name in (
            "_load_dflash_decoder",
            "_load_eagle_decoder",
            "_load_speculative_decoder",
        ):
            setattr(manager, name, MagicMock(return_value=f"{name}-result"))
        return manager

    def test_build_speculative_uses_bundled_draft_when_unset(
        self, tmp_path, registry, mock_store
    ):
        """When draft_model is None and a bundled subdir is present, it is used."""
        bundled = tmp_path / "dflash"
        bundled.mkdir()
        (bundled / "config.json").write_text("{}")
        (bundled / "model-00001-of-00001.safetensors").write_bytes(b"")

        mock_store.local_path = MagicMock(return_value=tmp_path)

        manager = self._manager_with_mocks(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="dflash"
        )
        manager._build_speculative_decoder(model, "ns/model", cfg)

        call_args = manager._load_dflash_decoder.call_args
        used_cfg = call_args[0][1]
        assert used_cfg.draft_model == str(bundled)

    def test_build_speculative_explicit_draft_wins_over_bundled(
        self, tmp_path, registry, mock_store
    ):
        """Explicit draft_model path takes precedence over a bundled probe."""
        bundled = tmp_path / "dflash"
        bundled.mkdir()
        (bundled / "config.json").write_text("{}")
        (bundled / "model.safetensors").write_bytes(b"")

        mock_store.local_path = MagicMock(return_value=tmp_path)

        manager = self._manager_with_mocks(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True,
            draft_model="/explicit/path",
            num_tokens=None,
            strategy="dflash",
        )
        manager._build_speculative_decoder(model, "ns/model", cfg)

        call_args = manager._load_dflash_decoder.call_args
        used_cfg = call_args[0][1]
        assert used_cfg.draft_model == "/explicit/path"

    def test_build_speculative_no_probe_for_classic_strategy(
        self, tmp_path, registry, mock_store
    ):
        """The bundled-probe is never run for the 'classic' strategy."""
        bundled = tmp_path / "dflash"
        bundled.mkdir()
        (bundled / "config.json").write_text("{}")
        (bundled / "model.safetensors").write_bytes(b"")

        mock_store.local_path = MagicMock(return_value=tmp_path)

        manager = self._manager_with_mocks(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="classic"
        )
        manager._build_speculative_decoder(model, "ns/model", cfg)

        # classic routes to _load_speculative_decoder, not dflash
        manager._load_speculative_decoder.assert_called_once()
        manager._load_dflash_decoder.assert_not_called()
        # The cfg passed through should be unchanged (draft_model still None)
        call_args = manager._load_speculative_decoder.call_args
        used_cfg = call_args[0][2]
        assert used_cfg.draft_model is None

    def test_build_speculative_uses_bundled_draft_eagle(
        self, tmp_path, registry, mock_store
    ):
        """Bundled probe also works for the eagle strategy."""
        bundled = tmp_path / "eagle"
        bundled.mkdir()
        (bundled / "config.json").write_text("{}")
        (bundled / "weights.safetensors").write_bytes(b"")

        mock_store.local_path = MagicMock(return_value=tmp_path)

        manager = self._manager_with_mocks(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="eagle"
        )
        manager._build_speculative_decoder(model, "ns/model", cfg)

        call_args = manager._load_eagle_decoder.call_args
        used_cfg = call_args[0][1]
        assert used_cfg.draft_model == str(bundled)

    def test_build_speculative_absolute_hf_path_uses_probe(
        self, tmp_path, registry, mock_store
    ):
        """For absolute hf_path, the probe uses the path directly (no store lookup)."""
        target_dir = tmp_path / "local_model"
        target_dir.mkdir()
        bundled = target_dir / "dflash"
        bundled.mkdir()
        (bundled / "config.json").write_text("{}")
        (bundled / "model.safetensors").write_bytes(b"")

        mock_local_path = MagicMock(return_value=tmp_path)
        mock_store.local_path = mock_local_path

        manager = self._manager_with_mocks(registry, mock_store)
        model = MagicMock()
        cfg = SpeculativeConfig(
            enabled=True, draft_model=None, num_tokens=None, strategy="dflash"
        )
        manager._build_speculative_decoder(model, str(target_dir), cfg)

        call_args = manager._load_dflash_decoder.call_args
        used_cfg = call_args[0][1]
        assert used_cfg.draft_model == str(bundled)
        # store.local_path must NOT have been called for absolute hf_path
        mock_local_path.assert_not_called()
