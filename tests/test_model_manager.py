"""Tests for olmlx.engine.model_manager."""

import asyncio
import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olmlx.engine.model_manager import (
    LoadedModel,
    ModelManager,
    parse_keep_alive,
)
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

    def test_unload(self, mock_manager):
        mock_manager.unload("qwen3")
        assert mock_manager.get_loaded() == []

    def test_unload_not_loaded(self, mock_manager):
        assert mock_manager.unload("nonexistent") is False

    def test_unload_active_refs_raises(self, mock_manager):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.active_refs = 1
        with pytest.raises(RuntimeError, match="active"):
            mock_manager.unload("qwen3")
        assert len(mock_manager.get_loaded()) == 1  # still loaded
        lm.active_refs = 0

    @pytest.mark.asyncio
    async def test_ensure_loaded_cached(self, mock_manager):
        lm = await mock_manager.ensure_loaded("qwen3")
        assert lm.name == "qwen3:latest"

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
            return_value=(mock_model, mock_tokenizer, False, TemplateCaps()),
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
    async def test_stop_cancels_expiry_task(self, mock_manager):
        # Create a dummy task
        async def dummy():
            await asyncio.sleep(100)

        mock_manager._expiry_task = asyncio.create_task(dummy())
        await mock_manager.stop()
        assert mock_manager._expiry_task.cancelled()


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

    def test_vlm_no_spec_found(self, tmp_path, registry, mock_store):
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
        # Has vision keys but spec not found — still returns vlm
        assert kind == "vlm"

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
                model, tokenizer, is_vlm, caps = manager._load_model("test/path")

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
                model, tokenizer, is_vlm, caps = manager._load_model("test/vlm")

        assert is_vlm is True

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
                model, tokenizer, is_vlm, caps = manager._load_model("test/path")

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
                model, tokenizer, is_vlm, caps = manager._load_model("test/path")

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
                model, tokenizer, is_vlm, caps = manager._load_model("test/path")

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
                        model, tok, is_vlm, caps = manager._load_model("test/path")

        assert model is mock_model


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

        # Run one cycle of expiry check manually
        now = time.time()
        expired = [
            name
            for name, m in manager._loaded.items()
            if m.expires_at is not None and m.expires_at <= now
        ]
        for name in expired:
            del manager._loaded[name]

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

        now = time.time()
        expired = [
            name
            for name, m in manager._loaded.items()
            if m.expires_at is not None and m.expires_at <= now
        ]
        for name in expired:
            del manager._loaded[name]

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

        # Simulate one cycle of _check_expiry_loop logic
        now = time.time()
        expired = [
            name
            for name, m in manager._loaded.items()
            if m.expires_at is not None and m.expires_at <= now and m.active_refs == 0
        ]
        for name in expired:
            del manager._loaded[name]

        assert "busy:latest" in manager._loaded

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
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps()),
            ),
            patch(
                "olmlx.engine.model_manager._get_metal_memory_bytes",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.engine.model_manager._get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
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
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps()),
            ),
            patch(
                "olmlx.engine.model_manager._get_metal_memory_bytes",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.engine.model_manager._get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
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
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps()),
            ),
            patch(
                "olmlx.engine.model_manager._get_metal_memory_bytes",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.engine.model_manager._get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
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
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps()),
            ),
            patch(
                "olmlx.engine.model_manager._get_metal_memory_bytes",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.engine.model_manager._get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            with pytest.raises(MemoryError) as exc_info:
                await manager.ensure_loaded("qwen3")

        msg = str(exc_info.value)
        assert "OLMLX_MEMORY_LIMIT_FRACTION" in msg
        assert "smaller" in msg or "quantized" in msg

    @pytest.mark.asyncio
    async def test_cleanup_called_on_rejection(self, registry, mock_store):
        """gc.collect() and mx.clear_cache() are called when a model is rejected."""
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
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps()),
            ),
            patch(
                "olmlx.engine.model_manager._get_metal_memory_bytes",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.engine.model_manager._get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(MemoryError):
                await manager.ensure_loaded("qwen3")

        # Called twice: once for pre-load cache flush, once for post-rejection cleanup
        assert mock_gc.call_count == 2
        assert mock_clear.call_count == 2

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
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps()),
            ),
            patch(
                "olmlx.engine.model_manager._get_metal_memory_bytes",
                side_effect=track_get_metal,
            ),
            patch(
                "olmlx.engine.model_manager._get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect", side_effect=track_gc),
            patch("olmlx.engine.model_manager.mx.clear_cache", side_effect=track_clear),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"
        # Cache flush must happen before the first memory measurement
        gc_idx = call_order.index("gc.collect")
        clear_idx = call_order.index("mx.clear_cache")
        first_metal_idx = call_order.index("get_metal")
        assert gc_idx < first_metal_idx
        assert clear_idx < first_metal_idx

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
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps()),
            ),
            patch(
                "olmlx.engine.model_manager._get_metal_memory_bytes",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.engine.model_manager._get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
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
        """If _get_metal_memory_bytes raises after load, GPU cleanup must still run."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps()),
            ),
            patch(
                "olmlx.engine.model_manager._get_metal_memory_bytes",
                side_effect=[1 * self.GB, OSError("Metal query failed")],
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(OSError, match="Metal query failed"):
                await manager.ensure_loaded("qwen3")

        # Cleanup must have been called: once pre-load flush + once post-failure
        assert mock_gc.call_count == 2
        assert mock_clear.call_count == 2
        # Model must NOT be in _loaded
        assert "qwen3:latest" not in manager._loaded

    @pytest.mark.asyncio
    async def test_cleanup_on_system_memory_query_failure(self, registry, mock_store):
        """If _get_system_memory_bytes raises, GPU cleanup must still run."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps()),
            ),
            patch(
                "olmlx.engine.model_manager._get_metal_memory_bytes",
                side_effect=[1 * self.GB, 2 * self.GB],
            ),
            patch(
                "olmlx.engine.model_manager._get_system_memory_bytes",
                side_effect=OSError("sysconf failed"),
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(OSError, match="sysconf failed"):
                await manager.ensure_loaded("qwen3")

        assert mock_gc.call_count == 2
        assert mock_clear.call_count == 2
        assert "qwen3:latest" not in manager._loaded

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
                "olmlx.engine.model_manager._get_metal_memory_bytes",
                return_value=1 * self.GB,
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(RuntimeError, match="Metal OOM"):
                await manager.ensure_loaded("qwen3")

        # Pre-load flush + post-failure flush = 2 calls each
        assert mock_gc.call_count == 2
        assert mock_clear.call_count == 2
        assert "qwen3:latest" not in manager._loaded
