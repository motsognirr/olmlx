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
    ModelLoadTimeoutError,
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
        exp.flash_moe = True
        exp.flash_moe_io_threads = 4
        exp.flash_moe_cache_budget_experts = 16
        exp.kv_cache_quant = None
        return exp

    def test_flash_moe_falls_back_to_vlm_on_unsupported_model_type(
        self, registry, mock_store
    ):
        """When mlx-lm can't load the model (e.g. gemma4), fall back to mlx-vlm."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/moe-vlm")
        flash_moe_dir = self._make_flash_moe_dir(mock_store, "test/moe-vlm")
        model_exp = self._mock_model_exp()

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
                            model_exp=model_exp,
                        )

        assert is_vlm is True
        mock_mlx_vlm.load.assert_called_once()

    def test_flash_moe_uses_language_model_from_vlm(self, registry, mock_store):
        """VLM fallback should extract language_model for the MoE wrapper."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/moe-vlm2")
        flash_moe_dir = self._make_flash_moe_dir(mock_store, "test/moe-vlm2")
        model_exp = self._mock_model_exp()

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
                            model_exp=model_exp,
                        )

        assert captured_model["model"] is mock_language_model

    def test_flash_moe_still_works_with_mlx_lm(self, registry, mock_store):
        """When mlx-lm succeeds, it should NOT fall back to mlx-vlm."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/moe-text")
        flash_moe_dir = self._make_flash_moe_dir(mock_store, "test/moe-text")
        model_exp = self._mock_model_exp()

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
                        model_exp=model_exp,
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
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
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
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
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
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
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
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(TimeoutError):
                await manager.ensure_loaded("qwen3")

        # Only pre-load flush — the BaseException handler skips gc/clear
        # when a deferred cleanup is pending (background thread still running).
        assert mock_gc.call_count == 1
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
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(TimeoutError):
                await manager.ensure_loaded("qwen3")

            # Only pre-load flush (BaseException handler skips when deferred)
            assert mock_gc.call_count == 1
            assert mock_clear.call_count == 1

            # Await the cleanup task directly (deterministic, no sleep needed)
            cleanup_task = manager._pending_cleanups.get("qwen3:latest")
            assert cleanup_task is not None
            await cleanup_task

            # Deferred cleanup adds one more call each
            assert mock_gc.call_count == 2
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
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
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
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
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
                "olmlx.engine.model_manager.gc.collect",
                side_effect=gc_collect_that_fails_second_time,
            ),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
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
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
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
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
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
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(MemoryError):
                await manager.ensure_loaded("qwen3")

            # gc/clear should have been called for cleanup
            assert mock_gc.call_count >= 1
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
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(RuntimeError, match="Metal OOM"):
                await manager.ensure_loaded("qwen3")

        # Pre-load flush + post-failure flush = 2 calls each
        assert mock_gc.call_count == 2
        assert mock_clear.call_count == 2
        assert "qwen3:latest" not in manager._loaded


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


class TestEvictLruIfNeeded:
    """Tests for ModelManager._evict_lru_if_needed."""

    def test_no_eviction_below_capacity(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 3)
        manager = ModelManager(registry, mock_store)
        lm = LoadedModel(
            name="a", hf_path="a/a", model=MagicMock(), tokenizer=MagicMock()
        )
        manager._loaded["a"] = lm
        manager._evict_lru_if_needed()
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
        manager._evict_lru_if_needed()
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
            manager._evict_lru_if_needed()

    def test_skips_gc_when_pending_cleanup(self, registry, mock_store, monkeypatch):
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
        # Simulate pending cleanup — use a truthy sentinel (dict only checks key presence)
        manager._pending_cleanups["other"] = True
        with patch("olmlx.engine.model_manager.gc.collect") as mock_gc:
            manager._evict_lru_if_needed()
            mock_gc.assert_not_called()
        assert "old" not in manager._loaded
        del manager._pending_cleanups["other"]

    def test_nulls_speculative_decoder(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)
        decoder = MagicMock()
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=MagicMock(),
            tokenizer=MagicMock(),
            speculative_decoder=decoder,
            loaded_at=time.time() - 100,
        )
        manager._loaded["old"] = old
        manager._evict_lru_if_needed()
        assert "old" not in manager._loaded


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
        spec_config = (True, "test/draft-model", 5)

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
        spec_config = (True, "test/draft-model", 4)

        with pytest.raises(ValueError, match="vocab_size"):
            manager._load_model(
                "test/target-model", model_exp=model_exp, spec_config=spec_config
            )

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
        spec_config = (True, None, 4)

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
            lambda hf_path, load_path, flash_dir, *, model_exp: sentinel,
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = (True, "test/draft", 4)

        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            result = manager._load_model(
                "test/flash-model", model_exp=model_exp, spec_config=spec_config
            )
        # Flash path returns its own tuple unchanged.
        assert result is sentinel
        assert "OLMLX_SPECULATIVE" in caplog.text
        assert "Flash" in caplog.text

    def test_flash_moe_path_warns_when_standalone_speculative_set(
        self, monkeypatch, caplog
    ):
        """Symmetric to the Flash test: a Flash-MoE model with the
        standalone speculative flag must log a warning."""
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
            lambda hf_path, load_path, flash_moe_dir, *, model_exp: sentinel_load,
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = (True, "test/draft", 4)

        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            model, tokenizer, is_vlm, caps, decoder = manager._load_model(
                "test/moe-model", model_exp=model_exp, spec_config=spec_config
            )
        # Flash-MoE path returns the sentinel followed by None.
        assert (model, tokenizer, is_vlm, caps) == sentinel_load
        assert decoder is None
        assert "OLMLX_SPECULATIVE" in caplog.text
        assert "Flash-MoE" in caplog.text


class TestDFlashLoading:
    """Tests for dflash decoder loading in _load_model."""

    def test_load_model_requires_dflash_draft_model(self, monkeypatch):
        """Should raise ValueError when dflash is enabled but no draft model."""
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

        model_exp = ExperimentalSettings(
            dflash=True,
            dflash_draft_model=None,
            _env_file=None,
        )

        with pytest.raises(ValueError, match="dflash_draft_model"):
            manager._load_model("test/target-model", model_exp=model_exp)

    def test_load_model_creates_dflash_decoder(self, monkeypatch):
        """When dflash is enabled, _load_model should return a DFlashDecoder."""
        from olmlx.config import ExperimentalSettings
        from olmlx.engine.dflash.decoder import DFlashDecoder

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

        model_exp = ExperimentalSettings(
            dflash=True,
            dflash_draft_model="test/dflash-draft",
            _env_file=None,
        )

        model, tok, is_vlm, caps_out, decoder = manager._load_model(
            "test/target-model", model_exp=model_exp
        )

        assert decoder is mock_decoder
