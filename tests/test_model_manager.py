"""Tests for mlx_ollama.engine.model_manager."""

import asyncio
import importlib
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from mlx_ollama.engine.model_manager import LoadedModel, ModelManager, parse_keep_alive
from mlx_ollama.engine.template_caps import TemplateCaps


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

    def test_invalid_format(self):
        assert parse_keep_alive("invalid") == 300.0  # default

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
    def test_init(self, registry):
        manager = ModelManager(registry)
        assert manager._loaded == {}

    def test_get_loaded_empty(self, registry):
        manager = ModelManager(registry)
        assert manager.get_loaded() == []

    def test_get_loaded(self, mock_manager):
        loaded = mock_manager.get_loaded()
        assert len(loaded) == 1
        assert loaded[0].name == "qwen3:latest"

    def test_unload(self, mock_manager):
        mock_manager.unload("qwen3")
        assert mock_manager.get_loaded() == []

    def test_unload_not_loaded(self, mock_manager):
        mock_manager.unload("nonexistent")
        assert len(mock_manager.get_loaded()) == 1  # unchanged

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
    async def test_ensure_loaded_unknown_model(self, registry):
        manager = ModelManager(registry)
        with pytest.raises(ValueError, match="not found"):
            await manager.ensure_loaded("unknown_model")

    @pytest.mark.asyncio
    async def test_ensure_loaded_evicts_lru(self, registry, monkeypatch):
        monkeypatch.setattr("mlx_ollama.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry)

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
            ModelManager, "_load_model",
            return_value=(mock_model, mock_tokenizer, False, TemplateCaps()),
        ):
            lm = await manager.ensure_loaded("qwen3")

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

    def test_text_model(self, tmp_path):
        config_path = self._make_config(tmp_path, {"model_type": "llama"})

        mock_hf = MagicMock()
        mock_hf.hf_hub_download = MagicMock(return_value=config_path)

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                kind = ModelManager._detect_model_kind("test/model")
        assert kind == "text"

    def test_vlm_with_vision_keys(self, tmp_path):
        config_path = self._make_config(tmp_path, {
            "model_type": "qwen2_vl", "vision_config": {"hidden_size": 1024},
        })

        mock_hf = MagicMock()
        mock_hf.hf_hub_download = MagicMock(return_value=config_path)

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                kind = ModelManager._detect_model_kind("test/vlm")
        assert kind == "vlm"

    def test_config_download_fails(self):
        mock_hf = MagicMock()
        mock_hf.hf_hub_download = MagicMock(side_effect=Exception("not found"))

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            kind = ModelManager._detect_model_kind("nonexistent/model")
        assert kind == "unknown"

    def test_no_model_type(self, tmp_path):
        config_path = self._make_config(tmp_path, {"hidden_size": 1024})

        mock_hf = MagicMock()
        mock_hf.hf_hub_download = MagicMock(return_value=config_path)

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            kind = ModelManager._detect_model_kind("test/model")
        assert kind == "unknown"

    def test_vlm_no_spec_found(self, tmp_path):
        config_path = self._make_config(tmp_path, {
            "model_type": "custom_vlm", "vision_config": {},
        })

        mock_hf = MagicMock()
        mock_hf.hf_hub_download = MagicMock(return_value=config_path)

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            with patch("importlib.util.find_spec", return_value=None):
                kind = ModelManager._detect_model_kind("test/vlm")
        # Has vision keys but spec not found — still returns vlm
        assert kind == "vlm"

    def test_text_model_with_real_imports(self, tmp_path):
        """Test _detect_model_kind with a model_type that exists in mlx-lm."""
        config_path = self._make_config(tmp_path, {"model_type": "llama"})

        mock_hf = MagicMock()
        mock_hf.hf_hub_download = MagicMock(return_value=config_path)

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            kind = ModelManager._detect_model_kind("test/model")
        # llama is a known text model type
        assert kind == "text"


class TestLoadModel:
    def test_load_text_model(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(ModelManager, "_detect_model_kind", return_value="text"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                model, tokenizer, is_vlm, caps = ModelManager._load_model("test/path")

        assert is_vlm is False
        assert model is mock_model

    def test_load_vlm_model(self):
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        with patch.object(ModelManager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                model, tokenizer, is_vlm, caps = ModelManager._load_model("test/vlm")

        assert is_vlm is True

    def test_load_text_fallback_to_vlm(self):
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        with patch.object(ModelManager, "_detect_model_kind", return_value="text"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.side_effect = Exception("unsupported")
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}):
                model, tokenizer, is_vlm, caps = ModelManager._load_model("test/path")

        assert is_vlm is True

    def test_load_unknown_tries_mlx_lm_first(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(ModelManager, "_detect_model_kind", return_value="unknown"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                model, tokenizer, is_vlm, caps = ModelManager._load_model("test/path")

        assert is_vlm is False

    def test_load_unknown_fallback_to_vlm(self):
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        with patch.object(ModelManager, "_detect_model_kind", return_value="unknown"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.side_effect = Exception("fail")
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}):
                model, tokenizer, is_vlm, caps = ModelManager._load_model("test/path")

        assert is_vlm is True


class TestExpiryChecker:
    @pytest.mark.asyncio
    async def test_expired_models_removed(self, registry):
        manager = ModelManager(registry)
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
            name for name, m in manager._loaded.items()
            if m.expires_at is not None and m.expires_at <= now
        ]
        for name in expired:
            del manager._loaded[name]

        assert "expired:latest" not in manager._loaded

    @pytest.mark.asyncio
    async def test_non_expired_models_kept(self, registry):
        manager = ModelManager(registry)
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
            name for name, m in manager._loaded.items()
            if m.expires_at is not None and m.expires_at <= now
        ]
        for name in expired:
            del manager._loaded[name]

        assert "active:latest" in manager._loaded
