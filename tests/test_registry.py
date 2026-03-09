"""Tests for olmlx.engine.registry."""

import json
from unittest.mock import patch

import pytest

from olmlx.engine.registry import ModelRegistry, _atomic_write_json


class TestModelRegistry:
    def test_load(self, registry):
        assert "qwen3:latest" in registry._mappings
        assert registry._mappings["qwen3:latest"] == "Qwen/Qwen3-8B-MLX"

    def test_normalize_name_no_tag(self):
        assert ModelRegistry.normalize_name("qwen3") == "qwen3:latest"

    def test_normalize_name_with_tag(self):
        assert ModelRegistry.normalize_name("qwen3:8b") == "qwen3:8b"

    def test_resolve_found(self, registry):
        assert registry.resolve("qwen3:latest") == "Qwen/Qwen3-8B-MLX"

    def test_resolve_without_tag(self, registry):
        assert registry.resolve("qwen3") == "Qwen/Qwen3-8B-MLX"

    def test_resolve_not_found(self, registry):
        assert registry.resolve("nonexistent") is None

    def test_resolve_hf_path_passthrough(self, registry):
        assert registry.resolve("org/model-name") == "org/model-name"

    def test_list_models(self, registry):
        models = registry.list_models()
        assert "qwen3:latest" in models
        assert "llama3:8b" in models

    def test_add_alias(self, registry, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "olmlx.engine.registry.settings.models_config",
            tmp_path / "models.json",
        )
        registry._aliases_path = tmp_path / "aliases.json"
        registry.add_alias("my-model", "qwen3:latest")
        assert registry.resolve("my-model") == "Qwen/Qwen3-8B-MLX"
        # Alias should be persisted
        assert (tmp_path / "aliases.json").exists()

    def test_add_alias_source_not_found(self, registry):
        with pytest.raises(ValueError, match="not found"):
            registry.add_alias("my-model", "nonexistent")

    def test_remove(self, registry, tmp_path):
        registry._aliases_path = tmp_path / "aliases.json"
        registry._aliases["test:latest"] = "some/path"
        registry.remove("test")
        assert "test:latest" not in registry._aliases

    def test_alias_priority_over_mapping(self, registry, tmp_path):
        registry._aliases_path = tmp_path / "aliases.json"
        registry._aliases["qwen3:latest"] = "custom/override"
        assert registry.resolve("qwen3") == "custom/override"

    def test_load_empty_config(self, tmp_path, monkeypatch):
        config_path = tmp_path / "models.json"
        # No file exists
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg.load()
        assert reg._mappings == {}

    def test_add_mapping(self, registry, tmp_path, monkeypatch):
        models_json = tmp_path / "models2.json"
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_json)
        registry.add_mapping("new-model", "org/new-model-MLX")
        assert registry.resolve("new-model") == "org/new-model-MLX"
        # Should be persisted
        assert models_json.exists()

        saved = json.loads(models_json.read_text())
        assert saved["new-model:latest"] == "org/new-model-MLX"

    def test_add_mapping_idempotent(self, registry, tmp_path, monkeypatch):
        models_json = tmp_path / "models2.json"
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_json)
        registry.add_mapping("qwen3", "Qwen/Qwen3-8B-MLX")
        # Should not raise, mapping already exists

    def test_list_models_combines_aliases(self, registry, tmp_path):
        registry._aliases_path = tmp_path / "aliases.json"
        registry._aliases["custom:latest"] = "custom/path"
        models = registry.list_models()
        assert "custom:latest" in models
        assert "qwen3:latest" in models


class TestAtomicWriteJson:
    def test_atomic_write_json_creates_valid_file(self, tmp_path):
        target = tmp_path / "data.json"
        data = {"key": "value", "nested": {"a": 1}}
        _atomic_write_json(data, target)
        assert json.loads(target.read_text()) == data

    def test_atomic_write_json_no_leftover_tmp_files(self, tmp_path):
        target = tmp_path / "data.json"
        _atomic_write_json({"a": 1}, target)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_atomic_write_json_cleans_up_on_failure(self, tmp_path):
        target = tmp_path / "data.json"
        with patch("olmlx.engine.registry.json.dump", side_effect=IOError("disk full")):
            with pytest.raises(IOError, match="disk full"):
                _atomic_write_json({"a": 1}, target)
        assert not target.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_atomic_write_json_preserves_original_on_failure(self, tmp_path):
        target = tmp_path / "data.json"
        original = {"original": True}
        target.write_text(json.dumps(original))
        with patch("olmlx.engine.registry.json.dump", side_effect=IOError("disk full")):
            with pytest.raises(IOError, match="disk full"):
                _atomic_write_json({"new": True}, target)
        assert json.loads(target.read_text()) == original
