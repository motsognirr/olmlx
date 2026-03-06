"""Tests for mlx_ollama.engine.registry."""

import json

import pytest

from mlx_ollama.engine.registry import ModelRegistry


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
        monkeypatch.setattr("mlx_ollama.engine.registry.settings.models_config",
                            tmp_path / "models.json")
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
        monkeypatch.setattr("mlx_ollama.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg.load()
        assert reg._mappings == {}

    def test_list_models_combines_aliases(self, registry, tmp_path):
        registry._aliases_path = tmp_path / "aliases.json"
        registry._aliases["custom:latest"] = "custom/path"
        models = registry.list_models()
        assert "custom:latest" in models
        assert "qwen3:latest" in models
