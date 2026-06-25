"""Tests for LoRA-adapter support in olmlx.engine.registry (issue #362)."""

import json

import pytest

from olmlx.engine.registry import AdapterConfig, ModelConfig, ModelRegistry


def _load_registry(tmp_path, monkeypatch, data):
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps(data))
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
    reg = ModelRegistry()
    reg._aliases_path = tmp_path / "aliases.json"
    reg.load()
    return reg, config_path


BASE_CONFIG = {
    "qwen3-8b": "Qwen/Qwen3-8B-MLX",
    "adapters": {
        "qwen3-8b:my-coder-lora": {
            "base": "qwen3-8b",
            "hf_path": "acme/my-coder-lora",
        },
        "qwen3-8b:persona-lora": {
            "base": "qwen3-8b",
            "hf_path": "acme/persona-lora",
            "keep_alive": "10m",
        },
    },
}


class TestAdapterConfigParsing:
    def test_adapters_section_round_trips(self, tmp_path, monkeypatch):
        reg, _ = _load_registry(tmp_path, monkeypatch, BASE_CONFIG)
        adapters = reg.list_adapters()
        assert set(adapters) == {
            "qwen3-8b:my-coder-lora",
            "qwen3-8b:persona-lora",
        }
        cfg = adapters["qwen3-8b:my-coder-lora"]
        assert isinstance(cfg, AdapterConfig)
        assert cfg.base == "qwen3-8b"
        assert cfg.hf_path == "acme/my-coder-lora"
        assert cfg.keep_alive is None
        assert adapters["qwen3-8b:persona-lora"].keep_alive == "10m"

    def test_adapters_section_not_parsed_as_model(self, tmp_path, monkeypatch):
        reg, _ = _load_registry(tmp_path, monkeypatch, BASE_CONFIG)
        # The reserved "adapters" key must not leak into model mappings.
        assert "adapters" not in reg._mappings
        assert "adapters:latest" not in reg._mappings
        assert reg.resolve("adapters") is None

    def test_is_adapter(self, tmp_path, monkeypatch):
        reg, _ = _load_registry(tmp_path, monkeypatch, BASE_CONFIG)
        assert reg.is_adapter("qwen3-8b:my-coder-lora") is True
        # A plain base tag is not an adapter.
        assert reg.is_adapter("qwen3-8b:latest") is False
        assert reg.is_adapter("qwen3-8b") is False
        assert reg.is_adapter("nonexistent:x") is False

    def test_resolve_adapter(self, tmp_path, monkeypatch):
        reg, _ = _load_registry(tmp_path, monkeypatch, BASE_CONFIG)
        cfg = reg.resolve_adapter("qwen3-8b:my-coder-lora")
        assert isinstance(cfg, AdapterConfig)
        assert cfg.hf_path == "acme/my-coder-lora"
        assert reg.resolve_adapter("qwen3-8b:latest") is None

    def test_resolve_still_returns_base_modelconfig(self, tmp_path, monkeypatch):
        reg, _ = _load_registry(tmp_path, monkeypatch, BASE_CONFIG)
        base = reg.resolve("qwen3-8b")
        assert isinstance(base, ModelConfig)
        assert base.hf_path == "Qwen/Qwen3-8B-MLX"

    def test_invalid_adapter_entry_skipped_on_load(self, tmp_path, monkeypatch):
        data = {
            "qwen3-8b": "Qwen/Qwen3-8B-MLX",
            "adapters": {
                "qwen3-8b:good": {"base": "qwen3-8b", "hf_path": "acme/good"},
                "qwen3-8b:bad": {"hf_path": "acme/bad"},  # missing base
            },
        }
        reg, _ = _load_registry(tmp_path, monkeypatch, data)
        assert "qwen3-8b:good" in reg.list_adapters()
        assert "qwen3-8b:bad" not in reg.list_adapters()


class TestAdapterConfigValidation:
    def test_missing_base_raises(self):
        with pytest.raises(ValueError, match="base"):
            AdapterConfig.from_entry("qwen3-8b:x", {"hf_path": "acme/x"})

    def test_missing_hf_path_raises(self):
        with pytest.raises(ValueError, match="hf_path"):
            AdapterConfig.from_entry("qwen3-8b:x", {"base": "qwen3-8b"})

    def test_non_dict_entry_raises(self):
        with pytest.raises(ValueError):
            AdapterConfig.from_entry("qwen3-8b:x", "acme/x")

    def test_bad_hf_path_raises(self):
        with pytest.raises(ValueError):
            AdapterConfig.from_entry(
                "qwen3-8b:x", {"base": "qwen3-8b", "hf_path": "not-a-valid-path"}
            )


class TestAdapterPersistence:
    def test_add_adapter_mapping_persists(self, tmp_path, monkeypatch):
        reg, config_path = _load_registry(
            tmp_path, monkeypatch, {"qwen3-8b": "Qwen/Qwen3-8B-MLX"}
        )
        reg.add_adapter_mapping(
            "qwen3-8b:new-lora", base="qwen3-8b", hf_path="acme/new-lora"
        )
        assert reg.is_adapter("qwen3-8b:new-lora")
        saved = json.loads(config_path.read_text())
        assert "adapters" in saved
        assert saved["adapters"]["qwen3-8b:new-lora"] == {
            "base": "qwen3-8b",
            "hf_path": "acme/new-lora",
        }
        # Base model entry is preserved alongside the adapters section.
        assert saved["qwen3-8b"] == "Qwen/Qwen3-8B-MLX"

    def test_add_adapter_then_reload(self, tmp_path, monkeypatch):
        reg, config_path = _load_registry(
            tmp_path, monkeypatch, {"qwen3-8b": "Qwen/Qwen3-8B-MLX"}
        )
        reg.add_adapter_mapping(
            "qwen3-8b:new-lora", base="qwen3-8b", hf_path="acme/new-lora"
        )
        reg2 = ModelRegistry()
        reg2._aliases_path = tmp_path / "aliases.json"
        reg2.load()
        assert reg2.is_adapter("qwen3-8b:new-lora")
        assert reg2.resolve("qwen3-8b").hf_path == "Qwen/Qwen3-8B-MLX"

    def test_saving_models_preserves_adapters(self, tmp_path, monkeypatch):
        reg, config_path = _load_registry(tmp_path, monkeypatch, BASE_CONFIG)
        # A model-side save must not clobber the adapters section.
        reg.add_mapping("another-model", "acme/another-model")
        saved = json.loads(config_path.read_text())
        assert "adapters" in saved
        assert "qwen3-8b:my-coder-lora" in saved["adapters"]

    def test_remove_adapter(self, tmp_path, monkeypatch):
        reg, config_path = _load_registry(tmp_path, monkeypatch, BASE_CONFIG)
        reg.remove("qwen3-8b:my-coder-lora")
        assert not reg.is_adapter("qwen3-8b:my-coder-lora")
        saved = json.loads(config_path.read_text())
        assert "qwen3-8b:my-coder-lora" not in saved.get("adapters", {})
