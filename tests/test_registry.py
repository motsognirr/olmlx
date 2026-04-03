"""Tests for olmlx.engine.registry."""

import json
import os
import stat
from unittest.mock import patch

import pytest

from olmlx.engine.registry import ModelConfig, ModelRegistry, _atomic_write_json


class TestModelRegistry:
    def test_load(self, registry):
        assert "qwen3:latest" in registry._mappings
        assert registry._mappings["qwen3:latest"].hf_path == "Qwen/Qwen3-8B-MLX"

    def test_normalize_name_no_tag(self):
        assert ModelRegistry.normalize_name("qwen3") == "qwen3:latest"

    def test_normalize_name_with_tag(self):
        assert ModelRegistry.normalize_name("qwen3:8b") == "qwen3:8b"

    def test_resolve_found(self, registry):
        assert registry.resolve("qwen3:latest").hf_path == "Qwen/Qwen3-8B-MLX"

    def test_resolve_without_tag(self, registry):
        assert registry.resolve("qwen3").hf_path == "Qwen/Qwen3-8B-MLX"

    def test_resolve_not_found(self, registry):
        assert registry.resolve("nonexistent") is None

    def test_resolve_hf_path_passthrough(self, registry):
        assert registry.resolve("org/model-name").hf_path == "org/model-name"

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
        assert registry.resolve("my-model").hf_path == "Qwen/Qwen3-8B-MLX"
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

    def test_remove_mapping(self, registry, tmp_path, monkeypatch):
        models_json = tmp_path / "models_rm.json"
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_json)
        registry._aliases_path = tmp_path / "aliases.json"
        registry.add_mapping("removable", "org/removable-model")
        assert registry.resolve("removable").hf_path == "org/removable-model"
        registry.remove("removable")
        assert registry.resolve("removable") is None

    def test_alias_priority_over_mapping(self, registry, tmp_path):
        registry._aliases_path = tmp_path / "aliases.json"
        registry._aliases["qwen3:latest"] = "custom/override"
        assert registry.resolve("qwen3").hf_path == "custom/override"

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
        assert registry.resolve("new-model").hf_path == "org/new-model-MLX"
        # Should be persisted
        assert models_json.exists()

        saved = json.loads(models_json.read_text())
        assert saved["new-model:latest"] == "org/new-model-MLX"

    def test_add_mapping_idempotent(self, registry, tmp_path, monkeypatch):
        models_json = tmp_path / "models2.json"
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_json)
        registry.add_mapping("qwen3", "Qwen/Qwen3-8B-MLX")
        # Should not raise, mapping already exists

    def test_validate_model_name_rejects_empty(self):
        from olmlx.engine.registry import validate_model_name

        with pytest.raises(ValueError, match="empty"):
            validate_model_name("")

    def test_validate_model_name_rejects_whitespace_only(self):
        from olmlx.engine.registry import validate_model_name

        with pytest.raises(ValueError, match="empty"):
            validate_model_name("   ")

    def test_validate_model_name_rejects_path_traversal(self):
        from olmlx.engine.registry import validate_model_name

        with pytest.raises(ValueError, match="path traversal"):
            validate_model_name("../etc/passwd")

    def test_validate_model_name_rejects_embedded_path_traversal(self):
        from olmlx.engine.registry import validate_model_name

        with pytest.raises(ValueError, match="path traversal"):
            validate_model_name("model/../secret")

    def test_validate_model_name_rejects_too_long(self):
        from olmlx.engine.registry import validate_model_name

        with pytest.raises(ValueError, match="256"):
            validate_model_name("a" * 257)

    def test_validate_model_name_accepts_ollama_style(self):
        from olmlx.engine.registry import validate_model_name

        validate_model_name("qwen3:8b")  # should not raise

    def test_validate_model_name_accepts_hf_style(self):
        from olmlx.engine.registry import validate_model_name

        validate_model_name("Qwen/Qwen3-8B")  # should not raise

    def test_validate_model_name_accepts_256_chars(self):
        from olmlx.engine.registry import validate_model_name

        validate_model_name("a" * 256)  # should not raise

    def test_resolve_rejects_empty_name(self, registry):
        with pytest.raises(ValueError, match="empty"):
            registry.resolve("")

    def test_resolve_rejects_path_traversal(self, registry):
        with pytest.raises(ValueError, match="path traversal"):
            registry.resolve("../etc/passwd")

    def test_add_mapping_rejects_empty_name(self, registry):
        with pytest.raises(ValueError, match="empty"):
            registry.add_mapping("", "org/model")

    def test_validate_model_name_rejects_absolute_path(self):
        from olmlx.engine.registry import validate_model_name

        with pytest.raises(ValueError, match="path traversal"):
            validate_model_name("/etc/passwd")

    def test_resolve_rejects_absolute_path(self, registry):
        with pytest.raises(ValueError, match="path traversal"):
            registry.resolve("/etc/passwd")

    def test_add_alias_rejects_empty_alias(self, registry):
        with pytest.raises(ValueError, match="empty"):
            registry.add_alias("", "qwen3:latest")

    def test_add_mapping_rejects_empty_hf_path(self, registry):
        with pytest.raises(ValueError, match="HuggingFace path must not be empty"):
            registry.add_mapping("my-model", "")

    def test_add_mapping_rejects_long_hf_path(self, registry):
        with pytest.raises(ValueError, match="512"):
            registry.add_mapping("my-model", "org/" + "a" * 510)

    def test_add_mapping_rejects_hf_path_no_slash(self, registry):
        with pytest.raises(ValueError, match="owner/repo"):
            registry.add_mapping("my-model", "localmodel")

    def test_remove_rejects_empty_name(self, registry):
        with pytest.raises(ValueError, match="empty"):
            registry.remove("")

    def test_add_mapping_rejects_hf_path_extra_slashes(self, registry):
        with pytest.raises(ValueError, match="owner/repo"):
            registry.add_mapping("my-model", "org/model/extra")

    def test_resolve_rejects_hf_path_extra_slashes(self, registry):
        with pytest.raises(ValueError, match="owner/repo"):
            registry.resolve("org/model/extra")

    def test_add_mapping_rejects_hf_path_traversal(self, registry):
        with pytest.raises(ValueError, match="HuggingFace path.*path traversal"):
            registry.add_mapping("my-model", "../evil/path")

    def test_validate_model_name_rejects_null_bytes(self):
        from olmlx.engine.registry import validate_model_name

        with pytest.raises(ValueError, match="null bytes"):
            validate_model_name("legit\x00evil")

    def test_validate_hf_path_rejects_null_bytes(self):
        from olmlx.engine.registry import validate_hf_path

        with pytest.raises(ValueError, match="null bytes"):
            validate_hf_path("org/model\x00evil")

    def test_validate_model_name_rejects_bare_dotdot(self):
        from olmlx.engine.registry import validate_model_name

        with pytest.raises(ValueError, match="path traversal"):
            validate_model_name("..")

    def test_validate_model_name_allows_double_dots_in_name(self):
        from olmlx.engine.registry import validate_model_name

        validate_model_name("gpt2..medium")  # should not raise

    def test_list_models_combines_aliases(self, registry, tmp_path):
        registry._aliases_path = tmp_path / "aliases.json"
        registry._aliases["custom:latest"] = "custom/path"
        models = registry.list_models()
        assert "custom:latest" in models
        assert "qwen3:latest" in models


class TestModelRegistrySearch:
    def test_search_exact_match(self, registry):
        """Searching 'qwen3' should return the qwen3 entry."""
        results = registry.search("qwen3")
        assert len(results) >= 1
        names = [r[0] for r in results]
        assert "qwen3:latest" in names

    def test_search_close_match(self, registry):
        """Searching 'qwem3' (typo) should still return qwen3."""
        results = registry.search("qwem3")
        assert len(results) >= 1
        names = [r[0] for r in results]
        assert "qwen3:latest" in names

    def test_search_no_match(self, registry):
        """Searching 'zzzzzzz' should return empty list."""
        results = registry.search("zzzzzzz")
        assert results == []

    def test_search_max_results(self, tmp_path, monkeypatch):
        """With many models, search respects the max_results limit."""
        config = {f"model{i}:latest": f"org/model{i}" for i in range(20)}
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg.load()
        results = reg.search("model", max_results=3)
        assert len(results) <= 3

    def test_search_base_name_without_tag(self, registry):
        """Searching 'llama3' should match 'llama3:8b'."""
        results = registry.search("llama3")
        assert len(results) >= 1
        names = [r[0] for r in results]
        assert "llama3:8b" in names


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

    def test_atomic_write_json_file_permissions(self, tmp_path):
        target = tmp_path / "data.json"
        _atomic_write_json({"a": 1}, target)
        mode = stat.S_IMODE(os.stat(target).st_mode)
        assert mode == 0o644


class TestModelConfig:
    def test_from_string(self):
        """String entries in models.json become ModelConfig with just hf_path."""
        mc = ModelConfig.from_entry("mlx-community/Llama-3.2-3B-Instruct-4bit")
        assert mc.hf_path == "mlx-community/Llama-3.2-3B-Instruct-4bit"
        assert mc.experimental == {}
        assert mc.options == {}
        assert mc.keep_alive is None

    def test_from_dict(self):
        """Dict entries in models.json become ModelConfig with full config."""
        entry = {
            "hf_path": "Qwen/Qwen3-8B-MLX",
            "experimental": {"flash": True, "kv_cache_quant": "turboquant:4"},
            "options": {"temperature": 0.7, "num_predict": 2048},
            "keep_alive": "30m",
        }
        mc = ModelConfig.from_entry(entry)
        assert mc.hf_path == "Qwen/Qwen3-8B-MLX"
        assert mc.experimental == {"flash": True, "kv_cache_quant": "turboquant:4"}
        assert mc.options == {"temperature": 0.7, "num_predict": 2048}
        assert mc.keep_alive == "30m"

    def test_from_dict_minimal(self):
        """Dict entry with only hf_path works."""
        mc = ModelConfig.from_entry({"hf_path": "org/model"})
        assert mc.hf_path == "org/model"
        assert mc.experimental == {}
        assert mc.options == {}

    def test_from_dict_missing_hf_path_raises(self):
        with pytest.raises(ValueError, match="hf_path"):
            ModelConfig.from_entry({"experimental": {"flash": True}})

    def test_from_invalid_type_raises(self):
        with pytest.raises(TypeError, match="str or dict"):
            ModelConfig.from_entry(42)

    def test_to_entry_plain(self):
        """ModelConfig with no overrides serializes to plain string."""
        mc = ModelConfig(hf_path="org/model")
        assert mc.to_entry() == "org/model"

    def test_to_entry_rich(self):
        """ModelConfig with overrides serializes to dict."""
        mc = ModelConfig(
            hf_path="org/model",
            experimental={"flash": True},
            options={"temperature": 0.5},
            keep_alive="10m",
        )
        entry = mc.to_entry()
        assert isinstance(entry, dict)
        assert entry["hf_path"] == "org/model"
        assert entry["experimental"] == {"flash": True}
        assert entry["options"] == {"temperature": 0.5}
        assert entry["keep_alive"] == "10m"

    def test_to_entry_omits_empty_sections(self):
        """Only non-empty sections are included in dict serialization."""
        mc = ModelConfig(hf_path="org/model", experimental={"flash": True})
        entry = mc.to_entry()
        assert isinstance(entry, dict)
        assert "options" not in entry
        assert "keep_alive" not in entry

    def test_invalid_experimental_key_rejected(self):
        """Unknown keys in experimental dict are rejected."""
        with pytest.raises(ValueError, match="Unknown experimental"):
            ModelConfig.from_entry({
                "hf_path": "org/model",
                "experimental": {"nonexistent_setting": True},
            })

    def test_invalid_kv_cache_quant_rejected(self):
        """Invalid kv_cache_quant value is rejected."""
        with pytest.raises(ValueError, match="kv_cache_quant"):
            ModelConfig.from_entry({
                "hf_path": "org/model",
                "experimental": {"kv_cache_quant": "bad_value"},
            })


class TestRegistryModelConfig:
    def test_load_mixed_format(self, tmp_path, monkeypatch):
        """models.json with both string and dict entries loads correctly."""
        config = {
            "llama3:latest": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "qwen3:latest": {
                "hf_path": "Qwen/Qwen3-8B-MLX",
                "experimental": {"flash": True},
                "options": {"temperature": 0.3},
            },
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg.load()
        assert isinstance(reg._mappings["llama3:latest"], ModelConfig)
        assert reg._mappings["llama3:latest"].hf_path == "mlx-community/Llama-3.2-3B-Instruct-4bit"
        assert isinstance(reg._mappings["qwen3:latest"], ModelConfig)
        assert reg._mappings["qwen3:latest"].experimental == {"flash": True}

    def test_resolve_returns_model_config(self, tmp_path, monkeypatch):
        """resolve() returns ModelConfig instead of plain string."""
        config = {"qwen3:latest": "Qwen/Qwen3-8B-MLX"}
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg.load()
        result = reg.resolve("qwen3")
        assert isinstance(result, ModelConfig)
        assert result.hf_path == "Qwen/Qwen3-8B-MLX"

    def test_resolve_rich_entry_returns_config(self, tmp_path, monkeypatch):
        """resolve() returns full ModelConfig for rich entries."""
        config = {
            "qwen3:latest": {
                "hf_path": "Qwen/Qwen3-8B-MLX",
                "experimental": {"kv_cache_quant": "turboquant:4"},
                "keep_alive": "30m",
            }
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg.load()
        result = reg.resolve("qwen3")
        assert result.hf_path == "Qwen/Qwen3-8B-MLX"
        assert result.experimental == {"kv_cache_quant": "turboquant:4"}
        assert result.keep_alive == "30m"

    def test_resolve_hf_path_passthrough_returns_model_config(self, tmp_path, monkeypatch):
        """Direct HF paths return a ModelConfig with just hf_path."""
        config_path = tmp_path / "models.json"
        config_path.write_text("{}")
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg.load()
        result = reg.resolve("org/model-name")
        assert isinstance(result, ModelConfig)
        assert result.hf_path == "org/model-name"
        assert result.experimental == {}

    def test_save_preserves_rich_config(self, tmp_path, monkeypatch):
        """Round-trip: load → save → load preserves experimental overrides."""
        config = {
            "qwen3:latest": {
                "hf_path": "Qwen/Qwen3-8B-MLX",
                "experimental": {"flash": True, "kv_cache_quant": "turboquant:4"},
                "options": {"temperature": 0.7},
            }
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg.load()
        # Trigger save by adding a new mapping
        reg.add_mapping("new-model", "org/new-model")
        # Reload and verify rich config preserved
        reg2 = ModelRegistry()
        reg2.load()
        qwen = reg2._mappings["qwen3:latest"]
        assert qwen.experimental == {"flash": True, "kv_cache_quant": "turboquant:4"}
        assert qwen.options == {"temperature": 0.7}

    def test_save_compacts_plain_models(self, tmp_path, monkeypatch):
        """Models with no overrides are saved as plain strings for readability."""
        config_path = tmp_path / "models.json"
        config_path.write_text("{}")
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg.load()
        reg.add_mapping("simple", "org/simple-model")
        saved = json.loads(config_path.read_text())
        assert saved["simple:latest"] == "org/simple-model"

    def test_list_models_returns_model_configs(self, tmp_path, monkeypatch):
        """list_models() returns dict of ModelConfig values."""
        config = {
            "llama3:latest": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "qwen3:latest": {"hf_path": "Qwen/Qwen3-8B-MLX", "experimental": {"flash": True}},
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg.load()
        models = reg.list_models()
        assert all(isinstance(v, ModelConfig) for v in models.values())

    def test_add_mapping_with_model_config(self, tmp_path, monkeypatch):
        """add_mapping() can accept a ModelConfig to store rich config."""
        config_path = tmp_path / "models.json"
        config_path.write_text("{}")
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg.load()
        mc = ModelConfig(
            hf_path="org/model",
            experimental={"flash": True},
            options={"temperature": 0.5},
        )
        reg.add_mapping("my-model", "org/model", model_config=mc)
        result = reg.resolve("my-model")
        assert result.experimental == {"flash": True}
        assert result.options == {"temperature": 0.5}

    def test_alias_resolves_to_model_config(self, tmp_path, monkeypatch):
        """Aliases resolve to the same ModelConfig as their source."""
        config = {
            "qwen3:latest": {
                "hf_path": "Qwen/Qwen3-8B-MLX",
                "experimental": {"flash": True},
            }
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg._aliases_path = tmp_path / "aliases.json"
        reg.load()
        reg.add_alias("my-qwen", "qwen3")
        result = reg.resolve("my-qwen")
        # Alias stores just the hf_path (no experimental config propagation)
        assert isinstance(result, ModelConfig)
        assert result.hf_path == "Qwen/Qwen3-8B-MLX"
