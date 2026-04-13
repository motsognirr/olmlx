"""Tests for olmlx.config."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from olmlx.config import ExperimentalSettings, Settings, resolve_experimental


class TestSettings:
    def test_defaults(self, monkeypatch):
        # Clear any env vars
        for key in (
            "OLMLX_HOST",
            "OLMLX_PORT",
            "OLMLX_MODELS_DIR",
            "OLMLX_MODELS_CONFIG",
            "OLMLX_DEFAULT_KEEP_ALIVE",
            "OLMLX_MAX_LOADED_MODELS",
            "OLMLX_MODEL_LOAD_TIMEOUT",
        ):
            monkeypatch.delenv(key, raising=False)
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 11434
        assert s.default_keep_alive == "5m"
        assert s.max_loaded_models == 1
        assert s.model_load_timeout is None
        assert isinstance(s.models_dir, Path)
        assert s.models_config == Path.home() / ".olmlx" / "models.json"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_HOST", "127.0.0.1")
        monkeypatch.setenv("OLMLX_PORT", "8080")
        monkeypatch.setenv("OLMLX_MAX_LOADED_MODELS", "3")
        monkeypatch.setenv("OLMLX_MODEL_LOAD_TIMEOUT", "120")
        s = Settings()
        assert s.host == "127.0.0.1"
        assert s.port == 8080
        assert s.max_loaded_models == 3
        assert s.model_load_timeout == 120.0

    def test_model_load_timeout_rejects_zero(self, monkeypatch):
        monkeypatch.setenv("OLMLX_MODEL_LOAD_TIMEOUT", "0")
        with pytest.raises(ValidationError):
            Settings()

    def test_model_load_timeout_rejects_negative(self, monkeypatch):
        monkeypatch.setenv("OLMLX_MODEL_LOAD_TIMEOUT", "-1")
        with pytest.raises(ValidationError):
            Settings()

    def test_anthropic_models_default(self, monkeypatch):
        monkeypatch.delenv("OLMLX_ANTHROPIC_MODELS", raising=False)
        s = Settings()
        assert s.anthropic_models == {}

    def test_anthropic_models_from_env(self, monkeypatch):
        monkeypatch.setenv(
            "OLMLX_ANTHROPIC_MODELS",
            '{"haiku": "qwen3:latest", "sonnet": "qwen3-8b:latest"}',
        )
        s = Settings()
        assert s.anthropic_models == {
            "haiku": "qwen3:latest",
            "sonnet": "qwen3-8b:latest",
        }

    def test_anthropic_models_rejects_dash_in_key(self, monkeypatch):
        monkeypatch.setenv(
            "OLMLX_ANTHROPIC_MODELS",
            '{"claude-sonnet": "qwen3:latest"}',
        )
        with pytest.raises(ValidationError, match="single segment"):
            Settings()

    def test_port_rejects_zero(self, monkeypatch):
        monkeypatch.setenv("OLMLX_PORT", "0")
        with pytest.raises(ValidationError):
            Settings()

    def test_port_rejects_negative(self, monkeypatch):
        monkeypatch.setenv("OLMLX_PORT", "-1")
        with pytest.raises(ValidationError):
            Settings()

    def test_port_rejects_above_65535(self, monkeypatch):
        monkeypatch.setenv("OLMLX_PORT", "65536")
        with pytest.raises(ValidationError):
            Settings()

    def test_port_accepts_boundary_values(self, monkeypatch):
        monkeypatch.setenv("OLMLX_PORT", "1")
        s = Settings()
        assert s.port == 1

        monkeypatch.setenv("OLMLX_PORT", "65535")
        s = Settings()
        assert s.port == 65535

    def test_anthropic_models_rejects_colon_in_key(self, monkeypatch):
        monkeypatch.setenv(
            "OLMLX_ANTHROPIC_MODELS",
            '{"sonnet:latest": "qwen3:latest"}',
        )
        with pytest.raises(ValidationError, match="single segment"):
            Settings()


class TestResolveExperimental:
    def test_empty_overrides_returns_global(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", raising=False)
        base = ExperimentalSettings()
        result = resolve_experimental(base, {})
        assert result.flash == base.flash
        assert result.kv_cache_quant == base.kv_cache_quant

    def test_flash_override(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH", raising=False)
        base = ExperimentalSettings()
        assert base.flash is False
        result = resolve_experimental(base, {"flash": True})
        assert result.flash is True
        # Original should be unchanged
        assert base.flash is False

    def test_kv_cache_quant_override(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", raising=False)
        base = ExperimentalSettings()
        result = resolve_experimental(base, {"kv_cache_quant": "turboquant:4"})
        assert result.kv_cache_quant == "turboquant:4"

    def test_partial_override_preserves_other_fields(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD", raising=False)
        base = ExperimentalSettings()
        result = resolve_experimental(base, {"flash": True})
        # flash changed, but sparsity_threshold stays at default
        assert result.flash is True
        assert result.flash_sparsity_threshold == base.flash_sparsity_threshold

    def test_multiple_overrides(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD", raising=False)
        base = ExperimentalSettings()
        result = resolve_experimental(
            base,
            {
                "flash": True,
                "flash_sparsity_threshold": 0.3,
                "flash_moe": True,
            },
        )
        assert result.flash is True
        assert result.flash_sparsity_threshold == 0.3
        assert result.flash_moe is True

    def test_env_var_does_not_leak_into_overrides(self, monkeypatch):
        """resolve_experimental should not re-read env vars for unrelated fields."""
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", raising=False)
        base = ExperimentalSettings()
        # Set a bad env var *after* base is created
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", "bad_value")
        # Overriding only 'flash' should not trigger kv_cache_quant env var parsing
        result = resolve_experimental(base, {"flash": True})
        assert result.flash is True
        assert result.kv_cache_quant is None


class TestSpeculativeConfig:
    """Tests for standalone speculative decoding config fields."""

    def test_speculative_defaults(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_SPECULATIVE", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_SPECULATIVE_TOKENS", raising=False)
        s = ExperimentalSettings()
        assert s.speculative is False
        assert s.speculative_draft_model is None
        assert s.speculative_tokens == 4

    def test_speculative_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_SPECULATIVE", "true")
        monkeypatch.setenv(
            "OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL", "Qwen/Qwen3-0.6B"
        )
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_SPECULATIVE_TOKENS", "6")
        s = ExperimentalSettings()
        assert s.speculative is True
        assert s.speculative_draft_model == "Qwen/Qwen3-0.6B"
        assert s.speculative_tokens == 6

    def test_speculative_tokens_rejects_zero(self):
        with pytest.raises(ValidationError):
            ExperimentalSettings(speculative_tokens=0, _env_file=None)

    def test_speculative_per_model_override(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_SPECULATIVE", raising=False)
        base = ExperimentalSettings()
        result = resolve_experimental(
            base,
            {"speculative": True, "speculative_draft_model": "Qwen/Qwen3-0.6B"},
        )
        assert result.speculative is True
        assert result.speculative_draft_model == "Qwen/Qwen3-0.6B"
        assert base.speculative is False


class TestDFlashConfig:
    """Tests for dflash config fields."""

    def test_dflash_defaults(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_DFLASH", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_DFLASH_DRAFT_MODEL", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_DFLASH_BLOCK_SIZE", raising=False)
        s = ExperimentalSettings()
        assert s.dflash is False
        assert s.dflash_draft_model is None
        assert s.dflash_block_size == 4

    def test_dflash_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DFLASH", "true")
        monkeypatch.setenv(
            "OLMLX_EXPERIMENTAL_DFLASH_DRAFT_MODEL", "aryagm/dflash-qwen3"
        )
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DFLASH_BLOCK_SIZE", "8")
        s = ExperimentalSettings()
        assert s.dflash is True
        assert s.dflash_draft_model == "aryagm/dflash-qwen3"
        assert s.dflash_block_size == 8

    def test_dflash_block_size_rejects_zero(self):
        with pytest.raises(ValidationError):
            ExperimentalSettings(dflash_block_size=0, _env_file=None)
