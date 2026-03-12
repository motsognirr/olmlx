"""Tests for olmlx.config."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from olmlx.config import Settings


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

    def test_anthropic_models_rejects_colon_in_key(self, monkeypatch):
        monkeypatch.setenv(
            "OLMLX_ANTHROPIC_MODELS",
            '{"sonnet:latest": "qwen3:latest"}',
        )
        with pytest.raises(ValidationError, match="single segment"):
            Settings()
