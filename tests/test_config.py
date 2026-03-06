"""Tests for mlx_ollama.config."""

from pathlib import Path

from mlx_ollama.config import Settings


class TestSettings:
    def test_defaults(self, monkeypatch):
        # Clear any env vars
        for key in (
            "MLX_OLLAMA_HOST",
            "MLX_OLLAMA_PORT",
            "MLX_OLLAMA_MODELS_DIR",
            "MLX_OLLAMA_MODELS_CONFIG",
            "MLX_OLLAMA_DEFAULT_KEEP_ALIVE",
            "MLX_OLLAMA_MAX_LOADED_MODELS",
        ):
            monkeypatch.delenv(key, raising=False)
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 11434
        assert s.default_keep_alive == "5m"
        assert s.max_loaded_models == 1
        assert isinstance(s.models_dir, Path)
        assert s.models_config == Path.home() / ".mlx_ollama" / "models.json"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MLX_OLLAMA_HOST", "127.0.0.1")
        monkeypatch.setenv("MLX_OLLAMA_PORT", "8080")
        monkeypatch.setenv("MLX_OLLAMA_MAX_LOADED_MODELS", "3")
        s = Settings()
        assert s.host == "127.0.0.1"
        assert s.port == 8080
        assert s.max_loaded_models == 3
