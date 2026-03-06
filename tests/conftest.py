"""Shared fixtures for mlx-ollama tests."""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from mlx_ollama.engine.model_manager import LoadedModel, ModelManager
from mlx_ollama.engine.registry import ModelRegistry
from mlx_ollama.engine.template_caps import TemplateCaps
from mlx_ollama.models.store import ModelStore
from mlx_ollama.utils.timing import TimingStats


@pytest.fixture
def tmp_models_config(tmp_path):
    """Create a temp models.json."""
    config = {
        "qwen3:latest": "Qwen/Qwen3-8B-MLX",
        "llama3:8b": "mlx-community/Llama-3-8B-Instruct",
    }
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def registry(tmp_models_config, monkeypatch):
    """A ModelRegistry loaded from a temp config."""
    monkeypatch.setattr("mlx_ollama.engine.registry.settings.models_config", tmp_models_config)
    reg = ModelRegistry()
    reg.load()
    return reg


@pytest.fixture
def mock_loaded_model():
    """A mock LoadedModel."""
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.chat_template = "{{ messages }}{{ tools }}"
    tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    return LoadedModel(
        name="qwen3:latest",
        hf_path="Qwen/Qwen3-8B-MLX",
        model=model,
        tokenizer=tokenizer,
        is_vlm=False,
        template_caps=TemplateCaps(supports_tools=True, supports_enable_thinking=True),
    )


@pytest.fixture
def mock_manager(registry, mock_loaded_model, mock_store):
    """A ModelManager with mocked loading."""
    manager = ModelManager(registry, mock_store)
    manager._loaded["qwen3:latest"] = mock_loaded_model
    return manager


@pytest.fixture
def mock_store(registry, tmp_path, monkeypatch):
    """A ModelStore using a temp directory."""
    monkeypatch.setattr("mlx_ollama.models.store.settings.models_dir", tmp_path / "models")
    return ModelStore(registry)


@pytest.fixture
async def app_client(mock_manager, mock_store, registry):
    """An async test client for the FastAPI app."""
    from mlx_ollama.app import create_app

    app = create_app()
    app.state.registry = registry
    app.state.model_manager = mock_manager
    app.state.model_store = mock_store

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
