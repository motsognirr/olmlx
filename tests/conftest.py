"""Shared fixtures for olmlx tests."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from olmlx.engine.model_manager import LoadedModel, ModelManager
from olmlx.engine.registry import ModelRegistry
from olmlx.engine.template_caps import TemplateCaps
from olmlx.models.store import ModelStore


def make_error_stream(chunks, error_msg="GPU error"):
    """Create an async iterator that yields chunks then raises RuntimeError.

    Each instance has a trackable ``aclose`` mock.
    """

    class ErrorStream:
        def __init__(self):
            self._idx = 0
            self.aclose = AsyncMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._idx < len(chunks):
                chunk = chunks[self._idx]
                self._idx += 1
                return chunk
            raise RuntimeError(error_msg)

    return ErrorStream()


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
    monkeypatch.setattr(
        "olmlx.engine.registry.settings.models_config", tmp_models_config
    )
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
    monkeypatch.setattr("olmlx.models.store.settings.models_dir", tmp_path / "models")
    return ModelStore(registry)


@pytest.fixture
async def app_client(mock_manager, mock_store, registry):
    """An async test client for the FastAPI app."""
    from olmlx.app import create_app

    app = create_app()
    app.state.registry = registry
    app.state.model_manager = mock_manager
    app.state.model_store = mock_store

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
