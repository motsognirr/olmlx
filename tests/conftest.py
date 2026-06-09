"""Shared fixtures for olmlx tests."""

import json
from unittest.mock import AsyncMock, MagicMock

import huggingface_hub
import mlx_lm
import mlx_vlm
import pytest
from httpx import ASGITransport, AsyncClient

import olmlx.engine.inference as _inf_mod
from olmlx.engine.model_manager import LoadedModel, ModelManager
from olmlx.engine.registry import ModelRegistry
from olmlx.engine.template_caps import TemplateCaps
from olmlx.models.store import ModelStore


@pytest.fixture(autouse=True)
def block_real_model_loads(request, monkeypatch):
    """Fail any test that reaches a real model-loading entry point
    without ``@pytest.mark.real_model`` (#470).

    CI deselects ``-m "not real_model"`` and the integration suite
    autouse-mocks MLX, so a forgotten marker either downloads gigabytes
    in CI or silently passes against a mock — the false-positive gap
    that let the VLM image-drop bug through (#429). Guarded seams:
    ``mlx_lm.load``, ``mlx_vlm.load`` (every olmlx text/VLM load goes
    through them; both modules are already imported transitively) and
    ``huggingface_hub.snapshot_download`` (every fresh weight download
    — olmlx resolves it at call time via function-local imports).
    Cached whisper/TTS weights are *not* covered; importing
    ``mlx_audio`` here would drag in spaCy.

    The single-file metadata fetches (``hf_hub_download`` for
    config.json / chat templates, ``model_info`` for model cards) get
    softer treatment: every production call site wraps them in a
    designed ``except Exception`` fallback, and dozens of unit tests
    exercise exactly those fallbacks — previously via a *real* network
    404 per test. Those entry points raise ``LocalEntryNotFoundError``
    (what genuine offline mode raises) instead of failing the test, so
    fallback tests stay meaningful, deterministic, and offline.

    Tests marked ``real_model`` get the real entry points; the
    integration suite's ``mock_mlx_primitives`` layers its mocks over
    this guard (inner fixture wins during the test, unwinds cleanly).
    """
    if request.node.get_closest_marker("real_model"):
        yield
        return

    def _blocked(entry: str):
        def _fail(*_args, **_kwargs):
            pytest.fail(
                f"{entry} was called by a test without "
                "@pytest.mark.real_model. Loading or downloading real "
                "models must be opted into via the marker so CI "
                "(-m 'not real_model') deselects it; otherwise the "
                "integration MLX mock can turn a real-model bug into a "
                "silent pass. Add @pytest.mark.real_model (and put live "
                "coverage under tests/live/), or mock the load."
            )

        _fail._olmlx_real_model_guard = True
        return _fail

    def _offline(entry: str):
        from huggingface_hub.errors import LocalEntryNotFoundError

        def _raise(*_args, **_kwargs):
            raise LocalEntryNotFoundError(
                f"{entry} blocked by the olmlx real-model guard (no "
                "@pytest.mark.real_model on this test); raising the "
                "offline-mode error so designed fallbacks engage."
            )

        _raise._olmlx_real_model_guard = True
        return _raise

    monkeypatch.setattr(mlx_lm, "load", _blocked("mlx_lm.load"))
    monkeypatch.setattr(mlx_vlm, "load", _blocked("mlx_vlm.load"))
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        _blocked("huggingface_hub.snapshot_download"),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        _offline("huggingface_hub.hf_hub_download"),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "model_info",
        _offline("huggingface_hub.model_info"),
    )
    yield


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
    # Real HF tokenizers always expose bos_token (None on models without one).
    # MagicMock would otherwise return a MagicMock here, which crashes
    # tokenize_for_cache's startswith heuristic when prompt caching runs for
    # non-streaming requests (issue #342).
    tokenizer.bos_token = None
    return LoadedModel(
        name="qwen3:latest",
        hf_path="Qwen/Qwen3-8B-MLX",
        model=model,
        tokenizer=tokenizer,
        is_vlm=False,
        template_caps=TemplateCaps(supports_tools=True, supports_enable_thinking=True),
        # supports_cache_persistence defaults to False (issue #284 safety
        # default).  This fixture stands in for a normal text model whose
        # KV cache is safe to persist, so opt in explicitly.
        supports_cache_persistence=True,
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


@pytest.fixture(autouse=True)
async def reset_inference_state():
    """Reset module-level inference state between tests for isolation."""
    await _inf_mod._reset_inference_state()
    yield
    await _inf_mod._reset_inference_state()
