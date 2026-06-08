"""Live TTS test against real Kokoro via the OpenAI SDK (#367).

Drives `AsyncOpenAI` over an in-process ASGI transport (no real server) so the
SDK's own client exercises our /v1/audio/speech surface across every output
format — the acceptance criterion for #367.

Outside tests/integration/ on purpose (that package's autouse
mock_mlx_primitives would replace the real model). real_model; skipped in CI
(`-m "not real_model"`), when the model isn't downloaded, and when ffmpeg is
missing (required for the compressed formats).
"""

import json
import shutil

import pytest

from olmlx.config import settings

KOKORO = "prince-canuma/Kokoro-82M"


def _model_present() -> bool:
    from olmlx.models.store import _safe_dir_name

    return (settings.models_dir / _safe_dir_name(KOKORO) / "config.json").exists()


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required"),
    pytest.mark.skipif(
        not _model_present(),
        reason=f"{KOKORO} not downloaded in {settings.models_dir}",
    ),
]


@pytest.fixture
async def sdk_client(tmp_path, monkeypatch):
    """Real app + ModelManager, exposed via the real OpenAI AsyncOpenAI SDK
    bound to an in-process ASGI transport."""
    models_config = tmp_path / "models.json"
    models_config.write_text(json.dumps({}))
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text("{}")
    monkeypatch.setattr("olmlx.config.settings.models_config", models_config)
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_config)

    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.registry import ModelRegistry
    from olmlx.models.store import ModelStore

    registry = ModelRegistry()
    registry._aliases_path = aliases_path
    registry.load()
    store = ModelStore(registry)
    manager = ModelManager(registry, store)
    manager.start_expiry_checker()

    from olmlx.app import create_app

    app = create_app()
    app.state.registry = registry
    app.state.model_manager = manager
    app.state.model_store = store

    import httpx
    from openai import AsyncOpenAI

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = AsyncOpenAI(
        base_url="http://test/v1", api_key="not-needed", http_client=http_client
    )
    try:
        yield client
    finally:
        await http_client.aclose()
        await manager.stop()


@pytest.mark.parametrize("fmt", ["mp3", "wav", "pcm", "opus", "aac", "flac"])
async def test_speech_create_all_formats(sdk_client, fmt):
    resp = await sdk_client.audio.speech.create(
        model=KOKORO,
        voice="alloy",
        input="Hello from olmlx.",
        response_format=fmt,
    )
    data = resp.content if hasattr(resp, "content") else resp.read()
    assert len(data) > 0


async def test_speech_streaming_first_chunk(sdk_client):
    async with sdk_client.audio.speech.with_streaming_response.create(
        model=KOKORO,
        voice="af_heart",
        input="Streaming first chunk latency check.",
        response_format="pcm",
    ) as resp:
        total = 0
        async for chunk in resp.iter_bytes():
            total += len(chunk)
            if total > 0:
                break
        assert total > 0
