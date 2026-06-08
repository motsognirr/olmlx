"""Live audio-input test (#426).

Loads the REAL Gemma 4 e2b VLM and verifies that a chat request carrying an
audio clip (a synthetic 440 Hz sine-tone WAV) returns a coherent (non-empty)
answer across all three API surfaces: OpenAI, Ollama, and Anthropic.

Lives OUTSIDE ``tests/integration/`` on purpose: that package has an
``autouse`` ``mock_mlx_primitives`` fixture that patches ``mlx_lm.load`` /
``mlx_lm.generate`` / ``snapshot_download``, so any test placed there runs
against mocks (never a real model).  Here only the top-level ``tests/conftest``
applies, which does not mock MLX.

Skipped in CI via ``-m "not real_model"``.  Additionally skipped when the model
is not already present in the local olmlx store, so it never triggers a
multi-GB download — run it on a machine where the model is downloaded.
"""

import base64
import io
import json
import math
import os
import struct
import wave

import pytest

from olmlx.config import settings

# Matches on-disk store dir: mlx-community_gemma-4-e2b-it-4bit
AUDIO_MODEL = "mlx-community/gemma-4-e2b-it-4bit"


def _model_present() -> bool:
    """True when the model is already downloaded in the local olmlx store."""
    from olmlx.models.store import _safe_dir_name

    return (settings.models_dir / _safe_dir_name(AUDIO_MODEL) / "config.json").exists()


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not _model_present(),
        reason=f"{AUDIO_MODEL} not downloaded in {settings.models_dir}",
    ),
]


def _wav_b64(seconds: float = 1.0, sr: int = 16000, freq: float = 440.0) -> str:
    """Synthesise a short mono 16 kHz sine-tone WAV and return base64.

    If the ``OLMLX_TEST_AUDIO`` env var points to a real speech file that file
    is used instead (so CI runners with a speech clip get more realistic
    coverage without the test needing a checked-in asset).
    """
    override = os.environ.get("OLMLX_TEST_AUDIO")
    if override:
        with open(override, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(int(seconds * sr)):
            val = int(32767 * 0.2 * math.sin(2 * math.pi * freq * i / sr))
            w.writeframes(struct.pack("<h", val))
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
async def live_client(tmp_path, monkeypatch):
    """Real app + ModelManager backed by the real model store.

    Only the writable config (models.json / aliases.json) is redirected to a
    tmp dir so the auto-register-on-load does not mutate the developer's real
    ``~/.olmlx/models.json``.  ``models_dir`` is left at the real location so
    the already-downloaded model loads without a download.
    """
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

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await manager.stop()


async def test_openai_chat_accepts_audio(live_client):
    """POST /v1/chat/completions with an input_audio content part returns 200
    and a non-empty assistant message."""
    resp = await live_client.post(
        "/v1/chat/completions",
        json={
            "model": AUDIO_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this audio briefly."},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": _wav_b64(), "format": "wav"},
                        },
                    ],
                }
            ],
            "max_tokens": 64,
            "stream": False,
        },
        timeout=600,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip(), (
        f"expected non-empty string content, got: {body}"
    )


async def test_ollama_chat_accepts_audio(live_client):
    """POST /api/chat with an audio data-URI in the message audio field returns
    200 and a non-empty assistant message."""
    resp = await live_client.post(
        "/api/chat",
        json={
            "model": AUDIO_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Describe this audio briefly.",
                    "audio": ["data:audio/wav;base64," + _wav_b64()],
                }
            ],
            "stream": False,
        },
        timeout=600,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    content = body["message"]["content"]
    assert isinstance(content, str) and content.strip(), (
        f"expected non-empty string content, got: {body}"
    )


async def test_anthropic_messages_accepts_audio(live_client):
    """POST /v1/messages with an audio source block returns 200 and a non-empty
    text content block in the response."""
    resp = await live_client.post(
        "/v1/messages",
        json={
            "model": AUDIO_MODEL,
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this audio briefly."},
                        {
                            "type": "audio",
                            "source": {
                                "type": "base64",
                                "media_type": "audio/wav",
                                "data": _wav_b64(),
                            },
                        },
                    ],
                }
            ],
        },
        timeout=600,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    text = ""
    for block in body.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    assert text.strip(), f"expected non-empty text in content blocks, got: {body}"
