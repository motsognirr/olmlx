"""Router tests for /v1/audio/speech (#367)."""

from unittest.mock import patch

import numpy as np
import pytest


def _floats():
    async def _gen(*args, **kwargs):
        yield np.array([0.0, 0.1, -0.1], dtype=np.float32)
        yield np.array([0.2], dtype=np.float32)

    return _gen


@pytest.mark.asyncio
async def test_speech_pcm(app_client):
    with patch("olmlx.routers.audio.generate_speech", _floats()):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "hi",
                "voice": "alloy",
                "response_format": "pcm",
            },
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/pcm")
    # 4 samples * 2 bytes
    assert len(resp.content) == 8


@pytest.mark.asyncio
async def test_speech_wav_has_riff(app_client):
    with patch("olmlx.routers.audio.generate_speech", _floats()):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "hi",
                "voice": "alloy",
                "response_format": "wav",
            },
        )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_speech_unknown_voice_422(app_client):
    resp = await app_client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "hi", "voice": "nope"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_speech_bad_format_422(app_client):
    resp = await app_client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "hi",
            "voice": "alloy",
            "response_format": "ogg2",
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_speech_empty_input_422(app_client):
    resp = await app_client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "", "voice": "alloy"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_speech_oversize_413(app_client, monkeypatch):
    monkeypatch.setattr("olmlx.routers.audio.settings.tts_max_input_chars", 5)
    resp = await app_client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "hello world", "voice": "alloy"},
    )
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_speech_non_tts_model_400(app_client):
    async def _raise(*a, **k):
        raise ValueError("Model 'qwen3' is not a TTS model.")
        yield  # pragma: no cover - makes this an async generator

    with patch("olmlx.routers.audio.generate_speech", _raise):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={"model": "qwen3", "input": "hi", "voice": "alloy"},
        )
    assert resp.status_code == 400
