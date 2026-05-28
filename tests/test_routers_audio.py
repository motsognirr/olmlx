"""Tests for olmlx.routers.audio (/v1/audio/transcriptions)."""

from unittest.mock import AsyncMock, patch

import pytest

RESULT = {
    "text": "Hello world",
    "language": "en",
    "segments": [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 2.5,
            "text": " Hello world",
            "tokens": [1, 2],
            "temperature": 0.0,
            "avg_logprob": -0.1,
            "compression_ratio": 1.0,
            "no_speech_prob": 0.01,
        }
    ],
}


def _file():
    return {"file": ("clip.wav", b"RIFFfakewavdata", "audio/wav")}


@pytest.mark.asyncio
async def test_transcription_json(app_client):
    with patch(
        "olmlx.routers.audio.generate_transcription",
        new_callable=AsyncMock,
        return_value=RESULT,
    ):
        resp = await app_client.post(
            "/v1/audio/transcriptions",
            files=_file(),
            data={"model": "whisper-turbo"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"text": "Hello world"}


@pytest.mark.asyncio
async def test_transcription_verbose_json(app_client):
    with patch(
        "olmlx.routers.audio.generate_transcription",
        new_callable=AsyncMock,
        return_value=RESULT,
    ):
        resp = await app_client.post(
            "/v1/audio/transcriptions",
            files=_file(),
            data={"model": "whisper-turbo", "response_format": "verbose_json"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    assert body["text"] == "Hello world"
    assert body["duration"] == 2.5
    assert len(body["segments"]) == 1
    assert body["segments"][0]["text"] == " Hello world"


@pytest.mark.asyncio
async def test_transcription_text(app_client):
    with patch(
        "olmlx.routers.audio.generate_transcription",
        new_callable=AsyncMock,
        return_value=RESULT,
    ):
        resp = await app_client.post(
            "/v1/audio/transcriptions",
            files=_file(),
            data={"model": "whisper-turbo", "response_format": "text"},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "Hello world"


@pytest.mark.asyncio
async def test_transcription_srt(app_client):
    with patch(
        "olmlx.routers.audio.generate_transcription",
        new_callable=AsyncMock,
        return_value=RESULT,
    ):
        resp = await app_client.post(
            "/v1/audio/transcriptions",
            files=_file(),
            data={"model": "whisper-turbo", "response_format": "srt"},
        )
    assert resp.status_code == 200
    assert "00:00:00,000 --> 00:00:02,500" in resp.text
    assert "Hello world" in resp.text


@pytest.mark.asyncio
async def test_transcription_unknown_format(app_client):
    resp = await app_client.post(
        "/v1/audio/transcriptions",
        files=_file(),
        data={"model": "whisper-turbo", "response_format": "bogus"},
    )
    assert resp.status_code == 400
    assert "response_format" in resp.text


@pytest.mark.asyncio
async def test_transcription_too_large(app_client, monkeypatch):
    monkeypatch.setattr("olmlx.routers.audio.settings.audio_max_bytes", 4)
    resp = await app_client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", b"way too many bytes", "audio/wav")},
        data={"model": "whisper-turbo"},
    )
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_transcription_word_timestamps(app_client):
    with patch(
        "olmlx.routers.audio.generate_transcription",
        new_callable=AsyncMock,
        return_value=RESULT,
    ) as mock_tx:
        resp = await app_client.post(
            "/v1/audio/transcriptions",
            files=_file(),
            data={
                "model": "whisper-turbo",
                "timestamp_granularities[]": "word",
            },
        )
    assert resp.status_code == 200
    _, kwargs = mock_tx.call_args
    assert kwargs["word_timestamps"] is True
