"""VoiceIO orchestrator."""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from olmlx.chat.voice.io import VoiceIO, split_into_sentences


def test_split_into_sentences():
    out = split_into_sentences("Hello world. How are you?  Fine!")
    assert out == ["Hello world.", "How are you?", "Fine!"]


def test_split_into_sentences_no_terminator():
    assert split_into_sentences("just one line") == ["just one line"]


def test_split_into_sentences_empty():
    assert split_into_sentences("   ") == []


@pytest.mark.asyncio
async def test_listen_records_and_transcribes(mock_manager):
    vio = VoiceIO(
        manager=mock_manager,
        stt_model="whisper",
        tts_model="kokoro",
        voice="af_heart",
    )
    with (
        patch("olmlx.chat.voice.io.capture.record_to_wav", return_value="/tmp/x.wav"),
        patch("olmlx.chat.voice.io.os.unlink") as unlink,
        patch(
            "olmlx.chat.voice.io.generate_transcription",
            new=AsyncMock(return_value={"text": "  hello there  "}),
        ),
    ):
        text = await vio.listen()
    assert text == "hello there"
    unlink.assert_called_once_with("/tmp/x.wav")


def _fake_speech_stream(segments):
    """Return a callable producing a fresh async generator of ``segments``.

    Mirrors the upstream streaming ``generate_speech`` (issue #367): a sync call
    that returns an async generator yielding 24 kHz float32 segments.
    """

    def _factory(*_a, **_k):
        async def _gen():
            for seg in segments:
                yield seg

        return _gen()

    return _factory


@pytest.mark.asyncio
async def test_speak_synthesizes_each_sentence(mock_manager):
    vio = VoiceIO(
        manager=mock_manager,
        stt_model="whisper",
        tts_model="kokoro",
        voice="af_heart",
    )
    calls = []
    seg = np.array([0.1], dtype=np.float32)
    with (
        patch(
            "olmlx.chat.voice.io.generate_speech",
            side_effect=_fake_speech_stream([seg]),
        ) as gs,
        patch(
            "olmlx.chat.voice.io.playback.play", side_effect=lambda *a: calls.append(a)
        ),
    ):
        await vio.speak("One. Two.")
    assert gs.call_count == 2  # one generate_speech call per sentence
    assert len(calls) == 2  # one segment played per sentence


@pytest.mark.asyncio
async def test_speak_plays_every_streamed_segment(mock_manager):
    """Multiple segments from a single sentence are each played."""
    vio = VoiceIO(
        manager=mock_manager,
        stt_model="whisper",
        tts_model="kokoro",
        voice="af_heart",
    )
    calls = []
    segs = [np.array([0.1], dtype=np.float32), np.array([0.2], dtype=np.float32)]
    with (
        patch(
            "olmlx.chat.voice.io.generate_speech",
            side_effect=_fake_speech_stream(segs),
        ),
        patch(
            "olmlx.chat.voice.io.playback.play", side_effect=lambda *a: calls.append(a)
        ),
    ):
        await vio.speak("One sentence only")
    assert len(calls) == 2  # both streamed segments played


@pytest.mark.asyncio
async def test_speak_empty_does_nothing(mock_manager):
    vio = VoiceIO(manager=mock_manager, stt_model="w", tts_model="k", voice="af_heart")
    with (
        patch("olmlx.chat.voice.io.generate_speech") as gs,
        patch("olmlx.chat.voice.io.playback.play") as play,
    ):
        await vio.speak("   ")
    gs.assert_not_called()
    play.assert_not_called()
