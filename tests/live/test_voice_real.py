"""Live STT+TTS round-trip (real models; skipped in CI).

Synthesize speech with Kokoro, transcribe it back with Whisper, and assert the
core word survives. Outside tests/integration/ to dodge its autouse MLX mock.
Requires the [voice] extra (mlx-audio) and ffmpeg on PATH.
"""

import wave

import numpy as np
import pytest

from olmlx.engine.inference import generate_speech, generate_transcription

pytestmark = [pytest.mark.real_model]

KOKORO = "prince-canuma/Kokoro-82M"
WHISPER = "mlx-community/whisper-large-v3-turbo"


@pytest.fixture
def real_manager():
    # Build a real ModelManager exactly as the CLI does.
    # Source: olmlx/cli.py _create_store() + _run_chat() setup.
    from olmlx.cli import _create_store
    from olmlx.engine.model_manager import ModelManager

    store = _create_store()
    return ModelManager(store.registry, store)


_TTS_SAMPLE_RATE = 24000


@pytest.mark.asyncio
async def test_tts_then_stt_roundtrip(real_manager, tmp_path):
    # generate_speech is a streaming async generator (issue #367): collect the
    # 24 kHz float32 segments into one buffer.
    segments = []
    async for seg in generate_speech(
        real_manager, KOKORO, "The quick brown fox.", voice="af_heart"
    ):
        segments.append(np.asarray(seg, dtype=np.float32))
    pcm = np.concatenate(segments)
    sr = _TTS_SAMPLE_RATE
    assert pcm.size > sr // 2  # at least ~0.5s of audio

    path = str(tmp_path / "tts.wav")
    pcm16 = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())

    result = await generate_transcription(real_manager, WHISPER, path)
    assert "fox" in result["text"].lower()
