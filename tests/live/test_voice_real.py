"""Live STT+TTS round-trip (real models; skipped in CI).

Synthesize speech with Kokoro, transcribe it back with Whisper, and assert the
core word survives. Outside tests/integration/ to dodge its autouse MLX mock.
Requires the [voice] extra (mlx-audio) and ffmpeg on PATH.
"""

import shutil
import wave

import numpy as np
import pytest

from olmlx.config import settings
from olmlx.engine.inference import generate_speech, generate_transcription

KOKORO = "prince-canuma/Kokoro-82M"
WHISPER = "mlx-community/whisper-large-v3-turbo"


def _models_present() -> bool:
    """True only when both models are already downloaded locally.

    Mirrors tests/live/test_tts_speech.py: CI runs `pytest` without
    `-m "not real_model"`, so a live test must skip itself when its models
    aren't present — otherwise it downloads + runs real inference on the CI
    runner (which is what `real_model` is meant to prevent).
    """
    from olmlx.models.store import _safe_dir_name

    return all(
        (settings.models_dir / _safe_dir_name(repo) / "config.json").exists()
        for repo in (KOKORO, WHISPER)
    )


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required"),
    pytest.mark.skipif(
        not _models_present(),
        reason=f"{KOKORO} / {WHISPER} not downloaded in {settings.models_dir}",
    ),
]


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
