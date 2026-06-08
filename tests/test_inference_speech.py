"""generate_speech engine path (#367)."""

import types
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from olmlx.engine.inference import generate_speech


class _Result:
    def __init__(self, audio):
        self.audio = audio
        self.sample_rate = 24000


def _fake_lm(is_tts=True):
    lm = MagicMock()
    lm.is_tts = is_tts
    lm.inference_queue_timeout = 5.0
    # "none" is a valid SyncMode (Literal["full","minimal","none"]); a bare
    # MagicMock model can't drive real Metal sync, so skip lock-boundary sync.
    lm.sync_mode = "none"
    lm.name = "kokoro"

    def _gen(text, voice=None, speed=1.0, **kw):
        yield _Result(np.array([0.1, 0.2], dtype=np.float32))
        yield _Result(np.array([0.3], dtype=np.float32))

    lm.model = types.SimpleNamespace(generate=_gen)
    return lm


@pytest.mark.asyncio
async def test_generate_speech_yields_float_segments(monkeypatch):
    lm = _fake_lm()
    manager = MagicMock()
    manager.ensure_loaded = AsyncMock(return_value=lm)
    manager.store = None

    chunks = []
    async for seg in generate_speech(
        manager, "kokoro", "hello world", voice="af_heart", speed=1.0
    ):
        chunks.append(seg)

    assert len(chunks) == 2
    assert np.allclose(chunks[0], [0.1, 0.2])
    assert np.allclose(chunks[1], [0.3])
    lm.release_ref.assert_called_once()


@pytest.mark.asyncio
async def test_generate_speech_rejects_non_tts():
    lm = _fake_lm(is_tts=False)
    manager = MagicMock()
    manager.ensure_loaded = AsyncMock(return_value=lm)
    manager.store = None

    with pytest.raises(ValueError, match="not a TTS model"):
        async for _ in generate_speech(manager, "qwen3", "hi", voice="af_heart"):
            pass
    lm.release_ref.assert_called_once()
