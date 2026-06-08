"""Audio playback via sounddevice (issue #444)."""

from __future__ import annotations

import numpy as np

from olmlx.chat.voice.capture import _require_sounddevice


def play(pcm: np.ndarray, samplerate: int) -> None:
    """Play a float32 PCM buffer and block until it finishes."""
    if pcm is None:
        return
    buf = np.asarray(pcm)
    if buf.size == 0:
        return
    sd = _require_sounddevice()
    sd.play(buf, samplerate)
    sd.wait()
