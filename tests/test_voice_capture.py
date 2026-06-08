"""Push-to-talk mic capture."""

import sys
import types
import wave

import numpy as np
import pytest

from olmlx.chat.voice import capture


def _fake_sounddevice(monkeypatch, frames):
    """Install a fake sounddevice whose InputStream feeds ``frames``."""
    sd = types.ModuleType("sounddevice")

    class FakeStream:
        def __init__(self, samplerate, channels, dtype, callback):
            self._callback = callback

        def __enter__(self):
            self._callback(frames, len(frames), None, None)
            return self

        def __exit__(self, *a):
            return False

    sd.InputStream = FakeStream
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def test_record_to_wav_writes_int16(monkeypatch, tmp_path):
    frames = np.array([[100], [-100], [200]], dtype=np.int16)
    _fake_sounddevice(monkeypatch, frames)
    # Stop immediately (simulate the user pressing Enter).
    monkeypatch.setattr(capture, "_wait_for_stop", lambda: None)

    path = capture.record_to_wav(samplerate=16000, out_dir=tmp_path)

    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 3


def test_record_requires_sounddevice(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(RuntimeError, match="sounddevice"):
        capture.record_to_wav()
