"""Audio playback."""

import sys
import types

import numpy as np
import pytest

from olmlx.chat.voice import playback


def _fake_sounddevice(monkeypatch, recorded):
    sd = types.ModuleType("sounddevice")
    sd.play = lambda data, sr: recorded.update(data=np.asarray(data), sr=sr)
    sd.wait = lambda: recorded.update(waited=True)
    sd.stop = lambda: recorded.update(stopped=True)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def test_play_invokes_sounddevice(monkeypatch):
    rec = {}
    _fake_sounddevice(monkeypatch, rec)
    playback.play(np.array([0.1, 0.2], dtype=np.float32), 24000)
    assert rec["sr"] == 24000
    assert rec["waited"] is True
    np.testing.assert_allclose(rec["data"], [0.1, 0.2])


def test_play_empty_is_noop(monkeypatch):
    rec = {}
    _fake_sounddevice(monkeypatch, rec)
    playback.play(np.zeros(0, dtype=np.float32), 24000)
    assert "data" not in rec  # never called sd.play


def test_play_requires_sounddevice(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(RuntimeError, match="sounddevice"):
        playback.play(np.array([0.1], dtype=np.float32), 24000)
