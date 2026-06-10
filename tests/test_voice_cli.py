"""CLI dependency check + voice flag parsing for `olmlx chat --voice`."""

import sys
import types

import pytest

from olmlx.cli import _check_voice_deps, _build_chat_arg_parser_voice_defaults


def test_voice_flags_registered():
    """The chat subparser exposes --voice and model overrides."""
    flags = _build_chat_arg_parser_voice_defaults()
    assert "--voice" in flags
    assert "--stt-model" in flags
    assert "--tts-model" in flags
    assert "--voice-name" in flags


def test_check_voice_deps_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(SystemExit):
        _check_voice_deps()


def test_check_voice_deps_missing_mlx_audio(monkeypatch):
    # mlx-audio moved to the [audio] extra (#469); --voice needs it for TTS,
    # so its absence must exit with the install hint too.
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    monkeypatch.setitem(sys.modules, "mlx_audio", None)
    with pytest.raises(SystemExit):
        _check_voice_deps()


@pytest.mark.parametrize("missing", ["misaki", "en_core_web_sm"])
def test_check_voice_deps_missing_audio_transitives(monkeypatch, missing):
    # A hand-installed mlx-audio (pip install, no extra) lacks misaki /
    # en_core_web_sm and would otherwise pass the gate, then fail at the
    # first TTS call. The gate must front-load that failure too (#469).
    for mod in ("sounddevice", "mlx_audio", "misaki", "en_core_web_sm"):
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(SystemExit):
        _check_voice_deps()


def test_check_voice_deps_present(monkeypatch):
    for mod in ("sounddevice", "mlx_audio", "misaki", "en_core_web_sm"):
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
    # Should not raise.
    _check_voice_deps()
