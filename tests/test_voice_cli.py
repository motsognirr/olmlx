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


def test_check_voice_deps_present(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    # Should not raise.
    _check_voice_deps()
