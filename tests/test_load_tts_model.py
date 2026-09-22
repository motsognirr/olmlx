"""Loading a TTS model sets is_tts and skips LLM-only paths (#367)."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from olmlx.engine.model_manager import LoadedModel


def _stub_mlx_audio(monkeypatch, load_model):
    """Install a fake mlx_audio.tts.utils into sys.modules.

    mlx-audio lives in the [audio] extra (#469), so tests must not require
    the real package to be installed.
    """
    utils = types.ModuleType("mlx_audio.tts.utils")
    utils.load_model = load_model
    tts = types.ModuleType("mlx_audio.tts")
    tts.utils = utils
    pkg = types.ModuleType("mlx_audio")
    pkg.tts = tts
    monkeypatch.setitem(sys.modules, "mlx_audio", pkg)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", tts)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", utils)


def test_loadedmodel_has_is_tts_default_false():
    lm = LoadedModel(name="x", hf_path="x", model=object(), tokenizer=None)
    assert lm.is_tts is False


def test_load_branch_uses_mlx_audio(tmp_path, monkeypatch):
    # The kind=="tts" branch must call mlx_audio.tts.utils.load_model and
    # return (model, None, False, TemplateCaps(), None).
    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.template_caps import TemplateCaps

    mgr = ModelManager.__new__(ModelManager)
    fake_model = MagicMock()
    load = MagicMock(return_value=fake_model)
    _stub_mlx_audio(monkeypatch, load)
    with patch.object(ModelManager, "_detect_model_kind", return_value="tts"):
        result = ModelManager._load_model_tts(mgr, "owner/Kokoro-82M", str(tmp_path))
    model, tok, is_vlm, caps, dec = result
    assert model is fake_model
    assert tok is None and is_vlm is False and dec is None
    assert isinstance(caps, TemplateCaps)
    load.assert_called_once()


def test_load_tts_without_mlx_audio_gives_install_hint(tmp_path, monkeypatch):
    # Without the [audio] extra, loading a TTS model must raise ValueError
    # (-> HTTP 400 on /v1/audio/speech) with the install command, not a bare
    # ModuleNotFoundError (-> opaque 500). Issue #469.
    from olmlx.engine.model_manager import ModelManager

    mgr = ModelManager.__new__(ModelManager)
    monkeypatch.setitem(sys.modules, "mlx_audio", None)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", None)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", None)
    with pytest.raises(ValueError, match=r"uv sync --extra audio"):
        ModelManager._load_model_tts(mgr, "owner/Kokoro-82M", str(tmp_path))
