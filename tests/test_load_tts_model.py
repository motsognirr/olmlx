"""Loading a TTS model sets is_tts and skips LLM-only paths (#367)."""

from unittest.mock import MagicMock, patch

from olmlx.engine.model_manager import LoadedModel


def test_loadedmodel_has_is_tts_default_false():
    lm = LoadedModel(name="x", hf_path="x", model=object(), tokenizer=None)
    assert lm.is_tts is False


def test_load_branch_uses_mlx_audio(tmp_path):
    # The kind=="tts" branch must call mlx_audio.tts.utils.load_model and
    # return (model, None, False, TemplateCaps(), None).
    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.template_caps import TemplateCaps

    mgr = ModelManager.__new__(ModelManager)
    fake_model = MagicMock()
    with (
        patch.object(ModelManager, "_detect_model_kind", return_value="tts"),
        patch("mlx_audio.tts.utils.load_model", return_value=fake_model) as load,
    ):
        result = ModelManager._load_model_tts(mgr, "owner/Kokoro-82M", str(tmp_path))
    model, tok, is_vlm, caps, dec = result
    assert model is fake_model
    assert tok is None and is_vlm is False and dec is None
    assert isinstance(caps, TemplateCaps)
    load.assert_called_once()
