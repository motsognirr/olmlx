"""TTS model-kind detection (#367)."""

from unittest.mock import patch

from olmlx.engine.model_manager import ModelManager


def _detect(config: dict) -> str:
    mgr = ModelManager.__new__(ModelManager)  # bypass __init__
    mgr.store = None
    with patch("huggingface_hub.hf_hub_download") as dl:
        import json
        import tempfile

        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(config, f)
        f.flush()
        dl.return_value = f.name
        return mgr._detect_model_kind("some/repo")


def test_kokoro_config_detected_as_tts():
    # Kokoro config signature: istftnet + plbert, no model_type.
    cfg = {"istftnet": {}, "plbert": {}, "n_mels": 24, "n_token": 178}
    assert _detect(cfg) == "tts"


def test_whisper_not_misdetected_as_tts():
    cfg = {"n_mels": 80, "n_audio_state": 768}
    assert _detect(cfg) == "whisper"


def test_plain_text_model_not_tts():
    cfg = {"model_type": "qwen3"}
    assert _detect(cfg) != "tts"
