"""Voice-mode settings for olmlx chat."""

from olmlx.config import Settings


def test_voice_settings_defaults():
    s = Settings()
    assert s.chat_stt_model == "mlx-community/whisper-large-v3-turbo"
    assert s.chat_tts_model == "prince-canuma/Kokoro-82M"
    assert s.chat_tts_voice == "af_heart"


def test_voice_settings_env_override(monkeypatch):
    monkeypatch.setenv("OLMLX_CHAT_TTS_VOICE", "am_adam")
    monkeypatch.setenv("OLMLX_CHAT_STT_MODEL", "whisper-turbo")
    s = Settings()
    assert s.chat_tts_voice == "am_adam"
    assert s.chat_stt_model == "whisper-turbo"
