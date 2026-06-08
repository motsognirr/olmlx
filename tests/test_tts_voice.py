"""OpenAI -> Kokoro voice resolution (#367)."""

import pytest

from olmlx.engine.tts import KOKORO_VOICES, UnknownVoiceError, resolve_voice


def test_openai_names_map_to_kokoro():
    assert resolve_voice("alloy") == "af_alloy"
    assert resolve_voice("echo") == "am_echo"
    assert resolve_voice("fable") == "bm_fable"
    assert resolve_voice("onyx") == "am_onyx"
    assert resolve_voice("nova") == "af_nova"
    assert resolve_voice("shimmer") == "af_sky"


def test_native_kokoro_voice_passes_through():
    assert resolve_voice("af_heart") == "af_heart"
    assert resolve_voice("bm_george") == "bm_george"


def test_unknown_voice_raises():
    with pytest.raises(UnknownVoiceError):
        resolve_voice("definitely_not_a_voice")


def test_all_map_targets_are_real_voices():
    from olmlx.engine.tts import OPENAI_VOICE_MAP

    for target in OPENAI_VOICE_MAP.values():
        assert target in KOKORO_VOICES
