import pytest

from olmlx.utils.audio_input import normalize_audio_block


def test_openai_input_audio_builds_data_uri():
    block = {"type": "input_audio", "input_audio": {"data": "QQ==", "format": "wav"}}
    assert normalize_audio_block(block) == "data:audio/wav;base64,QQ=="


def test_openai_input_audio_mp3():
    block = {"type": "input_audio", "input_audio": {"data": "QQ==", "format": "mp3"}}
    assert normalize_audio_block(block) == "data:audio/mp3;base64,QQ=="


def test_openai_input_audio_missing_data_raises():
    with pytest.raises(ValueError, match="input_audio"):
        normalize_audio_block({"type": "input_audio", "input_audio": {"format": "wav"}})


def test_anthropic_audio_base64_builds_data_uri():
    block = {
        "type": "audio",
        "source": {"type": "base64", "media_type": "audio/mpeg", "data": "QQ=="},
    }
    assert normalize_audio_block(block) == "data:audio/mpeg;base64,QQ=="


def test_anthropic_audio_base64_defaults_media_type():
    block = {"type": "audio", "source": {"type": "base64", "data": "QQ=="}}
    assert normalize_audio_block(block) == "data:audio/wav;base64,QQ=="


def test_anthropic_audio_url_source():
    block = {"type": "audio", "source": {"type": "url", "url": "http://x/a.wav"}}
    assert normalize_audio_block(block) == "http://x/a.wav"


def test_anthropic_audio_unsupported_source_raises():
    block = {"type": "audio", "source": {"type": "file", "id": "abc"}}
    with pytest.raises(ValueError, match="unsupported audio source"):
        normalize_audio_block(block)


def test_not_an_audio_block_raises():
    with pytest.raises(ValueError, match="not an audio block"):
        normalize_audio_block({"type": "text", "text": "hi"})
