import base64
import os

import pytest

from olmlx.utils.audio_input import (
    cleanup_temp_audio,
    materialize_audio,
    normalize_audio_block,
)


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


def test_materialize_passthrough_url_and_path():
    paths, temps = materialize_audio(["http://x/a.wav", "/tmp/local.mp3"])
    assert paths == ["http://x/a.wav", "/tmp/local.mp3"]
    assert temps == []


def test_materialize_data_uri_writes_temp_file_with_suffix():
    raw = b"RIFFblah"
    uri = "data:audio/wav;base64," + base64.b64encode(raw).decode()
    paths, temps = materialize_audio([uri])
    try:
        assert len(paths) == 1 and len(temps) == 1
        assert paths == temps
        assert paths[0].endswith(".wav")
        with open(paths[0], "rb") as fh:
            assert fh.read() == raw
    finally:
        cleanup_temp_audio(temps)
    assert not os.path.exists(temps[0])


def test_materialize_mpeg_maps_to_mp3_suffix():
    uri = "data:audio/mpeg;base64," + base64.b64encode(b"x").decode()
    paths, temps = materialize_audio([uri])
    try:
        assert paths[0].endswith(".mp3")
    finally:
        cleanup_temp_audio(temps)


def test_cleanup_is_idempotent_and_swallows_missing():
    cleanup_temp_audio(["/nonexistent/abc.wav"])  # must not raise


def test_materialize_none_returns_empty():
    paths, temps = materialize_audio(None)
    assert paths == [] and temps == []
