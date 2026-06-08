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


def test_materialize_invalid_base64_raises():
    with pytest.raises(ValueError, match="invalid base64"):
        materialize_audio(["data:audio/wav;base64,!!!notbase64!!!"])


def test_materialize_cleans_up_earlier_temps_on_later_failure(monkeypatch):
    """When a later item in a multi-item list fails, temp files written for the
    earlier (successful) items must be cleaned up, not leaked — the caller never
    receives the partial list so it cannot clean them up itself."""
    import olmlx.utils.audio_input as ai

    created: list[str] = []
    real_mkstemp = ai.tempfile.mkstemp

    def spy_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created.append(path)
        return fd, path

    monkeypatch.setattr(ai.tempfile, "mkstemp", spy_mkstemp)

    good = "data:audio/wav;base64," + base64.b64encode(b"hello").decode()
    bad = "data:audio/wav;base64,!!!notbase64!!!"
    with pytest.raises(ValueError, match="invalid base64"):
        materialize_audio([good, bad])

    assert created, "expected a temp file to be created for the first (good) item"
    for path in created:
        assert not os.path.exists(path), f"leaked temp file: {path}"


def test_materialize_non_data_uri_failure_cleans_temps(monkeypatch):
    """A write failure (OSError) on a later item also cleans up earlier temps."""
    import olmlx.utils.audio_input as ai

    created: list[str] = []
    real_mkstemp = ai.tempfile.mkstemp
    calls = {"n": 0}

    def spy_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created.append(path)
        return fd, path

    monkeypatch.setattr(ai.tempfile, "mkstemp", spy_mkstemp)

    real_fdopen = ai.os.fdopen

    def flaky_fdopen(fd, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # fail writing the second item
            os.close(fd)
            raise OSError("disk full")
        return real_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(ai.os, "fdopen", flaky_fdopen)

    item = "data:audio/wav;base64," + base64.b64encode(b"x").decode()
    with pytest.raises(OSError, match="disk full"):
        materialize_audio([item, item])

    assert len(created) == 2
    for path in created:
        assert not os.path.exists(path), f"leaked temp file: {path}"
