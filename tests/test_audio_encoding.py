"""Audio encoding helpers for /v1/audio/speech (#367)."""

import struct

import numpy as np
import pytest

from olmlx.utils.audio import (
    FORMAT_MEDIA_TYPES,
    SUPPORTED_FORMATS,
    UnsupportedFormatError,
    float_to_pcm16,
    validate_format,
    wav_bytes,
)


def test_float_to_pcm16_roundtrip():
    samples = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    raw = float_to_pcm16(samples)
    assert isinstance(raw, bytes)
    decoded = np.frombuffer(raw, dtype="<i2")
    assert decoded[0] == 0
    assert decoded[1] == 32767
    assert decoded[2] == -32767
    assert abs(int(decoded[3]) - 16383) <= 1


def test_float_to_pcm16_clips():
    samples = np.array([2.0, -2.0], dtype=np.float32)
    decoded = np.frombuffer(float_to_pcm16(samples), dtype="<i2")
    assert decoded[0] == 32767
    assert decoded[1] == -32767


def test_wav_bytes_has_valid_header():
    pcm = float_to_pcm16(np.zeros(100, dtype=np.float32))
    out = wav_bytes(pcm, sample_rate=24000)
    assert out[:4] == b"RIFF"
    assert out[8:12] == b"WAVE"
    # data chunk size == len(pcm)
    riff_size = struct.unpack("<I", out[4:8])[0]
    assert riff_size == 36 + len(pcm)


def test_validate_format_accepts_supported():
    for fmt in SUPPORTED_FORMATS:
        validate_format(fmt)  # no raise
        assert fmt in FORMAT_MEDIA_TYPES


def test_validate_format_rejects_unknown():
    with pytest.raises(UnsupportedFormatError):
        validate_format("ogg2")


@pytest.mark.asyncio
async def test_ffmpeg_encode_mp3_produces_bytes():
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    pcm = float_to_pcm16((0.1 * np.sin(np.arange(24000) / 10.0)).astype(np.float32))

    async def _chunks():
        yield pcm

    from olmlx.utils.audio import ffmpeg_encode

    out = b""
    async for b in ffmpeg_encode(_chunks(), fmt="mp3", sample_rate=24000):
        out += b
    assert len(out) > 0
    # MP3 frame sync (0xFF 0xEx) or an ID3 header.
    assert out[:3] == b"ID3" or out[0] == 0xFF
