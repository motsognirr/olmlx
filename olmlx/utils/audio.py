"""Audio encoding for /v1/audio/speech (#367).

Converts the float-audio segment stream from mlx-audio into the requested
OpenAI container:

- ``pcm``  : raw signed 16-bit little-endian PCM (no header), streamed.
- ``wav``  : pure-Python RIFF/WAVE (buffered so the header sizes are exact).
- ``mp3``/``opus``/``aac``/``flac`` : piped through ffmpeg, streamed.

ffmpeg is already a hard requirement for the Whisper path.
"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator

import numpy as np

SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {"mp3", "opus", "aac", "flac", "wav", "pcm"}
)

FORMAT_MEDIA_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

# ffmpeg output muxer per format (formats not listed are native-encoded).
_FFMPEG_MUXER: dict[str, str] = {
    "mp3": "mp3",
    "opus": "opus",
    "aac": "adts",
    "flac": "flac",
}


class UnsupportedFormatError(ValueError):
    """Raised for an unknown ``response_format``. Surfaced as HTTP 422."""


def validate_format(fmt: str) -> None:
    if fmt not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported response_format '{fmt}'. "
            f"Must be one of: {', '.join(sorted(SUPPORTED_FORMATS))}."
        )


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert a float waveform in [-1, 1] to signed 16-bit LE PCM bytes."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def wav_bytes(pcm: bytes, *, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw 16-bit PCM in a complete RIFF/WAVE header."""
    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    header = b"RIFF"
    header += struct.pack("<I", 36 + len(pcm))
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack(
        "<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits
    )
    header += b"data"
    header += struct.pack("<I", len(pcm))
    return header + pcm


async def ffmpeg_encode(
    pcm_chunks: AsyncIterator[bytes], *, fmt: str, sample_rate: int, channels: int = 1
) -> AsyncIterator[bytes]:
    """Stream PCM chunks through ffmpeg, yielding encoded container bytes.

    The ffmpeg subprocess is always reaped — on normal completion, on a feed
    error, and on generator close (client disconnect) — so a dropped stream
    cannot orphan an ffmpeg process.
    """
    muxer = _FFMPEG_MUXER[fmt]
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-f",
        muxer,
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def _feed() -> None:
        try:
            async for chunk in pcm_chunks:
                proc.stdin.write(chunk)
                await proc.stdin.drain()
        finally:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()

    feeder = asyncio.create_task(_feed())
    try:
        while True:
            data = await proc.stdout.read(65536)
            if not data:
                break
            yield data
        await feeder
        await proc.wait()
        if proc.returncode not in (0, None):
            err = (await proc.stderr.read()).decode(errors="replace")
            raise RuntimeError(f"ffmpeg encode failed ({proc.returncode}): {err}")
    finally:
        feeder.cancel()
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()
