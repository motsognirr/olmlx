"""Normalize and materialize audio references from API surfaces (#426).

``mlx_vlm.utils.load_audio`` accepts file paths, http(s) URLs, and numpy
arrays — but NOT base64 or ``data:`` URIs.  This module converts the OpenAI
(``input_audio``) and Anthropic (``audio`` + ``source``) content-block shapes
into one load-ready string (URL / path / ``data:`` URI), and separately
materializes ``data:`` URIs into temp files just before generation.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

# Map a ``data:`` URI subtype to a file suffix.  Extension matters: mlx_vlm's
# read_audio routes m4a/aac/ogg/opus to ffmpeg by suffix, and others to
# miniaudio.  Unmapped subtypes fall back to ``.<subtype>``.
_SUFFIX_BY_SUBTYPE = {
    "mpeg": ".mp3",
    "mp3": ".mp3",
    "wav": ".wav",
    "x-wav": ".wav",
    "mp4": ".m4a",
    "x-m4a": ".m4a",
    "m4a": ".m4a",
    "aac": ".aac",
    "ogg": ".ogg",
    "opus": ".opus",
    "flac": ".flac",
    "webm": ".webm",
}


def normalize_audio_block(block: dict[str, Any]) -> str:
    """Convert an OpenAI ``input_audio`` or Anthropic ``audio`` content block to
    a string the engine can load (URL, path, or ``data:`` URI).

    Raises ``ValueError`` for missing fields, unsupported source types, or a
    non-audio block.
    """
    btype = block.get("type")

    # OpenAI: {"type": "input_audio", "input_audio": {"data": "<b64>", "format": "wav"}}
    if btype == "input_audio":
        spec = block.get("input_audio") or {}
        data = spec.get("data")
        if not isinstance(data, str) or not data:
            raise ValueError("input_audio block missing input_audio.data")
        fmt = spec.get("format") or "wav"
        return f"data:audio/{fmt};base64,{data}"

    # Anthropic / internal: {"type": "audio", "source": {...}}  (olmlx extension)
    if btype == "audio":
        source = block.get("source") or {}
        stype = source.get("type")
        if stype == "url":
            url = source.get("url")
            if not isinstance(url, str) or not url:
                raise ValueError("audio source type=url missing 'url'")
            return url
        if stype == "base64":
            data = source.get("data")
            if not isinstance(data, str) or not data:
                raise ValueError("audio source type=base64 missing 'data'")
            media_type = source.get("media_type") or "audio/wav"
            return f"data:{media_type};base64,{data}"
        raise ValueError(f"unsupported audio source type: {stype!r}")

    raise ValueError(f"not an audio block: type={btype!r}")


def _suffix_for_media_type(media_type: str) -> str:
    subtype = media_type.split("/", 1)[-1].lower() if "/" in media_type else "wav"
    return _SUFFIX_BY_SUBTYPE.get(subtype, f".{subtype}")


def materialize_audio(
    audio: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve audio references to loadable paths.

    ``data:`` URIs are base64-decoded into temp files (mlx_vlm's ``load_audio``
    cannot read data URIs); URLs and filesystem paths pass through unchanged.

    Returns ``(paths, temp_paths)`` where ``paths`` is what to hand to
    ``mlx_vlm`` and ``temp_paths`` ⊆ ``paths`` must be deleted by the caller
    via ``cleanup_temp_audio`` once generation is done.
    """
    paths: list[str] = []
    temp_paths: list[str] = []
    for item in audio or []:
        if isinstance(item, str) and item.startswith("data:"):
            header, _, b64 = item.partition(",")
            media_type = header[len("data:") :].split(";", 1)[0] or "audio/wav"
            try:
                raw = base64.b64decode(b64, validate=False)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"invalid base64 audio data: {exc}") from exc
            fd, tmp = tempfile.mkstemp(
                suffix=_suffix_for_media_type(media_type), prefix="olmlx-audio-"
            )
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            paths.append(tmp)
            temp_paths.append(tmp)
        else:
            paths.append(item)
    return paths, temp_paths


def cleanup_temp_audio(temp_paths: list[str] | None) -> None:
    """Delete materialized temp audio files; never raises."""
    for path in temp_paths or []:
        with contextlib.suppress(OSError):
            os.unlink(path)
