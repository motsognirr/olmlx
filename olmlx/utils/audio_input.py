"""Normalize and materialize audio references from API surfaces (#426).

``mlx_vlm.utils.load_audio`` accepts file paths, http(s) URLs, and numpy
arrays — but NOT base64 or ``data:`` URIs.  This module converts the OpenAI
(``input_audio``) and Anthropic (``audio`` + ``source``) content-block shapes
into one load-ready string (URL / path / ``data:`` URI), and separately
materializes ``data:`` URIs into temp files just before generation.
"""

from __future__ import annotations

from typing import Any


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
