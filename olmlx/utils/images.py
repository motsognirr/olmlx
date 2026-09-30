"""Normalize image references from API surfaces into mlx_vlm-loadable strings.

``mlx_vlm.utils.load_image`` accepts file paths, http(s) URLs, and
``data:image/...;base64,...`` data URIs (PIL sniffs the real format, so the
declared media type in a data URI is cosmetic).  This module converts the
OpenAI (``image_url``) and Anthropic (``image`` + ``source``) content-block
shapes into one of those forms (issue #428), and wraps the Ollama-native
``images`` field's raw base64 strings into data URIs (issue #714).
"""

from __future__ import annotations

from typing import Any


def normalize_image_block(block: dict[str, Any]) -> str:
    """Convert an OpenAI ``image_url`` or Anthropic ``image`` content block to a
    string ``load_image`` accepts (URL, path, or data URI).

    Raises ``ValueError`` for missing fields, unsupported source types, or a
    non-image block.
    """
    btype = block.get("type")

    # OpenAI: {"type": "image_url", "image_url": {"url": "..."}}
    if btype == "image_url":
        url = (block.get("image_url") or {}).get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("image_url block missing image_url.url")
        return url

    # Anthropic: {"type": "image", "source": {...}}
    if btype == "image":
        source = block.get("source") or {}
        stype = source.get("type")
        if stype == "url":
            url = source.get("url")
            if not isinstance(url, str) or not url:
                raise ValueError("image source type=url missing 'url'")
            return url
        if stype == "base64":
            data = source.get("data")
            if not isinstance(data, str) or not data:
                raise ValueError("image source type=base64 missing 'data'")
            media_type = source.get("media_type") or "image/png"
            return f"data:{media_type};base64,{data}"
        raise ValueError(f"unsupported image source type: {stype!r}")

    raise ValueError(f"not an image block: type={btype!r}")


_LOADABLE_PREFIXES = ("data:", "http://", "https://")


def ensure_image_data_uri(ref: str) -> str:
    """Wrap a raw base64 image (Ollama's ``images`` field) in a data URI.

    Ollama clients send bare base64 with no ``data:`` prefix, which
    ``load_image`` would otherwise try to ``open()`` as a file path.  Refs that
    are already data URIs or http(s) URLs — what the OpenAI/Anthropic/Responses
    routers put in the same field — pass through unchanged.
    """
    if ref.startswith(_LOADABLE_PREFIXES):
        return ref
    return f"data:image/png;base64,{ref}"
