"""KV cache state stored cross-request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CachedPromptState:
    """KV cache state from a previous generation, for prompt cache reuse."""

    tokens: list[int]  # Full sequence: prompt + generated tokens
    cache: list[Any]  # Per-layer KV cache objects (mutated in-place by generate_step)
