"""Cached prompt state dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

CacheType = Literal["system", "user", "assistant", "tool", "developer"]


@dataclass
class CachedPromptState:
    """A snapshot of a prompt cache and the tokens it represents.

    ``cache_type`` records which message-role boundary this state ends at;
    used by the multi-checkpoint store for tier-aware LRU. ``is_checkpoint``
    distinguishes a mid-prompt boundary snapshot from a terminal (end-of-
    generation) entry; the two participate in different lookup paths.
    """

    tokens: list[int]
    cache: list[Any]
    cache_type: CacheType = "assistant"
    is_checkpoint: bool = False
