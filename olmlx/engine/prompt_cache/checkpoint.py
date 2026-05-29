"""Message-boundary checkpoint primitives for the prompt cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SegmentRole = Literal["system", "user", "assistant", "tool", "developer"]


@dataclass(frozen=True)
class Segment:
    """One contiguous run of tokens belonging to a single chat message."""

    tokens: list[int]
    role: SegmentRole


@dataclass(frozen=True)
class SegmentedPrompt:
    """A prompt split into per-message segments, in submission order."""

    segments: list[Segment]

    @property
    def total_tokens(self) -> int:
        return sum(len(s.tokens) for s in self.segments)

    def flatten(self) -> list[int]:
        out: list[int] = []
        for s in self.segments:
            out.extend(s.tokens)
        return out

    def boundary_offsets(self) -> list[int]:
        """Cumulative end-offset for each segment.

        Empty list when there are no segments. The last value equals
        ``total_tokens``.
        """
        out: list[int] = []
        running = 0
        for s in self.segments:
            running += len(s.tokens)
            out.append(running)
        return out
