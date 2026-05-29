"""Message-boundary checkpoint primitives for the prompt cache."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx

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


def snapshot_cache_for_persistence(
    cache: list[Any],
    *,
    eager_eval: bool,
) -> list[Any]:
    """Return a thread-safe deep copy of a prompt cache.

    When ``eager_eval`` is True, all layer state arrays are materialized
    via ``mx.eval`` before the deep copy. This closes issue #284: the
    ``gated_delta_kernel`` outputs in ``ArraysCache`` carry a lazy graph
    bound to the originating worker thread's Metal stream. Evaluating
    eagerly produces concrete arrays whose evaluation is thread-independent,
    so the snapshot can be reused from any worker thread on a later request.

    Args:
        cache: List of mlx-lm cache layer objects (one per model layer).
        eager_eval: If True, materialize each layer's ``state`` before copy.
            Set this for caches whose layers include ``ArraysCache`` or
            other types known to produce lazy outputs bound to a Metal stream.
            For pure ``KVCache``/``QuantizedKVCache`` layouts the deep copy
            alone is sufficient and ``eager_eval=False`` saves the eval cost.
    """
    if eager_eval:
        states: list[Any] = []
        for layer in cache:
            state = layer.state
            if isinstance(state, tuple):
                states.extend(state)
            else:
                states.append(state)
        if states:
            mx.eval(states)
    return copy.deepcopy(cache)
