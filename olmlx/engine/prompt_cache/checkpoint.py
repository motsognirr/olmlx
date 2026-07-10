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


def flatten_cache_state(cache: list[Any]) -> list[Any]:
    """Return the flat list of state arrays across all cache layers.

    Most mlx-lm cache classes expose ``.state`` as a tuple ``(keys, values)``;
    ``ArraysCache`` exposes it as a list. Flattening is needed because
    ``mx.eval`` on a list-of-tuples-of-arrays would walk the PyTree as a
    generic container — fine in current MLX but brittle if a layer ever
    exposes a non-list/tuple container (dict, dataclass) whose internal
    arrays mx.eval would silently skip.

    Used by ``snapshot_cache_for_persistence`` and by callers that need
    to ``mx.eval`` the cache state in-place (e.g. mid-prefill flushes).
    """
    states: list[Any] = []
    for layer in cache:
        state = layer.state
        if isinstance(state, tuple):
            states.extend(state)
        else:
            states.append(state)
    return states


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
        states = flatten_cache_state(cache)
        if states:
            mx.eval(states)
    snapshot = copy.deepcopy(cache)
    if eager_eval:
        _pin_snapshot_state(snapshot)
    return snapshot


def _pin_snapshot_state(snapshot: list[Any]) -> None:
    """Materialize each layer's ``.state`` on THIS thread and pin it so a
    later ``.state`` access — possibly from a different worker thread —
    returns already-materialized arrays instead of rebuilding a lazy slice.

    Under mlx's thread-local streams (mlx >= 0.31.2, #499), evaluating a
    lazy graph requires the stream of the thread that *built* the graph.
    A step-based cache layer (``KVCache``, ``QuantizedKVCache``, ...)
    over-allocates in ``step``-sized chunks, so its ``.state`` property
    rebuilds a fresh ``[..., :offset, :]`` slice op on *every* access
    whenever ``offset`` doesn't exactly fill the buffer — even if the
    buffer itself is fully materialized. Evaluating that fresh slice on a
    thread other than the one that built it reproduces the #284 hazard
    this function exists to close, so it isn't enough to eval before the
    deepcopy: the deepcopy's ``.state`` must also be pre-built here, once,
    on the creating thread.

    For layer types with a working ``state`` setter (the ``KVCache``
    family), writing the evaluated, exact-length tuple back trims the
    backing buffer to ``offset`` — the property's own fast path
    (``offset == keys.shape[2]``) then returns the raw, already-
    materialized arrays on every later access, building no op at all.
    ``TurboQuantKVCache`` deliberately rejects ``state`` writes (it does
    not support restoration), so it exposes ``_pin_state_to_offset``
    instead to trim its packed buffers the same way.
    """
    for layer in snapshot:
        state = layer.state
        if not state:
            continue
        if isinstance(state, tuple):
            mx.eval(state)
            try:
                layer.state = state
            except NotImplementedError:
                pass
        else:
            mx.eval(state)
            pin = getattr(layer, "_pin_state_to_offset", None)
            if pin is not None:
                pin()
