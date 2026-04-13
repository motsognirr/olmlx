"""Speculative prefetcher for LLM in a Flash.

Predicts which neurons will be needed for upcoming layers or tokens and
starts background SSD I/O before the forward pass reaches those layers.

Two prefetch paths:

- **Cross-layer** (always-on): while layer L computes, predict and prefetch
  neurons for layer L+1. Uses the current layer's pre-MLP hidden state as
  an approximate signal for the next layer's activation pattern.

- **Draft-informed** (speculative decoding only): after the draft model
  generates candidate tokens, predict neurons for all layers of the target
  verification pass and submit bulk I/O before the target forward pass starts.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import cast

import mlx.core as mx

from olmlx.engine.flash.predictor import LookaheadBank, PredictorBank
from olmlx.engine.flash.weight_store import FlashWeightStore

logger = logging.getLogger(__name__)


@dataclass
class PrefetchStats:
    """Counters for monitoring prefetch effectiveness."""

    submitted: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    failures: int = 0

    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


@dataclass
class _LayerPrefetchState:
    """In-flight prefetch state for a single layer."""

    done: threading.Event = field(default_factory=threading.Event)


class Prefetcher:
    """Orchestrates background neuron prefetching from SSD.

    Thread-safe: ``submit`` is called from the main forward-pass thread;
    the actual I/O runs on a dedicated thread pool.
    """

    def __init__(
        self,
        predictor_bank: PredictorBank,
        weight_store: FlashWeightStore,
        num_layers: int,
        *,
        lookahead_bank: LookaheadBank | None = None,
        confidence_threshold: float = 0.3,
        min_neurons: int = 64,
        max_neurons: int | None = None,
        io_threads: int = 16,
    ):
        self._predictor_bank = predictor_bank
        self._lookahead_bank = lookahead_bank
        self._weight_store = weight_store
        self._num_layers = num_layers
        self._threshold = confidence_threshold
        self._min_neurons = min_neurons
        self._max_neurons = max_neurons
        self._executor = ThreadPoolExecutor(max_workers=io_threads)
        self._predict_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="prefetch-predict"
        )

        self._lock = threading.Lock()
        self._pending: dict[int, _LayerPrefetchState] = {}
        self.stats = PrefetchStats()

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_size(self) -> int | None:
        """Input dimension the sparsity predictors expect, or None if unknown."""
        predictors = self._predictor_bank.predictors
        if predictors and hasattr(predictors[0], "down"):
            return predictors[0].down.weight.shape[1]
        return None

    # ------------------------------------------------------------------
    # Cross-layer prefetch (Path A)
    # ------------------------------------------------------------------

    def submit(self, layer_idx: int, hidden_state: mx.array) -> None:
        """Predict neurons for ``layer_idx + 1`` and start background I/O.

        Prediction runs on a dedicated single-thread executor so it overlaps
        with the current layer's SSD I/O and compute.  The hidden state is
        materialized on the calling thread first (``mx.eval`` is not safe
        for concurrent calls).

        The ``_pending`` entry is registered *before* enqueueing so that
        ``wait(layer_idx + 1)`` in the next layer blocks until both
        prediction and I/O complete — this prevents concurrent ``mx.eval``
        between the prediction thread and the main forward-pass thread.
        """
        next_layer = layer_idx + 1
        if next_layer >= self._num_layers:
            return

        # Check before mx.eval to avoid wasted materialization when
        # the previous prediction is still in flight.  submit() is only
        # called from the single forward-pass thread, so no TOCTOU race.
        with self._lock:
            if next_layer in self._pending:
                return  # already in flight

        mx.eval(hidden_state)

        state = _LayerPrefetchState()
        with self._lock:
            self._pending[next_layer] = state

        try:
            self._predict_executor.submit(
                self._do_predict_and_io, layer_idx, hidden_state, state
            )
        except RuntimeError:
            with self._lock:
                self._pending.pop(next_layer, None)
            state.done.set()  # unblock any concurrent wait()

    def wait(self, layer_idx: int) -> None:
        """Block until any pending prefetch for *layer_idx* completes."""
        with self._lock:
            state = self._pending.pop(layer_idx, None)
        if state is None:
            return
        state.done.wait()

    # ------------------------------------------------------------------
    # Bulk prefetch (Path B — draft-informed)
    # ------------------------------------------------------------------

    def submit_bulk(
        self,
        layer_hidden_states: dict[int, mx.array],
    ) -> None:
        """Predict and prefetch neurons for multiple layers at once.

        Unlike :meth:`submit`, predictions run synchronously on the calling
        thread so that all I/O is queued before the caller starts the target
        forward pass.  This is important for speculative decoding (Path B)
        where the target pass starts immediately after ``submit_bulk``
        returns — serialising predictions on the background thread would
        delay later layers' I/O.

        .. warning::
           Must not be called while a ``submit()``-based prediction thread
           is in-flight — ``_predict()`` calls ``mx.eval`` internally, which
           would deadlock with the concurrent ``mx.eval`` on the prediction
           thread.  Current callers (``_submit_draft_prefetch``) invoke this
           between forward-pass steps when no prediction is in-flight.
        """
        for layer_idx, hidden in layer_hidden_states.items():
            if layer_idx >= self._num_layers:
                continue
            indices = self._predict(layer_idx, hidden)
            self._submit_io(layer_idx, indices)

    def cancel(self) -> None:
        """Cancel all in-flight prefetch I/O (e.g. on draft rejection).

        Already-completed reads remain in the cache (harmless).
        In-flight futures are allowed to finish but their results are
        discarded by clearing the pending map.
        """
        with self._lock:
            self._pending.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _do_predict_and_io(
        self,
        layer_idx: int,
        hidden_state: mx.array,
        state: _LayerPrefetchState,
    ) -> None:
        """Run on the prediction thread: predict next layer then submit I/O."""
        next_layer = layer_idx + 1
        try:
            if self._lookahead_bank is not None:
                indices = self._predict_lookahead(layer_idx, hidden_state)
            else:
                indices = self._predict(next_layer, hidden_state)
            self._enqueue_io(next_layer, indices, state)
        except Exception:
            logger.warning("Prediction failed for layer %d", next_layer, exc_info=True)
            with self._lock:
                self.stats.failures += 1
            state.done.set()

    def _predict(self, layer_idx: int, hidden_state: mx.array) -> list[int]:
        flat = hidden_state.reshape(-1, hidden_state.shape[-1])
        indices = self._predictor_bank.predict_layer(
            layer_idx,
            flat,
            threshold=self._threshold,
            min_neurons=self._min_neurons,
            max_neurons=self._max_neurons,
        )
        return cast(list[int], indices.tolist())

    def _predict_lookahead(self, layer_idx: int, hidden_state: mx.array) -> list[int]:
        assert self._lookahead_bank is not None
        flat = hidden_state.reshape(-1, hidden_state.shape[-1])
        indices = self._lookahead_bank.predict_next_layer(
            layer_idx,
            flat,
            threshold=self._threshold,
            min_neurons=self._min_neurons,
            max_neurons=self._max_neurons,
        )
        return cast(list[int], indices.tolist())

    def _enqueue_io(
        self,
        layer_idx: int,
        neuron_indices: list[int],
        state: _LayerPrefetchState,
    ) -> None:
        """Submit I/O to the thread pool for a pre-registered pending entry."""
        if not neuron_indices:
            state.done.set()
            return

        with self._lock:
            self.stats.submitted += 1

        def _do_prefetch():
            try:
                cached, missing = self._weight_store.get_cached_indices(
                    layer_idx, neuron_indices
                )
                with self._lock:
                    self.stats.cache_hits += len(cached)
                    self.stats.cache_misses += len(missing)
                if missing:
                    self._weight_store.prefetch_neurons(layer_idx, missing)
            except Exception:
                logger.warning(
                    "Prefetch I/O failed for layer %d", layer_idx, exc_info=True
                )
                with self._lock:
                    self.stats.failures += 1
            finally:
                state.done.set()

        try:
            self._executor.submit(_do_prefetch)
        except Exception:
            state.done.set()
            # Don't pop _pending: the entry may belong to a newer submit()
            # after cancel(). wait() and cancel() own _pending cleanup.
            with self._lock:
                self.stats.submitted -= 1
            logger.warning("Failed to submit prefetch for layer %d", layer_idx)

    def _submit_io(self, layer_idx: int, neuron_indices: list[int]) -> None:
        """Register a pending entry and submit I/O (used by synchronous submit_bulk)."""
        if not neuron_indices:
            return

        state = _LayerPrefetchState()
        with self._lock:
            if layer_idx in self._pending:
                return  # already in flight
            self._pending[layer_idx] = state
        self._enqueue_io(layer_idx, neuron_indices, state)

    def close(self) -> None:
        """Shut down both the prediction and I/O thread pools.

        Prediction executor is drained first since it submits work to the
        I/O executor.
        """
        self._predict_executor.shutdown(wait=True)
        self._executor.shutdown(wait=True)
