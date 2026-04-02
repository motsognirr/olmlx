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

        Uses a ``LookaheadBank`` (cross-layer predictor) when available,
        otherwise falls back to the sparsity predictor for layer L+1
        applied to layer L's hidden state.
        """
        next_layer = layer_idx + 1
        if next_layer >= self._num_layers:
            return

        if self._lookahead_bank is not None:
            indices = self._predict_lookahead(layer_idx, hidden_state)
        else:
            indices = self._predict(next_layer, hidden_state)
        self._submit_io(next_layer, indices)

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
        """Predict and prefetch neurons for multiple layers at once."""
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

    def _predict(self, layer_idx: int, hidden_state: mx.array) -> list[int]:
        flat = hidden_state.reshape(-1, hidden_state.shape[-1])
        indices = self._predictor_bank.predict_layer(
            layer_idx,
            flat,
            threshold=self._threshold,
            min_neurons=self._min_neurons,
            max_neurons=self._max_neurons,
        )
        return indices.tolist()

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
        return indices.tolist()

    def _submit_io(self, layer_idx: int, neuron_indices: list[int]) -> None:
        if not neuron_indices:
            return

        state = _LayerPrefetchState()
        with self._lock:
            if layer_idx in self._pending:
                return  # already in flight — don't overwrite
            self._pending[layer_idx] = state
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
                    self._weight_store.prefetch_neurons(
                        layer_idx, missing, executor=self._executor
                    )
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
            with self._lock:
                self._pending.pop(layer_idx, None)
                self.stats.submitted -= 1
            logger.warning("Failed to submit prefetch for layer %d", layer_idx)

    def close(self) -> None:
        """Shut down the prefetch thread pool."""
        self._executor.shutdown(wait=True)
