"""Speculative expert prefetcher for Flash-MoE.

While MoE layer L computes, a trained lookahead head predicts which experts
the NEXT MoE layer will route to and starts background SSD reads, so the next
layer's ``load_experts`` finds them cached. The same prediction's score
vector steers cache eviction (``ScoredLayerCache``) when enabled.

Concurrency contract (identical to the dense ``Prefetcher`` in
``prefetch.py``): ``mx.eval`` is not safe for concurrent calls, so the single
prediction thread is the ONLY background thread that evaluates arrays; the
calling thread materializes the hidden state before enqueueing, and must not
call ``mx.eval`` between ``submit()`` and the next ``wait()``. Prediction can
never affect correctness — a misprediction or failure only costs latency.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx

from olmlx.engine.flash.moe_predictor import MoeLookaheadBank
from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore
from olmlx.engine.flash.prefetch import PrefetchStats, _LayerPrefetchState

logger = logging.getLogger(__name__)


class MoePrefetcher:
    """Orchestrates background expert prefetching from SSD."""

    def __init__(
        self,
        bank: MoeLookaheadBank,
        weight_store: FlashMoeWeightStore,
        *,
        margin: float = 1.5,
        max_positions: int = 8,
        scored_eviction: bool = True,
        io_threads: int = 8,
    ):
        self._bank = bank
        self._weight_store = weight_store
        self._margin = margin
        self._max_positions = max_positions
        self._scored_eviction = scored_eviction
        self._io_executor = ThreadPoolExecutor(
            max_workers=io_threads, thread_name_prefix="moe-prefetch-io"
        )
        self._predict_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="moe-prefetch-predict"
        )
        self._lock = threading.Lock()
        self._pending: dict[int, _LayerPrefetchState] = {}
        self.stats = PrefetchStats()

    def submit(self, layer_idx: int, hidden_state: mx.array) -> None:
        """Predict the next MoE layer's experts and start background I/O.

        No-op when *layer_idx* has no successor head, the position count
        exceeds the prefill guard, or a prefetch for the successor is already
        pending. The hidden state is materialized on the calling thread
        (``mx.eval`` is not safe for concurrent calls).
        """
        next_layer = self._bank.next_moe_layer(layer_idx)
        if next_layer is None:
            return
        positions = hidden_state.size // hidden_state.shape[-1]
        if positions > self._max_positions:
            return  # prefill: most experts activate anyway, skip the work

        with self._lock:
            if next_layer in self._pending:
                return  # already in flight

        mx.eval(hidden_state)

        state = _LayerPrefetchState()
        with self._lock:
            self._pending[next_layer] = state
        try:
            self._predict_executor.submit(
                self._do_predict_and_io, layer_idx, next_layer, hidden_state, state
            )
        except RuntimeError:  # executor shut down
            with self._lock:
                self._pending.pop(next_layer, None)
            state.done.set()

    def wait(self, layer_idx: int) -> None:
        """Block until any pending prefetch targeting *layer_idx* completes."""
        with self._lock:
            state = self._pending.pop(layer_idx, None)
        if state is None:
            return
        state.done.wait()

    def _do_predict_and_io(
        self,
        layer_idx: int,
        next_layer: int,
        hidden_state: mx.array,
        state: _LayerPrefetchState,
    ) -> None:
        """Run on the prediction thread: predict, push scores, enqueue I/O."""
        try:
            result = self._bank.predict_next(
                layer_idx, hidden_state, margin=self._margin
            )
        except Exception:
            logger.warning(
                "Expert prediction failed for layer %d", next_layer, exc_info=True
            )
            with self._lock:
                self.stats.failures += 1
            result = None

        if result is None:
            state.done.set()
            return

        indices, scores = result
        if self._scored_eviction:
            # Push BEFORE the I/O lands so prefetch inserts already evict by
            # predicted need. The store clears these when next_layer's
            # forward consumes them. A failed push must not cancel the
            # prefetch itself — scores only steer eviction.
            try:
                self._weight_store.set_layer_scores(
                    next_layer, {i: float(scores[i]) for i in range(len(scores))}
                )
            except Exception:
                logger.warning(
                    "Expert score push failed for layer %d", next_layer, exc_info=True
                )
                with self._lock:
                    self.stats.failures += 1

        with self._lock:
            self.stats.submitted += 1

        def _do_prefetch() -> None:
            try:
                cached, fetched = self._weight_store.prefetch_experts(
                    next_layer, indices
                )
                with self._lock:
                    self.stats.cache_hits += cached
                    self.stats.cache_misses += fetched
            except Exception:
                logger.warning(
                    "Expert prefetch I/O failed for layer %d",
                    next_layer,
                    exc_info=True,
                )
                with self._lock:
                    self.stats.failures += 1
            finally:
                state.done.set()

        try:
            self._io_executor.submit(_do_prefetch)
        except RuntimeError:  # executor shut down
            state.done.set()
            with self._lock:
                self.stats.submitted -= 1

    def close(self) -> None:
        """Drain the prediction pool, then the I/O pool; log a summary.

        Prediction executor first — it submits work into the I/O executor.
        Idempotent: ThreadPoolExecutor.shutdown tolerates repeated calls.
        """
        self._predict_executor.shutdown(wait=True)
        self._io_executor.shutdown(wait=True)
        logger.info(
            "MoE prefetch stats: submitted=%d prefetched=%d already_cached=%d "
            "failures=%d",
            self.stats.submitted,
            self.stats.cache_misses,
            self.stats.cache_hits,
            self.stats.failures,
        )
