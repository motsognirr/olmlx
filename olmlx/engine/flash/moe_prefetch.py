"""Speculative expert prefetcher for Flash-MoE.

While MoE layer L computes, a trained lookahead head predicts which experts
the NEXT MoE layer will route to and starts background SSD reads, so the next
layer's ``load_experts`` finds them cached. The same prediction's score
vector steers cache eviction (``ScoredLayerCache``) when enabled.

Concurrency contract: prediction runs INLINE in ``submit()`` on the calling
thread, in pure NumPy (``MoeLookaheadBank.predict_next_np``) — no thread
other than the caller ever evaluates mx arrays, so there is no
background-eval rendezvous. Prefetch I/O is fully non-blocking: the forward
path never waits on it — ``load_experts`` races in-flight prefetch inserts
(the store's cache and protect() are built for that) and reads whatever is
still missing itself; a finished prefetch clears its own dedup slot. The
earlier design ran prediction on a dedicated thread whose ``mx.eval`` had
to be mutually excluded from the main thread's, and blocked each MoE layer
on the previous prediction+I/O — together ~40% of decode throughput (see
bench runs 20260710T*). Prediction can never affect correctness — a
misprediction or failure only costs latency (or a duplicate SSD read).

``submit()`` keeps its single-caller contract: it is only ever called from
the single forward-pass thread, so the pending-check and registration need
no atomicity across the two lock acquisitions.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

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
        # Build the NumPy head copies here, on the constructing thread —
        # the bank's mx weights were materialized on this thread by
        # load(), and predict_next_np must never touch mx afterwards.
        bank.ensure_np_heads()
        self._weight_store = weight_store
        self._margin = margin
        self._max_positions = max_positions
        self._scored_eviction = scored_eviction
        self._io_executor = ThreadPoolExecutor(
            max_workers=io_threads, thread_name_prefix="moe-prefetch-io"
        )
        self._lock = threading.Lock()
        self._pending: dict[int, _LayerPrefetchState] = {}
        self.stats = PrefetchStats()

    def submit(self, layer_idx: int, hidden_state: mx.array) -> None:
        """Predict the next MoE layer's experts and start background I/O.

        No-op when *layer_idx* has no successor head, the position count
        exceeds the prefill guard, or a prefetch for the successor is already
        pending. Prediction runs inline in NumPy on the calling thread; the
        hidden-state conversion below is the only (implicit) eval and it
        happens on the caller — nothing here touches mx from another thread.
        """
        next_layer = self._bank.next_moe_layer(layer_idx)
        if next_layer is None:
            return
        positions = hidden_state.size // hidden_state.shape[-1]
        if positions > self._max_positions:
            return  # prefill: most experts activate anyway, skip the work

        # submit() is only ever called from the single forward-pass thread
        # (single-caller contract), so check-then-register across the two
        # lock acquisitions below has no TOCTOU race.
        with self._lock:
            if next_layer in self._pending:
                return  # already in flight

        try:
            # float32 detour: numpy has no bfloat16. The np.array call is an
            # implicit eval on the calling thread — by the FlashMoE call
            # order (after ``mx.eval(inds)``) the hidden state is already
            # materialized, so this is a cast + copy, not a graph drive.
            flat = hidden_state.reshape(-1, hidden_state.shape[-1])
            hidden_np = np.array(flat.astype(mx.float32))
            result = self._bank.predict_next_np(
                layer_idx, hidden_np, margin=self._margin
            )
        except Exception:
            logger.warning(
                "Expert prediction failed for layer %d", next_layer, exc_info=True
            )
            with self._lock:
                self.stats.failures += 1
            return

        if result is None:
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

        state = _LayerPrefetchState()
        with self._lock:
            self._pending[next_layer] = state
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
                # Set done BEFORE clearing pending: a wait() that already
                # popped the state unblocks; one that arrives after the pop
                # finds nothing and returns — correct either way, since the
                # I/O is complete at both points.
                state.done.set()
                # Self-clear: nothing on the hot path calls wait() anymore,
                # so a finished prefetch must release its dedup slot or every
                # later submit for this layer would no-op against it. Guarded
                # by identity so a racing wait()+resubmit's fresh state is
                # never clobbered.
                with self._lock:
                    if self._pending.get(next_layer) is state:
                        del self._pending[next_layer]

        try:
            self._io_executor.submit(_do_prefetch)
        except RuntimeError:  # executor shut down
            with self._lock:
                self._pending.pop(next_layer, None)
                self.stats.submitted -= 1
            state.done.set()

    def wait(self, layer_idx: int) -> None:
        """Block until any pending prefetch I/O targeting *layer_idx* completes.

        NOT called on the forward path — prefetch is non-blocking there:
        ``load_experts`` races in-flight prefetch I/O and reads whatever is
        still missing itself (the store's cache tolerates the concurrent
        inserts; the cost is an occasional duplicate SSD read). Kept for
        tests and any caller that needs a completed prefetch as a barrier.
        """
        with self._lock:
            state = self._pending.pop(layer_idx, None)
        if state is None:
            return
        state.done.wait()

    def close(self) -> None:
        """Drain the I/O pool; log a summary.

        Idempotent: ThreadPoolExecutor.shutdown tolerates repeated calls.
        """
        self._io_executor.shutdown(wait=True)
        logger.info(
            "MoE prefetch stats: submitted=%d prefetched=%d already_cached=%d "
            "failures=%d",
            self.stats.submitted,
            self.stats.cache_misses,
            self.stats.cache_hits,
            self.stats.failures,
        )
