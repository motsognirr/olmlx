"""Continuous batching for concurrent text-chat requests (docs/batching-plan.md).

Two layers:

1. **Cache probes** (Phase 0). ``caches_batch_convertible`` mirrors
   mlx-lm's ``_make_cache`` (generate.py) exactly — that function raises
   for any cache type it cannot convert to a batch-aware cache, so a
   model is *mechanically* batchable only when every element of its
   prompt cache passes:

   - ``type(c) is KVCache`` (exact: subclasses like olmlx's quantized
     caches are rejected by mlx-lm and therefore here)
   - ``ArraysCache`` (batched natively via ``left_padding``)
   - ``RotatingKVCache`` with ``keep == 0`` (keep tokens unsupported)
   - ``CacheList`` of convertible elements

   ``caches_plain_kv`` is the stricter v1 *policy* probe: plain
   ``KVCache`` layers only. Hybrid ``ArraysCache`` models are
   mechanically batchable upstream but carry olmlx's #284/#396 GDN
   Metal-stream risk surface, so they stay on the exclusive path until
   they get dedicated parity coverage (plan §11).

2. **BatchScheduler** (Phase 1). One scheduler per LoadedModel. Requests
   are submitted from the event loop into a thread-safe inbox; a manager
   task acquires the global inference lock (injected — the scheduler is
   deliberately ignorant of ``engine.inference`` to avoid an import
   cycle) and runs one **worker thread** per busy period. The worker
   owns a fresh mlx-lm ``BatchGenerator`` for the period: it admits
   inbox sequences, ticks ``gen.next()`` (decode step + chunked-prefill
   interleave, all on mlx-lm's ``generation_stream`` — single thread,
   single stream, per the #284 discipline), dispatches per-uid events
   back to consumers via ``call_soon_threadsafe``, and exits when
   drained. ``BatchGenerator.close()`` synchronizes the generation
   stream before the manager releases the lock, preserving the "Metal
   work finished before release" invariant.

   Fairness: when an exclusive (non-batched) request starts waiting on
   the inference lock, ``exclusive_pending()`` turns true and the worker
   latches admission shut — running sequences finish, the lock is
   released to the FIFO waiter, and the manager re-queues behind it for
   the remaining inbox items.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable

from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)

logger = logging.getLogger(__name__)


def _cache_convertible(c: Any) -> bool:
    # Branch order matches mlx-lm's _make_cache.
    if type(c) is KVCache:
        return True
    if isinstance(c, ArraysCache):
        return True
    if isinstance(c, RotatingKVCache):
        return c.keep == 0
    if isinstance(c, CacheList):
        return all(_cache_convertible(sub) for sub in c.caches)
    return False


def caches_batch_convertible(caches: list[Any]) -> bool:
    """True when every per-layer cache can become a batch-aware cache.

    ``caches`` is what ``make_prompt_cache(model)`` / ``model.make_cache()``
    returns. An empty list means there is nothing to batch — not eligible.

    Materializes before checking: a Mock model's ``make_cache()`` returns
    a truthy object that iterates empty, which would otherwise vacuously
    pass ``all()`` — mocked models must never probe as batchable.
    """
    caches = list(caches)
    if not caches:
        return False
    return all(_cache_convertible(c) for c in caches)


def caches_plain_kv(caches: list[Any]) -> bool:
    """v1 policy probe: every layer is a plain ``KVCache`` (exact type).

    Dense full-attention models only — see module docstring for why
    hybrid/rotating layouts stay exclusive in v1. Materialized first for
    the same Mock-vacuity reason as ``caches_batch_convertible``.
    """
    caches = list(caches)
    if not caches:
        return False
    return all(type(c) is KVCache for c in caches)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


@dataclass
class BatchRequest:
    """Per-request inputs for a batched sequence."""

    tokens: list[int]
    max_tokens: int
    sampler: Callable[..., Any] | None = None
    logits_processors: list[Any] | None = None


class BatchSequence:
    """Handle linking one submitted request to its event stream.

    Events (dicts) arriving on ``out``:

    - ``{"type": "progress", "processed": int, "total": int}`` — prefill
    - ``{"type": "token", "token": int}`` — one generated token
    - ``{"type": "done", "reason": "stop" | "length" | "cancelled"}``
    - ``{"type": "error", "exc": BaseException}``
    """

    def __init__(self, request: BatchRequest, loop: asyncio.AbstractEventLoop):
        self.request = request
        self.loop = loop
        self.out: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.cancelled = threading.Event()
        self.uid: int | None = None

    def emit(self, event: dict[str, Any]) -> None:
        """Deliver an event to the consumer; safe from any thread."""
        try:
            self.loop.call_soon_threadsafe(self.out.put_nowait, event)
        except RuntimeError:
            # Event loop closed (shutdown) — the consumer is gone.
            pass


class BatchScheduler:
    """Continuous-batching scheduler for one model.

    All GPU work happens on a single worker thread per busy period; the
    inference lock is held (via the injected ``acquire_gpu``/
    ``release_gpu``) for exactly the worker's lifetime.
    """

    def __init__(
        self,
        *,
        generator_factory: Callable[[], Any],
        acquire_gpu: Callable[[], Awaitable[None]],
        release_gpu: Callable[[], None],
        exclusive_pending: Callable[[], bool] | None = None,
        name: str = "",
    ):
        self._generator_factory = generator_factory
        self._acquire_gpu = acquire_gpu
        self._release_gpu = release_gpu
        self._exclusive_pending = exclusive_pending or (lambda: False)
        self._name = name
        self._inbox: queue.Queue[BatchSequence] = queue.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wakeup: asyncio.Event | None = None
        self._manager_task: asyncio.Task[None] | None = None
        self._closing = False

    # -- consumer side (event loop) ------------------------------------

    async def submit(self, request: BatchRequest) -> BatchSequence:
        """Queue a request; returns the sequence handle to consume from."""
        if self._closing:
            raise RuntimeError("batch scheduler is closed")
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            if self._loop is not None and not self._loop.is_closed():
                raise RuntimeError(
                    "BatchScheduler is bound to a different running event loop"
                )
            # First use, or rebind after the previous loop was torn down
            # (a server keeps one loop for its lifetime; tests run one
            # loop per test). The old manager task died with its loop and
            # any queued sequences' consumers are gone — cancel them so
            # the new manager doesn't insert orphans.
            self._cancel_inbox()
            self._loop = loop
            self._wakeup = asyncio.Event()
            self._manager_task = None
        seq = BatchSequence(request, loop)
        self._inbox.put(seq)
        if self._manager_task is None or self._manager_task.done():
            self._manager_task = loop.create_task(self._manager())
        assert self._wakeup is not None
        self._wakeup.set()
        return seq

    def _cancel_inbox(self) -> None:
        try:
            while True:
                seq = self._inbox.get_nowait()
                seq.cancelled.set()
                seq.emit({"type": "done", "reason": "cancelled"})
        except queue.Empty:
            pass

    @staticmethod
    def cancel(seq: BatchSequence) -> None:
        """Request removal of a sequence (client disconnect / timeout).

        Idempotent; the worker frees the slot at its next tick and emits
        a ``done(cancelled)`` event."""
        seq.cancelled.set()

    def close(self) -> None:
        """Tear down (model unload). Cancels queued sequences and wakes
        the manager so it exits.

        Callers hold the active_refs contract — a model is only closed
        with no in-flight requests — so the worker is normally parked.
        If it is mid-drain, the cancellations flush it promptly.
        """
        self._closing = True
        self._cancel_inbox()
        if self._loop is not None and self._wakeup is not None:
            try:
                self._loop.call_soon_threadsafe(self._wakeup.set)
            except RuntimeError:
                pass

    # -- manager (event loop task) --------------------------------------

    async def _manager(self) -> None:
        assert self._wakeup is not None
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            if self._closing:
                return
            if self._inbox.empty():
                continue
            try:
                await self._acquire_gpu()
            except BaseException as exc:  # noqa: BLE001 — fail queued requests
                self._fail_inbox(exc)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                logger.exception("batch[%s]: GPU acquisition failed", self._name)
                continue
            try:
                await asyncio.to_thread(self._worker)
            finally:
                self._release_gpu()
            # Items may remain (fairness pause or arrivals during drain);
            # re-arm so the next loop iteration picks them up.
            if not self._inbox.empty() or self._closing:
                self._wakeup.set()

    def _fail_inbox(self, exc: BaseException) -> None:
        try:
            while True:
                seq = self._inbox.get_nowait()
                seq.emit({"type": "error", "exc": exc})
        except queue.Empty:
            pass

    # -- worker (dedicated thread, holds the GPU) ------------------------

    def _worker(self) -> None:
        from olmlx.engine.ropefix import safe_rope_patch

        active: dict[int, BatchSequence] = {}
        # Created lazily on first admission: a busy period that admits
        # nothing (pause latched at entry, or every inbox item already
        # cancelled) must not construct/close a BatchGenerator — that
        # would pointlessly toggle the wired limit and sync the stream.
        gen: Any = None
        paused = False
        try:
            with safe_rope_patch():
                while True:
                    # Latch: once an exclusive request is waiting on the
                    # lock, stop admitting for the rest of this busy
                    # period (it can only unblock after our release).
                    if not paused and self._exclusive_pending():
                        paused = True
                    if not paused and not self._closing:
                        gen = self._admit(gen, active)
                    if self._closing:
                        for seq in active.values():
                            seq.cancelled.set()
                    self._sweep_cancelled(gen, active)
                    if not active:
                        break
                    prompt_responses, gen_responses = gen.next()
                    for r in prompt_responses:
                        seq = active.get(r.uid)
                        if seq is not None:
                            seq.emit(
                                {
                                    "type": "progress",
                                    "processed": r.progress[0],
                                    "total": r.progress[1],
                                }
                            )
                    for r in gen_responses:
                        seq = active.get(r.uid)
                        if seq is None:
                            continue
                        # Mirror mlx-lm's batch_generate: the EOS step's
                        # token is not part of the output.
                        if r.finish_reason != "stop":
                            seq.emit({"type": "token", "token": r.token})
                        if r.finish_reason is not None:
                            seq.emit({"type": "done", "reason": r.finish_reason})
                            del active[r.uid]
        except BaseException as exc:  # noqa: BLE001 — fan failure out to consumers
            # A worker exception poisons the whole batch (shared forward
            # pass). Fail active AND queued sequences — retrying the
            # inbox against a generator that just blew up would loop.
            logger.exception("batch[%s]: worker failed", self._name)
            for seq in active.values():
                seq.emit({"type": "error", "exc": exc})
            self._fail_inbox(exc)
        finally:
            if gen is not None:
                try:
                    # Synchronizes generation_stream and restores the wired
                    # limit — the Metal barrier the lock release relies on.
                    gen.close()
                except Exception:
                    logger.exception("batch[%s]: generator close failed", self._name)

    def _admit(self, gen: Any, active: dict[int, BatchSequence]) -> Any:
        while True:
            try:
                seq = self._inbox.get_nowait()
            except queue.Empty:
                return gen
            if seq.cancelled.is_set():
                seq.emit({"type": "done", "reason": "cancelled"})
                continue
            if gen is None:
                gen = self._generator_factory()
            req = seq.request
            uid = gen.insert(
                [list(req.tokens)],
                [req.max_tokens],
                samplers=[req.sampler],
                # None entries break GenerationBatch's per-sequence
                # iteration when any *other* sequence has processors.
                logits_processors=[req.logits_processors or []],
            )[0]
            seq.uid = uid
            active[uid] = seq

    @staticmethod
    def _sweep_cancelled(gen: Any, active: dict[int, BatchSequence]) -> None:
        gone = [uid for uid, seq in active.items() if seq.cancelled.is_set()]
        if not gone:
            return
        gen.remove(gone)
        for uid in gone:
            seq = active.pop(uid)
            seq.emit({"type": "done", "reason": "cancelled"})
