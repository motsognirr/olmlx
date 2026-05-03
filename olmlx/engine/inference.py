import asyncio
import collections.abc
import contextlib
import dataclasses
import gc
import importlib
import itertools
import json
import logging
import threading
import time
import weakref
from collections.abc import AsyncGenerator
from typing import Any, Literal, overload

import mlx.core as mx

from olmlx.engine.model_manager import (
    CachedPromptState,
    LoadedModel,
    ModelManager,
    parse_keep_alive,
)
from olmlx.config import SyncMode, settings
from olmlx.utils import memory as memory_utils

try:
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    from mlx_lm.utils import common_prefix_len as _find_common_prefix
except ImportError:  # pragma: no cover
    make_prompt_cache = None  # type: ignore[assignment]
    trim_prompt_cache = None  # type: ignore[assignment]
    _find_common_prefix = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning(
        "mlx-lm prompt cache imports unavailable — prompt caching disabled"
    )

try:
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
except ImportError:  # pragma: no cover
    make_sampler = None  # type: ignore[assignment]
    make_logits_processors = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning(
        "mlx-lm sample_utils unavailable (mlx-lm < 0.30.7?) — sampler/logits_processors disabled"
    )
from olmlx.engine.template_caps import TemplateCaps
from olmlx.utils.streaming import async_mlx_stream
from olmlx.utils.timing import Timer, TimingStats

logger = logging.getLogger(__name__)

# gpt-oss special tokens used by the streaming filter
_GPT_OSS_STRUCTURAL_TOKENS = frozenset(
    {
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|call|>",
        "<|return|>",
    }
)


class _GptOssChannelFilter:
    """Stateful filter for gpt-oss channel tokens.

    Call ``should_yield(text)`` for each token. Returns True if the token's text
    should be sent to the client. After the stream ends, call
    ``get_fallback_texts()`` — if non-empty, yield those as fallback (the model
    produced analysis but no final channel).

    This is a class (not an async generator) so the caller can iterate the raw
    stream for prompt-cache token accumulation while only yielding filtered text.
    """

    _INIT = "init"
    _AFTER_START = "after_start"
    _EXPECT_CHANNEL = "expect_channel"
    _IN_BLOCK = "in_block"
    _CONTENT = "content"

    def __init__(self):
        self._state = self._INIT
        self._channel = None
        self._saw_any_channel = False
        self._saw_final = False
        self._analysis_texts: list[str] = []
        self._full_text_parts: list[str] = []

    def should_yield(self, text: str) -> bool:
        """Process one token's text and return whether it should be yielded."""
        self._full_text_parts.append(text)

        if text == "<|start|>":
            self._state = self._AFTER_START
            self._saw_any_channel = True
            return False

        if text == "<|channel|>":
            self._state = self._EXPECT_CHANNEL
            self._saw_any_channel = True
            return False

        if self._state == self._AFTER_START:
            return False

        if self._state == self._EXPECT_CHANNEL:
            self._channel = text.strip()
            self._state = self._IN_BLOCK
            if self._channel == "final":
                self._saw_final = True
            return False

        if text == "<|message|>" and self._state == self._IN_BLOCK:
            self._state = self._CONTENT
            return False

        if text in ("<|end|>", "<|call|>", "<|return|>"):
            self._state = self._INIT
            self._channel = None
            return False

        if self._state == self._CONTENT and self._channel == "final":
            return True

        if (
            self._state == self._CONTENT
            and self._channel == "analysis"
            and not self._saw_final
        ):
            self._analysis_texts.append(text)
            return False

        if (
            self._state == self._INIT
            and not self._saw_any_channel
            and text not in _GPT_OSS_STRUCTURAL_TOKENS
        ):
            return True

        return False

    def get_fallback_texts(self) -> list[str]:
        """Return buffered analysis texts if no final channel was seen."""
        if not self._saw_final and self._analysis_texts:
            return self._analysis_texts
        return []

    def get_full_text(self) -> str:
        """Return the complete raw text accumulated during streaming."""
        return "".join(self._full_text_parts)


async def _gpt_oss_filter(token_stream):
    """Async generator wrapper for backward compatibility with tests."""
    filt = _GptOssChannelFilter()
    buffered = []
    async for token in token_stream:
        if filt.should_yield(token.text):
            yield token
        else:
            buffered.append(token)
    for text in filt.get_fallback_texts():
        # Find matching token from buffer
        for tok in buffered:
            if tok.text == text:
                yield tok
                buffered.remove(tok)
                break


# -- Experimental: Distributed inference coordinator --
# Only set when OLMLX_EXPERIMENTAL_DISTRIBUTED=true; see set_distributed_coordinator().
_distributed_coordinator = None
_distributed_coordinator_lock = threading.Lock()


def set_distributed_coordinator(coordinator):
    """Set the distributed coordinator for broadcasting inference to workers."""
    global _distributed_coordinator
    with _distributed_coordinator_lock:
        _distributed_coordinator = coordinator


def _maybe_broadcast_distributed(
    lm,
    prompt_tokens: list[int],
    prompt_text: str,
    max_tokens: int,
    gen_kwargs: dict,
) -> None:
    """Broadcast inference params to distributed workers if applicable."""
    with _distributed_coordinator_lock:
        coord = _distributed_coordinator
    if coord is not None and lm.is_distributed:
        coord.broadcast_inference(
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_text,
            max_tokens=max_tokens,
            gen_kwargs=gen_kwargs,
        )
        from olmlx.engine.distributed import distributed_barrier

        distributed_barrier()


# Resolve generation streams at module load time to avoid repeated
# importlib.import_module() calls in the hot path (_safe_sync).
def _resolve_generation_streams() -> list[Any]:
    streams = []
    for mod_name in ("mlx_lm.generate", "mlx_vlm.generate"):
        try:
            mod = importlib.import_module(mod_name)
            streams.append(mod.generation_stream)
        except (ImportError, AttributeError):
            pass
    return streams


_generation_streams = _resolve_generation_streams()

# Metal does not support concurrent command buffer submission across any
# models — they all share the same Metal device and command queue.  A per-model
# lock would still allow interleaved GPU work from different models, risking
# crashes or corruption.  A single global lock is an intentional trade-off:
# we sacrifice parallelism for stability on Apple Silicon.
# A lock acquired on loop A and never released leaves ``_locked = True`` and
# a dead ``Future`` at the head of ``_waiters``.  Loop B then deadlocks trying
# to acquire — its own waiter Future is queued behind a Future whose
# ``set_result`` would need to run on loop A's (now-gone) scheduler.
# Test isolation relies on ``_reset_inference_state()`` force-releasing
# ``_inference_lock`` between tests.  Note: ``release()`` only clears
# ``_locked`` and wakes one pending waiter — it does NOT drain the
# ``_waiters`` deque.  Stale waiters there are normally cleaned up by the
# ``finally: self._waiters.remove(fut)`` clause inside cancelled
# ``acquire()`` calls; tests that need a truly fresh lock instance patch
# ``_inference_lock`` with ``asyncio.Lock()``.  ``_deferred_cleanup_locks``
# below uses a different strategy (per-loop WeakKeyDictionary) because
# force-releasing a lock held mid-cleanup would be unsafe.
_inference_lock = asyncio.Lock()
# A lock that was acquired on loop A (e.g. a test that crashed without
# releasing it) causes deadlock or "Future attached to a different loop"
# errors when another loop inherits it.  Per-loop keys ensure each test loop
# starts with a fresh, unlocked lock; WeakKeyDictionary lets closed test
# loops get garbage-collected without leaking.
_deferred_cleanup_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Lock
] = weakref.WeakKeyDictionary()
# Tasks are also keyed per-loop: a task bound to loop A cannot be cancelled or
# awaited from loop B (RuntimeError on Python 3.10+).  Keeping this per-loop
# keeps the reset path consistent with the lock scoping.
_deferred_cleanup_tasks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Task[None]
] = weakref.WeakKeyDictionary()
# Monotonic counter for cleanup-task names so the ERROR/WARNING log entries
# that key on ``task.get_name()`` have a collision-free identifier even when
# successive stream objects happen to reuse the same memory address.
_cleanup_counter = itertools.count()
# Tracks requests waiting for _inference_lock (not the _await_deferred_cleanup wait).
_queue_depth = 0


async def _reset_inference_state() -> None:
    """Reset module-level state for test isolation.

    Warning: Test-only. Must only be called when no coroutines are awaiting
    _inference_lock, otherwise they will block forever.

    Resets the global state variables that persist across tests. Call this in
    test fixtures to ensure test isolation.

    Cancels and awaits any in-flight deferred cleanup task to prevent
    orphaned release calls after the task completes.
    Force-releases _inference_lock if held to prevent test deadlocks.
    """
    global _queue_depth
    # Scope the reset to the calling loop so we don't wipe locks/tasks in use
    # by other loops (e.g. concurrent async test classes under loop_scope=session).
    loop = asyncio.get_running_loop()
    task = _deferred_cleanup_tasks.pop(loop, None)
    if task is not None:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError as cancel_exc:
                # If ``_cleanup`` raised in its body and the cancellation
                # arrived in its finally (e.g. at ``await lock.acquire()``),
                # the original exception lives on as ``__context__`` of the
                # ``CancelledError``.  Surface it so it isn't lost.  Nested
                # cancellation (``__context__`` is itself a ``CancelledError``)
                # is not an application error and shouldn't trigger this log.
                if cancel_exc.__context__ is not None and not isinstance(
                    cancel_exc.__context__, asyncio.CancelledError
                ):
                    logger.warning(
                        "Cleanup exception masked by cancellation during reset: %s",
                        cancel_exc.__context__,
                        exc_info=cancel_exc.__context__,
                    )
            except Exception as exc:
                # Race: the task was about to finish with a non-cancellation
                # exception when we called ``cancel()``, so ``await task``
                # re-raises the stored exception instead of ``CancelledError``.
                # Log and swallow — reset must not propagate user-code
                # exceptions up to the test teardown fixture.
                logger.warning(
                    "Cleanup task raised while being cancelled during reset: %s",
                    exc,
                    exc_info=exc,
                )
        elif not task.cancelled():
            # Consume any stored exception so asyncio doesn't log
            # "Task exception was never retrieved" to stderr.  Also log it
            # ourselves at warning level so fixture-level failures aren't
            # invisible (``_await_deferred_cleanup`` logs on its own path,
            # but reset is the only path for tests that aborted earlier).
            # ``task.exception()`` is safe here — the task is done and not
            # cancelled, so it can't raise ``InvalidStateError`` or ``CancelledError``.
            exc = task.exception()
            if exc is not None:
                logger.warning(
                    "Deferred cleanup task raised during reset: %s",
                    exc,
                    exc_info=exc,
                )
    # Also cleans up any lock entry ``_cleanup``'s finally block may have
    # created.  Both branches above can leave one behind:
    #   - ``if`` (cancel + await): ``_cleanup``'s finally runs during
    #     ``await task`` and calls ``_get_deferred_cleanup_lock()``, which
    #     creates a fresh lock entry before popping the task entry.
    #   - ``elif`` (already-done): we skip ``await task``, but the task
    #     may already have run its finally and left the same fresh entry.
    _deferred_cleanup_locks.pop(loop, None)
    # ``_queue_depth`` is intentionally global: it tracks waiters on the
    # global ``_inference_lock``, not per-loop cleanup state, so a per-loop
    # scope would make no sense here.
    _queue_depth = 0
    if _inference_lock.locked():
        _inference_lock.release()


def _get_inference_lock() -> asyncio.Lock:
    """Return the inference lock.

    Exists for API consistency with _get_deferred_cleanup_lock.
    The lock is created at module load time.
    """
    return _inference_lock


def _get_deferred_cleanup_lock() -> asyncio.Lock:
    """Lazily create a deferred cleanup lock keyed by the running event loop
    (Bug #119, Bug #243).

    A single cached lock breaks across event loops: stale ``_locked=True``
    state from a test that crashed without releasing it, or waiter Futures
    whose callbacks would fire on a closed loop, leak into the next test.
    Keying by the running loop keeps each loop's lock isolated; the weak
    dict lets closed test loops get garbage-collected.

    Safe within a single event loop: no await between the lookup and the
    assignment, so no two coroutines on the same loop can both observe
    ``None`` simultaneously.  WeakKeyDictionary is not thread-safe in the
    general case — GC-triggered key-removal callbacks can interleave with
    ``.get()`` / ``__setitem__`` from another thread.  Safe here because
    (a) asyncio is single-threaded and (b) on CPython the GIL serialises
    the GC callback against the dict operations.

    Must be called from within a running event loop — uses
    ``asyncio.get_running_loop()``, which raises ``RuntimeError`` if invoked
    from synchronous code.  All callers (``_await_deferred_cleanup``,
    ``_schedule_deferred_inference_cleanup``, ``_cleanup``'s finally) run
    inside running loops.  Test code accessing this directly must do so
    from an ``async def`` test or via ``loop.run_until_complete``.
    """
    loop = asyncio.get_running_loop()
    lock = _deferred_cleanup_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _deferred_cleanup_locks[loop] = lock
    return lock


def _sync_default_stream() -> None:
    try:
        mx.synchronize()
    except Exception:
        logger.debug("mx.synchronize() failed", exc_info=True)


def _sync_generation_streams() -> None:
    for stream in _generation_streams:
        try:
            mx.synchronize(stream)
        except Exception:
            logger.debug("generation_stream sync failed", exc_info=True)


def _safe_sync():
    """Synchronize Metal GPU state unconditionally, suppressing and logging errors.

    Syncs both the default stream and the generation stream (mlx_lm/mlx_vlm
    use a separate stream from the default). Callers rely on this being
    unconditional — notably cache-eviction and deferred-cleanup paths,
    which must sync regardless of ``settings.sync_mode``.
    """
    _sync_default_stream()
    _sync_generation_streams()


def _derive_timing_stats(
    stats: TimingStats,
    prompt_tps: float,
    gen_tps: float,
    eval_timer_ns: int,
) -> tuple[float, float]:
    """Populate ``stats.prompt_eval_duration`` and ``stats.eval_duration`` from
    mlx-lm's measured prefill/decode rates, with conservative fallbacks for
    cases where mlx-lm didn't report a rate.

    Returns the (possibly back-computed) ``(prompt_tps, gen_tps)`` so the
    caller's "Generation complete" log line remains informative when a rate
    was missing.

    Convention (matches Ollama): ``prompt_eval_duration`` covers prefill,
    ``eval_duration`` covers decode only. Both fields are clamped so their
    sum never exceeds ``eval_timer_ns``.
    """
    # Defensive coercion: third-party result objects sometimes carry
    # non-Python-numeric scalars (numpy/mlx) for these fields, and ad-hoc
    # MagicMock objects in tests would otherwise sneak through truthiness
    # checks. Treat anything non-numeric as missing.
    if not isinstance(prompt_tps, (int, float)):
        prompt_tps = 0.0
    if not isinstance(gen_tps, (int, float)):
        gen_tps = 0.0

    if prompt_tps > 0 and stats.prompt_eval_count > 0:
        # Clamp to wall-clock — rate noise can produce values exceeding the
        # actual elapsed time, which would otherwise break the sum invariant.
        stats.prompt_eval_duration = min(
            int(stats.prompt_eval_count / prompt_tps * 1e9),
            eval_timer_ns,
        )
    elif stats.prompt_eval_count > 0 and stats.eval_count == 0:
        # No decode happened — entire timer is prefill.
        stats.prompt_eval_duration = eval_timer_ns

    if gen_tps > 0 and stats.eval_count > 0:
        stats.eval_duration = min(
            int(stats.eval_count / gen_tps * 1e9),
            eval_timer_ns - stats.prompt_eval_duration,
        )
    elif stats.eval_count == 0:
        # No decode happened — match Ollama's convention.
        stats.eval_duration = 0
    else:
        # Subtract known prefill from the wall-clock timer. ``max(0, …)``
        # guards against the (post-clamp impossible) negative case.
        stats.eval_duration = max(0, eval_timer_ns - stats.prompt_eval_duration)

    # Symmetric back-compute: if only ``gen_tps`` was reported, recover
    # prefill from the wall-clock minus the (now known) decode duration.
    if (
        stats.prompt_eval_duration == 0
        and stats.prompt_eval_count > 0
        and stats.eval_duration > 0
        and eval_timer_ns > stats.eval_duration
    ):
        stats.prompt_eval_duration = eval_timer_ns - stats.eval_duration

    # Log-only fallback so the "Generation complete" line stays informative
    # when mlx-lm didn't report a rate directly.
    if gen_tps == 0 and stats.eval_duration and stats.eval_count:
        gen_tps = stats.eval_count / (stats.eval_duration / 1e9)
    if prompt_tps == 0 and stats.prompt_eval_duration and stats.prompt_eval_count:
        prompt_tps = stats.prompt_eval_count / (stats.prompt_eval_duration / 1e9)

    return prompt_tps, gen_tps


def _lock_boundary_sync(mode: SyncMode | None = None) -> None:
    """Sync Metal GPU state at inference-lock entry/exit with configurable scope.

    ``mode`` resolves per call (not cached) so a per-model override wins over
    the global default. Values:

    - ``"full"`` (default): identical to ``_safe_sync`` — sync default + all
      generation streams.
    - ``"minimal"``: sync the default stream only; skip the generation-stream
      loop. Safe because ``_generate_sync`` and mlx_lm's ``stream_generate``
      already synchronize the generation stream from inside the worker thread
      before it exits. **Same mlx_lm-internals assumption as ``"none"`` for
      the generation stream** — if mlx_lm ever drops that guarantee,
      ``"minimal"`` streaming is as unsafe as ``"none"`` for the
      generation-stream part (the default stream is still synced here).
    - ``"none"``: skip lock-boundary sync entirely. Safety depends on
      per-path guarantees that Metal work is complete before the lock
      releases:

      * ``_full_completion`` has ``_generate_sync`` → ``asyncio.to_thread``
        which blocks until the worker thread returns, and the thread
        calls ``mx.synchronize(generation_stream)`` before exit (see
        ``_generate_sync`` body below).
      * ``_stream_completion`` waits for ``drain_and_join``; mlx_lm's
        ``stream_generate`` synchronizes the generation stream inside
        its worker thread before exit. **This is an assumption about
        mlx_lm internals** — if that guarantee ever changes upstream,
        streaming under ``sync_mode="none"`` can reintroduce the
        "command encoder already encoding" Metal crash that the
        lock-boundary sync was added to prevent.
      * ``generate_embeddings`` runs synchronously with no worker thread
        and has its own load-bearing ``mx.synchronize()`` fallback
        specifically because the above assumption doesn't apply there.

    Cache-eviction and deferred-cleanup paths keep calling ``_safe_sync``
    directly — they are not lock-boundary calls and must always synchronize.
    """
    effective = mode if mode is not None else settings.sync_mode
    if effective == "none":
        return
    if effective == "minimal":
        _sync_default_stream()
        return
    if effective == "full":
        _sync_default_stream()
        _sync_generation_streams()
        return
    raise ValueError(f"Unknown sync_mode: {effective!r}")


class ServerBusyError(RuntimeError):
    """Raised when the server is recovering from a previous inference (deferred GPU cleanup)."""

    pass


_DEFERRED_CLEANUP_TIMEOUT = 600  # 10 minutes max wait for stuck thread
_DEFERRED_WAIT_TIMEOUT = 30.0  # max wait for deferred cleanup before rejecting


async def _await_deferred_cleanup():
    """Wait for any in-progress deferred GPU cleanup to complete.

    Raises ServerBusyError if cleanup doesn't finish within _DEFERRED_WAIT_TIMEOUT.
    Uses asyncio.wait() to avoid Python 3.11 wait_for race conditions.
    Uses _deferred_cleanup_lock to prevent TOCTOU races on _deferred_cleanup_tasks (Bug #119).
    """
    loop = asyncio.get_running_loop()
    async with _get_deferred_cleanup_lock():
        task = _deferred_cleanup_tasks.get(loop)
        if task is None or task.done():
            return
    # Wait outside the lock so _cleanup() can acquire it in its finally block
    # to remove the entry from _deferred_cleanup_tasks.  Holding the lock here
    # would deadlock.  Race safety: a concurrent _schedule_deferred_inference_cleanup
    # cannot replace the entry while we wait because _inference_lock is held by
    # the existing cleanup — no new inference (and thus no new cleanup) can be scheduled.
    logger.info("Waiting for deferred GPU cleanup to complete")
    done, _ = await asyncio.wait({task}, timeout=_DEFERRED_WAIT_TIMEOUT)
    if not done:
        raise ServerBusyError(
            f"Server busy: deferred GPU cleanup did not complete within {_DEFERRED_WAIT_TIMEOUT}s"
        )
    # ``asyncio.wait`` returns completed tasks regardless of whether they raised;
    # surface any exception so it isn't silently dropped on the return path.
    # Log-and-proceed is the same trade-off as the force-release in
    # ``_reset_inference_state``: if ``_cleanup`` raised, ``_safe_sync`` may
    # not have completed, so the next inference can run on top of dirty Metal
    # state (risking a Metal crash on the next request).  We accept that risk
    # because the alternative — refusing the next request — is also lose:
    # ``_inference_lock`` is already released and recovery would require
    # restarting the server.  The ERROR log is the only signal to the operator
    # that the cleanup failed; the request stream itself shows no failure.
    if not task.cancelled():
        exc = task.exception()
        if exc is not None:
            # ``task.get_name()`` lets operators correlate this entry with the
            # WARNING fired on the next request from
            # ``_schedule_deferred_inference_cleanup`` (same task name).
            logger.error(
                "Deferred inference cleanup [%s] raised; server state may be "
                "dirty, consider restart if this persists: %s",
                task.get_name(),
                exc,
                exc_info=exc,
            )


async def _schedule_deferred_inference_cleanup(stream) -> None:
    """Schedule deferred GPU cleanup when the inference thread is stuck.

    Polls the thread until it exits, then syncs Metal and releases the
    inference lock.  The lock remains held until the thread finishes to
    prevent concurrent Metal command buffer access.

    If the thread doesn't exit within _DEFERRED_CLEANUP_TIMEOUT seconds,
    releases the lock anyway (risk of Metal crash on next inference, but
    better than permanent deadlock).

    Uses _deferred_cleanup_lock to prevent TOCTOU races on _deferred_cleanup_tasks (Bug #119).
    """
    lock = _get_inference_lock()
    loop = asyncio.get_running_loop()

    async def _cleanup():
        thread = stream._thread
        deadline = time.monotonic() + _DEFERRED_CLEANUP_TIMEOUT
        try:
            while thread is not None and thread.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "Deferred inference cleanup: thread still alive after %ds — "
                        "releasing lock anyway (risk of Metal crash on next inference)",
                        _DEFERRED_CLEANUP_TIMEOUT,
                    )
                    break
                try:
                    wait = min(30, remaining)
                    await asyncio.to_thread(thread.join, wait)
                except BaseException as exc:
                    logger.warning(
                        "Deferred inference cleanup: poll loop aborted (%s) — "
                        "releasing lock (thread may still be alive)",
                        type(exc).__name__,
                    )
                    break  # finally will release the lock
            else:
                logger.info("Deferred inference cleanup: thread exited cleanly")
        finally:
            # ``_safe_sync`` already suppresses errors from its own
            # ``mx.synchronize`` calls, but wrap defensively so
            # ``lock.release()`` is truly unconditional — a future refactor
            # could change ``_safe_sync``'s error handling, and a failure
            # to release the inference lock would silently deadlock every
            # subsequent inference.
            try:
                if thread is None or not thread.is_alive():
                    _safe_sync()
            except Exception as sync_exc:
                logger.error(
                    "Deferred inference cleanup: _safe_sync raised; "
                    "Metal state may be dirty, consider restart if this "
                    "persists: %s",
                    sync_exc,
                    exc_info=sync_exc,
                )
            # Note: on timeout/abort with thread still alive, releasing the
            # lock risks a Metal crash on the next inference (the stuck thread
            # may still be issuing GPU commands).  Python can't kill CPU-bound
            # threads, so this is the "least bad" option vs permanent deadlock.
            lock.release()
            logger.info("Deferred inference cleanup: lock released")
            # Any ``CancelledError`` delivered at the ``__aenter__`` ``await``
            # below (cancellation during the acquire, or re-cancellation while
            # this finally is running) skips the pop.  In production that
            # leaves a stale done/cancelled entry in ``_deferred_cleanup_tasks``,
            # which is harmless: ``_inference_lock`` was already released
            # above so a new inference can proceed; ``_await_deferred_cleanup``
            # returns immediately for ``task.done() == True``; and the next
            # ``_schedule_deferred_inference_cleanup`` overwrites the entry.
            # In tests, ``_reset_inference_state`` pops both dicts explicitly
            # to keep per-test state clean — do not drop those fallback pops.
            # We deliberately do *not* wrap this with ``asyncio.shield`` —
            # the stale-entry path is benign, shielding would complicate
            # cancellation semantics, and the pop is not load-bearing.
            async with _get_deferred_cleanup_lock():
                # ``loop`` is captured from the enclosing function;
                # ``create_task`` binds the task to that loop, so using the
                # captured value is equivalent to ``asyncio.get_running_loop()``
                # here and avoids depending on that asyncio invariant.
                _deferred_cleanup_tasks.pop(loop, None)

    # IMPORTANT: ``create_task(_cleanup())`` must remain the last statement
    # in the ``async with _get_deferred_cleanup_lock()`` block below.
    # ``_cleanup``'s finally re-acquires the same lock to pop its dict entry;
    # this works because tasks don't preempt — the outer ``async with`` exits
    # and releases the lock before ``_cleanup`` is first scheduled.  Any
    # ``await`` inserted between ``create_task`` and end-of-block would let
    # the event loop schedule ``_cleanup`` into a deadlock on the same lock.
    async with _get_deferred_cleanup_lock():
        existing = _deferred_cleanup_tasks.get(loop)
        if existing is not None and not existing.done():
            logger.error(
                "Deferred inference cleanup already in progress — "
                "this should not happen while the inference lock is held"
            )
            return  # do not create a second task; the existing one will release the lock
        if existing is not None and not existing.cancelled():
            # ``existing`` is done — replacing the dict entry below drops the
            # last reference.  Consume any stored exception so asyncio doesn't
            # log "Task exception was never retrieved" when it's GC'd.  Log it
            # ourselves as a safety net in case the prior ``_await_deferred_cleanup``
            # log was missed (it normally fires first on the happy path).
            # ``existing.exception()`` is safe here — the task is done and not
            # cancelled, so it can't raise ``InvalidStateError`` or ``CancelledError``.
            # Note: if ``_await_deferred_cleanup`` already ran, this is the
            # second log of the same exception (ERROR there, WARNING here);
            # the matching ``[task name]`` prefix lets operators grep both
            # entries with one identifier.
            exc = existing.exception()
            if exc is not None:
                logger.warning(
                    "Replaced done cleanup task [%s] had raised "
                    "(may have been previously reported at ERROR by "
                    "_await_deferred_cleanup): %s",
                    existing.get_name(),
                    exc,
                    exc_info=exc,
                )

        # Name the task from a module-level monotonic counter so the
        # operator-facing log entries (ERROR from ``_await_deferred_cleanup``,
        # WARNING from the next request) share a collision-free identifier.
        # ``id(stream)`` would be reused after GC and could collide between
        # successive requests.
        _deferred_cleanup_tasks[loop] = asyncio.create_task(
            _cleanup(), name=f"deferred-cleanup-{next(_cleanup_counter)}"
        )


MEMORY_SAFETY_FACTOR = 1.3
"""Safety multiplier for KV cache memory estimates (Bug #125).

Metal alignment, intermediate buffers, and allocator overhead can cause actual
memory usage to exceed the raw 2-bytes-per-element calculation by 20-30%.
"""


def estimate_kv_cache_bytes(
    model: Any, num_tokens: int, *, kv_cache_quant: str | None = None
) -> int:
    """Estimate KV cache memory for a given number of tokens.

    Formula: sum_over_attn_layers(2 * kv_heads_i * head_dim) * num_tokens * bytes_per_element * MEMORY_SAFETY_FACTOR

    When *kv_cache_quant* is set (e.g. ``"turboquant:4"``), the per-head
    storage is reduced from ``head_dim * 2`` bytes (fp16) to the compressed
    size: ``head_dim / (8 / bits)`` packed-index bytes + 4 norm bytes.

    For NAS models (e.g. nemotron-nas) that have per-layer variable attention
    (some layers are no-op with self_attn=None, and KV head counts vary per
    layer), we introspect model.model.layers to count only actual attention
    layers and read their n_kv_heads.  Falls back to args-based estimation
    when layer introspection isn't possible.
    """
    if num_tokens <= 0:
        return 0

    # TurboQuant compression ratio.  Normal fp16 stores head_dim * 2 bytes
    # per K or V entry.  TurboQuant stores head_dim/(8/bits) packed-index
    # bytes + 4 float32 norm bytes.  We compute a multiplier <1 to scale
    # the fp16 estimate.  MLA models use a different cache layout so
    # TurboQuant does not apply there (ratio stays 1.0).
    _tq_ratio = 1.0  # applied to raw estimate before safety factor

    # mlx-lm text models: model.args
    # mlx-vlm vision-language models: model.language_model.args or .config
    # Wrapper args (e.g. Qwen3_5_MoE ModelArgs) carry only a ``text_config``
    # dict; the real attention fields live on ``model.language_model.args``.
    args = getattr(model, "args", None)
    args_owner: Any = model
    is_wrapper = (
        args is not None
        and hasattr(args, "text_config")
        and not hasattr(args, "num_attention_heads")
        and not hasattr(args, "kv_lora_rank")
    )
    if args is None or is_wrapper:
        lang_model = getattr(model, "language_model", None)
        inner_args = None
        if lang_model is not None:
            inner_args = getattr(lang_model, "args", None) or getattr(
                lang_model, "config", None
            )
        if inner_args is not None:
            args = inner_args
            args_owner = lang_model
        elif is_wrapper:
            # Fail loudly — otherwise we'd fall through to args.num_attention_heads
            # on the wrapper itself and crash with an opaque AttributeError.
            raise AttributeError(
                "model.args is a text_config wrapper but could not resolve "
                "inner attention config (model.language_model missing or has "
                "no 'args'/'config')"
            )
    if args is None:
        args = getattr(model, "config", None)
    if args is None:
        raise AttributeError(
            "Model has no 'args' attribute (checked model.args, "
            "model.language_model.args/config, model.config)"
        )

    # MLA (Multi-head Latent Attention) models like DeepSeek V3 compress the
    # KV cache to (kv_lora_rank + qk_rope_head_dim) per layer instead of
    # (2 * num_kv_heads * head_dim).  Detect via kv_lora_rank in model args.
    kv_lora_rank = getattr(args, "kv_lora_rank", None)
    if isinstance(kv_lora_rank, int) and kv_lora_rank > 0:
        qk_rope_head_dim = getattr(args, "qk_rope_head_dim", 0)
        num_layers = args.num_hidden_layers
        bytes_per_element = 2  # float16/bfloat16
        # MLA stores compressed_kv (kv_lora_rank dims) as keys and
        # k_pe (qk_rope_head_dim dims) as values, each with 1 effective head.
        raw = (
            num_layers
            * 2
            * (kv_lora_rank + qk_rope_head_dim)
            * num_tokens
            * bytes_per_element
        )
        return int(raw * MEMORY_SAFETY_FACTOR)

    num_heads = args.num_attention_heads
    head_dim = (
        args.head_dim if hasattr(args, "head_dim") else args.hidden_size // num_heads
    )
    bytes_per_element = 2  # float16/bfloat16

    if kv_cache_quant is not None:
        method, quant_bits = _parse_kv_cache_quant(kv_cache_quant)
        fp16_per_entry = head_dim * bytes_per_element
        if method == "turboquant":
            # TurboQuant: packed indices + float32 norm
            tq_per_entry = head_dim // (8 // quant_bits) + 4  # 4 bytes for f32 norm
            _tq_ratio = tq_per_entry / fp16_per_entry
        elif method == "spectral":
            # SpectralQuant: two packed regimes (semantic + tail) + float32 norm
            # Conservative estimate using avg_bits (actual varies per head)
            sq_per_entry = head_dim // (8 // quant_bits) + 4
            _tq_ratio = sq_per_entry / fp16_per_entry

    # Try layer introspection for NAS/variable-attention/hybrid models.
    # ``args_owner`` was set above to the component whose args we resolved
    # (model.language_model for VLMs/wrappers, else model) so we introspect
    # the correct layer tree and avoid hitting a vision encoder.
    inner = getattr(args_owner, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if isinstance(layers, (list, tuple)) and len(layers) > 0:
        # Per-layer accounting for hybrid attention (e.g. Gemma 4): some
        # layers may use sliding-window attention with a different
        # n_kv_heads/head_dim and a hard cap on cache depth, while others
        # use full attention with their own dimensions.
        sliding_window = getattr(args, "sliding_window", None)
        raw_total = 0
        found_attn_layer = False
        introspection_complete = True
        for layer in layers:
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                continue  # no-op attention layer — no KV cache
            layer_kv_heads = getattr(self_attn, "n_kv_heads", None)
            if not isinstance(layer_kv_heads, int):
                # Try alternate attribute name (e.g. Qwen3-Next uses
                # "num_key_value_heads" instead of "n_kv_heads")
                layer_kv_heads = getattr(self_attn, "num_key_value_heads", None)
            if not isinstance(layer_kv_heads, int):
                # Standard model — fall back to args
                introspection_complete = False
                break
            found_attn_layer = True
            # Per-layer head_dim falls back to the global head_dim if the
            # attention module doesn't expose its own as an int (most
            # uniform models).  isinstance check guards against test
            # MagicMocks auto-creating non-numeric attributes.
            attn_head_dim = getattr(self_attn, "head_dim", None)
            layer_head_dim = (
                attn_head_dim if isinstance(attn_head_dim, int) else head_dim
            )
            # Sliding-window attention: cap effective tokens at the window
            # size.  Use `is True` to avoid being fooled by truthy MagicMocks
            # in tests; production code sets a literal bool.  Prefer a
            # per-layer window if exposed (defensive — Gemma 4 today shares
            # a single window across all sliding layers via args, but a
            # future model could expose heterogeneous windows).
            is_sliding = getattr(self_attn, "is_sliding", None) is True
            layer_sw: int | None = None
            for attr in ("sliding_window_size", "sliding_window"):
                v = getattr(self_attn, attr, None)
                if isinstance(v, int) and v > 0:
                    layer_sw = v
                    break
            if layer_sw is None and isinstance(sliding_window, int):
                layer_sw = sliding_window
            if is_sliding and layer_sw is None:
                # A sliding-window layer with no resolvable window size
                # falls through to a full-prompt estimate (safe overestimate
                # — won't cause OOM, just a spurious 503 on long prompts).
                # Log so the condition is diagnosable without a debugger.
                logger.debug(
                    "Layer %d reports is_sliding=True but no window size "
                    "found on self_attn or args; using full token count for "
                    "KV estimation (safe overestimate)",
                    getattr(self_attn, "layer_idx", -1),
                )
            effective_tokens = (
                min(num_tokens, layer_sw)
                if is_sliding and layer_sw is not None
                else num_tokens
            )
            raw_total += (
                2
                * layer_kv_heads
                * layer_head_dim
                * effective_tokens
                * bytes_per_element
            )
        # Only trust introspection when every encountered layer reported its
        # KV heads.  found_attn_layer == False likely means the attention
        # module uses a different attribute name (e.g. "attention" instead of
        # "self_attn"); fall through to the args-based estimate in that case.
        if introspection_complete and found_attn_layer:
            return int(raw_total * _tq_ratio * MEMORY_SAFETY_FACTOR)

    # Fallback: uniform estimate from args
    num_layers = args.num_hidden_layers
    num_kv_heads = getattr(args, "num_key_value_heads", num_heads)
    raw = num_layers * 2 * num_kv_heads * head_dim * num_tokens * bytes_per_element
    return int(raw * _tq_ratio * MEMORY_SAFETY_FACTOR)


def tokenize_for_cache(tokenizer: Any, prompt_text: str) -> list[int]:
    """Tokenize prompt text matching stream_generate's tokenization logic.

    Must exactly replicate the BOS heuristic in mlx_lm.generate.stream_generate
    to avoid token sequence divergence (which would cause every request to be a
    cache miss).  stream_generate uses ``bos_token is None``, NOT ``not bos_token``.
    """
    bos = getattr(tokenizer, "bos_token", None)
    add_special = bos is None or not prompt_text.startswith(bos)
    return tokenizer.encode(prompt_text, add_special_tokens=add_special)


async def _acquire_inference_lock(timeout_override: float | None = None):
    """Acquire the inference lock with optional timeout.

    When *timeout_override* is provided it takes precedence over the global
    ``settings.inference_queue_timeout``.

    Uses asyncio.wait() instead of asyncio.wait_for() to avoid a known
    Python 3.11 race where wait_for can deliver the lock and then cancel,
    leaving the lock permanently held with no owner.
    """
    lock = _get_inference_lock()
    timeout = (
        timeout_override
        if timeout_override is not None
        else settings.inference_queue_timeout
    )
    if isinstance(timeout, (int, float)) and timeout > 0:
        acquire_task = asyncio.create_task(lock.acquire())
        try:
            done, _ = await asyncio.wait({acquire_task}, timeout=timeout)
        except BaseException:
            acquire_task.cancel()
            try:
                await acquire_task
                lock.release()
            except asyncio.CancelledError:
                pass
            raise
        if not done:
            acquire_task.cancel()
            try:
                await acquire_task
                lock.release()
            except asyncio.CancelledError:
                pass
            raise ServerBusyError(
                f"Server busy: inference queue timeout after {timeout}s"
            )
    else:
        await lock.acquire()


@contextlib.asynccontextmanager
async def _inference_locked(
    timeout_override: float | None = None,
    *,
    sync_mode: SyncMode | None = None,
):
    """Async context manager that acquires the inference lock with Metal sync on entry/exit.

    ``sync_mode`` controls lock-boundary sync behavior (see
    ``_lock_boundary_sync``). ``None`` defers to the global
    ``settings.sync_mode``.
    """
    global _queue_depth
    lock = _get_inference_lock()
    await _await_deferred_cleanup()
    _queue_depth += 1
    if _queue_depth > 1:
        logger.info("Request queued for inference lock (queue depth: %d)", _queue_depth)
    try:
        await _acquire_inference_lock(timeout_override)
    except BaseException:
        _queue_depth -= 1
        raise
    _queue_depth -= 1
    # Re-check after acquiring — a deferred cleanup task may have been
    # created between the pre-check and acquire (TOCTOU window).
    try:
        await _await_deferred_cleanup()
    except BaseException:
        lock.release()
        raise
    # Sync the default Metal stream so any pending GPU work from the previous
    # inference completes before we start a new one.
    try:
        _lock_boundary_sync(sync_mode)
    except BaseException:
        # BaseException (not ValueError): this handler re-raises, so
        # KeyboardInterrupt / SystemExit from mx.synchronize must be
        # caught here long enough to release the lock before they
        # propagate. Asymmetric with the exit handler below, which
        # narrows to ValueError because it suppresses (the broader catch
        # there would silently swallow shutdown signals).
        lock.release()
        raise
    try:
        yield
    finally:
        # Sync again on exit to ensure this inference's GPU work is fully
        # complete before releasing the lock to the next caller.
        try:
            _lock_boundary_sync(sync_mode)
        except ValueError:
            # Do NOT re-raise: that would mask any exception propagating
            # from the yield body. Fall back to _safe_sync() so unknown
            # modes fail-safe (sync anyway) instead of fail-open (no sync).
            # Narrow to ValueError (not BaseException) so KeyboardInterrupt /
            # SystemExit from mx.synchronize still propagate — those must
            # not be silently swallowed during shutdown.
            logger.error("exit _lock_boundary_sync failed", exc_info=True)
            _safe_sync()
        finally:
            lock.release()


@contextlib.contextmanager
def _inference_ref(lm: LoadedModel):
    """Track active inference on a model to prevent expiry during use.

    Note: there is a small window between ``ensure_loaded()`` (which returns
    the LoadedModel) and the point where ``_inference_ref`` increments
    ``active_refs``.  During that window the expiry checker could remove the
    model from ``_loaded``.  This is **safe**: the caller already holds a
    Python reference to the LoadedModel object, so the model and tokenizer
    remain alive in memory.  The only side-effect is that the next request
    would re-load the model into ``_loaded``.

    Bug #118: Python ``+=`` on int is not atomic. Concurrent async tasks can
    race on ``active_refs``. Use the model's ``_active_refs_lock`` to protect
    increments and decrements.
    """
    with lm._active_refs_lock:
        lm.active_refs += 1
    try:
        yield
    finally:
        with lm._active_refs_lock:
            lm.active_refs -= 1
        # Refresh expiry so the model doesn't expire immediately after inference
        ka = parse_keep_alive(settings.default_keep_alive)
        if ka is not None:
            lm.expires_at = time.time() + ka


def _build_generate_kwargs(options: dict | None, is_vlm: bool = False) -> dict:
    """Convert Ollama options dict to mlx_lm/mlx_vlm generate kwargs.

    For text models (mlx-lm ≥ 0.30.7), sampling params are folded into a
    ``sampler`` callable via ``make_sampler``, and penalty params into a
    ``logits_processors`` list via ``make_logits_processors``.

    For VLMs (mlx-vlm), params are passed directly as before.
    """
    if not options:
        return {}
    kwargs = {}

    if is_vlm:
        # mlx-vlm still accepts direct keyword arguments
        vlm_mappings = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "seed": "seed",
            "num_predict": "max_tokens",
            "repeat_penalty": "repetition_penalty",
            "repeat_last_n": "repetition_context_size",
            "min_p": "min_p",
        }
        for ollama_key, mlx_key in vlm_mappings.items():
            if ollama_key in options:
                kwargs[mlx_key] = options[ollama_key]
    else:
        # mlx-lm ≥ 0.30.7: sampling via make_sampler / make_logits_processors
        sampler_args = {}
        sampling_map = {
            "temperature": "temp",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
        }
        for ollama_key, sampler_key in sampling_map.items():
            if ollama_key in options:
                sampler_args[sampler_key] = options[ollama_key]
        # Only build sampler when temperature is explicitly set — make_sampler
        # defaults temp=0.0 (greedy), which makes top_k/top_p/min_p irrelevant.
        if sampler_args and "temp" in sampler_args:
            if make_sampler is None:
                raise RuntimeError("mlx-lm is not installed; cannot build sampler")
            kwargs["sampler"] = make_sampler(**sampler_args)
        elif sampler_args:
            logger.warning(
                "top_k/top_p/min_p provided without temperature; no sampler "
                "will be built and these params will have no effect"
            )

        # Collect penalty params — only build processors when repeat_penalty
        # is present; repeat_last_n alone is a no-op (no penalty to apply).
        if "repeat_penalty" in options:
            penalty_args = {"repetition_penalty": options["repeat_penalty"]}
            if "repeat_last_n" in options:
                penalty_args["repetition_context_size"] = options["repeat_last_n"]
            if make_logits_processors is None:
                raise RuntimeError(
                    "mlx-lm is not installed; cannot build logits processors"
                )
            kwargs["logits_processors"] = make_logits_processors(**penalty_args)
        elif "repeat_last_n" in options:
            logger.warning(
                "repeat_last_n without repeat_penalty has no effect; ignored"
            )

        if "num_predict" in options:
            kwargs["max_tokens"] = options["num_predict"]

        # Forward seed so _apply_seed can consume it before generation
        if "seed" in options:
            kwargs["seed"] = options["seed"]

        if "stop" in options:
            logger.warning("stop sequences not supported by mlx-lm >= 0.30.7; ignored")

        for penalty_key in ("frequency_penalty", "presence_penalty"):
            if penalty_key in options:
                logger.warning(
                    "%s not supported by mlx-lm >= 0.30.7; ignored", penalty_key
                )

    return kwargs


def _apply_seed(kwargs: dict, *, consume: bool = True) -> None:
    """Read ``seed`` from *kwargs* and set the MLX RNG state.

    Must be called from the inference thread, not the event loop.

    Args:
        kwargs: Generate kwargs dict (may contain ``seed``).
        consume: If True, pop the key so it is not forwarded to the
                 underlying generate call (required for mlx-lm which
                 does not accept a ``seed`` kwarg).  If False, the key
                 is left in place (VLMs forward it to mlx-vlm).
    """
    seed = kwargs.pop("seed", None) if consume else kwargs.get("seed", None)
    if seed is not None:
        mx.random.seed(seed)


def _inject_tools_into_system(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Inject tool descriptions into the system message when the template doesn't support tools natively."""
    tool_desc_parts = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        tool_desc_parts.append(
            f"- {name}: {desc}\n  Parameters: {json.dumps(params, indent=2)}"
        )
    tool_block = (
        "You have access to the following tools. To call a tool, output a JSON object "
        'with "name" and "arguments" keys.\n\n'
        "Available tools:\n" + "\n".join(tool_desc_parts)
    )

    messages = list(messages)  # shallow copy
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            **messages[0],
            "content": messages[0]["content"] + "\n\n" + tool_block,
        }
    else:
        messages.insert(0, {"role": "system", "content": tool_block})
    return messages


_NATIVE_TOOL_HINT = (
    "Disregard any tool call format instructions above. "
    "You MUST use only the native tool call format provided by the system."
)

# Substrings that indicate the system message contains client-injected
# tool-format instructions targeting a non-native format.  These are formats
# clients embed as a fallback for models without native tool support, but
# they conflict with templates that DO have native tool support and confuse
# the model at long prompt lengths.
_CLIENT_TOOL_FORMAT_PATTERNS = (
    "<function=",  # Llama 3.x style — used by opencode, Claude Code
    "[TOOL_CALLS]",  # Mistral style
    "<|python_tag|>",  # Llama 3.x JSON style
    # NB: `<tool_call>` is intentionally absent — it's Qwen's *native* tool
    # call format token, so it would false-positive on Qwen models where the
    # client's instructions match the native format.  The opencode/Claude
    # Code case is still caught by `<function=`, which appears alongside
    # `<tool_call>` in their format examples.
)


def _add_native_tool_hint(
    messages: list[dict], native_template_text: str = ""
) -> list[dict]:
    """Append a hint to the system message to use native tool call format.

    Clients like opencode and Claude Code embed their own tool-format
    instructions (e.g. ``<function=Name>``) in the system message.  These
    conflict with the model's native tool call format (e.g. Gemma 4's
    ``<|tool_call>call:Name{...}<tool_call|>``).  At long prompt lengths
    the model follows the client's text instructions instead of using native
    tokens, producing unparseable output.

    A short override at the end of the system message steers the model back
    to the native format without modifying the client's original content.

    **Scope: this is intentionally general, not Gemma 4-specific.**
    Any model with native tool support (Qwen3, Mistral, Llama 3.x, Gemma 4,
    etc.) can be confused by client-injected non-native format instructions
    at long prompt lengths.  Gemma 4 is just where the symptom was first
    observed.  The patterns matched are the formats clients inject as a
    fallback for models *without* native tool support — when the model DOES
    have native support, the template format is authoritative.

    Only applied when the system message contains a conflict pattern that
    is NOT also present in the model's own chat template.  This prevents
    false positives for models like Mistral whose template natively contains
    ``[TOOL_CALLS]``: in that case the client's instructions match the
    model's native format and the "Disregard" override would suppress
    legitimate guidance.  Pass ``native_template_text`` from the call site;
    omit it (or pass empty) to skip the suppression check.
    """
    if not messages or messages[0].get("role") != "system":
        return messages
    content = messages[0].get("content", "")
    # Multimodal content (list of parts) is not handled — the conflict pattern
    # is text-only and the hint targets text-only system messages.
    if not isinstance(content, str):
        return messages
    if _NATIVE_TOOL_HINT in content:
        return messages
    # A pattern is only a "conflict" if it appears in the system message
    # AND the model's own template doesn't use it natively.
    triggered = [
        p
        for p in _CLIENT_TOOL_FORMAT_PATTERNS
        if p in content and p not in native_template_text
    ]
    if not triggered:
        return messages
    messages = list(messages)  # shallow copy
    messages[0] = {
        **messages[0],
        "content": content + "\n\n" + _NATIVE_TOOL_HINT,
    }
    return messages


def _get_chat_template_text(tokenizer: Any) -> str:
    """Extract the chat template as a single string for substring matching.

    Handles both text tokenizers (chat_template directly) and VLM processors
    (chat_template on the wrapped tokenizer).  Lists of named templates are
    flattened into a single space-joined string.
    """
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl is None:
        sub = getattr(tokenizer, "tokenizer", None)
        if sub is not None:
            tpl = getattr(sub, "chat_template", None)
    if tpl is None:
        return ""
    if isinstance(tpl, list):
        return " ".join(t.get("template", "") for t in tpl if isinstance(t, dict))
    return tpl if isinstance(tpl, str) else ""


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None = None,
    caps: TemplateCaps | None = None,
    *,
    tokenize: bool = False,
    enable_thinking: bool | None = None,
) -> Any:
    """Core chat template application.

    Uses TemplateCaps to decide which kwargs to pass, avoiding blind try/except.
    Returns str when tokenize=False, token list/dict when tokenize=True.
    """
    if caps is None:
        caps = TemplateCaps()

    kwargs: dict[str, Any] = {"tokenize": tokenize, "add_generation_prompt": True}

    if tools and caps.supports_tools:
        kwargs["tools"] = tools
    elif tools and not caps.supports_tools:
        logger.info(
            "Template lacks tool support, injecting tool descriptions into system message"
        )
        messages = _inject_tools_into_system(messages, tools)

    if caps.supports_enable_thinking:
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        elif tools:
            kwargs["enable_thinking"] = (
                False  # backward compat for non-Anthropic callers
            )
        else:
            kwargs["enable_thinking"] = True

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as exc:
        # If tools kwarg caused the error, retry without it (injecting instead)
        if tools and "tools" in kwargs:
            logger.warning(
                "apply_chat_template failed with tools kwarg (%s), retrying with injection",
                exc,
            )
            del kwargs["tools"]
            # Keep enable_thinking — it's independent of the tools kwarg failure
            messages = _inject_tools_into_system(messages, tools)
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except Exception as exc2:
                raise RuntimeError(
                    f"Chat template failed even without tools: {exc2}"
                ) from exc2
        raise RuntimeError(f"Chat template failed: {exc}") from exc


def apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None = None,
    caps: TemplateCaps | None = None,
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Apply chat template for text-only models (mlx-lm), returning prompt text."""
    return _apply_chat_template(
        tokenizer,
        messages,
        tools,
        caps,
        tokenize=False,
        enable_thinking=enable_thinking,
    )


def _normalize_tool_calls_in_messages(messages: list[dict]) -> list[dict]:
    """Normalise tool_calls in assistant messages for chat templates.

    Different chat templates expect different tool_call layouts:

    * **Qwen / Llama**: flat ``{name, arguments: dict}``
    * **Gemma 4**: nested ``{function: {name, arguments}, id, type}``

    Rather than guessing which layout a template needs, this function
    produces a *union* dict that satisfies both::

        {
            "name": "read",
            "arguments": {"path": "/foo"},
            "function": {"name": "read", "arguments": {"path": "/foo"}},
            "id": "call_x",
            "type": "function",
        }

    It also ensures ``arguments`` is always a parsed dict (never a JSON
    string), which is what both Qwen's ``|items`` filter and Gemma's
    ``is mapping`` test require.
    """
    result = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m = dict(m)
            normalised = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                name = fn.get("name", tc.get("name", ""))
                args = fn.get("arguments", tc.get("arguments", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                normalised.append(
                    {
                        "name": name,
                        "arguments": args,
                        "function": {"name": name, "arguments": args},
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                    }
                )
            m["tool_calls"] = normalised
            result.append(m)
        else:
            result.append(m)
    return result


def _convert_tool_messages_to_responses(messages: list[dict]) -> list[dict]:
    """Convert ``role: "tool"`` messages to ``tool_responses`` format.

    Some models (e.g. Gemma 4) don't support OpenAI-style ``role: "tool"``
    messages.  Instead they expect a ``tool_responses`` array merged into the
    preceding assistant message that made the ``tool_calls``.  This keeps
    tool responses inside the model turn, which is critical — the template
    omits ``<turn|>`` after tool_responses so the model continues in the same
    turn.  Placing them on a separate user message would put the model's
    generation inside a user turn, producing degenerate output.

    For *intermediate* assistant messages (followed by more messages), a
    newline content placeholder ensures the template emits ``<turn|>`` to
    properly close the model turn before the next turn opens.  The last
    assistant message with tool_responses keeps empty content so the model
    continues generating in the same turn.
    """
    if not any(m.get("role") == "tool" for m in messages):
        return messages

    # Build a mapping from tool_call_id → function name across all assistant messages.
    id_to_name: dict[str, str] = {}
    for m in messages:
        for tc in m.get("tool_calls", []):
            tc_id = tc.get("id", "")
            fn = tc.get("function", {})
            name = fn.get("name", "") if isinstance(fn, dict) else ""
            if tc_id and name:
                id_to_name[tc_id] = name

    result: list[dict] = []
    for m in messages:
        if m.get("role") == "tool":
            tc_id = m.get("tool_call_id", "")
            name = id_to_name.get(tc_id, "unknown")
            resp = {"name": name, "response": m.get("content", "")}
            # Merge into the preceding assistant message.
            prev = result[-1] if result else None
            if prev and prev.get("role") == "assistant":
                prev.setdefault("tool_responses", []).append(resp)
            else:
                # No preceding assistant — shouldn't happen, but create a
                # model-role message to keep the turn correct.
                result.append(
                    {"role": "assistant", "content": "", "tool_responses": [resp]}
                )
        else:
            result.append(dict(m))  # shallow copy to avoid mutating input

    # Ensure intermediate assistant messages with tool_responses get their
    # model turn closed.  The template omits <turn|> when tool_responses is
    # present and content is falsy.  Setting content to "\n" makes it truthy
    # (triggering <turn|>) while rendering as empty after strip_thinking().
    # The *last* such message keeps empty content so the model can continue
    # generating in the same turn.
    for i in range(len(result) - 1):
        m = result[i]
        if (
            m.get("role") == "assistant"
            and m.get("tool_responses")
            and not m.get("content")
        ):
            m["content"] = "\n"

    return result


def _apply_chat_template_vlm(
    processor: Any,
    model: Any,
    messages: list[dict],
    images: list[str] | None = None,
    tools: list[dict] | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """Apply chat template for vision-language models (mlx-vlm).

    When tools are provided, bypasses mlx_vlm.apply_chat_template and calls
    the processor's tokenizer directly.  mlx_vlm's message processing wraps
    text content in ``[{type: text, text: ..., content: ...}]`` dicts, which
    the Jinja template renders as Python list repr — garbling the prompt.
    """
    if tools:
        if images:
            logger.warning(
                "VLM native-tools path does not support images; "
                "%d image(s) will be ignored",
                len(images),
            )
        # Use tokenizer directly to get clean native tool formatting.
        tok = (
            processor.tokenizer
            if hasattr(processor, "tokenizer")
            and hasattr(processor.tokenizer, "apply_chat_template")
            else processor
        )
        kwargs: dict = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

    import mlx_vlm

    config = model.config if hasattr(model, "config") else {}
    num_images = len(images) if images else 0
    # Pass the full message list so the model gets proper conversation context
    result = mlx_vlm.apply_chat_template(
        processor, config, messages, num_images=num_images
    )
    if not isinstance(result, str):
        raise TypeError(
            f"mlx_vlm.apply_chat_template returned non-str ({type(result).__name__}); "
            "expected tokenize=False default"
        )
    return result


def _get_model_for_cache(model: Any, is_vlm: bool) -> Any:
    """Get the language model for cache creation.

    For text models (mlx-lm), returns the model directly.
    For VLM models (mlx-vlm), returns model.language_model.
    """
    if is_vlm:
        return getattr(model, "language_model", model)
    return model


def _make_turboquant_prompt_cache(model: Any, bits: int, is_vlm: bool = False) -> list:
    """Create a TurboQuant-compressed prompt cache for the model."""
    from olmlx.engine.turboquant_cache import make_turboquant_cache

    cache_model = _get_model_for_cache(model, is_vlm)
    return make_turboquant_cache(cache_model, bits=bits)


def _make_spectral_prompt_cache(
    model: Any, bits: int, calibration_dir: Any, is_vlm: bool = False
) -> list:
    """Create a SpectralQuant-compressed prompt cache for the model."""
    from olmlx.engine.spectralquant_cache import make_spectral_cache

    cache_model = _get_model_for_cache(model, is_vlm)
    return make_spectral_cache(cache_model, calibration_dir, avg_bits=bits)


def _parse_kv_cache_quant(spec: str) -> tuple[str, int]:
    """Split an `OLMLX_EXPERIMENTAL_KV_CACHE_QUANT` value like `"spectral:4"`
    into `(method, bits)`.  Format is validated at config load time."""
    method, bits_str = spec.split(":")
    return method, int(bits_str)


def _make_prompt_cache_for_lm(lm: LoadedModel) -> list:
    """Create a fresh prompt cache for `lm` using the configured factory
    (plain mlx-lm, TurboQuant, or SpectralQuant).  Single source of truth
    for cache creation."""
    if lm.kv_cache_quant is not None:
        method, bits = _parse_kv_cache_quant(lm.kv_cache_quant)
        if method == "spectral":
            return _make_spectral_prompt_cache(
                lm.model, bits, lm.spectral_calibration_dir, is_vlm=lm.is_vlm
            )
        return _make_turboquant_prompt_cache(lm.model, bits, is_vlm=lm.is_vlm)
    cache_model = _get_model_for_cache(lm.model, lm.is_vlm)
    return make_prompt_cache(cache_model)


def _extract_images(messages: list[dict]) -> list[str] | None:
    """Extract image URLs/paths from message content."""
    images = []
    for msg in messages:
        if msg.get("images"):
            images.extend(msg["images"])
    return images if images else None


def count_chat_tokens(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None = None,
    caps: TemplateCaps | None = None,
    *,
    enable_thinking: bool | None = None,
) -> int:
    """Count input tokens by applying the chat template with tokenize=True.

    No GPU inference needed — CPU-only tokenization.  Uses
    add_generation_prompt=True so the count includes the assistant-turn
    opener tokens, matching what the model actually receives at inference.
    """
    result = _apply_chat_template(
        tokenizer, messages, tools, caps, tokenize=True, enable_thinking=enable_thinking
    )

    # Handle varied return types from apply_chat_template.
    # BatchEncoding (transformers) extends UserDict, not dict, so use Mapping.
    if isinstance(result, collections.abc.Mapping):
        tokens = result.get("input_ids")
        if tokens is None:
            raise TypeError(
                f"apply_chat_template returned dict without 'input_ids': keys={list(result.keys())}"
            )
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
    elif isinstance(result, list) and result and isinstance(result[0], list):
        tokens = result[0]
    elif isinstance(result, list):
        tokens = result
    else:
        raise TypeError(
            f"Unexpected return type from apply_chat_template: {type(result)}"
        )

    return len(tokens)


async def generate_completion(
    manager: ModelManager,
    model_name: str,
    prompt: str,
    options: dict | None = None,
    stream: bool = True,
    keep_alive: str | None = None,
    max_tokens: int = 512,
    images: list[str] | None = None,
    apply_chat_template: bool = False,
    system: str | None = None,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a text completion, streaming or not.

    When *apply_chat_template* is True the raw prompt is wrapped in chat
    messages and run through the model's chat template before generation.
    If *system* is provided, it becomes a ``{"role": "system"}`` message.
    This is needed for chat-only models (e.g. Nemotron-H) that require the
    template framing to produce meaningful output.
    """
    stats = TimingStats()

    with Timer() as load_timer:
        lm = await manager.ensure_loaded(model_name, keep_alive)
    stats.load_duration = load_timer.duration_ns

    if apply_chat_template and not lm.is_vlm:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            prompt = apply_chat_template_text(
                lm.text_tokenizer,
                messages,
                caps=lm.template_caps,
                enable_thinking=False,
            )
            logger.info(
                "Applied chat template for /api/generate (prompt length: %d chars)",
                len(prompt),
            )
            logger.debug("Templated prompt: %s", prompt[:500])
        except RuntimeError as exc:
            logger.warning(
                "Chat template failed for %s, falling back to raw prompt: %s",
                model_name,
                exc,
                exc_info=True,
            )
            if system:
                prompt = f"{system}\n\n{prompt}"
    elif apply_chat_template and lm.is_vlm:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            prompt = _apply_chat_template_vlm(lm.tokenizer, lm.model, messages)
            logger.info(
                "Applied VLM chat template for /api/generate (prompt length: %d chars)",
                len(prompt),
            )
            logger.debug("VLM templated prompt: %s", prompt[:500])
        except Exception as exc:
            logger.warning(
                "VLM chat template failed for %s, falling back to raw prompt: %s",
                model_name,
                exc,
                exc_info=True,
            )
            if system:
                prompt = f"{system}\n\n{prompt}"

    # Merge per-model defaults with request options.  options=None means
    # "use defaults"; options={} means "override all defaults" (empty).
    merged_options = (
        {**lm.default_options, **(options or {})}
        if lm.default_options and options is None
        else options
    )
    gen_kwargs = _build_generate_kwargs(merged_options, is_vlm=lm.is_vlm)
    mt = gen_kwargs.pop("max_tokens", max_tokens)

    if stream:
        return _stream_completion(lm, prompt, mt, gen_kwargs, stats, images)
    else:
        return await _full_completion(lm, prompt, mt, gen_kwargs, stats, images)


@dataclasses.dataclass
class _CacheSetupResult:
    """Result of prompt cache setup."""

    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    full_prompt_tokens: list[int] | None = None
    prompt: str | list[int] = ""
    cache_setup_done: bool = False


async def _setup_prompt_cache(
    lm: LoadedModel,
    prompt: str | list[int],
    gen_kwargs: dict,
    *,
    prompt_tokens: list[int] | None,
    cache_id: str,
) -> _CacheSetupResult:
    """Set up prompt cache for a streaming completion.

    Handles memory pressure eviction, cache lookup, prefix matching,
    and cache hit/miss logic. Mutates gen_kwargs in place (adds
    prompt_cache and optionally input_ids).
    """
    result = _CacheSetupResult(prompt=prompt)

    # Memory pressure check — invalidate cache to prevent Metal OOM
    memory_too_high = (
        prompt_tokens is not None
        and make_prompt_cache is not None
        and memory_utils.is_memory_pressure_high(settings.memory_limit_fraction)
    )
    if memory_too_high:
        logger.warning(
            "Memory pressure high, evicting prompt caches to free GPU memory"
        )
        await lm.prompt_cache_store.async_evict_all_to_disk()
        gc.collect()
        mx.clear_cache()
        _safe_sync()  # Bug #120: ensure freed buffers are reclaimed
        memory_too_high = memory_utils.is_memory_pressure_high(
            settings.memory_limit_fraction
        )

    if memory_too_high or prompt_tokens is None or make_prompt_cache is None:
        return result

    cached = await lm.prompt_cache_store.async_get(cache_id)
    logger.debug(
        "Cache lookup: cached=%s, new prompt=%d tokens",
        (
            f"{len(cached.tokens)} tokens (first 5: {cached.tokens[:5]})"
            if cached
            else "none"
        ),
        len(prompt_tokens),
    )

    prefix_len = (
        _find_common_prefix(prompt_tokens, cached.tokens) if cached is not None else 0
    )
    logger.debug(
        "Common prefix length: %d / %d prompt tokens",
        prefix_len,
        len(prompt_tokens),
    )

    # Set before mutation so finally guard can clean up on error
    result.full_prompt_tokens = prompt_tokens

    # Label for the fresh-cache log line.  Set to "trim-fallback" if
    # we discard a stale cache because trim_prompt_cache could not
    # align it (e.g. Qwen3-Next hybrid cache).  Otherwise the log
    # block below picks "miss" or "init" based on whether `cached`
    # was populated.
    fresh_cache_label: str | None = None

    # stream_generate requires at least 1 token, so we must back up
    # by one position on exact-match.  If that would mean suffix_start=0
    # (single-token prompt), the cache hit is useless — trimming the
    # entire cache to re-process the lone token is a cold start.  Treat
    # it as a miss and create a fresh cache instead.
    suffix_start = min(prefix_len, len(prompt_tokens) - 1) if prompt_tokens else 0

    if prefix_len > 0 and suffix_start > 0:
        # Bug #123: Remove cache from store before mutation so the
        # store's copy is not corrupted if the client disconnects
        # mid-stream.  The cache will be re-stored on successful
        # completion; on disconnect the finally block is a no-op.
        working_cache = cached.cache
        lm.prompt_cache_store.remove(cache_id)
        # Trim cache to suffix_start so it aligns with where we resume
        trim_amount = len(cached.tokens) - suffix_start
        trimmed = 0
        if trim_amount > 0 and lm.supports_cache_trim:
            trimmed = trim_prompt_cache(working_cache, trim_amount)

        if trim_amount > 0 and not lm.supports_cache_trim:
            # Hybrid sliding-window models (RotatingKVCache layers) cannot
            # be trimmed back; trim_prompt_cache would silently return 0.
            logger.debug(
                "Skipping trim on non-trimmable hybrid cache "
                "(would-be trim=%d); discarding and creating fresh",
                trim_amount,
            )
            del working_cache
            cached = None
            gc.collect()
            mx.clear_cache()
            _safe_sync()
            fresh_cache_label = "non-trimmable-hybrid"
        elif trim_amount > 0 and trimmed != trim_amount:
            # Defensive path: only reachable for models where
            # supports_cache_trim is True but trim_prompt_cache still
            # under-delivers (e.g. a custom cache type whose trim()
            # silently clamps, or a class the probe allowlist missed).
            # Known-hybrid caches (RotatingKVCache, ChunkedKVCache) are
            # flagged at load time and hit the skip-trim branch above.
            # A partial trim would leave the KV state misaligned with
            # the prompt, so fall through to fresh-cache creation.
            logger.warning(
                "Prompt cache trim incomplete (asked for %d, got %d); "
                "discarding stale cache and creating fresh",
                trim_amount,
                trimmed,
            )
            # Release the stale cache's GPU memory before allocating
            # the fresh one.  Both `working_cache` and `cached.cache`
            # hold references to the same KV tensors — both must be
            # broken or the tensors stay resident.  Then force GC +
            # Metal buffer reclamation: CPython's GC is non-deterministic
            # and Metal buffers won't be reclaimed in time for the
            # fresh cache allocation otherwise — for a 32B+ model this
            # could transiently double KV memory and trigger the
            # uncatchable Metal OOM the memory guards exist to prevent.
            del working_cache
            cached = None
            gc.collect()
            mx.clear_cache()
            _safe_sync()
            fresh_cache_label = "trim-fallback"
            # Fall through to fresh-cache creation below
        else:
            suffix_tokens = prompt_tokens[suffix_start:]

            result.cache_read_tokens = suffix_start
            result.cache_creation_tokens = len(suffix_tokens)
            logger.info(
                "Prompt cache hit: %d prefix tokens reused, %d new tokens to process (was %d total)",
                prefix_len,
                len(suffix_tokens),
                len(prompt_tokens),
            )
            gen_kwargs["prompt_cache"] = working_cache
            if lm.is_vlm:
                gen_kwargs["input_ids"] = mx.array([suffix_tokens])
            else:
                result.prompt = suffix_tokens

    if "prompt_cache" not in gen_kwargs:
        # Reached on either: (a) no usable prefix in cache miss, or
        # (b) trim-fallback above.  In the trim-fallback path the
        # cache was already removed before the trim attempt — this
        # remove is then a harmless no-op (PromptCacheStore.remove
        # is idempotent).  Kept for the cache-miss path.
        lm.prompt_cache_store.remove(cache_id)
        new_cache = _make_prompt_cache_for_lm(lm)
        gen_kwargs["prompt_cache"] = new_cache
        result.cache_creation_tokens = len(prompt_tokens)
        if fresh_cache_label is None:
            fresh_cache_label = "miss" if cached is not None else "init"
        logger.info(
            "Prompt cache %s: creating fresh cache for %d tokens",
            fresh_cache_label,
            len(prompt_tokens),
        )
        if lm.is_vlm:
            gen_kwargs["input_ids"] = mx.array([prompt_tokens])
        else:
            result.prompt = prompt_tokens

    result.cache_setup_done = True

    # Release the cached reference
    del cached

    return result


@dataclasses.dataclass
class _PreflightResult:
    """Result of KV cache pre-flight memory check."""

    prompt: str | list[int] = ""
    memory_limit: int = 0


async def _kv_cache_preflight_check(
    lm: LoadedModel,
    prompt: str | list[int],
    max_tokens: int,
    gen_kwargs: dict,
    *,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    full_prompt_tokens: list[int] | None,
    cache_id: str,
) -> _PreflightResult:
    """Estimate KV cache memory and reject if it would exceed the limit.

    Evicts prompt caches under pressure and restores prompt if needed.
    Mutates gen_kwargs in place (may remove prompt_cache and input_ids).
    Raises MemoryError if the KV cache would exceed the memory limit.
    """
    result = _PreflightResult(prompt=prompt)

    if cache_creation_tokens > 0:
        num_prefill_tokens = cache_creation_tokens
    elif isinstance(prompt, list):
        num_prefill_tokens = len(prompt)
    elif isinstance(prompt, str) and not lm.is_vlm:
        # Non-cached text path — tokenize to get a count.
        # VLMs excluded: text_tokenizer.encode() misses image patch
        # tokens, giving a systematic undercount.
        try:
            num_prefill_tokens = len(lm.text_tokenizer.encode(prompt))
        except Exception:
            num_prefill_tokens = 0
    else:
        num_prefill_tokens = 0

    total_physical = memory_utils.get_system_memory_bytes()
    result.memory_limit = (
        int(total_physical * settings.memory_limit_fraction)
        if total_physical > 0
        else 0
    )

    if total_physical <= 0 or num_prefill_tokens <= 0:
        return result

    try:
        kv_bytes = estimate_kv_cache_bytes(
            lm.model,
            num_prefill_tokens + max_tokens,
            kv_cache_quant=lm.kv_cache_quant,
        )
        current_metal = memory_utils.get_metal_memory()
        if current_metal + kv_bytes > result.memory_limit:
            # Drop cached KV tensors so eviction + gc can reclaim them
            working = gen_kwargs.pop("prompt_cache", None)
            had_cache = working is not None
            gen_kwargs.pop("input_ids", None)
            # Re-add cache temporarily so evict_all_to_disk() can persist it.
            # Use cache_read_tokens to match the trimmed KV state.
            if had_cache and full_prompt_tokens is not None:
                await lm.prompt_cache_store.async_set(
                    cache_id,
                    CachedPromptState(
                        tokens=list(full_prompt_tokens[:cache_read_tokens]),
                        cache=working,
                    ),
                )
            working = None
            await lm.prompt_cache_store.async_evict_all_to_disk()
            gc.collect()
            mx.clear_cache()
            # Re-estimate for the full generation window
            estimate_tokens = num_prefill_tokens
            if had_cache and cache_read_tokens > 0:
                estimate_tokens = cache_read_tokens + num_prefill_tokens
                if full_prompt_tokens is not None and not lm.is_vlm:
                    result.prompt = full_prompt_tokens
            estimate_total = estimate_tokens + max_tokens
            kv_bytes = estimate_kv_cache_bytes(
                lm.model,
                estimate_total,
                kv_cache_quant=lm.kv_cache_quant,
            )
            _safe_sync()
            current_metal = memory_utils.get_metal_memory()
            if current_metal + kv_bytes > result.memory_limit:
                available_gb = max(0.0, (result.memory_limit - current_metal) / 1024**3)
                raise MemoryError(
                    f"KV cache for {estimate_total} tokens estimated at "
                    f"{kv_bytes / 1024**3:.1f} GB, but only "
                    f"{available_gb:.1f} GB available "
                    f"— prompt too long, reduce context or use a smaller model"
                )
    except MemoryError:
        raise
    except Exception:
        logger.warning(
            "KV cache pre-flight check skipped — OOM protection inactive",
            exc_info=True,
        )

    return result


async def _store_prompt_cache_after_generation(
    lm: LoadedModel,
    gen_kwargs: dict,
    full_prompt_tokens: list[int] | None,
    generated_tokens: list[int],
    eval_count: int,
    cache_id: str,
) -> None:
    """Store prompt cache state after successful generation.

    Handles trimming to max_cache_tokens, eviction pressure cleanup,
    and cache invalidation on trim failure.

    Non-trimmable hybrid caches (e.g. Gemma 4, Qwen3-Next with
    RotatingKVCache layers) skip the trim path and are stored as-is
    even when they exceed max_cache_tokens.  The pre-generation setup
    path handles their eventual discard when alignment requires a trim;
    this storage just keeps them alive for strict-extension cache hits.
    """
    prompt_cache = gen_kwargs.get("prompt_cache")
    if prompt_cache is None or full_prompt_tokens is None:
        return

    stored_tokens = list(full_prompt_tokens) + generated_tokens
    actual_total = len(full_prompt_tokens) + eval_count
    max_cache_tokens = settings.prompt_cache_max_tokens

    # Non-trimmable hybrid caches (RotatingKVCache) can never be trimmed
    # back.  The pre-generation setup path handles cache alignment by
    # discarding and creating fresh; here we store as-is or skip entirely
    # when the metadata is known to be unreliable.
    if not lm.supports_cache_trim:
        if eval_count != len(generated_tokens):
            # None-ID tokens: the ring buffer has stale generation entries
            # at positions we can't trim out, and the metadata can't
            # faithfully represent them.  Remove any previous entry from
            # the store (the mutable cache object was shared) and skip
            # storage — a corrupt cache is worse than no cache.
            lm.prompt_cache_store.remove(cache_id)
            logger.debug(
                "Non-trimmable cache with misaligned eval_count "
                "(%d != %d); skipping storage",
                eval_count,
                len(generated_tokens),
            )
            return

        if max_cache_tokens is not None and actual_total > max_cache_tokens:
            logger.warning(
                "Storing non-trimmable cache at %d tokens "
                "(would-be trim=%d, exceeds limit of %d); "
                "max_cache_tokens is advisory for hybrid sliding-window models",
                len(stored_tokens),
                actual_total - max_cache_tokens,
                max_cache_tokens,
            )
        evicted = await lm.prompt_cache_store.async_set(
            cache_id,
            CachedPromptState(tokens=stored_tokens, cache=prompt_cache),
        )
        if evicted is not None:
            del evicted
            if memory_utils.is_memory_pressure_high(settings.memory_limit_fraction):
                gc.collect()
                mx.clear_cache()
        logger.debug(
            "Cache stored (non-trimmable): %d tokens (%d prompt + %d generated)",
            len(stored_tokens),
            len(full_prompt_tokens),
            len(generated_tokens),
        )
        return

    if max_cache_tokens is not None and actual_total > max_cache_tokens:
        trim_amount = actual_total - max_cache_tokens
        # cache_invalidated drives the post-trim flow.  Set when:
        # (a) trim returns the wrong amount (trimmable cache misreporting), or
        # (b) trim raises an unexpected exception.  In either case
        # the storage block is skipped and the cache reference is
        # released.  Using a flag avoids signalling a normal "this
        # cache type can't be trimmed" outcome via an exception.
        cache_invalidated = False
        try:
            trimmed = trim_prompt_cache(prompt_cache, trim_amount)
            if trimmed != trim_amount:
                # A trimmable cache that under-delivers on trim
                # (possible with unusual cache implementations).
                # Invalidate rather than carry a broken cache forward.
                logger.warning(
                    "Cache trim incomplete (asked for %d, got %d); invalidating cache",
                    trim_amount,
                    trimmed,
                )
                cache_invalidated = True
            elif eval_count != len(generated_tokens):
                # None-ID tokens present: can't map generated_tokens
                # to KV cache positions. Trim KV cache down to prompt
                # boundary so depth == len(stored_tokens).
                extra = max_cache_tokens - len(full_prompt_tokens)
                if extra > 0:
                    trimmed_extra = trim_prompt_cache(prompt_cache, extra)
                    if trimmed_extra != extra:
                        logger.warning(
                            "Cache trim incomplete (asked for %d, got %d); "
                            "invalidating cache",
                            extra,
                            trimmed_extra,
                        )
                        cache_invalidated = True
                if not cache_invalidated:
                    stored_tokens = list(full_prompt_tokens)[:max_cache_tokens]
            else:
                stored_tokens = stored_tokens[:max_cache_tokens]
        except Exception:
            cache_invalidated = True
            logger.warning(
                "Cache trim raised; invalidating cache",
                exc_info=True,
            )

        if cache_invalidated:
            lm.prompt_cache_store.remove(cache_id)
            prompt_cache = None
            gc.collect()
            mx.clear_cache()
            # No _safe_sync() needed here (unlike the pre-generation
            # trim-fallback path): no fresh cache allocation follows
            # immediately — the function is about to return.  The
            # next request's pre-generation path will sync before
            # its own allocation.
        else:
            evicted = await lm.prompt_cache_store.async_set(
                cache_id,
                CachedPromptState(tokens=stored_tokens, cache=prompt_cache),
            )
            if evicted is not None:
                del evicted
                if memory_utils.is_memory_pressure_high(settings.memory_limit_fraction):
                    gc.collect()
                    mx.clear_cache()
            logger.info(
                "Cache trimmed: %d → %d tokens (limit %d)",
                actual_total,
                len(stored_tokens),
                max_cache_tokens,
            )
    else:
        evicted = await lm.prompt_cache_store.async_set(
            cache_id,
            CachedPromptState(
                tokens=stored_tokens,
                cache=prompt_cache,
            ),
        )
        if evicted is not None:
            del evicted
            if memory_utils.is_memory_pressure_high(settings.memory_limit_fraction):
                gc.collect()
                mx.clear_cache()
        logger.debug(
            "Cache stored: %d tokens (%d prompt + %d generated)",
            len(stored_tokens),
            len(full_prompt_tokens),
            len(generated_tokens),
        )


async def _stream_completion(
    lm: LoadedModel,
    prompt: str | list[int],
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
    *,
    use_prompt_cache: bool = False,
    prompt_tokens: list[int] | None = None,
    cache_id: str = "",
) -> AsyncGenerator[dict, None]:
    # Use explicit acquire/release instead of `async with` to prevent
    # CancelledError from releasing the lock before cleanup completes.
    global _queue_depth
    lock = _get_inference_lock()
    await _await_deferred_cleanup()
    _queue_depth += 1
    if _queue_depth > 1:
        logger.info(
            "Streaming request queued for inference lock (queue depth: %d)",
            _queue_depth,
        )
    try:
        await _acquire_inference_lock(lm.inference_queue_timeout)
    except BaseException:
        _queue_depth -= 1
        raise
    _queue_depth -= 1
    # Re-check after acquiring — a deferred cleanup task may have been
    # created between the pre-check and acquire (TOCTOU window).
    try:
        await _await_deferred_cleanup()
    except BaseException:
        lock.release()
        raise
    # Sync default stream before starting — same purpose as _inference_locked entry.
    try:
        _lock_boundary_sync(lm.sync_mode)
    except BaseException:
        # BaseException (re-raise path): see _inference_locked entry for
        # the rationale. Summary: must release the lock on any exception,
        # including KeyboardInterrupt / SystemExit, so the shutdown
        # signal can propagate without leaving the lock held.
        lock.release()
        raise

    # Everything after lock acquisition must be in try/finally so the lock is
    # always released — even if the generator is closed at a yield point
    # (e.g. client disconnect during cache_info yield).
    stream = None
    generation_complete = False
    generated_tokens: list[int] = []
    full_prompt_tokens: list[int] | None = None
    # Save original string prompt before cache setup may replace it with token IDs.
    # prompt is always str at entry; cache setup may later reassign it to list[int].
    original_prompt = prompt
    try:
        # Cache setup — must happen after lock to prevent concurrent cache corruption
        if use_prompt_cache:
            cs = await _setup_prompt_cache(
                lm,
                prompt,
                gen_kwargs,
                prompt_tokens=prompt_tokens,
                cache_id=cache_id,
            )
        else:
            cs = _CacheSetupResult(prompt=prompt)
        prompt = cs.prompt
        cache_read_tokens = cs.cache_read_tokens
        cache_creation_tokens = cs.cache_creation_tokens
        full_prompt_tokens = cs.full_prompt_tokens
        cache_setup_done = cs.cache_setup_done

        # Pre-flight KV cache memory check
        pf = await _kv_cache_preflight_check(
            lm,
            prompt,
            max_tokens,
            gen_kwargs,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            full_prompt_tokens=full_prompt_tokens,
            cache_id=cache_id,
        )
        prompt = pf.prompt
        memory_limit = pf.memory_limit

        # Yield cache stats after the pre-flight check so routers can
        # use them.  This starts the HTTP response — no 503 after this.
        if cache_setup_done:
            yield {
                "cache_info": True,
                "cache_read_tokens": cache_read_tokens,
                "cache_creation_tokens": cache_creation_tokens,
            }

        # Broadcast to distributed workers before starting generation.
        # Workers need prompt_text (str) because mlx_lm.stream_generate
        # expects a string prompt, not token IDs. Always use original_prompt
        # (the original string before cache manipulation may have replaced it
        # with token IDs) to avoid tokenizer round-trip mismatches.
        # Strip prompt_cache and input_ids — these are local MLX objects
        # that cannot be serialized to JSON for the sideband protocol.
        if lm.is_distributed:
            tokens = (
                prompt_tokens
                if prompt_tokens is not None
                else tokenize_for_cache(lm.text_tokenizer, original_prompt)
            )
            broadcast_kwargs = {
                k: v
                for k, v in gen_kwargs.items()
                if k not in ("prompt_cache", "input_ids")
            }
            _maybe_broadcast_distributed(
                lm, tokens, original_prompt, max_tokens, broadcast_kwargs
            )

        if lm.is_speculative and not images:
            from olmlx.engine.speculative_stream import async_speculative_stream

            # Speculative decoding uses greedy argmax; sampling params are not supported.
            _sampling_keys = {
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "seed",
            }
            # Greedy-compatible defaults: temperature=0, top_p=1.0, top_k=0
            _greedy_defaults = {"temperature": 0, "top_p": 1.0, "top_k": 0}
            _dropped = {
                k
                for k in gen_kwargs
                if k in _sampling_keys
                and gen_kwargs[k] is not None
                and gen_kwargs[k] != _greedy_defaults.get(k)
            }
            if _dropped:
                logger.warning(
                    "Speculative decoding uses greedy argmax; ignoring sampling parameters: %s",
                    ", ".join(sorted(_dropped)),
                )
            stream = async_speculative_stream(
                lm.speculative_decoder,
                lm.text_tokenizer,
                prompt,
                max_tokens=max_tokens,
            )
        else:
            if lm.is_speculative:
                logger.debug("speculative decoding skipped: request includes images")
            stream = async_mlx_stream(
                lm.model,
                lm.tokenizer,
                prompt,
                max_tokens=max_tokens,
                is_vlm=lm.is_vlm,
                images=images,
                memory_limit=memory_limit,
                **gen_kwargs,
            )

        # Channel filter for gpt-oss models (decides which tokens to yield as text)
        channel_filter = (
            _GptOssChannelFilter() if lm.template_caps.has_channel_format else None
        )

        inf_timeout = (
            lm.inference_timeout
            if lm.inference_timeout is not None
            else settings.inference_timeout
        )
        timed_out = False

        with _inference_ref(lm), Timer() as total_timer:
            with Timer() as eval_timer:
                inf_start = time.monotonic()
                token = None
                async for token in stream:
                    # Always accumulate for prompt cache (raw stream, not filtered)
                    stats.eval_count = token.generation_tokens
                    stats.prompt_eval_count = token.prompt_tokens
                    if token.token is not None:
                        generated_tokens.append(token.token)
                    else:
                        logger.debug(
                            "Skipping token with None ID at generation step %d "
                            "(cache token sequence will be incomplete)",
                            token.generation_tokens,
                        )
                    # Yield text only if the filter allows it (or no filter).
                    # When channel filter is active, always include raw_text so
                    # downstream consumers (e.g. tool call parsers) can
                    # reconstruct the full unfiltered output.
                    if channel_filter is None:
                        yield {"text": token.text, "done": False}
                    elif channel_filter.should_yield(token.text):
                        yield {"text": token.text, "done": False}

                    if inf_timeout is not None:
                        elapsed = time.monotonic() - inf_start
                        if elapsed > inf_timeout:
                            logger.warning(
                                "Inference timeout after %.1fs (limit: %.1fs)",
                                elapsed,
                                inf_timeout,
                            )
                            timed_out = True
                            stream.cancel()
                            break

            # Fallback: yield analysis content if no final channel was produced
            if channel_filter is not None:
                for text in channel_filter.get_fallback_texts():
                    yield {"text": text, "done": False}
                # Capture raw text for tool call parsing (will be included in done chunk)
                raw_text = channel_filter.get_full_text()
            else:
                raw_text = ""

            prompt_tps, gen_tps = _derive_timing_stats(
                stats,
                getattr(token, "prompt_tps", 0) or 0,
                getattr(token, "generation_tps", 0) or 0,
                eval_timer.duration_ns,
            )

        stats.total_duration = total_timer.duration_ns
        if not timed_out:
            logger.info(
                "Generation complete: %d prompt tokens (%.1f tok/s), %d tokens generated (%.1f tok/s), %.2fs total",
                stats.prompt_eval_count,
                prompt_tps,
                stats.eval_count,
                gen_tps,
                total_timer.duration_ns / 1e9,
            )
            generation_complete = True

            # Store cache state after successful generation
            await _store_prompt_cache_after_generation(
                lm,
                gen_kwargs,
                full_prompt_tokens,
                generated_tokens,
                stats.eval_count,
                cache_id,
            )

        # raw_text contains the complete unfiltered output (e.g. gpt-oss channel tokens).
        # It is only present in the done chunk when gpt-oss channel format was used,
        # allowing consumers to parse tool calls from the full raw text.
        done_chunk: dict = {"text": "", "done": True, "stats": stats}
        if raw_text:
            done_chunk["raw_text"] = raw_text
        if timed_out:
            done_chunk["done_reason"] = "timeout"
        yield done_chunk
    finally:
        # Release GPU-backed references from gen_kwargs so they can be
        # garbage-collected.  prompt_cache is either stored in the cache
        # store (successful path) or should be freed; input_ids is a
        # temporary MLX array only needed during stream setup.
        gen_kwargs.pop("prompt_cache", None)
        gen_kwargs.pop("input_ids", None)
        # Invalidate cache on incomplete generation to avoid inconsistent state
        if not generation_complete and full_prompt_tokens is not None:
            logger.debug("Cache invalidated: generation did not complete")
            lm.prompt_cache_store.remove(cache_id)
        # We MUST wait for the Metal thread to finish before releasing
        # _inference_lock, otherwise the next inference will hit concurrent
        # Metal command buffer access.
        # stream may be None if generator was closed during cache setup.
        thread_alive = False
        if stream is not None:
            _drain_task = asyncio.ensure_future(stream.drain_and_join())
            try:
                await asyncio.shield(_drain_task)
            except (asyncio.CancelledError, Exception):
                # Shield was interrupted — cancel the inner drain task to
                # avoid a leaked coroutine that logs misleading warnings.
                _drain_task.cancel()
                # Ensure cancel_event is set even if _drain_task was cancelled
                # before drain_and_join() could set it (which would leave the
                # prefill callback returning True indefinitely).
                stream.cancel()
                # Fallback join — give the thread a chance to exit before
                # going to deferred cleanup.
                if stream._thread is not None and stream._thread.is_alive():
                    try:
                        await asyncio.to_thread(stream._thread.join, 10)
                    except (asyncio.CancelledError, Exception):
                        pass
            thread_alive = stream._thread is not None and stream._thread.is_alive()

        if thread_alive:
            # Thread is stuck (likely in long prefill).  Defer cleanup to
            # avoid calling _safe_sync() while the thread is still using
            # the GPU — that causes an uncatchable Metal assertion crash.
            logger.warning(
                "Inference thread still alive after cleanup attempts — "
                "deferring Metal sync and lock release until thread exits"
            )
            await _schedule_deferred_inference_cleanup(stream)
        else:
            # Normal path — thread exited, safe to sync and release.
            try:
                _lock_boundary_sync(lm.sync_mode)
            except ValueError:
                # Don't re-raise: a stream-body exception is already mid-
                # propagation through the outer finally, and masking it
                # with an unknown-mode ValueError would erase the cause.
                # Fall back to _safe_sync() so we still sync before
                # releasing the lock. Narrow to ValueError so interrupt
                # signals (KeyboardInterrupt / SystemExit) from
                # mx.synchronize propagate instead of being swallowed.
                logger.error(
                    "exit _lock_boundary_sync failed in _stream_completion",
                    exc_info=True,
                )
                _safe_sync()
            finally:
                lock.release()


async def _full_completion(
    lm: LoadedModel,
    prompt: str,
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
    has_tools: bool = False,
) -> dict:
    # inference_timeout is not enforced for non-streaming: the GPU thread
    # cannot be safely cancelled (releasing the lock while Metal is still
    # running causes concurrent command buffer access).  Streaming handles
    # this via CancellableStream.cancel() + drain_and_join().
    async with _inference_locked(lm.inference_queue_timeout, sync_mode=lm.sync_mode):
        with _inference_ref(lm):
            return await _full_completion_inner(
                lm, prompt, max_tokens, gen_kwargs, stats, images, has_tools=has_tools
            )


async def _full_completion_inner(
    lm: LoadedModel,
    prompt: str,
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
    has_tools: bool = False,
) -> dict:
    def _generate_sync():
        """Run generate + synchronize in the same thread so GPU work completes
        before the thread returns to the pool."""
        # Broadcast inside the thread so rank 0 and workers enter MLX
        # computation at the same time (avoids all_sum timeout).
        # Must happen before _apply_seed which pops seed from gen_kwargs.
        if lm.is_distributed:
            tokens = tokenize_for_cache(lm.text_tokenizer, prompt)
            _maybe_broadcast_distributed(lm, tokens, prompt, max_tokens, gen_kwargs)

        _apply_seed(gen_kwargs, consume=not lm.is_vlm)

        if lm.is_vlm and images:
            if lm.is_speculative:
                logger.debug("speculative decoding skipped: request includes images")
            import mlx_vlm

            # mlx_vlm.generate returns a plain str; prompt/generation token
            # counts are not exposed, so stats.prompt_eval_count /
            # stats.eval_count stay 0 on this path.
            result = mlx_vlm.generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                image=images,
                max_tokens=max_tokens,
                **gen_kwargs,
            )
            from mlx_vlm.generate import (
                generation_stream,
            )  # used by mx.synchronize below
        elif lm.is_speculative:
            import threading

            from olmlx.engine.speculative_stream import (
                speculative_stream_generate,
            )

            if isinstance(prompt, str):
                prompt_tokens = lm.text_tokenizer.encode(prompt)
            else:
                prompt_tokens = prompt

            cancel = threading.Event()
            eos_token_id = getattr(lm.text_tokenizer, "eos_token_id", None)
            result = None
            text_parts = []
            for response in speculative_stream_generate(
                lm.speculative_decoder,
                prompt_tokens,
                max_tokens=max_tokens,
                cancel_event=cancel,
                eos_token_id=eos_token_id,
                tokenizer=lm.text_tokenizer,
            ):
                text_parts.append(response.text)
                result = response
            if result is not None:
                result = (result, "".join(text_parts))
            # Speculative decoding does not use mlx_lm's generation_stream,
            # so sync the default stream only.
            mx.synchronize()
            return result
        elif lm.is_vlm:
            import mlx_vlm

            result = mlx_vlm.generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                image=images,
                max_tokens=max_tokens,
                **gen_kwargs,
            )
            from mlx_vlm.generate import (
                generation_stream,
            )  # used by mx.synchronize below
        else:
            import mlx_lm

            # Use stream_generate to capture token counts (generate() discards them).
            # Accumulate text segments since each yield is incremental.
            result = None
            text_parts = []
            for response in mlx_lm.stream_generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                **gen_kwargs,
            ):
                text_parts.append(response.text)
                result = response
            # Store full text on the result for downstream extraction
            if result is not None:
                result = (result, "".join(text_parts))
            from mlx_lm.generate import (
                generation_stream,
            )  # used by mx.synchronize below

        # Sync the generation_stream specifically — mlx_lm/mlx_vlm run GPU
        # work on this module-level stream, not the default stream.  Without
        # this, generate() may return before GPU work is actually done.
        mx.synchronize(generation_stream)
        return result

    with Timer() as total_timer:
        with Timer() as eval_timer:
            result = await asyncio.to_thread(_generate_sync)

    stats.total_duration = total_timer.duration_ns

    # Unpack (GenerationResult, full_text) tuple from stream_generate path
    full_text = None
    if isinstance(result, tuple):
        gen_result, full_text = result
        result = gen_result

    # Extract token counts from GenerationResult (stream_generate) or string
    if hasattr(result, "prompt_tokens"):
        stats.prompt_eval_count = result.prompt_tokens
    if hasattr(result, "generation_tokens"):
        stats.eval_count = result.generation_tokens

    prompt_tps, gen_tps = _derive_timing_stats(
        stats,
        getattr(result, "prompt_tps", 0) or 0,
        getattr(result, "generation_tps", 0) or 0,
        eval_timer.duration_ns,
    )
    total_secs = stats.total_duration / 1e9 if stats.total_duration else 0
    logger.info(
        "Generation complete: %d prompt tokens (%.1f tok/s), %d tokens generated (%.1f tok/s), %.2fs total",
        stats.prompt_eval_count,
        prompt_tps,
        stats.eval_count,
        gen_tps,
        total_secs,
    )

    # Extract text: prefer accumulated full_text, fall back to result
    if full_text is not None:
        text = full_text
    elif result is None:
        text = ""
    elif hasattr(result, "text"):
        text = result.text
    elif isinstance(result, str):
        text = result
    else:
        text = str(result)

    # Strip gpt-oss channel tokens for non-streaming path.
    # Keep raw_text so routers can parse tool calls from the full output.
    raw_text = None
    tool_uses = None
    thinking = ""
    if lm.template_caps.has_channel_format and "<|channel|>" in text:
        from olmlx.engine.tool_parser import _parse_gpt_oss_channels

        raw_text = text
        parsed = _parse_gpt_oss_channels(text, has_tools=has_tools)
        if parsed is not None:
            thinking, visible, tool_uses = parsed
            text = visible

    result_dict: dict = {"text": text, "done": True, "stats": stats}
    if raw_text is not None:
        result_dict["raw_text"] = raw_text
    if tool_uses:
        result_dict["tool_uses"] = tool_uses
    if thinking:
        result_dict["thinking"] = thinking
    return result_dict


@overload
async def generate_chat(
    manager: ModelManager,
    model_name: str,
    messages: list[dict],
    options: dict | None = ...,
    tools: list[dict] | None = ...,
    *,
    stream: Literal[True],
    keep_alive: str | None = ...,
    max_tokens: int = ...,
    cache_id: str = ...,
    enable_thinking: bool | None = ...,
) -> AsyncGenerator[dict, None]: ...


@overload
async def generate_chat(
    manager: ModelManager,
    model_name: str,
    messages: list[dict],
    options: dict | None = ...,
    tools: list[dict] | None = ...,
    *,
    stream: Literal[False],
    keep_alive: str | None = ...,
    max_tokens: int = ...,
    cache_id: str = ...,
    enable_thinking: bool | None = ...,
) -> dict: ...


@overload
async def generate_chat(
    manager: ModelManager,
    model_name: str,
    messages: list[dict],
    options: dict | None = ...,
    tools: list[dict] | None = ...,
    stream: bool = ...,
    keep_alive: str | None = ...,
    max_tokens: int = ...,
    cache_id: str = ...,
    enable_thinking: bool | None = ...,
) -> AsyncGenerator[dict, None] | dict: ...


async def generate_chat(
    manager: ModelManager,
    model_name: str,
    messages: list[dict],
    options: dict | None = None,
    tools: list[dict] | None = None,
    stream: bool = True,
    keep_alive: str | None = None,
    max_tokens: int = 512,
    cache_id: str = "",
    enable_thinking: bool | None = None,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a chat completion."""
    stats = TimingStats()

    with Timer() as load_timer:
        lm = await manager.ensure_loaded(model_name, keep_alive)
    stats.load_duration = load_timer.duration_ns

    images = _extract_images(messages)

    # Normalise OpenAI-format tool_calls ({function: {name, arguments: "json"}})
    # to the flat format chat templates expect ({name, arguments: {...}}).
    messages = _normalize_tool_calls_in_messages(messages)

    # When native tools are used, clients (e.g. opencode, Claude Code) may
    # include their own tool-format instructions in the system message that
    # conflict with the model's native tool call format.  With long prompts,
    # models like Gemma 4 follow the client's text instructions instead of
    # generating native tool call tokens.  Appending a short override to the
    # system message steers the model back to the native format.  We pass
    # the model's own chat template so the helper can suppress patterns the
    # template uses natively (e.g. Mistral's `[TOOL_CALLS]`).
    caps = lm.template_caps or TemplateCaps()
    if tools and caps.supports_tools:
        template_text = _get_chat_template_text(lm.tokenizer)
        messages = _add_native_tool_hint(messages, template_text)

    if lm.is_vlm:
        # VLM models must use the VLM generation path for tokenization.
        # Pass tools natively through the template when supported — this
        # produces model-native formatting (e.g. <|tool> tags for Gemma 4)
        # which is far more effective than injecting JSON into the system
        # message.  Fall back to system-message injection for models whose
        # template lacks tool support.
        # Resolve enable_thinking for templates that support it.
        vlm_thinking = enable_thinking if caps.supports_enable_thinking else None
        # Convert role="tool" messages to tool_responses format for models
        # that don't support OpenAI-style tool messages (e.g. Gemma 4).
        if caps.uses_tool_responses:
            messages = _convert_tool_messages_to_responses(messages)
        if tools and caps.supports_tools:
            prompt = _apply_chat_template_vlm(
                lm.tokenizer,
                lm.model,
                messages,
                images,
                tools=tools,
                enable_thinking=vlm_thinking,
            )
            logger.info("VLM chat prompt with %d tools (native template)", len(tools))
        elif tools:
            vlm_messages = _inject_tools_into_system(list(messages), tools)
            prompt = _apply_chat_template_vlm(
                lm.tokenizer, lm.model, vlm_messages, images
            )
            logger.info(
                "VLM chat prompt with %d tools (injected into system)", len(tools)
            )
        else:
            prompt = _apply_chat_template_vlm(lm.tokenizer, lm.model, messages, images)
        logger.debug("Prompt (first 2000 chars): %s", prompt[:2000])
        logger.debug("Prompt (last 2000 chars): %s", prompt[-2000:])
        if enable_thinking is not None and vlm_thinking is None:
            logger.debug(
                "enable_thinking=%s ignored for VLM model (template does not support it)",
                enable_thinking,
            )
    else:
        prompt = apply_chat_template_text(
            lm.text_tokenizer,
            messages,
            tools,
            caps=lm.template_caps,
            enable_thinking=enable_thinking,
        )
        if tools:
            logger.info("Chat prompt with %d tools", len(tools))
        logger.debug("Prompt (first 2000 chars): %s", prompt[:2000])
        logger.debug("Prompt (last 2000 chars): %s", prompt[-2000:])

    # Merge per-model defaults with request options.  options=None means
    # "use defaults"; options={} means "override all defaults" (empty).
    merged_options = (
        {**lm.default_options, **(options or {})}
        if lm.default_options and options is None
        else options
    )
    gen_kwargs = _build_generate_kwargs(merged_options, is_vlm=lm.is_vlm)
    mt = gen_kwargs.pop("max_tokens", max_tokens)

    # Prompt caching: streaming only, when enabled.
    # Disabled in distributed mode because rank 0 processes only suffix tokens
    # on cache hits while workers process the full prompt, causing all_sum
    # call count mismatch and deadlock.
    use_prompt_cache = (
        settings.prompt_cache
        and stream
        and make_prompt_cache is not None
        and not lm.is_distributed
    )
    prompt_tokens = None
    if use_prompt_cache:
        prompt_tokens = tokenize_for_cache(lm.text_tokenizer, prompt)
        # Memory-only peek for debug logging; the authoritative lookup happens
        # inside _stream_completion under the inference lock.
        cached_state = lm.prompt_cache_store.peek(cache_id)
        logger.debug(
            "Prompt cache enabled: %d prompt tokens, existing cache=%s",
            len(prompt_tokens),
            f"{len(cached_state.tokens)} tokens" if cached_state else "none",
        )
    else:
        logger.debug(
            "Prompt cache disabled: setting=%s stream=%s make_prompt_cache=%s",
            settings.prompt_cache,
            stream,
            make_prompt_cache is not None,
        )

    if stream:
        return _stream_completion(
            lm,
            prompt,
            mt,
            gen_kwargs,
            stats,
            images,
            use_prompt_cache=use_prompt_cache,
            prompt_tokens=prompt_tokens,
            cache_id=cache_id,
        )
    else:
        return await _full_completion(
            lm, prompt, mt, gen_kwargs, stats, images, has_tools=bool(tools)
        )


async def generate_embeddings(
    manager: ModelManager,
    model_name: str,
    texts: list[str],
    keep_alive: str | None = None,
) -> list[list[float]]:
    """Generate embeddings using the model's hidden states or embed_tokens layer."""
    lm = await manager.ensure_loaded(model_name, keep_alive)

    async with _inference_locked(lm.inference_queue_timeout, sync_mode=lm.sync_mode):
        with _inference_ref(lm):
            embeddings = []

            tokenizer = lm.text_tokenizer

            # Check if model has a static embedding layer we can use directly
            embed_layer = None
            model_inner = getattr(lm.model, "model", lm.model)
            if hasattr(model_inner, "embed_tokens"):
                embed_layer = model_inner.embed_tokens

            for text in texts:
                tokens = tokenizer.encode(text)
                input_ids = mx.array([tokens])

                if embed_layer is not None:
                    # Use static token embeddings — no forward pass needed
                    hidden = embed_layer(input_ids)
                else:
                    outputs = lm.model(input_ids)
                    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                        hidden = outputs.hidden_states[-1]
                    elif hasattr(outputs, "last_hidden_state"):
                        hidden = outputs.last_hidden_state
                    else:
                        hidden = outputs

                # Robust shape handling
                if hidden.ndim == 3:
                    # (batch, seq, dim) — mean-pool over sequence
                    embedding = mx.mean(hidden[0], axis=0)
                elif hidden.ndim == 2:
                    # (seq, dim) — mean-pool over sequence
                    embedding = mx.mean(hidden, axis=0)
                elif hidden.ndim == 1:
                    embedding = hidden
                else:
                    raise ValueError(
                        f"Unexpected embedding tensor shape: {hidden.shape}"
                    )

                embeddings.append(embedding.tolist())

            # Load-bearing when sync_mode="none": this function runs
            # synchronously under _inference_locked with no worker thread,
            # so the lock-boundary exit sync may be skipped. This sync is
            # then the only Metal barrier before the lock is released —
            # do not remove without providing an equivalent barrier.
            # Suppress+log rather than raise: the embeddings have already
            # been .tolist()'d so the caller should still get the result;
            # a Metal error here will surface on the next request.
            try:
                mx.synchronize()
            except Exception:
                # WARNING, not DEBUG: under sync_mode="none" this is the
                # only Metal barrier in the path; a silent failure here
                # typically surfaces as an uncatchable Metal crash on the
                # next inference. Operators need to see it now, not after.
                logger.warning(
                    "embeddings post-compute sync failed — next inference will crash",
                    exc_info=True,
                )
            return embeddings
