import asyncio
import collections.abc
import contextlib
import dataclasses
import gc
import itertools
import logging
import sys
import threading
import time
import weakref
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import mlx.core as mx
import numpy as np

from olmlx.engine.model_manager import (
    CachedPromptState,
    LoadedModel,
    ModelManager,
    parse_keep_alive,
)
from olmlx.config import SyncMode, settings
from olmlx.utils import memory as memory_utils

if TYPE_CHECKING:
    from olmlx.engine.prompt_cache.checkpoint import SegmentedPrompt

try:
    from mlx_lm.models.cache import (
        KVCache,
        RotatingKVCache,
        make_prompt_cache,
        trim_prompt_cache,
    )
    from mlx_lm.utils import common_prefix_len as _find_common_prefix
except ImportError:  # pragma: no cover
    make_prompt_cache = None  # type: ignore[assignment]
    trim_prompt_cache = None  # type: ignore[assignment]
    KVCache = None  # type: ignore[assignment]
    RotatingKVCache = None  # type: ignore[assignment]
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
from olmlx.engine.grammar import GrammarSpec

# Logits processors / decoding-output filters were extracted to a focused
# module (#454); re-exported here so existing call sites and the tests that
# import them from ``olmlx.engine.inference`` keep working.
from olmlx.engine.logits_processors import (
    _GptOssChannelFilter,
    _install_grammar_processor,
    _make_frequency_penalty_processor,
    _make_presence_penalty_processor,
    # Re-exported only for back-compat with tests that import them from here.
    _gpt_oss_filter as _gpt_oss_filter,
    _GPT_OSS_STRUCTURAL_TOKENS as _GPT_OSS_STRUCTURAL_TOKENS,
    _resolve_model_vocab_size as _resolve_model_vocab_size,
)

# Chat-template application + message normalization were extracted to a focused
# module (#454); re-exported here so existing call sites and the tests that
# import them from ``olmlx.engine.inference`` keep working.
from olmlx.engine.chat_templating import (
    _add_native_tool_hint,
    _apply_chat_template,
    _apply_chat_template_vlm,
    _convert_tool_messages_to_responses,
    _convert_tool_messages_to_user_text,
    _get_chat_template_text,
    _inject_tools_into_system,
    _normalize_tool_calls_in_messages,
    _resolve_thinking_active,
    apply_chat_template_text,
    tokenize_segmented_chat,
    # Re-exported only for back-compat with tests that import them from here.
    _message_boundary_token_ids as _message_boundary_token_ids,
    _NATIVE_TOOL_HINT as _NATIVE_TOOL_HINT,
)
from olmlx.context import surface_var
from olmlx.engine.stop_sequences import StopScanner, truncate_at_stop
from olmlx.engine.template_caps import TemplateCaps
from olmlx.utils import metrics as _metrics
from olmlx.utils import tracing as _tracing
from olmlx.utils.audio_input import cleanup_temp_audio, materialize_audio
from olmlx.utils.streaming import async_mlx_stream, materialize_lazy_cache_state
from olmlx.utils.timing import Timer, TimingStats


def _strategy_label(lm: "LoadedModel") -> str:
    """Tracing strategy attribute for a loaded model: ``"none"`` when not
    speculative, else the decoder's metrics strategy label (classic/pld/…)."""
    decoder = getattr(lm, "speculative_decoder", None)
    if decoder is None:
        return "none"
    return _metrics._STRATEGY_BY_CLASS.get(type(decoder).__name__, "unknown")


logger = logging.getLogger(__name__)


# -- Experimental: Distributed inference coordinator --
# Only set when OLMLX_DISTRIBUTED=true; see set_distributed_coordinator().
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


def _safe_sync():
    """Synchronize the calling thread's default Metal stream, suppressing
    and logging errors.

    Under mlx >= 0.31.2 streams are thread-local: worker-thread GPU work is
    fenced by the worker's own end-of-run sync (CancellableStream._run's
    finally / _generate_sync) plus the thread join in drain_and_join. A
    cross-thread sync of a generation stream would only sync THIS thread's
    idle instance of the proxy — so this helper fences exactly what the
    calling (event-loop) thread itself issued: cache deepcopy/eval,
    eviction cleanup. Callers rely on this being unconditional — notably
    cache-eviction and deferred-cleanup paths, which must sync regardless
    of ``settings.sync_mode``.
    """
    _sync_default_stream()


def _derive_timing_stats(
    stats: TimingStats,
    prompt_tps: float,
    gen_tps: float,
    eval_timer_ns: int,
    prefilled_count: int | None = None,
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

    Edge case worth knowing: when both counts are nonzero but neither rate
    is reported, the helper has no way to apportion wall-clock between
    prefill and decode. It splits ``eval_timer_ns`` 50/50 so neither phase
    ends up at 0 (which would cause divide-by-zero on clients computing
    tok/s). The sum invariant still holds; the split is just heuristic.
    mlx-lm reports both rates in practice, so this path is essentially
    unreached.

    ``prefilled_count`` is the number of tokens mlx-lm actually prefilled —
    which differs from ``stats.prompt_eval_count`` on a prompt-cache hit, where
    the caller reports the *full* prompt size but mlx-lm only re-prefilled the
    uncached suffix (and measured ``prompt_tps`` over that suffix). The prefill
    *duration* must be derived from the count the rate was measured over, else
    ``full_count / suffix_rate`` inflates ``raw_prompt_ns`` enough to crowd
    decode down to a sliver in the proportional clamp — producing an impossible
    decode tok/s. Defaults to ``stats.prompt_eval_count`` (no cache hit, or the
    non-streaming path where the reported count already matches the rate).
    """
    prefill_count = (
        prefilled_count
        if prefilled_count is not None and prefilled_count > 0
        else stats.prompt_eval_count
    )
    # Explicit zero of the fields this helper owns — several branches below
    # write only one of the two, so initialize both up front to make the
    # mutation contract independent of the caller's TimingStats state.
    stats.prompt_eval_duration = 0
    stats.eval_duration = 0

    # Defensive coercion: ``isinstance(x, (int, float))`` is the explicit
    # contract for what mlx-lm returns (Python floats). It rejects
    # MagicMock (whose ``__float__`` returns 1.0 and would otherwise
    # silently produce bogus durations in tests), and any future
    # non-Python-numeric type from upstream — at which point we'd want to
    # see real zeros and notice rather than silently coerce.
    if not isinstance(prompt_tps, (int, float)):
        prompt_tps = 0.0
    else:
        prompt_tps = float(prompt_tps)
    if not isinstance(gen_tps, (int, float)):
        gen_tps = 0.0
    else:
        gen_tps = float(gen_tps)

    raw_prompt_ns = (
        int(prefill_count / prompt_tps * 1e9)
        if prompt_tps > 0 and prefill_count > 0
        else 0
    )
    raw_decode_ns = (
        int(stats.eval_count / gen_tps * 1e9)
        if gen_tps > 0 and stats.eval_count > 0
        else 0
    )

    if raw_prompt_ns and raw_decode_ns:
        # Both rates known. If their sum exceeds wall-clock (rate noise) split
        # eval_timer_ns proportionally so each phase gets a non-zero share —
        # forcing one to 0 would create divide-by-zero on the client.
        raw_total = raw_prompt_ns + raw_decode_ns
        if raw_total > eval_timer_ns:
            # Use integer floor-division to keep the math exact — int(a*b/c)
            # would coerce through float and lose precision for large values.
            stats.prompt_eval_duration = eval_timer_ns * raw_prompt_ns // raw_total
            stats.eval_duration = eval_timer_ns - stats.prompt_eval_duration
        else:
            stats.prompt_eval_duration = raw_prompt_ns
            stats.eval_duration = raw_decode_ns
    elif raw_prompt_ns:
        # Only prompt rate known.
        if raw_prompt_ns >= eval_timer_ns and stats.eval_count > 0:
            # Rate noise: prefill alone would consume the full timer, leaving
            # nothing for decode despite eval_count > 0. Split 50/50 so
            # clients don't divide by zero on decode tok/s.
            stats.prompt_eval_duration = eval_timer_ns // 2
            stats.eval_duration = eval_timer_ns - stats.prompt_eval_duration
        else:
            stats.prompt_eval_duration = min(raw_prompt_ns, eval_timer_ns)
            if stats.eval_count == 0:
                stats.eval_duration = 0
            else:
                stats.eval_duration = max(0, eval_timer_ns - stats.prompt_eval_duration)
    elif raw_decode_ns:
        # Only decode rate known. Recover prefill by subtraction.
        if raw_decode_ns >= eval_timer_ns and stats.prompt_eval_count > 0:
            # Symmetric noise case.
            stats.eval_duration = eval_timer_ns // 2
            stats.prompt_eval_duration = eval_timer_ns - stats.eval_duration
        else:
            stats.eval_duration = min(raw_decode_ns, eval_timer_ns)
            if stats.prompt_eval_count > 0:
                stats.prompt_eval_duration = max(0, eval_timer_ns - stats.eval_duration)
    else:
        # Neither rate known.
        if stats.eval_count == 0:
            if stats.prompt_eval_count > 0:
                # Whole timer is prefill.
                stats.prompt_eval_duration = eval_timer_ns
            stats.eval_duration = 0
        elif stats.prompt_eval_count > 0:
            # Both counts > 0 with no rates: 50/50 to avoid divide-by-zero on
            # either side. mlx-lm reports rates in practice, so this path is
            # essentially unreached.
            stats.prompt_eval_duration = eval_timer_ns // 2
            stats.eval_duration = eval_timer_ns - stats.prompt_eval_duration
        else:
            # Only eval_count > 0: assign full timer to decode.
            stats.eval_duration = eval_timer_ns

    # Log-line rates: always derive from the final stored durations so the
    # "Generation complete" log agrees with the API response. This matters
    # when clamping kicks in — the raw mlx-lm rate would imply a different
    # duration than the one we actually report.
    if stats.eval_duration > 0 and stats.eval_count > 0:
        gen_tps = stats.eval_count / (stats.eval_duration / 1e9)
    if stats.prompt_eval_duration > 0 and prefill_count > 0:
        # Use prefill_count (what was actually prefilled), not the reported
        # prompt_eval_count, so the logged rate matches the work the duration
        # measures — otherwise a cache hit shows full_count / suffix_time, a
        # fictitious "rate" far above the hardware's real prefill throughput.
        prompt_tps = prefill_count / (stats.prompt_eval_duration / 1e9)

    return prompt_tps, gen_tps


def _lock_boundary_sync(mode: SyncMode | None = None) -> None:
    """Sync Metal GPU state at inference-lock entry/exit with configurable scope.

    ``mode`` resolves per call (not cached) so a per-model override wins over
    the global default. Values:

    - ``"full"`` / ``"minimal"``: sync the calling thread's default stream.
      The two modes are equivalent since mlx >= 0.31.2 made generation
      streams thread-local (worker-side sync + thread join fence worker GPU
      work; see ``_safe_sync``). ``"full"`` remains an accepted value for
      config compatibility.
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
    if effective in ("full", "minimal"):
        _sync_default_stream()
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


async def _schedule_deferred_inference_cleanup(
    stream, audio_temps: list[str] | None = None
) -> None:
    """Schedule deferred GPU cleanup when the inference thread is stuck.

    Polls the thread until it exits, then syncs Metal and releases the
    inference lock.  The lock remains held until the thread finishes to
    prevent concurrent Metal command buffer access.

    If the thread doesn't exit within _DEFERRED_CLEANUP_TIMEOUT seconds,
    releases the lock anyway (risk of Metal crash on next inference, but
    better than permanent deadlock).

    Uses _deferred_cleanup_lock to prevent TOCTOU races on _deferred_cleanup_tasks (Bug #119).

    ``audio_temps``: temp audio file paths to delete after the thread exits
    (or after timeout).  Passed from ``_stream_completion`` so that temp files
    are never deleted while the worker thread may still be reading them.
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
            # Clean up temp audio files now that the worker thread is done
            # (or we've given up waiting for it).  Must happen before lock
            # release so callers can rely on "lock released ⇒ temps gone".
            if audio_temps:
                from olmlx.utils.audio_input import cleanup_temp_audio

                cleanup_temp_audio(audio_temps)
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
        elif method == "shard":
            # ShardQuant: PCA-basis-projected packed indices + float32 norm.
            # Rank truncation makes the real footprint smaller than this, so
            # the turboquant-style estimate is a safe upper bound — but still
            # far below fp16. Without it, shard-quant models were estimated at
            # full fp16 KV size, 503-ing long prompts that would actually fit
            # (#634).
            shard_per_entry = head_dim // (8 // quant_bits) + 4
            _tq_ratio = shard_per_entry / fp16_per_entry

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


def build_context_input_tokens(
    tokenizer: Any, prompt_text: str, context: list[int] | None
) -> list[int]:
    """Build the full input token sequence for Ollama ``/api/generate`` context.

    Tokenizes *prompt_text* (via :func:`tokenize_for_cache`, so the ids match
    what generation would produce for the string) and, when *context* is a
    non-empty prior token sequence, prepends it — the legacy Ollama
    stateless-continuation mechanism (issue #656).  A leading BOS on the fresh
    prompt is dropped when *context* is supplied so the concatenated sequence
    doesn't repeat the sequence-initial BOS (whether the BOS came from
    ``add_special_tokens`` or from a chat template that emits it as literal
    text — both surface as ``bos_token_id`` at position 0).
    """
    prompt_tokens = tokenize_for_cache(tokenizer, prompt_text)
    if not context:
        return prompt_tokens
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and prompt_tokens and prompt_tokens[0] == bos_id:
        prompt_tokens = prompt_tokens[1:]
    return list(context) + prompt_tokens


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


async def _enter_inference_lock(
    timeout_override: float | None = None,
    *,
    sync_mode: SyncMode | None = None,
    queued_log: str = "Request queued for inference lock (queue depth: %d)",
) -> None:
    """Shared lock-entry protocol (#284 boundary discipline).

    Deferred-cleanup handshake, queue-depth accounting, FIFO acquire,
    post-acquire deferred-cleanup re-check (TOCTOU window: a deferred
    cleanup task may have been created between the pre-check and the
    acquire), then a Metal boundary sync so any pending GPU work from the
    previous lock holder completes before the caller starts new work.

    On any failure the lock is left released. Callers that complete the
    entry successfully own the lock and must pair with
    ``_exit_inference_lock``.
    """
    global _queue_depth
    lock = _get_inference_lock()
    await _await_deferred_cleanup()
    _queue_depth += 1
    if _queue_depth > 1:
        logger.info(queued_log, _queue_depth)
    try:
        await _acquire_inference_lock(timeout_override)
    finally:
        _queue_depth -= 1
    try:
        await _await_deferred_cleanup()
        _lock_boundary_sync(sync_mode)
    except BaseException:
        # BaseException (not ValueError): this handler re-raises, so
        # KeyboardInterrupt / SystemExit from mx.synchronize must be
        # caught here long enough to release the lock before they
        # propagate. Asymmetric with _exit_inference_lock, which narrows
        # to ValueError because it suppresses (the broader catch there
        # would silently swallow shutdown signals).
        lock.release()
        raise


def _exit_inference_lock(
    sync_mode: SyncMode | None = None, *, context: str = ""
) -> None:
    """Shared lock-exit protocol: Metal boundary sync, then release.

    Runs inside ``finally`` blocks, so an unknown sync mode must not
    raise (it would mask the in-flight exception): fall back to
    ``_safe_sync()`` so unknown modes fail-safe (sync anyway) instead of
    fail-open (no sync). Narrowed to ValueError so KeyboardInterrupt /
    SystemExit from mx.synchronize still propagate during shutdown.
    """
    try:
        _lock_boundary_sync(sync_mode)
    except ValueError:
        logger.error(
            "exit _lock_boundary_sync failed%s",
            f" in {context}" if context else "",
            exc_info=True,
        )
        _safe_sync()
    finally:
        _get_inference_lock().release()


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
    await _enter_inference_lock(timeout_override, sync_mode=sync_mode)
    try:
        yield
    finally:
        _exit_inference_lock(sync_mode, context="_inference_locked")


@contextlib.contextmanager
def _inference_ref(
    lm: LoadedModel, keep_alive: int | str | None = None, *, adopt: bool = False
):
    """Track active inference on a model to prevent expiry during use.

    When *adopt* is True the caller obtained ``lm`` via
    ``ensure_loaded(..., pin=True)``, which already incremented ``active_refs``
    under the lock. This context manager does NOT increment on entry and does
    NOT release on exit; the caller's outer try/finally owns the single pin
    release. This avoids races where an exception path here and the caller's
    cleanup could double-release. Expiry refresh still runs on exit so the
    model's deadline reflects the inference that just happened.

    With *adopt* False (the legacy path) this increments on entry and releases
    on exit as before; a small unprotected window then exists between
    ``ensure_loaded`` and here, but that path is only used where the caller
    does not hold the model across awaits.

    Bug #118: Python ``+=`` on int is not atomic. Concurrent async tasks can
    race on ``active_refs``. Use the model's ``_active_refs_lock`` to protect
    increments and decrements.

    Bug #338: *keep_alive* overrides the global default for expiry refresh.
    When a request passes a specific keep_alive, that value is honoured after
    inference instead of being silently replaced with the global default.
    """
    if not adopt:
        lm.acquire_ref()
    try:
        yield
    finally:
        if not adopt:
            lm.release_ref()
        # Refresh expiry so the model doesn't expire immediately after inference.
        # Honour per-request keep_alive when available (fix #338).
        # Falls back to settings.default_keep_alive when no per-request value
        # is given.  The else branch (ka is None → expires_at = None, i.e.
        # never expire) mirrors ensure_loaded's refresh path for already-loaded
        # models (both paths have the same pre-existing limitation: per-model
        # keep_alive from models.json is not checked during refresh — only at
        # initial load time).
        ka = parse_keep_alive(
            keep_alive if keep_alive is not None else settings.default_keep_alive
        )
        if ka is not None:
            lm.expires_at = time.time() + ka
        else:
            lm.expires_at = None


async def _release_pin_after_gen(
    gen: AsyncGenerator[dict, None], lm: LoadedModel
) -> AsyncGenerator[dict, None]:
    """Forward ``gen`` and release ``lm``'s pin on any generator exit.

    Used by the streaming entry points to release the pin acquired by
    ``ensure_loaded(pin=True)`` once the underlying generator finishes
    (normally, via exception, or via aclose on client disconnect). The
    inner generator's ``_inference_ref(adopt=True)`` does NOT release —
    only this wrapper does — so the pin is released exactly once.

    The finally explicitly closes ``gen`` so the wrapped generator's own
    finally blocks (e.g. the inference-lock release in
    ``_stream_completion``) fire when the client disconnects mid-stream.
    """
    try:
        async for chunk in gen:
            yield chunk
    finally:
        try:
            await gen.aclose()
        finally:
            lm.release_ref()


async def _trace_inference_gen(
    gen: AsyncGenerator[dict, None], lm: LoadedModel
) -> AsyncGenerator[dict, None]:
    """Hold a streaming-request ``inference`` span open for the generator's
    whole lifetime so the seam's ``prefill``/``decode`` spans (opened while the
    wrapped generator is being iterated) nest under it. Only wired in when
    tracing is enabled, so the off path keeps its exact generator topology.
    """
    with _tracing.span(
        "inference",
        model=lm.name,
        surface=surface_var.get(),
        strategy=_strategy_label(lm),
        **{"gen.stream": True},
    ):
        async for chunk in gen:
            yield chunk


def _merge_default_options(defaults: dict | None, request: dict | None) -> dict:
    """Merge per-model default options with per-request options.

    Request values win per-key; keys absent from the request fall back to
    model defaults.  ``request=None`` and ``request={}`` both mean "use
    defaults"; any non-None ``request`` is layered on top of ``defaults``.
    ``defaults=None`` is accepted symmetrically with ``request`` and treated
    as an empty dict so callers that haven't normalised the field can still
    use this helper without a guard.

    History: prior versions dropped *all* defaults whenever the request
    supplied *any* options — so a request that sent ``top_k`` without
    ``temperature`` silently lost the model's default temperature and ran
    greedy (no sampler built).  Surfaced via opencode + Qwen3-Coder-Next-4bit
    where opencode sent ``{top_k, top_p, min_p}`` and ``models.json``'s
    ``"temperature": 0.7`` was discarded.  The current always-merge form
    matches Ollama's per-model options semantics.
    """
    return {**(defaults or {}), **(request or {})}


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
        # Forward stop sequences for downstream (popped before passing to mlx-vlm)
        if "stop" in options and options["stop"]:
            raw = options["stop"]
            kwargs["stop"] = [raw] if isinstance(raw, str) else raw
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

        # Forward stop sequences for downstream (popped before passing to mlx-lm)
        if "stop" in options and options["stop"]:
            raw = options["stop"]
            kwargs["stop"] = [raw] if isinstance(raw, str) else raw

        # Build custom logits processors for frequency/presence penalty
        # and merge with any existing repeat_penalty processors.
        _existing = kwargs.pop("logits_processors", [])
        fp = options.get("frequency_penalty")
        if fp is not None and fp != 0:
            _existing.append(_make_frequency_penalty_processor(fp))
        pp = options.get("presence_penalty")
        if pp is not None and pp != 0:
            _existing.append(_make_presence_penalty_processor(pp))
        if _existing:
            kwargs["logits_processors"] = _existing

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


def _make_shard_prompt_cache(
    model: Any, bits: int, calibration_dir: Any, is_vlm: bool = False
) -> list:
    """Create a Shard-compressed prompt cache for the model (#377)."""
    from olmlx.engine.shardquant_cache import make_shard_cache

    cache_model = _get_model_for_cache(model, is_vlm)
    return make_shard_cache(
        cache_model, calibration_dir, bits=bits, fused=settings.shard_fused
    )


def _parse_kv_cache_quant(spec: str) -> tuple[str, int]:
    """Split an `OLMLX_KV_CACHE_QUANT` value like `"spectral:4"`
    into `(method, bits)`.  Format is validated at config load time."""
    method, bits_str = spec.split(":")
    return method, int(bits_str)


def _parse_kv_eviction(spec: str) -> tuple[int, int]:
    """Split an ``OLMLX_KV_EVICTION`` value like ``"4:512"`` into
    ``(sink, window)``. Format is validated at config load time (#505)."""
    sink_str, window_str = spec.split(":")
    return int(sink_str), int(window_str)


def _make_eviction_prompt_cache(model: Any, sink: int, window: int) -> list:
    """Build a StreamingLLM sink+window eviction cache for *model* (#505).

    ``RotatingKVCache(max_size=sink+window, keep=sink)`` keeps the first
    ``sink`` (attention-sink) tokens plus a rotating recent ``window`` and
    drops the middle — bounding KV count for long contexts. mlx-lm's
    ``create_attention_mask`` delegates to ``RotatingKVCache.make_mask``, so a
    full-attention model attends correctly to the retained keys with no custom
    mask/rope work.

    Eviction is applied **only to pure full-attention models** — those whose
    default cache is all plain ``KVCache``. Hybrid/SWA/GDN models (any
    ``RotatingKVCache`` already present, or ``ArraysCache`` recurrent state)
    are left untouched: a single per-forward mask is built from ``cache[0]``,
    so a mixed list would mis-mask, and recurrent layers can't be windowed.
    """
    default = make_prompt_cache(model)
    if RotatingKVCache is None or KVCache is None:
        return default
    if not default or not all(type(layer) is KVCache for layer in default):
        logger.warning(
            "kv_eviction requested but model is not pure full-attention "
            "(cache layers: %s); eviction skipped for this model.",
            sorted({type(layer).__name__ for layer in default}),
        )
        return default
    max_size = sink + window
    return [RotatingKVCache(max_size=max_size, keep=sink) for _ in default]


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
        if method == "shard":
            return _make_shard_prompt_cache(
                lm.model, bits, lm.shard_calibration_dir, is_vlm=lm.is_vlm
            )
        return _make_turboquant_prompt_cache(lm.model, bits, is_vlm=lm.is_vlm)
    cache_model = _get_model_for_cache(lm.model, lm.is_vlm)
    if lm.kv_eviction is not None:
        sink, window = _parse_kv_eviction(lm.kv_eviction)
        return _make_eviction_prompt_cache(cache_model, sink, window)
    return make_prompt_cache(cache_model)


def _extract_images(messages: list[dict]) -> list[str] | None:
    """Extract image URLs/paths from message content."""
    images = []
    for msg in messages:
        if msg.get("images"):
            images.extend(msg["images"])
    return images if images else None


def _extract_audio(messages: list[dict]) -> list[str] | None:
    """Extract audio refs (URLs/paths/data-URIs) from message content (#426)."""
    audio: list[str] = []
    for msg in messages:
        if msg.get("audio"):
            audio.extend(msg["audio"])
    return audio if audio else None


def _audio_capable(lm: Any) -> bool:
    """True when the loaded model can accept audio input.

    Audio models are VLMs whose mlx-vlm processor wires a ``feature_extractor``
    (Gemma 4, gemma3n, qwen3_omni_moe, phi4mm, minicpmo).  Read from the
    already-loaded processor — no new ModelManager kind needed.
    """
    if not getattr(lm, "is_vlm", False):
        return False
    return getattr(lm.tokenizer, "feature_extractor", None) is not None


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
    keep_alive: int | str | None = None,
    max_tokens: int = 512,
    images: list[str] | None = None,
    apply_chat_template: bool = False,
    system: str | None = None,
    enable_thinking: bool | None = None,
    grammar_spec: GrammarSpec | None = None,
    context: list[int] | None = None,
    return_context: bool = False,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a text completion, streaming or not.

    When *apply_chat_template* is True the raw prompt is wrapped in chat
    messages and run through the model's chat template before generation.
    If *system* is provided, it becomes a ``{"role": "system"}`` message.
    This is needed for chat-only models (e.g. Nemotron-H) that require the
    template framing to produce meaningful output.

    *context* / *return_context* implement Ollama ``/api/generate``'s legacy
    stateless multi-turn continuation (issue #656).  When *return_context* is
    True the (non-VLM) result carries a ``context`` key — the full token
    sequence that was prefilled followed by every generated token id — so a
    client can pass it back as *context* next turn.  A supplied *context* is
    prepended as a raw token prefix to the tokenized prompt (bypassing no
    template — the new turn is still templated/tokenized as usual, then
    appended).  Both are ignored for VLMs, whose token ids can't represent
    image patches.
    """
    stats = TimingStats()

    with Timer() as load_timer:
        lm = await manager.ensure_loaded(model_name, keep_alive, pin=True)
    stats.load_duration = load_timer.duration_ns

    # The pin acquired by ``ensure_loaded(pin=True)`` is released exactly
    # once: by the stream wrapper after the generator exits, by the
    # non-stream finally below, or by the outer except below if anything
    # raises before delegation. The outer try must scope from RIGHT AFTER
    # ensure_loaded (PR #394 aider review) so an exception in the chat
    # template / kwargs setup doesn't leak the pin.
    pin_released_or_transferred = False
    try:
        # /api/generate defaults thinking OFF when unspecified (None), unlike the
        # chat route's "think unless tools".  Coerce once so the template
        # instruction and the downstream thinking_expected signal stay consistent
        # (otherwise the splitter would arm the orphan-</think> buffer for thinking
        # the model was told not to produce).
        effective_thinking = enable_thinking if enable_thinking is not None else False

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
                    enable_thinking=effective_thinking,
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
                # Forward the *coerced* effective_thinking (not raw enable_thinking)
                # so VLM /api/generate is off-by-default like the text path, and so
                # the explicit instruction matches the thinking_expected signal
                # computed below.  We deliberately don't defer to the VLM template's
                # own default: we can't introspect it, so passing an explicit bool
                # is the only way to keep the thinking-splitter consistent.  This
                # differs intentionally from generate_chat's no-tools VLM path,
                # which passes None and lets _apply_chat_template_vlm's guard skip
                # the kwarg (preserving the template default).  Consequence: every
                # /api/generate VLM request passes enable_thinking (incl. False) to
                # the template even for non-thinking VLMs — harmless because HF
                # templates ignore unknown kwargs.
                prompt = _apply_chat_template_vlm(
                    lm.tokenizer, lm.model, messages, enable_thinking=effective_thinking
                )
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

        merged_options = _merge_default_options(lm.default_options, options)
        gen_kwargs = _build_generate_kwargs(merged_options, is_vlm=lm.is_vlm)
        mt = gen_kwargs.pop("max_tokens", max_tokens)

        grammar_active = _install_grammar_processor(lm, gen_kwargs, grammar_spec)

        # Tell routers whether to wait for a (possibly orphaned) `</think>` when
        # splitting thinking from the response (issue #307) — shares the rules
        # with `generate_chat`.  Completions never carry tools.  In raw mode no
        # chat template is applied, so no thinking instruction reached the model:
        # leave thinking_expected False so the router doesn't arm the orphan-close
        # heuristic on un-templated output (a literal `</think>` in code/prose).
        caps = lm.template_caps or TemplateCaps()
        thinking_expected = (
            _resolve_thinking_active(caps, None, effective_thinking)
            if apply_chat_template
            else False
        )

        # Ollama /api/generate context continuation (#656): tokenize the prompt,
        # prepend any prior `context` tokens, and feed generation the exact token
        # ids so the returned context is a faithful continuation prefix. Text
        # models only — VLM token ids can't represent image patches, and the
        # cache/token paths only replace a str prompt.
        context_input_tokens: list[int] | None = None
        gen_prompt: str | list[int] = prompt
        if return_context and not lm.is_vlm and isinstance(prompt, str):
            context_input_tokens = build_context_input_tokens(
                lm.text_tokenizer, prompt, context
            )
            gen_prompt = context_input_tokens
        collect_generated_tokens = context_input_tokens is not None

        if stream:
            gen = _stream_completion(
                lm,
                gen_prompt,
                mt,
                gen_kwargs,
                stats,
                images,
                keep_alive=keep_alive,
                grammar_active=grammar_active,
                adopt_pin=True,
                thinking_expected=thinking_expected,
                collect_generated_tokens=collect_generated_tokens,
            )
            # Ownership of the pin transfers to the wrapper; its finally
            # releases on any generator exit. Mark the flag for the outer
            # except so it doesn't double-release.
            pin_released_or_transferred = True
            streamed = _release_pin_after_gen(gen, lm)
            if _tracing.enabled():
                streamed = _trace_inference_gen(streamed, lm)
            if context_input_tokens is not None:
                streamed = _augment_stream_with_context(streamed, context_input_tokens)
            return _prepend_meta(
                streamed,
                {"thinking_expected": thinking_expected},
            )
        else:
            try:
                with _tracing.span(
                    "inference",
                    model=lm.name,
                    surface=surface_var.get(),
                    strategy=_strategy_label(lm),
                    **{"gen.stream": False},
                ):
                    result = await _full_completion(
                        lm,
                        gen_prompt,
                        mt,
                        gen_kwargs,
                        stats,
                        images,
                        keep_alive=keep_alive,
                        grammar_active=grammar_active,
                        adopt_pin=True,
                        thinking_expected=thinking_expected,
                        collect_generated_tokens=collect_generated_tokens,
                    )
                result["thinking_expected"] = thinking_expected
                if context_input_tokens is not None:
                    # Full continuation context: everything prefilled followed
                    # by every generated token id (#656).
                    generated = result.pop("generated_tokens", [])
                    result["context"] = context_input_tokens + generated
                return result
            finally:
                # Release exactly once on every exit path — success or
                # exception. Set the flag so the outer except doesn't try
                # to release again.
                if not pin_released_or_transferred:
                    lm.release_ref()
                    pin_released_or_transferred = True
    except BaseException:
        if not pin_released_or_transferred:
            lm.release_ref()
        raise


@dataclasses.dataclass
class _CacheSetupResult:
    """Result of prompt cache setup."""

    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    full_prompt_tokens: list[int] | None = None
    prompt: str | list[int] = ""
    cache_setup_done: bool = False
    # Checkpoint path only: closure that runs the segmented-prefill drive.
    # Executed on the generation worker thread (same thread-local
    # generation_stream as decode, #284/#499), never on the event loop.
    deferred_prefill: Callable[[threading.Event | None], None] | None = None


# Sub-chunk size for the segmented-prefill drive's per-chunk ``model()`` calls
# on non-pure-rotating (GatedDeltaNet/ArraysCache) caches. Matches mlx-lm's
# ``generate_step`` ``prefill_step_size`` default so the checkpoint path bounds
# activation memory the same way mlx-lm's native prefill does (avoids the
# 70k-token single-forward OOM).
_PREFILL_CHUNK = 2048


def _drive_segmented_prefill(
    *,
    model: Any,
    segmented: "SegmentedPrompt",
    cache: list[Any],
    insert_checkpoint: Callable[[CachedPromptState], None],
    already_covered_tokens: int = 0,
    cancel_event: threading.Event | None = None,
) -> list[int]:
    """Run prefill in at most two ``model(...)`` calls and snapshot the
    cache at the deepest interior message boundary; return the suffix to
    hand off to stream_generate.

    ``already_covered_tokens`` is the depth at which ``cache`` is already
    populated (from a checkpoint hit). Segments wholly inside that depth
    are skipped automatically — the first chunk's start is
    ``already_covered_tokens``.

    ``insert_checkpoint`` is called with each boundary snapshot; when the
    drive runs on the generation worker thread the caller passes a
    loop-marshalling wrapper because ``PromptCacheStore`` is loop-affine
    (#463).

    The returned suffix is ``[full_flat_tokens[-1]]`` — one token —
    because ``mlx_lm.stream_generate`` requires at least one prompt
    token to seed the decode step, and feeding the same token back into
    the prefilled cache is a no-op for the cache but produces the
    correct first-token logits.

    **Why two chunks, not one-per-segment.** mlx-lm's ``gated_delta_kernel``
    (GatedDeltaNet linear attention in Qwen3.5 / Qwen3-Next text towers)
    is not exactly chunking-invariant: feeding ``[A, B]`` and then ``[C]``
    yields a slightly different recurrent state than feeding ``[A, B, C]``
    in one call.  On dense full-precision targets the drift is invisible,
    but on MoE-quantized targets (e.g. Qwen3.6-35B-A3B-6bit) it crosses
    expert-routing thresholds and the model derails into verbatim
    paragraph repetition by the second turn.  The pre-fix per-segment
    drive made this scale linearly with conversation depth — every
    extra turn added another chunk boundary.  This implementation keeps
    drift bounded by driving the uncovered tail as a single
    ``[already_covered_tokens .. deepest_interior_boundary]`` chunk
    (snapshot point for future requests) followed by a single
    ``[deepest_interior_boundary .. len(flat)-1]`` chunk (the final
    message minus the reserved trailing token).  If there is no usable
    interior boundary the whole tail goes in one chunk and no snapshot
    is taken.  Trade-off: only the deepest interior boundary is
    snapshotted per request, not every boundary — for the
    sequentially-extending chat workload this is the only snapshot
    future requests actually use (earlier ones in the same request
    would be shadowed by the deeper hit anyway), and earlier requests'
    boundary snapshots remain in the store and still serve shallower
    branched lookups.

    The last boundary (``boundary == len(flat)``) is never a snapshot
    point: the final segment's trailing token is reserved for
    stream_generate, so KV depth is ``len(flat) - 1`` while the
    boundary claims ``len(flat)`` — that mismatch would silently
    misalign every future strict-extension lookup.
    """
    from olmlx.engine.prompt_cache.checkpoint import (
        flatten_cache_state,
        snapshot_cache_for_persistence,
    )

    flat = segmented.flatten()
    if not flat:
        return []
    boundaries = segmented.boundary_offsets()

    # Find the deepest interior boundary strictly greater than what's
    # already covered and strictly less than len(flat).  This is both the
    # snapshot point and the split between the two prefill chunks.
    deepest_boundary: int | None = None
    deepest_role: Any = None
    for boundary, seg in zip(boundaries, segmented.segments, strict=True):
        if already_covered_tokens < boundary < len(flat):
            deepest_boundary = boundary
            deepest_role = seg.role

    final_prefill_end = len(flat) - 1  # reserve trailing token for stream_generate

    # Prefill MUST run on the same stream ``stream_generate`` decodes on
    # (mlx-lm's ``generation_stream``).  Prefilling on ``mx.default_stream``
    # and then decoding on ``generation_stream`` leaves the GatedDeltaNet
    # recurrent state as a cross-stream lazy graph whose materialization
    # corrupts MoE expert routing on Qwen3-Next-family targets once the
    # prompt is large enough (~16k+ tokens) — the model emits coherent but
    # off-prompt pretraining-data dumps instead of answering (Qwen3-Coder-Next,
    # Qwen3.6-35B-A3B; same Metal-stream hazard family as #284).  Single-message
    # requests never hit this because they skip the drive entirely and let
    # ``stream_generate`` prefill on ``generation_stream``.  ``mx.clear_cache()``
    # per chunk mirrors mlx-lm's prefill loop.
    # Resolve mlx-lm's generation stream AT CALL TIME: under mlx >= 0.31.2 it
    # is a ThreadLocalStream proxy that resolves per-thread. This function
    # runs on the generation worker thread (deferred from cache setup), so
    # resolving here yields the same underlying stream ``stream_generate``
    # decodes on — the #284 same-stream contract, now expressed as
    # same-thread (#499).
    try:
        from mlx_lm.generate import generation_stream as prefill_stream
    except ImportError:  # mlx-lm absent (pure unit-test environments)
        prefill_stream = mx.default_stream(mx.default_device())
    pure_rotating = _is_pure_rotating_cache(cache)

    def _run(start: int, end: int) -> None:
        if end <= start:
            return
        with mx.stream(prefill_stream):
            if pure_rotating:
                # Sliding-window models (gpt-oss, Step-3.5, Gemma 3): feed the
                # span in ONE call, preserving the validated single-call
                # behavior their windowed attention depends on.
                model(mx.array(flat[start:end], dtype=mx.int32)[None, :], cache=cache)
                mx.eval(flatten_cache_state(cache))
                mx.clear_cache()
            else:
                # GatedDeltaNet/ArraysCache family: sub-chunk at _PREFILL_CHUNK
                # to bound activation memory — a ~70k-token chunk-1 prefill fed
                # as a single forward OOM'd Metal at 123 GB (Qwen3.6-35B-A3B-6bit).
                # Mirrors mlx-lm's prefill loop; eval + clear per sub-chunk caps
                # peak memory at roughly one sub-chunk's worth of activations.
                pos = start
                while pos < end:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    stop = min(pos + _PREFILL_CHUNK, end)
                    model(
                        mx.array(flat[pos:stop], dtype=mx.int32)[None, :], cache=cache
                    )
                    mx.eval(flatten_cache_state(cache))
                    mx.clear_cache()
                    pos = stop

    if pure_rotating:
        # Sliding-window models (gpt-oss, Step-3.5, Gemma 3) must be prefilled
        # in ONE call — splitting at an interior boundary corrupts their
        # windowed attention (coherent-but-unrelated output, no tool calls).
        # The two-chunk split only exists to bound GatedDeltaNet drift, which
        # these don't have.  Snapshot at the end (KV depth == len(flat)-1, so
        # tokens stored are flat[:final_prefill_end] — consistent, no trailing-
        # token mismatch) so a strict-extension follow-up can still warm-start.
        logger.debug(
            "Pure-rotating cache: single-chunk prefill [%d..%d] (no interior split)",
            already_covered_tokens,
            final_prefill_end,
        )
        _run(already_covered_tokens, final_prefill_end)
        if final_prefill_end > already_covered_tokens and segmented.segments:
            snap = snapshot_cache_for_persistence(cache, eager_eval=False)
            insert_checkpoint(
                CachedPromptState(
                    tokens=flat[:final_prefill_end],
                    cache=snap,
                    # Snapshot spans the whole prompt (not an interior
                    # boundary), so tier it by the leading role — matching how
                    # tokenize_segmented_chat tiers injected preamble segments.
                    cache_type=segmented.segments[0].role,
                    is_checkpoint=True,
                )
            )
    elif deepest_boundary is None:
        # No usable interior boundary — single chunk, no snapshot.
        _run(already_covered_tokens, final_prefill_end)
    else:
        # Chunk 1: uncovered prefix up to the deepest interior boundary.
        _run(already_covered_tokens, deepest_boundary)
        # Snapshot at the boundary.  eager_eval=False because ``_run``'s
        # inner ``mx.eval(flatten_cache_state(cache))`` just materialised
        # the state on the prefill stream; deepcopy alone is sufficient.
        snap = snapshot_cache_for_persistence(cache, eager_eval=False)
        insert_checkpoint(
            CachedPromptState(
                tokens=flat[:deepest_boundary],
                cache=snap,
                cache_type=deepest_role,
                is_checkpoint=True,
            )
        )
        # Chunk 2: boundary to the reserved trailing token.
        if cancel_event is not None and cancel_event.is_set():
            return [flat[-1]]
        _run(deepest_boundary, final_prefill_end)

    return [flat[-1]]


def _cache_list_contains_lazy_state(cache: list[Any]) -> bool:
    """True iff the cache contains layer types that hold thread-bound lazy
    state needing eager mx.eval before cross-thread reuse.

    Flags two families:

    - ``ArraysCache`` (hybrid SSM models like Qwen3.5): ``gated_delta_kernel``
      outputs carry a lazy graph bound to the building thread's Metal stream.
    - Quantized KV caches (TurboQuant / Spectral / Shard): their ``.state``
      property rebuilds a fresh ``[..., :offset, :]`` slice on every access, and
      ``trim`` can shrink the packed buffers via a lazy slice — both bound to the
      building thread's stream. Reusing/trimming such a cache on a *different*
      worker-pool thread crashes with "There is no Stream(gpu, N) in current
      thread". (These expose a ``_pin_state_to_offset`` marker method; matched by
      class name rather than ``hasattr`` so a MagicMock — which answers True to
      any ``hasattr`` — doesn't spuriously match in tests.)

    Pure KVCache/QuantizedKVCache/RotatingKVCache layouts return False because
    their ``.state`` arrays are produced by stock matmul/concat ops.
    """
    LAZY_STATE_CLASSES = {
        "ArraysCache",
        "TurboQuantKVCache",
        "SpectralQuantKVCache",
        "ShardKVCache",
    }
    return any(type(layer).__name__ in LAZY_STATE_CLASSES for layer in cache)


def _is_pure_rotating_cache(cache: list[Any]) -> bool:
    """True iff the cache is a sliding-window layout (has ``RotatingKVCache``)
    with no ``ArraysCache`` (GatedDeltaNet/SSM) layers.

    These models — gpt-oss, Step-3.5, Gemma 3 — must be prefilled in a SINGLE
    ``model(...)`` call.  The two-chunk checkpoint drive (which exists only to
    bound GatedDeltaNet recurrent-state drift on ``ArraysCache`` models)
    corrupts sliding-window attention here: splitting the prefill at an
    interior message boundary makes the model generate coherent-but-unrelated
    output and skip tool calls.  Mixed Rotating+Arrays layouts (Qwen3-Next)
    return False — they keep the two-chunk drive for the SSM part.
    """
    names = {type(layer).__name__ for layer in cache}
    return "RotatingKVCache" in names and "ArraysCache" not in names


async def _setup_via_checkpoint_path(
    lm: LoadedModel,
    prompt: str | list[int],
    gen_kwargs: dict,
    *,
    messages: list[dict],
    tokenizer: Any,
    prompt_tokens: list[int],
    template_kwargs: dict | None = None,
) -> _CacheSetupResult:
    """Checkpoint-mechanism path for non-trimmable cache layouts.

    Tokenizes per-message, looks up a strict-prefix match in the prompt
    cache store, drives prefill for the uncovered segments, and stores a
    checkpoint at each boundary. Returns a _CacheSetupResult with prompt
    reduced to the single token that mlx-lm's stream_generate needs to
    seed its first decode step.
    """
    from olmlx.engine.prompt_cache.checkpoint import snapshot_cache_for_persistence

    # Pass the caller's authoritative tokenization (prompt_tokens) directly
    # so segmentation never re-tokenizes — this avoids a BOS-handling
    # mismatch between apply_chat_template(tokenize=True) and
    # tokenize_for_cache that would silently disable the checkpoint path
    # on tokenizers like Llama 3's. template_kwargs is still threaded
    # through for the non-production no-prompt_tokens path used by tests.
    segmented = tokenize_segmented_chat(
        tokenizer,
        messages,
        full_tokens=prompt_tokens,
        **(template_kwargs or {}),
    )
    # Look up the longest stored prefix that is a STRICT prefix of our
    # tokens BEFORE the single-segment guard — a single-message request
    # that happens to be a strict extension of an existing checkpoint
    # should still warm-start, even though it can't itself contribute a
    # new mid-prompt checkpoint.
    hit = lm.prompt_cache_store.fetch_nearest(prompt_tokens)

    # No usable message boundaries AND no warm-start to inherit: a
    # single-segment prompt has only one boundary at len(prompt_tokens),
    # which the drive deliberately skips (the KV depth there is
    # len(prompt_tokens)-1 because the trailing token is reserved for
    # stream_generate). With nothing to gain on either side, skip the
    # checkpoint path entirely — cheaper than driving a no-op snapshot.
    if hit is None and len(segmented.segments) <= 1:
        logger.debug(
            "Checkpoint path: no usable message boundaries detected "
            "and no warm-start hit (segments=%d); falling back to fresh cache",
            len(segmented.segments),
        )
        new_cache = _make_prompt_cache_for_lm(lm)
        gen_kwargs["prompt_cache"] = new_cache
        # Return prompt_tokens (list[int]) to stay consistent with the
        # cold-start and warm-start branches below — handing the
        # original string back would let downstream code re-tokenize
        # via a different BOS path than prompt_tokens used.
        return _CacheSetupResult(
            prompt=list(prompt_tokens),
            full_prompt_tokens=prompt_tokens,
            cache_creation_tokens=len(prompt_tokens),
            cache_setup_done=True,
        )

    if hit is None:
        # Cold start with usable boundaries: build a fresh cache.  The
        # drive will then run at most two ``model(...)`` calls and store
        # one checkpoint at the deepest interior boundary (see
        # ``_drive_segmented_prefill``'s docstring for why per-segment
        # chunking was abandoned).
        cache = _make_prompt_cache_for_lm(lm)
        already_covered = 0
    else:
        # Warm start: deepcopy the stored snapshot (so subsequent
        # mutation during drive doesn't corrupt the stored entry).
        # eager_eval is necessary when the loaded entry contains
        # lazy-state layer types (ArraysCache) — otherwise re-evaluating
        # the lazy graph on this worker thread re-introduces the #284
        # Metal stream crash. Don't rely on "already materialised when
        # stored" — pre-PR disk entries from the old flat path were
        # stored before snapshot_cache_for_persistence existed.
        cached_state, _ = hit
        needs_eager_load = _cache_list_contains_lazy_state(cached_state.cache)
        cache = snapshot_cache_for_persistence(
            cached_state.cache,
            eager_eval=needs_eager_load,
        )
        already_covered = len(cached_state.tokens)

    # The drive is DEFERRED to the generation worker thread: under
    # thread-local streams (mlx >= 0.31.2) running it here, on the event
    # loop, would prefill on the loop thread's generation-stream instance
    # while decode uses the worker's — the #284 cross-stream GDN hazard.
    # Suffix and token counts are deterministic (the drive always reserves
    # exactly the trailing token for stream_generate), so they are computed
    # here; only the model() forwards move.
    flat = segmented.flatten()

    loop = asyncio.get_running_loop()
    store = lm.prompt_cache_store

    def _insert_checkpoint_threadsafe(state: CachedPromptState) -> None:
        # PromptCacheStore is loop-affine (assert_loop_thread, #463); the
        # drive runs on the worker thread, so marshal insertion back to the
        # loop. The snapshot is already materialized on the worker (mx.eval
        # per chunk), so the arrays may cross threads.
        def _do_insert() -> None:
            try:
                store.insert_checkpoint(state)
            except Exception:
                logger.warning("checkpoint insert failed", exc_info=True)

        loop.call_soon_threadsafe(_do_insert)

    model = lm.model

    def _deferred_prefill(cancel_event: threading.Event | None = None) -> None:
        _drive_segmented_prefill(
            model=model,
            segmented=segmented,
            cache=cache,
            insert_checkpoint=_insert_checkpoint_threadsafe,
            already_covered_tokens=already_covered,
            cancel_event=cancel_event,
        )

    gen_kwargs["prompt_cache"] = cache
    suffix: list[int] = [flat[-1]] if flat else []
    return _CacheSetupResult(
        prompt=suffix,
        full_prompt_tokens=prompt_tokens,
        cache_read_tokens=already_covered,
        cache_creation_tokens=max(0, len(prompt_tokens) - already_covered),
        cache_setup_done=True,
        deferred_prefill=_deferred_prefill if flat else None,
    )


async def _setup_vlm_prompt_cache(
    lm: LoadedModel,
    prompt_tokens: list[int] | None,
    gen_kwargs: dict,
    *,
    cache_id: str,
) -> tuple[int, int]:
    """Attach an mlx_vlm ``PromptCacheState`` to ``gen_kwargs`` for VLM KV reuse.

    Returns ``(cache_read_tokens, cache_creation_tokens)`` for metrics. mlx_vlm
    owns prefix matching / cache trimming / image-in-prefix detection and
    updates the state in place after generation; here we only fetch-or-create
    the per-cache_id state and report an *estimate* of the reused prefix.

    The estimate uses ``lm.text_tokenizer`` tokenization, which can differ
    slightly from mlx_vlm's internal ``input_ids`` (image placeholder
    expansion), so the counts are approximate — the actual reuse is whatever
    mlx_vlm's own ``find_prefix_length`` decides at generate time.
    """
    from mlx_vlm.generate import PromptCacheState

    store = getattr(lm, "vlm_prompt_cache_store", None)
    if store is None or not store.enabled() or prompt_tokens is None:
        return 0, len(prompt_tokens) if prompt_tokens is not None else 0

    # Memory pressure: drop the VLM store and fall back to a fresh state.
    if memory_utils.is_memory_pressure_high(settings.memory_limit_fraction):
        logger.warning("Memory pressure high, clearing VLM prompt cache")
        store.clear()
        mx.clear_cache()
        _safe_sync()

    state = await store.async_get(cache_id)
    if state is not None:
        read = state.find_prefix_length(prompt_tokens)
        # A full-prefix re-request reuses everything but the seed; clamp so we
        # never claim more reuse than there are new tokens to process.
        read = min(read, max(len(prompt_tokens) - 1, 0))
        store.note_hit(reused_tokens=read)
    else:
        state = PromptCacheState()
        await store.async_insert(cache_id, state)
        read = 0
        store.note_miss()

    gen_kwargs["prompt_cache_state"] = state
    creation = len(prompt_tokens) - read
    logger.info(
        "VLM prompt cache: ~%d prefix tokens reusable, ~%d new (cache_id=%s)",
        read,
        creation,
        cache_id,
    )
    return read, creation


async def _setup_prompt_cache(
    lm: LoadedModel,
    prompt: str | list[int],
    gen_kwargs: dict,
    *,
    prompt_tokens: list[int] | None,
    cache_id: str,
    messages: list[dict] | None = None,
    tokenizer: Any = None,
    template_kwargs: dict | None = None,
) -> _CacheSetupResult:
    """Set up prompt cache for a streaming or non-streaming completion.

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

    # Checkpoint path — non-trimmable cache layouts (set by probe in
    # Task 5.2). Drives prefill per message-segment and stores snapshots
    # at each boundary into the prompt cache store. Closes #284 (ArraysCache
    # via eager mx.eval) and #343 (RotatingKVCache via shorter-match lookup).
    if (
        lm.uses_checkpoint_persistence
        and messages is not None
        and tokenizer is not None
    ):
        return await _setup_via_checkpoint_path(
            lm,
            prompt,
            gen_kwargs,
            messages=messages,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            template_kwargs=template_kwargs,
        )

    # Cross-request prompt cache reuse is disabled for this model.  The
    # checkpoint path above handles the main non-trimmable families (#284
    # ArraysCache, #343 RotatingKVCache, #396 mixed Rotating+Arrays).  What
    # actually reaches here today:
    #   - Probe-failure models: _probe_cache_capabilities caught an exception
    #     and set supports_cache_persistence=False as the safe fallback.
    #   - Cache types not yet allowlisted by the probe.
    # We never store for these models, so the only path to a stale entry is
    # pre-PR data.  Skip the disk lookup entirely and clear any in-memory
    # entry.  Gated on peek() so we don't pay a blocking unlink(ENOENT)
    # syscall on every request.  Pre-PR on-disk files can persist
    # indefinitely — that's harmless because nothing reads from disk for
    # these models — but they're cleaned up by disk cache size eviction
    # in PromptCacheStore eventually.
    if not lm.supports_cache_persistence:
        if lm.prompt_cache_store.peek(cache_id) is not None:
            lm.prompt_cache_store.remove(cache_id)
        cached = None
    else:
        cached = await lm.prompt_cache_store.async_get(cache_id)
        # Issue #365: cache_id miss → look for a sibling that shares a long
        # token prefix. Takeover semantics — no KV copy. The old cache_id
        # loses its entry (documented limitation: concurrent two-stream
        # sibling branching falls back to fresh prefill on the loser).
        if cached is None and settings.prompt_cache_radix:
            found = lm.prompt_cache_store.find_by_prefix(
                prompt_tokens,
                min_prefix_tokens=settings.prompt_cache_radix_min_prefix_tokens,
            )
            if found is not None:
                old_cache_id, cached, prefix_len_hint = found
                lm.prompt_cache_store.takeover(old_cache_id, cache_id)
                logger.info(
                    "Radix prefix hit: %d tokens reused from cache_id=%s → %s",
                    prefix_len_hint,
                    old_cache_id,
                    cache_id,
                )
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
        # A lazy-state cache (TurboQuant/Spectral/Shard) trimmed HERE — on the
        # event-loop thread — leaves a lazy op bound to this thread's Metal
        # stream: a large trim takes the buffer-shrink branch
        # (``_key_indices = _key_indices[..., :cap, :]`` — a lazy slice) and the
        # quant ``.state`` rebuilds a lazy ``[:offset]`` slice on every access.
        # Under mlx thread-local streams (#499) the generation worker thread then
        # crashes evaluating it: "There is no Stream(gpu, N) in current thread"
        # at flash-MoE's ``mx.eval(inds)`` (pure-MoE + kv_cache_quant models —
        # GLM-4.5-Air, Qwen3-235B).  Defer the GPU trim to the worker via
        # ``deferred_prefill`` so it runs on the same thread that decodes.  A
        # trimmable cache trims exactly ``trim_amount`` whenever
        # ``trim_amount <= offset`` (always true here, ``suffix_start >= 0``), so
        # the count is predictable — the partial-trim fallback below is only for
        # non-lazy custom caches whose ``trim()`` clamps.
        defer_trim = lm.supports_cache_trim and _cache_list_contains_lazy_state(
            working_cache
        )
        if trim_amount > 0 and lm.supports_cache_trim:
            if defer_trim:
                # Predicted count (see comment): the actual trim runs on the
                # worker.  Defensive: a lazy-state cache here is always fully
                # trimmable, so a short trim would mean a broken invariant — log
                # it loudly rather than let a misaligned cache pass silently.
                trimmed = trim_amount

                def _deferred_trim(
                    _cancel: threading.Event | None = None,
                    _cache: list[Any] = working_cache,
                    _amount: int = trim_amount,
                ) -> None:
                    got = trim_prompt_cache(_cache, _amount)
                    if got != _amount:
                        logger.warning(
                            "Deferred prompt-cache trim under-delivered "
                            "(asked %d, got %d); cache may be misaligned",
                            _amount,
                            got,
                        )

                result.deferred_prefill = _deferred_trim
            else:
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
            result.prompt = suffix_tokens

    if "prompt_cache" not in gen_kwargs:
        # Reached on either: (a) no usable prefix in cache miss, or
        # (b) trim-fallback above.  In the trim-fallback path the
        # cache was already removed before the trim attempt — this
        # remove is then a harmless no-op (PromptCacheStore.remove
        # is idempotent).  Kept for the cache-miss path.  Skip for
        # non-persistable models — the lookup branch above already
        # ran the (peek-gated) cleanup, so any further remove() here
        # would be redundant disk I/O on the event loop hot path.
        if lm.supports_cache_persistence:
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
            # Use cache_read_tokens to match the trimmed KV state.  Issue
            # #284: for non-persistable models we skip the re-store and
            # clean up any stale entry instead — re-storing would
            # re-introduce the cross-request reuse path the persistence
            # guard exists to prevent, and _store_prompt_cache_after_generation
            # (which normally cleans up) isn't reached on the MemoryError
            # path.  Dropping the working reference still frees memory;
            # the eviction below runs regardless to flush other entries.
            if had_cache:
                if not lm.supports_cache_persistence:
                    lm.prompt_cache_store.remove(cache_id)
                elif full_prompt_tokens is not None:
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
                    # The trimmed cache was just dropped and the full prompt
                    # restored for a fresh re-prefill, so the prefix reuse the
                    # caller would otherwise report never happens — reset the
                    # counts to reflect the full re-prefill (#634).
                    result.cache_read_tokens = 0
                    result.cache_creation_tokens = len(full_prompt_tokens)
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

    Models routed through the checkpoint path (uses_checkpoint_persistence)
    return immediately — boundary snapshots were already taken during prefill
    by _drive_segmented_prefill.

    What actually reaches the body here:
      - Trimmable models (standard KVCache / QuantizedKVCache): the normal
        trim-and-store path.
      - Probe-failure models (supports_cache_persistence=False): caught by
        the early-return below, no storage occurs.
      - Other non-trimmable layouts (supports_cache_trim=False) that are
        persistable: stored as-is; strict-extension lookups reuse them.
    """
    if lm.uses_checkpoint_persistence:
        # Checkpoints already taken at boundaries during prefill by
        # _drive_segmented_prefill. Storing the post-generation state
        # would duplicate a longer entry that the next request's
        # fetch_nearest can't realign without trim — exactly what the
        # checkpoint path is designed to avoid.
        return

    prompt_cache = gen_kwargs.get("prompt_cache")
    if prompt_cache is None or full_prompt_tokens is None:
        return

    # Skip cross-request storage for non-persistable models.  After the
    # checkpoint-path port, the main #284 (ArraysCache), #343
    # (RotatingKVCache), and #396 (mixed Rotating+Arrays) families are
    # handled via uses_checkpoint_persistence and return above.  What
    # reaches here are probe-failure models whose _setup_prompt_cache
    # already cleared any stale entry.  No remove() call needed: nothing
    # was stored between setup and now.
    if not lm.supports_cache_persistence:
        logger.debug(
            "Cache not persistable (hybrid SSM/ArraysCache or non-trimmable "
            "sliding-window); skipping cross-request storage for %s",
            cache_id,
        )
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
            # The store-time trim above ran on THIS (event-loop) thread. For a
            # lazy-state cache (TurboQuant) a large trim takes `trim`'s
            # buffer-shrink branch (`_key_indices = _key_indices[..., :cap, :]`
            # — a lazy slice) and re-binds the packed buffers to the loop
            # thread's Metal stream, *after* the worker-side materialize already
            # ran. `_shed_transient_buffers` only nulls the dequant side
            # buffers, so that lazy slice would be stored as-is and crash the
            # next request's worker ("no Stream(gpu, N)"). Re-materialize here,
            # on the loop thread that owns the freshly-created slice op, so the
            # stored buffers are thread-agnostic leaves again.
            materialize_lazy_cache_state(prompt_cache)
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


def _batch_eligible(
    lm: LoadedModel,
    gen_kwargs: dict,
    *,
    max_tokens: int,
    images: list[str] | None,
    audio: list[str] | None,
) -> bool:
    """Route a chat request to the batch engine? (plan §3)

    Policy: opt-in (``OLMLX_BATCHING``), text-only dense models on plain
    ``KVCache`` layers. Everything else — VLM/audio kinds, speculative,
    distributed, flash, KV-quant, per-request seed (global PRNG →
    batched reproducibility impossible), gpt-oss channel format — stays
    on the exclusive-lock path unchanged.

    Grammar requests batch as of Phase 2: ``GenerationBatch._step``
    calls each sequence's processors with that sequence's ``[1, vocab]``
    logits row and a per-sequence ``TokenBuffer`` token history (prompt
    included on the first call), which is exactly the
    ``GrammarLogitsProcessor`` contract on the exclusive path.
    """
    # Per-model override (models.json ``batching``) replaces the global
    # opt-in gate; None defers to ``OLMLX_BATCHING`` — "default-on per-model"
    # without flipping the global default. Identity check, not truthiness:
    # tests patch ``inference.settings`` with a MagicMock whose every
    # attribute is truthy (and ``lm.batching`` is None there), which must
    # never switch a mocked request onto the batched path.
    batching = lm.batching if lm.batching is not None else settings.batching
    if batching is not True:
        return False
    # Ollama num_predict -1/-2 mean "unlimited"; mlx-lm's BatchGenerator
    # checks ``generated >= max_tokens``, which is instantly true for a
    # negative value (one-token response). stream_generate's ``n ==
    # max_tokens`` treats them as infinite, so those requests stay there.
    if max_tokens <= 0:
        return False
    if (
        lm.is_vlm
        or lm.is_whisper
        or lm.is_tts
        or lm.is_reranker
        or lm.is_distributed
        or lm.is_flash
        or lm.is_flash_moe
        or lm.is_speculative
    ):
        return False
    if lm.kv_cache_quant is not None:
        return False
    if images or audio:
        return False
    if gen_kwargs.get("seed") is not None:
        return False
    if lm.template_caps.has_channel_format:
        return False
    if lm.batch_convertible is None:
        try:
            from olmlx.engine.batching import caches_plain_kv

            lm.batch_convertible = caches_plain_kv(_make_prompt_cache_for_lm(lm))
        except Exception:
            logger.debug("batch cache probe failed for %s", lm.name, exc_info=True)
            lm.batch_convertible = False
        if not lm.batch_convertible:
            logger.info(
                "Model %s is not batch-eligible (cache layout); using the "
                "exclusive path",
                lm.name,
            )
    return bool(lm.batch_convertible)


def _get_batch_scheduler(lm: LoadedModel) -> Any:
    """Lazily create the per-model BatchScheduler.

    The lock callables injected here are the scheduler's only coupling to
    this module: one untimed FIFO acquisition per busy period (per-request
    queue timeouts are enforced consumer-side in
    ``_stream_completion_batched``), with the same deferred-cleanup
    handshake and boundary syncs as the exclusive paths.
    """
    if lm.batch_scheduler is not None:
        return lm.batch_scheduler

    from olmlx.engine.batching import BatchScheduler

    model = lm.model
    stop_tokens = [[t] for t in lm.tokenizer.eos_token_ids]
    # Per-model batch-size overrides (models.json) fall back to the globals.
    completion_size = (
        lm.batch_completion_size
        if lm.batch_completion_size is not None
        else settings.batch_completion_size
    )
    prefill_size = (
        lm.batch_prefill_size
        if lm.batch_prefill_size is not None
        else settings.batch_prefill_size
    )
    prefill_step = (
        lm.batch_prefill_step
        if lm.batch_prefill_step is not None
        else settings.batch_prefill_step
    )
    fairness_quantum = (
        lm.batch_fairness_quantum
        if lm.batch_fairness_quantum is not None
        else settings.batch_fairness_quantum
    )
    sync_mode = lm.sync_mode

    def factory() -> Any:
        from mlx_lm.generate import BatchGenerator

        return BatchGenerator(
            model,
            stop_tokens=stop_tokens,
            completion_batch_size=completion_size,
            prefill_batch_size=prefill_size,
            prefill_step_size=prefill_step,
        )

    async def acquire_gpu() -> None:
        # The shared entry counts the waiting manager in _queue_depth so
        # OTHER models' running batch workers see it via their
        # exclusive_pending latch and drain — otherwise a busy model
        # starves every other model's batched requests (they never bump
        # the depth on their own). Timeout 0 disables the acquire
        # timeout: the manager waits FIFO behind exclusive requests;
        # batched requests time out individually consumer-side.
        await _enter_inference_lock(
            0,
            sync_mode=sync_mode,
            queued_log="Batch manager queued for inference lock (queue depth: %d)",
        )

    def release_gpu() -> None:
        _exit_inference_lock(sync_mode, context="batch release")

    # Aggregate (cross-sequence) KV admission (plan §10). The worker calls
    # these on its own thread (which owns the GPU): ``kv_headroom`` is the
    # bytes free for new KV right now — the memory ceiling minus current
    # Metal use, so resident sequences are already netted out — and
    # ``kv_estimate`` is one request's new-KV cost (suffix prefill +
    # generated; a reused prefix is already-resident memory). Batched
    # eligibility excludes KV-quant, so the estimate is plain fp16.
    def kv_headroom() -> int:
        total_physical = memory_utils.get_system_memory_bytes()
        if total_physical <= 0:
            return 1 << 62  # memory unknown — gate self-disables
        limit = int(total_physical * settings.memory_limit_fraction)
        return limit - memory_utils.get_metal_memory()

    def kv_estimate(request: Any) -> int:
        n = len(request.tokens) + max(request.max_tokens, 0)
        try:
            return estimate_kv_cache_bytes(model, n)
        except Exception:
            return 0  # estimate unavailable — admit (per-request gate stands)

    admission = settings.batch_kv_admission
    lm.batch_scheduler = BatchScheduler(
        generator_factory=factory,
        acquire_gpu=acquire_gpu,
        release_gpu=release_gpu,
        exclusive_pending=lambda: _queue_depth > 0,
        consumer_lag_limit=settings.batch_consumer_lag_limit,
        kv_headroom=kv_headroom if admission else None,
        kv_estimate=kv_estimate if admission else None,
        fairness_quantum=fairness_quantum,
        name=lm.name,
    )
    return lm.batch_scheduler


async def _drain_to_terminal(seq: Any) -> None:
    """Consume events until the worker confirms the sequence left the
    batch (done/error). Keeps the model pinned via the caller's
    _inference_ref until no forward pass can still involve it."""
    while True:
        event = await seq.out.get()
        if event["type"] in ("done", "error"):
            return


def _batched_kv_preflight(lm: LoadedModel, num_tokens: int, max_tokens: int) -> None:
    """Per-request KV admission for the batched path.

    The exclusive path's ``_kv_cache_preflight_check`` can't run here (its
    remediation mutates prompt-cache state under the inference lock, which
    a batched consumer doesn't hold), but the pure estimate half can:
    reject a request whose own KV would blow ``OLMLX_MEMORY_LIMIT_FRACTION``
    with a MemoryError (→ HTTP 503) instead of risking the uncatchable
    Metal OOM. Aggregate (cross-sequence) admission — keeping the *sum* of
    co-tenant KV under the ceiling — is the batch worker's job via the
    scheduler's ``kv_headroom``/``kv_estimate`` gate (``OLMLX_BATCH_KV_ADMISSION``,
    plan §10/§11); this per-request check still stands as the first line.
    """
    total_physical = memory_utils.get_system_memory_bytes()
    if total_physical <= 0 or num_tokens <= 0:
        return
    memory_limit = int(total_physical * settings.memory_limit_fraction)
    try:
        kv_bytes = estimate_kv_cache_bytes(
            lm.model,
            num_tokens + max(max_tokens, 0),
            kv_cache_quant=lm.kv_cache_quant,
        )
        current_metal = memory_utils.get_metal_memory()
    except Exception:
        logger.warning(
            "Batched KV pre-flight check skipped — OOM protection inactive",
            exc_info=True,
        )
        return
    if current_metal + kv_bytes > memory_limit:
        available_gb = max(0.0, (memory_limit - current_metal) / 1024**3)
        raise MemoryError(
            f"KV cache for {num_tokens + max(max_tokens, 0)} tokens estimated "
            f"at {kv_bytes / 1024**3:.1f} GB, but only {available_gb:.1f} GB "
            "available — prompt too long, reduce context or use a smaller model"
        )


@dataclasses.dataclass
class _BatchedCacheSetup:
    """Result of batched prompt-cache setup.

    ``suffix_tokens`` is what the sequence prefills; on a hit, ``cache``
    + ``history_tokens`` seed it with the reused prefix. ``store`` is
    True when this model participates in cross-request storage (the
    worker is then asked to hand the final KV back via the done event).
    """

    suffix_tokens: list[int]
    cache: list[Any] | None = None
    history_tokens: list[int] | None = None
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    store: bool = False


async def _setup_batched_prompt_cache(
    lm: LoadedModel, tokens: list[int], cache_id: str
) -> _BatchedCacheSetup:
    """Prompt-cache lookup for the batched path (plan §4 item 2).

    Move semantics: the entry is *taken* out of the store (``async_take``)
    so a concurrent request can never share the mutable cache object —
    batched consumers are not serialized by the inference lock, so the
    exclusive path's get-then-remove sequence (Bug #123) is not
    race-free here. The cache re-enters the store when the sequence
    finishes (the worker hands it back via the done event).

    The trim runs on the event loop, but eligibility restricts the
    batched path to plain ``KVCache`` layers whose trim is offset
    bookkeeping only (no Metal work) — it cannot race the batch worker's
    GPU stream.
    """
    res = _BatchedCacheSetup(
        suffix_tokens=list(tokens), cache_creation_tokens=len(tokens)
    )
    # Persistence can still be off for probe-failure models even though
    # the batch probe passed; those keep Phase 1 behavior (fresh prefill,
    # nothing stored). Checkpoint-persistence models can't reach here
    # (their caches are not plain KVCache), checked for symmetry.
    if (
        not lm.supports_cache_persistence
        or not lm.supports_cache_trim
        or lm.uses_checkpoint_persistence
    ):
        return res
    res.store = True
    state = await lm.prompt_cache_store.async_take(cache_id)
    if state is None and settings.prompt_cache_radix:
        # Sibling-prefix takeover (issue #365): take the matched entry
        # directly under its old id — it will be re-stored under the new
        # cache_id at finish, which is what takeover would have done.
        found = lm.prompt_cache_store.find_by_prefix(
            tokens,
            min_prefix_tokens=settings.prompt_cache_radix_min_prefix_tokens,
        )
        if found is not None:
            old_cache_id, _, prefix_len_hint = found
            state = lm.prompt_cache_store.take(old_cache_id)
            if state is not None:
                logger.info(
                    "Radix prefix hit (batched): %d tokens reused from "
                    "cache_id=%s → %s",
                    prefix_len_hint,
                    old_cache_id,
                    cache_id,
                )
    if state is None:
        return res
    prefix_len = _find_common_prefix(tokens, state.tokens)
    # Back up one position on exact match so the sequence still has a
    # token to prefill (same rule as the exclusive path); a hit that
    # would leave suffix_start == 0 is useless — the taken entry is
    # simply dropped (the exclusive path's remove-on-miss equivalent).
    suffix_start = min(prefix_len, len(tokens) - 1) if tokens else 0
    if suffix_start <= 0:
        return res
    trim_amount = len(state.tokens) - suffix_start
    if trim_amount > 0:
        trimmed = trim_prompt_cache(state.cache, trim_amount)
        if trimmed != trim_amount:
            logger.warning(
                "Batched prompt cache trim incomplete (asked for %d, got "
                "%d); discarding and prefilling fresh",
                trim_amount,
                trimmed,
            )
            return res
    res.cache = state.cache
    res.history_tokens = list(tokens[:suffix_start])
    res.suffix_tokens = list(tokens[suffix_start:])
    res.cache_read_tokens = suffix_start
    res.cache_creation_tokens = len(tokens) - suffix_start
    logger.info(
        "Prompt cache hit (batched): %d prefix tokens reused, %d new "
        "tokens to process (was %d total)",
        suffix_start,
        len(res.suffix_tokens),
        len(tokens),
    )
    return res


async def _restore_taken_cache(
    lm: LoadedModel, cache_id: str, cache_setup: _BatchedCacheSetup | None
) -> None:
    """Give a taken prompt-cache entry back to the store.

    Used when a batched request fails before/without consuming its
    seeded prefix (batch queue timeout, preflight MemoryError) so a
    retry still gets the prefix — the exclusive path never loses the
    entry in those cases (its queue timeout fires before cache setup;
    its preflight re-stores the trimmed cache before evicting to disk).
    The trimmed object covers exactly ``history_tokens`` and is only
    ever *read* by the batch (merge copies), so re-storing is safe even
    if the sequence was admitted before being swept.
    """
    if cache_setup is None or not cache_setup.cache:
        return
    await lm.prompt_cache_store.async_set(
        cache_id,
        CachedPromptState(
            tokens=list(cache_setup.history_tokens or []),
            cache=cache_setup.cache,
        ),
    )


async def _stream_completion_batched(
    lm: LoadedModel,
    prompt: str | list[int],
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    *,
    keep_alive: str | None = None,
    prompt_tokens: list[int] | None = None,
    use_prompt_cache: bool = False,
    cache_id: str = "",
    thinking_expected: bool = False,
) -> AsyncGenerator[dict, None]:
    """Batched counterpart of ``_stream_completion`` (plan §2/§4).

    Submits to the per-model BatchScheduler instead of taking the
    inference lock for the whole request; yields the same chunk dicts so
    routers are unchanged. Phase 2 adds the prompt-cache round trip
    (take → seed the sequence → re-store from the done event) and
    per-sequence grammar; no channel filter (excluded by eligibility).
    """
    from olmlx.engine.batching import BatchRequest

    stop_sequences: list[str] | None = gen_kwargs.pop("stop", None)
    if prompt_tokens is not None:
        tokens = list(prompt_tokens)
    elif isinstance(prompt, str):
        tokens = tokenize_for_cache(lm.text_tokenizer, prompt)
    else:
        tokens = list(prompt)

    cache_setup: _BatchedCacheSetup | None = None
    if use_prompt_cache and make_prompt_cache is not None:
        cache_setup = await _setup_batched_prompt_cache(lm, tokens, cache_id)
        submit_tokens = cache_setup.suffix_tokens
    else:
        submit_tokens = tokens

    # New-KV estimate covers only what this sequence will prefill; a
    # reused prefix is already-resident memory moved out of the store.
    # On rejection, restore the taken entry — exclusive-path parity: its
    # preflight re-stores the trimmed cache (then spills under pressure)
    # so a retry keeps the prefix; the 503'd request frees its KV either
    # way (the store's own pressure machinery handles residency).
    try:
        _batched_kv_preflight(lm, len(submit_tokens), max_tokens)
    except MemoryError:
        await _restore_taken_cache(lm, cache_id, cache_setup)
        raise

    scheduler = _get_batch_scheduler(lm)
    detokenizer = lm.tokenizer.detokenizer  # fresh streaming instance
    stats.prompt_eval_count = len(tokens)

    queue_timeout = (
        lm.inference_queue_timeout
        if lm.inference_queue_timeout is not None
        else settings.inference_queue_timeout
    )
    inf_timeout = (
        lm.inference_timeout
        if lm.inference_timeout is not None
        else settings.inference_timeout
    )

    stop_scanner = StopScanner(stop_sequences, thinking_aware=thinking_expected)
    stop_hit = False
    timed_out = False
    lagged = False
    max_tokens_hit = False
    terminal_seen = False
    start_ns = time.monotonic_ns()
    first_token_ns: int | None = None
    last_token_ns: int | None = None

    seq: Any = None
    try:
        # Start of the response body — mirrors the exclusive path's
        # cache_info yield (routers surface read/creation counts).
        if cache_setup is not None:
            yield {
                "cache_info": True,
                "cache_read_tokens": cache_setup.cache_read_tokens,
                "cache_creation_tokens": cache_setup.cache_creation_tokens,
            }
        with _inference_ref(lm, keep_alive=keep_alive):
            try:
                seq = await scheduler.submit(
                    BatchRequest(
                        tokens=submit_tokens,
                        max_tokens=max_tokens,
                        sampler=gen_kwargs.get("sampler"),
                        logits_processors=gen_kwargs.get("logits_processors"),
                        cache=cache_setup.cache if cache_setup else None,
                        history_tokens=(
                            cache_setup.history_tokens if cache_setup else None
                        ),
                        return_cache=bool(cache_setup and cache_setup.store),
                    )
                )
                inf_start = time.monotonic()
                awaiting_first = True
                while True:
                    if awaiting_first and queue_timeout and queue_timeout > 0:
                        try:
                            event = await asyncio.wait_for(seq.out.get(), queue_timeout)
                        except (asyncio.TimeoutError, TimeoutError):
                            await _restore_taken_cache(lm, cache_id, cache_setup)
                            raise ServerBusyError(
                                "Server busy: batch queue timeout after "
                                f"{queue_timeout}s"
                            ) from None
                    else:
                        event = await seq.out.get()
                    awaiting_first = False
                    seq.note_consumed()  # keep the backpressure lag honest
                    etype = event["type"]
                    if etype == "progress":
                        continue
                    if etype == "error":
                        terminal_seen = True
                        # Re-raise the worker's exception with its original
                        # type so the app-level handlers keep their mapping
                        # (MemoryError → 503, ValueError → 422); a generic
                        # wrapper would turn every failure into a 500.
                        raise event["exc"]
                    if etype == "done":
                        terminal_seen = True
                        # A lag cancellation (worker dropped a slow consumer
                        # — plan §11) is a benign truncation, not the
                        # model-unload cancel; end the stream cleanly.
                        lagged = event.get("truncated") == "lag"
                        if lagged:
                            logger.warning(
                                "Batched generation truncated: consumer lag "
                                "exceeded the backpressure limit"
                            )
                        max_tokens_hit = event["reason"] == "length"
                        if event["reason"] == "cancelled" and not (
                            stop_hit or timed_out or lagged
                        ):
                            # Scheduler closed under us (model unload).
                            raise RuntimeError(
                                "Batched generation cancelled (model unloading)"
                            )
                        if not (stop_hit or timed_out or lagged):
                            detokenizer.finalize()
                            tail = detokenizer.last_segment
                            if tail:
                                tail, stop_hit = stop_scanner.feed(tail)
                                if tail:
                                    yield {"text": tail, "done": False}
                        # Re-store the sequence's KV under cache_id. The
                        # worker ships the cache only when asked
                        # (return_cache) and skips it on timeout/disconnect
                        # (want_cache cleared) — a stop-hit cancellation is
                        # a *successful* completion and keeps it. The
                        # prefix-length guard skips storage for a sequence
                        # cancelled mid-prefill (partial KV shorter than
                        # the prompt).
                        ev_cache = event.get("cache")
                        if (
                            ev_cache is not None
                            and not timed_out
                            and not lagged
                            and len(event.get("tokens") or []) >= len(tokens)
                        ):
                            all_toks = event["tokens"]
                            generated = list(all_toks[len(tokens) :])
                            await _store_prompt_cache_after_generation(
                                lm,
                                {"prompt_cache": ev_cache},
                                tokens,
                                generated,
                                len(generated),
                                cache_id,
                            )
                        break
                    # token
                    if stop_hit or timed_out:
                        # Already cancelled; drain to the terminal event.
                        continue
                    last_token_ns = time.monotonic_ns()
                    if first_token_ns is None:
                        first_token_ns = last_token_ns
                    stats.eval_count += 1
                    detokenizer.add_token(event["token"])
                    piece = detokenizer.last_segment
                    if piece:
                        piece, stop_hit = stop_scanner.feed(piece)
                        if piece:
                            yield {"text": piece, "done": False}
                    if stop_hit:
                        scheduler.cancel(seq)
                        continue
                    if (
                        inf_timeout is not None
                        and (time.monotonic() - inf_start) > inf_timeout
                    ):
                        logger.warning(
                            "Inference timeout after %.1fs (limit: %.1fs)",
                            time.monotonic() - inf_start,
                            inf_timeout,
                        )
                        timed_out = True
                        # The generation is incomplete; don't pay for the
                        # cache extraction (parity with the exclusive
                        # path's invalidate-on-incomplete).
                        seq.want_cache = False
                        scheduler.cancel(seq)
            finally:
                if seq is not None and not terminal_seen:
                    # Client disconnect / consumer exception: free the
                    # batch slot and wait (bounded) for the worker to
                    # confirm removal before dropping the model pin.
                    seq.want_cache = False
                    scheduler.cancel(seq)
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(_drain_to_terminal(seq), 30.0)

        end_ns = time.monotonic_ns()
        stats.total_duration = end_ns - start_ns
        if first_token_ns is not None:
            stats.prompt_eval_duration = first_token_ns - start_ns
            if last_token_ns is not None:
                stats.eval_duration = max(last_token_ns - first_token_ns, 0)
        logger.info(
            "Batched generation complete: %d prompt tokens, %d tokens "
            "generated, %.2fs total",
            stats.prompt_eval_count,
            stats.eval_count,
            stats.total_duration / 1e9,
        )

        done_chunk: dict = {"text": "", "done": True, "stats": stats}
        if timed_out:
            done_chunk["done_reason"] = "timeout"
        elif stop_hit:
            done_chunk["done_reason"] = "stop"
        elif max_tokens_hit:
            # Scheduler exhausted the token budget — mirrors exclusive path.
            done_chunk["done_reason"] = "length"
        try:
            _metrics.observe_inference(lm.name, surface_var.get(), stats)
        except Exception:
            logger.debug("metrics: observe_inference failed (batched)", exc_info=True)
        yield done_chunk
    finally:
        exc_type = sys.exc_info()[0]
        # Normal completion leaves no exception; a consumer closing the
        # generator after the done chunk raises GeneratorExit (excluded).
        # ServerBusyError = queue timeout, not an inference error — the
        # exclusive path raises it before its metered section.
        if exc_type is not None and not issubclass(
            exc_type,
            (GeneratorExit, asyncio.CancelledError, ServerBusyError),
        ):
            try:
                _metrics.observe_inference(
                    lm.name, surface_var.get(), stats, error=True
                )
            except Exception:
                logger.debug(
                    "metrics: error observe failed (batched)",
                    exc_info=True,
                )


async def _stream_completion(
    lm: LoadedModel,
    prompt: str | list[int],
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
    audio: list[str] | None = None,
    *,
    use_prompt_cache: bool = False,
    prompt_tokens: list[int] | None = None,
    cache_id: str = "",
    keep_alive: str | None = None,
    grammar_active: bool = False,
    adopt_pin: bool = False,
    messages: list[dict] | None = None,
    tokenizer: Any = None,
    template_kwargs: dict | None = None,
    thinking_expected: bool = False,
    collect_generated_tokens: bool = False,
) -> AsyncGenerator[dict, None]:
    # Continuous batching (docs/batching-plan.md): eligible requests join
    # the per-model batch instead of serializing on the inference lock.
    # A context request (#656) skips it — the batched path aggregates its
    # internal stream and never emits per-token ids, so it can't build the
    # continuation context; mirrors the guard in `_full_completion`.
    if not collect_generated_tokens and _batch_eligible(
        lm,
        gen_kwargs,
        max_tokens=max_tokens,
        images=images,
        audio=audio,
    ):
        async for chunk in _stream_completion_batched(
            lm,
            prompt,
            max_tokens,
            gen_kwargs,
            stats,
            keep_alive=keep_alive,
            prompt_tokens=prompt_tokens,
            use_prompt_cache=use_prompt_cache,
            cache_id=cache_id,
            thinking_expected=thinking_expected,
        ):
            yield chunk
        return

    # Use explicit enter/exit instead of `async with` to prevent
    # CancelledError from releasing the lock before cleanup completes.
    await _enter_inference_lock(
        lm.inference_queue_timeout,
        sync_mode=lm.sync_mode,
        queued_log="Streaming request queued for inference lock (queue depth: %d)",
    )

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
    # Pop stop sequences before cache setup / stream creation so they are not
    # forwarded to mlx-lm (which does not support them).
    stop_sequences: list[str] | None = gen_kwargs.pop("stop", None)
    # Initialize before try so the finally can always reference it.
    _audio_temps: list[str] = []
    try:
        audio_paths, _audio_temps = materialize_audio(audio)
        # Cache setup — must happen after lock to prevent concurrent cache corruption
        if use_prompt_cache and lm.is_vlm:
            # VLM: attach an mlx_vlm PromptCacheState. ``prompt`` stays the full
            # str — mlx_vlm tokenizes it and reuses the KV prefix internally.
            read, creation = await _setup_vlm_prompt_cache(
                lm, prompt_tokens, gen_kwargs, cache_id=cache_id
            )
            cs = _CacheSetupResult(
                prompt=prompt,
                cache_read_tokens=read,
                cache_creation_tokens=creation,
                # None: the text-tokenizer count misses image-patch expansion;
                # mlx_vlm's per-token prompt_tokens is the accurate full size, so
                # let it stand instead of overriding with an undercount.
                full_prompt_tokens=None,
                cache_setup_done=True,
            )
        elif use_prompt_cache:
            cs = await _setup_prompt_cache(
                lm,
                prompt,
                gen_kwargs,
                prompt_tokens=prompt_tokens,
                cache_id=cache_id,
                messages=messages,
                tokenizer=tokenizer,
                template_kwargs=template_kwargs,
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

        if lm.is_speculative and not (images or audio_paths) and grammar_active:
            # Speculative decoders do not consume gen_kwargs["logits_processors"],
            # so a grammar-constrained request would bypass the mask entirely.
            # Disable speculative for this request (issue #361). Constrain-after-
            # verify across the 4 speculative strategies is a follow-up.
            logger.warning(
                "Speculative decoding disabled for this request: "
                "grammar-constrained decoding is not yet plumbed through "
                "the speculative path"
            )
            use_speculative = False
        else:
            use_speculative = lm.is_speculative and not (images or audio_paths)

        # prompt is a str or a token-id list; only the latter gives a count here.
        _prefill_prompt_tokens = len(prompt) if isinstance(prompt, list) else None
        with _tracing.span(
            "prefill",
            prompt_tokens=_prefill_prompt_tokens,
            **{"gen.stream": True},
        ):
            if use_speculative:
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
                # Cross-request KV reuse (#421): when the decoder owns an
                # enabled snapshot store (classic / pld with
                # OLMLX_SPECULATIVE_CACHE_SLOTS > 0), build a SegmentedPrompt so
                # prefill reuses a stored prefix instead of re-prefilling the
                # whole growing conversation each turn. Pass the *token list*
                # (not the str) as the prompt and segment from those exact
                # tokens so segmented.flatten() == the tokens prefill sees —
                # the alignment the reuse path's boundary guard requires.
                spec_prompt: str | list[int] = prompt
                spec_segmented = None
                _spec_store = getattr(lm.speculative_decoder, "_cache_store", None)
                if (
                    _spec_store is not None
                    and _spec_store.enabled()
                    and messages is not None
                    and not lm.is_vlm
                ):
                    spec_tokens = (
                        # ``tokenize_for_cache`` replicates mlx-lm's BOS
                        # heuristic; plain ``.encode`` (add_special_tokens=True)
                        # would double the BOS on BOS-prefixed templates
                        # (Llama 3 / Gemma / Mistral), diverging from the
                        # non-speculative path (#633).
                        tokenize_for_cache(lm.text_tokenizer, prompt)
                        if isinstance(prompt, str)
                        else list(prompt)
                    )
                    spec_segmented = tokenize_segmented_chat(
                        lm.text_tokenizer,
                        messages,
                        full_tokens=spec_tokens,
                        **(template_kwargs or {}),
                    )
                    spec_prompt = spec_tokens
                stream = async_speculative_stream(
                    lm.speculative_decoder,
                    lm.text_tokenizer,
                    spec_prompt,
                    max_tokens=max_tokens,
                    segmented=spec_segmented,
                )
            else:
                if lm.is_speculative:
                    logger.debug(
                        "speculative decoding skipped: request includes images or audio"
                    )
                # The decode loop runs in the CancellableStream worker thread;
                # OTel context does not cross threads, so hand the current context
                # (under the request's inference span) to the stream for the worker
                # to re-attach — otherwise worker-thread spans (e.g. flash) orphan.
                stream = async_mlx_stream(
                    lm.model,
                    lm.tokenizer,
                    prompt,
                    max_tokens=max_tokens,
                    is_vlm=lm.is_vlm,
                    images=images,
                    audio=audio_paths,
                    memory_limit=memory_limit,
                    trace_context=_tracing.current_context(),
                    deferred_prefill=cs.deferred_prefill,
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
        stop_hit = False
        stopped = False

        with (
            _tracing.span("decode") as _decode_span,
            _inference_ref(lm, keep_alive=keep_alive),
            Timer() as total_timer,
        ):
            with Timer() as eval_timer:
                inf_start = time.monotonic()
                prefill_start_ns = time.perf_counter_ns()
                ttft_measured_ns: int | None = None
                token = None
                stop_scanner = StopScanner(
                    stop_sequences, thinking_aware=thinking_expected
                )
                async for token in stream:
                    if ttft_measured_ns is None:
                        ttft_measured_ns = time.perf_counter_ns() - prefill_start_ns
                    # Always accumulate for prompt cache (raw stream, not filtered)
                    stats.eval_count = token.generation_tokens
                    # Report the full prompt size to the caller — when the
                    # cache path (flat or checkpoint) hands a suffix to
                    # mlx-lm, `token.prompt_tokens` only counts what
                    # mlx-lm actually re-prefilled, not the full request.
                    stats.prompt_eval_count = (
                        len(full_prompt_tokens)
                        if full_prompt_tokens is not None
                        else token.prompt_tokens
                    )
                    if token.token is not None:
                        generated_tokens.append(token.token)
                    else:
                        logger.debug(
                            "Skipping token with None ID at generation step %d "
                            "(cache token sequence will be incomplete)",
                            token.generation_tokens,
                        )

                    # Check stop sequences before yielding so the current
                    # token can be truncated at the earliest match.
                    if stop_sequences:
                        token_part, stop_hit = stop_scanner.feed(token.text or "")
                        if stop_hit:
                            if token_part:
                                if channel_filter is None:
                                    yield {"text": token_part, "done": False}
                                elif channel_filter.should_yield(token_part):
                                    yield {"text": token_part, "done": False}
                            # Cancel the worker so it stops decoding past-stop
                            # tokens into the shared prompt_cache — otherwise it
                            # keeps mutating the very object we are about to
                            # store by reference (#604). Mirrors the timeout
                            # path's cancel below.
                            stopped = True
                            stream.cancel()
                            break

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
                # On a cache hit stats.prompt_eval_count is the full prompt but
                # token.prompt_tokens (and prompt_tps) cover only the re-prefilled
                # suffix — apportion the prefill duration against the suffix.
                prefilled_count=(
                    getattr(token, "prompt_tokens", None) if token is not None else None
                ),
            )
            # ttft_ns: the prefill forward runs lazily inside the stream's
            # first iteration (worker thread), so the prefill *span* can't time
            # it; surface the measured prefill duration here instead. cache_hit:
            # mlx re-prefilled fewer tokens than the full prompt → a prefix was
            # served from the prompt cache.
            _decode_span.set_attributes(
                {
                    "eval_count": stats.eval_count,
                    "decode_tok_s": _metrics._decode_tps(stats),
                    "ttft_ns": (
                        ttft_measured_ns
                        if ttft_measured_ns is not None
                        else stats.prompt_eval_duration
                    ),
                    "ttft_measured": ttft_measured_ns is not None,
                    "cache_hit": bool(full_prompt_tokens)
                    and token is not None
                    and (token.prompt_tokens or 0) < len(full_prompt_tokens),
                }
            )
            if ttft_measured_ns is not None and token is not None:
                _fresh = token.prompt_tokens or 0
                _full = (
                    len(full_prompt_tokens)
                    if full_prompt_tokens is not None
                    else _fresh
                )
                logger.info(
                    "prefill %.2fs (fresh %d/%d tok, cache-covered %d)",
                    ttft_measured_ns / 1e9,
                    _fresh,
                    _full,
                    max(0, _full - _fresh),
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

            if stopped:
                # Stop-sequence hit: the worker was cancelled mid-decode, so the
                # live prompt_cache holds tokens past the stop that our
                # generated_tokens metadata does not track. Storing it by
                # reference would misalign the next cache hit (#604). Drop any
                # entry for this cache_id (setup may have installed this same
                # mutated object) and skip storage — mirrors the timeout path,
                # which never stores. The prefix is re-prefilled next turn.
                if full_prompt_tokens is not None:
                    lm.prompt_cache_store.remove(cache_id)
            else:
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
        if collect_generated_tokens:
            # Surface the produced token ids so generate_completion can build
            # the Ollama continuation context (#656). Already accumulated above
            # for the prompt cache; expose a copy so a later cache-store trim
            # can't mutate what the router forwards.
            done_chunk["generated_tokens"] = list(generated_tokens)
        if timed_out:
            done_chunk["done_reason"] = "timeout"
        elif stop_hit:
            done_chunk["done_reason"] = "stop"
        elif max_tokens > 0 and stats.eval_count >= max_tokens:
            # Stream terminated at the token budget (no EOS / stop sequence).
            # "length" mirrors OpenAI's finish_reason and the batched path.
            done_chunk["done_reason"] = "length"
        try:
            _metrics.observe_inference(lm.name, surface_var.get(), stats)
        except Exception:
            logger.debug("metrics: observe_inference failed (stream)", exc_info=True)
        yield done_chunk
    finally:
        # Release GPU-backed references from gen_kwargs so they can be
        # garbage-collected.  prompt_cache is either stored in the cache
        # store (successful path) or should be freed; prompt_cache_state
        # (VLM path) is owned by the VLM store, so this just drops a
        # duplicate reference; input_ids is legacy (no longer set).
        gen_kwargs.pop("prompt_cache", None)
        gen_kwargs.pop("prompt_cache_state", None)
        gen_kwargs.pop("input_ids", None)
        # Invalidate cache on incomplete generation to avoid inconsistent state
        if not generation_complete and full_prompt_tokens is not None:
            logger.debug("Cache invalidated: generation did not complete")
            lm.prompt_cache_store.remove(cache_id)
        # Record a failed generation for any in-flight real exception —
        # independent of how far prefill got, so early failures (e.g. prefill
        # OOM, which leaves full_prompt_tokens=None) are still counted. Client
        # disconnects (GeneratorExit) and cancellations are not errors, so they
        # are excluded; a clean early return leaves no exception so is also
        # excluded.
        if not generation_complete:
            exc_type = sys.exc_info()[0]
            if exc_type is not None and not issubclass(
                exc_type, (GeneratorExit, asyncio.CancelledError)
            ):
                try:
                    _metrics.observe_inference(
                        lm.name, surface_var.get(), stats, error=True
                    )
                except Exception:
                    logger.debug("metrics: error observe failed", exc_info=True)
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
            # Pass _audio_temps so temp files are deleted only after the
            # thread exits (deleting them now would race the reader).
            logger.warning(
                "Inference thread still alive after cleanup attempts — "
                "deferring Metal sync and lock release until thread exits"
            )
            await _schedule_deferred_inference_cleanup(stream, audio_temps=_audio_temps)
        else:
            # Normal path — thread exited, safe to clean up and release.
            # Delete temp audio files now that the worker thread is done.
            cleanup_temp_audio(_audio_temps)
            _exit_inference_lock(lm.sync_mode, context="_stream_completion")


async def _full_completion_batched(
    lm: LoadedModel,
    prompt: str | list[int],
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    *,
    use_prompt_cache: bool = False,
    prompt_tokens: list[int] | None = None,
    cache_id: str = "",
    keep_alive: str | None = None,
    thinking_expected: bool = False,
) -> dict:
    """Non-streaming batched completion (plan §4 item 6).

    Internally consumes ``_stream_completion_batched`` (the
    ``collect_stream`` pattern) and aggregates the same result-dict shape
    as ``_full_completion``: stop sequences, prompt-cache round trip and
    metrics all happen inside the stream. Unlike the exclusive
    non-streaming path, ``inference_timeout`` IS enforced — the batch
    worker can drop a sequence at any tick, so cancellation is safe.
    """
    text_parts: list[str] = []
    result: dict = {"text": "", "done": True, "stats": stats}
    async for chunk in _stream_completion_batched(
        lm,
        prompt,
        max_tokens,
        gen_kwargs,
        stats,
        keep_alive=keep_alive,
        prompt_tokens=prompt_tokens,
        use_prompt_cache=use_prompt_cache,
        cache_id=cache_id,
        thinking_expected=thinking_expected,
    ):
        if chunk.get("cache_info"):
            result["cache_read_tokens"] = chunk.get("cache_read_tokens", 0)
            result["cache_creation_tokens"] = chunk.get("cache_creation_tokens", 0)
        elif chunk.get("done"):
            reason = chunk.get("done_reason")
            if reason:
                # Routers map done_reason themselves (timeout → "length");
                # finish_reason mirrors the exclusive path's stop marker.
                result["done_reason"] = reason
                if reason == "stop":
                    result["finish_reason"] = "stop"
        else:
            text_parts.append(chunk.get("text") or "")
    result["text"] = "".join(text_parts)
    return result


async def _full_completion(
    lm: LoadedModel,
    prompt: str | list[int],
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
    audio: list[str] | None = None,
    has_tools: bool = False,
    *,
    use_prompt_cache: bool = False,
    prompt_tokens: list[int] | None = None,
    cache_id: str = "",
    keep_alive: str | None = None,
    grammar_active: bool = False,
    adopt_pin: bool = False,
    messages: list[dict] | None = None,
    tokenizer: Any = None,
    template_kwargs: dict | None = None,
    thinking_expected: bool = False,
    collect_generated_tokens: bool = False,
) -> dict:
    # Continuous batching (plan §4 item 6): eligible non-streaming
    # requests internally consume the batched stream and aggregate —
    # stop sequences stay in gen_kwargs for the stream to handle.
    # The batched path aggregates its internal stream and does not surface
    # per-token ids; a context request (#656) skips it and takes the
    # token-tracked path below so `generated_tokens` can be collected.
    if not collect_generated_tokens and _batch_eligible(
        lm,
        gen_kwargs,
        max_tokens=max_tokens,
        images=images,
        audio=audio,
    ):
        return await _full_completion_batched(
            lm,
            prompt,
            max_tokens,
            gen_kwargs,
            stats,
            use_prompt_cache=use_prompt_cache,
            prompt_tokens=prompt_tokens,
            cache_id=cache_id,
            keep_alive=keep_alive,
            thinking_expected=thinking_expected,
        )

    # inference_timeout is not enforced for non-streaming: the GPU thread
    # cannot be safely cancelled (releasing the lock while Metal is still
    # running causes concurrent command buffer access).  Streaming handles
    # this via CancellableStream.cancel() + drain_and_join().
    # Pop stop sequences before passing gen_kwargs to mlx-lm (unsupported).
    stop_sequences: list[str] | None = gen_kwargs.pop("stop", None)
    # When adopt_pin is True the caller passed an ``lm`` already pinned via
    # ``ensure_loaded(pin=True)``. ``_inference_ref(adopt=True)`` does NOT
    # release the pin — the caller's outer finally is responsible for the
    # single release. This avoids races between an early exception path
    # here and the caller's cleanup.
    async with _inference_locked(lm.inference_queue_timeout, sync_mode=lm.sync_mode):
        with _inference_ref(lm, keep_alive=keep_alive, adopt=adopt_pin):
            # Cache setup must happen under the inference lock so two
            # concurrent requests can't race to read/mutate the same store
            # entry.  The gate at the call site has already excluded VLM
            # and speculative paths (which don't consume ``prompt_cache``).
            cache_read_tokens = 0
            cache_creation_tokens = 0
            full_prompt_tokens: list[int] | None = None
            cache_setup_done = False
            generation_complete = False
            generated_tokens: list[int] = []
            result_dict: dict = {}
            deferred_prefill: Callable[[threading.Event | None], None] | None = None
            try:
                if use_prompt_cache and lm.is_vlm:
                    # VLM: attach an mlx_vlm PromptCacheState. Leave ``prompt``
                    # as the full str — mlx_vlm tokenizes it and reuses the KV
                    # prefix internally (no suffix-only trimming on this path).
                    (
                        cache_read_tokens,
                        cache_creation_tokens,
                    ) = await _setup_vlm_prompt_cache(
                        lm, prompt_tokens, gen_kwargs, cache_id=cache_id
                    )
                    # Leave full_prompt_tokens None: the text-tokenizer count
                    # misses image-patch expansion, and mlx_vlm's GenerationResult
                    # already reports the accurate full prompt size (reused + new)
                    # on both hits and misses — let that engine count stand rather
                    # than overriding stats.prompt_eval_count with an undercount.
                    full_prompt_tokens = None
                    cache_setup_done = True
                elif use_prompt_cache:
                    cs = await _setup_prompt_cache(
                        lm,
                        prompt,
                        gen_kwargs,
                        prompt_tokens=prompt_tokens,
                        cache_id=cache_id,
                        messages=messages,
                        tokenizer=tokenizer,
                        template_kwargs=template_kwargs,
                    )
                    prompt = cs.prompt
                    cache_read_tokens = cs.cache_read_tokens
                    cache_creation_tokens = cs.cache_creation_tokens
                    full_prompt_tokens = cs.full_prompt_tokens
                    cache_setup_done = cs.cache_setup_done
                    deferred_prefill = cs.deferred_prefill

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

                result_dict = await _full_completion_inner(
                    lm,
                    prompt,
                    max_tokens,
                    gen_kwargs,
                    stats,
                    images,
                    audio,
                    has_tools=has_tools,
                    generated_tokens_out=(
                        generated_tokens
                        if (use_prompt_cache or collect_generated_tokens)
                        else None
                    ),
                    grammar_active=grammar_active,
                    deferred_prefill=deferred_prefill,
                    stop_sequences=stop_sequences,
                    thinking_expected=thinking_expected,
                )
                generation_complete = True
                if collect_generated_tokens:
                    # Surface produced token ids for the Ollama continuation
                    # context (#656); a copy so cache-store paths can't mutate
                    # what the router forwards.
                    result_dict["generated_tokens"] = list(generated_tokens)

                # Report the full prompt size to the caller — when the
                # cache path (flat or checkpoint) hands a suffix to mlx-lm,
                # `result.prompt_tokens` (captured into stats inside
                # _full_completion_inner) only counts what mlx-lm actually
                # re-prefilled, not the full request.
                if full_prompt_tokens is not None:
                    stats.prompt_eval_count = len(full_prompt_tokens)

                if cache_setup_done:
                    result_dict["cache_read_tokens"] = cache_read_tokens
                    result_dict["cache_creation_tokens"] = cache_creation_tokens
                    await _store_prompt_cache_after_generation(
                        lm,
                        gen_kwargs,
                        full_prompt_tokens,
                        generated_tokens,
                        stats.eval_count,
                        cache_id,
                    )
                _result = result_dict
            finally:
                # Drop GPU-backed references from gen_kwargs so they can be
                # garbage-collected.  ``prompt_cache`` (text path) is either
                # persisted in the store (success) or released here.
                # ``prompt_cache_state`` (VLM path) is owned by the VLM store,
                # so dropping the gen_kwargs reference just releases a duplicate.
                # ``input_ids`` is legacy (no longer set) — popped defensively.
                gen_kwargs.pop("prompt_cache", None)
                gen_kwargs.pop("prompt_cache_state", None)
                gen_kwargs.pop("input_ids", None)
                if not generation_complete and full_prompt_tokens is not None:
                    logger.debug(
                        "Cache invalidated: non-streaming generation did not complete"
                    )
                    lm.prompt_cache_store.remove(cache_id)
                # Symmetric with the streaming path: record a failed generation
                # for any in-flight real exception. ``_full_completion_inner``
                # records *success* on its normal return, so this fires only when
                # it (or cache setup) raised. Client cancellations are excluded.
                if not generation_complete:
                    exc_type = sys.exc_info()[0]
                    if exc_type is not None and not issubclass(
                        exc_type, (GeneratorExit, asyncio.CancelledError)
                    ):
                        try:
                            _metrics.observe_inference(
                                lm.name, surface_var.get(), stats, error=True
                            )
                        except Exception:
                            logger.debug(
                                "metrics: error observe failed (non-stream)",
                                exc_info=True,
                            )
    if stop_sequences and result_dict:
        text, hit = truncate_at_stop(
            result_dict.get("text", ""),
            stop_sequences,
            thinking_aware=thinking_expected,
        )
        if hit:
            result_dict["finish_reason"] = "stop"
            # The stop sequence is applied post-hoc, so mlx-lm may have run on to
            # max_tokens and set done_reason="length". A stop-sequence hit means
            # the visible generation ended at the stop, so "stop" wins — drop the
            # length marker that routers key on.
            result_dict.pop("done_reason", None)
            # mlx-lm generated (and counted) tokens past the stop sequence; the
            # client only sees the truncated text, so report its token count as
            # eval_count instead of the full max_tokens run. add_special_tokens
            # avoids inflating the count with a BOS that was never generated.
            stats.eval_count = len(
                lm.text_tokenizer.encode(text, add_special_tokens=False)
            )
        result_dict["text"] = text
    return result_dict


async def _full_completion_inner(
    lm: LoadedModel,
    prompt: str | list[int],
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
    audio: list[str] | None = None,
    has_tools: bool = False,
    *,
    generated_tokens_out: list[int] | None = None,
    grammar_active: bool = False,
    deferred_prefill: Callable[[threading.Event | None], None] | None = None,
    stop_sequences: list[str] | None = None,
    thinking_expected: bool = False,
) -> dict:
    audio_paths: list[str] = []
    _audio_temps: list[str] = []

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

        if deferred_prefill is not None:
            # Checkpoint-path prefill: run on this worker thread so prefill
            # and decode share its thread-local generation_stream (#284/#499).
            deferred_prefill(None)

        # Decide speculative use explicitly (mirrors _stream_completion).
        # The disable-on-grammar path is taken when both speculative is
        # loaded and grammar is requested for this request; otherwise
        # fall through to the speculative branch unchanged.
        if (
            lm.is_speculative
            and grammar_active
            and not (lm.is_vlm and (images or audio_paths))
        ):
            logger.warning(
                "Speculative decoding disabled for this request: "
                "grammar-constrained decoding is not yet plumbed through "
                "the speculative path"
            )
            use_speculative = False
        else:
            use_speculative = lm.is_speculative

        if lm.is_vlm and (images or audio_paths):
            if lm.is_speculative:
                logger.debug("speculative decoding skipped: request includes images")
            import mlx_vlm
            from mlx_vlm.generate import (
                generation_stream,
            )  # used by mx.synchronize below

            # Drain stream_generate (not generate): it forwards prompt_cache_state
            # + logits_processors and yields GenerationResult with real token
            # counts. Return (last_result, full_text) so the downstream tuple
            # unpacking captures prompt/generation token counts (#429).
            result = None
            text_parts = []
            for response in mlx_vlm.stream_generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                image=images,
                audio=audio_paths,
                max_tokens=max_tokens,
                **gen_kwargs,
            ):
                text_parts.append(response.text)
                result = response
            if result is not None:
                result = (result, "".join(text_parts))
        elif use_speculative:
            import threading

            from olmlx.engine.speculative_stream import (
                speculative_stream_generate,
            )

            if isinstance(prompt, str):
                # BOS heuristic (see the buffered speculative path / #633):
                # plain ``.encode`` would double the BOS on BOS-prefixed
                # templates, diverging from the non-speculative path.
                prompt_tokens = tokenize_for_cache(lm.text_tokenizer, prompt)
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
                # Collect produced token ids for the Ollama continuation
                # context (#656) — the speculative branch, unlike the mlx-lm
                # branch above, otherwise never populates this and would return
                # a prompt-only context for speculative models.
                if generated_tokens_out is not None:
                    tok_id = getattr(response, "token", None)
                    if tok_id is not None:
                        generated_tokens_out.append(tok_id)
                result = response
            if result is not None:
                result = (result, "".join(text_parts))
            # Speculative decoding does not use mlx_lm's generation_stream,
            # so sync the default stream only.
            mx.synchronize()
            return result
        elif lm.is_vlm:
            import mlx_vlm
            from mlx_vlm.generate import (
                generation_stream,
            )  # used by mx.synchronize below

            result = None
            text_parts = []
            for response in mlx_vlm.stream_generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                image=images,
                audio=audio_paths,
                max_tokens=max_tokens,
                **gen_kwargs,
            ):
                text_parts.append(response.text)
                result = response
            if result is not None:
                result = (result, "".join(text_parts))
        else:
            import mlx_lm

            # Use stream_generate to capture token counts (generate() discards them).
            # Accumulate text segments since each yield is incremental.
            # When prompt caching is active the caller passes a list buffer
            # via generated_tokens_out so _store_prompt_cache_after_generation
            # can persist the produced token IDs alongside the prompt prefix.
            result = None
            text_parts = []
            # Feed a StopScanner so a stop-sequence match halts decoding at the
            # earliest match instead of running on to max_tokens and truncating
            # post-hoc — the latter wastes GPU time under the inference lock and
            # stores post-stop tokens in the prompt cache (#613). The full text
            # (including the stop marker) is kept so the caller's post-hoc
            # truncate_at_stop still trims it and sets finish_reason.
            stop_scanner = (
                StopScanner(stop_sequences, thinking_aware=thinking_expected)
                if stop_sequences
                else None
            )
            mlx_gen = mlx_lm.stream_generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                **gen_kwargs,
            )
            try:
                for response in mlx_gen:
                    text_parts.append(response.text)
                    if generated_tokens_out is not None:
                        tok_id = getattr(response, "token", None)
                        if tok_id is not None:
                            generated_tokens_out.append(tok_id)
                        else:
                            logger.debug(
                                "Skipping token with None ID at generation step %d "
                                "(cache token sequence will be incomplete)",
                                len(generated_tokens_out),
                            )
                    result = response
                    if stop_scanner is not None:
                        _, stop_hit = stop_scanner.feed(response.text or "")
                        if stop_hit:
                            break
            finally:
                # Close the generator explicitly on an early (stop) break so
                # mlx-lm's wired_limit.__exit__ runs its generation-stream sync
                # now, on this worker thread — mirrors CancellableStream._run.
                # Guarded: mlx_lm.stream_generate returns a real generator, but
                # tests may substitute a plain iterator with no close(). The
                # close itself is wrapped like CancellableStream._run — GPU
                # teardown (wired_limit sync) can raise, and an exception here
                # would supersede a real in-flight error from the loop body.
                _close = getattr(mlx_gen, "close", None)
                if callable(_close):
                    try:
                        _close()
                    except Exception:
                        logger.debug(
                            "stream_generate close() failed during teardown",
                            exc_info=True,
                        )
            # Store full text on the result for downstream extraction
            if result is not None:
                result = (result, "".join(text_parts))
            from mlx_lm.generate import (
                generation_stream,
            )  # used by mx.synchronize below

        # Flat-path lazy-state caches (TurboQuant) leave their packed buffers
        # bound to this worker thread's Metal stream (the fetch path returns
        # from a dequant side buffer, so the packed writes never enter the
        # per-token eval). Materialize them here, on the generating worker,
        # before the cache is stored and reused from another worker thread —
        # symmetric with the streaming path's gen_factory finalizer. Skipped by
        # the speculative branch, which returns above and does not build these
        # caches. No-op for plain/Spectral/Shard caches.
        materialize_lazy_cache_state(gen_kwargs.get("prompt_cache"))

        # Sync the generation_stream specifically — mlx_lm/mlx_vlm run GPU
        # work on this module-level stream, not the default stream.  Without
        # this, generate() may return before GPU work is actually done.
        mx.synchronize(generation_stream)
        return result

    # Non-streaming fuses prefill + decode inside the single _generate_sync
    # thread call, so the two phases can't be time-separated here as they are
    # in _stream_completion. The prefill span marks prompt readiness; the decode
    # span wraps the generation work and carries the finalized token counts. Both
    # nest under the entry-point ``inference`` span via the active OTel context.
    try:
        audio_paths, _audio_temps = materialize_audio(audio)
        prompt_token_count = len(prompt) if isinstance(prompt, list) else None
        with _tracing.span(
            "prefill", prompt_tokens=prompt_token_count, **{"gen.stream": False}
        ):
            pass
        with _tracing.span("decode") as _decode_span:
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
                # Non-streaming reports the mlx-lm-prefilled count as prompt_eval_count
                # already, so this matches — passed explicitly for parity with the
                # streaming path's cache-hit handling.
                prefilled_count=getattr(result, "prompt_tokens", None),
            )
            # ttft_ns: prefill + decode are fused in the single _generate_sync
            # thread call here, so surface the measured prefill duration as the
            # time-to-first-token attribute (the prefill span only marks readiness).
            _decode_span.set_attributes(
                {
                    "eval_count": stats.eval_count,
                    "decode_tok_s": _metrics._decode_tps(stats),
                    "ttft_ns": stats.prompt_eval_duration,
                }
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
        # Propagate the stop reason from mlx-lm's final GenerationResponse so the
        # non-streaming routers can report max_tokens truncation. getattr(..., None)
        # is safe across all sub-paths (stream_generate / speculative / mlx_vlm).
        if getattr(result, "finish_reason", None) == "length":
            result_dict["done_reason"] = "length"
        if raw_text is not None:
            result_dict["raw_text"] = raw_text
        if tool_uses:
            result_dict["tool_uses"] = tool_uses
        if thinking:
            result_dict["thinking"] = thinking
        try:
            _metrics.observe_inference(lm.name, surface_var.get(), stats)
        except Exception:
            logger.debug(
                "metrics: observe_inference failed (non-stream)", exc_info=True
            )
        return result_dict
    finally:
        # Clean up any temp audio files created by materialize_audio.
        # The worker thread (_generate_sync) has returned via asyncio.to_thread,
        # so all audio reads are complete before this runs — no race condition.
        cleanup_temp_audio(_audio_temps)


@overload
async def generate_chat(
    manager: ModelManager,
    model_name: str,
    messages: list[dict],
    options: dict | None = ...,
    tools: list[dict] | None = ...,
    *,
    stream: Literal[True],
    keep_alive: int | str | None = ...,
    max_tokens: int = ...,
    cache_id: str = ...,
    enable_thinking: bool | None = ...,
    reasoning_effort: str | None = ...,
    grammar_spec: GrammarSpec | None = ...,
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
    keep_alive: int | str | None = ...,
    max_tokens: int = ...,
    cache_id: str = ...,
    enable_thinking: bool | None = ...,
    reasoning_effort: str | None = ...,
    grammar_spec: GrammarSpec | None = ...,
) -> dict: ...


@overload
async def generate_chat(
    manager: ModelManager,
    model_name: str,
    messages: list[dict],
    options: dict | None = ...,
    tools: list[dict] | None = ...,
    stream: bool = ...,
    keep_alive: int | str | None = ...,
    max_tokens: int = ...,
    cache_id: str = ...,
    enable_thinking: bool | None = ...,
    reasoning_effort: str | None = ...,
    grammar_spec: GrammarSpec | None = ...,
) -> AsyncGenerator[dict, None] | dict: ...


async def generate_chat(
    manager: ModelManager,
    model_name: str,
    messages: list[dict],
    options: dict | None = None,
    tools: list[dict] | None = None,
    stream: bool = True,
    keep_alive: int | str | None = None,
    max_tokens: int = 512,
    cache_id: str = "",
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    grammar_spec: GrammarSpec | None = None,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a chat completion."""
    stats = TimingStats()

    with Timer() as load_timer:
        lm = await manager.ensure_loaded(model_name, keep_alive, pin=True)
    stats.load_duration = load_timer.duration_ns

    # The pin acquired by ``ensure_loaded(pin=True)`` is released exactly
    # once: by the stream wrapper after the generator exits, by the
    # non-stream finally below, or by the outer except below if anything
    # raises before delegation. The outer try must scope from RIGHT AFTER
    # ensure_loaded (PR #394 aider review) so an exception in the chat
    # template / kwargs setup doesn't leak the pin.
    pin_released_or_transferred = False
    try:
        # Per-model default for enable_thinking applies when the request
        # didn't set the flag. Request value, when present, still wins.
        # See issue #400.
        if enable_thinking is None and lm.enable_thinking is not None:
            enable_thinking = lm.enable_thinking

        # Per-model default reasoning level (gpt-oss / Harmony) applies when the
        # caller didn't pass one. An explicit caller value still wins.
        if reasoning_effort is None and lm.reasoning_effort is not None:
            reasoning_effort = lm.reasoning_effort

        images = _extract_images(messages)
        audio = _extract_audio(messages)

        if audio and not _audio_capable(lm):
            raise ValueError(
                f"Model {lm.name!r} cannot accept audio input: it is not an "
                "audio-capable multimodal model. Load a model with an audio "
                "tower (e.g. a Gemma 4 checkpoint)."
            )
        # Reject tools + audio here (out of scope v1) so the contract holds on
        # every VLM template branch, not only the native-tools one: the
        # system-injection branch calls _apply_chat_template_vlm without tools=,
        # which would bypass that function's own audio+tools guard.
        if audio and tools:
            raise ValueError(
                "tools + audio is not supported in this version: combining "
                "native tool calling with audio input is out of scope (#426). "
                "Send the audio without tools, or the tools without audio."
            )

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

        # Rewrite OpenAI-style role="tool" turns for templates that can't render
        # them natively: Gemma-style templates take a tool_responses array;
        # minimal templates (Devstral/Mistral) only accept user/system/assistant
        # and raise otherwise, so tool turns are folded into user-message text.
        if any(m.get("role") == "tool" for m in messages):
            if caps.uses_tool_responses:
                messages = _convert_tool_messages_to_responses(messages)
            elif not caps.handles_tool_role:
                messages = _convert_tool_messages_to_user_text(messages)

        if lm.is_vlm:
            # VLM models must use the VLM generation path for tokenization.
            # Pass tools natively through the template when supported — this
            # produces model-native formatting (e.g. <|tool> tags for Gemma 4)
            # which is far more effective than injecting JSON into the system
            # message.  Fall back to system-message injection for models whose
            # template lacks tool support.
            # Resolve enable_thinking for templates that support it.
            vlm_thinking = enable_thinking if caps.supports_enable_thinking else None
            if tools and caps.supports_tools:
                prompt = _apply_chat_template_vlm(
                    lm.tokenizer,
                    lm.model,
                    messages,
                    images,
                    tools=tools,
                    enable_thinking=vlm_thinking,
                    audio=audio,
                )
                logger.info(
                    "VLM chat prompt with %d tools (native template)", len(tools)
                )
            elif tools:
                vlm_messages = _inject_tools_into_system(list(messages), tools)
                prompt = _apply_chat_template_vlm(
                    lm.tokenizer,
                    lm.model,
                    vlm_messages,
                    images,
                    enable_thinking=vlm_thinking,
                    audio=audio,
                )
                logger.info(
                    "VLM chat prompt with %d tools (injected into system)", len(tools)
                )
            else:
                prompt = _apply_chat_template_vlm(
                    lm.tokenizer,
                    lm.model,
                    messages,
                    images,
                    enable_thinking=vlm_thinking,
                    audio=audio,
                )
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
                reasoning_effort=reasoning_effort,
            )
            if tools:
                logger.info("Chat prompt with %d tools", len(tools))
            logger.debug("Prompt (first 2000 chars): %s", prompt[:2000])
            logger.debug("Prompt (last 2000 chars): %s", prompt[-2000:])

        merged_options = _merge_default_options(lm.default_options, options)
        gen_kwargs = _build_generate_kwargs(merged_options, is_vlm=lm.is_vlm)
        mt = gen_kwargs.pop("max_tokens", max_tokens)

        grammar_active = _install_grammar_processor(
            lm, gen_kwargs, grammar_spec, has_tools=bool(tools)
        )

        # Prompt caching applies to both streaming and non-streaming requests
        # (issue #342).  Disabled in distributed mode because rank 0 processes
        # only suffix tokens on cache hits while workers process the full
        # prompt, causing all_sum call count mismatch and deadlock.  Disabled
        # for speculative regardless of stream mode (issue #346): the
        # speculative decoder owns its own internal target/draft caches and
        # would receive a misaligned suffix-only prompt on a cache hit —
        # ``async_speculative_stream`` does not consume ``gen_kwargs['prompt_cache']``.
        # Cross-request reuse for speculative is instead self-managed by the
        # decoder via its own ``_SpecCacheStore`` (issue #421), driven by the
        # ``SegmentedPrompt`` built on the speculative branch above — so the
        # ``not lm.is_speculative`` gate here stays correct.
        # VLMs (stream and non-stream) take the separate ``vlm_cache_ok`` path
        # below — they reuse KV via mlx_vlm's ``PromptCacheState`` rather than
        # the text store's ``prompt_cache``/checkpoint machinery (#429).
        # Per-model ``prompt_cache`` (set in models.json) overrides the global
        # ``OLMLX_PROMPT_CACHE`` toggle. Surfaced for architectures that hit
        # checkpoint-path bugs (e.g. Qwen3-Coder-Next MoE-quantized GatedDeltaNet
        # targets where chunked prefill crosses expert-routing thresholds) while
        # other models on the same server keep caching enabled.
        effective_prompt_cache = (
            lm.prompt_cache if lm.prompt_cache is not None else settings.prompt_cache
        )
        # VLM now caches on both stream and non-stream via the VLM store path
        # (mlx_vlm PromptCacheState); its KV reuse is independent of the text
        # path's checkpoint/radix machinery. A VLM with its store disabled
        # (vlm_prompt_cache_slots=0) is excluded so prompt_tokens stays None.
        vlm_cache_ok = (
            lm.is_vlm
            and getattr(lm, "vlm_prompt_cache_store", None) is not None
            and lm.vlm_prompt_cache_store.enabled()
        )
        use_prompt_cache = (
            effective_prompt_cache
            and make_prompt_cache is not None
            and not lm.is_distributed
            and not lm.is_speculative
            and (not lm.is_vlm or vlm_cache_ok)
        )
        prompt_tokens = None
        if use_prompt_cache:
            prompt_tokens = tokenize_for_cache(lm.text_tokenizer, prompt)
            # Memory-only peek for debug logging; the authoritative lookup happens
            # inside _stream_completion/_full_completion under the inference lock.
            cached_state = lm.prompt_cache_store.peek(cache_id)
            logger.debug(
                "Prompt cache enabled: %d prompt tokens, existing cache=%s",
                len(prompt_tokens),
                f"{len(cached_state.tokens)} tokens" if cached_state else "none",
            )
        else:
            logger.debug(
                "Prompt cache disabled: setting=%s per_model=%s stream=%s vlm=%s "
                "speculative=%s make_prompt_cache=%s",
                settings.prompt_cache,
                lm.prompt_cache,
                stream,
                lm.is_vlm,
                lm.is_speculative,
                make_prompt_cache is not None,
            )

        # Tell streaming routers whether to wait for a (possibly orphaned, see
        # #307) `</think>` token — shares the rules with `_apply_chat_template`.
        thinking_expected = _resolve_thinking_active(caps, tools, enable_thinking)

        # Build the chat template kwargs used for segmented tokenization in the
        # checkpoint cache path.  These must match the kwargs that produced
        # ``prompt_tokens`` so that ``tokenize_segmented_chat`` produces the
        # same full token sequence and the EOM-boundary split is correctly
        # aligned.  Only applies to text (non-VLM) models; the checkpoint path
        # is never reached for VLMs.
        chat_template_kwargs: dict | None = None
        if not lm.is_vlm:
            chat_template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
            }
            if caps.supports_enable_thinking:
                chat_template_kwargs["enable_thinking"] = _resolve_thinking_active(
                    caps, tools, enable_thinking
                )

        if stream:
            gen = _stream_completion(
                lm,
                prompt,
                mt,
                gen_kwargs,
                stats,
                images,
                audio,
                use_prompt_cache=use_prompt_cache,
                prompt_tokens=prompt_tokens,
                cache_id=cache_id,
                keep_alive=keep_alive,
                grammar_active=grammar_active,
                adopt_pin=True,
                messages=messages,
                tokenizer=lm.tokenizer,
                template_kwargs=chat_template_kwargs,
                thinking_expected=thinking_expected,
            )
            # Ownership of the pin transfers to the wrapper; its finally
            # releases on any generator exit. Mark the flag for the outer
            # except so it doesn't double-release.
            pin_released_or_transferred = True
            streamed = _release_pin_after_gen(gen, lm)
            if _tracing.enabled():
                streamed = _trace_inference_gen(streamed, lm)
            return _prepend_meta(
                streamed,
                {"thinking_expected": thinking_expected},
            )
        else:
            try:
                with _tracing.span(
                    "inference",
                    model=lm.name,
                    surface=surface_var.get(),
                    strategy=_strategy_label(lm),
                    **{"gen.stream": False},
                ):
                    result = await _full_completion(
                        lm,
                        prompt,
                        mt,
                        gen_kwargs,
                        stats,
                        images,
                        audio,
                        has_tools=bool(tools),
                        use_prompt_cache=use_prompt_cache,
                        prompt_tokens=prompt_tokens,
                        cache_id=cache_id,
                        keep_alive=keep_alive,
                        grammar_active=grammar_active,
                        adopt_pin=True,
                        messages=messages,
                        tokenizer=lm.tokenizer,
                        template_kwargs=chat_template_kwargs,
                        thinking_expected=thinking_expected,
                    )
                # Mirror the streaming meta chunk so non-streaming routers
                # can gate orphan `</think>` handling on the same signal
                # (issue #307).
                result["thinking_expected"] = thinking_expected
                return result
            finally:
                if not pin_released_or_transferred:
                    lm.release_ref()
                    pin_released_or_transferred = True
    except BaseException:
        if not pin_released_or_transferred:
            lm.release_ref()
        raise


# Streaming routers consult this when the engine signals
# `thinking_expected=True` (issue #307): how many characters of leading
# output to buffer while waiting for an orphan `</think>` before giving
# up and emitting the held text as content.  Tuned for the tension
# between two concerns:
#   * Qwen3.5/3.6's orphan-thinking preamble for reasoning tasks is
#     typically a few hundred characters before `</think>` arrives.
#   * For thinking-capable models that produce a direct answer (no
#     thinking block at all), every byte of this buffer adds TTFB
#     latency — at ~1000 chars/s the worst case is roughly one second
#     before any text reaches the client.  Keep-alive pings cover the
#     wait but it's still a visible regression vs streaming a
#     non-thinking model.  1024 was picked over the reviewer-suggested
#     512 to keep margin for longer Qwen3.5 reasoning traces (the
#     observed in-issue example is ~280 chars but production traces
#     can exceed that for non-trivial prompts); revisit if real-world
#     direct-answer TTFB becomes a complaint.
#
# KNOWN LIMITATION: when a thinking preamble exceeds this limit before
# `</think>` arrives, the streaming routers fall through to text /
# passthrough state and emit the buffered content as visible text.  The
# `</think>` that arrives later is then surfaced as a literal token in
# the visible content (no retroactive reclassification is possible in a
# streaming path).  Practical impact: a complex multi-step reasoning
# trace that runs longer than ~1024 characters before closing its
# thinking block will leak the preamble into the response.  Mitigations
# the operator can apply: bump this constant (trades TTFB for
# correctness margin), use the non-streaming endpoint (always re-parses
# the full text), or use a model that emits the standard `<think>...`
# opener (which is detected at any position).
INIT_ORPHAN_DETECT_LIMIT = 1024


async def _prepend_meta(
    stream: AsyncGenerator[dict, None],
    meta: dict,
) -> AsyncGenerator[dict, None]:
    """Yield ``meta`` as the first chunk, then forward ``stream``.

    Used so routers learn streaming-level metadata (e.g. whether thinking is
    expected — issue #307) before any text chunks arrive.
    """
    try:
        yield meta
        async for chunk in stream:
            yield chunk
    finally:
        await stream.aclose()


async def _augment_stream_with_context(
    stream: AsyncGenerator[dict, None],
    input_tokens: list[int],
) -> AsyncGenerator[dict, None]:
    """Attach the Ollama continuation ``context`` to the terminal chunk (#656).

    The engine's done chunk carries the generated token ids under
    ``generated_tokens`` (present because generation was asked to collect
    them); replace that with ``context`` = the prefilled *input_tokens*
    followed by those generated ids, so the router can forward it verbatim.
    """
    try:
        async for chunk in stream:
            if chunk.get("done"):
                generated = chunk.pop("generated_tokens", [])
                chunk["context"] = input_tokens + generated
            yield chunk
    finally:
        await stream.aclose()


async def generate_embeddings(
    manager: ModelManager,
    model_name: str,
    texts: list[str],
    keep_alive: int | str | None = None,
) -> tuple[list[list[float]], int]:
    """Generate embeddings using the model's hidden states or embed_tokens layer.

    Returns ``(embeddings, total_tokens)`` where ``total_tokens`` is the summed
    token count across all inputs (for OpenAI ``usage.prompt_tokens``).
    """
    lm = await manager.ensure_loaded(model_name, keep_alive, pin=True)

    try:
        async with _inference_locked(
            lm.inference_queue_timeout, sync_mode=lm.sync_mode
        ):
            with (
                _tracing.span(
                    "inference",
                    model=lm.name,
                    surface=surface_var.get(),
                    strategy="none",
                ),
                _inference_ref(lm, keep_alive=keep_alive, adopt=True),
            ):
                embeddings = []
                total_tokens = 0

                tokenizer = lm.text_tokenizer

                # Check if model has a static embedding layer we can use directly
                embed_layer = None
                model_inner = getattr(lm.model, "model", lm.model)
                if hasattr(model_inner, "embed_tokens"):
                    embed_layer = model_inner.embed_tokens

                for text in texts:
                    tokens = tokenizer.encode(text)
                    total_tokens += len(tokens)
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
                return embeddings, total_tokens
    finally:
        lm.release_ref()


def _score_pairs(
    model,
    tokenizer,
    query: str,
    documents: list[str],
    *,
    max_tokens_per_doc: int,
    batch_size: int = 32,
) -> list[float]:
    """Tokenize (query, doc) pairs, batch-score, sigmoid -> [0,1] scores."""
    import numpy as np

    model_max = int(getattr(tokenizer, "model_max_length", 512) or 512)
    # transformers sometimes reports a sentinel (very large) model_max_length.
    if model_max > 100_000:
        model_max = 512
    max_len = min(max_tokens_per_doc, model_max)

    scores: list[float] = []
    for start in range(0, len(documents), batch_size):
        chunk = documents[start : start + batch_size]
        enc = tokenizer(
            [query] * len(chunk),
            chunk,
            truncation="only_second",
            max_length=max_len,
            padding=True,
            return_tensors="np",
        )
        input_ids = mx.array(np.asarray(enc["input_ids"]).astype(np.int32))
        attn = mx.array(np.asarray(enc["attention_mask"]).astype(np.int32))
        logits = model(input_ids, attn)  # [b, 1] (num_labels == 1)
        # Single relevance logit -> probability. Assumes num_labels == 1
        # (enforced at load by load_cross_encoder); column 0 is the score.
        probs = mx.sigmoid(logits[:, 0])
        mx.eval(probs)
        # probs is 1-D (one score per doc) so tolist() is a list; the isinstance
        # guard also narrows mlx's union return type for the type checker.
        batch_scores = probs.tolist()
        if not isinstance(batch_scores, list):
            batch_scores = [batch_scores]
        scores.extend(float(x) for x in batch_scores)
    return scores


def _build_rerank_results(
    *,
    scores: list[float],
    documents: list[str],
    top_n: int | None,
    return_documents: bool,
) -> list[dict]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if top_n is not None:
        order = order[: min(top_n, len(order))]
    results: list[dict] = []
    for idx in order:
        item: dict = {"index": idx, "relevance_score": scores[idx]}
        if return_documents:
            item["document"] = documents[idx]
        results.append(item)
    return results


async def generate_rerank(
    manager: ModelManager,
    model_name: str,
    query: str,
    documents: list[str],
    *,
    top_n: int | None = None,
    max_tokens_per_doc: int = 4096,
    return_documents: bool = False,
    keep_alive: int | str | None = None,
) -> dict:
    """Score documents against a query with a cross-encoder reranker (#369)."""
    lm = await manager.ensure_loaded(model_name, keep_alive, pin=True)
    try:
        if not getattr(lm, "is_reranker", False):
            raise ValueError(
                f"Model '{model_name}' is not a reranker. /v1/rerank requires "
                "an XLM-RoBERTa cross-encoder (e.g. bge-reranker-v2-m3)."
            )
        async with _inference_locked(
            lm.inference_queue_timeout, sync_mode=lm.sync_mode
        ):
            with (
                _tracing.span(
                    "inference",
                    model=lm.name,
                    surface=surface_var.get(),
                    strategy="none",
                ),
                _inference_ref(lm, keep_alive=keep_alive, adopt=True),
            ):
                scores = _score_pairs(
                    lm.model,
                    lm.text_tokenizer,
                    query,
                    documents,
                    max_tokens_per_doc=max_tokens_per_doc,
                )
                results = _build_rerank_results(
                    scores=scores,
                    documents=documents,
                    top_n=top_n,
                    return_documents=return_documents,
                )
                # Load-bearing when sync_mode="none": same rationale as
                # generate_embeddings — this is the only Metal barrier before
                # the lock is released. Suppress+log so caller still gets
                # results; Metal error surfaces on the next inference.
                try:
                    mx.synchronize()
                except Exception:
                    logger.warning(
                        "rerank post-compute sync failed — next inference will crash",
                        exc_info=True,
                    )
                return {"results": results}
    finally:
        lm.release_ref()


async def generate_transcription(
    manager: ModelManager,
    model_name: str,
    audio_path: str,
    *,
    language: str | None = None,
    prompt: str | None = None,
    temperature: float = 0.0,
    word_timestamps: bool = False,
    keep_alive: str | None = None,
) -> dict:
    """Transcribe an audio file with a whisper model managed by ModelManager.

    Loads (or reuses) the whisper model via ``ensure_loaded``, injects it into
    mlx_whisper's module-level ``ModelHolder`` so ``transcribe()`` reuses the
    managed model instead of loading its own, and runs the (synchronous)
    transcription in a worker thread. Returns the raw mlx-whisper result dict
    (``text``, ``segments``, ``language``). Issue #366.
    """
    import importlib

    # NOTE: ``import mlx_whisper.transcribe as ...`` binds to the *function*
    # ``transcribe``, not the submodule — mlx_whisper's __init__ does
    # ``from .transcribe import transcribe`` which shadows the submodule name on
    # the package. Fetch the genuine submodule object (the one unittest.mock's
    # patch() targets) via importlib so ModelHolder injection lands correctly.
    whisper_transcribe = importlib.import_module("mlx_whisper.transcribe")

    lm = await manager.ensure_loaded(model_name, keep_alive, pin=True)

    try:
        # Reject non-whisper models with a clear error (issue #366). Without this,
        # a text/VLM model name would load fine, get injected into mlx_whisper's
        # ModelHolder, and produce a cryptic failure inside transcribe(). Raising
        # ValueError surfaces as HTTP 400 via the app-level handler.
        if not lm.is_whisper:
            raise ValueError(
                f"Model '{model_name}' is not a Whisper model. "
                "/v1/audio/transcriptions requires a whisper-class model "
                "(e.g. whisper-turbo or an mlx-community/whisper-* repo)."
            )

        # Resolve the on-disk path used as path_or_hf_repo so the ModelHolder
        # cache key matches what we inject.
        if manager.store is not None:
            load_path = str(manager.store.local_path(lm.hf_path))
        else:
            load_path = lm.hf_path

        async with _inference_locked(
            lm.inference_queue_timeout, sync_mode=lm.sync_mode
        ):
            with (
                _tracing.span(
                    "inference",
                    model=lm.name,
                    surface=surface_var.get(),
                    strategy="none",
                ),
                _inference_ref(lm, adopt=True),
            ):
                # Inject the managed model so transcribe() reuses it. Safe because
                # _inference_locked serializes inference (no ModelHolder race).
                whisper_transcribe.ModelHolder.model = lm.model
                whisper_transcribe.ModelHolder.model_path = load_path

                def _run() -> dict:
                    try:
                        return whisper_transcribe.transcribe(
                            audio_path,
                            path_or_hf_repo=load_path,
                            language=language,
                            initial_prompt=prompt,
                            temperature=temperature,
                            word_timestamps=word_timestamps,
                        )
                    except FileNotFoundError as exc:
                        raise ValueError(
                            "Audio decoding failed: ffmpeg was not found on PATH. "
                            "Install ffmpeg (e.g. `brew install ffmpeg`) to use "
                            "/v1/audio/transcriptions."
                        ) from exc
                    except RuntimeError as exc:
                        # mlx_whisper.audio.load_audio raises RuntimeError on a
                        # failed ffmpeg decode (bad/corrupt/unsupported file).
                        raise ValueError(f"Audio decoding failed: {exc}") from exc

                return await asyncio.to_thread(_run)
    finally:
        lm.release_ref()


# Max TTS audio segments buffered between the mlx-audio worker thread and the
# streaming consumer. Bounds peak memory if the client stalls (see the
# semaphore backpressure in generate_speech).
_TTS_QUEUE_MAX_SEGMENTS = 4


async def generate_speech(
    manager: ModelManager,
    model_name: str,
    text: str,
    *,
    voice: str,
    speed: float = 1.0,
    keep_alive: str | None = None,
):
    """Stream TTS audio segments for a managed mlx-audio model (issue #367).

    Async generator yielding 1-D float32 numpy arrays (24 kHz mono), one per
    mlx-audio segment, under the serialized inference lock. mlx-audio's
    ``model.generate`` is a *synchronous* generator; we run the whole loop in
    one worker thread (keeping all MLX work on a single thread, per the #284
    stream hazards) and bridge segments to the event loop via a queue.

    Raises ``ValueError`` (-> HTTP 400) if the model is not a TTS model.
    """
    lm = await manager.ensure_loaded(model_name, keep_alive, pin=True)
    try:
        if not lm.is_tts:
            raise ValueError(
                f"Model '{model_name}' is not a TTS model. "
                "/v1/audio/speech requires a TTS model (e.g. a Kokoro repo "
                "such as prince-canuma/Kokoro-82M)."
            )

        async with _inference_locked(
            lm.inference_queue_timeout, sync_mode=lm.sync_mode
        ):
            with (
                _tracing.span(
                    "inference",
                    model=lm.name,
                    surface=surface_var.get(),
                    strategy="none",
                ),
                _inference_ref(lm, adopt=True),
            ):
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()
                stop = threading.Event()
                sentinel = object()
                # Bound in-flight segments so a slow/stalled consumer (or
                # ffmpeg encoder) can't make the worker buffer the whole
                # utterance in memory. The worker blocks on a free slot before
                # producing; the consumer releases one per segment taken.
                slots = threading.Semaphore(_TTS_QUEUE_MAX_SEGMENTS)

                def _worker() -> None:
                    try:
                        for result in lm.model.generate(text, voice=voice, speed=speed):
                            if stop.is_set():
                                break
                            slots.acquire()
                            if stop.is_set():  # woken by teardown, not a free slot
                                break
                            # np.asarray forces eval on THIS thread, keeping
                            # the MLX graph materialization off the loop.
                            audio = np.asarray(result.audio, dtype=np.float32)
                            loop.call_soon_threadsafe(queue.put_nowait, audio)
                        loop.call_soon_threadsafe(queue.put_nowait, sentinel)
                    except Exception as exc:  # noqa: BLE001 - re-raised on loop
                        loop.call_soon_threadsafe(queue.put_nowait, exc)

                worker = asyncio.create_task(asyncio.to_thread(_worker))
                try:
                    while True:
                        item = await queue.get()
                        if item is sentinel:
                            break
                        if isinstance(item, Exception):
                            raise item
                        slots.release()  # free the slot this segment occupied
                        yield item
                finally:
                    stop.set()
                    slots.release()  # unblock a worker parked in acquire()
                    await worker
    finally:
        lm.release_ref()
