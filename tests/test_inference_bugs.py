"""Tests for inference engine bug fixes (#118, #119, #120, #123, #124, #125)."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import olmlx.engine.inference as _inf_mod
from olmlx.engine.inference import (
    _estimate_kv_cache_bytes,
    _inference_ref,
)
from olmlx.engine.model_manager import LoadedModel


# ---------------------------------------------------------------------------
# Bug #118: active_refs race condition
# ---------------------------------------------------------------------------
class TestActiveRefsLock:
    """active_refs increments/decrements must be protected by a lock."""

    def test_inference_ref_uses_lock(self):
        """_inference_ref should use a threading.Lock, not bare +=/-=."""
        # Verify LoadedModel has _active_refs_lock
        real_lm = LoadedModel(
            name="test",
            hf_path="test/test",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        assert hasattr(real_lm, "_active_refs_lock"), (
            "LoadedModel must have _active_refs_lock (Bug #118)"
        )
        assert isinstance(real_lm._active_refs_lock, type(threading.Lock()))

        # Verify _inference_ref works with the lock
        with patch("olmlx.engine.inference.parse_keep_alive", return_value=300.0):
            with _inference_ref(real_lm):
                assert real_lm.active_refs == 1
        assert real_lm.active_refs == 0

    def test_lock_excluded_from_equality_and_repr(self):
        """_active_refs_lock must not affect equality or repr."""
        from dataclasses import fields

        lock_field = next(
            f for f in fields(LoadedModel) if f.name == "_active_refs_lock"
        )
        assert lock_field.compare is False, "_active_refs_lock must have compare=False"
        assert lock_field.repr is False, "_active_refs_lock must have repr=False"
        # Lock should not appear in repr
        lm = LoadedModel(
            name="test",
            hf_path="test/test",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        assert "_active_refs_lock" not in repr(lm)

    def test_concurrent_inference_ref_no_race(self):
        """Concurrent _inference_ref calls must not lose increments/decrements."""
        real_lm = LoadedModel(
            name="test",
            hf_path="test/test",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        errors = []
        barrier = threading.Barrier(10)

        def worker():
            try:
                barrier.wait(timeout=5)
                with patch(
                    "olmlx.engine.inference.parse_keep_alive", return_value=300.0
                ):
                    with _inference_ref(real_lm):
                        time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors in concurrent inference_ref: {errors}"
        assert real_lm.active_refs == 0, (
            f"active_refs should be 0 after all refs released, got {real_lm.active_refs}"
        )


# ---------------------------------------------------------------------------
# Bug #124: Token count and embeddings skip _inference_ref
# ---------------------------------------------------------------------------
class TestTokenCountUsesInferenceRef:
    """Anthropic token count endpoint must use _inference_ref, not manual active_refs."""

    @pytest.mark.asyncio
    async def test_anthropic_count_tokens_uses_inference_ref(self):
        """Token count endpoint should use _inference_ref context manager."""
        import olmlx.routers.anthropic as anthropic_mod

        # Check that the source code does NOT contain manual active_refs manipulation
        import inspect

        source = inspect.getsource(anthropic_mod.anthropic_count_tokens)
        assert "lm.active_refs += 1" not in source, (
            "Token count endpoint must use _inference_ref, not manual active_refs += 1 (Bug #124)"
        )
        assert "lm.active_refs -= 1" not in source, (
            "Token count endpoint must use _inference_ref, not manual active_refs -= 1 (Bug #124)"
        )


class TestEmbeddingsUseInferenceRef:
    """generate_embeddings must use _inference_ref to prevent model eviction."""

    @pytest.mark.asyncio
    async def test_embeddings_uses_inference_ref(self):
        """generate_embeddings should wrap with _inference_ref."""
        import inspect

        source = inspect.getsource(_inf_mod.generate_embeddings)
        assert "_inference_ref" in source, (
            "generate_embeddings must use _inference_ref to prevent eviction (Bug #124)"
        )


# ---------------------------------------------------------------------------
# Bug #119: deferred cleanup task global state race
# ---------------------------------------------------------------------------
class TestDeferredCleanupLock:
    """_schedule_deferred_inference_cleanup and _await_deferred_cleanup must be synchronized."""

    def test_deferred_cleanup_lock_exists(self):
        """Module should have a deferred cleanup lock getter for synchronization."""
        assert hasattr(_inf_mod, "_get_deferred_cleanup_lock"), (
            "Module must have _get_deferred_cleanup_lock (Bug #119)"
        )

    @pytest.mark.asyncio
    async def test_schedule_deferred_cleanup_is_async(self):
        """_schedule_deferred_inference_cleanup should be async to use the lock."""
        import inspect

        assert inspect.iscoroutinefunction(
            _inf_mod._schedule_deferred_inference_cleanup
        ), (
            "_schedule_deferred_inference_cleanup must be async to use asyncio.Lock (Bug #119)"
        )

    @pytest.mark.asyncio
    async def test_await_deferred_cleanup_uses_lock(self):
        """_await_deferred_cleanup should acquire _deferred_cleanup_lock."""
        import inspect

        source = inspect.getsource(_inf_mod._await_deferred_cleanup)
        assert "_get_deferred_cleanup_lock" in source, (
            "_await_deferred_cleanup must use _get_deferred_cleanup_lock (Bug #119)"
        )


# ---------------------------------------------------------------------------
# Bug #243: _deferred_cleanup_lock cached globally — breaks across event loops
# ---------------------------------------------------------------------------
class TestDeferredCleanupLockPerLoop:
    """_get_deferred_cleanup_lock must return a lock bound to the running loop.

    The multi-loop test is deliberately ``def``, not ``async def`` — its whole
    point is to exercise two separate ``asyncio.new_event_loop()`` instances,
    which cannot be done from inside a single running loop.  Each test pops
    its own entries from the module dicts explicitly: the loop locals stay
    alive until the test returns, and WeakKeyDictionary only collects entries
    once the key goes out of scope, so ``loop.close()`` alone isn't sufficient.
    """

    def test_separate_locks_for_separate_loops(self):
        """A fresh event loop must receive a lock bound to itself, not a stale one."""
        import asyncio

        loop_a = asyncio.new_event_loop()
        try:
            lock_a = loop_a.run_until_complete(self._get_lock())
        finally:
            # Explicit pop so loop_a's entry doesn't persist for the rest of
            # this test while ``loop_a`` is still in scope (WeakKeyDictionary
            # holds weak refs to *keys*; a live local variable keeps the key
            # alive).  Matches ``test_stale_locked_lock_not_inherited`` and
            # ``test_separate_tasks_for_separate_loops``.
            _inf_mod._deferred_cleanup_locks.pop(loop_a, None)
            loop_a.close()

        loop_b = asyncio.new_event_loop()
        try:
            lock_b = loop_b.run_until_complete(self._get_lock())
            # The acquire must succeed on loop B's own lock; if we leaked
            # loop A's lock, this would raise "attached to a different loop".
            loop_b.run_until_complete(self._acquire_and_release(lock_b))
        finally:
            _inf_mod._deferred_cleanup_locks.pop(loop_b, None)
            loop_b.close()

        assert lock_a is not lock_b, (
            "Each event loop must receive its own lock (Bug #243)"
        )

    @staticmethod
    async def _get_lock():
        return _inf_mod._get_deferred_cleanup_lock()

    @staticmethod
    async def _acquire_and_release(lock):
        async with lock:
            pass

    @pytest.mark.asyncio
    async def test_same_loop_returns_same_lock(self):
        """Within a single event loop, repeated calls must return the same lock."""
        import asyncio

        try:
            lock1 = _inf_mod._get_deferred_cleanup_lock()
            lock2 = _inf_mod._get_deferred_cleanup_lock()
            assert lock1 is lock2, (
                "Repeated calls on the same loop must return the same lock"
            )
        finally:
            # Explicit cleanup matches the sibling tests in this class;
            # the autouse ``_reset_inference_state`` fixture would also
            # handle this, but being explicit keeps the pattern uniform.
            _inf_mod._deferred_cleanup_locks.pop(asyncio.get_running_loop(), None)

    def test_separate_tasks_for_separate_loops(self):
        """A task registered on loop A must be invisible to loop B.

        ``_await_deferred_cleanup`` only observes the calling loop's task.
        If ``_deferred_cleanup_tasks`` were a single global like pre-fix,
        loop B would try to ``asyncio.wait`` on loop A's foreign task and
        raise ``RuntimeError: Task is attached to a different loop``.
        """
        import asyncio

        async def register_task_and_abandon():
            async def never():
                await asyncio.sleep(999)

            _inf_mod._deferred_cleanup_tasks[asyncio.get_running_loop()] = (
                asyncio.create_task(never())
            )

        loop_a = asyncio.new_event_loop()
        try:
            loop_a.run_until_complete(register_task_and_abandon())
        finally:
            # Drain the pending ``never()`` task to avoid leaking it on a
            # closed loop (ResourceWarning under Py 3.12+ filterwarnings=error).
            pending = asyncio.all_tasks(loop_a)
            for t in pending:
                t.cancel()
            if pending:
                loop_a.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            _inf_mod._deferred_cleanup_tasks.pop(loop_a, None)
            loop_a.close()

        async def await_cleanup_on_fresh_loop():
            # Loop B has no task registered.  _await_deferred_cleanup must
            # return immediately (task for this loop is None) rather than
            # touching loop A's orphaned task.
            await _inf_mod._await_deferred_cleanup()

        loop_b = asyncio.new_event_loop()
        try:
            loop_b.run_until_complete(
                asyncio.wait_for(await_cleanup_on_fresh_loop(), timeout=0.5)
            )
        finally:
            # ``_await_deferred_cleanup`` called ``_get_deferred_cleanup_lock``
            # which created an entry for loop_b.  Pop explicitly to keep the
            # class-wide cleanup pattern uniform.
            _inf_mod._deferred_cleanup_locks.pop(loop_b, None)
            loop_b.close()

    def test_stale_locked_lock_not_inherited(self):
        """A lock acquired on a closed loop must not block a fresh loop.

        This is the actual Bug #243 failure mode: if loop A acquires the
        deferred cleanup lock and closes without releasing it, a pre-fix
        module would cache that locked instance and loop B would deadlock
        on its next acquire.  The fix keys locks by loop, so loop B gets
        a fresh, unlocked lock.
        """
        import asyncio

        async def acquire_no_release():
            lock = _inf_mod._get_deferred_cleanup_lock()
            await lock.acquire()  # intentionally never released

        loop_a = asyncio.new_event_loop()
        try:
            loop_a.run_until_complete(acquire_no_release())
        finally:
            # Explicit pop for consistency with
            # ``test_separate_tasks_for_separate_loops``; WeakKeyDictionary
            # would also GC the entry when loop_a is collected, but being
            # explicit avoids a latent dependency on GC timing.
            _inf_mod._deferred_cleanup_locks.pop(loop_a, None)
            loop_a.close()  # closed with the lock still "held"

        async def acquire_with_timeout():
            lock = _inf_mod._get_deferred_cleanup_lock()
            # Pre-fix: inherits loop A's locked instance → times out.
            # Post-fix: loop B gets a fresh unlocked lock → returns immediately.
            await asyncio.wait_for(lock.acquire(), timeout=0.5)
            lock.release()

        loop_b = asyncio.new_event_loop()
        try:
            loop_b.run_until_complete(acquire_with_timeout())
        finally:
            _inf_mod._deferred_cleanup_locks.pop(loop_b, None)
            loop_b.close()


# ---------------------------------------------------------------------------
# Bug #120: GPU memory not freed on client disconnect during long prefill
# ---------------------------------------------------------------------------
class TestDrainAndJoinBlocksNewInference:
    """When drain_and_join times out with thread alive, new inference must be blocked."""

    @pytest.mark.asyncio
    async def test_kv_cache_eviction_calls_synchronize(self):
        """After KV cache eviction mx.clear_cache(), mx.synchronize() must follow."""
        import inspect

        # Check both _stream_completion and extracted helpers for eviction sites
        sources = [
            ("_stream_completion", inspect.getsource(_inf_mod._stream_completion)),
            ("_setup_prompt_cache", inspect.getsource(_inf_mod._setup_prompt_cache)),
            (
                "_kv_cache_preflight_check",
                inspect.getsource(_inf_mod._kv_cache_preflight_check),
            ),
        ]
        total_eviction_sites = 0
        for func_name, source in sources:
            lines = source.split("\n")
            eviction_indices = [
                i
                for i, line in enumerate(lines)
                if "evict_all_to_disk" in line and not line.lstrip().startswith("#")
            ]
            total_eviction_sites += len(eviction_indices)
            for idx, start in enumerate(eviction_indices):
                end = (
                    eviction_indices[idx + 1]
                    if idx + 1 < len(eviction_indices)
                    else len(lines)
                )
                block = "\n".join(lines[start:end])
                if "clear_cache" in block:
                    assert "synchronize" in block or "_safe_sync" in block, (
                        f"mx.synchronize() or _safe_sync() must follow mx.clear_cache() "
                        f"in {func_name} eviction path at source line {start} (Bug #120)"
                    )
        assert total_eviction_sites >= 1, (
            "Expected at least one evict_all_to_disk call across completion functions"
        )

    def test_eviction_uses_safe_sync_not_lock_boundary_sync(self):
        """Eviction and deferred-cleanup paths must use _safe_sync (unconditional)
        — not _lock_boundary_sync.

        _lock_boundary_sync honors per-model/global sync_mode and can be
        'none', which would silently skip the post-eviction sync needed to
        reclaim freed Metal buffers (Bug #120). The deferred cleanup path
        holds the inference lock while the worker thread may still be using
        the GPU and *must* sync before releasing the lock. Guard against a
        future edit that unifies the call sites.

        Uses bytecode-level name introspection rather than source text so
        nested helpers, refactors, or docstring references to the names
        don't produce false positives/negatives.
        """

        def names_referenced(fn):
            """Return the set of names referenced in fn's bytecode, including
            any nested (closure / inner-function) code objects.

            Note: CPython-specific. ``co_names`` contains names used by
            LOAD_GLOBAL / LOAD_ATTR bytecode instructions — it catches
            direct references like ``_safe_sync(...)`` but not aliases
            (e.g. ``sync_fn = _safe_sync; sync_fn(...)``). Good enough
            for the patterns this guard is intended to catch.
            """
            seen = set()
            stack = [fn.__code__]
            while stack:
                code = stack.pop()
                seen.update(code.co_names)
                for const in code.co_consts:
                    if hasattr(const, "co_names"):
                        stack.append(const)
            return seen

        funcs = [
            ("_setup_prompt_cache", _inf_mod._setup_prompt_cache),
            ("_kv_cache_preflight_check", _inf_mod._kv_cache_preflight_check),
            (
                "_schedule_deferred_inference_cleanup",
                _inf_mod._schedule_deferred_inference_cleanup,
            ),
        ]
        for func_name, fn in funcs:
            names = names_referenced(fn)
            assert "_safe_sync" in names, (
                f"{func_name}: must reference _safe_sync (Bug #120 / "
                f"deferred-cleanup correctness)"
            )
            assert "_lock_boundary_sync" not in names, (
                f"{func_name}: must NOT reference _lock_boundary_sync — "
                f"that helper honors sync_mode='none' and would silently skip "
                f"the Metal sync needed here"
            )

    @pytest.mark.asyncio
    async def test_eviction_still_syncs_under_sync_mode_none(self, mock_manager):
        """Integration guard: under lm.sync_mode='none' (where lock-boundary
        syncs are skipped), the memory-pressure eviction path must still
        call mx.synchronize() via _safe_sync. This complements the static
        bytecode test by exercising the code path functionally — catches
        regressions like aliasing or indirect dispatch that bytecode
        introspection would miss.
        """
        from unittest.mock import AsyncMock, patch
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.sync_mode = "none"
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30])
        lm.prompt_cache_store.set(
            "", CachedPromptState(tokens=[10, 20, 30], cache=[MagicMock()])
        )

        # Minimal stream so generate_chat returns.
        from olmlx.utils.streaming import CancellableStream, StreamToken

        mock_stream = MagicMock(spec=CancellableStream)
        mock_stream.drain_and_join = AsyncMock()
        mock_stream._thread = None
        token_iter = iter(
            [
                StreamToken(
                    text="hi",
                    token=1,
                    prompt_tokens=3,
                    generation_tokens=1,
                    prompt_tps=100.0,
                    generation_tps=50.0,
                )
            ]
        )

        async def anext_impl():
            try:
                return next(token_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch("olmlx.engine.inference.async_mlx_stream", return_value=mock_stream),
            patch(
                "olmlx.engine.inference.make_prompt_cache",
                MagicMock(return_value=[MagicMock()]),
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch("olmlx.utils.memory.is_memory_pressure_high", return_value=True),
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 32768
            mock_settings.default_keep_alive = "5m"
            mock_settings.inference_timeout = None
            mock_settings.memory_limit_fraction = 0.9
            mock_settings.sync_mode = "full"  # per-model "none" still wins
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for _ in gen:
                pass

        # Eviction must have run (memory pressure → clear_cache call)
        mock_mx.clear_cache.assert_called()
        # And despite sync_mode="none" skipping the lock-boundary sync, the
        # eviction path's _safe_sync() must still have produced a
        # synchronize() call.
        assert mock_mx.synchronize.call_count >= 1, (
            "eviction path under sync_mode='none' must still sync (Bug #120)"
        )


# ---------------------------------------------------------------------------
# Bug #123: Prompt cache state corruption on mid-stream disconnect
# ---------------------------------------------------------------------------
class TestPromptCacheRemoveBeforeMutation:
    """Cache must be removed from store before mutation to avoid corruption (Bug #123)."""

    @pytest.mark.asyncio
    async def test_cache_removed_from_store_before_mutation(self):
        """Cache setup must remove cache from store before trimming/passing to generator."""
        import inspect

        # Bug #123 fix now lives in _setup_prompt_cache (extracted helper)
        source = inspect.getsource(_inf_mod._setup_prompt_cache)
        assert "prompt_cache_store.remove" in source, (
            "Prompt cache must be removed from store before mutation (Bug #123)"
        )

    def test_finally_block_removes_cache_on_incomplete_generation(self):
        """The finally block in _stream_completion must call remove on incomplete generation."""
        import inspect

        source = inspect.getsource(_inf_mod._stream_completion)
        # The finally block should invalidate cache when generation_complete is False
        assert "not generation_complete" in source, (
            "_stream_completion must check generation_complete in finally block"
        )
        assert "prompt_cache_store.remove" in source, (
            "_stream_completion must remove cache on incomplete generation"
        )


# ---------------------------------------------------------------------------
# Bug #125: KV cache memory estimate lacks safety margin
# ---------------------------------------------------------------------------
class TestKVCacheMemorySafetyFactor:
    """KV cache memory estimate must include a safety margin."""

    def test_safety_factor_constant_exists(self):
        """Module should define MEMORY_SAFETY_FACTOR."""
        assert hasattr(_inf_mod, "MEMORY_SAFETY_FACTOR"), (
            "Module must define MEMORY_SAFETY_FACTOR constant (Bug #125)"
        )
        assert _inf_mod.MEMORY_SAFETY_FACTOR >= 1.2, (
            f"MEMORY_SAFETY_FACTOR should be >= 1.2, got {_inf_mod.MEMORY_SAFETY_FACTOR}"
        )

    def test_estimate_includes_safety_margin(self):
        """_estimate_kv_cache_bytes should include MEMORY_SAFETY_FACTOR multiplier."""
        model = MagicMock()
        model.args.num_hidden_layers = 32
        model.args.num_attention_heads = 32
        model.args.num_key_value_heads = 8
        model.args.head_dim = 128
        model.args.hidden_size = 4096

        result = _estimate_kv_cache_bytes(model, 1000)
        # Raw estimate: 32 layers * 2 * 8 heads * 128 dim * 1000 tokens * 2 bytes
        raw = 32 * 2 * 8 * 128 * 1000 * 2
        expected = int(raw * _inf_mod.MEMORY_SAFETY_FACTOR)
        assert result == expected, (
            f"Expected {expected} (raw {raw} * {_inf_mod.MEMORY_SAFETY_FACTOR}), got {result}"
        )

    def test_estimate_zero_tokens_unchanged(self):
        """Zero tokens should still return 0."""
        model = MagicMock()
        assert _estimate_kv_cache_bytes(model, 0) == 0
