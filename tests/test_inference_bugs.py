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
            any nested (closure / inner-function) code objects."""
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
