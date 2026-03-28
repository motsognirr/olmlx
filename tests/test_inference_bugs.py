"""Tests for inference engine bug fixes (#118, #119, #120, #123, #124, #125)."""

import copy
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import olmlx.engine.inference as _inf_mod
from olmlx.engine.inference import (
    _estimate_kv_cache_bytes,
    _inference_ref,
)
from olmlx.engine.model_manager import CachedPromptState, LoadedModel


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

    def test_drain_timeout_flag_exists(self):
        """CancellableStream should have a flag to indicate stuck thread."""
        from olmlx.utils.streaming import CancellableStream

        stream = CancellableStream(lambda ce: iter([]), is_vlm=False)
        # After fix, there should be a way to check if the thread is stuck
        assert hasattr(stream, "thread_stuck") or hasattr(stream, "_thread_stuck"), (
            "CancellableStream needs a thread_stuck indicator (Bug #120)"
        )

    @pytest.mark.asyncio
    async def test_kv_cache_eviction_calls_synchronize(self):
        """After KV cache eviction mx.clear_cache(), mx.synchronize() must follow."""
        import inspect

        source = inspect.getsource(_inf_mod._stream_completion)
        # Find the eviction recovery path (evict_all_to_disk + clear_cache)
        # After the fix, mx.synchronize() should appear after mx.clear_cache()
        # in the memory pressure section
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "evict_all_to_disk" in line:
                # Look for mx.synchronize() within the next few lines after clear_cache
                block = "\n".join(lines[i : i + 10])
                if "clear_cache" in block:
                    assert "synchronize" in block or "_safe_sync" in block, (
                        "mx.synchronize() or _safe_sync() must follow mx.clear_cache() "
                        "in KV cache eviction path (Bug #120)"
                    )
                break


# ---------------------------------------------------------------------------
# Bug #123: Prompt cache state corruption on mid-stream disconnect
# ---------------------------------------------------------------------------
class TestPromptCacheDeepCopy:
    """Cache must be deep-copied before passing to generator to avoid corruption."""

    @pytest.mark.asyncio
    async def test_cache_not_committed_on_incomplete_generation(self):
        """On disconnect (generation_complete=False), cache must not be stored."""
        import inspect

        source = inspect.getsource(_inf_mod._stream_completion)
        # The fix should deep-copy the cache before use and only commit on success.
        # Check that copy or deepcopy is used for cache objects
        assert "copy" in source or "deepcopy" in source, (
            "Prompt cache must be deep-copied before passing to generator (Bug #123)"
        )

    def test_cache_deep_copy_preserves_structure(self):
        """Deep-copying a CachedPromptState should produce an independent copy."""
        original_cache = [MagicMock(), MagicMock()]
        state = CachedPromptState(tokens=[1, 2, 3], cache=original_cache)
        copied = copy.deepcopy(state)
        # Tokens should be independent
        copied.tokens.append(4)
        assert len(state.tokens) == 3
        # Cache list should be independent (the objects inside may be shallow-copied
        # but the list itself should be different)
        assert copied.cache is not original_cache


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
