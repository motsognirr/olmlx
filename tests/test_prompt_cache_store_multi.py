"""Tests for the multi-checkpoint extension of PromptCacheStore."""

from olmlx.engine.prompt_cache.state import CachedPromptState
from olmlx.engine.prompt_cache.store import PromptCacheStore


def _state(tokens, *, cache_type="system", is_checkpoint=True):
    return CachedPromptState(
        tokens=list(tokens),
        cache=[],
        cache_type=cache_type,
        is_checkpoint=is_checkpoint,
    )


def test_insert_checkpoint_keys_by_tokens_not_cache_id():
    """Two checkpoints from different conversations sharing a system prefix
    must both be retrievable via fetch_nearest."""
    store = PromptCacheStore(max_slots=8)
    store.insert_checkpoint(_state([1, 2, 3]))
    store.insert_checkpoint(_state([1, 2, 3, 4, 5]))
    # Lookup with a 4th conversation that shares 5 tokens with the longer entry
    hit = store.fetch_nearest([1, 2, 3, 4, 5, 6, 7])
    assert hit is not None
    state, suffix = hit
    assert state.tokens == [1, 2, 3, 4, 5]
    assert suffix == [6, 7]


def test_fetch_nearest_returns_shorter_when_only_shorter_exists():
    store = PromptCacheStore(max_slots=8)
    store.insert_checkpoint(_state([1, 2, 3]))
    hit = store.fetch_nearest([1, 2, 3, 4, 5])
    assert hit is not None
    state, suffix = hit
    assert state.tokens == [1, 2, 3]
    assert suffix == [4, 5]


def test_fetch_nearest_returns_none_when_no_strict_prefix():
    store = PromptCacheStore(max_slots=8)
    store.insert_checkpoint(_state([1, 2, 9, 9]))
    assert store.fetch_nearest([1, 2, 3, 4]) is None


def test_fetch_nearest_exact_match_returns_empty_suffix():
    store = PromptCacheStore(max_slots=8)
    store.insert_checkpoint(_state([1, 2, 3]))
    hit = store.fetch_nearest([1, 2, 3])
    assert hit is not None
    state, suffix = hit
    assert state.tokens == [1, 2, 3]
    assert suffix == []


def test_insert_checkpoint_dedupes_on_identical_tokens():
    """Re-inserting a checkpoint with the same tokens replaces the entry,
    does not add a duplicate."""
    store = PromptCacheStore(max_slots=8)
    store.insert_checkpoint(_state([1, 2, 3]))
    store.insert_checkpoint(_state([1, 2, 3]))
    # Only one slot used.
    assert len(store) == 1


def test_estimate_state_bytes_counts_arrayscache_state():
    """Regression for Claude review Finding 2: ArraysCache layers expose
    state via ``.state`` (a list of arrays), not ``.keys``/``.values``.
    Without the fallback path, every checkpoint entry for a hybrid SSM
    model reports 0 bytes and the RAM eviction pressure never fires."""
    import mlx.core as mx
    from olmlx.engine.prompt_cache.state import CachedPromptState
    from olmlx.engine.prompt_cache.store import _estimate_state_bytes

    class _ArraysCacheLike:
        """Minimal stand-in for ArraysCache: no .keys/.values, only .state."""

        def __init__(self, arrays):
            self.state = arrays

    arr1 = mx.zeros((1, 4, 16, 32))
    arr2 = mx.zeros((1, 4, 16, 32))
    state = CachedPromptState(
        tokens=[1, 2, 3],
        cache=[_ArraysCacheLike([arr1, arr2])],
    )
    nbytes = _estimate_state_bytes(state)
    expected = arr1.nbytes + arr2.nbytes
    assert nbytes == expected, (
        f"expected {expected} bytes from ArraysCache.state, got {nbytes}"
    )


def test_estimate_state_bytes_still_counts_kvcache_keys_values():
    """Regression: the .keys/.values fast path must still work."""
    import mlx.core as mx
    from olmlx.engine.prompt_cache.state import CachedPromptState
    from olmlx.engine.prompt_cache.store import _estimate_state_bytes

    class _KvCacheLike:
        keys = mx.zeros((1, 4, 16, 32))
        values = mx.ones((1, 4, 16, 32))

    state = CachedPromptState(tokens=[1], cache=[_KvCacheLike()])
    nbytes = _estimate_state_bytes(state)
    expected = _KvCacheLike.keys.nbytes + _KvCacheLike.values.nbytes
    assert nbytes == expected
