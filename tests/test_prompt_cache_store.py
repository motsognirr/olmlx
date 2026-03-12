"""Tests for PromptCacheStore (per-agent KV cache LRU store)."""

from olmlx.engine.model_manager import CachedPromptState, PromptCacheStore


def _make_state(token_id: int = 1) -> CachedPromptState:
    """Create a minimal CachedPromptState for testing."""
    return CachedPromptState(tokens=[token_id], cache=[f"cache_{token_id}"])


class TestPromptCacheStore:
    def test_get_returns_none_for_missing_key(self):
        store = PromptCacheStore(max_slots=4)
        assert store.get("nonexistent") is None

    def test_set_and_get_roundtrip(self):
        store = PromptCacheStore(max_slots=4)
        state = _make_state(1)
        evicted = store.set("agent-a", state)
        assert evicted is None
        assert store.get("agent-a") is state

    def test_set_evicts_lru_when_full(self):
        store = PromptCacheStore(max_slots=2)
        state_a = _make_state(1)
        store.set("a", state_a)
        store.set("b", _make_state(2))
        evicted = store.set("c", _make_state(3))  # should evict "a"
        assert evicted is state_a
        assert store.get("a") is None
        assert store.get("b") is not None
        assert store.get("c") is not None

    def test_get_promotes_to_mru(self):
        store = PromptCacheStore(max_slots=2)
        store.set("a", _make_state(1))
        store.set("b", _make_state(2))
        store.get("a")  # promote "a" to MRU
        store.set("c", _make_state(3))  # should evict "b" (LRU), not "a"
        assert store.get("a") is not None
        assert store.get("b") is None
        assert store.get("c") is not None

    def test_remove_specific_key(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state(1))
        store.set("b", _make_state(2))
        store.remove("a")
        assert store.get("a") is None
        assert store.get("b") is not None

    def test_remove_missing_key_is_noop(self):
        store = PromptCacheStore(max_slots=4)
        store.remove("nonexistent")  # should not raise

    def test_clear_removes_all(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state(1))
        store.set("b", _make_state(2))
        store.clear()
        assert store.get("a") is None
        assert store.get("b") is None
        assert len(store) == 0

    def test_default_key_empty_string(self):
        store = PromptCacheStore(max_slots=4)
        state = _make_state(1)
        store.set("", state)
        assert store.get("") is state

    def test_max_slots_one_behaves_like_single_cache(self):
        store = PromptCacheStore(max_slots=1)
        state_a = _make_state(1)
        store.set("a", state_a)
        assert store.get("a") is state_a

        state_b = _make_state(2)
        evicted = store.set("b", state_b)
        assert evicted is state_a
        assert store.get("a") is None
        assert store.get("b") is state_b

    def test_len_tracks_entries(self):
        store = PromptCacheStore(max_slots=4)
        assert len(store) == 0
        store.set("a", _make_state(1))
        assert len(store) == 1
        store.set("b", _make_state(2))
        assert len(store) == 2
        store.remove("a")
        assert len(store) == 1

    def test_set_overwrites_existing_key(self):
        store = PromptCacheStore(max_slots=4)
        state1 = _make_state(1)
        state2 = _make_state(2)
        store.set("a", state1)
        old = store.set("a", state2)
        assert old is state1  # different .cache → returns old for cleanup
        assert store.get("a") is state2
        assert len(store) == 1

    def test_set_overwrite_same_cache_returns_none(self):
        """Overwrite with same .cache list returns None (no cleanup needed)."""
        store = PromptCacheStore(max_slots=4)
        shared_cache = ["kv_layer"]
        state1 = CachedPromptState(tokens=[1, 2], cache=shared_cache)
        state2 = CachedPromptState(tokens=[1, 2, 3], cache=shared_cache)
        store.set("a", state1)
        old = store.set("a", state2)
        assert old is None
        assert store.get("a") is state2
