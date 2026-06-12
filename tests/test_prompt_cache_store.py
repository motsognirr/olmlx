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


class TestTake:
    """Move semantics for the batched path (batching plan Phase 2): the
    entry leaves the store atomically on the loop thread — no await
    window between lookup and removal in which a concurrent (lockless)
    batched request could grab the same mutable cache object."""

    def test_take_returns_and_removes(self):
        store = PromptCacheStore(max_slots=4)
        state = _make_state(1)
        store.set("a", state)
        assert store.take("a") is state
        assert store.get("a") is None
        assert len(store) == 0

    def test_take_missing_returns_none(self):
        store = PromptCacheStore(max_slots=4)
        assert store.take("missing") is None

    def test_take_removes_from_radix(self):
        store = PromptCacheStore(max_slots=4)
        state = CachedPromptState(tokens=list(range(64)), cache=["kv"])
        store.set("a", state)
        store.take("a")
        assert store.find_by_prefix(list(range(64)), min_prefix_tokens=8) is None

    def test_take_updates_ram_bytes(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state(1))
        before = store.metrics.bytes_in_ram
        store.take("a")
        assert store.metrics.bytes_in_ram <= before

    def test_take_counts_no_cache_id_metrics(self):
        """take() is metrics-neutral: the radix path that uses it has
        already counted a radix hit, and async_take owns hit/miss
        accounting for the cache_id path."""
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state(1))
        store.take("a")
        store.take("missing")
        assert store.metrics.cache_id_hits == 0
        assert store.metrics.cache_id_misses == 0

    async def test_async_take_memory_hit(self):
        store = PromptCacheStore(max_slots=4)
        state = _make_state(1)
        store.set("a", state)
        assert await store.async_take("a") is state
        assert store.get("a") is None
        assert store.metrics.cache_id_hits == 1

    async def test_async_take_miss_counts_miss(self):
        store = PromptCacheStore(max_slots=4)
        assert await store.async_take("missing") is None
        assert store.metrics.cache_id_misses == 1

    async def test_async_take_disk_fallback_loads_and_unlinks(self, tmp_path):
        store = PromptCacheStore(max_slots=4, disk_path=tmp_path, model_name="m")
        disk_state = _make_state(7)
        fake_path = tmp_path / "m" / "a.safetensors"
        unlinked = []
        store._read_from_disk = lambda cid: (disk_state, fake_path)
        store._unlink_and_refresh = lambda p: unlinked.append(p)
        got = await store.async_take("a")
        assert got is disk_state
        # Move semantics: not re-inserted into memory, file removed.
        assert len(store) == 0
        assert unlinked == [fake_path]
        assert store.metrics.cache_id_hits == 1

    async def test_async_take_memory_hit_unlinks_stale_disk_copy(self, tmp_path):
        store = PromptCacheStore(max_slots=4, disk_path=tmp_path, model_name="m")
        unlinked = []
        store._unlink_and_refresh = lambda p: unlinked.append(p)
        store.set("a", _make_state(1))
        await store.async_take("a")
        assert unlinked == [store._disk_file_path("a")]
