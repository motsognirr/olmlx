"""Unit tests for VlmPromptCacheStore (#429). No real model required."""

from olmlx.engine.prompt_cache.vlm_state import VlmPromptCacheStore


class _FakeState:
    """Stand-in for mlx_vlm.generate.PromptCacheState (only identity matters)."""

    def __init__(self, tag):
        self.tag = tag


def test_disabled_when_capacity_zero():
    store = VlmPromptCacheStore(capacity=0)
    assert store.enabled() is False
    store.insert("a", _FakeState("a"))
    assert store.get("a") is None  # insert is a no-op when disabled


def test_insert_and_get():
    store = VlmPromptCacheStore(capacity=2)
    s = _FakeState("a")
    store.insert("a", s)
    assert store.get("a") is s
    assert store.get("missing") is None


def test_lru_eviction_past_capacity():
    store = VlmPromptCacheStore(capacity=2)
    store.insert("a", _FakeState("a"))
    store.insert("b", _FakeState("b"))
    store.insert("c", _FakeState("c"))  # evicts "a" (least-recently-used)
    assert store.get("a") is None
    assert store.get("b").tag == "b"
    assert store.get("c").tag == "c"


def test_get_promotes_to_mru():
    store = VlmPromptCacheStore(capacity=2)
    store.insert("a", _FakeState("a"))
    store.insert("b", _FakeState("b"))
    assert store.get("a").tag == "a"  # promote "a" -> MRU
    store.insert("c", _FakeState("c"))  # evicts "b" (now LRU), not "a"
    assert store.get("b") is None
    assert store.get("a").tag == "a"
    assert store.get("c").tag == "c"


def test_insert_same_id_replaces_without_growing():
    store = VlmPromptCacheStore(capacity=2)
    store.insert("a", _FakeState("first"))
    store.insert("a", _FakeState("second"))
    assert store.get("a").tag == "second"
    # Only one slot consumed: inserting two more should evict by capacity=2,
    # so "a" survives alongside the most recent.
    store.insert("b", _FakeState("b"))
    assert store.get("a").tag == "second"
    assert store.get("b").tag == "b"


def test_clear():
    store = VlmPromptCacheStore(capacity=2)
    store.insert("a", _FakeState("a"))
    store.clear()
    assert store.get("a") is None


def test_metrics_counters():
    store = VlmPromptCacheStore(capacity=2)
    assert store.metrics() == {
        "vlm_cache_hits": 0,
        "vlm_cache_misses": 0,
        "vlm_cache_tokens_reused": 0,
    }
    store.note_miss()
    store.note_hit(reused_tokens=10)
    store.note_hit(reused_tokens=5)
    assert store.metrics() == {
        "vlm_cache_hits": 2,
        "vlm_cache_misses": 1,
        "vlm_cache_tokens_reused": 15,
    }


# --- Disk spill (#491) -------------------------------------------------------

import mlx.core as mx  # noqa: E402


def _filled_state(token_ids):
    """A real mlx_vlm PromptCacheState backed by a serializable KVCache, so the
    safetensors round-trip exercises the real save/load helpers (no model)."""
    from mlx_lm.models.cache import KVCache
    from mlx_vlm.generate import PromptCacheState

    layer = KVCache()
    keys = mx.ones((1, 2, len(token_ids), 4), dtype=mx.float16)
    vals = mx.zeros((1, 2, len(token_ids), 4), dtype=mx.float16)
    layer.update_and_fetch(keys, vals)
    state = PromptCacheState()
    state.update(token_ids, [layer])
    return state


def _disk_store(tmp_path, capacity=1, max_bytes=None):
    return VlmPromptCacheStore(
        capacity=capacity,
        disk_path=tmp_path,
        model_name="test-vlm:latest",
        disk_max_bytes=max_bytes,
    )


def test_disk_disabled_without_path():
    store = VlmPromptCacheStore(capacity=1)
    assert store._disk_enabled is False


async def test_no_disk_write_when_disk_disabled(tmp_path):
    """A plain (no disk_path) store must not write files even on eviction."""
    store = VlmPromptCacheStore(capacity=1)
    await store.async_insert("a", _filled_state([1, 2, 3]))
    await store.async_insert("b", _filled_state([4, 5, 6]))  # evicts "a"
    assert list(tmp_path.iterdir()) == []


async def test_evicted_entry_spills_to_disk(tmp_path):
    store = _disk_store(tmp_path)
    await store.async_insert("a", _filled_state([1, 2, 3]))
    await store.async_insert("b", _filled_state([4, 5, 6]))  # evicts "a"
    # "a" left memory but its KV is now on disk.
    assert store.get("a") is None
    spilled = list((tmp_path / "test-vlm_latest").glob("*.safetensors"))
    assert len(spilled) == 1


async def test_restore_from_disk_round_trips(tmp_path):
    store = _disk_store(tmp_path)
    await store.async_insert("a", _filled_state([1, 2, 3]))
    await store.async_insert("b", _filled_state([4, 5, 6]))  # evicts "a" to disk

    restored = await store.async_get("a")
    assert restored is not None
    assert restored.token_ids == [1, 2, 3]
    assert restored.cache is not None
    assert restored.cache[0].offset == 3  # KV history length round-tripped


async def test_async_get_memory_hit_skips_disk(tmp_path):
    store = _disk_store(tmp_path, capacity=2)
    state = _filled_state([1, 2, 3])
    await store.async_insert("a", state)
    got = await store.async_get("a")
    assert got is state  # same in-memory object, not a disk reconstruction


async def test_async_get_miss_returns_none(tmp_path):
    store = _disk_store(tmp_path)
    assert await store.async_get("never-stored") is None


async def test_empty_state_not_spilled(tmp_path):
    """A never-filled state (cache is None) evicted before use writes nothing."""
    from mlx_vlm.generate import PromptCacheState

    store = _disk_store(tmp_path)
    await store.async_insert("a", PromptCacheState())  # cache=None
    await store.async_insert("b", _filled_state([4, 5, 6]))  # evicts empty "a"
    a_file = tmp_path / "test-vlm_latest" / "a.safetensors"
    assert not a_file.exists()


async def test_async_get_skips_reinsert_after_bulk_clear(tmp_path):
    """A memory-pressure ``clear()`` during the threaded disk restore must
    cancel the re-insert — otherwise the stale disk copy is written straight
    back in, partly defeating the flush (#618). Mirrors the text store's
    eviction-generation guard.
    """
    store = _disk_store(tmp_path, capacity=2)
    disk_state = _filled_state([1, 2, 3])

    def load_with_concurrent_clear(cid):
        # Simulate ``clear()`` (the memory-pressure path) landing while the
        # disk read is in flight: it bumps the eviction generation.
        store._evict_generation += 1
        return disk_state

    store._load_from_disk = load_with_concurrent_clear
    got = await store.async_get("a")
    # The restore is abandoned: nothing re-enters memory.
    assert got is None
    assert store.get("a") is None
    assert "a" not in store._entries


async def test_async_get_keeps_fresher_concurrent_insert(tmp_path):
    """If another coroutine inserts a fresher state for the same id during
    the threaded disk restore, ``async_get`` must return that fresher entry
    and discard the stale disk copy — not clobber it (#618)."""
    store = _disk_store(tmp_path, capacity=2)
    stale_disk = _filled_state([1, 2, 3])
    fresher = _filled_state([1, 2, 3, 4, 5])

    def load_with_concurrent_insert(cid):
        # Simulate a fresher insert for the same id completing during the
        # await (direct dict write bypasses the loop-thread assert, exactly
        # as the text-store guard test bumps _evict_generation directly).
        store._entries[cid] = fresher
        return stale_disk

    store._load_from_disk = load_with_concurrent_insert
    got = await store.async_get("a")
    assert got is fresher
    assert store.get("a") is fresher


async def test_cleanup_respects_byte_cap(tmp_path):
    # Cap below one entry's size → the oldest spilled file is reclaimed.
    store = _disk_store(tmp_path, capacity=1, max_bytes=1)
    await store.async_insert("a", _filled_state([1, 2, 3]))
    await store.async_insert("b", _filled_state([4, 5, 6]))  # evict a → save → cleanup
    await store.async_insert("c", _filled_state([7, 8, 9]))  # evict b → save → cleanup
    spilled = list((tmp_path / "test-vlm_latest").glob("*.safetensors"))
    # Byte cap keeps disk bounded; not every evictee survives on disk.
    assert len(spilled) <= 1
