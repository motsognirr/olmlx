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
