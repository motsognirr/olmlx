"""Tests for the Responses-API state store."""

from olmlx.engine.responses_state import ResponsesStore


def test_put_get_roundtrip():
    store = ResponsesStore(max_entries=4)
    store.put("resp_1", {"output_items": [{"a": 1}]})
    assert store.get("resp_1") == {"output_items": [{"a": 1}]}


def test_missing_returns_none():
    store = ResponsesStore(max_entries=4)
    assert store.get("nope") is None


def test_delete():
    store = ResponsesStore(max_entries=4)
    store.put("resp_1", {"x": 1})
    assert store.delete("resp_1") is True
    assert store.get("resp_1") is None
    assert store.delete("resp_1") is False


def test_lru_eviction():
    store = ResponsesStore(max_entries=2)
    store.put("a", {"v": 1})
    store.put("b", {"v": 2})
    store.put("c", {"v": 3})  # evicts "a"
    assert store.get("a") is None
    assert store.get("b") == {"v": 2}
    assert store.get("c") == {"v": 3}


def test_get_refreshes_lru():
    store = ResponsesStore(max_entries=2)
    store.put("a", {"v": 1})
    store.put("b", {"v": 2})
    store.get("a")  # "a" now most-recently used
    store.put("c", {"v": 3})  # evicts "b", not "a"
    assert store.get("a") == {"v": 1}
    assert store.get("b") is None
