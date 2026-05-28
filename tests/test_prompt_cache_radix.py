"""Tests for the cross-request radix prefix cache (issue #365)."""

import pytest
from unittest.mock import MagicMock, patch

from olmlx.engine.prompt_cache.radix import PrefixCacheIndex
from olmlx.engine.prompt_cache.metrics import CacheMetrics
from olmlx.engine.prompt_cache import CachedPromptState, PromptCacheStore


class TestPrefixCacheIndexInsertLookup:
    def test_empty_index_returns_no_match(self):
        idx = PrefixCacheIndex()
        assert idx.find_longest_prefix([1, 2, 3]) == (None, 0)

    def test_single_entry_exact_match(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        assert idx.find_longest_prefix([1, 2, 3]) == ("a", 3)

    def test_single_entry_longer_query(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        assert idx.find_longest_prefix([1, 2, 3, 4, 5]) == ("a", 3)

    def test_single_entry_shorter_query(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        # No terminal at depth 2 → no match
        assert idx.find_longest_prefix([1, 2]) == (None, 0)

    def test_no_overlap_returns_no_match(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        assert idx.find_longest_prefix([9, 8, 7]) == (None, 0)

    def test_two_entries_sibling_branches(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3, 7], "a")
        idx.insert([1, 2, 3, 8], "b")
        assert idx.find_longest_prefix([1, 2, 3, 7, 9]) == ("a", 4)
        assert idx.find_longest_prefix([1, 2, 3, 8, 9]) == ("b", 4)

    def test_returns_deepest_terminal_on_chain(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "short")
        idx.insert([1, 2, 3, 4, 5], "long")
        # Query that fully matches the long entry returns the long one
        assert idx.find_longest_prefix([1, 2, 3, 4, 5, 6]) == ("long", 5)
        # Query that only matches the short entry returns the short one
        assert idx.find_longest_prefix([1, 2, 3, 9]) == ("short", 3)


class TestPrefixCacheIndexRemove:
    def test_remove_then_lookup_misses(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        idx.remove([1, 2, 3], "a")
        assert idx.find_longest_prefix([1, 2, 3]) == (None, 0)

    def test_remove_unknown_is_noop(self):
        idx = PrefixCacheIndex()
        idx.remove([1, 2, 3], "a")  # should not raise
        assert idx.find_longest_prefix([1, 2, 3]) == (None, 0)

    def test_remove_one_sibling_keeps_other(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3, 7], "a")
        idx.insert([1, 2, 3, 8], "b")
        idx.remove([1, 2, 3, 7], "a")
        assert idx.find_longest_prefix([1, 2, 3, 7, 9]) == (None, 0)
        assert idx.find_longest_prefix([1, 2, 3, 8, 9]) == ("b", 4)

    def test_remove_deep_keeps_shallow_terminal(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "short")
        idx.insert([1, 2, 3, 4, 5], "long")
        idx.remove([1, 2, 3, 4, 5], "long")
        assert idx.find_longest_prefix([1, 2, 3, 4]) == ("short", 3)

    def test_overwrite_same_path_replaces_terminal(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        idx.insert([1, 2, 3], "b")
        assert idx.find_longest_prefix([1, 2, 3]) == ("b", 3)
        idx.remove([1, 2, 3], "b")
        assert idx.find_longest_prefix([1, 2, 3]) == (None, 0)


class TestCacheMetrics:
    def test_defaults_are_zero(self):
        m = CacheMetrics()
        assert m.cache_id_hits == 0
        assert m.cache_id_misses == 0
        assert m.radix_hits == 0
        assert m.radix_misses == 0
        assert m.evictions_ram == 0
        assert m.evictions_disk == 0
        assert m.bytes_in_ram == 0
        assert m.bytes_on_disk == 0

    def test_to_dict_round_trip(self):
        m = CacheMetrics(cache_id_hits=3, radix_hits=1, bytes_in_ram=2048)
        d = m.to_dict()
        assert d["cache_id_hits"] == 3
        assert d["radix_hits"] == 1
        assert d["bytes_in_ram"] == 2048
        # All defined keys present
        assert set(d.keys()) == {
            "cache_id_hits",
            "cache_id_misses",
            "radix_hits",
            "radix_misses",
            "evictions_ram",
            "evictions_disk",
            "bytes_in_ram",
            "bytes_on_disk",
        }


def _make_state(tokens: list[int]) -> CachedPromptState:
    return CachedPromptState(tokens=list(tokens), cache=[object()])


class TestPromptCacheStoreRadixIntegration:
    def test_set_inserts_into_radix(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state([1, 2, 3]))
        store_metrics = store.metrics
        assert store_metrics.bytes_in_ram >= 0  # estimator runs without crashing
        found = store.find_by_prefix([1, 2, 3, 4], min_prefix_tokens=0)
        assert found is not None
        old_cache_id, state, depth = found
        assert old_cache_id == "a"
        assert depth == 3
        assert state.tokens == [1, 2, 3]

    def test_remove_drops_radix_terminal(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state([1, 2, 3]))
        store.remove("a")
        assert store.find_by_prefix([1, 2, 3], min_prefix_tokens=0) is None

    def test_find_by_prefix_threshold_filters(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state([1, 2, 3]))
        # min_prefix_tokens=5 rejects the 3-token match
        assert store.find_by_prefix([1, 2, 3, 9, 9], min_prefix_tokens=5) is None
        # min_prefix_tokens=3 accepts
        assert store.find_by_prefix([1, 2, 3, 9, 9], min_prefix_tokens=3) is not None

    def test_takeover_rekeys_entry(self):
        store = PromptCacheStore(max_slots=4)
        original = _make_state([1, 2, 3])
        store.set("a", original)
        taken = store.takeover("a", "b")
        assert taken is original
        # Old key gone
        assert store.peek("a") is None
        # New key resolves
        assert store.peek("b") is original
        # Radix terminal now points to "b"
        cache_id, _ = store._radix.find_longest_prefix([1, 2, 3])
        assert cache_id == "b"

    def test_takeover_unknown_returns_none(self):
        store = PromptCacheStore(max_slots=4)
        assert store.takeover("missing", "new") is None

    def test_lru_eviction_removes_radix_terminal(self):
        store = PromptCacheStore(max_slots=2)
        store.set("a", _make_state([1, 1, 1]))
        store.set("b", _make_state([2, 2, 2]))
        store.set("c", _make_state([3, 3, 3]))  # evicts "a"
        assert store.peek("a") is None
        # Radix no longer has the "a" path
        cache_id, _ = store._radix.find_longest_prefix([1, 1, 1])
        assert cache_id is None
        # Metrics recorded the eviction
        assert store.metrics.evictions_ram == 1

    def test_metrics_tracks_hits_and_misses(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state([1, 2, 3]))
        assert store.peek("a") is not None  # peek does not affect metrics
        # get() records a hit
        store.get("a")
        assert store.metrics.cache_id_hits == 1
        # get() on unknown records a miss
        store.get("missing")
        assert store.metrics.cache_id_misses == 1


@pytest.fixture(autouse=True)
def _safe_memory_defaults():
    """Mock memory utils so prompt cache tests aren't affected by real Metal state."""
    with (
        patch("olmlx.utils.memory.mx.get_active_memory", return_value=0),
        patch("olmlx.utils.memory.mx.get_cache_memory", return_value=0),
    ):
        yield


def _make_lm_for_radix(store: PromptCacheStore) -> MagicMock:
    """Minimal LoadedModel stand-in for _setup_prompt_cache."""
    lm = MagicMock()
    lm.prompt_cache_store = store
    lm.supports_cache_persistence = True
    lm.supports_cache_trim = True
    lm.is_vlm = False
    lm.kv_cache_quant = None
    lm.is_distributed = False
    return lm


@pytest.mark.asyncio
class TestRadixFallbackInSetup:
    async def test_sibling_prefix_hit_triggers_takeover(self):
        from olmlx.engine.inference import _setup_prompt_cache

        store = PromptCacheStore(max_slots=4)
        shared_prompt = list(range(1, 1025))  # 1024 tokens
        # Seed the store with cache_id="A": shared system prompt + completed turn.
        # Simulates a previous request that processed a full conversation turn.
        seeded_tokens = shared_prompt + [42, 43, 44]  # 1027 tokens stored
        seeded = CachedPromptState(tokens=seeded_tokens, cache=[MagicMock()])
        store.set("A", seeded)

        # New request "B" extends "A"'s stored tokens with a new user turn.
        # This is the real Claude Code sibling case — same conversation history,
        # new cache_id, new user message appended.
        continuation_tokens = seeded_tokens + [99, 100]  # 1029 tokens total

        lm = _make_lm_for_radix(store)
        gen_kwargs: dict = {}

        with patch(
            "olmlx.engine.inference.make_prompt_cache", return_value=[MagicMock()]
        ):
            result = await _setup_prompt_cache(
                lm,
                prompt=continuation_tokens,
                gen_kwargs=gen_kwargs,
                prompt_tokens=continuation_tokens,
                cache_id="B",
            )

        # Radix hit: prefix matched 1027 tokens, those tokens are reused.
        assert result.cache_read_tokens >= 1024
        # Only the 2-token new user turn needs prefilling.
        assert result.cache_creation_tokens <= 3
        # Takeover: "A" gone (taken over then removed for mid-gen safety).
        assert store.peek("A") is None
        # KV state wired into gen_kwargs so mlx-lm reuses it.
        assert "prompt_cache" in gen_kwargs

    async def test_radix_below_threshold_falls_to_fresh(self):
        from olmlx.engine.inference import _setup_prompt_cache

        store = PromptCacheStore(max_slots=4)
        store.set("A", CachedPromptState(tokens=[1, 2, 3], cache=[MagicMock()]))
        lm = _make_lm_for_radix(store)

        with (
            patch(
                "olmlx.engine.inference.settings.prompt_cache_radix_min_prefix_tokens",
                1000,
            ),
            patch(
                "olmlx.engine.inference.make_prompt_cache", return_value=[MagicMock()]
            ),
        ):
            result = await _setup_prompt_cache(
                lm,
                prompt=[1, 2, 3, 4, 5],
                gen_kwargs={},
                prompt_tokens=[1, 2, 3, 4, 5],
                cache_id="B",
            )

        # Threshold not met → fresh cache, A stays in place
        assert result.cache_read_tokens == 0
        assert store.peek("A") is not None

    async def test_radix_disabled_skips_lookup(self):
        from olmlx.engine.inference import _setup_prompt_cache

        store = PromptCacheStore(max_slots=4)
        store.set("A", CachedPromptState(tokens=list(range(1024)), cache=[MagicMock()]))
        lm = _make_lm_for_radix(store)

        with (
            patch("olmlx.engine.inference.settings.prompt_cache_radix", False),
            patch(
                "olmlx.engine.inference.make_prompt_cache", return_value=[MagicMock()]
            ),
        ):
            result = await _setup_prompt_cache(
                lm,
                prompt=list(range(1024)) + [99],
                gen_kwargs={},
                prompt_tokens=list(range(1024)) + [99],
                cache_id="B",
            )

        # No fallback → fresh prefill, A still there
        assert result.cache_read_tokens == 0
        assert store.peek("A") is not None
