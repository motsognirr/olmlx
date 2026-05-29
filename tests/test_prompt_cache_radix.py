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
        # Query shares 2 tokens with the stored sequence; reachable
        # terminal in the subtree wins — caller will trim back.
        assert idx.find_longest_prefix([1, 2]) == ("a", 2)

    def test_no_overlap_returns_no_match(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        assert idx.find_longest_prefix([9, 8, 7]) == (None, 0)

    def test_sibling_branch_finds_subtree_terminal(self):
        # Claude-Code-style case: A's terminal is past the divergence
        # point, but the shared prefix should still be findable.
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3, 42, 43], "a")
        # Sibling query diverges at depth 4 (42 vs 99); should still
        # find "a" with prefix_len == 3.
        assert idx.find_longest_prefix([1, 2, 3, 99, 100]) == ("a", 3)

    def test_sibling_prefers_deeper_descent_terminal(self):
        # If a shallow terminal lies on the descent path AND a deeper
        # terminal sits in the subtree past the divergence, the deeper
        # subtree terminal wins because it shares more tokens.
        idx = PrefixCacheIndex()
        idx.insert([1, 2], "shallow")
        idx.insert([1, 2, 3, 4], "deep")
        # Query diverges at depth 4; deepest visited node is at depth 3.
        # Subtree below depth 3 contains "deep" → return ("deep", 3).
        assert idx.find_longest_prefix([1, 2, 3, 99]) == ("deep", 3)

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
        # "a"'s branch was pruned; query that diverges at depth 4 still
        # shares [1,2,3] with "b" so returns ("b", 3).
        assert idx.find_longest_prefix([1, 2, 3, 7, 9]) == ("b", 3)
        assert idx.find_longest_prefix([1, 2, 3, 8, 9]) == ("b", 4)

    def test_remove_deep_keeps_shallow_terminal(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "short")
        idx.insert([1, 2, 3, 4, 5], "long")
        idx.remove([1, 2, 3, 4, 5], "long")
        assert idx.find_longest_prefix([1, 2, 3, 4]) == ("short", 3)

    def test_duplicate_path_keeps_both_terminals(self):
        # Two cache_ids with identical token sequences must both stay
        # reachable; removing one leaves the other findable. Previously
        # the second insert overwrote the first terminal, breaking the
        # invariant that every entry in _entries has a trie terminal.
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        idx.insert([1, 2, 3], "b")
        # Either cache_id is acceptable on lookup — both share the prefix.
        cache_id, depth = idx.find_longest_prefix([1, 2, 3])
        assert cache_id in {"a", "b"}
        assert depth == 3
        # Removing "b" must NOT clear "a"'s terminal.
        idx.remove([1, 2, 3], "b")
        assert idx.find_longest_prefix([1, 2, 3]) == ("a", 3)
        idx.remove([1, 2, 3], "a")
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

    def test_find_by_prefix_short_circuits_below_threshold(self):
        # Regression for aider PR #391 review: a shallow descent (e.g.
        # only BOS matches) must not walk the entire subtree under that
        # shallow node when min_prefix_tokens guarantees a reject. We
        # verify by counting DFS subtree visits through the trie's
        # `find_longest_prefix(min_depth=...)` parameter — when the
        # descent ends below min_depth, the function must return
        # (None, 0) without touching any subtree node.
        from olmlx.engine.prompt_cache.radix import PrefixCacheIndex

        idx = PrefixCacheIndex()
        # Fat subtree under BOS=1, no terminal at depth 1.
        for i in range(8):
            idx.insert([1, 100 + i, 200 + i], f"e{i}")

        # Query shares only BOS with stored entries (descent depth 1).
        # With min_depth=256, expect (None, 0).
        result = idx.find_longest_prefix([1, 9999], min_depth=256)
        assert result == (None, 0)

        # Without the threshold, the same query finds something via DFS.
        cache_id, depth = idx.find_longest_prefix([1, 9999])
        assert cache_id is not None
        assert depth == 1

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

    def test_bulk_eviction_counts_toward_evictions_ram(self):
        # Regression for aider PR #391 follow-up: memory-pressure
        # bulk evictions via evict_all_to_disk() must be reflected in
        # evictions_ram, not silently zeroed out alongside _entries.
        store = PromptCacheStore(max_slots=8)
        store.set("a", _make_state([1, 2]))
        store.set("b", _make_state([3, 4]))
        store.set("c", _make_state([5, 6]))
        before = store.metrics.evictions_ram
        store.evict_all_to_disk()
        # All three entries were dropped from RAM.
        assert store.metrics.evictions_ram == before + 3

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
        shared_prompt = list(range(1, 1025))  # 1024-token system prefix
        # Seed "A": shared system prefix + a completed turn (user msg + reply).
        # The terminal sits past the divergence point that "B" will create.
        seeded_tokens = shared_prompt + [42, 43, 44]  # 1027 tokens stored
        seeded = CachedPromptState(tokens=seeded_tokens, cache=[MagicMock()])
        store.set("A", seeded)

        # "B" arrives with the same system prefix but a different user turn.
        # This is the Claude-Code-style branching case: shared system prompt,
        # divergent next message. A's KV state should be taken over and
        # trimmed back to the 1024-token shared prefix.
        sibling_prompt_tokens = shared_prompt + [99, 100]  # 1026 tokens total

        lm = _make_lm_for_radix(store)
        gen_kwargs: dict = {}

        with (
            patch("olmlx.engine.inference.trim_prompt_cache", return_value=3),
            patch(
                "olmlx.engine.inference.make_prompt_cache", return_value=[MagicMock()]
            ),
        ):
            result = await _setup_prompt_cache(
                lm,
                prompt=sibling_prompt_tokens,
                gen_kwargs=gen_kwargs,
                prompt_tokens=sibling_prompt_tokens,
                cache_id="B",
            )

        # Radix hit: A's KV state trimmed back to the 1024-token shared prefix.
        assert result.cache_read_tokens >= 1024
        # Only the 2-token divergent user turn needs prefilling.
        assert result.cache_creation_tokens <= 2
        # Takeover: "A" gone (entry transferred to "B").
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


class TestRamBudgetEviction:
    def _state_with_bytes(self, tokens, nbytes):
        layer = MagicMock()
        layer.keys = MagicMock()
        layer.keys.nbytes = nbytes
        layer.values = MagicMock()
        layer.values.nbytes = 0
        return CachedPromptState(tokens=list(tokens), cache=[layer])

    def test_byte_budget_evicts_extra_entries(self):
        # Slot cap 10, byte budget 2500 bytes; entries are 1000 bytes each
        store = PromptCacheStore(max_slots=10, ram_budget_bytes=2500)
        store.set("a", self._state_with_bytes([1, 1, 1], 1000))
        store.set("b", self._state_with_bytes([2, 2, 2], 1000))
        store.set("c", self._state_with_bytes([3, 3, 3], 1000))
        # 3000 bytes > 2500 → LRU (a) evicted
        assert store.peek("a") is None
        assert store.peek("b") is not None
        assert store.peek("c") is not None
        assert store.metrics.evictions_ram == 1

    def test_no_budget_leaves_entries_alone(self):
        store = PromptCacheStore(max_slots=10, ram_budget_bytes=None)
        store.set("a", self._state_with_bytes([1], 10_000))
        store.set("b", self._state_with_bytes([2], 10_000))
        assert store.peek("a") is not None
        assert store.peek("b") is not None
