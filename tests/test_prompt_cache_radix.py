"""Tests for the cross-request radix prefix cache (issue #365)."""

from olmlx.engine.prompt_cache.radix import PrefixCacheIndex
from olmlx.engine.prompt_cache.metrics import CacheMetrics


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
            "cache_id_hits", "cache_id_misses",
            "radix_hits", "radix_misses",
            "evictions_ram", "evictions_disk",
            "bytes_in_ram", "bytes_on_disk",
        }
