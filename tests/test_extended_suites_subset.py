"""Tests for olmlx.bench.extended_suites: deterministic subset selection + cache dir."""

from __future__ import annotations

from pathlib import Path

from olmlx.bench.extended_suites import bench_cache_dir, select_subset


class TestSelectSubset:
    def test_returns_n_items_when_n_lt_len(self):
        items = list(range(100))
        out = select_subset(items, 10)
        assert len(out) == 10

    def test_returns_all_items_when_n_ge_len(self):
        items = list(range(5))
        out = select_subset(items, 10)
        assert out == items

    def test_is_deterministic(self):
        items = list(range(100))
        a = select_subset(items, 10)
        b = select_subset(items, 10)
        assert a == b

    def test_spreads_evenly(self):
        items = list(range(164))
        out = select_subset(items, 50)
        assert out[0] == 0
        assert out[-1] >= 160

    def test_stratified_balances_strata(self):
        items = [(i, f"stratum-{i % 3}") for i in range(12)]
        out = select_subset(items, 6, key=lambda x: x[1])
        counts: dict[str, int] = {}
        for _, s in out:
            counts[s] = counts.get(s, 0) + 1
        assert counts == {"stratum-0": 2, "stratum-1": 2, "stratum-2": 2}

    def test_stratified_redistributes_when_bucket_too_small(self):
        # 3 strata: one large, two singletons. n=6 should yield 4 from the large
        # bucket + 1 + 1 from the singletons (or whatever distribution keeps
        # len == n). Total must equal n.
        items = [(i, "big") for i in range(10)] + [(99, "tiny1")] + [(100, "tiny2")]
        out = select_subset(items, 6, key=lambda x: x[1])
        assert len(out) == 6
        labels = [s for _, s in out]
        # The two singletons must each appear exactly once (they only have one item).
        assert labels.count("tiny1") == 1
        assert labels.count("tiny2") == 1
        # The remainder (4) all come from "big".
        assert labels.count("big") == 4


class TestBenchCacheDir:
    def test_default_path(self):
        d = bench_cache_dir()
        assert d == Path("~/.olmlx/bench-cache").expanduser()

    def test_creates_if_missing(self, tmp_path, monkeypatch):
        target = tmp_path / "bc"
        monkeypatch.setenv("OLMLX_BENCH_CACHE_DIR", str(target))
        d = bench_cache_dir()
        assert d == target
        assert d.is_dir()
