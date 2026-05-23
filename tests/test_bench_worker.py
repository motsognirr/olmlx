"""Tests for the bench worker's thinking toggle."""

from __future__ import annotations

import pytest

from olmlx.bench.worker import _resolve_bench_think


class TestResolveBenchThink:
    @pytest.mark.parametrize("value", ["true", "True", "1", "on", "yes"])
    def test_truthy_enables_thinking(self, value):
        assert _resolve_bench_think({"OLMLX_BENCH_THINK": value}) is True

    @pytest.mark.parametrize("value", ["false", "False", "0", "off", "no"])
    def test_falsy_disables_thinking(self, value):
        assert _resolve_bench_think({"OLMLX_BENCH_THINK": value}) is False

    def test_unset_returns_none(self):
        assert _resolve_bench_think({}) is None

    def test_empty_returns_none(self):
        assert _resolve_bench_think({"OLMLX_BENCH_THINK": ""}) is None

    def test_unrecognized_returns_none(self):
        assert _resolve_bench_think({"OLMLX_BENCH_THINK": "maybe"}) is None
