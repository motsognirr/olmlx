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

    def test_unset_returns_none_silently(self, caplog):
        with caplog.at_level("WARNING", logger="olmlx.bench.worker"):
            assert _resolve_bench_think({}) is None
        # Unset is the engine-default path; no warning should fire.
        assert caplog.records == []

    def test_empty_returns_none_silently(self, caplog):
        with caplog.at_level("WARNING", logger="olmlx.bench.worker"):
            assert _resolve_bench_think({"OLMLX_BENCH_THINK": ""}) is None
        assert caplog.records == []

    def test_unrecognized_returns_none_and_warns(self, caplog):
        # A typo'd value (e.g. ``"enabled"``) must not pass silently — an
        # A/B with one arm typo'd would otherwise run engine-default on
        # both arms with no signal anything went wrong.
        with caplog.at_level("WARNING", logger="olmlx.bench.worker"):
            assert _resolve_bench_think({"OLMLX_BENCH_THINK": "enabled"}) is None
        assert any(
            "OLMLX_BENCH_THINK" in rec.getMessage() and "enabled" in rec.getMessage()
            for rec in caplog.records
        )
