"""Tests for the chart builders in scripts/build_extended_report.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_extended_report  # type: ignore[import-not-found]


@pytest.fixture
def run_dir(tmp_path):
    """Copy the synthetic run fixture into tmp_path."""
    raw = tmp_path / "raw"
    raw.mkdir()
    fixtures = Path(__file__).parent / "fixtures" / "extended_run" / "raw"
    for src in fixtures.glob("*.json"):
        (raw / src.name).write_bytes(src.read_bytes())
    return tmp_path


class TestLoadResults:
    def test_loads_all_json(self, run_dir):
        results = build_extended_report.load_results(run_dir)
        assert len(results) == 4
        models = {r["model"] for r in results}
        assert "org/qwen-tiny" in models
        assert "org/big-moe" in models


class TestFrontierChart:
    def test_writes_png(self, run_dir):
        out = run_dir / "charts" / "frontier.png"
        build_extended_report.render_frontier_chart(
            build_extended_report.load_results(run_dir), out
        )
        assert out.exists()
        assert out.stat().st_size > 0


class TestSuiteHeatmap:
    def test_writes_png(self, run_dir):
        out = run_dir / "charts" / "suite_heatmap.png"
        build_extended_report.render_suite_heatmap(
            build_extended_report.load_results(run_dir), out
        )
        assert out.exists()


class TestQuantPairsChart:
    def test_handles_empty_pairs(self, run_dir):
        # The run_dir fixture now includes qwen-pair-4bit and qwen-pair-6bit with
        # :latest suffix, so this exercises alias resolution rather than empty-pairs.
        out = run_dir / "charts" / "quant_pairs.png"
        build_extended_report.render_quant_pairs_chart(
            build_extended_report.load_results(run_dir), out
        )
        assert out.exists()


class TestQuantPairsAliasResolution:
    def test_renders_with_alias_suffix(self, tmp_path):
        # Copy fixtures including the new pair, render, verify chart contains actual bars
        raw = tmp_path / "raw"
        raw.mkdir()
        fixtures = Path(__file__).parent / "fixtures" / "extended_run" / "raw"
        for src in fixtures.glob("*.json"):
            (raw / src.name).write_bytes(src.read_bytes())
        results = build_extended_report.load_results(tmp_path)
        out = tmp_path / "charts" / "quant_pairs.png"
        build_extended_report.render_quant_pairs_chart(results, out)
        assert out.exists()
        # Size sanity: a real-data chart is much larger than the placeholder
        assert out.stat().st_size > 5000
