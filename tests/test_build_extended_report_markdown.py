"""Tests for the markdown report renderer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_extended_report  # type: ignore[import-not-found]


def _fixture_run(tmp_path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    fixtures = Path(__file__).parent / "fixtures" / "extended_run" / "raw"
    for src in fixtures.glob("*.json"):
        (raw / src.name).write_bytes(src.read_bytes())
    return tmp_path


class TestHeadlineTable:
    def test_contains_all_models(self, tmp_path):
        run = _fixture_run(tmp_path)
        results = build_extended_report.load_results(run)
        md = build_extended_report.render_headline_table(results)
        assert "qwen-tiny" in md
        assert "big-moe" in md

    def test_em_dash_for_missing_extended_cells(self, tmp_path):
        # big-moe is core_only, so its extended suite columns should be em-dashes.
        run = _fixture_run(tmp_path)
        results = build_extended_report.load_results(run)
        md = build_extended_report.render_headline_table(results)
        # Find the big-moe row.
        big_row = next(line for line in md.splitlines() if "big-moe" in line)
        # The row should contain the em-dash placeholder.
        assert "—" in big_row


class TestBuildReport:
    def test_writes_readme_and_charts(self, tmp_path):
        run = _fixture_run(tmp_path)
        build_extended_report.build_report(run)
        assert (run / "README.md").exists()
        assert (run / "charts" / "frontier.png").exists()
        assert (run / "charts" / "suite_heatmap.png").exists()
        assert (run / "charts" / "quant_pairs.png").exists()
        text = (run / "README.md").read_text()
        assert "# Extended benchmark" in text
        assert "## Methodology" in text
        assert "## Findings" in text
        assert "## Future research directions" in text
