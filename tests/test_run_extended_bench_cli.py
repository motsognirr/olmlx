"""Smoke test the CLI orchestrator's argparse + tier-dispatch logic."""

from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import run_extended_bench  # type: ignore[import-not-found]


class TestArgParser:
    def test_default_output_dir(self):
        args = run_extended_bench.parse_args(["--models-config", "x.json"])
        assert args.output.name == "extended-2026-05"

    def test_only_flag(self):
        args = run_extended_bench.parse_args(
            ["--models-config", "x.json", "--only", "foo", "--only", "bar"]
        )
        assert args.only == ["foo", "bar"]


class TestSelectModels:
    def test_filters_to_only_list(self, tmp_path):
        config = {
            "alpha:latest": {"hf_path": "org/alpha"},
            "beta:latest": {"hf_path": "org/beta"},
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        selection = run_extended_bench.select_models(config_path, only=["org/alpha"])
        assert selection == [("org/alpha", "alpha:latest")]

    def test_returns_all_when_only_empty(self, tmp_path):
        config = {
            "alpha:latest": {"hf_path": "org/alpha"},
            "beta:latest": "org/beta",
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        selection = run_extended_bench.select_models(config_path, only=[])
        hfs = {hf for hf, _ in selection}
        assert hfs == {"org/alpha", "org/beta"}
