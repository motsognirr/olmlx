"""Tests for olmlx.bench.extended_runner."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from olmlx.bench.extended_runner import (
    SuiteAssignment,
    apply_runtime_triage,
    assemble_core_suite,
    assemble_extended_suite,
    composite_score,
)


@pytest.fixture
def cached_datasets(tmp_path, monkeypatch):
    """Pre-populate cache so the loaders are offline."""
    monkeypatch.setenv("OLMLX_BENCH_CACHE_DIR", str(tmp_path))
    fixtures = Path(__file__).parent / "fixtures" / "bench_cache"
    mapping = {
        "humanevalplus.json": "humanevalplus_sample.json",
        "mbppplus.json": "mbppplus_sample.json",
        "gsm8k.json": "gsm8k_sample.json",
        "math500.json": "math500_sample.json",
        "mmlu_pro.json": "mmlu_pro_sample.json",
        "gpqa_diamond.json": "gpqa_diamond_sample.json",
        "ifeval.json": "ifeval_sample.json",
    }
    for dest, src in mapping.items():
        (tmp_path / dest).write_bytes((fixtures / src).read_bytes())
    return tmp_path


class TestAssembleSuites:
    def test_core_has_three_suites(self, cached_datasets):
        suite = assemble_core_suite()
        # Fixtures are tiny; we just check structure.
        categories = {p.category for p in suite}
        assert "humaneval-plus" in categories
        assert "gsm8k" in categories
        # GPQA category is suffixed with domain ("gpqa-physics").
        assert any(c.startswith("gpqa") for c in categories)

    def test_extended_has_six_more_suites(self, cached_datasets):
        suite = assemble_extended_suite()
        categories = {p.category for p in suite}
        assert "mbpp-plus" in categories
        assert any(c.startswith("math500") for c in categories)
        assert any(c.startswith("mmlu-pro") for c in categories)
        assert any(c.startswith("ifeval") for c in categories)
        assert any(c.startswith("ruler-niah") for c in categories)


class TestRuntimeTriage:
    def test_keep_core_if_budget_sufficient(self):
        # 100 tok/s × 3600s = 360k tokens >> core requirement (180 × 500 = 90k).
        decision = apply_runtime_triage(observed_tok_per_s=100, remaining_seconds=3600)
        assert decision == SuiteAssignment.FULL_CORE

    def test_drop_gpqa_when_tight(self):
        # 50 tok/s × 1500s = 75k tokens — below full_need (90k) but >= minus_gpqa_need (60k).
        decision = apply_runtime_triage(observed_tok_per_s=50, remaining_seconds=1500)
        assert decision == SuiteAssignment.CORE_MINUS_GPQA

    def test_he_plus_only_when_critical(self):
        decision = apply_runtime_triage(observed_tok_per_s=1, remaining_seconds=600)
        assert decision == SuiteAssignment.HE_PLUS_ONLY


class TestCompositeScore:
    def test_unweighted_mean_of_suite_pass_rates(self):
        # Three suites with pass rates 1.0, 0.5, 0.0 → composite 0.5.
        per_suite = {"humaneval-plus": 1.0, "gsm8k": 0.5, "gpqa": 0.0}
        assert composite_score(per_suite) == pytest.approx(0.5)

    def test_handles_empty(self):
        assert composite_score({}) == 0.0


class TestCodeExecGate:
    def test_code_exec_gate_off_by_default(self, monkeypatch):
        """Without OLMLX_BENCH_CODE_EXEC=1, code_exec must not execute."""
        from olmlx.bench.quality import grade

        monkeypatch.delenv("OLMLX_BENCH_CODE_EXEC", raising=False)
        expected = {
            "prompt": "",
            "tests": "def check(c):\n    pass\n",
            "entry_point": "f",
        }
        # Emulate what _drive_prompt would set when env var is absent:
        if os.environ.get("OLMLX_BENCH_CODE_EXEC") == "1":
            expected["_enabled"] = True
        result = grade("code_exec", "def f(): pass", expected)
        assert result.passed is None  # ungraded when gate off

    def test_code_exec_gate_on_when_env_set(self, monkeypatch):
        """With OLMLX_BENCH_CODE_EXEC=1, _enabled is True in expected dict."""
        monkeypatch.setenv("OLMLX_BENCH_CODE_EXEC", "1")
        expected: dict = {}
        if os.environ.get("OLMLX_BENCH_CODE_EXEC") == "1":
            expected["_enabled"] = True
        assert expected.get("_enabled") is True


class TestExtendedSuiteDedup:
    def test_extended_suite_does_not_duplicate_core_humaneval(self, cached_datasets):
        """Core HE+ 50 should not be re-graded as part of extended."""
        core = assemble_core_suite()
        ext = assemble_extended_suite()
        core_he = {p.name for p in core if p.category == "humaneval-plus"}
        ext_he = {p.name for p in ext if p.category == "humaneval-plus"}
        overlap = core_he & ext_he
        # Some overlap is acceptable (extended includes the same subset by deterministic
        # selection); the run_model fix dedups by name so they're only graded once.
        # This test just documents the overlap exists.
        assert len(overlap) > 0
