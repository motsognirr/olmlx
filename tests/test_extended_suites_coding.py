"""Tests for olmlx.bench.extended_suites: coding suite loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmlx.bench.extended_suites import load_humaneval_plus, load_mbpp_plus


@pytest.fixture
def coding_cache(tmp_path, monkeypatch):
    """Populate a fake bench cache from tests/fixtures/bench_cache/."""
    monkeypatch.setenv("OLMLX_BENCH_CACHE_DIR", str(tmp_path))
    fixtures = Path(__file__).parent / "fixtures" / "bench_cache"
    (tmp_path / "humanevalplus.json").write_bytes(
        (fixtures / "humanevalplus_sample.json").read_bytes()
    )
    (tmp_path / "mbppplus.json").write_bytes(
        (fixtures / "mbppplus_sample.json").read_bytes()
    )
    return tmp_path


class TestHumanEvalPlus:
    def test_loads_from_cache(self, coding_cache):
        prompts = load_humaneval_plus(n=None)
        assert len(prompts) == 2

    def test_subset_size(self, coding_cache):
        prompts = load_humaneval_plus(n=1)
        assert len(prompts) == 1

    def test_each_prompt_has_code_exec_grader(self, coding_cache):
        for p in load_humaneval_plus(n=None):
            assert p.grader == "code_exec"
            assert p.expected["entry_point"]
            assert "def check" in p.expected["tests"]

    def test_max_tokens_4096(self, coding_cache):
        for p in load_humaneval_plus(n=None):
            assert p.max_tokens == 4096

    def test_category(self, coding_cache):
        for p in load_humaneval_plus(n=None):
            assert p.category == "humaneval-plus"


class TestMbppPlus:
    def test_loads_from_cache(self, coding_cache):
        prompts = load_mbpp_plus(n=None)
        assert len(prompts) == 1

    def test_each_prompt_has_code_exec_grader(self, coding_cache):
        for p in load_mbpp_plus(n=None):
            assert p.grader == "code_exec"
            assert "def check" in p.expected["tests"]
