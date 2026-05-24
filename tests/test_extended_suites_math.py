"""Tests for olmlx.bench.extended_suites: math suite loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmlx.bench.extended_suites import load_gsm8k, load_math500


@pytest.fixture
def math_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OLMLX_BENCH_CACHE_DIR", str(tmp_path))
    fixtures = Path(__file__).parent / "fixtures" / "bench_cache"
    (tmp_path / "gsm8k.json").write_bytes((fixtures / "gsm8k_sample.json").read_bytes())
    (tmp_path / "math500.json").write_bytes(
        (fixtures / "math500_sample.json").read_bytes()
    )
    return tmp_path


class TestGsm8k:
    def test_loads_from_cache(self, math_cache):
        prompts = load_gsm8k(n=None)
        assert len(prompts) == 3

    def test_extracts_integer_answer(self, math_cache):
        prompts = load_gsm8k(n=None)
        assert prompts[0].expected["answer"] == 18
        assert prompts[1].expected["answer"] == 16

    def test_grader_is_numeric(self, math_cache):
        for p in load_gsm8k(n=None):
            assert p.grader == "numeric"

    def test_max_tokens_4096(self, math_cache):
        for p in load_gsm8k(n=None):
            assert p.max_tokens == 4096


class TestMath500:
    def test_loads_from_cache(self, math_cache):
        prompts = load_math500(n=None)
        assert len(prompts) == 2

    def test_grader_is_regex_match_for_boxed(self, math_cache):
        prompts = load_math500(n=None)
        assert prompts[0].grader == "regex_match"
        assert prompts[0].expected["answer"] == "4"
