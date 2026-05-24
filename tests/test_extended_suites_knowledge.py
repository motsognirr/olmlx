"""Tests for MMLU-Pro, GPQA-Diamond, IFEval loaders + RULER generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmlx.bench.extended_suites import (
    load_gpqa_diamond,
    load_ifeval,
    load_mmlu_pro,
    make_ruler_niah,
)


@pytest.fixture
def knowledge_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OLMLX_BENCH_CACHE_DIR", str(tmp_path))
    fixtures = Path(__file__).parent / "fixtures" / "bench_cache"
    for name in ("mmlu_pro.json", "gpqa_diamond.json", "ifeval.json"):
        src = fixtures / name.replace(".json", "_sample.json")
        (tmp_path / name).write_bytes(src.read_bytes())
    return tmp_path


class TestMmluPro:
    def test_loads(self, knowledge_cache):
        prompts = load_mmlu_pro(n=None)
        assert len(prompts) == 2

    def test_regex_grader(self, knowledge_cache):
        for p in load_mmlu_pro(n=None):
            assert p.grader == "regex_match"
            assert p.expected["answer"] in (
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
            )

    def test_max_tokens_1024(self, knowledge_cache):
        for p in load_mmlu_pro(n=None):
            assert p.max_tokens == 1024


class TestGpqaDiamond:
    def test_loads(self, knowledge_cache):
        prompts = load_gpqa_diamond(n=None)
        assert len(prompts) == 1

    def test_answer_is_letter(self, knowledge_cache):
        for p in load_gpqa_diamond(n=None):
            assert p.expected["answer"] in ("A", "B", "C", "D")


class TestIfeval:
    def test_loads(self, knowledge_cache):
        prompts = load_ifeval(n=None)
        assert len(prompts) == 1

    def test_grader_is_ifeval(self, knowledge_cache):
        for p in load_ifeval(n=None):
            assert p.grader == "ifeval"
            assert p.expected["instruction_id_list"] == ["keywords:existence"]


class TestRulerNiah:
    def test_generates_n_prompts(self):
        prompts = make_ruler_niah(context_tokens=4096, n=5)
        assert len(prompts) == 5

    def test_grader_is_contains_with_needle(self):
        prompts = make_ruler_niah(context_tokens=4096, n=3)
        for p in prompts:
            assert p.grader == "contains"
            assert len(p.expected["substrings"]) == 1

    def test_is_deterministic(self):
        a = make_ruler_niah(context_tokens=4096, n=5)
        b = make_ruler_niah(context_tokens=4096, n=5)
        assert [p.name for p in a] == [p.name for p in b]
        assert [p.expected for p in a] == [p.expected for p in b]
