"""Tests for olmlx.bench.task_prompts — shape of the curated mini-sets."""

from __future__ import annotations

from olmlx.bench.task_prompts import (
    GSM8K_MINI,
    HUMANEVAL_MINI,
    MMLU_MINI,
    PROMPT_SETS,
)


class TestGSM8K:
    def test_size(self):
        assert len(GSM8K_MINI) == 20

    def test_unique_names(self):
        names = [p.name for p in GSM8K_MINI]
        assert len(names) == len(set(names))

    def test_all_use_numeric_grader(self):
        for p in GSM8K_MINI:
            assert p.grader == "numeric"
            assert "answer" in p.expected
            assert isinstance(p.expected["answer"], (int, float))


class TestMMLU:
    def test_size(self):
        assert len(MMLU_MINI) == 20

    def test_all_use_regex_grader_with_valid_answer(self):
        for p in MMLU_MINI:
            assert p.grader == "regex_match"
            assert p.expected["answer"] in ("A", "B", "C", "D")

    def test_answer_distribution_not_trivially_skewed(self):
        # If all answers were the same letter, a "always answer C" model
        # would pass 100% — not a useful benchmark. Sanity-check spread.
        answers = [p.expected["answer"] for p in MMLU_MINI]
        distinct = set(answers)
        assert len(distinct) >= 3


class TestHumanEval:
    def test_size(self):
        assert len(HUMANEVAL_MINI) == 10

    def test_all_use_code_exec_grader(self):
        for p in HUMANEVAL_MINI:
            assert p.grader == "code_exec"
            assert p.expected["entry_point"]
            assert "check(candidate)" in p.expected["tests"]
            assert "def " in p.expected["prompt"]


class TestPromptSetsRegistry:
    def test_has_all_sets(self):
        assert set(PROMPT_SETS) == {"gsm8k", "mmlu", "humaneval"}

    def test_values_are_prompt_lists(self):
        for key, prompts in PROMPT_SETS.items():
            assert len(prompts) > 0, key
