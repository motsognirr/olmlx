"""Tests for wiring quality graders into the bench runner.

Covers the integration glue (prompt-set selection, grader application,
quality serialization, scenario aggregation) — the graders themselves are
tested in test_bench_quality.py.
"""

from __future__ import annotations

from olmlx.bench.results import PromptResult, ScenarioResult
from olmlx.bench.runner import apply_graders, build_prompts
from olmlx.bench.quality import QualityResult


class TestBuildPrompts:
    def test_throughput_set_has_no_graders(self):
        prompts = build_prompts("throughput")
        assert len(prompts) == 7
        assert all(p.grader is None for p in prompts)

    def test_quality_set_is_all_graded(self):
        prompts = build_prompts("quality")
        # gsm8k(20) + mmlu(20) + humaneval(10)
        assert len(prompts) == 50
        assert all(p.grader is not None for p in prompts)

    def test_all_set_is_union(self):
        assert len(build_prompts("all")) == 57

    def test_unknown_set_raises(self):
        import pytest

        with pytest.raises(ValueError):
            build_prompts("bogus")


class TestApplyGraders:
    def _result(self, name, output, status=200):
        return PromptResult(
            prompt_name=name,
            category="gsm8k",
            output_text=output,
            status_code=status,
        )

    def test_numeric_pass_attaches_quality(self):
        results = [self._result("p1", "The answer is #### 18")]
        prompts = [
            {"name": "p1", "grader": "numeric", "expected": {"answer": 18, "tol": 0.0}}
        ]
        apply_graders(results, prompts, enable_code_exec=False)
        assert results[0].grader == "numeric"
        assert results[0].quality is not None
        assert results[0].quality.passed is True

    def test_numeric_fail(self):
        results = [self._result("p1", "#### 7")]
        prompts = [
            {"name": "p1", "grader": "numeric", "expected": {"answer": 18, "tol": 0.0}}
        ]
        apply_graders(results, prompts, enable_code_exec=False)
        assert results[0].quality.passed is False

    def test_ungraded_prompt_left_alone(self):
        results = [self._result("p1", "anything")]
        prompts = [{"name": "p1", "grader": None, "expected": {}}]
        apply_graders(results, prompts, enable_code_exec=False)
        assert results[0].grader is None
        assert results[0].quality is None

    def test_non_200_not_graded(self):
        results = [self._result("p1", "", status=503)]
        prompts = [{"name": "p1", "grader": "numeric", "expected": {"answer": 18}}]
        apply_graders(results, prompts, enable_code_exec=False)
        assert results[0].quality is None

    def test_code_exec_disabled_by_default(self):
        results = [self._result("p1", "```python\ndef f():\n    return 1\n```")]
        prompts = [
            {
                "name": "p1",
                "grader": "code_exec",
                "expected": {
                    "prompt": "",
                    "tests": "def check(f):\n    assert f() == 1\n",
                    "entry_point": "f",
                },
            }
        ]
        apply_graders(results, prompts, enable_code_exec=False)
        # Grader runs but returns ungraded (passed=None) when not enabled.
        assert results[0].quality is not None
        assert results[0].quality.passed is None

    def test_code_exec_enabled_injects_flag(self):
        results = [self._result("p1", "```python\ndef f():\n    return 1\n```")]
        prompts = [
            {
                "name": "p1",
                "grader": "code_exec",
                "expected": {
                    "prompt": "",
                    "tests": "def check(f):\n    assert f() == 1\n",
                    "entry_point": "f",
                },
            }
        ]
        apply_graders(results, prompts, enable_code_exec=True)
        assert results[0].quality.passed is True
        # Caller's expected dict must not be mutated with the private flag.
        assert "_enabled" not in prompts[0]["expected"]


class TestQualitySerialization:
    def test_prompt_result_round_trips_quality(self):
        q = QualityResult(
            grader="numeric", passed=True, score=1.0, detail="extracted=18"
        )
        pr = PromptResult(
            prompt_name="p1",
            category="gsm8k",
            output_text="#### 18",
            status_code=200,
            grader="numeric",
            quality=q,
        )
        restored = PromptResult.from_dict(pr.to_dict())
        assert restored.grader == "numeric"
        assert restored.quality is not None
        assert restored.quality.passed is True
        assert restored.quality.score == 1.0

    def test_prompt_result_without_quality_round_trips(self):
        pr = PromptResult("p1", "factual", "Paris", 200)
        restored = PromptResult.from_dict(pr.to_dict())
        assert restored.quality is None
        assert restored.grader is None


class TestScenarioQualitySummary:
    def _graded(self, name, passed):
        return PromptResult(
            name,
            "gsm8k",
            "x",
            200,
            grader="numeric",
            quality=QualityResult("numeric", passed, 1.0 if passed else 0.0, ""),
        )

    def test_counts_passed_over_graded(self):
        sc = ScenarioResult(
            scenario_name="baseline",
            scenario_description="",
            env_overrides={},
            prompt_results=[
                self._graded("a", True),
                self._graded("b", False),
                self._graded("c", True),
                PromptResult("d", "factual", "x", 200),  # ungraded
            ],
        )
        assert sc.quality_summary() == (2, 3)

    def test_no_graded_prompts_returns_zero_total(self):
        sc = ScenarioResult(
            scenario_name="baseline",
            scenario_description="",
            env_overrides={},
            prompt_results=[PromptResult("d", "factual", "x", 200)],
        )
        assert sc.quality_summary() == (0, 0)
