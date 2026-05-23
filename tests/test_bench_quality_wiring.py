"""Tests for wiring quality graders into the bench runner.

Covers the integration glue (prompt-set selection, grader application,
quality serialization, scenario aggregation) — the graders themselves are
tested in test_bench_quality.py.
"""

from __future__ import annotations

from olmlx.bench.prompts import PROMPTS
from olmlx.bench.results import PromptResult, RunResult, ScenarioResult
from olmlx.bench.runner import _worker_timeout, apply_graders, build_prompts
from olmlx.bench.quality import QualityResult
from olmlx.bench.task_prompts import PROMPT_SETS


class TestBuildPrompts:
    def test_throughput_set_matches_canonical_list(self):
        # Tie the assertion to the source list so adding a throughput probe
        # in prompts.py doesn't silently require a test edit.
        assert build_prompts("throughput") == list(PROMPTS)
        assert all(p.grader is None for p in build_prompts("throughput"))

    def test_quality_set_is_all_graded(self):
        prompts = build_prompts("quality")
        expected_size = sum(len(v) for v in PROMPT_SETS.values())
        assert len(prompts) == expected_size
        assert all(p.grader is not None for p in prompts)

    def test_all_set_is_throughput_plus_quality(self):
        # Relationship-based: survives any future addition to either set.
        assert len(build_prompts("all")) == len(build_prompts("throughput")) + len(
            build_prompts("quality")
        )

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
        # Invariant: r.grader is set iff r.quality is set. A failed request
        # leaves both as None so consumers don't have to special-case the
        # "graded prompt but no verdict" combination.
        assert results[0].grader is None
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

    def test_code_exec_enabled_injects_flag(self, monkeypatch):
        # Verifies the injection contract without actually running a
        # subprocess — the real code_exec execution is covered in
        # test_bench_quality.py. Stubs ``grade`` in the runner namespace
        # to capture what apply_graders hands it.
        captured: dict = {}

        def fake_grade(grader_name, output, expected):
            captured["grader"] = grader_name
            captured["expected"] = expected
            return QualityResult(grader=grader_name, passed=True, score=1.0, detail="")

        monkeypatch.setattr("olmlx.bench.runner.grade", fake_grade)

        results = [self._result("p1", "irrelevant")]
        original_expected = {
            "prompt": "",
            "tests": "def check(f):\n    assert f() == 1\n",
            "entry_point": "f",
        }
        prompts = [{"name": "p1", "grader": "code_exec", "expected": original_expected}]
        apply_graders(results, prompts, enable_code_exec=True)

        assert captured["grader"] == "code_exec"
        assert captured["expected"]["_enabled"] is True
        # Caller's expected dict must not be mutated with the private flag.
        assert "_enabled" not in original_expected
        assert results[0].quality.passed is True


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


class TestRunResultMetadata:
    """Saved runs must record which prompt set + bench env they ran with so
    an operator can tell a think run apart from a no-think run, or a
    throughput run apart from a quality run, from the JSON alone."""

    def _empty_run(self, **kwargs):
        return RunResult(
            model="m",
            timestamp="20260101T000000Z",
            git_sha=None,
            scenarios=[],
            **kwargs,
        )

    def test_prompt_set_round_trips(self):
        run = self._empty_run(prompt_set="quality")
        assert RunResult.from_dict(run.to_dict()).prompt_set == "quality"

    def test_bench_env_round_trips(self):
        run = self._empty_run(bench_env={"OLMLX_BENCH_THINK": "true"})
        restored = RunResult.from_dict(run.to_dict())
        assert restored.bench_env == {"OLMLX_BENCH_THINK": "true"}

    def test_legacy_run_without_new_fields_loads(self):
        # Old run JSONs (pre-this-PR) have no prompt_set or bench_env.
        # Loading must default both rather than raise.
        legacy = {
            "model": "m",
            "timestamp": "20260101T000000Z",
            "git_sha": None,
            "max_tokens_override": None,
            "scenarios": [],
        }
        run = RunResult.from_dict(legacy)
        assert run.prompt_set is None
        assert run.bench_env == {}


class TestWorkerTimeout:
    """`OLMLX_BENCH_WORKER_TIMEOUT` must never silently disable the kill
    switch (e.g. via ``inf``) and never silently swallow a typo'd value.
    """

    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("OLMLX_BENCH_WORKER_TIMEOUT", raising=False)
        assert _worker_timeout() == 600.0

    def test_positive_value_used(self, monkeypatch):
        monkeypatch.setenv("OLMLX_BENCH_WORKER_TIMEOUT", "1800")
        assert _worker_timeout() == 1800.0

    def test_inf_rejected(self, monkeypatch, caplog):
        # `subprocess.run(timeout=inf)` waits forever — must never get there.
        monkeypatch.setenv("OLMLX_BENCH_WORKER_TIMEOUT", "inf")
        with caplog.at_level("WARNING", logger="olmlx.bench.runner"):
            assert _worker_timeout() == 600.0
        assert any("inf" in rec.getMessage().lower() for rec in caplog.records)

    def test_nan_rejected(self, monkeypatch):
        monkeypatch.setenv("OLMLX_BENCH_WORKER_TIMEOUT", "nan")
        assert _worker_timeout() == 600.0

    def test_zero_rejected(self, monkeypatch):
        monkeypatch.setenv("OLMLX_BENCH_WORKER_TIMEOUT", "0")
        assert _worker_timeout() == 600.0

    def test_negative_rejected(self, monkeypatch):
        monkeypatch.setenv("OLMLX_BENCH_WORKER_TIMEOUT", "-5")
        assert _worker_timeout() == 600.0

    def test_unparseable_warns(self, monkeypatch, caplog):
        monkeypatch.setenv("OLMLX_BENCH_WORKER_TIMEOUT", "abc")
        with caplog.at_level("WARNING", logger="olmlx.bench.runner"):
            assert _worker_timeout() == 600.0
        assert any("abc" in rec.getMessage() for rec in caplog.records)
