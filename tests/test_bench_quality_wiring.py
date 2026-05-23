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

    def test_all_set_currently_has_unique_names(self):
        # Build succeeds today — no collisions. Catches future regressions
        # at construction rather than letting them mis-grade a saved run.
        build_prompts("all")

    def test_to_dict_carries_grader_fields_into_apply_graders(self):
        # Integration check: ``build_prompts("quality")`` →
        # ``BenchPrompt.to_dict()`` → ``apply_graders`` must round-trip
        # the ``grader`` and ``expected`` fields. If ``to_dict()`` ever
        # silently dropped either, the quality run would report
        # ``quality 0/0`` with no error. This test pins the contract.
        prompts = build_prompts("quality")
        prompts_data = [p.to_dict() for p in prompts]
        # Pick the first graded prompt and synthesise a "pass" result.
        first = prompts[0]
        assert first.grader is not None
        result = PromptResult(
            first.name,
            first.category,
            output_text="",  # filled below per grader
            status_code=200,
        )
        # Numeric is the easiest "pass" output to synthesise; if the first
        # prompt isn't numeric, skip the content check and just confirm the
        # grader was applied (verdict-shape, not verdict-value).
        if first.grader == "numeric":
            result.output_text = f"#### {first.expected['answer']}"
        apply_graders([result], prompts_data, enable_code_exec=False)
        assert result.grader == first.grader, (
            "BenchPrompt.to_dict() did not surface grader to apply_graders"
        )

    def test_quality_set_raises_on_within_set_duplicate(self, monkeypatch):
        # Same collision concern as ``"all"`` but within the graded set:
        # if two task_prompts entries ever share a name, apply_graders's
        # by-name dict silently keeps the last one. Synthesize a collision
        # via PROMPT_SETS monkeypatch and verify build_prompts catches it
        # for the ``"quality"`` path too — not just ``"all"``.
        import pytest

        from olmlx.bench.prompts import BenchPrompt

        a = BenchPrompt(
            name="dup",
            category="gsm8k",
            messages=[{"role": "user", "content": "x"}],
            grader="numeric",
            expected={"answer": 1},
        )
        b = BenchPrompt(
            name="dup",
            category="mmlu",
            messages=[{"role": "user", "content": "y"}],
            grader="numeric",
            expected={"answer": 2},
        )
        monkeypatch.setattr(
            "olmlx.bench.task_prompts.PROMPT_SETS", {"set_a": [a], "set_b": [b]}
        )
        with pytest.raises(ValueError, match="Duplicate prompt names within quality"):
            build_prompts("quality")

    def test_all_set_raises_on_duplicate_names(self, monkeypatch):
        import pytest

        from olmlx.bench.prompts import BenchPrompt
        from olmlx.bench.task_prompts import PROMPT_SETS

        # Synthesize a collision: pick any real graded prompt name and add
        # a fake throughput probe with the same name. ``build_prompts`` must
        # reject the union with a clear error rather than silently shipping
        # the collision into apply_graders.
        any_graded_name = next(iter(PROMPT_SETS.values()))[0].name
        dup = BenchPrompt(
            name=any_graded_name,
            category="throughput",
            messages=[{"role": "user", "content": "hi"}],
        )
        monkeypatch.setattr("olmlx.bench.runner.PROMPTS", (dup,))
        with pytest.raises(ValueError, match="Duplicate prompt names"):
            build_prompts("all")


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

    def test_code_exec_disabled_does_not_call_grade(self, monkeypatch):
        # Runner enforces the code_exec gate itself; ``grade()`` must not
        # be invoked at all when ``enable_code_exec=False``. Otherwise a
        # future change to the grader's internal opt-in flag would silently
        # let untrusted code execute. Stubs ``grade`` to assert non-call.
        called = []

        def watch_grade(*args, **kwargs):
            called.append(args)
            return QualityResult("code_exec", True, 1.0, "")

        monkeypatch.setattr("olmlx.bench.runner.grade", watch_grade)
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
        assert called == []
        assert results[0].quality is not None
        assert results[0].quality.passed is None
        assert results[0].grader == "code_exec"

    def test_returns_code_exec_excluded_count(self):
        # apply_graders should *report* the disabled-code_exec count
        # directly so the runner doesn't have to re-infer it from
        # ``passed is None`` (which would also match grader exceptions).
        prompts = [
            {
                "name": "p1",
                "grader": "code_exec",
                "expected": {
                    "prompt": "",
                    "tests": "",
                    "entry_point": "f",
                },
            },
            {"name": "p2", "grader": "numeric", "expected": {"answer": 1}},
        ]
        results = [
            self._result("p1", "anything"),
            self._result("p2", "#### 1"),
        ]
        stats = apply_graders(results, prompts, enable_code_exec=False)
        assert stats == {"code_exec_excluded": 1}

    def test_returns_zero_excluded_when_code_exec_enabled(self, monkeypatch):
        # Stub grade() so we don't actually spawn a subprocess.
        monkeypatch.setattr(
            "olmlx.bench.runner.grade",
            lambda g, o, e: QualityResult(g, True, 1.0, ""),
        )
        prompts = [{"name": "p1", "grader": "code_exec", "expected": {}}]
        results = [self._result("p1", "x")]
        stats = apply_graders(results, prompts, enable_code_exec=True)
        assert stats == {"code_exec_excluded": 0}

    def test_grade_exception_caught_and_logged(self, monkeypatch, caplog):
        # grade() is contracted not to raise, but a future regression
        # mustn't abort the whole bench run and lose accumulated results.
        # The fallback preserves the ``r.grader ⇔ r.quality`` invariant.
        def boom(grader_name, output, expected):
            raise RuntimeError("simulated grader regression")

        monkeypatch.setattr("olmlx.bench.runner.grade", boom)
        prompts = [{"name": "p1", "grader": "numeric", "expected": {"answer": 1}}]
        results = [self._result("p1", "#### 1")]
        with caplog.at_level("WARNING", logger="olmlx.bench.runner"):
            apply_graders(results, prompts, enable_code_exec=False)
        assert results[0].grader == "numeric"
        assert results[0].quality is not None
        assert results[0].quality.passed is None
        assert "simulated grader regression" in results[0].quality.detail
        assert any("Grader" in rec.getMessage() for rec in caplog.records)

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

    def test_indeterminate_graded_excluded_from_tally(self):
        # Prompts that have a grader + a QualityResult but ``passed is
        # None`` (e.g. ``code_exec`` when ``enable_code_exec`` is false)
        # must not count toward the denominator — otherwise the reported
        # pass rate is dragged down by prompts that were never actually
        # judged.
        sc = ScenarioResult(
            scenario_name="baseline",
            scenario_description="",
            env_overrides={},
            prompt_results=[
                self._graded("a", True),
                PromptResult(
                    "b",
                    "humaneval",
                    "x",
                    200,
                    grader="code_exec",
                    quality=QualityResult(
                        "code_exec", None, None, "code_exec disabled"
                    ),
                ),
            ],
        )
        assert sc.quality_summary() == (1, 1)


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


class TestBenchEnvCapture:
    """``bench_env`` should record only experiment-defining variables, not
    operational knobs — the timeout is a safety rail that doesn't change
    how to interpret an A/B result."""

    def test_think_is_captured(self, monkeypatch):
        from olmlx.bench.runner import _capture_bench_env

        monkeypatch.setenv("OLMLX_BENCH_THINK", "true")
        assert _capture_bench_env() == {"OLMLX_BENCH_THINK": "true"}

    def test_worker_timeout_is_not_captured(self, monkeypatch):
        from olmlx.bench.runner import _capture_bench_env

        monkeypatch.setenv("OLMLX_BENCH_WORKER_TIMEOUT", "1800")
        monkeypatch.delenv("OLMLX_BENCH_THINK", raising=False)
        # Operational knob, intentionally excluded from the experiment record.
        assert _capture_bench_env() == {}

    def test_unrecognized_think_is_not_captured(self, monkeypatch):
        # A typo'd value (e.g. ``"tru"``) falls back to engine default in
        # the worker. Recording it in bench_env would imply think was
        # toggled when the run actually used engine default — defeats the
        # self-describing-A/B purpose.
        from olmlx.bench.runner import _capture_bench_env

        monkeypatch.setenv("OLMLX_BENCH_THINK", "tru")
        assert _capture_bench_env() == {}

    def test_recognized_think_is_captured_verbatim(self, monkeypatch):
        from olmlx.bench.runner import _capture_bench_env

        monkeypatch.setenv("OLMLX_BENCH_THINK", "False")
        assert _capture_bench_env() == {"OLMLX_BENCH_THINK": "False"}


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
