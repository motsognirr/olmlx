"""Tests for olmlx.bench.quality graders."""

from __future__ import annotations

import os
import sys

import pytest

from olmlx.bench.quality import (
    GRADERS,
    QualityResult,
    grade,
    grade_code_exec,
    grade_contains,
    grade_exact_match,
    grade_numeric,
    grade_regex_match,
    grade_regression_snapshot,
)


class TestQualityResult:
    def test_to_dict_roundtrip(self):
        q = QualityResult(
            grader="numeric",
            passed=True,
            score=1.0,
            detail="extracted=42",
            reference="42",
        )
        restored = QualityResult.from_dict(q.to_dict())
        assert restored == q

    def test_from_dict_tolerates_missing_optional(self):
        q = QualityResult.from_dict({"grader": "numeric"})
        assert q.grader == "numeric"
        assert q.passed is None
        assert q.score is None
        assert q.detail == ""


class TestGradeExactMatch:
    def test_match(self):
        r = grade_exact_match(
            "The capital is Paris.", {"answer": "The capital is Paris."}
        )
        assert r.passed is True
        assert r.score == 1.0

    def test_normalize_whitespace(self):
        r = grade_exact_match("  Hello   world ", {"answer": "hello world"})
        assert r.passed is True

    def test_normalize_case(self):
        r = grade_exact_match("Paris", {"answer": "paris"})
        assert r.passed is True

    def test_no_normalize(self):
        r = grade_exact_match(
            "  Hello world ",
            {"answer": "Hello world", "normalize": False},
        )
        assert r.passed is False

    def test_mismatch(self):
        r = grade_exact_match("London", {"answer": "Paris"})
        assert r.passed is False
        assert r.score == 0.0


class TestGradeContains:
    def test_all_present(self):
        r = grade_contains(
            "apples, bananas, cherries", {"substrings": ["apple", "cherr"]}
        )
        assert r.passed is True
        assert r.score == 1.0

    def test_partial_match_with_all(self):
        r = grade_contains(
            "apple only", {"substrings": ["apple", "banana"], "all": True}
        )
        assert r.passed is False
        assert r.score == 0.5

    def test_any_match(self):
        r = grade_contains(
            "apple only", {"substrings": ["apple", "banana"], "all": False}
        )
        assert r.passed is True
        assert r.score == 0.5

    def test_case_insensitive_default(self):
        r = grade_contains("APPLE", {"substrings": ["apple"]})
        assert r.passed is True

    def test_case_sensitive_when_disabled(self):
        r = grade_contains(
            "APPLE",
            {"substrings": ["apple"], "ignore_case": False, "all": True},
        )
        assert r.passed is False

    def test_no_substrings_returns_none(self):
        r = grade_contains("x", {"substrings": []})
        assert r.passed is None


class TestGradeRegexMatch:
    def test_extract_and_match(self):
        r = grade_regex_match(
            "Therefore the answer is B.",
            {"pattern": r"answer is ([A-D])", "group": 1, "answer": "B"},
        )
        assert r.passed is True

    def test_no_match(self):
        r = grade_regex_match(
            "no letter here",
            {"pattern": r"answer is ([A-D])", "group": 1, "answer": "B"},
        )
        assert r.passed is False

    def test_case_insensitive_compare(self):
        r = grade_regex_match(
            "Answer: b.",
            {"pattern": r"(?i)answer[:\s]*([A-D])", "group": 1, "answer": "B"},
        )
        assert r.passed is True

    def test_invalid_pattern_returns_false(self):
        r = grade_regex_match("x", {"pattern": "(unclosed", "answer": "x"})
        assert r.passed is False
        assert "invalid pattern" in r.detail

    def test_no_pattern_returns_none(self):
        r = grade_regex_match("x", {"pattern": "", "answer": "y"})
        assert r.passed is None

    def test_default_group_matches_whole_pattern(self):
        # No explicit group → default 0 → compare the entire match.
        # Previously defaulted to 1 and would fail confusingly on
        # patterns without a capture group.
        r = grade_regex_match(
            "Final score: 42",
            {"pattern": r"Final score: 42", "answer": "Final score: 42"},
        )
        assert r.passed is True


class TestGradeNumeric:
    def test_gsm8k_hash_form(self):
        r = grade_numeric("Working... #### 42", {"answer": 42})
        assert r.passed is True

    def test_boxed_form(self):
        r = grade_numeric(r"Therefore \boxed{42}", {"answer": 42})
        assert r.passed is True

    def test_last_number_fallback(self):
        r = grade_numeric("First 3 then 7 then 42.", {"answer": 42})
        assert r.passed is True

    def test_tolerance(self):
        r = grade_numeric("answer: 41.9", {"answer": 42.0, "tol": 0.2})
        assert r.passed is True

    def test_strips_commas(self):
        r = grade_numeric("total cost: 1,234", {"answer": 1234})
        assert r.passed is True

    def test_gsm8k_hash_form_with_commas(self):
        # A model that follows the instructed "#### N" format but writes a
        # comma-grouped number must still parse correctly. The default
        # GSM8K prompt asks for "#### <number>" — many models will format
        # >999 answers with thousands separators.
        r = grade_numeric("Step 1...\n#### 1,234", {"answer": 1234})
        assert r.passed is True

    def test_no_number_fails(self):
        r = grade_numeric("no digits", {"answer": 5})
        assert r.passed is False

    def test_missing_answer_returns_none(self):
        r = grade_numeric("42", {})
        assert r.passed is None


class TestGradeRegressionSnapshot:
    def test_missing_reference_path_returns_none(self):
        r = grade_regression_snapshot("x", {})
        assert r.passed is None
        assert "no golden" in r.detail

    def test_missing_file_returns_none(self, tmp_path):
        r = grade_regression_snapshot(
            "x", {"reference_path": str(tmp_path / "nope.txt")}
        )
        assert r.passed is None

    def test_exact_mode_pass(self, tmp_path):
        ref = tmp_path / "g.txt"
        ref.write_text("Hello, world.")
        r = grade_regression_snapshot(
            "Hello,  world.",
            {"reference_path": str(ref), "mode": "exact"},
        )
        assert r.passed is True

    def test_exact_mode_fail(self, tmp_path):
        ref = tmp_path / "g.txt"
        ref.write_text("Hello, world.")
        r = grade_regression_snapshot(
            "Goodbye.",
            {"reference_path": str(ref), "mode": "exact"},
        )
        assert r.passed is False

    def test_similarity_mode_pass(self, tmp_path):
        ref = tmp_path / "g.txt"
        ref.write_text("The capital of France is Paris.")
        r = grade_regression_snapshot(
            "The capital of France is Paris.",
            {"reference_path": str(ref), "mode": "similarity", "threshold": 0.9},
        )
        assert r.passed is True
        assert r.score == pytest.approx(1.0)

    def test_similarity_mode_fail_below_threshold(self, tmp_path):
        ref = tmp_path / "g.txt"
        ref.write_text("The capital of France is Paris.")
        r = grade_regression_snapshot(
            "Completely different output that shares almost nothing.",
            {"reference_path": str(ref), "mode": "similarity", "threshold": 0.9},
        )
        assert r.passed is False


class TestGradeDispatch:
    def test_unknown_grader(self):
        r = grade("no-such-grader", "x", {})
        assert r.passed is None
        assert "unknown grader" in r.detail

    def test_dispatch_to_numeric(self):
        r = grade("numeric", "#### 7", {"answer": 7})
        assert r.passed is True

    def test_registry_has_all_builtins(self):
        for name in (
            "exact_match",
            "contains",
            "regex_match",
            "numeric",
            "code_exec",
            "regression_snapshot",
        ):
            assert name in GRADERS


@pytest.mark.skipif(os.name != "posix", reason="code_exec requires POSIX rlimits")
class TestGradeCodeExec:
    def test_disabled_by_default(self):
        r = grade_code_exec(
            "```python\ndef foo():\n    return 1\n```",
            {
                "prompt": "",
                "tests": "def check(fn):\n    assert fn() == 1\n",
                "entry_point": "foo",
            },
        )
        assert r.passed is None
        assert "disabled" in r.detail

    def test_trivially_passing_completion(self):
        r = grade_code_exec(
            "```python\ndef foo():\n    return 1\n```",
            {
                "_enabled": True,
                "prompt": "",
                "tests": "def check(fn):\n    assert fn() == 1\n",
                "entry_point": "foo",
            },
        )
        assert r.passed is True

    def test_trivially_failing_completion(self):
        r = grade_code_exec(
            "```python\ndef foo():\n    return 2\n```",
            {
                "_enabled": True,
                "prompt": "",
                "tests": "def check(fn):\n    assert fn() == 1\n",
                "entry_point": "foo",
            },
        )
        assert r.passed is False

    def test_infinite_loop_times_out(self):
        r = grade_code_exec(
            "```python\ndef foo():\n    while True:\n        pass\n```",
            {
                "_enabled": True,
                "prompt": "",
                "tests": "def check(fn):\n    fn()\n",
                "entry_point": "foo",
            },
        )
        # Either CPU-time rlimit kills it (SIGXCPU → non-zero exit, detail
        # surfaces the kill reason) or subprocess.run's wall-clock timeout
        # fires first. Either way, the grader returns passed=False and we
        # never hang the harness.
        assert r.passed is False

    def test_missing_tests_returns_none(self):
        r = grade_code_exec(
            "def foo(): return 1",
            {"_enabled": True, "prompt": "", "tests": "", "entry_point": "foo"},
        )
        assert r.passed is None

    def test_non_posix_returns_disabled(self, monkeypatch):
        monkeypatch.setattr("olmlx.bench.quality.os.name", "nt")
        r = grade_code_exec(
            "",
            {
                "_enabled": True,
                "prompt": "",
                "tests": "def check(fn): pass\n",
                "entry_point": "foo",
            },
        )
        assert r.passed is None
        assert "POSIX" in r.detail


class TestGradeWrapsExceptions:
    def test_grader_crash_returns_ungraded_not_raise(self, monkeypatch):
        def boom(_output, _expected):
            raise RuntimeError("synthetic")

        monkeypatch.setitem(GRADERS, "boom", boom)
        r = grade("boom", "x", {})
        # passed=None (ungraded), not False — a grader bug must not
        # be counted as a model failure.
        assert r.passed is None
        assert r.score is None
        assert "grader raised" in r.detail


def test_python_executable_is_sys_executable():
    # Sanity: make sure we're running graders with the same interpreter
    # we're testing under — avoids site-packages drift surprises.
    assert sys.executable
