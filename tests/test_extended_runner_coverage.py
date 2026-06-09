"""Regression coverage for olmlx.bench.extended_runner orchestration.

Focus: the HTTP driver (`_drive_prompt`, `_warmup`) and the suite
orchestration in `run_model` — scenario selection (triage), subset filtering,
result aggregation, per-prompt persistence, and timeout handling. All HTTP and
grading is mocked; no network, model, or GPU is touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import olmlx.bench.extended_runner as er
from olmlx.bench.extended_runner import (
    ModelRunResult,
    PromptResult,
    SuiteAssignment,
    _drive_prompt,
    _warmup,
    aggregate_per_suite,
    composite_score,
    run_model,
    run_model_sync,
    safe_model_name,
    suite_of,
    write_result,
)


# ---------------------------------------------------------------------------
# Lightweight stand-ins so we never depend on real datasets / loaders.
# ---------------------------------------------------------------------------


@dataclass
class FakePrompt:
    """Mirror of BenchPrompt's fields used by the runner."""

    name: str
    category: str
    messages: list[dict[str, str]] = field(default_factory=list)
    max_tokens: int = 64
    grader: str | None = "exact_match"
    expected: dict[str, Any] = field(default_factory=dict)


def _fake_grade(passed=True, score=1.0, detail="match"):
    gr = MagicMock()
    gr.passed = passed
    gr.score = score
    gr.detail = detail
    return gr


class _Resp:
    """Minimal httpx.Response stand-in."""

    def __init__(self, payload: dict[str, Any], raise_exc: Exception | None = None):
        self._payload = payload
        self._raise_exc = raise_exc

    def raise_for_status(self) -> None:
        if self._raise_exc is not None:
            raise self._raise_exc

    def json(self) -> dict[str, Any]:
        return self._payload


def _client_returning(*responses):
    """An AsyncClient mock whose .post() yields the given responses in order."""
    client = MagicMock()
    client.post = AsyncMock(side_effect=list(responses))
    return client


# ---------------------------------------------------------------------------
# suite_of / safe_model_name — small mappers feeding aggregation.
# ---------------------------------------------------------------------------


class TestSuiteOf:
    def test_known_prefixes_collapse_to_headline_suite(self):
        assert suite_of("gpqa-physics") == "gpqa"
        assert suite_of("math500-algebra") == "math500"
        assert suite_of("ruler-niah-4k") == "ruler-niah"
        assert suite_of("humaneval-plus") == "humaneval-plus"

    def test_unknown_category_passes_through(self):
        assert suite_of("totally-novel") == "totally-novel"


class TestSafeModelName:
    def test_slashes_and_specials_become_underscores(self):
        assert safe_model_name("Qwen/Qwen3-8B:latest") == "Qwen_Qwen3-8B_latest"

    def test_safe_chars_preserved(self):
        assert safe_model_name("model_v1.2-x") == "model_v1.2-x"


# ---------------------------------------------------------------------------
# aggregate_per_suite — ungraded (None) prompts must be skipped, not counted.
# ---------------------------------------------------------------------------


class TestAggregatePerSuite:
    def _pr(self, suite, passed):
        return PromptResult(
            name="n",
            category=suite,
            suite=suite,
            passed=passed,
            score=None,
            detail="",
            output_text_clip="",
        )

    def test_pass_rate_per_suite(self):
        results = [
            self._pr("gsm8k", True),
            self._pr("gsm8k", False),
            self._pr("gpqa", True),
        ]
        agg = aggregate_per_suite(results)
        assert agg == {"gsm8k": pytest.approx(0.5), "gpqa": pytest.approx(1.0)}

    def test_ungraded_none_excluded_entirely(self):
        # A suite with only ungraded results must not appear (no 0-div, no key).
        results = [
            self._pr("ifeval", None),
            self._pr("gsm8k", True),
            self._pr("gsm8k", None),
        ]
        agg = aggregate_per_suite(results)
        assert "ifeval" not in agg
        # gsm8k's lone None is dropped → 1/1 graded.
        assert agg["gsm8k"] == pytest.approx(1.0)

    def test_empty_input(self):
        assert aggregate_per_suite([]) == {}


# ---------------------------------------------------------------------------
# write_result — JSON persistence with enum serialization.
# ---------------------------------------------------------------------------


class TestWriteResult:
    def test_writes_under_raw_with_safe_name_and_enum_value(self, tmp_path):
        result = ModelRunResult(
            model="Qwen/Qwen3-8B",
            tier="core",
            assignment=SuiteAssignment.FULL_CORE,
            warmup_tok_per_s=42.0,
            per_suite_pass_rate={"gsm8k": 1.0},
            composite=1.0,
        )
        path = write_result(tmp_path, result)
        assert path == tmp_path / "raw" / "Qwen_Qwen3-8B.json"
        data = json.loads(path.read_text())
        # Enum is serialized as its .value, not the Enum repr.
        assert data["assignment"] == "full_core"
        assert data["model"] == "Qwen/Qwen3-8B"
        assert data["per_suite_pass_rate"] == {"gsm8k": 1.0}

    def test_creates_raw_dir_when_absent(self, tmp_path):
        out = tmp_path / "nested" / "out"
        result = ModelRunResult(
            model="m",
            tier="core",
            assignment=SuiteAssignment.HE_PLUS_ONLY,
            warmup_tok_per_s=1.0,
        )
        path = write_result(out, result)
        assert path.exists()
        assert (out / "raw").is_dir()


# ---------------------------------------------------------------------------
# _drive_prompt — HTTP request shaping, grading, transport-error handling.
# ---------------------------------------------------------------------------


class TestDrivePrompt:
    async def test_happy_path_grades_output(self):
        client = _client_returning(_Resp({"message": {"content": "ANSWER"}}))
        prompt = FakePrompt(name="p1", category="gsm8k-x", expected={"answer": "y"})
        with patch.object(er, "grade", return_value=_fake_grade(True, 1.0, "ok")):
            res = await _drive_prompt(client, "m", prompt, "gsm8k")
        assert res.passed is True
        assert res.score == 1.0
        assert res.detail == "ok"
        assert res.suite == "gsm8k"
        assert res.output_text_clip == "ANSWER"

    async def test_request_body_uses_deterministic_options(self):
        client = _client_returning(_Resp({"message": {"content": "x"}}))
        prompt = FakePrompt(
            name="p", category="gsm8k", messages=[{"role": "user", "content": "hi"}]
        )
        prompt.max_tokens = 123
        with patch.object(er, "grade", return_value=_fake_grade()):
            await _drive_prompt(client, "the-model", prompt, "gsm8k")
        _, kwargs = client.post.call_args
        body = kwargs["json"]
        assert kwargs["timeout"] == 600.0
        assert body["model"] == "the-model"
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0.0
        assert body["options"]["seed"] == 42
        assert body["options"]["num_predict"] == 123
        assert body["messages"] == prompt.messages

    async def test_output_clip_truncated_to_500_chars(self):
        long = "z" * 1000
        client = _client_returning(_Resp({"message": {"content": long}}))
        prompt = FakePrompt(name="p", category="gsm8k")
        with patch.object(er, "grade", return_value=_fake_grade()):
            res = await _drive_prompt(client, "m", prompt, "gsm8k")
        assert len(res.output_text_clip) == 500

    async def test_http_error_recorded_as_ungraded_not_failed(self):
        err = httpx.ConnectError("boom")
        client = MagicMock()
        client.post = AsyncMock(side_effect=err)
        prompt = FakePrompt(name="p", category="gsm8k")
        # grade must NOT be called on the transport-error path.
        with patch.object(er, "grade") as graded:
            res = await _drive_prompt(client, "m", prompt, "gsm8k")
        graded.assert_not_called()
        assert res.passed is None
        assert res.score is None
        assert "transport error" in res.detail

    async def test_raise_for_status_5xx_is_ungraded(self):
        resp = _Resp(
            {}, raise_exc=httpx.HTTPStatusError("500", request=None, response=None)
        )
        client = _client_returning(resp)
        prompt = FakePrompt(name="p", category="gsm8k")
        res = await _drive_prompt(client, "m", prompt, "gsm8k")
        assert res.passed is None

    async def test_code_exec_gate_off_sets_enabled_false(self, monkeypatch):
        monkeypatch.delenv("OLMLX_BENCH_CODE_EXEC", raising=False)
        client = _client_returning(_Resp({"message": {"content": "code"}}))
        prompt = FakePrompt(name="p", category="humaneval-plus", grader="code_exec")
        captured = {}

        def fake_grade(grader, output, expected):
            captured["expected"] = expected
            return _fake_grade(None, None, "ungraded")

        with patch.object(er, "grade", side_effect=fake_grade):
            await _drive_prompt(client, "m", prompt, "humaneval-plus")
        assert captured["expected"]["_enabled"] is False

    async def test_code_exec_gate_on_sets_enabled_true(self, monkeypatch):
        monkeypatch.setenv("OLMLX_BENCH_CODE_EXEC", "1")
        client = _client_returning(_Resp({"message": {"content": "code"}}))
        prompt = FakePrompt(name="p", category="humaneval-plus", grader="code_exec")
        captured = {}

        def fake_grade(grader, output, expected):
            captured["expected"] = expected
            return _fake_grade()

        with patch.object(er, "grade", side_effect=fake_grade):
            await _drive_prompt(client, "m", prompt, "humaneval-plus")
        assert captured["expected"]["_enabled"] is True

    async def test_grader_none_defaults_to_exact_match(self):
        client = _client_returning(_Resp({"message": {"content": "x"}}))
        prompt = FakePrompt(name="p", category="gsm8k", grader=None)
        with patch.object(er, "grade", return_value=_fake_grade()) as graded:
            await _drive_prompt(client, "m", prompt, "gsm8k")
        assert graded.call_args[0][0] == "exact_match"

    async def test_missing_message_content_grades_empty_string(self):
        client = _client_returning(_Resp({}))  # no "message" key
        prompt = FakePrompt(name="p", category="gsm8k")
        with patch.object(er, "grade", return_value=_fake_grade()) as graded:
            await _drive_prompt(client, "m", prompt, "gsm8k")
        # output defaults to "" → passed through to grade.
        assert graded.call_args[0][1] == ""


# ---------------------------------------------------------------------------
# _warmup — tok/s estimation and the missing/zero eval_duration fallback.
# ---------------------------------------------------------------------------


class TestWarmup:
    async def test_uses_server_decode_time_when_present(self):
        # 100 tokens / 0.5s decode = 200 tok/s.
        client = _client_returning(
            _Resp({"eval_count": 100, "eval_duration": 500_000_000})
        )
        tps = await _warmup(client, "m")
        assert tps == pytest.approx(200.0)

    async def test_zero_eval_duration_falls_back_to_wallclock(self):
        # eval_duration=0 must NOT yield ~1e9 tok/s (the old `or 1` bug).
        client = _client_returning(_Resp({"eval_count": 10, "eval_duration": 0}))
        with patch.object(er.time, "monotonic", side_effect=[0.0, 2.0]):
            tps = await _warmup(client, "m")
        # 10 tokens / 2s wall-clock = 5 tok/s — a finite, sane estimate.
        assert tps == pytest.approx(5.0)
        assert tps < 1e6

    async def test_missing_counts_fall_back_without_divzero(self):
        client = _client_returning(_Resp({}))
        with patch.object(er.time, "monotonic", side_effect=[0.0, 0.0]):
            tps = await _warmup(client, "m")
        # max(eval_count,1)/max(elapsed,0.001) = 1/0.001 = 1000.
        assert tps == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# run_model — full orchestration: triage selection, subset filtering,
# extended dedup, per-prompt persistence, aggregation, timeout breakout.
# ---------------------------------------------------------------------------


def _patch_suites(monkeypatch, core, extended=None):
    monkeypatch.setattr(er, "assemble_core_suite", lambda: list(core))
    monkeypatch.setattr(er, "assemble_extended_suite", lambda: list(extended or []))


def _patch_client(monkeypatch, warmup_tps):
    """Make httpx.AsyncClient a context manager; stub _warmup return."""

    class _CtxClient:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(er.httpx, "AsyncClient", lambda **kw: _CtxClient())
    monkeypatch.setattr(er, "_warmup", AsyncMock(return_value=warmup_tps))


class TestRunModelOrchestration:
    async def test_full_core_runs_all_core_prompts(self, tmp_path, monkeypatch):
        core = [
            FakePrompt("he1", "humaneval-plus"),
            FakePrompt("g1", "gsm8k"),
            FakePrompt("q1", "gpqa-physics"),
        ]
        _patch_suites(monkeypatch, core)
        _patch_client(monkeypatch, warmup_tps=1000.0)  # fast → FULL_CORE

        async def fake_drive(client, model, prompt, suite):
            return PromptResult(
                prompt.name, prompt.category, suite, True, 1.0, "ok", "out"
            )

        monkeypatch.setattr(er, "_drive_prompt", fake_drive)

        res = await run_model("m", "core", "http://x", tmp_path, remaining_seconds=1e9)
        assert res.assignment == SuiteAssignment.FULL_CORE
        assert len(res.prompts) == 3
        assert res.timed_out is False
        # All graded True → composite is mean of per-suite (all 1.0).
        assert res.composite == pytest.approx(1.0)
        assert set(res.per_suite_pass_rate) == {"humaneval-plus", "gsm8k", "gpqa"}

    async def test_he_plus_only_filters_to_humaneval(self, tmp_path, monkeypatch):
        core = [
            FakePrompt("he1", "humaneval-plus"),
            FakePrompt("g1", "gsm8k"),
            FakePrompt("q1", "gpqa-physics"),
        ]
        _patch_suites(monkeypatch, core)
        # Slow + tiny budget → HE_PLUS_ONLY.
        _patch_client(monkeypatch, warmup_tps=1.0)
        seen = []

        async def fake_drive(client, model, prompt, suite):
            seen.append(prompt.category)
            return PromptResult(prompt.name, prompt.category, suite, True, 1.0, "", "")

        monkeypatch.setattr(er, "_drive_prompt", fake_drive)
        res = await run_model("m", "core", "http://x", tmp_path, remaining_seconds=10)
        assert res.assignment == SuiteAssignment.HE_PLUS_ONLY
        assert seen == ["humaneval-plus"]

    async def test_core_minus_gpqa_filters_gpqa(self, tmp_path, monkeypatch):
        core = [
            FakePrompt("he1", "humaneval-plus"),
            FakePrompt("g1", "gsm8k"),
            FakePrompt("q1", "gpqa-physics"),
        ]
        _patch_suites(monkeypatch, core)
        # 50 tok/s × 1500s = 75k tokens → CORE_MINUS_GPQA band.
        _patch_client(monkeypatch, warmup_tps=50.0)
        seen = []

        async def fake_drive(client, model, prompt, suite):
            seen.append(prompt.category)
            return PromptResult(prompt.name, prompt.category, suite, True, 1.0, "", "")

        monkeypatch.setattr(er, "_drive_prompt", fake_drive)
        res = await run_model("m", "core", "http://x", tmp_path, remaining_seconds=1500)
        assert res.assignment == SuiteAssignment.CORE_MINUS_GPQA
        assert "gpqa-physics" not in seen
        assert "gsm8k" in seen and "humaneval-plus" in seen

    async def test_extended_tier_appends_and_dedups_by_name(
        self, tmp_path, monkeypatch
    ):
        core = [FakePrompt("he1", "humaneval-plus")]
        extended = [
            FakePrompt("he1", "humaneval-plus"),  # dup name → must be skipped
            FakePrompt("m1", "math500"),
        ]
        _patch_suites(monkeypatch, core, extended)
        _patch_client(monkeypatch, warmup_tps=1e9)
        seen = []

        async def fake_drive(client, model, prompt, suite):
            seen.append(prompt.name)
            return PromptResult(prompt.name, prompt.category, suite, True, 1.0, "", "")

        monkeypatch.setattr(er, "_drive_prompt", fake_drive)
        res = await run_model(
            "m", "extended", "http://x", tmp_path, remaining_seconds=1e9
        )
        # he1 graded once, m1 added; duplicate extended he1 dropped.
        assert seen == ["he1", "m1"]
        assert res.tier == "extended"

    async def test_partial_json_written_after_each_prompt(self, tmp_path, monkeypatch):
        core = [FakePrompt("a", "gsm8k"), FakePrompt("b", "gsm8k")]
        _patch_suites(monkeypatch, core)
        _patch_client(monkeypatch, warmup_tps=1e9)
        write_calls = []
        real_write = er.write_result

        def spy_write(output_dir, result):
            write_calls.append(len(result.prompts))
            return real_write(output_dir, result)

        monkeypatch.setattr(er, "write_result", spy_write)

        async def fake_drive(client, model, prompt, suite):
            return PromptResult(prompt.name, prompt.category, suite, True, 1.0, "", "")

        monkeypatch.setattr(er, "_drive_prompt", fake_drive)
        await run_model("m", "core", "http://x", tmp_path, remaining_seconds=1e9)
        # Two partial writes (1, 2 prompts) plus a final write.
        assert write_calls[:2] == [1, 2]
        # Final file exists on disk.
        assert (tmp_path / "raw" / "m.json").exists()

    async def test_per_model_timeout_breaks_loop_and_marks_timed_out(
        self, tmp_path, monkeypatch
    ):
        core = [FakePrompt(f"p{i}", "gsm8k") for i in range(5)]
        _patch_suites(monkeypatch, core)
        _patch_client(monkeypatch, warmup_tps=1e9)

        # t_start=0, then deadline check returns increasing time. With
        # per_model_timeout=10 and deadline=10, the 2nd loop check at t=20
        # has remaining<=0 → breakout after 1 prompt.
        #
        # ``er.time`` is the global ``time`` module, so this patch leaks
        # to *every* monotonic caller during the test (asyncio, fixture
        # teardown). Repeat the last value once exhausted — a bare
        # ``next(times)`` raised StopIteration out of unrelated yield
        # fixtures whenever a conftest change perturbed the call count.
        times = [0.0, 1.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        times_iter = iter(times)
        monkeypatch.setattr(er.time, "monotonic", lambda: next(times_iter, times[-1]))

        async def fake_drive(client, model, prompt, suite):
            return PromptResult(prompt.name, prompt.category, suite, True, 1.0, "", "")

        monkeypatch.setattr(er, "_drive_prompt", fake_drive)
        res = await run_model(
            "m",
            "core",
            "http://x",
            tmp_path,
            remaining_seconds=1e9,
            per_model_timeout=10,
        )
        assert res.timed_out is True
        assert len(res.prompts) == 1

    async def test_per_prompt_timeout_records_failed_result(
        self, tmp_path, monkeypatch
    ):
        core = [FakePrompt("slow", "gsm8k")]
        _patch_suites(monkeypatch, core)
        _patch_client(monkeypatch, warmup_tps=1e9)

        async def hang(client, model, prompt, suite):
            raise AssertionError("should be cancelled by wait_for")

        monkeypatch.setattr(er, "_drive_prompt", hang)

        async def fake_wait_for(coro, timeout):
            coro.close()  # avoid 'never awaited' warning
            raise er.asyncio.TimeoutError()

        monkeypatch.setattr(er.asyncio, "wait_for", fake_wait_for)
        res = await run_model("m", "core", "http://x", tmp_path, remaining_seconds=1e9)
        assert len(res.prompts) == 1
        pr = res.prompts[0]
        assert pr.passed is False
        assert pr.score == 0.0
        assert "wait_for timeout" in pr.detail


# ---------------------------------------------------------------------------
# run_model_sync — thin asyncio.run wrapper around run_model.
# ---------------------------------------------------------------------------


class TestRunModelSync:
    def test_delegates_to_run_model(self, monkeypatch):
        sentinel = object()

        async def fake_run_model(*args, **kwargs):
            return sentinel

        monkeypatch.setattr(er, "run_model", fake_run_model)
        out = run_model_sync("m", "core", "http://x", "/out", 100.0)
        assert out is sentinel


# ---------------------------------------------------------------------------
# composite_score edge — single suite, defensive against empty (already in
# sibling test, but pin the single-suite branch explicitly here).
# ---------------------------------------------------------------------------


class TestCompositeScoreSingle:
    def test_single_suite_returns_its_rate(self):
        assert composite_score({"gsm8k": 0.7}) == pytest.approx(0.7)
