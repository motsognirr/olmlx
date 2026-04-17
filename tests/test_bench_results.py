"""Tests for olmlx.bench.results."""

import pytest

from olmlx.bench.results import (
    PromptResult,
    RunResult,
    ScenarioResult,
    build_leaderboard,
    compare_runs,
    create_run_result,
    format_leaderboard,
    list_runs,
    load_run,
    save_run,
)


class TestPromptResult:
    def test_tokens_per_second(self):
        r = PromptResult(
            prompt_name="test",
            category="unit",
            output_text="hello",
            status_code=200,
            eval_count=100,
            eval_duration_ns=2_000_000_000,  # 2 seconds
        )
        assert r.tokens_per_second == 50.0

    def test_tokens_per_second_zero_duration(self):
        r = PromptResult(
            prompt_name="test",
            category="unit",
            output_text="",
            status_code=200,
        )
        assert r.tokens_per_second == 0.0

    def test_prompt_tokens_per_second(self):
        r = PromptResult(
            prompt_name="test",
            category="unit",
            output_text="",
            status_code=200,
            prompt_eval_count=50,
            prompt_eval_duration_ns=1_000_000_000,
        )
        assert r.prompt_tokens_per_second == 50.0

    def test_to_dict_roundtrip(self):
        r = PromptResult(
            prompt_name="test",
            category="unit",
            output_text="hello world",
            status_code=200,
            error=None,
            eval_count=10,
            eval_duration_ns=500_000_000,
            prompt_eval_count=5,
            prompt_eval_duration_ns=100_000_000,
            total_duration_ns=600_000_000,
        )
        d = r.to_dict()
        restored = PromptResult.from_dict(d)
        assert restored.prompt_name == r.prompt_name
        assert restored.output_text == r.output_text
        assert restored.eval_count == r.eval_count
        assert restored.eval_duration_ns == r.eval_duration_ns

    def test_to_dict_includes_computed_fields(self):
        r = PromptResult(
            prompt_name="test",
            category="unit",
            output_text="",
            status_code=200,
            eval_count=100,
            eval_duration_ns=1_000_000_000,
        )
        d = r.to_dict()
        assert d["tokens_per_second"] == 100.0


class TestScenarioResult:
    def test_to_dict_roundtrip(self):
        sr = ScenarioResult(
            scenario_name="baseline",
            scenario_description="Default settings",
            env_overrides={},
            prompt_results=[
                PromptResult(
                    prompt_name="test",
                    category="unit",
                    output_text="hi",
                    status_code=200,
                )
            ],
        )
        d = sr.to_dict()
        restored = ScenarioResult.from_dict(d)
        assert restored.scenario_name == sr.scenario_name
        assert len(restored.prompt_results) == 1

    def test_skipped_roundtrip(self):
        sr = ScenarioResult(
            scenario_name="flash",
            scenario_description="Flash inference",
            env_overrides={"OLMLX_EXPERIMENTAL_FLASH": "true"},
            prompt_results=[],
            skipped=True,
            skip_reason="No flash layout",
        )
        d = sr.to_dict()
        restored = ScenarioResult.from_dict(d)
        assert restored.skipped is True
        assert restored.skip_reason == "No flash layout"


class TestRunResult:
    def test_to_dict_roundtrip(self):
        run = RunResult(
            model="test-model",
            timestamp="20260329T120000Z",
            git_sha="abc1234",
            scenarios=[],
            max_tokens_override=128,
        )
        d = run.to_dict()
        restored = RunResult.from_dict(d)
        assert restored.model == run.model
        assert restored.timestamp == run.timestamp
        assert restored.git_sha == run.git_sha
        assert restored.max_tokens_override == 128


class TestSaveLoadRun:
    def test_save_and_load(self, tmp_path):
        run = create_run_result(
            model="test-model",
            scenarios=[
                ScenarioResult(
                    scenario_name="baseline",
                    scenario_description="Default",
                    env_overrides={},
                    prompt_results=[
                        PromptResult(
                            prompt_name="factual",
                            category="factual",
                            output_text="Paris is the capital of France.",
                            status_code=200,
                            eval_count=8,
                            eval_duration_ns=400_000_000,
                        )
                    ],
                )
            ],
        )
        run_dir = save_run(run, tmp_path)
        assert (run_dir / "results.json").exists()

        loaded = load_run(run_dir)
        assert loaded.model == "test-model"
        assert len(loaded.scenarios) == 1
        assert (
            loaded.scenarios[0].prompt_results[0].output_text
            == "Paris is the capital of France."
        )

    def test_load_from_json_file(self, tmp_path):
        run = create_run_result(model="m", scenarios=[])
        run_dir = save_run(run, tmp_path)
        loaded = load_run(run_dir / "results.json")
        assert loaded.model == "m"

    def test_save_run_collision_uses_suffix(self, tmp_path):
        """Two runs with the same timestamp get distinct directories."""
        run1 = create_run_result(model="m1", scenarios=[])
        run2 = create_run_result(model="m2", scenarios=[])
        run2.timestamp = run1.timestamp  # force collision

        dir1 = save_run(run1, tmp_path)
        dir2 = save_run(run2, tmp_path)

        assert dir1 != dir2
        assert dir1.name == run1.timestamp
        assert dir2.name == f"{run1.timestamp}-1"
        assert load_run(dir1).model == "m1"
        assert load_run(dir2).model == "m2"


class TestListRuns:
    def test_empty_dir(self, tmp_path):
        assert list_runs(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert list_runs(tmp_path / "nope") == []

    def test_lists_saved_runs(self, tmp_path):
        run1 = create_run_result(model="model-a", scenarios=[])
        run1.timestamp = "20260329T100000Z"
        save_run(run1, tmp_path)

        run2 = create_run_result(model="model-b", scenarios=[])
        run2.timestamp = "20260329T110000Z"
        save_run(run2, tmp_path)

        runs = list_runs(tmp_path)
        assert len(runs) == 2
        assert runs[0]["model"] == "model-a"
        assert runs[1]["model"] == "model-b"


class TestCompareRuns:
    def test_compare_basic(self):
        pr1 = PromptResult(
            prompt_name="factual",
            category="factual",
            output_text="Paris",
            status_code=200,
            eval_count=10,
            eval_duration_ns=1_000_000_000,
        )
        pr2 = PromptResult(
            prompt_name="factual",
            category="factual",
            output_text="Paris is the capital.",
            status_code=200,
            eval_count=10,
            eval_duration_ns=500_000_000,
        )
        run1 = RunResult(
            model="m",
            timestamp="t1",
            git_sha="aaa",
            scenarios=[
                ScenarioResult(
                    scenario_name="baseline",
                    scenario_description="Default",
                    env_overrides={},
                    prompt_results=[pr1],
                )
            ],
        )
        run2 = RunResult(
            model="m",
            timestamp="t2",
            git_sha="bbb",
            scenarios=[
                ScenarioResult(
                    scenario_name="baseline",
                    scenario_description="Default",
                    env_overrides={},
                    prompt_results=[pr2],
                )
            ],
        )
        output = compare_runs(run1, run2)
        assert "baseline" in output
        assert "factual" in output
        assert "+100.0%" in output
        assert "Output Differences" in output
        assert "Paris is the capital." in output

    def test_compare_skipped_scenarios_excluded(self):
        run1 = RunResult(
            model="m",
            timestamp="t1",
            git_sha=None,
            scenarios=[
                ScenarioResult(
                    scenario_name="flash",
                    scenario_description="Flash",
                    env_overrides={},
                    prompt_results=[],
                    skipped=True,
                )
            ],
        )
        run2 = RunResult(
            model="m",
            timestamp="t2",
            git_sha=None,
            scenarios=[],
        )
        output = compare_runs(run1, run2)
        assert "No output differences found." in output


def _prompt(tps: float, *, status_code: int = 200) -> PromptResult:
    """Build a PromptResult whose tokens_per_second equals `tps`."""
    if tps <= 0:
        return PromptResult(
            prompt_name="p",
            category="c",
            output_text="",
            status_code=status_code,
        )
    eval_count = 100
    eval_duration_ns = int(eval_count / tps * 1e9)
    return PromptResult(
        prompt_name="p",
        category="c",
        output_text="",
        status_code=status_code,
        eval_count=eval_count,
        eval_duration_ns=eval_duration_ns,
    )


def _scenario(name: str, prompts: list[PromptResult], *, skipped: bool = False):
    return ScenarioResult(
        scenario_name=name,
        scenario_description=name,
        env_overrides={},
        prompt_results=prompts,
        skipped=skipped,
    )


def _save_fake_run(
    tmp_path, *, model: str, timestamp: str, scenarios: list[ScenarioResult]
):
    run = RunResult(
        model=model,
        timestamp=timestamp,
        git_sha="abc1234",
        scenarios=scenarios,
    )
    save_run(run, tmp_path)


class TestBuildLeaderboard:
    def test_latest_per_model_ranks_by_best_tps(self, tmp_path):
        # model-a older run (slow), newer run (fast) — newer should win and rank first
        _save_fake_run(
            tmp_path,
            model="model-a",
            timestamp="20260101T000000Z",
            scenarios=[_scenario("baseline", [_prompt(20.0)])],
        )
        _save_fake_run(
            tmp_path,
            model="model-a",
            timestamp="20260102T000000Z",
            scenarios=[_scenario("baseline", [_prompt(80.0)])],
        )
        # model-b single run in the middle
        _save_fake_run(
            tmp_path,
            model="model-b",
            timestamp="20260102T120000Z",
            scenarios=[_scenario("baseline", [_prompt(50.0)])],
        )

        entries = build_leaderboard(tmp_path)

        assert [e.model for e in entries] == ["model-a", "model-b"]
        assert entries[0].best_tps == 80.0
        assert entries[0].timestamp == "20260102T000000Z"
        assert entries[1].best_tps == 50.0

    def test_all_runs_keeps_history(self, tmp_path):
        _save_fake_run(
            tmp_path,
            model="model-a",
            timestamp="20260101T000000Z",
            scenarios=[_scenario("baseline", [_prompt(20.0)])],
        )
        _save_fake_run(
            tmp_path,
            model="model-a",
            timestamp="20260102T000000Z",
            scenarios=[_scenario("baseline", [_prompt(80.0)])],
        )
        _save_fake_run(
            tmp_path,
            model="model-b",
            timestamp="20260102T120000Z",
            scenarios=[_scenario("baseline", [_prompt(50.0)])],
        )

        entries = build_leaderboard(tmp_path, latest_per_model=False)
        assert len(entries) == 3
        assert [e.best_tps for e in entries] == [80.0, 50.0, 20.0]

    def test_best_scenario_is_max_across_scenarios(self, tmp_path):
        _save_fake_run(
            tmp_path,
            model="model-a",
            timestamp="20260101T000000Z",
            scenarios=[
                _scenario("baseline", [_prompt(40.0), _prompt(60.0)]),  # avg 50
                _scenario("turboquant-4", [_prompt(70.0), _prompt(90.0)]),  # avg 80
            ],
        )

        entries = build_leaderboard(tmp_path)
        assert len(entries) == 1
        assert entries[0].best_tps == pytest.approx(80.0, rel=1e-6)
        assert entries[0].best_scenario == "turboquant-4"

    def test_skips_runs_with_no_valid_prompts(self, tmp_path):
        # One run has all-zero prompts (simulating server failures)
        _save_fake_run(
            tmp_path,
            model="broken",
            timestamp="20260101T000000Z",
            scenarios=[
                _scenario("baseline", [_prompt(0.0, status_code=0)]),
            ],
        )
        _save_fake_run(
            tmp_path,
            model="working",
            timestamp="20260102T000000Z",
            scenarios=[_scenario("baseline", [_prompt(50.0)])],
        )

        entries = build_leaderboard(tmp_path)
        assert [e.model for e in entries] == ["working"]

    def test_skipped_scenarios_excluded_from_counts(self, tmp_path):
        _save_fake_run(
            tmp_path,
            model="model-a",
            timestamp="20260101T000000Z",
            scenarios=[
                _scenario("flash", [], skipped=True),
                _scenario("baseline", [_prompt(50.0)]),
            ],
        )

        entries = build_leaderboard(tmp_path)
        assert len(entries) == 1
        assert entries[0].failed_scenarios == 0
        assert entries[0].total_scenarios == 1

    def test_failed_scenarios_counted(self, tmp_path):
        _save_fake_run(
            tmp_path,
            model="model-a",
            timestamp="20260101T000000Z",
            scenarios=[
                _scenario("baseline", [_prompt(50.0)]),
                _scenario("turboquant-4", [_prompt(0.0, status_code=0)]),
            ],
        )

        entries = build_leaderboard(tmp_path)
        assert len(entries) == 1
        assert entries[0].failed_scenarios == 1
        assert entries[0].total_scenarios == 2

    def test_empty_dir_returns_empty(self, tmp_path):
        assert build_leaderboard(tmp_path) == []

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        assert build_leaderboard(tmp_path / "nope") == []

    def test_bench_path_is_file_returns_empty(self, tmp_path):
        not_a_dir = tmp_path / "oops"
        not_a_dir.write_text("{}")
        assert build_leaderboard(not_a_dir) == []

    def test_invalid_shape_results_json_is_skipped(self, tmp_path):
        # Valid JSON but not a dict — load_run's from_dict raises TypeError,
        # which must not abort the whole leaderboard build.
        bad = tmp_path / "broken"
        bad.mkdir()
        (bad / "results.json").write_text("null")

        _save_fake_run(
            tmp_path,
            model="good",
            timestamp="20260101T000000Z",
            scenarios=[_scenario("baseline", [_prompt(50.0)])],
        )

        entries = build_leaderboard(tmp_path)
        assert [e.model for e in entries] == ["good"]

    def test_iterdir_oserror_returns_empty(self, tmp_path, monkeypatch):
        # Simulate a bench_dir we can stat() but not iterate (e.g. wrong
        # ownership). The function must not propagate the OSError.
        from pathlib import Path

        real_iterdir = Path.iterdir

        def fake_iterdir(self):
            if self == tmp_path:
                raise PermissionError("denied")
            yield from real_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", fake_iterdir)
        assert build_leaderboard(tmp_path) == []

    def test_results_json_exists_permission_error_is_skipped(
        self, tmp_path, monkeypatch
    ):
        # Python 3.12+: Path.exists() propagates non-FileNotFoundError OSErrors.
        # A subdirectory the process can stat but not inspect must skip that
        # run, not abort the whole build.
        from pathlib import Path

        _save_fake_run(
            tmp_path,
            model="good",
            timestamp="20260101T000000Z",
            scenarios=[_scenario("baseline", [_prompt(42.0)])],
        )
        bad_run_dir = tmp_path / "20260101T010000Z-restricted"
        bad_run_dir.mkdir()

        real_exists = Path.exists

        def fake_exists(self):
            if self == bad_run_dir / "results.json":
                raise PermissionError("stat denied")
            return real_exists(self)

        monkeypatch.setattr(Path, "exists", fake_exists)

        entries = build_leaderboard(tmp_path)
        assert [e.model for e in entries] == ["good"]

    def test_entry_equality_ignores_run_dir(self):
        from pathlib import Path

        from olmlx.bench.results import LeaderboardEntry

        common = dict(
            model="m",
            best_tps=10.0,
            best_scenario="s",
            timestamp="t",
            git_sha="g",
            failed_scenarios=0,
            total_scenarios=1,
        )
        a = LeaderboardEntry(**common, run_dir=Path("/a"))
        b = LeaderboardEntry(**common, run_dir=Path("/b"))
        assert a == b
        assert hash(a) == hash(b)

    def test_tiebreaker_handles_double_digit_collision_counter(self, tmp_path):
        # 11 runs share one timestamp; save_run writes them to ...Z, ...Z-1,
        # ..., ...Z-10. The last (counter=10) must win — naive string order
        # would rank it below ...Z-9.
        for i in range(11):
            run = RunResult(
                model="model-a",
                timestamp="20260101T000000Z",
                git_sha=f"rev{i:02d}",
                scenarios=[_scenario("baseline", [_prompt(10.0 + i)])],
            )
            save_run(run, tmp_path)

        entries = build_leaderboard(tmp_path)
        assert len(entries) == 1
        assert entries[0].git_sha == "rev10"
        assert entries[0].best_tps == pytest.approx(20.0, rel=1e-6)

    def test_same_timestamp_tiebreaker_is_deterministic(self, tmp_path):
        # save_run appends a -N suffix to the directory on sub-second collisions
        # but leaves the timestamp field untouched. The later (higher-suffix)
        # directory should win deterministically.
        run_early = RunResult(
            model="model-a",
            timestamp="20260101T000000Z",
            git_sha="early",
            scenarios=[_scenario("baseline", [_prompt(30.0)])],
        )
        run_late = RunResult(
            model="model-a",
            timestamp="20260101T000000Z",
            git_sha="late",
            scenarios=[_scenario("baseline", [_prompt(90.0)])],
        )
        save_run(run_early, tmp_path)
        save_run(run_late, tmp_path)

        entries = build_leaderboard(tmp_path)
        assert len(entries) == 1
        assert entries[0].git_sha == "late"
        assert entries[0].best_tps == pytest.approx(90.0, rel=1e-6)


class TestFormatLeaderboard:
    def _entries(self, n: int):
        from olmlx.bench.results import LeaderboardEntry

        return [
            LeaderboardEntry(
                model=f"model-{i}",
                best_tps=100.0 - i,
                best_scenario="baseline",
                timestamp="20260101T000000Z",
                git_sha="abc1234",
                failed_scenarios=0,
                total_scenarios=1,
                run_dir=__import__("pathlib").Path("/tmp/x"),
            )
            for i in range(n)
        ]

    def test_limit_truncates_rows(self):
        out = format_leaderboard(self._entries(5), limit=2)
        assert "model-0" in out
        assert "model-1" in out
        assert "model-2" not in out

    def test_no_limit_shows_all(self):
        out = format_leaderboard(self._entries(3))
        assert "model-0" in out
        assert "model-2" in out

    def test_header_present(self):
        out = format_leaderboard(self._entries(1))
        assert "Best tok/s" in out
        assert "Model" in out

    def test_long_model_name_does_not_break_alignment(self):
        from pathlib import Path

        from olmlx.bench.results import LeaderboardEntry

        long_name = "mlx-community/Meta-Llama-3.1-70B-Instruct-8bit-some-suffix"
        entries = [
            LeaderboardEntry(
                model=long_name,
                best_tps=45.2,
                best_scenario="baseline",
                timestamp="20260101T000000Z",
                git_sha="abc1234",
                failed_scenarios=0,
                total_scenarios=1,
                run_dir=Path("/tmp/x"),
            ),
        ]
        out = format_leaderboard(entries)
        lines = out.split("\n")
        assert long_name in out
        # Header, separator, and data row must share the same width so columns
        # stay aligned.
        assert len(lines[0]) == len(lines[1]) == len(lines[2])

    def test_long_scenario_name_does_not_break_alignment(self):
        from pathlib import Path

        from olmlx.bench.results import LeaderboardEntry

        long_scenario = "a-really-long-scenario-name-that-overflows"
        entries = [
            LeaderboardEntry(
                model="model-a",
                best_tps=45.2,
                best_scenario=long_scenario,
                timestamp="20260101T000000Z",
                git_sha="abc1234",
                failed_scenarios=0,
                total_scenarios=1,
                run_dir=Path("/tmp/x"),
            ),
        ]
        out = format_leaderboard(entries)
        lines = out.split("\n")
        assert long_scenario in out
        assert len(lines[0]) == len(lines[1]) == len(lines[2])

    def test_missing_git_sha_uses_ascii_placeholder(self):
        # The em dash (U+2014) is 1 Python character but can render as 2
        # visual columns in East_Asian_Width=Ambiguous terminals, silently
        # widening the Git column past the ':<10' header pad.
        from pathlib import Path

        from olmlx.bench.results import LeaderboardEntry

        entries = [
            LeaderboardEntry(
                model="model-a",
                best_tps=45.2,
                best_scenario="baseline",
                timestamp="20260101T000000Z",
                git_sha=None,
                failed_scenarios=0,
                total_scenarios=1,
                run_dir=Path("/tmp/x"),
            ),
        ]
        out = format_leaderboard(entries)
        assert "—" not in out

    def test_full_length_git_sha_does_not_break_alignment(self):
        from pathlib import Path

        from olmlx.bench.results import LeaderboardEntry

        full_sha = "a" * 40
        entries = [
            LeaderboardEntry(
                model="model-a",
                best_tps=45.2,
                best_scenario="baseline",
                timestamp="20260101T000000Z",
                git_sha=full_sha,
                failed_scenarios=0,
                total_scenarios=1,
                run_dir=Path("/tmp/x"),
            ),
        ]
        out = format_leaderboard(entries)
        lines = out.split("\n")
        assert len(lines[0]) == len(lines[1]) == len(lines[2])
