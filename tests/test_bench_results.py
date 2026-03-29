"""Tests for olmlx.bench.results."""

from olmlx.bench.results import (
    PromptResult,
    RunResult,
    ScenarioResult,
    compare_runs,
    create_run_result,
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
