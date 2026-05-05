"""Tests for olmlx.bench.runner."""

import json
from unittest.mock import MagicMock, patch


from olmlx.bench.prompts import PROMPTS
from olmlx.bench.results import PromptResult
from olmlx.bench.runner import (
    _run_prompts_over_http,
    _run_server_scenario,
    _run_worker,
    _wait_for_server,
    run_bench,
)
from olmlx.bench.scenarios import Scenario


class TestRunWorker:
    def test_worker_success(self, tmp_path, monkeypatch):
        """Mock subprocess.run to return pre-built results."""
        fake_results = [
            {
                "prompt_name": "factual",
                "category": "factual",
                "output_text": "Paris.",
                "status_code": 200,
                "eval_count": 5,
                "eval_duration_ns": 500_000_000,
                "prompt_eval_count": 10,
                "prompt_eval_duration_ns": 200_000_000,
                "total_duration_ns": 700_000_000,
            }
        ]

        def fake_subprocess_run(cmd, env, capture_output, text, timeout):
            # Write results to the path specified in cmd
            results_idx = cmd.index("--results-json") + 1
            results_path = cmd[results_idx]
            import pathlib

            pathlib.Path(results_path).write_text(json.dumps(fake_results))

            class FakeResult:
                returncode = 0
                stderr = ""
                stdout = ""

            return FakeResult()

        scenario = Scenario(name="test", description="Test scenario")
        prompts_data = [PROMPTS[0].to_dict()]

        with patch(
            "olmlx.bench.runner.subprocess.run", side_effect=fake_subprocess_run
        ):
            results = _run_worker("test-model", scenario, prompts_data, None)

        assert len(results) == 1
        assert results[0].prompt_name == "factual"
        assert results[0].output_text == "Paris."
        assert results[0].tokens_per_second == 10.0

    def test_worker_failure_returns_error(self):
        """Worker exit code != 0 produces error result."""
        scenario = Scenario(name="test", description="Test")

        class FakeResult:
            returncode = 1
            stderr = "ImportError: no module"
            stdout = ""

        with patch("olmlx.bench.runner.subprocess.run", return_value=FakeResult()):
            results = _run_worker("model", scenario, [], None)

        assert len(results) == 1
        assert results[0].prompt_name == "__worker_error__"
        assert "exit" in results[0].error.lower() or "code 1" in results[0].error

    def test_worker_timeout_returns_error(self):
        """Worker timeout produces error result."""
        import subprocess

        scenario = Scenario(name="test", description="Test")

        with patch(
            "olmlx.bench.runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="test", timeout=600),
        ):
            results = _run_worker("model", scenario, [], None)

        assert len(results) == 1
        assert "timed out" in results[0].error.lower()

    def test_worker_passes_env_overrides(self):
        """Verify env overrides are passed to subprocess."""
        scenario = Scenario(
            name="tq4",
            description="TurboQuant",
            env_overrides={"OLMLX_KV_CACHE_QUANT": "turboquant:4"},
        )

        captured_env = {}

        def capture_run(cmd, env, **kwargs):
            captured_env.update(env)
            results_idx = cmd.index("--results-json") + 1
            import pathlib

            pathlib.Path(cmd[results_idx]).write_text("[]")

            class FakeResult:
                returncode = 0
                stderr = ""

            return FakeResult()

        with patch("olmlx.bench.runner.subprocess.run", side_effect=capture_run):
            _run_worker("model", scenario, [], None)

        assert captured_env.get("OLMLX_KV_CACHE_QUANT") == "turboquant:4"

    def test_worker_passes_max_tokens(self):
        """Verify --max-tokens is passed when provided."""
        scenario = Scenario(name="test", description="Test")
        captured_cmd = []

        def capture_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            results_idx = cmd.index("--results-json") + 1
            import pathlib

            pathlib.Path(cmd[results_idx]).write_text("[]")

            class FakeResult:
                returncode = 0
                stderr = ""

            return FakeResult()

        with patch("olmlx.bench.runner.subprocess.run", side_effect=capture_run):
            _run_worker("model", scenario, [], max_tokens=64)

        assert "--max-tokens" in captured_cmd
        assert "64" in captured_cmd


class TestRunBench:
    def test_skipped_scenarios_recorded(self, tmp_path, monkeypatch):
        """Scenarios that fail skip check are recorded as skipped."""
        monkeypatch.setattr(
            "olmlx.bench.runner._resolve_model_path",
            lambda m: tmp_path / "model",
        )

        always_skip = Scenario(
            name="test-skip",
            description="Always skips",
            should_skip=lambda p: "test reason",
        )

        with patch("olmlx.bench.runner.get_scenarios", return_value=[always_skip]):
            result = run_bench("model", bench_dir=tmp_path / "bench")

        assert len(result.scenarios) == 1
        assert result.scenarios[0].skipped is True
        assert result.scenarios[0].skip_reason == "test reason"

    def test_results_saved_to_disk(self, tmp_path, monkeypatch):
        """Results are persisted as JSON."""
        monkeypatch.setattr(
            "olmlx.bench.runner._resolve_model_path",
            lambda m: tmp_path / "model",
        )

        always_skip = Scenario(
            name="baseline",
            description="Baseline",
            should_skip=lambda p: "skipped for test",
        )

        bench_dir = tmp_path / "bench"
        with patch("olmlx.bench.runner.get_scenarios", return_value=[always_skip]):
            result = run_bench("model", bench_dir=bench_dir)

        run_dir = bench_dir / result.timestamp
        assert (run_dir / "results.json").exists()

    def test_post_run_leaderboard_printed(self, tmp_path, monkeypatch, capsys):
        """A successful run prints the Leaderboard block to stderr."""
        monkeypatch.setattr(
            "olmlx.bench.runner._resolve_model_path",
            lambda m: tmp_path / "model",
        )

        fake_entry = MagicMock()
        monkeypatch.setattr(
            "olmlx.bench.runner.build_leaderboard",
            lambda _bench_dir: [fake_entry],
        )
        monkeypatch.setattr(
            "olmlx.bench.runner.format_leaderboard",
            lambda _entries, limit: "<leaderboard body>",
        )

        always_skip = Scenario(
            name="baseline",
            description="Baseline",
            should_skip=lambda p: "skipped for test",
        )
        with patch("olmlx.bench.runner.get_scenarios", return_value=[always_skip]):
            run_bench("model", bench_dir=tmp_path / "bench")

        err = capsys.readouterr().err
        assert "Leaderboard (top 5):" in err
        assert "<leaderboard body>" in err

    def test_post_run_leaderboard_failure_is_swallowed(
        self, tmp_path, monkeypatch, caplog
    ):
        """A broken build_leaderboard must not break a saved run."""
        import logging

        monkeypatch.setattr(
            "olmlx.bench.runner._resolve_model_path",
            lambda m: tmp_path / "model",
        )

        def _boom(_bench_dir):
            raise RuntimeError("synthetic")

        monkeypatch.setattr("olmlx.bench.runner.build_leaderboard", _boom)

        always_skip = Scenario(
            name="baseline",
            description="Baseline",
            should_skip=lambda p: "skipped for test",
        )
        with (
            patch("olmlx.bench.runner.get_scenarios", return_value=[always_skip]),
            caplog.at_level(logging.WARNING, logger="olmlx.bench.runner"),
        ):
            result = run_bench("model", bench_dir=tmp_path / "bench")

        assert result is not None
        assert any("Could not build leaderboard" in r.message for r in caplog.records)

    def test_server_mode_dispatched(self, tmp_path, monkeypatch):
        """Server-mode scenarios call _run_server_scenario, not _run_worker."""
        monkeypatch.setattr(
            "olmlx.bench.runner._resolve_model_path",
            lambda m: tmp_path / "model",
        )

        server_scenario = Scenario(
            name="distributed",
            description="Distributed",
            env_overrides={"OLMLX_EXPERIMENTAL_DISTRIBUTED": "true"},
            server_mode=True,
        )

        with (
            patch("olmlx.bench.runner.get_scenarios", return_value=[server_scenario]),
            patch(
                "olmlx.bench.runner._run_server_scenario",
                return_value=[
                    PromptResult(
                        prompt_name="factual",
                        category="factual",
                        output_text="Paris",
                        status_code=200,
                        eval_count=5,
                        eval_duration_ns=500_000_000,
                    )
                ],
            ) as mock_server,
            patch("olmlx.bench.runner._run_worker") as mock_worker,
        ):
            result = run_bench("model", bench_dir=tmp_path / "bench")

        mock_server.assert_called_once()
        mock_worker.assert_not_called()
        assert len(result.scenarios) == 1
        assert result.scenarios[0].prompt_results[0].output_text == "Paris"


class TestWaitForServer:
    def test_returns_true_when_server_responds(self):
        proc = MagicMock()
        proc.poll.return_value = None  # process is alive

        with patch("olmlx.bench.runner.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            assert _wait_for_server(11434, proc, timeout=5) is True

    def test_returns_false_when_process_dies(self):
        proc = MagicMock()
        proc.poll.return_value = 1  # process exited

        assert _wait_for_server(11434, proc, timeout=1) is False


class TestRunServerScenario:
    def test_server_startup_failure(self):
        """If server process exits immediately, returns error."""
        scenario = Scenario(
            name="distributed",
            description="Distributed",
            env_overrides={"OLMLX_EXPERIMENTAL_DISTRIBUTED": "true"},
            server_mode=True,
        )

        fake_proc = MagicMock()
        fake_proc.poll.return_value = 1  # already dead
        fake_proc.communicate.return_value = (b"", b"startup error")

        with patch("olmlx.bench.runner.subprocess.Popen", return_value=fake_proc):
            results = _run_server_scenario("model", scenario, [], None)

        assert len(results) == 1
        assert results[0].prompt_name == "__server_error__"
        assert "start" in results[0].error.lower()

    def test_server_success_runs_prompts(self):
        """If server starts, prompts are run over HTTP and server is terminated."""
        scenario = Scenario(
            name="distributed",
            description="Distributed",
            env_overrides={"OLMLX_EXPERIMENTAL_DISTRIBUTED": "true"},
            server_mode=True,
        )

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None  # alive

        expected_results = [
            PromptResult(
                prompt_name="factual",
                category="factual",
                output_text="Paris",
                status_code=200,
            )
        ]

        with (
            patch("olmlx.bench.runner.subprocess.Popen", return_value=fake_proc),
            patch("olmlx.bench.runner._wait_for_server", return_value=True),
            patch(
                "olmlx.bench.runner._run_prompts_over_http",
                return_value=expected_results,
            ),
        ):
            results = _run_server_scenario(
                "model",
                scenario,
                [{"name": "factual", "category": "factual", "messages": []}],
                None,
            )

        assert results == expected_results
        fake_proc.terminate.assert_called_once()

    def test_server_env_overrides_passed_to_popen(self):
        """Verify env overrides are passed when launching the server."""
        scenario = Scenario(
            name="distributed",
            description="Distributed",
            env_overrides={
                "OLMLX_EXPERIMENTAL_DISTRIBUTED": "true",
                "OLMLX_KV_CACHE_QUANT": "turboquant:4",
            },
            server_mode=True,
        )

        captured_env = {}
        fake_proc = MagicMock()
        fake_proc.poll.return_value = 1
        fake_proc.communicate.return_value = (b"", b"fail")

        def capture_popen(cmd, env, **kwargs):
            captured_env.update(env)
            return fake_proc

        with patch("olmlx.bench.runner.subprocess.Popen", side_effect=capture_popen):
            _run_server_scenario("model", scenario, [], None)

        assert captured_env.get("OLMLX_EXPERIMENTAL_DISTRIBUTED") == "true"
        assert captured_env.get("OLMLX_KV_CACHE_QUANT") == "turboquant:4"


class TestRunPromptsOverHttp:
    def test_success(self):
        """HTTP 200 response is parsed into PromptResult."""
        response_body = json.dumps(
            {
                "message": {"content": "Paris"},
                "eval_count": 5,
                "eval_duration": 500_000_000,
                "prompt_eval_count": 10,
                "prompt_eval_duration": 100_000_000,
                "total_duration": 600_000_000,
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        prompt = {
            "name": "factual",
            "category": "factual",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
        }

        with patch("olmlx.bench.runner.urllib.request.urlopen", return_value=mock_resp):
            results = _run_prompts_over_http("model", [prompt], None, 11434)

        assert len(results) == 1
        assert results[0].output_text == "Paris"
        assert results[0].status_code == 200
        assert results[0].eval_count == 5

    def test_http_error(self):
        """HTTP error responses are captured."""
        import urllib.error

        prompt = {
            "name": "factual",
            "category": "factual",
            "messages": [],
            "max_tokens": 64,
        }

        with patch(
            "olmlx.bench.runner.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="http://test",
                code=503,
                msg="Busy",
                hdrs=None,
                fp=MagicMock(read=lambda: b"overloaded"),
            ),
        ):
            results = _run_prompts_over_http("model", [prompt], None, 11434)

        assert len(results) == 1
        assert results[0].status_code == 503

    def test_connection_error(self):
        """Network errors are captured."""
        prompt = {
            "name": "factual",
            "category": "factual",
            "messages": [],
            "max_tokens": 64,
        }

        with patch(
            "olmlx.bench.runner.urllib.request.urlopen",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            results = _run_prompts_over_http("model", [prompt], None, 11434)

        assert len(results) == 1
        assert results[0].status_code == 0
        assert "refused" in results[0].error.lower()
