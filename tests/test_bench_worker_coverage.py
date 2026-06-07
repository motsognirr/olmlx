"""Regression coverage for olmlx.bench.worker.

Exercises the subprocess worker's server-wait poll loop, prompt dispatch over
HTTP (success / HTTPError / generic exception), the think-value recognizer, and
the ``main`` entry point — all with urllib and subprocess fully mocked. No
network, no real server, no real model.
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from olmlx.bench import worker
from olmlx.bench.worker import (
    _run_prompts,
    _wait_for_server,
    is_recognized_think_value,
)


# --------------------------------------------------------------------------
# is_recognized_think_value
# --------------------------------------------------------------------------
class TestIsRecognizedThinkValue:
    @pytest.mark.parametrize(
        "value", ["true", "True", " 1 ", "ON", "yes", "false", "0", "off", "no"]
    )
    def test_recognized(self, value):
        assert is_recognized_think_value(value) is True

    @pytest.mark.parametrize("value", ["tru", "enabled", "", "maybe", "2"])
    def test_unrecognized(self, value):
        assert is_recognized_think_value(value) is False


# --------------------------------------------------------------------------
# _wait_for_server
# --------------------------------------------------------------------------
class TestWaitForServer:
    def _proc(self, poll_return=None):
        proc = MagicMock()
        proc.poll.return_value = poll_return
        return proc

    def test_returns_true_on_http_200(self):
        proc = self._proc(poll_return=None)
        resp = MagicMock()
        resp.status = 200
        cm = MagicMock()
        cm.__enter__.return_value = resp
        cm.__exit__.return_value = False
        with patch.object(worker.urllib.request, "urlopen", return_value=cm):
            assert _wait_for_server(11435, proc, timeout=5) is True

    def test_returns_false_when_process_dead(self):
        # Process already exited -> no HTTP attempt, immediate False.
        proc = self._proc(poll_return=1)
        with patch.object(worker.urllib.request, "urlopen") as uo:
            assert _wait_for_server(11435, proc, timeout=5) is False
            uo.assert_not_called()

    def test_retries_then_succeeds(self):
        proc = self._proc(poll_return=None)
        resp = MagicMock()
        resp.status = 200
        cm = MagicMock()
        cm.__enter__.return_value = resp
        cm.__exit__.return_value = False
        calls = {"n": 0}

        def fake_urlopen(req, timeout=5):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.URLError("conn refused")
            return cm

        with (
            patch.object(worker.urllib.request, "urlopen", side_effect=fake_urlopen),
            patch.object(worker.time, "sleep") as sleep,
        ):
            assert _wait_for_server(11435, proc, timeout=5) is True
        assert calls["n"] == 2
        # The connection error path must sleep before retrying.
        sleep.assert_called()

    def test_returns_false_on_timeout(self):
        proc = self._proc(poll_return=None)
        # monotonic: first call sets deadline (=100+5), while-check is past it.
        times = iter([100.0, 200.0, 300.0])
        with (
            patch.object(worker.time, "monotonic", side_effect=lambda: next(times)),
            patch.object(worker.urllib.request, "urlopen") as uo,
        ):
            assert _wait_for_server(11435, proc, timeout=5) is False
            uo.assert_not_called()

    def test_non_200_status_keeps_polling_until_timeout(self):
        proc = self._proc(poll_return=None)
        resp = MagicMock()
        resp.status = 503
        cm = MagicMock()
        cm.__enter__.return_value = resp
        cm.__exit__.return_value = False
        # deadline: enter loop once, then expire.
        times = iter([0.0, 1.0, 1000.0])
        with (
            patch.object(worker.time, "monotonic", side_effect=lambda: next(times)),
            patch.object(worker.urllib.request, "urlopen", return_value=cm),
            patch.object(worker.time, "sleep"),
        ):
            assert _wait_for_server(11435, proc, timeout=5) is False


# --------------------------------------------------------------------------
# _run_prompts
# --------------------------------------------------------------------------
def _prompt(name="p1", category="general", max_tokens=None):
    p = {
        "name": name,
        "category": category,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if max_tokens is not None:
        p["max_tokens"] = max_tokens
    return p


def _ok_response(payload, status=200):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(payload).encode()
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


class TestRunPrompts:
    def test_success_maps_ollama_fields(self, monkeypatch):
        monkeypatch.delenv("OLMLX_BENCH_THINK", raising=False)
        payload = {
            "message": {"content": "hello world"},
            "eval_count": 12,
            "eval_duration": 999,
            "prompt_eval_count": 3,
            "prompt_eval_duration": 7,
            "total_duration": 1234,
        }
        captured = {}

        def fake_urlopen(req, timeout=300):
            captured["req"] = req
            return _ok_response(payload)

        with patch.object(worker.urllib.request, "urlopen", side_effect=fake_urlopen):
            results = _run_prompts(11435, "qwen3", [_prompt()], None)

        assert len(results) == 1
        r = results[0]
        assert r["output_text"] == "hello world"
        assert r["status_code"] == 200
        assert r["error"] is None
        assert r["eval_count"] == 12
        assert r["eval_duration_ns"] == 999
        assert r["prompt_eval_count"] == 3
        assert r["prompt_eval_duration_ns"] == 7
        assert r["total_duration_ns"] == 1234
        # Request body sanity: model, deterministic options, no think field.
        body = json.loads(captured["req"].data.decode())
        assert body["model"] == "qwen3"
        assert body["stream"] is False
        assert body["options"]["seed"] == 42
        assert body["options"]["temperature"] == 0.0
        assert body["options"]["num_predict"] == 256  # default fallback
        assert "think" not in body

    def test_max_tokens_override_beats_prompt(self, monkeypatch):
        monkeypatch.delenv("OLMLX_BENCH_THINK", raising=False)
        captured = {}

        def fake_urlopen(req, timeout=300):
            captured["body"] = json.loads(req.data.decode())
            return _ok_response({"message": {"content": "x"}})

        with patch.object(worker.urllib.request, "urlopen", side_effect=fake_urlopen):
            _run_prompts(11435, "m", [_prompt(max_tokens=99)], max_tokens_override=512)
        assert captured["body"]["options"]["num_predict"] == 512

    def test_prompt_max_tokens_used_when_no_override(self, monkeypatch):
        monkeypatch.delenv("OLMLX_BENCH_THINK", raising=False)
        captured = {}

        def fake_urlopen(req, timeout=300):
            captured["body"] = json.loads(req.data.decode())
            return _ok_response({"message": {"content": "x"}})

        with patch.object(worker.urllib.request, "urlopen", side_effect=fake_urlopen):
            _run_prompts(11435, "m", [_prompt(max_tokens=77)], None)
        assert captured["body"]["options"]["num_predict"] == 77

    def test_think_true_adds_think_field(self, monkeypatch):
        monkeypatch.setenv("OLMLX_BENCH_THINK", "true")
        captured = {}

        def fake_urlopen(req, timeout=300):
            captured["body"] = json.loads(req.data.decode())
            return _ok_response({"message": {"content": "x"}})

        with patch.object(worker.urllib.request, "urlopen", side_effect=fake_urlopen):
            _run_prompts(11435, "m", [_prompt()], None)
        assert captured["body"]["think"] is True

    def test_think_false_adds_think_field(self, monkeypatch):
        monkeypatch.setenv("OLMLX_BENCH_THINK", "off")
        captured = {}

        def fake_urlopen(req, timeout=300):
            captured["body"] = json.loads(req.data.decode())
            return _ok_response({"message": {"content": "x"}})

        with patch.object(worker.urllib.request, "urlopen", side_effect=fake_urlopen):
            _run_prompts(11435, "m", [_prompt()], None)
        assert captured["body"]["think"] is False

    def test_missing_message_content_defaults_empty(self, monkeypatch):
        monkeypatch.delenv("OLMLX_BENCH_THINK", raising=False)
        with patch.object(
            worker.urllib.request, "urlopen", return_value=_ok_response({})
        ):
            results = _run_prompts(11435, "m", [_prompt()], None)
        r = results[0]
        assert r["output_text"] == ""
        assert r["eval_count"] == 0
        assert r["total_duration_ns"] == 0

    def test_http_error_captures_error_body(self, monkeypatch):
        monkeypatch.delenv("OLMLX_BENCH_THINK", raising=False)
        err = urllib.error.HTTPError(
            url="http://x",
            code=500,
            msg="boom",
            hdrs=None,
            fp=io.BytesIO(b"internal explosion detail"),
        )
        with patch.object(worker.urllib.request, "urlopen", side_effect=err):
            results = _run_prompts(11435, "m", [_prompt(name="bad")], None)
        r = results[0]
        assert r["prompt_name"] == "bad"
        assert r["status_code"] == 500
        assert r["error"] == "internal explosion detail"
        assert r["output_text"] == ""

    def test_http_error_body_truncated_to_500_chars(self, monkeypatch):
        monkeypatch.delenv("OLMLX_BENCH_THINK", raising=False)
        big = b"E" * 2000
        err = urllib.error.HTTPError(
            url="http://x", code=502, msg="bad", hdrs=None, fp=io.BytesIO(big)
        )
        with patch.object(worker.urllib.request, "urlopen", side_effect=err):
            results = _run_prompts(11435, "m", [_prompt()], None)
        assert len(results[0]["error"]) == 500

    def test_http_error_read_failure_falls_back(self, monkeypatch):
        monkeypatch.delenv("OLMLX_BENCH_THINK", raising=False)
        err = urllib.error.HTTPError(
            url="http://x", code=503, msg="bad", hdrs=None, fp=None
        )
        # .read() on an HTTPError with fp=None raises; the except must catch it.
        with patch.object(err, "read", side_effect=ValueError("no fp")):
            with patch.object(worker.urllib.request, "urlopen", side_effect=err):
                results = _run_prompts(11435, "m", [_prompt()], None)
        assert results[0]["error"] == "HTTP 503"
        assert results[0]["status_code"] == 503

    def test_generic_exception_captured(self, monkeypatch):
        monkeypatch.delenv("OLMLX_BENCH_THINK", raising=False)
        with patch.object(
            worker.urllib.request,
            "urlopen",
            side_effect=urllib.error.URLError("conn refused"),
        ):
            results = _run_prompts(11435, "m", [_prompt(name="z")], None)
        r = results[0]
        assert r["status_code"] == 0
        assert "conn refused" in r["error"]
        assert r["prompt_name"] == "z"

    def test_multiple_prompts_preserve_order(self, monkeypatch):
        monkeypatch.delenv("OLMLX_BENCH_THINK", raising=False)
        responses = [
            _ok_response({"message": {"content": "a"}}),
            _ok_response({"message": {"content": "b"}}),
        ]
        with patch.object(worker.urllib.request, "urlopen", side_effect=responses):
            results = _run_prompts(
                11435, "m", [_prompt(name="p1"), _prompt(name="p2")], None
            )
        assert [r["prompt_name"] for r in results] == ["p1", "p2"]
        assert [r["output_text"] for r in results] == ["a", "b"]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
class TestMain:
    def _args(self, tmp_path, prompts):
        prompts_json = tmp_path / "prompts.json"
        results_json = tmp_path / "results.json"
        prompts_json.write_text(json.dumps(prompts))
        return prompts_json, results_json

    def test_happy_path_writes_results(self, tmp_path, monkeypatch):
        prompts = [_prompt(name="p1")]
        prompts_json, results_json = self._args(tmp_path, prompts)
        argv = [
            "worker",
            "--model",
            "qwen3",
            "--prompts-json",
            str(prompts_json),
            "--results-json",
            str(results_json),
        ]
        proc = MagicMock()
        proc.poll.return_value = None  # still running during finally
        fake_results = [{"prompt_name": "p1", "output_text": "done"}]

        with (
            patch.object(worker.sys, "argv", argv),
            patch.object(worker.subprocess, "Popen", return_value=proc),
            patch.object(worker, "_wait_for_server", return_value=True),
            patch.object(worker, "_run_prompts", return_value=fake_results) as run_mock,
        ):
            worker.main()

        written = json.loads(results_json.read_text())
        assert written == fake_results
        run_mock.assert_called_once()
        # finally block terminates a still-running server.
        proc.terminate.assert_called_once()
        proc.wait.assert_called()

    def test_server_failed_to_start_writes_error_record(self, tmp_path):
        prompts_json, results_json = self._args(tmp_path, [_prompt()])
        argv = [
            "worker",
            "--model",
            "m",
            "--prompts-json",
            str(prompts_json),
            "--results-json",
            str(results_json),
        ]
        proc = MagicMock()
        # poll() returns 7 (dead) so main records exit code, and finally skips term.
        proc.poll.return_value = 7
        proc.returncode = 7

        with (
            patch.object(worker.sys, "argv", argv),
            patch.object(worker.subprocess, "Popen", return_value=proc),
            patch.object(worker, "_wait_for_server", return_value=False),
            patch.object(worker, "_run_prompts") as run_mock,
        ):
            worker.main()

        written = json.loads(results_json.read_text())
        assert len(written) == 1
        rec = written[0]
        assert rec["prompt_name"] == "__server_error__"
        assert rec["category"] == "error"
        assert "exit=7" in rec["error"]
        # No prompts run when the server never came up.
        run_mock.assert_not_called()
        # Server already dead -> no terminate.
        proc.terminate.assert_not_called()

    def test_server_timeout_records_timeout_marker(self, tmp_path):
        prompts_json, results_json = self._args(tmp_path, [_prompt()])
        argv = [
            "worker",
            "--model",
            "m",
            "--prompts-json",
            str(prompts_json),
            "--results-json",
            str(results_json),
        ]
        proc = MagicMock()
        proc.poll.return_value = None  # never exited -> "timeout" string

        with (
            patch.object(worker.sys, "argv", argv),
            patch.object(worker.subprocess, "Popen", return_value=proc),
            patch.object(worker, "_wait_for_server", return_value=False),
        ):
            worker.main()

        rec = json.loads(results_json.read_text())[0]
        assert "exit=timeout" in rec["error"]
        # Process still alive at the failure branch -> finally terminates it.
        proc.terminate.assert_called_once()

    def test_kill_on_terminate_timeout(self, tmp_path):
        prompts_json, results_json = self._args(tmp_path, [_prompt()])
        argv = [
            "worker",
            "--model",
            "m",
            "--prompts-json",
            str(prompts_json),
            "--results-json",
            str(results_json),
        ]
        proc = MagicMock()
        proc.poll.return_value = None
        # First wait (after terminate) times out -> kill, second wait returns.
        proc.wait.side_effect = [
            worker.subprocess.TimeoutExpired(cmd="x", timeout=15),
            None,
        ]

        with (
            patch.object(worker.sys, "argv", argv),
            patch.object(worker.subprocess, "Popen", return_value=proc),
            patch.object(worker, "_wait_for_server", return_value=True),
            patch.object(worker, "_run_prompts", return_value=[]),
        ):
            worker.main()

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert proc.wait.call_count == 2

    def test_kill_wait_also_times_out_is_swallowed(self, tmp_path):
        prompts_json, results_json = self._args(tmp_path, [_prompt()])
        argv = [
            "worker",
            "--model",
            "m",
            "--prompts-json",
            str(prompts_json),
            "--results-json",
            str(results_json),
        ]
        proc = MagicMock()
        proc.poll.return_value = None
        # Both the terminate-wait and the post-kill wait time out; main must
        # not propagate the second TimeoutExpired.
        proc.wait.side_effect = [
            worker.subprocess.TimeoutExpired(cmd="x", timeout=15),
            worker.subprocess.TimeoutExpired(cmd="x", timeout=5),
        ]

        with (
            patch.object(worker.sys, "argv", argv),
            patch.object(worker.subprocess, "Popen", return_value=proc),
            patch.object(worker, "_wait_for_server", return_value=True),
            patch.object(worker, "_run_prompts", return_value=[]),
        ):
            worker.main()  # should return cleanly

        proc.kill.assert_called_once()
        assert proc.wait.call_count == 2
