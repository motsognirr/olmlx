"""Regression coverage for olmlx.bench.api_bench run/orchestration paths.

Focuses on the previously-uncovered logic: run_single (streaming + non-stream
timing, tps source selection, error handling, token estimation fallback),
_warmup / _unload best-effort wrappers, _print_table fallback rendering,
build_arg_parser defaults, _get_olmlx_version, and the main() sweep loop.

All HTTP is faked — no network, no real server. httpx is only used to build
Response objects and to be the type the code calls; a FakeClient stands in for
httpx.Client everywhere a request would go out.
"""

from __future__ import annotations

import contextlib
import json

import httpx
import pytest

from olmlx.bench import api_bench
from olmlx.bench.api_bench import (
    OllamaChatAdapter,
    OpenAIChatAdapter,
    RunRecord,
    _get_olmlx_version,
    _print_table,
    _unload,
    _warmup,
    build_arg_parser,
    run_single,
)
from olmlx.bench.prompts import BenchPrompt


_DUMMY_REQ = httpx.Request("POST", "http://x/api/chat")


def make_response(status=200, *, json=None, text=None):
    """Build an httpx.Response with a request attached so raise_for_status works."""
    if json is not None:
        return httpx.Response(status, json=json, request=_DUMMY_REQ)
    if text is not None:
        return httpx.Response(status, text=text, request=_DUMMY_REQ)
    return httpx.Response(status, request=_DUMMY_REQ)


@pytest.fixture
def prompt() -> BenchPrompt:
    return BenchPrompt(
        name="t",
        category="test",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=32,
    )


# --------------------------------------------------------------------------- #
# Fake httpx client                                                           #
# --------------------------------------------------------------------------- #


class _FakeStreamResponse:
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom",
                request=_DUMMY_REQ,
                response=httpx.Response(self.status_code, request=_DUMMY_REQ),
            )

    def iter_lines(self):
        yield from self._lines


class FakeClient:
    """Stands in for httpx.Client. Records calls; replays scripted responses."""

    def __init__(self, *, post_responses=None, stream_lines=None, stream_status=200):
        # post_responses: list of httpx.Response OR Exception to raise, popped FIFO.
        self._post_responses = list(post_responses or [])
        self._stream_lines = stream_lines or []
        self._stream_status = stream_status
        self.post_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def post(self, url, *, json=None, headers=None, timeout=None):
        self.post_calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        if not self._post_responses:
            return make_response(200, json={})
        nxt = self._post_responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    @contextlib.contextmanager
    def stream(self, method, url, *, json=None, headers=None, timeout=None):
        self.stream_calls.append(
            {"method": method, "url": url, "json": json, "timeout": timeout}
        )
        yield _FakeStreamResponse(self._stream_lines, status=self._stream_status)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------- #
# run_single — non-streaming                                                  #
# --------------------------------------------------------------------------- #


def test_run_single_nonstream_uses_eval_duration_for_tps(prompt):
    resp = make_response(
        200,
        json={
            "message": {"content": "hi there friend"},
            "prompt_eval_count": 5,
            "eval_count": 10,
            "eval_duration": 2_000_000_000,  # 2s for 10 tokens -> 5 tok/s
            "prompt_eval_duration": 1_000_000,
        },
    )
    client = FakeClient(post_responses=[resp])
    rec = run_single(
        client, "http://x", OllamaChatAdapter(), prompt, "m", 32, False, 1.0, 0
    )
    assert rec.error is None
    assert rec.output_tokens == 10
    assert rec.prompt_tokens == 5
    assert rec.tps_source == "eval_duration"
    assert rec.tokens_per_sec == pytest.approx(5.0)
    assert rec.tokens_estimated is False
    assert rec.ttft_ms is None  # non-stream never records ttft
    assert rec.mode == "nostream"
    assert rec.api == "ollama-chat"
    # URL joined without double slash.
    assert client.post_calls[0]["url"] == "http://x/api/chat"


def test_run_single_nonstream_estimates_tokens_when_usage_missing(prompt):
    # OpenAI adapter with no usage -> output_tokens estimated from text.
    resp = make_response(
        200,
        json={"choices": [{"message": {"content": "one two three four five"}}]},
    )
    client = FakeClient(post_responses=[resp])
    rec = run_single(
        client, "http://x", OpenAIChatAdapter(), prompt, "m", 32, False, 1.0, 0
    )
    assert rec.tokens_estimated is True
    assert rec.output_tokens is not None and rec.output_tokens > 0
    # No eval_duration and no prompt_eval_duration on this path -> rtt_fallback.
    assert rec.tps_source == "rtt_fallback"
    assert rec.tokens_per_sec is not None


def test_run_single_nonstream_decode_estimate_via_prompt_eval(prompt, monkeypatch):
    # Force total_ms large and prompt_eval_duration present so the
    # decode_estimate (non-stream) branch is taken instead of eval_duration.
    resp = make_response(
        200,
        json={
            "message": {"content": "abc"},
            "eval_count": 8,
            # Small prompt_eval_duration (ns) so total_ms - prompt_eval stays
            # positive even though the faked round-trip is near-instant. With no
            # eval_duration the tps falls through to the non-stream decode
            # estimate branch.
            "prompt_eval_duration": 100,
        },
    )
    client = FakeClient(post_responses=[resp])
    rec = run_single(
        client, "http://x", OllamaChatAdapter(), prompt, "m", 32, False, 1.0, 0
    )
    assert rec.tps_source == "decode_estimate"
    assert rec.tokens_per_sec is not None


def test_run_single_nonstream_http_error_recorded(prompt):
    err_resp = make_response(500, json={"error": "kaboom"})
    client = FakeClient(post_responses=[err_resp])
    rec = run_single(
        client, "http://x", OllamaChatAdapter(), prompt, "m", 32, False, 1.0, 0
    )
    assert rec.error is not None
    assert "HTTPStatusError" in rec.error or "Error" in rec.error
    assert rec.tokens_per_sec is None
    assert rec.total_ms >= 0


def test_run_single_nonstream_exception_before_response(prompt):
    boom = httpx.ConnectError("refused")
    client = FakeClient(post_responses=[boom])
    rec = run_single(
        client, "http://x", OllamaChatAdapter(), prompt, "m", 32, False, 1.0, 0
    )
    assert rec.error is not None
    assert "ConnectError" in rec.error
    assert rec.total_ms >= 0


def test_run_single_zero_output_tokens_no_tps(prompt):
    # Empty text, no usage -> output_tokens stays None, no tps computed.
    resp = make_response(200, json={"message": {"content": ""}})
    client = FakeClient(post_responses=[resp])
    rec = run_single(
        client, "http://x", OllamaChatAdapter(), prompt, "m", 32, False, 1.0, 0
    )
    assert rec.error is None
    assert rec.output_tokens is None
    assert rec.tokens_per_sec is None
    assert rec.tps_source is None


# --------------------------------------------------------------------------- #
# run_single — streaming                                                      #
# --------------------------------------------------------------------------- #


def test_run_single_stream_aggregates_text_and_ttft(prompt):
    lines = [
        json.dumps({"message": {"content": "he"}, "done": False}),
        json.dumps({"message": {"content": "llo"}, "done": False}),
        json.dumps(
            {
                "message": {"content": ""},
                "done": True,
                "eval_count": 4,
                "prompt_eval_count": 9,
                "eval_duration": 1_000_000_000,
            }
        ),
    ]
    client = FakeClient(stream_lines=lines)
    rec = run_single(
        client, "http://x/", OllamaChatAdapter(), prompt, "m", 32, True, 1.0, 0
    )
    assert rec.error is None
    assert rec.mode == "stream"
    assert rec.ttft_ms is not None and rec.ttft_ms >= 0
    assert rec.output_tokens == 4
    assert rec.prompt_tokens == 9
    assert rec.tps_source == "eval_duration"
    # stream call carries the POST body and joined url.
    assert client.stream_calls[0]["url"] == "http://x/api/chat"
    assert client.stream_calls[0]["method"] == "POST"


def test_run_single_stream_decode_estimate_when_no_eval_duration(prompt):
    lines = [
        json.dumps({"message": {"content": "x"}, "done": False}),
        json.dumps(
            {"message": {"content": ""}, "done": True, "eval_count": 3}
        ),  # no eval_duration
    ]
    client = FakeClient(stream_lines=lines)
    rec = run_single(
        client, "http://x", OllamaChatAdapter(), prompt, "m", 32, True, 1.0, 0
    )
    assert rec.output_tokens == 3
    assert rec.tps_source == "decode_estimate"


def test_run_single_stream_missing_done_sets_error(prompt):
    # No done event -> "stream ended without done event" error.
    lines = [json.dumps({"message": {"content": "partial"}, "done": False})]
    client = FakeClient(stream_lines=lines)
    rec = run_single(
        client, "http://x", OllamaChatAdapter(), prompt, "m", 32, True, 1.0, 0
    )
    assert rec.error is not None
    assert "done" in rec.error
    # Text still aggregated; tokens estimated since no usage seen.
    assert rec.tokens_estimated is True


def test_run_single_stream_http_error(prompt):
    client = FakeClient(stream_lines=[], stream_status=503)
    rec = run_single(
        client, "http://x", OllamaChatAdapter(), prompt, "m", 32, True, 1.0, 0
    )
    assert rec.error is not None
    assert "HTTPStatusError" in rec.error


# --------------------------------------------------------------------------- #
# _warmup / _unload                                                           #
# --------------------------------------------------------------------------- #


def test_warmup_posts_nonstream_request():
    client = FakeClient(post_responses=[make_response(200, json={})])
    _warmup(client, "http://x/", "m", 1.0, OllamaChatAdapter())
    assert len(client.post_calls) == 1
    call = client.post_calls[0]
    assert call["url"] == "http://x/api/chat"
    assert call["json"]["stream"] is False
    assert call["json"]["model"] == "m"


def test_warmup_swallows_errors(capsys):
    client = FakeClient(post_responses=[httpx.ConnectError("down")])
    # Must not raise.
    _warmup(client, "http://x", "m", 1.0, OllamaChatAdapter())
    err = capsys.readouterr().err
    assert "warmup" in err


def test_unload_posts_to_api_unload():
    client = FakeClient(post_responses=[make_response(200, json={})])
    _unload(client, "http://x/", "mymodel", 1.0)
    call = client.post_calls[0]
    assert call["url"] == "http://x/api/unload"
    assert call["json"] == {"model": "mymodel"}


def test_unload_logs_http_error_status(capsys):
    client = FakeClient(post_responses=[make_response(404, text="nope")])
    _unload(client, "http://x", "mymodel", 1.0)
    err = capsys.readouterr().err
    assert "unload" in err
    assert "404" in err


def test_unload_swallows_exceptions(capsys):
    client = FakeClient(post_responses=[httpx.ConnectError("x")])
    _unload(client, "http://x", "m", 1.0)
    err = capsys.readouterr().err
    assert "unload" in err


# --------------------------------------------------------------------------- #
# _print_table (fallback path), version, arg parser                           #
# --------------------------------------------------------------------------- #


def _summary_row():
    return {
        "api": "openai-chat",
        "mode": "stream",
        "model": "m",
        "prompt": "p",
        "runs": 2,
        "ttft_p50_ms": 12.5,
        "tps_p50": 50.0,
        "total_p50_ms": 200.0,
    }


def test_print_table_fallback_tabs(monkeypatch, capsys):
    # Force the ImportError fallback by making rich import fail.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("rich"):
            raise ImportError("no rich")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    errors = [
        RunRecord(
            api="x",
            mode="stream",
            model="m",
            prompt="p",
            run_index=0,
            ttft_ms=None,
            total_ms=1.0,
            prompt_tokens=None,
            output_tokens=None,
            tokens_per_sec=None,
            tokens_estimated=False,
            tps_source=None,
            error="boom",
        )
    ]
    _print_table([_summary_row()], errors)
    out = capsys.readouterr()
    assert "openai-chat" in out.out
    assert "12.50" in out.out  # _fmt float formatting
    assert "1 errored runs" in out.err


def test_print_table_fallback_handles_none_float(monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("rich"):
            raise ImportError("no rich")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    row = _summary_row()
    row["ttft_p50_ms"] = None
    row["tps_p50"] = None
    _print_table([row], [])
    out = capsys.readouterr().out
    # None floats render as "-" and there are no error lines.
    assert "-" in out


def test_get_olmlx_version_returns_string():
    v = _get_olmlx_version()
    assert isinstance(v, str)
    assert v != ""


def test_get_olmlx_version_unknown_on_lookup_failure(monkeypatch):
    import importlib.metadata as md

    def boom(_name):
        raise md.PackageNotFoundError("x")

    monkeypatch.setattr(md, "version", boom)
    assert _get_olmlx_version() == "unknown"


def test_arg_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args(["--models", "qwen3:8b"])
    assert args.models == "qwen3:8b"
    assert args.url == "http://localhost:11434"
    assert args.modes == "stream,nostream"
    assert args.runs == 1
    assert args.warmup == 1
    assert args.timeout == pytest.approx(300.0)
    assert args.no_json is False


def test_arg_parser_requires_models():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# --------------------------------------------------------------------------- #
# main() — full sweep with fully faked client                                 #
# --------------------------------------------------------------------------- #


def test_main_runs_sweep_and_writes_json(tmp_path, monkeypatch):
    """Drive main() end-to-end: warmup + one nostream run + unload, write JSON."""
    out_file = tmp_path / "bench.json"

    nostream_resp = make_response(
        200,
        json={
            "message": {"content": "hello world"},
            "prompt_eval_count": 3,
            "eval_count": 5,
            "eval_duration": 1_000_000_000,
            "prompt_eval_duration": 1_000_000,
        },
    )
    # Calls in order: warmup POST, run_single POST (nostream), unload POST.
    client = FakeClient(
        post_responses=[
            make_response(200, json={}),  # warmup
            nostream_resp,  # the single run
            make_response(200, json={}),  # unload
        ]
    )
    monkeypatch.setattr(api_bench.httpx, "Client", lambda *a, **k: client)

    main_argv = [
        "--models",
        "m",
        "--apis",
        "ollama-chat",
        "--modes",
        "nostream",
        "--prompts",
        "factual",
        "--runs",
        "1",
        "--warmup",
        "1",
        "--output",
        str(out_file),
    ]
    rc = api_bench.main(main_argv)
    assert rc == 0  # no errored runs
    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert payload["server_url"] == "http://localhost:11434"
    assert payload["records"]
    rec = payload["records"][0]
    assert rec["api"] == "ollama-chat"
    assert rec["output_tokens"] == 5
    assert payload["summary"]
    # Three POSTs total: warmup, run, unload.
    assert len(client.post_calls) == 3


def test_main_returns_1_on_errored_run(tmp_path, monkeypatch):
    # warmup=0 -> first POST is the run itself, second is the unload.
    client = FakeClient(
        post_responses=[
            make_response(500, json={"error": "boom"}),  # run -> error
            make_response(200, json={}),  # unload
        ]
    )
    monkeypatch.setattr(api_bench.httpx, "Client", lambda *a, **k: client)
    rc = api_bench.main(
        [
            "--models",
            "m",
            "--apis",
            "ollama-chat",
            "--modes",
            "nostream",
            "--prompts",
            "factual",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--no-json",
        ]
    )
    assert rc == 1


def test_main_empty_models_raises(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(api_bench.httpx, "Client", lambda *a, **k: client)
    with pytest.raises(SystemExit):
        api_bench.main(["--models", " , ", "--no-json"])


def test_main_default_output_path_under_home(tmp_path, monkeypatch):
    # Redirect Path.home() so the default ~/.olmlx/bench path lands in tmp_path.
    monkeypatch.setattr(api_bench.Path, "home", staticmethod(lambda: tmp_path))
    client = FakeClient(
        post_responses=[
            make_response(
                200,
                json={"message": {"content": "hi"}, "eval_count": 1},
            ),
            make_response(200, json={}),  # unload
        ]
    )
    monkeypatch.setattr(api_bench.httpx, "Client", lambda *a, **k: client)
    rc = api_bench.main(
        [
            "--models",
            "m",
            "--apis",
            "ollama-chat",
            "--modes",
            "nostream",
            "--prompts",
            "factual",
            "--runs",
            "1",
            "--warmup",
            "0",
        ]
    )
    assert rc == 0
    bench_dir = tmp_path / ".olmlx" / "bench"
    written = list(bench_dir.glob("api_bench_*.json"))
    assert len(written) == 1
