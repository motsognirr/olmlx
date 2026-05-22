"""Ad-hoc quality grading: run a model over the mini task-sets and grade.

Usage: python scripts/grade_quant_compare.py <hf-path> <out.json>
           [--sets gsm8k,mmlu,humaneval] [--max-tokens N]

Starts its own `olmlx serve` subprocess (like the bench worker), sends each
task prompt over HTTP with temp=0/seed=42, then grades the output with the
existing olmlx.bench.quality graders. code_exec is enabled for HumanEval.
A failed request is recorded (passed=null) and the run continues; results are
written incrementally after each prompt, so partial output survives an early
exit or hard crash (at most the in-flight prompt is lost).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from olmlx.bench.prompts import BenchPrompt
from olmlx.bench.quality import grade
from olmlx.bench.task_prompts import PROMPT_SETS


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(port: int, proc: subprocess.Popen, timeout: float = 180) -> bool:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(1)
    return False


def _start_server(attempts: int = 3) -> tuple[subprocess.Popen, int]:
    """Start `olmlx serve` on a free port, retrying on a fresh port if it
    fails to come up. `_free_port()` releases the socket before the subprocess
    binds it, so a collision is possible under load — retrying makes that race
    non-fatal instead of dying with a misleading "server failed to start"."""
    last_exit: int | str | None = None
    for _ in range(attempts):
        port = _free_port()
        proc = subprocess.Popen(
            [sys.executable, "-m", "olmlx", "serve"],
            env={**os.environ, "OLMLX_PORT": str(port)},
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )
        if _wait(port, proc):
            return proc, port
        # poll() is None when the server is alive but never answered (timeout);
        # report that distinctly rather than a misleading "exit=None".
        exit_code = proc.poll()
        last_exit = "timeout" if exit_code is None else exit_code
        if exit_code is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)  # reap so a retried attempt leaves no zombie
    raise SystemExit(
        f"server failed to start after {attempts} attempts (exit={last_exit})"
    )


def _chat(
    port: int,
    model: str,
    prompt: BenchPrompt,
    max_tokens: int | None = None,
    timeout: float = 1800,
) -> str:
    body = {
        "model": model,
        "stream": False,
        "messages": prompt.messages,
        "options": {
            "seed": 42,
            "temperature": 0.0,
            "num_predict": max_tokens if max_tokens is not None else prompt.max_tokens,
        },
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # stream=False means the socket is silent during the whole load+decode
    # window, so this is effectively a wall-clock budget for one request.
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()).get("message", {}).get("content", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model name or HF path to serve and grade")
    parser.add_argument("out_path", help="Where to write the results JSON")
    parser.add_argument(
        "--sets",
        default="gsm8k,mmlu,humaneval",
        help="Comma-separated task sets (default: gsm8k,mmlu,humaneval)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,  # low per-prompt defaults (MMLU 128) truncate reasoning models
        dest="max_tokens",
        help="num_predict for every prompt (default: 1024)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1800,
        dest="request_timeout",
        help="per-request wall-clock budget in seconds; raise for slow cold "
        "loads of large models (default: 1800)",
    )
    args = parser.parse_args()
    set_names = args.sets.split(",")
    if unknown := set(set_names) - PROMPT_SETS.keys():
        parser.error(
            f"unknown task sets: {', '.join(sorted(unknown))}; "
            f"valid: {', '.join(sorted(PROMPT_SETS))}"
        )

    # Fail fast on a bad output path before the (slow) model load, so a typo'd
    # or unwritable destination doesn't surface only after the first _save().
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Writability probe in append mode: raises PermissionError for a bad path
    # before the slow model load, without truncating an existing result file.
    with open(out_path, "a"):
        pass

    def _save(results: list[dict]) -> None:
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "results": results}, f, indent=2)
            f.write("\n")

    proc, port = _start_server()
    results: list[dict] = []
    code_exec_announced = False
    try:
        for set_name in set_names:
            for p in PROMPT_SETS[set_name]:
                expected = dict(p.expected)
                if p.grader == "code_exec":
                    expected["_enabled"] = True
                    if not code_exec_announced:
                        print(
                            "  [INFO] code_exec enabled — model-generated Python "
                            "will run in a sandboxed subprocess",
                            file=sys.stderr,
                        )
                        code_exec_announced = True
                try:
                    out = _chat(
                        port, args.model, p, args.max_tokens, args.request_timeout
                    )
                    q = grade(p.grader, out, expected)
                    entry = {
                        "set": set_name,
                        "name": p.name,
                        "grader": p.grader,
                        "passed": q.passed,
                        "score": q.score,
                        "detail": q.detail,
                    }
                    mark = {True: "PASS", False: "FAIL", None: "----"}[q.passed]
                    print(f"  [{mark}] {p.name}: {q.detail}", file=sys.stderr)
                except Exception as e:  # noqa: BLE001 — one bad request shouldn't lose the run
                    entry = {
                        "set": set_name,
                        "name": p.name,
                        "grader": p.grader,
                        "passed": None,
                        "score": None,  # ungraded ≠ graded-wrong (quality.py convention)
                        "detail": f"request failed: {type(e).__name__}: {e}",
                    }
                    print(f"  [ERR ] {p.name}: {entry['detail']}", file=sys.stderr)
                results.append(entry)
                _save(results)  # incremental: survives a hard crash mid-run
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if results:
            print(f"\nsaved {args.out_path} ({len(results)} results)", file=sys.stderr)


if __name__ == "__main__":
    main()
