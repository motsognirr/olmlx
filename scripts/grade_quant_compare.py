"""Ad-hoc quality grading: run a model over the mini task-sets and grade.

Usage: python scripts/grade_quant_compare.py <hf-path> <out.json> [--sets gsm8k,mmlu,humaneval]

Starts its own `olmlx serve` subprocess (like the bench worker), sends each
task prompt over HTTP with temp=0/seed=42, then grades the output with the
existing olmlx.bench.quality graders. code_exec is enabled for HumanEval.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

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


def _chat(port: int, model: str, prompt, max_tokens: int | None = None) -> str:
    body = {
        "model": model,
        "stream": False,
        "messages": prompt.messages,
        "options": {
            "seed": 42,
            "temperature": 0.0,
            "num_predict": max_tokens or prompt.max_tokens,
        },
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode()).get("message", {}).get("content", "")


def main() -> None:
    model = sys.argv[1]
    out_path = sys.argv[2]
    set_names = ["gsm8k", "mmlu", "humaneval"]
    max_tokens_override: int | None = None
    for i, a in enumerate(sys.argv):
        if a == "--sets":
            set_names = sys.argv[i + 1].split(",")
        if a == "--max-tokens":
            max_tokens_override = int(sys.argv[i + 1])

    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "olmlx", "serve"],
        env={**os.environ, "OLMLX_PORT": str(port)},
        stdout=subprocess.DEVNULL,
        stderr=sys.stderr,
    )
    results: list[dict] = []
    try:
        if not _wait(port, proc):
            raise SystemExit(f"server failed to start (exit={proc.poll()})")
        for set_name in set_names:
            for p in PROMPT_SETS[set_name]:
                expected = dict(p.expected)
                if p.grader == "code_exec":
                    expected["_enabled"] = True
                out = _chat(port, model, p, max_tokens_override)
                q = grade(p.grader, out, expected)
                results.append(
                    {
                        "set": set_name,
                        "name": p.name,
                        "grader": p.grader,
                        "passed": q.passed,
                        "score": q.score,
                        "detail": q.detail,
                    }
                )
                mark = {True: "PASS", False: "FAIL", None: "----"}[q.passed]
                print(f"  [{mark}] {p.name}: {q.detail}", file=sys.stderr)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    with open(out_path, "w") as f:
        json.dump({"model": model, "results": results}, f, indent=2)
    print(f"\nsaved {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
