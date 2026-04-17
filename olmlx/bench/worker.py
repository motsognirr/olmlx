"""Subprocess worker — runs prompts against a single scenario configuration.

Invoked as: python -m olmlx.bench.worker --model MODEL --prompts-json PATH --results-json PATH
Environment variables control the scenario's feature flags (set by runner before spawn).

Starts a real olmlx server (uvicorn) inside the subprocess and sends prompts
over HTTP with urllib. This tests the full stack (lifespan, middleware, routers)
without requiring httpx (a dev-only dependency).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_SERVER_PORT = 11435  # Use a non-default port to avoid conflicts
_SERVER_STARTUP_TIMEOUT = 120
_SERVER_POLL_INTERVAL = 1


def _wait_for_server(port: int, proc: subprocess.Popen, timeout: float) -> bool:
    """Poll until the server responds to GET / or the process dies."""
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(_SERVER_POLL_INTERVAL)
    return False


def _run_prompts(
    port: int, model: str, prompts: list[dict], max_tokens_override: int | None
) -> list[dict]:
    """Send prompts to a running server over HTTP."""
    url = f"http://127.0.0.1:{port}/api/chat"
    results = []
    for prompt in prompts:
        tok_limit = max_tokens_override or prompt.get("max_tokens", 256)
        body = {
            "model": model,
            "stream": False,
            "messages": prompt["messages"],
            "options": {
                "seed": 42,
                "temperature": 0.0,
                "num_predict": tok_limit,
            },
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
                results.append(
                    {
                        "prompt_name": prompt["name"],
                        "category": prompt["category"],
                        "output_text": data.get("message", {}).get("content", ""),
                        "status_code": resp.status,
                        "error": None,
                        "eval_count": data.get("eval_count", 0),
                        "eval_duration_ns": data.get("eval_duration", 0),
                        "prompt_eval_count": data.get("prompt_eval_count", 0),
                        "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
                        "total_duration_ns": data.get("total_duration", 0),
                    }
                )
        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode(errors="replace")[:500]
            except Exception:
                error_body = f"HTTP {e.code}"
            results.append(
                {
                    "prompt_name": prompt["name"],
                    "category": prompt["category"],
                    "output_text": "",
                    "status_code": e.code,
                    "error": error_body,
                    "eval_count": 0,
                    "eval_duration_ns": 0,
                    "prompt_eval_count": 0,
                    "prompt_eval_duration_ns": 0,
                    "total_duration_ns": 0,
                }
            )
        except Exception as e:
            results.append(
                {
                    "prompt_name": prompt["name"],
                    "category": prompt["category"],
                    "output_text": "",
                    "status_code": 0,
                    "error": str(e),
                    "eval_count": 0,
                    "eval_duration_ns": 0,
                    "prompt_eval_count": 0,
                    "prompt_eval_duration_ns": 0,
                    "total_duration_ns": 0,
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="Bench worker subprocess")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--results-json", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--port", type=int, default=_SERVER_PORT)
    args = parser.parse_args()

    prompts = json.loads(args.prompts_json.read_text())

    # Start olmlx serve inside this subprocess (inherits env vars set by runner)
    cmd = [sys.executable, "-m", "olmlx", "serve"]
    env = {**os.environ, "OLMLX_PORT": str(args.port)}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=sys.stderr,
        env=env,
    )

    try:
        if not _wait_for_server(args.port, proc, _SERVER_STARTUP_TIMEOUT):
            rc = proc.returncode if proc.poll() is not None else "timeout"
            args.results_json.write_text(
                json.dumps(
                    [
                        {
                            "prompt_name": "__server_error__",
                            "category": "error",
                            "output_text": "",
                            "status_code": 0,
                            "error": f"Server failed to start (exit={rc})",
                            "eval_count": 0,
                            "eval_duration_ns": 0,
                            "prompt_eval_count": 0,
                            "prompt_eval_duration_ns": 0,
                            "total_duration_ns": 0,
                        }
                    ]
                )
            )
            return

        results = _run_prompts(args.port, args.model, prompts, args.max_tokens)
        args.results_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass


if __name__ == "__main__":
    main()
