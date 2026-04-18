"""Orchestrates benchmark runs — spawns worker subprocesses per scenario."""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from olmlx.bench.prompts import PROMPTS
from olmlx.bench.results import (
    DEFAULT_BENCH_DIR,
    PromptResult,
    RunResult,
    ScenarioResult,
    build_leaderboard,
    create_run_result,
    format_leaderboard,
    save_run,
)
from olmlx.bench.scenarios import Scenario, get_scenarios

logger = logging.getLogger(__name__)

_WORKER_TIMEOUT = 600  # seconds before killing a worker subprocess


def _find_free_port() -> int:
    """Bind to port 0 to get an OS-assigned ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_SERVER_STARTUP_TIMEOUT = 300  # seconds to wait for olmlx serve to become ready
_SERVER_READY_POLL_INTERVAL = 2  # seconds between readiness checks


def _resolve_model_path(model: str) -> Path:
    """Resolve an HF path to its local directory."""
    from olmlx.config import settings
    from olmlx.models.store import _safe_dir_name

    return settings.models_dir / _safe_dir_name(model)


def run_bench(
    model: str,
    scenario_names: list[str] | None = None,
    max_tokens: int | None = None,
    bench_dir: Path = DEFAULT_BENCH_DIR,
) -> RunResult:
    """Run all scenarios and return aggregated results."""
    scenarios = get_scenarios(scenario_names)
    model_path = _resolve_model_path(model)
    prompts_data = [p.to_dict() for p in PROMPTS]

    scenario_results: list[ScenarioResult] = []

    for i, scenario in enumerate(scenarios, 1):
        print(
            f"[{i}/{len(scenarios)}] {scenario.name}: {scenario.description}",
            file=sys.stderr,
        )

        # Check skip condition
        skip_reason = scenario.should_skip(model_path)
        if skip_reason is not None:
            print(f"  SKIPPED: {skip_reason}", file=sys.stderr)
            scenario_results.append(
                ScenarioResult(
                    scenario_name=scenario.name,
                    scenario_description=scenario.description,
                    env_overrides=scenario.env_overrides,
                    prompt_results=[],
                    skipped=True,
                    skip_reason=skip_reason,
                )
            )
            continue

        # Run worker subprocess (or server for distributed scenarios)
        if scenario.server_mode:
            prompt_results = _run_server_scenario(
                model, scenario, prompts_data, max_tokens
            )
        else:
            prompt_results = _run_worker(model, scenario, prompts_data, max_tokens)

        # Report summary
        ok = sum(1 for r in prompt_results if r.status_code == 200)
        fail = len(prompt_results) - ok
        avg_tps = 0.0
        tps_values = [
            r.tokens_per_second for r in prompt_results if r.tokens_per_second > 0
        ]
        if tps_values:
            avg_tps = sum(tps_values) / len(tps_values)
        status = "OK" if fail == 0 else f"{fail} FAILED"
        print(
            f"  {status} — {ok}/{len(prompt_results)} prompts, avg {avg_tps:.1f} tok/s",
            file=sys.stderr,
        )

        scenario_results.append(
            ScenarioResult(
                scenario_name=scenario.name,
                scenario_description=scenario.description,
                env_overrides=scenario.env_overrides,
                prompt_results=prompt_results,
            )
        )

    run = create_run_result(
        model=model,
        scenarios=scenario_results,
        max_tokens_override=max_tokens,
    )
    run_dir = save_run(run, bench_dir)
    print(f"\nResults saved to {run_dir}", file=sys.stderr)

    try:
        entries = build_leaderboard(bench_dir)
        if entries:
            print("\nLeaderboard (top 5):", file=sys.stderr)
            print(format_leaderboard(entries, limit=5), file=sys.stderr)
    except Exception:
        logger.warning("Could not build leaderboard", exc_info=True)

    return run


def _run_worker(
    model: str,
    scenario: Scenario,
    prompts_data: list[dict],
    max_tokens: int | None,
) -> list[PromptResult]:
    """Spawn a subprocess for a single scenario and collect results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_path = Path(tmpdir) / "prompts.json"
        results_path = Path(tmpdir) / "results.json"
        prompts_path.write_text(json.dumps(prompts_data))

        # Build env: inherit current env + scenario overrides
        env = os.environ.copy()
        env.update(scenario.env_overrides)

        port = _find_free_port()
        cmd = [
            sys.executable,
            "-m",
            "olmlx.bench.worker",
            "--model",
            model,
            "--prompts-json",
            str(prompts_path),
            "--results-json",
            str(results_path),
            "--port",
            str(port),
        ]
        if max_tokens is not None:
            cmd.extend(["--max-tokens", str(max_tokens)])

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=_WORKER_TIMEOUT,
            )
            if result.returncode != 0:
                logger.error(
                    "Worker failed for %s (exit %d):\n%s",
                    scenario.name,
                    result.returncode,
                    result.stderr[-1000:] if result.stderr else "(no stderr)",
                )
                return [
                    PromptResult(
                        prompt_name="__worker_error__",
                        category="error",
                        output_text="",
                        status_code=0,
                        error=f"Worker exited with code {result.returncode}: {result.stderr[-500:] if result.stderr else ''}",
                    )
                ]

            if not results_path.exists():
                return [
                    PromptResult(
                        prompt_name="__worker_error__",
                        category="error",
                        output_text="",
                        status_code=0,
                        error="Worker did not produce results file",
                    )
                ]

            raw = json.loads(results_path.read_text())
            return [PromptResult.from_dict(r) for r in raw]

        except subprocess.TimeoutExpired:
            return [
                PromptResult(
                    prompt_name="__worker_error__",
                    category="error",
                    output_text="",
                    status_code=0,
                    error=f"Worker timed out after {_WORKER_TIMEOUT}s",
                )
            ]


def _get_server_port(scenario: Scenario) -> int:
    """Determine the port the server will listen on."""
    port_str = scenario.env_overrides.get(
        "OLMLX_PORT", os.environ.get("OLMLX_PORT", "11434")
    )
    return int(port_str)


def _wait_for_server(port: int, proc: subprocess.Popen, timeout: float) -> bool:
    """Poll the server until it responds to GET / or the process dies."""
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/"
    while time.monotonic() < deadline:
        # Check if the process has died
        if proc.poll() is not None:
            return False
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(_SERVER_READY_POLL_INTERVAL)
    return False


def _run_server_scenario(
    model: str,
    scenario: Scenario,
    prompts_data: list[dict],
    max_tokens: int | None,
) -> list[PromptResult]:
    """Launch olmlx serve, run prompts over HTTP, then shut down.

    Used for distributed scenarios where the full CLI startup sequence
    (ring init, SSH workers, sideband server) is required.
    """
    # For distributed, the model comes from the hostfile, but we still
    # use the CLI --model flag for the API requests.
    port = _get_server_port(scenario)

    env = os.environ.copy()
    env.update(scenario.env_overrides)

    cmd = [sys.executable, "-m", "olmlx", "serve", "--port", str(port)]
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )

        print(f"  Waiting for server on port {port}...", file=sys.stderr)
        if not _wait_for_server(port, proc, _SERVER_STARTUP_TIMEOUT):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            rc = proc.returncode
            return [
                PromptResult(
                    prompt_name="__server_error__",
                    category="error",
                    output_text="",
                    status_code=0,
                    error=f"Server failed to start within {_SERVER_STARTUP_TIMEOUT}s (exit code {rc})",
                )
            ]

        print("  Server ready, running prompts...", file=sys.stderr)

        # Run prompts over HTTP
        results = _run_prompts_over_http(model, prompts_data, max_tokens, port)
        return results

    except Exception as e:
        return [
            PromptResult(
                prompt_name="__server_error__",
                category="error",
                output_text="",
                status_code=0,
                error=f"Server scenario failed: {e}",
            )
        ]
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def _run_prompts_over_http(
    model: str,
    prompts_data: list[dict],
    max_tokens: int | None,
    port: int,
) -> list[PromptResult]:
    """Send prompts to a running olmlx server over HTTP."""
    url = f"http://127.0.0.1:{port}/api/chat"
    results = []
    for prompt in prompts_data:
        tok_limit = max_tokens or prompt.get("max_tokens", 256)
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
                    PromptResult(
                        prompt_name=prompt["name"],
                        category=prompt["category"],
                        output_text=data.get("message", {}).get("content", ""),
                        status_code=resp.status,
                        eval_count=data.get("eval_count", 0),
                        eval_duration_ns=data.get("eval_duration", 0),
                        prompt_eval_count=data.get("prompt_eval_count", 0),
                        prompt_eval_duration_ns=data.get("prompt_eval_duration", 0),
                        total_duration_ns=data.get("total_duration", 0),
                    )
                )
        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode(errors="replace")[:500]
            except Exception:
                error_body = str(e)
            results.append(
                PromptResult(
                    prompt_name=prompt["name"],
                    category=prompt["category"],
                    output_text="",
                    status_code=e.code,
                    error=error_body,
                )
            )
        except Exception as e:
            results.append(
                PromptResult(
                    prompt_name=prompt["name"],
                    category=prompt["category"],
                    output_text="",
                    status_code=0,
                    error=str(e),
                )
            )
    return results
