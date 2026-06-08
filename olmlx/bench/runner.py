"""Orchestrates benchmark runs — spawns worker subprocesses per scenario."""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from olmlx.bench.prompts import PROMPTS, BenchPrompt
from olmlx.bench.quality import QualityResult, grade
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


def _capture_bench_env() -> dict[str, str]:
    """Capture the bench-relevant env knobs set when ``run_bench`` was called.

    Limited to the experiment-defining toggles so a saved run's
    ``bench_env`` is a clean record of the A/B variable(s). Operational
    knobs like the worker kill timeout are intentionally excluded — they
    don't change how to interpret the result and are logged separately at
    run start.

    Only records values the worker will actually honour — a typo'd
    ``OLMLX_BENCH_THINK=tru`` (which the worker warns about and falls
    back to engine default) does **not** land in the saved JSON, so an
    operator reading the run record can trust that what's there reflects
    the run that happened.

    Maintenance: when adding a new experiment-defining env var, extend
    this function explicitly. The enumeration is deliberately not
    pattern-based (no ``OLMLX_BENCH_*`` glob) so operational variables
    aren't accidentally promoted to the experiment record.
    """
    from olmlx.bench.worker import is_recognized_think_value

    captured: dict[str, str] = {}
    think_raw = os.environ.get("OLMLX_BENCH_THINK", "")
    if think_raw and is_recognized_think_value(think_raw):
        captured["OLMLX_BENCH_THINK"] = think_raw
    return captured


_DEFAULT_WORKER_TIMEOUT = 1800.0


def _worker_timeout() -> float:
    """Per-scenario worker kill timeout, in seconds.

    Defaults to 1800s (30 min). The 7-prompt throughput set finishes in well
    under that, but the 50-prompt quality set on a mid/large model (especially
    with thinking on and a high ``--max-tokens``) can run many minutes per
    scenario — the previous 600s default made those scenarios reliably time out
    and lose their results. Still overridable via ``OLMLX_BENCH_WORKER_TIMEOUT``
    for even longer graded runs rather than hard-failing.

    Rejects ``inf`` / ``nan`` (these would silently disable the kill switch in
    ``communicate(timeout=...)``) and warns on unparseable values so an ignored
    override is diagnosable instead of vanishing into the default.
    """
    raw = os.environ.get("OLMLX_BENCH_WORKER_TIMEOUT", "")
    if not raw:
        return _DEFAULT_WORKER_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Ignoring OLMLX_BENCH_WORKER_TIMEOUT=%r: not a number; using %.0fs",
            raw,
            _DEFAULT_WORKER_TIMEOUT,
        )
        return _DEFAULT_WORKER_TIMEOUT
    # math.isfinite covers inf, -inf, and nan in one call. Negative or zero
    # values fall through to the same warning rather than silently flipping
    # to the default with no diagnostic.
    if not math.isfinite(value) or value <= 0:
        logger.warning(
            "Ignoring OLMLX_BENCH_WORKER_TIMEOUT=%r: must be a positive finite "
            "number; using %.0fs",
            raw,
            _DEFAULT_WORKER_TIMEOUT,
        )
        return _DEFAULT_WORKER_TIMEOUT
    return value


def build_prompts(prompt_set: str) -> list[BenchPrompt]:
    """Resolve a prompt-set name to its list of prompts.

    - ``throughput``: the 7 ungraded throughput probes (tok/s only).
    - ``quality``: GSM8K + MMLU + HumanEval mini-sets (all graded).
    - ``all``: throughput probes followed by the graded sets.

    The throughput path returns before importing ``task_prompts`` so a
    plain ``olmlx bench run`` doesn't construct the ~50 graded prompts.
    """
    if prompt_set == "throughput":
        return list(PROMPTS)

    from olmlx.bench.task_prompts import PROMPT_SETS

    graded = [p for sets in PROMPT_SETS.values() for p in sets]
    # Same collision rationale as the ``all`` branch below, but within the
    # graded set itself — if two entries across the GSM8K / MMLU / HumanEval
    # sub-sets ever share a name, ``apply_graders``'s ``by_name`` dict
    # silently keeps the last and mis-grades the first matching result.
    graded_dupes = sorted(
        n for n, c in Counter(p.name for p in graded).items() if c > 1
    )
    if graded_dupes:
        raise ValueError(
            f"Duplicate prompt names within quality set: {graded_dupes!r}. "
            f"Rename to keep prompt names unique."
        )
    if prompt_set == "quality":
        return graded
    if prompt_set == "all":
        combined = list(PROMPTS) + graded
        # apply_graders joins PromptResult → prompt by name. A duplicate
        # name across throughput and graded sets would silently grade a
        # throughput result against the colliding graded prompt's grader.
        # Catch the collision here, at construction, rather than letting it
        # slip through into a saved run.
        dupes = sorted(n for n, c in Counter(p.name for p in combined).items() if c > 1)
        if dupes:
            raise ValueError(
                f"Duplicate prompt names across throughput + quality sets: "
                f"{dupes!r}. Rename to keep prompt names unique."
            )
        return combined
    raise ValueError(
        f"Unknown prompt set {prompt_set!r}. Available: throughput, quality, all"
    )


def apply_graders(
    results: list[PromptResult],
    prompts: list[dict],
    enable_code_exec: bool,
) -> dict[str, int]:
    """Grade prompt results in place against their prompt metadata.

    Graders are pure functions run here in the parent (the worker only
    produces ``output_text``). Only successful (HTTP 200) responses to
    prompts that carry a grader are scored; non-200 responses and prompts
    without a grader leave ``r.grader`` and ``r.quality`` untouched (both
    stay ``None``). The pair always satisfies ``r.grader is not None ⇔
    r.quality is not None``.

    A grader that runs may still return ``passed=None`` to signal "no
    verdict reached" — e.g. ``code_exec`` when ``enable_code_exec`` is
    false. That is a graded verdict, distinct from never having run a
    grader at all; ``ScenarioResult.quality_summary`` excludes both from
    its passed/total tally.

    ``code_exec`` runs untrusted model code, so it stays disabled unless
    ``enable_code_exec`` is set, signalled to the grader via a private
    ``_enabled`` flag injected into a *copy* of the expected payload (the
    caller's dict is never mutated).

    ``prompts`` is a list of dicts (not ``BenchPrompt`` objects) because
    the same list is what gets JSON-serialised and shipped to the worker
    subprocess — keeping a single representation avoids parallel state.

    Returns a stats dict: ``{"code_exec_excluded": N}`` where N counts
    prompts that hit the ``code_exec`` grader while
    ``enable_code_exec=False``. The runner uses this for the
    "pass --enable-code-exec" notice rather than re-inferring the count
    from ``r.quality.passed is None``, which would also match grader
    exceptions or other future ``passed=None`` paths.
    """
    # Only graded prompts can match — excluding ungraded throughput
    # prompts from the lookup avoids a last-wins collision in the dict
    # comprehension if a graded and an ungraded prompt ever share a name
    # under ``--prompt-set all``.
    by_name = {p["name"]: p for p in prompts if p.get("grader")}
    stats = {"code_exec_excluded": 0}
    for r in results:
        prompt = by_name.get(r.prompt_name)
        if prompt is None:
            continue
        if r.status_code != 200:
            continue
        grader_name = prompt["grader"]
        # Enforce the code_exec opt-in *here*, not by trusting
        # ``grade_code_exec`` to look at an ``_enabled`` key. Otherwise
        # a future refactor of the grader's internal gate (rename, default
        # change) would silently let untrusted code run with
        # ``enable_code_exec=False``. Synthesise the disabled verdict and
        # skip the grader call entirely.
        if grader_name == "code_exec" and not enable_code_exec:
            stats["code_exec_excluded"] += 1
            r.grader = grader_name
            r.quality = QualityResult(
                grader="code_exec",
                passed=None,
                score=None,
                detail="code_exec disabled (pass --enable-code-exec)",
            )
            continue
        expected = dict(prompt.get("expected") or {})
        if grader_name == "code_exec":
            expected["_enabled"] = True
        # Wrap defensively: ``grade`` is contracted not to raise (it has
        # its own ``except Exception`` returning ``passed=None``), but a
        # future regression there shouldn't abort a long graded run mid-way
        # and lose every result accumulated so far. The fallback preserves
        # the ``r.grader ⇔ r.quality`` invariant.
        try:
            quality = grade(grader_name, r.output_text, expected)
        except Exception as exc:  # belt-and-suspenders, not the live path
            logger.warning(
                "Grader %r raised on prompt %r: %s",
                grader_name,
                r.prompt_name,
                exc,
            )
            quality = QualityResult(
                grader=grader_name,
                passed=None,
                score=None,
                detail=f"apply_graders caught: {exc!r}",
            )
        r.grader = grader_name
        r.quality = quality
    return stats


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
    prompt_set: str = "throughput",
    enable_code_exec: bool = False,
) -> RunResult:
    """Run all scenarios and return aggregated results."""
    scenarios = get_scenarios(scenario_names)
    model_path = _resolve_model_path(model)
    prompts_data = [p.to_dict() for p in build_prompts(prompt_set)]
    # Resolve the worker timeout once so the value logged and the value
    # actually passed to ``subprocess.run`` can never diverge — both come
    # from the same single read of ``OLMLX_BENCH_WORKER_TIMEOUT``.
    worker_timeout = _worker_timeout()
    logger.info("Bench worker timeout: %.0fs", worker_timeout)
    # If think is being toggled while running ``--prompt-set all``, the
    # throughput probes run in think mode too. The summary line later
    # combines tok/s and quality, but the tok/s figure won't be a
    # standard baseline comparable to a non-think throughput run; warn
    # so the operator doesn't misread.
    if prompt_set == "all" and os.environ.get("OLMLX_BENCH_THINK"):
        print(
            "  note: OLMLX_BENCH_THINK is set with --prompt-set all — throughput "
            "probes also run in thinking mode, so the tok/s figure is not a "
            "standard baseline.",
            file=sys.stderr,
        )

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
            prompt_results = _run_worker(
                model, scenario, prompts_data, max_tokens, worker_timeout
            )

        # Grade in the parent (the worker only returns raw output_text).
        grade_stats = apply_graders(prompt_results, prompts_data, enable_code_exec)

        sc_result = ScenarioResult(
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            env_overrides=scenario.env_overrides,
            prompt_results=prompt_results,
        )

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
        passed, graded = sc_result.quality_summary()
        quality_str = ""
        if graded:
            quality_str = f", quality {passed}/{graded} ({passed / graded:.0%})"
        print(
            f"  {status} — {ok}/{len(prompt_results)} prompts, "
            f"avg {avg_tps:.1f} tok/s{quality_str}",
            file=sys.stderr,
        )
        # Surface code_exec exclusion explicitly — otherwise an operator
        # who omits ``--enable-code-exec`` sees e.g. ``quality 20/40``
        # without realising 10 HumanEval prompts were excluded from the
        # denominator. The count comes directly from apply_graders so it
        # can't be confused with grader exceptions or any other future
        # ``passed=None`` path.
        excluded = grade_stats["code_exec_excluded"]
        if excluded:
            print(
                f"  ({excluded} code_exec prompts excluded — pass "
                "--enable-code-exec to grade them)",
                file=sys.stderr,
            )

        scenario_results.append(sc_result)

    run = create_run_result(
        model=model,
        scenarios=scenario_results,
        max_tokens_override=max_tokens,
        prompt_set=prompt_set,
        bench_env=_capture_bench_env(),
    )
    run_dir = save_run(run, bench_dir)
    print(f"\nResults saved to {run_dir}", file=sys.stderr)

    try:
        entries = build_leaderboard(bench_dir)
        if entries:
            print("\nLeaderboard (top 5):", file=sys.stderr)
            print(format_leaderboard(entries, limit=5), file=sys.stderr)
            print(
                "  (run 'olmlx bench leaderboard --all-runs' for full history)",
                file=sys.stderr,
            )
    except Exception:
        logger.warning("Could not build leaderboard", exc_info=True)

    return run


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Kill the worker's whole process group, reaping any server it spawned.

    The worker is launched with ``start_new_session=True``, so its PID is its
    own process-group leader and the uvicorn server it spawns inherits that
    group. Killing the group guarantees the server dies with the worker.

    Without this, a worker killed on timeout (CPython SIGKILLs it on
    ``TimeoutExpired``, so the worker's own ``finally`` cleanup cannot run)
    leaves its server orphaned with the model still resident in RAM. Across a
    multi-scenario / multi-model sweep these orphans accumulate to tens of GB
    and eventually starve later loads (observed: 61 GB, cascading load failures).
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _run_worker(
    model: str,
    scenario: Scenario,
    prompts_data: list[dict],
    max_tokens: int | None,
    worker_timeout: float,
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

        # start_new_session=True: run the worker (and the server it spawns) in a
        # dedicated process group so _terminate_process_group can reap the whole
        # tree — see that function for the orphaned-server leak this prevents.
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            try:
                _, stderr = proc.communicate(timeout=worker_timeout)
            except subprocess.TimeoutExpired:
                return [
                    PromptResult(
                        prompt_name="__worker_error__",
                        category="error",
                        output_text="",
                        status_code=0,
                        error=f"Worker timed out after {worker_timeout:.0f}s",
                    )
                ]

            if proc.returncode != 0:
                logger.error(
                    "Worker failed for %s (exit %d):\n%s",
                    scenario.name,
                    proc.returncode,
                    stderr[-1000:] if stderr else "(no stderr)",
                )
                return [
                    PromptResult(
                        prompt_name="__worker_error__",
                        category="error",
                        output_text="",
                        status_code=0,
                        error=f"Worker exited with code {proc.returncode}: {stderr[-500:] if stderr else ''}",
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

        finally:
            # Backstop on every path (timeout, crash, clean exit): ensure the
            # worker's server child can never outlive this call.
            _terminate_process_group(proc)


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
