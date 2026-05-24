"""CLI orchestrator for the May-2026 extended benchmark.

Usage::

    uv run scripts/run_extended_bench.py \\
        --models-config ~/.olmlx/models.json \\
        --output extended-2026-05 \\
        --only mlx-community/Qwen3-8B-4bit \\
        --budget-hours 6

The script:
1. Reads ``models.json`` to discover which models to bench.
2. Optionally launches ``olmlx serve`` as a subprocess (if ``--spawn-server``).
3. Dispatches each model through :func:`olmlx.bench.extended_runner.run_model`
   (the HTTP driver added in Task 7).
4. Writes per-model JSON under ``<output>/raw/`` (handled by the runner).
5. Emits a summary table to stdout when all models are done.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Public helpers — exposed so the test module can import them directly
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.  ``argv`` defaults to ``sys.argv[1:]``."""
    default_output = Path(f"extended-{time.strftime('%Y-%m')}")
    parser = argparse.ArgumentParser(
        description="Extended benchmark orchestrator for olmlx.",
    )
    parser.add_argument(
        "--models-config",
        required=True,
        metavar="PATH",
        help="Path to models.json (Ollama alias → {hf_path, …} or plain string).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        metavar="DIR",
        help="Output directory for per-model JSON results (default: extended-YYYY-MM).",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="HF_PATH",
        help="Restrict to this HF path (repeatable).  Empty = all models.",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=48.0,
        metavar="H",
        help="Total wall-clock budget in hours (default: 48).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        metavar="URL",
        help="Base URL of the olmlx server (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--spawn-server",
        action="store_true",
        help="Launch `olmlx serve` as a subprocess and shut it down at exit.",
    )
    parser.add_argument(
        "--enable-code-exec",
        action="store_true",
        help="Enable code-execution grading (sets OLMLX_BENCH_CODE_EXEC=1).",
    )
    return parser.parse_args(argv)


def select_models(
    config_path: Path,
    only: list[str],
) -> list[tuple[str, str]]:
    """Return ``[(hf_path, ollama_alias), …]`` from *config_path*.

    Values in models.json can be either a plain string (the hf_path) or a
    dict containing an ``"hf_path"`` key.  Entries without a resolvable
    hf_path are silently skipped.

    If *only* is non-empty, only entries whose hf_path appears in *only*
    are returned.
    """
    raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    result: list[tuple[str, str]] = []
    for alias, value in raw.items():
        if isinstance(value, str):
            hf_path = value
        elif isinstance(value, dict):
            hf_path = value.get("hf_path", "")
        else:
            continue
        if not hf_path:
            continue
        if only and hf_path not in only:
            continue
        result.append((hf_path, alias))
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _spawn_server(base_url: str) -> subprocess.Popen[bytes]:
    """Launch ``olmlx serve`` and wait for it to be reachable."""
    import urllib.request
    import urllib.error

    log = logging.getLogger(__name__)
    proc = subprocess.Popen(
        [sys.executable, "-m", "olmlx", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.info("Spawned olmlx serve (PID %d), waiting for readiness…", proc.returncode)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(base_url + "/", timeout=2)
            log.info("Server is up.")
            return proc
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    log.warning("Server did not become ready within 60 s; proceeding anyway.")
    return proc


def _kill_server(proc: subprocess.Popen[bytes]) -> None:
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        proc.kill()


def _print_summary(results: list[dict[str, Any]]) -> None:
    header = f"{'Model':<50} {'Tier':<12} {'Assignment':<20} {'Composite':>9}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<50} {r['tier']:<12} {r['assignment']:<20}"
            f" {r['composite']:>9.1%}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def _run_all(
    models: list[tuple[str, str]],
    base_url: str,
    output_dir: Path,
    budget_seconds: float,
) -> list[dict[str, Any]]:
    from olmlx.bench.extended_runner import run_model
    from olmlx.bench.tier_table import Tier, tier_for

    log = logging.getLogger(__name__)
    t0 = time.monotonic()
    summary: list[dict[str, Any]] = []

    # Import here so safe_model_name lives next to its only caller.
    from olmlx.bench.extended_runner import safe_model_name

    for hf_path, alias in models:
        elapsed = time.monotonic() - t0
        remaining = budget_seconds - elapsed
        if remaining <= 0:
            log.warning("Budget exhausted before reaching %s — skipping.", hf_path)
            continue

        # Skip if a result already exists on disk for this model.
        # The runner writes both partials and final JSONs to the same path,
        # so re-running after a kill/crash naturally resumes at the next model.
        existing = output_dir / "raw" / f"{safe_model_name(alias)}.json"
        if existing.exists():
            log.info("Skipping %s — result already exists at %s", hf_path, existing)
            continue

        tier_enum = tier_for(hf_path)
        if tier_enum is None:
            log.warning("No tier found for %s; defaulting to core_only.", hf_path)
            tier_str = Tier.CORE_ONLY.value
        else:
            tier_str = tier_enum.value

        log.info(
            "Starting %s  (tier=%s, remaining=%.0f s)", hf_path, tier_str, remaining
        )
        try:
            result = await run_model(
                model=alias,
                tier=tier_str,
                base_url=base_url,
                output_dir=output_dir,
                remaining_seconds=remaining,
            )
            summary.append(result.to_dict())
            log.info("Finished %s  composite=%.1f%%", hf_path, result.composite * 100)
        except Exception as exc:  # noqa: BLE001
            log.error("run_model failed for %s: %r", hf_path, exc)
            summary.append(
                {
                    "model": alias,
                    "tier": tier_str,
                    "assignment": "error",
                    "composite": 0.0,
                }
            )

    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    if args.enable_code_exec:
        os.environ["OLMLX_BENCH_CODE_EXEC"] = "1"

    config_path = Path(args.models_config)
    models = select_models(config_path, only=args.only)
    if not models:
        log.error("No models matched.  Check --models-config and --only filters.")
        sys.exit(1)

    log.info(
        "Running %d model(s), budget=%.1f h, output=%s",
        len(models),
        args.budget_hours,
        args.output,
    )

    server_proc: subprocess.Popen[bytes] | None = None
    if args.spawn_server:
        server_proc = _spawn_server(args.base_url)

    try:
        summary = asyncio.run(
            _run_all(
                models=models,
                base_url=args.base_url,
                output_dir=args.output,
                budget_seconds=args.budget_hours * 3600,
            )
        )
    finally:
        if server_proc is not None:
            _kill_server(server_proc)

    _print_summary(summary)


if __name__ == "__main__":
    main()
