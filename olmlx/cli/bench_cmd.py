"""Bench subcommands: run, compare, list, leaderboard."""

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from olmlx.config import (
    warn_legacy_flash_env as _warn_legacy_flash_env,
)

from olmlx.cli.config_cmd import _configure_logging
from olmlx.cli.serve import (
    _surface_legacy_dflash_env,
    _surface_legacy_kv_cache_quant_env,
    _surface_legacy_speculative_env,
)

logger = logging.getLogger(__name__)


def cmd_bench_run(args):
    """Run benchmark scenarios."""
    _configure_logging()
    _warn_legacy_flash_env()
    _surface_legacy_speculative_env()
    _surface_legacy_dflash_env()
    _surface_legacy_kv_cache_quant_env()

    from olmlx.bench.runner import run_bench
    from olmlx.bench.results import DEFAULT_BENCH_DIR

    scenario_names = (
        [s.strip() for s in args.scenarios.split(",") if s.strip()]
        if args.scenarios
        else None
    )
    bench_dir = Path(args.bench_dir) if args.bench_dir else DEFAULT_BENCH_DIR

    run_bench(
        model=args.model,
        scenario_names=scenario_names,
        max_tokens=args.max_tokens,
        bench_dir=bench_dir,
        prompt_set=args.prompt_set,
        enable_code_exec=args.enable_code_exec,
    )


def cmd_bench_compare(args):
    """Compare two benchmark runs."""

    from olmlx.bench.results import DEFAULT_BENCH_DIR, compare_runs, load_run

    def _resolve_run(ref: str) -> Path:
        p = Path(ref)
        if p.exists():
            return p
        # Try as timestamp under default bench dir
        candidate = DEFAULT_BENCH_DIR / ref
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Run not found: {ref}")

    run1 = load_run(_resolve_run(args.run1))
    run2 = load_run(_resolve_run(args.run2))
    print(compare_runs(run1, run2))


def cmd_bench_list(args):
    """List past benchmark runs."""

    from olmlx.bench.results import DEFAULT_BENCH_DIR, list_runs

    bench_dir = Path(args.bench_dir) if args.bench_dir else DEFAULT_BENCH_DIR
    runs = list_runs(bench_dir)
    if not runs:
        print("No benchmark runs found.")
        return

    print(
        f"{'Timestamp':<22} {'Model':<45} {'Git':<10} {'Scenarios':>9} {'Skipped':>7}"
    )
    print("-" * 95)
    for r in runs:
        print(
            f"{r['timestamp']:<22} {r['model']:<45} {r['git_sha'] or '—':<10} "
            f"{r['scenarios']:>9} {r['skipped']:>7}"
        )


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _non_empty_str(value: str) -> str:
    """argparse ``type`` validator that mirrors ``Field(min_length=1)``
    on the corresponding Settings field. Without it, ``--flag ""``
    propagates an empty string into Settings and surfaces as an
    unhandled ``ValidationError`` traceback at startup. Surrounding
    whitespace is stripped so that ``--flag " hf/path "`` doesn't
    later fail with a confusing path-not-found error."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("value must be a non-empty string")
    return stripped


def cmd_bench_leaderboard(args):
    """Show the model leaderboard derived from saved bench runs."""

    from olmlx.bench.results import (
        DEFAULT_BENCH_DIR,
        build_leaderboard,
        format_leaderboard,
    )

    bench_dir = Path(args.bench_dir) if args.bench_dir else DEFAULT_BENCH_DIR
    entries = build_leaderboard(bench_dir, latest_per_model=not args.all_runs)
    if not entries:
        print("No bench runs with valid measurements found.")
        return
    print(format_leaderboard(entries, limit=args.limit))
