"""Opt-in end-to-end smoke test for the extended benchmark pipeline.

Skipped unless ``OLMLX_RUN_E2E_BENCH=1`` is set in the environment **and**
``olmlx serve`` is reachable at ``localhost:11434``.

Run::

    OLMLX_RUN_E2E_BENCH=1 uv run pytest tests/integration/test_extended_bench_smoke.py -v

The test drives a single tiny draft model
(``mlx-community/Qwen3-0.6B-4bit``) through the extended-bench runner
with ``tier="core_only"`` and a tight 600-second budget.  That budget is
intentionally small enough to trigger the triage rules so only a sliver of
the core suite is actually executed.  After the run it calls ``build_report``
to render markdown + charts and asserts the expected output files exist.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

E2E_ENV = "OLMLX_RUN_E2E_BENCH"
BASE_URL = "http://localhost:11434"
SMOKE_MODEL = "mlx-community/Qwen3-0.6B-4bit"
SMOKE_ALIAS = "mlx-community/Qwen3-0.6B-4bit:latest"
SMOKE_TIER = "core_only"
SMOKE_BUDGET_SECONDS = 600.0


def _server_reachable() -> bool:
    try:
        urllib.request.urlopen(BASE_URL + "/", timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


if not os.environ.get(E2E_ENV):
    pytest.skip(
        f"Set {E2E_ENV}=1 to run end-to-end bench smoke tests.",
        allow_module_level=True,
    )

if not _server_reachable():
    pytest.skip(
        f"olmlx serve not reachable at {BASE_URL}; start it before running this test.",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Helpers — import scripts that live outside the package
# ---------------------------------------------------------------------------

_SCRIPTS = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import build_extended_report  # type: ignore[import-not-found]  # noqa: E402

from olmlx.bench.extended_runner import run_model  # noqa: E402

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_bench_smoke(tmp_path: Path) -> None:
    """Drive SMOKE_MODEL through run_model, render report, check outputs."""
    output_dir = tmp_path / "extended-smoke"

    result = await run_model(
        model=SMOKE_ALIAS,
        tier=SMOKE_TIER,
        base_url=BASE_URL,
        output_dir=output_dir,
        remaining_seconds=SMOKE_BUDGET_SECONDS,
    )

    # --- per-model JSON written by run_model ---
    from olmlx.bench.extended_runner import safe_model_name

    json_path = output_dir / "raw" / f"{safe_model_name(SMOKE_ALIAS)}.json"
    assert json_path.exists(), f"Expected per-model JSON at {json_path}"
    assert json_path.stat().st_size > 0

    # --- result sanity checks ---
    assert result.model == SMOKE_ALIAS
    assert result.tier == SMOKE_TIER
    assert result.warmup_tok_per_s >= 0.0

    # --- render report (markdown + charts) ---
    build_extended_report.build_report(output_dir)

    readme = output_dir / "README.md"
    assert readme.exists(), "build_report must produce README.md"
    assert readme.stat().st_size > 0

    charts_dir = output_dir / "charts"
    chart_files = list(charts_dir.glob("*.png"))
    assert len(chart_files) >= 1, (
        f"build_report must produce at least one .png chart; found {chart_files}"
    )
