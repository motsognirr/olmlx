"""Render the extended-bench report from per-model raw JSON.

Pure post-processor: reads ``<output>/raw/*.json``, draws matplotlib charts
into ``<output>/charts/``, and writes ``<output>/README.md``. Idempotent;
can be re-run without re-benching any model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering for CI / scripted use
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger("build_extended_report")

# Stable suite ordering across the report.
SUITES_CORE = ("humaneval-plus", "gsm8k", "gpqa")
SUITES_EXT = ("mbpp-plus", "math500", "mmlu-pro", "ifeval", "ruler-niah")
SUITES_ALL = SUITES_CORE + SUITES_EXT

# Matched quant pairs reported in the comparison chart.
QUANT_PAIRS = [
    (
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        "mlx-community/Qwen3.6-35B-A3B-6bit",
        "Qwen3.6-35B-A3B",
    ),
    (
        "mlx-community/Qwen3.6-27B-4bit",
        "unsloth/Qwen3.6-27B-MLX-8bit",
        "Qwen3.6-27B",
    ),
]


def load_results(output_dir: Path) -> list[dict[str, Any]]:
    raw = output_dir / "raw"
    results: list[dict[str, Any]] = []
    for path in sorted(raw.glob("*.json")):
        if path.parent.name == "ablation":
            continue
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def _short_name(hf_path: str) -> str:
    return hf_path.rsplit("/", 1)[-1]


def render_frontier_chart(results: list[dict[str, Any]], out_path: Path) -> None:
    """Scatter: decode tok/s vs composite quality, colored by tier."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for tier, color in (("extended", "tab:blue"), ("core_only", "tab:orange")):
        xs = [r.get("warmup_tok_per_s", 0) for r in results if r["tier"] == tier]
        ys = [r.get("composite", 0) for r in results if r["tier"] == tier]
        labels = [_short_name(r["model"]) for r in results if r["tier"] == tier]
        ax.scatter(xs, ys, c=color, label=tier, s=80, alpha=0.75)
        for x, y, lbl in zip(xs, ys, labels, strict=True):
            ax.annotate(
                lbl, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points"
            )
    ax.set_xlabel("Warmup decode tok/s (proxy for production throughput)")
    ax.set_ylabel("Composite quality (unweighted mean of suite pass rates)")
    ax.set_title("Speed-quality frontier")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_suite_heatmap(results: list[dict[str, Any]], out_path: Path) -> None:
    """Heatmap of per-suite pass rates: 23 rows × suite columns."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = sorted(results, key=lambda r: (r["tier"], -r.get("composite", 0)))
    suites = list(SUITES_ALL)
    grid = []
    labels = []
    for r in results:
        labels.append(_short_name(r["model"]))
        per_suite = r.get("per_suite_pass_rate", {})
        grid.append([per_suite.get(s, float("nan")) for s in suites])
    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.35)))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(suites)))
    ax.set_xticklabels(suites, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v == v:  # not NaN
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
    fig.colorbar(im, ax=ax, label="pass rate")
    ax.set_title("Per-suite pass rate by model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_quant_pairs_chart(results: list[dict[str, Any]], out_path: Path) -> None:
    """Grouped bars: per-suite pass rate for matched 4-bit vs higher-bit pairs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_model = {r["model"].removesuffix(":latest"): r for r in results}
    pairs_found = [
        (a, b, n) for a, b, n in QUANT_PAIRS if a in by_model and b in by_model
    ]
    if not pairs_found:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(
            0.5,
            0.5,
            "No matched quant pairs in this run",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return
    # One bar group per (pair, suite); two bars per group for the two quants.
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_w = 0.35
    x_base = 0
    xticks = []
    xtick_labels = []
    for a, b, name in pairs_found:
        a_rates = by_model[a].get("per_suite_pass_rate", {})
        b_rates = by_model[b].get("per_suite_pass_rate", {})
        common = [s for s in SUITES_ALL if s in a_rates and s in b_rates]
        for j, suite in enumerate(common):
            ax.bar(
                x_base + j - bar_w / 2, a_rates[suite], width=bar_w, color="tab:blue"
            )
            ax.bar(
                x_base + j + bar_w / 2, b_rates[suite], width=bar_w, color="tab:orange"
            )
            xticks.append(x_base + j)
            xtick_labels.append(f"{name}\n{suite}")
        x_base += len(common) + 1
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("pass rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Matched-pair quant comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_headline_table(results: list[dict[str, Any]]) -> str:
    """Render the 23-row headline table."""
    results = sorted(results, key=lambda r: (r["tier"], -r.get("composite", 0)))
    header = "| Model | Tier | tok/s | Composite | " + " | ".join(SUITES_ALL) + " |"
    sep = "|" + "|".join(["---"] * (len(SUITES_ALL) + 4)) + "|"
    lines = [header, sep]
    for r in results:
        per_suite = r.get("per_suite_pass_rate", {})
        cells = []
        for s in SUITES_ALL:
            if s in per_suite:
                cells.append(f"{per_suite[s]:.2f}")
            elif s in SUITES_CORE:
                cells.append("0.00")
            else:
                cells.append("—")
        lines.append(
            f"| {_short_name(r['model'])} | {r['tier']} | "
            f"{r.get('warmup_tok_per_s', 0):.1f} | "
            f"{r.get('composite', 0):.2f} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def render_findings(results: list[dict[str, Any]]) -> str:
    """Auto-generate findings bullets from the result set."""
    lines: list[str] = []
    extended = [r for r in results if r["tier"] == "extended"]
    if extended:
        top = max(extended, key=lambda r: r.get("composite", 0))
        lines.append(
            f"- **Top extended-tier composite:** "
            f"`{_short_name(top['model'])}` at {top.get('composite', 0):.2f}."
        )
        fastest = max(extended, key=lambda r: r.get("warmup_tok_per_s", 0))
        lines.append(
            f"- **Fastest extended row:** `{_short_name(fastest['model'])}` "
            f"at {fastest.get('warmup_tok_per_s', 0):.1f} tok/s "
            f"(composite {fastest.get('composite', 0):.2f})."
        )
    saturated = []
    for s in SUITES_ALL:
        rates = [
            r["per_suite_pass_rate"][s]
            for r in results
            if s in r.get("per_suite_pass_rate", {})
        ]
        if rates and min(rates) >= 0.95:
            saturated.append(s)
    if saturated:
        lines.append(
            "- **Saturated suites this run:** "
            + ", ".join(f"`{s}`" for s in saturated)
            + ". Retire or replace before the next report."
        )
    if not lines:
        lines.append("- No models produced grade-able results in this run.")
    return "\n".join(lines)


def render_future_research(results: list[dict[str, Any]]) -> str:
    """Heuristic future-research bullets derived from the run."""
    bullets: list[str] = []
    has_ext = any(r["tier"] == "extended" for r in results)
    if has_ext:
        bullets.append(
            "- Compare spectral-calibrated KV-quant vs TurboQuant on a target "
            "that has on-disk calibration data, using the Core suite as the "
            "common ground."
        )
        bullets.append(
            "- Run a same-version draft for the A3B target if/when one is "
            "released — the May 2026 report's cross-version draft hit 55% "
            "acceptance but still ran net-slower."
        )
    if any(
        r.get("per_suite_pass_rate", {}).get("ruler-niah", 0) < 0.5 for r in results
    ):
        bullets.append(
            "- RULER pass rate drops below 50% on at least one row; sweep KV "
            "quant bit width (off / TQ-4 / TQ-2 / spectral-4) at 4k/8k/16k "
            "context to map the degradation curve."
        )
    if any(
        r.get("per_suite_pass_rate", {}).get("humaneval-plus", 0) >= 0.95
        for r in results
    ):
        bullets.append(
            "- HumanEval+ is saturating at the top — swap to LiveCodeBench "
            "for the coding deep-dive in the next report."
        )
    if not bullets:
        bullets.append(
            "- No clear research direction surfaced by this run; consider a "
            "longer follow-up at full-split sizes."
        )
    return "\n".join(bullets)


_METHODOLOGY_BLOCK = """
- **Serving:** each model loaded via `olmlx serve` and hit through `/api/chat`
  at `temperature=0`, `seed=42`, `top_p=1.0`. Per-model `options` from
  `models.json` are overridden so models are compared at the same sampling
  distribution.
- **Token caps:** 4096 for HumanEval+/MBPP+/GSM8K/MATH-500 (avoids
  `<think>`-block truncation on verbose reasoners); 1024 for MMLU-Pro / GPQA /
  IFEval / RULER.
- **Composite score:** unweighted mean of per-suite pass rates. Each suite
  contributes equally regardless of prompt count.
- **Suite subset selection:** deterministic spread by source-ID order (or
  stratified by category/level/instruction-family); the selected ID list is
  recorded in each `raw/<model>.json` for reproducibility.
- **Runtime triage:** when a slow model can't finish Core within its
  remaining budget, GPQA is dropped first; HumanEval+ only is the last-resort
  slice. The assignment is recorded per row.
""".strip()


def build_report(output_dir: Path) -> None:
    """Render charts + README into output_dir."""
    results = load_results(output_dir)
    charts_dir = output_dir / "charts"
    render_frontier_chart(results, charts_dir / "frontier.png")
    render_suite_heatmap(results, charts_dir / "suite_heatmap.png")
    render_quant_pairs_chart(results, charts_dir / "quant_pairs.png")
    md = f"""# Extended benchmark — 23 configured models

Speed + quality comparison across the models in `~/.olmlx/models.json`.
See `docs/superpowers/specs/2026-05-24-extended-bench-design.md` for the
design that drove this run.

## Methodology

{_METHODOLOGY_BLOCK}

## Headline

![Speed-quality frontier](charts/frontier.png)

{render_headline_table(results)}

## Per-suite pass rates

![Suite heatmap](charts/suite_heatmap.png)

## Matched quant pairs

![Quant pairs](charts/quant_pairs.png)

## Findings

{render_findings(results)}

## Future research directions

{render_future_research(results)}

## Caveats

- HumanEval+ / MBPP+ run with sandboxed `code_exec`; acceptable for the
  single-user local olmlx tool per `CLAUDE.md`.
- KV-quant noise is length-dependent and not load-bearing at this set size;
  would be at full-split sizes.
- Flash-MoE rows reflect SSD I/O variance — re-run on a quiet machine for
  best comparability.
- IFEval covers the verifiable-constraint subset only; rubric-graded
  constraints are excluded.

## Reproducing

```bash
python scripts/run_extended_bench.py \\
    --models-config ~/.olmlx/models.json \\
    --output {output_dir} \\
    --enable-code-exec --start-server

python scripts/build_extended_report.py {output_dir}
```
"""
    (output_dir / "README.md").write_text(md, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Render extended bench report")
    parser.add_argument("output", type=Path, help="Run directory containing raw/")
    args = parser.parse_args(argv)
    build_report(args.output)
    logger.info("wrote %s", args.output / "README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
