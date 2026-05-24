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
    by_model = {r["model"]: r for r in results}
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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Render extended bench report")
    parser.add_argument("output", type=Path, help="Run directory containing raw/")
    args = parser.parse_args(argv)
    results = load_results(args.output)
    charts_dir = args.output / "charts"
    render_frontier_chart(results, charts_dir / "frontier.png")
    render_suite_heatmap(results, charts_dir / "suite_heatmap.png")
    render_quant_pairs_chart(results, charts_dir / "quant_pairs.png")
    logger.info("charts written to %s", charts_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
