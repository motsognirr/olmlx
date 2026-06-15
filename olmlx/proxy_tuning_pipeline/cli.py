"""CLI + orchestration for the proxy-tuning data pipeline.

Usage:
    OPENAI_API_KEY=... uv run python -m olmlx.proxy_tuning_pipeline.cli \
        --repo . --out data/proxy_tuning --n-per-unit 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from olmlx.proxy_tuning_pipeline.curate import (
    dedupe_examples,
    quality_filter,
    split_train_valid,
)
from olmlx.proxy_tuning_pipeline.expand import (
    DEFAULT_MODEL,
    Generator,
    OpenAIGenerator,
    expand_units,
)
from olmlx.proxy_tuning_pipeline.extract import extract_repo
from olmlx.proxy_tuning_pipeline.schema import write_jsonl

logger = logging.getLogger(__name__)


def run_pipeline(
    repo_root: str | Path,
    out_dir: str | Path,
    generator: Generator,
    n_per_unit: int,
    valid_frac: float,
    seed: int,
) -> dict[str, Any]:
    """Extract -> expand -> curate -> write. Returns a stats dict."""
    out_dir = Path(out_dir)
    units = extract_repo(repo_root)
    logger.info("extracted %d units", len(units))

    examples = expand_units(units, generator, n_per_unit=n_per_unit)
    generated = len(examples)

    # Track the two drop sources separately: a large quality-drop points at the
    # generator producing short/long junk; a large dedup-drop points at the
    # generator being repetitive. They need different fixes (filters vs prompt).
    filtered = quality_filter(examples)
    examples = dedupe_examples(filtered)
    kept = len(examples)

    train, valid = split_train_valid(examples, valid_frac=valid_frac, seed=seed)
    write_jsonl(out_dir / "train.jsonl", (e.to_chat_row() for e in train))
    write_jsonl(out_dir / "valid.jsonl", (e.to_chat_row() for e in valid))

    stats = {
        "units": len(units),
        "generated": generated,
        "quality_dropped": generated - len(filtered),
        "dedup_dropped": len(filtered) - kept,
        "kept": kept,
        "dropped": generated - kept,
        "train": len(train),
        "valid": len(valid),
    }
    logger.info("pipeline stats: %s", stats)
    if units and not kept:
        # Generator produced output but every pair was filtered/deduped away —
        # an empty dataset only otherwise noticed at training time.
        logger.error(
            "run_pipeline: 0 examples survived curation (generated=%d)", generated
        )
    return stats


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(
        description="Build proxy-tuning SFT data from the olmlx repo."
    )
    ap.add_argument("--repo", default=".", help="repo root to mine")
    ap.add_argument(
        "--out", default="data/proxy_tuning", help="output dir for train/valid jsonl"
    )
    ap.add_argument(
        "--n-per-unit", type=int, default=4, help="pairs requested per source unit"
    )
    ap.add_argument(
        "--valid-frac", type=float, default=0.08, help="validation fraction"
    )
    ap.add_argument("--seed", type=int, default=0, help="split shuffle seed")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI generator model id")
    args = ap.parse_args(argv)

    stats = run_pipeline(
        repo_root=args.repo,
        out_dir=args.out,
        generator=OpenAIGenerator(model=args.model),
        n_per_unit=args.n_per_unit,
        valid_frac=args.valid_frac,
        seed=args.seed,
    )
    print(f"Done. {stats}")


if __name__ == "__main__":
    main()
