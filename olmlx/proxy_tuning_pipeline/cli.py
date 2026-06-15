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
    expand_units_checkpointed,
    load_done_provenances,
    load_examples,
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
    limit_units: int | None = None,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Extract -> expand (checkpointed) -> curate -> write. Returns a stats dict.

    Generation is crash-safe and resumable: each unit's pairs are appended to
    ``<out_dir>/raw.jsonl`` as they complete, so an interrupted run loses nothing
    durable and re-running skips units already present. ``limit_units`` caps how
    many units are expanded (cheap smoke run); ``concurrency`` is the generator
    thread-pool size. Curation reads ``raw.jsonl`` back, so it works even on a
    partially-generated checkpoint.
    """
    out_dir = Path(out_dir)
    raw_path = out_dir / "raw.jsonl"
    units = extract_repo(repo_root)
    logger.info("extracted %d units", len(units))
    if limit_units is not None:
        units = units[:limit_units]
        logger.info("limiting to %d units for this run", len(units))

    done = load_done_provenances(raw_path)
    if done:
        logger.info("resume: %d units already in %s — skipping", len(done), raw_path)
    expand_units_checkpointed(
        units,
        generator,
        n_per_unit=n_per_unit,
        raw_path=raw_path,
        concurrency=concurrency,
        done=done,
    )

    examples = load_examples(raw_path)
    generated = len(examples)

    # Track the two drop sources separately: a large quality-drop points at the
    # generator producing short/long junk; a large dedup-drop points at the
    # generator being repetitive. They need different fixes (filters vs prompt).
    filtered = quality_filter(examples)
    examples = dedupe_examples(filtered)
    # Deterministic order regardless of concurrent generation order, so the
    # seeded train/valid split is reproducible across runs.
    examples.sort(key=lambda e: (e.provenance, e.user))
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
    ap.add_argument(
        "--limit-units",
        type=int,
        default=None,
        help="cap units expanded (cheap smoke run; omit to process the whole repo)",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="generator thread-pool size (parallel API calls)",
    )
    args = ap.parse_args(argv)

    stats = run_pipeline(
        repo_root=args.repo,
        out_dir=args.out,
        generator=OpenAIGenerator(model=args.model),
        n_per_unit=args.n_per_unit,
        valid_frac=args.valid_frac,
        seed=args.seed,
        limit_units=args.limit_units,
        concurrency=args.concurrency,
    )
    print(f"Done. {stats}")


if __name__ == "__main__":
    main()
