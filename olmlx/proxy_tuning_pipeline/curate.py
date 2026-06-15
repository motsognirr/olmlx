"""Curation: quality filtering, dedup, and train/valid split."""

from __future__ import annotations

import random

from olmlx.proxy_tuning_pipeline.schema import ChatExample

_MIN_USER_CHARS = 8
_MIN_ASSISTANT_CHARS = 12
_MAX_ASSISTANT_CHARS = 20_000


def quality_filter(examples: list[ChatExample]) -> list[ChatExample]:
    """Drop pairs that are too short to teach or implausibly long (likely junk)."""
    out: list[ChatExample] = []
    for e in examples:
        if len(e.user.strip()) < _MIN_USER_CHARS:
            continue
        if not (
            _MIN_ASSISTANT_CHARS <= len(e.assistant.strip()) <= _MAX_ASSISTANT_CHARS
        ):
            continue
        out.append(e)
    return out


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def dedupe_examples(examples: list[ChatExample]) -> list[ChatExample]:
    """Drop normalized-duplicate (user, assistant) pairs; order-preserving."""
    seen: set[tuple[str, str]] = set()
    out: list[ChatExample] = []
    for e in examples:
        key = (_norm(e.user), _norm(e.assistant))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def split_train_valid(
    examples: list[ChatExample],
    valid_frac: float,
    seed: int,
) -> tuple[list[ChatExample], list[ChatExample]]:
    """Shuffle deterministically and split off a validation fraction.

    Guarantees at least one validation example when the input is non-empty so
    ``mlx_lm.lora`` always has a ``valid.jsonl`` to evaluate against.
    """
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * valid_frac)) if shuffled else 0
    valid = shuffled[:n_valid]
    train = shuffled[n_valid:]
    return train, valid
