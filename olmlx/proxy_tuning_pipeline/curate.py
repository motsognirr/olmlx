"""Curation: quality filtering, dedup, and train/valid split."""

from __future__ import annotations

import math
import random

from olmlx.proxy_tuning_pipeline.schema import ChatExample

_MIN_USER_CHARS = 8
_MIN_ASSISTANT_CHARS = 12
_MAX_ASSISTANT_CHARS = 20_000


def cap_kind_fraction(
    examples: list[ChatExample],
    kind: str,
    max_fraction: float,
    seed: int,
) -> list[ChatExample]:
    """Deterministically downsample one over-represented ``kind``.

    Keeps every example of the other kinds; randomly samples ``kind`` down so it
    is at most ``max_fraction`` of the returned set. No-op when ``kind`` is
    already under the cap or ``max_fraction`` is not in (0, 1).
    """
    if not (0.0 < max_fraction < 1.0):
        return list(examples)
    others = [e for e in examples if e.kind != kind]
    targeted = [e for e in examples if e.kind == kind]
    # targeted / (targeted + others) <= max_fraction  ->  targeted <= f/(1-f)*others
    cap = math.floor(max_fraction / (1.0 - max_fraction) * len(others))
    if len(targeted) <= cap:
        return list(examples)
    kept = random.Random(seed).sample(targeted, cap)
    return others + kept


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

    Guarantees at least one validation example whenever the input is non-empty
    (regardless of ``valid_frac``, including ``0.0``) so ``mlx_lm.lora`` always
    has a ``valid.jsonl`` to evaluate against.
    """
    if not (0.0 <= valid_frac < 1.0):
        raise ValueError(f"valid_frac must be in [0, 1), got {valid_frac}")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * valid_frac)) if shuffled else 0
    valid = shuffled[:n_valid]
    train = shuffled[n_valid:]
    return train, valid
