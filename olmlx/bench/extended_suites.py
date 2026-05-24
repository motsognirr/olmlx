"""Extended quality suites.

Loaders fetch canonical datasets (HumanEval+, MBPP+, GSM8K, MATH-500,
MMLU-Pro, GPQA-Diamond, IFEval) once, cache them under
``~/.olmlx/bench-cache/`` (or ``OLMLX_BENCH_CACHE_DIR`` if set), and return
``list[BenchPrompt]`` ready for the existing bench worker.

Subset selection is deterministic — re-runs on a different machine grade the
same prompts. The selected ID list is recorded by the runner alongside
results so a third-party reviewer can verify which prompts contributed to
each score.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def bench_cache_dir() -> Path:
    """Directory for downloaded benchmark datasets."""
    env = os.environ.get("OLMLX_BENCH_CACHE_DIR")
    base = Path(env).expanduser() if env else Path("~/.olmlx/bench-cache").expanduser()
    base.mkdir(parents=True, exist_ok=True)
    return base


def select_subset(
    items: Sequence[T],
    n: int,
    *,
    key: Callable[[T], str] | None = None,
) -> list[T]:
    """Deterministically pick ``n`` items, spread evenly across ``items``.

    With ``key`` set, balances counts across the strata returned by ``key``
    (e.g. MMLU-Pro category labels), splitting ``n`` as evenly as possible
    and applying the same even-spread rule within each stratum.

    Picks indices ``round(i * len(items) / n)`` for i in 0..n-1, so the
    first item is always included; the last item is included when ``n`` is
    large relative to ``len(items)``.

    When ``key`` is provided and all strata can satisfy their allocation,
    returns exactly ``n`` items.  When some strata are smaller than their
    allocation, surplus is redistributed to other strata so the total still
    equals ``n``.
    """
    if n >= len(items):
        return list(items)
    if key is None:
        return [items[round(i * len(items) / n)] for i in range(n)]

    buckets: dict[str, list[T]] = {}
    for item in items:
        buckets.setdefault(key(item), []).append(item)
    labels = sorted(buckets)
    base = n // len(labels)
    remainder = n - base * len(labels)
    # Initial allocation per bucket.
    allotments: dict[str, int] = {
        label: base + (1 if i < remainder else 0) for i, label in enumerate(labels)
    }

    # Iteratively redistribute slack from exhausted buckets to non-exhausted ones.
    while True:
        slack = 0
        can_absorb: list[str] = []
        for label in labels:
            allot = allotments[label]
            bucket_len = len(buckets[label])
            if allot >= bucket_len:
                slack += allot - bucket_len
                allotments[label] = bucket_len  # cap at bucket size
            else:
                can_absorb.append(label)
        if slack == 0 or not can_absorb:
            break
        # Distribute slack evenly across absorbing buckets (priority to earlier labels).
        per = slack // len(can_absorb)
        extra = slack - per * len(can_absorb)
        for i, label in enumerate(can_absorb):
            allotments[label] += per + (1 if i < extra else 0)

    out: list[T] = []
    for label in labels:
        take = allotments[label]
        bucket = buckets[label]
        if take >= len(bucket):
            out.extend(bucket)
        else:
            out.extend(bucket[round(j * len(bucket) / take)] for j in range(take))
    return out
