"""Dataclasses + JSONL loader for the Stage-3 proxy-tuning eval harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# The three rubric categories the held-out prompt set spans.
VALID_CATEGORIES = frozenset(
    {"explain_invariant", "implement_convention", "convention_qa"}
)


@dataclass(frozen=True)
class EvalPrompt:
    id: str
    category: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class EvalScore:
    prompt_id: str
    category: str
    alpha: float
    convention: int  # 1-5
    coherence: int  # 1-5
    rationale: str
    output: str


@dataclass(frozen=True)
class AlphaSummary:
    alpha: float
    n: int
    mean_convention: float
    mean_coherence: float


@dataclass(frozen=True)
class ShipDecision:
    ship: bool
    best_alpha: float
    base_convention: float
    best_convention: float
    base_coherence: float
    best_coherence: float
    conv_margin: float
    coh_drop: float
    reason: str
    per_alpha: list[AlphaSummary] = field(default_factory=list)


def load_eval_prompts(path: str) -> list[EvalPrompt]:
    """Load + validate the held-out prompt JSONL.

    Raises ValueError on an unknown category or a duplicate id — both would
    silently skew the aggregate, so fail loud.
    """
    prompts: list[EvalPrompt] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            cat = row["category"]
            if cat not in VALID_CATEGORIES:
                raise ValueError(
                    f"prompt {row.get('id')!r} has unknown category {cat!r}; "
                    f"expected one of {sorted(VALID_CATEGORIES)}"
                )
            pid = row["id"]
            if pid in seen:
                raise ValueError(f"duplicate prompt id {pid!r}")
            seen.add(pid)
            prompts.append(EvalPrompt(id=pid, category=cat, messages=row["messages"]))
    return prompts
