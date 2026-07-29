"""REAP keep/drop (+bank) planning from calibrated saliency."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

PLAN_VERSION = 1


@dataclass
class ReapPlan:
    mode: str
    moe_layer_indices: list[int]
    keep: dict[int, list[int]]
    bank_high: dict[int, list[int]] | None
    high_bits: int | None
    low_bits: int | None
    num_experts_orig: int
    top_k: int
    meta: dict = field(default_factory=dict)


def _validate(saliency: np.ndarray, moe_layer_indices, num_experts: int) -> None:
    if saliency.shape != (len(moe_layer_indices), num_experts):
        raise ValueError(
            f"saliency shape {saliency.shape} != "
            f"({len(moe_layer_indices)}, {num_experts})"
        )


def _top_ascending(scores: np.ndarray, k: int) -> list[int]:
    order = np.lexsort(
        (np.arange(len(scores)), -scores)
    )  # score desc, index asc tiebreak
    return sorted(int(i) for i in order[:k])


def plan_uniform(saliency, moe_layer_indices, keep_count, *, top_k, num_experts):
    _validate(saliency, moe_layer_indices, num_experts)
    if not top_k <= keep_count <= num_experts:
        raise ValueError(
            f"keep_count={keep_count} must satisfy "
            f"top_k={top_k} <= keep <= num_experts={num_experts}"
        )
    keep = {
        layer: _top_ascending(saliency[pos], keep_count)
        for pos, layer in enumerate(moe_layer_indices)
    }
    return ReapPlan(
        "uniform", list(moe_layer_indices), keep, None, None, None, num_experts, top_k
    )


def plan_global(
    saliency,
    moe_layer_indices,
    keep_fraction,
    *,
    top_k,
    num_experts,
    floor_multiple: int = 4,
):
    _validate(saliency, moe_layer_indices, num_experts)
    L = len(moe_layer_indices)
    budget = round(keep_fraction * L * num_experts)
    floor = min(floor_multiple * top_k, num_experts)
    if budget < floor * L:
        raise ValueError(f"budget {budget} below floor {floor}*{L} layers")
    norm = saliency / np.maximum(saliency.sum(axis=1, keepdims=True), 1e-12)
    keep_sets: list[set[int]] = []
    for pos in range(L):
        keep_sets.append(set(_top_ascending(norm[pos], floor)))
    remaining = budget - sum(len(s) for s in keep_sets)
    flat = [
        (-norm[pos, e], pos, e)
        for pos in range(L)
        for e in range(num_experts)
        if e not in keep_sets[pos]
    ]
    flat.sort()
    for _, pos, e in flat[:remaining]:
        keep_sets[pos].add(e)
    keep = {
        layer: sorted(keep_sets[pos]) for pos, layer in enumerate(moe_layer_indices)
    }
    return ReapPlan(
        "global", list(moe_layer_indices), keep, None, None, None, num_experts, top_k
    )


def plan_graded(
    saliency,
    moe_layer_indices,
    keep_fraction,
    *,
    high_fraction,
    high_bits,
    low_bits,
    top_k,
    num_experts,
    floor_multiple: int = 4,
):
    base = plan_global(
        saliency,
        moe_layer_indices,
        keep_fraction,
        top_k=top_k,
        num_experts=num_experts,
        floor_multiple=floor_multiple,
    )
    bank_high: dict[int, list[int]] = {}
    for pos, layer in enumerate(moe_layer_indices):
        kept = base.keep[layer]
        n_high = math.ceil(high_fraction * len(kept))
        kept_scores = saliency[pos, kept]
        order = np.lexsort((np.arange(len(kept)), -kept_scores))
        bank_high[layer] = sorted(int(kept[i]) for i in order[:n_high])
    return ReapPlan(
        "graded",
        base.moe_layer_indices,
        base.keep,
        bank_high,
        high_bits,
        low_bits,
        num_experts,
        top_k,
    )


def save_plan(plan: ReapPlan, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(plan)
    payload["version"] = PLAN_VERSION
    payload["keep"] = {str(k): v for k, v in plan.keep.items()}
    if plan.bank_high is not None:
        payload["bank_high"] = {str(k): v for k, v in plan.bank_high.items()}
    path.write_text(json.dumps(payload, indent=2))


def load_plan(path: Path) -> ReapPlan:
    raw = json.loads(Path(path).read_text())
    if raw.get("version") != PLAN_VERSION:
        raise ValueError(f"unsupported reap plan version {raw.get('version')}")
    keep = {int(k): list(v) for k, v in raw["keep"].items()}
    bank_high = (
        {int(k): list(v) for k, v in raw["bank_high"].items()}
        if raw.get("bank_high")
        else None
    )
    return ReapPlan(
        raw["mode"],
        list(raw["moe_layer_indices"]),
        keep,
        bank_high,
        raw.get("high_bits"),
        raw.get("low_bits"),
        raw["num_experts_orig"],
        raw["top_k"],
        raw.get("meta", {}),
    )
