"""REAP reporting: per-source expert-overlap analysis and the held-out
perplexity gate the kimi-k3-mlx repo explicitly lacked."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from olmlx.engine.reap.calibrate import SaliencyAccumulator, load_saliency


def source_overlap(acc: SaliencyAccumulator, keep_count: int) -> dict:
    num_layers, _, num_experts = acc.sum.shape
    tops: dict[str, list[set[int]]] = {}
    for source in acc.sources:
        sal = acc.saliency([source])
        tops[source] = [
            set(np.argsort(-sal[layer])[:keep_count].tolist())
            for layer in range(num_layers)
        ]
    pairs = {}
    for a, b in itertools.combinations(acc.sources, 2):
        per_layer = [
            len(tops[a][layer] & tops[b][layer]) / keep_count
            for layer in range(num_layers)
        ]
        pairs[f"{a}|{b}"] = {
            "mean_overlap": float(np.mean(per_layer)),
            "per_layer": per_layer,
        }
    return {
        "pairs": pairs,
        "chance": keep_count / num_experts,
        "keep_count": keep_count,
    }


def streaming_perplexity(
    model_path: str,
    tagged_texts,
    *,
    max_tokens: int = 512,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, float]:
    from olmlx.engine.reap import calibrate as cal_mod

    totals: dict[str, list[float]] = {}

    def on_final(
        sample_idx: int, source: str, logits: mx.array, ids: list[int]
    ) -> None:
        if len(ids) < 2:
            return
        lg = logits.astype(mx.float32)[:, :-1]
        logprobs = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        tgt = mx.array([ids[1:]])
        nll = float(-mx.take_along_axis(logprobs, tgt[..., None], axis=-1).sum())
        acc = totals.setdefault(source, [0.0, 0])
        acc[0] += nll
        acc[1] += len(ids) - 1

    cal_mod.stream_layer_forward(
        model_path,
        list(tagged_texts),
        max_tokens=max_tokens,
        keep_head=True,
        per_sample_final=on_final,
        progress_callback=progress_callback,
    )
    return {source: math.exp(nll / max(n, 1)) for source, (nll, n) in totals.items()}


def build_report(
    saliency_path: Path,
    *,
    keep_count: int,
    full_model_dir: str | None = None,
    pruned_model_dir: str | None = None,
    held_out_texts=None,
    max_tokens: int = 512,
    progress_callback=None,
) -> dict:
    acc, meta = load_saliency(saliency_path)
    report: dict = {
        "overlap": source_overlap(acc, keep_count),
        "meta": meta,
        "perplexity": None,
    }
    if full_model_dir and pruned_model_dir and held_out_texts:
        full = streaming_perplexity(
            full_model_dir,
            held_out_texts,
            max_tokens=max_tokens,
            progress_callback=progress_callback,
        )
        pruned = streaming_perplexity(
            pruned_model_dir,
            held_out_texts,
            max_tokens=max_tokens,
            progress_callback=progress_callback,
        )
        ratio = {s: pruned[s] / full[s] for s in full if s in pruned}
        report["perplexity"] = {"full": full, "pruned": pruned, "ratio": ratio}
    return report
