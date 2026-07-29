"""REAP saliency calibration: taps, accumulator, full-forward + streaming drivers."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from olmlx.engine.reap.arch import MoeModuleInfo, applied_scores, find_moe_module

logger = logging.getLogger(__name__)


class SaliencyAccumulator:
    def __init__(
        self, num_moe_layers: int, num_experts: int, sources: list[str]
    ) -> None:
        self.sources = list(sources)
        self.sum = np.zeros(
            (num_moe_layers, len(self.sources), num_experts), dtype=np.float64
        )
        self.count = np.zeros_like(self.sum, dtype=np.int64)

    def add(
        self,
        layer_pos: int,
        source_idx: int,
        expert_ids: np.ndarray,
        contribs: np.ndarray,
    ) -> None:
        np.add.at(self.sum[layer_pos, source_idx], expert_ids, contribs)
        np.add.at(self.count[layer_pos, source_idx], expert_ids, 1)

    def _source_indices(self, sources: list[str] | None) -> list[int]:
        if sources is None:
            return list(range(len(self.sources)))
        missing = [s for s in sources if s not in self.sources]
        if missing:
            raise ValueError(
                f"unknown sources {missing}; calibrated sources: {self.sources}"
            )
        return [self.sources.index(s) for s in sources]

    def saliency(self, sources: list[str] | None = None) -> np.ndarray:
        idx = self._source_indices(sources)
        s = self.sum[:, idx].sum(axis=1)
        c = self.count[:, idx].sum(axis=1)
        return s / np.maximum(c, 1)

    def frequency(self, sources: list[str] | None = None) -> np.ndarray:
        idx = self._source_indices(sources)
        return self.count[:, idx].sum(axis=1)


def save_saliency(path: Path, acc: SaliencyAccumulator, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        sum=acc.sum,
        count=acc.count,
        sources=np.array(acc.sources),
        meta=json.dumps(meta),
    )


def load_saliency(path: Path) -> tuple[SaliencyAccumulator, dict]:
    data = np.load(path, allow_pickle=False)
    sources = [str(s) for s in data["sources"]]
    acc = SaliencyAccumulator(data["sum"].shape[0], data["sum"].shape[2], sources)
    acc.sum = data["sum"].astype(np.float64)
    acc.count = data["count"].astype(np.int64)
    return acc, json.loads(str(data["meta"]))


class _CaptureContainer(nn.Module):
    def __init__(self, inner: Any, captured: dict) -> None:
        super().__init__()
        self.inner = inner
        self._captured = captured

    def __call__(self, x, inds, **kwargs):
        out = self.inner(x, inds, **kwargs)
        self._captured["inds"] = inds
        self._captured["expert_outs"] = out
        return out


class _CaptureGate(nn.Module):
    def __init__(self, inner: Any, captured: dict) -> None:
        super().__init__()
        self.inner = inner
        self._captured = captured

    def __call__(self, x):
        out = self.inner(x)
        self._captured["gate_out"] = out
        return out


class SaliencyTap(nn.Module):
    """Replaces a MoE block attr on a decoder layer; records REAP contributions
    from the block's own forward (no routing reimplementation on the hot path)."""

    def __init__(
        self,
        info: MoeModuleInfo,
        layer_pos: int,
        acc: SaliencyAccumulator,
        source_ref: dict[str, int],
    ) -> None:
        super().__init__()
        self.inner = info.module
        self._info = info
        self._layer_pos = layer_pos
        self._acc = acc
        self._source_ref = source_ref

    def __call__(self, x, *args, **kwargs):
        info = self._info
        container = getattr(self.inner, info.container_name)
        gate = getattr(self.inner, info.gate_attr)
        captured: dict = {}
        setattr(self.inner, info.container_name, _CaptureContainer(container, captured))
        setattr(self.inner, info.gate_attr, _CaptureGate(gate, captured))
        try:
            y = self.inner(x, *args, **kwargs)
        finally:
            setattr(self.inner, info.container_name, container)
            setattr(self.inner, info.gate_attr, gate)
        inds = captured["inds"]
        scores = applied_scores(info.style, self.inner, captured["gate_out"], inds)
        norms = mx.linalg.norm(captured["expert_outs"].astype(mx.float32), axis=-1)
        contribs = np.asarray((scores * norms).reshape(-1), dtype=np.float64)
        experts = np.asarray(inds.reshape(-1))
        self._acc.add(self._layer_pos, self._source_ref["idx"], experts, contribs)
        return y


def install_taps(
    model_layers,
    acc: SaliencyAccumulator,
    source_ref: dict[str, int],
    *,
    top_k_hint: int | None = None,
) -> list[tuple[int, Any, str]]:
    installed: list[tuple[int, Any, str]] = []
    layer_pos = 0
    for i, layer in enumerate(model_layers):
        info = find_moe_module(layer, top_k_hint=top_k_hint)
        if info is None:
            continue
        tap = SaliencyTap(info, layer_pos, acc, source_ref)
        installed.append((i, info.module, info.attr_name))
        setattr(layer, info.attr_name, tap)
        layer_pos += 1
    return installed


def remove_taps(model_layers, installed: list[tuple[int, Any, str]]) -> None:
    for i, original, attr in installed:
        setattr(model_layers[i], attr, original)


def _encode(tokenizer, text: str, max_tokens: int) -> list[int]:
    tokens = tokenizer.encode(text)
    if isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    return list(tokens)[:max_tokens]


def _moe_scan(layers, top_k_hint=None) -> tuple[list[int], int, int]:
    indices, num_experts, top_k = [], 0, 0
    for i, layer in enumerate(layers):
        info = find_moe_module(layer, top_k_hint=top_k_hint)
        if info is not None:
            indices.append(i)
            num_experts, top_k = info.num_experts, info.top_k
    if not indices:
        raise ValueError(
            "model has no recognized MoE layers; REAP requires an MoE model"
        )
    return indices, num_experts, top_k


def _backbone(model):
    from olmlx.engine.flash.prepare import _get_backbone

    return _get_backbone(model)


def collect_saliency(
    model,
    tokenizer,
    tagged_texts: list[tuple[str, str]],
    *,
    max_tokens: int = 512,
    progress_callback: Callable[[str, float], None] | None = None,
) -> tuple[SaliencyAccumulator, dict]:
    """Full-forward (non-streaming) saliency collection for models that fit in RAM."""
    inner = _backbone(model)
    layers = inner.layers
    moe_layer_indices, num_experts, top_k = _moe_scan(layers)
    sources = list(dict.fromkeys(s for s, _ in tagged_texts))
    acc = SaliencyAccumulator(len(moe_layer_indices), num_experts, sources)
    source_ref = {"idx": 0}
    installed = install_taps(layers, acc, source_ref)
    try:
        for n, (source, text) in enumerate(tagged_texts):
            source_ref["idx"] = sources.index(source)
            ids = _encode(tokenizer, text, max_tokens)
            if not ids:
                continue
            mx.eval(model(mx.array([ids])))
            if progress_callback:
                progress_callback(
                    f"Calibrated {n + 1}/{len(tagged_texts)} samples",
                    (n + 1) / len(tagged_texts),
                )
    finally:
        remove_taps(layers, installed)
    meta = {
        "model_type": getattr(getattr(model, "args", None), "model_type", "unknown"),
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_layer_indices": moe_layer_indices,
        "sources": sources,
        "num_samples": len(tagged_texts),
        "max_tokens": max_tokens,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return acc, meta
