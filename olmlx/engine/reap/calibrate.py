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

from olmlx.engine.flash.prepare import load_model_with_strict_fallback
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


def _resolve_head(model, inner):
    """Returns (norm, head_fn) for final logits; head_fn may use tied embeddings."""
    norm = getattr(inner, "norm", None) or getattr(inner, "final_layernorm", None)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        return norm, lm_head
    return norm, inner.embed_tokens.as_linear


def _layer_window_size(inner, args, li: int, layer) -> int | None:
    """Best-effort per-layer sliding-window size, mirroring how the model's own
    full-forward decides it, so the streaming mask matches the hooked one exactly.

    Returns None for global/full-attention layers (plain causal, unbounded).
    Two conventions are checked (covers gpt_oss/olmo3-style backbones and
    gemma3/cohere2-style layers; falls back to full causal if neither applies —
    still correct for archs with no sliding attention, e.g. qwen3_moe/deepseek_v3):
    - a backbone- or args-level `layer_types` list, indexed by `li`, where any
      value other than "full_attention" means sliding (gpt_oss, olmo3);
    - a boolean `is_sliding`/`use_sliding_window` flag on the layer or its
      `self_attn` (gemma3_text, cohere2).
    """
    layer_types = getattr(inner, "layer_types", None) or getattr(
        args, "layer_types", None
    )
    if layer_types is not None and li < len(layer_types):
        is_sliding = layer_types[li] != "full_attention"
    else:
        is_sliding = False
        attn = getattr(layer, "self_attn", None)
        for holder in (attn, layer):
            if holder is None:
                continue
            for attr in ("is_sliding", "use_sliding_window"):
                flag = getattr(holder, attr, None)
                if flag is not None:
                    is_sliding = bool(flag)
                    break
            if is_sliding:
                break
    if not is_sliding:
        return None
    return getattr(inner, "window_size", None) or getattr(args, "sliding_window", None)


def stream_layer_forward(
    model_path,
    tagged_texts,
    *,
    max_tokens=512,
    acc: SaliencyAccumulator | None = None,
    keep_head=False,
    per_sample_final=None,
    progress_callback=None,
):
    """Layer-at-a-time streaming forward over a lazily-loaded model.

    One disk pass; peak RAM ~= one layer + all sample hidden states. With `acc`
    set, installs SaliencyTaps on MoE layers (calibration). With `keep_head`,
    applies final norm + lm_head per sample and calls
    per_sample_final(sample_idx, source, logits, token_ids) (perplexity).
    Consumes the model (layers are nullified as it advances).
    Returns the meta dict (same shape as collect_saliency's).
    """
    import gc

    from mlx_lm.models.base import create_causal_mask
    from mlx_lm.models.cache import make_prompt_cache

    from olmlx.engine.flash.prepare import (
        _embed_normalizer,
        _get_backbone,
        _nullify_module_params,
    )

    model, tokenizer = load_model_with_strict_fallback(model_path, lazy=True)
    inner = _get_backbone(model)
    args = getattr(model, "args", None)
    layers = inner.layers
    moe_layer_indices, num_experts, top_k = _moe_scan(layers)

    # Derived up front from every tagged text (not the post-encode-filter
    # sample list below) so a source whose samples all encode to empty still
    # shows up in meta, matching collect_saliency's up-front `sources`.
    all_sources = list(dict.fromkeys(s for s, _ in tagged_texts))

    embed = inner.embed_tokens
    mx.eval(embed.parameters())
    hidden_size = getattr(args, "hidden_size", 0) or 1
    scale = _embed_normalizer(model, inner, hidden_size)
    hidden, token_ids, sample_sources = [], [], []
    for source, text in tagged_texts:
        ids = _encode(tokenizer, text, max_tokens)
        if not ids:
            continue
        h = embed(mx.array([ids])) * scale
        mx.eval(h)
        hidden.append(h)
        token_ids.append(ids)
        sample_sources.append(source)

    tied = bool(getattr(args, "tie_word_embeddings", False))
    if not keep_head:
        head = getattr(model, "lm_head", None)
        if head is not None:
            _nullify_module_params(head)
        if not tied:
            _nullify_module_params(embed)
    # keep_head: embed must survive when tied (it IS the head); norm+lm_head stay.
    gc.collect()
    mx.clear_cache()

    source_ref = {"idx": 0}
    layer_pos = 0
    for li, layer in enumerate(layers):
        mx.eval(layer.parameters())
        tap_installed: list[tuple[int, Any, str]] = []
        if acc is not None:
            tap_installed = install_taps([layer], acc, source_ref)
            if tap_installed:
                tap = getattr(layer, tap_installed[0][2])
                tap._layer_pos = layer_pos  # renumber from per-call 0 to global ordinal
                layer_pos += 1
        for si in range(len(hidden)):
            if tap_installed:
                source_ref["idx"] = acc.sources.index(sample_sources[si])
            # Fresh per (sample, layer) cache: GDN-safe (recurrent state must
            # not carry across independent forward passes at this layer).
            cache_i = make_prompt_cache(model)[li]
            seq_len = hidden[si].shape[1]
            # Boolean mask (True=attend), matching what each arch's own
            # full-forward builds via create_attention_mask(..., return_array=True)
            # (or the equivalent "causal" fast-path for seq_len <= window_size).
            # A float additive mask (0/-1e9) breaks DeepSeek-V3's MLA attention,
            # which applies the mask via mx.where(mask, scores, min) — an
            # additive float mask is nonzero (truthy) exactly where it should
            # be falsy, inverting the causal direction. See task-4-report.md.
            # Per-layer window_size mirrors sliding-attention layers (gpt_oss
            # etc.) — a global unwindowed mask silently over-attends on those
            # layers once seq_len exceeds the window. See task-4-report.md.
            window = _layer_window_size(inner, args, li, layer)
            mask = (
                create_causal_mask(seq_len, window_size=window) if seq_len > 1 else None
            )
            out = layer(hidden[si], mask=mask, cache=cache_i)
            hidden[si] = out[0] if isinstance(out, (tuple, list)) else out
            mx.eval(hidden[si])
        if tap_installed:
            remove_taps([layer], tap_installed)
        _nullify_module_params(layer)
        gc.collect()
        mx.clear_cache()
        if progress_callback:
            progress_callback(
                f"Processed layer {li + 1}/{len(layers)}",
                0.05 + (li + 1) / len(layers) * 0.9,
            )

    if keep_head and per_sample_final is not None:
        norm, head = _resolve_head(model, inner)
        for si in range(len(hidden)):
            h = norm(hidden[si]) if norm is not None else hidden[si]
            per_sample_final(si, sample_sources[si], head(h), token_ids[si])

    return {
        "model_type": getattr(args, "model_type", "unknown"),
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_layer_indices": moe_layer_indices,
        "sources": all_sources,
        "num_samples": len(hidden),
        "max_tokens": max_tokens,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def calibrate_saliency_streaming(
    model_path, tagged_texts, *, max_tokens=512, progress_callback=None
):
    sources = list(dict.fromkeys(s for s, _ in tagged_texts))
    # geometry probe on a throwaway lazy skeleton is avoided: stream_layer_forward
    # scans MoE geometry itself, but the accumulator needs dims up front — so size
    # it lazily on first add via a thin pre-scan of the same lazy load.
    model, _tok = load_model_with_strict_fallback(model_path, lazy=True)
    from olmlx.engine.flash.prepare import _get_backbone

    moe_layer_indices, num_experts, _top_k = _moe_scan(_get_backbone(model).layers)
    del model
    acc = SaliencyAccumulator(len(moe_layer_indices), num_experts, sources)
    meta = stream_layer_forward(
        model_path,
        list(tagged_texts),
        max_tokens=max_tokens,
        acc=acc,
        progress_callback=progress_callback,
    )
    return acc, meta
