"""Offline training for Flash-MoE expert lookahead heads.

Records (hidden state at MoE layer L, router top-k at the next MoE layer)
traces by running the Flash-MoE-wrapped model over calibration texts, then
trains one low-rank head per consecutive MoE-layer pair with recall-biased
BCE. Router targets come free from the resident gates — no labels beyond the
forward pass itself.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from olmlx.engine.flash.moe_predictor import MoeLookaheadBank
from olmlx.engine.flash.prepare import _train_single_predictor

logger = logging.getLogger(__name__)


def build_multi_hot(inds: np.ndarray, num_experts: int) -> np.ndarray:
    """(P, K) integer expert indices -> (P, num_experts) float32 multi-hot."""
    out = np.zeros((inds.shape[0], num_experts), dtype=np.float32)
    out[np.arange(inds.shape[0])[:, None], inds] = 1.0
    return out


def recall_at_m(pred_scores: np.ndarray, true_inds: np.ndarray, m: int) -> float:
    """Mean fraction of true experts found in the top-m predicted experts."""
    m = min(m, pred_scores.shape[1])
    top_m = np.argpartition(-pred_scores, m - 1, axis=1)[:, :m]
    hits = 0
    for row_top, row_true in zip(top_m, true_inds):
        hits += len(set(row_top.tolist()) & set(row_true.tolist()))
    return hits / true_inds.size


def record_moe_router_traces(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    moe_layer_indices: list[int],
    *,
    max_positions_per_layer: int = 4096,
    max_tokens_per_text: int = 512,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Record per-MoE-layer (input hidden, router top-k) over *texts*.

    Temporarily wraps each MoE layer's replacement module (which must expose
    ``_route``) with a recorder, runs full forward passes, and restores the
    original modules. Positions are aligned across layers (each forward
    contributes the same positions everywhere), so pair training can zip
    layer L's hiddens with layer M's indices.

    Modules without ``_route`` (Gemma4-style pre-routed experts, or dense
    layers) are skipped with a warning.
    """
    import mlx.nn as nn

    from olmlx.engine.flash.flash_moe_model import _find_moe_module

    class _Recorder(nn.Module):
        def __init__(self, inner: Any, hidden_sink: list, inds_sink: list):
            super().__init__()
            # Bypass nn.Module attribute registration for the inner module —
            # the recorder must not appear to own its parameters.
            object.__setattr__(self, "_inner", inner)
            object.__setattr__(self, "_hidden_sink", hidden_sink)
            object.__setattr__(self, "_inds_sink", inds_sink)

        def __call__(self, x: mx.array) -> mx.array:
            recorded = sum(a.shape[0] for a in self._hidden_sink)
            if recorded < max_positions_per_layer:
                flat = x.reshape(-1, x.shape[-1])
                inds, _ = self._inner._route(x)
                flat_inds = inds.reshape(-1, inds.shape[-1])
                mx.eval(flat, flat_inds)
                budget = max_positions_per_layer - recorded
                self._hidden_sink.append(np.array(flat.astype(mx.float32))[:budget])
                self._inds_sink.append(np.array(flat_inds)[:budget])
            return self._inner(x)

    sinks: dict[int, tuple[list, list]] = {}
    originals: dict[int, tuple[Any, str]] = {}
    layers = model.layers

    for layer_idx in sorted(moe_layer_indices):
        layer = layers[layer_idx]
        # The Flash-MoE replacement sits where the original MoE module was.
        try:
            attr, mod = _find_moe_module(layer)
        except AttributeError:
            attr, mod = None, None
        if mod is None or not hasattr(mod, "_route"):
            logger.warning(
                "MoE layer %d has no _route-style module — skipping trace "
                "recording (Gemma4-style pre-routed layers are unsupported)",
                layer_idx,
            )
            continue
        hidden_sink: list = []
        inds_sink: list = []
        sinks[layer_idx] = (hidden_sink, inds_sink)
        originals[layer_idx] = (mod, attr)
        setattr(layer, attr, _Recorder(mod, hidden_sink, inds_sink))

    try:
        for text in texts:
            tokens = tokenizer.encode(text)[:max_tokens_per_text]
            if not tokens:
                continue
            out = model(mx.array([tokens]))
            mx.eval(out)
    finally:
        for layer_idx, (mod, attr) in originals.items():
            setattr(layers[layer_idx], attr, mod)

    traces: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer_idx, (hidden_sink, inds_sink) in sinks.items():
        if not hidden_sink:
            logger.warning("No positions recorded for MoE layer %d", layer_idx)
            continue
        traces[layer_idx] = (
            np.concatenate(hidden_sink, axis=0),
            np.concatenate(inds_sink, axis=0).astype(np.int32),
        )
    return traces


def train_from_traces(
    traces: dict[int, tuple[np.ndarray, np.ndarray]],
    moe_layer_indices: list[int],
    hidden_size: int,
    num_experts: int,
    *,
    num_experts_per_tok: int,
    rank: int = 128,
    epochs: int = 5,
    lr: float = 1e-3,
    holdout_fraction: float = 0.1,
    eval_margin: float = 1.5,
    progress_callback: Callable[[str, float], None] | None = None,
) -> tuple[MoeLookaheadBank, dict[str, float]]:
    """Train per-pair heads from recorded traces.

    Returns ``(bank, recalls)`` where ``recalls`` maps ``"L→M"`` to holdout
    recall@m (m = ceil(eval_margin * num_experts_per_tok)). Pairs with a
    missing trace are skipped (absent from ``recalls``).
    """
    bank = MoeLookaheadBank(
        moe_layer_indices,
        hidden_size,
        num_experts,
        rank=rank,
        num_experts_per_tok=num_experts_per_tok,
    )
    indices = bank.moe_layer_indices
    recalls: dict[str, float] = {}
    num_pairs = len(indices) - 1

    for pair_idx in range(num_pairs):
        src, dst = indices[pair_idx], indices[pair_idx + 1]
        if src not in traces or dst not in traces:
            logger.warning("No trace pair for MoE layers %d→%d, skipping", src, dst)
            continue
        hid = traces[src][0]
        next_inds = traces[dst][1]
        n = min(len(hid), len(next_inds))
        hid, next_inds = hid[:n], next_inds[:n]

        n_hold = int(n * holdout_fraction) if n > 10 else 0
        n_train = n - n_hold

        def _on_epoch(epoch: int, _p=pair_idx) -> None:
            if progress_callback:
                progress_callback(
                    f"Training pair {_p + 1}/{num_pairs} epoch {epoch + 1}/{epochs}",
                    (_p * epochs + epoch + 1) / (num_pairs * epochs),
                )

        # 2x recall bias, same as the dense lookahead: a false negative is a
        # synchronous SSD miss on the critical path; a false positive is one
        # wasted read.
        _train_single_predictor(
            bank.heads[pair_idx],
            mx.array(hid[:n_train]),
            mx.array(build_multi_hot(next_inds[:n_train], num_experts)),
            epochs=epochs,
            lr=lr,
            pos_weight_multiplier=2.0,
            epoch_callback=_on_epoch,
        )

        if n_hold:
            scores = bank.heads[pair_idx](mx.array(hid[n_train:]))
            mx.eval(scores)
            m = min(num_experts, math.ceil(eval_margin * num_experts_per_tok))
            recalls[f"{src}→{dst}"] = recall_at_m(
                np.array(scores, dtype=np.float32), next_inds[n_train:], m
            )

    return bank, recalls


def train_moe_lookahead(
    model_path: str,
    flash_moe_dir: Path,
    *,
    rank: int = 128,
    epochs: int = 5,
    lr: float = 1e-3,
    num_samples: int = 32,
    calibration_dataset: str | None = None,
    max_positions_per_layer: int = 4096,
    holdout_fraction: float = 0.1,
    io_threads: int = 16,
    cache_budget_experts: int = 48,
    progress_callback: Callable[[str, float], None] | None = None,
) -> Path:
    """End-to-end: record traces on the Flash-MoE model, train, save.

    Saves to ``<flash_moe_dir>/moe_lookahead`` and returns that path. Prints
    per-pair holdout recall via the logger; the CLI surfaces it to the user.
    """
    import json

    from olmlx.engine.flash.flash_moe_model import load_flash_moe_model
    from olmlx.engine.flash.prepare import (
        _get_c4_calibration_data,
        _get_calibration_data,
    )

    moe_config = json.loads((flash_moe_dir / "flash_moe_config.json").read_text())
    moe_layer_indices = moe_config["moe_layer_indices"]
    if len(moe_layer_indices) < 2:
        raise ValueError(
            f"Need at least 2 MoE layers for lookahead, got {moe_layer_indices}"
        )

    if calibration_dataset == "synthetic":
        texts = _get_calibration_data(num_samples)
    else:
        texts = _get_c4_calibration_data(num_samples)

    if progress_callback:
        progress_callback("Loading model (Flash-MoE, lazy)", 0.0)

    model, tokenizer, store = load_flash_moe_model(
        model_path,
        flash_moe_dir,
        cache_budget_experts=cache_budget_experts,
        io_threads=io_threads,
    )
    try:
        if progress_callback:
            progress_callback("Recording router traces", 0.05)
        traces = record_moe_router_traces(
            model,
            tokenizer,
            texts,
            moe_layer_indices,
            max_positions_per_layer=max_positions_per_layer,
        )
    finally:
        store.close()

    if not traces:
        raise RuntimeError(
            "No router traces recorded — model has no _route-style MoE layers"
        )

    bank, recalls = train_from_traces(
        traces,
        moe_layer_indices,
        hidden_size=moe_config["hidden_size"],
        num_experts=moe_config["num_experts"],
        num_experts_per_tok=moe_config["num_experts_per_tok"],
        rank=rank,
        epochs=epochs,
        lr=lr,
        holdout_fraction=holdout_fraction,
        progress_callback=lambda desc, frac: (
            progress_callback(desc, 0.1 + frac * 0.85) if progress_callback else None
        ),
    )

    out_dir = flash_moe_dir / "moe_lookahead"
    bank.save(out_dir)
    for pair, recall in recalls.items():
        logger.info("Holdout recall@m for pair %s: %.3f", pair, recall)
    if progress_callback:
        progress_callback("Done", 1.0)
    return out_dir
