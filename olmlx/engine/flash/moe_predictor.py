"""Expert lookahead predictors for Flash-MoE.

Per-MoE-layer-pair low-rank heads that predict which experts the NEXT MoE
layer's router will select, given the hidden state entering the current MoE
layer. Drives speculative expert prefetch and predicted-need cache eviction.

Same architecture family as the dense-path ``SparsityPredictor``
(``predictor.py``), with ``num_experts`` outputs instead of
``intermediate_size``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from olmlx.engine.flash.predictor import SparsityPredictor

SIDECAR_NAME = "moe_lookahead.json"


class MoeLookaheadBank:
    """Per-MoE-layer-pair expert lookahead heads.

    Head ``i`` predicts the expert set of ``moe_layer_indices[i + 1]`` from
    the hidden state entering ``moe_layer_indices[i]``. Pairs follow
    *consecutive MoE layers*, not consecutive layer indices — interleaved
    dense layers just add I/O lead time.
    """

    def __init__(
        self,
        moe_layer_indices: list[int],
        hidden_size: int,
        num_experts: int,
        rank: int = 128,
        num_experts_per_tok: int = 8,
        trained_pairs: set[int] | None = None,
    ):
        indices = sorted(moe_layer_indices)
        if len(indices) < 2:
            raise ValueError(f"Need at least 2 MoE layers for lookahead, got {indices}")
        self.moe_layer_indices = indices
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.rank = rank
        self.heads = [
            SparsityPredictor(hidden_size, num_experts, rank)
            for _ in range(len(indices) - 1)
        ]
        self._pair_for_layer = {indices[i]: i for i in range(len(indices) - 1)}
        self._next_for_layer = {
            indices[i]: indices[i + 1] for i in range(len(indices) - 1)
        }
        # Default: all pairs trained — matches prior behaviour for banks
        # built directly (not via train_from_traces), e.g. in tests or
        # hand-assembled banks.
        self.trained_pairs: set[int] = (
            set(range(len(indices) - 1))
            if trained_pairs is None
            else set(trained_pairs)
        )

    def next_moe_layer(self, layer_idx: int) -> int | None:
        """The MoE layer whose experts head(layer_idx) predicts, or None.

        Returns None when the pair is untrained (randomly-initialized head)
        even if a successor layer exists structurally — an untrained head's
        scores are garbage and must not drive prefetch or eviction.
        """
        pair_idx = self._pair_for_layer.get(layer_idx)
        if pair_idx is None or pair_idx not in self.trained_pairs:
            return None
        return self._next_for_layer.get(layer_idx)

    def predict_next(
        self,
        layer_idx: int,
        hidden_state: mx.array,
        *,
        margin: float = 1.5,
    ) -> tuple[list[int], np.ndarray] | None:
        """Predict the next MoE layer's expert set from *layer_idx*'s input.

        Returns ``(sorted top-m expert indices, full score vector)`` where
        ``m = min(num_experts, ceil(margin * num_experts_per_tok))``, or
        ``None`` if *layer_idx* has no successor head or the pair is
        untrained (routed through the same check as ``next_moe_layer`` so
        the two stay consistent). Calls ``mx.eval`` — only safe on the
        prediction thread (or when no prediction is in flight).
        """
        if self.next_moe_layer(layer_idx) is None:
            return None
        pair_idx = self._pair_for_layer[layer_idx]
        flat = hidden_state.reshape(-1, hidden_state.shape[-1])
        scores = self.heads[pair_idx](flat).mean(axis=0)
        mx.eval(scores)
        scores_np = np.array(scores, dtype=np.float32)
        m = min(self.num_experts, math.ceil(margin * self.num_experts_per_tok))
        top_m = (
            np.argpartition(-scores_np, m - 1)[:m]
            if m < len(scores_np)
            else np.arange(len(scores_np))
        )
        return sorted(int(i) for i in top_m), scores_np

    def save(self, path: Path) -> None:
        """Save heads + sidecar to a directory."""
        path.mkdir(parents=True, exist_ok=True)
        for i, head in enumerate(self.heads):
            mx.savez(
                str(path / f"head_{i:02d}.npz"),
                **{
                    f"pair_{i}.down.weight": head.down.weight,
                    f"pair_{i}.up.weight": head.up.weight,
                },
            )
        (path / SIDECAR_NAME).write_text(
            json.dumps(
                {
                    "hidden_size": self.hidden_size,
                    "num_experts": self.num_experts,
                    "num_experts_per_tok": self.num_experts_per_tok,
                    "rank": self.rank,
                    "moe_layer_indices": self.moe_layer_indices,
                    "trained_pairs": sorted(self.trained_pairs),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> MoeLookaheadBank:
        """Load a bank from a directory written by :meth:`save`.

        A sidecar missing ``trained_pairs`` (there are no released banks
        that predate this field — this only covers hand-edited/legacy
        sidecars) is treated as all-trained, matching the constructor
        default.
        """
        sidecar_path = path / SIDECAR_NAME
        if not sidecar_path.exists():
            raise FileNotFoundError(f"No {SIDECAR_NAME} in {path}")
        meta = json.loads(sidecar_path.read_text())
        num_pairs = len(meta["moe_layer_indices"]) - 1
        trained_pairs = (
            set(meta["trained_pairs"])
            if "trained_pairs" in meta
            else set(range(num_pairs))
        )
        bank = cls(
            meta["moe_layer_indices"],
            hidden_size=meta["hidden_size"],
            num_experts=meta["num_experts"],
            rank=meta["rank"],
            num_experts_per_tok=meta["num_experts_per_tok"],
            trained_pairs=trained_pairs,
        )
        for i, head in enumerate(bank.heads):
            weights = dict(mx.load(str(path / f"head_{i:02d}.npz")))  # pyright: ignore[reportCallIssue]
            head.down.weight = weights[f"pair_{i}.down.weight"]
            head.up.weight = weights[f"pair_{i}.up.weight"]
        return bank
