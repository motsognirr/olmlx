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
        # Materialize head weights on the constructing thread. predict_next
        # evals on the prefetcher's prediction thread, and a lazy weight
        # graph stays bound to the creating thread's stream — mlx >= 0.32
        # makes the stream registry thread-local, so evaluating it from the
        # prediction thread raises "There is no Stream(...) in current
        # thread". Same invariant as snapshot_cache_for_persistence.
        mx.eval([h.parameters() for h in self.heads])
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
        # NumPy copies of head weights for predict_next_np, built on first
        # use (load() replaces the constructor weights, so building here
        # would capture the wrong ones). ~rank*(hidden+experts) fp32 per
        # head — small next to the expert bundle.
        self._np_heads: list[tuple[np.ndarray, np.ndarray]] | None = None
        # Holdout recall@m per pair, set by the trainer and persisted in the
        # sidecar. Empty for banks trained before recall persistence.
        self.pair_recalls: dict[int, float] = {}

    def apply_recall_gate(self, min_recall: float) -> int:
        """Disable pairs whose holdout recall@m is below *min_recall*.

        A low-recall head mostly prefetches wrong experts — pure wasted SSD
        bandwidth — so gated pairs are removed from ``trained_pairs`` and
        behave exactly like untrained ones (``next_moe_layer`` returns None;
        no prediction, no I/O). Pairs with no recorded recall (legacy banks)
        are kept: gating on absent data would silently disable prefetch
        wholesale. Returns the number of pairs gated.

        DESTRUCTIVE: do not ``save()`` a gated bank — gated-but-trained
        pairs would persist as untrained, indistinguishable from
        never-trained. Serving only ever gates a fresh per-load instance
        (``_maybe_create_prefetcher``), so relaxing ``min_recall`` takes
        effect on the next model load. Note the recorded recall is measured
        at the trainer's eval margin (1.5); if the serve-time
        ``flash_moe_lookahead_margin`` differs, the gate compares against a
        slightly different m than prefetch actually uses.
        """
        if min_recall <= 0.0:
            return 0
        gated = {
            pair_idx
            for pair_idx, recall in self.pair_recalls.items()
            if recall < min_recall and pair_idx in self.trained_pairs
        }
        self.trained_pairs -= gated
        return len(gated)

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

    def ensure_np_heads(self) -> None:
        """Materialize NumPy copies of every head's weights.

        Must run on a thread that may evaluate the head arrays (the loading
        thread, or the generation thread) — the ``astype`` below is a fresh
        lazy op on the current thread's stream. Idempotent; call after
        :meth:`load` and before handing the bank to a prefetcher.
        """
        if self._np_heads is not None:
            return
        self._np_heads = [
            (
                np.array(head.down.weight.astype(mx.float32)),
                np.array(head.up.weight.astype(mx.float32)),
            )
            for head in self.heads
        ]

    def predict_next_np(
        self,
        layer_idx: int,
        hidden_np: np.ndarray,
        *,
        margin: float = 1.5,
    ) -> tuple[list[int], np.ndarray] | None:
        """NumPy twin of :meth:`predict_next` — no mx ops, no mx.eval.

        Takes an already-materialized float32 hidden state of shape
        ``(positions, hidden_size)``. Safe to call from any thread (the
        head weights are one-time NumPy copies), which is what lets the
        prefetcher predict inline on the forward-pass thread without a
        background-eval rendezvous. Same return contract as
        :meth:`predict_next`.
        """
        if self.next_moe_layer(layer_idx) is None:
            return None
        if self._np_heads is None:
            self.ensure_np_heads()
        assert self._np_heads is not None
        down_w, up_w = self._np_heads[self._pair_for_layer[layer_idx]]
        # SparsityPredictor forward: sigmoid(up(relu(down(x)))), nn.Linear
        # convention y = x @ W.T, averaged over positions.
        pre = np.maximum(hidden_np @ down_w.T, 0.0) @ up_w.T
        scores_np = (1.0 / (1.0 + np.exp(-pre))).mean(axis=0).astype(np.float32)
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
                    # JSON keys are strings; load() converts back to int.
                    "pair_recalls": {
                        str(k): v for k, v in sorted(self.pair_recalls.items())
                    },
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
        bank.pair_recalls = {
            int(k): float(v) for k, v in meta.get("pair_recalls", {}).items()
        }
        for i, head in enumerate(bank.heads):
            weights = dict(mx.load(str(path / f"head_{i:02d}.npz")))  # pyright: ignore[reportCallIssue]
            head.down.weight = weights[f"pair_{i}.down.weight"]
            head.up.weight = weights[f"pair_{i}.up.weight"]
        # mx.load arrays replace the eagerly-evaled constructor weights and
        # may themselves be lazy — re-materialize before the bank crosses to
        # the prediction thread (see the constructor comment).
        mx.eval([h.parameters() for h in bank.heads])
        return bank
