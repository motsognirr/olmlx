"""Sparsity predictor for LLM in a Flash.

Low-rank MLP that predicts which FFN neurons will activate for a given input,
enabling selective weight loading from SSD.
"""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class SparsityPredictor(nn.Module):
    """Low-rank predictor for FFN neuron activation.

    Architecture per the paper:
        input (hidden_size) -> Linear(hidden_size, rank) -> ReLU
        -> Linear(rank, intermediate_size) -> sigmoid
    """

    def __init__(self, hidden_size: int, intermediate_size: int, rank: int = 128):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Return activation scores for each neuron (0-1 range)."""
        return mx.sigmoid(self.up(mx.maximum(self.down(x), 0)))

    def predict_active(
        self,
        x: mx.array,
        threshold: float = 0.5,
        min_neurons: int = 128,
        max_neurons: int | None = None,
    ) -> mx.array:
        """Return sorted indices of predicted-active neurons.

        Args:
            x: Input hidden state, shape (batch, hidden_size) or (hidden_size,).
            threshold: Activation score threshold.
            min_neurons: Minimum number of neurons to select.
            max_neurons: Maximum number of neurons to select.

        Returns:
            1D array of sorted neuron indices.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Average scores across batch/sequence positions
        scores = self(x).mean(axis=0)  # (intermediate_size,)
        mx.eval(scores)

        n = scores.shape[0]
        min_neurons = min(min_neurons, n)

        # Determine how many neurons to select
        active_mask = scores > threshold
        num_active = int(mx.sum(active_mask).item())

        # Use max of num_active and min_neurons, capped by max_neurons
        k = max(num_active, min_neurons)
        if max_neurons is not None:
            k = min(k, max_neurons)
        k = min(k, n)

        # Select top-k neurons by score using argpartition
        indices = mx.argpartition(scores, kth=n - k)[n - k :]
        return mx.sort(indices)


def compute_layer_ranks(
    num_layers: int,
    base_rank: int = 128,
    sensitive_layers: int = 0,
    sensitive_rank_multiplier: int = 4,
) -> list[int]:
    """Compute per-layer predictor ranks with higher rank for sensitive layers.

    The last `sensitive_layers` layers get `base_rank * sensitive_rank_multiplier`.
    """
    ranks = [base_rank] * num_layers
    for i in range(max(0, num_layers - sensitive_layers), num_layers):
        ranks[i] = base_rank * sensitive_rank_multiplier
    return ranks


class PredictorBank:
    """Collection of per-layer sparsity predictors."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        rank: int = 128,
        ranks: list[int] | None = None,
    ):
        if ranks is not None:
            if len(ranks) != num_layers:
                raise ValueError(
                    f"ranks length {len(ranks)} != num_layers {num_layers}"
                )
            self.predictors = [
                SparsityPredictor(hidden_size, intermediate_size, r) for r in ranks
            ]
        else:
            self.predictors = [
                SparsityPredictor(hidden_size, intermediate_size, rank)
                for _ in range(num_layers)
            ]

    def save(self, path: Path) -> None:
        """Save all predictor weights to a directory."""
        path.mkdir(parents=True, exist_ok=True)
        for i, pred in enumerate(self.predictors):
            weights = dict(pred.parameters())
            flat = {}
            for key, val in weights.items():
                if isinstance(val, dict):
                    for subkey, subval in val.items():
                        flat[f"layer_{i}.{key}.{subkey}"] = subval
                else:
                    flat[f"layer_{i}.{key}"] = val
            mx.savez(str(path / f"predictor_{i:02d}.npz"), **flat)

    @classmethod
    def load(cls, path: Path) -> PredictorBank:
        """Load predictor bank from a directory."""
        files = sorted(
            path.glob("predictor_*.npz"),
            key=lambda f: int(re.search(r"(\d+)", f.name).group(1)),
        )
        if not files:
            raise FileNotFoundError(f"No predictor files found in {path}")

        bank = cls.__new__(cls)
        bank.predictors = []

        for f in files:
            m = re.search(r"predictor_(\d+)", f.stem)
            if m is None:
                raise ValueError(f"Cannot parse layer index from {f.name}")
            i = int(m.group(1))

            weights = dict(mx.load(str(f)))
            down_w = weights[f"layer_{i}.down.weight"]
            up_w = weights[f"layer_{i}.up.weight"]

            # Read rank per-file to support mixed ranks
            rank, hidden_size = down_w.shape
            intermediate_size, _ = up_w.shape

            pred = SparsityPredictor(hidden_size, intermediate_size, rank)
            pred.down.weight = down_w
            pred.up.weight = up_w
            bank.predictors.append(pred)

        return bank

    def predict_layer(
        self,
        layer_idx: int,
        hidden_state: mx.array,
        threshold: float = 0.5,
        min_neurons: int = 128,
        max_neurons: int | None = None,
    ) -> mx.array:
        """Predict active neuron indices for a layer."""
        return self.predictors[layer_idx].predict_active(
            hidden_state, threshold, min_neurons, max_neurons
        )


# Same architecture as SparsityPredictor, trained on cross-layer objective.
LookaheadPredictor = SparsityPredictor


class LookaheadBank:
    """Collection of per-layer-pair lookahead predictors.

    Predictor at index i predicts layer i+1 neurons given layer i hidden state.
    There are num_layers - 1 predictors (no predictor for the last layer).
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        rank: int = 64,
    ):
        self.num_layers = num_layers
        self.predictors = [
            SparsityPredictor(hidden_size, intermediate_size, rank)
            for _ in range(num_layers - 1)
        ]

    def save(self, path: Path) -> None:
        """Save all lookahead predictor weights to a directory."""
        path.mkdir(parents=True, exist_ok=True)
        for i, pred in enumerate(self.predictors):
            weights = dict(pred.parameters())
            flat = {}
            for key, val in weights.items():
                if isinstance(val, dict):
                    for subkey, subval in val.items():
                        flat[f"pair_{i}.{key}.{subkey}"] = subval
                else:
                    flat[f"pair_{i}.{key}"] = val
            mx.savez(str(path / f"lookahead_{i:02d}.npz"), **flat)

    @classmethod
    def load(cls, path: Path) -> LookaheadBank:
        """Load lookahead bank from a directory."""
        files = sorted(
            path.glob("lookahead_*.npz"),
            key=lambda f: int(re.search(r"(\d+)", f.name).group(1)),
        )
        if not files:
            raise FileNotFoundError(f"No lookahead predictor files found in {path}")

        bank = cls.__new__(cls)
        bank.predictors = []

        parsed: dict[int, Path] = {}
        for f in files:
            m = re.search(r"lookahead_(\d+)", f.stem)
            if m is None:
                raise ValueError(f"Cannot parse pair index from {f.name}")
            parsed[int(m.group(1))] = f

        expected = list(range(len(parsed)))
        if sorted(parsed.keys()) != expected:
            raise ValueError(
                f"Non-contiguous lookahead files: found indices {sorted(parsed.keys())}, "
                f"expected {expected}"
            )

        bank.num_layers = len(parsed) + 1

        for i in expected:
            f = parsed[i]
            weights = dict(mx.load(str(f)))
            down_w = weights[f"pair_{i}.down.weight"]
            up_w = weights[f"pair_{i}.up.weight"]

            rank, hidden_size = down_w.shape
            intermediate_size, _ = up_w.shape

            pred = SparsityPredictor(hidden_size, intermediate_size, rank)
            pred.down.weight = down_w
            pred.up.weight = up_w
            bank.predictors.append(pred)

        return bank

    def predict_next_layer(
        self,
        layer_idx: int,
        hidden_state: mx.array,
        threshold: float = 0.3,
        min_neurons: int = 64,
        max_neurons: int | None = None,
    ) -> mx.array:
        """Predict active neuron indices for layer_idx + 1."""
        if layer_idx >= len(self.predictors):
            return mx.array([], dtype=mx.int32)
        return self.predictors[layer_idx].predict_active(
            hidden_state, threshold, min_neurons, max_neurons
        )
