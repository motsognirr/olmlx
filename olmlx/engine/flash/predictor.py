"""Sparsity predictor for LLM in a Flash.

Low-rank MLP that predicts which FFN neurons will activate for a given input,
enabling selective weight loading from SSD.
"""

from __future__ import annotations

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


class PredictorBank:
    """Collection of per-layer sparsity predictors."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        rank: int = 128,
    ):
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
        files = sorted(path.glob("predictor_*.npz"))
        if not files:
            raise FileNotFoundError(f"No predictor files found in {path}")

        # Peek at first predictor to get dimensions
        first_weights = dict(mx.load(str(files[0])))
        # Expect keys like "layer_0.down.weight", "layer_0.up.weight"
        down_weight_key = [k for k in first_weights if "down.weight" in k][0]
        up_weight_key = [k for k in first_weights if "up.weight" in k][0]
        rank, hidden_size = first_weights[down_weight_key].shape
        intermediate_size, _ = first_weights[up_weight_key].shape

        bank = cls.__new__(cls)
        bank.predictors = []

        for i, f in enumerate(files):
            pred = SparsityPredictor(hidden_size, intermediate_size, rank)
            weights = dict(mx.load(str(f)))

            down_w = weights[f"layer_{i}.down.weight"]
            up_w = weights[f"layer_{i}.up.weight"]
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
