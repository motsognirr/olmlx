"""Flash-aware MLP and WindowManager for LLM in a Flash.

FlashMLP replaces the standard MLP in transformer blocks with a sparse version
that loads only predicted-active neuron weights from SSD.
"""

from __future__ import annotations

from collections import deque

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.flash.predictor import SparsityPredictor
from olmlx.engine.flash.weight_store import FlashWeightStore


class WindowManager:
    """Sliding window of recently activated neurons per layer.

    Keeps track of neurons activated in the last k tokens to reduce
    incremental SSD reads between successive tokens.
    """

    def __init__(self, num_layers: int, window_size: int = 5):
        self.window_size = window_size
        self._history: dict[int, deque[set[int]]] = {
            i: deque(maxlen=window_size) for i in range(num_layers)
        }
        # Cached union of recent window indices per layer
        self._cached_window: dict[int, set[int]] = {i: set() for i in range(num_layers)}
        self._dirty: dict[int, bool] = {i: False for i in range(num_layers)}

    def get_window(self, layer_idx: int) -> set[int]:
        """Return union of neuron indices from last window_size tokens."""
        if not self._dirty.get(layer_idx, False):
            return self._cached_window.get(layer_idx, set())

        history = self._history.get(layer_idx)
        if not history:
            self._cached_window[layer_idx] = set()
        else:
            result: set[int] = set()
            for indices_set in history:
                result |= indices_set
            self._cached_window[layer_idx] = result
        self._dirty[layer_idx] = False
        return self._cached_window[layer_idx]

    def update(self, layer_idx: int, active_indices: mx.array) -> None:
        """Record newly activated neurons for this token."""
        if layer_idx not in self._history:
            self._history[layer_idx] = deque(maxlen=self.window_size)
            self._cached_window[layer_idx] = set()
        mx.eval(active_indices)
        self._history[layer_idx].append(set(active_indices.tolist()))
        self._dirty[layer_idx] = True

    def reset(self) -> None:
        """Clear all window state (new conversation)."""
        for layer_idx in self._history:
            self._history[layer_idx].clear()
            self._cached_window[layer_idx] = set()
            self._dirty[layer_idx] = False


class FlashMLP(nn.Module):
    """Drop-in replacement for standard MLP that loads weights on demand.

    Instead of holding gate_proj/up_proj/down_proj in GPU memory, uses a
    SparsityPredictor + FlashWeightStore to load only active neurons.
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int,
        predictor: SparsityPredictor,
        weight_store: FlashWeightStore,
        window_manager: WindowManager,
        sparsity_threshold: float = 0.5,
        min_active_neurons: int = 128,
        max_active_neurons: int | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.predictor = predictor
        self.weight_store = weight_store
        self.window_manager = window_manager
        self.sparsity_threshold = sparsity_threshold
        self.min_active_neurons = min_active_neurons
        self.max_active_neurons = max_active_neurons

    def __call__(self, x: mx.array) -> mx.array:
        """Sparse FFN forward pass."""
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.hidden_size)

        # Predict active neurons
        predicted = self.predictor.predict_active(
            flat_x,
            threshold=self.sparsity_threshold,
            min_neurons=self.min_active_neurons,
            max_neurons=self.max_active_neurons,
        )
        mx.eval(predicted)
        predicted_list = predicted.tolist()

        # Union with window (sets, no MLX→Python→MLX roundtrip)
        window = self.window_manager.get_window(self.layer_idx)
        if window:
            combined = sorted(set(predicted_list) | window)
        else:
            combined = predicted_list

        # Load weights from store
        gate_cols, up_cols, down_rows = self.weight_store.load_neurons(
            self.layer_idx, combined
        )

        # Sparse SwiGLU forward
        gate_out = flat_x @ gate_cols
        up_out = flat_x @ up_cols
        act = mx.sigmoid(gate_out) * gate_out * up_out
        output = act @ down_rows

        # Update window
        self.window_manager.update(self.layer_idx, predicted)

        return output.reshape(orig_shape)
