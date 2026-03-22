"""Flash model wrapper for LLM in a Flash.

Wraps an mlx-lm model, replacing standard FFN layers with FlashMLP instances
that load weights on demand from SSD. Compatible with mlx_lm.stream_generate().
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.flash.flash_mlp import FlashMLP, WindowManager
from olmlx.engine.flash.predictor import PredictorBank
from olmlx.engine.flash.weight_store import FlashWeightStore

logger = logging.getLogger(__name__)


@dataclass
class FlashConfig:
    """Configuration for flash inference."""

    hidden_size: int
    intermediate_size: int
    num_layers: int
    sparsity_threshold: float = 0.5
    min_active_neurons: int = 128
    max_active_neurons: int | None = None
    window_size: int = 5
    io_threads: int = 32
    cache_budget_neurons: int = 1024


class FlashModelWrapper(nn.Module):
    """Wraps an mlx-lm model for flash inference.

    - Attention layers, embeddings, layernorms, lm_head stay fully in RAM
    - FFN layers are replaced with FlashMLP instances
    - Compatible with mlx_lm.stream_generate() without changes
    """

    def __init__(
        self,
        model: nn.Module,
        predictor_bank: PredictorBank,
        weight_store: FlashWeightStore,
        flash_config: FlashConfig,
    ):
        super().__init__()
        self._model = model
        self.window_manager = WindowManager(
            flash_config.num_layers, flash_config.window_size
        )
        self._replace_mlps(predictor_bank, weight_store, flash_config)

    def _replace_mlps(
        self,
        predictor_bank: PredictorBank,
        weight_store: FlashWeightStore,
        config: FlashConfig,
    ) -> None:
        """Replace each TransformerBlock.mlp with a FlashMLP."""
        layers = self._model.layers
        replaced = 0
        for i, layer in enumerate(layers):
            if not hasattr(layer, "mlp"):
                continue

            flash_mlp = FlashMLP(
                layer_idx=i,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                predictor=predictor_bank.predictors[i],
                weight_store=weight_store,
                window_manager=self.window_manager,
                sparsity_threshold=config.sparsity_threshold,
                min_active_neurons=config.min_active_neurons,
                max_active_neurons=config.max_active_neurons,
            )

            # Free original MLP weights to reclaim RAM
            original_mlp = layer.mlp
            if hasattr(original_mlp, "gate_proj"):
                del original_mlp.gate_proj
            if hasattr(original_mlp, "up_proj"):
                del original_mlp.up_proj
            if hasattr(original_mlp, "down_proj"):
                del original_mlp.down_proj

            layer.mlp = flash_mlp
            replaced += 1

        gc.collect()
        mx.clear_cache()
        logger.info("Replaced %d MLP layers with FlashMLP", replaced)

    def __call__(self, inputs, cache=None, **kwargs):
        """Forward pass — delegates to the wrapped model."""
        return self._model(inputs, cache=cache, **kwargs)

    def __getattr__(self, name):
        """Proxy attributes to the wrapped model."""
        if name.startswith("_") or name in (
            "window_manager",
            "training",
        ):
            raise AttributeError(name)
        return getattr(self._model, name)

    @property
    def layers(self):
        return self._model.layers

    @property
    def args(self):
        return self._model.args
