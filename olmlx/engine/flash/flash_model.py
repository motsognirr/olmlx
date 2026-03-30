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
    memory_budget_fraction: float | None = None


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
        # Store in __dict__ directly so __getattr__ can find it without
        # going through nn.Module's dict (which __getattr__ can't reach
        # for _-prefixed names).
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_sharded", False)
        self.window_manager = WindowManager(
            flash_config.num_layers,
            flash_config.window_size,
            memory_budget_fraction=flash_config.memory_budget_fraction,
            intermediate_size=flash_config.intermediate_size,
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

    def shard(self, group=None):
        """Shard attention layers only — FlashMLP handles its own weights via SSD.

        In distributed mode, attention projections are split across ranks with
        all_sum synchronization, while FlashMLP layers remain unsharded. Each
        rank independently loads active neurons from its local SSD. This is
        correct because o_proj (sharded-to-all) replicates its output via
        all_sum, so every rank feeds identical input to FlashMLP.
        """
        if self._sharded:
            raise RuntimeError(
                "shard() has already been called on this model — "
                "reload the model to retry"
            )
        if group is None:
            raise ValueError("shard() requires an explicit distributed group")
        from mlx.nn.layers.distributed import shard_linear

        N = group.size()

        # Validate all layers before any mutation so a failure doesn't
        # leave the model in a partially-sharded state.
        for i, layer in enumerate(self._model.layers):
            if not hasattr(layer, "self_attn"):
                continue
            attn = layer.self_attn
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if not hasattr(attn, proj):
                    raise ValueError(
                        f"Layer {i}: self_attn has no '{proj}' — "
                        "fused-projection architectures are not supported "
                        "for Flash+distributed"
                    )
            if attn.n_heads % N != 0:
                raise ValueError(
                    f"Layer {i}: n_heads ({attn.n_heads}) is not divisible "
                    f"by world size ({N})"
                )
            if attn.n_kv_heads % N != 0:
                raise ValueError(
                    f"Layer {i}: n_kv_heads ({attn.n_kv_heads}) is not "
                    f"divisible by world size ({N})"
                )

        # Mark sharded before mutation so a mid-loop failure still prevents
        # a second shard() call on a partially-mutated model.
        object.__setattr__(self, "_sharded", True)
        sharded_count = 0
        for layer in self._model.layers:
            if not hasattr(layer, "self_attn"):
                continue
            attn = layer.self_attn
            attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=group)
            attn.k_proj = shard_linear(attn.k_proj, "all-to-sharded", group=group)
            attn.v_proj = shard_linear(attn.v_proj, "all-to-sharded", group=group)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.n_heads //= N
            attn.n_kv_heads //= N
            sharded_count += 1
        logger.info(
            "Sharded %d attention layers (MLP handled by Flash SSD)",
            sharded_count,
        )

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
        model = object.__getattribute__(self, "_model")
        return getattr(model, name)

    def parameters(self):
        """Return combined parameters of inner model and wrapper submodules.

        ``_model`` is stored in ``__dict__`` (bypassing ``nn.Module``'s dict),
        so the base ``nn.Module.parameters()`` cannot reach it.  Merge the
        base class result (which covers ``window_manager`` and any future
        submodules registered via normal assignment) with the inner model's
        parameters so that ``mx.eval(wrapper.parameters())`` materializes
        all weights — including sharded attention projections.
        """
        params = dict(super().parameters())
        params["_model"] = self._model.parameters()
        return params

    def freeze(self, *args, **kwargs):
        """Freeze parameters — delegates to inner model since it is invisible to nn.Module."""
        super().freeze(*args, **kwargs)
        self._model.freeze(*args, **kwargs)

    def trainable(self):
        """Return combined trainable parameters (same blind spot as parameters)."""
        params = dict(super().trainable())
        params["_model"] = self._model.trainable()
        return params

    @property
    def layers(self):
        return self._model.layers

    @property
    def args(self):
        return self._model.args
