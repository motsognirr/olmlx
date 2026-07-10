"""Flash MoE model wrapper.

Wraps an mlx-lm model, replacing MoE layers' SwitchGLU dispatch with
FlashMoE instances that load expert weights on demand from SSD.
Compatible with mlx_lm.stream_generate().
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.flash.flash_moe import FlashMoE
from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

if TYPE_CHECKING:
    from olmlx.engine.flash.moe_prefetch import MoePrefetcher

logger = logging.getLogger(__name__)


class _FlashMoEBase(nn.Module):
    """Base class for Flash-MoE replacement layers.

    Subclasses implement _route() and optionally _combine().
    """

    def __init__(self, original_moe, flash_moe: FlashMoE):
        super().__init__()
        if getattr(original_moe, "sharding_group", None) is not None:
            raise NotImplementedError(
                "Flash-MoE does not support distributed tensor parallelism. "
                "Each rank loads all needed experts, so all_sum would produce "
                "incorrect results. Disable distributed or Flash-MoE."
            )
        self._flash_moe = flash_moe

    def _route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Return (inds, scores) for expert selection."""
        raise NotImplementedError

    def _combine(self, x: mx.array, y: mx.array) -> mx.array:
        """Combine expert output with input. Default: returns expert output unchanged."""
        return y

    def __call__(self, x):
        inds, scores = self._route(x)
        y = self._flash_moe(x, inds, scores).astype(x.dtype)
        return self._combine(x, y)


class _FlashMoEDeepSeek(_FlashMoEBase):
    """Replacement MoE layer for DeepSeek-V3 / Kimi-K2.5 style models.

    Keeps gate and shared_experts in RAM, uses FlashMoE for expert dispatch.
    """

    def __init__(self, original_moe, flash_moe: FlashMoE):
        super().__init__(original_moe, flash_moe)
        self.gate = original_moe.gate
        # Some models (Step-3.5) name the shared expert `share_expert` (singular).
        se = getattr(original_moe, "shared_experts", None)
        if se is None:
            se = getattr(original_moe, "share_expert", None)
        self.shared_experts = se

    def _route(self, x):
        return self.gate(x)

    def _combine(self, x, y):
        if self.shared_experts is not None:
            return y + self.shared_experts(x)
        return y


class _FlashMoEGptOss(_FlashMoEBase):
    """Replacement MoE layer for gpt-oss style models.

    Keeps router in RAM, uses FlashMoE for expert dispatch.
    """

    def __init__(self, original_mlp, flash_moe: FlashMoE):
        super().__init__(original_mlp, flash_moe)
        self.router = original_mlp.router
        self.num_experts_per_tok = original_mlp.num_experts_per_tok

    def _route(self, x):
        g = self.router(x)
        k = self.num_experts_per_tok
        # topk
        part_inds = mx.argpartition(g, kth=-k, axis=-1)
        inds = part_inds[..., -k:]
        scores = mx.take_along_axis(g, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)  # pyright: ignore[reportCallIssue]
        return inds, scores


class _FlashMoEQwen3Next(_FlashMoEBase):
    """Replacement MoE layer for Qwen3-Next style models.

    Gate is a plain nn.Linear (returns logits, not (inds, scores)).
    Keeps gate, shared_expert, and shared_expert_gate in RAM.
    """

    def __init__(self, original_moe, flash_moe: FlashMoE):
        super().__init__(original_moe, flash_moe)
        self.gate = original_moe.gate
        self.top_k = original_moe.top_k
        self.norm_topk_prob = original_moe.norm_topk_prob
        self.shared_expert = original_moe.shared_expert
        self.shared_expert_gate = original_moe.shared_expert_gate

    def _route(self, x):
        scores = mx.softmax(self.gate(x).astype(mx.float32), axis=-1)
        k = self.top_k
        inds = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return inds, scores

    def _combine(self, x, y):
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y


class _FlashMoEQwen3(_FlashMoEBase):
    """Replacement MoE layer for plain Qwen3-MoE style models.

    Covers Qwen3MoeSparseMoeBlock (e.g. Qwen3-235B-A22B): a plain nn.Linear
    gate (returns logits, not (inds, scores)), no shared experts, and no
    e_score_correction_bias. Routing mirrors mlx-lm's Qwen3MoeSparseMoeBlock:
    softmax over all experts, top-k, optional norm_topk_prob renormalization.
    """

    def __init__(self, original_moe, flash_moe: FlashMoE):
        super().__init__(original_moe, flash_moe)
        self.gate = original_moe.gate
        self.top_k = original_moe.top_k
        self.norm_topk_prob = original_moe.norm_topk_prob

    def _route(self, x):
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)  # pyright: ignore[reportCallIssue]
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        return inds, scores


class _FlashMoEMiniMax(_FlashMoEBase):
    """Replacement MoE layer for MiniMax-style models.

    Gate is nn.Linear returning logits; uses sigmoid scoring with
    e_score_correction_bias for expert selection.
    """

    def __init__(self, original_moe, flash_moe: FlashMoE):
        super().__init__(original_moe, flash_moe)
        self.gate = original_moe.gate
        self.num_experts_per_tok = original_moe.num_experts_per_tok
        self.e_score_correction_bias = original_moe.e_score_correction_bias
        self.shared_experts = getattr(original_moe, "shared_experts", None)

    def _route(self, x):
        gates = self.gate(x.astype(mx.float32))
        scores = mx.sigmoid(gates)
        orig_scores = scores
        scores = scores + self.e_score_correction_bias

        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = scores.astype(x.dtype)
        return inds, scores

    def _combine(self, x, y):
        if self.shared_experts is not None:
            return y + self.shared_experts(x)
        return y


class _FlashMoEGemma4(nn.Module):
    """Replacement experts module for Gemma4 VLM-style layers.

    In Gemma4 VLM, router and experts are separate attributes on the decoder
    layer.  The router stays untouched; this replaces ``layer.experts`` and
    accepts pre-routed ``(x, top_k_indices, top_k_weights)`` directly.
    """

    def __init__(self, flash_moe: FlashMoE):
        super().__init__()
        self._flash_moe = flash_moe

    def __call__(self, x, top_k_indices, top_k_weights):
        y = self._flash_moe(x, top_k_indices, top_k_weights)
        return y.astype(x.dtype)


class FlashMoeModelWrapper(nn.Module):
    """Wraps an mlx-lm model for Flash-MoE inference.

    - Router (gate), shared experts, attention, embeddings stay in RAM
    - SwitchGLU expert weights are replaced with FlashMoE (SSD-loaded)
    - Compatible with mlx_lm.stream_generate() without changes
    """

    def __init__(
        self,
        model: nn.Module,
        weight_store: FlashMoeWeightStore,
        moe_layer_indices: list[int],
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        prefetcher: MoePrefetcher | None = None,
    ):
        super().__init__()
        self._model = model
        self._weight_store = weight_store
        # Surfaced for ModelManager._close_model_resources, which closes
        # ``lm.model.prefetcher`` BEFORE the weight store (prefetch tasks
        # submit into the store's I/O pool). Mirrors FlashModelWrapper.
        self.prefetcher = prefetcher
        _replace_moe_layers(
            model,
            weight_store,
            moe_layer_indices,
            hidden_size,
            intermediate_size,
            num_experts,
            num_experts_per_tok,
            prefetcher=prefetcher,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Any | None = None,
        **kwargs: Any,
    ) -> mx.array:
        """Forward pass — delegates to the wrapped model."""
        return self._model(inputs, cache=cache, **kwargs)

    def __getattr__(self, name):
        """Proxy non-private attributes to the wrapped model."""
        if name.startswith("_") or name in ("training",):
            return super().__getattr__(name)
        return getattr(self._model, name)

    @property
    def layers(self):
        return self._model.layers

    @property
    def args(self):
        return self._model.args


def _maybe_create_prefetcher(
    flash_moe_dir: Path,
    moe_config: dict[str, Any],
    store: FlashMoeWeightStore,
    *,
    margin: float,
    max_positions: int,
    scored_eviction: bool,
    min_recall: float = 0.0,
) -> MoePrefetcher | None:
    """Load the trained lookahead bank and build a prefetcher, or None.

    Never raises: a missing/corrupt/stale ``moe_lookahead/`` directory is an
    optional accelerator, not a load failure. A sidecar that disagrees with
    the bundle (re-bundled model, wrong architecture) is rejected — serving a
    wrong-shaped predictor would prefetch garbage every token. Pairs whose
    trained holdout recall is below *min_recall* are gated off (their
    predictions are mostly wasted SSD reads); if that gates every pair, no
    prefetcher is built at all.
    """
    from olmlx.engine.flash.moe_predictor import MoeLookaheadBank
    from olmlx.engine.flash.moe_prefetch import MoePrefetcher

    lookahead_dir = flash_moe_dir / "moe_lookahead"
    if not lookahead_dir.exists():
        logger.debug("No moe_lookahead/ in %s — prefetch disabled", flash_moe_dir)
        return None
    try:
        bank = MoeLookaheadBank.load(lookahead_dir)
    except Exception:
        logger.warning(
            "Failed to load MoE lookahead bank from %s — prefetch disabled",
            lookahead_dir,
            exc_info=True,
        )
        return None

    expected = {
        "hidden_size": moe_config["hidden_size"],
        "num_experts": moe_config["num_experts"],
        "moe_layer_indices": sorted(moe_config["moe_layer_indices"]),
    }
    actual = {
        "hidden_size": bank.hidden_size,
        "num_experts": bank.num_experts,
        "moe_layer_indices": bank.moe_layer_indices,
    }
    if expected != actual:
        logger.warning(
            "MoE lookahead bank at %s does not match the bundle "
            "(expected %s, got %s) — prefetch disabled; retrain with "
            "`olmlx flash train-moe-lookahead`",
            lookahead_dir,
            expected,
            actual,
        )
        return None

    gated = bank.apply_recall_gate(min_recall)
    if gated:
        logger.info(
            "Recall gate (min_recall=%.2f): disabled %d of %d lookahead pairs",
            min_recall,
            gated,
            len(bank.heads),
        )
    if not bank.trained_pairs:
        logger.info(
            "All lookahead pairs gated below min_recall=%.2f — prefetch disabled",
            min_recall,
        )
        return None

    logger.info(
        "MoE expert prefetch enabled (margin=%.2f, max_positions=%d, "
        "scored_eviction=%s, active_pairs=%d/%d)",
        margin,
        max_positions,
        scored_eviction,
        len(bank.trained_pairs),
        len(bank.heads),
    )
    return MoePrefetcher(
        bank,
        store,
        margin=margin,
        max_positions=max_positions,
        scored_eviction=scored_eviction,
    )


def wrap_flash_moe(
    model: Any,
    flash_moe_dir: Path | str,
    *,
    io_threads: int,
    cache_budget_experts: int,
    prefetch: bool = False,
    lookahead_margin: float = 1.5,
    prefetch_max_positions: int = 8,
    scored_eviction: bool = True,
    prefetch_min_recall: float = 0.0,
) -> tuple[Any, Any]:
    """Wrap a lazily-loaded MoE model with SSD-streamed experts.

    The kernel shared by the serving loader
    (``ModelManager._load_flash_moe_model``) and the offline calibration
    loader (``load_flash_moe_model``): read the bundle config, open the
    weight store, replace MoE layers with streaming replacements, and eval
    only the non-expert params. Closes the store and re-raises if wrapping
    fails. Returns ``(wrapped_model, store)``; the caller owns the store.
    """
    import json

    # Call-time import so tests can patch the store on its home module.
    from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

    flash_moe_dir = Path(flash_moe_dir)
    moe_config = json.loads((flash_moe_dir / "flash_moe_config.json").read_text())

    store = FlashMoeWeightStore(
        flash_moe_dir,
        num_io_threads=io_threads,
        cache_budget_experts=cache_budget_experts,
    )
    prefetcher = None
    if prefetch:
        prefetcher = _maybe_create_prefetcher(
            flash_moe_dir,
            moe_config,
            store,
            margin=lookahead_margin,
            max_positions=prefetch_max_positions,
            scored_eviction=scored_eviction,
            min_recall=prefetch_min_recall,
        )
    try:
        wrapped = FlashMoeModelWrapper(
            model,
            store,
            moe_layer_indices=moe_config["moe_layer_indices"],
            hidden_size=moe_config["hidden_size"],
            intermediate_size=moe_config["intermediate_size"],
            num_experts=moe_config["num_experts"],
            num_experts_per_tok=moe_config["num_experts_per_tok"],
            prefetcher=prefetcher,
        )
        # Materialize only non-expert weights.
        mx.eval(wrapped.parameters())
    except Exception:
        if prefetcher is not None:
            prefetcher.close()
        store.close()
        raise
    return wrapped, store


def load_flash_moe_model(
    load_path: str,
    flash_moe_dir: Path | str,
    *,
    cache_budget_experts: int,
    io_threads: int,
) -> tuple[Any, Any, Any]:
    """Load a model in Flash-MoE mode for offline use (e.g. KV-quant calibration).

    Lazy-load so routed expert weights are never materialized, then
    ``wrap_flash_moe`` — the same kernel the serving loader uses. Returns
    ``(wrapped_model, tokenizer, store)``; the caller owns the store and must
    close it. Raises if the bundle is present but cannot be loaded.
    """
    from olmlx.engine.flash.prepare import load_model_with_strict_fallback

    try:
        model, tokenizer = load_model_with_strict_fallback(load_path, lazy=True)
    except ValueError:
        # VLM-shaped architectures mlx-lm rejects (Gemma4-class, Kimi-K2.5)
        # are served via the same fallback in the manager's Flash-MoE loader;
        # without it here, bundle-bearing VLMs could not be calibrated.
        from olmlx.engine.vlm_load import load_vlm

        model, processor = load_vlm(load_path, lazy=True)
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
    wrapped, store = wrap_flash_moe(
        model,
        flash_moe_dir,
        io_threads=io_threads,
        cache_budget_experts=cache_budget_experts,
    )
    return wrapped, tokenizer, store


def _find_moe_module(layer: nn.Module) -> tuple[str, nn.Module]:
    """Find the MoE module on a decoder layer.

    Returns (attr_name, module) — the attribute may be 'experts' (Gemma4 VLM,
    where router + experts live directly on the layer), 'mlp' (DeepSeek,
    Qwen3-Next, gpt-oss), or 'block_sparse_moe' (MiniMax).
    """
    # Gemma4 VLM: router + experts are on the layer, mlp is a separate dense path
    experts = getattr(layer, "experts", None)
    if (
        experts is not None
        and getattr(layer, "router", None) is not None
        and getattr(experts, "switch_glu", None) is not None
    ):
        return "experts", experts

    for attr in ("mlp", "block_sparse_moe", "mixer"):
        mod = getattr(layer, attr, None)
        if mod is not None:
            return attr, mod
    raise AttributeError(
        f"Layer {layer} has no 'mlp', 'block_sparse_moe', or 'mixer' attribute"
    )


def _replace_moe_layers(
    model: nn.Module,
    weight_store: FlashMoeWeightStore,
    moe_layer_indices: list[int],
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    prefetcher: MoePrefetcher | None = None,
) -> None:
    """Replace MoE layers with FlashMoE variants."""
    layers = model.layers
    replaced = 0

    for layer_idx in moe_layer_indices:
        layer = layers[layer_idx]
        moe_attr, moe_module = _find_moe_module(layer)

        # Extract activation function from the original SwitchGLU before we delete it
        activation = None
        # For Gemma4 VLM, moe_module is the Experts wrapper itself
        switch_glu = getattr(moe_module, "switch_glu", None)
        if switch_glu is not None and hasattr(switch_glu, "activation"):
            activation = switch_glu.activation
        else:
            for attr in ("switch_mlp", "experts"):
                switch = getattr(moe_module, attr, None)
                if switch is not None and hasattr(switch, "activation"):
                    activation = switch.activation
                    break

        # Create FlashMoE for this layer
        flash_moe = FlashMoE(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            weight_store=weight_store,
            activation=activation,
            prefetcher=prefetcher,
        )

        if moe_attr == "experts":
            # Gemma4 VLM: router stays on layer, only replace experts
            if getattr(moe_module, "sharding_group", None) is not None:
                raise NotImplementedError(
                    "Flash-MoE does not support distributed tensor parallelism. "
                    "Each rank loads all needed experts, so all_sum would produce "
                    "incorrect results. Disable distributed or Flash-MoE."
                )
            replacement = _FlashMoEGemma4(flash_moe)
            # Free SwitchGLU weights from the original Experts module
            if switch_glu is not None:
                for proj in ("gate_proj", "up_proj", "down_proj", "fc1", "fc2"):
                    if hasattr(switch_glu, proj):
                        delattr(switch_glu, proj)
                delattr(moe_module, "switch_glu")
        else:
            # Detect router style and create appropriate replacement.
            # Structural checks use gate type: nn.Linear (or QuantizedLinear)
            # returns logits; custom gate modules return (inds, scores) directly.
            gate = getattr(moe_module, "gate", None)
            gate_is_linear = isinstance(gate, (nn.Linear,)) or (
                hasattr(nn, "QuantizedLinear") and isinstance(gate, nn.QuantizedLinear)
            )
            if hasattr(moe_module, "shared_expert_gate") and gate_is_linear:
                # Qwen3-Next style: linear gate + shared_expert + shared_expert_gate
                replacement = _FlashMoEQwen3Next(moe_module, flash_moe)
            elif gate_is_linear and hasattr(moe_module, "e_score_correction_bias"):
                # MiniMax style: linear gate with sigmoid scoring + correction bias
                replacement = _FlashMoEMiniMax(moe_module, flash_moe)
            elif gate_is_linear:
                # Plain Qwen3-MoE style (e.g. Qwen3-235B-A22B): linear gate ->
                # softmax -> top-k. Must come before the DeepSeek branch — a
                # plain linear gate returns logits, never (inds, scores).
                replacement = _FlashMoEQwen3(moe_module, flash_moe)
            elif gate is not None:
                # DeepSeek-V3 / Kimi-K2.5 style: custom gate returns (inds, scores)
                replacement = _FlashMoEDeepSeek(moe_module, flash_moe)
            else:
                # gpt-oss style (has router + experts)
                replacement = _FlashMoEGptOss(moe_module, flash_moe)

            # Delete original SwitchGLU/expert weights before replacing
            for attr in ("switch_mlp", "experts"):
                if hasattr(moe_module, attr):
                    switch = getattr(moe_module, attr)
                    for proj in (
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                        "fc1",
                        "fc2",
                    ):
                        if hasattr(switch, proj):
                            delattr(switch, proj)
                    delattr(moe_module, attr)
                    break

        # Replace the MoE module on the layer
        setattr(layer, moe_attr, replacement)
        replaced += 1

    gc.collect()
    mx.clear_cache()
    logger.info("Replaced %d MoE layers with FlashMoE", replaced)
