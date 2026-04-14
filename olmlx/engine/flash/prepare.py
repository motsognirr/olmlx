"""Preparation pipeline for flash inference.

Prepares a model for LLM-in-a-Flash inference by:
1. Streaming calibration data through the model one layer at a time
2. Recording FFN activation patterns
3. Training per-layer sparsity predictors
4. Bundling FFN weights with row-column bundling
"""

from __future__ import annotations

import gc
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from mlx.utils import tree_map

from olmlx.engine.flash.bundler import bundle_ffn_weights
from olmlx.engine.flash.predictor import (
    LookaheadBank,
    PredictorBank,
    compute_layer_ranks,
)

logger = logging.getLogger(__name__)

# Error substring used by mlx-lm when safetensors contain keys not in the model.
# This is a fragile contract with mlx-lm internals — if the error wording changes,
# VL model loading will fall through to the mlx_vlm fallback instead of using
# strict=False, which still works but loads as a vision model unnecessarily.
_STRICT_LOAD_ERROR = "parameters not in model"


def _get_backbone(model: nn.Module) -> Any:
    """Navigate to the transformer backbone that has .layers and .embed_tokens.

    Handles both standard models (Model.model = backbone) and VL models
    (Model.language_model.model = backbone).
    """
    inner = model.model if hasattr(model, "model") else model
    lm = getattr(inner, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", lm)
    return inner


def load_model_with_strict_fallback(model_path: str, *, lazy: bool) -> tuple:
    """Load model via mlx-lm, retrying with strict=False for VL models.

    VL models (e.g. Qwen3.5) ship vision tower weights in safetensors that the
    text-only model class doesn't use. When mlx-lm raises ValueError containing
    "parameters not in model", retries with strict=False.

    Returns (model, tokenizer).
    """
    import mlx_lm

    try:
        return mlx_lm.load(model_path, lazy=lazy)
    except ValueError as exc:
        if _STRICT_LOAD_ERROR not in str(exc):
            raise
        logger.info(
            "Retrying with strict=False (extra weights in safetensors): %s", exc
        )
        model_dir = Path(model_path)
        model, config = mlx_lm.utils.load_model(model_dir, lazy=lazy, strict=False)
        # config may be a dict or a dataclass depending on mlx-lm version
        eos = (
            config.get("eos_token_id")
            if isinstance(config, dict)
            else getattr(config, "eos_token_id", None)
        )
        # None is intentionally passed through to let mlx-lm use the tokenizer default
        tokenizer = mlx_lm.utils.load_tokenizer(
            model_dir, eos_token_ids=[eos] if isinstance(eos, int) else eos
        )
        return model, tokenizer


def _encode_tokens(tokenizer, text: str) -> list[int]:
    """Encode text to token ids, handling both fast and slow tokenizers."""
    result = tokenizer.encode(text)
    tokens = result["input_ids"] if isinstance(result, dict) else result
    if len(tokens) > 512:
        tokens = tokens[:512]
    return tokens


def _create_causal_mask(h: mx.array) -> mx.array | None:
    """Create a causal attention mask for hidden states of shape (B, L, D)."""
    L = h.shape[1]
    if L <= 1:
        return None
    indices = mx.arange(L)
    mask = indices[:, None] < indices[None, :]
    return mask * -1e9


def _nullify_module_params(module: nn.Module) -> None:
    """Replace all parameter arrays with tiny placeholders to free VRAM."""
    new_params = tree_map(
        lambda x: mx.zeros((1,)) if isinstance(x, mx.array) else x,
        module.parameters(),
    )
    module.update(new_params)


def _get_c4_calibration_data(num_samples: int = 10000) -> list[str]:
    """Load calibration data from C4 dataset via HuggingFace datasets.

    Falls back to synthetic data if the datasets library is not installed.
    """
    try:
        import datasets as _ds_mod

        load_dataset = _ds_mod.load_dataset
    except ImportError:
        actual = min(num_samples, 256)
        logger.warning(
            "HuggingFace datasets not installed; "
            "falling back to %d synthetic calibration samples (requested %d). "
            "Install with: pip install datasets",
            actual,
            num_samples,
        )
        return _get_calibration_data(actual)

    try:
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        ds_iter = iter(ds)
        texts: list[str] = []
        try:
            for example in ds_iter:
                text = example.get("text", "")
                if len(text) < 100:
                    continue
                texts.append(text[:2048])
                if len(texts) >= num_samples:
                    break
        finally:
            # Close the streaming iterator to release HTTP connections
            if hasattr(ds_iter, "close"):
                ds_iter.close()
    except Exception as exc:
        actual = min(num_samples, 256)
        logger.warning(
            "Failed to stream C4 dataset (%s: %s); "
            "falling back to %d synthetic calibration samples",
            type(exc).__name__,
            exc,
            actual,
        )
        return _get_calibration_data(actual)

    if len(texts) < num_samples:
        logger.warning("Only got %d C4 samples (requested %d)", len(texts), num_samples)
    return texts


def _get_calibration_data(num_samples: int = 256) -> list[str]:
    """Generate simple calibration prompts."""
    # Simple diverse prompts for calibration
    templates = [
        "Explain the concept of {} in simple terms.",
        "What are the main differences between {} and {}?",
        "Write a short paragraph about {}.",
        "List three important facts about {}.",
        "How does {} work in practice?",
        "Describe the history of {}.",
        "What are the advantages of {}?",
        "Compare and contrast {} with {}.",
    ]
    topics = [
        "machine learning",
        "quantum computing",
        "climate change",
        "renewable energy",
        "artificial intelligence",
        "blockchain technology",
        "space exploration",
        "genetics",
        "economics",
        "philosophy",
        "mathematics",
        "computer science",
        "biology",
        "chemistry",
        "physics",
        "literature",
        "music theory",
        "architecture",
        "psychology",
        "sociology",
        "neuroscience",
        "robotics",
        "cryptography",
        "data science",
        "cloud computing",
        "cybersecurity",
        "nanotechnology",
        "biotechnology",
        "oceanography",
        "meteorology",
        "astronomy",
        "geology",
    ]

    samples = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        topic1 = topics[i % len(topics)]
        topic2 = topics[(i + 7) % len(topics)]
        if "{}" in template:
            count = template.count("{}")
            if count == 2:
                samples.append(template.format(topic1, topic2))
            else:
                samples.append(template.format(topic1))
        else:
            samples.append(template)

    return samples


def _record_activations(
    model: nn.Module,
    tokenizer,
    calibration_texts: list[str],
    activation_threshold: float = 0.01,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[int, tuple[list[mx.array], list[mx.array]]]:
    """Run calibration data through model and record FFN activations.

    Returns dict mapping layer_idx -> (inputs_list, targets_list).
    """

    layers = model.layers
    num_layers = len(layers)

    # Storage: layer_idx -> (list of input arrays, list of target mask arrays)
    recordings: dict[int, tuple[list[mx.array], list[mx.array]]] = {
        i: ([], []) for i in range(num_layers)
    }

    # Instance __call__ patching doesn't work (Python resolves __call__
    # on the type, not instance), so we replace layer.mlp entirely.
    original_mlps = {}
    for i, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue
        original_mlps[i] = layer.mlp

    for i in original_mlps:
        layers[i].mlp = _RecordingMLP(
            original_mlps[i], recordings[i], activation_threshold
        )

    try:
        total = len(calibration_texts)
        for idx, text in enumerate(calibration_texts):
            if progress_callback:
                progress_callback("Recording activations", (idx + 1) / total)

            tokens = _encode_tokens(tokenizer, text)
            input_ids = mx.array([tokens])

            try:
                model(input_ids)
            except (ValueError, RuntimeError, TypeError) as exc:
                logger.debug("Skipping sample %d (model error): %s", idx, exc)
    finally:
        for i, mlp in original_mlps.items():
            layers[i].mlp = mlp

    return recordings


class _RecordingMLP(nn.Module):
    """Wrapper that records FFN activation patterns during forward pass.

    Delegates to the original MLP for the forward result (preserving any
    custom gating or normalization), then separately computes gate/up
    projections for activation recording.

    Thread-safe within a single layer: uses a per-instance lock to prevent
    concurrent interception of down_proj. Note: mx.eval is called while holding
    the lock, and concurrent mx.eval calls across threads can deadlock.
    Different layers have separate locks, but parallel mlx.eval across layers
    is unsafe. Current single-threaded calibration pipeline avoids this issue.
    """

    def __init__(
        self,
        original: nn.Module,
        recordings: tuple[list[mx.array], list[mx.array]],
        activation_threshold: float,
    ):
        super().__init__()
        self._original = original
        self._recordings = recordings
        self._threshold = activation_threshold
        self._lock = threading.Lock()

    def __call__(self, x):
        orig = self._original
        if not (hasattr(orig, "gate_proj") and hasattr(orig, "up_proj")):
            return orig(x)

        with self._lock:
            captured = {}
            real_down_proj = orig.down_proj

            def intercepting_down_proj(activated):
                captured["activated"] = activated
                return real_down_proj(activated)

            orig.down_proj = intercepting_down_proj
            try:
                result = orig(x)
            finally:
                orig.down_proj = real_down_proj

        if "activated" not in captured:
            return result
        flat_input = x.reshape(-1, x.shape[-1])
        activated = captured["activated"]
        flat_activated = activated.reshape(-1, activated.shape[-1])
        mean_act = mx.abs(flat_activated).mean(axis=0)
        target = (mean_act > self._threshold).astype(mx.float32)
        mean_input = flat_input.mean(axis=0)

        mx.eval(mean_input, target)
        self._recordings[0].append(mean_input)
        self._recordings[1].append(target)

        return result


def _stream_record_activations(
    model_path: str,
    calibration_texts: list[str],
    activation_threshold: float = 0.01,
    progress_callback: Callable[[str, float], None] | None = None,
) -> tuple[dict[int, tuple[list[mx.array], list[mx.array]]], int, int, int]:
    """Stream calibration through model one layer at a time.

    Loads only one layer's weights into VRAM at a time, enabling preparation
    of models larger than available RAM.

    Note: All num_samples hidden states are materialized before the layer loop.
    For a 70B model (hidden_size=8192) with 256 samples x 512 tokens x fp16,
    this is ~2 GB. Memory scales linearly with num_samples.

    Returns:
        (recordings, hidden_size, intermediate_size, num_layers)
    """
    if progress_callback:
        progress_callback("Loading model skeleton", 0.0)

    try:
        model, tokenizer = load_model_with_strict_fallback(model_path, lazy=True)
    except ValueError:
        # mlx_lm doesn't support this model type (e.g. gemma4 VLM) —
        # fall back to mlx_vlm and extract the language model.
        import mlx_vlm

        vlm_model, processor = mlx_vlm.load(model_path, lazy=True)
        model = vlm_model.language_model
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

    inner = _get_backbone(model)
    layers = inner.layers
    num_layers = len(layers)

    # Find dimensions from first layer with gate_proj/up_proj (some MoE models
    # have dense layers at the start that lack these projections)
    hidden_size = intermediate_size = None
    for layer in layers:
        mlp = layer.mlp if hasattr(layer, "mlp") else None
        if mlp and hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
            gate = mlp.gate_proj
            intermediate_size = gate.weight.shape[0]
            # For quantized models, weight is packed — derive real dim from scales
            if hasattr(gate, "scales") and hasattr(gate, "group_size"):
                hidden_size = gate.scales.shape[1] * gate.group_size
            else:
                hidden_size = gate.weight.shape[1]
            break
    if hidden_size is None or intermediate_size is None:
        raise ValueError("No layer has gate_proj/up_proj — cannot determine dimensions")

    if progress_callback:
        progress_callback("Computing embeddings", 0.02)

    embed = (
        getattr(inner, "embed_tokens", None)
        or getattr(inner, "wte", None)
        or getattr(inner, "tok_embeddings", None)
    )
    if embed is None:
        raise ValueError(
            "Cannot find embedding layer (tried embed_tokens, wte, tok_embeddings)"
        )

    mx.eval(embed.parameters())

    hidden_states: list[mx.array | None] = []
    for text in calibration_texts:
        tokens = _encode_tokens(tokenizer, text)
        input_ids = mx.array([tokens])
        h = embed(input_ids)
        mx.eval(h)
        hidden_states.append(h)

    _nullify_module_params(embed)
    # Free LM head and final norm — not needed for layer-by-layer streaming
    for attr in ("lm_head", "norm", "output"):
        submod = getattr(inner, attr, None)
        if submod is None:
            submod = getattr(model, attr, None)
        if submod is not None and isinstance(submod, nn.Module):
            _nullify_module_params(submod)
    gc.collect()
    mx.clear_cache()

    if progress_callback:
        progress_callback("Streaming layers", 0.05)

    recordings: dict[int, tuple[list[mx.array], list[mx.array]]] = {}

    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        mx.eval(layer.parameters())

        original_mlp = layer.mlp
        recording_data: tuple[list[mx.array], list[mx.array]] = ([], [])

        if hasattr(original_mlp, "gate_proj") and hasattr(original_mlp, "up_proj"):
            layer.mlp = _RecordingMLP(
                original_mlp, recording_data, activation_threshold
            )

        for i in range(len(hidden_states)):
            if hidden_states[i] is None:
                continue
            mask = _create_causal_mask(hidden_states[i])
            try:
                out = layer(hidden_states[i], mask=mask, cache=None)
                hidden_states[i] = out[0] if isinstance(out, (tuple, list)) else out
                mx.eval(hidden_states[i])
            except (ValueError, RuntimeError, TypeError) as exc:
                logger.debug("Skipping sample %d at layer %d: %s", i, layer_idx, exc)
                hidden_states[i] = None  # free stale tensor
        recordings[layer_idx] = recording_data

        if not recording_data[0]:
            logger.warning("No activations recorded for layer %d", layer_idx)

        layer.mlp = original_mlp
        _nullify_module_params(layer)
        gc.collect()
        mx.clear_cache()

        if progress_callback:
            frac = 0.05 + ((layer_idx + 1) / num_layers) * 0.95
            progress_callback(f"Processed layer {layer_idx + 1}/{num_layers}", frac)

    return recordings, hidden_size, intermediate_size, num_layers


def _train_single_predictor(
    pred: nn.Module,
    inputs: mx.array,
    targets: mx.array,
    epochs: int,
    lr: float,
    pos_weight_multiplier: float | None = 1.0,
    epoch_callback: Callable[[int], None] | None = None,
) -> None:
    """Train a single predictor with balanced BCE loss.

    Args:
        pos_weight_multiplier: Extra scaling on pos_weight for recall bias
            (e.g. 2.0 for lookahead predictors). ``None`` disables class
            balancing entirely (pos_w = neg_w = 1.0).
        epoch_callback: Called after each epoch with the epoch index (0-based).
    """
    from mlx.optimizers import Adam

    optimizer = Adam(learning_rate=lr)

    if pos_weight_multiplier is not None:
        eps_w = 1e-7
        num_pos = float(mx.sum(targets).item()) + eps_w
        num_neg = float(mx.sum(1 - targets).item()) + eps_w
        pos_w = min(num_neg / num_pos, 1000.0) * pos_weight_multiplier
    else:
        pos_w = 1.0
    neg_w = 1.0

    def loss_fn(model, x, y):
        scores = model(x)
        eps = 1e-7
        return -mx.mean(
            pos_w * y * mx.log(scores + eps)
            + neg_w * (1 - y) * mx.log(1 - scores + eps)
        )

    loss_and_grad = nn.value_and_grad(pred, loss_fn)

    for epoch in range(epochs):
        loss, grads = loss_and_grad(pred, inputs, targets)
        optimizer.update(pred, grads)
        mx.eval(pred.parameters(), optimizer.state)
        if epoch_callback is not None:
            epoch_callback(epoch)


def _train_predictors(
    recordings: dict[int, tuple[list[mx.array], list[mx.array]]],
    hidden_size: int,
    intermediate_size: int,
    rank: int = 128,
    ranks: list[int] | None = None,
    epochs: int = 5,
    lr: float = 1e-3,
    balanced_loss: bool = True,
    progress_callback: Callable[[str, float], None] | None = None,
) -> PredictorBank:
    """Train per-layer sparsity predictors from recorded activations."""
    num_layers = len(recordings)
    bank = PredictorBank(num_layers, hidden_size, intermediate_size, rank, ranks=ranks)

    total_steps = num_layers * epochs
    step = 0

    for layer_idx in sorted(recordings.keys()):
        inputs_list, targets_list = recordings[layer_idx]
        if not inputs_list:
            logger.warning("No recordings for layer %d, skipping", layer_idx)
            step += epochs
            if progress_callback:
                progress_callback(f"Skipped layer {layer_idx}", step / total_steps)
            continue

        inputs = mx.stack(inputs_list)
        targets = mx.stack(targets_list)

        def _on_epoch(epoch: int, _li=layer_idx) -> None:
            nonlocal step
            step += 1
            if progress_callback:
                progress_callback(
                    f"Training layer {_li} epoch {epoch + 1}/{epochs}",
                    step / total_steps,
                )

        _train_single_predictor(
            bank.predictors[layer_idx],
            inputs,
            targets,
            epochs=epochs,
            lr=lr,
            pos_weight_multiplier=1.0 if balanced_loss else None,
            epoch_callback=_on_epoch,
        )

    return bank


def _train_lookahead_predictors(
    recordings: dict[int, tuple[list[mx.array], list[mx.array]]],
    hidden_size: int,
    intermediate_size: int,
    rank: int = 64,
    epochs: int = 5,
    lr: float = 1e-3,
    progress_callback: Callable[[str, float], None] | None = None,
) -> LookaheadBank | None:
    """Train cross-layer lookahead predictors (input_L → target_{L+1}).

    Returns None if insufficient recordings for cross-layer pairs.
    """
    num_layers = len(recordings)
    if num_layers < 2:
        logger.warning("Need at least 2 layers for lookahead predictors")
        return None

    bank = LookaheadBank(num_layers, hidden_size, intermediate_size, rank)

    total_steps = (num_layers - 1) * epochs
    step = 0

    for layer_idx in range(num_layers - 1):
        inputs_list = recordings[layer_idx][0]
        next_layer = layer_idx + 1
        targets_list = recordings[next_layer][1] if next_layer in recordings else []

        if not inputs_list or not targets_list:
            logger.warning(
                "No cross-layer pair for layers %d→%d, skipping",
                layer_idx,
                next_layer,
            )
            step += epochs
            if progress_callback:
                progress_callback(
                    f"Skipped lookahead {layer_idx}→{next_layer}",
                    step / total_steps,
                )
            continue

        n = min(len(inputs_list), len(targets_list))
        if n < max(len(inputs_list), len(targets_list)):
            logger.warning(
                "Recording count mismatch for lookahead %d→%d: %d inputs, %d targets; "
                "using %d samples",
                layer_idx,
                next_layer,
                len(inputs_list),
                len(targets_list),
                n,
            )
        inputs = mx.stack(inputs_list[:n])
        targets = mx.stack(targets_list[:n])

        def _on_epoch(epoch: int, _li=layer_idx, _nl=next_layer) -> None:
            nonlocal step
            step += 1
            if progress_callback:
                progress_callback(
                    f"Training lookahead {_li}→{_nl} epoch {epoch + 1}/{epochs}",
                    step / total_steps,
                )

        # 2x recall boost: false negatives cost latency, false positives
        # only waste a cache slot.
        _train_single_predictor(
            bank.predictors[layer_idx],
            inputs,
            targets,
            epochs=epochs,
            lr=lr,
            pos_weight_multiplier=2.0,
            epoch_callback=_on_epoch,
        )

    return bank


def prepare_model_for_flash(
    model_path: str,
    output_dir: Path | None = None,
    rank: int = 128,
    sensitive_layers: int = 0,
    sensitive_rank_multiplier: int = 4,
    num_samples: int = 256,
    calibration_dataset: str | None = None,
    activation_threshold: float = 0.01,
    epochs: int = 5,
    lookahead_rank: int = 64,
    train_lookahead: bool = False,
    progress_callback: Callable[[str, float], None] | None = None,
) -> Path:
    """Full preparation pipeline for flash inference.

    Streams calibration data through the model one layer at a time,
    so the full model never needs to fit in RAM.

    Steps:
    1. Stream calibration data layer-by-layer (one layer in RAM at a time)
    2. Train per-layer sparsity predictors
    3. Bundle FFN weights with row-column bundling
    4. Save predictor bank + config

    Args:
        model_path: HF model path or local directory.
        output_dir: Where to write flash files. Defaults to model_dir/flash.
        rank: Predictor rank (lower = smaller but less accurate).
        num_samples: Number of calibration samples.
        activation_threshold: Threshold for considering a neuron "active".
        epochs: Training epochs for predictors.
        progress_callback: Called with (description, progress_fraction).

    Returns:
        Path to the flash directory.
    """
    if output_dir is None:
        output_dir = Path(model_path) / "flash"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate calibration data
    if calibration_dataset == "synthetic":
        calibration_texts = _get_calibration_data(num_samples)
    else:
        # Default to C4 (also handles calibration_dataset="c4" or None)
        calibration_texts = _get_c4_calibration_data(num_samples)

    # Step 2: Stream-record activations (one layer at a time)
    recordings, hidden_size, intermediate_size, num_layers = _stream_record_activations(
        model_path,
        calibration_texts,
        activation_threshold=activation_threshold,
        progress_callback=lambda desc, frac: (
            progress_callback(desc, frac * 0.4) if progress_callback else None
        ),
    )

    # Step 3: Train predictors
    if progress_callback:
        progress_callback("Training predictors", 0.4)

    # Compute per-layer ranks if sensitive_layers is set
    layer_ranks = None
    if sensitive_layers > 0:
        layer_ranks = compute_layer_ranks(
            num_layers, rank, sensitive_layers, sensitive_rank_multiplier
        )

    predictor_bank = _train_predictors(
        recordings,
        hidden_size,
        intermediate_size,
        rank=rank,
        ranks=layer_ranks,
        epochs=epochs,
        progress_callback=lambda desc, frac: (
            progress_callback(desc, 0.4 + frac * 0.15) if progress_callback else None
        ),
    )

    # Step 3b: Train lookahead predictors (cross-layer)
    lookahead_bank = None
    if train_lookahead and num_layers >= 2:
        if progress_callback:
            progress_callback("Training lookahead predictors", 0.55)

        lookahead_bank = _train_lookahead_predictors(
            recordings,
            hidden_size,
            intermediate_size,
            rank=lookahead_rank,
            epochs=epochs,
            progress_callback=lambda desc, frac: (
                progress_callback(desc, 0.55 + frac * 0.1)
                if progress_callback
                else None
            ),
        )

    # Step 4: Bundle FFN weights
    if progress_callback:
        progress_callback("Bundling weights", 0.65)

    model_dir = Path(model_path)
    bundle_ffn_weights(model_dir, output_dir)

    # Step 5: Save predictors
    if progress_callback:
        progress_callback("Saving predictors", 0.8)

    predictor_bank.save(output_dir / "predictors")

    if lookahead_bank is not None:
        lookahead_bank.save(output_dir / "lookahead_predictors")

    # Step 6: Write config
    flash_config = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_layers": num_layers,
        "predictor_rank": rank,
        "predictor_ranks": layer_ranks,
        "sensitive_layers": sensitive_layers,
        "sensitive_rank_multiplier": sensitive_rank_multiplier,
        "lookahead_rank": lookahead_rank,
        "has_lookahead_predictors": lookahead_bank is not None,
        "num_calibration_samples": num_samples,
        "calibration_dataset": calibration_dataset or "c4",
        "activation_threshold": activation_threshold,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "flash_config.json").write_text(json.dumps(flash_config, indent=2))

    if progress_callback:
        progress_callback("Done", 1.0)

    logger.info("Flash preparation complete: %s", output_dir)
    return output_dir
