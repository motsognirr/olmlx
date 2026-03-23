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
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from mlx.utils import tree_map

import mlx_lm

from olmlx.engine.flash.bundler import bundle_ffn_weights
from olmlx.engine.flash.predictor import PredictorBank

logger = logging.getLogger(__name__)


def _nullify_module_params(module: nn.Module) -> None:
    """Replace all parameter arrays with tiny placeholders to free VRAM."""
    placeholder = mx.zeros((1,))
    new_params = tree_map(
        lambda x: placeholder if isinstance(x, mx.array) else x,
        module.parameters(),
    )
    module.update(new_params)


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

            tokens = tokenizer.encode(text)
            if len(tokens) > 512:
                tokens = tokens[:512]

            input_ids = mx.array([tokens])

            try:
                model(input_ids)
            except (ValueError, RuntimeError):
                logger.debug("Skipping sample %d (model error)", idx)
    finally:
        for i, mlp in original_mlps.items():
            layers[i].mlp = mlp

    return recordings


class _RecordingMLP(nn.Module):
    """Wrapper that records FFN activation patterns during forward pass.

    Computes gate/up projections once and reuses them for both the actual
    MLP output and the recording targets, avoiding redundant matmuls.
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

    def __call__(self, x):
        orig = self._original
        if not (hasattr(orig, "gate_proj") and hasattr(orig, "up_proj")):
            return orig(x)

        # Compute gate/up once, reuse for both MLP output and recording
        gate_out = orig.gate_proj(x)
        up_out = orig.up_proj(x)
        activated = nn.silu(gate_out) * up_out
        result = orig.down_proj(activated)

        # Record activation pattern from the same gate/up values
        flat_input = x.reshape(-1, x.shape[-1])
        flat_gate = gate_out.reshape(-1, gate_out.shape[-1])
        flat_up = up_out.reshape(-1, up_out.shape[-1])
        intermediate = mx.sigmoid(flat_gate) * flat_gate * flat_up
        mean_act = mx.abs(intermediate).mean(axis=0)
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

    Returns:
        (recordings, hidden_size, intermediate_size, num_layers)
    """
    from mlx_lm.models.base import create_attention_mask

    if progress_callback:
        progress_callback("Loading model skeleton", 0.0)

    model, tokenizer = mlx_lm.load(model_path, lazy=True)

    # Access inner model (mlx-lm wraps: Model.model = LlamaModel/Qwen3Model/etc.)
    inner = model.model if hasattr(model, "model") else model
    layers = inner.layers
    num_layers = len(layers)

    layer0_mlp = layers[0].mlp
    if hasattr(layer0_mlp, "gate_proj") and hasattr(layer0_mlp, "up_proj"):
        # Lazy arrays still have shape metadata
        intermediate_size = layer0_mlp.gate_proj.weight.shape[0]
        hidden_size = layer0_mlp.gate_proj.weight.shape[1]
    else:
        raise ValueError("First layer has no gate_proj/up_proj — cannot determine dimensions")

    if progress_callback:
        progress_callback("Computing embeddings", 0.02)

    mx.eval(inner.embed_tokens.parameters())

    hidden_states: list[mx.array] = []
    for text in calibration_texts:
        tokens = tokenizer.encode(text)
        if len(tokens) > 512:
            tokens = tokens[:512]
        input_ids = mx.array([tokens])
        h = inner.embed_tokens(input_ids)
        mx.eval(h)
        hidden_states.append(h)

    _nullify_module_params(inner.embed_tokens)
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
            mask = create_attention_mask(hidden_states[i], cache=None)
            hidden_states[i] = layer(hidden_states[i], mask=mask, cache=None)
            mx.eval(hidden_states[i])
        recordings[layer_idx] = recording_data

        layer.mlp = original_mlp
        _nullify_module_params(layer)
        gc.collect()
        mx.clear_cache()

        if progress_callback:
            frac = 0.05 + ((layer_idx + 1) / num_layers) * 0.95
            progress_callback(
                f"Processed layer {layer_idx + 1}/{num_layers}", frac
            )

    return recordings, hidden_size, intermediate_size, num_layers


def _train_predictors(
    recordings: dict[int, tuple[list[mx.array], list[mx.array]]],
    hidden_size: int,
    intermediate_size: int,
    rank: int = 128,
    epochs: int = 5,
    lr: float = 1e-3,
    progress_callback: Callable[[str, float], None] | None = None,
) -> PredictorBank:
    """Train per-layer sparsity predictors from recorded activations."""
    from mlx.optimizers import Adam

    num_layers = len(recordings)
    bank = PredictorBank(num_layers, hidden_size, intermediate_size, rank)

    total_steps = num_layers * epochs
    step = 0

    for layer_idx in sorted(recordings.keys()):
        inputs_list, targets_list = recordings[layer_idx]
        if not inputs_list:
            logger.warning("No recordings for layer %d, skipping", layer_idx)
            step += epochs
            continue

        # Stack into training tensors
        inputs = mx.stack(inputs_list)  # (N, hidden_size)
        targets = mx.stack(targets_list)  # (N, intermediate_size)

        pred = bank.predictors[layer_idx]
        optimizer = Adam(learning_rate=lr)

        def loss_fn(model, x, y):
            scores = model(x)
            # Binary cross-entropy
            eps = 1e-7
            return -mx.mean(
                y * mx.log(scores + eps) + (1 - y) * mx.log(1 - scores + eps)
            )

        loss_and_grad = nn.value_and_grad(pred, loss_fn)

        for epoch in range(epochs):
            loss, grads = loss_and_grad(pred, inputs, targets)
            optimizer.update(pred, grads)
            mx.eval(pred.parameters(), optimizer.state)
            step += 1

            if progress_callback:
                progress_callback(
                    f"Training layer {layer_idx} (epoch {epoch + 1}/{epochs})",
                    step / total_steps,
                )

    return bank


def prepare_model_for_flash(
    model_path: str,
    output_dir: Path | None = None,
    rank: int = 128,
    num_samples: int = 256,
    activation_threshold: float = 0.01,
    epochs: int = 5,
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
    calibration_texts = _get_calibration_data(num_samples)

    # Step 2: Stream-record activations (one layer at a time)
    recordings, hidden_size, intermediate_size, num_layers = (
        _stream_record_activations(
            model_path,
            calibration_texts,
            activation_threshold=activation_threshold,
            progress_callback=lambda desc, frac: (
                progress_callback(desc, frac * 0.4) if progress_callback else None
            ),
        )
    )

    # Step 3: Train predictors
    if progress_callback:
        progress_callback("Training predictors", 0.4)

    predictor_bank = _train_predictors(
        recordings,
        hidden_size,
        intermediate_size,
        rank=rank,
        epochs=epochs,
        progress_callback=lambda desc, frac: (
            progress_callback(desc, 0.4 + frac * 0.2) if progress_callback else None
        ),
    )

    # Step 4: Bundle FFN weights
    if progress_callback:
        progress_callback("Bundling weights", 0.6)

    model_dir = Path(model_path)
    bundle_ffn_weights(model_dir, output_dir)

    # Step 5: Save predictors
    if progress_callback:
        progress_callback("Saving predictors", 0.8)

    predictor_bank.save(output_dir / "predictors")

    # Step 6: Write config
    flash_config = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_layers": num_layers,
        "predictor_rank": rank,
        "num_calibration_samples": num_samples,
        "activation_threshold": activation_threshold,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "flash_config.json").write_text(json.dumps(flash_config, indent=2))

    if progress_callback:
        progress_callback("Done", 1.0)

    logger.info("Flash preparation complete: %s", output_dir)
    return output_dir
