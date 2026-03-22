"""Preparation pipeline for flash inference.

Prepares a model for LLM-in-a-Flash inference by:
1. Loading the model fully
2. Recording FFN activation patterns (calibration)
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

from olmlx.engine.flash.bundler import bundle_ffn_weights
from olmlx.engine.flash.predictor import PredictorBank

logger = logging.getLogger(__name__)


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

    # Install hooks on each MLP to record inputs and activations
    original_mlps = {}
    for i, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue
        original_mlps[i] = layer.mlp

    # Install hooks once (not per-sample) to capture MLP inputs/activations.
    # The hooks re-run gate+up projections because we need intermediate
    # activation magnitudes, which aren't exposed by the standard forward.
    def make_hook(layer_idx, orig_call):
        def hooked_call(x):
            result = orig_call(x)

            flat_input = x.reshape(-1, x.shape[-1])
            mlp = original_mlps[layer_idx]
            if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
                gate_out = mlp.gate_proj(flat_input)
                up_out = mlp.up_proj(flat_input)
                intermediate = mx.sigmoid(gate_out) * gate_out * up_out
                mean_act = mx.abs(intermediate).mean(axis=0)
                target = (mean_act > activation_threshold).astype(mx.float32)
                mean_input = flat_input.mean(axis=0)

                mx.eval(mean_input, target)
                recordings[layer_idx][0].append(mean_input)
                recordings[layer_idx][1].append(target)

            return result

        return hooked_call

    for i in original_mlps:
        layers[i].mlp.__call__ = make_hook(i, original_mlps[i].__call__)

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
        # Restore original MLPs
        for i, mlp in original_mlps.items():
            layers[i].mlp.__call__ = mlp.__call__

    return recordings


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

    Steps:
    1. Load model fully (requires enough RAM)
    2. Run calibration to record activations
    3. Train per-layer predictors
    4. Bundle FFN weights with row-column bundling
    5. Save predictor bank
    6. Write flash_config.json

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
    from olmlx import ensure_mlx_lm

    ensure_mlx_lm()
    import mlx_lm

    if progress_callback:
        progress_callback("Loading model", 0.0)

    model, tokenizer = mlx_lm.load(model_path)

    # Determine dimensions from model config
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        hidden_size = config.get("hidden_size")
        intermediate_size = config.get("intermediate_size")
        num_layers = config.get("num_hidden_layers")
    else:
        # Try to infer from model
        layer0 = model.layers[0]
        hidden_size = layer0.mlp.gate_proj.weight.shape[1]
        intermediate_size = layer0.mlp.gate_proj.weight.shape[0]
        num_layers = len(model.layers)

    if output_dir is None:
        output_dir = Path(model_path) / "flash"

    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Loading model", 0.1)

    # Step 1: Generate calibration data
    calibration_texts = _get_calibration_data(num_samples)

    # Step 2: Record activations
    if progress_callback:
        progress_callback("Recording activations", 0.1)

    recordings = _record_activations(
        model,
        tokenizer,
        calibration_texts,
        activation_threshold=activation_threshold,
        progress_callback=lambda desc, frac: (
            progress_callback(desc, 0.1 + frac * 0.3) if progress_callback else None
        ),
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

    # Free model to reclaim RAM before bundling
    del model
    gc.collect()
    mx.clear_cache()

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
