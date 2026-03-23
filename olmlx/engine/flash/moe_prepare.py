"""Preparation pipeline for Flash-MoE inference.

Prepares an MoE model for Flash-MoE inference by bundling expert weights
into .flashexperts files. Unlike dense flash preparation, this does NOT
require loading the full model into RAM — it reads safetensors directly.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def is_moe_model(model_path: str | Path) -> bool:
    """Check if a model is an MoE model based on config.json."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False
    config = json.loads(config_path.read_text())
    # Check for text_config (wrapper models like Kimi-K2.5)
    if "text_config" in config:
        config = config["text_config"]
    return (
        config.get("n_routed_experts") is not None
        or config.get("num_local_experts") is not None
    )


def prepare_moe_for_flash(
    model_path: str,
    output_dir: Path | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> Path:
    """Full preparation pipeline for Flash-MoE inference.

    Steps:
    1. Read config.json to detect MoE architecture
    2. Bundle expert weights from safetensors (no model loading)
    3. Write flash_moe_config.json

    Args:
        model_path: Local model directory path.
        output_dir: Where to write flash_moe files. Defaults to model_dir/flash_moe.
        progress_callback: Called with (description, progress_fraction).

    Returns:
        Path to the flash_moe directory.
    """
    from olmlx.engine.flash.moe_bundler import bundle_moe_experts

    model_dir = Path(model_path)

    if output_dir is None:
        output_dir = model_dir / "flash_moe"

    if progress_callback:
        progress_callback("Reading model config", 0.0)

    config = json.loads((model_dir / "config.json").read_text())
    # Handle wrapper models (e.g., Kimi-K2.5 wraps DeepSeek-V3)
    text_config = config.get("text_config", config)

    hidden_size = text_config["hidden_size"]
    intermediate_size = text_config.get("moe_intermediate_size") or text_config.get(
        "intermediate_size"
    )
    if intermediate_size is None:
        raise ValueError(
            f"config.json at {model_dir} is missing both "
            "'moe_intermediate_size' and 'intermediate_size'"
        )
    num_experts = text_config.get("n_routed_experts") or text_config.get(
        "num_local_experts"
    )
    num_layers = text_config.get("num_hidden_layers") or text_config.get("num_layers")
    first_dense = text_config.get("first_k_dense_replace", 0)
    moe_freq = text_config.get("moe_layer_freq", 1)
    num_experts_per_tok = text_config.get("num_experts_per_tok", 8)

    if progress_callback:
        progress_callback("Bundling expert weights", 0.1)

    layouts = bundle_moe_experts(model_dir, output_dir)

    if progress_callback:
        progress_callback("Writing config", 0.9)

    # Write flash_moe_config.json
    flash_config = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "num_layers": num_layers,
        "first_k_dense_replace": first_dense,
        "moe_layer_freq": moe_freq,
        "num_moe_layers": len(layouts),
        "moe_layer_indices": sorted(layouts.keys()),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "flash_moe_config.json").write_text(
        json.dumps(flash_config, indent=2)
    )

    if progress_callback:
        progress_callback("Done", 1.0)

    logger.info("Flash-MoE preparation complete: %s", output_dir)
    return output_dir
