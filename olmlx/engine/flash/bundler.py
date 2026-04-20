"""Repack model FFN weights into bundled format for flash inference.

Each neuron's gate_proj column, up_proj column, and down_proj row are stored
contiguously in a .flashweights file, enabling efficient sequential reads
(row-column bundling from the LLM in a Flash paper).

Supports both regular (float16) and quantized (4-bit/8-bit) models.
Quantized weights are dequantized to float16 during bundling.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from olmlx.engine.flash import _ssd_base
from olmlx.engine.flash._ssd_base import HeaderSpec

logger = logging.getLogger(__name__)

HEADER_MAGIC = 0x464C5348  # "FLSH"
HEADER_VERSION = 1
HEADER_SIZE = 64  # bytes

# Dtype string → byte size
_DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
}

# Body: num_neurons(4) hidden_size(4) dtype_len(4) dtype(16) = 28 bytes.
# Prefix (magic+version) is 8 bytes; padding to 64 bytes is handled by the codec.
_HEADER_SPEC = HeaderSpec(
    magic=HEADER_MAGIC,
    version=HEADER_VERSION,
    size=HEADER_SIZE,
    body_format="<III16s",
)


def _encode_header(num_neurons: int, hidden_size: int, dtype: str) -> bytes:
    dtype_bytes = dtype.encode("ascii")
    return _ssd_base.encode_header(
        _HEADER_SPEC,
        num_neurons,
        hidden_size,
        len(dtype_bytes),
        dtype_bytes.ljust(16, b"\x00"),
    )


def parse_header(data: bytes) -> dict:
    """Parse a .flashweights header. Raises ValueError on bad magic or version."""
    num_neurons, hidden_size, dtype_len, dtype_raw = _ssd_base.parse_header(
        _HEADER_SPEC, data
    )
    dtype = dtype_raw[:dtype_len].decode("ascii")
    return {
        "magic": HEADER_MAGIC,
        "version": HEADER_VERSION,
        "num_neurons": num_neurons,
        "hidden_size": hidden_size,
        "dtype": dtype,
    }


@dataclass
class BundledLayerLayout:
    """Describes the on-disk layout of one FFN layer's bundled weights."""

    layer_idx: int
    num_neurons: int
    hidden_size: int
    neuron_byte_size: int
    file_path: Path
    offsets: np.ndarray  # uint64 byte offsets for each neuron
    dtype: str


def _dequantize_weight(
    weight: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray | None,
    group_size: int,
    bits: int,
) -> np.ndarray:
    """Dequantize a weight tensor to float16 using MLX."""
    w_mx = mx.array(weight)
    s_mx = mx.array(scales)
    b_mx = mx.array(biases) if biases is not None else None
    result = mx.dequantize(w_mx, s_mx, b_mx, group_size, bits)
    result = result.astype(mx.float16)
    mx.eval(result)
    return np.array(result)


def _find_ffn_weights(
    model_dir: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict | None]:
    """Load all FFN weights from safetensors, grouped by layer index.

    Returns (layers_dict, quantization_config_or_none).
    Quantized models have .weight/.scales/.biases per projection;
    regular models have just .weight.
    """
    layers: dict[int, dict[str, np.ndarray]] = {}

    # Check for quantization config
    config_path = model_dir / "config.json"
    quant_config = None
    if config_path.exists():
        config = json.loads(config_path.read_text())
        quant_config = config.get("quantization")

    # Match both text models (model.layers.X.mlp...) and VLMs
    # (language_model.model.layers.X.mlp...)
    pattern = re.compile(
        r"(?:language_model\.)?model\.layers\.(\d+)\.mlp\."
        r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)"
    )

    for sf_path in sorted(model_dir.glob("*.safetensors")):
        if sf_path.name.startswith("flash_"):
            continue
        # Use mx.load to handle bfloat16 (safetensors.numpy can't).
        # mx.load allocates Metal-backed arrays; convert matched tensors to
        # numpy immediately and delete the rest to free Metal memory per shard.
        try:
            tensors = mx.load(str(sf_path))
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to load {sf_path.name} during flash bundling: {exc}"
            ) from exc
        for name, arr in tensors.items():
            m = pattern.match(name)
            if m:
                layer_idx = int(m.group(1))
                proj_name = m.group(2)
                component = m.group(3)  # weight, scales, or biases
                key = f"{proj_name}.{component}"
                if layer_idx not in layers:
                    layers[layer_idx] = {}
                # bfloat16 has no numpy equivalent — cast to float16
                # (the bundler writes float16 anyway)
                if arr.dtype == mx.bfloat16:
                    arr = arr.astype(mx.float16)
                layers[layer_idx][key] = np.array(arr)
        del tensors
        mx.clear_cache()

    return layers, quant_config


def _get_dense_weights(
    layer_weights: dict[str, np.ndarray],
    quant_config: dict | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Extract dense float16 weight matrices from layer weights.

    Handles both quantized and regular formats.

    Returns (gate_w, up_w, down_w, intermediate_size, hidden_size).
    """
    is_quantized = "gate_proj.scales" in layer_weights

    if is_quantized:
        if quant_config is None:
            raise ValueError(
                "Quantized weights found but no quantization config in config.json"
            )
        group_size = quant_config["group_size"]
        bits = quant_config["bits"]

        gate_w = _dequantize_weight(
            layer_weights["gate_proj.weight"],
            layer_weights["gate_proj.scales"],
            layer_weights.get("gate_proj.biases"),
            group_size,
            bits,
        )
        up_w = _dequantize_weight(
            layer_weights["up_proj.weight"],
            layer_weights["up_proj.scales"],
            layer_weights.get("up_proj.biases"),
            group_size,
            bits,
        )
        down_w = _dequantize_weight(
            layer_weights["down_proj.weight"],
            layer_weights["down_proj.scales"],
            layer_weights.get("down_proj.biases"),
            group_size,
            bits,
        )
    else:
        gate_w = layer_weights["gate_proj.weight"]
        up_w = layer_weights["up_proj.weight"]
        down_w = layer_weights["down_proj.weight"]

    inter, hidden = gate_w.shape
    return gate_w, up_w, down_w, inter, hidden


def bundle_ffn_weights(
    model_dir: Path,
    output_dir: Path,
    dtype: str = "float16",
) -> dict[int, BundledLayerLayout]:
    """Repack safetensors FFN weights into bundled .flashweights format.

    For each FFN layer, creates a file where neuron i's data is stored
    contiguously as: [gate_proj row i | up_proj row i | down_proj col i].

    Quantized weights are dequantized to float16 during bundling.

    Args:
        model_dir: Directory containing model safetensors files.
        output_dir: Directory to write .flashweights files.
        dtype: Target dtype string. Only "float16" is currently supported.

    Returns:
        Dict mapping layer index to BundledLayerLayout.
    """
    if dtype != "float16":
        raise ValueError(
            f"Only dtype='float16' is supported for bundling, got '{dtype}'"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    layers, quant_config = _find_ffn_weights(model_dir)
    if not layers:
        raise ValueError(f"No FFN weights found in {model_dir}")

    if quant_config:
        logger.info(
            "Quantized model detected (bits=%d, group_size=%d) — dequantizing to %s",
            quant_config["bits"],
            quant_config["group_size"],
            dtype,
        )

    layouts: dict[int, BundledLayerLayout] = {}
    hidden_size = None
    intermediate_size = None

    for layer_idx in sorted(layers.keys()):
        gate_w, up_w, down_w, inter, hidden = _get_dense_weights(
            layers[layer_idx], quant_config
        )
        if hidden_size is None:
            hidden_size = hidden
            intermediate_size = inter

        dtype_bytes = _DTYPE_BYTES[dtype]
        neuron_byte_size = hidden * dtype_bytes * 3

        # Build offset table
        offset_table_size = inter * 8
        data_start = HEADER_SIZE + offset_table_size
        offsets = np.array(
            [data_start + i * neuron_byte_size for i in range(inter)],
            dtype=np.uint64,
        )

        file_path = output_dir / f"layer_{layer_idx:02d}.flashweights"
        # Interleave rows/cols per neuron into one contiguous buffer so we
        # issue a single write(2) per layer instead of 3 per neuron.
        # Inline the casts so each temp is freed before the next is made.
        neuron_block = np.empty((inter, hidden * 3), dtype=np.float16)
        neuron_block[:, :hidden] = gate_w.astype(np.float16, copy=False)
        neuron_block[:, hidden : 2 * hidden] = up_w.astype(np.float16, copy=False)
        neuron_block[:, 2 * hidden :] = down_w.astype(np.float16, copy=False).T
        with open(file_path, "wb") as f:
            f.write(_encode_header(inter, hidden, dtype))
            f.write(offsets.tobytes())
            f.write(neuron_block)
        del neuron_block

        layouts[layer_idx] = BundledLayerLayout(
            layer_idx=layer_idx,
            num_neurons=inter,
            hidden_size=hidden,
            neuron_byte_size=neuron_byte_size,
            file_path=file_path,
            offsets=offsets,
            dtype=dtype,
        )

        logger.info("Bundled layer %d: %d neurons", layer_idx, inter)

        # Free memory for this layer's dense weights
        del gate_w, up_w, down_w
        layers[layer_idx] = {}  # Release source tensors

    config = {
        "num_layers": len(layouts),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": dtype,
        "quantized_source": quant_config is not None,
        "layers": {
            str(idx): {
                "file": layout.file_path.name,
                "num_neurons": layout.num_neurons,
                "neuron_byte_size": layout.neuron_byte_size,
            }
            for idx, layout in layouts.items()
        },
    }
    (output_dir / "flash_layout.json").write_text(json.dumps(config, indent=2))

    return layouts
