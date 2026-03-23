"""Repack MoE expert weights into bundled format for flash inference.

Each expert's gate_proj, up_proj, and down_proj weight matrices are stored
contiguously in a .flashexperts file, enabling efficient per-expert SSD reads.

Supports both regular (float16) and quantized (4-bit MXFP4, etc.) models.
Quantized weights are preserved in their packed form (no dequantization).
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MOE_HEADER_MAGIC = 0x464C4D45  # "FLME"
MOE_HEADER_VERSION = 1
MOE_HEADER_SIZE = 128  # bytes

# Header format:
# magic(4) version(4) num_experts(4) hidden_size(4) intermediate_size(4)
# is_quantized(4) bits(4) group_size(4) expert_byte_size(8)
# padding(88) = 128 bytes
_MOE_HEADER_STRUCT = struct.Struct("<IIIIIIII Q 88s")

# Map numpy dtype to string for manifest
_DTYPE_TO_STR = {
    np.dtype("float16"): "float16",
    np.dtype("float32"): "float32",
    np.dtype("uint32"): "uint32",
    np.dtype("uint8"): "uint8",
    np.dtype("int8"): "int8",
}


def _encode_moe_header(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    is_quantized: bool,
    bits: int,
    group_size: int,
    expert_byte_size: int,
) -> bytes:
    return _MOE_HEADER_STRUCT.pack(
        MOE_HEADER_MAGIC,
        MOE_HEADER_VERSION,
        num_experts,
        hidden_size,
        intermediate_size,
        1 if is_quantized else 0,
        bits,
        group_size,
        expert_byte_size,
        b"\x00" * 88,
    )


def parse_moe_header(data: bytes) -> dict:
    (
        magic,
        version,
        num_experts,
        hidden_size,
        intermediate_size,
        is_quantized_int,
        bits,
        group_size,
        expert_byte_size,
        _,
    ) = _MOE_HEADER_STRUCT.unpack(data[:MOE_HEADER_SIZE])
    if magic != MOE_HEADER_MAGIC:
        raise ValueError(
            f"Invalid .flashexperts magic: expected {MOE_HEADER_MAGIC:#010x}, got {magic:#010x}"
        )
    return {
        "magic": magic,
        "version": version,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "is_quantized": bool(is_quantized_int),
        "bits": bits,
        "group_size": group_size,
        "expert_byte_size": expert_byte_size,
    }


@dataclass
class MoeExpertLayout:
    """Describes the on-disk layout of one MoE layer's bundled expert weights."""

    layer_idx: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    expert_byte_size: int
    file_path: Path
    offsets: np.ndarray  # uint64 byte offsets for each expert
    is_quantized: bool
    bits: int
    group_size: int
    quant_mode: str = "affine"


# Cache for loaded shard files to avoid re-reading the same shard
_shard_cache: dict[str, dict] = {}


def _load_tensor(model_dir: Path, name: str, index: dict | None) -> np.ndarray:
    """Load a single tensor from safetensors, handling sharded models.

    Uses mx.load() which handles all dtypes including bfloat16.
    bfloat16 tensors are converted to float16 for numpy compatibility.
    Caches loaded shards to avoid redundant I/O.
    """
    import mlx.core as mx

    if index is not None:
        weight_map = index["weight_map"]
        shard_file = weight_map.get(name)
        if shard_file is None:
            raise KeyError(f"Tensor {name!r} not found in safetensors index")
        sf_path = str(model_dir / shard_file)
    else:
        sf_path = str(model_dir / "model.safetensors")

    if sf_path not in _shard_cache:
        _shard_cache[sf_path] = mx.load(sf_path)
    tensors = _shard_cache[sf_path]

    if name not in tensors:
        raise KeyError(f"Tensor {name!r} not found in {sf_path}")
    arr = tensors[name]

    # bfloat16 isn't supported by numpy — convert to float16
    if arr.dtype == mx.bfloat16:
        arr = arr.astype(mx.float16)

    mx.eval(arr)
    return np.array(arr)


def _clear_shard_cache():
    """Clear the shard cache to free memory."""
    _shard_cache.clear()


def _try_load_tensor(
    model_dir: Path, name: str, index: dict | None
) -> np.ndarray | None:
    """Load a tensor, returning None if not found."""
    try:
        return _load_tensor(model_dir, name, index)
    except (KeyError, FileNotFoundError):
        return None
    except Exception as e:
        # safetensors raises SafetensorError for missing tensors
        if "does not contain tensor" in str(e):
            return None
        raise


def _detect_expert_prefix(
    model_dir: Path, sample_layer: int, index: dict | None
) -> str:
    """Detect whether expert weights use 'switch_mlp' or 'experts' prefix."""
    for prefix in ("switch_mlp", "experts"):
        name = f"model.layers.{sample_layer}.mlp.{prefix}.gate_proj.weight"
        # Check index first (fast, no I/O)
        if index and name in index.get("weight_map", {}):
            return prefix

    # No index — scan single safetensors file keys
    import mlx.core as mx

    sf_path = model_dir / "model.safetensors"
    if sf_path.exists():
        sf_key = str(sf_path)
        if sf_key not in _shard_cache:
            _shard_cache[sf_key] = mx.load(sf_key)
        tensors = _shard_cache[sf_key]
        for prefix in ("switch_mlp", "experts"):
            name = f"model.layers.{sample_layer}.mlp.{prefix}.gate_proj.weight"
            if name in tensors:
                return prefix

    raise ValueError(
        f"Cannot find expert weights for layer {sample_layer} "
        f"(tried switch_mlp and experts prefixes)"
    )


def _detect_moe_layers(config: dict) -> list[int]:
    """Return list of layer indices that are MoE layers based on config."""
    num_layers = config.get("num_hidden_layers") or config.get("num_layers", 0)
    first_dense = config.get("first_k_dense_replace", 0)
    freq = config.get("moe_layer_freq") or 1

    moe_layers = []
    for i in range(num_layers):
        if i >= first_dense and (i - first_dense) % freq == 0:
            moe_layers.append(i)
    return moe_layers


def _collect_expert_components(
    model_dir: Path,
    prefix: str,
    index: dict | None,
) -> list[tuple[str, np.ndarray]]:
    """Collect all component arrays for one projection, in order.

    Returns list of (component_name, stacked_array) tuples.
    For quantized: [("weight", arr), ("scales", arr), ("biases", arr)]
    For non-quantized: [("weight", arr)]
    biases may be absent.
    """
    components = []
    weight = _load_tensor(model_dir, f"{prefix}.weight", index)
    components.append(("weight", weight))

    scales = _try_load_tensor(model_dir, f"{prefix}.scales", index)
    if scales is not None:
        components.append(("scales", scales))

    biases = _try_load_tensor(model_dir, f"{prefix}.biases", index)
    if biases is not None:
        components.append(("biases", biases))

    return components


def _collect_all_projections(
    model_dir: Path,
    prefix: str,
    index: dict | None,
) -> tuple[list[tuple[str, str, np.ndarray]], list[dict]]:
    """Collect all components for gate_proj, up_proj, down_proj.

    Returns:
        (all_components, manifest_entries)
        all_components: [(proj_name, comp_name, stacked_arr), ...]
        manifest_entries: [{"name": "gate_proj.weight", "shape": [...], "dtype": "uint32", "nbytes": N}, ...]
    """
    all_components = []
    manifest = []

    for proj in ("gate_proj", "up_proj", "down_proj"):
        components = _collect_expert_components(model_dir, f"{prefix}.{proj}", index)
        for comp_name, arr in components:
            all_components.append((proj, comp_name, arr))
            # Per-expert shape and size (first dim is num_experts)
            per_expert_shape = list(arr.shape[1:])
            dtype_str = _DTYPE_TO_STR.get(arr.dtype, str(arr.dtype))
            manifest.append(
                {
                    "name": f"{proj}.{comp_name}",
                    "shape": per_expert_shape,
                    "dtype": dtype_str,
                    "nbytes": int(arr[0].nbytes),
                }
            )

    # Also check for per-projection linear biases (e.g., gpt-oss has bias=True)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        bias = _try_load_tensor(model_dir, f"{prefix}.{proj}.bias", index)
        if bias is not None:
            all_components.append((proj, "bias", bias))
            per_expert_shape = list(bias.shape[1:])
            dtype_str = _DTYPE_TO_STR.get(bias.dtype, str(bias.dtype))
            manifest.append(
                {
                    "name": f"{proj}.bias",
                    "shape": per_expert_shape,
                    "dtype": dtype_str,
                    "nbytes": int(bias[0].nbytes),
                }
            )

    return all_components, manifest


def bundle_moe_experts(
    model_dir: Path,
    output_dir: Path,
) -> dict[int, MoeExpertLayout]:
    """Bundle MoE expert weights from safetensors into .flashexperts format.

    Reads stacked expert weights directly from safetensors (one tensor at a time)
    and writes per-expert data contiguously to .flashexperts files.

    Args:
        model_dir: Directory containing model safetensors and config.json.
        output_dir: Directory to write .flashexperts files.

    Returns:
        Dict mapping MoE layer index to MoeExpertLayout.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_shard_cache()

    # Read model config (handle wrapper models like Kimi-K2.5)
    raw_config = json.loads((model_dir / "config.json").read_text())
    config = raw_config.get("text_config", raw_config)
    hidden_size = config["hidden_size"]
    intermediate_size = config.get("moe_intermediate_size") or config.get(
        "intermediate_size"
    )
    if intermediate_size is None:
        raise ValueError(
            f"config.json at {model_dir} is missing both "
            "'moe_intermediate_size' and 'intermediate_size'"
        )
    num_experts = (
        config.get("n_routed_experts")
        or config.get("num_local_experts")  # gpt-oss uses this
    )
    if num_experts is None:
        raise ValueError(
            f"config.json at {model_dir} is missing both "
            "'n_routed_experts' and 'num_local_experts'"
        )

    # Check for safetensors index (sharded models)
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None

    # Detect quantization from config
    quant_config = config.get("quantization")

    moe_layers = _detect_moe_layers(config)
    if not moe_layers:
        raise ValueError(f"No MoE layers detected in config at {model_dir}")

    logger.info(
        "Bundling %d MoE layers (%d experts each) from %s",
        len(moe_layers),
        num_experts,
        model_dir,
    )

    layouts: dict[int, MoeExpertLayout] = {}

    try:
        # Detect expert weight prefix: "switch_mlp" (DeepSeek-V3) or "experts" (gpt-oss)
        expert_prefix = _detect_expert_prefix(model_dir, moe_layers[0], index)

        # Process first MoE layer to determine component manifest
        first_prefix = f"model.layers.{moe_layers[0]}.mlp.{expert_prefix}"
        _, manifest = _collect_all_projections(model_dir, first_prefix, index)
        expert_byte_size = sum(entry["nbytes"] for entry in manifest)

        bits = 0
        group_size = 0
        quant_mode = "affine"
        is_quantized = any(
            e["dtype"] == "uint32" for e in manifest if "weight" in e["name"]
        )
        if is_quantized and quant_config:
            bits = quant_config.get("bits", 4)
            group_size = quant_config.get("group_size", 32)
            quant_mode = quant_config.get("mode", "affine")
        for layer_idx in moe_layers:
            prefix = f"model.layers.{layer_idx}.mlp.{expert_prefix}"

            all_components, layer_manifest = _collect_all_projections(
                model_dir, prefix, index
            )
            if layer_manifest != manifest:
                raise ValueError(
                    f"MoE layer {layer_idx} has different component layout than "
                    f"layer {moe_layers[0]}. Heterogeneous MoE layers are not supported."
                )

            # Build offset table
            offset_table_size = num_experts * 8
            data_start = MOE_HEADER_SIZE + offset_table_size
            offsets = np.array(
                [data_start + i * expert_byte_size for i in range(num_experts)],
                dtype=np.uint64,
            )

            file_path = output_dir / f"layer_{layer_idx:02d}.flashexperts"
            with open(file_path, "wb") as f:
                f.write(
                    _encode_moe_header(
                        num_experts,
                        hidden_size,
                        intermediate_size,
                        is_quantized,
                        bits,
                        group_size,
                        expert_byte_size,
                    )
                )
                f.write(offsets.tobytes())

                for expert_idx in range(num_experts):
                    for _proj, _comp, arr in all_components:
                        f.write(arr[expert_idx].tobytes())

            layouts[layer_idx] = MoeExpertLayout(
                layer_idx=layer_idx,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                expert_byte_size=expert_byte_size,
                file_path=file_path,
                offsets=offsets,
                is_quantized=is_quantized,
                bits=bits,
                group_size=group_size,
            )

            logger.info("Bundled MoE layer %d: %d experts", layer_idx, num_experts)

            # Free memory for this layer
            del all_components
            _clear_shard_cache()
    finally:
        _clear_shard_cache()

    # Write layout JSON with component manifest
    layout_config = {
        "num_moe_layers": len(layouts),
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "is_quantized": is_quantized,
        "quant_mode": quant_mode,
        "expert_prefix": expert_prefix,
        "component_manifest": manifest,
        "layers": {
            str(idx): {
                "file": layout.file_path.name,
                "num_experts": layout.num_experts,
                "expert_byte_size": layout.expert_byte_size,
                "is_quantized": layout.is_quantized,
                "bits": layout.bits,
                "group_size": layout.group_size,
            }
            for idx, layout in layouts.items()
        },
    }
    (output_dir / "flash_moe_layout.json").write_text(
        json.dumps(layout_config, indent=2)
    )

    return layouts
