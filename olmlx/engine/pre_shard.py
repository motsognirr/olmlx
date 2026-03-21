"""Pre-sharding logic for distributed inference.

Downloads the model once on the coordinator, pre-computes each worker's
weight shard using a FakeGroup, and prepares directories for SCP transfer.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Weight file patterns to exclude when collecting non-weight files
_WEIGHT_PATTERNS = {
    re.compile(r"^model.*\.safetensors$"),
    re.compile(r"^model.*\.safetensors\.index\.json$"),
    re.compile(r"^pytorch_model.*\.bin$"),
    re.compile(r"^pytorch_model.*\.bin\.index\.json$"),
    re.compile(r"^\.pre_sharded$"),  # pre-shard marker file
}


_LAYER_KEY_RE = re.compile(r"^(model\.layers\.)(\d+)(\..*)")


def _filter_pipeline_weights(
    weights: dict[str, Any],
    start_idx: int,
    end_idx: int,
) -> dict[str, Any]:
    """Filter and renumber layer weights for a pipeline rank.

    Keeps weights for layers in [start_idx, end_idx), renumbered to 0-based.
    Non-layer weights (embed_tokens, norm, lm_head, etc.) are kept as-is.
    """
    filtered = {}
    for key, value in weights.items():
        m = _LAYER_KEY_RE.match(key)
        if m:
            layer_idx = int(m.group(2))
            if start_idx <= layer_idx < end_idx:
                new_key = f"{m.group(1)}{layer_idx - start_idx}{m.group(3)}"
                filtered[new_key] = value
        else:
            filtered[key] = value
    return filtered


class FakeGroup:
    """Mimics mx.distributed.Group for pre-sharding without a real distributed setup."""

    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def collect_non_weight_files(model_dir: Path) -> list[Path]:
    """Return paths to config, tokenizer, and other non-weight files."""
    files = []
    for f in model_dir.iterdir():
        if not f.is_file():
            continue
        if any(p.match(f.name) for p in _WEIGHT_PATTERNS):
            continue
        files.append(f)
    return files


def write_shard_marker(
    shard_dir: Path,
    rank: int,
    world_size: int,
    model_path: str,
    **extra: Any,
) -> None:
    """Write a .pre_sharded JSON marker for validation."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "rank": rank,
        "world_size": world_size,
        "model_path": model_path,
        **extra,
    }
    (shard_dir / ".pre_sharded").write_text(json.dumps(marker))


def read_shard_marker(shard_dir: Path) -> dict | None:
    """Read and parse the .pre_sharded marker. Returns None if missing/corrupt."""
    try:
        return json.loads((shard_dir / ".pre_sharded").read_text())
    except (json.JSONDecodeError, OSError):
        return None


def pre_shard_for_rank(
    model_dir: Path,
    rank: int,
    world_size: int,
    output_dir: Path,
) -> None:
    """Load model, shard with FakeGroup(rank, world_size), save weights."""
    # Lazy imports — mlx/mlx_lm are heavy and not available in test environments
    import mlx.core as mx  # noqa: F811
    import mlx.utils
    import mlx_lm  # noqa: F811

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files (config, tokenizer, etc.)
    for f in collect_non_weight_files(model_dir):
        shutil.copy2(f, output_dir / f.name)

    # Load model lazily and shard with fake group
    logger.info("Pre-sharding rank %d/%d from %s", rank, world_size, model_dir)
    model, _tokenizer = mlx_lm.load(str(model_dir))

    group = FakeGroup(rank=rank, size=world_size)
    model.shard(group)

    # Materialize and flatten nested parameter dict for save_safetensors
    params = model.parameters()
    mx.eval(params)
    flat_weights = dict(mlx.utils.tree_flatten(params))

    # Save as single safetensors file
    mx.save_safetensors(str(output_dir / "model.safetensors"), flat_weights)

    # Write marker
    write_shard_marker(output_dir, rank, world_size, str(model_dir))

    logger.info("Pre-sharded rank %d saved to %s", rank, output_dir)


def _load_safetensors_weights(
    model_dir: Path,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> dict:
    """Load safetensors weights from a model directory.

    Handles both single-file and multi-file (sharded) safetensors layouts.
    When start_idx/end_idx are provided, only loads shard files containing
    keys for layers in [start_idx, end_idx) or non-layer keys.
    """
    import mlx.core as mx

    index_path = model_dir / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        # No index file — try single-file layout
        return dict(mx.load(str(model_dir / "model.safetensors")))

    weight_map = index.get("weight_map")
    if weight_map is None:
        raise ValueError(
            f"model.safetensors.index.json in {model_dir} has no 'weight_map' key"
        )

    # For multi-file models, only load shard files containing needed keys
    if start_idx is not None and end_idx is not None:
        needed_files: set[str] = set()
        for key, shard_file in weight_map.items():
            m = _LAYER_KEY_RE.match(key)
            if m:
                layer_idx = int(m.group(2))
                if start_idx <= layer_idx < end_idx:
                    needed_files.add(shard_file)
            else:
                needed_files.add(shard_file)
    else:
        needed_files = set(weight_map.values())

    weights: dict = {}
    for shard_file in needed_files:
        weights.update(mx.load(str(model_dir / shard_file)))
    return weights


def pre_shard_pipeline_for_rank(
    model_dir: Path,
    rank: int,
    world_size: int,
    output_dir: Path,
    layer_counts: list[int] | None = None,
) -> None:
    """Filter and save only the layers owned by this rank for pipeline parallelism.

    Unlike tensor pre-sharding, this works at the weight-key level without
    instantiating the model. Layers are renumbered to 0-based, and config.json
    is modified to match the reduced layer count.
    """
    import mlx.core as mx

    from olmlx.engine.pipeline import _compute_layer_counts, _compute_layer_range

    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((model_dir / "config.json").read_text())
    total_layers = config["num_hidden_layers"]

    if layer_counts is None:
        layer_counts = _compute_layer_counts(total_layers, world_size)
    if sum(layer_counts) != total_layers:
        raise ValueError(
            f"layer_counts {layer_counts} sums to {sum(layer_counts)}, "
            f"but model has {total_layers} layers"
        )
    if len(layer_counts) != world_size:
        raise ValueError(
            f"layer_counts must have {world_size} entries (one per rank), "
            f"got {len(layer_counts)}"
        )

    start_idx, end_idx = _compute_layer_range(rank, layer_counts)

    logger.info(
        "Pipeline pre-sharding rank %d/%d: layers %d-%d (of %d total)",
        rank,
        world_size,
        start_idx,
        end_idx - 1,
        total_layers,
    )
    weights = _load_safetensors_weights(model_dir, start_idx, end_idx)
    filtered = _filter_pipeline_weights(weights, start_idx, end_idx)
    del weights

    mx.save_safetensors(str(output_dir / "model.safetensors"), filtered)

    # Copy non-weight files (skip config.json — we write a modified version)
    for f in collect_non_weight_files(model_dir):
        if f.name != "config.json":
            shutil.copy2(f, output_dir / f.name)

    config["num_hidden_layers"] = end_idx - start_idx
    if "layer_types" in config:
        if len(config["layer_types"]) != total_layers:
            raise ValueError(
                f"config.json layer_types length ({len(config['layer_types'])}) "
                f"does not match num_hidden_layers ({total_layers})"
            )
        config["layer_types"] = config["layer_types"][start_idx:end_idx]
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    write_shard_marker(
        output_dir,
        rank,
        world_size,
        str(model_dir),
        strategy="pipeline",
        layer_counts=layer_counts,
    )

    logger.info("Pipeline pre-sharded rank %d saved to %s", rank, output_dir)


def pre_shard_pipeline_all_workers(
    model_dir: Path,
    world_size: int,
    output_base: Path,
    layer_counts: list[int] | None = None,
    progress_cb=None,
) -> dict[int, Path]:
    """Pipeline pre-shard for all worker ranks (1..N-1). Returns {rank: shard_dir}."""
    result = {}
    for rank in range(1, world_size):
        shard_dir = output_base / f"rank{rank}"
        pre_shard_pipeline_for_rank(
            model_dir,
            rank=rank,
            world_size=world_size,
            output_dir=shard_dir,
            layer_counts=layer_counts,
        )
        result[rank] = shard_dir
        if progress_cb:
            progress_cb(rank, world_size)
    return result


def pre_shard_all_workers(
    model_dir: Path,
    world_size: int,
    output_base: Path,
    progress_cb=None,
) -> dict[int, Path]:
    """Pre-shard for all worker ranks (1..N-1). Returns {rank: shard_dir}.

    Note: each rank loads the full model from disk independently. Loading once
    and re-sharding in memory would be faster, but MLX model deep-copy behavior
    is unverified and sequential processing avoids accumulating multiple ranks'
    materialized weights.
    """
    result = {}
    for rank in range(1, world_size):
        shard_dir = output_base / f"rank{rank}"
        pre_shard_for_rank(
            model_dir, rank=rank, world_size=world_size, output_dir=shard_dir
        )
        result[rank] = shard_dir
        if progress_cb:
            progress_cb(rank, world_size)
    return result
