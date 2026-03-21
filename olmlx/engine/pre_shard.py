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

logger = logging.getLogger(__name__)

# Weight file patterns to exclude when collecting non-weight files
_WEIGHT_PATTERNS = {
    re.compile(r"^model.*\.safetensors$"),
    re.compile(r"^model.*\.safetensors\.index\.json$"),
    re.compile(r"^pytorch_model.*\.bin$"),
    re.compile(r"^pytorch_model.*\.bin\.index\.json$"),
    re.compile(r"^\.pre_sharded$"),  # pre-shard marker file
}


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
    shard_dir: Path, rank: int, world_size: int, model_path: str
) -> None:
    """Write a .pre_sharded JSON marker for validation."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "rank": rank,
        "world_size": world_size,
        "model_path": model_path,
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
