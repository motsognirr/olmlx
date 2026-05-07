"""Sharded precompute of target hidden states for DFlash draft training.

Saves ``(input_ids, target_hidden)`` pairs to ``.safetensors`` shards
on SSD so subsequent training runs can skip the target forward pass —
the dominant cost when training a draft for a large target.

Shard format: each shard contains a single batch as
``{"input_ids": (B, L) int32, "target_hidden": (B, L, num_target_layers
* hidden_size)}``. One batch per shard keeps the on-disk layout
trivial; multi-batch packing is unnecessary because shard count
follows ``steps`` directly. ``index.json`` in the output directory
records ``{batch_size, seq_len, hidden_size, num_shards}`` so the
reader can validate shape compatibility before training starts.

Storage cost (rough): ``shards * batch_size * seq_len *
num_target_layers * hidden_size * 2`` bytes for bf16 hiddens. A 27B
target with ``num_target_layers=4`` and ``hidden_size=4096`` consumes
~32 KB per token of hiddens — tractable for ~100k tokens (~3 GB), gets
unwieldy past 1M tokens. The CLI exposes ``--precompute-shards`` so
users can scope the precompute pass to what fits.

Distillation (``--distill``) is incompatible with precomputed shards
because storing vocab-size logits per token blows up the storage cost
~100x. Run distillation in the online mode.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.json"
SHARD_PATTERN = "shard-{:05d}.safetensors"


def precompute_target_hiddens(
    target: nn.Module,
    batches: Iterable[mx.array],
    output_dir: str | Path,
    *,
    num_shards: int | None = None,
    progress_callback: Any = None,
) -> Path:
    """Run the target on each batch and dump ``(input_ids, hidden)`` shards.

    ``target`` must already have ``_patch_model`` installed by the
    caller — this function does not own the hook lifecycle so the
    caller can also use the same patched target for online training in
    the same session.

    ``num_shards``, if set, caps the number of shards written; the
    function returns early once the cap is hit. ``None`` consumes the
    full ``batches`` iterator.
    """
    if num_shards is not None and num_shards <= 0:
        raise ValueError(
            f"num_shards must be a positive integer or None, got {num_shards}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(target, "_hidden_states"):
        raise RuntimeError(
            "precompute_target_hiddens requires _patch_model(target, ...) "
            "to be installed before calling"
        )

    written = 0
    batch_size: int | None = None
    seq_len: int | None = None
    hidden_size: int | None = None

    for shard_idx, input_ids in enumerate(batches):
        if num_shards is not None and shard_idx >= num_shards:
            break

        # Reset slots before each forward so a hook that fires zero times
        # this iteration cannot leak a stale tensor from the previous one
        # — the ``any(h is None ...)`` guard below only catches hooks
        # that *never* fired.
        for i in range(len(target._hidden_states)):  # type: ignore[attr-defined]
            target._hidden_states[i] = None  # type: ignore[attr-defined]
        target(input_ids, cache=None)
        captured = list(target._hidden_states)  # type: ignore[attr-defined]
        if any(h is None for h in captured):
            raise RuntimeError(
                "Target forward did not populate all configured target_layer_ids"
            )
        hidden = mx.concatenate(captured, axis=-1)
        # Detach and force evaluation so the safetensors writer sees a
        # materialized array rather than a lazy expression that would
        # rerun the target forward.
        hidden = mx.stop_gradient(hidden)
        mx.eval(hidden, input_ids)

        # Capture shape metadata from the first shard so the reader can
        # validate compatibility before training starts.
        if shard_idx == 0:
            batch_size = int(input_ids.shape[0])
            seq_len = int(input_ids.shape[1])
            hidden_size = int(hidden.shape[-1])

        shard_path = output_dir / SHARD_PATTERN.format(shard_idx)
        mx.save_safetensors(
            str(shard_path),
            {
                "input_ids": input_ids.astype(mx.int32),
                "target_hidden": hidden,
            },
        )
        written += 1

        if progress_callback is not None and num_shards is not None:
            progress_callback(
                f"Precomputed shard {written}/{num_shards}",
                written / num_shards,
            )

    index = {
        "num_shards": written,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
    }
    (output_dir / INDEX_FILENAME).write_text(json.dumps(index, indent=2))
    logger.info("Wrote %d precomputed shards to %s", written, output_dir)
    return output_dir


_REQUIRED_INDEX_KEYS = ("num_shards", "batch_size", "seq_len", "hidden_size")


def read_precomputed_index(shard_dir: str | Path) -> dict[str, Any]:
    """Parse ``index.json`` from a precompute directory and validate keys.

    Raises ``FileNotFoundError`` if the index is missing, ``ValueError``
    on JSON corruption or missing required keys. Callers can use the
    returned dict to validate ``batch_size``/``seq_len``/``hidden_size``
    against the current training configuration before iteration starts.
    """
    index_path = Path(shard_dir) / INDEX_FILENAME
    if not index_path.exists():
        raise FileNotFoundError(f"Precompute index missing: {index_path}")
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt precompute index at {index_path}: {exc}") from exc
    missing = [k for k in _REQUIRED_INDEX_KEYS if index.get(k) is None]
    if missing:
        raise ValueError(
            f"Precompute index at {index_path} missing required keys: {missing}"
        )
    return index


def iter_precomputed_shards(
    shard_dir: str | Path,
    *,
    max_examples: int | None = None,
) -> Iterator[tuple[mx.array, mx.array]]:
    """Yield ``(input_ids, target_hidden)`` tuples from precomputed shards.

    Shards are read in lexicographic order, which matches the writer's
    ``shard-NNNNN`` zero-padded naming.

    ``max_examples=None`` yields each shard exactly once (one-shot dump
    mode). When ``max_examples`` exceeds the on-disk shard count, the
    iterator cycles back to the first shard — this provides "free"
    multi-epoch behavior without complicating the training loop's
    bookkeeping. The training pipeline always passes ``max_examples=
    steps``, so the cycling path is the production code path.
    """
    shard_dir = Path(shard_dir)
    if not shard_dir.exists():
        raise FileNotFoundError(f"Precomputed shard directory not found: {shard_dir}")

    index_path = shard_dir / INDEX_FILENAME
    if index_path.exists():
        # Parsed for validation only — callers needing the metadata
        # should call ``read_precomputed_index`` directly.
        read_precomputed_index(shard_dir)

    shard_paths = sorted(shard_dir.glob("shard-*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard-*.safetensors files in {shard_dir}; was "
            "precompute_target_hiddens run?"
        )

    yielded = 0
    while True:
        for shard_path in shard_paths:
            tensors = mx.load(str(shard_path))
            try:
                input_ids = tensors["input_ids"]
                target_hidden = tensors["target_hidden"]
            except KeyError as exc:
                raise ValueError(
                    f"Shard {shard_path} is missing required key {exc}"
                ) from exc
            yield input_ids, target_hidden
            yielded += 1
            if max_examples is not None and yielded >= max_examples:
                return
        if max_examples is None:
            # No cap → yield each shard exactly once and stop.
            return
