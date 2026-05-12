"""Sharded precompute of target hidden states for DFlash draft training.

Saves ``(input_ids, target_hidden)`` pairs to ``.safetensors`` shards
on SSD so subsequent training runs can skip the target forward pass —
the dominant cost when training a draft for a large target.

Shard format: each shard contains a single batch as
``{"input_ids": (B, L) int32, "target_hidden": (B, L, num_target_layers
* model_hidden_size)}``. One batch per shard keeps the on-disk layout
trivial; multi-batch packing is unnecessary because shard count
follows ``steps`` directly. ``index.json`` in the output directory
records ``{batch_size, seq_len, concat_hidden_size, num_shards}``
(the ``concat_`` prefix on the hidden dim is intentional: this is
``num_target_layers * model_hidden_size``, not the per-layer dim
written to ``config.json``) so the reader can validate shape
compatibility before training starts.

Storage cost (rough): ``shards * batch_size * seq_len *
num_target_layers * model_hidden_size * 2`` bytes for bf16 hiddens. A
27B target with ``num_target_layers=4`` and ``model_hidden_size=4096``
consumes ~32 KB per token of hiddens — tractable for ~100k tokens
(~3 GB), gets unwieldy past 1M tokens. The CLI exposes
``--precompute-shards`` so users can scope the precompute pass to
what fits.

I/O efficiency note: the training loop currently reads each shard
fully but only consumes the ``[:, :p+1, :]`` prefix where ``p`` is a
random pivot. With a uniform pivot and the default ``seq_len=2048``
that wastes ~50% of every read on average. A future improvement
would store fixed-pivot windows so each read is fully consumed; for
now, the trade-off pays off when target forwards dominate cost.

Distillation (``--distill``) is incompatible with precomputed shards
because storing vocab-size logits per token blows up the storage cost
~100x. Run distillation in the online mode.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.json"
SHARD_PATTERN = "shard-{:05d}.safetensors"

# Transient mx.load() failures under sustained mmap / file-descriptor
# pressure on macOS are rare but real (observed mid-run during a 2000-
# step EAGLE training). Retry up to 3 times with linear backoff
# (0s → 0.5s → 1.0s sleeps; ~1.5s total wait across 2 sleeps before
# the 3rd attempt) before giving up — plenty for a transient hiccup
# and trivial vs. losing an 8h training run.
_SHARD_OPEN_RETRIES = 3
_SHARD_OPEN_BACKOFF_S = 0.5


def precompute_target_hiddens(
    target: nn.Module,
    batches: Iterable[mx.array],
    output_dir: str | Path,
    storage: list[Any],
    *,
    target_layer_ids: list[int],
    num_shards: int | None = None,
    progress_callback: Any = None,
) -> Path:
    """Run the target on each batch and dump ``(input_ids, hidden)`` shards.

    ``target`` must already have ``_patch_model`` installed by the
    caller, with the same *storage* list passed here — this function
    does not own the hook lifecycle so the caller can also use the same
    patched target for online training in the same session.

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
        # that *never* fired. Slice-assign to keep the same list object
        # the installed hooks reference.
        storage[:] = [None] * len(storage)
        target(input_ids, cache=None)
        captured = list(storage)
        if any(h is None for h in captured):
            raise RuntimeError(
                "Target forward did not populate all configured target_layer_ids"
            )
        hidden = mx.concatenate(captured, axis=-1)
        # Force evaluation so the safetensors writer sees a materialized
        # array rather than a lazy expression that would rerun the
        # target forward. (No ``stop_gradient`` needed — there's no
        # value_and_grad tape active in this function, so nothing
        # would propagate through ``hidden`` anyway.)
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

    if written == 0:
        # Refuse to write an index with ``null`` for ``batch_size`` /
        # ``seq_len`` / ``concat_hidden_size`` (their initial value):
        # the file would land on disk and confuse anyone inspecting
        # it, and a subsequent ``read_precomputed_index`` would reject
        # it on missing-key grounds anyway.
        raise RuntimeError(
            "precompute_target_hiddens produced no shards — check that "
            "the 'batches' iterator yields at least one item (an exhausted "
            "iterator, an empty dataset, or all examples filtered out "
            "would all lead here)."
        )
    index = {
        "num_shards": written,
        "batch_size": batch_size,
        "seq_len": seq_len,
        # Named ``concat_hidden_size`` (not ``hidden_size``) so it
        # cannot be confused with the per-layer model dimension stored
        # in ``config.json``. This value is
        # ``num_target_layers * model_hidden_size``.
        "concat_hidden_size": hidden_size,
        # The exact layer indices that produced the captures.
        # ``concat_hidden_size`` alone can't tell two runs with the
        # same *count* but different *indices* apart (e.g.
        # ``[6,12,19,25]`` vs ``[1,2,3,4]`` both yield ``4 *
        # model_hidden_size``); without this key the training loop
        # would silently consume hiddens from the wrong layers.
        "target_layer_ids": list(target_layer_ids),
    }
    (output_dir / INDEX_FILENAME).write_text(json.dumps(index, indent=2))
    logger.info("Wrote %d precomputed shards to %s", written, output_dir)
    return output_dir


_REQUIRED_INDEX_KEYS = (
    "num_shards",
    "batch_size",
    "seq_len",
    "concat_hidden_size",
    "target_layer_ids",
)


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
    # Treat both absent keys and explicit ``null``/empty values as
    # missing. ``index.get(k) is None`` alone would accept
    # ``"target_layer_ids": null`` from a hand-edited JSON, and
    # ``"target_layer_ids": []`` would slip through the shape-check
    # downstream and surface as a confusing ``mx.array`` shape error.
    missing = [
        k
        for k in _REQUIRED_INDEX_KEYS
        if k not in index or index[k] is None or index[k] == [] or index[k] == ""
    ]
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

    shard_paths = sorted(shard_dir.glob("shard-*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard-*.safetensors files in {shard_dir}; was "
            "precompute_target_hiddens run?"
        )

    index_path = shard_dir / INDEX_FILENAME
    if index_path.exists():
        # Validate format AND cross-check the recorded shard count
        # against the actual files. A stale ``index.json`` (e.g. left
        # over after a manual shard delete) would otherwise let the
        # iterator silently cycle through fewer shards than the
        # training loop expects.
        meta = read_precomputed_index(shard_dir)
        if meta["num_shards"] != len(shard_paths):
            raise ValueError(
                f"Shard directory {shard_dir} is inconsistent: "
                f"index.json records num_shards={meta['num_shards']} but "
                f"found {len(shard_paths)} shard-*.safetensors files. "
                "Re-run precompute or delete the stale index.json."
            )

    yielded = 0
    while True:
        for shard_path in shard_paths:
            # ``mx.load`` is overloaded to return either a dict (safetensors)
            # or a single ``mx.array`` (npy); we always write safetensors,
            # so narrow the type for pyright.
            #
            # Retry transient open() failures. mlx's ``mx.load`` on macOS
            # has been observed to spuriously fail to open a valid shard
            # under sustained mmap / file-descriptor pressure mid-run
            # (single-shot reload of the same file via the same API
            # succeeds). Don't kill an 8h training run for a transient
            # I/O hiccup — retry a few times with backoff. Truly missing
            # / corrupt shards still surface after the retries are
            # exhausted, with the original RuntimeError.
            tensors: dict[str, mx.array] | None = None
            last_exc: Exception | None = None
            for attempt in range(_SHARD_OPEN_RETRIES):
                try:
                    tensors = mx.load(str(shard_path))  # type: ignore[assignment]
                    break
                except RuntimeError as exc:
                    # Retry any ``RuntimeError`` from ``mx.load``. The
                    # backoff + retry-count protect against infinite
                    # loops on genuinely corrupt shards: a real
                    # corruption fails identically all 3 attempts and
                    # propagates after the loop. We previously gated
                    # this on ``"Failed to open file" in str(exc)``
                    # pinned against mlx 0.30.x's safetensors error
                    # message, but that substring is fragile across
                    # mlx versions — a phrasing change would silently
                    # stop the retry from firing and the training run
                    # would die on the first transient hiccup as if
                    # the retry logic weren't there at all. Widening
                    # the catch trades a small amount of redundant
                    # retry work on genuinely-broken shards for
                    # robustness against mlx error-message churn.
                    last_exc = exc
                    if attempt < _SHARD_OPEN_RETRIES - 1:
                        # Log every retry so an operator running an 8h
                        # training session can see degraded-storage
                        # behaviour as it happens instead of only on
                        # final failure. ``warning`` level so it
                        # surfaces under default log config without
                        # spamming under ``DEBUG``.
                        #
                        # Phrasing: we catch any ``RuntimeError`` from
                        # ``mx.load`` (transient I/O is the common
                        # case but the catch is intentionally wide —
                        # see comment above). The message names the
                        # likely cause without claiming it: a corrupt
                        # safetensors header or shape mismatch would
                        # also surface as ``RuntimeError`` and would
                        # fail identically all 3 attempts before
                        # propagating, so the retry is harmless either
                        # way.
                        logger.warning(
                            "iter_precomputed_shards: mx.load(%s) raised "
                            "RuntimeError on attempt %d/%d: %s — sleeping "
                            "%.1fs before retry (likely transient I/O "
                            "pressure; a persistent failure will propagate "
                            "after %d attempts)",
                            shard_path,
                            attempt + 1,
                            _SHARD_OPEN_RETRIES,
                            exc,
                            _SHARD_OPEN_BACKOFF_S * (attempt + 1),
                            _SHARD_OPEN_RETRIES,
                        )
                        time.sleep(_SHARD_OPEN_BACKOFF_S * (attempt + 1))
            if tensors is None:
                # Log at ERROR before raising so an operator parsing
                # CI/training logs can distinguish a persistent failure
                # (this message) from the transient hiccups the per-
                # retry WARNING surfaces. Same ``last_exc`` context is
                # also preserved via ``raise ... from last_exc``.
                logger.error(
                    "iter_precomputed_shards: persistent failure loading "
                    "shard %s after %d attempts; last exception: %s",
                    shard_path,
                    _SHARD_OPEN_RETRIES,
                    last_exc,
                )
                raise RuntimeError(
                    f"Failed to load shard {shard_path} after "
                    f"{_SHARD_OPEN_RETRIES} attempts (see prior warnings "
                    "for the underlying ``RuntimeError``). Common causes: "
                    "sustained macOS mmap/fd pressure on the I/O subsystem, "
                    "a corrupt safetensors header, or a shape mismatch in "
                    "the shard payload. The retry loop has exhausted; "
                    "reduce concurrent workload, re-run precompute, or "
                    "check the shard directory for damage."
                ) from last_exc
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
