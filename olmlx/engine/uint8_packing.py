"""Normalize uint8-packed MLX quantized checkpoints to stock mlx-lm layout.

Some exports (e.g. ``RecursiveSI/flint-int4``) store affine-quantized weights
packed in ``uint8`` instead of mlx's ``uint32``, and leave per-module bit
widths (INT8 embeddings next to INT4 projections) out of ``config.json``.
mlx-lm then fails with a shape mismatch on the first such parameter.

The bytes are identical either way (little-endian, low bits first), so the fix
is a dtype *view*: rewrite each affected shard with ``uint32`` weights and add
the per-module ``bits`` overrides the shapes imply. Done once, in place, before
the first load or Flash-MoE prepare; later calls find no ``uint8`` weights and
return immediately after reading the shard headers.

Crash safety: config.json is updated first (atomically) (overrides are derived from shapes,
which read the same for uint8 and uint32), then each shard is replaced
atomically. A rerun after an interruption converts whatever shards remain.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

_BYTES_PER_ELEM = {"U8": 1, "U32": 4}
# mx.save_safetensors appends ".safetensors" to any other name.
_TMP_SUFFIX = ".u32tmp.safetensors"
_LOCK_NAME = ".uint8-normalize.lock"


def _read_header(path: Path) -> dict | None:
    """Shard header, or None if the file isn't a readable safetensors shard.

    A bad shard is not this module's to report: skipping it lets the loader
    raise its own error for it.
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            if n > size - 8:
                return None
            header = json.loads(f.read(n))
    except (OSError, struct.error, ValueError):
        return None
    if not isinstance(header, dict):
        return None
    header.pop("__metadata__", None)
    return header


def _packed_weights(headers: dict[Path, dict]) -> dict[str, tuple[str, list, list]]:
    """Map module path -> (weight dtype, weight shape, scales shape) for quantized weights."""
    scales = {
        name[: -len(".scales")]: info["shape"]
        for h in headers.values()
        for name, info in h.items()
        if name.endswith(".scales")
    }
    out = {}
    for h in headers.values():
        for name, info in h.items():
            if not name.endswith(".weight"):
                continue
            module = name[: -len(".weight")]
            if module in scales and info["dtype"] in _BYTES_PER_ELEM:
                out[module] = (info["dtype"], info["shape"], scales[module])
    return out


class PackedWeightsError(RuntimeError):
    """A uint8-packed checkpoint whose layout can't be normalized.

    Deliberately not a ValueError: load paths treat ValueError from mlx-lm as
    "unsupported model type" and fall back to the VLM loader.
    """


def _bits_overrides(packed: dict, quant: dict) -> dict:
    """Per-module overrides the packed shapes imply, relative to *quant*.

    A module's existing override supplies its group_size; only modules whose
    inferred width differs from what the config already says are returned.
    """
    default_gs, default_bits = quant.get("group_size", 64), quant.get("bits", 4)
    overrides = {}
    for module, (dtype, w_shape, s_shape) in packed.items():
        current = quant.get(module)
        current = current if isinstance(current, dict) else {}
        group_size = current.get("group_size", default_gs)
        in_features = s_shape[-1] * group_size
        bits, rem = divmod(w_shape[-1] * _BYTES_PER_ELEM[dtype] * 8, in_features)
        if rem or bits not in (2, 3, 4, 5, 6, 8):
            raise PackedWeightsError(
                f"Cannot infer bit width for {module}: packed {w_shape} {dtype}, "
                f"scales {s_shape}, group_size {group_size}"
            )
        if bits != current.get("bits", default_bits):
            # Merge: keep any other per-module keys (e.g. mode) the config has.
            overrides[module] = {**current, "group_size": group_size, "bits": bits}
    return overrides


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _scan(model_dir: Path) -> tuple[dict[Path, dict], dict, set[str]]:
    shards = sorted(
        p for p in model_dir.glob("*.safetensors") if not p.name.endswith(_TMP_SUFFIX)
    )
    headers = {p: h for p in shards if (h := _read_header(p)) is not None}
    packed = _packed_weights(headers)
    u8_modules = {m for m, (dtype, _, _) in packed.items() if dtype == "U8"}
    return headers, packed, u8_modules


def normalize_uint8_packing_in_place(model_dir: str | Path) -> bool:
    """Rewrite uint8-packed quantized weights as uint32. Returns True if anything changed.

    Concurrent callers (two first loads, or a load racing Flash-MoE prepare) are
    serialized by an flock on ``model_dir/.uint8-normalize.lock``; whoever gets
    it second re-scans and finds nothing left to do.
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists() or not _scan(model_dir)[2]:
        return False  # fast path: headers only, no lock

    import fcntl

    try:
        lock = open(model_dir / _LOCK_NAME, "w")
    except OSError as exc:
        logger.warning(
            "%s has uint8-packed weights but can't be normalized in place (%s); "
            "loading will likely fail with a shape mismatch",
            model_dir,
            exc,
        )
        return False
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _normalize_locked(model_dir, config_path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _normalize_locked(model_dir: Path, config_path: Path) -> bool:
    for stale in [*model_dir.glob(f"*{_TMP_SUFFIX}"), model_dir / "config.json.tmp"]:
        # Left by an interrupted rewrite; the original file is intact.
        stale.unlink(missing_ok=True)
    headers, packed, u8_modules = _scan(model_dir)
    if not u8_modules:
        return False  # another caller finished while we waited for the lock

    try:
        cfg = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise PackedWeightsError(f"Unreadable {config_path}: {exc}") from exc
    quant = dict(cfg.get("quantization") or cfg.get("quantization_config") or {})
    # Only the modules this pass converts: uint32 ones are already loadable as
    # configured (or were covered by the config write of an interrupted pass).
    overrides = _bits_overrides({m: packed[m] for m in u8_modules}, quant)
    if overrides:
        # Shapes give bits x group_size, not each separately: a module quantized
        # with an undeclared group_size reads as a different bit width. Surface
        # what was inferred so a wrong guess is visible in the log.
        logger.warning(
            "Inferred per-module quantization for %s from packed shapes "
            "(assumes declared group sizes): %s",
            model_dir,
            overrides,
        )
        quant.update(overrides)
        cfg["quantization"] = quant
        cfg["quantization_config"] = quant
        _write_json_atomic(config_path, cfg)
    logger.info(
        "Converting %d uint8-packed weights in %s to uint32",
        len(u8_modules),
        model_dir,
    )

    import mlx.core as mx

    for path, header in headers.items():
        names = {f"{m}.weight" for m in u8_modules} & header.keys()
        if not names:
            continue
        weights, meta = mx.load(str(path), return_metadata=True)
        weights = {
            k: v.view(mx.uint32) if k in names else v for k, v in weights.items()
        }
        # Rewrite the real file: replacing a symlinked shard (HF hub cache
        # snapshots point into blobs/) would turn the link into a copy.
        target = path.resolve()
        tmp = target.with_name(target.stem + _TMP_SUFFIX)
        mx.save_safetensors(str(tmp), weights, metadata=meta or {"format": "mlx"})
        os.replace(tmp, target)
        del weights
    return True
