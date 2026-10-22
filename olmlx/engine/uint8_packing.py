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

Crash safety: config.json is updated first (overrides are derived from shapes,
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


def _read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
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


def _bits_overrides(packed: dict, group_size: int, default_bits: int) -> dict:
    overrides = {}
    for module, (dtype, w_shape, s_shape) in packed.items():
        in_features = s_shape[-1] * group_size
        bits, rem = divmod(w_shape[-1] * _BYTES_PER_ELEM[dtype] * 8, in_features)
        if rem or bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(
                f"Cannot infer bit width for {module}: packed {w_shape} {dtype}, "
                f"scales {s_shape}, group_size {group_size}"
            )
        if bits != default_bits:
            overrides[module] = {"group_size": group_size, "bits": bits}
    return overrides


def normalize_uint8_packing_in_place(model_dir: str | Path) -> bool:
    """Rewrite uint8-packed quantized weights as uint32. Returns True if anything changed."""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    for stale in model_dir.glob(f"*{_TMP_SUFFIX}"):
        stale.unlink()  # left by an interrupted rewrite; the original shard is intact
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards or not config_path.exists():
        return False
    headers = {p: _read_header(p) for p in shards}
    packed = _packed_weights(headers)
    u8_modules = {m for m, (dtype, _, _) in packed.items() if dtype == "U8"}
    if not u8_modules:
        return False

    cfg = json.loads(config_path.read_text())
    quant = dict(cfg.get("quantization") or cfg.get("quantization_config") or {})
    group_size, default_bits = quant.get("group_size", 64), quant.get("bits", 4)
    overrides = _bits_overrides(packed, group_size, default_bits)
    if any(quant.get(m) != q for m, q in overrides.items()):
        quant.update(overrides)
        cfg["quantization"] = quant
        cfg["quantization_config"] = quant
        config_path.write_text(json.dumps(cfg, indent=2))
    logger.info(
        "Converting %d uint8-packed weights in %s to uint32 (%d bit-width overrides)",
        len(u8_modules),
        model_dir,
        len(overrides),
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
        tmp = path.with_name(path.stem + _TMP_SUFFIX)
        mx.save_safetensors(str(tmp), weights, metadata=meta or {"format": "mlx"})
        os.replace(tmp, path)
        del weights
    return True
