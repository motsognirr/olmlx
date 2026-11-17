"""Memory-mapped FP8 n-gram (PLE) table for Edge0 packs.

Qwen4-Exp's per-layer embedding reads 16 hashed n-gram rows per token from a
~320M-row table. An Edge0 pack ships it as one int8 GGUF tensor of FP8
(E4M3) bytes plus a single float scale — 48 GiB that must never be loaded.
mlx-vlm can serve the table by row lookup, but only from its own MLX-affine
layout (``QuantizedMMapNGramEmbedding``); this module provides the same
interface over the pack's bytes and teaches mlx-vlm to pick it from the
manifest.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import mlx.core as mx
import numpy as np

MANIFEST_FORMAT = "edge0_fp8"
MANIFEST_NAME = "edge0_ple.json"

_install_lock = threading.Lock()


def _e4m3_table() -> np.ndarray:
    """All 256 FP8 E4M3 values (the two NaN encodings read as 0)."""
    table = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        sign, exponent, mantissa = byte >> 7, (byte >> 3) & 15, byte & 7
        if exponent == 15 and mantissa == 7:
            value = 0.0
        elif exponent == 0:
            value = math.ldexp(mantissa / 8.0, -6)
        else:
            value = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
        table[byte] = -value if sign else value
    return table


E4M3_TABLE = _e4m3_table()


def write_manifest(
    path: Path,
    *,
    data_file: str,
    data_offset: int,
    row_count: int,
    row_width: int,
    scale: float,
) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "encoding": "f8_e4m3",
                "data_file": data_file,
                "data_offset": int(data_offset),
                "row_count": int(row_count),
                "row_width": int(row_width),
                "scale": float(scale),
            },
            indent=2,
        )
        + "\n"
    )


class Fp8NGramEmbedding:
    """Read-only FP8 table served by mmap row lookup.

    Call-compatible with mlx-vlm's ``QuantizedMMapNGramEmbedding``:
    ``table(row_ids) -> [..., row_width]`` bfloat16, plus ``row_count`` /
    ``row_width``.
    """

    def __init__(self, manifest_path: Path | str, *, cache_rows=None):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"{manifest_path} is not an {MANIFEST_FORMAT} manifest")
        if manifest.get("encoding") != "f8_e4m3":
            raise ValueError(f"unsupported PLE encoding {manifest.get('encoding')!r}")
        self.row_count = int(manifest["row_count"])
        self.row_width = int(manifest["row_width"])
        # The manifest sits in a downloaded directory: never let it name a
        # file outside that directory.
        root = manifest_path.parent.resolve()
        data_path = (root / str(manifest["data_file"])).resolve()
        if not data_path.is_relative_to(root):
            raise ValueError(
                f"PLE manifest data_file {manifest['data_file']!r} escapes the "
                "model directory"
            )
        offset = int(manifest["data_offset"])
        expected = offset + self.row_count * self.row_width
        if data_path.stat().st_size < expected:
            raise ValueError(
                f"{data_path.name} is smaller than its {self.row_count} rows "
                "(incomplete download?)"
            )
        self._rows = np.memmap(
            data_path,
            dtype=np.uint8,
            mode="r",
            offset=offset,
            shape=(self.row_count, self.row_width),
        )
        self._values = E4M3_TABLE * np.float32(manifest["scale"])

    def __call__(self, row_ids) -> mx.array:
        ids = np.asarray(row_ids).astype(np.int64, copy=False)
        if ids.size and (ids.min() < 0 or ids.max() >= self.row_count):
            raise IndexError("n-gram row outside the PLE table")
        flat = ids.reshape(-1)
        if not flat.size:
            return mx.zeros((*ids.shape, self.row_width), dtype=mx.bfloat16)
        # Read each distinct row once, in file order.
        unique, inverse = np.unique(flat, return_inverse=True)
        # Expand to request order while the rows are still bytes, so the
        # float32 values are materialized once rather than twice.
        values = self._values[self._rows[unique][inverse]]
        return mx.array(values).astype(mx.bfloat16).reshape(*ids.shape, self.row_width)

    lookup = __call__


def install_mlx_vlm_hook() -> None:
    """Make mlx-vlm build :class:`Fp8NGramEmbedding` for ``edge0_fp8`` manifests.

    ``Qwen4ExpNGramEmbedding`` imports ``QuantizedMMapNGramEmbedding`` from
    ``ple_storage`` when it is constructed, so replacing that name with a
    dispatching subclass is enough. Any other manifest still gets mlx-vlm's own class.
    Idempotent.
    """
    from mlx_vlm.models.qwen4_exp import ple_storage

    with _install_lock:
        original = ple_storage.QuantizedMMapNGramEmbedding
        if getattr(original, "_olmlx_edge0_dispatch", False):
            return

        # A subclass, not a factory function: mlx-vlm also reads class
        # attributes (``MANIFEST_VERSION``) off this name.
        class _DispatchingNGramEmbedding(original):  # type: ignore[misc, valid-type]
            _olmlx_edge0_dispatch = True

            def __new__(cls, manifest_path, *, cache_rows=None):
                try:
                    fmt = json.loads(Path(manifest_path).read_text()).get("format")
                except (OSError, ValueError, AttributeError, TypeError):
                    fmt = None
                if fmt == MANIFEST_FORMAT:
                    # Not an instance of ``cls``, so ``__init__`` is skipped.
                    return Fp8NGramEmbedding(manifest_path, cache_rows=cache_rows)
                return super().__new__(cls)

        _DispatchingNGramEmbedding.__name__ = original.__name__
        _DispatchingNGramEmbedding.__qualname__ = original.__qualname__
        ple_storage.QuantizedMMapNGramEmbedding = _DispatchingNGramEmbedding  # type: ignore[misc]
