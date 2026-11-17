"""Minimal read-only GGUF container reader.

Just enough of the format to read an Edge0 pack's skeleton, n-gram table and
vision projector: the key/value metadata and the tensor directory, with tensor
data served as zero-copy views of a memory map. Decoding the ggml block
formats lives in :mod:`olmlx.engine.edge0.ggml_quants`.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from olmlx.engine.edge0 import ggml_quants

_MAGIC = b"GGUF"
_DEFAULT_ALIGNMENT = 32
_SUPPORTED_VERSIONS = (2, 3)

# GGUF metadata value types.
_SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_TYPE_STRING = 8
_TYPE_ARRAY = 9


@dataclass(frozen=True)
class GGUFTensor:
    """One entry of the tensor directory."""

    name: str
    #: Outermost dimension first (numpy order; GGUF stores it reversed).
    shape: tuple[int, ...]
    ggml_type: int
    #: Absolute byte offset of the tensor data in the file.
    offset: int
    nbytes: int


class _Cursor:
    def __init__(self, buf: mmap.mmap):
        self._buf = buf
        self.pos = 0

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        if self.pos + size > len(self._buf):
            raise ValueError("truncated GGUF header")
        (value,) = struct.unpack_from(fmt, self._buf, self.pos)
        self.pos += size
        return value

    def string(self) -> str:
        length = self.unpack("<Q")
        if self.pos + length > len(self._buf):
            raise ValueError("truncated GGUF header")
        text = self._buf[self.pos : self.pos + length].decode("utf-8", "replace")
        self.pos += length
        return text

    def value(self, value_type: int) -> Any:
        if value_type == _TYPE_STRING:
            return self.string()
        if value_type == _TYPE_ARRAY:
            item_type = self.unpack("<I")
            count = self.unpack("<Q")
            return [self.value(item_type) for _ in range(count)]
        try:
            return self.unpack(_SCALAR_FORMATS[value_type])
        except KeyError:
            raise ValueError(f"unknown GGUF metadata type {value_type}") from None


class GGUFFile:
    """A memory-mapped GGUF file: ``metadata`` dict + ``tensors`` directory."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._file = open(self.path, "rb")
        try:
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._file.close()
            raise
        try:
            self.metadata, self.tensors = self._parse()
        except Exception:
            self.close()
            raise

    def _parse(self) -> tuple[dict[str, Any], dict[str, GGUFTensor]]:
        if self._map[:4] != _MAGIC:
            raise ValueError(f"{self.path} is not a GGUF file")
        cursor = _Cursor(self._map)
        cursor.pos = 4
        version = cursor.unpack("<I")
        if version not in _SUPPORTED_VERSIONS:
            raise ValueError(f"{self.path}: unsupported GGUF version {version}")
        tensor_count = cursor.unpack("<Q")
        kv_count = cursor.unpack("<Q")

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = cursor.string()
            metadata[key] = cursor.value(cursor.unpack("<I"))

        directory = []
        for _ in range(tensor_count):
            name = cursor.string()
            n_dims = cursor.unpack("<I")
            dims = [cursor.unpack("<Q") for _ in range(n_dims)]
            ggml_type = cursor.unpack("<I")
            relative_offset = cursor.unpack("<Q")
            directory.append((name, tuple(reversed(dims)), ggml_type, relative_offset))

        alignment = int(metadata.get("general.alignment", _DEFAULT_ALIGNMENT))
        data_start = cursor.pos + (-cursor.pos) % alignment

        tensors: dict[str, GGUFTensor] = {}
        for name, shape, ggml_type, relative_offset in directory:
            per_block, block_bytes = ggml_quants.block_geometry(ggml_type)
            count = int(np.prod(shape, dtype=np.int64)) if shape else 1
            if count % per_block:
                raise ValueError(
                    f"{self.path}: tensor {name!r} has {count} values, not a "
                    f"multiple of its block size {per_block}"
                )
            nbytes = count // per_block * block_bytes
            offset = data_start + relative_offset
            if offset + nbytes > len(self._map):
                raise ValueError(
                    f"{self.path}: tensor {name!r} extends past the end of the file"
                )
            tensors[name] = GGUFTensor(name, shape, ggml_type, offset, nbytes)
        return metadata, tensors

    def raw(self, name: str) -> np.ndarray:
        """Zero-copy ``uint8`` view of a tensor's stored bytes."""
        tensor = self.tensors[name]
        return np.frombuffer(
            self._map, dtype=np.uint8, count=tensor.nbytes, offset=tensor.offset
        )

    def load(self, name: str) -> np.ndarray:
        """Decode a tensor to float32 in its numpy shape."""
        tensor = self.tensors[name]
        return ggml_quants.dequantize(self.raw(name), tensor.ggml_type).reshape(
            tensor.shape
        )

    def close(self) -> None:
        try:
            self._map.close()
        except (BufferError, ValueError):
            # A live ``raw()`` view still exports the buffer; the map is
            # released when the last view is collected.
            pass
        self._file.close()

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
