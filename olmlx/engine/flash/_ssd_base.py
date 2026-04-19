"""Shared infrastructure for Flash SSD-streamed weight stores.

This module is internal to the ``olmlx.engine.flash`` package. It exposes
three reusable pieces:

* ``full_pread`` — a pread helper that retries on short reads.
* ``HeaderSpec`` + ``encode_header`` / ``parse_header`` — a magic/version
  header codec parameterised by struct format; concrete bundler modules
  hold module-level HeaderSpec constants.
* ``LayerLruCache[K, V]`` — a thread-safe per-layer LRU cache used by both
  the dense (``FlashWeightStore``) and MoE (``FlashMoeWeightStore``)
  paths.

The two weight stores share enough I/O plumbing that a single point of
failure is worth maintaining; the bundle bodies and cached value types
differ and are *not* abstracted here.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, Hashable, Sequence, TypeVar

logger = logging.getLogger(__name__)

# macOS fcntl command to bypass OS page cache (for accurate flash benchmarks).
F_NOCACHE = 48


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def full_pread(fd: int, size: int, offset: int) -> bytes:
    """Read exactly *size* bytes from *fd* at *offset*, retrying on short reads."""
    buf = bytearray()
    pos = offset
    remaining = size
    while remaining > 0:
        chunk = os.pread(fd, remaining, pos)
        if not chunk:
            raise OSError(f"Unexpected EOF: wanted {size} bytes at offset {offset}")
        buf.extend(chunk)
        pos += len(chunk)
        remaining -= len(chunk)
    return bytes(buf)


def open_fds(
    layer_paths: dict[int, os.PathLike | str], *, bypass_cache: bool
) -> dict[int, int]:
    """Open read-only fds for each layer file, optionally with F_NOCACHE on macOS.

    On error, any fds already opened are closed before the exception propagates.
    """
    if bypass_cache and sys.platform != "darwin":
        logger.warning(
            "bypass_cache is only supported on macOS (F_NOCACHE); "
            "OS page cache will not be bypassed on %s",
            sys.platform,
        )
    fds: dict[int, int] = {}
    try:
        for layer_idx, path in layer_paths.items():
            fd = os.open(str(path), os.O_RDONLY)
            if bypass_cache and sys.platform == "darwin":
                import fcntl

                fcntl.fcntl(fd, F_NOCACHE, 1)
            fds[layer_idx] = fd
    except Exception:
        for fd in fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    return fds


def close_fds(fds: dict[int, int]) -> None:
    """Close all file descriptors in the mapping, ignoring close errors."""
    for fd in fds.values():
        try:
            os.close(fd)
        except OSError:
            pass
    fds.clear()


# ---------------------------------------------------------------------------
# Header codec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeaderSpec:
    """Describes the wire format of a bundle file header.

    ``body_format`` is the struct format for the payload fields *after* the
    fixed ``magic (I)`` and ``version (I)`` prefix. The encoder zero-pads
    the remainder of the buffer out to ``size`` bytes.
    """

    magic: int
    version: int
    size: int
    body_format: str

    @property
    def body_struct(self) -> struct.Struct:
        return struct.Struct(self.body_format)


# Magic + version are always little-endian uint32s at the head of the buffer.
_PREFIX = struct.Struct("<II")


def encode_header(spec: HeaderSpec, *values: object) -> bytes:
    """Pack magic + version + *values*, then zero-pad to ``spec.size``."""
    body = spec.body_struct.pack(*values)
    head = _PREFIX.pack(spec.magic, spec.version) + body
    if len(head) > spec.size:
        raise ValueError(
            f"Header payload ({len(head)} bytes) exceeds spec.size ({spec.size})"
        )
    return head + b"\x00" * (spec.size - len(head))


def parse_header(spec: HeaderSpec, data: bytes) -> tuple[object, ...]:
    """Verify magic + version and unpack the body. Returns the body tuple."""
    if len(data) < spec.size:
        raise ValueError(
            f"Header buffer too short: got {len(data)} bytes, need {spec.size}"
        )
    magic, version = _PREFIX.unpack_from(data, 0)
    if magic != spec.magic:
        raise ValueError(
            f"Invalid bundle magic: expected {spec.magic:#010x}, got {magic:#010x}"
        )
    if version != spec.version:
        raise ValueError(
            f"Unsupported bundle version: expected {spec.version}, got {version}"
        )
    body_size = spec.body_struct.size
    body_end = _PREFIX.size + body_size
    return spec.body_struct.unpack(data[_PREFIX.size : body_end])


# ---------------------------------------------------------------------------
# Layer LRU cache
# ---------------------------------------------------------------------------


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LayerLruCache(Generic[K, V]):
    """Thread-safe per-layer LRU cache.

    Backs both ``FlashWeightStore`` (K=neuron index, V=(gate, up, down))
    and ``FlashMoeWeightStore`` (K=expert index, V=component dict).
    """

    def __init__(self, max_per_layer: int):
        self._max = max_per_layer
        self._lock = threading.Lock()
        self._cache: dict[int, OrderedDict[K, V]] = {}

    def get(self, layer_idx: int, key: K) -> V | None:
        with self._lock:
            layer_cache = self._cache.get(layer_idx)
            if layer_cache is None or key not in layer_cache:
                return None
            layer_cache.move_to_end(key)
            return layer_cache[key]

    def put(self, layer_idx: int, key: K, value: V) -> None:
        if self._max <= 0:
            return
        with self._lock:
            layer_cache = self._cache.setdefault(layer_idx, OrderedDict())
            layer_cache[key] = value
            layer_cache.move_to_end(key)
            while len(layer_cache) > self._max:
                layer_cache.popitem(last=False)

    def get_batch(self, layer_idx: int, keys: Sequence[K]) -> dict[K, V]:
        """Return a dict with cached entries for *keys* (refreshes their LRU order)."""
        with self._lock:
            result: dict[K, V] = {}
            layer_cache = self._cache.get(layer_idx)
            if layer_cache is None:
                return result
            for k in keys:
                if k in layer_cache:
                    layer_cache.move_to_end(k)
                    result[k] = layer_cache[k]
            return result

    def get_cached_indices(
        self, layer_idx: int, keys: Sequence[K]
    ) -> tuple[list[K], list[K]]:
        """Return ``(cached, missing)`` — does not update LRU order."""
        with self._lock:
            layer_cache = self._cache.get(layer_idx)
            if layer_cache is None:
                return [], list(keys)
            cached: list[K] = []
            missing: list[K] = []
            for k in keys:
                (cached if k in layer_cache else missing).append(k)
            return cached, missing
