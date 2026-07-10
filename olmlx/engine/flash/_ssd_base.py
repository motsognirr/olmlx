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
from typing import Any, Generic, Hashable, Sequence, TypeVar

logger = logging.getLogger(__name__)

# macOS fcntl command to bypass OS page cache (for accurate flash benchmarks).
F_NOCACHE = 48


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def full_pread(fd: int, size: int, offset: int) -> bytes:
    """Read exactly *size* bytes from *fd* at *offset*, retrying on short reads."""
    chunk = os.pread(fd, size, offset)
    if len(chunk) == size:
        return chunk
    if not chunk:
        raise OSError(f"Unexpected EOF: wanted {size} bytes at offset {offset}")
    buf = bytearray(chunk)
    pos = offset + len(chunk)
    remaining = size - len(chunk)
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
    apply_nocache = bypass_cache and sys.platform == "darwin"
    if bypass_cache and not apply_nocache:
        logger.warning(
            "bypass_cache is only supported on macOS (F_NOCACHE); "
            "OS page cache will not be bypassed on %s",
            sys.platform,
        )
    if apply_nocache:
        import fcntl

    fds: dict[int, int] = {}
    try:
        for layer_idx, path in layer_paths.items():
            fd = os.open(str(path), os.O_RDONLY)
            # Register the fd before any further call that might raise, so the
            # except-block cleanup below is guaranteed to close it.
            fds[layer_idx] = fd
            if apply_nocache:
                # fcntl is imported above when apply_nocache is True
                fcntl.fcntl(fd, F_NOCACHE, 1)  # pyright: ignore[reportPossiblyUnboundVariable]
    except Exception:
        for fd in fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    return fds


def close_fds(fds: dict[int, int]) -> None:
    """Close all file descriptors in the mapping and clear it.

    Clearing the mapping prevents a second close attempt when ``close()`` is
    followed by ``__del__`` on the same store.
    """
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

    def __post_init__(self) -> None:
        # Compile once; HeaderSpec is frozen, so body_struct is a constant property.
        object.__setattr__(self, "_body_struct", struct.Struct(self.body_format))

    @property
    def body_struct(self) -> struct.Struct:
        return self._body_struct  # type: ignore[attr-defined]


# Magic + version are always little-endian uint32s at the head of the buffer.
_PREFIX = struct.Struct("<II")


def encode_header(spec: HeaderSpec, *values: Any) -> bytes:
    """Pack magic + version + *values*, then zero-pad to ``spec.size``."""
    body = spec.body_struct.pack(*values)
    head = _PREFIX.pack(spec.magic, spec.version) + body
    if len(head) > spec.size:
        raise ValueError(
            f"Header payload ({len(head)} bytes) exceeds spec.size ({spec.size})"
        )
    return head + b"\x00" * (spec.size - len(head))


def parse_header(spec: HeaderSpec, data: bytes) -> tuple[Any, ...]:
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


class ScoredLayerCache(LayerLruCache[K, V]):
    """LRU cache whose eviction victim can be steered by predicted-need scores.

    Used by the MoE store: a lookahead predictor pushes per-expert scores via
    ``set_scores`` before its prefetch I/O lands; on overflow the victim is
    the lowest-scored non-protected key instead of the LRU-oldest. Behaves as
    a plain ``LayerLruCache`` for layers without scores, so it can back the
    store unconditionally.

    Scores are consume-once: the store clears them when the layer's forward
    pass loads its experts, so a prediction from a previous token can never
    steer a later eviction.

    Victim hierarchy on overflow:

    1. Among keys that are non-protected AND not the just-inserted key:
       lowest ``scores.get(k, 0.0)`` when scores are set, else LRU-oldest.
       Ties resolve to LRU-oldest.
    2. If no such key exists and the just-inserted key is not protected,
       the just-inserted key itself is evicted (a wasted insert, but
       protected keys survive).
    3. If everything including the just-inserted key is protected, the
       LRU-oldest key overall is evicted — the cache must not grow
       unbounded.
    """

    def __init__(self, max_per_layer: int):
        super().__init__(max_per_layer)
        # Guarded by the parent class's self._lock.
        self._scores: dict[int, dict[K, float]] = {}
        self._protected: dict[int, set[K]] = {}

    def set_scores(self, layer_idx: int, scores: dict[K, float]) -> None:
        """Replace the predicted-need scores for *layer_idx*."""
        with self._lock:
            self._scores[layer_idx] = scores

    def clear_scores(self, layer_idx: int) -> None:
        """Drop scores for *layer_idx* (consume-once staleness guard)."""
        with self._lock:
            self._scores.pop(layer_idx, None)

    def protect(self, layer_idx: int, keys: set[K]) -> None:
        """Replace the eviction-protected set for *layer_idx* (in-flight experts)."""
        with self._lock:
            self._protected[layer_idx] = keys

    def put(self, layer_idx: int, key: K, value: V) -> None:
        if self._max <= 0:
            return
        with self._lock:
            layer_cache = self._cache.setdefault(layer_idx, OrderedDict())
            is_new_key = key not in layer_cache
            layer_cache[key] = value
            layer_cache.move_to_end(key)
            while len(layer_cache) > self._max:
                victim = self._pick_victim(
                    layer_idx, layer_cache, new_key=key if is_new_key else None
                )
                del layer_cache[victim]

    def _pick_victim(
        self,
        layer_idx: int,
        layer_cache: "OrderedDict[K, V]",
        new_key: K | None = None,
    ) -> K:
        """Choose the eviction victim. Caller holds self._lock.

        Implements the victim hierarchy documented on the class.
        """
        protected = self._protected.get(layer_idx, set())
        candidates = [k for k in layer_cache if k not in protected and k != new_key]
        if candidates:
            scores = self._scores.get(layer_idx)
            if not scores:
                return candidates[0]  # LRU-oldest non-protected (and non-new)
            # min() keeps the first (LRU-oldest) key on score ties.
            return min(candidates, key=lambda k: scores.get(k, 0.0))
        if new_key is not None and new_key not in protected:
            # Every pre-existing key is protected: the just-inserted key is the
            # victim of last resort (wasted insert, but protected keys survive).
            return new_key
        # Everything including the just-inserted key is protected: the cache
        # must not grow unbounded, so fall back to plain LRU-oldest.
        return next(iter(layer_cache))
