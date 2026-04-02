"""Flash weight store: load neuron weights from SSD with caching and parallel I/O."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mlx.core as mx
import numpy as np

from olmlx.engine.flash.bundler import (
    HEADER_SIZE,
    BundledLayerLayout,
    _DTYPE_BYTES,
    parse_header,
)

logger = logging.getLogger(__name__)

_NP_DTYPE = {"float16": np.float16, "float32": np.float32, "bfloat16": np.uint16}

# MLX dtype for reinterpreting uint16 storage back to bfloat16
_MX_DTYPE = {"float16": mx.float16, "float32": mx.float32, "bfloat16": mx.bfloat16}

# macOS fcntl command to bypass OS page cache for accurate flash benchmarks
_F_NOCACHE = 48


class NeuronCache:
    """Thread-safe per-layer LRU cache for loaded neuron weight chunks."""

    def __init__(self, max_neurons_per_layer: int):
        self._max = max_neurons_per_layer
        self._lock = threading.Lock()
        # layer_idx -> OrderedDict[neuron_idx -> (gate, up, down)]
        self._cache: dict[
            int, OrderedDict[int, tuple[mx.array, mx.array, mx.array]]
        ] = {}

    def get(
        self, layer_idx: int, neuron_idx: int
    ) -> tuple[mx.array, mx.array, mx.array] | None:
        with self._lock:
            layer_cache = self._cache.get(layer_idx)
            if layer_cache is None:
                return None
            if neuron_idx not in layer_cache:
                return None
            layer_cache.move_to_end(neuron_idx)
            return layer_cache[neuron_idx]

    def put(
        self,
        layer_idx: int,
        neuron_idx: int,
        data: tuple[mx.array, mx.array, mx.array],
    ) -> None:
        if self._max <= 0:
            return
        with self._lock:
            if layer_idx not in self._cache:
                self._cache[layer_idx] = OrderedDict()
            layer_cache = self._cache[layer_idx]
            layer_cache[neuron_idx] = data
            layer_cache.move_to_end(neuron_idx)
            while len(layer_cache) > self._max:
                layer_cache.popitem(last=False)

    def get_batch(
        self, layer_idx: int, indices: list[int]
    ) -> dict[int, tuple[mx.array, mx.array, mx.array]]:
        """Return dict mapping neuron_idx -> data for cached neurons."""
        with self._lock:
            result = {}
            layer_cache = self._cache.get(layer_idx)
            if layer_cache is None:
                return result
            for idx in indices:
                if idx in layer_cache:
                    layer_cache.move_to_end(idx)
                    result[idx] = layer_cache[idx]
            return result

    def get_cached_indices(
        self, layer_idx: int, neuron_indices: list[int]
    ) -> tuple[list[int], list[int]]:
        """Return (cached_indices, missing_indices)."""
        with self._lock:
            layer_cache = self._cache.get(layer_idx)
            if layer_cache is None:
                return [], list(neuron_indices)
            cached = [idx for idx in neuron_indices if idx in layer_cache]
            missing = [idx for idx in neuron_indices if idx not in layer_cache]
            return cached, missing


class PreallocatedNeuronBuffer:
    """Pre-allocated DRAM buffer for neuron weights (Paper Section 3.3).

    Maintains fixed-size numpy arrays per gate/up/down projection.
    Uses swap-based eviction and append-based insertion to avoid
    per-call allocation (no mx.stack).
    """

    def __init__(
        self,
        max_neurons: int,
        hidden_size: int,
        dtype=np.float16,
        mx_dtype: mx.Dtype | None = None,
    ):
        self._max = max_neurons
        self._hidden = hidden_size
        self._gate = np.zeros((max_neurons, hidden_size), dtype=dtype)
        self._up = np.zeros((max_neurons, hidden_size), dtype=dtype)
        self._down = np.zeros((max_neurons, hidden_size), dtype=dtype)
        # mx dtype for reinterpretation (e.g. uint16 storage → bfloat16)
        self._mx_dtype = mx_dtype
        self._neuron_to_slot: dict[int, int] = {}
        self._num_used = 0
        # OrderedDict for O(1) LRU: neuron_idx -> None, oldest first
        self._access_order: OrderedDict[int, None] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        """Expose lock for callers that need atomic check-fetch-insert-read."""
        return self._lock

    @property
    def num_used(self) -> int:
        return self._num_used

    def contains(self, neuron_idx: int) -> bool:
        with self._lock:
            return neuron_idx in self._neuron_to_slot

    def insert(
        self,
        neuron_idx: int,
        gate_data: np.ndarray,
        up_data: np.ndarray,
        down_data: np.ndarray,
    ) -> int:
        """Insert a neuron, evicting LRU if full. Returns slot index."""
        with self._lock:
            # If already present, update in-place
            if neuron_idx in self._neuron_to_slot:
                slot = self._neuron_to_slot[neuron_idx]
                self._gate[slot] = gate_data
                self._up[slot] = up_data
                self._down[slot] = down_data
                self._access_order.move_to_end(neuron_idx)
                return slot

            if self._num_used < self._max:
                slot = self._num_used
                self._num_used += 1
            else:
                # Evict oldest (first item in OrderedDict)
                evict_idx, _ = self._access_order.popitem(last=False)
                slot = self._neuron_to_slot.pop(evict_idx)

            self._gate[slot] = gate_data
            self._up[slot] = up_data
            self._down[slot] = down_data
            self._neuron_to_slot[neuron_idx] = slot
            self._access_order[neuron_idx] = None
            return slot

    def get_matrices(
        self, neuron_indices: list[int]
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Return gate_cols, up_cols, down_rows for given neurons."""
        with self._lock:
            slots = [self._neuron_to_slot[idx] for idx in neuron_indices]
            for idx in neuron_indices:
                if idx in self._access_order:
                    self._access_order.move_to_end(idx)
            # Copy numpy data while lock is held to avoid races with concurrent inserts
            gate_data = self._gate[slots].T.copy()
            up_data = self._up[slots].T.copy()
            down_data = self._down[slots].copy()

        gate = mx.array(gate_data)
        up = mx.array(up_data)
        down = mx.array(down_data)
        # Reinterpret uint16 storage as bfloat16 (or other non-native numpy dtype)
        if self._mx_dtype is not None:
            gate = gate.view(self._mx_dtype)
            up = up.view(self._mx_dtype)
            down = down.view(self._mx_dtype)
        return gate, up, down

    def get_cached_indices(
        self, neuron_indices: list[int]
    ) -> tuple[list[int], list[int]]:
        """Return (cached_indices, missing_indices)."""
        with self._lock:
            cached = [idx for idx in neuron_indices if idx in self._neuron_to_slot]
            missing = [idx for idx in neuron_indices if idx not in self._neuron_to_slot]
            return cached, missing


class FlashWeightStore:
    """Manages bundled FFN weights on SSD with parallel I/O and RAM caching."""

    def __init__(
        self,
        flash_dir: Path,
        num_io_threads: int = 32,
        cache_budget_neurons: int = 1024,
        bypass_cache: bool = False,
        use_preallocated_buffer: bool = False,
    ):
        self._flash_dir = flash_dir
        self._bypass_cache = bypass_cache
        self._use_preallocated = use_preallocated_buffer
        self._executor = ThreadPoolExecutor(max_workers=num_io_threads)
        self._cache: NeuronCache | None = None
        self._buffers: dict[int, PreallocatedNeuronBuffer] = {}
        self._layouts = self._load_layouts()

        if use_preallocated_buffer:
            for layer_idx, layout in self._layouts.items():
                np_dtype = _NP_DTYPE[layout.dtype]
                # Pass mx_dtype for reinterpretation when numpy can't represent
                # the model dtype natively (e.g. bfloat16 stored as uint16)
                needs_reinterpret = np_dtype != np.float16 and np_dtype != np.float32
                self._buffers[layer_idx] = PreallocatedNeuronBuffer(
                    max_neurons=cache_budget_neurons,
                    hidden_size=layout.hidden_size,
                    dtype=np_dtype,
                    mx_dtype=_MX_DTYPE[layout.dtype] if needs_reinterpret else None,
                )
        else:
            self._cache = NeuronCache(max_neurons_per_layer=cache_budget_neurons)
        if bypass_cache and sys.platform != "darwin":
            logger.warning(
                "bypass_cache is only supported on macOS (F_NOCACHE); "
                "OS page cache will not be bypassed on %s",
                sys.platform,
            )
        self._fds: dict[int, int] = {}
        try:
            for layer_idx, layout in self._layouts.items():
                fd = os.open(str(layout.file_path), os.O_RDONLY)
                if bypass_cache and sys.platform == "darwin":
                    import fcntl

                    fcntl.fcntl(fd, _F_NOCACHE, 1)
                self._fds[layer_idx] = fd
        except Exception:
            self.close()
            raise

    def _load_layouts(self) -> dict[int, BundledLayerLayout]:
        config_path = self._flash_dir / "flash_layout.json"
        config = json.loads(config_path.read_text())

        layouts = {}
        for layer_str, layer_info in config["layers"].items():
            layer_idx = int(layer_str)
            file_path = self._flash_dir / layer_info["file"]

            # Read header and offset table in a single open
            with open(file_path, "rb") as f:
                header = parse_header(f.read(HEADER_SIZE))
                num_neurons = header["num_neurons"]
                offset_table_size = num_neurons * 8
                offsets = np.frombuffer(
                    f.read(offset_table_size), dtype=np.uint64
                ).copy()

            hidden_size = header["hidden_size"]
            dtype = header["dtype"]
            dtype_bytes = _DTYPE_BYTES[dtype]
            neuron_byte_size = hidden_size * dtype_bytes * 3

            layouts[layer_idx] = BundledLayerLayout(
                layer_idx=layer_idx,
                num_neurons=num_neurons,
                hidden_size=hidden_size,
                neuron_byte_size=neuron_byte_size,
                file_path=file_path,
                offsets=offsets,
                dtype=dtype,
            )

        return layouts

    @staticmethod
    def _full_pread(fd: int, size: int, offset: int) -> bytes:
        """Read exactly *size* bytes via pread, retrying on short reads."""
        buf = b""
        pos = offset
        remaining = size
        while remaining > 0:
            chunk = os.pread(fd, remaining, pos)
            if not chunk:
                raise OSError(f"Unexpected EOF: wanted {size} bytes at offset {offset}")
            buf += chunk
            pos += len(chunk)
            remaining -= len(chunk)
        return buf

    def _read_neuron_raw(
        self, layer_idx: int, neuron_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read a single neuron's bundled weights from SSD as numpy arrays."""
        layout = self._layouts[layer_idx]
        fd = self._fds[layer_idx]
        offset = int(layout.offsets[neuron_idx])
        raw = self._full_pread(fd, layout.neuron_byte_size, offset)

        hidden = layout.hidden_size
        np_dtype = _NP_DTYPE[layout.dtype]
        chunk = hidden * _DTYPE_BYTES[layout.dtype]

        gate_col = np.frombuffer(raw[:chunk], dtype=np_dtype).copy()
        up_col = np.frombuffer(raw[chunk : 2 * chunk], dtype=np_dtype).copy()
        down_row = np.frombuffer(raw[2 * chunk : 3 * chunk], dtype=np_dtype).copy()

        return gate_col, up_col, down_row

    def _read_neuron(
        self, layer_idx: int, neuron_idx: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Read a single neuron's bundled weights from SSD as mx arrays."""
        gate, up, down = self._read_neuron_raw(layer_idx, neuron_idx)
        mx_gate, mx_up, mx_down = mx.array(gate), mx.array(up), mx.array(down)
        # Reinterpret uint16 storage as bfloat16 when needed
        target_dtype = _MX_DTYPE.get(self._layouts[layer_idx].dtype)
        if target_dtype is not None and mx_gate.dtype != target_dtype:
            mx_gate = mx_gate.view(target_dtype)
            mx_up = mx_up.view(target_dtype)
            mx_down = mx_down.view(target_dtype)
        return mx_gate, mx_up, mx_down

    def load_neurons(
        self,
        layer_idx: int,
        neuron_indices: list[int],
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Load gate_proj columns, up_proj columns, down_proj rows for given neurons.

        Returns:
            gate_cols: (hidden_size, len(neuron_indices))
            up_cols:   (hidden_size, len(neuron_indices))
            down_rows: (len(neuron_indices), hidden_size)
        """
        if self._use_preallocated:
            return self._load_neurons_preallocated(layer_idx, neuron_indices)
        return self._load_neurons_cache(layer_idx, neuron_indices)

    def _load_neurons_preallocated(
        self, layer_idx: int, neuron_indices: list[int]
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Load neurons using the preallocated buffer path.

        Determines cached/missing under the lock to prevent TOCTOU races,
        then releases for I/O, then re-acquires for insert + read.
        """
        buf = self._buffers[layer_idx]

        # Determine missing under lock to prevent concurrent eviction
        with buf.lock:
            _, missing = buf.get_cached_indices(neuron_indices)

        # Fetch missing neurons via parallel I/O (outside lock)
        loaded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if missing:
            futures = {
                idx: self._executor.submit(self._read_neuron_raw, layer_idx, idx)
                for idx in missing
            }
            for idx, future in futures.items():
                loaded[idx] = future.result()

        # Insert under lock, then check for eviction races
        with buf.lock:
            for idx, (gate, up, down) in loaded.items():
                buf.insert(idx, gate, up, down)
            _, still_missing = buf.get_cached_indices(neuron_indices)
            if not still_missing:
                return buf.get_matrices(neuron_indices)

        # Re-fetch evicted neurons outside the lock (rare path)
        extra = {idx: self._read_neuron_raw(layer_idx, idx) for idx in still_missing}
        with buf.lock:
            for idx, (gate, up, down) in extra.items():
                buf.insert(idx, gate, up, down)
            return buf.get_matrices(neuron_indices)

    def _load_neurons_cache(
        self, layer_idx: int, neuron_indices: list[int]
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Load neurons using the NeuronCache path (original)."""
        cached = self._cache.get_batch(layer_idx, neuron_indices)
        missing = [idx for idx in neuron_indices if idx not in cached]

        if missing:
            futures = {
                idx: self._executor.submit(self._read_neuron, layer_idx, idx)
                for idx in missing
            }
            for idx, future in futures.items():
                data = future.result()
                cached[idx] = data
                self._cache.put(layer_idx, idx, data)

        gate_cols = []
        up_cols = []
        down_rows = []
        for idx in neuron_indices:
            g, u, d = cached[idx]
            gate_cols.append(g)
            up_cols.append(u)
            down_rows.append(d)

        return (
            mx.stack(gate_cols, axis=1),
            mx.stack(up_cols, axis=1),
            mx.stack(down_rows, axis=0),
        )

    def get_cached_indices(
        self, layer_idx: int, neuron_indices: list[int]
    ) -> tuple[list[int], list[int]]:
        """Check which neurons are already cached without loading.

        Returns:
            (cached, missing) — two lists of neuron indices.
        """
        if self._use_preallocated:
            buf = self._buffers[layer_idx]
            with buf.lock:
                return buf.get_cached_indices(neuron_indices)
        assert self._cache is not None
        return self._cache.get_cached_indices(layer_idx, neuron_indices)

    def prefetch_neurons(
        self,
        layer_idx: int,
        neuron_indices: list[int],
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        """Load neurons into cache without returning matrices.

        Used by the speculative prefetcher to warm the cache in the background.
        An external *executor* can be passed to avoid contention with the main
        I/O thread pool; if ``None``, the store's own pool is used.
        """
        pool = executor or self._executor

        if self._use_preallocated:
            buf = self._buffers[layer_idx]
            with buf.lock:
                _, missing = buf.get_cached_indices(neuron_indices)
            if not missing:
                return
            futures = {
                pool.submit(self._read_neuron_raw, layer_idx, idx): idx
                for idx in missing
            }
            results = {}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
            with buf.lock:
                for idx, (gate, up, down) in results.items():
                    buf.insert(idx, gate, up, down)
        else:
            assert self._cache is not None
            _, missing = self._cache.get_cached_indices(layer_idx, neuron_indices)
            if not missing:
                return
            futures = {
                idx: pool.submit(self._read_neuron, layer_idx, idx) for idx in missing
            }
            for idx, future in futures.items():
                data = future.result()
                self._cache.put(layer_idx, idx, data)

    def close(self) -> None:
        """Release file descriptors and shut down the I/O thread pool."""
        self._executor.shutdown(wait=True)
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
