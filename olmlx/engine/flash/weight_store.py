"""Flash weight store: load neuron weights from SSD with caching and parallel I/O."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

from olmlx.engine.flash.bundler import (
    HEADER_SIZE,
    BundledLayerLayout,
    _DTYPE_BYTES,
    parse_header,
)


class NeuronCache:
    """Per-layer LRU cache for loaded neuron weight chunks."""

    def __init__(self, max_neurons_per_layer: int):
        self._max = max_neurons_per_layer
        # layer_idx -> OrderedDict[neuron_idx -> (gate, up, down)]
        self._cache: dict[
            int, OrderedDict[int, tuple[mx.array, mx.array, mx.array]]
        ] = {}

    def get(
        self, layer_idx: int, neuron_idx: int
    ) -> tuple[mx.array, mx.array, mx.array] | None:
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
        result = {}
        for idx in indices:
            data = self.get(layer_idx, idx)
            if data is not None:
                result[idx] = data
        return result


class FlashWeightStore:
    """Manages bundled FFN weights on SSD with parallel I/O and RAM caching."""

    def __init__(
        self,
        flash_dir: Path,
        num_io_threads: int = 32,
        cache_budget_neurons: int = 1024,
    ):
        self._flash_dir = flash_dir
        self._executor = ThreadPoolExecutor(max_workers=num_io_threads)
        self._cache = NeuronCache(max_neurons_per_layer=cache_budget_neurons)
        self._layouts = self._load_layouts()
        self._fds: dict[int, int] = {}
        for layer_idx, layout in self._layouts.items():
            self._fds[layer_idx] = os.open(str(layout.file_path), os.O_RDONLY)

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

    def _read_neuron(
        self, layer_idx: int, neuron_idx: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Read a single neuron's bundled weights from SSD using pread."""
        layout = self._layouts[layer_idx]
        fd = self._fds[layer_idx]
        offset = int(layout.offsets[neuron_idx])
        raw = os.pread(fd, layout.neuron_byte_size, offset)

        hidden = layout.hidden_size
        chunk = hidden * _DTYPE_BYTES[layout.dtype]

        gate_col = mx.array(np.frombuffer(raw[:chunk], dtype=np.float16).copy())
        up_col = mx.array(
            np.frombuffer(raw[chunk : 2 * chunk], dtype=np.float16).copy()
        )
        down_row = mx.array(
            np.frombuffer(raw[2 * chunk : 3 * chunk], dtype=np.float16).copy()
        )

        return gate_col, up_col, down_row

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
        # Check cache — returns dict of idx -> data for hits
        cached = self._cache.get_batch(layer_idx, neuron_indices)
        missing = [idx for idx in neuron_indices if idx not in cached]

        # Load missing neurons via parallel I/O
        if missing:
            futures = {
                idx: self._executor.submit(self._read_neuron, layer_idx, idx)
                for idx in missing
            }
            for idx, future in futures.items():
                data = future.result()
                cached[idx] = data
                self._cache.put(layer_idx, idx, data)

        # Assemble results in input order using the combined dict
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

    def __del__(self):
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._executor.shutdown(wait=False)
