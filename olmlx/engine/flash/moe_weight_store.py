"""Flash MoE weight store: load expert weights from SSD with caching and parallel I/O."""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from olmlx.engine.flash.moe_bundler import (
    MOE_HEADER_SIZE,
    MoeExpertLayout,
    parse_moe_header,
)

# Map dtype string to numpy dtype (for parsing raw bytes)
_STR_TO_NP_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "uint32": np.uint32,
    "uint8": np.uint8,
    "int8": np.int8,
}

# Map dtype string to mlx dtype
_STR_TO_MX_DTYPE = {
    "float16": mx.float16,
    "float32": mx.float32,
    "uint32": mx.uint32,
    "uint8": mx.uint8,
    "int8": mx.int8,
}


@dataclass
class LoadedExperts:
    """Stacked expert weights for a subset of experts."""

    gate_weight: mx.array
    gate_scales: mx.array | None
    gate_biases: mx.array | None
    gate_bias: mx.array | None
    up_weight: mx.array
    up_scales: mx.array | None
    up_biases: mx.array | None
    up_bias: mx.array | None
    down_weight: mx.array
    down_scales: mx.array | None
    down_biases: mx.array | None
    down_bias: mx.array | None
    is_quantized: bool
    bits: int
    group_size: int
    quant_mode: str = "affine"
    expert_index_map: dict[int, int] = field(default_factory=dict)


class ExpertCache:
    """Thread-safe per-layer LRU cache for loaded expert data."""

    def __init__(self, max_experts_per_layer: int):
        self._max = max_experts_per_layer
        self._lock = threading.Lock()
        self._cache: dict[int, OrderedDict[int, dict]] = {}

    def get(self, layer_idx: int, expert_idx: int) -> dict | None:
        with self._lock:
            layer_cache = self._cache.get(layer_idx)
            if layer_cache is None or expert_idx not in layer_cache:
                return None
            layer_cache.move_to_end(expert_idx)
            return layer_cache[expert_idx]

    def put(self, layer_idx: int, expert_idx: int, data: dict) -> None:
        if self._max <= 0:
            return
        with self._lock:
            if layer_idx not in self._cache:
                self._cache[layer_idx] = OrderedDict()
            layer_cache = self._cache[layer_idx]
            layer_cache[expert_idx] = data
            layer_cache.move_to_end(expert_idx)
            while len(layer_cache) > self._max:
                layer_cache.popitem(last=False)

    def get_batch(self, layer_idx: int, indices: list[int]) -> dict[int, dict]:
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


class FlashMoeWeightStore:
    """Manages bundled MoE expert weights on SSD with parallel I/O and RAM caching."""

    def __init__(
        self,
        flash_dir: Path,
        num_io_threads: int = 32,
        cache_budget_experts: int = 48,
    ):
        self._flash_dir = flash_dir
        self._executor = ThreadPoolExecutor(max_workers=num_io_threads)
        self._cache = ExpertCache(max_experts_per_layer=cache_budget_experts)
        self._layout_config = json.loads(
            (flash_dir / "flash_moe_layout.json").read_text()
        )
        self._manifest = self._layout_config.get("component_manifest")
        self._quant_mode = self._layout_config.get("quant_mode", "affine")
        self._layouts = self._load_layouts()
        self._fds: dict[int, int] = {}
        try:
            for layer_idx, layout in self._layouts.items():
                self._fds[layer_idx] = os.open(str(layout.file_path), os.O_RDONLY)
        except Exception:
            self.close()
            raise

    def _load_layouts(self) -> dict[int, MoeExpertLayout]:
        layouts = {}
        for layer_str, layer_info in self._layout_config["layers"].items():
            layer_idx = int(layer_str)
            file_path = self._flash_dir / layer_info["file"]

            # Read header
            with open(file_path, "rb") as f:
                header = parse_moe_header(f.read(MOE_HEADER_SIZE))
                num_experts = header["num_experts"]
                offset_table_size = num_experts * 8
                offsets = np.frombuffer(
                    f.read(offset_table_size), dtype=np.uint64
                ).copy()

            layouts[layer_idx] = MoeExpertLayout(
                layer_idx=layer_idx,
                num_experts=header["num_experts"],
                hidden_size=header["hidden_size"],
                intermediate_size=header["intermediate_size"],
                expert_byte_size=header["expert_byte_size"],
                file_path=file_path,
                offsets=offsets,
                is_quantized=header["is_quantized"],
                bits=header["bits"],
                group_size=header["group_size"],
                quant_mode=self._quant_mode,
            )

        return layouts

    @staticmethod
    def _full_pread(fd: int, size: int, offset: int) -> bytes:
        """Read exactly *size* bytes via pread, retrying on short reads."""
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

    def _read_expert(self, layer_idx: int, expert_idx: int) -> dict:
        """Read a single expert's weights from SSD."""
        layout = self._layouts[layer_idx]
        fd = self._fds[layer_idx]
        offset = int(layout.offsets[expert_idx])
        raw = self._full_pread(fd, layout.expert_byte_size, offset)

        if self._manifest:
            return self._parse_expert_with_manifest(raw, self._manifest)
        elif layout.is_quantized:
            return self._parse_quantized_expert(raw, layout)
        else:
            return self._parse_float16_expert(
                raw, layout.hidden_size, layout.intermediate_size
            )

    @staticmethod
    def _parse_expert_with_manifest(raw: bytes, manifest: list[dict]) -> dict:
        """Parse expert from raw bytes using the component manifest."""
        result = {}
        pos = 0
        for entry in manifest:
            name = entry["name"]
            nbytes = entry["nbytes"]
            shape = entry["shape"]
            dtype_str = entry["dtype"]

            np_dtype = _STR_TO_NP_DTYPE.get(dtype_str)
            if np_dtype is None:
                raise ValueError(f"Unsupported dtype {dtype_str!r} in manifest")

            arr = np.frombuffer(raw[pos : pos + nbytes], dtype=np_dtype).copy()
            if shape:
                arr = arr.reshape(shape)

            # Map manifest name to dict key
            # e.g., "gate_proj.weight" -> key "gate_weight"
            # "gate_proj.scales" -> key "gate_scales"
            # "gate_proj.bias" -> key "gate_bias"
            proj, comp = name.split(".", 1)
            proj_short = proj.replace("_proj", "")
            key = f"{proj_short}_{comp}"

            result[key] = mx.array(arr)
            pos += nbytes

        # Ensure all expected keys exist (fill None for missing)
        for proj in ("gate", "up", "down"):
            for comp in ("weight", "scales", "biases", "bias"):
                key = f"{proj}_{comp}"
                if key not in result:
                    result[key] = None

        return result

    @staticmethod
    def _parse_float16_expert(raw: bytes, hidden: int, inter: int) -> dict:
        """Parse non-quantized expert from raw bytes (legacy, no manifest)."""
        gate_size = inter * hidden * 2
        up_size = inter * hidden * 2
        down_size = hidden * inter * 2

        pos = 0
        gate_w = mx.array(
            np.frombuffer(raw[pos : pos + gate_size], dtype=np.float16)
            .copy()
            .reshape(inter, hidden)
        )
        pos += gate_size

        up_w = mx.array(
            np.frombuffer(raw[pos : pos + up_size], dtype=np.float16)
            .copy()
            .reshape(inter, hidden)
        )
        pos += up_size

        down_w = mx.array(
            np.frombuffer(raw[pos : pos + down_size], dtype=np.float16)
            .copy()
            .reshape(hidden, inter)
        )

        return {
            "gate_weight": gate_w,
            "gate_scales": None,
            "gate_biases": None,
            "gate_bias": None,
            "up_weight": up_w,
            "up_scales": None,
            "up_biases": None,
            "up_bias": None,
            "down_weight": down_w,
            "down_scales": None,
            "down_biases": None,
            "down_bias": None,
        }

    @staticmethod
    def _parse_quantized_expert(raw: bytes, layout: MoeExpertLayout) -> dict:
        """Parse quantized expert from raw bytes (legacy, no manifest)."""
        hidden = layout.hidden_size
        inter = layout.intermediate_size
        bits = layout.bits
        group_size = layout.group_size

        gate_packed_dim = hidden * bits // 32
        down_packed_dim = inter * bits // 32

        result = {}
        pos = 0

        for proj, out_dim, packed_dim in [
            ("gate", inter, gate_packed_dim),
            ("up", inter, gate_packed_dim),
            ("down", hidden, down_packed_dim),
        ]:
            in_dim = hidden if proj != "down" else inter

            w_size = out_dim * packed_dim * 4
            w = np.frombuffer(raw[pos : pos + w_size], dtype=np.uint32).copy()
            w = w.reshape(out_dim, packed_dim)
            result[f"{proj}_weight"] = mx.array(w)
            pos += w_size

            s_dim = in_dim // group_size
            s_size = out_dim * s_dim * 2
            s = np.frombuffer(raw[pos : pos + s_size], dtype=np.float16).copy()
            s = s.reshape(out_dim, s_dim)
            result[f"{proj}_scales"] = mx.array(s)
            pos += s_size

            b_size = out_dim * s_dim * 2
            b = np.frombuffer(raw[pos : pos + b_size], dtype=np.float16).copy()
            b = b.reshape(out_dim, s_dim)
            result[f"{proj}_biases"] = mx.array(b)
            pos += b_size

            result[f"{proj}_bias"] = None

        return result

    def load_experts(
        self,
        layer_idx: int,
        expert_indices: list[int],
    ) -> LoadedExperts:
        """Load expert weights for given indices, using cache and parallel I/O."""
        layout = self._layouts[layer_idx]

        # Check cache
        cached = self._cache.get_batch(layer_idx, expert_indices)
        missing = [idx for idx in expert_indices if idx not in cached]

        # Load missing experts via parallel I/O
        if missing:
            futures = {
                idx: self._executor.submit(self._read_expert, layer_idx, idx)
                for idx in missing
            }
            for idx, future in futures.items():
                data = future.result()
                cached[idx] = data
                self._cache.put(layer_idx, idx, data)

        # Build index map and stack arrays in input order
        expert_index_map = {eidx: i for i, eidx in enumerate(expert_indices)}

        # Collect per-component lists
        components: dict[str, list] = {
            k: []
            for k in (
                "gate_weight",
                "gate_scales",
                "gate_biases",
                "gate_bias",
                "up_weight",
                "up_scales",
                "up_biases",
                "up_bias",
                "down_weight",
                "down_scales",
                "down_biases",
                "down_bias",
            )
        }

        for eidx in expert_indices:
            d = cached[eidx]
            for key in components:
                val = d.get(key)
                if val is not None:
                    components[key].append(val)

        def _stack_or_none(lst):
            return mx.stack(lst) if lst else None

        return LoadedExperts(
            gate_weight=mx.stack(components["gate_weight"]),
            gate_scales=_stack_or_none(components["gate_scales"]),
            gate_biases=_stack_or_none(components["gate_biases"]),
            gate_bias=_stack_or_none(components["gate_bias"]),
            up_weight=mx.stack(components["up_weight"]),
            up_scales=_stack_or_none(components["up_scales"]),
            up_biases=_stack_or_none(components["up_biases"]),
            up_bias=_stack_or_none(components["up_bias"]),
            down_weight=mx.stack(components["down_weight"]),
            down_scales=_stack_or_none(components["down_scales"]),
            down_biases=_stack_or_none(components["down_biases"]),
            down_bias=_stack_or_none(components["down_bias"]),
            is_quantized=layout.is_quantized,
            bits=layout.bits,
            group_size=layout.group_size,
            quant_mode=layout.quant_mode,
            expert_index_map=expert_index_map,
        )

    def close(self) -> None:
        """Release file descriptors and shut down the I/O thread pool."""
        self._executor.shutdown(wait=True)
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    def __del__(self) -> None:
        # Use wait=False to avoid deadlock during interpreter shutdown
        self._executor.shutdown(wait=False)
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
