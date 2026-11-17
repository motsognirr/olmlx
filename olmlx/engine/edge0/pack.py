"""Reader for an Edge0 "strata" latent-MoE model pack.

An Edge0 pack (e.g. ``Edge0/Edge0-145B-q3k``) is Qwen3.8-Flash-Next
(``qwen4_exp``) with every layer's routed experts rebuilt as *latent* experts:
each expert belongs to one of ``G`` groups, a shared per-group projection
``D_g`` maps the hidden state down to a narrower latent width, the expert is a
SwiGLU MLP at that width, and a shared ``U_g`` maps the result back.

The pack is a directory of raw files rather than a single checkpoint:

``model.json``
    Launch manifest; names the other files (``args`` / ``env``).
``native_experts.txt`` + ``experts.bin``
    Routed experts, one fixed-size record per (layer, expert):
    ``[gate | up | down]`` in ggml block formats, no header or padding.
``latent_dense_q8.bin`` / ``latent_dense_q4.bin``
    Per layer ``D`` (``G*latent`` rows x ``hidden``) then ``U``
    (``G*hidden`` rows x ``latent``), as ``Q8_0`` / ``Q4_0`` rows.
``labels.bin``
    int32 ``[layers, experts]``: each expert's group.
``scales/l{N}.f32``
    float32 ``[latent]`` column scale ``s``. The stored gate/up rows are
    ``A * diag(s)``, so the latent input is multiplied by ``1/s``.
``index.txt``
    Tensor directory; read here only for the router's shape.
``model-skeleton.gguf`` / ``ple-fp8.gguf`` / ``mmproj-*.gguf``
    The whole trunk, the n-gram embedding table, the vision tower. (The
    pack's ``dense.bin`` repeats the trunk's float tensors and is not read.)

This module only describes and validates the pack and hands out raw bytes;
decoding lives in :mod:`olmlx.engine.edge0.ggml_quants`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from olmlx.engine.edge0 import ggml_quants

logger = logging.getLogger(__name__)

MANIFEST_NAME = "model.json"
NATIVE_EXPERTS_NAME = "native_experts.txt"
INDEX_NAME = "index.txt"

_LATENT_DENSE_FMT_TO_GGML = {8: ggml_quants.GGML_Q8_0, 4: ggml_quants.GGML_Q4_0}

_EXPERT_EMBD_RE = re.compile(r"expert_embd\s+(\d+)")
_EXPERT_FF_RE = re.compile(r"expert_ff\s+(\d+)")
_N_EXPERT_RE = re.compile(r"\(n_expert\s+(\d+)")


class Edge0PackError(ValueError):
    """The directory is not a readable/consistent Edge0 latent pack."""


@dataclass(frozen=True)
class LatentGeometry:
    """Shape of the latent expert pool."""

    n_layers: int
    n_experts: int
    n_groups: int
    hidden: int
    latent: int
    ff: int
    gu_type: int
    down_type: int

    @property
    def gu_row_bytes(self) -> int:
        per_block, block_bytes = ggml_quants.block_geometry(self.gu_type)
        return self.latent // per_block * block_bytes

    @property
    def down_row_bytes(self) -> int:
        per_block, block_bytes = ggml_quants.block_geometry(self.down_type)
        return self.ff // per_block * block_bytes

    @property
    def gate_bytes(self) -> int:
        """Bytes of the gate matrix (and equally of the up matrix)."""
        return self.gu_row_bytes * self.ff

    @property
    def down_offset(self) -> int:
        return 2 * self.gate_bytes

    @property
    def record_bytes(self) -> int:
        return self.down_offset + self.down_row_bytes * self.latent


def is_edge0_pack(model_dir: Path | str) -> bool:
    """True when *model_dir* looks like an Edge0 latent pack (cheap check)."""
    model_dir = Path(model_dir)
    return (model_dir / MANIFEST_NAME).is_file() and (
        model_dir / NATIVE_EXPERTS_NAME
    ).is_file()


def _flag_value(args: list, flag: str) -> str | None:
    for position, value in enumerate(args[:-1]):
        if value == flag:
            return str(args[position + 1])
    return None


class Edge0Pack:
    """A validated Edge0 latent pack directory."""

    def __init__(self, root: Path | str, *, projection_bits: int | None = None):
        """*projection_bits* (4 or 8) picks which of the pack's D/U files to
        use (``latent_dense_q4.bin`` / ``latent_dense_q8.bin``); ``None`` keeps
        the one the pack's manifest names."""
        self.root = Path(root)
        if not is_edge0_pack(self.root):
            raise Edge0PackError(f"{self.root} is not an Edge0 pack")
        try:
            self.manifest = json.loads((self.root / MANIFEST_NAME).read_text())
        except (OSError, ValueError) as exc:
            raise Edge0PackError(f"unreadable {MANIFEST_NAME}: {exc}") from exc
        if not isinstance(self.manifest, dict):
            raise Edge0PackError(f"{MANIFEST_NAME} is not a JSON object")
        args = self.manifest.get("args") or []
        env = self.manifest.get("env") or {}

        self.experts_path = self._resolve(env.get("STRATA_LATENT_A"), "experts.bin")
        self.labels_path = self._resolve(env.get("STRATA_LATENT_LABELS"), "labels.bin")
        self.latent_dense_path = self._resolve(
            env.get("STRATA_LATENT_DENSE"), "latent_dense_q8.bin"
        )
        fmt = int(env.get("STRATA_LATENT_DENSE_FMT", 8))
        if projection_bits is not None and projection_bits != fmt:
            # Packs ship both quantizations of the same D/U side by side.
            sibling = self.latent_dense_path.with_name(
                f"latent_dense_q{projection_bits}.bin"
            )
            if sibling.is_file():
                self.latent_dense_path, fmt = sibling, projection_bits
            else:
                # A preference, not a requirement: a pack that ships only the
                # file its manifest names is still perfectly usable.
                logger.warning(
                    "Edge0 pack %s has no %s; using its %d-bit projections "
                    "(%s) instead of the requested %d-bit ones",
                    self.root,
                    sibling.name,
                    fmt,
                    self.latent_dense_path.name,
                    projection_bits,
                )
        if fmt not in _LATENT_DENSE_FMT_TO_GGML:
            raise Edge0PackError(f"unsupported latent projection format q{fmt}")
        self.latent_dense_type = _LATENT_DENSE_FMT_TO_GGML[fmt]
        #: Bits of the D/U file actually in use (4 or 8).
        self.projection_bits = fmt
        self.skeleton_path = self._resolve(
            _flag_value(args, "--native"), "model-skeleton.gguf"
        )
        self.ple_path = self._resolve(_flag_value(args, "--ple-gguf"), "ple-fp8.gguf")

        self.index = self._read_index()
        self.geometry = self._read_geometry()
        self._validate_sizes()

    def _resolve(self, value, default: str) -> Path:
        path = (self.root / str(value or default)).resolve()
        if not path.is_relative_to(self.root.resolve()):
            # The manifest is untrusted download content; never follow it out
            # of the pack directory.
            raise Edge0PackError(f"manifest path {value!r} escapes the pack directory")
        return path

    # -- index.txt -------------------------------------------------------

    def _read_index(self) -> dict[str, tuple[int, ...]]:
        """Tensor shapes (numpy order) from ``index.txt``.

        Only the shapes are used: the trunk weights themselves are read from
        the skeleton GGUF, which holds every tensor ``dense.bin`` does.
        """
        path = self.root / INDEX_NAME
        shapes: dict[str, tuple[int, ...]] = {}
        try:
            lines = path.read_text().splitlines()
        except OSError as exc:
            raise Edge0PackError(f"unreadable {INDEX_NAME}: {exc}") from exc
        for line in lines:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 19:
                raise Edge0PackError(f"{INDEX_NAME}: malformed row {line!r}")
            try:
                ne0, ne1 = int(fields[7]), int(fields[8])
            except ValueError as exc:
                raise Edge0PackError(f"{INDEX_NAME}: malformed row {line!r}") from exc
            # ne0 is the contiguous (input) axis; ne1 == 0 marks a vector.
            shapes[fields[0]] = (ne1, ne0) if ne1 else (ne0,)
        if not shapes:
            raise Edge0PackError(f"{INDEX_NAME} lists no tensors")
        return shapes

    # -- latent pool -------------------------------------------------------

    def _read_geometry(self) -> LatentGeometry:
        text = (self.root / NATIVE_EXPERTS_NAME).read_text()
        header = "\n".join(ln for ln in text.splitlines() if ln.startswith("#"))
        rows = [ln.split() for ln in text.splitlines() if ln.strip() and ln[0] != "#"]
        if not rows:
            raise Edge0PackError(f"{NATIVE_EXPERTS_NAME} lists no layers")

        router = self.index.get("blk.0.ffn_gate_inp.weight")
        if router is None or len(router) != 2:
            raise Edge0PackError(f"{INDEX_NAME} has no router (blk.0.ffn_gate_inp)")
        n_experts, hidden = router
        if match := _N_EXPERT_RE.search(header):
            n_experts = int(match.group(1))
        latent = int(m.group(1)) if (m := _EXPERT_EMBD_RE.search(header)) else hidden
        ff = int(m.group(1)) if (m := _EXPERT_FF_RE.search(header)) else 640
        if latent == hidden:
            raise Edge0PackError(
                "pack experts are not latent (expert_embd equals the hidden size); "
                "only latent-MoE packs are supported"
            )

        try:
            parsed = [tuple(int(v) for v in row[:5]) for row in rows]
        except ValueError as exc:
            raise Edge0PackError(f"{NATIVE_EXPERTS_NAME}: malformed row") from exc
        gu_type, down_type = parsed[0][1], parsed[0][2]
        for ggml_type in (gu_type, down_type):
            try:
                ggml_quants.block_geometry(ggml_type)
            except ValueError as exc:
                raise Edge0PackError(
                    f"expert weights use unsupported ggml type {ggml_type}"
                ) from exc

        # The group count is not written anywhere; recover it from the size of
        # the D/U file, which holds 2 * G * latent * hidden weights per layer.
        per_block, block_bytes = ggml_quants.block_geometry(self.latent_dense_type)
        per_group_layer = 2 * latent * hidden // per_block * block_bytes
        try:
            dense_size = self.latent_dense_path.stat().st_size
        except OSError as exc:
            raise Edge0PackError(f"missing {self.latent_dense_path.name}") from exc
        n_groups, remainder = divmod(dense_size, per_group_layer * len(parsed))
        if remainder or n_groups < 1:
            raise Edge0PackError(
                f"{self.latent_dense_path.name} size {dense_size} does not hold a "
                f"whole number of projection groups"
            )

        geometry = LatentGeometry(
            n_layers=len(parsed),
            n_experts=n_experts,
            n_groups=n_groups,
            hidden=hidden,
            latent=latent,
            ff=ff,
            gu_type=gu_type,
            down_type=down_type,
        )
        for expected_layer, (layer, gu, down, offset, blob) in enumerate(parsed):
            if (
                layer != expected_layer
                or (gu, down) != (gu_type, down_type)
                or blob != geometry.record_bytes
                or offset != expected_layer * n_experts * geometry.record_bytes
            ):
                raise Edge0PackError(
                    f"{NATIVE_EXPERTS_NAME}: layer {expected_layer} is not a "
                    "contiguous fixed-size record run"
                )
        return geometry

    def _validate_sizes(self) -> None:
        g = self.geometry
        expected = {
            self.experts_path: g.n_layers * g.n_experts * g.record_bytes,
            self.labels_path: g.n_layers * g.n_experts * 4,
        }
        for path, size in expected.items():
            try:
                actual = path.stat().st_size
            except OSError as exc:
                raise Edge0PackError(f"missing {path.name}") from exc
            if actual != size:
                raise Edge0PackError(
                    f"{path.name} is {actual} bytes, expected {size} "
                    "(incomplete download?)"
                )

    def expert_offset(self, layer: int, expert: int) -> int:
        g = self.geometry
        if not (0 <= layer < g.n_layers and 0 <= expert < g.n_experts):
            raise IndexError(f"no expert ({layer}, {expert}) in the pack")
        return (layer * g.n_experts + expert) * g.record_bytes

    @cached_property
    def labels(self) -> np.ndarray:
        """int32 ``[layers, experts]`` group id of every expert."""
        g = self.geometry
        labels = np.fromfile(self.labels_path, dtype="<i4").reshape(
            g.n_layers, g.n_experts
        )
        if labels.min() < 0 or labels.max() >= g.n_groups:
            raise Edge0PackError(
                "labels.bin holds a group id outside the pack's groups"
            )
        return labels

    def inverse_column_scale(self, layer: int) -> np.ndarray:
        """float32 ``[latent]`` multiplier applied to the latent expert input.

        ``scales/l{N}.f32`` stores ``s``; a pack without the file has no
        column scaling (all ones).
        """
        g = self.geometry
        path = self.root / "scales" / f"l{layer}.f32"
        if not path.exists():
            return np.ones(g.latent, dtype=np.float32)
        scale = np.fromfile(path, dtype="<f4")
        if scale.size != g.latent:
            raise Edge0PackError(
                f"{path.name} holds {scale.size} values, not {g.latent}"
            )
        inverse = np.ones_like(scale)
        np.divide(1.0, scale, out=inverse, where=scale > 0)
        return inverse

    @cached_property
    def _latent_dense(self) -> np.memmap:
        return np.memmap(self.latent_dense_path, dtype=np.uint8, mode="r")

    def _projection_bytes(self) -> int:
        g = self.geometry
        per_block, block_bytes = ggml_quants.block_geometry(self.latent_dense_type)
        return g.n_groups * g.latent * g.hidden // per_block * block_bytes

    def down_projection_raw(self, layer: int) -> np.ndarray:
        """Raw blocks of ``D``: ``G*latent`` rows of ``hidden`` weights."""
        size = self._projection_bytes()
        start = layer * 2 * size
        return self._latent_dense[start : start + size]

    def up_projection_raw(self, layer: int) -> np.ndarray:
        """Raw blocks of ``U``: ``G*hidden`` rows of ``latent`` weights."""
        size = self._projection_bytes()
        start = layer * 2 * size + size
        return self._latent_dense[start : start + size]
