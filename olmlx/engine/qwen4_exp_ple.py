"""External (mmap) PLE storage for Qwen4-Exp checkpoints (Qwen3.8-Flash-Next).

Qwen4-Exp's Per-Layer Embedding carries ~51B hashed n-gram embedding
parameters, shipped as 128 Q4 shards (~30 GiB at 4-bit). Loaded as ordinary
MLX parameters they are materialized in RAM: the flash-MoE loader evals every
non-expert parameter (``wrap_flash_moe``), and a plain VLM load evals the
whole tree. Each token only reads a handful of rows, so that is ~30 GiB of
cold memory.

mlx-vlm (>=0.6.17) can instead serve the tables by mmap row lookup when
``text_config.ple_storage`` names a manifest. It only reads that from
``config.json``, so :func:`ensure_external_ple_view` builds — via mlx-vlm's own
``prepare_external_ple_model`` — a sibling *view* of the checkpoint: hard links
to the safetensors (no extra disk), a filtered index without the PLE keys, and
a config carrying ``ple_storage``. The manifest points back at the PLE byte
ranges in the original files. The view lives inside the model's store
directory (``<model>/external_ple``), so ``olmlx models delete`` removes it.

:func:`olmlx.engine.vlm_load.load_vlm` routes qwen4_exp loads through the view,
so every mlx-vlm load path (plain VLM, flash-MoE VLM fallback, calibration)
picks it up without per-call-site wiring.

mlx-vlm's reader only knows Q4/group-32 affine and NVFP4 tables. 8-bit
checkpoints (``Qwen3.8-Flash-Next-oQ8e-mtp``: ~57 GiB of PLE, which could
never be wired on a 64 GB machine) get an olmlx-built view instead, whose
manifest carries ``"format": Q8_MANIFEST_FORMAT`` and is served by
:class:`AffineQ8NGramEmbedding`. mlx-vlm picks the class by manifest format
through :func:`register_ple_table`, which Edge0's FP8 table uses too.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import struct
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

VIEW_DIRNAME = "external_ple"
_PLE_MARKER = ".ple.ple_embedding.ngram_embedding.shards."
# The manifest name mlx-vlm's view builder uses; the Q8 view matches it.
_MANIFEST_NAME = "ple-store.json"


def _needs_view(model_dir: Path) -> bool:
    try:
        config = json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict) or config.get("model_type") != "qwen4_exp":
        return False
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        return False  # malformed; let the loader report it
    if text_config.get("ple_storage"):
        return False  # checkpoint is already an external-PLE view
    try:
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict):
            raise TypeError("weight_map is not a mapping")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # mlx-vlm's view builder needs the shard index; without it the PLE
        # tables load as ordinary (wired) parameters — make that visible.
        logger.warning(
            "qwen4_exp checkpoint %s has no usable safetensors index (%s); "
            "loading without the external PLE view",
            model_dir,
            exc,
        )
        return False
    if any(_PLE_MARKER in key for key in weight_map):
        return True
    # A qwen4_exp checkpoint without recognizable PLE shards: either a variant
    # without PLE, or an upstream key rename that stopped the marker matching —
    # in which case the ~30 GiB tables would be wired as parameters again.
    logger.warning(
        "qwen4_exp checkpoint %s has no PLE shards matching %r; loading "
        "without the external PLE view",
        model_dir,
        _PLE_MARKER,
    )
    return False


_FINGERPRINT = ".olmlx_source_fingerprint.json"
_NOT_MODEL_CONTENT = frozenset({"manifest.json"})


def _source_fingerprint(model_dir: Path) -> dict[str, list[int]]:
    """``name -> [size, mtime_ns, inode]`` for every top-level source file.

    Covers everything the view depends on: the hard-linked weight shards, the
    copied sidecars (config, tokenizer, chat template — a re-pull may fix only
    those), and the PLE-only shards the manifest addresses by file name + byte
    offset (a rewrite can shift the header and therefore every offset).
    """
    fingerprint = {}
    for entry in model_dir.iterdir():
        # Skip olmlx's own bookkeeping (the store refreshes manifest.json
        # independently of model content) and dotfiles.
        if entry.name in _NOT_MODEL_CONTENT or entry.name.startswith("."):
            continue
        if entry.is_file():
            st = entry.stat()
            fingerprint[entry.name] = [st.st_size, st.st_mtime_ns, st.st_ino]
    return fingerprint


def _view_is_current(model_dir: Path, view: Path) -> bool:
    """True when the view was built from exactly the current source files."""
    try:
        recorded = json.loads((view / _FINGERPRINT).read_text())
        # The manifest name is mlx-vlm's choice; read it from the view's own
        # config rather than hard-coding it.
        view_config = json.loads((view / "config.json").read_text())
        manifest = view_config["text_config"]["ple_storage"]["manifest"]
        for name in (manifest, "model.safetensors.index.json"):
            if not (view / name).is_file():
                return False
        return recorded == _source_fingerprint(model_dir)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _build_view(model_dir: Path, target: Path) -> None:
    fingerprint = _source_fingerprint(model_dir)
    if _ple_is_q8(model_dir):
        _build_q8_view(model_dir, target)
    else:
        from mlx_vlm.models.qwen4_exp.ple_storage import prepare_external_ple_model

        prepare_external_ple_model(model_dir, target)
    (target / _FINGERPRINT).write_text(json.dumps(fingerprint))


def ensure_external_ple_view(model_path: str | Path) -> Path | None:
    """Return the external-PLE view for a qwen4_exp checkpoint, building it if needed.

    Returns ``None`` (load the original path) for anything else: non-qwen4_exp
    models, checkpoints without PLE shards or already configured for external
    storage, and paths that are not local directories (repo ids).
    """
    model_dir = Path(model_path)
    if not model_dir.is_dir() or not _needs_view(model_dir):
        return None

    view = model_dir / VIEW_DIRNAME
    if view.is_dir() and _view_is_current(model_dir, view):
        return view

    # Build beside the final location, then swap in by rename: an interrupted
    # build never leaves a half-populated view, and a stale view is moved
    # aside atomically rather than rmtree'd in place (a concurrent loader
    # never sees a half-deleted directory). Same parent, so the manifest's
    # relative source_root ("..") stays valid after the rename.
    tmp = Path(tempfile.mkdtemp(prefix=f".{VIEW_DIRNAME}.", dir=model_dir))
    stale = model_dir / f".{VIEW_DIRNAME}.stale-{uuid.uuid4().hex}"
    try:
        _build_view(model_dir, tmp)
        tmp.chmod(0o755)  # mkdtemp is 0700; match the rest of the store
        if view.exists():
            logger.info("Replacing stale external PLE view at %s", view)
            try:
                view.rename(stale)
            except FileNotFoundError:
                pass  # a concurrent loader already moved it aside
        try:
            tmp.rename(view)
        except OSError as exc:
            # ENOTEMPTY/EEXIST: a concurrent builder renamed theirs in first —
            # use it if sound. Anything else is a real failure.
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
            if not _view_is_current(model_dir, view):
                raise
    finally:
        for leftover in (tmp, stale):
            if leftover.exists():
                shutil.rmtree(leftover, ignore_errors=True)
    logger.info("Built external PLE view for %s at %s", model_dir.name, view)
    return view


# ---------------------------------------------------------------------------
# Table dispatch: which class mlx-vlm builds for a given PLE manifest.
# ---------------------------------------------------------------------------

_PLE_TABLES: dict[str, Callable[..., Any]] = {}
_hook_lock = threading.Lock()


def _manifest_format(manifest_path: Any) -> str | None:
    try:
        fmt = json.loads(Path(manifest_path).read_text()).get("format")
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    return fmt if isinstance(fmt, str) else None


def register_ple_table(fmt: str, factory: Callable[..., Any]) -> None:
    """Make mlx-vlm build ``factory(manifest, cache_rows=...)`` for *fmt* manifests.

    ``Qwen4ExpNGramEmbedding`` imports ``QuantizedMMapNGramEmbedding`` from
    ``ple_storage`` when it is constructed, so replacing that name with a
    dispatching subclass is enough; a manifest without a registered
    ``"format"`` still gets mlx-vlm's own class. Every olmlx table format
    registers here, so the name is wrapped exactly once whatever the order in
    which formats are installed. Idempotent.
    """
    from mlx_vlm.models.qwen4_exp import ple_storage

    with _hook_lock:
        _PLE_TABLES[fmt] = factory
        original = ple_storage.QuantizedMMapNGramEmbedding
        if getattr(original, "_olmlx_ple_dispatch", False):
            return

        # A subclass, not a factory function: mlx-vlm also reads class
        # attributes (``MANIFEST_VERSION``) off this name.
        class _DispatchingNGramEmbedding(original):  # type: ignore[misc, valid-type]
            _olmlx_ple_dispatch = True

            def __new__(cls, manifest_path, *, cache_rows=None):
                fmt = _manifest_format(manifest_path)
                table = _PLE_TABLES.get(fmt) if fmt is not None else None
                if table is not None:
                    # Not an instance of ``cls``, so ``__init__`` is skipped.
                    return table(manifest_path, cache_rows=cache_rows)
                return super().__new__(cls)

        _DispatchingNGramEmbedding.__name__ = original.__name__
        _DispatchingNGramEmbedding.__qualname__ = original.__qualname__
        ple_storage.QuantizedMMapNGramEmbedding = _DispatchingNGramEmbedding  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 8-bit affine PLE tables.
# ---------------------------------------------------------------------------

Q8_MANIFEST_FORMAT = "olmlx_affine_q8"
_Q8_QUANTIZATION = {"bits": 8, "group_size": 32, "mode": "affine"}
_Q8_GROUP = _Q8_QUANTIZATION["group_size"]
_Q8_VALUES_PER_WORD = 32 // _Q8_QUANTIZATION["bits"]  # MLX packs into uint32
_Q8_TENSORS = (("weight", "U32"), ("scales", "BF16"), ("biases", "BF16"))
# Everything under the n-gram table, its shards and oQ8e's leftover FP8
# ``weight_scale`` alike, leaves the view's index. mlx-vlm still loads that
# scale (it reads whole files), which :func:`_drop_unit_ple_scales` handles.
_PLE_TABLE = ".ple.ple_embedding.ngram_embedding."
_NP_DTYPES = {"U32": np.dtype("<u4"), "BF16": np.dtype("<u2")}
# Checksums of the source files; they do not describe the view.
_NOT_COPIED = frozenset({"MANIFEST.sha256"})


def _require_bare_filename(name: str) -> None:
    """Reject a file name that could reach outside its directory."""
    path = Path(name)
    if path.is_absolute() or path.name != name or name in {".", ".."}:
        raise ValueError(f"PLE view file must be a bare filename: {name!r}")


def _quantization(config: dict) -> dict:
    # mlx-vlm quantizes modules from ``quantization`` when present; an
    # HF-style ``quantization_config`` may be a stub without the per-module
    # overrides that mark the PLE table, so it is only the fallback.
    quant = config.get("quantization") or config.get("quantization_config")
    return quant if isinstance(quant, dict) else {}


def _ple_prefixes(weight_map: dict[str, str]) -> list[str]:
    prefixes = {key.rsplit(".", 1)[0] for key in weight_map if _PLE_MARKER in key}
    return sorted(prefixes, key=lambda prefix: int(prefix.rsplit(".", 1)[1]))


def _ple_is_q8(model_dir: Path) -> bool:
    """True when every PLE shard is quantized as 8-bit / group-32 affine."""
    config = json.loads((model_dir / "config.json").read_text())
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    quant = _quantization(config)
    prefixes = _ple_prefixes(index["weight_map"])
    found = set()
    for prefix in prefixes:
        params = quant.get(prefix, quant)
        # Configs written before MLX had quantization modes omit "mode"; it
        # means affine.
        found.add(
            (params.get("bits"), params.get("group_size"), params.get("mode", "affine"))
        )
    expected = (
        _Q8_QUANTIZATION["bits"],
        _Q8_QUANTIZATION["group_size"],
        _Q8_QUANTIZATION["mode"],
    )
    return found == {expected}


def _q8_manifest(model_dir: Path, target: Path, weight_map: dict[str, str]) -> dict:
    """Index the PLE shards' byte ranges in the source safetensors (no copies)."""
    headers: dict[str, tuple[int, dict]] = {}

    def descriptor(key: str) -> dict:
        file_name = weight_map[key]
        if file_name not in headers:
            with (model_dir / file_name).open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                headers[file_name] = (8 + size, json.loads(stream.read(size)))
        payload_start, header = headers[file_name]
        info = header[key]
        return {
            "file": file_name,
            "offset": payload_start + int(info["data_offsets"][0]),
            "dtype": info["dtype"],
            "shape": [int(dim) for dim in info["shape"]],
        }

    shards = []
    row_start = 0
    row_width = None
    for prefix in _ple_prefixes(weight_map):
        tensors = {name: descriptor(f"{prefix}.{name}") for name, _ in _Q8_TENSORS}
        for name, dtype in _Q8_TENSORS:
            if tensors[name]["dtype"] != dtype:
                raise ValueError(f"PLE tensor {prefix}.{name} is not {dtype}")
        weight_shape = tensors["weight"]["shape"]
        if len(weight_shape) != 2:
            raise ValueError(f"PLE tensor {prefix}.weight is not two-dimensional")
        rows, width = weight_shape[0], weight_shape[1] * _Q8_VALUES_PER_WORD
        if row_width is None:
            row_width = width
        if width != row_width or width % _Q8_GROUP:
            raise ValueError(f"PLE shard {prefix} has an unexpected row width")
        for name in ("scales", "biases"):
            if tensors[name]["shape"] != [rows, width // _Q8_GROUP]:
                raise ValueError(f"PLE tensor {prefix}.{name} has the wrong shape")
        shards.append({"row_start": row_start, "row_count": rows, **tensors})
        row_start += rows
    if not shards:
        raise ValueError(f"{model_dir} has no PLE shards")
    return {
        "format": Q8_MANIFEST_FORMAT,
        "version": 1,
        "source_root": os.path.relpath(model_dir.resolve(), target.resolve()),
        "row_width": row_width,
        "row_count": row_start,
        "quantization": dict(_Q8_QUANTIZATION),
        "shards": shards,
    }


def _build_q8_view(model_dir: Path, target: Path) -> None:
    """mlx-vlm's ``prepare_external_ple_model`` for an 8-bit table.

    Same view layout: hard-linked weight files (and, unlike mlx-vlm's
    builder, hard-linked sidecars), an index and config without the PLE
    tensors, and ``text_config.ple_storage`` naming the manifest.
    """
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    # Every name here is read from and hard-linked into the view.
    for file_name in set(weight_map.values()):
        _require_bare_filename(file_name)
    manifest = _q8_manifest(model_dir, target, weight_map)
    (target / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    kept = {key: name for key, name in weight_map.items() if _PLE_TABLE not in key}
    for file_name in sorted(set(kept.values())):
        os.link(model_dir / file_name, target / file_name)
    index["weight_map"] = kept
    (target / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )

    for entry in model_dir.iterdir():
        if (
            not entry.is_file()
            or entry.name.startswith(".")
            or entry.suffix == ".safetensors"
            or entry.name in _NOT_COPIED
            or entry.name in _NOT_MODEL_CONTENT  # store bookkeeping
            or entry.name in {"config.json", "model.safetensors.index.json"}
        ):
            continue
        # Linked, not copied: the view costs no disk even for a large sidecar.
        os.link(entry, target / entry.name)

    config = json.loads((model_dir / "config.json").read_text())
    config["text_config"]["ple_storage"] = {"manifest": _MANIFEST_NAME, "cache_rows": 0}
    # The PLE modules no longer exist, so their per-module overrides go too.
    for key in ("quantization", "quantization_config"):
        if isinstance(config.get(key), dict):
            config[key] = {k: v for k, v in config[key].items() if _PLE_TABLE not in k}
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")


class AffineQ8NGramEmbedding:
    """mmap row lookup over an 8-bit affine PLE table.

    Call-compatible with mlx-vlm's ``QuantizedMMapNGramEmbedding`` (mlx-vlm
    reads ``row_count``/``row_width`` and calls it with n-gram row ids). Each
    call reads its distinct rows from the source safetensors through the page
    cache and dequantizes them with MLX. Nothing is cached in-process, so
    ``cache_rows`` is accepted only for call compatibility.
    """

    def __init__(self, manifest_path: Path | str, *, cache_rows: int | None = None):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != Q8_MANIFEST_FORMAT:
            raise ValueError(f"not an {Q8_MANIFEST_FORMAT} manifest: {manifest_path}")
        if manifest.get("quantization") != _Q8_QUANTIZATION:
            raise ValueError(
                f"unsupported PLE quantization: {manifest.get('quantization')}"
            )
        # A manifest must not be able to mmap arbitrary local files. The
        # source is the manifest's own directory, or, for a view (which sits
        # inside the checkpoint it indexes), that view's parent.
        source_root = Path(manifest["source_root"])
        base = manifest_path.parent.resolve()
        root = (base / source_root).resolve()
        allowed = {base, base.parent} if base.name == VIEW_DIRNAME else {base}
        if source_root.is_absolute() or root not in allowed:
            raise ValueError(
                "source_root must be the manifest's directory or its parent: "
                f"{manifest['source_root']!r}"
            )
        self.row_width = int(manifest["row_width"])
        if self.row_width <= 0 or self.row_width % _Q8_GROUP:
            raise ValueError("row_width must be a positive multiple of the group size")
        widths = {
            "weight": self.row_width // _Q8_VALUES_PER_WORD,
            "scales": self.row_width // _Q8_GROUP,
            "biases": self.row_width // _Q8_GROUP,
        }
        self._widths = widths

        self._shards: list[tuple[int, dict[str, np.memmap]]] = []
        ends = []
        expected_start = 0
        for entry in manifest["shards"]:
            start, count = int(entry["row_start"]), int(entry["row_count"])
            if start != expected_start or count <= 0:
                raise ValueError("PLE shards must be positive, contiguous and ordered")
            arrays = {}
            for name, dtype in _Q8_TENSORS:
                tensor = entry[name]
                if tensor["dtype"] != dtype:
                    raise ValueError(f"unexpected {name} dtype: {tensor['dtype']}")
                _require_bare_filename(tensor["file"])
                file_name = Path(tensor["file"])
                shape = (count, widths[name])
                if tuple(int(dim) for dim in tensor["shape"]) != shape:
                    raise ValueError(f"{name} shape does not match the row layout")
                # A bare name can still be a symlink out of the checkpoint.
                path = (root / file_name).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f"PLE tensor file resolves outside {root}: {path}")
                offset = int(tensor["offset"])
                nbytes = count * widths[name] * _NP_DTYPES[dtype].itemsize
                if offset < 0 or path.stat().st_size < offset + nbytes:
                    raise ValueError(f"{name} byte range exceeds {path}")
                arrays[name] = np.memmap(
                    path, dtype=_NP_DTYPES[dtype], mode="r", offset=offset, shape=shape
                )
            self._shards.append((start, arrays))
            expected_start += count
            ends.append(expected_start)
        if not self._shards:
            raise ValueError("PLE manifest lists no shards")
        self.row_count = expected_start
        self._ends = np.asarray(ends, dtype=np.int64)

    def __call__(self, row_ids: Any) -> mx.array:
        ids = np.asarray(row_ids).astype(np.int64, copy=False)
        if ids.size == 0:
            return mx.zeros((*ids.shape, self.row_width), dtype=mx.bfloat16)
        if ids.min() < 0 or ids.max() >= self.row_count:
            raise IndexError("n-gram row outside mmap table")
        unique, inverse = np.unique(ids.reshape(-1), return_inverse=True)
        gathered = {
            name: np.empty((unique.size, self._widths[name]), dtype=_NP_DTYPES[dtype])
            for name, dtype in _Q8_TENSORS
        }
        # ``unique`` is sorted, so each shard's rows are read in file order.
        shard_of = np.searchsorted(self._ends, unique, side="right")
        for shard in np.unique(shard_of):
            selected = shard_of == shard
            start, arrays = self._shards[shard]
            local = unique[selected] - start
            for name, out in gathered.items():
                out[selected] = arrays[name][local]
        rows = mx.dequantize(
            mx.array(gathered["weight"]),
            mx.array(gathered["scales"]).view(mx.bfloat16),
            mx.array(gathered["biases"]).view(mx.bfloat16),
            group_size=_Q8_GROUP,
            bits=_Q8_QUANTIZATION["bits"],
        )
        order = mx.array(inverse.reshape(-1).astype(np.int64))
        return rows[order].reshape(*ids.shape, self.row_width)

    lookup = __call__


_PLE_SCALE_SUFFIX = _PLE_TABLE.rstrip(".") + ".weight_scale"
_converter_guard: Callable[[dict], dict] | None = None


def _drop_unit_ple_scales(weights: dict) -> dict:
    """Remove PLE table scales of exactly 1.0 that have no FP8 shards to scale.

    oQ8e ships its n-gram table as MLX-quantized shards plus a leftover FP8
    table ``weight_scale`` of 1.0. mlx-vlm loads every tensor of each indexed
    file, so the view cannot hide it, and the FP8 converter ``sanitize`` runs
    (before it drops the external table) takes it for an FP8 table and fails
    on the missing ``shard_N`` tensors. Multiplying by 1.0 is a no-op, so
    dropping it changes no weight; any other value is left for mlx-vlm to
    reject.
    """
    for key in [key for key in weights if key.endswith(_PLE_SCALE_SUFFIX)]:
        prefix = key[: -len("weight_scale")]
        if any(other.startswith(prefix + "shard_") for other in weights):
            continue  # a real FP8 table: mlx-vlm applies the scale
        scale = weights[key]
        if scale.size == 1 and scale.astype(mx.float32).item() == 1.0:
            weights = {k: v for k, v in weights.items() if k != key}
    return weights


def _install_fp8_converter_guard() -> None:
    """Run :func:`_drop_unit_ple_scales` ahead of mlx-vlm's qwen4_exp FP8 converter.

    ``Model.sanitize`` calls the converter through its module global, so
    wrapping that name is enough. Idempotent.
    """
    global _converter_guard
    from mlx_vlm.models.qwen4_exp import qwen4_exp as model_module

    with _hook_lock:
        original = getattr(model_module, "convert_qwen4_exp_fp8_weights", None)
        if original is None:
            # A release without the converter has nothing to guard.
            logger.debug("mlx-vlm qwen4_exp has no FP8 converter; guard not installed")
            return
        if original is _converter_guard:
            return

        def guarded(weights: dict) -> dict:
            return original(_drop_unit_ple_scales(weights))

        _converter_guard = guarded
        model_module.convert_qwen4_exp_fp8_weights = guarded  # pyright: ignore[reportPrivateImportUsage]


def uses_external_ple(model_path: str | Path) -> bool:
    """True for a qwen4_exp checkpoint whose config names an external PLE manifest.

    That is every view :func:`ensure_external_ple_view` builds, and also a view
    directory loaded directly.
    """
    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict) or config.get("model_type") != "qwen4_exp":
        return False
    text_config = config.get("text_config")
    return isinstance(text_config, dict) and bool(text_config.get("ple_storage"))


def install_mlx_vlm_hook() -> None:
    """Make mlx-vlm load 8-bit PLE views (see :class:`AffineQ8NGramEmbedding`)."""
    register_ple_table(Q8_MANIFEST_FORMAT, AffineQ8NGramEmbedding)
    _install_fp8_converter_guard()
