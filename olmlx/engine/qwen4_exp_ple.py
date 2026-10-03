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
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

VIEW_DIRNAME = "external_ple"
_PLE_MARKER = ".ple.ple_embedding.ngram_embedding.shards."


def _needs_view(model_dir: Path) -> bool:
    try:
        config = json.loads((model_dir / "config.json").read_text())
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    except (OSError, ValueError):
        return False
    if config.get("model_type") != "qwen4_exp":
        return False
    if (config.get("text_config") or {}).get("ple_storage"):
        return False  # checkpoint is already an external-PLE view
    return any(_PLE_MARKER in key for key in index.get("weight_map", {}))


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
        for name in ("config.json", "ple-store.json", "model.safetensors.index.json"):
            if not (view / name).is_file():
                return False
    except (OSError, ValueError):
        return False
    return recorded == _source_fingerprint(model_dir)


def _build_view(model_dir: Path, target: Path) -> None:
    from mlx_vlm.models.qwen4_exp.ple_storage import prepare_external_ple_model

    fingerprint = _source_fingerprint(model_dir)
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
        except OSError:
            # Lost a race with a concurrent builder; use theirs if it is sound.
            if not _view_is_current(model_dir, view):
                raise
    finally:
        for leftover in (tmp, stale):
            if leftover.exists():
                shutil.rmtree(leftover, ignore_errors=True)
    logger.info("Built external PLE view for %s at %s", model_dir.name, view)
    return view
