import asyncio
import json
import logging
import re
import shutil
import threading
from collections.abc import AsyncGenerator
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from olmlx.config import settings
from olmlx.engine.awq_gptq_converter import (
    convert_to_mlx,
    converting_marker,
    detect_format,
)
from olmlx.engine.registry import KnownDraft, ModelConfig, ModelRegistry
from olmlx.models.manifest import ModelManifest

logger = logging.getLogger(__name__)


def _safe_dir_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def _strip_ollama_tag(hf_path: str) -> str:
    """Strip Ollama-style tag from HF path.

    Transforms "owner/repo:tag" → "owner/repo". Tags with "/" in them fail
    HF validation because they exceed reasonable length and contain special chars.
    Returns the original path if no single-slash tag is present.
    """
    if ":" not in hf_path:
        return hf_path
    base, _, tag = hf_path.partition(":")
    if "/" in base and base.count("/") == 1:
        return base
    return hf_path


def _extract_metadata(model_dir: Path) -> dict:
    """Extract model metadata from config.json if available."""
    config_path = model_dir / "config.json"
    meta = {"family": "", "parameter_size": "", "quantization_level": ""}
    cfg = None
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            meta["family"] = cfg.get("model_type", "")
            # Estimate parameter size from hidden_size and num_layers
            hidden = cfg.get("hidden_size", 0)
            layers = cfg.get("num_hidden_layers", 0)
            if hidden and layers:
                params = hidden * hidden * layers * 4  # rough estimate
                if params > 1e9:
                    meta["parameter_size"] = f"{params / 1e9:.0f}B"
                else:
                    meta["parameter_size"] = f"{params / 1e6:.0f}M"
        except Exception:
            pass
    # Check for quantization — GPTQ models use a separate file
    quant_config = model_dir / "quantize_config.json"
    if quant_config.exists():
        try:
            with open(quant_config) as f:
                qcfg = json.load(f)
            meta["quantization_level"] = qcfg.get("bits", "")
            if meta["quantization_level"]:
                meta["quantization_level"] = f"{meta['quantization_level']}-bit"
        except Exception:
            pass
    # Also check config.json for MLX quantization info (reuse already-loaded cfg)
    if cfg and "quantization" in cfg:
        q = cfg["quantization"]
        bits = q.get("bits", "")
        if bits:
            meta["quantization_level"] = f"{bits}-bit"
    return meta


def _converted_path(models_dir: Path, hf_path: str) -> Path:
    """Return the canonical directory for the MLX-converted form of *hf_path*.

    The ``@`` separator can never appear in a raw download directory name
    (``_safe_dir_name`` maps it to ``_``), so this path cannot collide with the
    download directory of any real HF repo — including one literally named
    ``owner/model@mlx`` or ``owner/model-mlx-converted``.
    """
    return models_dir / (_safe_dir_name(hf_path) + "@mlx")


def _is_valid_mlx_dir(path: Path) -> bool:
    """True when *path* is a complete, ready-to-load MLX model directory."""
    return (
        path.exists()
        and (path / "config.json").exists()
        and not converting_marker(path).exists()
    )


def _dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _derive_manifest(local_dir: Path, name: str, hf_path: str) -> ModelManifest:
    """Dynamically build a ModelManifest from on-disk files.

    Reads ``config.json`` and ``quantize_config.json`` for metadata,
    computes directory size, and generates a digest.  Used as a fallback
    when ``manifest.json`` is missing (e.g. models downloaded via mlx-lm
    rather than ``/api/pull``).

    When *local_dir* does not exist on disk, returns a manifest with
    size=0 and an empty modified_at.
    """
    meta = _extract_metadata(local_dir)
    size = _dir_size(local_dir) if local_dir.exists() else 0
    modified_at = ""
    if local_dir.exists():
        try:
            modified_at = datetime.fromtimestamp(
                local_dir.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        except OSError:
            pass
    return ModelManifest(
        name=name,
        hf_path=hf_path,
        size=size,
        modified_at=modified_at,
        digest=ModelManifest.compute_digest(name),
        format="mlx",
        family=meta["family"],
        parameter_size=meta["parameter_size"],
        quantization_level=meta["quantization_level"],
    )


class ModelStore:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.models_dir = settings.models_dir
        self._download_locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        self._pull_locks: dict[str, asyncio.Lock] = {}

    def local_path(self, hf_path: str) -> Path:
        """Return the local directory for a HF repo ID."""
        return self.models_dir / _safe_dir_name(hf_path)

    def adapter_local_path(self, hf_path: str) -> Path:
        """Return the local directory for a LoRA adapter repo ID (issue #362).

        Adapters live under a dedicated ``adapters/`` subtree so they never
        collide with a base model that happens to share the same repo id, and
        so ``list_local`` (which scans ``models_dir`` top-level) doesn't treat
        them as base models.
        """
        return self.models_dir / "adapters" / _safe_dir_name(hf_path)

    def is_adapter_downloaded(self, hf_path: str) -> bool:
        """True when the adapter's ``adapter_config.json`` is present locally."""
        local = self.adapter_local_path(hf_path)
        return (local / "adapter_config.json").exists() and not (
            local / ".downloading"
        ).exists()

    def is_downloaded(self, hf_path: str) -> bool:
        """Check if a model is already downloaded locally."""
        local = self.local_path(hf_path)
        return (local / "config.json").exists() and not (
            local / ".downloading"
        ).exists()

    def _resolve_model_dir(self, name: str) -> tuple[Path, bool] | None:
        """Resolve a model name to its local directory.

        Returns ``(path, has_manifest)`` where *has_manifest* is ``True``
        when a ``manifest.json`` exists, ``False`` when only ``config.json``
        was found (model is downloaded but no manifest was written yet).

        Returns ``None`` when the model cannot be found at all.
        """
        resolved = self.registry.resolve(name)
        hf_path = resolved.hf_path if resolved is not None else None
        if hf_path is not None:
            hf_path = _strip_ollama_tag(hf_path)
            # An AWQ/GPTQ model may have been converted to a sibling directory
            # (and its raw download removed), so check the converted path before
            # the raw one — otherwise delete()/show() can't find it by name.
            converted = _converted_path(self.models_dir, hf_path)
            if (converted / "manifest.json").exists():
                return (converted, True)
            if _is_valid_mlx_dir(converted):
                return (converted, False)
            d = self.local_path(hf_path)
            if (d / "manifest.json").exists():
                return (d, True)
            if (d / "config.json").exists():
                return (d, False)
        # Fall back to old name-based directories
        normalized = self.registry.normalize_name(name)
        for candidate in [_safe_dir_name(normalized), _safe_dir_name(name)]:
            d = self.models_dir / candidate
            if (d / "manifest.json").exists():
                return (d, True)
            if (d / "config.json").exists():
                return (d, False)
        return None

    def _pull_complete(self, hf_path: str) -> bool:
        """True when no (further) pull work is needed for *hf_path*.

        Either a valid converted MLX artifact exists, or the model is already
        downloaded AND needs no conversion. A downloaded-but-unconverted
        AWQ/GPTQ model (e.g. a prior conversion failed) returns False so the
        conversion is retried rather than reported as already complete.
        """
        if _is_valid_mlx_dir(_converted_path(self.models_dir, hf_path)):
            return True
        if self.is_downloaded(hf_path):
            return detect_format(self.local_path(hf_path)) is None
        return False

    def _pull_lock(self, hf_path: str) -> asyncio.Lock:
        """Return a per-path async lock, creating one if needed."""
        return self._pull_locks.setdefault(hf_path, asyncio.Lock())

    def _download_lock(self, hf_path: str) -> threading.Lock:
        """Return a per-path lock, creating one if needed."""
        with self._locks_lock:
            if hf_path not in self._download_locks:
                self._download_locks[hf_path] = threading.Lock()
            return self._download_locks[hf_path]

    def ensure_downloaded(self, hf_path: str) -> Path:
        """Download a model if not already present. Returns the local directory.

        Checks the converted directory first: if a prior pull already
        converted this model to MLX format, return that immediately without
        triggering a new download.

        Uses a .downloading marker to track incomplete downloads.
        Partial directories are kept on failure so snapshot_download can resume.
        Thread-safe: concurrent calls for the same hf_path are serialized.
        """
        converted_dir = _converted_path(self.models_dir, hf_path)
        if _is_valid_mlx_dir(converted_dir):
            return converted_dir

        local_dir = self.local_path(hf_path)
        if self.is_downloaded(hf_path):
            return local_dir

        with self._download_lock(hf_path):
            # Re-check after acquiring lock — another thread may have
            # completed the download while we waited.
            if self.is_downloaded(hf_path):
                return local_dir

            from huggingface_hub import snapshot_download

            local_dir.mkdir(parents=True, exist_ok=True)
            marker = local_dir / ".downloading"
            # Let touch() propagate on failure — without the marker we can't
            # safely track download state, so failing fast is correct.
            marker.touch()
            # Don't rmtree on failure: partial dir lets snapshot_download
            # resume on retry.  The .downloading marker keeps is_downloaded()
            # safe either way.
            snapshot_download(repo_id=hf_path, local_dir=str(local_dir))
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                # Rename so is_downloaded() isn't permanently poisoned —
                # without this, every future call would re-enter the download
                # path and hit snapshot_download (a network round-trip).
                try:
                    marker.rename(marker.with_name(".downloading.failed"))
                except OSError:
                    pass
                logger.error(
                    "Failed to remove .downloading marker %s; renamed to .downloading.failed"
                    " if possible — model may need manual cleanup",
                    marker,
                )
            return local_dir

    def ensure_adapter_downloaded(self, hf_path: str) -> Path:
        """Download a LoRA adapter if not present. Returns its local directory.

        Mirrors :meth:`ensure_downloaded` but targets the ``adapters/`` subtree
        and performs no AWQ/GPTQ conversion (adapters are small delta weights).
        Thread-safe: concurrent calls for the same adapter are serialized.
        """
        local_dir = self.adapter_local_path(hf_path)
        if self.is_adapter_downloaded(hf_path):
            return local_dir

        with self._download_lock("adapter:" + hf_path):
            if self.is_adapter_downloaded(hf_path):
                return local_dir

            from huggingface_hub import snapshot_download

            local_dir.mkdir(parents=True, exist_ok=True)
            marker = local_dir / ".downloading"
            marker.touch()
            snapshot_download(repo_id=hf_path, local_dir=str(local_dir))
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                try:
                    marker.rename(marker.with_name(".downloading.failed"))
                except OSError:
                    pass
                logger.error(
                    "Failed to remove .downloading marker %s; renamed to "
                    ".downloading.failed if possible — adapter may need manual cleanup",
                    marker,
                )
            return local_dir

    async def pull_adapter(self, name: str) -> AsyncGenerator[dict, None]:
        """Pull a registered LoRA adapter, yielding progress dicts (issue #362).

        The adapter must already be declared in the ``adapters`` section of
        ``models.json`` (so its base and HF repo are known). Downloads the
        adapter weights into the ``adapters/`` subtree.
        """
        cfg = self.registry.resolve_adapter(name)
        if cfg is None:
            raise ValueError(f"Adapter '{name}' not found in config")

        hf_path = cfg.hf_path
        if self.is_adapter_downloaded(hf_path):
            yield {"status": "pulling manifest"}
            yield {"status": "already downloaded"}
            yield {"status": "success"}
            return

        async with self._pull_lock("adapter:" + hf_path):
            if self.is_adapter_downloaded(hf_path):
                yield {"status": "pulling manifest"}
                yield {"status": "already downloaded"}
                yield {"status": "success"}
                return
            yield {"status": "pulling manifest"}
            yield {"status": f"downloading adapter {hf_path}"}
            await asyncio.to_thread(self.ensure_adapter_downloaded, hf_path)
            yield {"status": "verifying"}
            yield {"status": "success"}

    async def pull(self, name: str) -> AsyncGenerator[dict, None]:
        """Pull a model from HuggingFace, yielding progress dicts.

        If the downloaded model is detected as AWQ or GPTQ, it is
        automatically re-quantised to MLX int4 (configurable via
        OLMLX_AWQ_GPTQ_CONVERT_BITS / OLMLX_AWQ_GPTQ_CONVERT_GROUP_SIZE).
        The converted artifact lands in a sibling directory and the original
        is removed when OLMLX_AWQ_GPTQ_REMOVE_SOURCE=true (default).

        ``base:adapter`` names dispatch to :meth:`pull_adapter` (issue #362).
        """
        if self.registry.is_adapter(name):
            async for event in self.pull_adapter(name):
                yield event
            return

        resolved = self.registry.resolve(name)
        hf_path = resolved.hf_path if resolved is not None else None
        if hf_path is None:
            if "/" in name:
                hf_path = name
            else:
                raise ValueError(f"Model '{name}' not found in config")

        # Strip Ollama-style tag before passing to HF.
        hf_path = _strip_ollama_tag(hf_path)

        converted_dir = _converted_path(self.models_dir, hf_path)

        # Fast path: skip lock if the model is fully ready (downloaded with no
        # conversion needed, or already converted). A downloaded-but-unconverted
        # AWQ/GPTQ model is NOT complete — fall through so conversion is retried.
        if self._pull_complete(hf_path):
            yield {"status": "pulling manifest"}
            yield {"status": "already downloaded"}
            yield {"status": "success"}
            if "/" not in name:
                self.registry.add_mapping(name, hf_path)
            return

        async with self._pull_lock(hf_path):
            # Re-check after acquiring lock — another coroutine may have
            # completed the download (or conversion) while we waited.
            converted_dir = _converted_path(self.models_dir, hf_path)
            if self._pull_complete(hf_path):
                yield {"status": "pulling manifest"}
                yield {"status": "already downloaded"}
                yield {"status": "success"}
                if "/" not in name:
                    self.registry.add_mapping(name, hf_path)
                return

            yield {"status": "pulling manifest"}
            yield {"status": f"downloading {hf_path}"}

            # ensure_downloaded acquires its own threading.Lock for the same
            # hf_path — that lock serializes the sync download path used by
            # ModelManager._load_model().  The asyncio lock here serializes the
            # async pull path (is_downloaded check → download → manifest write).
            raw_local_dir = await asyncio.to_thread(self.ensure_downloaded, hf_path)
            local_dir = raw_local_dir

            fmt = detect_format(raw_local_dir)
            if fmt:
                yield {"status": f"converting {fmt} → mlx"}
                await asyncio.to_thread(
                    convert_to_mlx,
                    raw_local_dir,
                    converted_dir,
                    settings.awq_gptq_convert_bits,
                    settings.awq_gptq_convert_group_size,
                )
                local_dir = converted_dir
                if settings.awq_gptq_remove_source:
                    await asyncio.to_thread(shutil.rmtree, raw_local_dir)

            yield {"status": "verifying"}

            # Write manifest — after conversion, local_dir is the MLX artifact
            # and _extract_metadata reads its config.json for quant info.
            meta = _extract_metadata(local_dir)
            size = _dir_size(local_dir)
            normalized = self.registry.normalize_name(name)
            manifest = ModelManifest(
                name=normalized,
                hf_path=hf_path,
                size=size,
                modified_at=datetime.now(timezone.utc).isoformat(),
                digest=ModelManifest.compute_digest(normalized),
                format="mlx",
                family=meta["family"],
                parameter_size=meta["parameter_size"],
                quantization_level=meta["quantization_level"],
            )
            manifest.save(local_dir / "manifest.json")

            # Auto-register in models.json
            if "/" not in name:
                self.registry.add_mapping(name, hf_path)

            yield {"status": "success"}

    def register_speculative_draft(
        self, name: str, hf_path: str, draft: KnownDraft
    ) -> None:
        """Persist a curated speculative draft onto the target's models.json
        entry (#514).

        Writes ``speculative=True`` plus the draft's strategy / repo / block
        size, preserving any existing per-model config on the entry. Like the
        auto-registration in :meth:`pull`, this mutates the registry and must
        run on the event-loop thread.
        """
        existing = self.registry.resolve(name)
        base = existing if existing is not None else ModelConfig(hf_path=hf_path)
        mc = replace(
            base,
            hf_path=hf_path,
            speculative=True,
            speculative_strategy=draft.strategy,
            speculative_draft_model=draft.draft_repo,
            speculative_tokens=draft.block_size,
        )
        self.registry.add_mapping(name, hf_path, model_config=mc)

    def list_local(self) -> list[ModelManifest]:
        """List all locally stored models, deriving manifests for directories that lack one."""
        models = []
        if not self.models_dir.exists():
            return models
        # Build a reverse map: safe_dir_name → (short_name, hf_path) from the
        # registry so we can match unknown model directories to their Ollama name.
        dir_to_info: dict[str, tuple[str, str]] = {}
        for short_name, cfg in self.registry.list_models().items():
            safe = _safe_dir_name(cfg.hf_path)
            if safe not in dir_to_info:
                dir_to_info[safe] = (short_name, cfg.hf_path)
        for d in self.models_dir.iterdir():
            if d.is_dir():
                manifest_path = d / "manifest.json"
                if manifest_path.exists():
                    try:
                        models.append(ModelManifest.load(manifest_path))
                        continue
                    except Exception:
                        logger.warning("Failed to load manifest: %s", manifest_path)
                # No valid manifest — try to derive one from config.json
                config_path = d / "config.json"
                if config_path.exists():
                    manifest: ModelManifest | None = None
                    try:
                        info = dir_to_info.get(d.name)
                        if info is not None:
                            short_name, hf_path = info
                        else:
                            # Unknown model — reverse the safe_dir_name heuristic
                            hf_path = (
                                d.name.replace("_", "/", 1) if "_" in d.name else d.name
                            )
                            short_name = hf_path
                        manifest = _derive_manifest(d, short_name, hf_path)
                        manifest.save(manifest_path)
                    except Exception:
                        logger.debug("Failed to derive manifest for %s", d)
                    # Include the model even if save failed (disk full, etc.)
                    if manifest is not None:
                        models.append(manifest)
        return models

    def show(self, name: str) -> ModelManifest | None:
        resolved = self._resolve_model_dir(name)
        if resolved is None:
            return None
        model_dir, has_manifest = resolved
        if has_manifest:
            return ModelManifest.load(model_dir / "manifest.json")
        # Derive manifest on demand and backfill so subsequent calls are fast.
        normalized = self.registry.normalize_name(name)
        resolved_cfg = self.registry.resolve(name)
        hf_path = resolved_cfg.hf_path if resolved_cfg is not None else name
        manifest = _derive_manifest(model_dir, normalized, hf_path)
        manifest.save(model_dir / "manifest.json")
        return manifest

    def delete(self, name: str) -> bool:
        import shutil

        if self.registry.is_adapter(name):
            cfg = self.registry.resolve_adapter(name)
            if cfg is not None:
                adapter_dir = self.adapter_local_path(cfg.hf_path)
                if adapter_dir.exists():
                    shutil.rmtree(adapter_dir)
                    return True
            return False

        resolved = self._resolve_model_dir(name)
        if resolved is not None:
            shutil.rmtree(resolved[0])
            return True
        return False

    def has_blob(self, digest: str) -> bool:
        blobs_dir = self.models_dir / "blobs"
        return (blobs_dir / digest).exists()

    async def save_blob(self, digest: str, data: bytes):
        blobs_dir = self.models_dir / "blobs"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        blob_path = blobs_dir / digest
        await asyncio.to_thread(blob_path.write_bytes, data)
