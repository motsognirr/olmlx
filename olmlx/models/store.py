import asyncio
import json
import logging
import re
import threading
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path

from olmlx.config import settings
from olmlx.engine.registry import ModelRegistry
from olmlx.models.manifest import ModelManifest

logger = logging.getLogger(__name__)


def _safe_dir_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


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


def _dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


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

    def is_downloaded(self, hf_path: str) -> bool:
        """Check if a model is already downloaded locally."""
        local = self.local_path(hf_path)
        return (local / "config.json").exists() and not (
            local / ".downloading"
        ).exists()

    def _resolve_model_dir(self, name: str) -> Path | None:
        """Resolve a model name to its local directory, trying HF-path-based naming first."""
        hf_path = self.registry.resolve(name)
        if hf_path is not None:
            d = self.local_path(hf_path)
            if (d / "manifest.json").exists():
                return d
        # Fall back to old name-based directories
        normalized = self.registry.normalize_name(name)
        for candidate in [_safe_dir_name(normalized), _safe_dir_name(name)]:
            d = self.models_dir / candidate
            if (d / "manifest.json").exists():
                return d
        return None

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

        Uses a .downloading marker to track incomplete downloads.
        Partial directories are kept on failure so snapshot_download can resume.
        Thread-safe: concurrent calls for the same hf_path are serialized.
        """
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

    async def pull(self, name: str) -> AsyncGenerator[dict, None]:
        """Pull a model from HuggingFace, yielding progress dicts."""
        hf_path = self.registry.resolve(name)
        if hf_path is None:
            if "/" in name:
                hf_path = name
            else:
                raise ValueError(f"Model '{name}' not found in config")

        # Fast path: skip lock if already downloaded
        if self.is_downloaded(hf_path):
            yield {"status": "pulling manifest"}
            yield {"status": "already downloaded"}
            yield {"status": "success"}
            if "/" not in name:
                self.registry.add_mapping(name, hf_path)
            return

        async with self._pull_lock(hf_path):
            # Re-check after acquiring lock — another coroutine may have
            # completed the download while we waited.
            if self.is_downloaded(hf_path):
                yield {"status": "pulling manifest"}
                yield {"status": "already downloaded"}
                yield {"status": "success"}
                # Ensure registered
                if "/" not in name:
                    self.registry.add_mapping(name, hf_path)
                return

            yield {"status": "pulling manifest"}
            yield {"status": f"downloading {hf_path}"}

            # ensure_downloaded acquires its own threading.Lock for the same
            # hf_path — that lock serializes the sync download path used by
            # ModelManager._load_model().  The asyncio lock here serializes the
            # async pull path (is_downloaded check → download → manifest write).
            local_dir = await asyncio.to_thread(self.ensure_downloaded, hf_path)

            yield {"status": "verifying"}

            # Write manifest
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

    def list_local(self) -> list[ModelManifest]:
        """List all locally stored models."""
        models = []
        if not self.models_dir.exists():
            return models
        for d in self.models_dir.iterdir():
            if d.is_dir():
                manifest_path = d / "manifest.json"
                if manifest_path.exists():
                    try:
                        models.append(ModelManifest.load(manifest_path))
                    except Exception:
                        logger.warning("Failed to load manifest: %s", manifest_path)
        return models

    def show(self, name: str) -> ModelManifest | None:
        model_dir = self._resolve_model_dir(name)
        if model_dir is not None:
            return ModelManifest.load(model_dir / "manifest.json")
        return None

    def delete(self, name: str) -> bool:
        import shutil

        model_dir = self._resolve_model_dir(name)
        if model_dir is not None:
            shutil.rmtree(model_dir)
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
