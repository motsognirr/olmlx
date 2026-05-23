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


def _dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _read_hf_path_sidecar(model_dir: Path) -> str | None:
    """Read the .hf_path sidecar written by ensure_downloaded, if present.

    The sidecar exists for dirs we created via /api/pull or model load — it
    carries the exact HF repo id so _derive_manifest doesn't have to reverse
    the lossy _safe_dir_name encoding for orgs whose names contain '_'.
    """
    sidecar = model_dir / ".hf_path"
    if not sidecar.exists():
        return None
    try:
        text = sidecar.read_text().strip()
    except OSError:
        return None
    return text or None


def _write_hf_path_sidecar(model_dir: Path, hf_path: str) -> None:
    sidecar = model_dir / ".hf_path"
    try:
        sidecar.write_text(hf_path)
    except OSError:
        logger.warning("Failed to write hf_path sidecar to %s", sidecar)


def _derive_manifest(
    model_dir: Path,
    *,
    name: str,
    hf_path: str,
) -> ModelManifest:
    """Synthesize a ModelManifest from on-disk files when manifest.json is absent.

    Covers model dirs created outside of /api/pull — mlx-lm's own download
    path, manual moves, etc. — which have config.json + weights but no
    manifest.json (issue #340).  Caller resolves name/hf_path so the result
    is deterministic across endpoints (see ModelStore._canonicalize_dir).
    """
    meta = _extract_metadata(model_dir)
    size = _dir_size(model_dir)
    mtime = model_dir.stat().st_mtime
    modified_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
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

    def is_downloaded(self, hf_path: str) -> bool:
        """Check if a model is already downloaded locally."""
        local = self.local_path(hf_path)
        return (local / "config.json").exists() and not (
            local / ".downloading"
        ).exists()

    def _canonicalize_dir(self, model_dir: Path) -> tuple[str, str]:
        """Return the canonical (manifest_name, hf_path) for a model dir.

        Resolution order for hf_path: .hf_path sidecar (authoritative for
        dirs we created) → reverse the dir-name encoding (lossy fallback
        for orgs whose names contain '_').  manifest_name prefers the
        registry's normalized alias for this hf_path, so digests are stable
        across /api/show, /api/tags, and /api/ps regardless of how the
        caller spelled the model.
        """
        hf_path = _read_hf_path_sidecar(model_dir)
        if hf_path is None:
            d = model_dir.name
            hf_path = d.replace("_", "/", 1) if "_" in d else d
        canonical_name: str | None = None
        for cfg_name, cfg in self.registry.list_models().items():
            if cfg.hf_path == hf_path:
                canonical_name = self.registry.normalize_name(cfg_name)
                break
        if canonical_name is None:
            canonical_name = hf_path if ":" in hf_path else f"{hf_path}:latest"
        return canonical_name, hf_path

    def _resolve_model_dir(self, name: str) -> Path | None:
        """Resolve a model name to its local directory, trying HF-path-based naming first.

        A directory qualifies if it has either manifest.json (created by
        /api/pull) or config.json (created by mlx-lm download paths or
        manual moves — issue #340).
        """

        def _is_model_dir(d: Path) -> bool:
            return (d / "manifest.json").exists() or (d / "config.json").exists()

        resolved = self.registry.resolve(name)
        hf_path = resolved.hf_path if resolved is not None else None
        if hf_path is not None:
            d = self.local_path(hf_path)
            if _is_model_dir(d):
                return d
        # Fall back to old name-based directories
        normalized = self.registry.normalize_name(name)
        for candidate in [_safe_dir_name(normalized), _safe_dir_name(name)]:
            d = self.models_dir / candidate
            if _is_model_dir(d):
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
            # Persist the exact hf_path so future _derive_manifest calls
            # don't have to reverse the lossy dir-name encoding (issue #340).
            _write_hf_path_sidecar(local_dir, hf_path)
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
        resolved = self.registry.resolve(name)
        hf_path = resolved.hf_path if resolved is not None else None
        if hf_path is None:
            if "/" in name:
                hf_path = name
            else:
                raise ValueError(f"Model '{name}' not found in config")

        # Strip Ollama-style tag before passing to HF.
        hf_path = _strip_ollama_tag(hf_path)

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

    def _derive_and_cache(self, model_dir: Path) -> ModelManifest:
        """Build a ModelManifest from disk and persist it as manifest.json.

        Caching to disk caps the _dir_size walk to one per model per
        install (subsequent calls hit the manifest.json fast path).  Write
        failures are tolerated — the manifest is still returned for this
        call, the walk just repeats next time.
        """
        name, hf_path = self._canonicalize_dir(model_dir)
        manifest = _derive_manifest(model_dir, name=name, hf_path=hf_path)
        try:
            manifest.save(model_dir / "manifest.json")
        except OSError:
            logger.warning(
                "Failed to cache derived manifest at %s", model_dir / "manifest.json"
            )
        return manifest

    def list_local(self) -> list[ModelManifest]:
        """List all locally stored models.

        Dirs without manifest.json but with config.json are still surfaced
        with metadata derived from disk and cached as manifest.json on
        first sight (issue #340).
        """
        models = []
        if not self.models_dir.exists():
            return models
        for d in self.models_dir.iterdir():
            if not d.is_dir():
                continue
            manifest_path = d / "manifest.json"
            if manifest_path.exists():
                try:
                    models.append(ModelManifest.load(manifest_path))
                    continue
                except Exception:
                    logger.warning(
                        "Failed to load manifest %s; deriving from disk",
                        manifest_path,
                    )
            if not (d / "config.json").exists():
                continue
            try:
                models.append(self._derive_and_cache(d))
            except Exception:
                logger.warning("Failed to derive manifest for %s", d)
        return models

    def show(self, name: str) -> ModelManifest | None:
        model_dir = self._resolve_model_dir(name)
        if model_dir is None:
            return None
        manifest_path = model_dir / "manifest.json"
        if manifest_path.exists():
            try:
                return ModelManifest.load(manifest_path)
            except Exception:
                logger.warning(
                    "Failed to load manifest %s; deriving from disk", manifest_path
                )
        return self._derive_and_cache(model_dir)

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
