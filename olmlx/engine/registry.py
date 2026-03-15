import json
import os
import tempfile
from pathlib import Path

from olmlx.config import settings


def _atomic_write_json(data: dict, path: Path) -> None:
    """Write JSON data to path atomically using temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w") as f:
            fd = -1  # ownership transferred to the file object
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        if fd != -1:
            os.close(fd)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def validate_model_name(name: str) -> None:
    """Validate a model name. Raises ValueError for invalid names."""
    if not name or not name.strip():
        raise ValueError("Model name must not be empty")
    if "\x00" in name:
        raise ValueError("Model name must not contain null bytes")
    if (
        name == ".."
        or name.startswith("/")
        or "/../" in name
        or name.startswith("../")
        or name.endswith("/..")
    ):
        raise ValueError(f"Model name {name!r} contains path traversal sequence")
    if len(name) > 256:
        raise ValueError(f"Model name must be at most 256 characters, got {len(name)}")


def validate_hf_path(hf_path: str) -> None:
    """Validate a HuggingFace repo path. Raises ValueError for invalid paths."""
    if not hf_path or not hf_path.strip():
        raise ValueError("HuggingFace path must not be empty")
    if "\x00" in hf_path:
        raise ValueError("HuggingFace path must not contain null bytes")
    if (
        hf_path == ".."
        or hf_path.startswith("/")
        or "/../" in hf_path
        or hf_path.startswith("../")
        or hf_path.endswith("/..")
    ):
        raise ValueError(
            f"HuggingFace path {hf_path!r} contains path traversal sequence"
        )
    if len(hf_path) > 512:
        raise ValueError(
            f"HuggingFace path must be at most 512 characters, got {len(hf_path)}"
        )
    if hf_path.count("/") != 1:
        raise ValueError(f"HuggingFace path {hf_path!r} must be in 'owner/repo' format")


class ModelRegistry:
    """Resolves Ollama model names to HuggingFace paths via config file."""

    def __init__(self):
        self._mappings: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._aliases_path = settings.models_config.parent / "aliases.json"

    def load(self):
        """Load model mappings from config file and aliases."""
        if settings.models_config.exists():
            with open(settings.models_config) as f:
                self._mappings = json.load(f)
        if self._aliases_path.exists():
            with open(self._aliases_path) as f:
                self._aliases = json.load(f)

    @staticmethod
    def normalize_name(name: str) -> str:
        """Append :latest if no tag present."""
        if ":" not in name:
            return f"{name}:latest"
        return name

    def resolve(self, name: str) -> str | None:
        """Resolve an Ollama model name to a HuggingFace path.

        Returns the HF path or None if not found.
        Direct HF paths (containing '/') are passed through.
        """
        if "/" in name:
            validate_hf_path(name)
            return name
        validate_model_name(name)
        normalized = self.normalize_name(name)
        # Check aliases first, then mappings
        if normalized in self._aliases:
            return self._aliases[normalized]
        if normalized in self._mappings:
            return self._mappings[normalized]
        # Try without tag normalization
        if name in self._mappings:
            return self._mappings[name]
        return None

    def list_models(self) -> dict[str, str]:
        """Return all known model name → HF path mappings."""
        combined = {**self._mappings, **self._aliases}
        return combined

    def add_alias(self, alias: str, source: str):
        """Create an alias from source model."""
        validate_model_name(alias)
        alias = self.normalize_name(alias)
        hf_path = self.resolve(source)
        if hf_path is None:
            raise ValueError(f"Source model '{source}' not found")
        self._aliases[alias] = hf_path
        self._save_aliases()

    def add_mapping(self, name: str, hf_path: str):
        """Add a name → HF path mapping and persist to models.json."""
        validate_model_name(name)
        validate_hf_path(hf_path)
        normalized = self.normalize_name(name)
        if self._mappings.get(normalized) == hf_path:
            return  # already exists
        self._mappings[normalized] = hf_path
        self._save_mappings()

    def _save_mappings(self):
        _atomic_write_json(self._mappings, settings.models_config)

    def remove(self, name: str):
        """Remove a model alias."""
        validate_model_name(name)
        normalized = self.normalize_name(name)
        self._aliases.pop(normalized, None)
        self._save_aliases()

    def _save_aliases(self):
        _atomic_write_json(self._aliases, self._aliases_path)
