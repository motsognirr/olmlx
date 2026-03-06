import json
from pathlib import Path

from mlx_ollama.config import settings


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
            return name
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
        alias = self.normalize_name(alias)
        hf_path = self.resolve(source)
        if hf_path is None:
            raise ValueError(f"Source model '{source}' not found")
        self._aliases[alias] = hf_path
        self._save_aliases()

    def add_mapping(self, name: str, hf_path: str):
        """Add a name → HF path mapping and persist to models.json."""
        normalized = self.normalize_name(name)
        if self._mappings.get(normalized) == hf_path:
            return  # already exists
        self._mappings[normalized] = hf_path
        self._save_mappings()

    def _save_mappings(self):
        settings.models_config.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.models_config, "w") as f:
            json.dump(self._mappings, f, indent=2)

    def remove(self, name: str):
        """Remove a model alias."""
        normalized = self.normalize_name(name)
        self._aliases.pop(normalized, None)
        self._save_aliases()

    def _save_aliases(self):
        self._aliases_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._aliases_path, "w") as f:
            json.dump(self._aliases, f, indent=2)
