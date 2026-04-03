from __future__ import annotations

import difflib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olmlx.config import settings

# Experimental keys that can be overridden per-model.
# Distributed settings are excluded — they affect process startup, not per-model behavior.
PER_MODEL_EXPERIMENTAL_KEYS: frozenset[str] = frozenset({
    # Flash inference
    "flash",
    "flash_sparsity_threshold",
    "flash_min_active_neurons",
    "flash_max_active_neurons",
    "flash_window_size",
    "flash_io_threads",
    "flash_cache_budget_neurons",
    "flash_bypass_os_cache",
    "flash_preallocated_buffer",
    "flash_memory_budget_fraction",
    # Flash prefetch
    "flash_prefetch",
    "flash_prefetch_confidence_threshold",
    "flash_prefetch_min_neurons",
    "flash_prefetch_max_neurons",
    "flash_prefetch_io_threads",
    # Flash speculative
    "flash_speculative",
    "flash_speculative_draft_model",
    "flash_speculative_tokens",
    # KV cache quantization
    "kv_cache_quant",
    # Flash MoE
    "flash_moe",
    "flash_moe_cache_budget_experts",
    "flash_moe_io_threads",
})


def _validate_experimental_overrides(overrides: dict[str, Any]) -> None:
    """Validate per-model experimental overrides."""
    unknown = set(overrides) - PER_MODEL_EXPERIMENTAL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown experimental keys: {sorted(unknown)}. "
            f"Allowed per-model keys: {sorted(PER_MODEL_EXPERIMENTAL_KEYS)}"
        )
    # Validate kv_cache_quant format if present
    kv_quant = overrides.get("kv_cache_quant")
    if kv_quant is not None:
        _valid_methods = {"turboquant"}
        _valid_bits = {"2", "4"}
        parts = str(kv_quant).split(":", 1)
        if (
            len(parts) != 2
            or parts[0] not in _valid_methods
            or parts[1] not in _valid_bits
        ):
            raise ValueError(
                f"Invalid kv_cache_quant={kv_quant!r}. "
                f"Expected '<method>:<bits>' where method is one of {_valid_methods} "
                f"and bits is one of {_valid_bits}."
            )


@dataclass
class ModelConfig:
    """Per-model configuration resolved from models.json."""

    hf_path: str
    experimental: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    keep_alive: str | None = None

    @classmethod
    def from_entry(cls, entry: str | dict) -> ModelConfig:
        """Create a ModelConfig from a models.json entry (string or dict)."""
        if isinstance(entry, str):
            return cls(hf_path=entry)
        if isinstance(entry, dict):
            hf_path = entry.get("hf_path")
            if not hf_path:
                raise ValueError("Model config dict must contain 'hf_path'")
            experimental = dict(entry.get("experimental", {}))
            if experimental:
                _validate_experimental_overrides(experimental)
            return cls(
                hf_path=hf_path,
                experimental=experimental,
                options=dict(entry.get("options", {})),
                keep_alive=entry.get("keep_alive"),
            )
        raise TypeError(f"Model config entry must be str or dict, got {type(entry).__name__}")

    def to_entry(self) -> str | dict:
        """Serialize to models.json format. Plain models become strings."""
        if not self.experimental and not self.options and self.keep_alive is None:
            return self.hf_path
        result: dict[str, Any] = {"hf_path": self.hf_path}
        if self.experimental:
            result["experimental"] = self.experimental
        if self.options:
            result["options"] = self.options
        if self.keep_alive is not None:
            result["keep_alive"] = self.keep_alive
        return result


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
        self._mappings: dict[str, ModelConfig] = {}
        self._aliases: dict[str, str] = {}
        self._aliases_path = settings.models_config.parent / "aliases.json"

    def load(self):
        """Load model mappings from config file and aliases."""
        if settings.models_config.exists():
            with open(settings.models_config) as f:
                raw = json.load(f)
            self._mappings = {
                k: ModelConfig.from_entry(v) for k, v in raw.items()
            }
        if self._aliases_path.exists():
            with open(self._aliases_path) as f:
                self._aliases = json.load(f)

    @staticmethod
    def normalize_name(name: str) -> str:
        """Append :latest if no tag present."""
        if ":" not in name:
            return f"{name}:latest"
        return name

    def resolve(self, name: str) -> ModelConfig | None:
        """Resolve an Ollama model name to a ModelConfig.

        Returns the ModelConfig or None if not found.
        Direct HF paths (containing '/') are wrapped in a plain ModelConfig.
        """
        if "/" in name:
            validate_hf_path(name)
            return ModelConfig(hf_path=name)
        validate_model_name(name)
        normalized = self.normalize_name(name)
        # Check aliases first, then mappings
        if normalized in self._aliases:
            hf_path = self._aliases[normalized]
            # Look up the target model's config if it exists in mappings
            for mc in self._mappings.values():
                if mc.hf_path == hf_path:
                    return mc
            return ModelConfig(hf_path=hf_path)
        if normalized in self._mappings:
            return self._mappings[normalized]
        # Try without tag normalization
        if name in self._mappings:
            return self._mappings[name]
        return None

    def list_models(self) -> dict[str, ModelConfig]:
        """Return all known model name → ModelConfig mappings."""
        combined: dict[str, ModelConfig] = {**self._mappings}
        for alias_name, hf_path in self._aliases.items():
            if alias_name not in combined:
                combined[alias_name] = ModelConfig(hf_path=hf_path)
        return combined

    def add_alias(self, alias: str, source: str):
        """Create an alias from source model."""
        validate_model_name(alias)
        alias = self.normalize_name(alias)
        resolved = self.resolve(source)
        if resolved is None:
            raise ValueError(f"Source model '{source}' not found")
        self._aliases[alias] = resolved.hf_path
        self._save_aliases()

    def add_mapping(
        self,
        name: str,
        hf_path: str,
        *,
        model_config: ModelConfig | None = None,
    ):
        """Add a name → HF path mapping and persist to models.json.

        If *model_config* is provided, its experimental/options/keep_alive
        are stored.  Otherwise a plain mapping is created.
        """
        validate_model_name(name)
        validate_hf_path(hf_path)
        normalized = self.normalize_name(name)
        existing = self._mappings.get(normalized)
        if model_config is not None:
            mc = model_config
        else:
            mc = ModelConfig(hf_path=hf_path)
        if existing is not None and existing.hf_path == mc.hf_path and existing.experimental == mc.experimental and existing.options == mc.options and existing.keep_alive == mc.keep_alive:
            return  # already exists
        self._mappings[normalized] = mc
        self._save_mappings()

    def _save_mappings(self):
        serialized = {k: v.to_entry() for k, v in self._mappings.items()}
        _atomic_write_json(serialized, settings.models_config)

    def remove(self, name: str):
        """Remove a model alias or mapping."""
        validate_model_name(name)
        normalized = self.normalize_name(name)
        if self._aliases.pop(normalized, None) is not None:
            self._save_aliases()
        if self._mappings.pop(normalized, None) is not None:
            self._save_mappings()

    def search(self, query: str, max_results: int = 5) -> list[tuple[str, str]]:
        """Fuzzy search models by name. Returns [(name, hf_path), ...]."""
        if not query or len(query) > 200:
            return []
        all_models = self.list_models()
        all_names = list(all_models.keys())
        # Map each base name (without tag) to all full names that share it
        base_to_full: dict[str, list[str]] = {}
        for n in all_names:
            base_to_full.setdefault(n.split(":")[0], []).append(n)
        # Deduplicated candidate list: full names + base names
        candidates = list(dict.fromkeys(all_names + list(base_to_full.keys())))
        matches = difflib.get_close_matches(
            query, candidates, n=max(max_results * 4, 20), cutoff=0.4
        )
        seen: set[str] = set()
        results: list[tuple[str, str]] = []
        for m in matches:
            # Expand base name matches to all full names
            full_names = base_to_full.get(m, [m])
            for full in full_names:
                if full not in seen and full in all_models:
                    seen.add(full)
                    results.append((full, all_models[full].hf_path))
        return results[:max_results]

    def _save_aliases(self):
        _atomic_write_json(self._aliases, self._aliases_path)
