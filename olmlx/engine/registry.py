from __future__ import annotations

import difflib
import json
import os
import tempfile
import threading
import dataclasses
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, get_args

import logging

from olmlx.config import SyncMode, settings

logger = logging.getLogger(__name__)

# Experimental keys that can be overridden per-model.
# Distributed settings are excluded — they affect process startup, not per-model behavior.
PER_MODEL_EXPERIMENTAL_KEYS: frozenset[str] = frozenset(
    {
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
        # DFlash block-diffusion speculative decoding
        "dflash",
        "dflash_draft_model",
        "dflash_block_size",
        # KV cache quantization
        "kv_cache_quant",
        # Flash MoE
        "flash_moe",
        "flash_moe_cache_budget_experts",
        "flash_moe_io_threads",
    }
)


# Keys that were once experimental but have since been promoted out of the
# ``experimental`` block. The dict is kept for forward compatibility (a
# future promotion may rename the key); today every entry maps to itself
# because the keys' names are unchanged — only their location moved.
PROMOTED_EXPERIMENTAL_KEYS: dict[str, str] = {
    "speculative": "speculative",
    "speculative_draft_model": "speculative_draft_model",
    "speculative_tokens": "speculative_tokens",
}


def _validate_experimental_overrides(overrides: dict[str, Any]) -> None:
    """Validate per-model experimental overrides.

    Checks key names against the whitelist, then validates values by
    calling ``resolve_experimental`` with the global defaults as base.
    This uses ``model_dump()`` + ``model_validate()`` on a complete dict
    (all fields present), so pydantic-settings env var resolution cannot
    override any values or produce confusing errors for unrelated fields.
    """
    promoted = set(overrides) & PROMOTED_EXPERIMENTAL_KEYS.keys()
    if promoted:
        renamed = sorted(k for k in promoted if k != PROMOTED_EXPERIMENTAL_KEYS[k])
        unchanged = sorted(k for k in promoted if k == PROMOTED_EXPERIMENTAL_KEYS[k])
        parts: list[str] = []
        if unchanged:
            parts.append(
                "move " + ", ".join(repr(k) for k in unchanged) + " to the top level"
            )
        if renamed:
            renames = ", ".join(
                f"{k!r} → top-level {PROMOTED_EXPERIMENTAL_KEYS[k]!r}" for k in renamed
            )
            parts.append(f"rename {renames}")
        raise ValueError(
            "These keys have been promoted out of 'experimental': "
            + "; ".join(parts)
            + ". Update the models.json entry accordingly."
        )
    unknown = set(overrides) - PER_MODEL_EXPERIMENTAL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown experimental keys: {sorted(unknown)}. "
            f"Allowed per-model keys: {sorted(PER_MODEL_EXPERIMENTAL_KEYS)}"
        )
    if overrides:
        from olmlx.config import experimental as _global, resolve_experimental

        try:
            resolve_experimental(_global, overrides)
        except Exception as e:
            raise ValueError(f"Invalid experimental override: {e}") from e


VALID_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "num_predict",
        "repeat_penalty",
        "repeat_last_n",
        "stop",
        "frequency_penalty",
        "presence_penalty",
    }
)


_OPTION_TYPES: dict[str, type | tuple[type, ...]] = {
    "temperature": (int, float),
    "top_p": (int, float),
    "top_k": int,
    "min_p": (int, float),
    "seed": int,
    "num_predict": int,
    "repeat_penalty": (int, float),
    "repeat_last_n": int,
    "stop": list,
    "frequency_penalty": (int, float),
    "presence_penalty": (int, float),
}


def _validate_options(options: dict) -> None:
    """Validate option keys and value types."""
    unknown = set(options) - VALID_OPTION_KEYS
    if unknown:
        raise ValueError(
            f"Unknown option key(s): {sorted(unknown)}. "
            f"Valid keys: {sorted(VALID_OPTION_KEYS)}"
        )
    for key, value in options.items():
        expected = _OPTION_TYPES.get(key)
        if expected is not None and (
            not isinstance(value, expected) or isinstance(value, bool)
        ):
            raise ValueError(
                f"Option '{key}' must be {expected}, got {type(value).__name__}"
            )


def _validate_timeout(name: str, value: Any) -> float:
    """Validate a timeout value: must be a positive number, not bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"'{name}' must be a positive number, got {value!r}")
    return float(value)


_VALID_SYNC_MODES: frozenset[str] = frozenset(get_args(SyncMode))


def _validate_sync_mode(value: Any) -> SyncMode:
    if not isinstance(value, str) or value not in _VALID_SYNC_MODES:
        raise ValueError(
            f"'sync_mode' must be one of {sorted(_VALID_SYNC_MODES)}, got {value!r}"
        )
    return value  # type: ignore[return-value]


def _validate_keep_alive(value: str) -> None:
    """Validate keep_alive format at parse time."""
    import re

    v = str(value).strip()
    if v in ("-1", "0"):
        return
    if v.isdigit():
        return  # bare integer seconds, consistent with Ollama API
    if not re.match(r"^\d+(s|m|h)$", v):
        raise ValueError(
            f"Invalid keep_alive format: {value!r}. "
            f"Expected a duration like '5m', '1h', '300s', '0', or '-1'."
        )


_KNOWN_CONFIG_KEYS: frozenset[str] = frozenset()  # set after ModelConfig is defined


@dataclass
class ModelConfig:
    """Per-model configuration resolved from models.json."""

    hf_path: str
    experimental: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    #: Per-model keep_alive duration (e.g. "30m"). Only applied on initial
    #: model load; changes to models.json are not picked up while the model
    #: is already loaded — an explicit unload is required.
    keep_alive: str | None = None
    inference_queue_timeout: float | None = None
    inference_timeout: float | None = None
    #: Metal sync behavior at inference lock boundaries: "full" (default),
    #: "minimal" (skip mlx_lm/mlx_vlm generation-stream sync), or "none"
    #: (skip lock-boundary sync entirely). None means use the global
    #: ``settings.sync_mode``.
    sync_mode: SyncMode | None = None
    #: Per-model speculative decoding overrides. ``None`` means inherit the
    #: global ``Settings.speculative*`` value.
    speculative: bool | None = None
    speculative_draft_model: str | None = None
    speculative_tokens: int | None = None
    #: Unrecognized keys from the JSON entry, preserved for round-trip fidelity.
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # ``from_entry`` already validates JSON inputs, but direct
        # construction (tests, programmatic callers) bypasses it. Keep
        # this in lockstep with ``Settings.speculative_tokens``'s
        # ``Field(gt=0)`` and the empty-string check in ``from_entry``.
        if self.speculative_tokens is not None and (
            isinstance(self.speculative_tokens, bool)
            or not isinstance(self.speculative_tokens, int)
            or self.speculative_tokens < 1
        ):
            raise ValueError(
                f"'speculative_tokens' must be a positive integer or None, "
                f"got {self.speculative_tokens!r}"
            )
        if (
            self.speculative_draft_model is not None
            and not self.speculative_draft_model.strip()
        ):
            raise ValueError(
                "'speculative_draft_model' must be a non-empty HuggingFace path or None"
            )

    def resolved_speculative(self) -> tuple[bool, str | None, int]:
        """Resolve speculative config: per-model overrides global settings.

        Returns ``(enabled, draft_model, tokens)``. When ``enabled`` is
        ``False`` the draft slot is forced to ``None`` even if a global
        ``speculative_draft_model`` is configured — callers should never
        see a non-None draft for a disabled model. The token count is
        always returned so callers that flip enabled at runtime keep a
        sensible default.
        """
        from olmlx.config import settings

        enabled = (
            self.speculative if self.speculative is not None else settings.speculative
        )
        if not enabled:
            tokens = (
                self.speculative_tokens
                if self.speculative_tokens is not None
                else settings.speculative_tokens
            )
            return (False, None, tokens)
        draft = (
            self.speculative_draft_model
            if self.speculative_draft_model is not None
            else settings.speculative_draft_model
        )
        tokens = (
            self.speculative_tokens
            if self.speculative_tokens is not None
            else settings.speculative_tokens
        )
        return (True, draft, tokens)

    @classmethod
    def from_entry(cls, entry: str | dict) -> ModelConfig:
        """Create a ModelConfig from a models.json entry (string or dict)."""
        if isinstance(entry, str):
            validate_hf_path(entry)
            return cls(hf_path=entry)
        if isinstance(entry, dict):
            hf_path = entry.get("hf_path")
            if not hf_path:
                raise ValueError("Model config dict must contain 'hf_path'")
            validate_hf_path(hf_path)
            experimental = dict(entry.get("experimental", {}))
            if experimental:
                _validate_experimental_overrides(experimental)
            options = dict(entry.get("options", {}))
            if options:
                _validate_options(options)
            keep_alive_raw = entry.get("keep_alive")
            if keep_alive_raw is not None:
                keep_alive = str(keep_alive_raw)
                _validate_keep_alive(keep_alive)
            else:
                keep_alive = None
            iqt_raw = entry.get("inference_queue_timeout")
            inference_queue_timeout = (
                _validate_timeout("inference_queue_timeout", iqt_raw)
                if iqt_raw is not None
                else None
            )
            it_raw = entry.get("inference_timeout")
            inference_timeout = (
                _validate_timeout("inference_timeout", it_raw)
                if it_raw is not None
                else None
            )
            sync_mode_raw = entry.get("sync_mode")
            sync_mode = (
                _validate_sync_mode(sync_mode_raw)
                if sync_mode_raw is not None
                else None
            )
            speculative_raw = entry.get("speculative")
            if speculative_raw is not None and not isinstance(speculative_raw, bool):
                raise ValueError(
                    f"'speculative' must be a bool, got {speculative_raw!r}"
                )
            speculative = speculative_raw

            speculative_draft_model_raw = entry.get("speculative_draft_model")
            if speculative_draft_model_raw is not None:
                if not isinstance(speculative_draft_model_raw, str):
                    raise ValueError(
                        f"'speculative_draft_model' must be a string, "
                        f"got {speculative_draft_model_raw!r}"
                    )
                if not speculative_draft_model_raw.strip():
                    # Empty/whitespace-only would slip past the load
                    # check and surface as the misleading "draft model
                    # not set" error. Reject it at parse time.
                    raise ValueError(
                        "'speculative_draft_model' must be a non-empty "
                        "HuggingFace path; use ``null`` to inherit from "
                        "the global setting."
                    )
            speculative_draft_model = speculative_draft_model_raw

            speculative_tokens_raw = entry.get("speculative_tokens")
            if speculative_tokens_raw is not None:
                if (
                    isinstance(speculative_tokens_raw, bool)
                    or not isinstance(speculative_tokens_raw, int)
                    or speculative_tokens_raw < 1
                ):
                    raise ValueError(
                        f"'speculative_tokens' must be a positive integer, "
                        f"got {speculative_tokens_raw!r}"
                    )
            speculative_tokens = speculative_tokens_raw

            extra = {k: v for k, v in entry.items() if k not in _KNOWN_CONFIG_KEYS}
            return cls(
                hf_path=hf_path,
                experimental=experimental,
                options=options,
                keep_alive=keep_alive,
                inference_queue_timeout=inference_queue_timeout,
                inference_timeout=inference_timeout,
                sync_mode=sync_mode,
                speculative=speculative,
                speculative_draft_model=speculative_draft_model,
                speculative_tokens=speculative_tokens,
                _extra=extra,
            )
        raise TypeError(
            f"Model config entry must be str or dict, got {type(entry).__name__}"
        )

    def to_entry(self) -> str | dict:
        """Serialize to models.json format. Plain models become strings."""
        if (
            not self.experimental
            and not self.options
            and self.keep_alive is None
            and self.inference_queue_timeout is None
            and self.inference_timeout is None
            and self.sync_mode is None
            and self.speculative is None
            and self.speculative_draft_model is None
            and self.speculative_tokens is None
            and not self._extra
        ):
            return self.hf_path
        # Put hf_path first for readability, then known keys, then extra
        result: dict[str, Any] = {"hf_path": self.hf_path}
        if self.experimental:
            result["experimental"] = self.experimental
        if self.options:
            result["options"] = self.options
        if self.keep_alive is not None:
            result["keep_alive"] = self.keep_alive
        if self.inference_queue_timeout is not None:
            result["inference_queue_timeout"] = self.inference_queue_timeout
        if self.inference_timeout is not None:
            result["inference_timeout"] = self.inference_timeout
        if self.sync_mode is not None:
            result["sync_mode"] = self.sync_mode
        if self.speculative is not None:
            result["speculative"] = self.speculative
        if self.speculative_draft_model is not None:
            result["speculative_draft_model"] = self.speculative_draft_model
        if self.speculative_tokens is not None:
            result["speculative_tokens"] = self.speculative_tokens
        # Filter known keys defensively — from_entry() already excludes them,
        # but _extra can be set directly via ModelConfig construction.
        result.update(
            {k: v for k, v in self._extra.items() if k not in _KNOWN_CONFIG_KEYS}
        )
        return result


# Derived from ModelConfig fields so it stays in sync automatically.
# Used to separate known config keys from _extra in from_entry()/to_entry().
_KNOWN_CONFIG_KEYS = frozenset(
    f.name for f in dataclasses.fields(ModelConfig) if f.name != "_extra"
)


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
        self._raw_unrecognized: dict[str, Any] = {}
        self._dirty_keys: set[str] = set()
        self._removed_keys: set[str] = set()
        self._save_lock = threading.Lock()
        self._aliases: dict[str, str] = {}
        self._aliases_path = settings.models_config.parent / "aliases.json"

    def load(self):
        """Load model mappings from config file and aliases."""
        if settings.models_config.exists():
            try:
                with open(settings.models_config) as f:
                    raw = json.load(f)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Corrupted %s, starting with empty config: %s",
                    settings.models_config,
                    exc,
                )
                raw = {}
            self._mappings = {}
            self._raw_unrecognized = {}
            self._dirty_keys = set()
            self._removed_keys = set()
            for k, v in raw.items():
                try:
                    self._mappings[k] = ModelConfig.from_entry(v)
                except (ValueError, TypeError) as exc:
                    logger.warning("Skipping invalid models.json entry %r: %s", k, exc)
                    self._raw_unrecognized[k] = v
        if self._aliases_path.exists():
            try:
                with open(self._aliases_path) as f:
                    self._aliases = json.load(f)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Corrupted %s, ignoring aliases: %s",
                    self._aliases_path,
                    exc,
                )
                self._aliases = {}

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
            if name in self._mappings:
                return self._mappings[name]
            # Try with :latest tag appended (models.json keys include tags)
            normalized = self.normalize_name(name)
            if normalized in self._mappings:
                return self._mappings[normalized]
            return ModelConfig(hf_path=name)
        validate_model_name(name)
        normalized = self.normalize_name(name)
        # Check aliases first, then mappings
        if normalized in self._aliases:
            canonical = self._aliases[normalized]
            # Direct lookup by canonical model name
            if canonical in self._mappings:
                return self._mappings[canonical]
            # Backward compat: old aliases.json stored hf_path instead of name
            return ModelConfig(hf_path=canonical)
        if normalized in self._mappings:
            return self._mappings[normalized]
        # Try without tag normalization
        if name in self._mappings:
            return self._mappings[name]
        return None

    def list_models(self) -> dict[str, ModelConfig]:
        """Return all known model name → ModelConfig mappings.

        Aliases take priority over mappings with the same name,
        matching the behavior of ``resolve()``.
        """
        combined: dict[str, ModelConfig] = {}
        # Aliases first (matching resolve() priority)
        for alias_name, canonical in self._aliases.items():
            if canonical in self._mappings:
                combined[alias_name] = self._mappings[canonical]
            else:
                # Backward compat: old aliases.json stored hf_path
                combined[alias_name] = ModelConfig(hf_path=canonical)
        # Then mappings for names not already covered by aliases
        for name, mc in self._mappings.items():
            if name not in combined:
                combined[name] = mc
        return combined

    def add_alias(self, alias: str, source: str):
        """Create an alias from source model."""
        validate_model_name(alias)
        alias = self.normalize_name(alias)
        resolved = self.resolve(source)
        if resolved is None:
            raise ValueError(f"Source model '{source}' not found")
        # Store canonical model name (key in _mappings) for deterministic lookup
        source_normalized = self.normalize_name(source)
        if source_normalized in self._mappings:
            self._aliases[alias] = source_normalized
        elif source in self._mappings:
            self._aliases[alias] = source
        else:
            # Source is itself an alias — walk the chain to find the _mappings key
            canonical = source_normalized
            seen: set[str] = set()
            while canonical in self._aliases and canonical not in seen:
                seen.add(canonical)
                canonical = self._aliases[canonical]
            if canonical in self._mappings:
                self._aliases[alias] = canonical
            else:
                # Ultimate fallback: store hf_path
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
        are stored.  When *model_config* is ``None`` and an existing entry
        already has the same ``hf_path``, the existing entry is preserved
        (avoids erasing per-model config from callers that don't carry it).
        """
        validate_model_name(name)
        validate_hf_path(hf_path)
        normalized = self.normalize_name(name)
        existing = self._mappings.get(normalized)
        if model_config is not None:
            if model_config.hf_path != hf_path:
                raise ValueError(
                    f"hf_path mismatch: argument {hf_path!r} != "
                    f"model_config.hf_path {model_config.hf_path!r}"
                )
            if existing is not None and existing._extra:
                merged = {**existing._extra, **model_config._extra}
                mc = replace(model_config, _extra=merged)
            else:
                mc = model_config
        else:
            # No rich config supplied — preserve existing entry if hf_path matches
            if existing is not None and existing.hf_path == hf_path:
                return
            if existing is not None:
                mc = replace(existing, hf_path=hf_path)
            else:
                mc = ModelConfig(hf_path=hf_path)
        if existing is not None and existing == mc:
            return  # identical, no save needed
        self._mappings[normalized] = mc
        self._raw_unrecognized.pop(normalized, None)
        self._dirty_keys.add(normalized)
        self._removed_keys.discard(normalized)
        self._save_mappings()

    def _save_mappings(self):
        with self._save_lock:
            self._save_mappings_locked()

    def _save_mappings_locked(self):
        # The lock is defensive — it serializes disk I/O but does not cover
        # dict mutations in add_mapping/remove. This is safe today because
        # all callers run on the single-threaded asyncio event loop. If
        # multi-threaded access is ever needed, the lock must be expanded
        # to cover all _mappings/_dirty_keys/_removed_keys mutations.
        #
        # Snapshot dirty/removed sets so only keys visible at flush-start are
        # cleared afterward; additions arriving after the snapshot survive.
        dirty_snapshot = set(self._dirty_keys)
        removed_snapshot = set(self._removed_keys)

        # Re-read disk state as base to preserve external edits (e.g. user
        # editing models.json while the server is running).
        disk_data: dict[str, Any] = {}
        disk_read_ok = False
        file_exists = settings.models_config.exists()
        if file_exists:
            try:
                with open(settings.models_config) as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError(f"Expected dict, got {type(loaded).__name__}")
                disk_data = loaded
                disk_read_ok = True
            except (json.JSONDecodeError, ValueError, OSError):
                pass  # corrupt, wrong type, or unreadable

        if not disk_read_ok and self._mappings:
            # File was missing or corrupt — write full in-memory state to
            # avoid silently dropping live entries. Removed keys are already
            # absent from _mappings (remove() pops them), so they won't
            # reappear. dirty/removed snapshots are not needed here.
            if file_exists:
                logger.warning("models.json corrupt; writing full in-memory state")
            disk_data = {k: v.to_entry() for k, v in self._mappings.items()}
        else:
            # Remove explicitly deleted keys
            for key in removed_snapshot:
                disk_data.pop(key, None)

            # Overlay only keys that were modified in this process.
            # For dirty keys, the in-memory state is authoritative for known
            # fields (hf_path, experimental, options, etc.) — external edits
            # to those fields are overwritten. Unknown extra keys from the
            # disk entry ARE merged so user-added keys survive.
            for k in dirty_snapshot:
                if k in self._mappings:
                    mc = self._mappings[k]
                    disk_entry = disk_data.get(k)
                    if isinstance(disk_entry, dict):
                        disk_extras = {
                            ek: ev
                            for ek, ev in disk_entry.items()
                            if ek not in _KNOWN_CONFIG_KEYS
                        }
                        if disk_extras:
                            merged = {**mc._extra, **disk_extras}
                            mc = replace(mc, _extra=merged)
                    disk_data[k] = mc.to_entry()

        # Preserve unrecognized entries (forward/backward compatibility).
        # These may be valid entries written by a newer version. Only inject
        # when the disk file was unreadable (recovery) — if the file was read
        # successfully and an entry is absent, the user intentionally removed it.
        if not disk_read_ok:
            for k, v in self._raw_unrecognized.items():
                if k not in disk_data:
                    disk_data[k] = v

        _atomic_write_json(disk_data, settings.models_config)
        # Only clear the keys we actually flushed — concurrent additions
        # that arrived after the snapshot are preserved for the next save.
        self._dirty_keys -= dirty_snapshot
        self._removed_keys -= removed_snapshot

    def remove(self, name: str):
        """Remove a model alias or mapping."""
        validate_model_name(name)
        normalized = self.normalize_name(name)
        if self._aliases.pop(normalized, None) is not None:
            self._save_aliases()
        raw_removed = self._raw_unrecognized.pop(normalized, None)
        if self._mappings.pop(normalized, None) is not None or raw_removed is not None:
            self._removed_keys.add(normalized)
            self._dirty_keys.discard(normalized)
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
