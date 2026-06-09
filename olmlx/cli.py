"""CLI for olmlx with serve, service, models, and config subcommands."""

import argparse
import asyncio
import json
import logging
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmlx.engine.registry import ModelRegistry

from olmlx.config import (
    settings,
    surface_legacy_flash_env as _surface_legacy_flash_env,
    surface_legacy_flash_moe_env as _surface_legacy_flash_moe_env,
    surface_legacy_flash_prefetch_speculative_env as _surface_legacy_flash_prefetch_speculative_env,
)

logger = logging.getLogger(__name__)

PLIST_LABEL = "com.dpalmqvist.olmlx"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_LABEL}.plist"

DEFAULT_MODELS = {
    "llama3.2:latest": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistral:7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "qwen2.5:3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "gemma2:2b": "mlx-community/gemma-2-2b-it-4bit",
    "whisper-turbo:latest": "mlx-community/whisper-large-v3-turbo",
    "whisper-large:latest": "mlx-community/whisper-large-v3-mlx",
}


def ensure_config():
    """Create ~/.olmlx/ and seed models.json if missing."""
    config_dir = settings.models_config.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    if not settings.models_config.exists():
        with open(settings.models_config, "w") as f:
            json.dump(DEFAULT_MODELS, f, indent=2)
        print(f"Created {settings.models_config} with example models")


def cmd_serve(args):
    """Start the olmlx server."""
    import uvicorn

    # ensure_config() must run before override validation so the registry
    # walk in _apply_serve_overrides sees a real models.json on first run.
    ensure_config()
    _configure_logging()
    _apply_serve_overrides(args)

    from olmlx.config import settings

    _surface_legacy_distributed_env()  # must run before the guard — legacy env var
    if settings.distributed:
        _hosts, strategy, hostfile_layers = _launch_distributed_workers()
        # The ring backend's init() blocks until all ranks connect. Both the
        # coordinator and workers must call init() within each other's retry
        # window (~31s). Workers start ~5-10s after SSH launch. We delay
        # the coordinator by 3s to overlap with the worker's init() window.
        print("  Waiting 3s for workers to start...")
        time.sleep(3)
        import mlx.core as mx

        try:
            group = mx.distributed.init(backend=settings.distributed_backend)
        except Exception:
            _cleanup_workers()
            raise
        print(f"  Ring initialized: rank {group.rank()}, world_size {group.size()}")

        # Start the sideband server NOW, before uvicorn imports the app.
        # The app import triggers transformers which can be very slow.
        # Workers need the sideband to be available during that time.
        from olmlx.engine.distributed import DistributedCoordinator

        coordinator = DistributedCoordinator(
            world_size=group.size(),
            port=settings.distributed_sideband_port,
            secret=settings.distributed_secret or None,
        )
        # Store for the app lifespan to retrieve
        global _cli_distributed_group, _cli_distributed_coordinator
        global _cli_distributed_strategy, _cli_distributed_layer_counts
        _cli_distributed_group = group
        _cli_distributed_coordinator = coordinator
        _cli_distributed_strategy = strategy
        _cli_distributed_layer_counts = hostfile_layers

    uvicorn.run(
        "olmlx.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


# DEPRECATION: drop _DEPRECATED_SPECULATIVE_ENV_VARS,
# _LEGACY_SPECULATIVE_FORWARD, _forward_legacy_speculative_env, the
# warning + forwarding call site in _apply_serve_overrides, and the
# legacy fallback in olmlx/bench/scenarios._requires_speculative_draft
# in the next release after this PR ships. The promotion in PR #270
# included a one-release deprecation window; once it passes, leaving
# this code in place silently keeps a now-unsupported alias alive.
_DEPRECATED_SPECULATIVE_ENV_VARS = (
    "OLMLX_EXPERIMENTAL_SPECULATIVE",
    "OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL",
    "OLMLX_EXPERIMENTAL_SPECULATIVE_TOKENS",
)

# Legacy → new env var mapping with parsers. Only applied when the
# matching new env var is unset, so users with the old names in their
# shell profile keep working through the deprecation window.
_LEGACY_SPECULATIVE_FORWARD: tuple[tuple[str, str, str, Callable[[str], Any]], ...] = (
    (
        "OLMLX_EXPERIMENTAL_SPECULATIVE",
        "OLMLX_SPECULATIVE",
        "speculative",
        lambda v: v.strip().lower() in ("1", "true", "yes", "on"),
    ),
    (
        "OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL",
        "OLMLX_SPECULATIVE_DRAFT_MODEL",
        "speculative_draft_model",
        str,
    ),
    (
        "OLMLX_EXPERIMENTAL_SPECULATIVE_TOKENS",
        "OLMLX_SPECULATIVE_TOKENS",
        "speculative_tokens",
        int,
    ),
)


def _legacy_speculative_values_in_dotenv() -> dict[str, str]:
    """Return ``{name: value}`` for any ``_DEPRECATED_SPECULATIVE_ENV_VARS``
    found in the project ``.env`` file.

    ``Settings.model_config`` declares ``env_file=".env"`` (cwd-relative);
    pydantic-settings reads it into Settings without touching
    ``os.environ``, so a shell-only scan would miss legacy values in the
    file. The format accepted is a subset of pydantic-settings' own
    ``.env`` parser: ``KEY=value`` lines (optionally ``export KEY=...``),
    with ``#`` comments and blank lines ignored, and surrounding single
    or double quotes stripped from the value.
    """
    dotenv_path = Path(".env")
    try:
        text = dotenv_path.read_text()
    except (FileNotFoundError, OSError):
        return {}
    found: dict[str, str] = {}
    legacy = set(_DEPRECATED_SPECULATIVE_ENV_VARS)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        # Require length >= 2 so a literal single quote ``"`` doesn't
        # collapse to the empty string.
        is_quoted = len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        )
        if is_quoted:
            value = value[1:-1]
        else:
            # Strip inline ``# comment`` from unquoted values. Without
            # this, a line like ``KEY=true  # enable`` would parse
            # ``true  # enable``, which the boolean forwarder coerces
            # to ``False`` — the opposite of intent. Limitation: an
            # unquoted value containing a literal ``#`` (e.g. a path
            # fragment) is silently truncated. Quote the value to
            # disable this stripping.
            comment_idx = value.find("#")
            if comment_idx != -1:
                value = value[:comment_idx].rstrip()
        if key in legacy and key not in found:
            found[key] = value
    return found


def _forward_legacy_speculative_env(
    settings_obj,
    dotenv_values: dict[str, str] | None = None,
) -> None:
    """Apply legacy env var values to the new Settings when the new env
    var is unset. Logs and swallows parse errors per-field so a single
    bad legacy value never blocks startup.

    "Unset" is determined by comparing the live Settings value against
    the field default — checking ``os.environ`` alone would miss values
    pydantic-settings already loaded from a ``.env`` file and silently
    let the legacy shell var clobber them.

    *dotenv_values* lets callers pass in pre-parsed ``.env`` values so
    the file isn't read twice when the deprecation banner already
    needed it. Defaults to a fresh parse for direct callers.
    """
    from olmlx.config import Settings

    if dotenv_values is None:
        dotenv_values = _legacy_speculative_values_in_dotenv()
    for legacy, new, attr, parse in _LEGACY_SPECULATIVE_FORWARD:
        # Shell wins over .env if both have the legacy var set, mirroring
        # pydantic-settings' precedence for the new names.
        legacy_val = os.environ.get(legacy, dotenv_values.get(legacy))
        if legacy_val is None:
            continue
        if os.environ.get(new) is not None:
            # The new shell var was set explicitly (even if its value
            # happens to equal the schema default).
            continue
        field_default = Settings.model_fields[attr].default
        if getattr(settings_obj, attr) != field_default:
            # pydantic-settings already loaded a non-default value into
            # the field (from a ``.env`` file or programmatic write at
            # import time). CLI flags can't be the source here — they
            # are applied later in ``_apply_serve_overrides``. The
            # remaining blind spot is a ``.env`` entry that happens to
            # match the schema default, which the legacy value would
            # still overwrite — an acceptable tradeoff during the
            # deprecation window.
            continue
        try:
            value = parse(legacy_val)
            setattr(settings_obj, attr, value)
            # Per-field log so the override is visible alongside the
            # bulk deprecation banner. Notable when a ``.env`` file set
            # the new field to its schema default and the legacy shell
            # var clobbers it — the operator gets a clear "X → Y"
            # trail, not just the up-front banner.
            logger.warning(
                "Forwarding legacy %s=%r → settings.%s. The new env var "
                "%s would take precedence if explicitly set in the shell. "
                "Note: a value in .env that equals the schema default "
                "cannot be distinguished from 'unset' and may be silently "
                "overridden by the legacy var — rename the .env entry to "
                "%s to avoid this.",
                legacy,
                legacy_val,
                attr,
                new,
                new,
            )
        except Exception as exc:
            # Catches both parse errors (ValueError/TypeError) and the
            # ``pydantic_core.ValidationError`` raised on assignment when
            # ``validate_assignment=True`` rejects the value (e.g.
            # speculative_tokens=0). A bad legacy value must never block
            # startup — fall back to the new Settings default.
            logger.warning(
                "Could not forward legacy env var %s=%r to %s: %s",
                legacy,
                legacy_val,
                new,
                exc,
            )


_DEPRECATED_DFLASH_ENV_VARS = (
    "OLMLX_EXPERIMENTAL_DFLASH",
    "OLMLX_EXPERIMENTAL_DFLASH_DRAFT_MODEL",
    "OLMLX_EXPERIMENTAL_DFLASH_BLOCK_SIZE",
)


def _surface_legacy_dflash_env() -> None:
    """Detect and forward legacy ``OLMLX_EXPERIMENTAL_DFLASH*`` env vars.

    DFlash has been folded into the unified speculative path
    (``OLMLX_SPECULATIVE`` + ``OLMLX_SPECULATIVE_STRATEGY=dflash`` +
    ``OLMLX_SPECULATIVE_DRAFT_MODEL`` + ``OLMLX_SPECULATIVE_TOKENS``).
    Honour the old env vars for one release, with a warning, then drop
    them in the next promotion cycle.
    """
    from olmlx.config import settings as _settings

    legacy_dflash = os.environ.get("OLMLX_EXPERIMENTAL_DFLASH", "").strip().lower()
    legacy_draft = os.environ.get("OLMLX_EXPERIMENTAL_DFLASH_DRAFT_MODEL")
    legacy_block = os.environ.get("OLMLX_EXPERIMENTAL_DFLASH_BLOCK_SIZE")

    stale = [v for v in _DEPRECATED_DFLASH_ENV_VARS if os.environ.get(v)]
    if not stale:
        return
    logger.warning(
        "Deprecated env vars detected: %s. DFlash is now a strategy of the "
        "unified speculative path. Set OLMLX_SPECULATIVE=true, "
        "OLMLX_SPECULATIVE_STRATEGY=dflash, "
        "OLMLX_SPECULATIVE_DRAFT_MODEL=<hf-path>, and (optionally) "
        "OLMLX_SPECULATIVE_TOKENS=<N> instead. The old vars will be removed "
        "in a future release.",
        ", ".join(stale),
    )

    if legacy_dflash in ("1", "true", "yes", "on"):
        # Also gate on ``_settings.speculative`` so that
        # ``_surface_legacy_speculative_env`` (which runs first and may
        # have already forwarded a legacy ``OLMLX_EXPERIMENTAL_SPECULATIVE
        # =true`` to enable classic speculative) is not silently
        # overridden to dflash by a coexisting legacy DFlash var. Explicit
        # speculative wins over implicit dflash.
        if not os.environ.get("OLMLX_SPECULATIVE") and not _settings.speculative:
            try:
                _settings.speculative = True
                _settings.speculative_strategy = "dflash"
            except Exception as exc:
                logger.warning("Could not forward legacy DFlash settings: %s", exc)
        elif _settings.speculative and _settings.speculative_strategy != "dflash":
            # Both ``OLMLX_EXPERIMENTAL_SPECULATIVE=true`` and
            # ``OLMLX_EXPERIMENTAL_DFLASH=true`` were set; the former
            # already forwarded into classic speculative. Surface the
            # conflict so the operator knows their DFlash flag was
            # dropped (the deprecation banner alone doesn't say which
            # of the two won).
            logger.warning(
                "Conflicting legacy flags: OLMLX_EXPERIMENTAL_SPECULATIVE "
                "(forwarded to classic speculative) takes precedence over "
                "OLMLX_EXPERIMENTAL_DFLASH. The DFlash strategy was NOT "
                "applied. Set OLMLX_SPECULATIVE_STRATEGY=dflash explicitly "
                "if DFlash is what you want."
            )
    if legacy_draft and not os.environ.get("OLMLX_SPECULATIVE_DRAFT_MODEL"):
        try:
            _settings.speculative_draft_model = legacy_draft
        except Exception as exc:
            logger.warning("Could not forward legacy DFlash draft model: %s", exc)
    if legacy_block and not os.environ.get("OLMLX_SPECULATIVE_TOKENS"):
        try:
            _settings.speculative_tokens = int(legacy_block)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Could not forward OLMLX_EXPERIMENTAL_DFLASH_BLOCK_SIZE=%r: %s",
                legacy_block,
                exc,
            )


def _surface_legacy_speculative_env() -> None:
    """Warn about and forward legacy ``OLMLX_EXPERIMENTAL_SPECULATIVE*``
    env vars (shell or ``.env``) to the new Settings.

    Called from every subcommand that touches speculative decoding so
    the deprecation window is honoured uniformly — ``serve``, ``chat``,
    and any future surface that reads ``settings.speculative*``. Reads
    ``.env`` once and threads the result into the forwarder so the
    file is only opened once per startup.
    """
    from olmlx.config import settings as _settings

    dotenv_values = _legacy_speculative_values_in_dotenv()
    shell_stale = [v for v in _DEPRECATED_SPECULATIVE_ENV_VARS if os.environ.get(v)]
    stale = sorted({*shell_stale, *dotenv_values.keys()})
    if stale:
        logger.warning(
            "Deprecated env vars detected: %s. They will be honoured for "
            "this release but should be renamed to OLMLX_SPECULATIVE, "
            "OLMLX_SPECULATIVE_DRAFT_MODEL, OLMLX_SPECULATIVE_TOKENS.",
            ", ".join(stale),
        )
        # Forward legacy values to the new Settings only when the new env
        # var is unset, so user-facing behaviour doesn't silently change
        # on upgrade. Drop this once the deprecation window closes.
        _forward_legacy_speculative_env(_settings, dotenv_values)


def _legacy_kv_cache_quant_in_dotenv() -> str | None:
    """Return the value of ``OLMLX_EXPERIMENTAL_KV_CACHE_QUANT`` from the
    project ``.env`` file, or None if not present or the file is unreadable."""
    dotenv_path = Path(".env")
    try:
        text = dotenv_path.read_text()
    except (FileNotFoundError, OSError):
        return None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if key != "OLMLX_EXPERIMENTAL_KV_CACHE_QUANT":
            continue
        value = value.strip()
        # Detect surrounding quotes first; a ``#`` inside quotes is
        # preserved.  If no quotes are found, strip trailing inline
        # comments and re-check — ``"turboquant:4" # comment`` has no
        # terminating quote until the comment is removed.
        is_quoted = len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        )
        if not is_quoted:
            comment_idx = value.find("#")
            if comment_idx != -1:
                value = value[:comment_idx].rstrip()
                is_quoted = len(value) >= 2 and (
                    (value.startswith('"') and value.endswith('"'))
                    or (value.startswith("'") and value.endswith("'"))
                )
        if is_quoted:
            value = value[1:-1]
        return value
    return None


def _surface_legacy_kv_cache_quant_env() -> None:
    """Forward legacy OLMLX_EXPERIMENTAL_KV_CACHE_QUANT to OLMLX_KV_CACHE_QUANT
    when the new env var is unset.

    Mirrors the speculative legacy forwarding pattern. Called from every
    subcommand that reads ``settings.kv_cache_quant`` so the deprecation
    window is honoured uniformly — ``serve`` and ``chat``.
    """
    from olmlx.config import settings as _settings

    legacy_val = os.environ.get("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT")
    if legacy_val is None:
        # Only read .env when the shell env var is absent, mirroring the
        # speculative forwarding pattern.
        legacy_val = _legacy_kv_cache_quant_in_dotenv()
    if legacy_val is None:
        return
    if os.environ.get("OLMLX_KV_CACHE_QUANT") is not None:
        return  # new env var takes precedence
    if _settings.kv_cache_quant is not None:
        # Already set via .env or a CLI flag applied earlier in
        # _apply_serve_overrides.
        return
    try:
        _settings.kv_cache_quant = legacy_val
        logger.warning(
            "Forwarding legacy OLMLX_EXPERIMENTAL_KV_CACHE_QUANT=%r "
            "→ settings.kv_cache_quant. Rename to OLMLX_KV_CACHE_QUANT.",
            legacy_val,
        )
    except Exception as exc:
        logger.warning(
            "Could not forward legacy env var OLMLX_EXPERIMENTAL_KV_CACHE_QUANT=%r: %s",
            legacy_val,
            exc,
        )


def _warn_kv_cache_quant_incompatibilities() -> None:
    """Warn about tracked incompatibilities at startup."""
    from olmlx.config import settings as _settings

    if _settings.prompt_cache_disk and _settings.kv_cache_quant:
        logger.warning(
            "Prompt cache disk offload is enabled (OLMLX_PROMPT_CACHE_DISK=true) "
            "together with KV cache quantization (OLMLX_KV_CACHE_QUANT=%s). "
            "Quantized KV caches cannot be serialized to disk — disk saves "
            "will be silently skipped. Disable one of these options.",
            _settings.kv_cache_quant,
        )
    # Per-model kv_cache_quant overrides are not checked here — walking
    # the full registry at startup is noisy (every entry with a per-model
    # override would warn, even for models not being loaded). The runtime
    # guard in PromptCacheStore._save_to_disk silently skips disk saves
    # for any non-serializable cache regardless of the source, so there
    # is no silent data loss.


#: Distributed fields with their ``OLMLX_EXPERIMENTAL_DISTRIBUTED_*`` legacy
#: name and the corresponding ``OLMLX_DISTRIBUTED_*`` new name (without
#: prefix), keyed by the Python attribute name on ``Settings``.
_DISTRIBUTED_LEGACY_ENV_MAP: dict[str, tuple[str, str]] = {
    "distributed": ("OLMLX_EXPERIMENTAL_DISTRIBUTED", "OLMLX_DISTRIBUTED"),
    "distributed_strategy": (
        "OLMLX_EXPERIMENTAL_DISTRIBUTED_STRATEGY",
        "OLMLX_DISTRIBUTED_STRATEGY",
    ),
    "distributed_hostfile": (
        "OLMLX_EXPERIMENTAL_DISTRIBUTED_HOSTFILE",
        "OLMLX_DISTRIBUTED_HOSTFILE",
    ),
    "distributed_backend": (
        "OLMLX_EXPERIMENTAL_DISTRIBUTED_BACKEND",
        "OLMLX_DISTRIBUTED_BACKEND",
    ),
    "distributed_port": (
        "OLMLX_EXPERIMENTAL_DISTRIBUTED_PORT",
        "OLMLX_DISTRIBUTED_PORT",
    ),
    "distributed_sideband_port": (
        "OLMLX_EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT",
        "OLMLX_DISTRIBUTED_SIDEBAND_PORT",
    ),
    "distributed_secret": (
        "OLMLX_EXPERIMENTAL_DISTRIBUTED_SECRET",
        "OLMLX_DISTRIBUTED_SECRET",
    ),
    "distributed_remote_working_dir": (
        "OLMLX_EXPERIMENTAL_DISTRIBUTED_REMOTE_WORKING_DIR",
        "OLMLX_DISTRIBUTED_REMOTE_WORKING_DIR",
    ),
    "distributed_remote_python": (
        "OLMLX_EXPERIMENTAL_DISTRIBUTED_REMOTE_PYTHON",
        "OLMLX_DISTRIBUTED_REMOTE_PYTHON",
    ),
    "distributed_pre_shard": (
        "OLMLX_EXPERIMENTAL_DISTRIBUTED_PRE_SHARD",
        "OLMLX_DISTRIBUTED_PRE_SHARD",
    ),
    "distributed_shard_dir": (
        "OLMLX_EXPERIMENTAL_DISTRIBUTED_SHARD_DIR",
        "OLMLX_DISTRIBUTED_SHARD_DIR",
    ),
    "distributed_worker_shard_dir": (
        "OLMLX_EXPERIMENTAL_DISTRIBUTED_WORKER_SHARD_DIR",
        "OLMLX_DISTRIBUTED_WORKER_SHARD_DIR",
    ),
}


def _surface_legacy_distributed_env() -> None:
    """Forward legacy ``OLMLX_EXPERIMENTAL_DISTRIBUTED_*`` env vars to the new
    ``OLMLX_DISTRIBUTED_*`` names when the new env var is unset.

    Mirrors the kv_cache_quant / speculative legacy forwarding pattern.
    Called from ``cmd_serve`` and worker applications that read
    ``settings.distributed_*`` so the deprecation window is honoured uniformly.
    """
    from olmlx.config import settings as _settings

    for attr_name, (legacy_name, new_name) in _DISTRIBUTED_LEGACY_ENV_MAP.items():
        # New env var takes precedence over legacy.
        if os.environ.get(new_name) is not None:
            continue
        legacy_val = os.environ.get(legacy_name)
        if legacy_val is None:
            continue
        current = getattr(_settings, attr_name)
        if isinstance(current, bool):
            legacy_val = legacy_val.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            try:
                legacy_val = int(legacy_val)
            except (ValueError, TypeError):
                logger.warning(
                    "Could not forward legacy %s=%r: not a valid int",
                    legacy_name,
                    legacy_val,
                )
                continue
        elif isinstance(current, Path):
            legacy_val = Path(legacy_val).expanduser()
        try:
            setattr(_settings, attr_name, legacy_val)
            logger.warning(
                "Forwarding legacy %s=%r → settings.%s. Rename to %s.",
                legacy_name,
                legacy_val,
                attr_name,
                new_name,
            )
        except Exception as exc:
            logger.warning(
                "Could not forward legacy %s=%r: %s",
                legacy_name,
                legacy_val,
                exc,
            )


def _apply_serve_overrides(args) -> None:
    """Apply CLI flags to the global Settings before the server starts.

    The flags are written to the ``settings`` instance so that the rest of
    the codebase (which reads ``from olmlx.config import settings``) picks
    them up without needing extra plumbing.
    """
    from olmlx.config import settings as _settings

    _surface_legacy_speculative_env()
    _surface_legacy_dflash_env()
    _surface_legacy_flash_env()
    _surface_legacy_flash_moe_env()
    _surface_legacy_flash_prefetch_speculative_env()

    # ``getattr`` defends programmatic callers that hand a bare
    # ``argparse.Namespace`` (e.g. tests) without populating these
    # attributes. The parser-derived defaults already cover bare
    # ``olmlx`` invocation; this is just a safety net.
    spec = getattr(args, "speculative", None)
    spec_strategy = getattr(args, "speculative_strategy", None)
    spec_draft = getattr(args, "speculative_draft_model", None)
    spec_tokens = getattr(args, "speculative_tokens", None)
    spec_layers_skip = getattr(args, "speculative_layers_skip", None)
    if spec is not None:
        _settings.speculative = spec
    if spec_strategy is not None:
        _settings.speculative_strategy = spec_strategy
    if spec_draft is not None:
        _settings.speculative_draft_model = spec_draft
    if spec_tokens is not None:
        _settings.speculative_tokens = spec_tokens
    if spec_layers_skip is not None:
        _settings.speculative_layers_skip = spec_layers_skip

    kvq = getattr(args, "kv_cache_quant", None)
    if kvq is not None:
        _settings.kv_cache_quant = kvq

    flash_flag = getattr(args, "flash", None)
    if flash_flag is not None:
        _settings.flash = flash_flag

    fs = getattr(args, "flash_speculative", None)
    fs_draft = getattr(args, "flash_speculative_draft_model", None)
    fs_tokens = getattr(args, "flash_speculative_tokens", None)
    fp = getattr(args, "flash_prefetch", None)
    if fs is not None:
        _settings.flash_speculative = fs
    if fs_draft is not None:
        _settings.flash_speculative_draft_model = fs_draft
    if fs_tokens is not None:
        _settings.flash_speculative_tokens = fs_tokens
    if fp is not None:
        _settings.flash_prefetch = fp

    if _settings.flash_speculative and not _settings.flash:
        logger.warning(
            "flash_speculative is set but flash is not enabled globally; "
            "it will only take effect for models with flash:true in models.json."
        )
    if _settings.flash_prefetch and not _settings.flash:
        logger.warning(
            "flash_prefetch is set but flash is not enabled globally; "
            "it will only take effect for models with flash:true in models.json."
        )

    _surface_legacy_kv_cache_quant_env()

    _warn_kv_cache_quant_incompatibilities()

    # Surface speculative misconfigurations at startup by walking the
    # registry and checking each model's ``resolved_speculative()``. This
    # is precise: it accepts ``OLMLX_SPECULATIVE=true`` with no global
    # draft as long as every registered model supplies its own. It also
    # catches per-model entries that enable speculative without a draft.
    # The "global speculative=true and zero registered models" case is
    # not flagged here — the first model load will raise a clear error.
    needs_migration = _models_with_promoted_keys_in_experimental()
    if needs_migration:
        from olmlx.engine.registry import PROMOTED_EXPERIMENTAL_KEYS

        promoted_list = ", ".join(repr(k) for k in sorted(PROMOTED_EXPERIMENTAL_KEYS))
        print(
            "Error: the following models in models.json still place "
            "speculative settings under 'experimental' — these keys have "
            f"been promoted to top-level fields. Move {promoted_list} "
            "out of the 'experimental' block. Affected entries: "
            f"{', '.join(needs_migration)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Share one ``ModelRegistry`` instance across the two audit helpers
    # that consume it (``_audit_speculative_config`` and
    # ``_audit_per_model_flash_in_distributed``) so the disk read
    # happens once *and* both audits see the same registry state. If
    # the centralised load fails, both skip together — letting each
    # helper retry on its own could leave one succeeding while another
    # silently skips on the same transient I/O failure, producing an
    # asymmetric view of the operator's config and requiring multiple
    # restart cycles to surface every problem. Note: the earlier
    # ``_models_with_promoted_keys_in_experimental`` call uses a raw
    # ``json.load`` (no registry instance), so it's not part of this
    # coordination and runs independently.
    audit_registry = _load_registry_for_audit()
    if audit_registry is None:
        return
    bad, dormant_drafts, flash_conflicts, dflash_moe_conflicts, global_draft_used = (
        _audit_speculative_config(audit_registry)
    )
    if dormant_drafts:
        logger.warning(
            "speculative_draft_model is configured for the following "
            "models but speculative decoding is disabled "
            "(speculative=false), so the draft model will be ignored: %s. "
            "Set speculative=true (per-model or globally) to enable.",
            ", ".join(dormant_drafts),
        )
    # Warn whenever the global draft is set but no model actually
    # consumes it. ``global_draft_used`` already encodes "any model
    # resolves to the global draft", so it's the only signal we need —
    # global ``speculative=True`` paired with per-model drafts on
    # every entry is just as dormant as ``speculative=False``. The
    # message wording is narrow on purpose: it claims only that the
    # global draft is unused.
    if _settings.speculative_draft_model and not global_draft_used:
        logger.warning(
            "OLMLX_SPECULATIVE_DRAFT_MODEL is set to %r but no model "
            "consumes it: nothing inherits the global draft. The "
            "setting has no effect until a model with ``speculative=true`` "
            "(and no per-model draft override) is configured.",
            _settings.speculative_draft_model,
        )
    # Suppress the flash-conflict warning for models that are also in
    # ``bad`` — telling the user "use flash_speculative" is misleading
    # when the actual error is the missing draft model.
    flash_conflicts_actionable = [m for m in flash_conflicts if m not in set(bad)]
    if flash_conflicts_actionable:
        # Note: standalone speculative is only dropped when the Flash
        # bundle actually loads — if the bundle directory is missing
        # ``_load_model`` falls through to the standard load path and
        # the standalone speculative decoder still runs. The startup
        # warning is advisory; the authoritative runtime warning lives
        # in ``_load_model`` for the case where Flash actually wins.
        logger.warning(
            "The following models combine speculative=true with Flash: "
            "%s. Once Flash is prepared and loads, standalone "
            "speculative decoding is dropped — use the per-model "
            "``flash_speculative`` field (or "
            "OLMLX_FLASH_SPECULATIVE / "
            "OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL / "
            "OLMLX_FLASH_SPECULATIVE_TOKENS) instead.",
            ", ".join(flash_conflicts_actionable),
        )
    # dflash on Flash-MoE models will raise ValueError at load time once
    # the Flash-MoE bundle is prepared and loaded; warn at startup so
    # users see the incompatibility early. Filter models also in ``bad``
    # (missing draft) — the "use classic strategy" suggestion is
    # misleading when there is no draft model configured at all.
    dflash_moe_actionable = [m for m in dflash_moe_conflicts if m not in set(bad)]
    if dflash_moe_actionable:
        logger.warning(
            "The following models combine speculative_strategy='dflash' "
            "with Flash-MoE. Once Flash-MoE is prepared and loads, this "
            "will raise a ValueError at load time (dflash requires "
            "hidden-state capture that does not generalize to MoE "
            "routing): %s. Use speculative_strategy='classic', "
            "'self_speculative', or set speculative=false.",
            ", ".join(dflash_moe_actionable),
        )
    if bad:
        print(
            "Error: the following models in models.json enable speculative "
            "decoding but have no draft model configured (per-model or "
            f"global): {', '.join(bad)}. Set 'speculative_draft_model' on "
            "each entry or set OLMLX_SPECULATIVE_DRAFT_MODEL globally.",
            file=sys.stderr,
        )
        sys.exit(1)

    _audit_per_model_flash_in_distributed(audit_registry)


def _audit_per_model_flash_in_distributed(
    registry: "ModelRegistry | None" = None,
) -> None:
    """Audit per-model Flash settings against distributed-mode invariants.

    Two failure modes to surface at startup:

    * **Hard error — coordinator/worker model-structure mismatch.** If
      any per-model ``resolved_flash().enabled`` differs from
      ``settings.flash``, the coordinator and workers would load
      structurally different models (one with ``FlashModelWrapper``
      replacing FFN layers, the other dense). The ring ``all_sum``
      then operates over mismatched layer shapes and crashes
      inference. Bail at startup with a clear migration nudge instead
      of letting the cluster spin up only to die during the first
      request. Mirrors the existing flash + pipeline-strategy guard.

    * **Warning — silently ignored per-model overrides.** When
      ``resolved_flash().enabled`` matches the global but a model has
      per-model values for the four numeric Flash knobs
      (``sparsity_threshold``, ``min/max_active_neurons``,
      ``memory_budget_fraction``), those overrides are honoured only
      on the coordinator. The worker path reads globals via
      ``_load_flash_tensor_worker`` and never consults the registry,
      so the per-model values are silently dropped. Log a warning so
      the operator notices before debugging "why is the neuron cap
      different on rank 1".
    """
    if not settings.distributed:
        return
    if registry is None:
        registry = _load_registry_for_audit()
        if registry is None:
            return
    numeric_fields = (
        "flash_sparsity_threshold",
        "flash_min_active_neurons",
        "flash_max_active_neurons",
        "flash_memory_budget_fraction",
    )
    mismatched: list[str] = []
    numeric_only: list[str] = []
    numeric_fields_by_model: dict[str, list[str]] = {}
    for name, mc in registry.list_models().items():
        try:
            resolved = mc.resolved_flash()
        except Exception as exc:
            logger.warning(
                "Could not audit Flash config for %s in distributed "
                "mode: %s. This model may fail to load on workers — "
                "check ``flash_min_active_neurons`` / "
                "``flash_max_active_neurons`` for cross-field "
                "violations against the global Settings.",
                name,
                exc,
            )
            continue
        if resolved.enabled != settings.flash:
            mismatched.append(name)
            continue
        # Only warn about numeric overrides when Flash is actually
        # enabled — overrides on a model that resolves to ``flash=False``
        # are inert on both coordinator and worker, so claiming they
        # are "silently dropped on workers" would mislead the user
        # into thinking Flash was running.
        if resolved.enabled:
            set_fields = [
                field
                for field in numeric_fields
                if getattr(mc, field, None) is not None
            ]
            if set_fields:
                numeric_only.append(name)
                numeric_fields_by_model[name] = set_fields

    if mismatched:
        print(
            "Error: distributed mode is enabled and the following "
            "models.json entries have a per-model Flash on/off that "
            "disagrees with the global OLMLX_FLASH setting "
            f"({settings.flash}): {', '.join(mismatched)}. The "
            "coordinator and workers would load structurally different "
            "models (one Flash-wrapped, one dense) and crash on the "
            "ring all_sum. Either remove the per-model 'flash' override "
            "from these entries, or set OLMLX_FLASH globally to match.",
            file=sys.stderr,
        )
        sys.exit(1)

    if numeric_only:
        details = ", ".join(
            f"{name} [{', '.join(numeric_fields_by_model[name])}]"
            for name in numeric_only
        )
        logger.warning(
            "Distributed mode is enabled and the following models.json "
            "entries have per-model Flash numeric overrides "
            "(sparsity_threshold/min/max/memory_budget_fraction): %s. "
            "Per-model numeric overrides are honoured only on the "
            "coordinator; the distributed worker path uses the global "
            "OLMLX_FLASH_* settings. Set the desired values globally "
            "for them to take effect on every rank.",
            details,
        )


def _models_with_promoted_keys_in_experimental() -> list[str]:
    """Return models.json entry names whose ``experimental`` block still
    contains the promoted speculative keys.

    Such entries are dropped by ``ModelRegistry.load()`` with a buried
    log warning; surfacing them as a hard startup error makes the
    migration actionable instead of mysterious. The set of promoted
    keys is taken directly from ``registry.PROMOTED_EXPERIMENTAL_KEYS``
    so the next promotion wires through automatically.
    """
    from olmlx.engine.registry import PROMOTED_EXPERIMENTAL_KEYS

    try:
        with open(settings.models_config) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return []
    except OSError as exc:
        # Permission denied, IsADirectoryError, etc. — degrade
        # gracefully like ``_audit_speculative_config`` does for the
        # registry load. Crashing startup over an unreadable
        # models.json is worse than skipping the migration check.
        logger.warning(
            "Skipping speculative migration check: could not read models.json: %s",
            exc,
        )
        return []
    except json.JSONDecodeError as exc:
        # A corrupt models.json hides any pending migration; surface it
        # as a warning here so the operator notices, rather than letting
        # the audit's broad except swallow the same failure later.
        logger.warning(
            "Skipping speculative migration check: models.json is invalid JSON: %s",
            exc,
        )
        return []
    if not isinstance(raw, dict):
        return []
    promoted_keys = set(PROMOTED_EXPERIMENTAL_KEYS.keys())
    bad: list[str] = []
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        exp = entry.get("experimental")
        if isinstance(exp, dict) and promoted_keys & exp.keys():
            bad.append(name)
    return bad


def _load_registry_for_audit() -> "ModelRegistry | None":
    """Load the ModelRegistry once for the startup-audit helpers.

    Returns the loaded registry or ``None`` if the load failed.
    ``_apply_serve_overrides`` skips *all* audits together when this
    returns ``None`` so a transient I/O failure can't let one audit
    succeed while another silently skips on the same startup
    (asymmetric output → multiple restart cycles to surface every
    problem). Standalone callers of the audit helpers (e.g. tests)
    still get the helper's own retry-and-skip fallback.

    Note: ``ModelsConfigError`` (raised by ``ModelRegistry.load`` when
    ``models.json`` is corrupt/unreadable) is intentionally NOT caught
    here. Returning ``None`` for that case would silently skip startup
    and let a later save clobber the file — exactly the behaviour the
    strict load was added to prevent. The exception propagates to
    ``cli_main``, which prints it and exits cleanly. Do not add a
    ``ModelsConfigError`` branch here.
    """
    from olmlx.engine.registry import ModelRegistry

    registry = ModelRegistry()
    try:
        registry.load()
    except ValueError as exc:
        # Validation errors (a malformed entry that survived the
        # ``_models_with_promoted_keys_in_experimental`` raw-JSON
        # check) are operator errors, not transient I/O issues —
        # flag them distinctly so the log makes the cause clear.
        # ``ModelRegistry.load`` itself catches per-entry ValueError
        # today and logs them, so this branch fires only if validation
        # moves earlier in the load sequence.
        logger.warning(
            "Skipping startup registry audit: invalid models.json entry: %s", exc
        )
        return None
    except OSError as exc:
        logger.warning(
            "Skipping startup registry audit: could not load registry: %s", exc
        )
        return None
    return registry


def _audit_speculative_config(
    registry: "ModelRegistry | None" = None,
) -> tuple[list[str], list[str], list[str], list[str], bool]:
    """Walk the registry and audit each model's resolved speculative
    config.

    Returns ``(bad, dormant_drafts, flash_conflicts, global_draft_used)``:
    - ``bad`` — models with ``speculative=True`` but no draft model
      anywhere. Triggers a startup error.
    - ``dormant_drafts`` — models with a per-model ``speculative_draft_model``
      set but the resolved ``enabled`` flag is False. Triggers a
      warning so users don't silently lose the draft they configured.
    - ``flash_conflicts`` — models that combine ``speculative=True``
      with Flash in the same entry. Standalone speculative decoding is
      silently dropped on the Flash load path; the model's own
      ``flash_speculative`` knob is the right one. Triggers a warning
      so users see the redirect. Flash-MoE supports standalone speculative
      (classic strategy only) and is excluded from this check.
    - ``dflash_moe_conflicts`` — models that combine a feature-conditioned
      speculative strategy (``dflash``/``eagle``/``mtp``, i.e.
      ``_FLASH_MOE_INCOMPATIBLE_STRATEGIES``) with Flash-MoE. Triggers a
      warning since these are unsupported on Flash-MoE targets (raises
      ValueError at load time).
    - ``global_draft_used`` — True if at least one model resolves to
      the global ``speculative_draft_model`` (i.e. has ``speculative=True``
      and no per-model draft override). Used to suppress the global
      dormant-draft warning when the global draft actually has consumers.

    The registry is loaded from disk; failures are logged and treated
    as "nothing to validate" so this never blocks startup on its own.
    """
    from olmlx.engine.registry import _FLASH_MOE_INCOMPATIBLE_STRATEGIES

    if registry is None:
        registry = _load_registry_for_audit()
    if registry is None:
        return [], [], [], [], False
    bad: list[str] = []
    dormant: list[str] = []
    flash_conflicts: list[str] = []
    dflash_moe_conflicts: list[str] = []
    global_draft_used = False
    for name, mc in registry.list_models().items():
        try:
            resolved = mc.resolved_speculative()
            enabled = resolved.enabled
            draft = resolved.draft_model
            strategy = resolved.strategy
        except ValueError as exc:
            # Configuration error (e.g. PLD ngram/window cross-field
            # invariant violated by the global+per-model combination).
            # Surface as an error rather than a warning so it's not
            # lost in startup chatter; the request itself will fail
            # later with the same message, but seeing it at startup
            # is more diagnosable.
            logger.error(
                "Speculative config for %s is invalid and will fail "
                "at model-load time: %s",
                name,
                exc,
            )
            continue
        except Exception as exc:
            # Unexpected (e.g. settings in an unexpected state).
            # Skipping the entry is safer than killing startup with
            # a stack trace from an unrelated cause.
            logger.warning(
                "Skipping audit of %s: could not resolve speculative config: %s",
                name,
                exc,
                exc_info=True,
            )
            continue
        if enabled and not draft and strategy != "self_speculative":
            bad.append(name)
        elif not enabled and mc.speculative_draft_model:
            # Use the raw per-model field rather than the resolved
            # ``draft``: the global dormant-draft case is already
            # surfaced separately in ``_apply_serve_overrides``.
            dormant.append(name)
        if enabled and mc.speculative_draft_model is None and draft is not None:
            # This model enables speculative without a per-model draft,
            # so it is consuming the global ``speculative_draft_model``.
            # Note: a per-model entry that copies the global draft path
            # verbatim into its own ``speculative_draft_model`` looks
            # "not consuming the global" here, even though the values
            # are identical. That's intentional — the user wrote a
            # per-model override, so the global setting is still
            # logically unused for that model.
            global_draft_used = True
        if enabled and draft and strategy == "self_speculative":
            # ``self_speculative`` uses the target's own layers as
            # draft — an external draft model set in the config is
            # silently ignored by ``_load_self_speculative_decoder``.
            # Warn so the operator knows the draft model setting has
            # no effect while this strategy is active.
            logger.warning(
                "speculative_draft_model is set for %r but "
                "strategy='self_speculative' uses the target's own "
                "layers — the draft model will be ignored.",
                name,
            )
        if enabled:
            # Resolve the full experimental config (global defaults
            # merged with per-model overrides) for ``flash`` and via
            # ``mc.resolved_flash_moe()`` for ``flash_moe``. Both are
            # promoted to top-level fields; ``experimental`` no longer
            # carries them.
            resolved_flash = None
            try:
                resolved_flash = mc.resolved_flash()
            except Exception as exc:
                logger.warning(
                    "Skipping flash conflict check for %s: could not "
                    "resolve flash overrides: %s",
                    name,
                    exc,
                    exc_info=True,
                )
            if resolved_flash is not None and resolved_flash.enabled:
                flash_conflicts.append(name)
            if (
                mc.resolved_flash_moe().enabled
                and strategy in _FLASH_MOE_INCOMPATIBLE_STRATEGIES
            ):
                dflash_moe_conflicts.append(name)
    return bad, dormant, flash_conflicts, dflash_moe_conflicts, global_draft_used


# Module-level state set by cmd_serve() for the app lifespan to retrieve.
_cli_distributed_group = None
_cli_distributed_coordinator = None
_cli_distributed_strategy = "tensor"
_cli_distributed_layer_counts = None

_VALID_HOSTNAME_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


_worker_procs: list[subprocess.Popen] = []
_worker_log_fhs: list = []
_atexit_registered = False


def _cleanup_workers():
    """Terminate all distributed worker processes and close log file handles."""
    for proc in _worker_procs:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait()
            except Exception:
                pass
        except Exception:
            pass
    for fh in _worker_log_fhs:
        try:
            fh.close()
        except Exception:
            pass
    _worker_procs.clear()
    _worker_log_fhs.clear()


def _pre_shard_and_distribute(
    hosts, model, world_size, settings, strategy="tensor", layer_counts=None
) -> bool:
    """Pre-shard model weights and distribute to workers via SCP.

    Returns True on success, False on failure (caller should fall back).
    """
    import shlex

    from olmlx.engine.pre_shard import (
        pre_shard_all_workers,
        pre_shard_pipeline_all_workers,
        read_shard_marker,
    )
    from olmlx.models.store import _safe_dir_name

    store = _create_store()
    try:
        model_dir = store.ensure_downloaded(model)
    except Exception as e:
        logger.warning("Failed to download model for pre-sharding: %s", e)
        return False

    safe_name = _safe_dir_name(model)
    shard_base = Path(settings.distributed_shard_dir).expanduser() / safe_name

    # Resolve default layer_counts for pipeline so marker comparison works
    # when the hostfile omits an explicit "layers" key.
    if strategy == "pipeline" and layer_counts is None:
        try:
            from olmlx.engine.pipeline import _compute_layer_counts

            cfg = json.loads((model_dir / "config.json").read_text())
            layer_counts = _compute_layer_counts(cfg["num_hidden_layers"], world_size)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning("Failed to read config.json for layer_counts: %s", e)
            return False

    # Check if valid shards already exist
    all_valid = True
    for rank in range(1, world_size):
        shard_dir = shard_base / f"rank{rank}"
        marker = read_shard_marker(shard_dir)
        if (
            marker is None
            or marker.get("model_path") != str(model_dir)
            or marker.get("world_size") != world_size
            or marker.get("rank") != rank
            or marker.get("strategy", "tensor") != strategy
        ):
            all_valid = False
            break
        if strategy == "pipeline" and marker.get("layer_counts") != layer_counts:
            all_valid = False
            break

    if all_valid:
        print("  Pre-sharded weights already exist, skipping re-shard")
    else:
        print(f"  Pre-sharding model for {world_size - 1} worker(s)...")
        try:
            if strategy == "pipeline":
                pre_shard_pipeline_all_workers(
                    model_dir,
                    world_size=world_size,
                    output_base=shard_base,
                    layer_counts=layer_counts,
                    progress_cb=lambda r, ws: print(f"    Sharded rank {r}/{ws - 1}"),
                )
            else:
                pre_shard_all_workers(
                    model_dir,
                    world_size=world_size,
                    output_base=shard_base,
                    progress_cb=lambda r, ws: print(f"    Sharded rank {r}/{ws - 1}"),
                )
        except Exception as e:
            logger.warning("Pre-sharding failed: %s", e)
            return False

    # SCP shards to each worker
    # Resolve ~ to absolute path so we can safely shlex.quote for SSH commands.
    worker_shard_dir = str(Path(settings.distributed_worker_shard_dir).expanduser())
    for rank, host in enumerate(hosts[1:], start=1):
        shard_dir = shard_base / f"rank{rank}"
        remote_dir = f"{worker_shard_dir}/{safe_name}/rank{rank}"

        # Create remote directory
        mkdir_cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            f"mkdir -p {shlex.quote(remote_dir)}",
        ]
        try:
            subprocess.run(mkdir_cmd, check=True, capture_output=True, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("Failed to create remote dir on %s: %s", host, e)
            return False

        # SCP with compression — no shlex.quote: SCP args aren't shell-processed
        scp_cmd = [
            "scp",
            "-C",
            "-o",
            "BatchMode=yes",
            "-r",
            f"{shard_dir}/.",
            f"{host}:{remote_dir}/",
        ]
        print(f"  Transferring shard to {host} rank {rank}...")
        try:
            subprocess.run(scp_cmd, check=True, capture_output=True, timeout=600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("SCP to %s failed: %s", host, e)
            return False

    print("  Pre-sharded weights distributed to all workers")
    return True


def validate_remote_python(remote_python: str) -> None:
    """Validate remote_python against a strict allowlist to prevent shell injection.

    The remote_python config value is intentionally not shell-quoted (to allow
    multi-word values like "uv run python"), so it must be validated to prevent
    command injection via SSH.  Note: remote_working_dir does not need a similar
    allowlist because it is passed through shlex.quote() before interpolation.
    """
    if not re.fullmatch(r"[a-zA-Z0-9_ /.@-]+", remote_python):
        raise ValueError(f"Invalid remote_python value: {remote_python!r}")


def _launch_distributed_workers() -> tuple[list[str], str, list[int] | None]:
    """Launch worker processes on remote hosts via SSH for distributed inference.

    Returns the list of hosts from the hostfile.
    Stores Popen handles in _worker_procs for cleanup on failure/shutdown.
    Requires passwordless SSH with pre-accepted host keys — run
    `ssh-keyscan -H <host> >> ~/.ssh/known_hosts` for each worker first.
    """
    import atexit
    import shlex

    from olmlx.config import settings

    hostfile_path = Path(settings.distributed_hostfile).expanduser()
    if not hostfile_path.exists():
        print(
            f"Error: distributed hostfile not found at {hostfile_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(hostfile_path) as f:
        hostfile = json.load(f)

    hosts = hostfile.get("hosts", [])
    if len(hosts) < 2:
        print(
            "Error: hostfile must contain at least 2 hosts for distributed inference",
            file=sys.stderr,
        )
        sys.exit(1)

    model = hostfile.get("model", "")
    if not model:
        print(
            "Error: hostfile must contain a 'model' field with the HF model path",
            file=sys.stderr,
        )
        sys.exit(1)

    strategy = hostfile.get("strategy", "tensor")
    if strategy != "tensor":
        print(
            f"Error: distributed inference is tensor-only; hostfile strategy "
            f"must be 'tensor', got {strategy!r}. The pipeline strategy is not "
            f"supported.",
            file=sys.stderr,
        )
        sys.exit(1)

    hostfile_layers = hostfile.get("layers")
    if hostfile_layers is not None:
        if not isinstance(hostfile_layers, list) or not all(
            isinstance(x, int) and x > 0 for x in hostfile_layers
        ):
            print(
                "Error: hostfile 'layers' must be a list of positive integers",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(hostfile_layers) != len(hosts):
            print(
                f"Error: hostfile 'layers' has {len(hostfile_layers)} entries "
                f"but there are {len(hosts)} hosts (must match)",
                file=sys.stderr,
            )
            sys.exit(1)

    # Validate hostnames to prevent command injection
    for host in hosts:
        if not _VALID_HOSTNAME_RE.match(host):
            print(
                f"Error: invalid hostname {host!r} in hostfile",
                file=sys.stderr,
            )
            sys.exit(1)

    world_size = len(hosts)
    coordinator_host = hosts[0]
    print(f"Distributed mode: {world_size} nodes, coordinator={coordinator_host}")

    log_dir = Path.home() / ".olmlx"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate ring hostfile for MLX distributed backend
    ring_hostfile_data = [
        [f"{h}:{settings.distributed_port + i}"] for i, h in enumerate(hosts)
    ]
    max_port = settings.distributed_port + len(hosts) - 1
    if max_port > 65535:
        print(
            f"Error: distributed_port {settings.distributed_port} + "
            f"{len(hosts)} hosts exceeds port limit 65535",
            file=sys.stderr,
        )
        sys.exit(1)
    ring_hostfile_path = log_dir / "ring_hostfile.json"
    with open(ring_hostfile_path, "w") as f:
        json.dump(ring_hostfile_data, f)

    # Set coordinator env vars for MLX ring backend
    os.environ["MLX_RANK"] = "0"
    os.environ["MLX_HOSTFILE"] = str(ring_hostfile_path)

    ring_hostfile_json = json.dumps(ring_hostfile_data)

    global _atexit_registered

    if not _atexit_registered:
        atexit.register(_cleanup_workers)
        _atexit_registered = True

    remote_python = settings.distributed_remote_python
    validate_remote_python(remote_python)
    remote_working_dir = settings.distributed_remote_working_dir

    if settings.flash_moe:
        print(
            "Error: Flash-MoE + distributed is not supported. "
            "Disable OLMLX_FLASH_MOE or OLMLX_DISTRIBUTED.",
            file=sys.stderr,
        )
        sys.exit(1)

    if settings.flash and strategy == "pipeline":
        print(
            "Error: Flash + pipeline distributed strategy is not supported. "
            "Use tensor strategy or disable Flash.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Pre-shard and distribute weights to workers if enabled
    pre_sharded = False
    if settings.distributed_pre_shard:
        if settings.flash:
            logger.info(
                "Skipping pre-sharding: Flash mode shards only attention "
                "layers at runtime, MLP weights are loaded from SSD on "
                "each node independently."
            )
        else:
            pre_sharded = _pre_shard_and_distribute(
                hosts,
                model,
                world_size,
                settings,
                strategy=strategy,
                layer_counts=hostfile_layers,
            )

    # Pre-compute safe model name for env var paths (used when pre-sharded)
    from olmlx.config import PRE_SHARDED_DIR_ENV
    from olmlx.models.store import _safe_dir_name

    safe_name = _safe_dir_name(model) if pre_sharded else ""
    # Keep ~ as-is: the worker calls expanduser() on the received path
    worker_shard_dir = settings.distributed_worker_shard_dir if pre_sharded else ""

    # Launch workers on remote hosts (rank 1..N)
    for rank, host in enumerate(hosts[1:], start=1):
        env = {
            "OLMLX_DISTRIBUTED_MODEL": model,
            "OLMLX_DISTRIBUTED_BACKEND": settings.distributed_backend,
            "OLMLX_DISTRIBUTED_COORDINATOR_HOST": coordinator_host,
            "OLMLX_DISTRIBUTED_SIDEBAND_PORT": str(settings.distributed_sideband_port),
            "OLMLX_DISTRIBUTED_STRATEGY": strategy,
            "MLX_RANK": str(rank),
        }
        if hostfile_layers is not None:
            env["OLMLX_DISTRIBUTED_LAYER_COUNTS"] = ",".join(
                str(x) for x in hostfile_layers
            )
        if pre_sharded:
            env[PRE_SHARDED_DIR_ENV] = f"{worker_shard_dir}/{safe_name}/rank{rank}"
        # Forward promoted settings so workers use the same config as the
        # coordinator. Resolve per-model overrides so a models.json entry
        # with ``kv_cache_quant: "turboquant:2"`` reaches workers even
        # when the global OLMLX_KV_CACHE_QUANT is unset.
        _resolved_kvq = settings.kv_cache_quant
        if model:
            try:
                from olmlx.engine.registry import ModelRegistry

                reg = ModelRegistry()
                reg.load()
                mc = reg.resolve(model)
                if mc is not None:
                    _resolved_kvq = mc.resolved_kv_cache_quant()
            except Exception as exc:
                logger.debug(
                    "Skipping per-model kv_cache_quant resolution for distributed "
                    "worker: %s",
                    exc,
                )
        if _resolved_kvq:
            env["OLMLX_KV_CACHE_QUANT"] = _resolved_kvq
        _resolved_wq = settings.weight_quant
        if model:
            try:
                from olmlx.engine.registry import ModelRegistry

                reg = ModelRegistry()
                reg.load()
                mc = reg.resolve(model)
                if mc is not None:
                    _resolved_wq = mc.resolved_weight_quant()
            except Exception as exc:
                logger.debug(
                    "Skipping per-model weight_quant resolution for distributed "
                    "worker: %s",
                    exc,
                )
        if _resolved_wq:
            env["OLMLX_WEIGHT_QUANT"] = _resolved_wq
        if settings.flash:
            env["OLMLX_FLASH"] = "true"
            # Forward the *resolved* primary knobs (from ``settings``)
            # rather than relying on os.environ passthrough. The worker
            # process does not run ``_surface_legacy_flash_env``, so a
            # user with only legacy env vars set (e.g.
            # ``OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD``) would
            # otherwise see the worker fall back to schema defaults for
            # the numeric knobs. Sourcing from ``settings`` mirrors
            # whatever the coordinator's legacy shim already applied.
            env["OLMLX_FLASH_SPARSITY_THRESHOLD"] = str(
                settings.flash_sparsity_threshold
            )
            env["OLMLX_FLASH_MIN_ACTIVE_NEURONS"] = str(
                settings.flash_min_active_neurons
            )
            if settings.flash_max_active_neurons is not None:
                env["OLMLX_FLASH_MAX_ACTIVE_NEURONS"] = str(
                    settings.flash_max_active_neurons
                )
            if settings.flash_memory_budget_fraction is not None:
                env["OLMLX_FLASH_MEMORY_BUDGET_FRACTION"] = str(
                    settings.flash_memory_budget_fraction
                )
            # Forward all OLMLX_EXPERIMENTAL_FLASH_* vars verbatim.
            # This intentionally includes:
            #   - the advanced tuning knobs that still live under the
            #     experimental prefix (window_size, io_threads,
            #     cache_budget_neurons, predictor_*, prefetch_*,
            #     bypass_os_cache, preallocated_buffer);
            #   - the five *promoted* legacy primary knobs (e.g.
            #     OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD) when
            #     the user hasn't renamed them yet. Each worker also
            #     runs ``surface_legacy_flash_env``, which prefers
            #     the new-name vars already added above — so the
            #     legacy copies are harmless redundancy during the
            #     one-release deprecation window.
            # ``OLMLX_EXPERIMENTAL_FLASH_MOE`` also matches this
            # prefix but is safe: the flash_moe guard above already
            # exited if it was true.
            for key, val in os.environ.items():
                if key in env:
                    continue
                if key.startswith("OLMLX_EXPERIMENTAL_FLASH_"):
                    env[key] = val
        env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())

        script_parts = [
            "HOSTFILE=$(mktemp)",
            'trap "rm -f $HOSTFILE ${SECRET_FILE:-}" EXIT',
            f"echo {shlex.quote(ring_hostfile_json)} > $HOSTFILE",
            "export MLX_HOSTFILE=$HOSTFILE",
        ]
        if remote_working_dir:
            script_parts.append(f"cd {shlex.quote(remote_working_dir)}")

        if settings.distributed_secret:
            script_parts.extend(
                [
                    "SECRET_FILE=$(mktemp)",
                    f"printf '%s' {shlex.quote(settings.distributed_secret)} > $SECRET_FILE",
                    "chmod 600 $SECRET_FILE",
                    "export OLMLX_DISTRIBUTED_SECRET_FILE=$SECRET_FILE",
                ]
            )

        script_parts.append(
            f"{env_str} {remote_python} -m olmlx.engine.distributed_worker"
        )
        remote_cmd = "; ".join(script_parts)

        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            host,
            remote_cmd,
        ]
        log_file = log_dir / f"worker-{rank}.log"
        print(f"  Launching worker rank {rank} on {host} (log: {log_file})")
        log_fh = open(log_file, "w")
        try:
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)
        except Exception:
            log_fh.close()
            raise
        _worker_log_fhs.append(log_fh)
        _worker_procs.append(proc)

    # Check for immediate SSH failures (auth errors, bad host, etc.)
    time.sleep(0.2)
    for proc in _worker_procs:
        if proc.poll() is not None:
            logger.error("Worker process exited immediately (rc=%d)", proc.returncode)
            _cleanup_workers()
            raise RuntimeError("Worker SSH launch failed — check worker logs")

    return hosts, strategy, hostfile_layers


def _find_executable() -> str:
    """Find the olmlx executable path."""
    exe = shutil.which("olmlx")
    if exe:
        return exe
    # Fallback: use the current Python interpreter with -m
    return sys.executable


# Heuristics for spotting credential-bearing env vars so they are never
# persisted to the cleartext plist. Suffix matching (not substring) for the
# generic markers so the LLM-token config keys this codebase is full of
# (OLMLX_SPECULATIVE_TOKENS, OLMLX_PROMPT_CACHE_MAX_TOKENS, …) are NOT mistaken
# for credentials — only an actual trailing _TOKEN/_KEY/_PASSWORD counts.
_SECRET_ENV_SUFFIXES = (
    "_SECRET",
    "_TOKEN",
    "_KEY",
    "_APIKEY",
    "_PASSWORD",
    "_PASSWD",
    "_CREDENTIAL",
    "_CREDENTIALS",
)
# Substrings that are unambiguous credentials regardless of position.
_SECRET_ENV_SUBSTRINGS = ("SECRET", "PASSWORD", "CREDENTIAL")


def _is_secret_env_key(key: str) -> bool:
    """True if an env var name looks like it carries a credential."""
    upper = key.upper()
    return upper.endswith(_SECRET_ENV_SUFFIXES) or any(
        marker in upper for marker in _SECRET_ENV_SUBSTRINGS
    )


def _build_plist() -> dict:
    """Build a launchd plist dict for the olmlx service."""
    exe = _find_executable()
    if exe == sys.executable:
        program_args = [exe, "-m", "olmlx"]
    else:
        program_args = [exe]

    env_vars = {}
    # Forward OLMLX_ env vars if set, but never persist secrets into the
    # cleartext launchd plist — ~/Library/LaunchAgents/com.olmlx.plist is
    # readable by any local process and is not a safe credential store (#454).
    for key, value in os.environ.items():
        if key.startswith("OLMLX_") and not _is_secret_env_key(key):
            env_vars[key] = value
    # Ensure PATH includes common tool locations
    env_vars["PATH"] = os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin")

    plist = {
        "Label": PLIST_LABEL,
        "ProgramArguments": program_args,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(Path.home() / ".olmlx" / "olmlx.log"),
        "StandardErrorPath": str(Path.home() / ".olmlx" / "olmlx.log"),
        "EnvironmentVariables": env_vars,
    }
    return plist


def cmd_service_install(_args):
    """Install and load the launchd service."""
    ensure_config()
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    plist = _build_plist()
    with open(PLIST_PATH, "wb") as f:
        plistlib.dump(plist, f)
    print(f"Wrote {PLIST_PATH}")

    try:
        subprocess.run(
            ["launchctl", "load", str(PLIST_PATH)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else "(no output)"
        print(
            f"Plist was written to {PLIST_PATH} but the service could not be loaded.\n"
            f"launchctl stderr: {stderr}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Service {PLIST_LABEL} loaded")


def cmd_service_uninstall(_args):
    """Unload and remove the launchd service."""
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], check=False)
        PLIST_PATH.unlink()
        print(f"Service {PLIST_LABEL} unloaded and plist removed")
    else:
        print(f"No plist found at {PLIST_PATH}")


def cmd_service_status(_args):
    """Show the status of the launchd service."""
    result = subprocess.run(
        ["launchctl", "list", PLIST_LABEL],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"Service {PLIST_LABEL} is loaded")
        print(result.stdout.strip())
    else:
        print(f"Service {PLIST_LABEL} is not loaded")


def _create_store():
    """Create a ModelStore instance for CLI use.

    Raises on failure — callers are responsible for catching and exiting.
    """
    from olmlx.engine.registry import ModelRegistry
    from olmlx.models.store import ModelStore

    ensure_config()
    registry = ModelRegistry()
    registry.load()
    return ModelStore(registry)


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.1f} GB"
    elif size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f} MB"
    elif size_bytes >= 1e3:
        return f"{size_bytes / 1e3:.1f} KB"
    return f"{size_bytes} B"


def cmd_models_list(_args):
    """List locally downloaded models."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        models = store.list_local()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if not models:
        print("No models downloaded.")
        return
    # Header
    print(f"{'NAME':<30} {'SIZE':<12} {'PARAMS':<10} {'QUANT':<10} {'HF PATH'}")
    print("-" * 90)
    for m in sorted(models, key=lambda x: x.name or ""):
        name = (m.name or "")[:30]
        params = (m.parameter_size or "")[:10]
        quant = (m.quantization_level or "")[:10]
        print(
            f"{name:<30} {_format_size(m.size):<12} "
            f"{params:<10} {quant:<10} {m.hf_path}"
        )


def cmd_models_search(args):
    """Search for models by name."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    results = store.registry.search(args.query)
    if not results:
        print(f"No models matching '{args.query}'.")
        return
    print(f"{'NAME':<30} {'HF PATH'}")
    print("-" * 60)
    for name, hf_path in results:
        print(f"{name:<30} {hf_path}")


def cmd_models_show(args):
    """Show details for a specific model."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        manifest = store.show(args.model_name)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if manifest is None:
        print(
            f"Model '{args.model_name}' not found locally.\n"
            f"Try: olmlx models search {args.model_name}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Name:           {manifest.name}")
    print(f"HF Path:        {manifest.hf_path}")
    print(f"Size:           {_format_size(manifest.size)}")
    print(f"Format:         {manifest.format}")
    print(f"Family:         {manifest.family}")
    print(f"Parameters:     {manifest.parameter_size}")
    print(f"Quantization:   {manifest.quantization_level}")
    print(f"Modified:       {manifest.modified_at}")
    print(f"Digest:         {manifest.digest}")


def cmd_models_pull(args):
    """Pull/download a model."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    async def _pull():
        async for status in store.pull(args.model_name):
            if msg := status.get("status", ""):
                print(msg, flush=True)

    try:
        asyncio.run(_pull())
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        msg = f"Error: {e}"
        if "not found" in str(e).lower():
            msg += f"\nTry: olmlx models search {args.model_name}"
        print(msg, file=sys.stderr, flush=True)
        sys.exit(1)


def cmd_models_delete(args):
    """Delete a locally downloaded model."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if not args.yes:
        try:
            confirm = input(f"Delete model '{args.model_name}'? [y/N] ")
        except EOFError:
            print("Aborted.")
            return
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return
    try:
        deleted = store.delete(args.model_name)
    except Exception as e:
        print(f"Error deleting model '{args.model_name}': {e}", file=sys.stderr)
        sys.exit(1)
    if deleted:
        print(f"Model '{args.model_name}' deleted.")
    else:
        print(f"Model '{args.model_name}' not found locally.", file=sys.stderr)
        sys.exit(1)


def _configure_logging():
    """Configure logging from settings."""
    from olmlx.context import RequestIDFormatter

    handler = logging.StreamHandler()
    handler.setFormatter(
        RequestIDFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level))
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)


_VOICE_FLAGS = ("--voice", "--stt-model", "--tts-model", "--voice-name")


def _build_chat_arg_parser_voice_defaults() -> tuple[str, ...]:
    """Return the voice flag strings the chat subparser registers (test seam)."""
    return _VOICE_FLAGS


def _check_voice_deps() -> None:
    """Exit with an install hint if voice deps are unavailable."""
    import importlib.util

    def _absent(name: str) -> bool:
        mod = sys.modules.get(name, "x")
        if mod is None:
            # Explicitly stubbed-out as missing.
            return True
        if mod != "x":
            # Already imported as a real module object -> present.
            return False
        try:
            return importlib.util.find_spec(name) is None
        except Exception:
            return True

    # Only sounddevice lives in the [voice] extra; mlx-audio (Kokoro TTS) is a
    # core dependency for /v1/audio/speech, so `--extra voice` wouldn't install
    # it. Gate on sounddevice alone — the dep the hint can actually fix.
    if _absent("sounddevice"):
        print(
            "--voice needs the 'sounddevice' package (PortAudio). "
            "Install with: uv sync --extra voice",
            file=sys.stderr,
        )
        raise SystemExit(1)


def cmd_chat(args):
    """Start an interactive chat session."""
    from olmlx.chat.config import ChatConfig, load_mcp_config, load_tool_safety_config
    from olmlx.chat.mcp_client import MCPClientManager
    from olmlx.chat.session import ChatSession
    from olmlx.chat.tool_safety import ToolPolicy, ToolSafetyPolicy
    from olmlx.chat.tui import ChatTUI
    from olmlx.engine.model_manager import ModelManager

    ensure_config()
    _configure_logging()
    # ``olmlx chat`` reads ``settings.speculative*`` via ModelManager
    # too, so honour the deprecation window here. Without this a user
    # who only runs chat would silently lose forwarding even though
    # ``serve`` handles it correctly.
    _surface_legacy_speculative_env()
    _surface_legacy_kv_cache_quant_env()
    _surface_legacy_flash_env()
    _surface_legacy_flash_moe_env()
    _surface_legacy_flash_prefetch_speculative_env()
    _warn_kv_cache_quant_incompatibilities()

    model_name = args.model_name
    if model_name is None:
        print("Error: model name required. Usage: olmlx chat <model>", file=sys.stderr)
        sys.exit(1)

    chat_kwargs: dict[str, Any] = dict(
        model_name=model_name,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        thinking=not args.no_thinking,
        mcp_enabled=not args.no_mcp,
        repeat_penalty=args.repeat_penalty,
        repeat_last_n=args.repeat_last_n,
        skills_enabled=not args.no_skills,
        builtin_tools_enabled=not args.no_builtin_tools,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tool_timeout=args.tool_timeout,
        mcp_connect_retries=args.mcp_connect_retries,
        local_tool_safety=args.local_tool_safety,
        tool_result_truncation=args.tool_result_truncation,
        max_consecutive_tool_failures=args.max_consecutive_tool_failures,
    )
    # Filter out None for nullable args so ChatConfig defaults apply.
    # Boolean flags (store_true) are never None — only filter numeric args.
    for _key in (
        "temperature",
        "top_p",
        "top_k",
        "tool_timeout",
        "mcp_connect_retries",
        "tool_result_truncation",
        "max_consecutive_tool_failures",
    ):
        if chat_kwargs[_key] is None:
            del chat_kwargs[_key]
    # Boolean store_true flags default to False. Drop them when False so
    # ChatConfig's default wins — if the default ever flips to True, the CLI
    # won't silently override it back to False.
    for _key in ("local_tool_safety",):
        if not chat_kwargs.get(_key):
            chat_kwargs.pop(_key, None)  # ChatConfig default wins
    if args.mcp_config:
        chat_kwargs["mcp_config_path"] = Path(args.mcp_config)
    if args.skills_dir:
        chat_kwargs["skills_dir"] = Path(args.skills_dir)
    config = ChatConfig(**chat_kwargs)

    async def _run_chat():
        from olmlx.chat.builtin_tools import BuiltinToolManager
        from olmlx.chat.skills import SkillManager

        store = _create_store()
        manager = ModelManager(store.registry, store)

        tui = ChatTUI(tool_result_truncation=config.tool_result_truncation)
        mcp = None
        skills = None
        builtin = None

        try:
            tui.console.print(f"[dim]Loading {model_name}...[/dim]")
            await manager.ensure_loaded(model_name, keep_alive="-1")

            if config.mcp_enabled:
                mcp_cfg = load_mcp_config(config.mcp_config_path)
                if mcp_cfg:
                    mcp = MCPClientManager()
                    await mcp.connect_all(
                        mcp_cfg, max_attempts=config.mcp_connect_retries
                    )

            if config.skills_enabled:
                skills = SkillManager(config.skills_dir)
                skills.load()
                if skills.list_skills():
                    tui.console.print(
                        f"[dim]Loaded {len(skills.list_skills())} skill(s)[/dim]"
                    )

            if config.builtin_tools_enabled:
                builtin = BuiltinToolManager(config)

            # Load tool safety policy
            safety_config = load_tool_safety_config(config.mcp_config_path)
            active_stream_ctx = None

            async def confirm_decider(name: str, args: dict) -> bool:
                nonlocal active_stream_ctx
                if active_stream_ctx and active_stream_ctx.is_active:
                    active_stream_ctx.finish()
                return await asyncio.to_thread(tui.confirm_tool_call, name, args)

            def _build_judge():
                from olmlx.chat.llm_judge import SafeJudge

                return SafeJudge(
                    manager,
                    model_name=lambda: (
                        safety_config.judge_model
                        if safety_config.judge_model
                        else config.model_name
                    ),
                )

            llm_judge = None
            uses_auto = safety_config.default_policy == ToolPolicy.AUTO or any(
                p == ToolPolicy.AUTO for p in safety_config.tool_policies.values()
            )
            if uses_auto:
                llm_judge = _build_judge()
                if safety_config.judge_model:
                    tui.console.print(
                        f"[dim]LLM judge using separate model: "
                        f"{safety_config.judge_model}[/dim]"
                    )

            policy = ToolSafetyPolicy(
                safety_config,
                decider=confirm_decider,
                llm_judge=llm_judge,
            )

            session = ChatSession(
                config=config,
                manager=manager,
                mcp=mcp,
                skills=skills,
                builtin=builtin,
                tool_safety=policy,
            )
            tools = mcp.get_tools_for_chat() if mcp else []
            if builtin:
                tools = tools + builtin.get_tool_definitions()
            tui.display_welcome(model_name, tools)

            voice_io = None
            if getattr(args, "voice", False):
                _check_voice_deps()
                from olmlx.chat.voice.io import VoiceIO

                stt_model = args.stt_model or settings.chat_stt_model
                tts_model = args.tts_model or settings.chat_tts_model
                voice_name = args.voice_name or settings.chat_tts_voice
                tui.console.print(f"[dim]Loading STT {stt_model}...[/dim]")
                await manager.ensure_loaded(stt_model, keep_alive="-1")
                tui.console.print(f"[dim]Loading TTS {tts_model}...[/dim]")
                await manager.ensure_loaded(tts_model, keep_alive="-1")
                voice_io = VoiceIO(
                    manager=manager,
                    stt_model=stt_model,
                    tts_model=tts_model,
                    voice=voice_name,
                )
                tui.console.print(
                    "[dim]Voice on. Press Enter at the prompt to talk; "
                    "type to send text.[/dim]"
                )

            while True:
                user_input = tui.get_user_input()
                if user_input is None:
                    break

                # Empty line in voice mode => push-to-talk.
                if voice_io is not None and not user_input.strip():
                    try:
                        user_input = await voice_io.listen()
                    except RuntimeError as exc:  # device/dep failure
                        tui.display_error(str(exc))
                        continue
                    if not user_input:
                        continue
                    tui.console.print(f"[dim]heard:[/dim] {user_input}")

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    cmd_parts = user_input.split(None, 1)
                    command = cmd_parts[0].lower()
                    arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                    if command in ("/exit", "/quit"):
                        break
                    elif command == "/clear":
                        session.clear_history()
                        tui.console.print("[dim]History cleared[/dim]")
                    elif command == "/tools":
                        tui.display_tools(tools)
                    elif command == "/skills":
                        if skills and skills.list_skills():
                            tui.console.print("[bold]Skills:[/bold]")
                            for s in skills.list_skills():
                                desc = f" — {s.description}" if s.description else ""
                                tui.console.print(f"  [cyan]{s.name}[/cyan]{desc}")
                        else:
                            tui.console.print("[dim]No skills loaded[/dim]")
                    elif command == "/safety":
                        tui.display_safety_policy(policy)
                    elif command == "/mode":
                        if arg == "auto":
                            new_default = ToolPolicy.AUTO
                        elif arg == "confirm":
                            new_default = ToolPolicy.CONFIRM
                        else:
                            tui.display_error("Usage: /mode auto|confirm")
                            continue
                        safety_config.default_policy = new_default
                        if new_default == ToolPolicy.CONFIRM:
                            # Clear per-tool AUTO overrides so tools that
                            # were auto-judged are now confirmed manually.
                            # ALLOW and DENY overrides are intentionally
                            # preserved — the user explicitly configured
                            # those and switching to confirm mode shouldn't
                            # undo that policy.
                            cleared = [
                                name
                                for name, pol in list(
                                    safety_config.tool_policies.items()
                                )
                                if pol == ToolPolicy.AUTO
                            ]
                            for name in cleared:
                                del safety_config.tool_policies[name]
                            if cleared:
                                tui.console.print(
                                    f"[dim]Cleared AUTO override(s): "
                                    f"{', '.join(cleared)}[/dim]"
                                )
                        if new_default == ToolPolicy.AUTO and llm_judge is None:
                            llm_judge = _build_judge()
                            policy.llm_judge = llm_judge
                            tui.console.print("[dim]LLM judge initialised[/dim]")
                        tui.console.print(
                            f"[dim]Default policy: {new_default.value}[/dim]"
                        )
                    elif command == "/system":
                        if arg:
                            config.system_prompt = arg
                            session.clear_history()
                            tui.console.print(
                                "[dim]System prompt set. History cleared.[/dim]"
                            )
                        else:
                            current = config.system_prompt or "(none)"
                            tui.console.print(f"[dim]System prompt: {current}[/dim]")
                    elif command == "/model":
                        model_parts = arg.split(None, 1)
                        if model_parts and model_parts[0] == "thinking":
                            if len(model_parts) == 2 and model_parts[1] in (
                                "on",
                                "off",
                            ):
                                config.thinking = model_parts[1] == "on"
                                tui.console.print(
                                    f"[dim]Thinking: {'on' if config.thinking else 'off'}[/dim]"
                                )
                            else:
                                thinking_str = "on" if config.thinking else "off"
                                tui.console.print(
                                    f"[dim]Thinking: {thinking_str}. Use: /model thinking on|off[/dim]"
                                )
                        elif arg:
                            tui.console.print(f"[dim]Loading {arg}...[/dim]")
                            try:
                                await manager.ensure_loaded(arg, keep_alive="-1")
                                config.model_name = arg
                                session.clear_history()
                                tui.console.print(
                                    f"[dim]Switched to {arg}. History cleared.[/dim]"
                                )
                            except Exception as exc:
                                tui.display_error(str(exc))
                        else:
                            thinking_str = "on" if config.thinking else "off"
                            tui.console.print(
                                f"[dim]Current model: {config.model_name} | thinking: {thinking_str}[/dim]"
                            )
                    else:
                        tui.display_error(f"Unknown command: {command}")
                    continue

                # Collect events while streaming tokens
                pending_events = []
                confirmed_tool_ids: set[str] = set()
                spoken_parts: list[str] = []
                stream_ctx = tui.stream_response()
                active_stream_ctx = stream_ctx
                try:
                    with stream_ctx:
                        async for event in session.send_message(user_input):
                            if event["type"] == "thinking_start":
                                stream_ctx.start_thinking()
                            elif event["type"] == "thinking_end":
                                stream_ctx.end_thinking()
                            elif event["type"] == "thinking_token":
                                stream_ctx.update(event["text"])
                            elif event["type"] == "token":
                                stream_ctx.update(event["text"])
                                if voice_io is not None:
                                    spoken_parts.append(event["text"])
                            elif event["type"] == "tool_approved":
                                # Track confirmed IDs to avoid duplicate display
                                confirmed_tool_ids.add(event["id"])
                            else:
                                pending_events.append(event)
                    # Track if question was asked for post-processing
                    question_asked = any(
                        e.get("type") == "question" for e in pending_events
                    )
                    if question_asked:
                        # Find the question event and ask user
                        for event in pending_events:
                            if event.get("type") == "question":
                                answer = tui.ask_question(
                                    event.get("header", ""),
                                    event.get("question", ""),
                                    options=event.get("options"),
                                    multiple=event.get("multiple", False),
                                )
                                if answer is not None:
                                    user_input = answer
                                    break
                finally:
                    active_stream_ctx = None

                # Display collected events
                for event in pending_events:
                    if event["type"] == "tool_call":
                        # Skip display for confirmed tools — already shown
                        # by confirm_tool_call during the prompt
                        if event["id"] not in confirmed_tool_ids:
                            tui.display_tool_call(event["name"], event["arguments"])
                    elif event["type"] == "tool_result":
                        tui.display_tool_result(event["name"], event["result"])
                    elif event["type"] == "tool_error":
                        tui.display_tool_error(event["name"], event["error"])
                    elif event["type"] == "tool_denied":
                        # Only show panel for policy-denied or auto-denied
                        # tools; user-denied tools were already shown
                        # at the confirm prompt
                        if event.get("reason") != "user":
                            tui.display_tool_denied(
                                event["name"], reason=event.get("reason", "policy")
                            )
                    elif event["type"] == "tool_auto_judging":
                        tui.display_tool_auto_judging(event["name"])
                    elif event["type"] == "tool_confirmation_needed":
                        pass  # handled inline by decider callback
                    elif event["type"] == "max_turns_exceeded":
                        tui.display_error("Max tool turns reached")
                    elif event["type"] == "tool_failures_exceeded":
                        tui.display_tool_failures_exceeded(event["message"])
                    elif event["type"] == "memory_truncated":
                        tui.display_memory_truncated(event["message"])
                    elif event["type"] == "repetition_detected":
                        tui.display_repetition_detected()
                    elif event["type"] == "model_load_error":
                        tui.display_model_load_error(event["error"])
                        break

                if voice_io is not None and spoken_parts:
                    try:
                        await voice_io.speak("".join(spoken_parts))
                    except RuntimeError as exc:
                        tui.display_error(str(exc))

        except MemoryError as exc:
            tui.display_error(str(exc))
            sys.exit(1)
        except ValueError as exc:
            tui.display_error(str(exc))
            sys.exit(1)
        finally:
            if mcp is not None:
                await mcp.disconnect_all()
            await manager.stop()

    try:
        asyncio.run(_run_chat())
    except KeyboardInterrupt:
        print("\nBye!", file=sys.stderr)


def cmd_config_show(_args):
    """Show current configuration."""
    _surface_legacy_kv_cache_quant_env()
    _surface_legacy_distributed_env()
    _surface_legacy_flash_env()
    _surface_legacy_flash_moe_env()
    _surface_legacy_flash_prefetch_speculative_env()

    print(f"Host:                   {settings.host}")
    print(f"Port:                   {settings.port}")
    print(f"Models dir:             {settings.models_dir}")
    print(f"Models config:          {settings.models_config}")
    print(f"Default keep-alive:     {settings.default_keep_alive}")
    print(f"Max loaded models:      {settings.max_loaded_models}")
    print(f"Memory limit fraction:  {settings.memory_limit_fraction}")
    print(f"Log level:              {settings.log_level}")
    print(f"Prompt cache:           {settings.prompt_cache}")
    print(f"Prompt cache max tokens: {settings.prompt_cache_max_tokens}")
    print(f"CORS origins:           {settings.cors_origins}")
    if settings.kv_cache_quant:
        print(f"KV cache quant:         {settings.kv_cache_quant}")
    if settings.flash:
        print("Flash inference:        enabled")
        print(f"  Sparsity threshold:   {settings.flash_sparsity_threshold}")
        print(f"  Min active neurons:   {settings.flash_min_active_neurons}")
        if settings.flash_max_active_neurons is not None:
            print(f"  Max active neurons:   {settings.flash_max_active_neurons}")
        if settings.flash_memory_budget_fraction is not None:
            print(f"  Memory budget frac:   {settings.flash_memory_budget_fraction}")
    if settings.distributed:
        print()
        print("Distributed inference:")
        print(f"  Hostfile:             {settings.distributed_hostfile}")
        print(f"  Backend:              {settings.distributed_backend}")
        print(f"  Port:                 {settings.distributed_port}")
        print(f"  Sideband port:        {settings.distributed_sideband_port}")


def cmd_bench_run(args):
    """Run benchmark scenarios."""
    _configure_logging()
    _surface_legacy_flash_env()
    _surface_legacy_flash_moe_env()
    _surface_legacy_flash_prefetch_speculative_env()
    _surface_legacy_speculative_env()
    _surface_legacy_dflash_env()
    _surface_legacy_kv_cache_quant_env()
    from pathlib import Path

    from olmlx.bench.runner import run_bench
    from olmlx.bench.results import DEFAULT_BENCH_DIR

    scenario_names = (
        [s.strip() for s in args.scenarios.split(",") if s.strip()]
        if args.scenarios
        else None
    )
    bench_dir = Path(args.bench_dir) if args.bench_dir else DEFAULT_BENCH_DIR

    run_bench(
        model=args.model,
        scenario_names=scenario_names,
        max_tokens=args.max_tokens,
        bench_dir=bench_dir,
        prompt_set=args.prompt_set,
        enable_code_exec=args.enable_code_exec,
    )


def cmd_bench_compare(args):
    """Compare two benchmark runs."""
    from pathlib import Path

    from olmlx.bench.results import DEFAULT_BENCH_DIR, compare_runs, load_run

    def _resolve_run(ref: str) -> Path:
        p = Path(ref)
        if p.exists():
            return p
        # Try as timestamp under default bench dir
        candidate = DEFAULT_BENCH_DIR / ref
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Run not found: {ref}")

    run1 = load_run(_resolve_run(args.run1))
    run2 = load_run(_resolve_run(args.run2))
    print(compare_runs(run1, run2))


def cmd_bench_list(args):
    """List past benchmark runs."""
    from pathlib import Path

    from olmlx.bench.results import DEFAULT_BENCH_DIR, list_runs

    bench_dir = Path(args.bench_dir) if args.bench_dir else DEFAULT_BENCH_DIR
    runs = list_runs(bench_dir)
    if not runs:
        print("No benchmark runs found.")
        return

    print(
        f"{'Timestamp':<22} {'Model':<45} {'Git':<10} {'Scenarios':>9} {'Skipped':>7}"
    )
    print("-" * 95)
    for r in runs:
        print(
            f"{r['timestamp']:<22} {r['model']:<45} {r['git_sha'] or '—':<10} "
            f"{r['scenarios']:>9} {r['skipped']:>7}"
        )


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _non_empty_str(value: str) -> str:
    """argparse ``type`` validator that mirrors ``Field(min_length=1)``
    on the corresponding Settings field. Without it, ``--flag ""``
    propagates an empty string into Settings and surfaces as an
    unhandled ``ValidationError`` traceback at startup. Surrounding
    whitespace is stripped so that ``--flag " hf/path "`` doesn't
    later fail with a confusing path-not-found error."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("value must be a non-empty string")
    return stripped


def cmd_bench_leaderboard(args):
    """Show the model leaderboard derived from saved bench runs."""
    from pathlib import Path

    from olmlx.bench.results import (
        DEFAULT_BENCH_DIR,
        build_leaderboard,
        format_leaderboard,
    )

    bench_dir = Path(args.bench_dir) if args.bench_dir else DEFAULT_BENCH_DIR
    entries = build_leaderboard(bench_dir, latest_per_model=not args.all_runs)
    if not entries:
        print("No bench runs with valid measurements found.")
        return
    print(format_leaderboard(entries, limit=args.limit))


def _flash_progress(desc, frac):
    bar_len = 30
    filled = int(bar_len * frac)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  {desc:<40s} [{bar}] {frac:5.1%}", end="", flush=True)
    if frac >= 1.0:
        print()


def cmd_spectral_prepare(args):
    """Prepare a model for spectral quant (eigenspectral calibration)."""
    _configure_logging()

    store = _create_store()
    _resolved = store.registry.resolve(args.model)
    hf_path = _resolved.hf_path if _resolved is not None else args.model
    local_dir = store.ensure_downloaded(hf_path)
    model_path = str(local_dir)

    print(f"Running spectral calibration for {args.model}...")
    print(f"  Model path: {model_path}")
    print(f"  Average bits: {args.avg_bits}")
    dataset_label = args.calibration_dataset or "c4"
    print(f"  Calibration dataset: {dataset_label}")
    print(f"  Calibration samples: {args.samples}")
    print(f"  Max tokens per head: {args.max_tokens}")
    print()

    from olmlx.engine.spectralquant_calibrate import calibrate_model

    output_dir = calibrate_model(
        model_path=model_path,
        num_samples=args.samples,
        calibration_dataset=args.calibration_dataset,
        avg_bits=args.avg_bits,
        max_tokens_per_head=args.max_tokens,
        progress_callback=_flash_progress,
    )

    print("\nSpectral calibration complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use spectral quant:")
    print(f"  OLMLX_KV_CACHE_QUANT=spectral:{args.avg_bits} olmlx serve")


def cmd_flash_prepare(args):
    """Prepare a model for flash inference (auto-detects MoE vs dense)."""
    _configure_logging()
    # Forward legacy env vars so OLMLX_EXPERIMENTAL_FLASH_PREFETCH=true
    # reaches settings.flash_prefetch before _cmd_flash_dense_prepare reads it
    # via train_lookahead=settings.flash_prefetch.
    _surface_legacy_flash_env()
    _surface_legacy_flash_moe_env()
    _surface_legacy_flash_prefetch_speculative_env()

    store = _create_store()
    _resolved = store.registry.resolve(args.model)
    hf_path = _resolved.hf_path if _resolved is not None else args.model
    local_dir = store.ensure_downloaded(hf_path)
    model_path = str(local_dir)

    # Auto-detect MoE model
    from olmlx.engine.flash.moe_prepare import is_moe_model

    if is_moe_model(model_path):
        _cmd_flash_moe_prepare(args, model_path)
    else:
        _cmd_flash_dense_prepare(args, model_path)


def _cmd_flash_moe_prepare(args, model_path):
    """Prepare an MoE model for Flash-MoE inference."""
    from olmlx.engine.flash.moe_prepare import prepare_moe_for_flash

    print("Detected MoE model — using Flash-MoE preparation (no model loading needed)")
    print(f"  Model path: {model_path}")
    print()

    output_dir = prepare_moe_for_flash(
        model_path=model_path,
        progress_callback=_flash_progress,
    )

    print("\nFlash-MoE preparation complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use Flash-MoE inference:")
    print("  OLMLX_EXPERIMENTAL_FLASH_MOE=true olmlx serve")


def _cmd_flash_dense_prepare(args, model_path):
    """Prepare a dense model for flash inference."""
    from olmlx.engine.flash.prepare import prepare_model_for_flash

    print(f"Preparing {args.model} for flash inference...")
    print(f"  Model path: {model_path}")
    print(f"  Predictor rank: {args.rank}")
    if args.sensitive_layers > 0:
        print(
            f"  Sensitive layers: last {args.sensitive_layers} (rank x{args.sensitive_rank_multiplier})"
        )
    dataset_label = args.calibration_dataset or "c4"
    print(f"  Calibration dataset: {dataset_label}")
    print(f"  Calibration samples: {args.samples}")
    print(f"  Activation threshold: {args.threshold}")
    print(f"  Training epochs: {args.epochs}")
    print()

    output_dir = prepare_model_for_flash(
        model_path=model_path,
        rank=args.rank,
        sensitive_layers=args.sensitive_layers,
        sensitive_rank_multiplier=args.sensitive_rank_multiplier,
        num_samples=args.samples,
        calibration_dataset=args.calibration_dataset,
        activation_threshold=args.threshold,
        epochs=args.epochs,
        train_lookahead=settings.flash_prefetch,
        progress_callback=_flash_progress,
    )

    print("\nFlash preparation complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use flash inference:")
    print("  olmlx serve --flash")
    print("  # or set OLMLX_FLASH=true")


def cmd_dflash_precompute(args):
    """Precompute target hidden states for DFlash draft training."""
    _configure_logging()

    store = _create_store()
    _resolved = store.registry.resolve(args.model)
    hf_path = _resolved.hf_path if _resolved is not None else args.model
    local_dir = store.ensure_downloaded(hf_path)
    model_path = str(local_dir)

    target_layer_ids: list[int] | None = None
    if args.target_layer_ids:
        try:
            target_layer_ids = [int(x) for x in args.target_layer_ids.split(",")]
        except ValueError as exc:
            raise SystemExit(
                f"--target-layer-ids must be a comma-separated list of "
                f"integers, got {args.target_layer_ids!r}: {exc}"
            ) from exc

    output_dir = Path(args.output) if args.output else Path(model_path) / "dflash_cache"

    print(f"Precomputing target hidden states for {args.model}...")
    print(f"  Target path: {model_path}")
    print(f"  Output: {output_dir}")
    print(f"  Shards: {args.shards}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq len: {args.seq_len}")
    if target_layer_ids:
        print(f"  Target layer ids: {target_layer_ids}")
    else:
        print(f"  Target layers: {args.num_target_layers} (evenly spaced)")
    print()

    from mlx_lm import load as _mlx_lm_load

    from olmlx.engine.dflash.decoder import _patch_model, _unpatch_model
    from olmlx.engine.dflash.precompute import precompute_target_hiddens
    from olmlx.engine.dflash.prepare import _resolve_target_layer_ids
    from olmlx.engine.dflash.training_data import stream_training_batches

    # ``mlx_lm.load`` returns a 2-tuple in current versions; older
    # variants returned 3-tuples. Slice to the first two so either
    # works.
    loaded = _mlx_lm_load(model_path)
    target, tokenizer = loaded[0], loaded[1]
    target.eval()
    if hasattr(target, "freeze"):
        target.freeze()

    target_layers_attr = (
        target.layers if hasattr(target, "layers") else target.model.layers
    )
    layer_ids = _resolve_target_layer_ids(
        target_layer_ids, args.num_target_layers, len(target_layers_attr)
    )
    print(f"  Resolved target_layer_ids: {layer_ids}\n")

    # Caller-owned hidden-state storage — kept off the ``nn.Module`` so
    # mlx's parameter tracker doesn't pick the captures up.
    hidden_capture: list[Any] = [None] * len(layer_ids)
    _patch_model(target, layer_ids, hidden_capture)
    try:
        # ``max_examples`` is intentionally omitted — ``num_shards``
        # below is the load-bearing cap (it's also what writes the
        # correct count into ``index.json``); a duplicate cap on the
        # iterator side would just mask off-by-one issues.
        batches = stream_training_batches(
            tokenizer,
            dataset=args.data or "HuggingFaceH4/ultrachat_200k",
            split=args.split or "train_sft",
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        precompute_target_hiddens(
            target,
            batches,
            output_dir,
            hidden_capture,
            target_layer_ids=layer_ids,
            num_shards=args.shards,
            progress_callback=_flash_progress,
        )
    finally:
        _unpatch_model(target)

    print("\nPrecompute complete!")
    print(f"  Output: {output_dir}")
    print(
        f"  Re-use with: olmlx dflash prepare {args.model} "
        f"--use-precomputed {output_dir}"
    )


def cmd_dflash_prepare(args):
    """Train a DFlash draft model for a target."""
    _configure_logging()

    # Validate up-front so the user gets a clear message before we
    # download the target and run hook-installation. The same check
    # exists deep inside the training loop, but surfacing it after a
    # multi-GB download is a poor UX.
    if args.block_size < 1:
        raise SystemExit(f"--block-size must be >= 1, got {args.block_size}")
    min_seq_len = 2 * args.block_size + 1
    if args.seq_len < min_seq_len:
        raise SystemExit(
            f"--seq-len ({args.seq_len}) too small for --block-size "
            f"({args.block_size}); need at least 2*block_size + 1 = "
            f"{min_seq_len} tokens per sequence."
        )
    if args.train_windows_per_step < 1:
        raise SystemExit(
            f"--train-windows-per-step must be >= 1, got {args.train_windows_per_step}"
        )

    store = _create_store()
    _resolved = store.registry.resolve(args.model)
    hf_path = _resolved.hf_path if _resolved is not None else args.model
    local_dir = store.ensure_downloaded(hf_path)
    model_path = str(local_dir)

    target_layer_ids: list[int] | None = None
    if args.target_layer_ids:
        try:
            target_layer_ids = [int(x) for x in args.target_layer_ids.split(",")]
        except ValueError as exc:
            raise SystemExit(
                f"--target-layer-ids must be a comma-separated list of "
                f"integers, got {args.target_layer_ids!r}: {exc}"
            ) from exc

    print(f"Training DFlash draft for {args.model}...")
    print(f"  Target path: {model_path}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq len: {args.seq_len}")
    print(f"  Block size (draft tokens): {args.block_size}")
    print(f"  Draft layers: {args.num_hidden_layers}")
    if target_layer_ids:
        print(f"  Target layer ids: {target_layer_ids}")
    else:
        print(f"  Target layers: {args.num_target_layers} (evenly spaced)")
    print(f"  Dataset: {args.data}")
    print(f"  LR: {args.lr}")
    if args.distill:
        print(f"  Distillation: alpha={args.distill_alpha} temp={args.distill_temp}")
    if args.train_windows_per_step != 1:
        print(f"  Train windows per step: {args.train_windows_per_step}")
    if args.use_precomputed:
        print(f"  Precomputed shards: {args.use_precomputed}")
    print()

    from olmlx.engine.dflash.prepare import prepare_dflash_draft

    output_dir = prepare_dflash_draft(
        model_path=model_path,
        dataset=args.data,
        dataset_split=args.split,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        block_size=args.block_size,
        num_hidden_layers=args.num_hidden_layers,
        target_layer_ids=target_layer_ids,
        num_target_layers=args.num_target_layers,
        mask_token_id=args.mask_token_id,
        lr=args.lr,
        output_dir=args.output,
        distill=args.distill,
        distill_alpha=args.distill_alpha,
        distill_temp=args.distill_temp,
        position_decay_gamma=args.position_decay_gamma,
        train_windows_per_step=args.train_windows_per_step,
        use_precomputed=args.use_precomputed,
        progress_callback=_flash_progress,
    )

    print("\nDFlash draft training complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use the trained draft:")
    print("  OLMLX_SPECULATIVE=true \\")
    print("  OLMLX_SPECULATIVE_STRATEGY=dflash \\")
    print(f"  OLMLX_SPECULATIVE_DRAFT_MODEL={output_dir} \\")
    print("  olmlx serve")


def cmd_eagle_prepare(args):
    """Train an EAGLE draft model for a target.

    Phase D supports only ``--use-precomputed`` mode: pass the
    directory of (input_ids, target_hidden) shards produced by
    ``olmlx dflash precompute`` (the shard format is shared between
    DFlash and EAGLE — EAGLE consumes the deepest captured layer).
    """
    _configure_logging()

    if args.block_size < 1:
        raise SystemExit(f"--block-size must be >= 1, got {args.block_size}")
    if not args.use_precomputed:
        raise SystemExit(
            "--use-precomputed is required for EAGLE training. Run "
            "`olmlx dflash precompute <target>` first to dump target hidden "
            "states; the same shards work for EAGLE (it just slices the "
            "deepest layer from the concatenated ladder)."
        )

    store = _create_store()
    _resolved = store.registry.resolve(args.model)
    hf_path = _resolved.hf_path if _resolved is not None else args.model
    local_dir = store.ensure_downloaded(hf_path)
    model_path = str(local_dir)

    print(f"Training EAGLE draft for {args.model}...")
    print(f"  Target path: {model_path}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq len: {args.seq_len}")
    print(f"  Block size (draft tokens): {args.block_size}")
    print(f"  Draft layers: {args.num_hidden_layers}")
    print(f"  LR: {args.lr}")
    sample_positions = args.sample_positions if args.sample_positions > 0 else None
    print(f"  Sample positions/step: {sample_positions or 'all'}")
    print(f"  Precomputed shards: {args.use_precomputed}")
    print()

    from olmlx.engine.eagle.prepare import prepare_eagle_draft

    output_dir = prepare_eagle_draft(
        model_path=model_path,
        use_precomputed=args.use_precomputed,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        block_size=args.block_size,
        num_hidden_layers=args.num_hidden_layers,
        lr=args.lr,
        sample_positions=sample_positions,
        seed=args.seed,
        output_dir=args.output,
        progress_callback=_flash_progress,
    )

    print("\nEAGLE draft training complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use the trained draft:")
    print("  OLMLX_SPECULATIVE=true \\")
    print("  OLMLX_SPECULATIVE_STRATEGY=eagle \\")
    print(f"  OLMLX_SPECULATIVE_DRAFT_MODEL={output_dir} \\")
    print("  olmlx serve")


def cmd_flash_info(args):
    """Show flash preparation info for a model."""
    # Honour the legacy ``OLMLX_EXPERIMENTAL_FLASH*`` shim so that
    # ``olmlx flash info`` reflects the same effective Flash settings
    # an operator would see from ``olmlx config show`` / ``olmlx serve``
    # when they've only renamed the new env vars in some shells.
    _surface_legacy_flash_env()
    _surface_legacy_flash_moe_env()
    _surface_legacy_flash_prefetch_speculative_env()
    store = _create_store()
    _resolved = store.registry.resolve(args.model)
    hf_path = _resolved.hf_path if _resolved is not None else args.model
    local_dir = store.local_path(hf_path)

    # Check for Flash-MoE first
    flash_moe_dir = local_dir / "flash_moe"
    flash_dir = local_dir / "flash"

    if flash_moe_dir.exists() and (flash_moe_dir / "flash_moe_config.json").exists():
        _show_flash_moe_info(args.model, flash_moe_dir)
    elif flash_dir.exists():
        _show_flash_dense_info(args.model, flash_dir)
    else:
        print(f"Model '{args.model}' has not been prepared for flash inference.")
        print(f"\nRun: olmlx flash prepare {args.model}")


def _show_flash_moe_info(model_name, flash_moe_dir):
    config_path = flash_moe_dir / "flash_moe_config.json"
    config = json.loads(config_path.read_text())
    print(f"Flash-MoE info for '{model_name}':")
    print("  Status:             prepared")
    print("  Type:               MoE (expert offloading)")
    print(f"  Flash directory:    {flash_moe_dir}")
    print(f"  Hidden size:        {config.get('hidden_size')}")
    print(f"  Intermediate size:  {config.get('intermediate_size')}")
    print(f"  Num experts:        {config.get('num_experts')}")
    print(f"  Experts per token:  {config.get('num_experts_per_tok')}")
    print(f"  MoE layers:         {config.get('num_moe_layers')}")
    print(f"  Prepared at:        {config.get('prepared_at')}")

    fe_files = list(flash_moe_dir.glob("*.flashexperts"))
    print(f"  Expert files:       {len(fe_files)}")

    total_bytes = sum(f.stat().st_size for f in flash_moe_dir.rglob("*") if f.is_file())
    if total_bytes > 1024**3:
        print(f"  Total size:         {total_bytes / (1024**3):.1f} GB")
    else:
        print(f"  Total size:         {total_bytes / (1024**2):.1f} MB")

    print("\nTo use Flash-MoE inference:")
    print("  OLMLX_EXPERIMENTAL_FLASH_MOE=true olmlx serve")


def _show_flash_dense_info(model_name, flash_dir):
    config_path = flash_dir / "flash_config.json"
    if not config_path.exists():
        print(f"Flash directory exists but no config found: {flash_dir}")
        return

    config = json.loads(config_path.read_text())
    print(f"Flash info for '{model_name}':")
    print("  Status:             prepared")
    print("  Type:               Dense (neuron offloading)")
    print(f"  Flash directory:    {flash_dir}")
    print(f"  Hidden size:        {config.get('hidden_size')}")
    print(f"  Intermediate size:  {config.get('intermediate_size')}")
    print(f"  Num layers:         {config.get('num_layers')}")
    print(f"  Predictor rank:     {config.get('predictor_rank')}")
    print(f"  Calibration samples:{config.get('num_calibration_samples')}")
    print(f"  Prepared at:        {config.get('prepared_at')}")

    fw_files = list(flash_dir.glob("*.flashweights"))
    print(f"  Weight files:       {len(fw_files)}")

    pred_dir = flash_dir / "predictors"
    if pred_dir.exists():
        pred_files = list(pred_dir.glob("*.npz"))
        print(f"  Predictor files:    {len(pred_files)}")

    total_bytes = sum(f.stat().st_size for f in flash_dir.rglob("*") if f.is_file())
    print(f"  Total size:         {total_bytes / (1024**2):.1f} MB")

    print("\nTo use flash inference:")
    print("  olmlx serve --flash")
    print("  # or set OLMLX_FLASH=true")


def build_parser() -> argparse.ArgumentParser:
    # Each subparser group MUST use ``{cmd}_command`` as its ``dest``
    # so that ``cli_main()`` can look up the subcommand via
    # ``getattr(args, f"{cmd}_command", None)``.
    parser = argparse.ArgumentParser(
        prog="olmlx",
        description="Ollama-compatible API server using Apple MLX",
    )
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Start the server (default)")
    serve_p.add_argument(
        "--speculative",
        dest="speculative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable speculative decoding (overrides OLMLX_SPECULATIVE)",
    )
    serve_p.add_argument(
        "--speculative-strategy",
        dest="speculative_strategy",
        choices=("classic", "dflash", "eagle", "pld", "self_speculative"),
        default=None,
        help=(
            "Speculative decoding strategy: 'classic' (standalone draft LM), "
            "'dflash' (block-diffusion draft conditioned on target hidden "
            "states), 'eagle' (autoregressive draft head conditioned on "
            "target last-layer hidden, arxiv 2401.15077), 'pld' "
            "(prompt-lookup decoding — n-gram lookup in the prompt+generated "
            "history, no draft model required), or 'self_speculative' "
            "(LayerSkip-style — uses target's own early layers as draft; "
            "no external draft model required). Default: classic."
        ),
    )
    serve_p.add_argument(
        "--speculative-draft-model",
        dest="speculative_draft_model",
        type=_non_empty_str,
        default=None,
        help="HuggingFace path of the draft model used for speculative decoding",
    )
    serve_p.add_argument(
        "--speculative-tokens",
        dest="speculative_tokens",
        type=_positive_int,
        default=None,
        help=(
            "Number of tokens drafted per verification step (default: 4 "
            "for classic, 10 for PLD). For DFlash this is the block size "
            "(excluding the pending token); for PLD it's the max draft "
            "length (actual draft is bounded by the longest n-gram match)."
        ),
    )
    serve_p.add_argument(
        "--speculative-layers-skip",
        dest="speculative_layers_skip",
        type=_positive_int,
        default=None,
        help=(
            "Number of layers skipped during self_speculative draft "
            "(default: L//4, where L is the total number of layers). "
            "Only applies to strategy='self_speculative'."
        ),
    )
    serve_p.add_argument(
        "--kv-cache-quant",
        dest="kv_cache_quant",
        type=str,
        default=None,
        help="KV cache quantization method and bits (e.g. turboquant:4, spectral:2)",
    )
    serve_p.add_argument(
        "--flash",
        dest="flash",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable Flash inference (LLM in a Flash; sparse FFN with SSD-"
            "backed neuron loading). Overrides OLMLX_FLASH. Requires the "
            "model to be prepared first via 'olmlx flash prepare'."
        ),
    )
    serve_p.add_argument(
        "--flash-prefetch",
        dest="flash_prefetch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Flash speculative neuron prefetch (overrides OLMLX_FLASH_PREFETCH).",
    )
    serve_p.add_argument(
        "--flash-speculative",
        dest="flash_speculative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Flash + speculative decoding (overrides OLMLX_FLASH_SPECULATIVE).",
    )
    serve_p.add_argument(
        "--flash-speculative-draft-model",
        dest="flash_speculative_draft_model",
        type=_non_empty_str,
        default=None,
        help="HuggingFace path of the draft model used for Flash speculative decoding.",
    )
    serve_p.add_argument(
        "--flash-speculative-tokens",
        dest="flash_speculative_tokens",
        type=_positive_int,
        default=None,
        help="Tokens drafted per verification step for Flash speculative (default: 4).",
    )

    svc = sub.add_parser("service", help="Manage the launchd service")
    svc_sub = svc.add_subparsers(dest="service_command")
    svc_sub.add_parser("install", help="Install and start the launchd service")
    svc_sub.add_parser("uninstall", help="Stop and remove the launchd service")
    svc_sub.add_parser("status", help="Show service status")

    mdl = sub.add_parser("models", help="Manage local models")
    mdl_sub = mdl.add_subparsers(dest="models_command")
    mdl_sub.add_parser("list", help="List locally downloaded models")
    pull_p = mdl_sub.add_parser("pull", help="Pull/download a model")
    pull_p.add_argument("model_name", help="Model name or HF path")
    show_p = mdl_sub.add_parser("show", help="Show model details")
    show_p.add_argument("model_name", help="Model name or HF path")
    del_p = mdl_sub.add_parser("delete", help="Delete a local model")
    del_p.add_argument("model_name", help="Model name or HF path")
    del_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    search_p = mdl_sub.add_parser("search", help="Search for models by name")
    search_p.add_argument("query", help="Search query")

    chat_p = sub.add_parser("chat", help="Interactive chat")
    chat_p.add_argument("model_name", nargs="?", help="Model name or HF path")
    chat_p.add_argument("--system", "-s", help="System prompt")
    chat_p.add_argument(
        "--mcp-config", help="MCP config path (default: ~/.olmlx/mcp.json)"
    )
    chat_p.add_argument(
        "--no-mcp", action="store_true", default=False, help="Disable MCP"
    )
    chat_p.add_argument(
        "--no-thinking", action="store_true", default=False, help="Disable thinking"
    )
    chat_p.add_argument("--max-tokens", type=int, default=4096)
    chat_p.add_argument("--max-turns", type=int, default=25)
    chat_p.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (1.0 = disabled, default: 1.1)",
    )
    chat_p.add_argument(
        "--repeat-last-n",
        type=int,
        default=64,
        help="Context window for repetition penalty (default: 64)",
    )
    chat_p.add_argument(
        "--no-skills", action="store_true", default=False, help="Disable skills"
    )
    chat_p.add_argument(
        "--no-builtin-tools",
        action="store_true",
        default=False,
        help="Disable built-in tools",
    )
    chat_p.add_argument(
        "--skills-dir", help="Skills directory (default: ~/.olmlx/skills)"
    )
    chat_p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: model default)",
    )
    chat_p.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling (default: model default)",
    )
    chat_p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (default: model default)",
    )
    chat_p.add_argument(
        "--tool-timeout",
        type=float,
        default=None,
        help=(
            "Timeout in seconds for tool calls, MCP and builtin "
            "(default: per-tool defaults — MCP 30, bash 120)"
        ),
    )
    chat_p.add_argument(
        "--mcp-connect-retries",
        type=int,
        default=None,
        help="MCP server connection retry attempts (default: 3)",
    )
    chat_p.add_argument(
        "--local-tool-safety",
        action="store_true",
        default=False,
        help="Apply tool safety policy to local tools (builtins, skills)",
    )
    chat_p.add_argument(
        "--tool-result-truncation",
        type=int,
        default=None,
        help="Max chars for tool result display (default: 2000)",
    )
    chat_p.add_argument(
        "--max-consecutive-tool-failures",
        type=int,
        default=None,
        help="Max consecutive tool failure turns before stopping (default: 3, 0=unlimited)",
    )
    chat_p.add_argument(
        "--voice",
        action="store_true",
        help="Enable push-to-talk STT input and Kokoro TTS output (issue #444).",
    )
    chat_p.add_argument(
        "--stt-model", default=None, help="Override STT (Whisper) model."
    )
    chat_p.add_argument(
        "--tts-model", default=None, help="Override TTS (Kokoro) model."
    )
    chat_p.add_argument(
        "--voice-name", default=None, help="Kokoro voice (e.g. af_heart)."
    )

    # Flash inference
    flash = sub.add_parser("flash", help="Flash inference (LLM in a Flash)")
    flash_sub = flash.add_subparsers(dest="flash_command")

    prepare_p = flash_sub.add_parser(
        "prepare", help="Prepare a model for flash inference"
    )
    prepare_p.add_argument("model", help="Model name or HF path")
    prepare_p.add_argument(
        "--rank", type=int, default=128, help="Predictor rank (default: 128)"
    )
    prepare_p.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256)",
    )
    prepare_p.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Activation threshold (default: 0.01)",
    )
    prepare_p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Predictor training epochs (default: 5)",
    )
    prepare_p.add_argument(
        "--calibration-dataset",
        type=str,
        default=None,
        help="Calibration dataset: 'c4' (default) or 'synthetic'",
    )
    prepare_p.add_argument(
        "--sensitive-layers",
        type=int,
        default=0,
        help="Number of last layers to use higher predictor rank (default: 0, disabled)",
    )
    prepare_p.add_argument(
        "--sensitive-rank-multiplier",
        type=int,
        default=4,
        help="Rank multiplier for sensitive layers (default: 4)",
    )
    info_p = flash_sub.add_parser(
        "info", help="Show flash preparation info for a model"
    )
    info_p.add_argument("model", help="Model name or HF path")

    # DFlash draft training
    dflash = sub.add_parser("dflash", help="DFlash block-diffusion draft training")
    dflash_sub = dflash.add_subparsers(dest="dflash_command")
    dflash_prepare_p = dflash_sub.add_parser(
        "prepare", help="Train a DFlash draft model for a target"
    )
    dflash_prepare_p.add_argument("model", help="Target model name or HF path")
    dflash_prepare_p.add_argument(
        "--data",
        type=str,
        default=None,
        help="HuggingFace dataset path (default: HuggingFaceH4/ultrachat_200k)",
    )
    dflash_prepare_p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (default: train_sft)",
    )
    dflash_prepare_p.add_argument(
        "--steps", type=int, default=2000, help="Training steps (default: 2000)"
    )
    dflash_prepare_p.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: 4)"
    )
    dflash_prepare_p.add_argument(
        "--seq-len", type=int, default=2048, help="Sequence length (default: 2048)"
    )
    dflash_prepare_p.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Number of draft tokens per step (default: 16, per paper)",
    )
    dflash_prepare_p.add_argument(
        "--num-hidden-layers",
        type=int,
        default=5,
        help="Draft model hidden layer count (default: 5, per paper)",
    )
    dflash_prepare_p.add_argument(
        "--num-target-layers",
        type=int,
        default=5,
        help=(
            "Number of target hidden states to extract (default: 5, "
            "per paper). Ignored when --target-layer-ids is set."
        ),
    )
    dflash_prepare_p.add_argument(
        "--target-layer-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated target layer indices to extract hidden states "
            "from (e.g. '5,11,17,23'). Defaults to evenly spaced layers."
        ),
    )
    dflash_prepare_p.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help=(
            "Token id used as MASK in the block-diffusion draft input. "
            "Defaults to the tokenizer's pad_token_id (or eos_token_id if "
            "no pad). For tokenizers with neither, this flag is required — "
            "token 0 is not a safe fallback (often <bos>/<unk>)."
        ),
    )
    dflash_prepare_p.add_argument(
        "--lr", type=float, default=5e-4, help="Peak learning rate (default: 5e-4)"
    )
    dflash_prepare_p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <target-model-dir>/dflash)",
    )
    dflash_prepare_p.add_argument(
        "--distill",
        action="store_true",
        default=False,
        help=(
            "Enable Hinton-style KL distillation against the target's "
            "logits at the masked positions. Incompatible with "
            "--use-precomputed (precomputed shards do not store logits). "
            "Memory note: peak usage ~2x the CE-only path because both "
            "target and draft probability tensors of shape "
            "(batch_size, block_size, vocab_size) are live during the "
            "KL reduction. For high-vocab targets (e.g. Gemma, Command R "
            "at ~256k vocab) lower --batch-size accordingly."
        ),
    )
    dflash_prepare_p.add_argument(
        "--distill-alpha",
        type=float,
        default=0.5,
        help=(
            "Distillation mixing weight: loss = (1-alpha)*CE + "
            "alpha*T^2*KL. Default 0.5; ignored when --distill is unset."
        ),
    )
    dflash_prepare_p.add_argument(
        "--distill-temp",
        type=float,
        default=2.0,
        help="Distillation temperature T (default: 2.0)",
    )
    dflash_prepare_p.add_argument(
        "--position-decay-gamma",
        type=float,
        default=None,
        help=(
            "Per-position loss weight decay: w_k = exp(-(k-1)/gamma) for "
            "k=1..block_size. Emphasises early positions because "
            "acceptance length compounds. Pass 0 (or negative) to "
            "disable and use the uniform-mean reduction (the default "
            "when this flag is omitted). Suggested starting value: "
            "block_size/2."
        ),
    )
    dflash_prepare_p.add_argument(
        "--train-windows-per-step",
        type=int,
        default=1,
        help=(
            "Number of non-overlapping masked windows to train on per "
            "batch (per optimizer step). Default 1 reproduces the "
            "legacy single-window behaviour bit-for-bit. K > 1 "
            "amortises the target forward across K draft-loss windows "
            "in a single optimizer step; the optimizer-step budget "
            "(--steps) is unchanged but each step sees K times more "
            "training signal. When the batch's shared unpadded prefix "
            "is too short for K non-overlapping windows, fewer are "
            "used (K is a target, not a guarantee). See gh#382."
        ),
    )
    dflash_prepare_p.add_argument(
        "--use-precomputed",
        type=str,
        default=None,
        help=(
            "Read (input_ids, target_hidden) shards from this directory "
            "instead of running the target each step. Produced by "
            "`olmlx dflash precompute`."
        ),
    )

    dflash_precompute_p = dflash_sub.add_parser(
        "precompute",
        help="Precompute target hidden states for DFlash draft training",
    )
    dflash_precompute_p.add_argument("model", help="Target model name or HF path")
    dflash_precompute_p.add_argument(
        "--data", type=str, default=None, help="HuggingFace dataset path"
    )
    dflash_precompute_p.add_argument(
        "--split", type=str, default=None, help="Dataset split"
    )
    dflash_precompute_p.add_argument(
        "--shards",
        type=int,
        default=500,
        help="Number of shards to write (default: 500)",
    )
    dflash_precompute_p.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: 4)"
    )
    dflash_precompute_p.add_argument(
        "--seq-len", type=int, default=2048, help="Sequence length (default: 2048)"
    )
    dflash_precompute_p.add_argument(
        "--num-target-layers",
        type=int,
        default=5,
        help="Number of target hidden states to extract (default: 5, per paper)",
    )
    dflash_precompute_p.add_argument(
        "--target-layer-ids",
        type=str,
        default=None,
        help="Comma-separated target layer indices (overrides --num-target-layers)",
    )
    dflash_precompute_p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <target-model-dir>/dflash_cache)",
    )

    # EAGLE draft training (arxiv 2401.15077)
    eagle = sub.add_parser(
        "eagle", help="EAGLE autoregressive speculative draft training"
    )
    eagle_sub = eagle.add_subparsers(dest="eagle_command")
    eagle_prepare_p = eagle_sub.add_parser(
        "prepare", help="Train an EAGLE draft model for a target"
    )
    eagle_prepare_p.add_argument("model", help="Target model name or HF path")
    eagle_prepare_p.add_argument(
        "--use-precomputed",
        type=str,
        required=True,
        help=(
            "Directory of (input_ids, target_hidden) shards produced by "
            "`olmlx dflash precompute`. EAGLE consumes the deepest captured "
            "layer; the same shards work for both DFlash and EAGLE."
        ),
    )
    eagle_prepare_p.add_argument(
        "--steps", type=int, default=2000, help="Training steps (default: 2000)"
    )
    eagle_prepare_p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Batch size (default: 4). Under --use-precomputed this is "
            "validated against the shard layout (shards are written at a "
            "fixed batch shape; pass the value that matches or rerun "
            "`olmlx dflash precompute` at the desired batch size)."
        ),
    )
    eagle_prepare_p.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help=(
            "Sequence length (default: 2048). Under --use-precomputed "
            "this is validated against the shard layout (shards are "
            "written at a fixed sequence length; pass the value that "
            "matches or rerun precompute at the desired length)."
        ),
    )
    eagle_prepare_p.add_argument(
        "--block-size",
        type=int,
        default=4,
        help=(
            "Number of draft tokens per verify (default: 4). Note: each "
            "drafted token forces one Metal command-buffer flush via "
            "``.item()`` (autoregressive feature-space drafting needs the "
            "integer token id for the next iteration). On Apple Silicon "
            "that's ~0.5–1 ms per flush, so block_size=4 adds ~2–4 ms of "
            "sync overhead per verify before the target's parallel forward. "
            "If real-bench acceptance is low for your draft, smaller "
            "block_size (1 or 2) may be Pareto-optimal — it halves/quarters "
            "the sync overhead and avoids deeper compounding-error positions."
        ),
    )
    eagle_prepare_p.add_argument(
        "--num-hidden-layers",
        type=int,
        default=1,
        help="Draft model decoder layer count (default: 1, EAGLE-1 default)",
    )
    eagle_prepare_p.add_argument(
        "--lr", type=float, default=5e-4, help="Peak learning rate (default: 5e-4)"
    )
    eagle_prepare_p.add_argument(
        "--sample-positions",
        type=int,
        default=256,
        help=(
            "Per-step subsample of positions where lm_head is applied "
            "during loss computation. The full sequence still runs through "
            "draft self-attention; only the final vocab projection is "
            "subsampled. Set to 0 to disable subsampling and score every "
            "position (~10x slower on large vocabs). Default: 256."
        ),
    )
    eagle_prepare_p.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "PRNG seed for reproducible training. Seeds both mlx (weight "
            "init etc.) and stdlib random (the per-step position subsample "
            "used by sample-positions). Default: 0."
        ),
    )
    eagle_prepare_p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <target-model-dir>/eagle)",
    )

    # Spectral quant calibration
    spectral = sub.add_parser("spectral", help="SpectralQuant KV cache compression")
    spectral_sub = spectral.add_subparsers(dest="spectral_command")

    spectral_prepare_p = spectral_sub.add_parser(
        "prepare", help="Run spectral calibration for a model"
    )
    spectral_prepare_p.add_argument("model", help="Model name or HF path")
    spectral_prepare_p.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256)",
    )
    spectral_prepare_p.add_argument(
        "--avg-bits",
        type=int,
        default=4,
        choices=[2, 4],
        help="Target average bits per dimension (default: 4)",
    )
    spectral_prepare_p.add_argument(
        "--calibration-dataset",
        type=str,
        default=None,
        help="Calibration dataset: 'c4' (default) or 'synthetic'",
    )
    spectral_prepare_p.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens to collect per head (default: 8192)",
    )

    # Bench (benchmarking)
    bench = sub.add_parser("bench", help="Benchmarking and functional tests")
    bench_sub = bench.add_subparsers(dest="bench_command")

    bench_run = bench_sub.add_parser("run", help="Run benchmark scenarios")
    bench_run.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model name or HF path (default: mlx-community/Qwen2.5-0.5B-Instruct-4bit)",
    )
    bench_run.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max tokens for all prompts",
    )
    bench_run.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated scenario names (default: all)",
    )
    bench_run.add_argument(
        "--prompt-set",
        choices=["throughput", "quality", "all"],
        default="throughput",
        help=(
            "Which prompts to run: 'throughput' (7 tok/s probes, default), "
            "'quality' (GSM8K+MMLU+HumanEval graded sets), or 'all'"
        ),
    )
    bench_run.add_argument(
        "--enable-code-exec",
        action="store_true",
        help=(
            "Run model-generated code for the HumanEval code_exec grader in a "
            "resource-limited subprocess (off by default; only the local user's "
            "chosen model produces this code)"
        ),
    )
    bench_run.add_argument(
        "--bench-dir",
        "--output-dir",
        dest="bench_dir",
        type=str,
        default=None,
        help="Directory to save the run in (default: ~/.olmlx/bench/runs)",
    )

    bench_compare = bench_sub.add_parser("compare", help="Compare two benchmark runs")
    bench_compare.add_argument("run1", help="First run (timestamp or path)")
    bench_compare.add_argument("run2", help="Second run (timestamp or path)")

    bench_list = bench_sub.add_parser("list", help="List past benchmark runs")
    bench_list.add_argument(
        "--bench-dir",
        type=str,
        default=None,
        help="Directory to read runs from (default: ~/.olmlx/bench/runs)",
    )

    bench_lb = bench_sub.add_parser(
        "leaderboard", help="Show model leaderboard derived from past runs"
    )
    bench_lb.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Show every run instead of latest per model. Use this if a "
            "recent regression run has displaced an earlier faster result."
        ),
    )
    bench_lb.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Limit rows (default: all); must be >= 1",
    )
    bench_lb.add_argument(
        "--bench-dir",
        type=str,
        default=None,
        help="Directory to read runs from (default: ~/.olmlx/bench/runs)",
    )

    cfg = sub.add_parser("config", help="Show configuration")
    cfg_sub = cfg.add_subparsers(dest="config_command")
    cfg_sub.add_parser("show", help="Show current configuration")

    return parser


# Registry: (command, subcommand) → handler name.
# Handler names are resolved via globals() at call time so test
# monkeypatching (``monkeypatch.setattr("olmlx.cli.cmd_serve", mock)``)
# works.  _validate_command_handlers() catches typos at import time.
# Subcommand=None for commands that take no subcommand (serve, chat).
#
# IMPORTANT: every cmd_* handler referenced below must be defined
# above this point in the file (import-time globals() resolution).
_COMMAND_HANDLERS: dict[tuple[str, str | None], str] = {
    ("serve", None): "cmd_serve",
    ("chat", None): "cmd_chat",
    ("service", "install"): "cmd_service_install",
    ("service", "uninstall"): "cmd_service_uninstall",
    ("service", "status"): "cmd_service_status",
    ("models", "list"): "cmd_models_list",
    ("models", "pull"): "cmd_models_pull",
    ("models", "show"): "cmd_models_show",
    ("models", "delete"): "cmd_models_delete",
    ("models", "search"): "cmd_models_search",
    ("flash", "prepare"): "cmd_flash_prepare",
    ("flash", "info"): "cmd_flash_info",
    ("dflash", "prepare"): "cmd_dflash_prepare",
    ("dflash", "precompute"): "cmd_dflash_precompute",
    ("eagle", "prepare"): "cmd_eagle_prepare",
    ("spectral", "prepare"): "cmd_spectral_prepare",
    ("bench", "run"): "cmd_bench_run",
    ("bench", "compare"): "cmd_bench_compare",
    ("bench", "list"): "cmd_bench_list",
    ("bench", "leaderboard"): "cmd_bench_leaderboard",
    ("config", "show"): "cmd_config_show",
}


def _validate_command_handlers() -> None:
    """Verify every handler name in the registry refers to a callable.

    Called at module load to catch typos before runtime dispatch.
    """
    for (cmd, sub), name in _COMMAND_HANDLERS.items():
        handler = globals().get(name)
        if handler is None:
            raise NameError(
                f"Handler {name!r} (registered for ({cmd!r}, {sub!r})) "
                f"not found in module globals"
            )
        if not callable(handler):
            raise TypeError(f"Handler {name!r} for ({cmd!r}, {sub!r}) is not callable")


_validate_command_handlers()


def _resolve_handler(cmd: str, sub_name: str | None) -> Callable[..., Any] | None:
    """Look up a handler by (command, subcommand) in the registry.

    Resolves via ``globals()`` so that test monkeypatching
    (``monkeypatch.setattr("olmlx.cli.cmd_serve", mock_fn)``) works.
    """
    name = _COMMAND_HANDLERS.get((cmd, sub_name))
    if name is None:
        return None
    handler = globals().get(name)
    if handler is None:
        raise NameError(
            f"Handler {name!r} (registered for ({cmd!r}, {sub_name!r})) "
            f"not found in module globals"
        )
    return handler


def cli_main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        # Bare invocation: derive serve-subparser defaults from the
        # parser itself rather than hardcoding the flag list. New serve
        # flags wire through automatically; any top-level flags already
        # on ``args`` win because ``hasattr`` short-circuits the copy.
        # Invariant: top-level parser flag names must not overlap with
        # serve-only flag names — otherwise this loop would suppress the
        # serve default for the colliding name. The root parser only
        # declares the ``command`` dest today, so this holds.
        serve_defaults = vars(parser.parse_args(["serve"]))
        for _name, _default in serve_defaults.items():
            if not hasattr(args, _name):
                setattr(args, _name, _default)

    cmd = args.command or "serve"  # default
    sub_name = getattr(args, f"{cmd}_command", None)

    handler = _resolve_handler(cmd, sub_name)
    if handler:
        # ``ModelsConfigError`` surfaces from ``registry.load()`` /
        # ``_save_mappings_locked()`` when ``models.json`` is unreadable
        # or corrupt — refusing to clobber the file. Convert to a clean
        # exit with the error text instead of dumping a traceback.
        from olmlx.engine.registry import ModelsConfigError

        try:
            return handler(args)
        except ModelsConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    # Unknown subcommand — print help for the parent command
    parser.parse_args([cmd, "--help"])
