"""The serve subcommand: overrides, legacy env surfacing, config audits."""

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmlx.engine.registry import ModelRegistry

from olmlx.config import (
    settings,
    warn_legacy_flash_env as _warn_legacy_flash_env,
)

from olmlx.cli.config_cmd import _configure_logging, ensure_config
from olmlx.cli.distributed_launch import (
    _cleanup_workers,
    _launch_distributed_workers,
)

logger = logging.getLogger(__name__)


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
    file. Delegates to the single canonical ``.env`` parser (#635).
    """
    from olmlx.config import parse_dotenv_values

    return parse_dotenv_values(_DEPRECATED_SPECULATIVE_ENV_VARS)


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
    project ``.env`` file, or None if not present or the file is unreadable.

    Delegates to the single canonical ``.env`` parser (#635)."""
    from olmlx.config import parse_dotenv_values

    return parse_dotenv_values(("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT",)).get(
        "OLMLX_EXPERIMENTAL_KV_CACHE_QUANT"
    )


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
    _warn_legacy_flash_env()

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
        # Strategies that legitimately run without a ``speculative_draft_model``:
        # ``self_speculative`` (target's own early layers), ``pld`` (n-gram
        # prompt lookup), ``lookahead`` (draft-free Jacobi iteration), and
        # ``proxy_tuning`` (steers via expert/anti-expert models configured
        # under their own ``speculative_proxy_*`` knobs).
        _draftless = {"self_speculative", "pld", "lookahead", "proxy_tuning"}
        if enabled and not draft and strategy not in _draftless:
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


def cmd_config_show(_args):
    """Show current configuration."""
    _surface_legacy_kv_cache_quant_env()
    _surface_legacy_distributed_env()
    _warn_legacy_flash_env()

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
