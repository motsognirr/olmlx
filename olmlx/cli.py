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
from typing import Any

from olmlx.config import settings

logger = logging.getLogger(__name__)

PLIST_LABEL = "com.dpalmqvist.olmlx"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_LABEL}.plist"

DEFAULT_MODELS = {
    "llama3.2:latest": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistral:7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "qwen2.5:3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "gemma2:2b": "mlx-community/gemma-2-2b-it-4bit",
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

    from olmlx.config import experimental

    if experimental.distributed:
        _hosts, strategy, hostfile_layers = _launch_distributed_workers()
        # The ring backend's init() blocks until all ranks connect. Both the
        # coordinator and workers must call init() within each other's retry
        # window (~31s). Workers start ~5-10s after SSH launch. We delay
        # the coordinator by 3s to overlap with the worker's init() window.
        print("  Waiting 3s for workers to start...")
        time.sleep(3)
        import mlx.core as mx

        try:
            group = mx.distributed.init(backend=experimental.distributed_backend)
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
            port=experimental.distributed_sideband_port,
            secret=experimental.distributed_secret or None,
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


def _forward_legacy_speculative_env(_settings) -> None:
    """Apply legacy env var values to the new Settings when the new env
    var is unset. Logs and swallows parse errors per-field so a single
    bad legacy value never blocks startup.

    "Unset" is determined by comparing the live Settings value against
    the field default — checking ``os.environ`` alone would miss values
    pydantic-settings already loaded from a ``.env`` file and silently
    let the legacy shell var clobber them.
    """
    from olmlx.config import Settings

    for legacy, new, attr, parse in _LEGACY_SPECULATIVE_FORWARD:
        legacy_val = os.environ.get(legacy)
        if legacy_val is None:
            continue
        field_default = Settings.model_fields[attr].default
        if getattr(_settings, attr) != field_default:
            # The new value already came from somewhere — env, .env,
            # CLI flag, or programmatic write. Don't override it with
            # the legacy value.
            continue
        try:
            value = parse(legacy_val)
            setattr(_settings, attr, value)
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


def _apply_serve_overrides(args) -> None:
    """Apply CLI flags to the global Settings before the server starts.

    The flags are written to the ``settings`` instance so that the rest of
    the codebase (which reads ``from olmlx.config import settings``) picks
    them up without needing extra plumbing.
    """
    from olmlx.config import settings as _settings

    # Surface the env-var rename so a user upgrading with the old names
    # in their shell profile doesn't silently lose speculative decoding —
    # pydantic-settings drops unknown OLMLX_EXPERIMENTAL_* keys without
    # warning.
    stale = [v for v in _DEPRECATED_SPECULATIVE_ENV_VARS if os.environ.get(v)]
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
        _forward_legacy_speculative_env(_settings)

    if args.speculative is not None:
        _settings.speculative = args.speculative
    if args.speculative_draft_model is not None:
        _settings.speculative_draft_model = args.speculative_draft_model
    if args.speculative_tokens is not None:
        _settings.speculative_tokens = args.speculative_tokens

    # Surface speculative misconfigurations at startup by walking the
    # registry and checking each model's ``resolved_speculative()``. This
    # is precise: it accepts ``OLMLX_SPECULATIVE=true`` with no global
    # draft as long as every registered model supplies its own. It also
    # catches per-model entries that enable speculative without a draft.
    # The "global speculative=true and zero registered models" case is
    # not flagged here — the first model load will raise a clear error.
    needs_migration = _models_with_promoted_keys_in_experimental()
    if needs_migration:
        print(
            "Error: the following models in models.json still place "
            "speculative settings under 'experimental' — these keys have "
            "been promoted to top-level fields. Move 'speculative', "
            "'speculative_draft_model', and 'speculative_tokens' out of "
            f"the 'experimental' block. Affected entries: {', '.join(needs_migration)}.",
            file=sys.stderr,
        )
        sys.exit(2)

    bad, dormant_drafts, flash_conflicts, global_draft_used = (
        _audit_speculative_config()
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
    if flash_conflicts:
        # Note: standalone speculative is only dropped when the Flash
        # bundle actually loads — if the bundle directory is missing
        # ``_load_model`` falls through to the standard load path and
        # the standalone speculative decoder still runs. The startup
        # warning is advisory; the authoritative runtime warning lives
        # in ``_load_model`` for the case where Flash actually wins.
        logger.warning(
            "The following models combine speculative=true with Flash or "
            "Flash-MoE: %s. Once Flash is prepared and loads, standalone "
            "speculative decoding is dropped — use the model's "
            "flash_speculative settings "
            "(OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_*; flash-speculative "
            "remains experimental) instead.",
            ", ".join(flash_conflicts),
        )
    if bad:
        print(
            "Error: the following models in models.json enable speculative "
            "decoding but have no draft model configured (per-model or "
            f"global): {', '.join(bad)}. Set 'speculative_draft_model' on "
            "each entry or set OLMLX_SPECULATIVE_DRAFT_MODEL globally.",
            file=sys.stderr,
        )
        sys.exit(2)


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


def _audit_speculative_config() -> tuple[list[str], list[str], list[str], bool]:
    """Walk the registry and audit each model's resolved speculative
    config.

    Returns ``(bad, dormant_drafts, flash_conflicts, global_draft_used)``:
    - ``bad`` — models with ``speculative=True`` but no draft model
      anywhere. Triggers a startup error.
    - ``dormant_drafts`` — models with a per-model ``speculative_draft_model``
      set but the resolved ``enabled`` flag is False. Triggers a
      warning so users don't silently lose the draft they configured.
    - ``flash_conflicts`` — models that combine ``speculative=True``
      with Flash or Flash-MoE in the same entry. Standalone speculative
      decoding is silently dropped on the Flash load path; the model's
      own ``flash_speculative`` knob is the right one. Triggers a
      warning so users see the redirect.
    - ``global_draft_used`` — True if at least one model resolves to
      the global ``speculative_draft_model`` (i.e. has ``speculative=True``
      and no per-model draft override). Used to suppress the global
      dormant-draft warning when the global draft actually has consumers.

    The registry is loaded from disk; failures are logged and treated
    as "nothing to validate" so this never blocks startup on its own.
    """
    try:
        from olmlx.config import experimental as global_exp
        from olmlx.config import resolve_experimental
        from olmlx.engine.registry import ModelRegistry

        registry = ModelRegistry()
        registry.load()
    except ValueError as exc:
        # Validation errors (e.g. a malformed entry that survived
        # ``_models_with_promoted_keys_in_experimental``) are operator
        # errors, not transient I/O issues — flag them distinctly.
        logger.warning(
            "Skipping speculative config validation: invalid models.json entry: %s",
            exc,
        )
        return [], [], [], False
    except Exception as exc:
        logger.warning(
            "Skipping speculative config validation: could not load registry: %s",
            exc,
        )
        return [], [], [], False
    bad: list[str] = []
    dormant: list[str] = []
    flash_conflicts: list[str] = []
    global_draft_used = False
    for name, mc in registry.list_models().items():
        enabled, draft, _ = mc.resolved_speculative()
        if enabled and not draft:
            bad.append(name)
        elif not enabled and mc.speculative_draft_model:
            # Use the raw per-model field rather than the resolved
            # ``draft``: the global dormant-draft case is already
            # surfaced separately in ``_apply_serve_overrides``.
            dormant.append(name)
        if enabled and mc.speculative_draft_model is None and draft is not None:
            # This model enables speculative without a per-model draft,
            # so it is consuming the global ``speculative_draft_model``.
            global_draft_used = True
        if enabled:
            # Resolve the full experimental config (global defaults
            # merged with per-model overrides) so a globally enabled
            # OLMLX_EXPERIMENTAL_FLASH still trips the conflict check.
            try:
                resolved_exp = resolve_experimental(global_exp, mc.experimental)
            except Exception as exc:
                logger.warning(
                    "Skipping flash-conflict check for %s: could not "
                    "resolve experimental overrides: %s",
                    name,
                    exc,
                )
                continue
            if resolved_exp.flash or resolved_exp.flash_moe:
                flash_conflicts.append(name)
    return bad, dormant, flash_conflicts, global_draft_used


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
    hosts, model, world_size, experimental, strategy="tensor", layer_counts=None
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
    shard_base = Path(experimental.distributed_shard_dir).expanduser() / safe_name

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
    worker_shard_dir = str(Path(experimental.distributed_worker_shard_dir).expanduser())
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

    from olmlx.config import experimental

    hostfile_path = Path(experimental.distributed_hostfile).expanduser()
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
    if strategy not in ("tensor", "pipeline"):
        print(
            f"Error: hostfile strategy must be 'tensor' or 'pipeline', got {strategy!r}",
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
        [f"{h}:{experimental.distributed_port + i}"] for i, h in enumerate(hosts)
    ]
    max_port = experimental.distributed_port + len(hosts) - 1
    if max_port > 65535:
        print(
            f"Error: distributed_port {experimental.distributed_port} + "
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

    remote_python = experimental.distributed_remote_python
    validate_remote_python(remote_python)
    remote_working_dir = experimental.distributed_remote_working_dir

    if experimental.flash_moe:
        print(
            "Error: Flash-MoE + distributed is not supported. "
            "Disable OLMLX_EXPERIMENTAL_FLASH_MOE or OLMLX_EXPERIMENTAL_DISTRIBUTED.",
            file=sys.stderr,
        )
        sys.exit(1)

    if experimental.flash and strategy == "pipeline":
        print(
            "Error: Flash + pipeline distributed strategy is not supported. "
            "Use tensor strategy or disable Flash.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Pre-shard and distribute weights to workers if enabled
    pre_sharded = False
    if experimental.distributed_pre_shard:
        if experimental.flash:
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
                experimental,
                strategy=strategy,
                layer_counts=hostfile_layers,
            )

    # Pre-compute safe model name for env var paths (used when pre-sharded)
    from olmlx.config import PRE_SHARDED_DIR_ENV
    from olmlx.models.store import _safe_dir_name

    safe_name = _safe_dir_name(model) if pre_sharded else ""
    # Keep ~ as-is: the worker calls expanduser() on the received path
    worker_shard_dir = experimental.distributed_worker_shard_dir if pre_sharded else ""

    # Launch workers on remote hosts (rank 1..N)
    for rank, host in enumerate(hosts[1:], start=1):
        env = {
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_MODEL": model,
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_BACKEND": experimental.distributed_backend,
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_COORDINATOR_HOST": coordinator_host,
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT": str(
                experimental.distributed_sideband_port
            ),
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_STRATEGY": strategy,
            "MLX_RANK": str(rank),
        }
        if hostfile_layers is not None:
            env["OLMLX_EXPERIMENTAL_DISTRIBUTED_LAYER_COUNTS"] = ",".join(
                str(x) for x in hostfile_layers
            )
        if pre_sharded:
            env[PRE_SHARDED_DIR_ENV] = f"{worker_shard_dir}/{safe_name}/rank{rank}"
        if experimental.flash:
            env["OLMLX_EXPERIMENTAL_FLASH"] = "true"
            # Forward all flash tuning params so worker FlashConfig matches.
            # OLMLX_EXPERIMENTAL_FLASH_MOE also matches this prefix but is
            # safe: the flash_moe guard above already exited if it was true.
            for key, val in os.environ.items():
                if key.startswith("OLMLX_EXPERIMENTAL_FLASH_") and key not in env:
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

        if experimental.distributed_secret:
            script_parts.extend(
                [
                    "SECRET_FILE=$(mktemp)",
                    f"printf '%s' {shlex.quote(experimental.distributed_secret)} > $SECRET_FILE",
                    "chmod 600 $SECRET_FILE",
                    "export OLMLX_EXPERIMENTAL_DISTRIBUTED_SECRET_FILE=$SECRET_FILE",
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


def _build_plist() -> dict:
    """Build a launchd plist dict for the olmlx service."""
    exe = _find_executable()
    if exe == sys.executable:
        program_args = [exe, "-m", "olmlx"]
    else:
        program_args = [exe]

    env_vars = {}
    # Forward OLMLX_ env vars if set
    for key, value in os.environ.items():
        if key.startswith("OLMLX_"):
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

            while True:
                user_input = tui.get_user_input()
                if user_input is None:
                    break

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
    from olmlx.config import experimental

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
    if experimental.distributed:
        print()
        print("Experimental distributed inference:")
        print(f"  Hostfile:             {experimental.distributed_hostfile}")
        print(f"  Backend:              {experimental.distributed_backend}")
        print(f"  Port:                 {experimental.distributed_port}")
        print(f"  Sideband port:        {experimental.distributed_sideband_port}")


def cmd_bench_run(args):
    """Run benchmark scenarios."""
    _configure_logging()
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
    print(f"  OLMLX_EXPERIMENTAL_KV_CACHE_QUANT=spectral:{args.avg_bits} olmlx serve")


def cmd_flash_prepare(args):
    """Prepare a model for flash inference (auto-detects MoE vs dense)."""
    _configure_logging()

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

    from olmlx.config import experimental

    output_dir = prepare_model_for_flash(
        model_path=model_path,
        rank=args.rank,
        sensitive_layers=args.sensitive_layers,
        sensitive_rank_multiplier=args.sensitive_rank_multiplier,
        num_samples=args.samples,
        calibration_dataset=args.calibration_dataset,
        activation_threshold=args.threshold,
        epochs=args.epochs,
        train_lookahead=experimental.flash_prefetch,
        progress_callback=_flash_progress,
    )

    print("\nFlash preparation complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use flash inference:")
    print("  OLMLX_EXPERIMENTAL_FLASH=true olmlx serve")


def cmd_flash_info(args):
    """Show flash preparation info for a model."""
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
    print("  OLMLX_EXPERIMENTAL_FLASH=true olmlx serve")


def build_parser() -> argparse.ArgumentParser:
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
        "--speculative-draft-model",
        dest="speculative_draft_model",
        default=None,
        help="HuggingFace path of the draft model used for speculative decoding",
    )
    serve_p.add_argument(
        "--speculative-tokens",
        dest="speculative_tokens",
        type=_positive_int,
        default=None,
        help="Number of tokens drafted per verification step (default: 4)",
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
        help="Timeout for MCP tool calls in seconds (default: 30)",
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


def cli_main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "serve":
        # Bare invocation: derive serve-subparser defaults from the
        # parser itself rather than hardcoding the flag list. New serve
        # flags wire through automatically; any top-level flags already
        # on ``args`` win because ``hasattr`` short-circuits the copy.
        # Invariant: top-level parser flag names must not overlap with
        # serve-only flag names — otherwise this loop would suppress the
        # serve default for the colliding name. The root parser only
        # declares the ``command`` dest today, so this holds.
        if args.command is None:
            serve_defaults = vars(parser.parse_args(["serve"]))
            for _name, _default in serve_defaults.items():
                if not hasattr(args, _name):
                    setattr(args, _name, _default)
        cmd_serve(args)
    elif args.command == "service":
        if args.service_command == "install":
            cmd_service_install(args)
        elif args.service_command == "uninstall":
            cmd_service_uninstall(args)
        elif args.service_command == "status":
            cmd_service_status(args)
        else:
            parser.parse_args(["service", "--help"])
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "models":
        if args.models_command == "list":
            cmd_models_list(args)
        elif args.models_command == "pull":
            cmd_models_pull(args)
        elif args.models_command == "show":
            cmd_models_show(args)
        elif args.models_command == "delete":
            cmd_models_delete(args)
        elif args.models_command == "search":
            cmd_models_search(args)
        else:
            parser.parse_args(["models", "--help"])
    elif args.command == "flash":
        if args.flash_command == "prepare":
            cmd_flash_prepare(args)
        elif args.flash_command == "info":
            cmd_flash_info(args)
        else:
            parser.parse_args(["flash", "--help"])
    elif args.command == "spectral":
        if args.spectral_command == "prepare":
            cmd_spectral_prepare(args)
        else:
            parser.parse_args(["spectral", "--help"])
    elif args.command == "bench":
        if args.bench_command == "run":
            cmd_bench_run(args)
        elif args.bench_command == "compare":
            cmd_bench_compare(args)
        elif args.bench_command == "list":
            cmd_bench_list(args)
        elif args.bench_command == "leaderboard":
            cmd_bench_leaderboard(args)
        else:
            parser.parse_args(["bench", "--help"])
    elif args.command == "config":
        if args.config_command == "show":
            cmd_config_show(args)
        else:
            parser.parse_args(["config", "--help"])
