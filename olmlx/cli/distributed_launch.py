"""Distributed worker launch, sharding, and cleanup for serve."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from olmlx.config import (
    PROMOTED_FLASH_ENV_RENAMES,
)

logger = logging.getLogger(__name__)


_VALID_HOSTNAME_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


_worker_procs: list[subprocess.Popen] = []
_worker_log_fhs: list = []
_atexit_registered = False
_signal_handlers_installed = False


def _install_signal_handlers() -> None:
    """Install SIGTERM/SIGINT handlers that clean up distributed workers.

    Python's default SIGTERM disposition terminates the process *without*
    running atexit handlers, so the atexit-only cleanup orphans the SSH
    worker processes when the coordinator is killed during the long
    pre-uvicorn startup window (worker launch + slow ``import transformers``).
    Once ``uvicorn.run()`` takes over it installs its own signal handlers
    (replacing these) and shuts down gracefully, which runs atexit — so
    these handlers only need to cover the startup window.

    A pre-existing non-default Python handler is chained after cleanup;
    ``signal.default_int_handler`` is intentionally not chained (it raises
    KeyboardInterrupt, preempting the ``128 + signum`` exit).
    """
    import signal

    global _signal_handlers_installed
    if _signal_handlers_installed:
        return
    _signal_handlers_installed = True

    def _make_handler(previous):
        def _signal_cleanup(signum, frame):
            # The exit must happen even if cleanup or the chained handler
            # raises — surviving the signal would leave the coordinator
            # alive in an undefined state.
            try:
                _cleanup_workers()
                if callable(previous) and previous is not signal.default_int_handler:
                    previous(signum, frame)
            except Exception:
                logger.exception("Worker cleanup failed during signal shutdown")
            finally:
                sys.exit(128 + signum)

        return _signal_cleanup

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, _make_handler(signal.getsignal(signum)))


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
    try:
        (Path.home() / ".olmlx" / "ring_hostfile.json").unlink(missing_ok=True)
    except Exception:
        pass


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

    # Local import: breaks the distributed_launch → models_cmd → serve →
    # distributed_launch cycle; binds at call time so patching
    # olmlx.cli.models_cmd._create_store still takes effect here.
    from olmlx.cli.models_cmd import _create_store

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

    # Late-bound on purpose: tests rebind olmlx.config.settings wholesale
    # (monkeypatch.setattr("olmlx.config.settings", MagicMock(...))), which a
    # module-level `from olmlx.config import settings` would not see.
    from olmlx.config import settings  # noqa: PLC0415

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

    # atexit alone is insufficient: the default SIGTERM disposition skips
    # atexit handlers, orphaning SSH workers if the coordinator is killed
    # before uvicorn (which installs its own handlers) takes over (#461).
    _install_signal_handlers()

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
            # rather than relying on os.environ passthrough, so the worker
            # gets the coordinator's effective config under the canonical
            # ``OLMLX_FLASH_*`` names. ``settings`` honors only the new
            # names; a stale legacy ``OLMLX_EXPERIMENTAL_FLASH_*`` knob is
            # no longer applied (warn-only), so its value does not flow
            # here — the worker sees the schema default for that knob,
            # matching the coordinator.
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
            # Forward the advanced tuning knobs that still live under the
            # experimental prefix (window_size, io_threads,
            # cache_budget_neurons, predictor_*, prefetch_*,
            # bypass_os_cache, preallocated_buffer) verbatim. The
            # *promoted* legacy names (in ``PROMOTED_FLASH_ENV_RENAMES``,
            # e.g. OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD,
            # OLMLX_EXPERIMENTAL_FLASH_MOE*) are skipped: they are no
            # longer honoured and their effective values already flow
            # under the new OLMLX_FLASH_* names above — forwarding them
            # would only make each worker warn about a name the operator
            # set on the *coordinator*.
            for key, val in os.environ.items():
                if key in env or key in PROMOTED_FLASH_ENV_RENAMES:
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
