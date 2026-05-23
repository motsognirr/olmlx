"""Worker entry point for distributed inference (non-rank-0 nodes).

Launched on remote hosts via SSH. Connects to the rank 0 coordinator,
loads the model, shards it, then runs stream_generate in lockstep
with rank 0 for each inference request.

Environment variables:
    OLMLX_DISTRIBUTED_MODEL: HF model path to load
    OLMLX_DISTRIBUTED_COORDINATOR_HOST: rank 0 hostname
    OLMLX_DISTRIBUTED_SIDEBAND_PORT: coordinator sideband port
    OLMLX_DISTRIBUTED_SECRET: shared secret for authentication

Legacy (OLMLX_EXPERIMENTAL_DISTRIBUTED_*) names are also accepted
for backward compatibility.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


# New env var name → legacy env var name for backward compatibility.
_DISTRIBUTED_ENV_MAP: dict[str, str] = {
    "OLMLX_DISTRIBUTED_MODEL": "OLMLX_EXPERIMENTAL_DISTRIBUTED_MODEL",
    "OLMLX_DISTRIBUTED_COORDINATOR_HOST": "OLMLX_EXPERIMENTAL_DISTRIBUTED_COORDINATOR_HOST",
    "OLMLX_DISTRIBUTED_SIDEBAND_PORT": "OLMLX_EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT",
    "OLMLX_DISTRIBUTED_SECRET": "OLMLX_EXPERIMENTAL_DISTRIBUTED_SECRET",
    "OLMLX_DISTRIBUTED_SECRET_FILE": "OLMLX_EXPERIMENTAL_DISTRIBUTED_SECRET_FILE",
    "OLMLX_DISTRIBUTED_BACKEND": "OLMLX_EXPERIMENTAL_DISTRIBUTED_BACKEND",
    "OLMLX_DISTRIBUTED_STRATEGY": "OLMLX_EXPERIMENTAL_DISTRIBUTED_STRATEGY",
    "OLMLX_DISTRIBUTED_LAYER_COUNTS": "OLMLX_EXPERIMENTAL_DISTRIBUTED_LAYER_COUNTS",
}


def _get_env(name: str, default: str | None = None) -> str | None:
    """Read *name* from the environment with legacy fallback."""
    val = os.environ.get(name)
    if val is not None:
        return val
    legacy_name = _DISTRIBUTED_ENV_MAP.get(name)
    if legacy_name is not None:
        val = os.environ.get(legacy_name)
        if val is not None:
            logger.warning("Using legacy env var %s — rename to %s", legacy_name, name)
            return val
    return default


def _load_pre_sharded(shard_dir_str, group):
    """Load model from pre-sharded weights directory.

    1. Load model with pre-sharded weights (smaller shapes)
    2. model.shard(group) for structural distributed layer conversion
    3. Reload pre-sharded weights to overwrite double-split values
    4. Materialize parameters
    """
    from pathlib import Path

    import mlx.core as mx
    import mlx_lm

    shard_dir = Path(shard_dir_str).expanduser()
    logger.info("Loading pre-sharded weights from %s", shard_dir)

    model, tokenizer = mlx_lm.load(str(shard_dir))  # pyright: ignore[reportAssignmentType]
    shard_fn = getattr(model, "shard", None)
    if shard_fn is None:
        raise RuntimeError(f"Model in {shard_dir} does not support shard()")
    shard_fn(group)

    # Overwrite double-split weights with correct pre-sharded values
    weights_path = shard_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Pre-sharded weights not found at {weights_path}. "
            "The shard directory may be corrupt — delete it to trigger re-sharding."
        )
    model.load_weights(str(weights_path), strict=False)

    mx.eval(model.parameters())
    logger.info("Pre-sharded model loaded and materialized")
    return model, tokenizer


def _load_flash_tensor_worker(model_path: str, group) -> tuple:
    """Load a Flash model for tensor-parallel worker.

    Loads the model, wraps it with FlashModelWrapper (replacing MLP layers
    with FlashMLP instances), then shards only attention layers. Each worker
    must have flash-prepared data on its local SSD.
    """
    import json
    from pathlib import Path

    import mlx.core as mx
    import mlx_lm

    from olmlx.config import experimental, settings
    from olmlx.engine.flash.flash_model import FlashConfig, FlashModelWrapper
    from olmlx.engine.flash.predictor import PredictorBank
    from olmlx.engine.flash.weight_store import FlashWeightStore
    from olmlx.models.store import _safe_dir_name

    flash_dir = Path(settings.models_dir) / _safe_dir_name(model_path) / "flash"
    if not flash_dir.exists() or not (flash_dir / "flash_layout.json").exists():
        raise FileNotFoundError(
            f"Flash data not found at {flash_dir}. "
            f"Run 'olmlx flash prepare {model_path}' on this worker node first."
        )

    # Verify the model is fully cached locally. Any machine that ran
    # `olmlx flash prepare` will have it cached, but the cache could have
    # been cleared since then.  Fail fast rather than hanging the ring for
    # hours on a silent download.
    if not Path(model_path).is_dir():
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            snapshot_download(model_path, local_files_only=True)
        except (LocalEntryNotFoundError, FileNotFoundError) as e:
            raise FileNotFoundError(
                f"Model {model_path!r} not fully cached. "
                f"Run 'olmlx flash prepare {model_path}' on this worker node "
                "first."
            ) from e

    logger.info("Loading flash model %s from %s", model_path, flash_dir)
    model, tokenizer = mlx_lm.load(model_path)  # pyright: ignore[reportAssignmentType]

    predictor_bank = PredictorBank.load(flash_dir / "predictors")
    layout_config = json.loads((flash_dir / "flash_layout.json").read_text())

    flash_config = FlashConfig(
        hidden_size=layout_config["hidden_size"],
        intermediate_size=layout_config["intermediate_size"],
        num_layers=layout_config["num_layers"],
        sparsity_threshold=settings.flash_sparsity_threshold,
        min_active_neurons=settings.flash_min_active_neurons,
        max_active_neurons=settings.flash_max_active_neurons,
        window_size=experimental.flash_window_size,
        io_threads=experimental.flash_io_threads,
        cache_budget_neurons=experimental.flash_cache_budget_neurons,
        memory_budget_fraction=settings.flash_memory_budget_fraction,
    )

    weight_store = FlashWeightStore(
        flash_dir,
        num_io_threads=flash_config.io_threads,
        cache_budget_neurons=flash_config.cache_budget_neurons,
        bypass_cache=experimental.flash_bypass_os_cache,
        use_preallocated_buffer=experimental.flash_preallocated_buffer,
    )

    wrapped = FlashModelWrapper(model, predictor_bank, weight_store, flash_config)
    wrapped.shard(group)
    mx.eval(wrapped.parameters())
    logger.info("Flash model loaded and sharded (attention-only)")
    return wrapped, tokenizer


def worker_main() -> None:
    """Main loop for distributed worker nodes."""
    import mlx.core as mx

    from olmlx.engine.distributed import DistributedWorker, distributed_barrier

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model_path = _get_env("OLMLX_DISTRIBUTED_MODEL")
    if not model_path:
        logger.error("OLMLX_DISTRIBUTED_MODEL not set")
        sys.exit(1)

    # Check Flash-MoE before ring init — exiting after init hangs the
    # coordinator on ring collectives.
    from olmlx.config import (
        settings as _settings_early,
        surface_legacy_flash_env,
        surface_legacy_flash_moe_env,
        surface_legacy_flash_prefetch_speculative_env,
    )

    # Honour the one-release deprecation window for ``OLMLX_EXPERIMENTAL_FLASH*``
    # and ``OLMLX_EXPERIMENTAL_FLASH_MOE*`` on the direct-worker path. The
    # coordinator runs the same shims in its own startup and forwards resolved
    # ``OLMLX_FLASH*``/``OLMLX_FLASH_MOE*`` values to workers it launches via
    # SSH, so these matter only when ``python -m olmlx.engine.distributed_worker``
    # is invoked directly with legacy env vars set. Both helpers live in
    # ``olmlx.config`` so the worker does not need to import ``olmlx.cli``
    # (and its argparse/uvicorn baggage).
    surface_legacy_flash_env()
    surface_legacy_flash_moe_env()
    surface_legacy_flash_prefetch_speculative_env()

    if _settings_early.flash_moe:
        logger.error(
            "Flash-MoE + distributed is not supported. "
            "Disable OLMLX_FLASH_MOE or "
            "OLMLX_DISTRIBUTED."
        )
        sys.exit(1)

    strategy = _get_env("OLMLX_DISTRIBUTED_STRATEGY", "tensor")
    if strategy == "pipeline" and _settings_early.flash:
        logger.error(
            "Flash + pipeline distributed strategy is not supported. "
            "Use tensor strategy or disable Flash."
        )
        sys.exit(1)

    if _settings_early.flash:
        from pathlib import Path

        from olmlx.config import settings
        from olmlx.models.store import _safe_dir_name

        _flash_dir = Path(settings.models_dir) / _safe_dir_name(model_path) / "flash"
        if not _flash_dir.exists() or not (_flash_dir / "flash_layout.json").exists():
            logger.error(
                "Flash data not found at %s. "
                "Run 'olmlx flash prepare %s' on this worker node first.",
                _flash_dir,
                model_path,
            )
            sys.exit(1)

    coordinator_host = _get_env("OLMLX_DISTRIBUTED_COORDINATOR_HOST", "127.0.0.1")
    sideband_port = int(_get_env("OLMLX_DISTRIBUTED_SIDEBAND_PORT", "32400"))

    secret_file = _get_env("OLMLX_DISTRIBUTED_SECRET_FILE")
    if secret_file:
        from pathlib import Path

        secret_path = Path(secret_file).expanduser()
        try:
            secret = secret_path.read_text().strip()
            secret_path.unlink()
        except OSError as e:
            logger.error("Failed to read secret from %s: %s", secret_file, e)
            sys.exit(1)
    else:
        secret = _get_env("OLMLX_DISTRIBUTED_SECRET", "") or None

    # Initialize MLX distributed
    backend = _get_env("OLMLX_DISTRIBUTED_BACKEND", "ring")
    group = mx.distributed.init(backend=backend)
    rank = group.rank()
    world_size = group.size()
    logger.info("Worker rank %d/%d starting", rank, world_size)

    if rank == 0:
        logger.info("Rank 0 should run the full server, not the worker")
        return

    # Connect to coordinator sideband
    worker = DistributedWorker(
        coordinator_host=coordinator_host,
        port=sideband_port,
    )

    # Load and shard the model
    import mlx_lm

    from olmlx.config import PRE_SHARDED_DIR_ENV, settings

    if strategy == "pipeline":
        # Pipeline mode: load model, apply pipeline partitioning
        layer_counts_str = _get_env("OLMLX_DISTRIBUTED_LAYER_COUNTS", "")
        try:
            layer_counts = (
                [int(x) for x in layer_counts_str.split(",") if x]
                if layer_counts_str
                else None
            )
        except ValueError:
            logger.error(
                "Invalid OLMLX_DISTRIBUTED_LAYER_COUNTS=%r, "
                "expected comma-separated integers",
                layer_counts_str,
            )
            worker.close()
            sys.exit(1)

        from olmlx.engine.pipeline import apply_pipeline

        pre_shard_dir = os.environ.get(PRE_SHARDED_DIR_ENV)
        pre_sharded = False
        if pre_shard_dir:
            try:
                from pathlib import Path

                shard_path = Path(pre_shard_dir).expanduser()
                logger.info("Loading pre-sharded pipeline weights from %s", shard_path)
                model, tokenizer = mlx_lm.load(str(shard_path))  # pyright: ignore[reportAssignmentType]
                pre_sharded = True
            except Exception as e:
                logger.warning(
                    "Pre-sharded pipeline load failed (%s), "
                    "falling back to full model download",
                    e,
                )
                model, tokenizer = mlx_lm.load(model_path)  # pyright: ignore[reportAssignmentType]
        else:
            logger.info("Loading model %s (pipeline strategy)", model_path)
            model, tokenizer = mlx_lm.load(model_path)  # pyright: ignore[reportAssignmentType]

        try:
            apply_pipeline(
                model, group, layer_counts=layer_counts, pre_sharded=pre_sharded
            )
        except ValueError as e:
            logger.error("Pipeline setup failed: %s", e)
            worker.close()
            sys.exit(1)
        mx.eval(model.parameters())  # materialize owned weights on GPU
    elif strategy == "tensor":
        if settings.flash:
            try:
                model, tokenizer = _load_flash_tensor_worker(model_path, group)
            except Exception:
                logger.exception("Flash model load failed")
                worker.close()
                sys.exit(1)
        else:
            pre_shard_dir = os.environ.get(PRE_SHARDED_DIR_ENV)
            # `model is None` is the fallback signal: if _load_pre_sharded
            # raises before returning, the assignment never executes, so
            # model stays None from the pre-declaration above — the second
            # branch then runs the HF download.
            model = tokenizer = None
            if pre_shard_dir:
                try:
                    model, tokenizer = _load_pre_sharded(pre_shard_dir, group)
                except Exception as e:
                    logger.warning(
                        "Pre-sharded load failed (%s), falling back to HF download",
                        e,
                    )
            if model is None:
                logger.info("Loading model %s", model_path)
                model, tokenizer = mlx_lm.load(model_path)  # pyright: ignore[reportAssignmentType]

                shard_fn = getattr(model, "shard", None)
                if shard_fn is None:
                    logger.error("Model %s does not support shard()", model_path)
                    worker.close()
                    sys.exit(1)

                shard_fn(group)
                # Materialize all lazy weight slices before entering inference.
                # Without this, the combined lazy eval + all_sum Metal command
                # buffer can exceed the ~10s GPU timeout for large models (32B+).
                mx.eval(model.parameters())
    else:
        worker.close()
        raise AssertionError(
            "unreachable: distributed_strategy is Literal['tensor']; "
            "pydantic rejects any other value at config parse"
        )
    worker.send_ready(secret=secret)
    logger.info("Model sharded, ready signal sent, entering inference loop")

    # Main loop: wait for broadcast → run stream_generate → repeat
    # Note: if rank 0 fails mid-inference (OOM, device error), workers will
    # hang on all_sum indefinitely. This is an inherent MLX limitation —
    # there is no timeout on collective operations. The worker must be
    # killed externally (the atexit handler in cli.py handles this).
    try:
        while True:
            req = worker.wait_for_inference()
            if req is None:
                logger.info("Received shutdown, exiting")
                break

            # Barrier: synchronize with coordinator before heavy compute.
            # The coordinator broadcasts via sideband then hits the same
            # barrier.  This prevents Metal GPU timeouts from one rank
            # starting all_sum ops before the other is ready.
            distributed_barrier()

            # Run stream_generate in lockstep with rank 0.
            # The sharded model's all_sum ops synchronize with other ranks.
            # Output is discarded — only rank 0 returns results.
            for _ in mlx_lm.stream_generate(
                model,
                tokenizer,
                prompt=req.prompt_text,
                max_tokens=req.max_tokens,
                **req.gen_kwargs,
            ):
                pass
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
    finally:
        worker.close()
        logger.info("Worker rank %d exiting", rank)


if __name__ == "__main__":
    worker_main()
