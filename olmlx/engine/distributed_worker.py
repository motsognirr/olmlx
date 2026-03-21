"""Worker entry point for distributed inference (non-rank-0 nodes).

Launched on remote hosts via SSH. Connects to the rank 0 coordinator,
loads the model, shards it, then runs stream_generate in lockstep
with rank 0 for each inference request.

Environment variables:
    OLMLX_EXPERIMENTAL_DISTRIBUTED_MODEL: HF model path to load
    OLMLX_EXPERIMENTAL_DISTRIBUTED_COORDINATOR_HOST: rank 0 hostname
    OLMLX_EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT: coordinator sideband port
    OLMLX_EXPERIMENTAL_DISTRIBUTED_SECRET: shared secret for authentication
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


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

    model, tokenizer = mlx_lm.load(str(shard_dir))
    if not hasattr(model, "shard"):
        raise RuntimeError(f"Model in {shard_dir} does not support shard()")
    model.shard(group)

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


def worker_main() -> None:
    """Main loop for distributed worker nodes."""
    import mlx.core as mx

    from olmlx.engine.distributed import DistributedWorker, distributed_barrier

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model_path = os.environ.get("OLMLX_EXPERIMENTAL_DISTRIBUTED_MODEL")
    if not model_path:
        logger.error("OLMLX_EXPERIMENTAL_DISTRIBUTED_MODEL not set")
        sys.exit(1)

    coordinator_host = os.environ.get(
        "OLMLX_EXPERIMENTAL_DISTRIBUTED_COORDINATOR_HOST", "127.0.0.1"
    )
    sideband_port = int(
        os.environ.get("OLMLX_EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT", "32400")
    )
    secret = os.environ.get("OLMLX_EXPERIMENTAL_DISTRIBUTED_SECRET", "") or None

    # Initialize MLX distributed
    backend = os.environ.get("OLMLX_EXPERIMENTAL_DISTRIBUTED_BACKEND", "ring")
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

    # Read distributed strategy config
    strategy = os.environ.get("OLMLX_EXPERIMENTAL_DISTRIBUTED_STRATEGY", "tensor")

    # Load and shard the model
    import mlx_lm

    from olmlx.config import PRE_SHARDED_DIR_ENV

    if strategy == "pipeline":
        # Pipeline mode: load full model, apply pipeline partitioning
        layer_counts_str = os.environ.get(
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_LAYER_COUNTS", ""
        )
        try:
            layer_counts = (
                [int(x) for x in layer_counts_str.split(",") if x]
                if layer_counts_str
                else None
            )
        except ValueError:
            logger.error(
                "Invalid OLMLX_EXPERIMENTAL_DISTRIBUTED_LAYER_COUNTS=%r, "
                "expected comma-separated integers",
                layer_counts_str,
            )
            worker.close()
            sys.exit(1)
        logger.info("Loading model %s (pipeline strategy)", model_path)
        model, tokenizer = mlx_lm.load(model_path)

        from olmlx.engine.pipeline import apply_pipeline

        try:
            apply_pipeline(model, group, layer_counts=layer_counts)
        except ValueError as e:
            logger.error("Pipeline setup failed: %s", e)
            worker.close()
            sys.exit(1)
        mx.eval(model.parameters())  # materialize owned weights on GPU
    else:
        pre_shard_dir = os.environ.get(PRE_SHARDED_DIR_ENV)
        if pre_shard_dir:
            try:
                model, tokenizer = _load_pre_sharded(pre_shard_dir, group)
            except Exception as e:
                logger.warning(
                    "Pre-sharded load failed (%s), falling back to HF download", e
                )
                pre_shard_dir = None
        if not pre_shard_dir:
            logger.info("Loading model %s", model_path)
            model, tokenizer = mlx_lm.load(model_path)

            if not hasattr(model, "shard"):
                logger.error("Model %s does not support shard()", model_path)
                worker.close()
                sys.exit(1)

            model.shard(group)
            # Materialize all lazy weight slices before entering inference.
            # Without this, the combined lazy eval + all_sum Metal command
            # buffer can exceed the ~10s GPU timeout for large models (32B+).
            mx.eval(model.parameters())
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
