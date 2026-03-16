"""Worker entry point for distributed inference (non-rank-0 nodes).

Launched on remote hosts via SSH. Connects to the rank 0 coordinator,
loads the model, shards it, then runs stream_generate in lockstep
with rank 0 for each inference request.

Environment variables:
    EXPERIMENTAL_DISTRIBUTED_MODEL: HF model path to load
    EXPERIMENTAL_DISTRIBUTED_COORDINATOR_HOST: rank 0 hostname
    EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT: coordinator sideband port
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def worker_main() -> None:
    """Main loop for distributed worker nodes."""
    import mlx.core as mx

    from olmlx.engine.distributed import DistributedWorker

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model_path = os.environ.get("EXPERIMENTAL_DISTRIBUTED_MODEL")
    if not model_path:
        logger.error("EXPERIMENTAL_DISTRIBUTED_MODEL not set")
        sys.exit(1)

    coordinator_host = os.environ.get(
        "EXPERIMENTAL_DISTRIBUTED_COORDINATOR_HOST", "127.0.0.1"
    )
    sideband_port = int(
        os.environ.get("EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT", "32400")
    )

    # Initialize MLX distributed
    group = mx.distributed.init()
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

    logger.info("Loading model %s", model_path)
    model, tokenizer = mlx_lm.load(model_path)

    if not hasattr(model, "shard"):
        logger.error("Model %s does not support shard()", model_path)
        worker.close()
        sys.exit(1)

    model.shard(group)
    logger.info("Model sharded, entering inference loop")

    # Main loop: wait for broadcast → run stream_generate → repeat
    try:
        while True:
            req = worker.wait_for_inference()
            if req is None:
                logger.info("Received shutdown, exiting")
                break

            # Run stream_generate in lockstep with rank 0.
            # The sharded model's all_sum ops synchronize with other ranks.
            # Output is discarded — only rank 0 returns results.
            for _ in mlx_lm.stream_generate(
                model,
                tokenizer,
                prompt=req.prompt_tokens,
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
