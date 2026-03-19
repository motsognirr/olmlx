# Streaming Architecture

This document describes olmlx's streaming architecture for bridging synchronous MLX inference with async FastAPI.

## Overview

MLX inference (`mlx_lm.stream_generate()`, `mlx_vlm.stream_generate()`) runs synchronously on a background thread. olmlx bridges this to async via `CancellableStream`, which:

1. Runs the sync generator in a daemon thread
2. Yields tokens through an `asyncio.Queue`
3. Synchronizes Metal GPU streams in the background thread before signaling completion

## Key Components

### CancellableStream

Located in `olmlx/utils/streaming.py`.

- **Purpose**: Wrap a sync generator as an async iterable
- **Cancellation**: Uses `threading.Event` to signal the background thread
- **Thread Safety**: The `_run()` method's `finally` block synchronizes Metal GPU streams before posting the sentinel

### _run()

The background thread method that:

1. Iterates the generator, yielding tokens to the queue
2. On exit (normal or exception), executes a `finally` block that:
   - Closes the generator (which syncs the generation stream)
   - Calls `mx.synchronize(generation_stream)` and `mx.synchronize()` to ensure all GPU work completes
   - Sets `_stream_done` event
   - Posts the sentinel to the queue

This ordering is critical: Metal sync happens **before** the sentinel is posted, so `drain_and_join()` can safely wait for the sentinel as a signal that GPU work is complete.

### drain_and_join()

Must be called in `try/finally` to ensure proper cleanup:

1. Sets the cancel event
2. Checks `_stream_done.is_set()` first — if already set, skips the drain entirely (this is the fast path when the stream ran to completion and the sentinel was consumed by the `async for` loop before `aclose()` was called)
3. Drains the queue until the sentinel (or timeout) if `_stream_done` is not set
4. Joins the background thread

By the time the sentinel is received, Metal synchronization has already completed in `_run()`. The join ensures the thread has fully exited before the caller releases `_inference_lock`.

If the thread doesn't exit within the timeout (default 60s), logs an error indicating potential GPU resource leak.

### async_mlx_stream()

Factory function that creates a `CancellableStream` for MLX inference:

- Takes `is_vlm` parameter (caller determines model type)
- For text models: uses `mlx_lm.stream_generate()` with optional `prompt_progress_callback` for prefill cancellation
- For VLMs: uses `mlx_vlm.stream_generate()`
- Returns a started stream ready to iterate

## Metal GPU Safety

The streaming architecture exists to prevent Metal crashes:

- **Concurrent command buffer access**: Multiple threads issuing GPU commands simultaneously causes uncatchable Metal assertion failures
- **Lock release before cleanup**: Releasing `_inference_lock` before the background thread finishes GPU work leads to race conditions
- **Solution**: The background thread's `_run()` method synchronizes Metal in its `finally` block before posting the sentinel. `drain_and_join()` waits for the sentinel and joins the thread, ensuring all GPU work completes before the lock is released.

## Deferred Cleanup

When a thread is stuck (e.g., long prefill that doesn't respond to cancellation), `_schedule_deferred_inference_cleanup()` in `olmlx/engine/inference.py`:

1. Polls the thread until it exits (up to 10 minutes)
2. Synchronizes Metal only after thread exits
3. Releases the lock

This avoids calling `mx.synchronize()` while the thread is still using the GPU, which would crash.

## Router Integration

All streaming routers (`routers/chat.py`, `routers/generate.py`, `routers/openai.py`, `routers/anthropic.py`) follow this pattern:

```python
try:
    async for token in result:
        yield {...}
finally:
    await result.aclose()  # wraps drain_and_join() in asyncio.shield with 10s to_thread fallback on CancelledError; without shield, client disconnect cancels drain_and_join before it can join the thread, causing the GPU race this architecture prevents
```

The `try/finally` ensures cleanup happens even on client disconnect.
