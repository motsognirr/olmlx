# Streaming Architecture

This document describes olmlx's streaming architecture for bridging synchronous MLX inference with async FastAPI.

## Overview

MLX inference (`mlx_lm.stream_generate()`, `mlx_vlm.stream_generate()`) runs synchronously on a background thread. olmlx bridges this to async via `CancellableStream`, which:

1. Runs the sync generator in a daemon thread
2. Yields tokens through an `asyncio.Queue`
3. Provides `drain_and_join()` to wait for thread completion before releasing locks

## Key Components

### CancellableStream

Located in `olmlx/utils/streaming.py`.

- **Purpose**: Wrap a sync generator as an async iterable
- **Cancellation**: Uses `threading.Event` to signal the background thread
- **Thread Safety**: Synchronizes Metal GPU streams before signaling completion

### drain_and_join()

Critical for Metal GPU safety. Must be called in `try/finally` to ensure:

1. Queue is drained until sentinel
2. Background thread is joined (waits for exit)
3. Metal streams are synchronized before lock release

If the thread doesn't exit within the timeout (default 60s), logs an error indicating potential GPU resource leak.

### async_mlx_stream()

Factory function that creates a `CancellableStream` for MLX inference:

- Detects VLM vs text model
- Handles `prompt_progress_callback` for prefill cancellation (mlx-lm only)
- Returns a started stream ready to iterate

## Metal GPU Safety

The streaming architecture exists to prevent Metal crashes:

- **Concurrent command buffer access**: Multiple threads issuing GPU commands simultaneously causes uncatchable Metal assertion failures
- **Lock release before cleanup**: Releasing `_inference_lock` before the background thread finishes GPU work leads to race conditions
- **Solution**: `drain_and_join()` synchronizes both the generation stream and default stream, then joins the thread, all before the lock is released

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
    async for token in stream:
        yield {...}
finally:
    await stream.aclose()  # ensures drain_and_join() is called
```

The `try/finally` ensures cleanup happens even on client disconnect.