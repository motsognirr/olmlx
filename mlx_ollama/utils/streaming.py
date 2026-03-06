import asyncio
import logging
import threading
import traceback
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class StreamToken:
    text: str
    token: int | None
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    finish_reason: str | None = None


_SENTINEL = object()
_ERROR_KEY = "__error__"
_QUEUE_PUT_TIMEOUT = 10.0  # seconds


class CancellableStream:
    """Async iterable wrapping a sync generator in a background thread.

    Provides cancellation via a threading.Event and drain_and_join() to wait
    for the background thread to finish (ensuring Metal operations complete
    before releasing locks).
    """

    def __init__(self, gen_factory: Callable[[threading.Event], Generator], is_vlm: bool = False):
        """
        Args:
            gen_factory: Called with a cancel_event; should return a generator
                         that yields response objects with text/token/etc attrs.
            is_vlm: Whether the model is a VLM (affects which generation_stream to sync).
        """
        self._gen_factory = gen_factory
        self._is_vlm = is_vlm
        self._cancel_event = threading.Event()
        self._stream_done = threading.Event()
        self._queue: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=32)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancel_event.set()

    def _run(self):
        gen = None
        try:
            gen = self._gen_factory(self._cancel_event)
            for resp in gen:
                if self._cancel_event.is_set():
                    break
                tok = StreamToken(
                    text=resp.text,
                    token=getattr(resp, "token", None),
                    prompt_tokens=resp.prompt_tokens,
                    generation_tokens=resp.generation_tokens,
                    prompt_tps=resp.prompt_tps,
                    generation_tps=resp.generation_tps,
                    finish_reason=getattr(resp, "finish_reason", None),
                )
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._queue.put(tok), self._loop
                    ).result(timeout=_QUEUE_PUT_TIMEOUT)
                except Exception:
                    break
        except Exception as exc:
            tb = traceback.format_exc()
            try:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put({
                        _ERROR_KEY: str(exc),
                        "__exc_type__": type(exc).__name__,
                        "__traceback__": tb,
                    }), self._loop
                ).result(timeout=_QUEUE_PUT_TIMEOUT)
            except Exception:
                pass
        finally:
            # Explicitly close the generator FIRST — this triggers
            # wired_limit.__exit__ inside mlx_lm/mlx_vlm which calls
            # mx.synchronize(generation_stream), ensuring all GPU work on
            # the generation stream completes before we signal done.
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass

            # Sync both the generation stream and the default stream.
            # mlx_lm and mlx_vlm run GPU work on their own module-level
            # generation_stream. We must also sync the default stream to
            # catch any Metal operations not on the generation stream.
            try:
                import mlx.core as mx
                if self._is_vlm:
                    from mlx_vlm.generate import generation_stream
                else:
                    from mlx_lm.generate import generation_stream
                mx.synchronize(generation_stream)
                mx.synchronize()  # default stream too
            except Exception:
                try:
                    import mlx.core as mx
                    mx.synchronize()
                except Exception:
                    pass

            # Signal completion before posting sentinel
            self._stream_done.set()

            try:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(_SENTINEL), self._loop
                ).result(timeout=_QUEUE_PUT_TIMEOUT)
            except Exception:
                pass

    async def drain_and_join(self):
        """Drain remaining items from the queue and wait for the thread to finish.

        IMPORTANT: This must wait for the thread to truly finish before returning,
        otherwise Metal operations from the dying thread can overlap with a new
        inference, causing '[_MTLCommandBuffer addCompletedHandler:] failed assertion'.
        """
        self._cancel_event.set()
        # If the stream already finished (sentinel posted), skip queue draining.
        # This avoids the 10s timeout when the sentinel was already consumed
        # by the async for loop (causing StopAsyncIteration).
        if not self._stream_done.is_set() and self._queue is not None:
            # Drain the queue until we see the sentinel.
            # Keep waiting as long as the background thread is alive.
            while True:
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=10.0)
                    if item is _SENTINEL:
                        break
                except asyncio.TimeoutError:
                    if self._thread is None or not self._thread.is_alive():
                        break
                    # Thread still running (e.g. long prefill) — keep waiting
                    logger.debug("drain_and_join: thread still alive, continuing to wait")
                    continue
        if self._thread is not None:
            # Wait for the thread to fully exit — no timeout
            await asyncio.to_thread(self._thread.join)

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamToken:
        item = await self._queue.get()
        if item is _SENTINEL:
            raise StopAsyncIteration
        if isinstance(item, dict) and _ERROR_KEY in item:
            exc_type = item.get("__exc_type__", "RuntimeError")
            tb = item.get("__traceback__", "")
            if tb:
                logger.error("Inference error (%s): %s\n%s", exc_type, item[_ERROR_KEY], tb)
            raise RuntimeError(f"{exc_type}: {item[_ERROR_KEY]}")
        return item


def async_mlx_stream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 512,
    is_vlm: bool = False,
    images: list[str] | None = None,
    **kwargs: Any,
) -> CancellableStream:
    """Bridge sync mlx_lm/mlx_vlm stream_generate into an async iterable.

    Returns a CancellableStream (started and ready to iterate).
    """
    def gen_factory(cancel_event: threading.Event):
        if is_vlm:
            import mlx_vlm
            return mlx_vlm.stream_generate(
                model, tokenizer, prompt=prompt, image=images,
                max_tokens=max_tokens, **kwargs,
            )
        else:
            import mlx_lm
            return mlx_lm.stream_generate(
                model, tokenizer, prompt=prompt,
                max_tokens=max_tokens, **kwargs,
            )

    stream = CancellableStream(gen_factory, is_vlm=is_vlm)
    stream.start()
    return stream
