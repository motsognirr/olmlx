import asyncio
import logging
import threading
import time
import traceback
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable, cast

logger = logging.getLogger(__name__)


# --- Streaming thinking-block stripping ---
#
# Open/close tag pairs recognized by ``strip_thinking_streaming``.  Each model
# family emits its own delimiter pair; we strip them all.
#
# - ``<think>...</think>``: Qwen, DeepSeek, and others
# - ``<|channel>thought\n...<channel|>``: Gemma 4 (issue #306)
#
# Tags are stored as paired ``(open, close)`` so that, once we enter an
# ``in_think`` block, we only look for the *matching* close tag.  Matching any
# close tag here would let a ``<think>`` block whose thinking content mentions
# the literal string ``<channel|>`` (common in code-gen prompts that discuss
# token formats) exit prematurely and leak thinking content.
#
# The Gemma 4 open tag pins the literal channel name ``thought`` because that
# is the only channel emitted by every current Gemma 4 checkpoint (validated
# against the ``mlx-community/gemma-4-*-OptiQ-4bit`` family in issue #306).
# Future variants that introduce additional channel names (``thinking``,
# ``reasoning``, …) need a new entry here or the streaming filter will miss
# the open and rely on the orphan-close path within ``_STREAM_DETECT_LIMIT``.
#
# Limitation 1: when ``skip_special_tokens`` strips the Gemma 4 ``<|channel>``
# opener, the decoded stream starts with bare ``thought\n``.  We do not list
# ``thought\n`` as a standalone open tag because it would false-positive on
# legitimate prose (e.g. "I had a thought\n…") and silently drop the rest of
# the response.  The orphan-close handling on ``<channel|>`` still recovers
# short thinking blocks; only thinking longer than ``_STREAM_DETECT_LIMIT``
# bytes with a stripped opener leaks through this filter.  The non-streaming
# path uses ``parse_model_output``'s optional-prefix regex and handles that
# case correctly.
#
# Limitation 2: in ``detect`` we strip an "orphaned" close tag when
# ``thinking_expected`` is set (the chat template can pre-open a thinking
# block in the prompt so the generated text starts mid-think with only a
# closing tag).  PR #314 added the ``thinking_expected`` gate so a
# non-thinking response that legitimately mentions ``<channel|>`` or
# ``</think>`` in prose is no longer truncated — the orphan branch
# requires the engine to have signalled incoming thinking.
# ``test_orphaned_channel_close_in_prose_preserved_when_not_thinking``
# in ``tests/test_routers_openai.py`` pins this.  When
# ``thinking_expected=True`` and the response's first
# ``detect_limit`` bytes do happen to contain a literal close tag in
# prose, the prefix is still dropped — accepted because the
# ``thinking_expected`` signal is per-model and a thinking-capable
# checkpoint that emits non-thinking content discussing its own
# delimiters is a corner case.
_THINKING_TAG_PAIRS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<|channel>thought\n", "<channel|>"),
)
_THINKING_OPEN_TAGS: tuple[str, ...] = tuple(p[0] for p in _THINKING_TAG_PAIRS)
_THINKING_CLOSE_TAGS: tuple[str, ...] = tuple(p[1] for p in _THINKING_TAG_PAIRS)
_CLOSE_FOR_OPEN: dict[str, str] = dict(_THINKING_TAG_PAIRS)

# Maximum buffer size in the ``detect`` phase before giving up on finding an
# orphaned close tag and transitioning to ``passthrough``.  Sized to let
# non-thinking models start streaming quickly (the user sees output after
# ~50 tokens) while still catching the orphaned ``</think>`` that a chat
# template may have pre-opened in the prompt.  A larger limit would close
# Limitation 1 below for the stripped-Gemma-4-opener case but would also
# blank-screen the much more common non-thinking response, so we keep the
# tight bound here and accept the limitation.
#
# Cost: every streaming response (Ollama ``/api/chat`` and OpenAI
# ``/v1/chat/completions`` without tools) now pays the up-to-200-byte detect
# buffer on first connect.  A targeted optimization that consults
# ``template_caps.py`` to disable thinking-strip for non-thinking models is
# possible but requires plumbing the per-model capability through to the
# router, which is out of scope for issue #306.
#
# This limit only applies on the "no tag seen at all" branch.  Once an open
# OR close tag is found the corresponding branch fires immediately, so an
# orphaned close tag well beyond this offset is still consumed correctly
# (and the preceding bytes still discarded) without ever consulting this
# constant.
_STREAM_DETECT_LIMIT = 200


def _find_earliest(tags: tuple[str, ...], s: str) -> tuple[str, int]:
    """Return (tag, index) of the earliest occurrence of any *tag* in *s*.

    Returns ``("", -1)`` when no tag is found; callers gate on the index.
    """
    best_idx = -1
    best_tag = ""
    for t in tags:
        i = s.find(t)
        if i != -1 and (best_idx == -1 or i < best_idx):
            best_idx = i
            best_tag = t
    return best_tag, best_idx


def _longest_partial_suffix(buf: str, tags: tuple[str, ...]) -> int:
    """Largest k such that ``buf[-k:]`` is a prefix of some tag.

    Used to retain enough tail bytes that a tag spanning a chunk boundary is
    still recognized on the next chunk — symmetrically for open tags
    (passthrough phase) and close tags (in_think phase).
    """
    longest = 0
    for tag in tags:
        for i in range(min(len(tag), len(buf)), longest, -1):
            if tag.startswith(buf[-i:]):
                longest = i
                break
    return longest


def strip_thinking_streaming(text: str, state: dict) -> str:
    """Strip thinking blocks from streaming text chunks.

    Recognizes two formats and any orphaned closing tag:

    - ``<think>...</think>`` (Qwen, DeepSeek, ...)
    - ``<|channel>thought\\n...<channel|>`` (Gemma 4)

    Uses *state* dict to track position across calls.  Keys:

    - ``phase``: one of ``"detect"``, ``"in_think"``, ``"passthrough"``
    - ``buffer``: accumulated text waiting to be resolved
    - ``expected_close``: the close tag paired with the open tag that
      started the current ``in_think`` block.  Set on entry to
      ``in_think``; cleared on exit.  Looking only for this specific
      tag (instead of any close tag) prevents a cross-format close
      mentioned inside thinking content from ending the block early.
    - ``thinking_expected``: when True, the detect phase fires the
      orphan-close heuristic (template-pre-opened thinking) and
      raises the detect buffer to ``detect_limit`` (default
      ``INIT_ORPHAN_DETECT_LIMIT``); when False (default), an orphan
      close tag is left in the output so a non-thinking model that
      mentions the literal token isn't truncated (issue #307).
    - ``detect_limit``: bytes to buffer in detect phase before giving
      up the orphan-close hunt and transitioning to passthrough.
      Defaults to ``_STREAM_DETECT_LIMIT`` (200) when thinking is not
      expected; callers that know thinking is incoming should set
      this to ``INIT_ORPHAN_DETECT_LIMIT`` so a multi-thousand-char
      reasoning preamble is still recognized.

    **Phases:**

    ``detect`` (initial) — The chat template may have opened a thinking
    block inside the prompt, so the generated text could start mid-think
    with only a closing tag.  We buffer all content until we can
    determine which case we're in:

    * A close tag seen first → discard buffer (orphaned thinking),
      switch to ``passthrough``.
    * An open tag seen first → emit text before it, switch to ``in_think``.
    * Neither tag after the buffer grows large → emit buffer (no thinking).

    ``in_think`` — Inside a thinking block; discard until the matching
    close tag (``expected_close``).

    ``passthrough`` — Emit everything, but still strip any new thinking
    blocks that appear later.
    """
    buf = state.get("buffer", "") + text
    out_parts: list[str] = []
    phase = state.get("phase", "detect")

    expected_close = state.get("expected_close", "")
    thinking_expected = bool(state.get("thinking_expected"))
    detect_limit = int(state.get("detect_limit", _STREAM_DETECT_LIMIT))

    while buf:
        if phase == "detect":
            open_tag, open_idx = _find_earliest(_THINKING_OPEN_TAGS, buf)
            close_tag, close_idx = _find_earliest(_THINKING_CLOSE_TAGS, buf)

            if (
                close_idx != -1
                and (open_idx == -1 or close_idx < open_idx)
                and thinking_expected
            ):
                # Orphaned close — discard everything before it.  No ``break``
                # here: any remaining buffer after the close must be
                # processed immediately in ``passthrough`` on the next loop
                # iteration so a follow-up open tag in the same chunk is not
                # missed.
                #
                # Debug-logged because this branch fires in two legitimate
                # scenarios (chat template pre-opened thinking; Gemma 4
                # stream with stripped ``<|channel>`` opener) and one
                # pathological one ("Limitation 2": a literal
                # ``<channel|>`` / ``</think>`` in non-thinking prose,
                # which silently truncates the prefix).  Logged at DEBUG
                # rather than WARNING so the legitimate cases don't spam
                # ops; raise the level if missing content is reported and
                # ``skip_special_tokens`` may be eating the opener.
                logger.debug(
                    "strip_thinking_streaming: orphaned %r at byte %d, "
                    "discarding %d-byte prefix",
                    close_tag,
                    close_idx,
                    close_idx,
                )
                buf = buf[close_idx + len(close_tag) :]
                # Clear ``expected_close`` for symmetry with the
                # ``in_think → passthrough`` transition at line 233.  The
                # current state machine never re-reads it from
                # ``passthrough`` (it is overwritten on the next ``in_think``
                # entry), but leaving stale state across phase transitions
                # invites bugs if the machine is later extended.
                expected_close = ""
                phase = "passthrough"
            elif open_idx != -1:
                # Normal open — emit text before it, enter in_think with the
                # matching close tag remembered so cross-format mentions
                # inside the thinking content can't end the block early.
                out_parts.append(buf[:open_idx])
                buf = buf[open_idx + len(open_tag) :]
                expected_close = _CLOSE_FOR_OPEN[open_tag]
                phase = "in_think"
            else:
                if len(buf) > detect_limit:
                    # Hold back any partial *open*-tag suffix so a tag
                    # straddling the detect→passthrough transition is still
                    # recognized on the next chunk.  Close-tag partials are
                    # deliberately not retained here: ``detect_limit`` bytes
                    # without any tag means this is a non-thinking response,
                    # so a later ``<channel|>`` or ``</think>`` is literal
                    # text and should pass through unmodified.
                    partial = _longest_partial_suffix(buf, _THINKING_OPEN_TAGS)
                    out_parts.append(buf[:-partial] if partial else buf)
                    buf = buf[-partial:] if partial else ""
                    phase = "passthrough"
                break

        elif phase == "in_think":
            close_idx = buf.find(expected_close)
            if close_idx == -1:
                # The expected close may straddle a chunk boundary.  Keep just
                # enough tail so the suffix can complete on the next chunk;
                # without this, splits like ``"</thi" + "nk>"`` would leave
                # the filter stuck in_think and silently drop the rest of the
                # stream.
                longest_partial = _longest_partial_suffix(buf, (expected_close,))
                buf = buf[-longest_partial:] if longest_partial else ""
                break
            else:
                buf = buf[close_idx + len(expected_close) :]
                expected_close = ""
                phase = "passthrough"

        else:  # passthrough
            open_tag, open_idx = _find_earliest(_THINKING_OPEN_TAGS, buf)
            if open_idx == -1:
                longest_partial = _longest_partial_suffix(buf, _THINKING_OPEN_TAGS)
                if longest_partial:
                    out_parts.append(buf[:-longest_partial])
                    buf = buf[-longest_partial:]
                    break
                else:
                    out_parts.append(buf)
                    buf = ""
            else:
                out_parts.append(buf[:open_idx])
                buf = buf[open_idx + len(open_tag) :]
                expected_close = _CLOSE_FOR_OPEN[open_tag]
                phase = "in_think"

    state["buffer"] = buf
    state["phase"] = phase
    state["expected_close"] = expected_close
    return "".join(out_parts)


def flush_thinking_buffer(state: dict) -> str:
    """Flush any remaining buffer when the stream ends.

    In ``detect`` the buffer holds the entire pre-tag output; in
    ``passthrough`` it may hold a partial open-tag suffix that was withheld
    in case a tag straddled a chunk boundary.  Both are real visible bytes
    once we know no more chunks are coming, so we return them.  ``in_think``
    state is thinking content and stays dropped.

    Note: returning the ``passthrough`` buffer is a fix relative to the
    earlier router-local implementation, which only flushed in ``detect`` and
    silently dropped a held partial-open suffix at end-of-stream.  The
    OpenAI streaming path picks up this fix transitively now that both
    routers share this implementation.

    All managed state keys are reset to their initial values before
    returning so the post-flush dict is consistent (an empty buffer
    implies the ``detect`` phase with no expected close), in case a
    caller reuses the state dict.
    """
    buf = state.get("buffer", "")
    phase = state.get("phase", "detect")
    state["buffer"] = ""
    state["phase"] = "detect"
    state["expected_close"] = ""
    if phase in ("detect", "passthrough"):
        return buf
    return ""


async def safe_ndjson_stream(
    source, format_chunk, format_error, log, log_prefix="streaming"
):
    """Wrap an async source with error handling and guaranteed cleanup.

    Args:
        source: Async iterator to consume (must support aclose()).
        format_chunk: Callable(item) -> ``str | list[str] | None`` for each
            yielded item.  Return ``None`` to skip an item; return a list to
            emit multiple NDJSON lines for a single source item (the
            terminal done chunk on the Ollama chat path uses this to flush a
            trailing thinking-detect buffer alongside the done marker).
        format_error: Callable(Exception) -> str for error formatting.
        log: Logger instance for error reporting.
        log_prefix: Prefix for error log messages.
    """
    try:
        async for item in source:
            # Note: format_chunk exceptions are also caught below.  This is
            # intentional — once streaming has started the HTTP status is 200,
            # so the best we can do is emit an error payload and log it.
            formatted = format_chunk(item)
            if formatted is None:
                continue
            if isinstance(formatted, list):
                for line in formatted:
                    # Defensive: drop ``None`` items even though the
                    # documented element type is ``str``.  Cheap to filter
                    # here so a future caller returning ``[None, "ok"]``
                    # doesn't propagate ``None`` into Starlette's writer.
                    if line is not None:
                        yield line
            else:
                yield formatted
    except Exception as exc:
        log.error("Error during %s: %s", log_prefix, exc, exc_info=True)
        try:
            yield format_error(exc)
        except Exception:
            log.error(
                "format_error raised during %s error handling",
                log_prefix,
                exc_info=True,
            )
    finally:
        # When called via aclose() (client disconnect), GeneratorExit is
        # swallowed after this cleanup completes — the intended behaviour.
        await source.aclose()


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
_has_prefill_callback: bool | None = None


class CancellableStream:
    """Async iterable wrapping a sync generator in a background thread.

    Provides cancellation via a threading.Event and drain_and_join() to wait
    for the background thread to finish (ensuring Metal operations complete
    before releasing locks).
    """

    def __init__(
        self, gen_factory: Callable[[threading.Event], Generator], is_vlm: bool = False
    ):
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
        # Queue items are one of: a StreamToken, an error dict, or _SENTINEL.
        self._queue: asyncio.Queue[StreamToken | dict[str, Any] | object] | None = None
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
                    self._queue.put(
                        {
                            _ERROR_KEY: str(exc),
                            "__exc_type__": type(exc).__name__,
                            "__traceback__": tb,
                        }
                    ),
                    self._loop,
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
                gen = None

            # Release the factory closure which captures model, tokenizer,
            # and prompt — these can be large and should not be pinned by
            # the stream object after the thread exits.
            self._gen_factory = None

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

    async def drain_and_join(self, timeout: float = 60.0):
        """Drain remaining items from the queue and wait for the thread to finish.

        IMPORTANT: This must wait for the thread to truly finish before returning,
        otherwise Metal operations from the dying thread can overlap with a new
        inference, causing '[_MTLCommandBuffer addCompletedHandler:] failed assertion'.

        Args:
            timeout: Maximum total seconds to wait across drain loop and thread join.
                     If exceeded, logs an error and returns (potential GPU resource leak).
        """
        self._cancel_event.set()
        deadline = time.monotonic() + timeout

        # If the stream already finished (sentinel posted), skip queue draining.
        # This avoids the 10s timeout when the sentinel was already consumed
        # by the async for loop (causing StopAsyncIteration).
        if not self._stream_done.is_set() and self._queue is not None:
            # Drain the queue until we see the sentinel.
            # Keep waiting as long as the background thread is alive and time remains.
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if self._thread is not None and self._thread.is_alive():
                        logger.warning(
                            "drain_and_join: drain loop timed out after %.1fs, "
                            "thread still alive — proceeding to join",
                            timeout,
                        )
                    else:
                        logger.debug(
                            "drain_and_join: drain timed out but thread already exited"
                        )
                    break
                try:
                    wait_time = min(10.0, remaining)
                    item = await asyncio.wait_for(self._queue.get(), timeout=wait_time)
                    if item is _SENTINEL:
                        break
                except asyncio.TimeoutError:
                    if self._thread is None or not self._thread.is_alive():
                        break
                    # Thread still running (e.g. long prefill) — keep waiting
                    logger.debug(
                        "drain_and_join: thread still alive, continuing to wait"
                    )
                    continue

        if self._thread is not None:
            remaining = deadline - time.monotonic()
            join_attempted = remaining > 0
            if join_attempted:
                try:
                    await asyncio.to_thread(self._thread.join, remaining)
                except Exception:
                    pass
            if self._thread.is_alive():
                if join_attempted:
                    logger.error(
                        "drain_and_join: thread still alive after %.1fs timeout — "
                        "potential GPU resource leak. The background inference thread "
                        "could not be stopped, which may cause Metal errors on the "
                        "next inference.",
                        timeout,
                    )
                else:
                    logger.error(
                        "drain_and_join: drain loop exhausted %.1fs budget, join skipped "
                        "— thread still alive, potential GPU resource leak.",
                        timeout,
                    )

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamToken:
        assert self._queue is not None
        item = await self._queue.get()
        if item is _SENTINEL:
            raise StopAsyncIteration
        if isinstance(item, dict) and _ERROR_KEY in item:
            exc_type = item.get("__exc_type__", "RuntimeError")
            tb = item.get("__traceback__", "")
            if tb:
                logger.error(
                    "Inference error (%s): %s\n%s", exc_type, item[_ERROR_KEY], tb
                )
            raise RuntimeError(f"{exc_type}: {item[_ERROR_KEY]}")
        return cast(StreamToken, item)


def _make_prefill_progress(
    cancel_event: threading.Event,
    memory_limit: int = 0,
    mx_module: Any = None,
) -> Callable[[float], bool]:
    """Create a prefill progress callback that checks cancellation and memory.

    Note: when the callback aborts prefill by returning False, mlx_lm stops
    early and the client receives a truncated (likely empty) 200 response.
    The bool return contract doesn't allow surfacing an error. The pre-flight
    check in inference.py handles the common case with a clean 503; this
    callback is a last-resort safety net for inaccurate estimates.

    Args:
        cancel_event: Threading event to signal cancellation.
        memory_limit: Metal memory limit in bytes. 0 disables memory checking.
        mx_module: The mlx.core module (injectable for testing).

    Returns:
        Callback that returns False to abort prefill.
    """
    # Resolve mx once at creation time, not per-callback invocation
    if mx_module is None and memory_limit > 0:
        import mlx.core as mx_module
    last_check_progress = -0.05  # trigger check on first callback invocation

    def _prefill_progress(progress: float) -> bool:
        nonlocal last_check_progress
        if cancel_event.is_set():
            return False
        # Check memory periodically (every ~5% progress)
        if memory_limit > 0 and progress - last_check_progress >= 0.05:
            last_check_progress = progress
            try:
                current = mx_module.get_active_memory() + mx_module.get_cache_memory()
                if current > memory_limit:
                    logger.warning(
                        "Aborting prefill at %.0f%%: Metal memory %.1f GB exceeds limit %.1f GB",
                        progress * 100,
                        current / 1024**3,
                        memory_limit / 1024**3,
                    )
                    return False
            except Exception:
                logger.debug(
                    "Prefill memory check failed at %.0f%%",
                    progress * 100,
                    exc_info=True,
                )
        return True

    return _prefill_progress


def async_mlx_stream(
    model: Any,
    tokenizer: Any,
    prompt: str | list[int],
    max_tokens: int = 512,
    is_vlm: bool = False,
    images: list[str] | None = None,
    memory_limit: int = 0,
    **kwargs: Any,
) -> CancellableStream:
    """Bridge sync mlx_lm/mlx_vlm stream_generate into an async iterable.

    Returns a CancellableStream (started and ready to iterate).
    """

    # Cache the signature check on the event loop thread (not the background
    # inference thread where gen_factory runs) to avoid any GIL ambiguity.
    if not is_vlm:
        global _has_prefill_callback
        if _has_prefill_callback is None:
            import inspect

            import mlx_lm

            _has_prefill_callback = (
                "prompt_progress_callback"
                in inspect.signature(mlx_lm.stream_generate).parameters
            )
    # VLMs excluded: mlx_vlm.stream_generate doesn't support prompt_progress_callback.
    # VLMs still get the pre-flight check in inference.py before streaming starts.
    use_prefill_callback = not is_vlm and _has_prefill_callback

    def gen_factory(cancel_event: threading.Event):
        from olmlx.engine.inference import _apply_seed

        _apply_seed(kwargs, consume=not is_vlm)

        if is_vlm:
            import mlx_vlm

            return mlx_vlm.stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                image=images,
                max_tokens=max_tokens,
                **kwargs,
            )
        else:
            import mlx_lm

            gen_kwargs = dict(prompt=prompt, max_tokens=max_tokens, **kwargs)
            gen_kwargs.pop("prompt_progress_callback", None)  # we control this below

            if use_prefill_callback:
                gen_kwargs["prompt_progress_callback"] = _make_prefill_progress(
                    cancel_event, memory_limit=memory_limit
                )

            return mlx_lm.stream_generate(model, tokenizer, **gen_kwargs)

    stream = CancellableStream(gen_factory, is_vlm=is_vlm)
    stream.start()
    return stream
