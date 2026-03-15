import asyncio
import contextlib
import gc
import importlib
import json
import logging
import os
import time
import collections.abc
from collections.abc import AsyncGenerator
from typing import Any

import mlx.core as mx

from olmlx.engine.model_manager import (
    CachedPromptState,
    LoadedModel,
    ModelManager,
    parse_keep_alive,
)
from olmlx.config import settings

try:
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    from mlx_lm.utils import common_prefix_len as _find_common_prefix
except ImportError:  # pragma: no cover
    make_prompt_cache = None  # type: ignore[assignment]
    trim_prompt_cache = None  # type: ignore[assignment]
    _find_common_prefix = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning(
        "mlx-lm prompt cache imports unavailable — prompt caching disabled"
    )
from olmlx.engine.template_caps import TemplateCaps
from olmlx.utils.streaming import async_mlx_stream
from olmlx.utils.timing import Timer, TimingStats

logger = logging.getLogger(__name__)


# Resolve generation streams at module load time to avoid repeated
# importlib.import_module() calls in the hot path (_safe_sync).
def _resolve_generation_streams() -> list[Any]:
    streams = []
    for mod_name in ("mlx_lm.generate", "mlx_vlm.generate"):
        try:
            mod = importlib.import_module(mod_name)
            streams.append(mod.generation_stream)
        except (ImportError, AttributeError):
            pass
    return streams


_generation_streams = _resolve_generation_streams()

# Metal does not support concurrent command buffer submission across any
# models — they all share the same Metal device and command queue.  A per-model
# lock would still allow interleaved GPU work from different models, risking
# crashes or corruption.  A single global lock is an intentional trade-off:
# we sacrifice parallelism for stability on Apple Silicon.
_inference_lock = asyncio.Lock()
_deferred_cleanup_task: asyncio.Task | None = None


def _safe_sync():
    """Synchronize Metal GPU state, suppressing and logging any errors.

    Also syncs the generation stream (mlx_lm/mlx_vlm use a separate stream
    from the default stream). This is critical to prevent 'command encoder
    already encoding' errors when the background inference thread is still
    writing to the GPU while a new request tries to start.
    """
    try:
        mx.synchronize()
    except Exception:
        logger.debug("mx.synchronize() failed", exc_info=True)

    for stream in _generation_streams:
        try:
            mx.synchronize(stream)
        except Exception:
            logger.debug("generation_stream sync failed", exc_info=True)


class ServerBusyError(RuntimeError):
    """Raised when the server is recovering from a previous inference (deferred GPU cleanup)."""

    pass


_DEFERRED_CLEANUP_TIMEOUT = 600  # 10 minutes max wait for stuck thread


def _schedule_deferred_inference_cleanup(stream) -> None:
    """Schedule deferred GPU cleanup when the inference thread is stuck.

    Polls the thread until it exits, then syncs Metal and releases the
    inference lock.  The lock remains held until the thread finishes to
    prevent concurrent Metal command buffer access.

    If the thread doesn't exit within _DEFERRED_CLEANUP_TIMEOUT seconds,
    releases the lock anyway (risk of Metal crash on next inference, but
    better than permanent deadlock).
    """
    global _deferred_cleanup_task

    if _deferred_cleanup_task is not None and not _deferred_cleanup_task.done():
        logger.error(
            "Deferred inference cleanup already in progress — "
            "this should not happen while the inference lock is held"
        )
        return  # do not create a second task; the existing one will release the lock

    async def _cleanup():
        thread = stream._thread
        deadline = time.monotonic() + _DEFERRED_CLEANUP_TIMEOUT
        try:
            while thread is not None and thread.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "Deferred inference cleanup: thread still alive after %ds — "
                        "releasing lock anyway (risk of Metal crash on next inference)",
                        _DEFERRED_CLEANUP_TIMEOUT,
                    )
                    break
                try:
                    wait = min(30, remaining)
                    await asyncio.to_thread(thread.join, wait)
                except BaseException as exc:
                    logger.warning(
                        "Deferred inference cleanup: poll loop aborted (%s) — "
                        "releasing lock (thread may still be alive)",
                        type(exc).__name__,
                    )
                    break  # finally will release the lock
            else:
                logger.info("Deferred inference cleanup: thread exited cleanly")
        finally:
            if thread is None or not thread.is_alive():
                _safe_sync()
            # Note: on timeout/abort with thread still alive, releasing the
            # lock risks a Metal crash on the next inference (the stuck thread
            # may still be issuing GPU commands).  Python can't kill CPU-bound
            # threads, so this is the "least bad" option vs permanent deadlock.
            _inference_lock.release()
            logger.info("Deferred inference cleanup: lock released")
            global _deferred_cleanup_task
            _deferred_cleanup_task = None

    _deferred_cleanup_task = asyncio.create_task(_cleanup())


# Fraction of memory_limit_fraction at which we shed the prompt cache to
# free Metal memory before hitting the hard model-load rejection limit.
_MEMORY_PRESSURE_THRESHOLD = 0.9

try:
    _TOTAL_PHYSICAL_MEMORY = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
except (OSError, ValueError):
    _TOTAL_PHYSICAL_MEMORY = 0


def _is_memory_pressure_high() -> bool:
    """Check if Metal memory is approaching the safety limit."""
    if _TOTAL_PHYSICAL_MEMORY == 0:
        return False
    try:
        mem = mx.get_active_memory() + mx.get_cache_memory()
        limit = int(_TOTAL_PHYSICAL_MEMORY * settings.memory_limit_fraction)
        return mem > int(limit * _MEMORY_PRESSURE_THRESHOLD)
    except Exception:
        return False


def _tokenize_for_cache(tokenizer: Any, prompt_text: str) -> list[int]:
    """Tokenize prompt text matching stream_generate's tokenization logic.

    Must exactly replicate the BOS heuristic in mlx_lm.generate.stream_generate
    to avoid token sequence divergence (which would cause every request to be a
    cache miss).  stream_generate uses ``bos_token is None``, NOT ``not bos_token``.
    """
    bos = getattr(tokenizer, "bos_token", None)
    add_special = bos is None or not prompt_text.startswith(bos)
    return tokenizer.encode(prompt_text, add_special_tokens=add_special)


@contextlib.asynccontextmanager
async def _inference_locked():
    """Async context manager that acquires the inference lock with Metal sync on entry/exit."""
    if _deferred_cleanup_task is not None and not _deferred_cleanup_task.done():
        raise ServerBusyError(
            "Server busy: recovering from previous inference — "
            "deferred GPU cleanup in progress"
        )
    await _inference_lock.acquire()
    # Re-check after acquiring — a deferred cleanup task may have been
    # created between the pre-check and acquire (TOCTOU window).
    if _deferred_cleanup_task is not None and not _deferred_cleanup_task.done():
        _inference_lock.release()
        raise ServerBusyError(
            "Server busy: recovering from previous inference — "
            "deferred GPU cleanup in progress"
        )
    # Sync the default Metal stream so any pending GPU work from the previous
    # inference completes before we start a new one.
    _safe_sync()
    try:
        yield
    finally:
        # Sync again on exit to ensure this inference's GPU work is fully
        # complete before releasing the lock to the next caller.
        _safe_sync()
        _inference_lock.release()


@contextlib.contextmanager
def _inference_ref(lm: LoadedModel):
    """Track active inference on a model to prevent expiry during use.

    Note: there is a small window between ``ensure_loaded()`` (which returns
    the LoadedModel) and the point where ``_inference_ref`` increments
    ``active_refs``.  During that window the expiry checker could remove the
    model from ``_loaded``.  This is **safe**: the caller already holds a
    Python reference to the LoadedModel object, so the model and tokenizer
    remain alive in memory.  The only side-effect is that the next request
    would re-load the model into ``_loaded``.
    """
    lm.active_refs += 1
    try:
        yield
    finally:
        lm.active_refs -= 1
        # Refresh expiry so the model doesn't expire immediately after inference
        ka = parse_keep_alive(settings.default_keep_alive)
        if ka is not None:
            lm.expires_at = time.time() + ka


def _build_generate_kwargs(options: dict | None, is_vlm: bool = False) -> dict:
    """Convert Ollama options dict to mlx_lm/mlx_vlm generate kwargs."""
    if not options:
        return {}
    kwargs = {}
    # mlx-lm uses "temp", mlx-vlm uses "temperature"
    temp_key = "temperature" if is_vlm else "temp"
    mappings = {
        "temperature": temp_key,
        "top_p": "top_p",
        "top_k": "top_k",
        "seed": "seed",
        "num_predict": "max_tokens",
        "repeat_penalty": "repetition_penalty",
        "repeat_last_n": "repetition_context_size",
        "min_p": "min_p",
    }
    for ollama_key, mlx_key in mappings.items():
        if ollama_key in options:
            kwargs[mlx_key] = options[ollama_key]
    # stop is only supported by mlx-lm
    if not is_vlm and "stop" in options:
        kwargs["stop"] = options["stop"]
    # frequency_penalty / presence_penalty — pass through if present
    for penalty_key in ("frequency_penalty", "presence_penalty"):
        if penalty_key in options and options[penalty_key]:
            kwargs[penalty_key] = options[penalty_key]
    return kwargs


def _inject_tools_into_system(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Inject tool descriptions into the system message when the template doesn't support tools natively."""
    tool_desc_parts = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        tool_desc_parts.append(
            f"- {name}: {desc}\n  Parameters: {json.dumps(params, indent=2)}"
        )
    tool_block = (
        "You have access to the following tools. To call a tool, output a JSON object "
        'with "name" and "arguments" keys.\n\n'
        "Available tools:\n" + "\n".join(tool_desc_parts)
    )

    messages = list(messages)  # shallow copy
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            **messages[0],
            "content": messages[0]["content"] + "\n\n" + tool_block,
        }
    else:
        messages.insert(0, {"role": "system", "content": tool_block})
    return messages


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None = None,
    caps: TemplateCaps | None = None,
    *,
    tokenize: bool = False,
    enable_thinking: bool | None = None,
) -> Any:
    """Core chat template application.

    Uses TemplateCaps to decide which kwargs to pass, avoiding blind try/except.
    Returns str when tokenize=False, token list/dict when tokenize=True.
    """
    if caps is None:
        caps = TemplateCaps()

    kwargs: dict[str, Any] = {"tokenize": tokenize, "add_generation_prompt": True}

    if tools and caps.supports_tools:
        kwargs["tools"] = tools
    elif tools and not caps.supports_tools:
        logger.info(
            "Template lacks tool support, injecting tool descriptions into system message"
        )
        messages = _inject_tools_into_system(messages, tools)

    if caps.supports_enable_thinking:
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        elif tools:
            kwargs["enable_thinking"] = (
                False  # backward compat for non-Anthropic callers
            )
        else:
            kwargs["enable_thinking"] = True

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as exc:
        # If tools kwarg caused the error, retry without it (injecting instead)
        if tools and "tools" in kwargs:
            logger.warning(
                "apply_chat_template failed with tools kwarg (%s), retrying with injection",
                exc,
            )
            del kwargs["tools"]
            # Keep enable_thinking — it's independent of the tools kwarg failure
            messages = _inject_tools_into_system(messages, tools)
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except Exception as exc2:
                raise RuntimeError(
                    f"Chat template failed even without tools: {exc2}"
                ) from exc2
        raise RuntimeError(f"Chat template failed: {exc}") from exc


def _apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None = None,
    caps: TemplateCaps | None = None,
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Apply chat template for text-only models (mlx-lm), returning prompt text."""
    return _apply_chat_template(
        tokenizer,
        messages,
        tools,
        caps,
        tokenize=False,
        enable_thinking=enable_thinking,
    )


def _apply_chat_template_vlm(
    processor: Any,
    model: Any,
    messages: list[dict],
    images: list[str] | None = None,
) -> str:
    """Apply chat template for vision-language models (mlx-vlm)."""
    import mlx_vlm

    config = model.config if hasattr(model, "config") else {}
    num_images = len(images) if images else 0
    # Pass the full message list so the model gets proper conversation context
    return mlx_vlm.apply_chat_template(
        processor, config, messages, num_images=num_images
    )


def _get_model_for_cache(model: Any, is_vlm: bool) -> Any:
    """Get the language model for cache creation.

    For text models (mlx-lm), returns the model directly.
    For VLM models (mlx-vlm), returns model.language_model.
    """
    if is_vlm:
        return getattr(model, "language_model", model)
    return model


def _extract_images(messages: list[dict]) -> list[str] | None:
    """Extract image URLs/paths from message content."""
    images = []
    for msg in messages:
        if msg.get("images"):
            images.extend(msg["images"])
    return images if images else None


def count_chat_tokens(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None = None,
    caps: TemplateCaps | None = None,
    *,
    enable_thinking: bool | None = None,
) -> int:
    """Count input tokens by applying the chat template with tokenize=True.

    No GPU inference needed — CPU-only tokenization.  Uses
    add_generation_prompt=True so the count includes the assistant-turn
    opener tokens, matching what the model actually receives at inference.
    """
    result = _apply_chat_template(
        tokenizer, messages, tools, caps, tokenize=True, enable_thinking=enable_thinking
    )

    # Handle varied return types from apply_chat_template.
    # BatchEncoding (transformers) extends UserDict, not dict, so use Mapping.
    if isinstance(result, collections.abc.Mapping):
        tokens = result.get("input_ids")
        if tokens is None:
            raise TypeError(
                f"apply_chat_template returned dict without 'input_ids': keys={list(result.keys())}"
            )
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
    elif isinstance(result, list) and result and isinstance(result[0], list):
        tokens = result[0]
    elif isinstance(result, list):
        tokens = result
    else:
        raise TypeError(
            f"Unexpected return type from apply_chat_template: {type(result)}"
        )

    return len(tokens)


async def generate_completion(
    manager: ModelManager,
    model_name: str,
    prompt: str,
    options: dict | None = None,
    stream: bool = True,
    keep_alive: str | None = None,
    max_tokens: int = 512,
    images: list[str] | None = None,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a text completion, streaming or not."""
    stats = TimingStats()

    with Timer() as load_timer:
        lm = await manager.ensure_loaded(model_name, keep_alive)
    stats.load_duration = load_timer.duration_ns

    gen_kwargs = _build_generate_kwargs(options, is_vlm=lm.is_vlm)
    mt = gen_kwargs.pop("max_tokens", max_tokens)

    if stream:
        return _stream_completion(lm, prompt, mt, gen_kwargs, stats, images)
    else:
        return await _full_completion(lm, prompt, mt, gen_kwargs, stats, images)


async def _stream_completion(
    lm: LoadedModel,
    prompt: str | list[int],
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
    *,
    use_prompt_cache: bool = False,
    prompt_tokens: list[int] | None = None,
    cache_id: str = "",
) -> AsyncGenerator[dict, None]:
    # Use explicit acquire/release instead of `async with` to prevent
    # CancelledError from releasing the lock before cleanup completes.
    if _deferred_cleanup_task is not None and not _deferred_cleanup_task.done():
        raise ServerBusyError(
            "Server busy: recovering from previous inference — "
            "deferred GPU cleanup in progress"
        )
    await _inference_lock.acquire()
    # Re-check after acquiring — a deferred cleanup task may have been
    # created between the pre-check and acquire (TOCTOU window).
    if _deferred_cleanup_task is not None and not _deferred_cleanup_task.done():
        _inference_lock.release()
        raise ServerBusyError(
            "Server busy: recovering from previous inference — "
            "deferred GPU cleanup in progress"
        )
    # Sync default stream before starting — same purpose as _inference_locked entry.
    _safe_sync()

    # Everything after lock acquisition must be in try/finally so the lock is
    # always released — even if the generator is closed at a yield point
    # (e.g. client disconnect during cache_info yield).
    stream = None
    generation_complete = False
    generated_tokens: list[int] = []
    cache_read_tokens = 0
    cache_creation_tokens = 0
    full_prompt_tokens: list[int] | None = None
    try:
        # Memory pressure check — invalidate cache to prevent Metal OOM
        memory_too_high = (
            use_prompt_cache
            and prompt_tokens is not None
            and make_prompt_cache is not None
            and _is_memory_pressure_high()
        )
        if memory_too_high:
            logger.warning(
                "Memory pressure high, invalidating prompt cache to prevent OOM"
            )
            lm.prompt_cache_store.clear()
            gc.collect()
            mx.clear_cache()
            memory_too_high = _is_memory_pressure_high()

        # Cache setup — must happen after lock to prevent concurrent cache corruption
        if (
            use_prompt_cache
            and not memory_too_high
            and prompt_tokens is not None
            and make_prompt_cache is not None
        ):
            cached = lm.prompt_cache_store.get(cache_id)
            logger.debug(
                "Cache lookup: cached=%s, new prompt=%d tokens",
                (
                    f"{len(cached.tokens)} tokens (first 5: {cached.tokens[:5]})"
                    if cached
                    else "none"
                ),
                len(prompt_tokens),
            )

            prefix_len = (
                _find_common_prefix(prompt_tokens, cached.tokens)
                if cached is not None
                else 0
            )
            logger.debug(
                "Common prefix length: %d / %d prompt tokens",
                prefix_len,
                len(prompt_tokens),
            )

            # Set before mutation so finally guard can clean up on error
            full_prompt_tokens = prompt_tokens

            # stream_generate requires at least 1 token, so we must back up
            # by one position on exact-match.  If that would mean suffix_start=0
            # (single-token prompt), the cache hit is useless — trimming the
            # entire cache to re-process the lone token is a cold start.  Treat
            # it as a miss and create a fresh cache instead.
            suffix_start = (
                min(prefix_len, len(prompt_tokens) - 1) if prompt_tokens else 0
            )

            if prefix_len > 0 and suffix_start > 0:
                # Trim cache to suffix_start so it aligns with where we resume
                trim_amount = len(cached.tokens) - suffix_start
                if trim_amount > 0:
                    trim_prompt_cache(cached.cache, trim_amount)

                suffix_tokens = prompt_tokens[suffix_start:]

                # Report suffix_start as cache_read — the number of tokens
                # whose KV entries are actually reused from cache.  On exact
                # match, suffix_start = prefix_len - 1 because stream_generate
                # re-processes the token at suffix_start (its KV is not reused).
                cache_read_tokens = suffix_start
                cache_creation_tokens = len(suffix_tokens)
                logger.info(
                    "Prompt cache hit: %d prefix tokens reused, %d new tokens to process (was %d total)",
                    prefix_len,
                    len(suffix_tokens),
                    len(prompt_tokens),
                )
                gen_kwargs["prompt_cache"] = cached.cache
                if lm.is_vlm:
                    # VLM stream_generate expects a string prompt; pass
                    # pre-tokenized tokens via input_ids to bypass prepare_inputs.
                    gen_kwargs["input_ids"] = mx.array([suffix_tokens])
                else:
                    prompt = suffix_tokens
            else:
                # No usable prefix — free old cache and create fresh
                lm.prompt_cache_store.remove(cache_id)
                cache_model = _get_model_for_cache(lm.model, lm.is_vlm)
                new_cache = make_prompt_cache(cache_model)
                gen_kwargs["prompt_cache"] = new_cache
                cache_creation_tokens = len(prompt_tokens)
                logger.info(
                    "Prompt cache %s: creating fresh cache for %d tokens",
                    "miss" if cached is not None else "init",
                    len(prompt_tokens),
                )
                if lm.is_vlm:
                    gen_kwargs["input_ids"] = mx.array([prompt_tokens])
                else:
                    prompt = prompt_tokens

            # Yield cache stats as first chunk so routers can use them
            yield {
                "cache_info": True,
                "cache_read_tokens": cache_read_tokens,
                "cache_creation_tokens": cache_creation_tokens,
            }

        stream = async_mlx_stream(
            lm.model,
            lm.tokenizer,
            prompt,
            max_tokens=max_tokens,
            is_vlm=lm.is_vlm,
            images=images,
            **gen_kwargs,
        )

        with _inference_ref(lm), Timer() as total_timer:
            with Timer() as eval_timer:
                async for token in stream:
                    yield {"text": token.text, "done": False}
                    stats.eval_count = token.generation_tokens
                    stats.prompt_eval_count = token.prompt_tokens
                    if token.token is not None:
                        generated_tokens.append(token.token)
                    else:
                        logger.debug(
                            "Skipping token with None ID at generation step %d "
                            "(cache token sequence will be incomplete)",
                            token.generation_tokens,
                        )

            stats.eval_duration = eval_timer.duration_ns
            prompt_tps = getattr(token, "prompt_tps", 0) or 0
            gen_tps = getattr(token, "generation_tps", 0) or 0

        stats.total_duration = total_timer.duration_ns
        logger.info(
            "Generation complete: %d prompt tokens (%.1f tok/s), %d tokens generated (%.1f tok/s), %.2fs total",
            stats.prompt_eval_count,
            prompt_tps,
            stats.eval_count,
            gen_tps,
            total_timer.duration_ns / 1e9,
        )
        generation_complete = True

        # Store cache state after successful generation
        prompt_cache = gen_kwargs.get("prompt_cache")
        if prompt_cache is not None and full_prompt_tokens is not None:
            stored_tokens = list(full_prompt_tokens) + generated_tokens
            # The KV cache has an entry for every generation step, including
            # steps where the token ID was None (skipped in generated_tokens).
            # Use stats.eval_count for the real generation depth.
            actual_total = len(full_prompt_tokens) + stats.eval_count
            max_cache_tokens = settings.prompt_cache_max_tokens
            if max_cache_tokens is not None and actual_total > max_cache_tokens:
                trim_amount = actual_total - max_cache_tokens
                try:
                    trim_prompt_cache(prompt_cache, trim_amount)
                    if stats.eval_count != len(generated_tokens):
                        # None-ID tokens present: can't map generated_tokens
                        # to KV cache positions. Trim KV cache down to prompt
                        # boundary so depth == len(stored_tokens).
                        extra = max_cache_tokens - len(full_prompt_tokens)
                        if extra > 0:
                            trim_prompt_cache(prompt_cache, extra)
                        stored_tokens = list(full_prompt_tokens)[:max_cache_tokens]
                    else:
                        stored_tokens = stored_tokens[:max_cache_tokens]
                    evicted = lm.prompt_cache_store.set(
                        cache_id,
                        CachedPromptState(tokens=stored_tokens, cache=prompt_cache),
                    )
                    if evicted is not None:
                        del evicted
                        if _is_memory_pressure_high():
                            gc.collect()
                            mx.clear_cache()
                    logger.info(
                        "Cache trimmed: %d → %d tokens (limit %d)",
                        actual_total,
                        len(stored_tokens),
                        max_cache_tokens,
                    )
                except Exception:
                    lm.prompt_cache_store.remove(cache_id)
                    prompt_cache = None
                    gc.collect()
                    mx.clear_cache()
                    logger.warning(
                        "Cache trim failed; invalidating cache",
                        exc_info=True,
                    )
            else:
                evicted = lm.prompt_cache_store.set(
                    cache_id,
                    CachedPromptState(
                        tokens=stored_tokens,
                        cache=prompt_cache,
                    ),
                )
                if evicted is not None:
                    del evicted
                    if _is_memory_pressure_high():
                        gc.collect()
                        mx.clear_cache()
                logger.debug(
                    "Cache stored: %d tokens (%d prompt + %d generated)",
                    len(stored_tokens),
                    len(full_prompt_tokens),
                    len(generated_tokens),
                )

        yield {
            "text": "",
            "done": True,
            "stats": stats,
        }
    finally:
        # Invalidate cache on incomplete generation to avoid inconsistent state
        if not generation_complete and full_prompt_tokens is not None:
            logger.debug("Cache invalidated: generation did not complete")
            lm.prompt_cache_store.remove(cache_id)
        # We MUST wait for the Metal thread to finish before releasing
        # _inference_lock, otherwise the next inference will hit concurrent
        # Metal command buffer access.
        # stream may be None if generator was closed during cache setup.
        thread_alive = False
        if stream is not None:
            _drain_task = asyncio.ensure_future(stream.drain_and_join())
            try:
                await asyncio.shield(_drain_task)
            except (asyncio.CancelledError, Exception):
                # Shield was interrupted — cancel the inner drain task to
                # avoid a leaked coroutine that logs misleading warnings.
                _drain_task.cancel()
                # Ensure cancel_event is set even if _drain_task was cancelled
                # before drain_and_join() could set it (which would leave the
                # prefill callback returning True indefinitely).
                stream.cancel()
                # Fallback join — give the thread a chance to exit before
                # going to deferred cleanup.
                if stream._thread is not None and stream._thread.is_alive():
                    try:
                        await asyncio.to_thread(stream._thread.join, 10)
                    except (asyncio.CancelledError, Exception):
                        pass
            thread_alive = stream._thread is not None and stream._thread.is_alive()

        if thread_alive:
            # Thread is stuck (likely in long prefill).  Defer cleanup to
            # avoid calling _safe_sync() while the thread is still using
            # the GPU — that causes an uncatchable Metal assertion crash.
            logger.warning(
                "Inference thread still alive after cleanup attempts — "
                "deferring Metal sync and lock release until thread exits"
            )
            _schedule_deferred_inference_cleanup(stream)
        else:
            # Normal path — thread exited, safe to sync and release.
            _safe_sync()
            _inference_lock.release()


async def _full_completion(
    lm: LoadedModel,
    prompt: str,
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
) -> dict:
    async with _inference_locked():
        with _inference_ref(lm):
            return await _full_completion_inner(
                lm,
                prompt,
                max_tokens,
                gen_kwargs,
                stats,
                images,
            )


async def _full_completion_inner(
    lm: LoadedModel,
    prompt: str,
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
) -> dict:
    def _generate_sync():
        """Run generate + synchronize in the same thread so GPU work completes
        before the thread returns to the pool."""
        if lm.is_vlm:
            import mlx_vlm

            result = mlx_vlm.generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                image=images,
                max_tokens=max_tokens,
                **gen_kwargs,
            )
            from mlx_vlm.generate import generation_stream
        else:
            import mlx_lm

            result = mlx_lm.generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                **gen_kwargs,
            )
            from mlx_lm.generate import generation_stream
        # Sync the generation_stream specifically — mlx_lm/mlx_vlm run GPU
        # work on this module-level stream, not the default stream.  Without
        # this, generate() may return before GPU work is actually done.
        mx.synchronize(generation_stream)
        return result

    with Timer() as total_timer:
        with Timer() as eval_timer:
            result = await asyncio.to_thread(_generate_sync)

    stats.eval_duration = eval_timer.duration_ns
    stats.total_duration = total_timer.duration_ns

    eval_secs = stats.eval_duration / 1e9 if stats.eval_duration else 0
    gen_tps = stats.eval_count / eval_secs if eval_secs > 0 else 0
    total_secs = stats.total_duration / 1e9 if stats.total_duration else 0
    logger.info(
        "Generation complete: %d prompt tokens, %d tokens generated (%.1f tok/s), %.2fs total",
        stats.prompt_eval_count,
        stats.eval_count,
        gen_tps,
        total_secs,
    )

    # mlx_vlm.generate returns GenerationResult dataclass
    if hasattr(result, "text"):
        text = result.text
    elif isinstance(result, str):
        text = result
    else:
        text = str(result)
    return {"text": text, "done": True, "stats": stats}


async def generate_chat(
    manager: ModelManager,
    model_name: str,
    messages: list[dict],
    options: dict | None = None,
    tools: list[dict] | None = None,
    stream: bool = True,
    keep_alive: str | None = None,
    max_tokens: int = 512,
    cache_id: str = "",
    enable_thinking: bool | None = None,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a chat completion."""
    stats = TimingStats()

    with Timer() as load_timer:
        lm = await manager.ensure_loaded(model_name, keep_alive)
    stats.load_duration = load_timer.duration_ns

    images = _extract_images(messages)

    if lm.is_vlm and not tools:
        # VLM template doesn't support enable_thinking. When the user
        # explicitly sets it, there are no images, and the text template
        # supports it, fall back to the text template path so thinking
        # can be controlled. Images require the VLM template.
        # Note: the text template prompt is tokenized by lm.tokenizer
        # (VLM processor) during generation. This works for Qwen models
        # where text and VLM tokenizers share the same vocabulary.
        if (
            enable_thinking is not None
            and not images
            and lm.template_caps
            and lm.template_caps.supports_enable_thinking
        ):
            prompt = _apply_chat_template_text(
                lm.text_tokenizer,
                messages,
                tools,
                caps=lm.template_caps,
                enable_thinking=enable_thinking,
            )
        else:
            if enable_thinking is not None:
                logger.warning(
                    "enable_thinking=%s ignored for VLM model (not supported by mlx-vlm template)",
                    enable_thinking,
                )
            prompt = _apply_chat_template_vlm(lm.tokenizer, lm.model, messages, images)
    else:
        # Use text template path when tools are needed, even for VLM-loaded models,
        # because _apply_chat_template_vlm doesn't support tool definitions.
        prompt = _apply_chat_template_text(
            lm.text_tokenizer,
            messages,
            tools,
            caps=lm.template_caps,
            enable_thinking=enable_thinking,
        )
        if tools:
            logger.info("Chat prompt with %d tools", len(tools))
        logger.debug("Prompt (first 1000 chars): %s", prompt[:1000])

    gen_kwargs = _build_generate_kwargs(options, is_vlm=lm.is_vlm)
    mt = gen_kwargs.pop("max_tokens", max_tokens)

    # Prompt caching: streaming only, when enabled
    use_prompt_cache = (
        settings.prompt_cache and stream and make_prompt_cache is not None
    )
    prompt_tokens = None
    if use_prompt_cache:
        prompt_tokens = _tokenize_for_cache(lm.text_tokenizer, prompt)
        cached_state = lm.prompt_cache_store.get(cache_id)
        logger.debug(
            "Prompt cache enabled: %d prompt tokens, existing cache=%s",
            len(prompt_tokens),
            f"{len(cached_state.tokens)} tokens" if cached_state else "none",
        )
    else:
        logger.debug(
            "Prompt cache disabled: setting=%s stream=%s make_prompt_cache=%s",
            settings.prompt_cache,
            stream,
            make_prompt_cache is not None,
        )

    if stream:
        return _stream_completion(
            lm,
            prompt,
            mt,
            gen_kwargs,
            stats,
            images,
            use_prompt_cache=use_prompt_cache,
            prompt_tokens=prompt_tokens,
            cache_id=cache_id,
        )
    else:
        return await _full_completion(lm, prompt, mt, gen_kwargs, stats, images)


async def generate_embeddings(
    manager: ModelManager,
    model_name: str,
    texts: list[str],
    keep_alive: str | None = None,
) -> list[list[float]]:
    """Generate embeddings using the model's hidden states or embed_tokens layer."""
    lm = await manager.ensure_loaded(model_name, keep_alive)

    async with _inference_locked():
        embeddings = []

        tokenizer = lm.text_tokenizer

        # Check if model has a static embedding layer we can use directly
        embed_layer = None
        model_inner = getattr(lm.model, "model", lm.model)
        if hasattr(model_inner, "embed_tokens"):
            embed_layer = model_inner.embed_tokens

        for text in texts:
            tokens = tokenizer.encode(text)
            input_ids = mx.array([tokens])

            if embed_layer is not None:
                # Use static token embeddings — no forward pass needed
                hidden = embed_layer(input_ids)
            else:
                outputs = lm.model(input_ids)
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    hidden = outputs.hidden_states[-1]
                elif hasattr(outputs, "last_hidden_state"):
                    hidden = outputs.last_hidden_state
                else:
                    hidden = outputs

            # Robust shape handling
            if hidden.ndim == 3:
                # (batch, seq, dim) — mean-pool over sequence
                embedding = mx.mean(hidden[0], axis=0)
            elif hidden.ndim == 2:
                # (seq, dim) — mean-pool over sequence
                embedding = mx.mean(hidden, axis=0)
            elif hidden.ndim == 1:
                embedding = hidden
            else:
                raise ValueError(f"Unexpected embedding tensor shape: {hidden.shape}")

            embeddings.append(embedding.tolist())

        # Defensive sync — _inference_locked exit also syncs, but this
        # ensures embedding tensors are fully evaluated before .tolist().
        mx.synchronize()
        return embeddings
