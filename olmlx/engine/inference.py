import asyncio
import collections.abc
import contextlib
import gc
import importlib
import json
import logging
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any

import mlx.core as mx

from olmlx.engine.model_manager import (
    CachedPromptState,
    LoadedModel,
    ModelManager,
    parse_keep_alive,
)
from olmlx.config import experimental, settings
from olmlx.utils import memory as memory_utils

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

try:
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
except ImportError:  # pragma: no cover
    make_sampler = None  # type: ignore[assignment]
    make_logits_processors = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning(
        "mlx-lm sample_utils unavailable (mlx-lm < 0.30.7?) — sampler/logits_processors disabled"
    )
from olmlx.engine.template_caps import TemplateCaps
from olmlx.utils.streaming import async_mlx_stream
from olmlx.utils.timing import Timer, TimingStats

logger = logging.getLogger(__name__)

# gpt-oss special tokens used by the streaming filter
_GPT_OSS_STRUCTURAL_TOKENS = frozenset(
    {
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|call|>",
        "<|return|>",
    }
)


class _GptOssChannelFilter:
    """Stateful filter for gpt-oss channel tokens.

    Call ``should_yield(text)`` for each token. Returns True if the token's text
    should be sent to the client. After the stream ends, call
    ``get_fallback_texts()`` — if non-empty, yield those as fallback (the model
    produced analysis but no final channel).

    This is a class (not an async generator) so the caller can iterate the raw
    stream for prompt-cache token accumulation while only yielding filtered text.
    """

    _INIT = "init"
    _AFTER_START = "after_start"
    _EXPECT_CHANNEL = "expect_channel"
    _IN_BLOCK = "in_block"
    _CONTENT = "content"

    def __init__(self):
        self._state = self._INIT
        self._channel = None
        self._saw_any_channel = False
        self._saw_final = False
        self._analysis_texts: list[str] = []

    def should_yield(self, text: str) -> bool:
        """Process one token's text and return whether it should be yielded."""
        if text == "<|start|>":
            self._state = self._AFTER_START
            self._saw_any_channel = True
            return False

        if text == "<|channel|>":
            self._state = self._EXPECT_CHANNEL
            self._saw_any_channel = True
            return False

        if self._state == self._AFTER_START:
            return False

        if self._state == self._EXPECT_CHANNEL:
            self._channel = text.strip()
            self._state = self._IN_BLOCK
            if self._channel == "final":
                self._saw_final = True
            return False

        if text == "<|message|>" and self._state == self._IN_BLOCK:
            self._state = self._CONTENT
            return False

        if text in ("<|end|>", "<|call|>", "<|return|>"):
            self._state = self._INIT
            self._channel = None
            return False

        if self._state == self._CONTENT and self._channel == "final":
            return True

        if (
            self._state == self._CONTENT
            and self._channel == "analysis"
            and not self._saw_final
        ):
            self._analysis_texts.append(text)
            return False

        if (
            self._state == self._INIT
            and not self._saw_any_channel
            and text not in _GPT_OSS_STRUCTURAL_TOKENS
        ):
            return True

        return False

    def get_fallback_texts(self) -> list[str]:
        """Return buffered analysis texts if no final channel was seen."""
        if not self._saw_final and self._analysis_texts:
            return self._analysis_texts
        return []


async def _gpt_oss_filter(token_stream):
    """Async generator wrapper for backward compatibility with tests."""
    filt = _GptOssChannelFilter()
    buffered = []
    async for token in token_stream:
        if filt.should_yield(token.text):
            yield token
        else:
            buffered.append(token)
    for text in filt.get_fallback_texts():
        # Find matching token from buffer
        for tok in buffered:
            if tok.text == text:
                yield tok
                buffered.remove(tok)
                break


# -- Experimental: Distributed inference coordinator --
# Only set when OLMLX_EXPERIMENTAL_DISTRIBUTED=true; see set_distributed_coordinator().
_distributed_coordinator = None
_distributed_coordinator_lock = threading.Lock()


def set_distributed_coordinator(coordinator):
    """Set the distributed coordinator for broadcasting inference to workers."""
    global _distributed_coordinator
    with _distributed_coordinator_lock:
        _distributed_coordinator = coordinator


def _maybe_broadcast_distributed(
    lm,
    prompt_tokens: list[int],
    prompt_text: str,
    max_tokens: int,
    gen_kwargs: dict,
) -> None:
    """Broadcast inference params to distributed workers if applicable."""
    with _distributed_coordinator_lock:
        coord = _distributed_coordinator
    if coord is not None and lm.is_distributed:
        coord.broadcast_inference(
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_text,
            max_tokens=max_tokens,
            gen_kwargs=gen_kwargs,
        )
        from olmlx.engine.distributed import distributed_barrier

        distributed_barrier()


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
_deferred_cleanup_lock: asyncio.Lock | None = None
# Tracks requests waiting for _inference_lock (not the _await_deferred_cleanup wait).
_queue_depth = 0


def _get_deferred_cleanup_lock() -> asyncio.Lock:
    """Lazily create the deferred cleanup lock in the current event loop (Bug #119).

    Module-level asyncio.Lock() binds to the loop at creation time, which
    breaks in tests that create fresh event loops.

    Safe: asyncio is single-threaded; no await between the ``is None`` check
    and the assignment, so no two coroutines can both observe ``None``
    simultaneously.
    """
    global _deferred_cleanup_lock
    if _deferred_cleanup_lock is None:
        _deferred_cleanup_lock = asyncio.Lock()
    return _deferred_cleanup_lock


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
_DEFERRED_WAIT_TIMEOUT = 30.0  # max wait for deferred cleanup before rejecting


async def _await_deferred_cleanup():
    """Wait for any in-progress deferred GPU cleanup to complete.

    Raises ServerBusyError if cleanup doesn't finish within _DEFERRED_WAIT_TIMEOUT.
    Uses asyncio.wait() to avoid Python 3.11 wait_for race conditions.
    Uses _deferred_cleanup_lock to prevent TOCTOU races on _deferred_cleanup_task (Bug #119).
    """
    async with _get_deferred_cleanup_lock():
        task = _deferred_cleanup_task
        if task is None or task.done():
            return
    # Wait outside the lock so _cleanup() can acquire it in its finally block
    # to set _deferred_cleanup_task = None.  Holding the lock here would deadlock.
    # Race safety: a concurrent _schedule_deferred_inference_cleanup cannot replace
    # _deferred_cleanup_task while we wait because _inference_lock is held by the
    # existing cleanup — no new inference (and thus no new cleanup) can be scheduled.
    logger.info("Waiting for deferred GPU cleanup to complete")
    done, _ = await asyncio.wait({task}, timeout=_DEFERRED_WAIT_TIMEOUT)
    if not done:
        raise ServerBusyError(
            f"Server busy: deferred GPU cleanup did not complete within {_DEFERRED_WAIT_TIMEOUT}s"
        )


async def _schedule_deferred_inference_cleanup(stream) -> None:
    """Schedule deferred GPU cleanup when the inference thread is stuck.

    Polls the thread until it exits, then syncs Metal and releases the
    inference lock.  The lock remains held until the thread finishes to
    prevent concurrent Metal command buffer access.

    If the thread doesn't exit within _DEFERRED_CLEANUP_TIMEOUT seconds,
    releases the lock anyway (risk of Metal crash on next inference, but
    better than permanent deadlock).

    Uses _deferred_cleanup_lock to prevent TOCTOU races on _deferred_cleanup_task (Bug #119).
    """
    global _deferred_cleanup_task

    async with _get_deferred_cleanup_lock():
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
                async with _get_deferred_cleanup_lock():
                    global _deferred_cleanup_task
                    _deferred_cleanup_task = None

        _deferred_cleanup_task = asyncio.create_task(_cleanup())


MEMORY_SAFETY_FACTOR = 1.3
"""Safety multiplier for KV cache memory estimates (Bug #125).

Metal alignment, intermediate buffers, and allocator overhead can cause actual
memory usage to exceed the raw 2-bytes-per-element calculation by 20-30%.
"""


def _estimate_kv_cache_bytes(model: Any, num_tokens: int) -> int:
    """Estimate KV cache memory for a given number of tokens.

    Formula: sum_over_attn_layers(2 * kv_heads_i * head_dim) * num_tokens * bytes_per_element * MEMORY_SAFETY_FACTOR

    For NAS models (e.g. nemotron-nas) that have per-layer variable attention
    (some layers are no-op with self_attn=None, and KV head counts vary per
    layer), we introspect model.model.layers to count only actual attention
    layers and read their n_kv_heads.  Falls back to args-based estimation
    when layer introspection isn't possible.
    """
    if num_tokens <= 0:
        return 0
    # mlx-lm text models: model.args
    # mlx-vlm vision-language models: model.language_model.args
    args = getattr(model, "args", None)
    if args is None:
        lang_model = getattr(model, "language_model", None)
        if lang_model is not None:
            args = getattr(lang_model, "args", None)
    if args is None:
        raise AttributeError(
            "Model has no 'args' attribute (checked model.args and model.language_model.args)"
        )
    num_heads = args.num_attention_heads
    head_dim = (
        args.head_dim if hasattr(args, "head_dim") else args.hidden_size // num_heads
    )
    bytes_per_element = 2  # float16/bfloat16

    # Try layer introspection for NAS/variable-attention models.
    # Track which sub-model owns the args so we introspect the right layers
    # (for VLMs, args came from model.language_model — introspect that, not
    # model.model which could be a vision encoder).
    lang_model_component = getattr(model, "language_model", None)
    args_owner = (
        lang_model_component
        if (lang_model_component is not None and getattr(model, "args", None) is None)
        else model
    )
    inner = getattr(args_owner, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if isinstance(layers, (list, tuple)) and len(layers) > 0:
        per_layer_kv_sum = 0
        for layer in layers:
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                continue  # no-op attention layer — no KV cache
            layer_kv_heads = getattr(self_attn, "n_kv_heads", None)
            if layer_kv_heads is None:
                # Standard model — fall back to args
                break
            per_layer_kv_sum += layer_kv_heads
        else:
            # All layers inspected successfully — but only trust the result
            # if we actually found some attention layers.  per_layer_kv_sum == 0
            # likely means the attention module uses a different attribute name
            # (e.g. "attention" instead of "self_attn"); fall through to the
            # args-based estimate in that case.
            if per_layer_kv_sum > 0:
                raw = 2 * per_layer_kv_sum * head_dim * num_tokens * bytes_per_element
                return int(raw * MEMORY_SAFETY_FACTOR)

    # Fallback: uniform estimate from args
    num_layers = args.num_hidden_layers
    num_kv_heads = getattr(args, "num_key_value_heads", num_heads)
    raw = num_layers * 2 * num_kv_heads * head_dim * num_tokens * bytes_per_element
    return int(raw * MEMORY_SAFETY_FACTOR)


def _tokenize_for_cache(tokenizer: Any, prompt_text: str) -> list[int]:
    """Tokenize prompt text matching stream_generate's tokenization logic.

    Must exactly replicate the BOS heuristic in mlx_lm.generate.stream_generate
    to avoid token sequence divergence (which would cause every request to be a
    cache miss).  stream_generate uses ``bos_token is None``, NOT ``not bos_token``.
    """
    bos = getattr(tokenizer, "bos_token", None)
    add_special = bos is None or not prompt_text.startswith(bos)
    return tokenizer.encode(prompt_text, add_special_tokens=add_special)


async def _acquire_inference_lock():
    """Acquire the inference lock with optional timeout from settings.

    Uses asyncio.wait() instead of asyncio.wait_for() to avoid a known
    Python 3.11 race where wait_for can deliver the lock and then cancel,
    leaving the lock permanently held with no owner.
    """
    timeout = settings.inference_queue_timeout
    if isinstance(timeout, (int, float)) and timeout > 0:
        acquire_task = asyncio.create_task(_inference_lock.acquire())
        try:
            done, _ = await asyncio.wait({acquire_task}, timeout=timeout)
        except BaseException:
            # Caller was cancelled (e.g. client disconnect, TaskGroup teardown).
            # Clean up the orphaned acquire task to prevent a lock leak.
            acquire_task.cancel()
            try:
                await acquire_task
                _inference_lock.release()
            except asyncio.CancelledError:
                pass
            raise
        if not done:
            acquire_task.cancel()
            # If acquire completed between wait() returning and cancel(),
            # we now own the lock — must release it before raising.
            try:
                await acquire_task
                _inference_lock.release()
            except asyncio.CancelledError:
                pass
            raise ServerBusyError(
                f"Server busy: inference queue timeout after {timeout}s"
            )
    else:
        await _inference_lock.acquire()


@contextlib.asynccontextmanager
async def _inference_locked():
    """Async context manager that acquires the inference lock with Metal sync on entry/exit."""
    global _queue_depth
    await _await_deferred_cleanup()
    _queue_depth += 1
    if _queue_depth > 1:
        logger.info("Request queued for inference lock (queue depth: %d)", _queue_depth)
    try:
        await _acquire_inference_lock()
    except BaseException:
        _queue_depth -= 1
        raise
    _queue_depth -= 1
    # Re-check after acquiring — a deferred cleanup task may have been
    # created between the pre-check and acquire (TOCTOU window).
    try:
        await _await_deferred_cleanup()
    except BaseException:
        _inference_lock.release()
        raise
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

    Bug #118: Python ``+=`` on int is not atomic. Concurrent async tasks can
    race on ``active_refs``. Use the model's ``_active_refs_lock`` to protect
    increments and decrements.
    """
    with lm._active_refs_lock:
        lm.active_refs += 1
    try:
        yield
    finally:
        with lm._active_refs_lock:
            lm.active_refs -= 1
        # Refresh expiry so the model doesn't expire immediately after inference
        ka = parse_keep_alive(settings.default_keep_alive)
        if ka is not None:
            lm.expires_at = time.time() + ka


def _build_generate_kwargs(options: dict | None, is_vlm: bool = False) -> dict:
    """Convert Ollama options dict to mlx_lm/mlx_vlm generate kwargs.

    For text models (mlx-lm ≥ 0.30.7), sampling params are folded into a
    ``sampler`` callable via ``make_sampler``, and penalty params into a
    ``logits_processors`` list via ``make_logits_processors``.

    For VLMs (mlx-vlm), params are passed directly as before.
    """
    if not options:
        return {}
    kwargs = {}

    if is_vlm:
        # mlx-vlm still accepts direct keyword arguments
        vlm_mappings = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "seed": "seed",
            "num_predict": "max_tokens",
            "repeat_penalty": "repetition_penalty",
            "repeat_last_n": "repetition_context_size",
            "min_p": "min_p",
        }
        for ollama_key, mlx_key in vlm_mappings.items():
            if ollama_key in options:
                kwargs[mlx_key] = options[ollama_key]
    else:
        # mlx-lm ≥ 0.30.7: sampling via make_sampler / make_logits_processors
        sampler_args = {}
        sampling_map = {
            "temperature": "temp",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
        }
        for ollama_key, sampler_key in sampling_map.items():
            if ollama_key in options:
                sampler_args[sampler_key] = options[ollama_key]
        # Only build sampler when temperature is explicitly set — make_sampler
        # defaults temp=0.0 (greedy), which makes top_k/top_p/min_p irrelevant.
        if sampler_args and "temp" in sampler_args:
            if make_sampler is None:
                raise RuntimeError("mlx-lm is not installed; cannot build sampler")
            kwargs["sampler"] = make_sampler(**sampler_args)
        elif sampler_args:
            logger.warning(
                "top_k/top_p/min_p provided without temperature; no sampler "
                "will be built and these params will have no effect"
            )

        # Collect penalty params — only build processors when repeat_penalty
        # is present; repeat_last_n alone is a no-op (no penalty to apply).
        if "repeat_penalty" in options:
            penalty_args = {"repetition_penalty": options["repeat_penalty"]}
            if "repeat_last_n" in options:
                penalty_args["repetition_context_size"] = options["repeat_last_n"]
            if make_logits_processors is None:
                raise RuntimeError(
                    "mlx-lm is not installed; cannot build logits processors"
                )
            kwargs["logits_processors"] = make_logits_processors(**penalty_args)
        elif "repeat_last_n" in options:
            logger.warning(
                "repeat_last_n without repeat_penalty has no effect; ignored"
            )

        if "num_predict" in options:
            kwargs["max_tokens"] = options["num_predict"]

        # Forward seed so _apply_seed can consume it before generation
        if "seed" in options:
            kwargs["seed"] = options["seed"]

        if "stop" in options:
            logger.warning("stop sequences not supported by mlx-lm >= 0.30.7; ignored")

        for penalty_key in ("frequency_penalty", "presence_penalty"):
            if penalty_key in options:
                logger.warning(
                    "%s not supported by mlx-lm >= 0.30.7; ignored", penalty_key
                )

    return kwargs


def _apply_seed(kwargs: dict, *, consume: bool = True) -> None:
    """Read ``seed`` from *kwargs* and set the MLX RNG state.

    Must be called from the inference thread, not the event loop.

    Args:
        kwargs: Generate kwargs dict (may contain ``seed``).
        consume: If True, pop the key so it is not forwarded to the
                 underlying generate call (required for mlx-lm which
                 does not accept a ``seed`` kwarg).  If False, the key
                 is left in place (VLMs forward it to mlx-vlm).
    """
    seed = kwargs.pop("seed", None) if consume else kwargs.get("seed", None)
    if seed is not None:
        mx.random.seed(seed)


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


def _make_turboquant_prompt_cache(model: Any, bits: int, is_vlm: bool = False) -> list:
    """Create a TurboQuant-compressed prompt cache for the model."""
    from olmlx.engine.turboquant_cache import make_turboquant_cache

    cache_model = _get_model_for_cache(model, is_vlm)
    return make_turboquant_cache(cache_model, bits=bits)


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
    apply_chat_template: bool = False,
    system: str | None = None,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a text completion, streaming or not.

    When *apply_chat_template* is True the raw prompt is wrapped in chat
    messages and run through the model's chat template before generation.
    If *system* is provided, it becomes a ``{"role": "system"}`` message.
    This is needed for chat-only models (e.g. Nemotron-H) that require the
    template framing to produce meaningful output.
    """
    stats = TimingStats()

    with Timer() as load_timer:
        lm = await manager.ensure_loaded(model_name, keep_alive)
    stats.load_duration = load_timer.duration_ns

    if apply_chat_template and not lm.is_vlm:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            prompt = _apply_chat_template_text(
                lm.text_tokenizer,
                messages,
                caps=lm.template_caps,
                enable_thinking=False,
            )
            logger.info(
                "Applied chat template for /api/generate (prompt length: %d chars)",
                len(prompt),
            )
            logger.debug("Templated prompt: %s", prompt[:500])
        except RuntimeError as exc:
            logger.warning(
                "Chat template failed for %s, falling back to raw prompt: %s",
                model_name,
                exc,
                exc_info=True,
            )
            if system:
                prompt = f"{system}\n\n{prompt}"
    elif apply_chat_template and lm.is_vlm:
        if system:
            prompt = f"{system}\n\n{prompt}"
            logger.warning(
                "apply_chat_template not supported for VLM %s; "
                "system prepended as plain text",
                model_name,
            )

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
    global _queue_depth
    await _await_deferred_cleanup()
    _queue_depth += 1
    if _queue_depth > 1:
        logger.info(
            "Streaming request queued for inference lock (queue depth: %d)",
            _queue_depth,
        )
    try:
        await _acquire_inference_lock()
    except BaseException:
        _queue_depth -= 1
        raise
    _queue_depth -= 1
    # Re-check after acquiring — a deferred cleanup task may have been
    # created between the pre-check and acquire (TOCTOU window).
    try:
        await _await_deferred_cleanup()
    except BaseException:
        _inference_lock.release()
        raise
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
    # Save original string prompt before cache setup may replace it with token IDs.
    # prompt is always str at entry; cache setup may later reassign it to list[int].
    original_prompt = prompt
    cache_setup_done = False
    try:
        # Memory pressure check — invalidate cache to prevent Metal OOM
        memory_too_high = (
            use_prompt_cache
            and prompt_tokens is not None
            and make_prompt_cache is not None
            and memory_utils.is_memory_pressure_high(settings.memory_limit_fraction)
        )
        if memory_too_high:
            logger.warning(
                "Memory pressure high, evicting prompt caches to free GPU memory"
            )
            await lm.prompt_cache_store.async_evict_all_to_disk()
            gc.collect()
            mx.clear_cache()
            _safe_sync()  # Bug #120: ensure freed buffers are reclaimed
            memory_too_high = memory_utils.is_memory_pressure_high(
                settings.memory_limit_fraction
            )

        # Cache setup — must happen after lock to prevent concurrent cache corruption
        if (
            use_prompt_cache
            and not memory_too_high
            and prompt_tokens is not None
            and make_prompt_cache is not None
        ):
            cached = await lm.prompt_cache_store.async_get(cache_id)
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
                # Bug #123: Remove cache from store before mutation so the
                # store's copy is not corrupted if the client disconnects
                # mid-stream.  The cache will be re-stored on successful
                # completion; on disconnect the finally block is a no-op.
                working_cache = cached.cache
                lm.prompt_cache_store.remove(cache_id)
                # Trim cache to suffix_start so it aligns with where we resume
                trim_amount = len(cached.tokens) - suffix_start
                if trim_amount > 0:
                    trim_prompt_cache(working_cache, trim_amount)

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
                gen_kwargs["prompt_cache"] = working_cache
                if lm.is_vlm:
                    # VLM stream_generate expects a string prompt; pass
                    # pre-tokenized tokens via input_ids to bypass prepare_inputs.
                    gen_kwargs["input_ids"] = mx.array([suffix_tokens])
                else:
                    prompt = suffix_tokens
            else:
                # No usable prefix — free old cache and create fresh
                lm.prompt_cache_store.remove(cache_id)
                kv_quant = experimental.kv_cache_quant
                if kv_quant is not None:
                    bits = int(kv_quant.split(":")[1])
                    new_cache = _make_turboquant_prompt_cache(
                        lm.model, bits, is_vlm=lm.is_vlm
                    )
                else:
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

            cache_setup_done = True

            # Release the cached reference — on cache miss the old
            # CachedPromptState was removed from the store but this local
            # variable still pins its KV cache tensors in GPU memory.
            del cached

        # Pre-flight KV cache memory check — estimate how much GPU memory
        # the KV cache will need and reject if it would exceed the limit.
        # This prevents uncatchable Metal OOM crashes during prefill.
        # MUST run before yielding cache_info, because that yield starts
        # the HTTP response — after which we can't return a clean 503.
        if cache_creation_tokens > 0:
            num_prefill_tokens = cache_creation_tokens
        elif isinstance(prompt, list):
            num_prefill_tokens = len(prompt)
        elif isinstance(prompt, str) and not lm.is_vlm:
            # Non-cached text path — tokenize to get a count.
            # VLMs excluded: text_tokenizer.encode() misses image patch
            # tokens, giving a systematic undercount.
            try:
                num_prefill_tokens = len(lm.text_tokenizer.encode(prompt))
            except Exception:
                num_prefill_tokens = 0
        else:
            num_prefill_tokens = 0
        # Compute memory_limit before try so the streaming callback always
        # has the correct limit, even when the pre-flight estimate throws.
        total_physical = memory_utils.get_system_memory_bytes()
        memory_limit = (
            int(total_physical * settings.memory_limit_fraction)
            if total_physical > 0
            else 0
        )
        if total_physical > 0 and num_prefill_tokens > 0:
            try:
                kv_bytes = _estimate_kv_cache_bytes(
                    lm.model, num_prefill_tokens + max_tokens
                )
                current_metal = memory_utils.get_metal_memory()
                if current_metal + kv_bytes > memory_limit:
                    # Drop cached KV tensors so eviction + gc can reclaim them
                    working = gen_kwargs.pop("prompt_cache", None)
                    had_cache = working is not None
                    gen_kwargs.pop(
                        "input_ids", None
                    )  # VLMs: force re-tokenize from string prompt
                    # Bug #123 removes cache from store before mutation.
                    # Re-add it temporarily so evict_all_to_disk() can persist it.
                    # Use full_prompt_tokens[:suffix_start] to match the trimmed
                    # KV state — working_cache was trimmed to suffix_start tokens.
                    if had_cache and full_prompt_tokens is not None:
                        await lm.prompt_cache_store.async_set(
                            cache_id,
                            CachedPromptState(
                                tokens=list(full_prompt_tokens[:suffix_start]),
                                cache=working,
                            ),
                        )
                    # Drop Python references so gc.collect() + mx.clear_cache()
                    # can actually reclaim GPU memory.
                    working_cache = None  # noqa: F841
                    working = None
                    await lm.prompt_cache_store.async_evict_all_to_disk()
                    gc.collect()
                    mx.clear_cache()
                    # After evicting the prompt cache, mlx_lm will prefill
                    # the full prompt from scratch — re-estimate for all tokens
                    # and restore prompt from suffix_tokens to the full sequence.
                    estimate_tokens = num_prefill_tokens
                    if had_cache and cache_read_tokens > 0:
                        estimate_tokens = cache_read_tokens + num_prefill_tokens
                        if full_prompt_tokens is not None and not lm.is_vlm:
                            prompt = full_prompt_tokens
                    # Re-estimate after eviction for the full generation window
                    estimate_total = estimate_tokens + max_tokens
                    kv_bytes = _estimate_kv_cache_bytes(lm.model, estimate_total)
                    # Sync Metal to ensure freed buffers are reclaimed before re-reading
                    _safe_sync()
                    current_metal = memory_utils.get_metal_memory()
                    if current_metal + kv_bytes > memory_limit:
                        available_gb = max(
                            0.0, (memory_limit - current_metal) / 1024**3
                        )
                        raise MemoryError(
                            f"KV cache for {estimate_total} tokens estimated at "
                            f"{kv_bytes / 1024**3:.1f} GB, but only "
                            f"{available_gb:.1f} GB available "
                            f"— prompt too long, reduce context or use a smaller model"
                        )
            except MemoryError:
                raise
            except Exception:
                logger.warning(
                    "KV cache pre-flight check skipped — OOM protection inactive",
                    exc_info=True,
                )

        # Yield cache stats after the pre-flight check so routers can
        # use them.  This starts the HTTP response — no 503 after this.
        if cache_setup_done:
            yield {
                "cache_info": True,
                "cache_read_tokens": cache_read_tokens,
                "cache_creation_tokens": cache_creation_tokens,
            }

        # Broadcast to distributed workers before starting generation.
        # Workers need prompt_text (str) because mlx_lm.stream_generate
        # expects a string prompt, not token IDs. Always use original_prompt
        # (the original string before cache manipulation may have replaced it
        # with token IDs) to avoid tokenizer round-trip mismatches.
        # Strip prompt_cache and input_ids — these are local MLX objects
        # that cannot be serialized to JSON for the sideband protocol.
        if lm.is_distributed:
            tokens = (
                prompt_tokens
                if prompt_tokens is not None
                else _tokenize_for_cache(lm.text_tokenizer, original_prompt)
            )
            broadcast_kwargs = {
                k: v
                for k, v in gen_kwargs.items()
                if k not in ("prompt_cache", "input_ids")
            }
            _maybe_broadcast_distributed(
                lm, tokens, original_prompt, max_tokens, broadcast_kwargs
            )

        if lm.is_speculative:
            from olmlx.engine.flash.speculative_stream import async_speculative_stream

            # Speculative decoding uses greedy argmax; sampling params are not supported.
            _sampling_keys = {
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "seed",
            }
            # Greedy-compatible defaults: temperature=0, top_p=1.0, top_k=0
            _greedy_defaults = {"temperature": 0, "top_p": 1.0, "top_k": 0}
            _dropped = {
                k
                for k in gen_kwargs
                if k in _sampling_keys
                and gen_kwargs[k] is not None
                and gen_kwargs[k] != _greedy_defaults.get(k)
            }
            if _dropped:
                logger.warning(
                    "Speculative decoding uses greedy argmax; ignoring sampling parameters: %s",
                    ", ".join(sorted(_dropped)),
                )
            stream = async_speculative_stream(
                lm.speculative_decoder,
                lm.tokenizer,
                prompt,
                max_tokens=max_tokens,
            )
        else:
            stream = async_mlx_stream(
                lm.model,
                lm.tokenizer,
                prompt,
                max_tokens=max_tokens,
                is_vlm=lm.is_vlm,
                images=images,
                memory_limit=memory_limit,
                **gen_kwargs,
            )

        # Channel filter for gpt-oss models (decides which tokens to yield as text)
        channel_filter = (
            _GptOssChannelFilter() if lm.template_caps.has_channel_format else None
        )

        with _inference_ref(lm), Timer() as total_timer:
            with Timer() as eval_timer:
                async for token in stream:
                    # Always accumulate for prompt cache (raw stream, not filtered)
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
                    # Yield text only if the filter allows it (or no filter)
                    if channel_filter is None or channel_filter.should_yield(
                        token.text
                    ):
                        yield {"text": token.text, "done": False}

            # Fallback: yield analysis content if no final channel was produced
            if channel_filter is not None:
                for text in channel_filter.get_fallback_texts():
                    yield {"text": text, "done": False}

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
                    evicted = await lm.prompt_cache_store.async_set(
                        cache_id,
                        CachedPromptState(tokens=stored_tokens, cache=prompt_cache),
                    )
                    if evicted is not None:
                        del evicted
                        if memory_utils.is_memory_pressure_high(
                            settings.memory_limit_fraction
                        ):
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
                evicted = await lm.prompt_cache_store.async_set(
                    cache_id,
                    CachedPromptState(
                        tokens=stored_tokens,
                        cache=prompt_cache,
                    ),
                )
                if evicted is not None:
                    del evicted
                    if memory_utils.is_memory_pressure_high(
                        settings.memory_limit_fraction
                    ):
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
        # Release GPU-backed references from gen_kwargs so they can be
        # garbage-collected.  prompt_cache is either stored in the cache
        # store (successful path) or should be freed; input_ids is a
        # temporary MLX array only needed during stream setup.
        gen_kwargs.pop("prompt_cache", None)
        gen_kwargs.pop("input_ids", None)
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
            await _schedule_deferred_inference_cleanup(stream)
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
    has_tools: bool = False,
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
                has_tools=has_tools,
            )


async def _full_completion_inner(
    lm: LoadedModel,
    prompt: str,
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
    has_tools: bool = False,
) -> dict:
    def _generate_sync():
        """Run generate + synchronize in the same thread so GPU work completes
        before the thread returns to the pool."""
        # Broadcast inside the thread so rank 0 and workers enter MLX
        # computation at the same time (avoids all_sum timeout).
        # Must happen before _apply_seed which pops seed from gen_kwargs.
        if lm.is_distributed:
            tokens = _tokenize_for_cache(lm.text_tokenizer, prompt)
            _maybe_broadcast_distributed(lm, tokens, prompt, max_tokens, gen_kwargs)

        _apply_seed(gen_kwargs, consume=not lm.is_vlm)

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
        elif lm.is_speculative:
            import threading

            from olmlx.engine.flash.speculative_stream import (
                speculative_stream_generate,
            )

            if isinstance(prompt, str):
                prompt_tokens = lm.text_tokenizer.encode(prompt)
            else:
                prompt_tokens = prompt

            cancel = threading.Event()
            eos_token_id = getattr(lm.text_tokenizer, "eos_token_id", None)
            result = None
            text_parts = []
            for response in speculative_stream_generate(
                lm.speculative_decoder,
                prompt_tokens,
                max_tokens=max_tokens,
                cancel_event=cancel,
                eos_token_id=eos_token_id,
                tokenizer=lm.text_tokenizer,
            ):
                text_parts.append(response.text)
                result = response
            if result is not None:
                result = (result, "".join(text_parts))
            # Speculative decoding does not use mlx_lm's generation_stream,
            # so sync the default stream only.
            mx.synchronize()
            return result
        else:
            import mlx_lm

            # Use stream_generate to capture token counts (generate() discards them).
            # Accumulate text segments since each yield is incremental.
            result = None
            text_parts = []
            for response in mlx_lm.stream_generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                **gen_kwargs,
            ):
                text_parts.append(response.text)
                result = response
            # Store full text on the result for downstream extraction
            if result is not None:
                result = (result, "".join(text_parts))
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

    # Unpack (GenerationResult, full_text) tuple from stream_generate path
    full_text = None
    if isinstance(result, tuple):
        gen_result, full_text = result
        result = gen_result

    # Extract token counts from GenerationResult (stream_generate) or string
    if hasattr(result, "prompt_tokens"):
        stats.prompt_eval_count = result.prompt_tokens
    if hasattr(result, "generation_tokens"):
        stats.eval_count = result.generation_tokens

    eval_secs = stats.eval_duration / 1e9 if stats.eval_duration else 0
    gen_tps = stats.eval_count / eval_secs if eval_secs > 0 else 0
    prompt_tps = stats.prompt_eval_count / eval_secs if eval_secs > 0 else 0
    total_secs = stats.total_duration / 1e9 if stats.total_duration else 0
    logger.info(
        "Generation complete: %d prompt tokens (%.1f tok/s), %d tokens generated (%.1f tok/s), %.2fs total",
        stats.prompt_eval_count,
        prompt_tps,
        stats.eval_count,
        gen_tps,
        total_secs,
    )

    # Extract text: prefer accumulated full_text, fall back to result
    if full_text is not None:
        text = full_text
    elif result is None:
        text = ""
    elif hasattr(result, "text"):
        text = result.text
    elif isinstance(result, str):
        text = result
    else:
        text = str(result)

    # Strip gpt-oss channel tokens for non-streaming path
    if lm.template_caps.has_channel_format and "<|channel|>" in text:
        from olmlx.engine.tool_parser import _parse_gpt_oss_channels

        parsed = _parse_gpt_oss_channels(text, has_tools=has_tools)
        if parsed is not None:
            _, visible, _ = parsed
            text = visible

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

    # Prompt caching: streaming only, when enabled.
    # Disabled in distributed mode because rank 0 processes only suffix tokens
    # on cache hits while workers process the full prompt, causing all_sum
    # call count mismatch and deadlock.
    use_prompt_cache = (
        settings.prompt_cache
        and stream
        and make_prompt_cache is not None
        and not lm.is_distributed
    )
    prompt_tokens = None
    if use_prompt_cache:
        prompt_tokens = _tokenize_for_cache(lm.text_tokenizer, prompt)
        # Memory-only peek for debug logging; the authoritative lookup happens
        # inside _stream_completion under the inference lock.
        cached_state = lm.prompt_cache_store.peek(cache_id)
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
        return await _full_completion(
            lm, prompt, mt, gen_kwargs, stats, images, has_tools=bool(tools)
        )


async def generate_embeddings(
    manager: ModelManager,
    model_name: str,
    texts: list[str],
    keep_alive: str | None = None,
) -> list[list[float]]:
    """Generate embeddings using the model's hidden states or embed_tokens layer."""
    lm = await manager.ensure_loaded(model_name, keep_alive)

    async with _inference_locked():
        with _inference_ref(lm):
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
                    raise ValueError(
                        f"Unexpected embedding tensor shape: {hidden.shape}"
                    )

                embeddings.append(embedding.tolist())

            # Defensive sync — _inference_locked exit also syncs, but this
            # ensures embedding tensors are fully evaluated before .tolist().
            mx.synchronize()
            return embeddings
