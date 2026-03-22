from __future__ import annotations

import asyncio
import gc
import importlib
import json
import logging
import re
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from olmlx.config import settings
from olmlx.engine.registry import ModelRegistry
from olmlx.utils import memory as memory_utils
from olmlx.engine.template_caps import TemplateCaps, detect_caps

if TYPE_CHECKING:
    from olmlx.models.store import ModelStore

logger = logging.getLogger(__name__)

try:
    from mlx_lm.models.cache import (
        load_prompt_cache,
        save_prompt_cache,
    )
except ImportError:  # pragma: no cover
    save_prompt_cache = None  # type: ignore[assignment]
    load_prompt_cache = None  # type: ignore[assignment]

# Exceptions from mlx-lm.load() that indicate the model simply isn't
# compatible with mlx-lm and should be retried with mlx-vlm.  Errors like
# ImportError, MemoryError, and RuntimeError indicate real problems that
# should propagate immediately.
_FALLBACK_EXCEPTIONS = (
    ValueError,
    KeyError,
    FileNotFoundError,
    OSError,
    json.JSONDecodeError,
)


class ModelLoadTimeoutError(TimeoutError):
    """Raised when model loading exceeds OLMLX_MODEL_LOAD_TIMEOUT."""


@dataclass
class CachedPromptState:
    """KV cache state from a previous generation, for prompt cache reuse."""

    tokens: list[int]  # Full sequence: prompt + generated tokens
    cache: list[Any]  # Per-layer KV cache objects (mutated in-place by generate_step)


class PromptCacheStore:
    """LRU store for per-agent KV caches with optional disk offload.

    When disk_path and model_name are provided, evicted caches are saved to
    disk instead of deleted.  On cache miss, the disk is checked before
    returning None.
    """

    def __init__(
        self,
        max_slots: int,
        disk_path: Path | None = None,
        model_name: str = "",
        disk_max_bytes: int | None = None,
    ) -> None:
        self._max_slots = max_slots
        self._entries: OrderedDict[str, CachedPromptState] = OrderedDict()
        self._disk_path = disk_path
        self._model_name = model_name.replace("/", "--")
        self._disk_max_bytes = disk_max_bytes

    @property
    def _disk_enabled(self) -> bool:
        return (
            self._disk_path is not None
            and self._model_name != ""
            and save_prompt_cache is not None
            and load_prompt_cache is not None
        )

    def _disk_dir(self) -> Path:
        """Return the model-specific disk cache directory."""
        assert self._disk_path is not None
        return self._disk_path / self._model_name

    def _disk_file_path(self, cache_id: str) -> Path:
        """Return the disk file path for a given cache_id."""
        safe_name = re.sub(r"[^\w\-.]", "_", cache_id) or "_default"
        return self._disk_dir() / f"{safe_name}.safetensors"

    def _save_to_disk(self, cache_id: str, state: CachedPromptState) -> None:
        """Save a cache entry to disk."""
        if not self._disk_enabled:
            return
        try:
            disk_dir = self._disk_dir()
            disk_dir.mkdir(parents=True, exist_ok=True)
            file_path = self._disk_file_path(cache_id)
            metadata = {"tokens": json.dumps(state.tokens)}
            save_prompt_cache(str(file_path), state.cache, metadata)
            logger.info(
                "Saved evicted cache '%s' to disk (%s)",
                cache_id,
                file_path,
            )
            self._cleanup_disk()
        except Exception:
            logger.warning(
                "Failed to save cache '%s' to disk",
                cache_id,
                exc_info=True,
            )

    def _load_from_disk(self, cache_id: str) -> CachedPromptState | None:
        """Try to load a cache entry from disk. Returns None if not found."""
        if not self._disk_enabled:
            return None
        file_path = self._disk_file_path(cache_id)
        if not file_path.exists():
            return None
        try:
            cache, metadata = load_prompt_cache(str(file_path), return_metadata=True)
            tokens = json.loads(metadata.get("tokens", "[]"))
            state = CachedPromptState(tokens=tokens, cache=cache)
            # Store in memory via set() to respect max_slots capacity.
            # Delete the evicted entry to release GPU buffers promptly.
            evicted = self.set(cache_id, state)
            if evicted is not None:
                del evicted
            # Remove disk file only after set() succeeded
            file_path.unlink(missing_ok=True)
            logger.info(
                "Restored cache '%s' from disk (%d tokens)",
                cache_id,
                len(tokens),
            )
            return state
        except Exception:
            logger.warning(
                "Failed to load cache '%s' from disk",
                cache_id,
                exc_info=True,
            )
            # Remove corrupt file
            file_path.unlink(missing_ok=True)
            return None

    def _cleanup_disk(self) -> None:
        """Remove oldest disk cache files if total size exceeds the limit."""
        if self._disk_max_bytes is None or self._disk_path is None:
            return
        disk_dir = self._disk_dir()
        if not disk_dir.exists():
            return
        # Single stat pass: collect (path, size, mtime) to avoid double-stat
        file_info = []
        for f in disk_dir.glob("*.safetensors"):
            st = f.stat()
            file_info.append((f, st.st_size, st.st_mtime))
        file_info.sort(key=lambda x: x[2])  # sort by mtime
        total = sum(size for _, size, _ in file_info)
        while total > self._disk_max_bytes and file_info:
            path, size, _ = file_info.pop(0)
            total -= size
            path.unlink(missing_ok=True)
            logger.info("Disk cache cleanup: removed %s", path)

    def get(self, cache_id: str) -> CachedPromptState | None:
        """Get a cache entry, promoting it to MRU.

        Checks memory first, then disk.  Returns None if not found in either.
        """
        state = self._entries.get(cache_id)
        if state is not None:
            self._entries.move_to_end(cache_id)
            return state
        # Memory miss — try disk
        return self._load_from_disk(cache_id)

    def set(self, cache_id: str, state: CachedPromptState) -> CachedPromptState | None:
        """Set a cache entry, evicting LRU if at capacity.

        Returns the displaced CachedPromptState when its GPU resources need
        cleanup (different .cache object), or None when no cleanup is needed.
        Evicted entries are saved to disk if disk offload is enabled.
        """
        if cache_id in self._entries:
            self._entries.move_to_end(cache_id)
            old = self._entries[cache_id]
            self._entries[cache_id] = state
            return old if old.cache is not state.cache else None
        evicted: CachedPromptState | None = None
        evicted_id: str | None = None
        if len(self._entries) >= self._max_slots:
            evicted_id, evicted = self._entries.popitem(last=False)
        self._entries[cache_id] = state
        # Save evicted entry to disk
        if evicted is not None and evicted_id is not None:
            self._save_to_disk(evicted_id, evicted)
        return evicted

    def remove(self, cache_id: str) -> None:
        """Remove a specific cache entry from memory and disk."""
        self._entries.pop(cache_id, None)
        if self._disk_enabled:
            self._disk_file_path(cache_id).unlink(missing_ok=True)

    def evict_all_to_disk(self) -> None:
        """Save all in-memory entries to disk, then clear memory.

        Used during memory pressure to free GPU memory while preserving
        cache state on disk for later restoration.
        """
        if self._disk_enabled:
            for cache_id, state in list(self._entries.items()):
                self._save_to_disk(cache_id, state)
        self._entries.clear()

    def clear(self) -> None:
        """Remove all cache entries and disk cache files."""
        self._entries.clear()
        if self._disk_path is not None and self._model_name:
            disk_dir = self._disk_dir()
            if disk_dir.exists():
                shutil.rmtree(disk_dir, ignore_errors=True)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class LoadedModel:
    name: str
    hf_path: str
    model: Any
    tokenizer: Any  # tokenizer (mlx-lm) or processor (mlx-vlm)
    is_vlm: bool = False
    is_distributed: bool = False
    is_flash: bool = False
    template_caps: TemplateCaps = field(default_factory=TemplateCaps)
    loaded_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    size_bytes: int = 0
    active_refs: int = 0
    prompt_cache_store: PromptCacheStore = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.prompt_cache_store is None:
            disk_path = (
                Path(settings.prompt_cache_disk_path).expanduser()
                if settings.prompt_cache_disk
                else None
            )
            disk_max_bytes = (
                int(settings.prompt_cache_disk_max_gb * 1024**3)
                if settings.prompt_cache_disk
                else None
            )
            self.prompt_cache_store = PromptCacheStore(
                max_slots=settings.prompt_cache_max_slots,
                disk_path=disk_path,
                model_name=self.name,
                disk_max_bytes=disk_max_bytes,
            )

    @property
    def text_tokenizer(self) -> Any:
        """Return the underlying text tokenizer, unwrapping VLM processor if needed.

        mlx-vlm's load() returns a processor whose .tokenizer attribute is
        the actual HuggingFace tokenizer with the chat template.
        """
        tok = self.tokenizer
        if self.is_vlm and hasattr(tok, "tokenizer"):
            return tok.tokenizer
        return tok


def parse_keep_alive(value: str | int) -> float | None:
    """Parse keep_alive to seconds. Returns None for never-expire (-1)."""
    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return float(value)
    value = str(value).strip()
    if value == "-1":
        return None
    if value == "0":
        return 0.0
    match = re.match(r"^(\d+)(s|m|h)$", value)
    if not match:
        logger.warning("Invalid keep_alive format: %r, defaulting to 5m", value)
        return 300.0  # default 5m
    num, unit = int(match.group(1)), match.group(2)
    multipliers = {"s": 1, "m": 60, "h": 3600}
    return float(num * multipliers[unit])


class ModelManager:
    """Manages loading/unloading of MLX models with LRU eviction."""

    def __init__(
        self,
        registry: ModelRegistry,
        store: ModelStore | None = None,
        distributed_group: Any = None,
        distributed_strategy: str = "tensor",
        distributed_layer_counts: list[int] | None = None,
    ):
        self.registry = registry
        self.store = store
        self._distributed_group = distributed_group
        self._distributed_strategy = distributed_strategy
        self._distributed_layer_counts = distributed_layer_counts
        self._loaded: dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()
        self._expiry_task: asyncio.Task | None = None
        self._pending_cleanups: dict[str, asyncio.Task] = {}
        self._pending_load_tasks: dict[str, asyncio.Task] = {}

    def start_expiry_checker(self):
        self._expiry_task = asyncio.create_task(self._check_expiry_loop())

    async def stop(self):
        if self._expiry_task:
            self._expiry_task.cancel()
            try:
                await self._expiry_task
            except asyncio.CancelledError:
                pass
        for task in self._pending_cleanups.values():
            task.cancel()
        if self._pending_cleanups:
            await asyncio.gather(
                *self._pending_cleanups.values(), return_exceptions=True
            )
        self._pending_cleanups.clear()
        # Drain orphaned load tasks so their exceptions are retrieved.
        # The underlying threads can't be interrupted (Python limitation)
        # so use a bounded timeout to avoid blocking shutdown indefinitely.
        if self._pending_load_tasks:
            drained = True
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *self._pending_load_tasks.values(), return_exceptions=True
                    ),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                drained = False
                logger.warning(
                    "Timed out waiting for %d orphaned load thread(s) to finish; "
                    "they will be abandoned on process exit",
                    len(self._pending_load_tasks),
                )
            # Only flush when all threads finished — if the drain timed out,
            # threads are still allocating and mx.clear_cache() is unsafe.
            if drained:
                gc.collect()
                mx.clear_cache()
        self._pending_load_tasks.clear()
        self._loaded.clear()

    def _resolve_keep_alive(self, keep_alive: str | None) -> float | None:
        """Parse keep_alive, falling back to the global default."""
        return parse_keep_alive(
            keep_alive if keep_alive is not None else settings.default_keep_alive
        )

    async def ensure_loaded(
        self, name: str, keep_alive: str | None = None
    ) -> LoadedModel:
        """Ensure a model is loaded and return it."""
        normalized = self.registry.normalize_name(name)

        while True:
            # If a previous load timed out and the background thread is still
            # running, wait for its deferred cleanup to finish BEFORE acquiring
            # the lock — the cleanup may take minutes (waiting for a background
            # download thread), and holding the lock would block all model ops.
            cleanup = self._pending_cleanups.get(normalized)
            if cleanup is not None:
                logger.info(
                    "Waiting for deferred cleanup of '%s' before retry",
                    normalized,
                )
                try:
                    await cleanup
                except BaseException:
                    logger.warning(
                        "Deferred cleanup for '%s' failed; proceeding with retry",
                        normalized,
                        exc_info=True,
                    )

            async with self._lock:
                # A cleanup may have been scheduled while we were waiting
                # for the lock (another caller timed out).  Release the lock
                # and loop back to await it outside.
                if normalized in self._pending_cleanups:
                    continue

                if normalized in self._loaded:
                    lm = self._loaded[normalized]
                    # Refresh expiry
                    ka = self._resolve_keep_alive(keep_alive)
                    if ka is not None:
                        lm.expires_at = time.time() + ka
                    else:
                        lm.expires_at = None
                    return lm

                # Evict LRU if at capacity (skip models with active inference)
                while len(self._loaded) >= settings.max_loaded_models:
                    evictable = {
                        k: v for k, v in self._loaded.items() if v.active_refs == 0
                    }
                    if not evictable:
                        raise RuntimeError(
                            "All loaded models are in use, cannot evict to load a new model"
                        )
                    oldest_name = min(evictable, key=lambda k: evictable[k].loaded_at)
                    logger.info("Evicting model %s", oldest_name)
                    del self._loaded[oldest_name]

                # Flush Metal allocator cache so that buffers from evicted models
                # don't inflate the mem_before measurement below.  Skip when
                # any deferred cleanup is pending — a different model's
                # background thread may still be allocating Metal memory.
                if not self._pending_cleanups:
                    gc.collect()
                    mx.clear_cache()

                hf_path = self.registry.resolve(name)
                if hf_path is None:
                    raise ValueError(
                        f"Model '{name}' not found. "
                        f"Add it to {settings.models_config} or use a HuggingFace path like 'mlx-community/Qwen2.5-3B-Instruct-4bit'"
                    )

                # Auto-register direct HF paths so future requests find them
                if "/" in name:
                    self.registry.add_mapping(name, hf_path)

                logger.info("Loading model %s from %s", normalized, hf_path)
                mem_before = memory_utils.get_metal_memory()

                # Initialize before try so the except handler can always
                # clean up, whether _load_model or the post-load check fails.
                model = tokenizer = None
                load_task = lm = None
                try:
                    coro = asyncio.to_thread(self._load_model_and_shard, hf_path)
                    timeout = settings.model_load_timeout
                    is_distributed = False
                    if timeout is not None:
                        load_task = asyncio.create_task(coro)
                        try:
                            (
                                model,
                                tokenizer,
                                is_vlm,
                                caps,
                                is_distributed,
                            ) = await asyncio.wait_for(
                                asyncio.shield(load_task), timeout=timeout
                            )
                        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
                            # The background thread continues running — Python
                            # cannot interrupt native threads.  Schedule GPU
                            # cleanup for when it finishes to prevent Metal
                            # memory leaks from orphaned model weights.
                            # This handles both explicit timeouts AND external
                            # cancellations (e.g. client disconnect), since
                            # asyncio.shield protects load_task from being
                            # cancelled but CancelledError still propagates
                            # to the caller.
                            self._schedule_deferred_cleanup(load_task, normalized)
                            if isinstance(exc, asyncio.TimeoutError):
                                raise ModelLoadTimeoutError(
                                    f"Loading model '{normalized}' timed out after {timeout}s. "
                                    f"Increase OLMLX_MODEL_LOAD_TIMEOUT or unset it to disable."
                                )
                            raise
                    else:
                        model, tokenizer, is_vlm, caps, is_distributed = await coro

                    # Check if the model fits safely in memory.  On Apple Silicon
                    # the GPU shares system RAM — if total Metal memory exceeds the
                    # configured fraction of system RAM, generation will almost
                    # certainly OOM and crash the process (Metal abort, not catchable
                    # in Python).  Reject early with a clear error instead.
                    #
                    # Note: this is a post-hoc check — the model is already loaded.
                    # If loading itself triggers an OOM the process will still crash.
                    # In practice, loading succeeds; it is the KV cache allocation
                    # during generation that causes the abort.
                    mem_after = memory_utils.get_metal_memory()
                    total = memory_utils.get_system_memory_bytes()
                    if total > 0 and mem_after > int(
                        total * settings.memory_limit_fraction
                    ):
                        limit = int(total * settings.memory_limit_fraction)
                        model_mb = max(0, (mem_after - mem_before)) // (1024 * 1024)
                        total_mb = total // (1024 * 1024)
                        limit_mb = limit // (1024 * 1024)
                        logger.error(
                            "Model %s uses %d MB, pushing Metal memory to %d MB "
                            "which exceeds the limit of %d MB "
                            "(%.0f%% of %d MB system RAM). Refusing to load.",
                            normalized,
                            model_mb,
                            mem_after // (1024 * 1024),
                            limit_mb,
                            settings.memory_limit_fraction * 100,
                            total_mb,
                        )
                        raise MemoryError(
                            f"Model '{normalized}' requires ~{model_mb} MB but the memory limit "
                            f"is {limit_mb} MB ({settings.memory_limit_fraction:.0%} of "
                            f"{total_mb} MB system RAM). Use a smaller/more quantized model, "
                            f"or increase OLMLX_MEMORY_LIMIT_FRACTION (current: "
                            f"{settings.memory_limit_fraction})."
                        )

                    # Memory check passed — register the model
                    ka = self._resolve_keep_alive(keep_alive)
                    expires = time.time() + ka if ka is not None else None

                    logger.info(
                        "Model %s caps: tools=%s, thinking=%s, thinking_tags=%s",
                        normalized,
                        caps.supports_tools,
                        caps.supports_enable_thinking,
                        caps.has_thinking_tags,
                    )

                    # Detect if flash mode was used
                    is_flash = False
                    try:
                        from olmlx.engine.flash.flash_model import (
                            FlashModelWrapper,
                        )

                        is_flash = isinstance(model, FlashModelWrapper)
                    except ImportError:
                        pass

                    lm = LoadedModel(
                        name=normalized,
                        hf_path=hf_path,
                        model=model,
                        tokenizer=tokenizer,
                        is_vlm=is_vlm,
                        is_distributed=is_distributed,
                        is_flash=is_flash,
                        template_caps=caps,
                        expires_at=expires,
                    )
                    self._loaded[normalized] = lm
                    return lm
                except BaseException:
                    # Drop references and flush Metal allocator so the memory
                    # is actually reclaimed before we raise.  Also clean up
                    # lm if it was already constructed (exception between
                    # LoadedModel() and return), and pop from _loaded to
                    # prevent a zombie entry holding GPU memory.
                    if lm is not None:
                        self._loaded.pop(normalized, None)
                        del lm
                    # When _load_model fails, model/tokenizer are still None —
                    # only bother deleting if they hold actual GPU resources.
                    if model is not None:
                        del model, tokenizer
                    # Release load_task's stored result tuple so the model
                    # weights can actually be freed by gc.collect below.
                    if load_task is not None:
                        del load_task
                    # Skip immediate cache flush when a deferred cleanup is
                    # pending — the background thread is still running and
                    # mx.clear_cache() is not safe to call concurrently with
                    # active Metal allocations.  The deferred cleanup handles
                    # it after the thread finishes.
                    if normalized not in self._pending_cleanups:
                        gc.collect()
                        mx.clear_cache()
                    raise

    def _schedule_deferred_cleanup(
        self, load_task: asyncio.Task, model_name: str
    ) -> None:
        """Schedule GPU cleanup for when a timed-out load thread finishes.

        asyncio.to_thread() threads cannot be interrupted, so after a timeout
        the thread continues running and may allocate GPU memory.  This method
        schedules a task that waits for the thread to complete, then flushes
        the Metal allocator to reclaim the memory.  The task is tracked in
        _pending_cleanups so that a retry for the same model waits for
        cleanup to finish before starting a new load.
        """

        async def _cleanup() -> None:
            cancelled = False
            try:
                await load_task
            except asyncio.CancelledError:
                # The background thread keeps running (Python can't
                # interrupt native threads).  Don't call load_task.cancel()
                # — it can't stop the thread, and the task's result/exception
                # will be drained by stop() via _pending_load_tasks.
                cancelled = True
                raise
            except BaseException:
                pass
            finally:
                # Flush Metal allocator before removing the entry.
                # gc/clear must complete before pop — otherwise a
                # concurrent retry could start its own pre-load gc/clear
                # while ours is still running, which is unsafe for the
                # Metal allocator.  If gc/clear itself fails, still pop
                # so the model slot isn't permanently bricked.
                #
                # Skip gc/clear when cancelled — the background thread is
                # still active and mx.clear_cache() is not safe to call
                # concurrently with Metal allocations.
                try:
                    if not cancelled:
                        gc.collect()
                        mx.clear_cache()
                finally:
                    self._pending_cleanups.pop(model_name, None)
                    # Only pop load_task when not cancelled — stop()
                    # needs the reference to drain orphaned threads.
                    if not cancelled:
                        self._pending_load_tasks.pop(model_name, None)
            logger.info(
                "Deferred GPU cleanup after timeout of '%s' completed", model_name
            )

        self._pending_cleanups[model_name] = asyncio.create_task(_cleanup())
        self._pending_load_tasks[model_name] = load_task

    def get_loaded(self) -> list[LoadedModel]:
        return list(self._loaded.values())

    def invalidate_prompt_cache(self, model_name: str, cache_id: str) -> None:
        """Remove a prompt cache entry for the given model and cache_id."""
        normalized = self.registry.normalize_name(model_name)
        lm = self._loaded.get(normalized)
        if lm is not None:
            lm.prompt_cache_store.remove(cache_id)

    def unload(self, name: str) -> bool:
        """Unload a model. Returns True if unloaded, False if not loaded.

        Raises RuntimeError if model has active requests.
        """
        normalized = self.registry.normalize_name(name)
        lm = self._loaded.get(normalized)
        if lm is None:
            return False
        if lm.active_refs > 0:
            raise RuntimeError(
                f"Model '{normalized}' has {lm.active_refs} active request(s)"
            )
        self._loaded.pop(normalized)
        return True

    # Config keys that indicate a vision-language model
    _VLM_CONFIG_KEYS = frozenset(
        {
            "vision_config",
            "visual",
            "vision_tower",
            "vision_model",
            "image_token_id",
            "vision_feature_layer",
            "mm_vision_tower",
            "visual_config",
            "vit_config",
        }
    )

    def _detect_model_kind(self, hf_path: str) -> str:
        """Return 'text', 'vlm', or 'unknown' by checking config.json against installed libraries."""
        config = None
        # Check local store first
        if self.store is not None:
            local_config = self.store.local_path(hf_path) / "config.json"
            if local_config.exists():
                try:
                    with open(local_config) as f:
                        config = json.load(f)
                except Exception:
                    pass
        # Fall back to HF hub
        if config is None:
            try:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(hf_path, "config.json")
                with open(config_path) as f:
                    config = json.load(f)
            except Exception:
                return "unknown"

        model_type = config.get("model_type", "").lower()
        if not model_type:
            return "unknown"

        # Check for vision-related config keys — these indicate a VLM regardless
        # of whether the base model_type also exists in mlx-lm
        has_vision_keys = bool(self._VLM_CONFIG_KEYS & config.keys())
        if has_vision_keys:
            # Verify mlx-vlm can handle it
            try:
                from mlx_vlm.utils import MODEL_REMAPPING as VLM_REMAP

                mapped = VLM_REMAP.get(model_type, model_type)
                spec = importlib.util.find_spec(f"mlx_vlm.models.{mapped}")
                if spec is not None:
                    return "vlm"
            except (ImportError, ModuleNotFoundError):
                pass
            # Has vision keys but mlx-vlm doesn't recognize it — still try as VLM
            logger.info(
                "Config has vision keys but model_type '%s' not in mlx-vlm, will try anyway",
                model_type,
            )
            return "vlm"

        # No vision keys — check mlx-lm
        try:
            from mlx_lm.utils import MODEL_REMAPPING as LM_REMAP

            mapped = LM_REMAP.get(model_type, model_type)
            spec = importlib.util.find_spec(f"mlx_lm.models.{mapped}")
            if spec is not None:
                return "text"
        except (ImportError, ModuleNotFoundError):
            pass

        # Fallback: check mlx-vlm even without vision keys
        try:
            from mlx_vlm.utils import MODEL_REMAPPING as VLM_REMAP

            mapped = VLM_REMAP.get(model_type, model_type)
            spec = importlib.util.find_spec(f"mlx_vlm.models.{mapped}")
            if spec is not None:
                return "vlm"
        except (ImportError, ModuleNotFoundError):
            pass

        return "unknown"

    def _try_lm_then_vlm(
        self, load_path: str, label: str
    ) -> tuple[Any, Any, bool, TemplateCaps]:
        """Try loading with mlx-lm first, fall back to mlx-vlm on failure."""
        try:
            import mlx_lm

            model, tokenizer = mlx_lm.load(load_path)
            caps = detect_caps(tokenizer)
            return model, tokenizer, False, caps
        except _FALLBACK_EXCEPTIONS as exc:
            logger.warning("mlx-lm failed for %s (%s), trying mlx-vlm", label, exc)
            import mlx_vlm

            model, processor = mlx_vlm.load(load_path)
            tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            caps = detect_caps(tok)
            return model, processor, True, caps

    def _flash_dir(self, hf_path: str) -> Path | None:
        """Return the flash-prepared directory for a model, if it exists."""
        if self.store is None:
            return None
        flash_path = self.store.local_path(hf_path) / "flash"
        if flash_path.exists() and (flash_path / "flash_layout.json").exists():
            return flash_path
        return None

    def _is_flash_enabled(self) -> bool:
        from olmlx.config import experimental

        return experimental.flash

    def _load_flash_model(
        self, hf_path: str, load_path: str, flash_dir: Path
    ) -> tuple[Any, Any, bool, TemplateCaps]:
        """Load a model in flash mode (LLM in a Flash).

        1. Load model normally via mlx-lm
        2. Create FlashWeightStore, PredictorBank, WindowManager
        3. Wrap model with FlashModelWrapper (replaces FFN layers)
        """
        from olmlx.config import experimental
        from olmlx.engine.flash.flash_model import FlashConfig, FlashModelWrapper
        from olmlx.engine.flash.predictor import PredictorBank
        from olmlx.engine.flash.weight_store import FlashWeightStore

        import mlx_lm

        logger.info("Loading model %s in flash mode from %s", hf_path, flash_dir)

        # Load the full model first (we'll free FFN weights after wrapping)
        model, tokenizer = mlx_lm.load(load_path)
        caps = detect_caps(tokenizer)

        # Load predictor bank
        predictor_path = flash_dir / "predictors"
        predictor_bank = PredictorBank.load(predictor_path)

        # Read flash layout for dimensions
        layout_config = json.loads((flash_dir / "flash_layout.json").read_text())

        flash_config = FlashConfig(
            hidden_size=layout_config["hidden_size"],
            intermediate_size=layout_config["intermediate_size"],
            num_layers=layout_config["num_layers"],
            sparsity_threshold=experimental.flash_sparsity_threshold,
            min_active_neurons=experimental.flash_min_active_neurons,
            max_active_neurons=experimental.flash_max_active_neurons,
            window_size=experimental.flash_window_size,
            io_threads=experimental.flash_io_threads,
            cache_budget_neurons=experimental.flash_cache_budget_neurons,
        )

        weight_store = FlashWeightStore(
            flash_dir,
            num_io_threads=flash_config.io_threads,
            cache_budget_neurons=flash_config.cache_budget_neurons,
        )

        # Wrap model — this replaces FFN layers and frees original weights
        wrapped = FlashModelWrapper(model, predictor_bank, weight_store, flash_config)

        return wrapped, tokenizer, False, caps

    def _load_model(self, hf_path: str) -> tuple[Any, Any, bool, TemplateCaps]:
        """Load a model, using config.json inspection to choose the right library."""
        # Ensure model is downloaded to the store
        load_path: str = hf_path
        if self.store is not None:
            local_dir = self.store.ensure_downloaded(hf_path)
            load_path = str(local_dir)

        # Check for flash-prepared model
        if self._is_flash_enabled():
            flash_dir = self._flash_dir(hf_path)
            if flash_dir is not None:
                return self._load_flash_model(hf_path, load_path, flash_dir)

        kind = self._detect_model_kind(hf_path)
        logger.info("Detected model kind for %s: %s", hf_path, kind)

        if kind == "vlm":
            # VLM detected — load with mlx-vlm directly
            import mlx_vlm

            model, processor = mlx_vlm.load(load_path)
            tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            caps = detect_caps(tok)
            return model, processor, True, caps

        # Text or unknown — try mlx-lm first, fall back to mlx-vlm
        return self._try_lm_then_vlm(load_path, hf_path)

    def _load_model_and_shard(
        self, hf_path: str
    ) -> tuple[Any, Any, bool, TemplateCaps, bool]:
        """Load a model and optionally shard it for distributed inference.

        Returns (model, tokenizer, is_vlm, caps, is_distributed).
        """
        model, tokenizer, is_vlm, caps = self._load_model(hf_path)
        is_distributed = False

        if self._distributed_group is not None:
            if is_vlm:
                del model, tokenizer
                gc.collect()
                mx.clear_cache()
                raise ValueError(
                    f"VLM models are not supported in distributed mode. "
                    f"Model {hf_path} is a vision-language model which cannot "
                    f"be sharded correctly across workers."
                )

            if self._distributed_strategy == "pipeline":
                from olmlx.engine.pipeline import apply_pipeline

                try:
                    apply_pipeline(
                        model,
                        self._distributed_group,
                        layer_counts=self._distributed_layer_counts,
                    )
                except ValueError:
                    del model, tokenizer
                    gc.collect()
                    mx.clear_cache()
                    raise
                # Materialize parameters — pipeline nullified unowned layers
                # so only owned weights are evaluated (no cross-rank ops).
                mx.eval(model.parameters())
                is_distributed = True
                logger.info(
                    "Model %s pipeline-sharded for distributed inference", hf_path
                )
            elif self._distributed_strategy == "tensor":
                if not hasattr(model, "shard"):
                    del model, tokenizer
                    gc.collect()
                    mx.clear_cache()
                    raise ValueError(
                        f"Model {hf_path} does not support distributed inference "
                        f"(no shard() method). Supported architectures include: "
                        f"llama, qwen2, qwen3, deepseek_v2, deepseek_v3, etc."
                    )
                model.shard(self._distributed_group)
                # Materialize all lazy weight slices BEFORE any forward pass.
                # model.shard() creates lazy array slices; if they're first
                # evaluated during inference (with all_sum), the combined Metal
                # command buffer can exceed the ~10s GPU timeout for large
                # models (32B+).  Evaluating here is purely local (no all_sum).
                mx.eval(model.parameters())
                is_distributed = True
                logger.info("Model %s sharded for distributed inference", hf_path)
            else:
                del model, tokenizer
                gc.collect()
                mx.clear_cache()
                raise ValueError(
                    f"Unknown distributed strategy {self._distributed_strategy!r}. "
                    f"Supported: 'tensor', 'pipeline'."
                )

        return model, tokenizer, is_vlm, caps, is_distributed

    async def _expire_stale(self):
        """Unload models whose keep-alive has expired (active_refs == 0)."""
        now = time.time()
        async with self._lock:
            # Models with active_refs > 0 are skipped — they are currently
            # serving requests.  Even if a model slips through with
            # active_refs == 0 between ensure_loaded() and _inference_ref(),
            # the caller still holds a Python reference, so the model/
            # tokenizer stay alive; only the _loaded dict entry is removed.
            expired = [
                name
                for name, lm in self._loaded.items()
                if lm.expires_at is not None
                and lm.expires_at <= now
                and lm.active_refs == 0
            ]
            for name in expired:
                logger.info("Unloading expired model %s", name)
                del self._loaded[name]

    async def _check_expiry_loop(self):
        while True:
            await asyncio.sleep(30)
            await self._expire_stale()
