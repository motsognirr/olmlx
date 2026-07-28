"""LoadedModel dataclass, structural_copy, and load errors (extracted from model_manager.py)."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.nn as nn

from olmlx.config import SyncMode
from olmlx.config import settings
from olmlx.engine.template_caps import TemplateCaps
from olmlx.engine.prompt_cache import CachedPromptState, PromptCacheStore  # noqa: F401

# Speculative-decoder draft loaders live in a focused mixin module (#454).
# ``_resolve_attention_causal`` is re-exported for back-compat with tests that
# import it from here.
from olmlx.engine.speculative_loaders import (
    _resolve_attention_causal as _resolve_attention_causal,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class ModelLoadTimeoutError(TimeoutError):
    """Raised when model loading exceeds OLMLX_MODEL_LOAD_TIMEOUT."""


class SpectralCalibrationMissingError(Exception):
    """Raised when SpectralQuant is configured but calibration data is absent."""


class ShardCalibrationMissingError(SpectralCalibrationMissingError):
    """Shard quant configured but calibration artifacts missing/mismatched.

    Subclasses SpectralCalibrationMissingError so the existing app.py
    exception handler (HTTP 400) catches it via Starlette's MRO walk.
    """


class ActiveRequestsError(RuntimeError):
    """Raised by ``ModelManager.unload`` when a model has in-flight requests.

    Subclasses ``RuntimeError`` so legacy ``except RuntimeError:`` keeps
    working, but the dedicated type lets HTTP routers narrow the 409 path
    to exactly this condition. Without it, an unrelated ``RuntimeError``
    from ``_close_loaded_model`` would be misreported as 409.
    """


def structural_copy(module: "nn.Module") -> "nn.Module":
    """Copy an ``nn.Module`` *tree* while sharing every weight by reference.

    LoRA hot-swap (issue #362) needs one base model's weights resident once in
    RAM with multiple adapters layered on top. ``mlx_lm.tuner.utils.load_adapters``
    mutates the module tree in place (replacing ``nn.Linear`` with ``LoRALinear``),
    so each adapter needs its *own* module objects — but the heavy ``mx.array``
    weights must stay shared.

    Neither ``copy.copy`` nor ``copy.deepcopy`` works:

    * ``copy.copy`` is shallow — the ``layers`` list and every submodule are
      shared by reference, so ``load_adapters`` would corrupt the base model.
    * ``copy.deepcopy`` duplicates the ``mx.array`` weights (defeating sharing)
      and crashes on quantized models — modules store an ``mx.Dtype`` attribute
      and ``Dtype`` is not picklable.

    So this walks the tree explicitly: fresh ``Module`` / list / dict / tuple
    containers, with ``mx.array`` leaves and scalar attributes (``bits``,
    ``group_size``, dtypes, …) shared by reference. Mutable per-module Python
    state (the ``_no_grad`` / freeze-tracking sets) is copied so freezing one
    instance can't affect another. ``nn.Module`` overrides ``__setattr__`` to
    route into its backing dict, so attributes are copied by mutating
    ``__dict__`` in place rather than reassigning it. Verified to share weights
    (no extra Metal allocation) and isolate the base through a real
    ``load_adapters`` call on a 4-bit-quantized model.

    A by-id memo makes the walk robust to module graphs that aren't strict
    trees: a submodule instance shared across layers (e.g. one rotary-embedding
    object) is copied once and re-shared, and any back-reference / cycle
    resolves to the in-progress copy instead of recursing forever. ``id()`` is
    stable here because the whole source tree stays referenced for the walk's
    duration, so no object is collected and no address is recycled.
    """

    memo: dict[int, Any] = {}

    def _copy(obj: Any) -> Any:
        oid = id(obj)
        cached = memo.get(oid)
        if cached is not None:
            return cached
        if isinstance(obj, nn.Module):
            new = obj.__class__.__new__(obj.__class__)
            memo[oid] = new  # register before recursing so cycles terminate
            for k, v in obj.__dict__.items():
                new.__dict__[k] = v.copy() if isinstance(v, (set, dict, list)) else v
            for k, v in obj.items():
                dict.__setitem__(new, k, _copy(v))
            return new
        if isinstance(obj, dict):
            new_dict: dict = {}
            memo[oid] = new_dict
            for k, v in obj.items():
                new_dict[k] = _copy(v)
            return new_dict
        if isinstance(obj, list):
            new_list: list = []
            memo[oid] = new_list
            for v in obj:
                new_list.append(_copy(v))
            return new_list
        if isinstance(obj, tuple):
            # Immutable, so it can't be pre-registered before its contents are
            # built — but any reference cycle reaching a tuple must pass through
            # a mutable container (list/dict/Module), which IS pre-registered,
            # so recursion still terminates. Register after building to dedupe a
            # tuple shared from several places.
            new_tuple = tuple(_copy(v) for v in obj)
            memo[oid] = new_tuple
            return new_tuple
        return obj

    return _copy(module)


@dataclass
class LoadedModel:
    name: str
    hf_path: str
    model: Any
    tokenizer: Any  # tokenizer (mlx-lm) or processor (mlx-vlm)
    is_vlm: bool = False
    is_distributed: bool = False
    is_flash: bool = False
    is_flash_moe: bool = False
    is_whisper: bool = False
    is_tts: bool = False
    is_reranker: bool = False
    speculative_decoder: Any = None
    weight_store: Any = None
    # LoRA-adapter hot-swap (issue #362). On an adapter entry, ``adapter_base``
    # names the base model's key in ``_loaded`` (whose weights this entry
    # shares via a structural copy); None on ordinary base models.
    # ``_adapter_child_refs`` counts adapters currently sharing THIS entry's
    # weights — a base with refs > 0 is pinned against LRU eviction. Guarded by
    # the manager's ``_lock`` (mutated only on the loop thread).
    adapter_base: str | None = None
    _adapter_child_refs: int = field(default=0, compare=False, repr=False)
    # Continuous-batching scheduler (engine/batching.py), created lazily on
    # the first batch-eligible request. None for ineligible/idle models.
    batch_scheduler: Any = None
    # Memoized result of the batch cache-probe (None = not yet probed).
    batch_convertible: bool | None = None
    template_caps: TemplateCaps = field(default_factory=TemplateCaps)
    loaded_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    size_bytes: int = 0
    active_refs: int = 0
    _active_refs_lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False
    )
    prompt_cache_store: PromptCacheStore = field(default=None)  # type: ignore[assignment]
    # Per-cache_id LRU of mlx_vlm PromptCacheState objects for cross-turn
    # image-prefix KV reuse. Only populated for VLMs (None otherwise). #429.
    vlm_prompt_cache_store: Any = None
    kv_cache_quant: str | None = None
    #: StreamingLLM sink+window KV eviction "<sink>:<window>" (#505), or None.
    kv_eviction: str | None = None
    #: Weight quantization string (e.g. "hqq:4") applied at load time.
    #: ``None`` means no weight quantization was applied.
    weight_quant: str | None = None
    # False for hybrid sliding-window models (RotatingKVCache layers).
    # Set by the loader; default True covers direct construction in tests.
    supports_cache_trim: bool = True
    # False for hybrid SSM-style models (ArraysCache layers, e.g. Qwen3.5,
    # Qwen3-Next).  When False, prompt cache state is not persisted across
    # requests because cross-request reuse crashes mlx-lm with a Metal
    # stream error.  Set by the loader's _probe_cache_capabilities call.
    # Issue #284.
    #
    # Defaults to False (unsafe-by-default) — unlike supports_cache_trim,
    # a false-positive here crashes the next request rather than wasting a
    # trim_prompt_cache() call.  Direct LoadedModel construction must
    # explicitly opt in if the model's cache layout is known to be safe.
    supports_cache_persistence: bool = False
    # True for models whose cache layout is non-trimmable (ArraysCache,
    # RotatingKVCache) but can still benefit from cross-request reuse via
    # the checkpoint path: prefill per message-segment, snapshot at each
    # boundary.  Set by _probe_cache_capabilities (Tasks 5.2/5.3).
    # Defaults to False until the probe populates it.
    uses_checkpoint_persistence: bool = False
    spectral_calibration_dir: Any = None  # Path | None, typed as Any to avoid import
    shard_calibration_dir: Any = None  # Path | None, typed as Any to avoid import
    default_options: dict = field(default_factory=dict)
    inference_queue_timeout: float | None = None
    inference_timeout: float | None = None
    sync_mode: SyncMode | None = None
    # Per-model default for the chat-template ``enable_thinking`` kwarg.
    # ``generate_chat`` consults this when the request didn't set the
    # flag.  None means defer to the engine default.
    enable_thinking: bool | None = None
    # Per-model default reasoning level for channel-format reasoners
    # (gpt-oss / Harmony): "low"/"medium"/"high". ``generate_chat`` consults
    # this when the caller didn't pass ``reasoning_effort``.
    reasoning_effort: str | None = None
    # Per-model override for the cross-request prompt cache toggle.
    # ``generate_chat`` honours this in place of ``settings.prompt_cache``
    # when set. None means defer to the global setting. Surfaced for
    # architectures that crash or degrade on the checkpoint path while
    # other models continue to benefit from caching (e.g. Qwen3-Coder-Next
    # MoE-quantized targets where chunked prefill triggers GatedDeltaNet
    # numerical drift across expert-routing thresholds).
    prompt_cache: bool | None = None
    # Per-model override for continuous batching. ``_batch_eligible``
    # consults this in place of ``settings.batching`` when set; None defers
    # to the global toggle. Mechanical eligibility (cache layout, model
    # kind, KV-quant, …) still applies on top. The batch-size knobs below
    # likewise override ``settings.batch_*`` in ``_get_batch_scheduler``
    # when set; None means inherit the global value.
    batching: bool | None = None
    batch_completion_size: int | None = None
    batch_prefill_size: int | None = None
    batch_prefill_step: int | None = None
    batch_fairness_quantum: float | None = None

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
            ram_budget_bytes = int(settings.prompt_cache_ram_budget_gb * 1024**3)
            self.prompt_cache_store = PromptCacheStore(
                max_slots=settings.prompt_cache_max_slots,
                disk_path=disk_path,
                model_name=self.name,
                disk_max_bytes=disk_max_bytes,
                ram_budget_bytes=ram_budget_bytes,
            )
        if self.is_vlm and self.vlm_prompt_cache_store is None:
            from olmlx.engine.prompt_cache.vlm_state import VlmPromptCacheStore

            vlm_disk_path = (
                settings.vlm_prompt_cache_disk_path
                if settings.vlm_prompt_cache_disk
                else None
            )
            vlm_disk_max_bytes = (
                int(settings.vlm_prompt_cache_disk_max_gb * 1024**3)
                if settings.vlm_prompt_cache_disk
                else None
            )
            self.vlm_prompt_cache_store = VlmPromptCacheStore(
                capacity=settings.vlm_prompt_cache_slots,
                disk_path=vlm_disk_path,
                model_name=self.name,
                disk_max_bytes=vlm_disk_max_bytes,
            )

    def acquire_ref(self) -> None:
        """Pin the model so eviction/expiry skip it (active_refs += 1)."""
        with self._active_refs_lock:
            self.active_refs += 1

    def release_ref(self) -> None:
        """Release a pin taken by :meth:`acquire_ref` (active_refs -= 1)."""
        with self._active_refs_lock:
            self.active_refs -= 1

    @property
    def is_speculative(self) -> bool:
        return self.speculative_decoder is not None

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


def _is_cross_encoder_config(config: dict) -> bool:
    """True for an XLM-RoBERTa-family sequence-classification reranker (#369).

    Requires both a ``*ForSequenceClassification`` architecture AND a
    roberta-family ``model_type`` — the encoder and weight remaps only support
    XLM-RoBERTa/RoBERTa, so a plain BERT/DistilBERT text classifier must NOT be
    routed here (it would otherwise be detected as a reranker and fail at load).
    """
    archs = config.get("architectures") or []
    if not any(
        isinstance(a, str) and a.endswith("ForSequenceClassification") for a in archs
    ):
        return False
    return "roberta" in (config.get("model_type") or "").lower()


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
    # Bare integer string → treat as seconds (consistent with Ollama API)
    if value.isdigit():
        return float(int(value))
    match = re.match(r"^(\d+)(s|m|h)$", value)
    if not match:
        logger.warning("Invalid keep_alive format: %r, defaulting to 5m", value)
        return 300.0  # default 5m
    num, unit = int(match.group(1)), match.group(2)
    multipliers = {"s": 1, "m": 60, "h": 3600}
    return float(num * multipliers[unit])
