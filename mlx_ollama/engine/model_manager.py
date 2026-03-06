from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mlx_ollama.config import settings
from mlx_ollama.engine.registry import ModelRegistry
from mlx_ollama.engine.template_caps import TemplateCaps, detect_caps

if TYPE_CHECKING:
    from mlx_ollama.models.store import ModelStore

logger = logging.getLogger(__name__)

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


@dataclass
class LoadedModel:
    name: str
    hf_path: str
    model: Any
    tokenizer: Any  # tokenizer (mlx-lm) or processor (mlx-vlm)
    is_vlm: bool = False
    template_caps: TemplateCaps = field(default_factory=TemplateCaps)
    loaded_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    size_bytes: int = 0
    active_refs: int = 0


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
        return 300.0  # default 5m
    num, unit = int(match.group(1)), match.group(2)
    multipliers = {"s": 1, "m": 60, "h": 3600}
    return float(num * multipliers[unit])


class ModelManager:
    """Manages loading/unloading of MLX models with LRU eviction."""

    def __init__(self, registry: ModelRegistry, store: ModelStore | None = None):
        self.registry = registry
        self.store = store
        self._loaded: dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()
        self._expiry_task: asyncio.Task | None = None

    def start_expiry_checker(self):
        self._expiry_task = asyncio.create_task(self._check_expiry_loop())

    async def stop(self):
        if self._expiry_task:
            self._expiry_task.cancel()
            try:
                await self._expiry_task
            except asyncio.CancelledError:
                pass
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
        async with self._lock:
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

            hf_path = self.registry.resolve(name)
            if hf_path is None:
                raise ValueError(
                    f"Model '{name}' not found. Add it to models.json or use a HuggingFace path."
                )

            # Auto-register direct HF paths so future requests find them
            if "/" in name:
                self.registry.add_mapping(name, hf_path)

            logger.info("Loading model %s from %s", normalized, hf_path)
            model, tokenizer, is_vlm, caps = await asyncio.to_thread(
                self._load_model, hf_path
            )

            ka = self._resolve_keep_alive(keep_alive)
            expires = time.time() + ka if ka is not None else None

            logger.info(
                "Model %s caps: tools=%s, thinking=%s, thinking_tags=%s",
                normalized,
                caps.supports_tools,
                caps.supports_enable_thinking,
                caps.has_thinking_tags,
            )

            lm = LoadedModel(
                name=normalized,
                hf_path=hf_path,
                model=model,
                tokenizer=tokenizer,
                is_vlm=is_vlm,
                template_caps=caps,
                expires_at=expires,
            )
            self._loaded[normalized] = lm
            return lm

    def get_loaded(self) -> list[LoadedModel]:
        return list(self._loaded.values())

    def unload(self, name: str):
        normalized = self.registry.normalize_name(name)
        self._loaded.pop(normalized, None)

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

    def _load_model(self, hf_path: str) -> tuple[Any, Any, bool, TemplateCaps]:
        """Load a model, using config.json inspection to choose the right library."""
        # Ensure model is downloaded to the store
        load_path: str = hf_path
        if self.store is not None:
            local_dir = self.store.local_path(hf_path)
            if not self.store.is_downloaded(hf_path):
                from huggingface_hub import snapshot_download

                logger.info("Downloading %s to %s", hf_path, local_dir)
                local_dir.mkdir(parents=True, exist_ok=True)
                snapshot_download(repo_id=hf_path, local_dir=str(local_dir))
            load_path = str(local_dir)

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

    async def _check_expiry_loop(self):
        while True:
            await asyncio.sleep(30)
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
