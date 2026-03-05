import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from mlx_ollama.config import settings
from mlx_ollama.engine.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    name: str
    hf_path: str
    model: Any
    tokenizer: Any  # tokenizer (mlx-lm) or processor (mlx-vlm)
    is_vlm: bool = False
    loaded_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    size_bytes: int = 0


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

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
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

    async def ensure_loaded(
        self, name: str, keep_alive: str | None = None
    ) -> LoadedModel:
        """Ensure a model is loaded and return it."""
        normalized = self.registry.normalize_name(name)
        async with self._lock:
            if normalized in self._loaded:
                lm = self._loaded[normalized]
                # Refresh expiry
                ka = parse_keep_alive(
                    keep_alive if keep_alive is not None else settings.default_keep_alive
                )
                if ka is not None:
                    lm.expires_at = time.time() + ka
                else:
                    lm.expires_at = None
                return lm

            # Evict LRU if at capacity
            while len(self._loaded) >= settings.max_loaded_models:
                oldest_name = min(
                    self._loaded, key=lambda k: self._loaded[k].loaded_at
                )
                logger.info("Evicting model %s", oldest_name)
                del self._loaded[oldest_name]

            hf_path = self.registry.resolve(name)
            if hf_path is None:
                raise ValueError(
                    f"Model '{name}' not found. Add it to models.json or use a HuggingFace path."
                )

            logger.info("Loading model %s from %s", normalized, hf_path)
            model, tokenizer, is_vlm = await asyncio.to_thread(self._load_model, hf_path)

            ka = parse_keep_alive(
                keep_alive if keep_alive is not None else settings.default_keep_alive
            )
            expires = time.time() + ka if ka is not None else None

            lm = LoadedModel(
                name=normalized,
                hf_path=hf_path,
                model=model,
                tokenizer=tokenizer,
                is_vlm=is_vlm,
                expires_at=expires,
            )
            self._loaded[normalized] = lm
            return lm

    def get_loaded(self) -> list[LoadedModel]:
        return list(self._loaded.values())

    def unload(self, name: str):
        normalized = self.registry.normalize_name(name)
        self._loaded.pop(normalized, None)

    @staticmethod
    def _load_model(hf_path: str) -> tuple[Any, Any, bool]:
        """Try mlx-lm first; fall back to mlx-vlm for vision-language models."""
        try:
            import mlx_lm
            model, tokenizer = mlx_lm.load(hf_path)
            return model, tokenizer, False
        except (ValueError, Exception) as exc:
            if "not in model" not in str(exc) and "vision" not in str(exc).lower():
                raise
            logger.info("mlx-lm failed (%s), trying mlx-vlm", exc.__class__.__name__)
            import mlx_vlm
            model, processor = mlx_vlm.load(hf_path)
            return model, processor, True

    async def _check_expiry_loop(self):
        while True:
            await asyncio.sleep(30)
            now = time.time()
            async with self._lock:
                expired = [
                    name
                    for name, lm in self._loaded.items()
                    if lm.expires_at is not None and lm.expires_at <= now
                ]
                for name in expired:
                    logger.info("Unloading expired model %s", name)
                    del self._loaded[name]
