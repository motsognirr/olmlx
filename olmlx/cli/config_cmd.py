"""Config bootstrap and logging setup for the olmlx CLI."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from olmlx.config import (
    settings,
)

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "llama3.2:latest": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistral:7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "qwen2.5:3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "gemma2:2b": "mlx-community/gemma-2-2b-it-4bit",
    "whisper-turbo:latest": "mlx-community/whisper-large-v3-turbo",
    "whisper-large:latest": "mlx-community/whisper-large-v3-mlx",
}


def ensure_config():
    """Create ~/.olmlx/ and seed models.json if missing."""
    config_dir = settings.models_config.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    if not settings.models_config.exists():
        # Atomic temp-file-plus-rename: a crash mid-write must not leave a
        # truncated models.json, which the strict loader would then
        # hard-refuse to start on every subsequent run (#635).
        from olmlx.engine.registry import _atomic_write_json

        _atomic_write_json(DEFAULT_MODELS, settings.models_config)
        print(f"Created {settings.models_config} with example models")


def _configure_logging():
    """Configure logging from settings."""
    from olmlx.context import RequestIDFormatter

    handler = logging.StreamHandler()
    handler.setFormatter(
        RequestIDFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level))
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)
