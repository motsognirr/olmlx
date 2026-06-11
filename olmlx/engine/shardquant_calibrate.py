"""Shard calibration: per-head no-RoPE PCA for K, product-VQ for V (#377)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_shard_calibration(
    calibration_dir: Path,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Load shard calibration artifacts. Implemented with the calibration
    pipeline (next task); make_shard_cache tests monkeypatch this."""
    raise NotImplementedError
