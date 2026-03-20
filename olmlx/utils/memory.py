"""Consolidated Metal/system memory helpers.

Shared by engine.inference and engine.model_manager to avoid duplication.
"""

import os

import mlx.core as mx


def get_metal_memory() -> int:
    """Return total Metal memory in use (active + cached)."""
    return mx.get_active_memory() + mx.get_cache_memory()


def get_system_memory_bytes() -> int:
    """Return total system memory in bytes, or 0 on failure."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        return 0


# Fraction of memory_limit_fraction at which we shed the prompt cache to
# free Metal memory before hitting the hard model-load rejection limit.
MEMORY_PRESSURE_THRESHOLD = 0.9


def is_memory_pressure_high(
    fraction: float, threshold: float = MEMORY_PRESSURE_THRESHOLD
) -> bool:
    """Check if Metal memory is approaching the safety limit.

    Returns True when Metal memory exceeds ``fraction * threshold`` of system RAM.
    Returns False when system memory cannot be determined or on any error.
    """
    try:
        total = get_system_memory_bytes()
        if total == 0:
            return False
        limit = int(total * fraction)
        return get_metal_memory() > int(limit * threshold)
    except Exception:
        return False
