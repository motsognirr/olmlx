"""Single chokepoint for loading models through mlx-vlm.

Every ``mlx_vlm.load`` call in olmlx goes through :func:`load_vlm` so that
load-time workarounds for upstream regressions are applied exactly once and
structurally — a new call site cannot forget them.
``tests/test_vlm_load.py`` enforces the invariant with a source grep.
"""

from __future__ import annotations

from typing import Any


def load_vlm(path: str, **kwargs: Any) -> tuple[Any, Any]:
    """``mlx_vlm.load`` with olmlx's load-time workarounds applied.

    Returns ``(model, processor)`` exactly like ``mlx_vlm.load``.
    """
    import mlx_vlm

    from olmlx.engine.gemma4_sanitize_fix import ensure_gemma4_sanitize_patch

    ensure_gemma4_sanitize_patch()
    return mlx_vlm.load(path, **kwargs)
