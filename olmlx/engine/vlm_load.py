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

    # Function-local import: model_manager imports this module (lazily), so a
    # top-level import back would form a cycle. Matches flash.prepare's pattern.
    from olmlx.engine.model_manager import _materialize_module_buffers

    ensure_gemma4_sanitize_patch()
    model, processor = mlx_vlm.load(path, **kwargs)
    # Materialize non-parameter buffers (scaled-RoPE ``_freqs``, ...) on THIS
    # (load) thread. mlx-vlm's parameter eval — like mlx-lm's — skips underscore
    # buffers, so left lazy they crash when first evaluated on the generation
    # worker thread (#499). Only touches small computed constants, never the
    # (possibly intentionally-lazy) weights. Covers the main VLM path plus the
    # lm-then-vlm / flash / flash-MoE VLM fallbacks that all route through here.
    _materialize_module_buffers(model)
    return model, processor
