"""Workaround for the mlx 0.31.x batched-decode ``mx.fast.rope`` bug.

mlx 0.31.x (pinned ``<0.31.2``, see pyproject) corrupts batch rows >= 1
when ``mx.fast.rope`` is called with B > 1 at L == 1 — the batched
single-token decode shape — **with a scalar offset**. The per-row
vector-offset path (what mlx-lm's BatchKVCache passes during batched
decode with left padding) is correct; only scalar-offset callers (e.g.
dflash self-generation's uniform-length batches over plain KVCache) hit
the broken kernel. Verified empirically on mlx 0.31.1 — see
tests/test_batching.py's removal-gate test.

Folding the batch dim into the heads dim before the kernel call is
exact because RoPE only rotates along the position axis (-2), and is
only possible for a scalar offset (a per-row offset vector has no
single fold). Only the buggy shape is redirected; prefill (L > 1),
unbatched, and vector-offset calls go straight through. The patch is
process-global while active, so hold it only around the generation
loop.

Remove once mlx is fixed (#499 tracks moving past the 0.31.2 pin); the
removal-gate unit test and the tests/live batched parity canary fail
loudly when that time comes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlx.core as mx


def _scalar_offset(offset: Any) -> bool:
    if isinstance(offset, int):
        return True
    if isinstance(offset, mx.array):
        return offset.ndim == 0 or offset.size == 1
    return False


@contextmanager
def safe_rope_patch() -> Iterator[None]:
    """Patch ``mx.fast.rope`` to sidestep the scalar-offset B > 1, L == 1
    corruption."""
    orig = mx.fast.rope

    def _folded(x: mx.array, dims: int, *args: Any, **kwargs: Any) -> mx.array:
        if (
            getattr(x, "ndim", 0) == 4
            and x.shape[0] > 1
            and x.shape[2] == 1
            # offset is keyword-only in mx.fast.rope; absent means 0 (scalar)
            and _scalar_offset(kwargs.get("offset", 0))
        ):
            B, H, L, D = x.shape
            out = orig(x.reshape(1, B * H, L, D), dims, *args, **kwargs)
            return out.reshape(B, H, L, D)
        return orig(x, dims, *args, **kwargs)

    mx.fast.rope = _folded
    try:
        yield
    finally:
        mx.fast.rope = orig
