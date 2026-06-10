"""Per-cache_id LRU of mlx_vlm PromptCacheState objects for VLM prompt caching.

mlx_vlm's ``stream_generate`` owns the hard parts of cross-turn KV reuse:
prefix matching against a stored token sequence, trimming the cache to the
common prefix, detecting whether the new tokens still contain image
placeholders (so vision features are only recomputed when needed), and
updating the state in place after generation.  This store just bounds how
many ``PromptCacheState`` lineages are retained, keyed by the request's
``cache_id``.

Mirrors the speculative ``_SpecCacheStore`` ergonomics: all inference is
serialized under the inference lock, so no internal locking is needed;
``capacity == 0`` is the kill switch (``OLMLX_VLM_PROMPT_CACHE_SLOTS=0``).

v1 limits (documented in CLAUDE.md): in-memory only — no radix-takeover, no
disk spill, no KV-quant.  At the small default slot count, keying on
``cache_id`` (rather than a longest-prefix scan) is sufficient.
"""

from __future__ import annotations

from typing import Any

from olmlx.utils.loop_affinity import assert_loop_thread


class VlmPromptCacheStore:
    def __init__(self, capacity: int) -> None:
        self._capacity = max(int(capacity), 0)
        # Insertion-ordered dict as an LRU: first key is least-recently-used.
        self._entries: dict[str, Any] = {}
        # Cumulative reuse counters surfaced on /api/ps (acceptance criterion 1).
        self._hits = 0
        self._misses = 0
        self._tokens_reused = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def enabled(self) -> bool:
        return self._capacity > 0

    def clear(self) -> None:
        """Drop retained states. Counters are cumulative and NOT reset here, so
        /api/ps reuse totals survive memory-pressure flushes.

        Deliberately NOT loop-affine (#463): like PromptCacheStore.clear, the
        production caller is the worker-thread close path in
        ``_close_loaded_model``, which owns the store exclusively by then.
        """
        self._entries.clear()

    def note_hit(self, reused_tokens: int) -> None:
        self._hits += 1
        self._tokens_reused += max(int(reused_tokens), 0)

    def note_miss(self) -> None:
        self._misses += 1

    def metrics(self) -> dict[str, int]:
        return {
            "vlm_cache_hits": self._hits,
            "vlm_cache_misses": self._misses,
            "vlm_cache_tokens_reused": self._tokens_reused,
        }

    def get(self, cache_id: str) -> Any | None:
        """Return the PromptCacheState for ``cache_id`` and promote it to
        most-recently-used, or ``None`` on miss / when disabled."""
        assert_loop_thread("VlmPromptCacheStore.get")
        if not self.enabled():
            return None
        state = self._entries.pop(cache_id, None)
        if state is None:
            return None
        self._entries[cache_id] = state  # re-insert at MRU end
        return state

    def insert(self, cache_id: str, state: Any) -> None:
        """Store ``state`` under ``cache_id`` as most-recently-used, evicting
        the least-recently-used entries past ``capacity``. No-op when disabled."""
        assert_loop_thread("VlmPromptCacheStore.insert")
        if not self.enabled():
            return
        self._entries.pop(cache_id, None)  # refresh position if already present
        self._entries[cache_id] = state
        while len(self._entries) > self._capacity:
            # Pop the oldest (least-recently-used) entry.
            oldest = next(iter(self._entries))
            del self._entries[oldest]
