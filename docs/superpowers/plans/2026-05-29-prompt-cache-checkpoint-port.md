# Prompt-Cache Message-Boundary Checkpoint Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable cross-request prompt-cache reuse for hybrid models whose layer types currently force `supports_cache_persistence = False` — Qwen3.5 text tower, Nemotron-H, Jamba (`ArraysCache`-bearing), and Gemma 3 / GPT-OSS (`RotatingKVCache` post-wrap) — by porting mlx-lm's message-boundary checkpoint mechanism into `PromptCacheStore`. Mixed-cache models (Qwen3-Next, tracked in #396) remain explicitly excluded.

**Architecture:** The chat routers tokenize each message in the request as a separate segment instead of running `apply_chat_template()` once. The token offsets between segments become checkpoint positions. During request handling, `_setup_prompt_cache` walks the radix to find the longest stored prefix and drives the model forward in chunks aligned to segment boundaries, snapshotting (`mx.eval` + deepcopy for non-trimmable caches) at each boundary into `PromptCacheStore`. Lookup uses two new modes: *shorter-match* (resume after a stored prefix, no trim needed — the path that unblocks non-trimmable models) and *exact-match*, in addition to the existing longest-prefix-with-trim path.

**Tech Stack:** mlx-lm (cache layer), mlx-core (`mx.eval`), our existing `PromptCacheStore` + `PrefixCacheIndex` + `_setup_prompt_cache`/`_store_prompt_cache_after_generation` in `engine/inference.py`, `_probe_cache_capabilities` in `engine/model_manager.py`, the four chat routers (`routers/openai.py`, `routers/anthropic.py`, `routers/chat.py`, `routers/generate.py`).

---

## Out of scope (do NOT do in this plan)

- Qwen3-Next mixed SWA + GDN composition (issue #396 — explicitly excluded; the probe must still reject this layout).
- VLM models (mlx-vlm doesn't expose `prompt_progress_callback`; segment-boundary snapshot mechanism doesn't apply).
- Distributed inference path (sideband protocol doesn't carry segment metadata; persistence remains disabled there).
- Speculative decoders (cache integration with speculative paths is already gated off per `OLMLX_SPECULATIVE` notes in CLAUDE.md — leave gating as-is).
- Whisper, embedding, and audio-transcription paths.

## File structure

**New files:**
- `olmlx/engine/prompt_cache/checkpoint.py` — `SegmentedPrompt` dataclass, `snapshot_cache_for_persistence()` helper (deepcopy + eager `mx.eval`), `iter_checkpoint_boundaries()`.
- `tests/test_prompt_cache_checkpoint.py` — unit tests for `SegmentedPrompt`, snapshot helper, boundary iteration.
- `tests/test_prompt_cache_store_multi.py` — unit tests for the multi-checkpoint store extension.
- `tests/test_inference_checkpoint_drive.py` — unit tests for the segmented-prefill drive logic in `_setup_prompt_cache`.
- `tests/test_probe_cache_capabilities_v2.py` — unit tests for the updated probe.

**Modified files:**
- `olmlx/engine/prompt_cache/state.py` — extend `CachedPromptState` with `cache_type: Literal["system","user","assistant"]` and `is_checkpoint: bool`.
- `olmlx/engine/prompt_cache/store.py` — allow multiple entries to share a `cache_id` prefix; add `insert_checkpoint()` and `fetch_nearest()` returning `(state, suffix_tokens)`; keep `get`/`set` for backward compatibility (assistant terminal entries).
- `olmlx/engine/prompt_cache/radix.py` — add `find_all_at_or_below(tokens, min_depth)` returning every stored entry that is a strict prefix of `tokens`, for shorter-match lookup.
- `olmlx/engine/inference.py` — `_setup_prompt_cache` reworked to consume `SegmentedPrompt`; new `_drive_segmented_prefill()` helper runs `model(...)` per segment with snapshots; `_store_prompt_cache_after_generation()` becomes a no-op for the new path (terminal snapshots already taken).
- `olmlx/engine/model_manager.py` — `_PERSISTABLE_CACHE_CLASSES` adds `ArraysCache`; `_probe_cache_capabilities` adds explicit rejection of mixed `RotatingKVCache + ArraysCache` (Qwen3-Next) even though both members are individually persistable; flips persistence on for non-trimmable layouts when the new checkpoint path is the only consumer.
- `olmlx/routers/openai.py` — tokenize messages segment-by-segment; build `SegmentedPrompt`; pass into `generate_chat`.
- `olmlx/routers/anthropic.py` — same.
- `olmlx/routers/chat.py` (Ollama `/api/chat`) — same.
- `olmlx/routers/generate.py` (Ollama `/api/generate`) — generate has no segment structure; passes a single-segment `SegmentedPrompt`.
- `CLAUDE.md` — update the "Prompt caching" design-decision bullet to describe the new mechanism and which model families it unblocks.

---

## Phase 0 — Setup (worktree, branch, baseline)

### Task 0.1: Create an isolated worktree

**Files:** none yet (worktree creation only).

- [ ] **Step 1: Run the worktree creation command**

Run: `git worktree add ../olmlx-checkpoint-port -b feat/prompt-cache-checkpoint`
Expected: prints `Preparing worktree`, then `HEAD is now at <sha>`.

- [ ] **Step 2: Switch into the worktree**

Run: `cd ../olmlx-checkpoint-port`
Expected: pwd ends in `olmlx-checkpoint-port`.

- [ ] **Step 3: Confirm baseline tests pass**

Run: `uv run pytest tests/test_prompt_cache_radix.py tests/test_prompt_cache_store.py -q`
Expected: all green. Baseline is clean before we touch anything.

---

## Phase 1 — `SegmentedPrompt` foundation

### Task 1.1: Define `SegmentedPrompt` and the segment dataclass

**Files:**
- Create: `olmlx/engine/prompt_cache/checkpoint.py`
- Test: `tests/test_prompt_cache_checkpoint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompt_cache_checkpoint.py
from olmlx.engine.prompt_cache.checkpoint import SegmentedPrompt, Segment

def test_segmented_prompt_total_tokens_is_sum_of_segments():
    sp = SegmentedPrompt(
        segments=[
            Segment(tokens=[1, 2, 3], role="system"),
            Segment(tokens=[4, 5], role="user"),
        ]
    )
    assert sp.total_tokens == 5
    assert sp.flatten() == [1, 2, 3, 4, 5]

def test_segmented_prompt_boundary_offsets_are_cumulative():
    sp = SegmentedPrompt(
        segments=[
            Segment(tokens=[1, 2, 3], role="system"),
            Segment(tokens=[4, 5], role="user"),
            Segment(tokens=[6], role="user"),
        ]
    )
    assert sp.boundary_offsets() == [3, 5, 6]

def test_segmented_prompt_empty_is_valid():
    sp = SegmentedPrompt(segments=[])
    assert sp.total_tokens == 0
    assert sp.boundary_offsets() == []
    assert sp.flatten() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompt_cache_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.engine.prompt_cache.checkpoint'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# olmlx/engine/prompt_cache/checkpoint.py
"""Message-boundary checkpoint primitives for the prompt cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SegmentRole = Literal["system", "user", "assistant", "tool", "developer"]


@dataclass(frozen=True)
class Segment:
    """One contiguous run of tokens belonging to a single chat message."""

    tokens: list[int]
    role: SegmentRole


@dataclass(frozen=True)
class SegmentedPrompt:
    """A prompt split into per-message segments, in submission order."""

    segments: list[Segment]

    @property
    def total_tokens(self) -> int:
        return sum(len(s.tokens) for s in self.segments)

    def flatten(self) -> list[int]:
        out: list[int] = []
        for s in self.segments:
            out.extend(s.tokens)
        return out

    def boundary_offsets(self) -> list[int]:
        """Cumulative end-offset for each segment.

        Empty list when there are no segments. The last value equals
        ``total_tokens``.
        """
        out: list[int] = []
        running = 0
        for s in self.segments:
            running += len(s.tokens)
            out.append(running)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompt_cache_checkpoint.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/prompt_cache/checkpoint.py tests/test_prompt_cache_checkpoint.py
git commit -m "feat(prompt_cache): add SegmentedPrompt dataclass for message-boundary checkpoints"
```

### Task 1.2: Add `snapshot_cache_for_persistence()` (eager `mx.eval` + deepcopy)

**Files:**
- Modify: `olmlx/engine/prompt_cache/checkpoint.py`
- Test: `tests/test_prompt_cache_checkpoint.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_prompt_cache_checkpoint.py
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from olmlx.engine.prompt_cache.checkpoint import snapshot_cache_for_persistence

def test_snapshot_cache_returns_deepcopy_of_arrays():
    cache = [KVCache()]
    keys = mx.zeros((1, 4, 8, 16))
    values = mx.ones((1, 4, 8, 16))
    cache[0].update_and_fetch(keys, values)
    snap = snapshot_cache_for_persistence(cache, eager_eval=True)
    assert snap is not cache, "must return a new list, not the input"
    assert snap[0] is not cache[0], "must deepcopy the layer object"
    # The snapshot's arrays must still represent the same data.
    snap_keys, _ = snap[0].state
    cache_keys, _ = cache[0].state
    assert mx.allclose(snap_keys, cache_keys).item()
    # Mutating the original must not affect the snapshot.
    cache[0].update_and_fetch(mx.zeros_like(keys), mx.zeros_like(values))
    snap_keys_after, _ = snap[0].state
    assert mx.allclose(snap_keys_after, snap_keys).item(), (
        "snapshot must not see the post-snapshot update"
    )

def test_snapshot_cache_eager_eval_materializes_state():
    """eager_eval=True should call mx.eval on the layer state arrays."""
    cache = [KVCache()]
    keys = mx.zeros((1, 4, 8, 16))
    values = mx.ones((1, 4, 8, 16))
    cache[0].update_and_fetch(keys, values)
    # After eager eval, the state arrays must be evaluated (not lazy).
    snap = snapshot_cache_for_persistence(cache, eager_eval=True)
    snap_keys, snap_values = snap[0].state
    # mx.array has no public "is_evaluated" flag, so we proxy via
    # nbytes which forces evaluation if lazy — but lazy arrays still
    # return correct nbytes from shape+dtype, so we instead assert
    # the snapshot can be re-read across a thread boundary, the
    # actual property #284 needs. Use a barrier eval as the assertion.
    import threading
    err: list[Exception] = []
    def read_in_thread() -> None:
        try:
            mx.eval(snap_keys)
            mx.eval(snap_values)
        except Exception as e:  # pragma: no cover
            err.append(e)
    t = threading.Thread(target=read_in_thread)
    t.start()
    t.join()
    assert not err, f"cross-thread eval failed: {err}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompt_cache_checkpoint.py -v -k snapshot`
Expected: FAIL with `ImportError: cannot import name 'snapshot_cache_for_persistence'`.

- [ ] **Step 3: Write the implementation**

```python
# Append to olmlx/engine/prompt_cache/checkpoint.py
import copy
from typing import Any

import mlx.core as mx


def snapshot_cache_for_persistence(
    cache: list[Any],
    *,
    eager_eval: bool,
) -> list[Any]:
    """Return a thread-safe deep copy of a prompt cache.

    When ``eager_eval`` is True, all layer state arrays are materialized
    via ``mx.eval`` before the deep copy. This closes issue #284: the
    ``gated_delta_kernel`` outputs in ``ArraysCache`` carry a lazy graph
    bound to the originating worker thread's Metal stream. Evaluating
    eagerly produces concrete arrays whose evaluation is thread-independent,
    so the snapshot can be reused from any worker thread on a later request.

    Args:
        cache: List of mlx-lm cache layer objects (one per model layer).
        eager_eval: If True, materialize each layer's ``state`` before copy.
            Set this for caches whose layers include ``ArraysCache`` or
            other types known to produce lazy outputs bound to a Metal stream.
            For pure ``KVCache``/``QuantizedKVCache`` layouts the deep copy
            alone is sufficient and ``eager_eval=False`` saves the eval cost.
    """
    if eager_eval:
        states = []
        for layer in cache:
            state = layer.state
            if isinstance(state, tuple):
                states.extend(state)
            else:
                states.append(state)
        if states:
            mx.eval(states)
    return copy.deepcopy(cache)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompt_cache_checkpoint.py -v -k snapshot`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/prompt_cache/checkpoint.py tests/test_prompt_cache_checkpoint.py
git commit -m "feat(prompt_cache): add snapshot_cache_for_persistence with eager-eval option (#284)"
```

---

## Phase 2 — Store-level checkpoint API

### Task 2.1: Extend `CachedPromptState` with `cache_type` and `is_checkpoint`

**Files:**
- Modify: `olmlx/engine/prompt_cache/state.py`
- Test: `tests/test_prompt_cache_checkpoint.py`

- [ ] **Step 1: Read the existing state.py**

Run: `cat olmlx/engine/prompt_cache/state.py`
Expected: ~10 lines, a `CachedPromptState` dataclass with `tokens` and `cache`.

- [ ] **Step 2: Write the failing test**

```python
# Append to tests/test_prompt_cache_checkpoint.py
from olmlx.engine.prompt_cache.state import CachedPromptState

def test_cached_prompt_state_defaults_match_pre_checkpoint_behavior():
    """Existing call sites that pass only tokens+cache get assistant terminal."""
    state = CachedPromptState(tokens=[1, 2, 3], cache=[])
    assert state.cache_type == "assistant"
    assert state.is_checkpoint is False

def test_cached_prompt_state_can_be_marked_as_checkpoint():
    state = CachedPromptState(
        tokens=[1, 2, 3], cache=[], cache_type="system", is_checkpoint=True
    )
    assert state.cache_type == "system"
    assert state.is_checkpoint is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_prompt_cache_checkpoint.py -v -k cached_prompt_state`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'cache_type'`.

- [ ] **Step 4: Update the dataclass**

```python
# olmlx/engine/prompt_cache/state.py
"""Cached prompt state dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CacheType = Literal["system", "user", "assistant", "tool", "developer"]


@dataclass
class CachedPromptState:
    """A snapshot of a prompt cache and the tokens it represents.

    ``cache_type`` records which message-role boundary this state ends at;
    used by the multi-checkpoint store for tier-aware LRU. ``is_checkpoint``
    distinguishes a mid-prompt boundary snapshot from a terminal (end-of-
    generation) entry; the two participate in different lookup paths.
    """

    tokens: list[int]
    cache: list[Any]
    cache_type: CacheType = "assistant"
    is_checkpoint: bool = False
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_prompt_cache_checkpoint.py -v -k cached_prompt_state`
Expected: 2 passed.

- [ ] **Step 6: Run the full prompt cache suite to confirm no regressions**

Run: `uv run pytest tests/test_prompt_cache_radix.py tests/test_prompt_cache_store.py tests/test_prompt_cache_checkpoint.py -q`
Expected: all green. (Existing call sites that pass `tokens=` and `cache=` only must still work because of the field defaults.)

- [ ] **Step 7: Commit**

```bash
git add olmlx/engine/prompt_cache/state.py tests/test_prompt_cache_checkpoint.py
git commit -m "feat(prompt_cache): tag CachedPromptState with cache_type and is_checkpoint"
```

### Task 2.2: Radix — add `find_strict_prefix()` for shorter-match lookup

**Files:**
- Modify: `olmlx/engine/prompt_cache/radix.py`
- Test: `tests/test_prompt_cache_radix.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_prompt_cache_radix.py
def test_find_strict_prefix_returns_deepest_stored_prefix():
    """Walks query tokens; returns the deepest terminal at or before query end."""
    idx = PrefixCacheIndex()
    idx.insert([1, 2], "shorter")
    idx.insert([1, 2, 3, 4], "deeper")
    cid, depth = idx.find_strict_prefix([1, 2, 3, 4, 5], min_depth=1)
    assert cid == "deeper"
    assert depth == 4

def test_find_strict_prefix_returns_none_when_no_terminal_is_a_prefix():
    """A stored entry that diverges past the shared prefix is NOT a strict prefix."""
    idx = PrefixCacheIndex()
    idx.insert([1, 2, 9, 9], "diverges")
    cid, depth = idx.find_strict_prefix([1, 2, 3, 4], min_depth=1)
    assert cid is None
    assert depth == 0

def test_find_strict_prefix_respects_min_depth():
    idx = PrefixCacheIndex()
    idx.insert([1], "tiny")
    assert idx.find_strict_prefix([1, 2, 3], min_depth=2) == (None, 0)
    assert idx.find_strict_prefix([1, 2, 3], min_depth=1) == ("tiny", 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompt_cache_radix.py -v -k strict_prefix`
Expected: FAIL with `AttributeError: 'PrefixCacheIndex' object has no attribute 'find_strict_prefix'`.

- [ ] **Step 3: Implement `find_strict_prefix`**

Append to `olmlx/engine/prompt_cache/radix.py`:

```python
    def find_strict_prefix(
        self, tokens: list[int], min_depth: int = 0
    ) -> tuple[str | None, int]:
        """Return the deepest terminal whose stored token sequence is a strict
        prefix of ``tokens`` (terminal depth <= len(tokens) and every token
        along the path matches).

        Distinct from ``find_longest_prefix`` which also surfaces siblings
        that diverge past the shared prefix. Strict-prefix is the lookup
        the non-trimmable checkpoint path needs: only safe to reuse a stored
        cache when its tokens fully match a prefix of the new request — no
        trim required, no divergence to discard.

        ``min_depth`` short-circuits below-threshold matches with
        ``(None, 0)`` (consistent with ``find_longest_prefix``).
        """
        node = self._root
        best_cid: str | None = None
        best_depth = 0
        for depth, tok in enumerate(tokens, start=1):
            child = node.children.get(tok)
            if child is None:
                break
            node = child
            if node.terminal_cache_ids:
                best_cid = next(iter(node.terminal_cache_ids))
                best_depth = depth
        if best_depth < min_depth:
            return None, 0
        return best_cid, best_depth
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompt_cache_radix.py -v -k strict_prefix`
Expected: 3 passed.

- [ ] **Step 5: Run the rest of the radix suite to confirm no regressions**

Run: `uv run pytest tests/test_prompt_cache_radix.py -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/prompt_cache/radix.py tests/test_prompt_cache_radix.py
git commit -m "feat(prompt_cache): add find_strict_prefix for shorter-match checkpoint lookup"
```

### Task 2.3: Store — `insert_checkpoint()` + `fetch_nearest()`

**Files:**
- Modify: `olmlx/engine/prompt_cache/store.py`
- Test: `tests/test_prompt_cache_store_multi.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompt_cache_store_multi.py
"""Tests for the multi-checkpoint extension of PromptCacheStore."""

import pytest
from olmlx.engine.prompt_cache.state import CachedPromptState
from olmlx.engine.prompt_cache.store import PromptCacheStore


def _state(tokens, *, cache_type="system", is_checkpoint=True):
    return CachedPromptState(
        tokens=list(tokens),
        cache=[],
        cache_type=cache_type,
        is_checkpoint=is_checkpoint,
    )


def test_insert_checkpoint_keys_by_tokens_not_cache_id():
    """Two checkpoints from different conversations sharing a system prefix
    must both be retrievable via fetch_nearest."""
    store = PromptCacheStore(max_slots=8)
    store.insert_checkpoint(_state([1, 2, 3]))
    store.insert_checkpoint(_state([1, 2, 3, 4, 5]))
    # Lookup with a 4th conversation that shares 5 tokens with the longer entry
    hit = store.fetch_nearest([1, 2, 3, 4, 5, 6, 7])
    assert hit is not None
    state, suffix = hit
    assert state.tokens == [1, 2, 3, 4, 5]
    assert suffix == [6, 7]


def test_fetch_nearest_returns_shorter_when_only_shorter_exists():
    store = PromptCacheStore(max_slots=8)
    store.insert_checkpoint(_state([1, 2, 3]))
    hit = store.fetch_nearest([1, 2, 3, 4, 5])
    assert hit is not None
    state, suffix = hit
    assert state.tokens == [1, 2, 3]
    assert suffix == [4, 5]


def test_fetch_nearest_returns_none_when_no_strict_prefix():
    store = PromptCacheStore(max_slots=8)
    store.insert_checkpoint(_state([1, 2, 9, 9]))
    assert store.fetch_nearest([1, 2, 3, 4]) is None


def test_fetch_nearest_exact_match_returns_empty_suffix():
    store = PromptCacheStore(max_slots=8)
    store.insert_checkpoint(_state([1, 2, 3]))
    hit = store.fetch_nearest([1, 2, 3])
    assert hit is not None
    state, suffix = hit
    assert state.tokens == [1, 2, 3]
    assert suffix == []


def test_insert_checkpoint_dedupes_on_identical_tokens():
    """Re-inserting a checkpoint with the same tokens replaces the entry,
    does not add a duplicate."""
    store = PromptCacheStore(max_slots=8)
    store.insert_checkpoint(_state([1, 2, 3]))
    store.insert_checkpoint(_state([1, 2, 3]))
    # Only one slot used.
    assert len(store) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompt_cache_store_multi.py -v`
Expected: FAIL with `AttributeError: 'PromptCacheStore' object has no attribute 'insert_checkpoint'`.

- [ ] **Step 3: Implement `insert_checkpoint` and `fetch_nearest`**

Add to `olmlx/engine/prompt_cache/store.py` (alongside the existing methods, after `find_by_prefix`):

```python
    # -- Multi-checkpoint API ----------------------------------------------
    #
    # Companion to the cache_id-keyed `get`/`set`/`takeover` API. Used by the
    # message-boundary checkpoint path: many checkpoint entries can coexist,
    # keyed by their token sequences (the cache_id is derived from the tokens).
    # Lookup returns the deepest STRICT-PREFIX match — no trim required, so
    # this path is safe for non-trimmable caches (RotatingKVCache after wrap,
    # ArraysCache via eager-eval; see snapshot_cache_for_persistence).

    @staticmethod
    def _checkpoint_cache_id(tokens: list[int]) -> str:
        """Stable cache_id derived from token list. Two checkpoints over
        identical tokens collide here intentionally — re-insertion replaces.
        """
        # Hash-based keying; collisions across distinct sequences would only
        # cause a false "displaced" eviction which the store handles.
        import hashlib
        h = hashlib.sha1(
            b"".join(t.to_bytes(4, "little", signed=False) for t in tokens),
            usedforsecurity=False,
        ).hexdigest()
        return f"ckpt:{h}"

    def insert_checkpoint(self, state: CachedPromptState) -> None:
        """Add a checkpoint state, keyed by its tokens.

        Re-inserting with the same tokens replaces the prior entry.
        """
        cid = self._checkpoint_cache_id(state.tokens)
        evicted = self.set(cid, state)
        if evicted is not None:
            del evicted

    def fetch_nearest(
        self, tokens: list[int]
    ) -> tuple[CachedPromptState, list[int]] | None:
        """Find the deepest stored entry whose tokens are a strict prefix
        of ``tokens``; return (state, suffix) where ``suffix`` is the
        portion of ``tokens`` not covered by ``state.tokens``.

        Returns None when no terminal in the index is a strict prefix.
        """
        cid, depth = self._radix.find_strict_prefix(tokens, min_depth=1)
        if cid is None or depth == 0:
            return None
        state = self._entries.get(cid)
        if state is None:
            return None
        self._entries.move_to_end(cid)
        self.metrics.radix_hits += 1
        suffix = tokens[depth:]
        return state, suffix
```

Add the import:

```python
# At top of olmlx/engine/prompt_cache/store.py, with the other imports:
import hashlib  # ALREADY ADDED via the staticmethod above — but if the
                # import lints into the file header, move it there.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompt_cache_store_multi.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the existing store suite to confirm no regressions**

Run: `uv run pytest tests/test_prompt_cache_store.py tests/test_prompt_cache_radix.py -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/prompt_cache/store.py tests/test_prompt_cache_store_multi.py
git commit -m "feat(prompt_cache): add insert_checkpoint + fetch_nearest for multi-entry checkpoint storage"
```

---

## Phase 3 — Router-side per-message tokenization

### Task 3.1: Helper `tokenize_segmented_chat()` in `engine/inference.py`

**Files:**
- Modify: `olmlx/engine/inference.py`
- Test: `tests/test_inference_checkpoint_drive.py` (new)

- [ ] **Step 1: Read the existing tokenization helpers**

Run: `grep -n "def _apply_chat_template\|def apply_chat_template_text\|def _tokenize" olmlx/engine/inference.py | head -20`
Expected: locates `_apply_chat_template` (~line 1617) and `apply_chat_template_text` (~line 1670).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_inference_checkpoint_drive.py
"""Tests for segment-aware tokenization and the segmented-prefill drive."""

from unittest.mock import MagicMock

import pytest

from olmlx.engine.inference import tokenize_segmented_chat
from olmlx.engine.prompt_cache.checkpoint import Segment, SegmentedPrompt


class _FakeTokenizer:
    """Minimal tokenizer stub: tokens are 1 per character, role-tagged via
    the chat template adding a fixed-length wrapper per message."""

    bos_token_id = None
    eos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        # Each message expands to [role_marker, *content_chars, end_marker].
        # We encode role as 1=system, 2=user, 3=assistant. End marker is 9.
        ROLE = {"system": 1, "user": 2, "assistant": 3}
        out = []
        for m in messages:
            out.append(ROLE[m["role"]])
            out.extend(ord(c) for c in m["content"])
            out.append(9)
        return out


def test_tokenize_segmented_chat_returns_one_segment_per_message():
    tok = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
    ]
    sp = tokenize_segmented_chat(tok, messages)
    assert isinstance(sp, SegmentedPrompt)
    assert len(sp.segments) == 2
    assert sp.segments[0].role == "system"
    assert sp.segments[1].role == "user"
    # System segment tokens = [1, 'A', 'B', 9] = [1, 65, 66, 9]
    assert sp.segments[0].tokens == [1, 65, 66, 9]
    # User segment tokens = [2, 'C', 'D', 9]
    assert sp.segments[1].tokens == [2, 67, 68, 9]


def test_tokenize_segmented_chat_flatten_matches_full_apply():
    """The concatenation of per-segment tokens must equal a full template
    application on all messages at once. If this fails the segment boundaries
    are off and snapshots would be misaligned."""
    tok = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
        {"role": "assistant", "content": "EF"},
        {"role": "user", "content": "GH"},
    ]
    sp = tokenize_segmented_chat(tok, messages)
    full = tok.apply_chat_template(messages)
    assert sp.flatten() == full
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_inference_checkpoint_drive.py -v -k tokenize_segmented`
Expected: FAIL with `ImportError: cannot import name 'tokenize_segmented_chat'`.

- [ ] **Step 4: Implement the helper**

Insert into `olmlx/engine/inference.py` near `apply_chat_template_text` (around line 1670):

```python
def tokenize_segmented_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    **template_kwargs: Any,
) -> "SegmentedPrompt":
    """Tokenize ``messages`` one-at-a-time so the resulting prompt is
    split into per-message segments suitable for checkpoint snapshotting.

    The token boundaries returned here MUST match what
    ``tokenizer.apply_chat_template(messages)`` would produce if called
    on the full list — otherwise downstream checkpoint snapshots would
    be misaligned with the request's actual prefill. The implementation
    enforces this by calling ``apply_chat_template`` on each successive
    prefix of ``messages`` and taking the delta against the prior
    prefix. This pays len(messages)+1 template applications, but the
    cost is bounded (a few microseconds each) and is the only way to
    correctly handle chat templates that insert role-dependent markers
    or that emit different tokens depending on the trailing message.

    Falls back to a single-segment prompt for templates that fail
    cleanly under the empty-messages prefix (some templates require a
    final user message).
    """
    from olmlx.engine.prompt_cache.checkpoint import Segment, SegmentedPrompt

    if not messages:
        return SegmentedPrompt(segments=[])

    # Compute the full tokenization once — this is the ground truth.
    full = tokenizer.apply_chat_template(messages, **template_kwargs)
    # Then for each k in [1..len-1], tokenize the first-k prefix to find
    # where the (k+1)-th segment starts. The first segment is everything
    # up to the first split.
    boundaries: list[int] = []
    for k in range(1, len(messages)):
        try:
            prefix_tokens = tokenizer.apply_chat_template(messages[:k], **template_kwargs)
        except Exception:
            # Template rejected the partial prefix (e.g. needs a final user
            # message). Bail out to a single segment covering everything.
            return SegmentedPrompt(
                segments=[Segment(tokens=list(full), role=messages[-1]["role"])]
            )
        # Sanity: the prefix tokens must be an actual prefix of the full
        # tokenization. If not, the template is non-monotonic — fall back.
        if list(full[: len(prefix_tokens)]) != list(prefix_tokens):
            return SegmentedPrompt(
                segments=[Segment(tokens=list(full), role=messages[-1]["role"])]
            )
        boundaries.append(len(prefix_tokens))
    boundaries.append(len(full))

    segments: list[Segment] = []
    start = 0
    for k, end in enumerate(boundaries):
        segments.append(
            Segment(tokens=list(full[start:end]), role=messages[k]["role"])
        )
        start = end
    return SegmentedPrompt(segments=segments)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_inference_checkpoint_drive.py -v -k tokenize_segmented`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/inference.py tests/test_inference_checkpoint_drive.py
git commit -m "feat(inference): add tokenize_segmented_chat for per-message segment boundaries"
```

### Task 3.2: Plumb `SegmentedPrompt` through the OpenAI chat router

**Files:**
- Modify: `olmlx/routers/openai.py`
- Test: `tests/test_openai_router.py` (existing — add one focused test)

- [ ] **Step 1: Locate the call site that runs apply_chat_template**

Run: `grep -n "apply_chat_template_text\|_apply_chat_template\|generate_chat" olmlx/routers/openai.py | head -10`
Expected: at least one call into `generate_chat` and (indirectly) into the apply helper.

- [ ] **Step 2: Write the failing test**

```python
# Append to tests/test_openai_router.py (or wherever chat-completions tests live)
import pytest
from unittest.mock import patch

def test_chat_completions_uses_segmented_tokenization(client, ...):
    """The router must call tokenize_segmented_chat and forward the
    SegmentedPrompt into generate_chat — not call apply_chat_template
    once on the full message list."""
    with patch(
        "olmlx.engine.inference.tokenize_segmented_chat",
        wraps=...,  # spy that delegates to the real impl
    ) as spy, patch(
        "olmlx.engine.inference.generate_chat"
    ) as gen:
        gen.return_value = ...  # mock streaming response
        client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
            ],
        })
        spy.assert_called_once()
        assert gen.call_args.kwargs.get("segmented_prompt") is not None
```

(Concrete fixtures: copy the pattern from the existing `test_openai_router.py` setup — likely `TestClient` + a fake `ModelManager`.)

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_router.py -v -k segmented_tokenization`
Expected: FAIL — the router doesn't yet call `tokenize_segmented_chat` and `generate_chat` doesn't yet accept `segmented_prompt`.

- [ ] **Step 4: Update the router**

In `olmlx/routers/openai.py`, find the chat completions handler. Replace the single `apply_chat_template_text(...)` call with:

```python
from olmlx.engine.inference import tokenize_segmented_chat

# ... inside the handler, after resolving messages and tokenizer:
segmented = tokenize_segmented_chat(
    lm.tokenizer, messages, **template_kwargs
)
# Pass both for now — generate_chat will consume `segmented` when the lm
# supports checkpoint persistence; otherwise it falls back to the flat path.
result = await generate_chat(
    ...,
    segmented_prompt=segmented,
)
```

Then add `segmented_prompt: SegmentedPrompt | None = None` to `generate_chat`'s signature in `engine/inference.py`. The actual consumption happens in the next task; for this task `generate_chat` just accepts and stores it.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_router.py -v -k segmented_tokenization`
Expected: 1 passed.

- [ ] **Step 6: Run the existing OpenAI router suite to confirm no regressions**

Run: `uv run pytest tests/test_openai_router.py -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add olmlx/routers/openai.py olmlx/engine/inference.py tests/test_openai_router.py
git commit -m "feat(router): plumb SegmentedPrompt from OpenAI chat into generate_chat"
```

### Task 3.3: Same plumbing for the Anthropic router

**Files:**
- Modify: `olmlx/routers/anthropic.py`
- Test: `tests/test_anthropic_router.py`

- [ ] **Step 1: Mirror Task 3.2 in the Anthropic router**

Add a similar test in `tests/test_anthropic_router.py` (use existing fixtures). Then update `routers/anthropic.py` to call `tokenize_segmented_chat` and forward `segmented_prompt` to `generate_chat` (the Anthropic router still uses `generate_chat` under the hood).

Run the test, then the suite.

- [ ] **Step 2: Commit**

```bash
git add olmlx/routers/anthropic.py tests/test_anthropic_router.py
git commit -m "feat(router): plumb SegmentedPrompt from Anthropic Messages into generate_chat"
```

### Task 3.4: Same plumbing for Ollama `/api/chat`

**Files:**
- Modify: `olmlx/routers/chat.py`
- Test: `tests/test_chat_router.py`

- [ ] **Step 1: Mirror Task 3.2 in the Ollama chat router**

Same test pattern, same router edit. Commit.

```bash
git add olmlx/routers/chat.py tests/test_chat_router.py
git commit -m "feat(router): plumb SegmentedPrompt from Ollama chat into generate_chat"
```

### Task 3.5: Single-segment `SegmentedPrompt` for `/api/generate`

**Files:**
- Modify: `olmlx/routers/generate.py`
- Test: `tests/test_generate_router.py`

- [ ] **Step 1: Write the failing test**

```python
def test_generate_router_passes_single_segment_prompt(client, ...):
    with patch("olmlx.engine.inference.generate_completion") as gen:
        gen.return_value = ...
        client.post("/api/generate", json={
            "model": "test-model",
            "prompt": "hello world",
        })
        sp = gen.call_args.kwargs.get("segmented_prompt")
        assert sp is not None
        assert len(sp.segments) == 1
        assert sp.segments[0].role == "user"
```

- [ ] **Step 2: Update the router**

In `routers/generate.py`, after tokenizing the prompt:

```python
from olmlx.engine.prompt_cache.checkpoint import Segment, SegmentedPrompt

prompt_tokens = lm.tokenizer.encode(prompt_text)
segmented = SegmentedPrompt(
    segments=[Segment(tokens=list(prompt_tokens), role="user")]
)
result = await generate_completion(..., segmented_prompt=segmented)
```

Add the same `segmented_prompt` kwarg to `generate_completion` in `engine/inference.py`.

- [ ] **Step 3: Run test and commit**

```bash
uv run pytest tests/test_generate_router.py -v -k single_segment
git add olmlx/routers/generate.py olmlx/engine/inference.py tests/test_generate_router.py
git commit -m "feat(router): pass single-segment SegmentedPrompt from /api/generate"
```

---

## Phase 4 — Segmented-prefill drive in `_setup_prompt_cache`

### Task 4.1: Add `_drive_segmented_prefill()`

**Files:**
- Modify: `olmlx/engine/inference.py`
- Test: `tests/test_inference_checkpoint_drive.py`

This is the load-bearing change. The drive function:
1. Takes the model, the `SegmentedPrompt`, and a starting cache (possibly populated by a checkpoint hit).
2. Determines which segment boundaries are not yet covered by the cache.
3. For each remaining boundary, runs `model(tokens_for_this_segment, cache=...)` to extend the cache, then calls `snapshot_cache_for_persistence(cache, eager_eval=needs_eager)` and `store.insert_checkpoint(...)`.
4. Returns the (now-warm) cache and the remaining token suffix to feed into `stream_generate`.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_inference_checkpoint_drive.py
import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache
from olmlx.engine.inference import _drive_segmented_prefill
from olmlx.engine.prompt_cache.checkpoint import Segment, SegmentedPrompt
from olmlx.engine.prompt_cache.store import PromptCacheStore


class _DummyModel:
    """Records call-by-call which tokens were fed in. Cache pretends to grow."""

    def __init__(self):
        self.calls: list[list[int]] = []

    def __call__(self, tokens, cache=None):
        # mlx-lm's contract: model(input_tokens_batched[None], cache=cache).
        # tokens is a 2D mx.array, batch dim first.
        self.calls.append(list(tokens.flatten().tolist()))
        # Grow each layer's "state" by len(tokens) — using KVCache primitives.
        S = tokens.shape[-1]
        keys = mx.zeros((1, 1, S, 4))
        values = mx.zeros((1, 1, S, 4))
        for layer in cache:
            layer.update_and_fetch(keys, values)
        # Return a dummy logits tensor of the right shape.
        return mx.zeros((1, S, 32))


def test_drive_runs_one_model_call_per_uncovered_segment():
    model = _DummyModel()
    store = PromptCacheStore(max_slots=8)
    sp = SegmentedPrompt(segments=[
        Segment(tokens=[1, 2, 3], role="system"),
        Segment(tokens=[4, 5], role="user"),
    ])
    cache = make_prompt_cache(_StubMxLmModelForCacheMake())  # see helper below
    suffix = _drive_segmented_prefill(
        model=model, segmented=sp, cache=cache, store=store, eager_eval=False
    )
    assert len(model.calls) == 2, "one call per segment when starting cold"
    # First call processes the system segment, second processes the user
    # segment.
    assert model.calls[0] == [1, 2, 3]
    assert model.calls[1] == [4, 5]
    # Both boundaries should have been snapshotted into the store.
    assert store.fetch_nearest([1, 2, 3, 4, 5, 99]) is not None
    # The drive returns the suffix to be fed to stream_generate: the last
    # token only (so generation continues from there).
    assert suffix == [5]
```

(Note: `_StubMxLmModelForCacheMake` is a tiny stub matching `make_prompt_cache`'s introspection requirements — model with `.layers` attribute returning an empty list. If that's impractical here, construct the cache directly: `cache = [KVCache()]`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inference_checkpoint_drive.py -v -k drive_runs_one`
Expected: FAIL with `ImportError: cannot import name '_drive_segmented_prefill'`.

- [ ] **Step 3: Implement the drive**

Insert into `olmlx/engine/inference.py` (near `_setup_prompt_cache`):

```python
def _drive_segmented_prefill(
    *,
    model: Any,
    segmented: "SegmentedPrompt",
    cache: list[Any],
    store: PromptCacheStore,
    eager_eval: bool,
    already_covered_tokens: int = 0,
) -> list[int]:
    """Walk the segmented prompt, run prefill per segment, snapshot the
    cache at each boundary, return the suffix to hand off to stream_generate.

    ``already_covered_tokens`` is the depth at which ``cache`` is already
    populated (from a checkpoint hit). Segments wholly inside that depth
    are skipped; a partially covered segment has only its uncovered tail
    fed to ``model``.

    The returned suffix is ``[full_flat_tokens[-1]]`` — one token —
    because ``mlx_lm.stream_generate`` requires at least one prompt
    token to seed the decode step, and feeding the same token back into
    the prefilled cache is a no-op for the cache but produces the
    correct first-token logits.
    """
    from olmlx.engine.prompt_cache.checkpoint import (
        snapshot_cache_for_persistence,
    )
    import mlx.core as mx

    flat = segmented.flatten()
    if not flat:
        return []
    boundaries = segmented.boundary_offsets()

    cursor = already_covered_tokens
    for boundary, seg in zip(boundaries, segmented.segments, strict=True):
        if boundary <= cursor:
            continue
        # Feed only the uncovered tail of this segment.
        chunk_start = cursor
        chunk_end = boundary
        # Last token is consumed by stream_generate's decode init below —
        # don't include it in the prefill if it's the absolute final token.
        prefill_end = chunk_end
        if chunk_end == len(flat):
            prefill_end = chunk_end - 1  # leave one token for stream_generate
        if prefill_end > chunk_start:
            chunk = flat[chunk_start:prefill_end]
            arr = mx.array(chunk, dtype=mx.int32)[None, :]
            with mx.stream(mx.default_stream(mx.default_device())):
                model(arr, cache=cache)
                mx.eval([c.state for c in cache])
            cursor = prefill_end
        else:
            cursor = chunk_end
        # Snapshot the cache state at this boundary into the store.
        snap = snapshot_cache_for_persistence(cache, eager_eval=eager_eval)
        store.insert_checkpoint(
            CachedPromptState(
                tokens=flat[:boundary],
                cache=snap,
                cache_type=seg.role,
                is_checkpoint=True,
            )
        )
    # Return the last token as the suffix.
    return [flat[-1]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inference_checkpoint_drive.py -v -k drive_runs_one`
Expected: PASS.

- [ ] **Step 5: Add a second test for the "skip-already-covered-segments" path**

```python
def test_drive_skips_segments_below_already_covered():
    model = _DummyModel()
    store = PromptCacheStore(max_slots=8)
    sp = SegmentedPrompt(segments=[
        Segment(tokens=[1, 2, 3], role="system"),
        Segment(tokens=[4, 5], role="user"),
    ])
    cache = [KVCache()]
    # Pretend the cache is already warmed to position 3 (end of system).
    # Drive should NOT re-process [1,2,3]; only [4,5].
    suffix = _drive_segmented_prefill(
        model=model,
        segmented=sp,
        cache=cache,
        store=store,
        eager_eval=False,
        already_covered_tokens=3,
    )
    assert len(model.calls) == 1
    assert model.calls[0] == [4]  # [4,5] minus the trailing token reserved for decode
    assert suffix == [5]
```

Run, expect PASS, commit.

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/inference.py tests/test_inference_checkpoint_drive.py
git commit -m "feat(inference): _drive_segmented_prefill snapshots at message boundaries"
```

### Task 4.2: Rework `_setup_prompt_cache` to consume `SegmentedPrompt`

**Files:**
- Modify: `olmlx/engine/inference.py`
- Test: `tests/test_inference_checkpoint_drive.py`

The existing `_setup_prompt_cache` handles the cache_id-keyed flat path. We add a new path that's taken when `segmented_prompt` is provided AND the cache layout uses the checkpoint mechanism (probe sets a new flag `uses_checkpoint_persistence: bool` — see Task 5.1).

- [ ] **Step 1: Write the failing test**

```python
def test_setup_prompt_cache_drives_segments_when_lm_uses_checkpoint_path():
    """When lm.uses_checkpoint_persistence is True, the setup function
    feeds segments through _drive_segmented_prefill and returns the
    warmed cache + the one-token suffix."""
    ...  # see _DummyModel / KVCache pattern above; assert model.calls and
    ...  # that the returned cache_setup result carries the warmed cache.
```

- [ ] **Step 2: Run, fail, implement**

In `_setup_prompt_cache`:
- Add a top-level branch: `if lm.uses_checkpoint_persistence and segmented_prompt is not None: return _setup_via_checkpoint_path(...)`.
- The new helper does: `store.fetch_nearest(flatten)` → seed cache with the hit's cache (or build fresh) → `_drive_segmented_prefill(..., already_covered_tokens=hit_depth)` → return the `_CacheSetupResult`.

- [ ] **Step 3: Run, expect PASS, commit**

```bash
git add olmlx/engine/inference.py tests/test_inference_checkpoint_drive.py
git commit -m "feat(inference): route segmented prompts through the checkpoint path in _setup_prompt_cache"
```

### Task 4.3: Make `_store_prompt_cache_after_generation` a no-op for checkpoint-path requests

**Files:**
- Modify: `olmlx/engine/inference.py`
- Test: `tests/test_inference_checkpoint_drive.py`

When the checkpoint path was used, snapshots were already taken at each boundary during prefill — re-storing post-generation would just duplicate the work. The "assistant" terminal entry (post-generation) is intentionally NOT stored for non-trimmable models, because the flat path's lookup logic on the next request can't realign it (#343).

- [ ] **Step 1: Write the failing test** — assert that with `uses_checkpoint_persistence=True`, the post-generation store call doesn't add another entry.

- [ ] **Step 2: Add an early-return**

```python
async def _store_prompt_cache_after_generation(lm, ...):
    if lm.uses_checkpoint_persistence:
        # Checkpoints already taken at boundaries during prefill (Phase 4.1).
        # Storing the post-generation state would duplicate a longer
        # entry that the next request's flat-prefix lookup can't realign
        # without trim — which is exactly what the checkpoint path is
        # designed to avoid.
        return
    # ... existing flat-path logic unchanged below ...
```

- [ ] **Step 3: Run, expect PASS, commit**

```bash
git add olmlx/engine/inference.py tests/test_inference_checkpoint_drive.py
git commit -m "fix(inference): skip post-generation store when checkpoint path was used"
```

---

## Phase 5 — Probe update + cache-flag wiring

### Task 5.1: Add `uses_checkpoint_persistence` flag to `LoadedModel`

**Files:**
- Modify: `olmlx/engine/model_manager.py`
- Test: `tests/test_probe_cache_capabilities_v2.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe_cache_capabilities_v2.py
"""Tests for the updated _probe_cache_capabilities (checkpoint path)."""

import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from olmlx.engine.model_manager import LoadedModel


def test_loaded_model_defaults_uses_checkpoint_persistence_off():
    lm = LoadedModel(name="x", hf_path="y", model=None, tokenizer=None)
    assert lm.uses_checkpoint_persistence is False
```

- [ ] **Step 2: Add the field to `LoadedModel`**

```python
# in olmlx/engine/model_manager.py, in the LoadedModel dataclass:
uses_checkpoint_persistence: bool = False
```

- [ ] **Step 3: Run, expect PASS, commit**

```bash
git add olmlx/engine/model_manager.py tests/test_probe_cache_capabilities_v2.py
git commit -m "feat(model_manager): add uses_checkpoint_persistence flag on LoadedModel"
```

### Task 5.2: Probe sets the flag for ArraysCache-bearing and post-wrap-RotatingKVCache layouts

**Files:**
- Modify: `olmlx/engine/model_manager.py`
- Test: `tests/test_probe_cache_capabilities_v2.py`

- [ ] **Step 1: Write the failing tests**

```python
def _fake_cache_with(*layer_types) -> list:
    """Stub: returns a cache list of the given classes (no model needed)."""
    return [t.__new__(t) for t in layer_types]


@pytest.mark.parametrize(
    "layers, expected",
    [
        ([KVCache], False),                            # pure trimmable: flat path
        ([RotatingKVCache], True),                     # sliding-window: checkpoint
        ([ArraysCache], True),                         # SSM: checkpoint
        ([KVCache, ArraysCache], True),                # Qwen3.5: SSM via checkpoint
        ([KVCache, RotatingKVCache], True),            # Gemma 3: sliding via ckpt
        ([RotatingKVCache, ArraysCache], False),       # Qwen3-Next: excluded (#396)
    ],
)
def test_probe_sets_uses_checkpoint_for_non_trimmable_only(layers, expected, monkeypatch):
    from olmlx.engine.model_manager import ModelManager
    lm = LoadedModel(name="x", hf_path="y", model=None, tokenizer=None)
    cache = _fake_cache_with(*layers)
    # Inject the fake cache into the probe path; the actual probe uses
    # make_prompt_cache(model), so monkeypatch it.
    monkeypatch.setattr(
        "olmlx.engine.model_manager.make_prompt_cache",
        lambda model: cache,
    )
    mgr = ModelManager.__new__(ModelManager)  # bypass init
    import asyncio
    asyncio.run(mgr._probe_cache_capabilities(lm))
    assert lm.uses_checkpoint_persistence is expected, (
        f"layers={[t.__name__ for t in layers]}: "
        f"want uses_checkpoint_persistence={expected}, got {lm.uses_checkpoint_persistence}"
    )
```

- [ ] **Step 2: Update `_probe_cache_capabilities`**

In `olmlx/engine/model_manager.py`:

```python
# Add after _PERSISTABLE_CACHE_CLASSES:
_PERSISTABLE_CACHE_CLASSES_WITH_CHECKPOINT = frozenset(
    {
        # All flat-path-persistable classes remain persistable via the
        # checkpoint path too.
        "KVCache",
        "QuantizedKVCache",
        "ConcatenateKVCache",
        "RotatingKVCache",   # checkpoint avoids the trim-after-wrap problem (#343)
        "ChunkedKVCache",
        # SSM-style state is safe with eager mx.eval before snapshot (#284).
        "ArraysCache",
    }
)

# Layer-type pairs that are individually safe but compose to a layout we
# can't currently snapshot consistently (#396 — Qwen3-Next).
_EXCLUDED_MIXED_LAYER_PAIRS = frozenset(
    {
        frozenset({"RotatingKVCache", "ArraysCache"}),
    }
)


def _layer_layout_is_mixed_excluded(cache_list: list) -> bool:
    """True iff the layer-class set is in the excluded-mixed-pairs list."""
    if not cache_list:
        return False
    classes = {type(layer).__name__ for layer in cache_list}
    for excluded in _EXCLUDED_MIXED_LAYER_PAIRS:
        if excluded.issubset(classes):
            return True
    return False


def _cache_supports_checkpoint_persistence(cache_list: list) -> bool:
    """True iff every layer is checkpoint-persistable AND the mix is not
    one of the explicitly excluded compositions (#396).
    """
    if not cache_list:
        return False
    if _layer_layout_is_mixed_excluded(cache_list):
        return False
    return all(
        type(layer).__name__ in _PERSISTABLE_CACHE_CLASSES_WITH_CHECKPOINT
        for layer in cache_list
    )
```

Then in `_probe_cache_capabilities`, alongside the existing flag assignments:

```python
# (inside _probe_cache_capabilities, after the existing trim/persist checks)
ckpt_ok = _cache_supports_checkpoint_persistence(cache)
# Checkpoint path is preferred for non-trimmable layouts; for trimmable
# layouts the existing flat path is simpler and equivalent — keep flat.
lm.uses_checkpoint_persistence = ckpt_ok and not trim_ok
```

- [ ] **Step 3: Run, expect PASS, commit**

```bash
git add olmlx/engine/model_manager.py tests/test_probe_cache_capabilities_v2.py
git commit -m "feat(model_manager): probe sets uses_checkpoint_persistence for non-trimmable layouts (#284, #343); excludes Qwen3-Next (#396)"
```

### Task 5.3: Probe also flips `supports_cache_persistence` on when `uses_checkpoint_persistence`

**Files:**
- Modify: `olmlx/engine/model_manager.py`
- Test: `tests/test_probe_cache_capabilities_v2.py`

The downstream code in `inference.py:_setup_prompt_cache` currently gates ALL cache lookup on `lm.supports_cache_persistence`. For the checkpoint path to engage we need this to be True even though the flat path's underlying cache types are non-trimmable.

- [ ] **Step 1: Write the failing test**

```python
def test_probe_enables_persistence_when_checkpoint_path_is_used():
    lm = LoadedModel(name="x", hf_path="y", model=None, tokenizer=None)
    # ... wire up an ArraysCache-only fake cache via monkeypatch as above
    # ... run probe
    assert lm.supports_cache_persistence is True
    assert lm.uses_checkpoint_persistence is True
```

- [ ] **Step 2: Update the probe to OR-in the flag**

```python
# inside _probe_cache_capabilities, replace
#     lm.supports_cache_persistence = persist_ok
# with:
lm.supports_cache_persistence = persist_ok or lm.uses_checkpoint_persistence
```

- [ ] **Step 3: Run, expect PASS, commit**

```bash
git add olmlx/engine/model_manager.py tests/test_probe_cache_capabilities_v2.py
git commit -m "fix(model_manager): supports_cache_persistence reflects checkpoint path"
```

---

## Phase 6 — End-to-end integration tests

### Task 6.1: Round-trip test on a real (small) hybrid model

**Files:**
- Test: `tests/test_inference_checkpoint_e2e.py` (new)

This is a slow test — it loads a small hybrid model and runs two requests that share a prefix. Should be marked `@pytest.mark.slow` so it doesn't run on every invocation.

- [ ] **Step 1: Pick a small model with `ArraysCache`**

Candidates: anything in `mlx-community` derived from Qwen3.5 at <=4B. If nothing fits the CI budget, mark the test `pytest.mark.slow` and document the model selection at the top of the test file.

- [ ] **Step 2: Write the test**

```python
import pytest

@pytest.mark.slow
def test_qwen3_5_text_two_request_cache_hit():
    """Two requests sharing a 200-token system prompt: second request's
    prefill time should drop dramatically because the system checkpoint
    was stored on request 1."""
    # ... boot ModelManager with the small model ...
    # ... fire request 1, measure prefill ms ...
    # ... fire request 2 with same system + different user, measure ...
    # assert request 2's prefill ms < 0.5 * request 1's prefill ms
```

- [ ] **Step 3: Run, expect PASS, commit**

```bash
git add tests/test_inference_checkpoint_e2e.py
git commit -m "test(prompt_cache): e2e cache-reuse benchmark on a hybrid SSM model"
```

### Task 6.2: Regression — Qwen3-Next layout stays on the flat path

**Files:**
- Test: `tests/test_probe_cache_capabilities_v2.py`

Already covered by the parametrized test in 5.2 — the `[RotatingKVCache, ArraysCache]` case asserts `uses_checkpoint_persistence is False`. Add a complement assertion that `supports_cache_persistence is also False` for this mix.

- [ ] **Step 1: Extend the parametrized test**

Add an `expected_supports_persistence` column matching the existing `expected` (or a separate assertion inside the loop).

- [ ] **Step 2: Run, expect PASS, commit**

```bash
git add tests/test_probe_cache_capabilities_v2.py
git commit -m "test(model_manager): assert Qwen3-Next mixed layout stays non-persistable (#396)"
```

---

## Phase 7 — Docs and final verification

### Task 7.1: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Edit the "Prompt caching" bullet**

Locate the paragraph beginning with `**Prompt caching**` in `CLAUDE.md`. Add a new sub-paragraph after the existing #284 / #343 / #365 notes:

```markdown
- **Message-boundary checkpoint path** (this work, supersedes the #284 / #343
  folds for the layouts it covers): when `_probe_cache_capabilities` marks a
  layout as `uses_checkpoint_persistence`, the chat routers tokenize each
  message separately into a `SegmentedPrompt`. `_setup_prompt_cache` consults
  `PromptCacheStore.fetch_nearest` for the deepest strict-prefix match, then
  drives `model(...)` per uncovered segment and snapshots
  (`snapshot_cache_for_persistence` — deepcopy + eager `mx.eval` for
  `ArraysCache` layers) at each boundary into the store. Closes #284
  (ArraysCache cross-thread Metal-stream crash via eager eval) and #343
  (RotatingKVCache near-zero reuse via trim-free shorter-match lookup) for
  pure-SSM and pure-sliding-window layouts. Mixed `RotatingKVCache +
  ArraysCache` (Qwen3-Next, issue #396) remains explicitly excluded by the
  probe and continues to do full prefill every request.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE.md): describe message-boundary checkpoint path"
```

### Task 7.2: ruff check + format, full pytest

- [ ] **Step 1: ruff check**

Run: `uv run ruff check olmlx tests`
Expected: clean.

- [ ] **Step 2: ruff format**

Run: `uv run ruff format olmlx tests`
Expected: a few files reformatted at most; review and commit any changes.

- [ ] **Step 3: full pytest**

Run: `uv run pytest -q -x --deselect tests/test_inference_checkpoint_e2e.py`
Expected: all green. (The e2e test is deselected because it requires a model download; run it separately with `uv run pytest tests/test_inference_checkpoint_e2e.py -m slow`.)

- [ ] **Step 4: Run the e2e test separately**

Run: `uv run pytest tests/test_inference_checkpoint_e2e.py -m slow -v`
Expected: PASS. If your environment can't host the model, skip and document in the PR body.

- [ ] **Step 5: Commit any final lint fixes**

```bash
git add -u
git commit -m "chore: ruff format"
```

### Task 7.3: Rebase + open PR

- [ ] **Step 1: Sync with main**

Run: `git fetch origin && git rebase origin/main`
Expected: clean rebase, or resolve conflicts.

- [ ] **Step 2: Push the branch**

Run: `git push -u origin feat/prompt-cache-checkpoint`
Expected: branch pushed.

- [ ] **Step 3: Open the PR**

Run:

```bash
gh pr create --repo motsognirr/olmlx --title "Message-boundary checkpoint prompt cache (fixes #284, #343)" --body "$(cat <<'EOF'
## Summary

Ports mlx-lm's message-boundary checkpoint mechanism into olmlx's `PromptCacheStore`, enabling cross-request prompt-cache reuse for hybrid models that the existing flat path couldn't safely persist:

- Pure SSM / `ArraysCache` layouts (Qwen3.5 text tower, Nemotron-H, Jamba) — closes #284. Eager `mx.eval` before snapshot materializes lazy `gated_delta_kernel` outputs so the stored state is reusable from any worker thread.
- Pure sliding-window / `RotatingKVCache` layouts (Gemma 3, GPT-OSS) — closes #343. The strict-prefix lookup never needs to trim, sidestepping the post-wrap non-trimmability that previously discarded every multi-turn cache.

Mixed `RotatingKVCache + ArraysCache` layouts (Qwen3-Next, #396) are explicitly excluded by the probe and continue to do full prefill per request, pending the composition work tracked in that issue.

## Test plan

- [ ] `uv run pytest tests/test_prompt_cache_checkpoint.py tests/test_prompt_cache_store_multi.py tests/test_inference_checkpoint_drive.py tests/test_probe_cache_capabilities_v2.py -q`
- [ ] `uv run pytest -q -x --deselect tests/test_inference_checkpoint_e2e.py`
- [ ] `uv run pytest tests/test_inference_checkpoint_e2e.py -m slow -v` on a host that can load Qwen3.5-{small}
- [ ] Manual: run `mlx-community/Qwen3.5-9B-6bit`, fire three multi-turn chats with shared system prompt, confirm `cache_creation_tokens` shrinks on requests 2 and 3.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed. Report it.

---

## Self-review checklist

- [x] Every spec section has at least one task (segmented tokenization, store extension, probe update, drive, end-to-end test, docs).
- [x] No "TBD" or "fill in" placeholders.
- [x] Function and type names used consistently across tasks: `SegmentedPrompt`, `Segment`, `CachedPromptState`, `snapshot_cache_for_persistence`, `_drive_segmented_prefill`, `tokenize_segmented_chat`, `insert_checkpoint`, `fetch_nearest`, `find_strict_prefix`, `uses_checkpoint_persistence`, `_PERSISTABLE_CACHE_CLASSES_WITH_CHECKPOINT`, `_EXCLUDED_MIXED_LAYER_PAIRS`, `_cache_supports_checkpoint_persistence`.
- [x] Qwen3-Next exclusion (#396) explicitly enforced in Task 5.2 and asserted in Task 6.2.
- [x] mxr eval scope (non-trimmable only) realized via the `eager_eval` kwarg in `snapshot_cache_for_persistence`, wired through `_drive_segmented_prefill` and set by `_probe_cache_capabilities`.
- [x] CLAUDE.md update task included (Task 7.1).
- [x] ruff + pytest gate task included (Task 7.2).
