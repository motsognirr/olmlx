# Cross-request Radix Prompt Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a token-prefix radix index over the existing per-`cache_id` `PromptCacheStore` so sibling continuations (same system prompt, different `cache_id`) reuse KV state via takeover instead of full prefill, and expose hit/miss metrics.

**Architecture:** New `olmlx/engine/prompt_cache/` package holding the moved `CachedPromptState`/`PromptCacheStore`, a new `PrefixCacheIndex` trie, and `CacheMetrics`. `_setup_prompt_cache` in `inference.py` gains a single fallback block: on `cache_id` miss, walk the trie for the longest token-prefix match, then re-key the entry to the new `cache_id` and proceed through the existing trim+extend path unchanged.

**Tech Stack:** Python 3.11+, mlx-lm `KVCache`, existing `OrderedDict` LRU. No new third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-05-28-radix-prompt-cache-design.md`

---

## File Map

| File | Responsibility | Action |
| --- | --- | --- |
| `olmlx/engine/prompt_cache/__init__.py` | Re-exports `PromptCacheStore`, `CachedPromptState`, `CacheMetrics` | Create |
| `olmlx/engine/prompt_cache/state.py` | `CachedPromptState` dataclass | Create (move from `model_manager.py`) |
| `olmlx/engine/prompt_cache/radix.py` | `PrefixCacheIndex` trie + `_TrieNode` | Create |
| `olmlx/engine/prompt_cache/metrics.py` | `CacheMetrics` counters dataclass | Create |
| `olmlx/engine/prompt_cache/store.py` | `PromptCacheStore` with radix + metrics | Create (move from `model_manager.py`) |
| `olmlx/engine/model_manager.py` | Re-export shim, delete moved classes | Modify |
| `olmlx/engine/inference.py` | Add radix fallback to `_setup_prompt_cache` | Modify |
| `olmlx/config.py` | Add `prompt_cache_radix*`, `prompt_cache_ram_budget_gb` settings | Modify |
| `olmlx/routers/status.py` | Surface `cache_metrics` on `/api/ps` | Modify |
| `olmlx/schemas/status.py` | Add `cache_metrics: dict` field on `RunningModel` | Modify |
| `tests/test_prompt_cache_radix.py` | Unit + integration tests for new behaviour | Create |
| `CLAUDE.md` | Update Prompt caching bullet | Modify |

---

## Task 1: Settings

**Files:**
- Modify: `olmlx/config.py:98-103` (insert after existing `prompt_cache_*` settings)
- Test: `tests/test_config.py` (verify no regression — existing test file)

- [ ] **Step 1: Add settings**

Edit `olmlx/config.py`, locate the block at lines 98–103:

```python
    prompt_cache: bool = True
    prompt_cache_max_tokens: Annotated[int, Field(gt=0)] | None = 32768
    prompt_cache_max_slots: Annotated[int, Field(gt=0)] = 4
    prompt_cache_disk: bool = False
    prompt_cache_disk_path: Path = Path.home() / ".olmlx" / "cache" / "kv"
    prompt_cache_disk_max_gb: Annotated[float, Field(gt=0)] = 10.0
```

Append after `prompt_cache_disk_max_gb`:

```python
    # Cross-request radix prefix cache (issue #365). When enabled, a
    # cache_id miss falls back to a token-prefix lookup over the in-memory
    # store; on hit, the matched entry is re-keyed to the new cache_id
    # (takeover semantics — no KV copy). The old cache_id loses its entry.
    prompt_cache_radix: bool = True
    # Soft RAM byte budget for the in-memory tier. Best-effort estimate;
    # slot count (prompt_cache_max_slots) is the hard cap.
    prompt_cache_ram_budget_gb: Annotated[float, Field(gt=0)] = 8.0
    # Below this token count, a radix-prefix hit falls back to fresh
    # prefill rather than taking over a near-empty match.
    prompt_cache_radix_min_prefix_tokens: Annotated[int, Field(ge=0)] = 256
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `uv run python -c "from olmlx.config import settings; print(settings.prompt_cache_radix, settings.prompt_cache_ram_budget_gb, settings.prompt_cache_radix_min_prefix_tokens)"`

Expected output: `True 8.0 256`

- [ ] **Step 3: Commit**

```bash
git add olmlx/config.py
git commit -m "feat(config): add prompt_cache_radix settings (#365)"
```

---

## Task 2: `PrefixCacheIndex` — tests

**Files:**
- Test: `tests/test_prompt_cache_radix.py` (create)
- Source target (not yet existing): `olmlx/engine/prompt_cache/radix.py`

- [ ] **Step 1: Write the failing trie tests**

Create `tests/test_prompt_cache_radix.py`:

```python
"""Tests for the cross-request radix prefix cache (issue #365)."""

from olmlx.engine.prompt_cache.radix import PrefixCacheIndex


class TestPrefixCacheIndexInsertLookup:
    def test_empty_index_returns_no_match(self):
        idx = PrefixCacheIndex()
        assert idx.find_longest_prefix([1, 2, 3]) == (None, 0)

    def test_single_entry_exact_match(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        assert idx.find_longest_prefix([1, 2, 3]) == ("a", 3)

    def test_single_entry_longer_query(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        assert idx.find_longest_prefix([1, 2, 3, 4, 5]) == ("a", 3)

    def test_single_entry_shorter_query(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        # No terminal at depth 2 → no match
        assert idx.find_longest_prefix([1, 2]) == (None, 0)

    def test_no_overlap_returns_no_match(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        assert idx.find_longest_prefix([9, 8, 7]) == (None, 0)

    def test_two_entries_sibling_branches(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3, 7], "a")
        idx.insert([1, 2, 3, 8], "b")
        assert idx.find_longest_prefix([1, 2, 3, 7, 9]) == ("a", 4)
        assert idx.find_longest_prefix([1, 2, 3, 8, 9]) == ("b", 4)

    def test_returns_deepest_terminal_on_chain(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "short")
        idx.insert([1, 2, 3, 4, 5], "long")
        # Query that fully matches the long entry returns the long one
        assert idx.find_longest_prefix([1, 2, 3, 4, 5, 6]) == ("long", 5)
        # Query that only matches the short entry returns the short one
        assert idx.find_longest_prefix([1, 2, 3, 9]) == ("short", 3)


class TestPrefixCacheIndexRemove:
    def test_remove_then_lookup_misses(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        idx.remove([1, 2, 3], "a")
        assert idx.find_longest_prefix([1, 2, 3]) == (None, 0)

    def test_remove_unknown_is_noop(self):
        idx = PrefixCacheIndex()
        idx.remove([1, 2, 3], "a")  # should not raise
        assert idx.find_longest_prefix([1, 2, 3]) == (None, 0)

    def test_remove_one_sibling_keeps_other(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3, 7], "a")
        idx.insert([1, 2, 3, 8], "b")
        idx.remove([1, 2, 3, 7], "a")
        assert idx.find_longest_prefix([1, 2, 3, 7, 9]) == (None, 0)
        assert idx.find_longest_prefix([1, 2, 3, 8, 9]) == ("b", 4)

    def test_remove_deep_keeps_shallow_terminal(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "short")
        idx.insert([1, 2, 3, 4, 5], "long")
        idx.remove([1, 2, 3, 4, 5], "long")
        assert idx.find_longest_prefix([1, 2, 3, 4]) == ("short", 3)

    def test_overwrite_same_path_replaces_terminal(self):
        idx = PrefixCacheIndex()
        idx.insert([1, 2, 3], "a")
        idx.insert([1, 2, 3], "b")
        assert idx.find_longest_prefix([1, 2, 3]) == ("b", 3)
        idx.remove([1, 2, 3], "b")
        assert idx.find_longest_prefix([1, 2, 3]) == (None, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompt_cache_radix.py -v`

Expected: every test fails with `ModuleNotFoundError: No module named 'olmlx.engine.prompt_cache'`.

---

## Task 3: `PrefixCacheIndex` — implementation

**Files:**
- Create: `olmlx/engine/prompt_cache/__init__.py`
- Create: `olmlx/engine/prompt_cache/radix.py`

- [ ] **Step 1: Create package init**

Create `olmlx/engine/prompt_cache/__init__.py` (empty for now — we'll add re-exports in later tasks):

```python
"""Cross-request prompt cache: RAM tier with radix prefix index + disk spill.

Issue #365.
"""
```

- [ ] **Step 2: Implement the trie**

Create `olmlx/engine/prompt_cache/radix.py`:

```python
"""Token-prefix trie for cross-request prompt cache lookup (issue #365)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    terminal_cache_id: str | None = None


class PrefixCacheIndex:
    """Trie over token IDs. Each terminal node maps to one cache_id.

    Lookups return the deepest terminal that lies on the descent path of
    the query tokens (longest-prefix match).

    Complexity: insert/remove/find are O(len(tokens)) with O(1) per step.
    """

    def __init__(self) -> None:
        self._root = _TrieNode()

    def insert(self, tokens: list[int], cache_id: str) -> None:
        """Mark the path tokens[0..len-1] as a terminal for cache_id.

        Overwrites any existing terminal at the same path.
        """
        node = self._root
        for tok in tokens:
            child = node.children.get(tok)
            if child is None:
                child = _TrieNode()
                node.children[tok] = child
            node = child
        node.terminal_cache_id = cache_id

    def find_longest_prefix(self, tokens: list[int]) -> tuple[str | None, int]:
        """Walk the trie matching tokens, return the deepest terminal seen.

        Returns (cache_id, prefix_len) or (None, 0) if no terminal lies on
        the descent path.
        """
        node = self._root
        best_id: str | None = None
        best_depth = 0
        for depth, tok in enumerate(tokens, start=1):
            child = node.children.get(tok)
            if child is None:
                break
            node = child
            if node.terminal_cache_id is not None:
                best_id = node.terminal_cache_id
                best_depth = depth
        return best_id, best_depth

    def remove(self, tokens: list[int], cache_id: str) -> None:
        """Clear the terminal at the tokens path if it matches cache_id,
        then prune now-empty branches upward.

        No-op if the path doesn't exist or the terminal belongs to a
        different cache_id.
        """
        path: list[tuple[_TrieNode, int]] = []
        node = self._root
        for tok in tokens:
            child = node.children.get(tok)
            if child is None:
                return
            path.append((node, tok))
            node = child
        if node.terminal_cache_id != cache_id:
            return
        node.terminal_cache_id = None
        # Prune upward while nodes are empty.
        for parent, tok in reversed(path):
            child = parent.children[tok]
            if child.children or child.terminal_cache_id is not None:
                return
            del parent.children[tok]
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompt_cache_radix.py -v`

Expected: all 11 trie tests pass.

- [ ] **Step 4: Commit**

```bash
git add olmlx/engine/prompt_cache/__init__.py olmlx/engine/prompt_cache/radix.py tests/test_prompt_cache_radix.py
git commit -m "feat(prompt_cache): add PrefixCacheIndex trie (#365)"
```

---

## Task 4: `CacheMetrics`

**Files:**
- Test: `tests/test_prompt_cache_radix.py` (append)
- Create: `olmlx/engine/prompt_cache/metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompt_cache_radix.py`:

```python
from olmlx.engine.prompt_cache.metrics import CacheMetrics


class TestCacheMetrics:
    def test_defaults_are_zero(self):
        m = CacheMetrics()
        assert m.cache_id_hits == 0
        assert m.cache_id_misses == 0
        assert m.radix_hits == 0
        assert m.radix_misses == 0
        assert m.evictions_ram == 0
        assert m.evictions_disk == 0
        assert m.bytes_in_ram == 0
        assert m.bytes_on_disk == 0

    def test_to_dict_round_trip(self):
        m = CacheMetrics(cache_id_hits=3, radix_hits=1, bytes_in_ram=2048)
        d = m.to_dict()
        assert d["cache_id_hits"] == 3
        assert d["radix_hits"] == 1
        assert d["bytes_in_ram"] == 2048
        # All defined keys present
        assert set(d.keys()) == {
            "cache_id_hits", "cache_id_misses",
            "radix_hits", "radix_misses",
            "evictions_ram", "evictions_disk",
            "bytes_in_ram", "bytes_on_disk",
        }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompt_cache_radix.py::TestCacheMetrics -v`

Expected: fail with `ModuleNotFoundError: No module named 'olmlx.engine.prompt_cache.metrics'`.

- [ ] **Step 3: Implement metrics dataclass**

Create `olmlx/engine/prompt_cache/metrics.py`:

```python
"""Per-store hit/miss/eviction counters for prompt cache (issue #365)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class CacheMetrics:
    cache_id_hits: int = 0
    cache_id_misses: int = 0
    radix_hits: int = 0
    radix_misses: int = 0
    evictions_ram: int = 0
    evictions_disk: int = 0
    bytes_in_ram: int = 0
    bytes_on_disk: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompt_cache_radix.py::TestCacheMetrics -v`

Expected: both metrics tests pass.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/prompt_cache/metrics.py tests/test_prompt_cache_radix.py
git commit -m "feat(prompt_cache): add CacheMetrics dataclass (#365)"
```

---

## Task 5: Move `CachedPromptState` to its own module

This is a pure refactor — no behaviour change. Existing tests must continue to pass.

**Files:**
- Create: `olmlx/engine/prompt_cache/state.py`
- Modify: `olmlx/engine/model_manager.py:386-395`
- Modify: `olmlx/engine/prompt_cache/__init__.py`

- [ ] **Step 1: Create state module**

Create `olmlx/engine/prompt_cache/state.py`:

```python
"""KV cache state stored cross-request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CachedPromptState:
    """KV cache state from a previous generation, for prompt cache reuse."""

    tokens: list[int]  # Full sequence: prompt + generated tokens
    cache: list[Any]  # Per-layer KV cache objects (mutated in-place by generate_step)
```

- [ ] **Step 2: Re-export from package**

Edit `olmlx/engine/prompt_cache/__init__.py` — replace contents with:

```python
"""Cross-request prompt cache: RAM tier with radix prefix index + disk spill.

Issue #365.
"""

from olmlx.engine.prompt_cache.state import CachedPromptState

__all__ = ["CachedPromptState"]
```

- [ ] **Step 3: Replace definition in model_manager.py with import**

In `olmlx/engine/model_manager.py`, find this block (around lines 386–395):

```python
@dataclass
class CachedPromptState:
    """KV cache state from a previous generation, for prompt cache reuse."""

    tokens: list[int]  # Full sequence: prompt + generated tokens
    cache: list[Any]  # Per-layer KV cache objects (mutated in-place by generate_step)
```

Delete that block. Add to the imports section at the top of the file (after the other `from olmlx.engine` imports around line 23):

```python
from olmlx.engine.prompt_cache import CachedPromptState
```

- [ ] **Step 4: Run all prompt cache tests to verify no regression**

Run: `uv run pytest tests/test_prompt_cache.py tests/test_prompt_cache_store.py tests/test_prompt_cache_radix.py -v`

Expected: all existing tests pass; trie + metrics tests still pass.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/prompt_cache/state.py olmlx/engine/prompt_cache/__init__.py olmlx/engine/model_manager.py
git commit -m "refactor(prompt_cache): move CachedPromptState to dedicated module (#365)"
```

---

## Task 6: Move `PromptCacheStore` to its own module

Pure refactor. We move the existing 330-line class verbatim; no semantic changes.

**Files:**
- Create: `olmlx/engine/prompt_cache/store.py`
- Modify: `olmlx/engine/model_manager.py` (delete `PromptCacheStore`, add re-export)
- Modify: `olmlx/engine/prompt_cache/__init__.py`

- [ ] **Step 1: Copy the class to store.py**

Read `olmlx/engine/model_manager.py:397-735` (the entire `class PromptCacheStore` block). Create `olmlx/engine/prompt_cache/store.py` with:

```python
"""Two-tier prompt cache store: in-memory LRU with optional disk spill.

This module owns the `PromptCacheStore` class previously defined in
`model_manager.py`. Behaviour is unchanged in this commit; radix index
and metrics integration land in subsequent tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from collections import OrderedDict
from pathlib import Path

from olmlx.engine.prompt_cache.state import CachedPromptState

logger = logging.getLogger(__name__)

try:
    from mlx_lm.models.cache import (
        load_prompt_cache,
        save_prompt_cache,
    )
except ImportError:  # pragma: no cover
    save_prompt_cache = None  # type: ignore[assignment]
    load_prompt_cache = None  # type: ignore[assignment]


def _is_serializable_cache(cache: list) -> bool:
    """Return True if every layer in *cache* can round-trip through safetensors.

    TurboQuant/SpectralQuant caches carry packed buffers + per-layer
    metadata that mlx-lm's save_prompt_cache cannot represent.
    """
    # Implementation copied verbatim from model_manager.py — see that
    # module for historical context. Will share definition with
    # model_manager.py in step 2.
```

**STOP.** Don't actually write a duplicate of `_is_serializable_cache` — it's used by `model_manager.py` too. Instead, the cleaner move is to keep `_is_serializable_cache` in `model_manager.py` for now (so other call sites don't break) and import it from the store module.

Replace the placeholder at the bottom of the new `store.py` file with this real import-based approach. The full body of `store.py` should be:

```python
"""Two-tier prompt cache store: in-memory LRU with optional disk spill.

This module owns the `PromptCacheStore` class previously defined in
`model_manager.py`. Behaviour is unchanged in this commit; radix index
and metrics integration land in subsequent tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from collections import OrderedDict
from pathlib import Path

from olmlx.engine.prompt_cache.state import CachedPromptState

logger = logging.getLogger(__name__)

try:
    from mlx_lm.models.cache import (
        load_prompt_cache,
        save_prompt_cache,
    )
except ImportError:  # pragma: no cover
    save_prompt_cache = None  # type: ignore[assignment]
    load_prompt_cache = None  # type: ignore[assignment]


class PromptCacheStore:
    """LRU store for per-agent KV caches with optional disk offload.

    When disk_path and model_name are provided, evicted caches are saved to
    disk instead of deleted. On cache miss, the disk is checked before
    returning None.
    """

    def __init__(
        self,
        max_slots: int,
        disk_path: Path | None = None,
        model_name: str = "",
        disk_max_bytes: int | None = None,
    ) -> None:
        self._max_slots = max_slots
        self._entries: OrderedDict[str, CachedPromptState] = OrderedDict()
        self._disk_path = disk_path
        self._model_name = model_name.replace("/", "--")
        self._disk_max_bytes = disk_max_bytes
        self._evict_generation = 0  # bumped by async_evict_all_to_disk

    # ALL existing methods — peek, get, set, remove, clear, clear_disk,
    # evict_all_to_disk, _evict_all_to_disk_sync, _save_entries_to_disk,
    # _read_from_disk, async_get, async_set, async_evict_all_to_disk,
    # _save_to_disk, _load_from_disk, _cleanup_disk, _disk_enabled,
    # _disk_dir, _disk_file_path, _set_in_memory, __len__ — go here.
    # Copy them VERBATIM from olmlx/engine/model_manager.py lines 419-735.

    # NOTE: _save_to_disk currently calls `_is_serializable_cache(state.cache)`.
    # That helper lives in model_manager.py. Import it lazily inside the
    # method to avoid a circular import at module load time. See step 2.
```

Now do the actual copy: in your editor, copy lines 419–735 of `olmlx/engine/model_manager.py` (everything from `@property def _disk_enabled` through `def __len__`) into `store.py` after the `__init__` method shown above. Make ONE textual change while pasting: the line inside `_save_to_disk` that reads

```python
        if state.cache and not _is_serializable_cache(state.cache):
```

becomes:

```python
        if state.cache:
            from olmlx.engine.model_manager import _is_serializable_cache
            if not _is_serializable_cache(state.cache):
```

(restructured so the import is only paid when there's a cache to check).

- [ ] **Step 2: Replace class with import in model_manager.py**

In `olmlx/engine/model_manager.py`, delete the entire `class PromptCacheStore:` block (lines 397–735, ending at the blank line before `@dataclass class LoadedModel:`).

Then in the import block at the top (around line 23), update the existing prompt_cache import to:

```python
from olmlx.engine.prompt_cache import CachedPromptState, PromptCacheStore
```

(replacing the earlier single-symbol import from Task 5).

- [ ] **Step 3: Re-export from the package**

Edit `olmlx/engine/prompt_cache/__init__.py`:

```python
"""Cross-request prompt cache: RAM tier with radix prefix index + disk spill.

Issue #365.
"""

from olmlx.engine.prompt_cache.state import CachedPromptState
from olmlx.engine.prompt_cache.store import PromptCacheStore

__all__ = ["CachedPromptState", "PromptCacheStore"]
```

- [ ] **Step 4: Run full test suite to verify no regression**

Run: `uv run pytest tests/test_prompt_cache.py tests/test_prompt_cache_store.py tests/test_prompt_cache_radix.py tests/test_model_manager.py tests/test_inference.py -v`

Expected: all tests pass. If any fail with `ImportError`, fix the missing module imports in `store.py` (likely the `re`, `json`, or `shutil` modules used by methods we copied — already included in the import block above).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/prompt_cache/store.py olmlx/engine/prompt_cache/__init__.py olmlx/engine/model_manager.py
git commit -m "refactor(prompt_cache): move PromptCacheStore to dedicated module (#365)"
```

---

## Task 7: Wire radix index + metrics into `PromptCacheStore`

We now teach the moved `PromptCacheStore` to maintain its trie + metric counters in sync with `_entries`. Public method signatures unchanged.

**Files:**
- Modify: `olmlx/engine/prompt_cache/store.py`
- Modify: `olmlx/engine/prompt_cache/__init__.py`
- Test: `tests/test_prompt_cache_radix.py` (append)

- [ ] **Step 1: Write failing integration tests for the store**

Append to `tests/test_prompt_cache_radix.py`:

```python
from olmlx.engine.prompt_cache import CachedPromptState, PromptCacheStore


def _make_state(tokens: list[int]) -> CachedPromptState:
    return CachedPromptState(tokens=list(tokens), cache=[object()])


class TestPromptCacheStoreRadixIntegration:
    def test_set_inserts_into_radix(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state([1, 2, 3]))
        # cache_id miss + radix lookup finds the entry
        store_metrics = store.metrics
        assert store_metrics.bytes_in_ram >= 0  # estimator runs without crashing
        found = store.find_by_prefix([1, 2, 3, 4], min_prefix_tokens=0)
        assert found is not None
        old_cache_id, state, depth = found
        assert old_cache_id == "a"
        assert depth == 3
        assert state.tokens == [1, 2, 3]

    def test_remove_drops_radix_terminal(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state([1, 2, 3]))
        store.remove("a")
        assert store.find_by_prefix([1, 2, 3], min_prefix_tokens=0) is None

    def test_find_by_prefix_threshold_filters(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state([1, 2, 3]))
        # min_prefix_tokens=5 rejects the 3-token match
        assert store.find_by_prefix([1, 2, 3, 9, 9], min_prefix_tokens=5) is None
        # min_prefix_tokens=3 accepts
        assert store.find_by_prefix([1, 2, 3, 9, 9], min_prefix_tokens=3) is not None

    def test_takeover_rekeys_entry(self):
        store = PromptCacheStore(max_slots=4)
        original = _make_state([1, 2, 3])
        store.set("a", original)
        taken = store.takeover("a", "b")
        assert taken is original
        # Old key gone
        assert store.peek("a") is None
        # New key resolves
        assert store.peek("b") is original
        # Radix terminal now points to "b"
        cache_id, _ = store._radix.find_longest_prefix([1, 2, 3])
        assert cache_id == "b"

    def test_takeover_unknown_returns_none(self):
        store = PromptCacheStore(max_slots=4)
        assert store.takeover("missing", "new") is None

    def test_lru_eviction_removes_radix_terminal(self):
        store = PromptCacheStore(max_slots=2)
        store.set("a", _make_state([1, 1, 1]))
        store.set("b", _make_state([2, 2, 2]))
        store.set("c", _make_state([3, 3, 3]))  # evicts "a"
        assert store.peek("a") is None
        # Radix no longer has the "a" path
        cache_id, _ = store._radix.find_longest_prefix([1, 1, 1])
        assert cache_id is None
        # Metrics recorded the eviction
        assert store.metrics.evictions_ram == 1

    def test_metrics_tracks_hits_and_misses(self):
        store = PromptCacheStore(max_slots=4)
        store.set("a", _make_state([1, 2, 3]))
        assert store.peek("a") is not None  # peek does not affect metrics
        # get() records a hit
        store.get("a")
        assert store.metrics.cache_id_hits == 1
        # get() on unknown records a miss
        store.get("missing")
        assert store.metrics.cache_id_misses == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompt_cache_radix.py::TestPromptCacheStoreRadixIntegration -v`

Expected: every test fails — `PromptCacheStore` has no `find_by_prefix`, `takeover`, `metrics`, or `_radix` attributes yet.

- [ ] **Step 3: Add radix + metrics to the store**

Edit `olmlx/engine/prompt_cache/store.py`. First, update the top-of-file imports:

```python
from olmlx.engine.prompt_cache.metrics import CacheMetrics
from olmlx.engine.prompt_cache.radix import PrefixCacheIndex
from olmlx.engine.prompt_cache.state import CachedPromptState
```

In `PromptCacheStore.__init__`, after the existing `self._evict_generation = 0` line, append:

```python
        self._radix = PrefixCacheIndex()
        self.metrics = CacheMetrics()
```

Add a small helper above the class definition:

```python
def _estimate_state_bytes(state: CachedPromptState) -> int:
    """Best-effort byte estimate of a cached state.

    Walks the per-layer cache list looking for mlx.core.array-shaped
    objects exposing `nbytes`. Unknown layer types contribute 0.
    """
    total = 0
    for layer in state.cache or ():
        for attr in ("keys", "values"):
            arr = getattr(layer, attr, None)
            nbytes = getattr(arr, "nbytes", None)
            if isinstance(nbytes, int):
                total += nbytes
    return total
```

Now wire mutations. The existing private methods are the choke points:

In `_set_in_memory`, after `self._entries[cache_id] = state` (both branches — the collision branch and the new-entry branch), add:

```python
            self._radix.insert(state.tokens, cache_id)
            self.metrics.bytes_in_ram += _estimate_state_bytes(state)
```

When the collision branch displaces an old state, subtract its bytes first:

```python
        if cache_id in self._entries:
            self._entries.move_to_end(cache_id)
            old = self._entries[cache_id]
            self.metrics.bytes_in_ram -= _estimate_state_bytes(old)
            self._radix.remove(old.tokens, cache_id)
            self._entries[cache_id] = state
            self._radix.insert(state.tokens, cache_id)
            self.metrics.bytes_in_ram += _estimate_state_bytes(state)
            displaced = old if old.cache is not state.cache else None
            return None, displaced
```

In the LRU-eviction branch, before `evicted_id, evicted = self._entries.popitem(last=False)`, capture and update metrics:

```python
        evicted: CachedPromptState | None = None
        evicted_id: str | None = None
        if len(self._entries) >= self._max_slots:
            evicted_id, evicted = self._entries.popitem(last=False)
            self._radix.remove(evicted.tokens, evicted_id)
            self.metrics.bytes_in_ram -= _estimate_state_bytes(evicted)
            self.metrics.evictions_ram += 1
        self._entries[cache_id] = state
        self._radix.insert(state.tokens, cache_id)
        self.metrics.bytes_in_ram += _estimate_state_bytes(state)
        return evicted_id, evicted
```

In `remove`:

```python
    def remove(self, cache_id: str) -> None:
        """Remove a specific cache entry from memory and disk."""
        existing = self._entries.pop(cache_id, None)
        if existing is not None:
            self._radix.remove(existing.tokens, cache_id)
            self.metrics.bytes_in_ram -= _estimate_state_bytes(existing)
        if self._disk_enabled:
            self._disk_file_path(cache_id).unlink(missing_ok=True)
```

In `_evict_all_to_disk_sync` and `async_evict_all_to_disk`, after clearing `_entries`, reset the index and the in-RAM byte count:

```python
        self._entries.clear()
        self._radix = PrefixCacheIndex()
        self.metrics.bytes_in_ram = 0
        self._evict_generation += 1
```

In `clear`:

```python
    def clear(self) -> None:
        """Remove all cache entries and disk cache files."""
        self._entries.clear()
        self._radix = PrefixCacheIndex()
        self.metrics.bytes_in_ram = 0
        if self._disk_path is not None and self._model_name:
            disk_dir = self._disk_dir()
            if disk_dir.exists():
                shutil.rmtree(disk_dir, ignore_errors=True)
```

In `get` (which calls `_load_from_disk` on memory miss), record hit/miss metrics:

```python
    def get(self, cache_id: str) -> CachedPromptState | None:
        """Get a cache entry, promoting it to MRU.

        Checks memory first, then disk.  Returns None if not found in either.
        """
        state = self._entries.get(cache_id)
        if state is not None:
            self._entries.move_to_end(cache_id)
            self.metrics.cache_id_hits += 1
            return state
        # Memory miss — try disk
        loaded = self._load_from_disk(cache_id)
        if loaded is not None:
            self.metrics.cache_id_hits += 1
        else:
            self.metrics.cache_id_misses += 1
        return loaded
```

Apply the same change to `async_get`: increment `cache_id_hits` on memory or disk hit, `cache_id_misses` on full miss.

`_load_from_disk` already calls `self.set(...)` internally, which now also inserts into the trie and increments bytes_in_ram — that's correct (a disk-restored entry is now a RAM entry).

Add the two new public methods at the end of the class (before `__len__`):

```python
    def find_by_prefix(
        self,
        tokens: list[int],
        min_prefix_tokens: int,
    ) -> tuple[str, CachedPromptState, int] | None:
        """Longest-prefix lookup against in-memory entries.

        Returns (old_cache_id, state, prefix_len) or None if no terminal
        on the descent path meets `min_prefix_tokens`.
        """
        cache_id, depth = self._radix.find_longest_prefix(tokens)
        if cache_id is None or depth < min_prefix_tokens:
            self.metrics.radix_misses += 1
            return None
        state = self._entries.get(cache_id)
        if state is None:
            # Trie out of sync — defensive.
            self.metrics.radix_misses += 1
            return None
        self.metrics.radix_hits += 1
        return cache_id, state, depth

    def takeover(
        self,
        old_cache_id: str,
        new_cache_id: str,
    ) -> CachedPromptState | None:
        """Re-key an existing in-memory entry. No KV copy.

        The trie terminal moves from old_cache_id to new_cache_id; the
        entry is promoted to MRU under the new key. Returns the state,
        or None if old_cache_id is no longer present.
        """
        state = self._entries.pop(old_cache_id, None)
        if state is None:
            return None
        self._radix.remove(state.tokens, old_cache_id)
        # The destination key may already exist; honour LRU semantics by
        # going through _set_in_memory which handles displacement + radix.
        evicted_id, evicted = self._set_in_memory(new_cache_id, state)
        if evicted_id is not None and evicted is not None:
            # Caller doesn't expect a side-channel disk save here — it
            # would mean the takeover destination collided with a third
            # entry. Mirror set()'s behaviour for symmetry.
            self._save_to_disk(evicted_id, evicted)
        return state
```

- [ ] **Step 4: Re-export `CacheMetrics` from the package**

Edit `olmlx/engine/prompt_cache/__init__.py`:

```python
"""Cross-request prompt cache: RAM tier with radix prefix index + disk spill.

Issue #365.
"""

from olmlx.engine.prompt_cache.metrics import CacheMetrics
from olmlx.engine.prompt_cache.state import CachedPromptState
from olmlx.engine.prompt_cache.store import PromptCacheStore

__all__ = ["CacheMetrics", "CachedPromptState", "PromptCacheStore"]
```

- [ ] **Step 5: Run the new + existing prompt cache tests**

Run: `uv run pytest tests/test_prompt_cache_radix.py tests/test_prompt_cache.py tests/test_prompt_cache_store.py -v`

Expected: all radix integration tests pass. Existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/prompt_cache/store.py olmlx/engine/prompt_cache/__init__.py tests/test_prompt_cache_radix.py
git commit -m "feat(prompt_cache): radix index + metrics in PromptCacheStore (#365)"
```

---

## Task 8: Wire radix fallback into `_setup_prompt_cache`

Single insertion point in `inference.py` after the existing `async_get` returns None.

**Files:**
- Modify: `olmlx/engine/inference.py:2167-2168`
- Test: `tests/test_prompt_cache_radix.py` (append)

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_prompt_cache_radix.py`:

```python
import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from olmlx.engine.inference import _setup_prompt_cache


@pytest.fixture(autouse=True)
def _safe_memory_defaults():
    """Mock memory utils so prompt cache tests aren't affected by real Metal state."""
    with (
        patch("olmlx.utils.memory.mx.get_active_memory", return_value=0),
        patch("olmlx.utils.memory.mx.get_cache_memory", return_value=0),
    ):
        yield


def _make_lm_for_radix(store: PromptCacheStore) -> MagicMock:
    """Minimal LoadedModel stand-in for _setup_prompt_cache."""
    lm = MagicMock()
    lm.prompt_cache_store = store
    lm.supports_cache_persistence = True
    lm.supports_cache_trim = True
    lm.is_vlm = False
    lm.kv_cache_quant = None
    lm.is_distributed = False
    return lm


@pytest.mark.asyncio
class TestRadixFallbackInSetup:
    async def test_sibling_prefix_hit_triggers_takeover(self):
        store = PromptCacheStore(max_slots=4)
        shared_prompt = list(range(1, 1025))  # 1024 tokens
        # Seed the store with cache_id="A" + a continuation
        seeded_tokens = shared_prompt + [42, 43, 44]
        seeded = CachedPromptState(tokens=seeded_tokens, cache=[MagicMock()])
        store.set("A", seeded)

        # Build a sibling prompt: same 1024-token system prefix, different turn
        sibling_prompt_tokens = shared_prompt + [99, 100]

        lm = _make_lm_for_radix(store)

        with patch(
            "olmlx.engine.inference.trim_prompt_cache", return_value=3
        ), patch(
            "olmlx.engine.inference.make_prompt_cache", return_value=[MagicMock()]
        ):
            result = await _setup_prompt_cache(
                lm,
                prompt=sibling_prompt_tokens,
                gen_kwargs={},
                prompt_tokens=sibling_prompt_tokens,
                cache_id="B",
            )

        # Cache_read_tokens reflects the radix hit + trim alignment
        assert result.cache_read_tokens >= 1024
        # Cache_creation_tokens is just the suffix (2 tokens of new turn)
        assert result.cache_creation_tokens <= 2
        # Takeover: "A" gone, "B" owns the entry
        assert store.peek("A") is None
        assert store.peek("B") is not None

    async def test_radix_below_threshold_falls_to_fresh(self):
        store = PromptCacheStore(max_slots=4)
        store.set("A", CachedPromptState(tokens=[1, 2, 3], cache=[MagicMock()]))
        lm = _make_lm_for_radix(store)

        with patch(
            "olmlx.engine.inference.settings.prompt_cache_radix_min_prefix_tokens",
            1000,
        ), patch(
            "olmlx.engine.inference.make_prompt_cache", return_value=[MagicMock()]
        ):
            result = await _setup_prompt_cache(
                lm,
                prompt=[1, 2, 3, 4, 5],
                gen_kwargs={},
                prompt_tokens=[1, 2, 3, 4, 5],
                cache_id="B",
            )

        # Threshold not met → fresh cache, A stays in place
        assert result.cache_read_tokens == 0
        assert store.peek("A") is not None

    async def test_radix_disabled_skips_lookup(self):
        store = PromptCacheStore(max_slots=4)
        store.set("A", CachedPromptState(tokens=list(range(1024)), cache=[MagicMock()]))
        lm = _make_lm_for_radix(store)

        with patch(
            "olmlx.engine.inference.settings.prompt_cache_radix", False
        ), patch(
            "olmlx.engine.inference.make_prompt_cache", return_value=[MagicMock()]
        ):
            result = await _setup_prompt_cache(
                lm,
                prompt=list(range(1024)) + [99],
                gen_kwargs={},
                prompt_tokens=list(range(1024)) + [99],
                cache_id="B",
            )

        # No fallback → fresh prefill, A still there
        assert result.cache_read_tokens == 0
        assert store.peek("A") is not None
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/test_prompt_cache_radix.py::TestRadixFallbackInSetup -v`

Expected: tests fail — the radix fallback is not yet wired in. The sibling test will report `cache_read_tokens == 0`.

- [ ] **Step 3: Insert the radix fallback in `_setup_prompt_cache`**

Open `olmlx/engine/inference.py`. Find the block at lines ~2163–2168:

```python
    if not lm.supports_cache_persistence:
        if lm.prompt_cache_store.peek(cache_id) is not None:
            lm.prompt_cache_store.remove(cache_id)
        cached = None
    else:
        cached = await lm.prompt_cache_store.async_get(cache_id)
```

Replace with:

```python
    if not lm.supports_cache_persistence:
        if lm.prompt_cache_store.peek(cache_id) is not None:
            lm.prompt_cache_store.remove(cache_id)
        cached = None
    else:
        cached = await lm.prompt_cache_store.async_get(cache_id)
        # Issue #365: cache_id miss → look for a sibling that shares a long
        # token prefix. Takeover semantics — no KV copy. The old cache_id
        # loses its entry (documented limitation: concurrent two-stream
        # sibling branching falls back to fresh prefill on the loser).
        if cached is None and settings.prompt_cache_radix:
            found = lm.prompt_cache_store.find_by_prefix(
                prompt_tokens,
                min_prefix_tokens=settings.prompt_cache_radix_min_prefix_tokens,
            )
            if found is not None:
                old_cache_id, cached, prefix_len_hint = found
                lm.prompt_cache_store.takeover(old_cache_id, cache_id)
                logger.info(
                    "Radix prefix hit: %d tokens reused from cache_id=%s → %s",
                    prefix_len_hint,
                    old_cache_id,
                    cache_id,
                )
```

Note: `settings` is already imported in `inference.py` (it's used elsewhere in `_setup_prompt_cache`); confirm by running `grep -n "from olmlx.config import settings" olmlx/engine/inference.py | head -1` — it should appear in the existing imports.

- [ ] **Step 4: Run the integration tests**

Run: `uv run pytest tests/test_prompt_cache_radix.py::TestRadixFallbackInSetup -v`

Expected: all three tests pass.

- [ ] **Step 5: Run the broader prompt cache regression suite**

Run: `uv run pytest tests/test_prompt_cache.py tests/test_prompt_cache_store.py tests/test_prompt_cache_radix.py tests/test_inference.py tests/test_inference_bugs.py -v`

Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/inference.py tests/test_prompt_cache_radix.py
git commit -m "feat(inference): radix prefix fallback in _setup_prompt_cache (#365)"
```

---

## Task 9: RAM byte budget as soft eviction trigger

Slot count is the existing hard cap. The spec promised a byte budget as the primary knob. Implement it as a *soft* trigger that runs additional LRU evictions until `bytes_in_ram ≤ budget`, evaluated after each `set`. Slot count remains the absolute backstop.

**Files:**
- Modify: `olmlx/engine/prompt_cache/store.py`
- Modify: `olmlx/engine/model_manager.py` (pass budget at construction)
- Test: `tests/test_prompt_cache_radix.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prompt_cache_radix.py`:

```python
class TestRamBudgetEviction:
    def _state_with_bytes(self, tokens, nbytes):
        layer = MagicMock()
        layer.keys = MagicMock()
        layer.keys.nbytes = nbytes
        layer.values = MagicMock()
        layer.values.nbytes = 0
        return CachedPromptState(tokens=list(tokens), cache=[layer])

    def test_byte_budget_evicts_extra_entries(self):
        # Slot cap 10, byte budget 2500 bytes; entries are 1000 bytes each
        store = PromptCacheStore(max_slots=10, ram_budget_bytes=2500)
        store.set("a", self._state_with_bytes([1, 1, 1], 1000))
        store.set("b", self._state_with_bytes([2, 2, 2], 1000))
        store.set("c", self._state_with_bytes([3, 3, 3], 1000))
        # 3000 bytes > 2500 → LRU (a) evicted
        assert store.peek("a") is None
        assert store.peek("b") is not None
        assert store.peek("c") is not None
        assert store.metrics.evictions_ram == 1

    def test_no_budget_leaves_entries_alone(self):
        store = PromptCacheStore(max_slots=10, ram_budget_bytes=None)
        store.set("a", self._state_with_bytes([1], 10_000))
        store.set("b", self._state_with_bytes([2], 10_000))
        assert store.peek("a") is not None
        assert store.peek("b") is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompt_cache_radix.py::TestRamBudgetEviction -v`

Expected: fail — `PromptCacheStore.__init__` does not accept `ram_budget_bytes`.

- [ ] **Step 3: Add budget parameter and enforcement**

In `olmlx/engine/prompt_cache/store.py`, update `__init__`:

```python
    def __init__(
        self,
        max_slots: int,
        disk_path: Path | None = None,
        model_name: str = "",
        disk_max_bytes: int | None = None,
        ram_budget_bytes: int | None = None,
    ) -> None:
        self._max_slots = max_slots
        self._entries: OrderedDict[str, CachedPromptState] = OrderedDict()
        self._disk_path = disk_path
        self._model_name = model_name.replace("/", "--")
        self._disk_max_bytes = disk_max_bytes
        self._ram_budget_bytes = ram_budget_bytes
        self._evict_generation = 0
        self._radix = PrefixCacheIndex()
        self.metrics = CacheMetrics()
```

Add a helper method:

```python
    def _enforce_ram_budget(self) -> list[tuple[str, CachedPromptState]]:
        """Evict LRU entries until bytes_in_ram <= ram_budget_bytes.

        Returns the list of (cache_id, state) tuples that were evicted,
        for the caller to spill to disk if applicable.
        """
        evicted: list[tuple[str, CachedPromptState]] = []
        if self._ram_budget_bytes is None:
            return evicted
        while (
            self._entries
            and self.metrics.bytes_in_ram > self._ram_budget_bytes
        ):
            cache_id, state = self._entries.popitem(last=False)
            self._radix.remove(state.tokens, cache_id)
            self.metrics.bytes_in_ram -= _estimate_state_bytes(state)
            self.metrics.evictions_ram += 1
            evicted.append((cache_id, state))
        return evicted
```

Call it at the end of `set`:

```python
    def set(self, cache_id: str, state: CachedPromptState) -> CachedPromptState | None:
        """Set a cache entry, evicting LRU if at capacity.
        ...existing docstring...
        """
        evicted_id, evicted = self._set_in_memory(cache_id, state)
        if evicted_id is not None and evicted is not None:
            self._save_to_disk(evicted_id, evicted)
        # Byte budget overflow → additional evictions
        for over_id, over_state in self._enforce_ram_budget():
            self._save_to_disk(over_id, over_state)
        return evicted
```

And at the end of `async_set`:

```python
    async def async_set(
        self, cache_id: str, state: CachedPromptState
    ) -> CachedPromptState | None:
        """Async version of set(). Memory ops are sync; disk save runs in a thread."""
        evicted_id, evicted = self._set_in_memory(cache_id, state)
        if evicted_id is not None and evicted is not None:
            await asyncio.to_thread(self._save_to_disk, evicted_id, evicted)
        for over_id, over_state in self._enforce_ram_budget():
            await asyncio.to_thread(self._save_to_disk, over_id, over_state)
        return evicted
```

- [ ] **Step 4: Wire the budget from settings in `LoadedModel.__post_init__`**

Edit `olmlx/engine/model_manager.py`. Find `LoadedModel.__post_init__` (around line 783). Update the `PromptCacheStore` construction:

```python
        if self.prompt_cache_store is None:
            disk_path = (
                Path(settings.prompt_cache_disk_path).expanduser()
                if settings.prompt_cache_disk
                else None
            )
            disk_max_bytes = (
                int(settings.prompt_cache_disk_max_gb * 1024**3)
                if settings.prompt_cache_disk
                else None
            )
            ram_budget_bytes = int(settings.prompt_cache_ram_budget_gb * 1024**3)
            self.prompt_cache_store = PromptCacheStore(
                max_slots=settings.prompt_cache_max_slots,
                disk_path=disk_path,
                model_name=self.name,
                disk_max_bytes=disk_max_bytes,
                ram_budget_bytes=ram_budget_bytes,
            )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_prompt_cache_radix.py::TestRamBudgetEviction tests/test_prompt_cache.py tests/test_prompt_cache_store.py -v`

Expected: budget tests pass; existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/prompt_cache/store.py olmlx/engine/model_manager.py tests/test_prompt_cache_radix.py
git commit -m "feat(prompt_cache): RAM byte budget soft-eviction (#365)"
```

---

## Task 10: Expose `cache_metrics` on `/api/ps`

**Files:**
- Modify: `olmlx/schemas/status.py`
- Modify: `olmlx/routers/status.py`
- Test: `tests/test_prompt_cache_radix.py` (append) or `tests/test_routers.py` if present

- [ ] **Step 1: Check whether a router test for /api/ps exists**

Run: `grep -rn "api/ps\|RunningModel" tests/ | head`

Note what comes back. If there's an existing `/api/ps` test file (likely `tests/test_status.py` or similar), append to that. Otherwise put the test in `tests/test_prompt_cache_radix.py`.

- [ ] **Step 2: Add the failing test**

Append to the appropriate test file (using `tests/test_prompt_cache_radix.py` for concreteness):

```python
from fastapi.testclient import TestClient


def _build_test_app(loaded_models):
    """Minimal FastAPI app exposing the status router and a fake manager."""
    from fastapi import FastAPI
    from olmlx.routers.status import router as status_router

    app = FastAPI()
    app.include_router(status_router)
    app.state.model_manager = MagicMock()
    app.state.model_manager.get_loaded.return_value = loaded_models
    app.state.model_store = None
    return app


def test_api_ps_includes_cache_metrics():
    lm = MagicMock()
    lm.name = "test-model"
    lm.hf_path = "hf/test-model"
    lm.size_bytes = 1024
    lm.expires_at = None
    lm.weight_quant = None
    lm.active_refs = 0
    store = PromptCacheStore(max_slots=4)
    store.set("a", CachedPromptState(tokens=[1, 2, 3], cache=[MagicMock()]))
    store.get("a")  # bump cache_id_hits
    lm.prompt_cache_store = store

    app = _build_test_app([lm])
    client = TestClient(app)
    resp = client.get("/api/ps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["models"][0]["cache_metrics"]["cache_id_hits"] == 1
    assert "radix_hits" in data["models"][0]["cache_metrics"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_prompt_cache_radix.py::test_api_ps_includes_cache_metrics -v`

Expected: fails with KeyError on `cache_metrics` (field not defined on `RunningModel`).

- [ ] **Step 4: Add field to schema**

Edit `olmlx/schemas/status.py`:

```python
from pydantic import BaseModel

from olmlx.schemas.models import ModelDetails


class RunningModel(BaseModel):
    name: str
    model: str = ""
    size: int = 0
    digest: str = ""
    details: ModelDetails = ModelDetails()
    expires_at: str = ""
    size_vram: int = 0
    active_refs: int = 0
    cache_metrics: dict[str, int] = {}


class PsResponse(BaseModel):
    models: list[RunningModel]


class VersionResponse(BaseModel):
    version: str
```

- [ ] **Step 5: Populate the field in the router**

Edit `olmlx/routers/status.py`, in the `for lm in loaded:` loop, just before the `models.append(RunningModel(...))` call:

```python
        cache_metrics = {}
        store = getattr(lm, "prompt_cache_store", None)
        if store is not None and hasattr(store, "metrics"):
            cache_metrics = store.metrics.to_dict()
```

And include the new field in the `RunningModel(...)` constructor:

```python
        models.append(
            RunningModel(
                name=lm.name,
                model=lm.hf_path,
                size=size,
                digest=digest,
                details=ModelDetails(
                    format="mlx",
                    family=meta["family"],
                    parameter_size=meta["parameter_size"],
                    quantization_level=meta["quantization_level"],
                ),
                expires_at=expires,
                size_vram=size,
                active_refs=lm.active_refs,
                cache_metrics=cache_metrics,
            )
        )
```

- [ ] **Step 6: Run the test**

Run: `uv run pytest tests/test_prompt_cache_radix.py::test_api_ps_includes_cache_metrics -v`

Expected: pass.

- [ ] **Step 7: Run the full status-router test set**

Run: `uv run pytest tests/ -k "status or ps" -v`

Expected: no regressions.

- [ ] **Step 8: Commit**

```bash
git add olmlx/schemas/status.py olmlx/routers/status.py tests/test_prompt_cache_radix.py
git commit -m "feat(api/ps): expose prompt_cache metrics per loaded model (#365)"
```

---

## Task 11: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Project Structure section**

Edit `CLAUDE.md`. In the project-structure tree under `engine/`, add a sub-tree:

Find:
```
│   ├── grammar.py      # xgrammar JSON-mode / JSON-Schema logits processor (issue #361)
│   ├── inference.py    # generate_chat, generate_completion, generate_embeddings
│   ├── model_manager.py # Model loading/unloading, keep-alive, LRU eviction
```

Add right above `grammar.py`:

```
│   ├── prompt_cache/
│   │   ├── radix.py    # PrefixCacheIndex: token-prefix trie for cross-request sibling lookup
│   │   ├── store.py    # PromptCacheStore: RAM tier (cache_id LRU + radix index) + disk spill
│   │   ├── state.py    # CachedPromptState dataclass
│   │   └── metrics.py  # CacheMetrics counters (hits, misses, evictions, bytes_in_ram)
```

- [ ] **Step 2: Update the Prompt caching design-decisions bullet**

Find the line in CLAUDE.md beginning `**Prompt caching**: KV cache reuse across requests when prompts share a common prefix.`

At the end of that bullet (after the existing sentence about within-request reuse still working for hybrid models), append:

```
 Cross-request radix prefix cache (issue #365): on a `cache_id` miss the store walks an in-memory token-prefix trie (`PrefixCacheIndex`) for the longest-prefix sibling, and on hit **takes over** the matched entry by re-keying it to the new `cache_id` (the old `cache_id` loses its entry — no KV copy). Catches the common Claude-Code-style branching workload where many sessions share the same large system prompt. Concurrent two-stream sibling branching is not supported in v1; the loser falls back to fresh prefill. Tunables: `OLMLX_PROMPT_CACHE_RADIX` (default `true`), `OLMLX_PROMPT_CACHE_RADIX_MIN_PREFIX_TOKENS` (default `256` — below this the takeover is skipped), `OLMLX_PROMPT_CACHE_RAM_BUDGET_GB` (default `8.0`, soft eviction trigger; `OLMLX_PROMPT_CACHE_MAX_SLOTS` remains the hard cap). Per-store metrics (`cache_id_hits`/`cache_id_misses`/`radix_hits`/`radix_misses`/`evictions_ram`/`evictions_disk`/`bytes_in_ram`/`bytes_on_disk`) surface on `/api/ps` under each model's `cache_metrics` field. Non-persistable caches (issues #284, #343) remain skipped — the radix fallback is gated on `supports_cache_persistence`.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE.md): document radix prompt cache (#365)"
```

---

## Task 12: Final regression sweep + lint

**Files:**
- None (verification only)

- [ ] **Step 1: Run full pytest suite**

Run: `uv run pytest -q`

Expected: all tests pass. If a test fails, fix the regression before proceeding.

- [ ] **Step 2: Run ruff**

Run: `uv run ruff check olmlx/ tests/ && uv run ruff format --check olmlx/ tests/`

Expected: no diagnostics. If `format --check` reports diffs, run `uv run ruff format olmlx/ tests/` and amend the most recent commit with the formatting changes.

- [ ] **Step 3: Smoke test end-to-end**

Run: `uv run olmlx --help`

Expected: CLI starts cleanly with no import errors. This catches missed `from olmlx.engine.model_manager import PromptCacheStore` callers.

- [ ] **Step 4 (optional): Run inference smoke test if a model is locally available**

If `~/.olmlx/models.json` has a configured model:

```bash
uv run olmlx serve &
SERVER_PID=$!
sleep 5
curl -s http://localhost:11434/api/ps | jq '.models[0].cache_metrics'
kill $SERVER_PID
```

Expected: returns a JSON object with the eight metric keys, all zero.

- [ ] **Step 5: Final commit (if any formatting fixes pending)**

If step 2 introduced format changes:

```bash
git add -u
git commit -m "style: ruff format after radix prompt cache (#365)"
```

---

## Acceptance check

- [ ] All tasks marked complete.
- [ ] `tests/test_prompt_cache_radix.py::TestRadixFallbackInSetup::test_sibling_prefix_hit_triggers_takeover` passes — proves ≥50% prefill saved on the sibling-branch case (the assertion `cache_creation_tokens <= 2` on a 1026-token prompt is ≥99.8% saved).
- [ ] `tests/test_prompt_cache.py` and `tests/test_prompt_cache_store.py` pass unchanged — no behaviour regression on cache_id-exact-match flows.
- [ ] `/api/ps` returns `cache_metrics` per model.
- [ ] CLAUDE.md updated.
