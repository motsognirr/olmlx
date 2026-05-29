# Cross-request radix prefix cache

Issue: [#365](https://github.com/motsognirr/olmlx/issues/365)
Date: 2026-05-28
Status: Approved

## Problem

The existing prompt cache (`PromptCacheStore` in `olmlx/engine/model_manager.py`) is keyed by a per-request `cache_id` string (HTTP header `x-cache-id`). Two requests with the same large system prompt but different `cache_id`s never find each other and each pays a full prefill, even though they share a long token prefix.

Concrete failure mode: Claude Code opens a fresh session per tool call branch. Each branch is a sibling continuation of the same 10–30k system prompt. The current store misses every time.

Issue #343 documented a related waste: storing non-trimmable cache state cross-request is pure I/O cost because the next request's retokenized prompt can never realign with the stored state via `trim_prompt_cache`. A token-prefix-indexed lookup over exact token IDs side-steps the realignment problem in principle, but rescuing #343 caches is out of scope here.

## Goals

- Token-prefix lookup for cache_id misses, so sibling continuations share KV cache state.
- Two-tier store: bounded RAM tier with disk spill for persistable + trimmable caches.
- Hit-rate metrics exposed for debugging.
- No quality or behaviour regression for existing cache_id-exact-match flows.
- ≥50% prefill saved on Claude-Code-style sibling-branch workloads.

## Non-goals

- KV-cache cloning. Concurrent two-stream sibling branching is documented as a follow-up (one stream wins via takeover; the other falls back to fresh cache).
- Rescuing non-persistable caches (issue #284 SSM ArraysCache, issue #343 RotatingKVCache/ChunkedKVCache). Tracked separately.
- Compressed prompt cache (LLMLingua-style).
- Distributed cache sharing across workers.

## Approach: takeover, not clone

On a sibling-prefix hit (different `cache_id`, shared token prefix), v1 **transfers ownership** of the matched entry to the requesting `cache_id`. The old `cache_id` loses its entry. Justification:

- Single-user server: concurrent two-stream sibling branching is rare.
- No per-cache-type clone primitive needed. A correct clone for hybrid caches is non-trivial.
- Net throughput win for the common case (session A finishes → session B branches from the same prefix) is identical to clone.
- The losing `cache_id` recovers next request via either (a) its own takeover from a deeper sibling or (b) fresh prefill.

The concurrent-sibling regression is bounded: at most one cache_id per shared prefix can be in RAM at a time, vs. two with cloning. Throughput loss is one fresh prefill per concurrent collision.

## Architecture

New package `olmlx/engine/prompt_cache/`:

```
olmlx/engine/prompt_cache/
├── __init__.py          # re-exports PromptCacheStore, CachedPromptState
├── state.py             # CachedPromptState (moved from model_manager.py)
├── radix.py             # PrefixCacheIndex — token-trie with longest-prefix lookup
├── store.py             # PromptCacheStore (refactored from model_manager.py)
└── metrics.py           # CacheMetrics: hit/miss/eviction counters
```

`olmlx/engine/model_manager.py` keeps `CachedPromptState` and `PromptCacheStore` as one-line re-exports from the new package so existing imports (`from olmlx.engine.model_manager import PromptCacheStore`) continue to work for one release. Internal call sites migrate to importing from `olmlx.engine.prompt_cache` directly.

### `PrefixCacheIndex` (radix.py)

A trie over token IDs. Each node is:

```python
@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"]   # next token → child node
    terminal_cache_ids: set[str]       # cache_ids that claim this exact token-prefix
```

Public API:

```python
class PrefixCacheIndex:
    def insert(self, tokens: list[int], cache_id: str) -> None: ...
    def remove(self, tokens: list[int], cache_id: str) -> None: ...
    def find_longest_prefix(
        self, tokens: list[int], min_depth: int = 0
    ) -> tuple[str | None, int]:
        """Return (cache_id_of_match, prefix_length)."""
```

Behaviour:

- `insert`: walks/creates nodes for each token; adds `cache_id` to the terminal set at the final node. Multiple distinct entries with identical token sequences all stay reachable (set semantics, no clobbering).
- `find_longest_prefix`: descends matching `tokens` to the deepest visited node, then returns any cache_id reachable from that node — either a terminal on the descent path or, if the descent ended on a non-terminal interior node, any terminal found via DFS in the subtree below. This catches the sibling case where a stored entry's terminal sits *past* the query's divergence point (e.g. shared system prompt, different next turn). The returned `prefix_len` is the descent depth, not the terminal's depth, so the caller knows how much of the query is shared and can trim the borrowed KV cache to align. `min_depth` short-circuits the subtree DFS and returns `(None, 0)` immediately when the descent ends below the caller's threshold — relevant for queries that only share a BOS token with a large stored subtree. Returns `(None, 0)` when nothing was matched.
- `remove`: walks to the leaf, discards `cache_id` from the terminal set, prunes upward until a node has either children or a non-empty terminal set.

Complexity: insert/remove/lookup are O(len(tokens)) with O(1) dict ops per step.

Memory: a single trie node is ~120 bytes (dict + small ints). For 4 active sessions × 32k tokens with high prefix overlap, the trie is well under 50 MB — negligible vs. the KV state it indexes.

### `PromptCacheStore` (store.py)

Refactored from `model_manager.py:397`. Holds both indexes plus the existing OrderedDict-based LRU and disk-spill machinery. Public API unchanged: `peek`, `get`, `set`, `remove`, `clear`, `evict_all_to_disk`, plus the `async_*` wrappers.

New internal field: `_radix: PrefixCacheIndex`.

Mutations stay in sync: every `_entries[cache_id] = state` is paired with `self._radix.insert(state.tokens, cache_id)`; every `_entries.pop(cache_id)` is paired with `self._radix.remove(state.tokens, cache_id)`. Disk-tier moves don't touch the radix — the radix indexes RAM only.

New public methods:

```python
def find_by_prefix(
    self,
    tokens: list[int],
    min_prefix_tokens: int,
) -> tuple[str, CachedPromptState, int] | None:
    """Longest-prefix lookup. Returns (old_cache_id, state, prefix_len) or None.

    Sync because the lookup is pure in-memory work (radix walk + dict
    get); no disk I/O is involved, so wrapping it in async_* would only
    add overhead. Caller is responsible for re-keying via takeover()
    if it intends to use the result under a different cache_id.
    """
```

```python
def takeover(
    self,
    old_cache_id: str,
    new_cache_id: str,
) -> CachedPromptState | None:
    """Re-key an existing entry. Old cache_id removed from both indexes;
    new cache_id maps to the same state (moved to MRU). No KV copy.
    Returns the state for caller convenience, or None if old_cache_id
    is no longer present (lost a race)."""
```

Takeover preserves LRU order by moving the entry to MRU under the new key.

### Two-tier semantics

- **RAM tier**: persistable caches only. Bounded by `prompt_cache_ram_budget_gb`. LRU eviction when the budget is exceeded.
- **Disk tier**: spill destination for evicted RAM entries that are persistable **and** trimmable **and** serializable (existing `_is_serializable_cache` check). Bounded by `prompt_cache_disk_max_gb`.
- Caches that are non-persistable (`supports_cache_persistence=False`) bypass the store entirely. Existing behaviour, unchanged. The two-tier framing makes this rule explicit rather than scattered across `_setup_prompt_cache` / `_store_prompt_cache_after_generation`.

`prompt_cache_max_slots` is kept as a secondary slot count cap (default unchanged) so existing tests that assert slot-count behaviour continue to pass. The RAM byte budget is the primary trigger for eviction.

### `CacheMetrics` (metrics.py)

```python
@dataclass
class CacheMetrics:
    cache_id_hits: int = 0
    cache_id_misses: int = 0
    radix_hits: int = 0       # exact miss, prefix lookup found usable entry
    radix_misses: int = 0     # exact miss, prefix lookup empty/below threshold
    evictions_ram: int = 0
    evictions_disk: int = 0
    bytes_in_ram: int = 0     # updated on each set/evict; snapshot estimate
    bytes_on_disk: int = 0    # updated lazily during _cleanup_disk
```

Exposed via:
- `LoadedModel.prompt_cache_store.metrics` (already accessible to callers).
- A new optional `cache_metrics` field on `/api/ps` response per loaded model. Field is debug-flavoured (snake_case keys, no Ollama-compat constraint since `/api/ps` already carries non-Ollama fields).
- Logged at `DEBUG` level on every store mutation.

## Lookup flow change in `_setup_prompt_cache`

Current logic (`olmlx/engine/inference.py:2108`) unchanged through the cache_id exact-match path. Insertion point for the radix lookup: after the existing `cached = await lm.prompt_cache_store.async_get(cache_id)` call returns `None`.

```python
if cached is None and settings.prompt_cache_radix and lm.supports_cache_persistence:
    found = lm.prompt_cache_store.find_by_prefix(
        prompt_tokens,
        min_prefix_tokens=settings.prompt_cache_radix_min_prefix_tokens,
    )
    if found is not None:
        old_cache_id, cached, prefix_len_hint = found
        lm.prompt_cache_store.takeover(old_cache_id, cache_id)
        logger.info(
            "Radix prefix hit: %d tokens reused from cache_id=%s → %s",
            prefix_len_hint, old_cache_id, cache_id,
        )
```

After this block, the existing `prefix_len = _find_common_prefix(prompt_tokens, cached.tokens)` path runs as today — it will find `prefix_len == prefix_len_hint` and proceed through the trim+extend logic with no further special-case handling.

The radix fallback is gated on `supports_cache_persistence` for the same reason the existing `cache_id` lookup is gated: non-persistable caches were never inserted into the radix index, so the lookup would be a guaranteed miss; the gate just skips the work.

`_store_prompt_cache_after_generation` (line 2435) requires no logic change. Its existing `async_set` already inserts into both indexes via the refactored store.

## Settings

Added to `olmlx/config.py`:

```python
prompt_cache_radix: bool = True
prompt_cache_ram_budget_gb: Annotated[float, Field(gt=0)] = 8.0
prompt_cache_radix_min_prefix_tokens: Annotated[int, Field(ge=0)] = 256
```

Env: `OLMLX_PROMPT_CACHE_RADIX`, `OLMLX_PROMPT_CACHE_RAM_BUDGET_GB`, `OLMLX_PROMPT_CACHE_RADIX_MIN_PREFIX_TOKENS`.

Existing `prompt_cache_max_slots` retained; documented as secondary cap.

## Testing

New file `tests/test_prompt_cache_radix.py`:

**Unit (trie):**
- Insert two non-overlapping token sequences → `find_longest_prefix` returns the right cache_id for prompts that extend either.
- Insert a sequence, then a deeper sequence sharing its prefix → query with each prompt returns the correct (deepest) match.
- Remove a sequence → its prefix branch is pruned where it has no other terminal/children.
- Prefix below threshold → caller can apply the floor; trie itself returns the raw depth.

**Unit (store):**
- `takeover`: state survives, new cache_id resolves, old cache_id is `None`.
- `find_by_prefix`: returns the deepest-matching entry; respects `min_prefix_tokens`; returns `None` on empty store or under-threshold match.
- Byte budget evicts LRU entry across the budget boundary; metrics increment.
- Index invariant: after any sequence of set/remove/takeover, the trie's terminals exactly match the keys of `_entries`.

**Integration (mocked engine):**
- Two cache_ids, same 1024-token system prompt, different 64-token user turns. Second request must report `cache_read_tokens ≥ 1024` (radix hit) and ≤ 64 `cache_creation_tokens`.
- Takeover invalidates first cache_id: a third request with the first cache_id and the original prompt must miss exact and either radix-hit the new owner or cold-prefill, never silently use stale state.
- `prompt_cache_radix=False` disables the new path entirely; behaviour matches today.

**Regression:** all existing `tests/test_prompt_cache.py` and `tests/test_prompt_cache_store.py` tests pass unchanged.

## Risks

| Risk | Mitigation |
| --- | --- |
| Trie memory bloat under pathological many-session workloads | Trie size is bounded by sum-of-stored-cache token counts; same order as the existing `_entries` keying. Worst case ~50 MB at 4-slot, 32k-token defaults. |
| Takeover corrupts an active concurrent request on the old cache_id | The store currently has no per-entry active-ref guard. If this proves a real problem in practice, the follow-up is hybrid takeover-or-clone gated on per-entry refs. v1 documents the limitation. |
| Stale terminal in the trie after `_entries` mutation | Every `_entries` mutation routes through helpers that update both indexes atomically (no public path mutates `_entries` directly post-refactor). Invariant asserted in tests. |
| Behaviour drift in disk-tier code paths during the refactor | The refactor is a move-and-rename, not a redesign. Existing tests gate the diff. The disk-tier byte counters are new but additive. |

## Rollout

Single PR. The feature is on by default (`prompt_cache_radix=True`) and gated by the existing `prompt_cache=True` flag for full disable. CLAUDE.md gains a paragraph under the Prompt caching bullet describing the radix tier and its takeover semantics.
