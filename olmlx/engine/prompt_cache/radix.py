"""Token-prefix trie for cross-request prompt cache lookup (issue #365)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    # A path can be claimed by more than one cache_id when distinct
    # entries in the store happen to land on identical token sequences
    # (rare in practice but the store doesn't prevent it). Tracking a
    # set keeps every entry in _entries reachable through the trie.
    terminal_cache_ids: set[str] = field(default_factory=set)


class PrefixCacheIndex:
    """Trie over token IDs. Each terminal node tracks the set of
    cache_ids that claim that exact path.

    Lookups walk the query tokens as far as possible, then return any
    reachable terminal in the subtree of the deepest visited node —
    so a sibling whose stored sequence diverges past the shared prefix
    is still found. The returned ``prefix_len`` is the descent depth
    (the actual shared-prefix length), not the terminal's depth.

    Complexity: insert/remove are O(len(tokens)) with O(1) per step.
    find_longest_prefix is O(len(tokens)) on the happy path plus a
    bounded DFS into one subtree on divergence.
    """

    def __init__(self) -> None:
        self._root = _TrieNode()

    def insert(self, tokens: list[int], cache_id: str) -> None:
        """Add cache_id to the terminal set at the tokens path."""
        node = self._root
        for tok in tokens:
            child = node.children.get(tok)
            if child is None:
                child = _TrieNode()
                node.children[tok] = child
            node = child
        node.terminal_cache_ids.add(cache_id)

    def find_longest_prefix(
        self, tokens: list[int], min_depth: int = 0
    ) -> tuple[str | None, int]:
        """Walk the trie matching tokens, return any cache_id reachable
        from the deepest visited node.

        The shared prefix length is the depth of the deepest visited
        node — not the depth of the returned terminal. A stored
        ``[1, 2, 3, 4, 5]`` (terminal at depth 5) and a query
        ``[1, 2, 3, 9]`` share three tokens; this returns the stored
        cache_id with ``prefix_len == 3`` so the caller can take it
        over and trim its KV cache back to align.

        ``min_depth`` short-circuits the subtree DFS when the descent
        depth is already below the caller's acceptance threshold —
        cheap reject for queries that only share a BOS token with a
        fat subtree of stored entries.

        Returns ``(None, 0)`` when the query has no token in common
        with anything stored, or when the descent depth is below
        ``min_depth``.
        """
        node = self._root
        deepest_node = node
        deepest_depth = 0
        for depth, tok in enumerate(tokens, start=1):
            child = node.children.get(tok)
            if child is None:
                break
            node = child
            deepest_node = node
            deepest_depth = depth
        if deepest_depth == 0 or deepest_depth < min_depth:
            return None, 0
        # Prefer a terminal at the deepest visited node itself; otherwise
        # any terminal in its subtree shares the same prefix-length with
        # the query (the descent path), so DFS for one. The remove()
        # pruning invariant guarantees a terminal exists somewhere below
        # any non-leaf node.
        if deepest_node.terminal_cache_ids:
            return next(iter(deepest_node.terminal_cache_ids)), deepest_depth
        stack = [deepest_node]
        while stack:
            n = stack.pop()
            if n.terminal_cache_ids:
                return next(iter(n.terminal_cache_ids)), deepest_depth
            stack.extend(n.children.values())
        return None, 0  # unreachable if pruning invariant holds

    def find_strict_prefix(
        self, tokens: list[int], min_depth: int = 0
    ) -> tuple[str | None, int]:
        """Return the deepest terminal whose stored token sequence is a
        *proper* prefix of ``tokens`` (terminal depth < len(tokens) and
        every token along the path matches).

        Distinct from ``find_longest_prefix`` which also surfaces siblings
        that diverge past the shared prefix. Strict-prefix is the lookup
        the non-trimmable checkpoint path needs: only safe to reuse a stored
        cache when its tokens are a proper prefix of the new request — no
        trim required, no divergence to discard, and *no exact-length
        match* either. An exact-length match would leave the checkpoint
        driver with no prefill work to do (``already_covered == len(flat)``)
        while still handing the last prompt token back to stream_generate as
        the decode seed — re-feeding that token at an extra position
        relative to the cache yields wrong output (the model sees the
        prompt's last token twice).

        ``min_depth`` short-circuits below-threshold matches with
        ``(None, 0)`` (consistent with ``find_longest_prefix``).
        """
        node = self._root
        best_cid: str | None = None
        best_depth = 0
        # Walk at most len(tokens) - 1 tokens: a proper prefix of ``tokens``
        # has length strictly less than len(tokens).
        max_depth = len(tokens) - 1
        for depth, tok in enumerate(tokens, start=1):
            if depth > max_depth:
                break
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

    def remove(self, tokens: list[int], cache_id: str) -> None:
        """Discard cache_id from the terminal set at the tokens path,
        then prune now-empty branches upward.

        No-op if the path doesn't exist or cache_id isn't claiming it.
        """
        path: list[tuple[_TrieNode, int]] = []
        node = self._root
        for tok in tokens:
            child = node.children.get(tok)
            if child is None:
                return
            path.append((node, tok))
            node = child
        if cache_id not in node.terminal_cache_ids:
            return
        node.terminal_cache_ids.discard(cache_id)
        # Prune upward while nodes have no children and no terminals.
        for parent, tok in reversed(path):
            child = parent.children[tok]
            if child.children or child.terminal_cache_ids:
                return
            del parent.children[tok]
