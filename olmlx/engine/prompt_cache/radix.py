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
        """Walk the trie matching tokens, return any cache_id reachable
        from the deepest visited node.

        The shared prefix length is the depth of the deepest visited
        node — not the depth of the returned terminal. A stored
        ``[1, 2, 3, 4, 5]`` (terminal at depth 5) and a query
        ``[1, 2, 3, 9]`` share three tokens; this returns the stored
        cache_id with ``prefix_len == 3`` so the caller can take it
        over and trim its KV cache back to align.

        Returns ``(None, 0)`` only when the query has no token in
        common with anything stored.
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
        if deepest_depth == 0:
            return None, 0
        # Prefer a terminal at the deepest visited node itself; otherwise
        # any terminal in its subtree shares the same prefix-length with
        # the query (the descent path), so DFS for one. The remove()
        # pruning invariant guarantees a terminal exists somewhere below
        # any non-leaf node.
        if deepest_node.terminal_cache_id is not None:
            return deepest_node.terminal_cache_id, deepest_depth
        stack = [deepest_node]
        while stack:
            n = stack.pop()
            if n.terminal_cache_id is not None:
                return n.terminal_cache_id, deepest_depth
            stack.extend(n.children.values())
        return None, 0  # unreachable if pruning invariant holds

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
