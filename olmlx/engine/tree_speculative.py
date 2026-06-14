"""Tree-structured speculative verification.

Builds a tree of draft alternatives and verifies them against the target
model in a single forward pass using a sparse attention mask.

The tree is organised as a flat pre-order sequence where each position
can only attend to its ancestors (via the mask), so sibling branches are
isolated from each other while sharing the common prefix context.

Reference: issue #358.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Sentinel: a large negative value used in additive attention masks to
# block attention between positions that are not in an ancestor
# relationship.
_MASK_BLOCK = -1e9


@dataclass
class TreeDraft:
    """A tree of draft token alternatives for speculative verification.

    The tree is stored in pre-order (parent before children).  Position 0
    is always the root (the pending/seed token).  ``parent_indices[i]``
    gives the flat index of the parent of node *i*; -1 for the root.

    ``primary_branch`` records the indices that form the main (most-likely)
    path from root to leaf, used as the starting point during verification
    before falling back to siblings.
    """

    tokens: list[int]
    parent_indices: list[int]
    primary_branch: list[int] = field(default_factory=list)
    # depth of each node (root=0); populated by build_comb_tree
    depths: list[int] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.tokens)

    @property
    def num_draft_levels(self) -> int:
        """Number of draft levels (excluding root)."""
        if not self.tokens:
            return 0
        return max(self.depths) if self.depths else 0

    def get_ancestors(self, node_idx: int) -> list[int]:
        """Return the ancestor indices of *node_idx* (including itself)."""
        path = [node_idx]
        p = self.parent_indices[node_idx]
        while p >= 0:
            path.append(p)
            p = self.parent_indices[p]
        path.reverse()
        return path


def build_tree_attention_mask(
    tree: TreeDraft,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Build an additive attention mask for a tree-structured sequence.

    Returns a mask of shape ``(1, 1, n, n)`` where *n* is the number of
    tree nodes.  Values are 0.0 (allowed to attend) or ``_MASK_BLOCK``
    (blocked).  Position *i* can attend to position *j* iff *j* is an
    ancestor of *i* (or *j == i* for self-attention).

    The mask shape ``(batch=1, n_heads=1, query_len, key_len)``
    broadcasts correctly with ``mx.fast.scaled_dot_product_attention``.
    """
    n = tree.num_nodes

    # Build the mask as a plain Python/numpy structure first (MLX arrays
    # are immutable — we can't scatter individual entries efficiently with
    # the older ArrayAt API that lacks .set()).
    import numpy as np

    mask_np = np.full((n, n), _MASK_BLOCK, dtype=np.float32)
    for i in range(n):
        mask_np[i, i] = 0.0
        p = tree.parent_indices[i]
        while p >= 0:
            mask_np[i, p] = 0.0
            p = tree.parent_indices[p]

    mask = mx.array(mask_np, dtype=dtype)
    return mask.reshape(1, 1, n, n)


def _build_causal_mask(seq_len: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Build a standard causal attention mask of shape (1, 1, n, n)."""
    mask = mx.full((seq_len, seq_len), _MASK_BLOCK, dtype=dtype)
    mask = mx.triu(mask, k=1)
    return mask.reshape(1, 1, seq_len, seq_len)


def build_comb_tree(
    pending_token: int,
    primary_tokens: list[int],
    alt_tokens_per_step: list[list[int]],
    max_nodes: int = 16,
) -> TreeDraft:
    """Build a tree where the primary path has sibling alternatives at
    each draft level.

    Tree structure (example w=2, λ=3)::

        Root (pending)
        ├── D1⁰ ── D2⁰ ── D3⁰     (primary path)
        ├── D1¹                     (sibling at level 1, leaf)
        ├── D2¹                     (sibling at level 2, leaf)
        └── D3¹                     (sibling at level 3, leaf)

    Sibling nodes are leaves: they have no children in the tree.
    Their "children" are the bonus tokens produced by the target's
    verification forward.

    Args:
        pending_token: The seed token (root).
        primary_tokens: Main draft path ``[D1, D2, ..., Dλ]``.
        alt_tokens_per_step: For each level 0..λ-1, the alternative
            tokens that should appear as siblings.  ``len(alt_tokens_per_step)``
            must equal ``len(primary_tokens)``.  An empty list at a level
            means no siblings for that level.

    Returns:
        A ``TreeDraft`` in pre-order with the root at index 0.
    """
    n_levels = len(primary_tokens)
    if len(alt_tokens_per_step) != n_levels:
        raise ValueError(
            f"alt_tokens_per_step length ({len(alt_tokens_per_step)}) must "
            f"match primary_tokens length ({n_levels})"
        )

    # Phase 1: primary path — indices 0..n_levels (root + primary).
    # Stop building the primary path early if it would exceed max_nodes.
    tokens: list[int] = [pending_token]
    parent_indices: list[int] = [-1]
    depths: list[int] = [0]
    primary_branch: list[int] = [0]

    n_primary = min(n_levels, max_nodes - 1)  # -1 for the already-added root
    for i in range(n_primary):
        tok = primary_tokens[i]
        parent_idx = i
        tokens.append(tok)
        parent_indices.append(parent_idx)
        depths.append(i + 1)
        primary_branch.append(len(tokens) - 1)

    # Phase 2: siblings — one group per level, after the primary path.
    # Stop early when the tree would exceed max_nodes.
    for level in range(n_levels):
        alts = alt_tokens_per_step[level]
        if not alts:
            continue
        parent_idx = level  # parent is the primary node at the SAME depth
        for alt_tok in alts:
            if len(tokens) >= max_nodes:
                break
            tokens.append(alt_tok)
            parent_indices.append(parent_idx)
            depths.append(level + 1)
        if len(tokens) >= max_nodes:
            break

    return TreeDraft(
        tokens=tokens,
        parent_indices=parent_indices,
        primary_branch=primary_branch,
        depths=depths,
    )


def build_full_tree(
    pending_token: int,
    draft_tokens_by_step: list[list[tuple[int, int]]],
    max_nodes: int = 16,
) -> TreeDraft:
    """Build a full K-ary tree from draft alternatives at each level.

    Each level's alternatives list has ``(token_id, parent_index)`` pairs.
    Parent indices refer to the *previous* level's position in the flat
    sequence.  Built breadth-first so that each level's nodes are
    contiguous (important for the final positional layout).

    The tree is pruned to at most *max_nodes* nodes (including the root).

    Args:
        pending_token: The seed token (root).
        draft_tokens_by_step: For each depth 1..D, a list of
            ``(token_id, parent_pos_in_prev_level)`` pairs where the
            parent position is 0-based within the previous level.
        max_nodes: Hard limit on total nodes.

    Returns:
        A ``TreeDraft`` where each node has exactly one parent and the
        primary branch follows the first child at each level.
    """
    # Level 0: root
    tokens: list[int] = [pending_token]
    parent_indices: list[int] = [-1]
    depths: list[int] = [0]
    primary_branch: list[int] = [0]

    level_start_indices: list[int] = [0]  # start index of each level
    # Number of nodes at each level so far (populated incrementally)
    level_sizes: list[int] = [1]

    for depth_idx, level_entries in enumerate(draft_tokens_by_step):
        if len(tokens) >= max_nodes:
            break
        depth = depth_idx + 1
        prev_start = level_start_indices[depth_idx]
        prev_size = level_sizes[depth_idx]
        level_start_indices.append(len(tokens))
        level_node_count = 0

        # Determine which parent gets the primary continuation
        primary_parent_offset = 0  # first parent at this level → primary child

        for parent_offset, (tok, _parent_rel_idx) in enumerate(level_entries):
            if len(tokens) >= max_nodes:
                break
            parent_abs_idx = prev_start + _parent_rel_idx
            if parent_abs_idx < prev_start or parent_abs_idx >= prev_start + prev_size:
                raise ValueError(
                    f"Parent index {_parent_rel_idx} at depth {depth} is out of "
                    f"bounds [0, {prev_size})"
                )
            tokens.append(tok)
            parent_indices.append(parent_abs_idx)
            depths.append(depth)
            level_node_count += 1

            # First child of the first parent is the primary branch
            if parent_offset == 0 and _parent_rel_idx == primary_parent_offset:
                primary_branch.append(len(tokens) - 1)

        level_sizes.append(level_node_count)

    return TreeDraft(
        tokens=tokens,
        parent_indices=parent_indices,
        primary_branch=primary_branch,
        depths=depths,
    )


def verify_tree_greedy(
    tree: TreeDraft,
    target_logits: mx.array,
) -> tuple[list[int], bool]:
    """Verify a tree of draft tokens against target logits (greedy).

    Walks the primary path first.  At each step, if the target's argmax
    at the node's parent position matches the draft token, the token is
    accepted and we continue.  On a mismatch at the primary path, we scan
    sibling nodes at the same depth — if any sibling matches, we accept it
    plus the target's bonus token at the sibling's position and return.

    All draft tokens accepted → take the bonus token from the primary
    path's final position.

    The sibling scan assumes all siblings at depth *d* share the same
    parent as the primary node at depth *d* (true for ``build_comb_tree``,
    but not guaranteed for ``build_full_tree``).  If ``build_full_tree``
    is wired in, the sibling loop must additionally validate that the
    sibling's parent is on the currently accepted prefix path.
    """
    n_logits = target_logits.shape[0]
    n_nodes = tree.num_nodes
    if n_logits != n_nodes:
        # Should never happen if the caller fed the correct tree tokens
        # to the target, but guard defensively.
        raise ValueError(
            f"target_logits has {n_logits} positions but tree has {n_nodes} nodes"
        )

    target_choices = mx.argmax(target_logits, axis=-1)
    mx.eval(target_choices)

    # Index the nodes by depth for fast sibling lookup.
    nodes_by_depth: dict[int, list[int]] = {}
    for i, d in enumerate(tree.depths):
        nodes_by_depth.setdefault(d, []).append(i)

    accepted: list[int] = []

    # Walk the primary branch.
    # Each node's acceptance is checked against the *parent* position's
    # logit: the logit at position p predicts what comes after position p
    # (the token that should occupy position p's child).
    for depth in range(1, tree.num_draft_levels + 1):
        if depth >= len(tree.primary_branch):
            break
        node_idx = tree.primary_branch[depth]
        parent_idx = tree.parent_indices[node_idx]
        draft_token = tree.tokens[node_idx]
        # target_choices[parent_idx] is "what comes after parent"
        target_token = int(target_choices[parent_idx].item())

        if draft_token == target_token:
            accepted.append(draft_token)
        else:
            # Primary mismatch — try siblings at the same depth.
            # The sibling scan assumes all siblings at depth *d* share
            # the same parent as the primary node (true for comb trees,
            # not guaranteed for full trees).  Validate explicitly.
            siblings = nodes_by_depth.get(depth, [])
            primary_parent = parent_idx
            for sib_idx in siblings:
                if sib_idx == node_idx:
                    continue
                sib_parent = tree.parent_indices[sib_idx]
                if sib_parent != primary_parent:
                    continue  # skip siblings from other branches
                sib_token = tree.tokens[sib_idx]
                if sib_token == int(target_choices[sib_parent].item()):
                    accepted.append(sib_token)
                    # Bonus: what comes after the sibling (target_choices[sib_idx]
                    # is the prediction at the sibling's own position).
                    accepted.append(int(target_choices[sib_idx].item()))
                    return accepted, True

            # No sibling matched — accept target's choice at the
            # rejected position and return.
            accepted.append(target_token)
            return accepted, False

    # All primary tokens matched — add bonus token.
    # The bonus is the prediction at the last primary node's position.
    primary_last_idx = tree.primary_branch[-1]
    accepted.append(int(target_choices[primary_last_idx].item()))
    return accepted, False


def _patch_target_for_tree_forward(
    target: Any,
    tree_mask: mx.array,
) -> Any:
    """Temporarily patch a model to use a sparse tree attention mask.

    Replaces the inner transformer's ``__call__`` with a version that
    passes *tree_mask* to each attention layer instead of creating a
    causal mask.  Returns the original ``__call__`` so the caller can
    restore it.

    Supports Qwen3-family models via two common patterns:
    - Text models (``Qwen3ForCausalLM``): the inner transformer lives at
      ``target.model``; ``target.lm_head`` is applied by the outer wrapper.
    - VLM models: the inner transformer lives at
      ``target.language_model``.

    The caller MUST restore the original ``__call__`` after the forward
    pass.  Failure to restore will leak the tree mask into subsequent
    generations.
    """
    # Resolve the inner module that actually runs the transformer layers.
    inner = target
    if hasattr(target, "language_model") and target.language_model is not None:
        inner = target.language_model
    elif hasattr(target, "model") and target.model is not None:
        inner = target.model

    # Detect model structure.
    layers = getattr(inner, "layers", None)
    norm_fn = getattr(inner, "norm", None)
    embed_fn = getattr(inner, "embed_tokens", None)

    if layers is None or norm_fn is None or embed_fn is None:
        raise NotImplementedError(
            f"Target model {type(target).__name__} does not expose "
            "embed_tokens/layers/norm; tree mask injection not supported "
            "for this architecture."
        )

    orig_call = type(inner).__call__

    def _tree_call(
        self_: Any,
        inputs: mx.array,
        cache: Any = None,
        input_embeddings: Any = None,
    ) -> Any:
        h = input_embeddings if input_embeddings is not None else embed_fn(inputs)
        # Cast the additive mask to the hidden-state dtype: current mlx's
        # scaled_dot_product_attention requires the mask to promote to the
        # output dtype, and a float32 mask cannot downcast to a bf16 model.
        # Mirrors mlx-lm's create_attention_mask, which builds in h.dtype.
        mask = tree_mask.astype(h.dtype)
        if cache is None:
            cache = [None] * len(layers)
        for layer, c in zip(layers, cache):
            h = layer(h, mask, c)
        return norm_fn(h)

    type(inner).__call__ = _tree_call  # type: ignore[method-assign]
    return orig_call


def _restore_target(target: Any, orig_call: Any) -> None:
    """Restore the model's original __call__ after tree forward."""
    inner = target
    if hasattr(target, "language_model") and target.language_model is not None:
        inner = target.language_model
    elif hasattr(target, "model") and target.model is not None:
        inner = target.model
    type(inner).__call__ = orig_call  # type: ignore[method-assign]


def extract_top_k_from_logits(
    logits: mx.array,
    k: int,
) -> list[int]:
    """Extract the top-K token IDs from a logit vector.

    Uses ``mx.argpartition`` for k ≪ vocab_size (avoids a full sort).

    Args:
        logits: ``(vocab,)`` or ``(batch, vocab)`` logits.
        k: Number of candidates to return.

    Returns:
        Token IDs of the top-K candidates, sorted by descending score.
    """
    if k <= 1:
        return [int(mx.argmax(logits, axis=-1).item())]

    vocab = logits.shape[-1]
    # argpartition places the smallest kth elements before index kth.
    # We want the largest k, so negate and partition on (k - 1).
    kth = min(k - 1, vocab - 1)
    part_idx = mx.argpartition(-logits, kth=kth, axis=-1)[..., :k]
    # part_idx is not sorted — pull the top-k values and argsort them.
    top_vals = mx.take(logits, part_idx, axis=-1)
    sort_idx = mx.argsort(-top_vals, axis=-1)
    sorted_indices = mx.take(part_idx, sort_idx, axis=-1)
    mx.eval(sorted_indices)
    flat = sorted_indices.flatten()
    return [int(flat[i].item()) for i in range(k)]
