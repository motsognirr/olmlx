"""Tests for tree-structured speculative verification (engine/tree_speculative.py)."""

import mlx.core as mx
import pytest

from olmlx.engine.tree_speculative import (
    build_comb_tree,
    build_full_tree,
    build_tree_attention_mask,
    extract_top_k_from_logits,
    verify_tree_greedy,
)


class TestBuildCombTree:
    def test_basic_structure(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20, 30],
            alt_tokens_per_step=[[11], [21], [31]],
        )
        # 1 root + 3 primary + 3 siblings = 7
        assert tree.num_nodes == 7
        assert tree.tokens == [1, 10, 20, 30, 11, 21, 31]
        assert tree.parent_indices == [-1, 0, 1, 2, 0, 1, 2]
        assert tree.depths == [0, 1, 2, 3, 1, 2, 3]
        assert tree.primary_branch == [0, 1, 2, 3]
        assert tree.num_draft_levels == 3

    def test_empty_alternatives(self):
        tree = build_comb_tree(
            pending_token=5,
            primary_tokens=[10, 20],
            alt_tokens_per_step=[[], []],
        )
        # Root + 2 primary = 3 (no siblings)
        assert tree.num_nodes == 3
        assert tree.tokens == [5, 10, 20]

    def test_mixed_alternatives(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20, 30],
            alt_tokens_per_step=[[11, 12], [], [31]],
        )
        # Root + 3 primary + 2 (level 1) + 0 (level 2) + 1 (level 3) = 7
        assert tree.num_nodes == 7
        assert tree.parent_indices == [-1, 0, 1, 2, 0, 0, 2]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_comb_tree(
                pending_token=1,
                primary_tokens=[10, 20],
                alt_tokens_per_step=[[11]],  # too short
            )

    def test_ancestor_path(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20, 30],
            alt_tokens_per_step=[[11], [21], [31]],
        )
        # D3 primary is at index 3
        ancestors = tree.get_ancestors(3)
        assert ancestors == [0, 1, 2, 3]  # root, D1, D2, D3

        # Sibling at level 1 (index 4)
        ancestors_4 = tree.get_ancestors(4)
        assert ancestors_4 == [0, 4]  # root, D1_sib

        # Sibling at level 2 (index 5)
        ancestors_5 = tree.get_ancestors(5)
        assert ancestors_5 == [0, 1, 5]  # root, D1, D2_sib


class TestBuildFullTree:
    def test_simple_tree(self):
        tree = build_full_tree(
            pending_token=100,
            draft_tokens_by_step=[
                [(10, 0), (11, 0)],  # depth 1: 2 children of root
                [(20, 0), (21, 1)],  # depth 2: child of 10, child of 11
            ],
            max_nodes=16,
        )
        # depth 0: 1 node (root)
        # depth 1: 2 nodes
        # depth 2: 2 nodes
        # total: 5
        assert tree.num_nodes == 5

    def test_max_nodes_prune(self):
        tree = build_full_tree(
            pending_token=1,
            draft_tokens_by_step=[
                [(10, 0), (11, 0), (12, 0)],
                [(20, 0), (21, 0)],
            ],
            max_nodes=4,
        )
        # depth 0: 1
        # depth 1: 3 (total 4 → hit max_nodes)
        # depth 2: 0 (pruned)
        assert tree.num_nodes == 4
        assert tree.num_draft_levels == 1


class TestAttentionMask:
    def test_mask_shape(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20, 30],
            alt_tokens_per_step=[[11], [21], [31]],
        )
        mask = build_tree_attention_mask(tree)
        assert mask.shape == (1, 1, 7, 7)

    def test_root_only_sees_itself(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20],
            alt_tokens_per_step=[[], []],
        )
        mask = build_tree_attention_mask(tree)
        mask_2d = mask[0, 0]
        n = tree.num_nodes  # 3

        # Root (pos 0) can attend to itself only
        assert mask_2d[0, 0].item() == 0.0
        for j in range(1, n):
            assert mask_2d[0, j].item() < -1e8, f"root should not see pos {j}"

    def test_primary_path_causal(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20, 30],
            alt_tokens_per_step=[[], [], []],
        )
        mask = build_tree_attention_mask(tree)
        mask_2d = mask[0, 0]
        n = 4  # root + 3 primary

        # Position 1 (D1) sees root + itself
        for j in range(n):
            if j <= 1:
                assert mask_2d[1, j].item() == 0.0, f"D1 should see pos {j}"
            else:
                assert mask_2d[1, j].item() < -1e8

        # Position 2 (D2) sees root + D1 + itself
        for j in range(n):
            if j <= 2:
                assert mask_2d[2, j].item() == 0.0, f"D2 should see pos {j}"
            else:
                assert mask_2d[2, j].item() < -1e8

    def test_sibling_isolation(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20],
            alt_tokens_per_step=[[11], [21]],
        )
        mask = build_tree_attention_mask(tree)
        mask_2d = mask[0, 0]
        # D1_sib is at index 3 (after primary path)
        # It should see root (0) and itself (3), NOT D1 (1) or D2 (2)

        # Root: ok
        assert mask_2d[3, 0].item() == 0.0
        # Itself: ok
        assert mask_2d[3, 3].item() == 0.0
        # D1: should NOT see
        assert mask_2d[3, 1].item() < -1e8, "D1_sib should NOT see D1"
        # D2: should NOT see
        assert mask_2d[3, 2].item() < -1e8, "D1_sib should NOT see D2"
        # D2_sib (index 4): should NOT see
        assert mask_2d[3, 4].item() < -1e8

        # D2_sib is at index 4. It should see root (0) and D1 (1), NOT D2 (2)
        assert mask_2d[4, 0].item() == 0.0
        assert mask_2d[4, 1].item() == 0.0, "D2_sib should see D1"
        assert mask_2d[4, 2].item() < -1e8, "D2_sib should NOT see D2"
        assert mask_2d[4, 3].item() < -1e8, "D2_sib should NOT see D1_sib"
        assert mask_2d[4, 4].item() == 0.0


class TestVerifyTreeGreedy:
    def test_full_primary_acceptance(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20, 30],
            alt_tokens_per_step=[[11], [21], [31]],
        )
        vocab_size = 50
        # Set logits: target_choices[0]=10, [1]=20, [2]=30, [3]=42(bonus)
        logits = mx.zeros((tree.num_nodes, vocab_size))
        logits = logits.at[0, 10].add(100.0)
        logits = logits.at[1, 20].add(100.0)
        logits = logits.at[2, 30].add(100.0)
        logits = logits.at[3, 42].add(100.0)

        accepted, used_sibling = verify_tree_greedy(tree, logits)
        assert accepted == [10, 20, 30, 42]
        assert not used_sibling

    def test_primary_rejection_at_position_2(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20, 30],
            alt_tokens_per_step=[[11], [21], [31]],
        )
        vocab_size = 50
        logits = mx.zeros((tree.num_nodes, vocab_size))
        logits = logits.at[0, 10].add(100.0)  # D1 matches
        logits = logits.at[1, 15].add(100.0)  # D2: target picks 15 (mismatch!)

        accepted, used_sibling = verify_tree_greedy(tree, logits)
        assert accepted == [10, 15]
        assert not used_sibling

    def test_sibling_match_at_position_2(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20, 30],
            alt_tokens_per_step=[[11], [21], [31]],
        )
        vocab_size = 50
        logits = mx.zeros((tree.num_nodes, vocab_size))
        logits = logits.at[0, 10].add(100.0)  # D1 matches
        logits = logits.at[1, 15].add(50.0)  # D2 mismatches
        logits = logits.at[1, 21].add(100.0)  # D2_sib matches at parent pos 1

        accepted, used_sibling = verify_tree_greedy(tree, logits)
        assert used_sibling
        assert accepted[0] == 10  # D1
        assert accepted[1] == 21  # D2_sib
        # bonus from target_choices[sib_idx] = target_choices[5] = 0
        assert len(accepted) == 3

    def test_sibling_match_bonus(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20, 30],
            alt_tokens_per_step=[[11], [21], [31]],
        )
        vocab_size = 50
        logits = mx.zeros((tree.num_nodes, vocab_size))
        logits = logits.at[0, 10].add(100.0)  # D1 matches
        logits = logits.at[1, 15].add(50.0)  # D2 mismatches
        logits = logits.at[1, 21].add(100.0)  # D2_sib matches
        # Bonus: target_choices[5] (sibling position 5)
        logits = logits.at[5, 42].add(100.0)  # bonus token for sibling

        accepted, used_sibling = verify_tree_greedy(tree, logits)
        assert used_sibling
        assert accepted == [10, 21, 42]

    def test_sibling_at_position_1(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20, 30],
            alt_tokens_per_step=[[11], [21], [31]],
        )
        vocab_size = 50
        logits = mx.zeros((tree.num_nodes, vocab_size))
        # D1: target_choices[0]=15 (mismatch!)
        logits = logits.at[0, 15].add(50.0)
        # D1_sib: target_choices[0]=11 (larger → match!)
        logits = logits.at[0, 11].add(100.0)
        # Bonus for sibling: target_choices[4] (D1_sib's position = 4)
        logits = logits.at[4, 42].add(100.0)

        accepted, used_sibling = verify_tree_greedy(tree, logits)
        assert used_sibling
        assert accepted == [11, 42]

    def test_logit_count_mismatch_raises(self):
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20],
            alt_tokens_per_step=[[], []],
        )
        logits = mx.zeros((1, 50))  # wrong size
        with pytest.raises(ValueError):
            verify_tree_greedy(tree, logits)


class TestExtractTopK:
    def test_k_1_returns_argmax(self):
        logits = mx.array([0.1, 0.5, 0.3, 0.9, 0.05])
        result = extract_top_k_from_logits(logits, 1)
        assert result == [3]

    def test_k_3_returns_top_3(self):
        logits = mx.array([0.1, 0.5, 0.3, 0.02, 0.9, 0.05])
        result = extract_top_k_from_logits(logits, 3)
        assert result[0] == 4  # 0.9
        assert result[1] == 1  # 0.5
        assert result[2] == 2  # 0.3

    def test_2d_logits(self):
        logits = mx.array([[0.1, 0.5, 0.3, 0.9]])
        result = extract_top_k_from_logits(logits, 2)
        assert result[0] == 3  # 0.9
        assert result[1] == 1  # 0.5


class TestTreeWidth1Degenerate:
    """When tree_width=1, the tree is a single path (degenerate tree).
    verify_tree_greedy should produce the same result as verify_draft_greedy."""

    def test_matches_verify_draft_greedy(self):
        from olmlx.engine.speculative import verify_draft_greedy

        # Build a tree with width=1 (no siblings): just the primary path
        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20, 30, 40],
            alt_tokens_per_step=[[], [], [], []],
        )
        # tree has 5 nodes: root + 4 primary
        # Tree verify checks using parent positions (target_choices[parent_idx])
        # Linear verify checks using consecutive positions (target_choices[i])

        vocab_size = 50
        logits = mx.zeros((tree.num_nodes, vocab_size))
        # In tree: D1 uses target_choices[0], D2 uses target_choices[1], etc.
        # In linear: D1 uses target_choices[0], D2 uses target_choices[1], etc.
        # These match because positions are consecutive in a single-path tree.
        logits = logits.at[0, 10].add(100.0)
        logits = logits.at[1, 20].add(100.0)
        logits = logits.at[2, 30].add(100.0)
        logits = logits.at[3, 15].add(100.0)  # D4 mismatch → accept 15
        # Bonus irrelevant here (no full acceptance)

        tree_accepted, _ = verify_tree_greedy(tree, logits)

        # Linear equivalent
        lin_logits = logits[: len(tree.primary_branch), :]  # 5 positions
        lin_accepted = verify_draft_greedy([10, 20, 30, 40], lin_logits)

        assert tree_accepted == lin_accepted, (
            f"tree: {tree_accepted}, linear: {lin_accepted}"
        )

    def test_full_acceptance_matches(self):
        from olmlx.engine.speculative import verify_draft_greedy

        tree = build_comb_tree(
            pending_token=1,
            primary_tokens=[10, 20],
            alt_tokens_per_step=[[], []],
        )
        vocab_size = 50
        logits = mx.zeros((tree.num_nodes, vocab_size))
        logits = logits.at[0, 10].add(100.0)  # D1
        logits = logits.at[1, 20].add(100.0)  # D2
        logits = logits.at[2, 42].add(100.0)  # bonus

        tree_accepted, _ = verify_tree_greedy(tree, logits)
        lin_logits = logits[:3, :]
        lin_accepted = verify_draft_greedy([10, 20], lin_logits)

        assert tree_accepted == lin_accepted, (
            f"tree: {tree_accepted}, linear: {lin_accepted}"
        )
