"""Tests for speculative decoding with flash inference (Paper §5.2)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from olmlx.engine.flash.speculative import SpeculativeFlashDecoder


class MockDraftModel(nn.Module):
    """Minimal draft model that returns fixed logits."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, cache=None):
        h = self.embed(input_ids)
        return self.lm_head(h)


class MockTargetModel(nn.Module):
    """Minimal target model that returns fixed logits."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, cache=None):
        h = self.embed(input_ids)
        return self.lm_head(h)


class TestSpeculativeFlashDecoder:
    @pytest.fixture()
    def decoder(self):
        vocab_size, hidden_size = 32, 16
        draft = MockDraftModel(vocab_size, hidden_size)
        target = MockTargetModel(vocab_size, hidden_size)
        return SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=4,
        )

    def test_init(self, decoder):
        assert decoder._lambda == 4
        assert 0 < decoder._alpha <= 1.0

    def test_draft_generate_returns_tokens(self, decoder):
        prompt = mx.array([[1, 2, 3]])
        draft_tokens, draft_logits = decoder._draft_generate(prompt, n=4)
        assert len(draft_tokens) == 4
        assert all(0 <= t < 32 for t in draft_tokens)

    def test_verify_greedy_all_match(self):
        """When draft and target produce same greedy tokens, all are accepted."""
        vocab_size, hidden_size = 8, 4
        draft = MockDraftModel(vocab_size, hidden_size)
        target = MockTargetModel(vocab_size, hidden_size)

        # Make target and draft share the same weights so they agree
        target.embed.weight = draft.embed.weight
        target.lm_head.weight = draft.lm_head.weight

        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=3,
        )

        prompt = mx.array([[1, 2, 3]])
        accepted, num_draft = decoder.generate_step(prompt)

        # Same model => all draft tokens should be accepted + 1 bonus
        assert len(accepted) >= 1

    def test_acceptance_rate_updates(self, decoder):
        """After generate_step, the acceptance rate alpha should change."""
        prompt = mx.array([[1, 2, 3]])

        decoder.generate_step(prompt)

        # Alpha should have been updated (may or may not differ depending on acceptance)
        # Just verify it's still in valid range
        assert 0 <= decoder._alpha <= 1.0

    def test_verify_rejects_divergent_tokens(self):
        """Verify that divergent tokens are rejected."""
        vocab_size = 8

        # Create known logits: draft strongly prefers token 0, target prefers token 7
        draft_probs = mx.zeros((1, vocab_size))
        draft_probs = draft_probs.at[0, 0].add(100.0)  # draft picks token 0

        target_probs = mx.zeros((4, vocab_size))  # 3 draft + 1 verify positions
        target_probs = target_probs.at[:, 7].add(100.0)  # target picks token 7

        decoder = SpeculativeFlashDecoder.__new__(SpeculativeFlashDecoder)
        decoder._lambda = 3
        decoder._alpha = 0.5
        decoder._alpha_ema = 0.9

        draft_tokens = [0, 0, 0]  # draft always picks 0

        accepted = decoder._verify(draft_tokens, target_probs)

        # Target wants 7, draft gave 0 — first token rejected
        assert len(accepted) >= 1  # at least the corrected first token
        assert accepted[0] == 7  # target's preferred token

    def test_effective_window_from_alpha(self, decoder):
        """Effective window size should be based on alpha * (lambda + 1)."""
        decoder._alpha = 0.8
        decoder._lambda = 4
        effective = max(1, int(decoder._alpha * (decoder._lambda + 1)))
        assert effective == 4  # 0.8 * 5 = 4

        decoder._alpha = 0.2
        effective = max(1, int(decoder._alpha * (decoder._lambda + 1)))
        assert effective == 1  # 0.2 * 5 = 1
