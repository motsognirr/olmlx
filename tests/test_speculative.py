"""Tests for model-agnostic speculative decoder (engine/speculative.py)."""

import mlx.core as mx
import pytest

from tests.test_flash_speculative import MockModel


class TestSpeculativeDecoder:
    """Tests for the base SpeculativeDecoder class (no Flash dependencies)."""

    @pytest.fixture()
    def decoder(self):
        from olmlx.engine.speculative import SpeculativeDecoder

        vocab_size, hidden_size = 32, 16
        draft = MockModel(vocab_size, hidden_size)
        target = MockModel(vocab_size, hidden_size)
        return SpeculativeDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=4,
        )

    @pytest.fixture()
    def shared_decoder(self):
        """Decoder with shared weights so draft and target always agree."""
        from olmlx.engine.speculative import SpeculativeDecoder

        vocab_size, hidden_size = 32, 16
        draft = MockModel(vocab_size, hidden_size)
        target = MockModel(vocab_size, hidden_size)
        target.embed.weight = draft.embed.weight
        target.lm_head.weight = draft.lm_head.weight
        for i in range(len(draft.layers)):
            target.layers[i] = draft.layers[i]
        return SpeculativeDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=3,
        )

    def test_init(self, decoder):
        assert decoder._lambda == 4
        assert 0 < decoder._alpha <= 1.0

    def test_no_prefetcher_attribute(self, decoder):
        """Base class should not have a _prefetcher attribute."""
        assert not hasattr(decoder, "_prefetcher")

    def test_prefill_creates_caches(self, shared_decoder):
        prompt = mx.array([[1, 2, 3, 4, 5]])
        first_token = shared_decoder.prefill(prompt)

        assert shared_decoder._target_cache is not None
        assert shared_decoder._draft_cache is not None
        assert shared_decoder._cache_seq_len == 5
        assert isinstance(first_token, int)
        assert 0 <= first_token < 32

    def test_step_returns_accepted_tokens(self, shared_decoder):
        prompt = mx.array([[1, 2, 3]])
        shared_decoder.prefill(prompt)

        accepted, num_draft = shared_decoder.step()

        assert len(accepted) >= 1
        assert num_draft == 3
        assert all(isinstance(t, int) for t in accepted)

    def test_verify_rejects_divergent_tokens(self):
        from olmlx.engine.speculative import SpeculativeDecoder

        vocab_size = 8
        target_probs = mx.zeros((4, vocab_size))
        target_probs = target_probs.at[:, 7].add(100.0)

        decoder = SpeculativeDecoder.__new__(SpeculativeDecoder)
        decoder._lambda = 3
        decoder._alpha = 0.5
        decoder._alpha_ema = 0.9

        draft_tokens = [0, 0, 0]
        accepted = decoder._verify(draft_tokens, target_probs)

        assert len(accepted) >= 1
        assert accepted[0] == 7

    def test_full_acceptance_with_shared_weights(self, shared_decoder):
        prompt = mx.array([[1, 2, 3]])
        shared_decoder.prefill(prompt)

        accepted, _ = shared_decoder.step()
        assert len(accepted) == 4  # 3 draft + 1 bonus

    def test_cache_trimmed_on_rejection(self):
        from olmlx.engine.speculative import SpeculativeDecoder

        vocab_size, hidden_size = 32, 16
        draft = MockModel(vocab_size, hidden_size)
        target = MockModel(vocab_size, hidden_size)
        decoder = SpeculativeDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=4,
        )

        prompt = mx.array([[1, 2, 3]])
        decoder.prefill(prompt)
        accepted, _ = decoder.step()

        assert decoder._cache_seq_len == 3 + len(accepted)
        assert decoder._target_cache[0].offset == decoder._cache_seq_len

    def test_reset_clears_state(self, shared_decoder):
        prompt = mx.array([[1, 2, 3]])
        shared_decoder.prefill(prompt)
        shared_decoder.step()

        shared_decoder.reset()

        assert shared_decoder._target_cache is None
        assert shared_decoder._draft_cache is None
        assert shared_decoder._cache_seq_len == 0
        assert shared_decoder._last_target_logit is None

    def test_generate_step_stateless(self, shared_decoder):
        prompt = mx.array([[1, 2, 3]])
        accepted, num_draft = shared_decoder.generate_step(prompt)

        assert len(accepted) >= 1
        assert num_draft == 3

    def test_multi_step_consistency(self):
        from olmlx.engine.speculative import SpeculativeDecoder

        vocab_size, hidden_size = 32, 16
        draft = MockModel(vocab_size, hidden_size)
        target = MockModel(vocab_size, hidden_size)
        decoder = SpeculativeDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=3,
        )

        prompt = mx.array([[1, 2, 3, 4, 5]])
        decoder.prefill(prompt)

        for _ in range(5):
            accepted, _ = decoder.step()
            assert len(accepted) >= 1
            assert decoder._target_cache[0].offset == decoder._cache_seq_len

    def test_acceptance_rate_in_valid_range(self, decoder):
        prompt = mx.array([[1, 2, 3]])
        decoder.generate_step(prompt)
        assert 0 <= decoder._alpha <= 1.0
