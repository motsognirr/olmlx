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

    def test_prefill_long_prompt_gives_valid_token(self, shared_decoder):
        """prefill() with a long prompt must return a valid token and populate caches."""
        prompt = mx.array([list(range(20))])
        first_token = shared_decoder.prefill(prompt)

        assert 0 <= first_token < 32
        assert shared_decoder._cache_seq_len == 20
        assert shared_decoder._target_cache is not None


class TestPrefillLastLogit:
    """Tests for _prefill_last_logit: two-pass prefill to avoid materialising
    the full [batch, seq_len, vocab] logit matrix."""

    @pytest.fixture()
    def model(self):
        return MockModel(32, 16)

    def test_single_token_returns_vocab_shaped_logit(self, model):
        from mlx_lm.models.cache import make_prompt_cache

        from olmlx.engine.speculative import _prefill_last_logit

        cache = make_prompt_cache(model)
        result = _prefill_last_logit(model, mx.array([[7]]), cache)
        assert result.shape == (32,)

    def test_multi_token_returns_vocab_shaped_logit(self, model):
        from mlx_lm.models.cache import make_prompt_cache

        from olmlx.engine.speculative import _prefill_last_logit

        cache = make_prompt_cache(model)
        result = _prefill_last_logit(model, mx.array([[1, 2, 3, 4, 5]]), cache)
        assert result.shape == (32,)

    def test_matches_naive_single_token(self, model):
        """Single-token result must match naive forward."""
        from mlx_lm.models.cache import make_prompt_cache

        from olmlx.engine.speculative import _logits, _prefill_last_logit

        prompt = mx.array([[7]])
        cache_naive = make_prompt_cache(model)
        naive = _logits(model(prompt, cache=cache_naive))[0, -1, :]
        mx.eval(naive)

        cache_2p = make_prompt_cache(model)
        result = _prefill_last_logit(model, prompt, cache_2p)
        mx.eval(result)

        assert mx.allclose(result, naive, atol=1e-5)

    def test_matches_naive_multi_token(self, model):
        """Smoke test: two-pass result shape matches naive forward on MockModel.

        Note: MockModel is cache-agnostic, so this does not verify numerical
        correctness of the KV-cache split for real causal transformers. It
        validates that ``_prefill_last_logit`` runs end-to-end and returns
        the expected shape, not that pass 2 sees the correct cached state.
        """
        from mlx_lm.models.cache import make_prompt_cache

        from olmlx.engine.speculative import _logits, _prefill_last_logit

        prompt = mx.array([[1, 2, 3, 4, 5]])
        cache_naive = make_prompt_cache(model)
        naive = _logits(model(prompt, cache=cache_naive))[0, -1, :]
        mx.eval(naive)

        cache_2p = make_prompt_cache(model)
        result = _prefill_last_logit(model, prompt, cache_2p)
        mx.eval(result)

        assert mx.allclose(result, naive, atol=1e-5)

    def test_cache_offset_equals_prompt_length(self, model):
        """After _prefill_last_logit, KV cache offset must equal prompt length."""
        from mlx_lm.models.cache import make_prompt_cache

        from olmlx.engine.speculative import _prefill_last_logit

        prompt = mx.array([[1, 2, 3, 4, 5]])
        cache = make_prompt_cache(model)
        _prefill_last_logit(model, prompt, cache)
        # Force evaluation so offset is committed.
        mx.eval(cache[0].keys, cache[0].values)

        assert cache[0].offset == prompt.shape[1]
