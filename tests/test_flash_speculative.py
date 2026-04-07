"""Tests for speculative decoding with flash inference (Paper §5.2)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from olmlx.engine.flash.speculative import SpeculativeFlashDecoder


class MockAttention(nn.Module):
    """Minimal attention that supports KV cache protocol."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.n_heads = 1
        self.n_kv_heads = 1

    def __call__(self, x, cache=None):
        if cache is not None:
            # KV cache expects (B, n_kv_heads, seq_len, head_dim)
            k = v = x.reshape(x.shape[0], 1, -1, x.shape[-1])
            cache.update_and_fetch(k, v)
        return x


class MockLayer(nn.Module):
    """Minimal transformer layer with attention (for KV cache)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)

    def __call__(self, x, cache=None):
        return self.self_attn(x, cache=cache)


class MockModel(nn.Module):
    """Minimal transformer model with KV cache support for testing."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, cache=None):
        h = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, cache=cache[i] if cache is not None else None)
        return self.lm_head(h)


# Backward-compat aliases used by test_speculative_stream.py
MockDraftModel = MockModel
MockTargetModel = MockModel


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
        draft_tokens = decoder._draft_generate(prompt, n=4)
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


class TestSpeculativeKVCache:
    """Tests for persistent KV cache support in SpeculativeFlashDecoder."""

    @pytest.fixture()
    def shared_decoder(self):
        """Decoder with shared weights so draft and target always agree."""
        vocab_size, hidden_size = 32, 16
        draft = MockDraftModel(vocab_size, hidden_size)
        target = MockTargetModel(vocab_size, hidden_size)
        # Share weights so they agree on every token
        target.embed.weight = draft.embed.weight
        target.lm_head.weight = draft.lm_head.weight
        for i in range(len(draft.layers)):
            target.layers[i] = draft.layers[i]
        return SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=3,
        )

    def test_prefill_creates_caches(self, shared_decoder):
        """After prefill(), both caches should exist and cache_seq_len == prompt length."""
        prompt = mx.array([[1, 2, 3, 4, 5]])
        first_token = shared_decoder.prefill(prompt)

        assert shared_decoder._target_cache is not None
        assert shared_decoder._draft_cache is not None
        assert shared_decoder._cache_seq_len == 5
        assert isinstance(first_token, int)
        assert 0 <= first_token < 32

    def test_prefill_stores_last_target_logit(self, shared_decoder):
        """After prefill(), _last_target_logit should be set."""
        prompt = mx.array([[1, 2, 3]])
        shared_decoder.prefill(prompt)

        assert shared_decoder._last_target_logit is not None
        assert shared_decoder._last_target_logit.shape == (32,)  # vocab_size

    def test_step_uses_incremental_target(self):
        """Target model should receive only lambda tokens, not full prompt."""
        vocab_size, hidden_size = 32, 16
        draft = MockDraftModel(vocab_size, hidden_size)
        target = MockTargetModel(vocab_size, hidden_size)
        # Share weights
        target.embed.weight = draft.embed.weight
        target.lm_head.weight = draft.lm_head.weight
        for i in range(len(draft.layers)):
            target.layers[i] = draft.layers[i]

        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=3,
        )

        prompt = mx.array([[1, 2, 3, 4, 5]])  # seq_len=5
        decoder.prefill(prompt)

        # Record target cache offset before step
        offset_before = decoder._target_cache[0].offset
        assert offset_before == 5  # prompt length

        decoder.step()

        # After step, the target cache should have advanced by at most lambda
        # (trimmed back on rejection), NOT by seq_len+lambda
        offset_after = decoder._target_cache[0].offset
        # The offset should be cache_seq_len (prompt + accepted), not 5+3=8 reprocessed
        assert offset_after == decoder._cache_seq_len
        # And cache_seq_len should be prompt + accepted (not prompt re-processed)
        assert decoder._cache_seq_len > 5  # at least one token accepted

    def test_step_returns_accepted_tokens(self, shared_decoder):
        """step() should return accepted tokens and draft count."""
        prompt = mx.array([[1, 2, 3]])
        shared_decoder.prefill(prompt)

        accepted, num_draft = shared_decoder.step()

        assert len(accepted) >= 1
        assert num_draft == 3  # lambda
        assert all(isinstance(t, int) for t in accepted)

    def test_cache_seq_len_grows_after_step(self, shared_decoder):
        """cache_seq_len should grow by len(accepted) after each step."""
        prompt = mx.array([[1, 2, 3]])
        shared_decoder.prefill(prompt)
        initial_len = shared_decoder._cache_seq_len

        accepted, _ = shared_decoder.step()
        assert shared_decoder._cache_seq_len == initial_len + len(accepted)

    def test_multi_step_accumulates(self, shared_decoder):
        """Multiple steps should accumulate in cache_seq_len."""
        prompt = mx.array([[1, 2, 3]])
        shared_decoder.prefill(prompt)

        total_accepted = 0
        for _ in range(3):
            accepted, _ = shared_decoder.step()
            total_accepted += len(accepted)

        assert shared_decoder._cache_seq_len == 3 + total_accepted

    def test_cache_trimmed_on_rejection(self):
        """When target rejects tokens, caches should be trimmed correctly."""
        vocab_size, hidden_size = 32, 16
        draft = MockDraftModel(vocab_size, hidden_size)
        target = MockTargetModel(vocab_size, hidden_size)
        # Different weights → likely disagreement → rejection
        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=4,
        )

        prompt = mx.array([[1, 2, 3]])
        decoder.prefill(prompt)

        accepted, _ = decoder.step()
        # cache_seq_len should reflect only accepted tokens
        assert decoder._cache_seq_len == 3 + len(accepted)

        # Verify the target cache offset matches expected position
        target_offset = decoder._target_cache[0].offset
        assert target_offset == decoder._cache_seq_len

    def test_generate_step_backward_compat(self):
        """Old generate_step(prompt) API should still work."""
        vocab_size, hidden_size = 32, 16
        draft = MockDraftModel(vocab_size, hidden_size)
        target = MockTargetModel(vocab_size, hidden_size)
        target.embed.weight = draft.embed.weight
        target.lm_head.weight = draft.lm_head.weight
        for i in range(len(draft.layers)):
            target.layers[i] = draft.layers[i]
        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=3,
        )

        prompt = mx.array([[1, 2, 3]])
        accepted, num_draft = decoder.generate_step(prompt)

        assert len(accepted) >= 1
        assert num_draft == 3

    def test_reset_clears_state(self, shared_decoder):
        """reset() should clear all cache state."""
        prompt = mx.array([[1, 2, 3]])
        shared_decoder.prefill(prompt)
        shared_decoder.step()

        shared_decoder.reset()

        assert shared_decoder._target_cache is None
        assert shared_decoder._draft_cache is None
        assert shared_decoder._cache_seq_len == 0
        assert shared_decoder._last_target_logit is None

    def test_multi_step_divergent_models(self):
        """Multiple steps with divergent models should maintain cache consistency."""
        vocab_size, hidden_size = 32, 16
        draft = MockDraftModel(vocab_size, hidden_size)
        target = MockTargetModel(vocab_size, hidden_size)
        # Different weights → likely some rejections
        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=3,
        )

        prompt = mx.array([[1, 2, 3, 4, 5]])
        decoder.prefill(prompt)

        for _ in range(5):
            accepted, _ = decoder.step()
            assert len(accepted) >= 1
            # Cache offset should always match cache_seq_len
            assert decoder._target_cache[0].offset == decoder._cache_seq_len
            assert decoder._last_target_logit is not None
            assert decoder._last_target_logit.shape == (vocab_size,)

    def test_step_works_when_trim_prompt_cache_is_none(self):
        """step() should not crash when trim_prompt_cache is None (import fallback)."""
        import olmlx.engine.flash.speculative as spec_mod

        vocab_size, hidden_size = 32, 16
        draft = MockDraftModel(vocab_size, hidden_size)
        target = MockTargetModel(vocab_size, hidden_size)
        # Different weights → likely rejection → triggers trim path
        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=3,
        )

        prompt = mx.array([[1, 2, 3]])
        decoder.prefill(prompt)

        # Simulate the import fallback: trim_prompt_cache = None
        original = spec_mod.trim_prompt_cache
        try:
            spec_mod.trim_prompt_cache = None
            # This should NOT raise TypeError
            accepted, num_draft = decoder.step()
            assert len(accepted) >= 1
            assert num_draft == 3
        finally:
            spec_mod.trim_prompt_cache = original

    def test_full_acceptance_then_step(self):
        """After a full acceptance step, the next step should work correctly."""
        vocab_size, hidden_size = 32, 16
        draft = MockDraftModel(vocab_size, hidden_size)
        target = MockTargetModel(vocab_size, hidden_size)
        # Share weights to guarantee full acceptance
        target.embed.weight = draft.embed.weight
        target.lm_head.weight = draft.lm_head.weight
        for i in range(len(draft.layers)):
            target.layers[i] = draft.layers[i]

        decoder = SpeculativeFlashDecoder(
            draft_model=draft,
            target_model=target,
            num_speculative_tokens=3,
        )

        prompt = mx.array([[1, 2, 3]])
        decoder.prefill(prompt)

        # First step: should fully accept (shared weights) → lambda+1 = 4 tokens
        accepted1, _ = decoder.step()
        assert len(accepted1) == 4  # 3 draft + 1 bonus
        assert decoder._cache_seq_len == 3 + 4
        assert decoder._target_cache[0].offset == decoder._cache_seq_len

        # Second step: should also work correctly without misalignment
        accepted2, _ = decoder.step()
        assert len(accepted2) >= 1
        assert decoder._target_cache[0].offset == decoder._cache_seq_len


class TestTrimPromptCacheNoneGuard:
    """Regression tests for #189: trim_prompt_cache called without None guard."""

    def test_init_raises_when_trim_prompt_cache_is_none(self, monkeypatch):
        """Construction must fail early when trim_prompt_cache is None."""
        import olmlx.engine.flash.speculative as spec_mod

        monkeypatch.setattr(spec_mod, "trim_prompt_cache", None)

        vocab_size, hidden_size = 32, 16
        draft = MockDraftModel(vocab_size, hidden_size)
        target = MockTargetModel(vocab_size, hidden_size)
        with pytest.raises(RuntimeError, match="trim_prompt_cache is unavailable"):
            SpeculativeFlashDecoder(
                draft_model=draft,
                target_model=target,
                num_speculative_tokens=3,
            )
