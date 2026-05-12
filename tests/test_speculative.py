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


class _CausalAttention:
    """Minimal scaled-dot-product attention over a real KV cache.

    The pass-2 output for the last position depends on K/V from positions
    0..N-2 via the attention mechanism, so a broken cache split (e.g. pass 1
    not materialised before pass 2) produces numerically different logits
    than a single-pass forward.
    """

    def __init__(self, hidden_size: int):
        rng = mx.random.key(0)
        self.wqkv = mx.random.normal((hidden_size, hidden_size * 3), key=rng) * 0.1
        self.n_heads = 1

    def __call__(self, x, cache=None):
        B, T, D = x.shape
        qkv = x @ self.wqkv
        q, k, v = mx.split(qkv, 3, axis=-1)
        # (B, n_heads=1, T, D)
        q = q.reshape(B, 1, T, D)
        k = k.reshape(B, 1, T, D)
        v = v.reshape(B, 1, T, D)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        scale = 1.0 / (D**0.5)
        scores = (q @ mx.swapaxes(k, -1, -2)) * scale
        # Causal mask over the appended-to history.
        S = k.shape[2]
        offset = S - T
        i = mx.arange(T).reshape(T, 1) + offset
        j = mx.arange(S).reshape(1, S)
        mask = mx.where(j <= i, 0.0, -1e9)
        scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).reshape(B, T, D)
        return out


class _CausalMockModel:
    """Model with real causal attention so KV cache state matters for output."""

    def __init__(self, vocab_size: int, hidden_size: int):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        rng = mx.random.key(1)
        self.embed_w = mx.random.normal((vocab_size, hidden_size), key=rng) * 0.1
        self.attn = _CausalAttention(hidden_size)
        self.lm_head_w = mx.random.normal((hidden_size, vocab_size), key=rng) * 0.1
        # KV cache protocol expects model.layers iterable for make_prompt_cache
        self.layers = [self.attn]

    def __call__(self, input_ids, cache=None):
        h = self.embed_w[input_ids]
        layer_cache = cache[0] if cache is not None else None
        h = self.attn(h, cache=layer_cache)
        return h @ self.lm_head_w


class TestPrefillLastLogitCausal:
    """Validates that _prefill_last_logit's two-pass split produces the same
    last-position logit as a single-pass forward on a model where attention
    actually consults the KV cache."""

    def test_two_pass_matches_naive_with_causal_attention(self):
        from mlx_lm.models.cache import KVCache

        from olmlx.engine.speculative import _prefill_last_logit

        model = _CausalMockModel(vocab_size=32, hidden_size=16)
        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7]])

        naive_cache = [KVCache()]
        naive = model(prompt, cache=naive_cache)[0, -1, :]
        mx.eval(naive)

        two_pass_cache = [KVCache()]
        result = _prefill_last_logit(model, prompt, two_pass_cache)
        mx.eval(result)

        assert mx.allclose(result, naive, atol=1e-4)
