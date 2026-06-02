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

    def test_generate_step_long_prompt_two_pass(self, shared_decoder):
        """generate_step() with a long prompt exercises the two-pass split
        (temporary KV cache + _prefill_last_logit) added to avoid the
        [batch, seq_len+lambda, vocab] materialisation OOM."""
        prompt = mx.array([list(range(20))])
        accepted, num_draft = shared_decoder.generate_step(prompt)

        assert len(accepted) >= 1
        assert num_draft == 3
        assert all(0 <= t < 32 for t in accepted)

    def test_draft_generate_n_zero_returns_empty(self, decoder):
        """_draft_generate(prompt, n=0) must return [] (no tokens)."""
        prompt = mx.array([[1, 2, 3]])
        assert decoder._draft_generate(prompt, n=0) == []


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

    def test_multi_token_smoke(self, model):
        """Smoke test: ``_prefill_last_logit`` runs end-to-end on a multi-token
        prompt and returns a vocab-shaped logit.

        MockModel is cache-agnostic (``MockAttention`` discards the
        ``update_and_fetch`` return), so this only exercises the code path.
        Numerical correctness of the two-pass split is covered by
        ``TestPrefillLastLogitCausal``.
        """
        from mlx_lm.models.cache import make_prompt_cache

        from olmlx.engine.speculative import _prefill_last_logit

        cache_2p = make_prompt_cache(model)
        result = _prefill_last_logit(model, mx.array([[1, 2, 3, 4, 5]]), cache_2p)
        mx.eval(result)

        assert result.shape == (32,)

    def test_cache_offset_equals_prompt_length(self, model):
        """After _prefill_last_logit, KV cache offset must equal prompt length.

        ``offset`` is a Python int incremented by ``update_and_fetch``
        regardless of lazy evaluation, so reading it directly is enough —
        no explicit ``mx.eval`` to mask whether the two passes ran.
        """
        from mlx_lm.models.cache import make_prompt_cache

        from olmlx.engine.speculative import _prefill_last_logit

        prompt = mx.array([[1, 2, 3, 4, 5]])
        cache = make_prompt_cache(model)
        _prefill_last_logit(model, prompt, cache)

        assert cache[0].offset == prompt.shape[1]


class _RecordingModel:
    """Wraps a model and records the seq_len of every ``__call__``.

    Used to assert that prefill bounds each forward pass to at most
    ``_PREFILL_CHUNK`` tokens, the way mlx-lm's native prefill loop does —
    a single forward over a very long prompt OOMs Metal (the
    attention-score intermediate for a ~38k-token prefill exceeds the
    ~41 GB single-buffer limit).
    """

    def __init__(self, inner):
        self._inner = inner
        self.layers = inner.layers
        self.calls: list[int] = []

    def __call__(self, input_ids, cache=None):
        self.calls.append(int(input_ids.shape[1]))
        return self._inner(input_ids, cache=cache)

    def __getattr__(self, name):
        # Delegate anything not set on the wrapper (e.g. named_modules, used
        # by find_gdn_class during SpeculativeDecoder construction) to inner.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class TestChunkedPrefill:
    """Prefill must sub-chunk long prompts so no single ``model()`` forward
    materialises an activation graph large enough to OOM Metal."""

    def test_prefix_forward_bounded_to_chunk_size(self):
        from mlx_lm.models.cache import make_prompt_cache

        from olmlx.engine.speculative import _PREFILL_CHUNK, _prefill_last_logit

        inner = MockModel(32, 16)
        model = _RecordingModel(inner)
        cache = make_prompt_cache(inner)
        n = _PREFILL_CHUNK * 2 + 10
        prompt = mx.zeros((1, n), dtype=mx.int32)

        result = _prefill_last_logit(model, prompt, cache)
        mx.eval(result)

        # No single forward may exceed the chunk size (pass 2 is 1 token).
        assert max(model.calls) <= _PREFILL_CHUNK
        # Cache must still be fully populated to the prompt length.
        assert cache[0].offset == n

    def test_chunked_prefill_matches_naive_causal(self, monkeypatch):
        """Sub-chunking must not change the last-position logit: standard
        KV-cached attention is mathematically chunking-invariant."""
        from mlx_lm.models.cache import KVCache

        from olmlx.engine import speculative
        from olmlx.engine.speculative import _prefill_last_logit

        # Force several sub-chunks over a short prompt.
        monkeypatch.setattr(speculative, "_PREFILL_CHUNK", 2)

        model = _CausalMockModel(vocab_size=32, hidden_size=16)
        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7]])

        naive = model(prompt, cache=[KVCache()])[0, -1, :]
        mx.eval(naive)

        result = _prefill_last_logit(model, prompt, [KVCache()])
        mx.eval(result)

        assert mx.allclose(result, naive, atol=1e-4)

    def test_draft_prefill_chunks_long_prompt(self):
        """SpeculativeDecoder.prefill must also chunk the draft forward."""
        from olmlx.engine.speculative import _PREFILL_CHUNK, SpeculativeDecoder

        draft = _RecordingModel(MockModel(32, 16))
        target = _RecordingModel(MockModel(32, 16))
        decoder = SpeculativeDecoder(
            draft_model=draft, target_model=target, num_speculative_tokens=2
        )
        n = _PREFILL_CHUNK * 2 + 5
        decoder.prefill(mx.zeros((1, n), dtype=mx.int32))

        assert max(draft.calls) <= _PREFILL_CHUNK
        assert max(target.calls) <= _PREFILL_CHUNK


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
        # Separate PRNG keys so embed_w and lm_head_w are independent.
        # Reusing one key would tie them (MLX PRNGs are deterministic per key).
        rng_embed = mx.random.key(1)
        rng_head = mx.random.key(2)
        self.embed_w = mx.random.normal((vocab_size, hidden_size), key=rng_embed) * 0.1
        self.attn = _CausalAttention(hidden_size)
        self.lm_head_w = mx.random.normal((hidden_size, vocab_size), key=rng_head) * 0.1
        # KV cache protocol expects model.layers iterable for make_prompt_cache
        self.layers = [self.attn]

    def __call__(self, input_ids, cache=None):
        h = self.embed_w[input_ids]
        layer_cache = cache[0] if cache is not None else None
        h = self.attn(h, cache=layer_cache)
        return h @ self.lm_head_w


class TestEvalCacheCacheTypes:
    """Coverage for the cache-type dispatch in _eval_cache. The function must
    surface arrays from each supported cache shape (KVCache via .keys/.values,
    ArraysCache-style via .state, TurboQuantKVCache via _key_dequant /
    _value_dequant) so pass-1 state is forced before pass-2 runs."""

    def test_arrays_cache_state_branch(self):
        """ArraysCache-style cache (no .keys/.values; .state returns a list of
        arrays) must reach the .state branch and be eval'd."""

        from olmlx.engine.speculative import _eval_cache

        class _ArraysCacheStub:
            """Minimal stand-in for mlx-lm's ArraysCache: no .keys/.values,
            exposes .state as a list of mx.arrays. Mirrors what hybrid
            linear-attention layers (e.g. Qwen3.5 GatedDeltaNet) use."""

            def __init__(self, arrs):
                self._arrs = arrs

            @property
            def state(self):
                return self._arrs

        a = mx.zeros((4, 4)) + 1.0
        b = mx.zeros((4, 4)) + 2.0
        cache = [_ArraysCacheStub([a, b])]
        # Should not raise and should not log an error (arrays were found).
        _eval_cache(cache)
        # Sanity: arrays are still valid and have their expected values.
        assert mx.allclose(a, mx.ones((4, 4)))
        assert mx.allclose(b, mx.full((4, 4), 2.0))

    def test_turboquant_dequant_buffer_probe(self):
        """TurboQuantKVCache exposes _key_dequant / _value_dequant rather
        than .keys/.values. _eval_cache must probe those attributes so the
        dequant chain is forced as a separate graph (otherwise pass-2 would
        fuse pass-1's dequant into one Metal command buffer)."""

        from olmlx.engine.speculative import _eval_cache

        class _TurboQuantStub:
            """No .keys/.values/.state, but exposes dequant side buffers."""

            def __init__(self, kd, vd):
                self._key_dequant = kd
                self._value_dequant = vd

        kd = mx.zeros((2, 4, 4)) + 0.5
        vd = mx.zeros((2, 4, 4)) + 0.25
        # Should reach the dequant-probe branch and NOT log an error.
        _eval_cache([_TurboQuantStub(kd, vd)])
        # Sanity: values are still those we set.
        assert mx.allclose(kd, mx.full((2, 4, 4), 0.5))
        assert mx.allclose(vd, mx.full((2, 4, 4), 0.25))

    def test_real_spectralquant_cache_does_not_log_error(self, caplog):
        """Regression: a real ``SpectralQuantKVCache`` must hit the ``.state``
        probe branch. If ``.state`` is removed or restructured on the cache
        class, the helper would silently fall through to the unrecognised
        cache error path and the OOM protection would degrade to a no-op."""
        import logging

        import numpy as np

        from olmlx.engine.spectralquant import (
            SpectralRotation,
            fit_codebook,
        )
        from olmlx.engine.spectralquant_cache import SpectralQuantKVCache
        from olmlx.engine.speculative import _eval_cache

        head_dim = 8
        d_eff = 4
        bits_high = 4
        bits_low = 2
        rng = np.random.RandomState(42)
        q, _ = np.linalg.qr(rng.randn(head_dim, head_dim).astype(np.float32))
        rot_k = SpectralRotation(mx.array(q))
        q2, _ = np.linalg.qr(rng.randn(head_dim, head_dim).astype(np.float32))
        rot_v = SpectralRotation(mx.array(q2))
        data = mx.random.normal((500, head_dim))
        norms = mx.linalg.norm(data, axis=-1, keepdims=True)
        data_n = data / mx.maximum(norms, mx.array(1e-8))
        rotated = rot_k.rotate(data_n)
        cb_sem = fit_codebook(rotated[..., :d_eff].reshape(-1), bits=bits_high)
        cb_tail = fit_codebook(rotated[..., d_eff:].reshape(-1), bits=bits_low)
        cache = SpectralQuantKVCache(
            rotation_key=rot_k,
            rotation_value=rot_v,
            codebook_sem_key=cb_sem,
            codebook_tail_key=cb_tail,
            codebook_sem_value=cb_sem,
            codebook_tail_value=cb_tail,
            d_eff=d_eff,
            bits_high=bits_high,
            bits_low=bits_low,
        )
        k = mx.random.normal((1, 2, 4, head_dim))
        v = mx.random.normal((1, 2, 4, head_dim))
        cache.update_and_fetch(k, v)

        with caplog.at_level(logging.ERROR, logger="olmlx.engine.speculative"):
            _eval_cache([cache])
        assert not any(
            "no mx.array entries found" in r.message for r in caplog.records
        ), (
            "_eval_cache hit the unrecognised-cache branch for a real "
            "SpectralQuantKVCache. Check that .state still returns the "
            "packed-storage arrays."
        )

    def test_real_turboquant_cache_does_not_log_error(self, caplog):
        """Regression: a real ``TurboQuantKVCache`` must hit a known probe
        branch in ``_eval_cache``. If anyone renames ``_key_dequant`` /
        ``_value_dequant`` on the cache class, the helper would silently
        fall through to the unrecognised-cache error path and the OOM
        protection would degrade to a no-op without test failure. This
        test pins the contract by instantiating the real cache."""
        import logging

        from olmlx.engine.speculative import _eval_cache
        from olmlx.engine.turboquant import TurboQuantRotation
        from olmlx.engine.turboquant_cache import TurboQuantKVCache

        rk = TurboQuantRotation(head_dim=64, seed=0)
        rv = TurboQuantRotation(head_dim=64, seed=1)
        cache = TurboQuantKVCache(bits=4, rotation_key=rk, rotation_value=rv)
        k = mx.random.normal((1, 8, 4, 64))
        v = mx.random.normal((1, 8, 4, 64))
        cache.update_and_fetch(k, v)

        with caplog.at_level(logging.ERROR, logger="olmlx.engine.speculative"):
            _eval_cache([cache])
        assert not any(
            "no mx.array entries found" in r.message for r in caplog.records
        ), (
            "_eval_cache hit the unrecognised-cache branch for a real "
            "TurboQuantKVCache. Check that _key_dequant / _value_dequant "
            "(or any replacement attributes) are probed in _eval_cache."
        )

    def test_unrecognised_cache_logs_error(self, caplog):
        """A cache with no probed arrays must log at ERROR level so the
        OOM-avoidance no-op surfaces on first encounter."""
        import logging

        from olmlx.engine.speculative import _eval_cache

        class _OpaqueCache:
            pass

        with caplog.at_level(logging.ERROR, logger="olmlx.engine.speculative"):
            _eval_cache([_OpaqueCache()])
        assert any("no mx.array entries found" in r.message for r in caplog.records)


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


class _Qwen3_5StyleModel(_CausalMockModel):
    """Mock model that mimics mlx-vlm 0.4.4 Qwen3_5/Qwen2-VL position-id
    caching. Pass 1 (cache_offset==0, _rope_deltas is None) computes positions
    from scratch and stores _position_ids / _rope_deltas on the instance.
    Subsequent calls with cache_offset > 0 take the cache-extension code path
    that builds positions on the fly from cache_offset + _rope_deltas, without
    consulting (or updating) _position_ids. Resetting _rope_deltas between
    those calls would force the model back onto the from-scratch code path
    with the wrong starting position.
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__(vocab_size, hidden_size)
        self._position_ids: mx.array | None = None
        self._rope_deltas: mx.array | None = None

    def __call__(self, input_ids, cache=None):
        cache_offset = (
            cache[0].offset if cache is not None and cache[0] is not None else 0
        )
        if cache_offset == 0 or self._rope_deltas is None or cache is None:
            # "Inner" branch (Qwen3_5 language.py:619): compute fresh, cache.
            seq_length = input_ids.shape[1]
            self._position_ids = mx.arange(seq_length).reshape(1, -1)
            self._rope_deltas = mx.array(0)
        else:
            # "Else" branch (Qwen3_5 language.py:631): cache-extension. Does
            # NOT update _position_ids or _rope_deltas — they persist from
            # the first call. This is the path pass-2 and the draft forward
            # take.
            pass
        return super().__call__(input_ids, cache=cache)


class TestVLMStateAcrossPasses:
    """Regression: the two-pass split must not require resetting cached
    VLM position state between passes, because Qwen3_5/Qwen2-VL only update
    that state on the cache_offset==0 call and use it as a delta thereafter.
    Resetting between passes forces the model onto the wrong code path."""

    def test_two_pass_does_not_overwrite_position_state_at_pass2(self):
        from mlx_lm.models.cache import KVCache

        from olmlx.engine.speculative import _prefill_last_logit

        model = _Qwen3_5StyleModel(vocab_size=32, hidden_size=16)
        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7]])

        # Naive single-pass: pass 1 sets _position_ids with prompt length.
        naive_cache = [KVCache()]
        naive = model(prompt, cache=naive_cache)[0, -1, :]
        mx.eval(naive)

        # Reset model and run the two-pass split: pass 1 is on prompt[:, :-1]
        # (N-1 tokens, cache_offset=0), pass 2 is on prompt[:, -1:] (1 token,
        # cache_offset=N-1). Pass 2 must take the cache-extension else branch.
        model._position_ids = None
        model._rope_deltas = None
        two_pass_cache = [KVCache()]
        result = _prefill_last_logit(model, prompt, two_pass_cache)
        mx.eval(result)

        # Output matches naive (numerical correctness).
        assert mx.allclose(result, naive, atol=1e-4)
        # _position_ids retains its pass-1 shape (N-1 = 6), confirming pass 2
        # did NOT enter the inner branch (which would have overwritten it
        # with a 1-element tensor). A reset between passes — as suggested by
        # reviewers — would force pass 2 into the inner branch and break this.
        assert model._position_ids is not None
        assert model._position_ids.shape == (1, prompt.shape[1] - 1)
