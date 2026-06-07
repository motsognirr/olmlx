"""Regression coverage for the LayerSkip self-speculative decoder
(``olmlx.engine.self_speculative.decoder``).

The decoder uses the target model's own early layers as an
autoregressive draft, then verifies with the full model. These tests
build stub MLX models (no real weights, no GPU dependence beyond MLX's
CPU eval) that satisfy the module's path-probing for ``embed_tokens`` /
``norm`` / ``lm_head`` and the trimmable-KVCache contract, so we can
drive prefill/step/reset/stats bookkeeping and the constructor guards
without a real model load.

The single ``step()``/full-generation paths that require a *real* model
(numerically meaningful acceptance behaviour on production weights) are
covered structurally here; correctness on real weights is out of scope
for a hermetic unit test and noted in the summary.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

import olmlx.engine.self_speculative.decoder as ssd
from olmlx.engine.self_speculative.decoder import (
    SelfSpeculativeDecoder,
    _eval_cache,
    _find_module,
    _logits,
)

# Reuse the cache-backed mock layer from the flash-speculative suite so
# the trimmable-KVCache probe in the constructor passes.
from tests.test_flash_speculative import MockLayer


@pytest.fixture(autouse=True)
def _seed_mlx_rng():
    """Pin MLX's global RNG before every test.

    The stub models below initialise their weights randomly, so a few
    tests (notably the rejection-trim path, which needs the draft to
    *diverge* from the forced verify token) depend on the RNG state.
    Without this, the outcome leaks across tests: an earlier test in the
    full suite can leave the RNG in a state where the draft coincidentally
    agrees, flipping a rejection into full acceptance. Seeding here makes
    the file deterministic and order-independent.
    """
    mx.random.seed(0)


class _SelfSpecModel(nn.Module):
    """Stub target whose attribute layout matches the decoder's path
    probing: top-level ``embed_tokens`` / ``norm`` / ``lm_head`` and a
    ``layers`` list (found by ``get_model_layers`` via ``.layers``).

    ``MockLayer`` writes the input straight into a trimmable ``KVCache``
    so ``make_prompt_cache`` yields one trimmable cache per layer and the
    constructor's non-trimmable guard does not fire.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, cache=None):
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, cache=cache[i] if cache is not None else None)
        return self.lm_head(self.norm(h))


class _TiedEmbedModel(nn.Module):
    """Stub with no ``lm_head`` but an ``embed_tokens`` that exposes
    ``as_linear`` — exercises the tied-embeddings fallback branch."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 3):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)

    def __call__(self, input_ids: mx.array, cache=None):
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, cache=cache[i] if cache is not None else None)
        # Tied head: project hidden back through the embedding matrix.
        return self.embed_tokens.as_linear(self.norm(h))


@pytest.fixture()
def model():
    return _SelfSpecModel(vocab_size=16, hidden_size=8, num_layers=4)


@pytest.fixture()
def decoder(model):
    return SelfSpeculativeDecoder(
        target_model=model,
        num_early_layers=2,
        num_speculative_tokens=3,
    )


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


class TestHelpers:
    def test_find_module_first_path_wins(self):
        class Holder:
            pass

        m = Holder()
        m.embed_tokens = "top"
        m.model = Holder()
        m.model.embed_tokens = "nested"
        paths = (("embed_tokens",), ("model", "embed_tokens"))
        assert _find_module(m, paths) == "top"

    def test_find_module_falls_through_to_nested(self):
        class Holder:
            pass

        m = Holder()
        m.model = Holder()
        m.model.embed_tokens = "nested"
        paths = (("embed_tokens",), ("model", "embed_tokens"))
        assert _find_module(m, paths) == "nested"

    def test_find_module_returns_none_when_absent(self):
        class Holder:
            pass

        assert _find_module(Holder(), (("missing",), ("a", "b"))) is None

    def test_logits_prefers_logits_attr(self):
        class Out:
            logits = mx.array([1.0, 2.0])

        assert mx.allclose(_logits(Out()), mx.array([1.0, 2.0]))

    def test_logits_passthrough_when_no_attr(self):
        arr = mx.array([3.0, 4.0])
        assert _logits(arr) is arr


class TestEvalCache:
    """Branch coverage for the cache-materialisation helper in this module
    (a near-copy of speculative._eval_cache). Each supported cache shape
    must surface its arrays so pass-1 KV state is forced before pass-2."""

    def test_keys_values_as_arrays(self):
        class C:
            keys = mx.zeros((2, 2)) + 1.0
            values = mx.zeros((2, 2)) + 2.0

        # Should not raise; arrays remain valid after eval.
        _eval_cache([C()])
        assert mx.allclose(C.keys, mx.ones((2, 2)))

    def test_keys_values_as_lists(self):
        class C:
            keys = [mx.zeros((2, 2)) + 1.0, "not-an-array"]
            values = (mx.zeros((2, 2)) + 2.0,)

        _eval_cache([C()])  # exercises the list/tuple extend branches

    def test_state_branch_when_no_keys_values(self):
        class C:
            keys = None
            values = None
            state = [mx.zeros((2, 2)) + 3.0, None]

        _eval_cache([C()])  # exercises the .state branch

    def test_dequant_side_buffers(self):
        class C:
            keys = None
            values = None
            state = None
            _key_dequant = mx.zeros((2, 2)) + 0.5
            _value_dequant = mx.zeros((2, 2)) + 0.25

        _eval_cache([C()])  # exercises the dequant-probe branch

    def test_empty_cache_is_noop(self):
        _eval_cache([])  # arrs stays empty; mx.eval not called


# ----------------------------------------------------------------------
# Constructor / guards
# ----------------------------------------------------------------------


class TestConstructor:
    def test_init_records_layer_counts_and_modules(self, decoder, model):
        assert decoder._total_layers == 4
        assert decoder._N == 2
        assert decoder._lambda == 3
        assert decoder._embed is model.embed_tokens
        assert decoder._norm is model.norm
        assert decoder._lm_head is model.lm_head

    def test_init_rejects_zero_speculative_tokens(self, model):
        with pytest.raises(ValueError, match="num_speculative_tokens"):
            SelfSpeculativeDecoder(model, num_early_layers=2, num_speculative_tokens=0)

    def test_init_rejects_early_layers_below_one(self, model):
        with pytest.raises(ValueError, match="num_early_layers"):
            SelfSpeculativeDecoder(model, num_early_layers=0)

    def test_init_rejects_early_layers_at_or_above_total(self, model):
        # num_early_layers == total_layers is out of [1, total-1].
        with pytest.raises(ValueError, match="num_early_layers"):
            SelfSpeculativeDecoder(model, num_early_layers=4)

    def test_init_tied_embedding_fallback_for_lm_head(self):
        m = _TiedEmbedModel(vocab_size=16, hidden_size=8, num_layers=3)
        dec = SelfSpeculativeDecoder(m, num_early_layers=1, num_speculative_tokens=2)
        # Fallback assigned the embedding's as_linear callable as lm_head.
        # (as_linear is a bound method, so a fresh object per access — assert
        # behaviour, not identity.)
        assert callable(dec._lm_head)
        h = mx.zeros((1, 1, 8))
        assert dec._lm_head(h).shape == (1, 1, 16)
        # And the decoder drives end-to-end on a tied-head model.
        dec.prefill(mx.array([[1, 2, 3]]))
        accepted, _ = dec.step()
        assert len(accepted) >= 1

    def test_init_missing_embed_raises(self):
        class NoEmbed(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [MockLayer(8) for _ in range(3)]
                self.norm = nn.RMSNorm(8)
                self.lm_head = nn.Linear(8, 16, bias=False)

        with pytest.raises(ValueError, match="embed_tokens"):
            SelfSpeculativeDecoder(NoEmbed(), num_early_layers=1)

    def test_init_missing_norm_raises(self):
        class NoNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(16, 8)
                self.layers = [MockLayer(8) for _ in range(3)]
                self.lm_head = nn.Linear(8, 16, bias=False)

        with pytest.raises(ValueError, match="norm"):
            SelfSpeculativeDecoder(NoNorm(), num_early_layers=1)

    def test_init_rejects_gdn_hybrid_target(self, model, monkeypatch):
        # If find_gdn_class reports a GDN layer, the constructor must
        # refuse the model (ArraysCache RNN-state can't be rolled back).
        class _FakeGDN:
            __module__ = "fake.mod"
            __name__ = "GatedDeltaNet"

        monkeypatch.setattr(ssd, "find_gdn_class", lambda m: _FakeGDN)
        with pytest.raises(NotImplementedError, match="linear-attention"):
            SelfSpeculativeDecoder(model, num_early_layers=2)

    def test_init_rejects_non_trimmable_early_cache(self, model, monkeypatch):
        # A cache whose early entries lack a callable trim() must be
        # rejected (RotatingKVCache / ChunkedKVCache in issue #343).
        class _NoTrim:
            trim = None  # not callable

        monkeypatch.setattr(
            ssd, "make_prompt_cache", lambda m: [_NoTrim() for _ in range(4)]
        )
        with pytest.raises(NotImplementedError, match="non-trimmable"):
            SelfSpeculativeDecoder(model, num_early_layers=2)

    def test_init_rejects_when_mlx_cache_helpers_unavailable(self, model, monkeypatch):
        # The module captures make_prompt_cache/trim_prompt_cache at import;
        # if either is None, construction must raise RuntimeError.
        monkeypatch.setattr(ssd, "make_prompt_cache", None)
        with pytest.raises(RuntimeError, match="mlx_lm.models.cache"):
            SelfSpeculativeDecoder(model, num_early_layers=2)

    def test_init_missing_lm_head_without_tied_raises(self):
        # embed_tokens without as_linear and no lm_head -> ValueError.
        class PlainEmbed:
            pass

        class NoHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [MockLayer(8) for _ in range(3)]
                self.norm = nn.RMSNorm(8)

            # embed_tokens is a plain object lacking as_linear.
            embed_tokens = PlainEmbed()

        with pytest.raises(ValueError, match="lm_head"):
            SelfSpeculativeDecoder(NoHead(), num_early_layers=1)


# ----------------------------------------------------------------------
# Lifecycle: prefill / reset / stats
# ----------------------------------------------------------------------


class TestLifecycle:
    def test_prefill_returns_int_token_and_populates_state(self, decoder):
        prompt = mx.array([[1, 2, 3, 4, 5]])
        tok = decoder.prefill(prompt)
        assert isinstance(tok, int)
        assert 0 <= tok < 16
        assert decoder._cache is not None
        # One cache entry per layer.
        assert len(decoder._cache) == decoder._total_layers
        assert decoder._prompt_len == 5
        assert decoder._pending_token == tok
        assert decoder._last_logit is not None

    def test_prefill_resets_vlm_rope_state(self):
        # VLM targets (mlx-vlm 0.4.4) cache _position_ids / _rope_deltas;
        # prefill must clear them so a fresh request starts from scratch.
        m = _SelfSpecModel(vocab_size=16, hidden_size=8, num_layers=4)
        m._position_ids = mx.array([[1, 2, 3]])
        m._rope_deltas = mx.array(5)
        dec = SelfSpeculativeDecoder(m, num_early_layers=2)
        dec.prefill(mx.array([[1, 2, 3]]))
        assert m._position_ids is None
        assert m._rope_deltas is None

    def test_prefill_single_token_prompt(self, decoder):
        # Exercises the prompt.shape[1] <= 1 branch in _prefill_last_logit.
        tok = decoder.prefill(mx.array([[7]]))
        assert 0 <= tok < 16
        assert decoder._prompt_len == 1

    def test_prefill_resets_prior_stats(self, decoder):
        decoder.prefill(mx.array([[1, 2, 3]]))
        decoder.step()
        assert decoder._stats_steps == 1
        # A fresh prefill clears accumulated stats.
        decoder.prefill(mx.array([[4, 5, 6]]))
        assert decoder._stats_steps == 0
        assert decoder._stats_proposed == 0
        assert decoder._stats_accepted_draft == 0

    def test_reset_clears_all_state(self, decoder):
        decoder.prefill(mx.array([[1, 2, 3]]))
        decoder.step()
        decoder.reset()
        assert decoder._cache is None
        assert decoder._last_logit is None
        assert decoder._pending_token is None
        assert decoder._prompt_len == 0
        assert decoder._stats_steps == 0
        assert decoder._stats_proposed == 0
        assert decoder._stats_accepted_draft == 0

    def test_close_is_noop(self, decoder):
        # close() must not raise and must not disturb state.
        decoder.prefill(mx.array([[1, 2, 3]]))
        decoder.close()
        assert decoder._cache is not None

    def test_step_before_prefill_raises(self, decoder):
        with pytest.raises(AssertionError):
            decoder.step()

    def test_stats_summary_zero_before_steps(self, decoder):
        summary = decoder.stats_summary()
        assert summary["steps"] == 0
        assert summary["proposed"] == 0
        assert summary["accepted_draft"] == 0
        # Division-by-zero guards return 0.0, not NaN/raise.
        assert summary["acceptance_rate"] == 0.0
        assert summary["avg_tokens_per_step"] == 0.0
        assert summary["lambda"] == 3


# ----------------------------------------------------------------------
# step(): bookkeeping and cache-length invariant
# ----------------------------------------------------------------------


class TestStep:
    def test_step_returns_accepted_and_lambda(self, decoder):
        decoder.prefill(mx.array([[1, 2, 3]]))
        accepted, num_draft = decoder.step()
        assert num_draft == 3  # == lambda
        assert len(accepted) >= 1  # greedy accept guarantees at least 1
        assert all(isinstance(t, int) for t in accepted)

    def test_cache_length_invariant_after_step(self, decoder):
        # After each step the cache holds exactly _prompt_len entries on
        # every layer (prompt + accepted tokens; pending excluded).
        decoder.prefill(mx.array([[1, 2, 3, 4]]))
        accepted, _ = decoder.step()
        for c in decoder._cache:
            assert c.offset == decoder._prompt_len

    def test_prompt_len_grows_by_num_accepted(self, decoder):
        decoder.prefill(mx.array([[1, 2, 3]]))
        before = decoder._prompt_len
        accepted, _ = decoder.step()
        assert decoder._prompt_len == before + len(accepted)

    def test_stats_accumulate_across_steps(self, decoder):
        decoder.prefill(mx.array([[1, 2, 3]]))
        decoder.step()
        decoder.step()
        assert decoder._stats_steps == 2
        # proposed grows by lambda each step.
        assert decoder._stats_proposed == 2 * decoder._lambda
        summary = decoder.stats_summary()
        assert summary["steps"] == 2
        assert summary["avg_tokens_per_step"] >= 1.0

    def test_acceptance_ema_stays_in_unit_range(self, decoder):
        decoder.prefill(mx.array([[1, 2, 3]]))
        for _ in range(4):
            decoder.step()
            assert 0.0 <= decoder._alpha <= 1.0

    def test_multi_step_keeps_cache_invariant(self, decoder):
        decoder.prefill(mx.array([[1, 2, 3, 4, 5]]))
        for _ in range(5):
            accepted, _ = decoder.step()
            assert len(accepted) >= 1
            for c in decoder._cache:
                assert c.offset == decoder._prompt_len

    def test_pending_token_in_range_after_step(self, decoder):
        decoder.prefill(mx.array([[1, 2, 3]]))
        decoder.step()
        assert isinstance(decoder._pending_token, int)
        assert 0 <= decoder._pending_token < 16

    def test_rejection_trims_cache_and_keeps_invariant(self):
        # Force the full-model verify to disagree with the early-layer
        # draft so >=1 draft token is rejected and the rejection-trim
        # path (trim_count > 0) runs. The full forward always argmaxes to
        # a fixed token; the draft loop uses the (untied) lm_head, which
        # picks something else, guaranteeing a position-0 mismatch.
        class _DivergeModel(_SelfSpecModel):
            def __call__(self, input_ids, cache=None):
                out = super().__call__(input_ids, cache=cache)
                B, T, V = out.shape
                forced = mx.zeros((B, T, V))
                # Pin every verify position to token 0.
                forced = forced.at[:, :, 0].add(1000.0)
                return forced

        m = _DivergeModel(vocab_size=16, hidden_size=8, num_layers=4)
        dec = SelfSpeculativeDecoder(m, num_early_layers=2, num_speculative_tokens=3)
        dec.prefill(mx.array([[1, 2, 3]]))
        accepted, num_draft = dec.step()
        assert num_draft == 3
        # Verify always picks token 0; the accepted list ends with it.
        assert accepted[-1] == 0
        # Cache invariant still holds after a rejection-driven trim.
        for c in dec._cache:
            assert c.offset == dec._prompt_len
        # A rejection means fewer than lambda+1 tokens were accepted.
        assert len(accepted) <= num_draft

    def test_lambda_one_step(self, model):
        # Minimal lambda exercises the single-draft loop / trim of 1.
        dec = SelfSpeculativeDecoder(
            model, num_early_layers=2, num_speculative_tokens=1
        )
        dec.prefill(mx.array([[1, 2, 3]]))
        accepted, num_draft = dec.step()
        assert num_draft == 1
        assert 1 <= len(accepted) <= 2
        for c in dec._cache:
            assert c.offset == dec._prompt_len
