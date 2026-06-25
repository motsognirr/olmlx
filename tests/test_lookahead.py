"""Tests for lookahead (Jacobi) decoding (engine/lookahead_decoder.py, #502).

Uses the CPU-friendly MockModel (a real nn.Module with KV-cache support, shared
with the PLD tests) so the full prefill/step/trim loop runs without a GPU or a
model download. The headline test is exactness: lookahead output must equal
plain greedy decoding token-for-token, which is the strategy's core guarantee.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from tests.test_flash_speculative import MockModel


class TestLookaheadConstruction:
    def test_rejects_zero_window(self):
        from olmlx.engine.lookahead_decoder import LookaheadDecoder

        with pytest.raises(ValueError, match="num_speculative_tokens must be >= 1"):
            LookaheadDecoder(target_model=MockModel(32, 16), num_speculative_tokens=0)

    def test_no_gdn_capture_for_plain_model(self):
        from olmlx.engine.lookahead_decoder import LookaheadDecoder

        dec = LookaheadDecoder(target_model=MockModel(32, 16))
        # MockModel is not a hybrid GatedDeltaNet target.
        assert dec._gdn_capture is None
        assert dec._target_gdn_buffer is None


class TestLookaheadDecoder:
    @pytest.fixture()
    def decoder(self):
        from olmlx.engine.lookahead_decoder import LookaheadDecoder

        return LookaheadDecoder(
            target_model=MockModel(32, 16), num_speculative_tokens=4
        )

    def test_prefill_populates_state(self, decoder):
        prompt = mx.array([[1, 2, 3, 4, 5]])
        first_token = decoder.prefill(prompt)
        assert decoder._target_cache is not None
        assert decoder._cache_seq_len == 5
        assert isinstance(first_token, int)
        assert 0 <= first_token < 32
        assert decoder._pending_token == first_token
        # Jacobi window seeded from the last `window` prompt tokens.
        assert decoder._guess == [2, 3, 4, 5]

    def test_prefill_seed_shorter_than_window(self, decoder):
        first = decoder.prefill(mx.array([[7, 9]]))
        assert decoder._guess == [7, 9]
        assert isinstance(first, int)

    def test_step_returns_at_least_one_token(self, decoder):
        decoder.prefill(mx.array([[1, 2, 3, 4, 5]]))
        accepted, num_drafted = decoder.step()
        assert len(accepted) >= 1
        assert 0 <= num_drafted <= decoder._window
        assert len(accepted) <= num_drafted + 1

    def test_step_advances_cache(self, decoder):
        decoder.prefill(mx.array([[1, 2, 3, 4, 5]]))
        accepted, _ = decoder.step()
        # Cache grows by exactly num_accepted positions and stays aligned.
        assert decoder._cache_seq_len == 5 + len(accepted)
        assert decoder._target_cache[0].offset == decoder._cache_seq_len

    def test_reset_clears_state(self, decoder):
        decoder.prefill(mx.array([[1, 2, 3]]))
        decoder.step()
        decoder.reset()
        assert decoder._target_cache is None
        assert decoder._cache_seq_len == 0
        assert decoder._pending_token is None
        assert decoder._guess == []
        assert decoder._pool == {}

    def test_multi_step_cache_alignment(self, decoder):
        prompt_ids = [1, 2, 3, 4, 5]
        decoder.prefill(mx.array([prompt_ids]))
        emitted = 1  # the prefill bonus token
        for _ in range(6):
            accepted, _ = decoder.step()
            assert len(accepted) >= 1
            emitted += len(accepted)
            # KV cache stays in lockstep with emitted-minus-pending.
            assert decoder._target_cache[0].offset == decoder._cache_seq_len
            assert decoder._cache_seq_len == len(prompt_ids) + emitted - 1

    def test_stats_summary_shape(self, decoder):
        decoder.prefill(mx.array([[1, 2, 3]]))
        decoder.step()
        s = decoder.stats_summary()
        for k in (
            "steps",
            "proposed",
            "accepted_draft",
            "acceptance_rate",
            "avg_tokens_per_step",
            "window",
        ):
            assert k in s, f"missing stats key: {k}"
        assert s["steps"] == 1
        assert 0 <= s["acceptance_rate"] <= 1
        assert s["window"] == decoder._window

    def test_protocol_conformance(self, decoder):
        assert callable(getattr(decoder, "prefill", None))
        assert callable(getattr(decoder, "step", None))
        assert callable(getattr(decoder, "reset", None))


class TestLookaheadExactness:
    """The core guarantee: lookahead emits exactly the greedy sequence."""

    def test_output_matches_greedy_reference(self):
        from olmlx.engine.lookahead_decoder import LookaheadDecoder
        from olmlx.engine.speculative import _logits, _prefill_last_logit

        target = MockModel(vocab_size=32, hidden_size=16)
        # Repetition gives the Jacobi guess something to converge on, exercising
        # the accept (not just no-match) branches.
        prompt = mx.array([[3, 7, 1, 5, 9, 2, 4, 6, 8, 3, 7, 1]])
        max_steps = 16

        # Greedy reference on the same model instance.
        from mlx_lm.models.cache import make_prompt_cache

        ref_cache = make_prompt_cache(target)
        first_logit = _prefill_last_logit(target, prompt, ref_cache)
        mx.eval(first_logit)
        ref_tokens = [int(mx.argmax(first_logit).item())]
        for _ in range(max_steps - 1):
            inp = mx.array([[ref_tokens[-1]]])
            out = _logits(target(inp, cache=ref_cache))[0, -1, :]
            mx.eval(out)
            ref_tokens.append(int(mx.argmax(out).item()))

        # Lookahead: prefill + repeated step() until we have >= max_steps.
        dec = LookaheadDecoder(target_model=target, num_speculative_tokens=4)
        la_tokens = [dec.prefill(prompt)]
        while len(la_tokens) < max_steps:
            accepted, _ = dec.step()
            la_tokens.extend(accepted)
        la_tokens = la_tokens[:max_steps]

        assert la_tokens == ref_tokens, (
            "lookahead diverged from greedy reference: "
            f"la={la_tokens} greedy={ref_tokens}"
        )

    def test_exactness_with_window_one(self):
        """Window=1 still matches greedy (degenerate single-token guess)."""
        from olmlx.engine.lookahead_decoder import LookaheadDecoder
        from olmlx.engine.speculative import _logits, _prefill_last_logit
        from mlx_lm.models.cache import make_prompt_cache

        target = MockModel(vocab_size=24, hidden_size=12)
        prompt = mx.array([[2, 4, 6, 2, 4, 6]])
        max_steps = 10

        ref_cache = make_prompt_cache(target)
        fl = _prefill_last_logit(target, prompt, ref_cache)
        mx.eval(fl)
        ref = [int(mx.argmax(fl).item())]
        for _ in range(max_steps - 1):
            out = _logits(target(mx.array([[ref[-1]]]), cache=ref_cache))[0, -1, :]
            mx.eval(out)
            ref.append(int(mx.argmax(out).item()))

        dec = LookaheadDecoder(target_model=target, num_speculative_tokens=1)
        got = [dec.prefill(prompt)]
        while len(got) < max_steps:
            accepted, _ = dec.step()
            got.extend(accepted)
        assert got[:max_steps] == ref


class TestLookaheadRegistration:
    def test_registered_strategy(self):
        from olmlx.engine.registry import (
            _FLASH_MOE_INCOMPATIBLE_STRATEGIES,
            _VALID_SPECULATIVE_STRATEGIES,
        )

        assert "lookahead" in _VALID_SPECULATIVE_STRATEGIES
        # Draft-free → composes with Flash-MoE, like PLD.
        assert "lookahead" not in _FLASH_MOE_INCOMPATIBLE_STRATEGIES

    def test_settings_accepts_lookahead(self):
        from olmlx.config import Settings

        s = Settings(speculative_strategy="lookahead", _env_file=None)
        assert s.speculative_strategy == "lookahead"

    def test_metrics_strategy_label(self):
        from olmlx.engine.lookahead_decoder import LookaheadDecoder
        from olmlx.utils.metrics import _STRATEGY_BY_CLASS

        assert _STRATEGY_BY_CLASS[LookaheadDecoder.__name__] == "lookahead"
