"""Regression coverage for olmlx.engine.speculative.

Focus areas (currently-uncovered branches):
- ``verify_draft_greedy``: accept / first-mismatch / full-acceptance bonus.
- ``SpeculativeDecoder`` acceptance-rate EMA + stats bookkeeping + reset().
- ``PromptLookupDecoder`` n-gram lookup (``_lookup_draft``), EMA "skip on
  no-match" rule, validation guards, prefill/step/reset state.

Hermetic: tiny fake mx-array models/stubs, no real model, no GPU, no I/O.
pytest runs with asyncio_mode=auto so async tests need no decorator (none here).
"""

import logging

import mlx.core as mx
import mlx.nn as nn
import pytest

from olmlx.engine.speculative import (
    PromptLookupDecoder,
    SpeculativeDecoder,
    verify_draft_greedy,
)
from tests.test_flash_speculative import MockModel


def _one_hot_logits(tokens: list[int], vocab: int, peak: float = 100.0) -> mx.array:
    """(len(tokens), vocab) logits whose argmax at row i is tokens[i]."""
    rows = []
    for t in tokens:
        row = mx.zeros((vocab,))
        row = row.at[t].add(peak)
        rows.append(row[None, :])
    return mx.concatenate(rows, axis=0)


class _ScriptedTarget(nn.Module):
    """Target stub whose forward returns argmax-fixed logits.

    For an input ``(1, T)`` it returns ``(1, T, vocab)`` logits where the
    argmax of every position is taken from ``self.next_tokens`` in order,
    cycling as needed. This makes ``PromptLookupDecoder.step`` deterministic
    without any real model maths. It is *not* wired to a KV cache content
    check — that is irrelevant for the lookup/EMA logic under test — but it
    still drives ``cache[0].update_and_fetch`` so ``trim_prompt_cache`` works.

    Subclasses ``nn.Module`` so ``find_gdn_class`` (which walks
    ``named_modules()``) sees a non-hybrid model and returns ``None``.
    """

    def __init__(self, vocab: int, next_tokens: list[int]):
        super().__init__()
        self.vocab = vocab
        self.next_tokens = next_tokens
        self._cursor = 0
        # make_prompt_cache walks model.layers; reuse MockModel's cache-aware
        # attention by composing one.
        self._inner = MockModel(vocab, 8, num_layers=1)
        self.layers = self._inner.layers

    def __call__(self, input_ids, cache=None):
        # Drive the KV cache so offsets/trim behave like a real forward.
        if cache is not None:
            self._inner(input_ids, cache=cache)
        T = input_ids.shape[1]
        toks = []
        for _ in range(T):
            toks.append(self.next_tokens[self._cursor % len(self.next_tokens)])
            self._cursor += 1
        return _one_hot_logits(toks, self.vocab)[None, :, :]


# ---------------------------------------------------------------------------
# verify_draft_greedy
# ---------------------------------------------------------------------------


class TestVerifyDraftGreedy:
    def test_all_accepted_appends_bonus(self):
        # target argmax = [5, 6, 7, 9]; draft proposed first three.
        vocab = 16
        target = _one_hot_logits([5, 6, 7, 9], vocab)
        accepted = verify_draft_greedy([5, 6, 7], target)
        # 3 draft tokens accepted + 1 bonus from the (n+1)th logit row.
        assert accepted == [5, 6, 7, 9]

    def test_first_mismatch_returns_target_token_and_stops(self):
        vocab = 16
        # target argmax = [5, 2, 7, 9]; draft says [5, 6, 7] -> mismatch at i=1.
        target = _one_hot_logits([5, 2, 7, 9], vocab)
        accepted = verify_draft_greedy([5, 6, 7], target)
        # Accept first match (5), then replace mismatch with target's 2, stop.
        assert accepted == [5, 2]

    def test_immediate_mismatch_returns_single_token(self):
        vocab = 16
        target = _one_hot_logits([3, 4, 5, 6], vocab)
        accepted = verify_draft_greedy([0, 0, 0], target)
        assert accepted == [3]

    def test_empty_draft_returns_only_bonus(self):
        vocab = 16
        # n == 0: loop body never runs; the single (n+1=1) row is the bonus.
        target = _one_hot_logits([8], vocab)
        accepted = verify_draft_greedy([], target)
        assert accepted == [8]

    def test_returns_at_least_one_token_invariant(self):
        vocab = 8
        target = _one_hot_logits([7, 7, 7, 7], vocab)
        for draft in ([], [0], [0, 1, 2]):
            assert len(verify_draft_greedy(draft, target)) >= 1


# ---------------------------------------------------------------------------
# SpeculativeDecoder acceptance-rate EMA + stats
# ---------------------------------------------------------------------------


def _bare_speculative(lam=4, ema=0.9):
    """A SpeculativeDecoder without running __init__ (no model wiring).

    Mirrors the style of the existing test_verify_rejects_divergent_tokens.
    """
    dec = SpeculativeDecoder.__new__(SpeculativeDecoder)
    dec._lambda = lam
    dec._alpha = 0.5
    dec._alpha_ema = ema
    return dec


class TestAcceptanceRateBookkeeping:
    def test_full_acceptance_pushes_ema_up(self):
        dec = _bare_speculative(lam=4, ema=0.9)
        # num_accepted = lambda + 1 -> all 4 draft accepted -> acceptance 1.0
        n_draft = dec._update_acceptance_rate(5)
        assert n_draft == 4  # min(5-1, 4)
        # alpha = 0.9*0.5 + 0.1*1.0 = 0.55
        assert dec._alpha == pytest.approx(0.55)

    def test_zero_draft_accepted_pushes_ema_down(self):
        dec = _bare_speculative(lam=4, ema=0.9)
        # num_accepted = 1 -> only target's correction token, 0 draft.
        n_draft = dec._update_acceptance_rate(1)
        assert n_draft == 0
        # alpha = 0.9*0.5 + 0.1*0.0 = 0.45
        assert dec._alpha == pytest.approx(0.45)

    def test_num_accepted_draft_clamped_to_lambda(self):
        dec = _bare_speculative(lam=3, ema=0.9)
        # num_accepted=5 would give 4 draft, but lambda=3 clamps it.
        n_draft = dec._update_acceptance_rate(5)
        assert n_draft == 3

    def test_update_rejects_zero_accepted(self):
        dec = _bare_speculative()
        with pytest.raises(AssertionError):
            dec._update_acceptance_rate(0)

    def test_stats_summary_empty(self):
        from tests.test_flash_speculative import MockModel as MM

        dec = SpeculativeDecoder(
            draft_model=MM(16, 8),
            target_model=MM(16, 8),
            num_speculative_tokens=4,
        )
        s = dec.stats_summary()
        # No steps run: acceptance / avg are guarded against div-by-zero.
        assert s["steps"] == 0
        assert s["proposed"] == 0
        assert s["acceptance_rate"] == 0.0
        assert s["avg_tokens_per_step"] == 0.0
        assert s["lambda"] == 4
        dec.close()

    def test_stats_summary_after_steps(self):
        draft = MockModel(16, 8)
        target = MockModel(16, 8)
        # Share weights so draft and target agree -> full acceptance.
        target.embed.weight = draft.embed.weight
        target.lm_head.weight = draft.lm_head.weight
        for i in range(len(draft.layers)):
            target.layers[i] = draft.layers[i]
        dec = SpeculativeDecoder(
            draft_model=draft, target_model=target, num_speculative_tokens=3
        )
        dec.prefill(mx.array([[1, 2, 3]]))
        dec.step()
        dec.step()
        s = dec.stats_summary()
        assert s["steps"] == 2
        assert s["proposed"] == 6  # 2 steps * lambda 3
        # avg_tokens_per_step = (accepted_draft + steps) / steps
        assert s["avg_tokens_per_step"] == pytest.approx((s["accepted_draft"] + 2) / 2)
        assert s["acceptance_rate"] == pytest.approx(s["accepted_draft"] / 6)
        dec.close()


class TestSpeculativeReset:
    def test_reset_zeroes_stats_and_state(self):
        draft = MockModel(16, 8)
        target = MockModel(16, 8)
        dec = SpeculativeDecoder(
            draft_model=draft, target_model=target, num_speculative_tokens=2
        )
        dec.prefill(mx.array([[1, 2, 3]]))
        dec.step()
        assert dec._stats_steps == 1
        dec.reset()
        assert dec._target_cache is None
        assert dec._draft_cache is None
        assert dec._cache_seq_len == 0
        assert dec._last_target_logit is None
        assert dec._pending_token is None
        assert dec._stats_steps == 0
        assert dec._stats_proposed == 0
        assert dec._stats_accepted_draft == 0
        dec.close()

    def test_step_before_prefill_raises(self):
        dec = SpeculativeDecoder(
            draft_model=MockModel(16, 8),
            target_model=MockModel(16, 8),
            num_speculative_tokens=2,
        )
        with pytest.raises(AssertionError):
            dec.step()
        dec.close()


class TestSpeculativeStepIntegration:
    """Drive prefill+step with independent (non-shared) models so partial
    rejection and cache-trim bookkeeping in _step_linear get exercised."""

    def test_partial_rejection_trims_and_advances(self):
        draft = MockModel(16, 8)
        target = MockModel(16, 8)
        dec = SpeculativeDecoder(
            draft_model=draft, target_model=target, num_speculative_tokens=4
        )
        dec.prefill(mx.array([[1, 2, 3]]))
        accepted, num_draft = dec.step()
        # Independent random weights almost surely reject early, so the
        # accepted prefix is short but always >= 1 (target correction).
        assert 1 <= len(accepted) <= 5
        assert num_draft == 4
        # Cache offset must equal the new sequence length after trimming.
        assert dec._target_cache[0].offset == dec._cache_seq_len
        assert dec._cache_seq_len == 3 + len(accepted)
        # Stats incremented exactly once.
        assert dec._stats_steps == 1
        assert dec._stats_proposed == 4
        assert 0 <= dec._stats_accepted_draft <= 4
        dec.close()

    def test_generate_step_updates_stats(self):
        draft = MockModel(16, 8)
        target = MockModel(16, 8)
        dec = SpeculativeDecoder(
            draft_model=draft, target_model=target, num_speculative_tokens=3
        )
        accepted, num_draft = dec.generate_step(mx.array([[1, 2, 3]]))
        assert len(accepted) >= 1
        assert num_draft == 3
        assert dec._stats_steps == 1
        assert dec._stats_proposed == 3
        assert 0 <= dec._alpha <= 1.0
        dec.close()

    def test_draft_generate_cached_length(self):
        draft = MockModel(16, 8)
        target = MockModel(16, 8)
        dec = SpeculativeDecoder(
            draft_model=draft, target_model=target, num_speculative_tokens=4
        )
        dec.prefill(mx.array([[1, 2, 3]]))
        tokens, ctx = dec._draft_generate_cached(pending_token=2, n=5)
        assert len(tokens) == 5
        assert all(0 <= t < 16 for t in tokens)
        assert ctx == []  # base class returns empty context
        dec.close()


# ---------------------------------------------------------------------------
# PromptLookupDecoder construction guards
# ---------------------------------------------------------------------------


class TestPLDValidation:
    def test_rejects_zero_speculative_tokens(self):
        with pytest.raises(ValueError, match="num_speculative_tokens must be >= 1"):
            PromptLookupDecoder(MockModel(16, 8), num_speculative_tokens=0)

    def test_rejects_bad_ngram_range(self):
        with pytest.raises(ValueError, match="Invalid n-gram range"):
            PromptLookupDecoder(MockModel(16, 8), max_ngram_size=2, min_ngram_size=3)

    def test_rejects_min_ngram_below_one(self):
        with pytest.raises(ValueError, match="Invalid n-gram range"):
            PromptLookupDecoder(MockModel(16, 8), min_ngram_size=0)

    def test_rejects_window_smaller_than_max_ngram(self):
        with pytest.raises(ValueError, match="must be >= max_ngram_size"):
            PromptLookupDecoder(MockModel(16, 8), max_ngram_size=4, lookup_window=2)

    def test_none_window_is_allowed(self):
        dec = PromptLookupDecoder(MockModel(16, 8), lookup_window=None)
        assert dec._lookup_window is None
        dec.close()


# ---------------------------------------------------------------------------
# PromptLookupDecoder._lookup_draft (n-gram match)
# ---------------------------------------------------------------------------


def _bare_pld(
    tokens, pending, lam=10, max_ng=3, min_ng=1, window=8192
) -> PromptLookupDecoder:
    """A PLD configured for direct _lookup_draft testing (no forward needed)."""
    dec = PromptLookupDecoder.__new__(PromptLookupDecoder)
    dec._lambda = lam
    dec._max_ngram = max_ng
    dec._min_ngram = min_ng
    dec._lookup_window = window
    dec._tokens = list(tokens)
    dec._pending_token = pending
    return dec


class TestPLDLookupDraft:
    def test_basic_unigram_match(self):
        # seq = [..., 5, 6, 7, pending=5]; trailing 1-gram "5" last matched
        # at index 0, draft = tokens after it = [6, 7].
        dec = _bare_pld([5, 6, 7], pending=5, max_ng=1, min_ng=1, lam=10)
        draft = dec._lookup_draft()
        assert draft == [6, 7]

    def test_longest_ngram_preferred(self):
        # seq = [1, 2, 3, 9, 1, 2, 3, pending=9]? Build so a 3-gram matches.
        # tokens=[1,2,3,4,1,2,3], pending=4 -> seq trailing 3-gram is [2,3,4]?
        # Actually trailing 3-gram = seq[-3:] = [2,3,4]. That occurs only once
        # so 3-gram fails; fall to 2-gram [3,4] -> not repeated -> 1-gram "4"
        # matches at index 3 -> draft = [1,2,3]. Verify max-ngram-first walk.
        dec = _bare_pld([1, 2, 3, 4, 1, 2, 3], pending=4, max_ng=3, min_ng=1)
        draft = dec._lookup_draft()
        # 1-gram "4" matched at idx 3, tokens after = [1, 2, 3].
        assert draft == [1, 2, 3]

    def test_three_gram_match(self):
        # tokens=[1,2,3,7,8,1,2,3], pending=7 -> trailing 3-gram = [3, ... ]
        # seq = [1,2,3,7,8,1,2,3,7]; trailing 3-gram = [2,3,7] matches at idx 1
        # -> draft starts after idx 1+3=4? no. Let's construct deliberately:
        # seq trailing 3-gram = seq[-3:] = [3, 7] no, len issue. Use explicit.
        dec = _bare_pld([9, 1, 2, 3, 9, 1, 2], pending=3, max_ng=3, min_ng=1)
        # seq = [9,1,2,3,9,1,2,3]; trailing 3-gram = [1,2,3] occurs at idx 1.
        # draft = tokens after idx 1+3=4 => [9,1,2] capped at L-1.
        draft = dec._lookup_draft()
        assert draft == [9, 1, 2]

    def test_no_match_returns_empty(self):
        dec = _bare_pld([1, 2, 3, 4], pending=99, max_ng=3, min_ng=1)
        assert dec._lookup_draft() == []

    def test_draft_capped_at_lambda(self):
        # Long repeat so plenty of follow tokens, but lambda caps the draft.
        body = [5] + list(range(10, 30))
        # seq = body + body + [pending=5]; unigram "5" match yields up to lambda.
        dec = _bare_pld(body + body, pending=5, max_ng=1, min_ng=1, lam=3)
        draft = dec._lookup_draft()
        assert len(draft) == 3

    def test_pending_excluded_from_draft(self):
        # The closest match (start = query_start-1) would propose [pending];
        # draft_end is capped at L-1 so that match yields an empty draft.
        # seq = [5]; pending=5 -> seq=[5, 5]. unigram "5": query_start=1.
        # The only candidate start=0 gives draft_start=1, draft_end=
        # min(1+lam, L-1=1)=1 -> empty (would otherwise propose [pending]).
        dec = _bare_pld([5], pending=5, max_ng=1, min_ng=1)
        assert dec._lookup_draft() == []

    def test_short_seq_returns_empty(self):
        # L < 2 -> early return.
        dec = _bare_pld([], pending=7, max_ng=3, min_ng=1)
        assert dec._lookup_draft() == []

    def test_lookup_draft_none_pending_raises(self):
        dec = _bare_pld([1, 2, 3], pending=None)
        with pytest.raises(RuntimeError, match="_pending_token=None"):
            dec._lookup_draft()


# ---------------------------------------------------------------------------
# PromptLookupDecoder prefill / step / EMA via scripted target
# ---------------------------------------------------------------------------


class TestPLDStep:
    def test_prefill_seeds_tokens_and_returns_token(self):
        target = _ScriptedTarget(16, next_tokens=[5])
        dec = PromptLookupDecoder(target, num_speculative_tokens=4)
        first = dec.prefill(mx.array([[1, 2, 3]]))
        assert first == 5
        assert dec._tokens == [1, 2, 3]
        assert dec._cache_seq_len == 3
        assert dec._pending_token == 5
        dec.close()

    def test_prefill_caps_seed_to_window(self):
        target = _ScriptedTarget(16, next_tokens=[0])
        dec = PromptLookupDecoder(
            target, num_speculative_tokens=2, max_ngram_size=2, lookup_window=4
        )
        dec.prefill(mx.array([[1, 2, 3, 4, 5, 6, 7]]))
        # Only the last 4 prompt tokens are seeded into the lookup table.
        assert dec._tokens == [4, 5, 6, 7]
        dec.close()

    def test_step_before_prefill_raises(self):
        dec = PromptLookupDecoder(MockModel(16, 8))
        with pytest.raises(RuntimeError, match="called before prefill"):
            dec.step()
        dec.close()

    def test_step_no_match_does_not_update_ema(self):
        # pending never appears in history -> no draft -> EMA stays 0.
        target = _ScriptedTarget(16, next_tokens=[15])
        dec = PromptLookupDecoder(target, num_speculative_tokens=4)
        dec.prefill(mx.array([[1, 2, 3]]))  # pending -> 15
        accepted, num_drafted = dec.step()
        assert num_drafted == 0
        assert len(accepted) == 1  # only the target's own next token
        # EMA untouched (skip-on-no-match rule).
        assert dec._alpha == 0.0
        assert dec.stats_summary()["acceptance_rate"] == 0.0
        dec.close()

    def test_step_with_match_updates_state_and_ema(self):
        # Build history that makes a unigram match. We control acceptance by
        # making the target's argmax agree with the drafted tokens.
        #
        # prefill prompt = [7, 8, 7] -> _prefill_last_logit runs two passes
        # consuming 3 scripted slots (cursor 0,1 on [7,8]; cursor 2 on [7]),
        # so the returned pending = next_tokens[2]. We set that to 8 so that
        # pending=8 matches the unigram at history index 1 -> draft = [7].
        # The step() target forward over [pending=8, draft=7] then consumes
        # next_tokens[3], next_tokens[4]: set to [7, 9] so draft 7 is
        # accepted and 9 is the bonus.
        target = _ScriptedTarget(16, next_tokens=[1, 1, 8, 7, 9])
        dec = PromptLookupDecoder(
            target, num_speculative_tokens=4, max_ngram_size=1, min_ngram_size=1
        )
        first = dec.prefill(mx.array([[7, 8, 7]]))
        assert first == 8  # pending
        accepted, num_drafted = dec.step()
        # draft was [7]; target argmax for [pending=8, draft=7] forward is
        # [7, 9] -> accept draft 7, bonus 9.
        assert num_drafted == 1
        assert accepted == [7, 9]
        # EMA updated because something was drafted (acceptance 1/1 = 1.0).
        # alpha = 0.9*0.0 + 0.1*1.0 = 0.1
        assert dec._alpha == pytest.approx(0.1)
        # Stats reflect the accepted draft.
        assert dec._stats_steps == 1
        assert dec._stats_proposed == 1
        assert dec._stats_accepted_draft == 1
        # Pending becomes the bonus token.
        assert dec._pending_token == 9
        dec.close()

    def test_reset_clears_pld_state_and_ema(self):
        target = _ScriptedTarget(16, next_tokens=[3])
        dec = PromptLookupDecoder(target, num_speculative_tokens=2)
        dec.prefill(mx.array([[1, 2, 3]]))
        dec.step()
        dec.reset()
        assert dec._target_cache is None
        assert dec._cache_seq_len == 0
        assert dec._pending_token is None
        assert dec._tokens == []
        assert dec._stats_steps == 0
        assert dec._alpha == 0.0
        dec.close()


# ---------------------------------------------------------------------------
# _eval_cache: warning path for unrecognised non-empty cache (extra branch)
# ---------------------------------------------------------------------------


class TestEvalCacheEmpty:
    def test_empty_cache_list_no_log(self, caplog):
        from olmlx.engine.speculative import _eval_cache

        with caplog.at_level(logging.ERROR, logger="olmlx.engine.speculative"):
            _eval_cache([])
        assert not any("no mx.array entries found" in r.message for r in caplog.records)
