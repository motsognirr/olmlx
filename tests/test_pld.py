"""Tests for prompt-lookup decoding (engine/speculative.py:PromptLookupDecoder)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from tests.test_flash_speculative import MockModel


class TestLookupDraft:
    """Unit tests for the n-gram lookup primitive (``_lookup_draft``).

    These tests bypass the model forward and exercise the pure-Python
    search by constructing a decoder with no caches and seeding state
    directly. They cover the matching rules the property tests rely on:
    most-recent match wins, longest n-gram size wins, no-match returns
    [], and the trailing query position is excluded.
    """

    @pytest.fixture()
    def decoder(self):
        from olmlx.engine.speculative import PromptLookupDecoder

        # MockModel only exists so PromptLookupDecoder.__init__ can probe
        # for a GDN class (it won't find one). The forward path isn't
        # exercised here.
        target = MockModel(32, 16)
        return PromptLookupDecoder(
            target_model=target,
            num_speculative_tokens=5,
            max_ngram_size=3,
            min_ngram_size=1,
        )

    def _seed(self, decoder, tokens: list[int]) -> None:
        """Set ``_tokens`` (history) and ``_pending_token`` (last token).

        Mirrors decoder state after some prefill+steps without actually
        running them. The pending token is the most recent emitted token
        and participates in the n-gram search.
        """
        assert len(tokens) >= 1
        decoder._tokens = list(tokens[:-1])
        decoder._pending_token = tokens[-1]

    def test_no_match_returns_empty(self, decoder):
        """No occurrence of the trailing n-gram → empty draft."""
        # Distinct tokens, no n-gram of any size has a duplicate.
        self._seed(decoder, [1, 2, 3, 4, 5])
        assert decoder._lookup_draft() == []

    def test_unigram_match_returns_following_tokens(self, decoder):
        """A 1-gram match returns the tokens after the matching position."""
        # Sequence: [9, 7, 8, 5, 4, 9]. Trailing query=[9] at index 5;
        # matches at position 0 (only prior occurrence). Tokens after
        # position 0 are seq[1:6] = [7, 8, 5, 4, 9] — capped at
        # lambda=5. The trailing 9 reappears as the 5th draft token by
        # coincidence; that's fine — PLD will let target verification
        # accept or reject it.
        self._seed(decoder, [9, 7, 8, 5, 4, 9])
        assert decoder._lookup_draft() == [7, 8, 5, 4, 9]

    def test_ngram_size_preference(self, decoder):
        """Longest n-gram wins: a 3-gram match preempts a shorter 2-gram match."""
        # Sequence: [1, 2, 3, 4, 7, 1, 2, 3]. Trailing 3-gram=[1, 2, 3]
        # at index 5; matches at position 0; draft_start=3; draft
        # extends seq[3:8] = [4, 7, 1, 2, 3] (capped at lambda=5).
        # The shorter 2-gram [2, 3] also matches at position 1 but is
        # not consulted because a longer n-gram already won.
        self._seed(decoder, [1, 2, 3, 4, 7, 1, 2, 3])
        assert decoder._lookup_draft() == [4, 7, 1, 2, 3]

    def test_most_recent_match_wins(self, decoder):
        """When multiple positions match the trailing n-gram, prefer the latest."""
        # Sequence: [5, 6, 9, 5, 6, 7, 8, 5, 6] (L=9). The trailing
        # 3-gram [8, 5, 6] doesn't recur, so we drop to 2-gram. Trailing
        # 2-gram=[5, 6] at index 7 matches at positions 0 and 3; the
        # most recent (position 3) wins. draft_start=5; seq[5:9] =
        # [7, 8, 5, 6] (bounded by sequence length, not lambda).
        self._seed(decoder, [5, 6, 9, 5, 6, 7, 8, 5, 6])
        assert decoder._lookup_draft() == [7, 8, 5, 6]

    def test_trailing_query_excluded(self, decoder):
        """The trailing n-gram itself must not be returned as its own match."""
        # Sequence: [9, 9]. Trailing 1-gram=[9]; the only other position
        # is 0 (also a 9), which counts. Tokens after position 0 are
        # [9] — that's the trailing query, but as a *following* token
        # of a valid prior match it's a legitimate draft.
        self._seed(decoder, [9, 9])
        assert decoder._lookup_draft() == [9]

    def test_caps_at_lambda(self, decoder):
        """Draft is capped at ``num_speculative_tokens``."""
        # Sequence: 0..9 then 0. Trailing 1-gram=[0] matches at position
        # 0; tokens after are [1, 2, 3, 4, 5, 6, 7, 8, 9] — capped at
        # lambda=5 → [1, 2, 3, 4, 5].
        self._seed(decoder, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        assert decoder._lookup_draft() == [1, 2, 3, 4, 5]

    def test_short_history_returns_empty(self, decoder):
        """A single pending token with no history has nothing to search."""
        self._seed(decoder, [7])
        assert decoder._lookup_draft() == []

    def test_min_ngram_floor(self):
        """``min_ngram_size=2`` rejects unigram-only matches."""
        from olmlx.engine.speculative import PromptLookupDecoder

        target = MockModel(32, 16)
        decoder = PromptLookupDecoder(
            target_model=target,
            num_speculative_tokens=5,
            max_ngram_size=3,
            min_ngram_size=2,
        )
        # Trailing token=9 has a 1-gram match earlier, but no 2-gram or
        # 3-gram match. With min_ngram=2 the unigram is ignored.
        self._seed(decoder, [9, 7, 8, 5, 4, 9])
        assert decoder._lookup_draft() == []

    def test_invalid_ngram_range_rejected(self):
        from olmlx.engine.speculative import PromptLookupDecoder

        target = MockModel(32, 16)
        with pytest.raises(ValueError, match="n-gram range"):
            PromptLookupDecoder(
                target_model=target,
                num_speculative_tokens=4,
                max_ngram_size=1,
                min_ngram_size=3,
            )
        with pytest.raises(ValueError, match="n-gram range"):
            PromptLookupDecoder(
                target_model=target,
                num_speculative_tokens=4,
                max_ngram_size=3,
                min_ngram_size=0,
            )

    def test_invalid_lambda_rejected(self):
        from olmlx.engine.speculative import PromptLookupDecoder

        target = MockModel(32, 16)
        with pytest.raises(ValueError, match="num_speculative_tokens"):
            PromptLookupDecoder(
                target_model=target,
                num_speculative_tokens=0,
            )

    def test_lookup_window_caps_search(self):
        """A lookup_window bounds the search to the most recent N tokens.

        Construct a history where the only matching n-gram lives *outside*
        the window — the lookup must miss because the window excludes it.
        With the window large enough to include the match, the same query
        finds the same draft. This is the load-bearing cap that keeps
        per-step cost from growing unbounded with context length.
        """
        from olmlx.engine.speculative import PromptLookupDecoder

        target = MockModel(32, 16)
        # Sequence: [1, 2, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1] (L=13).
        # Trailing 1-gram=[1] matches at position 0. With
        # lookup_window=5, history becomes the last 5 of _tokens =
        # [8, 8, 8, 8, 8] + pending=1; no match for 1 in [8*5]; draft=[].
        tokens = [1, 2, 3] + [8] * 9 + [1]
        windowed = PromptLookupDecoder(
            target_model=target,
            num_speculative_tokens=5,
            max_ngram_size=1,
            min_ngram_size=1,
            lookup_window=5,
        )
        windowed._tokens = tokens[:-1]
        windowed._pending_token = tokens[-1]
        assert windowed._lookup_draft() == []

        # Without the window, the same history hits the position-0 match.
        unbounded = PromptLookupDecoder(
            target_model=target,
            num_speculative_tokens=5,
            max_ngram_size=1,
            min_ngram_size=1,
            lookup_window=None,
        )
        unbounded._tokens = tokens[:-1]
        unbounded._pending_token = tokens[-1]
        # Position 0 (token=1); draft = tokens after = [2, 3, 8, 8, 8] (capped at λ=5).
        assert unbounded._lookup_draft() == [2, 3, 8, 8, 8]

    def test_lookup_window_too_small_rejected(self):
        """``lookup_window < max_ngram_size`` is rejected at construction."""
        from olmlx.engine.speculative import PromptLookupDecoder

        target = MockModel(32, 16)
        with pytest.raises(ValueError, match="lookup_window"):
            PromptLookupDecoder(
                target_model=target,
                num_speculative_tokens=4,
                max_ngram_size=3,
                min_ngram_size=1,
                lookup_window=2,
            )


class TestPromptLookupDecoder:
    """End-to-end tests against MockModel: caches populate, step()
    returns at least one token, state stays consistent across steps,
    output matches the greedy reference."""

    @pytest.fixture()
    def decoder(self):
        from olmlx.engine.speculative import PromptLookupDecoder

        target = MockModel(32, 16)
        return PromptLookupDecoder(
            target_model=target,
            num_speculative_tokens=4,
            max_ngram_size=3,
            min_ngram_size=1,
        )

    def test_prefill_populates_state(self, decoder):
        prompt = mx.array([[1, 2, 3, 4, 5]])
        first_token = decoder.prefill(prompt)

        assert decoder._target_cache is not None
        assert decoder._cache_seq_len == 5
        assert isinstance(first_token, int)
        assert 0 <= first_token < 32
        assert decoder._tokens == [1, 2, 3, 4, 5]
        assert decoder._pending_token == first_token

    def test_step_returns_at_least_one_token(self, decoder):
        prompt = mx.array([[1, 2, 3, 4, 5]])
        decoder.prefill(prompt)

        accepted, num_drafted = decoder.step()
        assert len(accepted) >= 1
        assert 0 <= num_drafted <= decoder._lambda
        # With variable draft length, accepted length is bounded above
        # by num_drafted + 1 (the bonus from verify_draft_greedy).
        assert len(accepted) <= num_drafted + 1

    def test_step_advances_cache(self, decoder):
        prompt = mx.array([[1, 2, 3, 4, 5]])
        decoder.prefill(prompt)
        accepted, _ = decoder.step()

        # Cache grows by exactly num_accepted positions.
        assert decoder._cache_seq_len == 5 + len(accepted)
        assert decoder._target_cache[0].offset == decoder._cache_seq_len

    def test_history_tracks_pending_and_accepted(self, decoder):
        """After each step, _tokens grows by pending+accepted[:-1] and
        _pending_token is set to accepted[-1]."""
        prompt = mx.array([[1, 2, 3, 4, 5]])
        first_token = decoder.prefill(prompt)
        accepted, _ = decoder.step()

        # The pending token from prefill is now in history; so are all
        # accepted tokens except the last (which is the new pending).
        expected_tokens = [1, 2, 3, 4, 5, first_token] + list(accepted[:-1])
        assert decoder._tokens == expected_tokens
        assert decoder._pending_token == accepted[-1]

    def test_reset_clears_state(self, decoder):
        prompt = mx.array([[1, 2, 3]])
        decoder.prefill(prompt)
        decoder.step()

        decoder.reset()
        assert decoder._target_cache is None
        assert decoder._cache_seq_len == 0
        assert decoder._last_target_logit is None
        assert decoder._pending_token is None
        assert decoder._tokens == []

    def test_multi_step_consistency(self, decoder):
        prompt = mx.array([[1, 2, 3, 4, 5]])
        decoder.prefill(prompt)
        emitted = [decoder._pending_token]
        for _ in range(6):
            accepted, _ = decoder.step()
            assert len(accepted) >= 1
            assert decoder._target_cache[0].offset == decoder._cache_seq_len
            # Sequence visible-so-far is exactly history + pending.
            assert (
                decoder._tokens[-1] == emitted[-1]
                or decoder._tokens[-len(accepted) :] == list(accepted[:-1])
                or len(accepted) == 1
            )
            emitted.extend(accepted)

    def test_output_matches_greedy_reference(self):
        """PLD output must equal greedy decoding token-for-token.

        PLD uses greedy verification, so the emitted token sequence is
        identical to running the target greedily one token at a time
        (as long as the target is deterministic). This is the core
        correctness guarantee stated in the issue's acceptance criteria.

        The prompt is deliberately constructed with repeated n-grams so
        PLD actually drafts something to verify, exercising the accept/
        reject branches rather than the no-match degenerate path.
        """
        from mlx_lm.models.cache import make_prompt_cache

        from olmlx.engine.speculative import (
            PromptLookupDecoder,
            _logits,
            _prefill_last_logit,
        )

        target = MockModel(vocab_size=32, hidden_size=16)
        # Repeated 3-gram [3, 7, 1] gives PLD a hit on the first step.
        prompt = mx.array([[3, 7, 1, 5, 9, 2, 4, 6, 8, 3, 7, 1]])
        max_steps = 12

        # Greedy reference: prefill, then one-at-a-time greedy.
        ref_cache = make_prompt_cache(target)
        first_logit = _prefill_last_logit(target, prompt, ref_cache)
        mx.eval(first_logit)
        ref_tokens = [int(mx.argmax(first_logit).item())]
        for _ in range(max_steps - 1):
            inp = mx.array([[ref_tokens[-1]]])
            out = _logits(target(inp, cache=ref_cache))[0, -1, :]
            mx.eval(out)
            ref_tokens.append(int(mx.argmax(out).item()))

        # PLD: prefill + repeated step() until we have >= max_steps.
        pld = PromptLookupDecoder(
            target_model=target,
            num_speculative_tokens=4,
            max_ngram_size=3,
            min_ngram_size=1,
        )
        pld_tokens = [pld.prefill(prompt)]
        while len(pld_tokens) < max_steps:
            accepted, _ = pld.step()
            pld_tokens.extend(accepted)
        pld_tokens = pld_tokens[:max_steps]

        assert pld_tokens == ref_tokens, (
            f"PLD diverged from greedy reference at step "
            f"{next(i for i, (a, b) in enumerate(zip(pld_tokens, ref_tokens)) if a != b)}: "
            f"pld={pld_tokens} greedy={ref_tokens}"
        )

    def test_stats_summary_shape(self, decoder):
        """``stats_summary`` returns the keys the streaming layer reads."""
        prompt = mx.array([[1, 2, 3]])
        decoder.prefill(prompt)
        decoder.step()
        s = decoder.stats_summary()
        for k in (
            "steps",
            "proposed",
            "accepted_draft",
            "acceptance_rate",
            "avg_tokens_per_step",
            "ema_acceptance_rate",
            "lambda",
        ):
            assert k in s, f"missing stats key: {k}"
        assert s["steps"] == 1
        assert 0 <= s["acceptance_rate"] <= 1
        assert s["lambda"] == decoder._lambda

    def test_protocol_conformance(self, decoder):
        """PLD must satisfy SpeculativeDecoderProtocol so the existing
        streaming adapter works without modification."""
        # SpeculativeDecoderProtocol isn't @runtime_checkable, so duck-
        # check the three method names by hand. If a future refactor
        # renames any of these, the streaming adapter would break
        # silently — fail loudly here instead.
        assert callable(getattr(decoder, "prefill", None))
        assert callable(getattr(decoder, "step", None))
        assert callable(getattr(decoder, "reset", None))

    def test_streaming_handles_no_match_step(self):
        """PLD ``step()`` can return ``num_drafted=0`` when no n-gram
        match is found. ``SpeculativeDecoder.step()`` never produces 0
        (it always drafts ``lambda``), so the streaming adapter is
        being asked to handle a new value. This test drives the
        adapter through a prompt of distinct tokens (forcing no
        match) to confirm the no-match step doesn't crash, divide by
        zero, or stall the stream.
        """
        import threading

        from olmlx.engine.speculative import PromptLookupDecoder
        from olmlx.engine.speculative_stream import speculative_stream_generate

        target = MockModel(vocab_size=32, hidden_size=16)
        pld = PromptLookupDecoder(
            target_model=target,
            num_speculative_tokens=4,
            # max_ngram=1 with all-distinct prompt tokens guarantees
            # the first lookup yields no match: every unigram appears
            # exactly once (as the trailing query).
            max_ngram_size=1,
            min_ngram_size=1,
        )
        prompt_tokens = [1, 2, 3, 4, 5, 6, 7]  # all distinct
        tokens = list(
            speculative_stream_generate(
                pld,
                prompt_tokens=prompt_tokens,
                max_tokens=5,
                cancel_event=threading.Event(),
                eos_token_id=None,
                tokenizer=None,
            )
        )
        # 5 yields, regardless of how many internal steps were no-match.
        assert len(tokens) == 5
        # Stats must not divide by zero: a stream where every step
        # was a no-match yields acceptance_rate=0.0, not NaN/raise.
        stats = pld.stats_summary()
        assert isinstance(stats["acceptance_rate"], float)
        assert 0.0 <= stats["acceptance_rate"] <= 1.0
