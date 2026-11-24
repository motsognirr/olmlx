"""Tests for model-agnostic speculative streaming (engine/speculative_stream.py)."""

import threading

import pytest

from tests.test_flash_speculative import MockModel


class TestSpeculativeStreamBase:
    """Verify that the base speculative_stream module works with SpeculativeDecoder."""

    @pytest.fixture()
    def shared_decoder(self):
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

    def test_stream_yields_tokens(self, shared_decoder):
        from olmlx.engine.speculative_stream import speculative_stream_generate

        cancel = threading.Event()
        responses = list(
            speculative_stream_generate(
                shared_decoder, [1, 2, 3], max_tokens=5, cancel_event=cancel
            )
        )

        assert len(responses) >= 1
        for resp in responses:
            assert hasattr(resp, "text")
            assert hasattr(resp, "token")

    def test_stream_respects_max_tokens(self, shared_decoder):
        from olmlx.engine.speculative_stream import speculative_stream_generate

        cancel = threading.Event()
        responses = list(
            speculative_stream_generate(
                shared_decoder, [1, 2, 3], max_tokens=8, cancel_event=cancel
            )
        )

        assert len(responses) <= 8

    def test_async_stream_importable(self):
        from olmlx.engine.speculative_stream import async_speculative_stream

        assert callable(async_speculative_stream)

    def test_flash_module_reexports(self):
        """Flash speculative_stream module should re-export from base."""
        from olmlx.engine.speculative_stream import speculative_stream_generate as base
        from olmlx.engine.flash.speculative_stream import (
            speculative_stream_generate as flash,
        )

        assert base is flash


class TestSpecialTokenTextStripping:
    """Stop tokens (e.g. Qwen ``<|im_end|>``) must not leak into streamed text.

    Pre-fix, ``tokenizer.decode(generated)`` was called without
    ``skip_special_tokens=True``, so the chunk that carried the EOS token also
    carried its literal string form. Clients (opencode, anthropic SDK, raw
    SSE consumers) then rendered ``<|im_end|>`` as plain text right before the
    ``finish_reason=stop`` signal.

    The non-speculative path used by every other model went through mlx-lm's
    ``stream_generate`` which sets ``skip_special_tokens=True`` by default,
    hiding the bug to non-speculative configurations.
    """

    class _StubDecoder:
        """Emits a fixed token sequence including a final EOS-special token."""

        def __init__(self, tokens: list[int]) -> None:
            self._tokens = tokens
            self._idx = 0

        def prefill(self, prompt, *, segmented=None, cancel_event=None):  # noqa: ARG002 - protocol compliance
            self._idx = 1
            return self._tokens[0]

        def step(self) -> tuple[list[int], int]:
            if self._idx >= len(self._tokens):
                # Decoder exhausted — caller should have stopped on EOS earlier.
                return ([], 0)
            tok = self._tokens[self._idx]
            self._idx += 1
            return ([tok], 1)

        def reset(self) -> None:
            self._idx = 0

    class _MockTokenizer:
        """Models a Qwen-style tokenizer where token 99 is a special EOS marker.

        Honours ``skip_special_tokens`` like real HuggingFace tokenizers do —
        without it, token 99 renders as the literal string ``<|im_end|>``.
        """

        eos_token_id = 99

        def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
            parts: list[str] = []
            for t in tokens:
                if t == 99:
                    if not skip_special_tokens:
                        parts.append("<|im_end|>")
                else:
                    parts.append(f"w{t}")
            return "".join(parts)

    def test_streamed_text_omits_eos_special_token(self):
        from olmlx.engine.speculative_stream import speculative_stream_generate

        decoder = self._StubDecoder([1, 2, 3, 99])
        tok = self._MockTokenizer()
        cancel = threading.Event()

        responses = list(
            speculative_stream_generate(
                decoder,
                [10, 11],
                max_tokens=8,
                cancel_event=cancel,
                eos_token_id=tok.eos_token_id,
                tokenizer=tok,
            )
        )

        full_streamed = "".join(r.text for r in responses)
        assert "<|im_end|>" not in full_streamed, (
            f"special token leaked into streamed text: {full_streamed!r}"
        )
        # And the substantive content tokens did make it through.
        for marker in ("w1", "w2", "w3"):
            assert marker in full_streamed, (
                f"expected content token {marker!r} missing from {full_streamed!r}"
            )
        # The final chunk should carry finish_reason=stop on the EOS token.
        assert responses[-1].token == 99
        assert responses[-1].finish_reason == "stop"

    def test_streamed_text_omits_eos_when_first_token_is_eos(self):
        """Edge case: prefill returns EOS as the very first token."""
        from olmlx.engine.speculative_stream import speculative_stream_generate

        decoder = self._StubDecoder([99])
        tok = self._MockTokenizer()
        cancel = threading.Event()

        responses = list(
            speculative_stream_generate(
                decoder,
                [10, 11],
                max_tokens=8,
                cancel_event=cancel,
                eos_token_id=tok.eos_token_id,
                tokenizer=tok,
            )
        )

        assert len(responses) == 1
        assert responses[0].token == 99
        assert "<|im_end|>" not in responses[0].text, (
            f"prefill EOS leaked: {responses[0].text!r}"
        )


class TestMultipleEosIds:
    """#757: speculative decoding must stop on *any* of the tokenizer's EOS
    ids (Gemma 3 ``<end_of_turn>``, Llama 3.x eot/eom), like the mlx-lm path
    does via ``tokenizer.eos_token_ids`` — not just ``eos_token_id``."""

    class _StubDecoder(TestSpecialTokenTextStripping._StubDecoder):
        """Pads with a non-EOS filler once exhausted, so a missed stop runs
        to ``max_tokens`` instead of spinning on empty steps forever."""

        def step(self) -> tuple[list[int], int]:
            accepted, n = super().step()
            return (accepted, n) if accepted else ([0], 1)

    def test_stops_on_secondary_eos_id(self):
        from olmlx.engine.speculative_stream import speculative_stream_generate

        decoder = self._StubDecoder([1, 2, 98, 5, 6, 99])
        responses = list(
            speculative_stream_generate(
                decoder,
                [10],
                max_tokens=16,
                cancel_event=threading.Event(),
                eos_token_ids={98, 99},
            )
        )
        assert [r.token for r in responses] == [1, 2, 98]
        assert responses[-1].finish_reason == "stop"

    def test_first_token_secondary_eos_stops(self):
        from olmlx.engine.speculative_stream import speculative_stream_generate

        decoder = self._StubDecoder([98, 5, 6])
        responses = list(
            speculative_stream_generate(
                decoder,
                [10],
                max_tokens=16,
                cancel_event=threading.Event(),
                eos_token_ids={98, 99},
            )
        )
        assert [r.token for r in responses] == [98]

    def test_tokenizer_eos_ids_union(self):
        from olmlx.engine.speculative_stream import tokenizer_eos_ids

        class _Tok:
            eos_token_id = 1
            eos_token_ids = {106, 1}

        assert tokenizer_eos_ids(_Tok()) == frozenset({1, 106})

    def test_tokenizer_eos_ids_single_only(self):
        from olmlx.engine.speculative_stream import tokenizer_eos_ids

        class _Tok:
            eos_token_id = 7

        assert tokenizer_eos_ids(_Tok()) == frozenset({7})
        assert tokenizer_eos_ids(None) == frozenset()

    def test_tokenizer_eos_ids_from_vlm_stopping_criteria(self):
        """mlx-vlm loads attach the config's ``eos_token_id`` list (Gemma 3:
        ``[1, 106]``) to the raw HF tokenizer as ``stopping_criteria`` — the
        HF tokenizer itself has no ``eos_token_ids``. Speculative VLM targets
        pass that raw tokenizer, so the criteria must be folded in."""
        from olmlx.engine.speculative_stream import tokenizer_eos_ids

        class _Criteria:
            eos_token_ids = [1, 106]
            additional_eos_token_ids = [107]

        class _HFTok:
            eos_token_id = 1
            stopping_criteria = _Criteria()

        assert tokenizer_eos_ids(_HFTok()) == frozenset({1, 106, 107})

    def test_tokenizer_eos_ids_list_valued_eos_token_id(self):
        from olmlx.engine.speculative_stream import tokenizer_eos_ids

        class _Tok:
            eos_token_id = [1, 106]

        assert tokenizer_eos_ids(_Tok()) == frozenset({1, 106})

    def test_tokenizer_eos_ids_numpy_ints(self):
        import numpy as np

        from olmlx.engine.speculative_stream import tokenizer_eos_ids

        class _Tok:
            eos_token_id = np.int64(1)
            eos_token_ids = [np.int64(106)]

        ids = tokenizer_eos_ids(_Tok())
        assert ids == frozenset({1, 106})
        assert all(type(t) is int for t in ids)

    def test_tokenizer_eos_ids_list_valued_stopping_criteria(self):
        from olmlx.engine.speculative_stream import tokenizer_eos_ids

        class _Tok:
            eos_token_id = 1
            stopping_criteria = [1, 106]

        assert tokenizer_eos_ids(_Tok()) == frozenset({1, 106})

    async def test_async_stream_uses_all_eos_ids(self):
        from olmlx.engine.speculative_stream import async_speculative_stream

        class _Tok:
            eos_token_id = 99
            eos_token_ids = {98, 99}

            def decode(self, tokens, skip_special_tokens=False):
                return "".join(f"w{t}" for t in tokens)

        stream = async_speculative_stream(
            self._StubDecoder([1, 98, 5, 6]), _Tok(), [10], max_tokens=16
        )
        tokens = [item.token async for item in stream]
        await stream.drain_and_join()
        assert tokens == [1, 98]


class _ClockedDecoder:
    """Speculative decoder stand-in on a fake clock: prefill costs
    ``prefill_s`` seconds, each step costs ``step_s`` and accepts
    ``tokens_per_step`` tokens."""

    def __init__(self, clock, prefill_s, step_s, tokens_per_step):
        self.clock = clock
        self.prefill_s = prefill_s
        self.step_s = step_s
        self.tokens_per_step = tokens_per_step

    def prefill(self, prompt, *, segmented=None, cancel_event=None):
        self.clock.t += self.prefill_s
        return 100

    def step(self):
        self.clock.t += self.step_s
        return list(range(101, 101 + self.tokens_per_step)), self.tokens_per_step - 1

    def reset(self):
        pass


class TestSpeculativeStreamTiming:
    """The decode rate must cover decode only. A rate that also spans the
    prefill makes the server count prefill twice: _derive_timing_stats then
    splits the wall time proportionally, halving the reported prefill and
    moving the other half into decode (a 14k-token MTP request showed
    ~1 tok/s decode while it really ran at ~25 tok/s)."""

    def _run(self, monkeypatch, max_tokens):
        from types import SimpleNamespace

        from olmlx.engine import speculative_stream as ss

        clock = SimpleNamespace(t=0.0)
        monkeypatch.setattr(ss, "time", SimpleNamespace(perf_counter=lambda: clock.t))
        dec = _ClockedDecoder(clock, prefill_s=10.0, step_s=0.1, tokens_per_step=2)
        return list(
            ss.speculative_stream_generate(
                dec,
                list(range(1000)),
                max_tokens=max_tokens,
                cancel_event=threading.Event(),
            )
        )

    def test_generation_rate_excludes_prefill(self, monkeypatch):
        last = self._run(monkeypatch, max_tokens=9)[-1]
        # 1 token from prefill + 4 steps x 2 tokens; decode took 4 x 0.1 s.
        assert last.generation_tokens == 9
        assert last.prompt_tps == pytest.approx(1000 / 10.0)
        assert last.generation_tps == pytest.approx(9 / 0.4)

    def test_server_durations_match_true_prefill_and_decode(self, monkeypatch):
        from olmlx.engine.inference import _derive_timing_stats
        from olmlx.utils.timing import TimingStats

        last = self._run(monkeypatch, max_tokens=9)[-1]
        stats = TimingStats(prompt_eval_count=1000, eval_count=last.generation_tokens)
        _derive_timing_stats(
            stats,
            last.prompt_tps,
            last.generation_tps,
            eval_timer_ns=10_400_000_000,
            prefilled_count=last.prompt_tokens,
        )
        assert stats.prompt_eval_duration == pytest.approx(10_000_000_000, rel=1e-6)
        assert stats.eval_duration == pytest.approx(400_000_000, rel=1e-6)

    def test_first_token_stop_reports_no_decode_time(self, monkeypatch):
        """A response that ends on the prefill's own token did no decoding.
        Its rate must still be a decode rate (mlx-lm times the first token
        from the end of prefill too): with no rate at all, a wall clock that
        equals the prefill sends _derive_timing_stats into its 50/50 split."""
        from types import SimpleNamespace

        from olmlx.engine import speculative_stream as ss
        from olmlx.engine.inference import _derive_timing_stats
        from olmlx.utils.timing import TimingStats

        clock = SimpleNamespace(t=0.0)
        monkeypatch.setattr(ss, "time", SimpleNamespace(perf_counter=lambda: clock.t))
        dec = _ClockedDecoder(clock, prefill_s=10.0, step_s=0.1, tokens_per_step=2)
        out = list(
            ss.speculative_stream_generate(
                dec,
                list(range(1000)),
                max_tokens=50,
                cancel_event=threading.Event(),
                eos_token_id=100,  # the token prefill returns
            )
        )
        assert [t.token for t in out] == [100]
        stats = TimingStats(prompt_eval_count=1000, eval_count=1)
        _derive_timing_stats(
            stats,
            out[-1].prompt_tps,
            out[-1].generation_tps,
            eval_timer_ns=10_000_000_000,
            prefilled_count=out[-1].prompt_tokens,
        )
        assert stats.prompt_eval_duration == pytest.approx(10_000_000_000, rel=1e-6)
        assert stats.eval_duration < 1_000_000
