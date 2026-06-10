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
