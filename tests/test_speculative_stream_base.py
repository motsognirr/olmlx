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
