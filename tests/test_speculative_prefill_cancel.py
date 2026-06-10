"""Cancellation of speculative prefill.

A client disconnect during a multi-minute speculative prefill must interrupt
the prefill within one sub-chunk instead of running it to completion (which
pins the GPU and 503s the next request via deferred cleanup). The decode loop
already checks ``cancel_event`` between steps; these tests cover the prefill
phase, which previously had no checkpoint.
"""

import threading

import mlx.core as mx

from tests.test_flash_speculative import MockModel


class _CancelOnNthCall:
    """Wraps a model; records each forward's seq_len and sets ``cancel_event``
    after the ``n``-th forward, simulating a client disconnect mid-prefill."""

    def __init__(self, inner, cancel_event, n=1):
        self._inner = inner
        self.layers = inner.layers
        self.calls: list[int] = []
        self._cancel_event = cancel_event
        self._n = n

    def __call__(self, input_ids, cache=None):
        self.calls.append(int(input_ids.shape[1]))
        out = self._inner(input_ids, cache=cache)
        if len(self.calls) >= self._n:
            self._cancel_event.set()
        return out

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class TestChunkedPrefillCancellation:
    def test_chunked_prefill_raises_when_cancelled_upfront(self):
        from mlx_lm.models.cache import make_prompt_cache

        from olmlx.engine.speculative import (
            PrefillCancelled,
            _chunked_prefill,
        )
        import pytest

        inner = MockModel(32, 16)
        cancel = threading.Event()
        cancel.set()  # already cancelled before any work
        model = _CancelOnNthCall(inner, cancel)
        cache = make_prompt_cache(inner)
        prompt = mx.zeros((1, 100), dtype=mx.int32)

        with pytest.raises(PrefillCancelled):
            _chunked_prefill(model, prompt, cache, cancel_event=cancel)

        # The top-of-loop check must fire before any forward runs.
        assert model.calls == []

    def test_chunked_prefill_stops_after_chunk_where_cancel_set(self, monkeypatch):
        """Cancel set during prefill stops at the next chunk boundary, not at
        the end — at most one extra forward after the signal."""
        from mlx_lm.models.cache import make_prompt_cache

        import pytest

        from olmlx.engine import speculative
        from olmlx.engine.speculative import PrefillCancelled, _chunked_prefill

        monkeypatch.setattr(speculative, "_PREFILL_CHUNK", 2)
        cancel = threading.Event()
        inner = MockModel(32, 16)
        # Set cancel after the 1st forward; loop must raise at the top of the
        # 2nd iteration instead of grinding through all chunks.
        model = _CancelOnNthCall(inner, cancel, n=1)
        cache = make_prompt_cache(inner)
        prompt = mx.zeros((1, 20), dtype=mx.int32)  # 10 chunks of 2

        with pytest.raises(PrefillCancelled):
            _chunked_prefill(model, prompt, cache, cancel_event=cancel)

        assert len(model.calls) == 1  # stopped right after the cancel fired

    def test_chunked_prefill_completes_without_cancel(self):
        """Backward compat: no cancel_event behaves exactly as before."""
        from mlx_lm.models.cache import make_prompt_cache

        from olmlx.engine.speculative import _chunked_prefill

        inner = MockModel(32, 16)
        cache = make_prompt_cache(inner)
        prompt = mx.zeros((1, 50), dtype=mx.int32)

        _chunked_prefill(inner, prompt, cache)  # no cancel_event kwarg
        assert cache[0].offset == 50

    def test_prefill_last_logit_raises_when_cancelled(self):
        from mlx_lm.models.cache import make_prompt_cache

        import pytest

        from olmlx.engine.speculative import (
            PrefillCancelled,
            _prefill_last_logit,
        )

        inner = MockModel(32, 16)
        cache = make_prompt_cache(inner)
        cancel = threading.Event()
        cancel.set()
        prompt = mx.zeros((1, 100), dtype=mx.int32)

        with pytest.raises(PrefillCancelled):
            _prefill_last_logit(inner, prompt, cache, cancel_event=cancel)

    def test_prefill_last_logit_skips_final_forward_when_cancelled_after_prefix(self):
        """Cancel set during the prefix loop must skip the final-token forward,
        not just stop the prefix — bound post-cancel work to one sub-chunk."""
        from mlx_lm.models.cache import make_prompt_cache

        import pytest

        from olmlx.engine.speculative import PrefillCancelled, _prefill_last_logit

        cancel = threading.Event()
        inner = MockModel(32, 16)
        # Sets cancel after the first (prefix) forward; the final 1-token
        # forward must then be skipped rather than run unchecked.
        model = _CancelOnNthCall(inner, cancel, n=1)
        cache = make_prompt_cache(inner)
        prompt = mx.zeros((1, 3), dtype=mx.int32)  # prefix=2 (one chunk), last=1

        with pytest.raises(PrefillCancelled):
            _prefill_last_logit(model, prompt, cache, cancel_event=cancel)

        assert model.calls == [2]  # only the prefix forward; final token skipped


class TestDecoderPrefillCancellation:
    def test_speculative_decoder_prefill_raises_when_cancelled(self):
        import pytest

        from olmlx.engine.speculative import PrefillCancelled, SpeculativeDecoder
        from tests.test_flash_speculative import MockDraftModel, MockTargetModel

        draft = MockDraftModel(32, 16)
        target = MockTargetModel(32, 16)
        decoder = SpeculativeDecoder(draft, target, num_speculative_tokens=2)
        cancel = threading.Event()
        cancel.set()

        with pytest.raises(PrefillCancelled):
            decoder.prefill(mx.zeros((1, 100), dtype=mx.int32), cancel_event=cancel)


class TestStreamPrefillCancellation:
    def test_stream_returns_cleanly_on_prefill_cancel(self):
        """When prefill is cancelled, the stream generator yields no tokens and
        returns cleanly (no exception propagates to the caller / lock holder)."""
        from olmlx.engine.speculative import PrefillCancelled
        from olmlx.engine.speculative_stream import speculative_stream_generate

        cancel = threading.Event()

        class _FakeDecoder:
            def prefill(self, prompt, *, segmented=None, cancel_event=None):
                # Simulate the cancel firing during prefill.
                raise PrefillCancelled()

            def step(self):  # pragma: no cover - never reached
                raise AssertionError("step() must not run after prefill cancel")

            def reset(self):
                pass

        gen = speculative_stream_generate(
            _FakeDecoder(),
            [1, 2, 3],
            max_tokens=10,
            cancel_event=cancel,
            eos_token_id=None,
            tokenizer=None,
        )
        assert list(gen) == []

    def test_stream_resets_decoder_on_prefill_cancel(self):
        """Cancelling prefill must release the decoder's partial KV caches
        eagerly via reset(), not leave them on the long-lived decoder instance
        until the next request's prefill()."""
        from olmlx.engine.speculative import PrefillCancelled
        from olmlx.engine.speculative_stream import speculative_stream_generate

        cancel = threading.Event()

        class _FakeDecoder:
            def __init__(self):
                self.reset_calls = 0
                self.cache = None

            def prefill(self, prompt, *, segmented=None, cancel_event=None):
                self.cache = ["partial-kv"]  # simulate populated caches
                raise PrefillCancelled()

            def step(self):  # pragma: no cover - never reached
                raise AssertionError("step() must not run after prefill cancel")

            def reset(self):
                self.reset_calls += 1
                self.cache = None

        decoder = _FakeDecoder()
        gen = speculative_stream_generate(
            decoder,
            [1, 2, 3],
            max_tokens=10,
            cancel_event=cancel,
            eos_token_id=None,
            tokenizer=None,
        )
        assert list(gen) == []
        assert decoder.reset_calls == 1
        assert decoder.cache is None
