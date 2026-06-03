"""``cancel_event`` conformance for ``SelfSpeculativeDecoder.prefill``.

``speculative_stream_generate`` invokes every decoder as
``decoder.prefill(prompt_arr, cancel_event=cancel_event)`` (the
``SpeculativeDecoderProtocol`` signature). ``SelfSpeculativeDecoder.prefill``
must therefore accept the ``cancel_event`` keyword — omitting it raises
``TypeError: prefill() got an unexpected keyword argument 'cancel_event'`` the
moment a self-speculative request reaches the stream. Like dflash/eagle, the
decoder prefills in a single forward, so the event is accepted for conformance
(honored only as far as the single forward allows).
"""

import threading

import mlx.core as mx
import mlx.nn as nn

from tests.test_flash_speculative import MockLayer


class _SelfSpecTarget(nn.Module):
    """Minimal target satisfying ``SelfSpeculativeDecoder.__init__`` discovery:
    ``embed_tokens`` / ``norm`` / ``lm_head`` / ``layers`` with trimmable
    (plain ``KVCache``) attention and no GatedDeltaNet."""

    def __init__(
        self, vocab_size: int = 32, hidden_size: int = 16, num_layers: int = 2
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, cache=None):
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, cache=cache[i] if cache is not None else None)
        return self.lm_head(self.norm(h))


def _make_decoder():
    from olmlx.engine.self_speculative import SelfSpeculativeDecoder

    return SelfSpeculativeDecoder(
        target_model=_SelfSpecTarget(),
        num_early_layers=1,
        num_speculative_tokens=2,
    )


class TestSelfSpeculativePrefillCancelConformance:
    def test_prefill_accepts_cancel_event_kwarg(self):
        """The exact call ``speculative_stream_generate`` makes must not raise
        ``TypeError`` for an unexpected ``cancel_event`` kwarg."""
        decoder = _make_decoder()
        prompt = mx.zeros((1, 8), dtype=mx.int32)

        first_token = decoder.prefill(prompt, cancel_event=threading.Event())

        assert isinstance(first_token, int)

    def test_prefill_without_cancel_event_still_works(self):
        """Backward compat: positional-only call is unchanged."""
        decoder = _make_decoder()
        prompt = mx.zeros((1, 8), dtype=mx.int32)

        first_token = decoder.prefill(prompt)

        assert isinstance(first_token, int)

    def test_prefill_raises_when_cancelled_upfront(self):
        """An already-set event interrupts before the prefix forward, so a
        disconnected request releases the GPU instead of running prefill out."""
        import pytest

        from olmlx.engine.speculative import PrefillCancelled

        decoder = _make_decoder()
        prompt = mx.zeros((1, 8), dtype=mx.int32)
        cancel = threading.Event()
        cancel.set()

        with pytest.raises(PrefillCancelled):
            decoder.prefill(prompt, cancel_event=cancel)

    def test_prefill_skips_last_token_forward_when_cancelled_after_prefix(self):
        """A cancel that fires during the prefix forward must skip the
        second (last-token) forward — verifying the stated guarantee end-to-end
        through ``SelfSpeculativeDecoder.prefill``, not just the helper."""
        import pytest

        from olmlx.engine.speculative import PrefillCancelled

        decoder = _make_decoder()
        prompt = mx.zeros((1, 8), dtype=mx.int32)
        cancel = threading.Event()

        # Wrap the target to count forwards and set the event after the first
        # (prefix) pass — the last-token forward must never run. Plain proxy
        # (mirrors _CancelOnNthCall in test_speculative_prefill_cancel.py):
        # exposes ``.layers`` for make_prompt_cache and delegates the rest.
        inner = decoder._target
        calls: list[int] = []

        class _CancelAfterPrefix:
            def __init__(self):
                self.layers = inner.layers

            def __call__(self, input_ids, cache=None):
                calls.append(int(input_ids.shape[1]))
                out = inner(input_ids, cache=cache)
                cancel.set()
                return out

            def __getattr__(self, name):
                return getattr(inner, name)

        decoder._target = _CancelAfterPrefix()

        with pytest.raises(PrefillCancelled):
            decoder.prefill(prompt, cancel_event=cancel)

        assert calls == [7]  # prefix forward only (8 - 1 reserved last token)
