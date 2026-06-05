"""Tests for MTPDecoder: prefill / step / reset protocol over an MTP draft.

Reuses the synthetic target harness from test_eagle.py
(_SimpleAttn, _SimpleLayer, _Inner, _SyntheticTarget) — they are
EAGLE-independent and exercise the same _patch_model / trim_prompt_cache
paths that MTPDecoder relies on.
"""

import inspect
import threading

import mlx.core as mx
import pytest

from olmlx.engine.mtp.decoder import MTPDecoder
from olmlx.engine.mtp.draft_model import MTPConfig, MTPDraftModel
from tests.test_eagle import _SyntheticTarget


def _make_mtp_decoder(vocab=64, hidden=32, num_layers=3, block_size=2):
    target = _SyntheticTarget(vocab=vocab, hidden=hidden, num_layers=num_layers)
    cfg = MTPConfig(
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=hidden // 2,
        rms_norm_eps=1e-6,
        vocab_size=vocab,
        max_position_embeddings=512,
        full_attention_interval=4,
        block_size=block_size,
    )
    draft = MTPDraftModel(cfg)
    mx.eval(draft.parameters())
    decoder = MTPDecoder(target, draft, block_size=block_size)
    return decoder, target, draft


def test_mtp_decoder_protocol_surface():
    for name in ("prefill", "step", "reset", "stats_summary", "close"):
        assert callable(getattr(MTPDecoder, name)), name
    assert "cancel_event" in inspect.signature(MTPDecoder.prefill).parameters


def test_mtp_decoder_block_size_validation():
    target = _SyntheticTarget()
    cfg = MTPConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=512,
        full_attention_interval=4,
        block_size=2,
    )
    try:
        MTPDecoder(target, MTPDraftModel(cfg), block_size=0)
        assert False, "expected ValueError for block_size=0"
    except ValueError:
        pass


def test_mtp_prefill_then_step_produces_accepted_tokens():
    decoder, _, _ = _make_mtp_decoder()
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    first = decoder.prefill(prompt)
    assert isinstance(first, int)
    accepted, num_drafts = decoder.step()
    assert len(accepted) == num_drafts + 1
    stats = decoder.stats_summary()
    assert stats["steps"] == 1 and stats["block_size"] == 2


def test_mtp_step_before_prefill_raises():
    decoder, _, _ = _make_mtp_decoder()
    try:
        decoder.step()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_mtp_prefill_subchunks_long_prefix(monkeypatch):
    """Pass-1 of prefill must fill the KV cache in sub-chunks of at most
    ``_PREFILL_CHUNK`` tokens, never a single forward over the whole prefix.

    A single full-prefix forward materialises ``lm_head`` over every position
    (a ``[1, seq-1, vocab]`` tensor) which OOMs Metal's single-buffer limit on
    long agentic prompts (#360). Sub-chunking bounds peak activation memory.
    """
    from olmlx.engine import speculative

    monkeypatch.setattr(speculative, "_PREFILL_CHUNK", 4)

    decoder, target, _ = _make_mtp_decoder()

    seq_lens: list[int] = []
    orig_call = type(target).__call__

    def recording_call(self, input_ids, cache=None):
        seq_lens.append(input_ids.shape[1])
        return orig_call(self, input_ids, cache=cache)

    monkeypatch.setattr(type(target), "__call__", recording_call)

    prompt = mx.arange(1, 21, dtype=mx.int32)[None, :]  # (1, 20)
    decoder.prefill(prompt)

    # No single target forward may span more than one sub-chunk's worth of
    # prefix tokens; the trailing single-token pass-2 forward is allowed.
    assert seq_lens, "target was never called"
    assert max(seq_lens) <= 4, f"un-chunked prefix forward: {seq_lens}"
    assert seq_lens[-1] == 1, f"pass-2 should be a single token: {seq_lens}"


def test_mtp_prefill_honors_cancel_event():
    """A cancel_event set before prefill aborts with PrefillCancelled and
    leaves the decoder reset (caches dropped, target un-patched)."""
    from olmlx.engine.speculative import PrefillCancelled

    decoder, _, _ = _make_mtp_decoder()
    cancel = threading.Event()
    cancel.set()

    prompt = mx.arange(1, 21, dtype=mx.int32)[None, :]
    with pytest.raises(PrefillCancelled):
        decoder.prefill(prompt, cancel_event=cancel)

    assert decoder._target_cache is None
    assert decoder._patched is False
