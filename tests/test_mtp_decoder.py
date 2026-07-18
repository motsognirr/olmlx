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


def test_mtp_draft_cache_growth_is_bounded_per_step():
    """The MTP draft cache must grow by exactly ``num_accepted`` positions
    per ``step()`` — one entry per committed token, keeping the draft's KV
    history and RoPE positions aligned with the committed sequence (#617).

    The pre-#617 arithmetic trimmed ``block_size + 1 - num_accepted`` on the
    draft cache (keeping only ``num_accepted - 1``), dropping the last
    accepted token's KV entry every step and compressing RoPE positions by
    one per step — cumulative acceptance-rate degradation on long runs.
    """
    decoder, _, _ = _make_mtp_decoder(block_size=2)
    decoder.prefill(mx.array([[1, 2, 3]], dtype=mx.int32))
    assert decoder._draft_cache is not None

    def _cache_offset() -> int:
        assert decoder._draft_cache is not None
        return decoder._draft_cache[0].offset

    baseline = _cache_offset()
    steps = 3
    total_accepted = 0
    for _ in range(steps):
        accepted, _ = decoder.step()
        total_accepted += len(accepted)

    expected = total_accepted
    actual = _cache_offset() - baseline
    assert actual == expected, (
        f"MTP draft cache offset grew by {actual} after {steps} steps "
        f"with {total_accepted} total accepted tokens; expected {expected}. "
        "Regression: the per-step draft trim no longer keeps exactly "
        "``num_accepted`` committed entries."
    )


def test_mtp_full_acceptance_aligns_draft_cache(monkeypatch):
    """On full acceptance the MTP draft cache is one entry short of the
    committed prefix, so ``step`` must run an align-forward to append the
    last accepted draft token's KV (#617)."""
    decoder, _, _ = _make_mtp_decoder(block_size=3)
    decoder.prefill(mx.array([[1, 2, 3]], dtype=mx.int32))
    assert decoder._draft_cache is not None
    baseline = decoder._draft_cache[0].offset

    monkeypatch.setattr(decoder, "_verify_greedy", lambda drafts, logits: [*drafts, 7])
    accepted, _ = decoder.step()
    assert len(accepted) == decoder._block_size + 1  # full acceptance
    grew = decoder._draft_cache[0].offset - baseline
    assert grew == len(accepted), (
        f"MTP draft cache grew by {grew} on full acceptance; expected "
        f"{len(accepted)} — the align-forward did not append the last "
        "accepted draft token's KV entry."
    )


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


def test_mtp_prefill_suppresses_gdn_capture_during_chunked_prefix(monkeypatch):
    """On a non-trimmable (GDN) target the capture buffer must be INACTIVE
    during the chunked prefix forward.

    ``_capturing_gdn_call`` appends per forward, so leaving the buffer active
    would pile every sub-chunk's GDN q/k/v/conv tensors into it and hold them
    live across all chunks — re-bloating exactly the memory the chunking
    bounds (#360), on the hybrid GDN targets that are MTP's primary use case.
    The buffer is re-enabled before the pass-2 single-token forward because
    ``step()`` relies on an active buffer (it only clears between verifies).
    """
    from olmlx.engine import spec_decoder_base as base_mod, speculative
    from olmlx.engine.mtp import decoder as decoder_mod

    monkeypatch.setattr(speculative, "_PREFILL_CHUNK", 4)
    monkeypatch.setattr(decoder_mod, "can_trim_prompt_cache", lambda _: False)
    monkeypatch.setattr(decoder_mod, "_HAS_GDN", True)
    monkeypatch.setattr(decoder_mod, "_find_gdn_class", lambda _m: object)

    state: dict[str, bool | None] = {"active": None}

    class _FakeBuffer:
        def clear(self):
            pass

    class _FakeCapture:
        @classmethod
        def for_model(cls, _model):
            return cls(), _FakeBuffer()

        def use_buffer(self, buf):
            state["active"] = buf is not None

        def close(self):
            pass

    # The capture is installed via ``SpecDecoderBase._install_gdn_capture``,
    # so the patch seam lives on the base module (#467).
    monkeypatch.setattr(base_mod, "GDNStateCapture", _FakeCapture)

    decoder, target, _ = _make_mtp_decoder()

    forwards: list[tuple[int, bool | None]] = []
    orig_call = type(target).__call__

    def recording_call(self, input_ids, cache=None):
        forwards.append((input_ids.shape[1], state["active"]))
        return orig_call(self, input_ids, cache=cache)

    monkeypatch.setattr(type(target), "__call__", recording_call)

    prompt = mx.arange(1, 21, dtype=mx.int32)[None, :]  # (1, 20)
    decoder.prefill(prompt)

    assert decoder._target_can_trim is False
    assert len(forwards) >= 2, forwards
    # Every prefix sub-chunk forward ran with the capture buffer suppressed.
    for _seq_len, active in forwards[:-1]:
        assert active is False, f"GDN capture active during prefix forward: {forwards}"
    # The final pass-2 (single-token) forward ran with the buffer re-enabled.
    last_seq, last_active = forwards[-1]
    assert last_seq == 1, forwards
    assert last_active is True, forwards


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
