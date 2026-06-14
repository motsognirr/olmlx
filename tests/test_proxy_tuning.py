"""Tests for proxy-tuning decode mode (engine/proxy_tuning.py)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from olmlx.engine.proxy_tuning import (
    ProxyTuningDecoder,
    check_vocab_identity,
    combine_proxy_logits,
)


def test_combine_proxy_logits_basic():
    base = mx.array([1.0, 2.0, 3.0])
    expert = mx.array([0.0, 5.0, 0.0])
    antiexpert = mx.array([0.0, 1.0, 0.0])
    # base + alpha*(expert - antiexpert) with alpha=1.0
    out = combine_proxy_logits(base, expert, antiexpert, 1.0)
    assert out.tolist() == [1.0, 6.0, 3.0]


def test_combine_proxy_logits_alpha_scales_delta():
    base = mx.array([0.0, 0.0])
    expert = mx.array([0.0, 4.0])
    antiexpert = mx.array([0.0, 0.0])
    out = combine_proxy_logits(base, expert, antiexpert, 0.5)
    assert out.tolist() == [0.0, 2.0]


def test_combine_proxy_logits_alpha_zero_is_base():
    base = mx.array([7.0, -3.0])
    expert = mx.array([100.0, 100.0])
    antiexpert = mx.array([-100.0, -100.0])
    out = combine_proxy_logits(base, expert, antiexpert, 0.0)
    assert out.tolist() == [7.0, -3.0]


class _FakeTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def test_check_vocab_identity_passes_on_match():
    tok_a = _FakeTokenizer({"a": 0, "b": 1, "c": 2})
    tok_b = _FakeTokenizer({"a": 0, "b": 1, "c": 2})
    # No raise == pass.
    check_vocab_identity(tok_a, tok_b, reference_label="base", other_label="expert")


def test_check_vocab_identity_raises_on_token_mapping_diff():
    # Same SIZE, different mapping — the exact case vocab_size-only checks miss.
    tok_a = _FakeTokenizer({"a": 0, "b": 1, "c": 2})
    tok_b = _FakeTokenizer({"a": 0, "b": 2, "c": 1})
    with pytest.raises(ValueError, match="vocab"):
        check_vocab_identity(tok_a, tok_b, reference_label="base", other_label="expert")


def test_check_vocab_identity_warns_when_unavailable(caplog):
    class _NoVocab:
        pass

    # Missing get_vocab() -> warn-and-return (loader's vocab_size check is the floor).
    with caplog.at_level("WARNING"):
        check_vocab_identity(
            _NoVocab(), _NoVocab(), reference_label="base", other_label="expert"
        )
    assert any("vocab" in r.message.lower() for r in caplog.records)


class _StubModel(nn.Module):
    """Returns a fixed last-position logit vector on every forward.

    ``make_cache`` returns ``[]`` so ``make_prompt_cache(model)`` produces an
    empty cache the stub ignores — keeps the test off real attention/KV code.
    The decoder only reads ``out[0, -1, :]``, so broadcasting the fixed vector
    across all positions is sufficient and deterministic.
    """

    def __init__(self, vocab_size: int, logit_vec: mx.array):
        super().__init__()
        self._vocab_size = vocab_size
        self._logit_vec = logit_vec
        self.calls = 0

    def make_cache(self) -> list:
        return []

    def __call__(self, tokens: mx.array, cache: Any = None) -> mx.array:
        self.calls += 1
        seq = tokens.shape[1]
        return mx.broadcast_to(
            self._logit_vec.reshape(1, 1, -1), (1, seq, self._vocab_size)
        )


def _make_decoder(vocab=4, base=None, expert=None, anti=None, alpha=1.0):
    base = base if base is not None else mx.zeros((vocab,))
    expert = expert if expert is not None else mx.zeros((vocab,))
    anti = anti if anti is not None else mx.zeros((vocab,))
    return ProxyTuningDecoder(
        base_model=_StubModel(vocab, base),
        expert_model=_StubModel(vocab, expert),
        antiexpert_model=_StubModel(vocab, anti),
        alpha=alpha,
    )


def test_decoder_construction_sets_target_and_alpha():
    dec = _make_decoder(alpha=2.0)
    assert dec._alpha == 2.0
    # _target must be the base model (base teardown reference; never patched).
    assert dec._target is dec._base
    assert dec._patched is False
    assert dec._bound is False
    assert dec._capture is None


def test_reset_clears_caches_and_pending():
    dec = _make_decoder()
    dec._base_cache = ["x"]
    dec._expert_cache = ["y"]
    dec._antiexpert_cache = ["z"]
    dec._pending_token = 3
    dec.reset()
    assert dec._base_cache is None
    assert dec._expert_cache is None
    assert dec._antiexpert_cache is None
    assert dec._pending_token is None


def test_prefill_returns_combined_argmax():
    # base favors idx 0, expert favors idx 2, antiexpert favors idx 1.
    # combined = base + (expert - antiexpert):
    #   idx0: 3 + (0-0) = 3
    #   idx1: 0 + (0-5) = -5
    #   idx2: 0 + (5-0) = 5   <- argmax
    base = mx.array([3.0, 0.0, 0.0])
    expert = mx.array([0.0, 0.0, 5.0])
    anti = mx.array([0.0, 5.0, 0.0])
    dec = _make_decoder(vocab=3, base=base, expert=expert, anti=anti, alpha=1.0)
    prompt = mx.array([[7, 8, 9]])  # any 3-token prompt
    first = dec.prefill(prompt)
    assert first == 2
    assert dec._pending_token == 2
    # All three caches must be populated.
    assert dec._base_cache is not None
    assert dec._expert_cache is not None
    assert dec._antiexpert_cache is not None


def test_prefill_single_token_prompt():
    base = mx.array([0.0, 9.0])
    dec = _make_decoder(vocab=2, base=base)
    first = dec.prefill(mx.array([[5]]))
    assert first == 1


def test_step_returns_combined_argmax():
    base = mx.array([3.0, 0.0, 0.0])
    expert = mx.array([0.0, 0.0, 5.0])
    anti = mx.array([0.0, 5.0, 0.0])
    dec = _make_decoder(vocab=3, base=base, expert=expert, anti=anti, alpha=1.0)
    dec.prefill(mx.array([[7, 8, 9]]))
    accepted, num_draft = dec.step()
    assert accepted == [2]
    assert num_draft == 0
    assert dec._pending_token == 2
    assert dec._stats_steps == 1


def test_step_before_prefill_raises():
    dec = _make_decoder()
    with pytest.raises(RuntimeError, match="prefill"):
        dec.step()


def test_alpha_changes_winner():
    # With alpha=0 the base wins (idx0); with alpha=1 the delta flips it to idx2.
    base = mx.array([1.0, 0.0, 0.0])
    expert = mx.array([0.0, 0.0, 10.0])
    anti = mx.array([0.0, 0.0, 0.0])
    dec0 = _make_decoder(vocab=3, base=base, expert=expert, anti=anti, alpha=0.0)
    assert dec0.prefill(mx.array([[1, 2]])) == 0
    dec1 = _make_decoder(vocab=3, base=base, expert=expert, anti=anti, alpha=1.0)
    assert dec1.prefill(mx.array([[1, 2]])) == 2


def test_multi_step_decode_advances_and_counts():
    # Fixed per-model logits → every step yields the same winning index, but the
    # caches must advance and _stats_steps must increment across calls.
    base = mx.array([3.0, 0.0, 0.0])
    expert = mx.array([0.0, 0.0, 5.0])
    anti = mx.array([0.0, 5.0, 0.0])
    dec = _make_decoder(vocab=3, base=base, expert=expert, anti=anti, alpha=1.0)
    dec.prefill(mx.array([[7, 8, 9]]))
    base_calls_after_prefill = dec._base.calls
    out = [dec.step() for _ in range(3)]
    assert out == [([2], 0), ([2], 0), ([2], 0)]
    assert dec._stats_steps == 3
    # Each step issues exactly one forward per model.
    assert dec._base.calls == base_calls_after_prefill + 3


def test_stats_summary_exposes_alpha():
    dec = _make_decoder(alpha=1.5)
    summary = dec.stats_summary()
    assert summary["alpha"] == 1.5


def test_strategy_label_is_proxy_tuning():
    from olmlx.engine.spec_decoder_base import _strategy_label_for

    assert _strategy_label_for(ProxyTuningDecoder) == "proxy_tuning"
