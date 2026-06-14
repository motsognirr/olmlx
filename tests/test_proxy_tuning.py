"""Tests for proxy-tuning decode mode (engine/proxy_tuning.py)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from olmlx.engine.proxy_tuning import check_vocab_identity, combine_proxy_logits


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
        check_vocab_identity(
            tok_a, tok_b, reference_label="base", other_label="expert"
        )


def test_check_vocab_identity_warns_when_unavailable(caplog):
    class _NoVocab:
        pass

    # Missing get_vocab() -> warn-and-return (loader's vocab_size check is the floor).
    with caplog.at_level("WARNING"):
        check_vocab_identity(
            _NoVocab(), _NoVocab(), reference_label="base", other_label="expert"
        )
    assert any("vocab" in r.message.lower() for r in caplog.records)
