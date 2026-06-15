"""Tests for the Stage-2 M-/M+ serveability verifier."""

from __future__ import annotations

import pytest

from olmlx.proxy_tuning_pipeline.verify import assert_serveable_pair


class _FakeTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def _loader_for(mapping: dict[str, dict[str, int]]):
    def _load(path: str):
        return _FakeTokenizer(mapping[path])

    return _load


def test_assert_serveable_pair_passes_for_matching_pair():
    vocab = {tok: i for i, tok in enumerate(["a", "b", "c"])}
    loader = _loader_for({"m_minus": dict(vocab), "m_plus": dict(vocab)})
    # No raise == pass.
    assert_serveable_pair("m_minus", "m_plus", base_vocab_size=3, loader=loader)


def test_assert_serveable_pair_raises_on_token_mapping_diff():
    loader = _loader_for(
        {"m_minus": {"a": 0, "b": 1, "c": 2}, "m_plus": {"a": 0, "b": 2, "c": 1}}
    )
    with pytest.raises(ValueError, match="vocab"):
        assert_serveable_pair("m_minus", "m_plus", base_vocab_size=3, loader=loader)


def test_assert_serveable_pair_raises_on_base_vocab_mismatch():
    vocab = {"a": 0, "b": 1, "c": 2}
    loader = _loader_for({"m_minus": dict(vocab), "m_plus": dict(vocab)})
    with pytest.raises(ValueError, match="base"):
        assert_serveable_pair(
            "m_minus", "m_plus", base_vocab_size=151936, loader=loader
        )
