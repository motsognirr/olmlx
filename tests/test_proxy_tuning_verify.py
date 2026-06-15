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


def _config_loader_for(sizes: dict[str, int]):
    def _load(path: str) -> int:
        return sizes[path]

    return _load


def test_assert_serveable_pair_passes_for_matching_pair():
    vocab = {tok: i for i, tok in enumerate(["a", "b", "c"])}
    loader = _loader_for({"m_minus": dict(vocab), "m_plus": dict(vocab)})
    config_loader = _config_loader_for({"m_minus": 3, "m_plus": 3})
    # No raise == pass.
    assert_serveable_pair(
        "m_minus",
        "m_plus",
        base_vocab_size=3,
        loader=loader,
        config_loader=config_loader,
    )


def test_assert_serveable_pair_raises_on_token_mapping_diff():
    loader = _loader_for(
        {"m_minus": {"a": 0, "b": 1, "c": 2}, "m_plus": {"a": 0, "b": 2, "c": 1}}
    )
    config_loader = _config_loader_for({"m_minus": 3, "m_plus": 3})
    with pytest.raises(ValueError, match="vocab"):
        assert_serveable_pair(
            "m_minus",
            "m_plus",
            base_vocab_size=3,
            loader=loader,
            config_loader=config_loader,
        )


def test_assert_serveable_pair_raises_on_base_vocab_mismatch():
    vocab = {"a": 0, "b": 1, "c": 2}
    loader = _loader_for({"m_minus": dict(vocab), "m_plus": dict(vocab)})
    config_loader = _config_loader_for({"m_minus": 3, "m_plus": 3})
    with pytest.raises(ValueError, match="base"):
        assert_serveable_pair(
            "m_minus",
            "m_plus",
            base_vocab_size=151936,
            loader=loader,
            config_loader=config_loader,
        )


def test_assert_serveable_pair_uses_config_vocab_not_tokenizer_length():
    # Real Qwen3: the tokenizer has 151669 entries, but config vocab_size is the
    # padded logits width 151936. The gate must compare the *config* width to the
    # base (151936) and pass — not the get_vocab() length (which would mismatch).
    vocab = {str(i): i for i in range(151669)}
    loader = _loader_for({"m_minus": dict(vocab), "m_plus": dict(vocab)})
    config_loader = _config_loader_for({"m_minus": 151936, "m_plus": 151936})
    assert_serveable_pair(
        "m_minus",
        "m_plus",
        base_vocab_size=151936,
        loader=loader,
        config_loader=config_loader,
    )
