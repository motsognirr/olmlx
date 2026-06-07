"""VLM vocab resolution for grammar bitmask sizing (#429)."""

import types

from olmlx.engine.inference import _resolve_model_vocab_size


class _Weight:
    def __init__(self, rows):
        self.shape = (rows, 4096)


class _Head:
    def __init__(self, rows):
        self.weight = _Weight(rows)


def _fake_lm(model):
    # Minimal stand-in: _resolve_model_vocab_size only reads lm.model.
    return types.SimpleNamespace(model=model)


def test_resolves_vocab_under_language_model_for_vlm():
    # VLM layout: lm_head lives under model.language_model, not model/model.model.
    language_model = types.SimpleNamespace(lm_head=_Head(151936))
    vlm = types.SimpleNamespace(language_model=language_model)
    assert _resolve_model_vocab_size(_fake_lm(vlm)) == 151936


def test_text_model_vocab_still_resolves():
    text = types.SimpleNamespace(lm_head=_Head(32000), model=None)
    assert _resolve_model_vocab_size(_fake_lm(text)) == 32000
