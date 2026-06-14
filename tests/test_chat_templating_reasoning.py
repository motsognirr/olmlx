"""reasoning_effort plumbing in olmlx.engine.chat_templating."""

from unittest.mock import MagicMock

from olmlx.engine.chat_templating import apply_chat_template_text
from olmlx.engine.template_caps import TemplateCaps


def _capturing_tokenizer():
    tok = MagicMock()
    captured: dict = {}

    def _apply(messages, **kwargs):
        captured.update(kwargs)
        return "PROMPT"

    tok.apply_chat_template = _apply
    return tok, captured


_MSGS = [{"role": "user", "content": "hi"}]


class TestReasoningEffortPlumbing:
    def test_passes_reasoning_effort_when_supported(self):
        tok, captured = _capturing_tokenizer()
        caps = TemplateCaps(supports_reasoning_effort=True)
        apply_chat_template_text(tok, _MSGS, None, caps, reasoning_effort="low")
        assert captured.get("reasoning_effort") == "low"

    def test_omits_when_template_unsupported(self):
        tok, captured = _capturing_tokenizer()
        caps = TemplateCaps(supports_reasoning_effort=False)
        apply_chat_template_text(tok, _MSGS, None, caps, reasoning_effort="low")
        assert "reasoning_effort" not in captured

    def test_omits_when_none(self):
        tok, captured = _capturing_tokenizer()
        caps = TemplateCaps(supports_reasoning_effort=True)
        apply_chat_template_text(tok, _MSGS, None, caps, reasoning_effort=None)
        assert "reasoning_effort" not in captured
