"""Tests for olmlx.engine.template_caps."""

from unittest.mock import MagicMock

from olmlx.engine.template_caps import TemplateCaps, detect_caps


class TestTemplateCaps:
    def test_defaults(self):
        caps = TemplateCaps()
        assert caps.supports_tools is False
        assert caps.supports_enable_thinking is False
        assert caps.has_thinking_tags is False

    def test_custom_values(self):
        caps = TemplateCaps(
            supports_tools=True,
            supports_enable_thinking=True,
            has_thinking_tags=True,
        )
        assert caps.supports_tools is True
        assert caps.supports_enable_thinking is True
        assert caps.has_thinking_tags is True


class TestDetectCaps:
    def test_no_template(self):
        tok = MagicMock(spec=[])  # no chat_template attribute
        caps = detect_caps(tok)
        assert caps.supports_tools is False
        assert caps.supports_enable_thinking is False
        assert caps.has_thinking_tags is False

    def test_none_template(self):
        tok = MagicMock()
        tok.chat_template = None
        caps = detect_caps(tok)
        assert caps.supports_tools is False

    def test_basic_template(self):
        tok = MagicMock()
        tok.chat_template = "{{ messages }}"
        caps = detect_caps(tok)
        assert caps.supports_tools is False
        assert caps.supports_enable_thinking is False
        assert caps.has_thinking_tags is False

    def test_tools_template(self):
        tok = MagicMock()
        tok.chat_template = "{{ messages }}{% if tools %}{{ tools }}{% endif %}"
        caps = detect_caps(tok)
        assert caps.supports_tools is True
        assert caps.supports_enable_thinking is False

    def test_thinking_template(self):
        tok = MagicMock()
        tok.chat_template = "{% if enable_thinking %}<think>{% endif %}{{ messages }}"
        caps = detect_caps(tok)
        assert caps.supports_enable_thinking is True
        assert caps.has_thinking_tags is True

    def test_qwen_style_template(self):
        tok = MagicMock()
        tok.chat_template = (
            "{{ messages }}{% if tools %}{{ tools }}{% endif %}"
            "{% if enable_thinking %}<think>{% endif %}"
        )
        caps = detect_caps(tok)
        assert caps.supports_tools is True
        assert caps.supports_enable_thinking is True
        assert caps.has_thinking_tags is True

    def test_list_template(self):
        tok = MagicMock()
        tok.chat_template = [
            {"name": "default", "template": "{{ messages }}"},
            {"name": "tool_use", "template": "{{ tools }}"},
        ]
        caps = detect_caps(tok)
        assert caps.supports_tools is True

    def test_list_template_empty(self):
        tok = MagicMock()
        tok.chat_template = []
        caps = detect_caps(tok)
        assert caps.supports_tools is False

    def test_list_template_with_non_dicts(self):
        tok = MagicMock()
        tok.chat_template = [
            {"template": "{{ messages }} with tools"},
            "not a dict",
        ]
        caps = detect_caps(tok)
        # "tools" is static text, not a Jinja2 variable — correctly False
        assert caps.supports_tools is False

    def test_thinking_in_lowercase(self):
        tok = MagicMock()
        tok.chat_template = "{{ messages }} Thinking block"
        caps = detect_caps(tok)
        assert caps.has_thinking_tags is True

    def test_tools_variable_detected(self):
        """Jinja2 variable `tools` in template → supports_tools=True."""
        tok = MagicMock()
        tok.chat_template = "{% if tools %}{{ tools }}{% endif %}"
        caps = detect_caps(tok)
        assert caps.supports_tools is True

    def test_tools_in_comment_not_detected(self):
        """The word 'tools' only in a Jinja2 comment → supports_tools=False."""
        tok = MagicMock()
        tok.chat_template = "{# tools #}{{ messages }}"
        caps = detect_caps(tok)
        assert caps.supports_tools is False

    def test_tools_in_static_text_not_detected(self):
        """The word 'tools' only in static text → supports_tools=False."""
        tok = MagicMock()
        tok.chat_template = "The word tools appears here. {{ messages }}"
        caps = detect_caps(tok)
        assert caps.supports_tools is False

    def test_enable_thinking_variable_detected(self):
        """Jinja2 variable `enable_thinking` → supports_enable_thinking=True."""
        tok = MagicMock()
        tok.chat_template = "{% if enable_thinking %}think{% endif %}{{ messages }}"
        caps = detect_caps(tok)
        assert caps.supports_enable_thinking is True

    def test_enable_thinking_in_comment_not_detected(self):
        """The word 'enable_thinking' only in a comment → supports_enable_thinking=False."""
        tok = MagicMock()
        tok.chat_template = "{# enable_thinking #}{{ messages }}"
        caps = detect_caps(tok)
        assert caps.supports_enable_thinking is False

    def test_malformed_template_falls_back_to_substring(self):
        """Invalid Jinja2 → fall back to substring matching."""
        tok = MagicMock()
        tok.chat_template = "{% if tools %}{{ tools }}{%"  # unclosed block
        caps = detect_caps(tok)
        # Substring fallback finds "tools"
        assert caps.supports_tools is True

    def test_list_template_jinja_parsing(self):
        """List-of-dicts format works with AST-based parsing."""
        tok = MagicMock()
        tok.chat_template = [
            {"name": "default", "template": "{{ messages }}"},
            {"name": "tool_use", "template": "{% if tools %}{{ tools }}{% endif %}"},
        ]
        caps = detect_caps(tok)
        assert caps.supports_tools is True

    def test_uses_tool_responses_detected(self):
        """Template accessing message.tool_responses → uses_tool_responses=True."""
        tok = MagicMock()
        tok.chat_template = (
            "{% for msg in messages %}"
            "{% if msg.tool_responses %}{{ msg.tool_responses }}{% endif %}"
            "{% endfor %}"
        )
        caps = detect_caps(tok)
        assert caps.uses_tool_responses is True

    def test_uses_tool_responses_false_by_default(self):
        """Standard template without tool_responses → uses_tool_responses=False."""
        tok = MagicMock()
        tok.chat_template = "{{ messages }}{% if tools %}{{ tools }}{% endif %}"
        caps = detect_caps(tok)
        assert caps.uses_tool_responses is False

    def test_uses_tool_responses_false_for_comment(self):
        """A Jinja comment mentioning tool_responses must not trigger the flag."""
        tok = MagicMock()
        tok.chat_template = (
            "{# This model does not support tool_responses #}{{ messages }}"
        )
        caps = detect_caps(tok)
        assert caps.uses_tool_responses is False

    def test_uses_tool_responses_defaults(self):
        caps = TemplateCaps()
        assert caps.uses_tool_responses is False

    def test_uses_tool_responses_bracket_notation(self):
        """Gemma 4 template uses message['tool_responses'] (bracket notation)."""
        tok = MagicMock()
        tok.chat_template = (
            "{% for msg in messages %}"
            "{% if msg['tool_responses'] %}{{ msg['tool_responses'] }}{% endif %}"
            "{% endfor %}"
        )
        caps = detect_caps(tok)
        assert caps.uses_tool_responses is True
