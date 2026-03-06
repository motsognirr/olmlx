"""Tests for mlx_ollama.engine.template_caps."""

from unittest.mock import MagicMock

from mlx_ollama.engine.template_caps import TemplateCaps, detect_caps


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
        assert caps.supports_tools is True

    def test_thinking_in_lowercase(self):
        tok = MagicMock()
        tok.chat_template = "{{ messages }} Thinking block"
        caps = detect_caps(tok)
        assert caps.has_thinking_tags is True
