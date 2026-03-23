"""Tests for gpt-oss channel token parsing and streaming filter."""

import asyncio
from unittest.mock import MagicMock

from olmlx.engine.template_caps import TemplateCaps, detect_caps
from olmlx.engine.tool_parser import parse_model_output


# ---------------------------------------------------------------------------
# TemplateCaps detection
# ---------------------------------------------------------------------------


class TestChannelFormatDetection:
    def test_gpt_oss_template_detected(self):
        """Templates with <|channel|> should set has_channel_format=True."""
        tok = MagicMock()
        tok.chat_template = (
            "<|start|>system<|message|>You are helpful<|end|>"
            "{% for m in messages %}<|start|>{{ m.role }}<|channel|>final<|message|>{{ m.content }}<|end|>{% endfor %}"
        )
        caps = detect_caps(tok)
        assert caps.has_channel_format is True

    def test_normal_template_not_detected(self):
        """Templates without <|channel|> should have has_channel_format=False."""
        tok = MagicMock()
        tok.chat_template = (
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
        )
        caps = detect_caps(tok)
        assert caps.has_channel_format is False

    def test_no_template(self):
        tok = MagicMock(spec=[])
        caps = detect_caps(tok)
        assert caps.has_channel_format is False

    def test_defaults(self):
        caps = TemplateCaps()
        assert caps.has_channel_format is False


# ---------------------------------------------------------------------------
# Buffered parsing (_parse_gpt_oss_channels via parse_model_output)
# ---------------------------------------------------------------------------


class TestParseGptOssChannels:
    def test_analysis_and_final(self):
        """Analysis channel -> thinking, final channel -> visible text."""
        text = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Let me think about this carefully."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "The answer is 42."
            "<|end|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=False)
        assert thinking == "Let me think about this carefully."
        assert visible == "The answer is 42."
        assert tools == []

    def test_final_only(self):
        """Output with only final channel should produce visible text, no thinking."""
        text = "<|start|>assistant<|channel|>final<|message|>Hello world!<|end|>"
        thinking, visible, tools = parse_model_output(text, has_tools=False)
        assert thinking == ""
        assert visible == "Hello world!"

    def test_analysis_only_falls_back_to_visible(self):
        """Output with only analysis channel should promote to visible text."""
        text = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Hmm interesting question."
            "<|end|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=False)
        assert thinking == ""
        assert visible == "Hmm interesting question."

    def test_multiple_final_blocks(self):
        """Multiple final blocks should be concatenated."""
        text = (
            "<|start|>assistant<|channel|>final<|message|>Part 1.<|end|>"
            "<|start|>assistant<|channel|>final<|message|> Part 2.<|end|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=False)
        assert visible == "Part 1. Part 2."

    def test_return_token_as_end(self):
        """<|return|> should also terminate a block."""
        text = "<|start|>assistant<|channel|>final<|message|>Done.<|return|>"
        thinking, visible, tools = parse_model_output(text, has_tools=False)
        assert visible == "Done."

    def test_no_channel_tokens_passthrough(self):
        """Text without gpt-oss tokens should pass through unchanged."""
        text = "Just a normal response without any special tokens."
        thinking, visible, tools = parse_model_output(text, has_tools=False)
        assert visible == text
        assert thinking == ""

    def test_mixed_with_think_tags(self):
        """gpt-oss channels should be preferred over <think> tags when present."""
        text = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Deep thought here."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "My answer."
            "<|end|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=False)
        assert thinking == "Deep thought here."
        assert visible == "My answer."

    def test_whitespace_in_channel_type(self):
        """Channel type may have trailing whitespace (e.g. 'analysis ')."""
        text = (
            "<|start|>assistant<|channel|>analysis <|message|>"
            "Thinking."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "Answer."
            "<|end|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=False)
        assert thinking == "Thinking."
        assert visible == "Answer."

    def test_commentary_channel_with_tool_call(self):
        """Commentary channel should be parsed as a tool call."""
        text = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "I need to search."
            "<|end|>"
            "<|start|>assistant to=functions.search<|channel|>commentary json<|message|>"
            '{"query": "test"}'
            "<|call|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=True)
        assert thinking == "I need to search."
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["input"] == {"query": "test"}


# ---------------------------------------------------------------------------
# Streaming filter
# ---------------------------------------------------------------------------


class TestGptOssStreamFilter:
    def _make_token(self, text):
        """Create a mock StreamToken."""
        from olmlx.utils.streaming import StreamToken

        return StreamToken(
            text=text,
            token=None,
            prompt_tokens=0,
            generation_tokens=0,
            prompt_tps=0.0,
            generation_tps=0.0,
        )

    def _run_filter(self, token_texts):
        """Run the streaming filter on a list of token text strings, return yielded texts."""
        from olmlx.engine.inference import _gpt_oss_filter

        async def mock_stream():
            for t in token_texts:
                yield self._make_token(t)

        async def collect():
            result = []
            async for tok in _gpt_oss_filter(mock_stream()):
                result.append(tok.text)
            return result

        return asyncio.run(collect())

    def test_final_channel_passes_through(self):
        """Only text from final channel should be yielded."""
        tokens = [
            "<|start|>",
            "assistant",
            "<|channel|>",
            "analysis",
            "<|message|>",
            "thinking",
            " here",
            "<|end|>",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "visible",
            " text",
            "<|end|>",
        ]
        result = self._run_filter(tokens)
        assert result == ["visible", " text"]

    def test_only_analysis_falls_back(self):
        """If only analysis channel, analysis content should be yielded as fallback."""
        tokens = [
            "<|start|>",
            "assistant",
            "<|channel|>",
            "analysis",
            "<|message|>",
            "just",
            " thinking",
            "<|end|>",
        ]
        result = self._run_filter(tokens)
        assert result == ["just", " thinking"]

    def test_no_channel_tokens_passthrough(self):
        """Plain text without channel tokens should pass through."""
        tokens = ["Hello", " world", "!"]
        result = self._run_filter(tokens)
        assert result == ["Hello", " world", "!"]

    def test_return_token_ends_block(self):
        """<|return|> should end a block like <|end|>."""
        tokens = [
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "answer",
            "<|return|>",
        ]
        result = self._run_filter(tokens)
        assert result == ["answer"]

    def test_commentary_channel_suppressed(self):
        """Commentary channel content should be suppressed."""
        tokens = [
            "<|start|>",
            "assistant",
            "<|channel|>",
            "commentary",
            "<|message|>",
            '{"tool": "call"}',
            "<|call|>",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "done",
            "<|end|>",
        ]
        result = self._run_filter(tokens)
        assert result == ["done"]
