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
        """Commentary channel should be parsed as a tool call (harmony format).

        The correct harmony format has to=functions.* AFTER <|channel|>:
        <|start|>assistant<|channel|>commentary to=functions.search<|constrain|>json<|message|>{...}<|call|>
        """
        text = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "I need to search."
            "<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.search<|constrain|>json<|message|>"
            '{"query": "test"}'
            "<|call|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=True)
        assert thinking == "I need to search."
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["input"] == {"query": "test"}

    def test_tool_call_without_constrain(self):
        """Tool call without <|constrain|> should still parse correctly."""
        text = (
            "<|start|>assistant<|channel|>commentary to=functions.get_weather<|message|>"
            '{"location": "Paris"}'
            "<|call|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=True)
        assert thinking == ""
        assert visible == ""
        assert len(tools) == 1
        assert tools[0]["name"] == "get_weather"
        assert tools[0]["input"] == {"location": "Paris"}

    def test_multiple_tool_calls(self):
        """Multiple tool calls should all be parsed."""
        text = (
            "<|start|>assistant<|channel|>commentary to=functions.search<|message|>"
            '{"query": "weather"}'
            "<|call|>"
            "<|start|>assistant<|channel|>commentary to=functions.get_location<|message|>"
            "{}"
            "<|call|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=True)
        assert len(tools) == 2
        assert tools[0]["name"] == "search"
        assert tools[0]["input"] == {"query": "weather"}
        assert tools[1]["name"] == "get_location"
        assert tools[1]["input"] == {}

    def test_tool_call_with_complex_args(self):
        """Tool call with nested objects and arrays should parse correctly."""
        text = (
            "<|start|>assistant<|channel|>commentary to=functions.search<|message|>"
            '{"query": "restaurants", "filters": {"cuisine": "italian", "price": ["$", "$$"]}, "limit": 5}'
            "<|call|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=True)
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["input"] == {
            "query": "restaurants",
            "filters": {"cuisine": "italian", "price": ["$", "$$"]},
            "limit": 5,
        }

    def test_tool_call_without_tools_flag(self):
        """Tool calls should be ignored when has_tools=False.

        When tools are not provided, commentary channel content is discarded
        (it's metadata, not visible text). The content doesn't appear in
        visible text because it's not meant for end users.
        """
        text = (
            "<|start|>assistant<|channel|>commentary to=functions.search<|message|>"
            '{"query": "test"}'
            "<|call|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=False)
        assert thinking == ""
        assert visible == ""  # Commentary content is discarded when has_tools=False
        assert tools == []

    def test_tool_call_with_return_token(self):
        """Tool call ending with <|return|> should work (end-of-string case)."""
        text = (
            "<|start|>assistant<|channel|>commentary to=functions.search<|message|>"
            '{"query": "test"}'
            "<|return|>"  # mlx-lm strips EOS, using return as terminator
        )
        thinking, visible, tools = parse_model_output(text, has_tools=True)
        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    def test_tool_call_name_extraction(self):
        """Tool name should be correctly extracted from to=functions.XYZ format."""
        text = (
            "<|start|>assistant<|channel|>commentary to=functions.get_current_weather<|message|>"
            '{"location": "NYC"}'
            "<|call|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=True)
        assert tools[0]["name"] == "get_current_weather"

    def test_mixed_final_and_tool_call(self):
        """Final channel content and tool calls should coexist."""
        text = (
            "<|start|>assistant<|channel|>commentary to=functions.search<|message|>"
            '{"query": "test"}'
            "<|call|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "I found some results!"
            "<|end|>"
        )
        thinking, visible, tools = parse_model_output(text, has_tools=True)
        assert visible == "I found some results!"
        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    def test_final_without_terminal_end_marker(self):
        """Final channel without <|end|> or <|return|> should still be parsed.

        mlx-lm strips EOS tokens, so the last block may end at end-of-string.
        """
        text = "<|start|>assistant<|channel|>final<|message|>Just a final message"
        thinking, visible, tools = parse_model_output(text, has_tools=True)
        assert thinking == ""
        assert visible == "Just a final message"
        assert tools == []

    def test_tool_call_without_terminal_marker(self):
        """Tool call without <|call|> terminator should still be parsed.

        When EOS is stripped, tool calls may end at end-of-string.
        """
        text = (
            "<|start|>assistant<|channel|>commentary to=functions.search<|message|>"
            '{"query": "test"}'
        )
        thinking, visible, tools = parse_model_output(text, has_tools=True)
        assert thinking == ""
        assert visible == ""
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
