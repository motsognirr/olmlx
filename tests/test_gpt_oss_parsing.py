"""Tests for gpt-oss channel token parsing and streaming filter."""

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
    """``_GptOssChannelFilter.feed`` returns ``(thinking, content)`` per token.

    Analysis-channel text streams live on the thinking side (#713) — the
    engine yields it as structured ``{"thinking": ...}`` chunks, never as
    in-band ``<think>`` markup a reasoning trace could break out of.
    """

    @staticmethod
    def _run_filter(token_texts):
        """Feed token texts through the filter; return the non-empty
        ``(channel, text)`` outputs in order."""
        from olmlx.engine.logits_processors import _GptOssChannelFilter

        filt = _GptOssChannelFilter()
        out = []
        for t in token_texts:
            thinking, content = filt.feed(t)
            if thinking:
                out.append(("thinking", thinking))
            if content:
                out.append(("content", content))
        return out

    def test_analysis_streams_as_thinking_then_final_as_content(self):
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
        assert self._run_filter(tokens) == [
            ("thinking", "thinking"),
            ("thinking", " here"),
            ("content", "visible"),
            ("content", " text"),
        ]

    def test_think_close_tag_in_analysis_stays_thinking(self):
        """Reasoning that mentions ``</think>`` must not leak into content."""
        tokens = [
            "<|channel|>",
            "analysis",
            "<|message|>",
            "The tag </think>",
            " ends reasoning.",
            "<|end|>",
            "<|channel|>",
            "final",
            "<|message|>",
            "answer",
        ]
        assert self._run_filter(tokens) == [
            ("thinking", "The tag </think>"),
            ("thinking", " ends reasoning."),
            ("content", "answer"),
        ]

    def test_analysis_only_streams_as_thinking(self):
        """Analysis with no final channel is still thinking — it was already
        streamed live, so it can't be promoted to content afterwards."""
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
        assert self._run_filter(tokens) == [
            ("thinking", "just"),
            ("thinking", " thinking"),
        ]

    def test_analysis_after_final_is_thinking(self):
        tokens = [
            "<|channel|>",
            "final",
            "<|message|>",
            "answer",
            "<|end|>",
            "<|channel|>",
            "analysis",
            "<|message|>",
            "late",
            "<|end|>",
        ]
        assert self._run_filter(tokens) == [
            ("content", "answer"),
            ("thinking", "late"),
        ]

    def test_no_channel_tokens_passthrough(self):
        """Plain text without channel tokens should pass through."""
        assert self._run_filter(["Hello", " world", "!"]) == [
            ("content", "Hello"),
            ("content", " world"),
            ("content", "!"),
        ]

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
            "trailing",
        ]
        assert self._run_filter(tokens) == [("content", "answer")]

    def test_commentary_tool_call_suppressed(self):
        """Commentary addressed to a function is a tool call, not text."""
        tokens = [
            "<|start|>",
            "assistant",
            "<|channel|>",
            "commentary",
            " to",
            "=functions",
            ".get_weather",
            " ",
            "<|constrain|>",
            "json",
            "<|message|>",
            '{"city": "Paris"}',
            "<|call|>",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "done",
            "<|end|>",
        ]
        assert self._run_filter(tokens) == [("content", "done")]

    def test_recipientless_commentary_is_visible_preamble(self):
        """Commentary without ``to=functions.X`` is user-visible preamble
        (#621), same as the non-streaming parser."""
        tokens = [
            "<|start|>",
            "assistant",
            "<|channel|>",
            "commentary",
            "<|message|>",
            "I will now",
            " update the files.",
            "<|end|>",
        ]
        assert self._run_filter(tokens) == [
            ("content", "I will now"),
            ("content", " update the files."),
        ]

    def test_full_text_accumulates_all_tokens(self):
        from olmlx.engine.logits_processors import _GptOssChannelFilter

        filt = _GptOssChannelFilter()
        for t in ["<|channel|>", "final", "<|message|>", "hi"]:
            filt.feed(t)
        assert filt.get_full_text() == "<|channel|>final<|message|>hi"

    def test_streamed_split_matches_non_streaming_parse(self):
        """Streamed thinking/content must match ``_parse_gpt_oss_channels`` on
        the raw text for the normal analysis + preamble + final shape."""
        from olmlx.engine.tool_parser import _parse_gpt_oss_channels

        tokens = [
            "<|start|>",
            "assistant",
            "<|channel|>",
            "analysis",
            "<|message|>",
            "7 times 8",
            " is 56.",
            "<|end|>",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "The answer",
            " is 56.",
            "<|return|>",
        ]
        out = self._run_filter(tokens)
        thinking = "".join(t for ch, t in out if ch == "thinking")
        content = "".join(t for ch, t in out if ch == "content")
        expected = _parse_gpt_oss_channels("".join(tokens), has_tools=False)
        assert expected is not None
        assert (thinking, content) == expected[:2]


# ---------------------------------------------------------------------------
# Streaming completion end-to-end (engine) — #713
# ---------------------------------------------------------------------------


class TestGptOssStreamingThinking:
    async def test_stream_completion_surfaces_analysis_as_thinking(self, mock_manager):
        """The streaming engine path must not drop the analysis channel: it
        streams as structured ``{"thinking": ...}`` chunks, with only the
        final-channel text in ``text`` chunks."""
        from dataclasses import replace
        from unittest.mock import AsyncMock, patch

        from olmlx.engine.inference import _stream_completion
        from olmlx.utils.streaming import CancellableStream, StreamToken
        from olmlx.utils.timing import TimingStats

        lm = mock_manager._loaded["qwen3:latest"]
        lm.template_caps = replace(lm.template_caps, has_channel_format=True)

        texts = [
            "<|channel|>",
            "analysis",
            "<|message|>",
            "7*8",
            "=56",
            "<|end|>",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "It is",
            " 56.",
            "<|return|>",
        ]
        token_iter = iter(
            StreamToken(
                text=t,
                token=i,
                prompt_tokens=3,
                generation_tokens=i + 1,
                prompt_tps=0.0,
                generation_tps=0.0,
            )
            for i, t in enumerate(texts)
        )

        async def anext_impl():
            try:
                return next(token_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_stream = MagicMock(spec=CancellableStream)
        mock_stream.drain_and_join = AsyncMock()
        mock_stream._thread = None
        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        chunks = []
        with (
            patch("olmlx.engine.inference.mx", MagicMock()),
            patch("olmlx.engine.inference.async_mlx_stream", return_value=mock_stream),
        ):
            async for c in _stream_completion(lm, "Hi", 32, {}, TimingStats()):
                chunks.append(c)

        thinking = "".join(c.get("thinking", "") for c in chunks)
        content = "".join(c.get("text", "") for c in chunks if not c.get("done"))

        assert thinking == "7*8=56"
        assert content == "It is 56."
        # raw_text on the done chunk is unchanged (tools-mode parsing).
        done = [c for c in chunks if c.get("done")]
        assert done and done[-1]["raw_text"] == "".join(texts)
