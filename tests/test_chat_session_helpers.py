"""Unit tests for the helpers extracted from ChatSession.send_message (issue #480).

- ``_parse_turn_output``: the parse step (parse_model_output + error fallback)
- ``ChatSession._stream_one_turn``: per-turn generate_chat consume loop +
  ThinkingTracker handling

All tests are hermetic: generate_chat is mocked, no model, no GPU.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from olmlx.chat.config import ChatConfig
from olmlx.chat.session import (
    ChatSession,
    ThinkingTracker,
    _parse_turn_output,
    _ToolMarkupStripper,
    _longest_partial_tag_suffix,
)
from olmlx.engine.template_caps import TemplateCaps


def _make_session(
    *,
    thinking=True,
    template_has_thinking=False,
):
    config = ChatConfig(model_name="test:latest", thinking=thinking)
    manager = MagicMock()
    loaded_model = MagicMock()
    loaded_model.template_caps = TemplateCaps(
        supports_enable_thinking=template_has_thinking,
        has_thinking_tags=template_has_thinking,
    )
    manager.ensure_loaded = AsyncMock(return_value=loaded_model)
    return ChatSession(config=config, manager=manager)


def _stream(*chunks):
    """Build an async generator factory yielding the given chunks."""

    async def gen(*args, **kwargs):
        for c in chunks:
            yield c

    return gen


def _new_result():
    return {
        "full_text": "",
        "thinking_expected": False,
        "repetition_stopped": False,
    }


async def _drain_turn(session, result, **kwargs):
    kwargs.setdefault("options", {})
    kwargs.setdefault("mcp_tools", None)
    kwargs.setdefault("implicit_thinking", False)
    kwargs.setdefault("template_has_thinking", False)
    events = []
    async for ev in session._stream_one_turn(result=result, **kwargs):
        events.append(ev)
    return events


class TestParseTurnOutput:
    def test_plain_text(self):
        visible, tool_uses, error = _parse_turn_output(
            "Hello world", has_tools=False, thinking_expected=False
        )
        assert visible == "Hello world"
        assert tool_uses == []
        assert error is None

    def test_thinking_stripped(self):
        visible, tool_uses, error = _parse_turn_output(
            "<think>hmm</think>Answer", has_tools=False, thinking_expected=True
        )
        assert visible == "Answer"
        assert tool_uses == []
        assert error is None

    def test_tool_call_parsed(self):
        text = (
            'Let me check.<tool_call>{"name": "get_weather", '
            '"arguments": {"city": "Paris"}}</tool_call>'
        )
        visible, tool_uses, error = _parse_turn_output(
            text, has_tools=True, thinking_expected=False
        )
        assert error is None
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "get_weather"
        assert tool_uses[0]["input"] == {"city": "Paris"}

    def test_parse_failure_falls_back_to_stripped_text(self):
        with patch(
            "olmlx.chat.session.parse_model_output",
            side_effect=ValueError("boom"),
        ):
            visible, tool_uses, error = _parse_turn_output(
                "<think>secret</think>The text",
                has_tools=True,
                thinking_expected=True,
            )
        assert visible == "The text"
        assert tool_uses == []
        assert error is not None
        assert error["type"] == "tool_error"
        assert error["name"] == "(parse error)"
        assert error["is_user_error"] is False
        assert "boom" in error["error"]


class TestStreamOneTurn:
    async def test_plain_tokens(self):
        session = _make_session()
        gen = _stream(
            {"text": "Hello ", "done": False},
            {"text": "world", "done": False},
            {"text": "", "done": True},
        )
        result = _new_result()
        with patch("olmlx.chat.session.generate_chat", side_effect=gen):
            events = await _drain_turn(session, result)

        token_text = "".join(e["text"] for e in events if e["type"] == "token")
        assert token_text == "Hello world"
        assert result["full_text"] == "Hello world"
        assert result["thinking_expected"] is False
        assert result["repetition_stopped"] is False

    async def test_thinking_expected_meta_captured(self):
        session = _make_session()
        gen = _stream(
            {"thinking_expected": True},
            {"text": "Hi", "done": False},
            {"text": "", "done": True},
        )
        result = _new_result()
        with patch("olmlx.chat.session.generate_chat", side_effect=gen):
            events = await _drain_turn(session, result)

        assert result["thinking_expected"] is True
        # The meta chunk must not surface as an event
        assert all("thinking_expected" not in e for e in events)

    async def test_cache_info_chunk_skipped(self):
        session = _make_session()
        gen = _stream(
            {"cache_info": {"hit": True}},
            {"text": "Hi", "done": False},
            {"text": "", "done": True},
        )
        result = _new_result()
        with patch("olmlx.chat.session.generate_chat", side_effect=gen):
            events = await _drain_turn(session, result)

        assert [e["type"] for e in events] == ["token"]
        assert result["full_text"] == "Hi"

    async def test_explicit_thinking_events(self):
        session = _make_session(thinking=True)
        gen = _stream(
            {"text": "<think>let me think", "done": False},
            {"text": "</think>", "done": False},
            {"text": "Answer", "done": False},
            {"text": "", "done": True},
        )
        result = _new_result()
        with patch("olmlx.chat.session.generate_chat", side_effect=gen):
            events = await _drain_turn(session, result)

        types = [e["type"] for e in events]
        assert "thinking_start" in types
        assert "thinking_token" in types
        assert "thinking_end" in types
        think_text = "".join(e["text"] for e in events if e["type"] == "thinking_token")
        assert "let me think" in think_text
        token_text = "".join(e["text"] for e in events if e["type"] == "token")
        assert token_text == "Answer"
        # full_text keeps the raw output including tags for the parse step
        assert "<think>" in result["full_text"]

    async def test_repetition_detected_stops_stream(self):
        session = _make_session()
        chunks = [{"text": "repeat phrase here. ", "done": False} for _ in range(60)]
        chunks.append({"text": "", "done": True})
        gen = _stream(*chunks)
        result = _new_result()
        with patch("olmlx.chat.session.generate_chat", side_effect=gen):
            events = await _drain_turn(session, result)

        types = [e["type"] for e in events]
        assert "repetition_detected" in types
        assert result["repetition_stopped"] is True
        # Stream stopped early — far fewer token events than the 60 chunks
        assert types.count("token") < 60

    async def test_disabled_thinking_implicit_flush(self):
        """Implicit thinking with thinking disabled and no </think> flushes as visible."""
        session = _make_session(thinking=False, template_has_thinking=True)
        gen = _stream(
            {"text": "buffered content", "done": False},
            {"text": "", "done": True},
        )
        result = _new_result()
        with patch("olmlx.chat.session.generate_chat", side_effect=gen):
            events = await _drain_turn(
                session,
                result,
                implicit_thinking=True,
                template_has_thinking=True,
            )

        token_text = "".join(e["text"] for e in events if e["type"] == "token")
        assert token_text == "buffered content"
        assert not any(e["type"] == "thinking_token" for e in events)

    async def test_repetition_mid_thinking_strips_and_closes(self):
        """Repetition inside an open <think> block strips it and emits thinking_end."""
        session = _make_session(thinking=True)
        chunks = [{"text": "<think>", "done": False}]
        chunks += [{"text": "repeat phrase here. ", "done": False} for _ in range(60)]
        chunks.append({"text": "", "done": True})
        gen = _stream(*chunks)
        result = _new_result()
        with patch("olmlx.chat.session.generate_chat", side_effect=gen):
            events = await _drain_turn(session, result)

        types = [e["type"] for e in events]
        assert "repetition_detected" in types
        assert "thinking_end" in types
        assert result["repetition_stopped"] is True
        # Incomplete think block stripped from the accumulated text
        assert "<think>" not in result["full_text"]


class TestLongestPartialTagSuffix:
    def test_no_partial(self):
        assert _longest_partial_tag_suffix("hello", "<|tool_call>") == 0

    def test_full_partial(self):
        assert _longest_partial_tag_suffix("abc<|to", "<|tool_call>") == 4

    def test_does_not_count_full_tag(self):
        # A complete tag is not a "partial" — must return exactly 0.
        assert _longest_partial_tag_suffix("<|tool_call>", "<|tool_call>") == 0

    def test_partial_close_tag(self):
        assert _longest_partial_tag_suffix("done<tool_", "<tool_call|>") == 6
        assert _longest_partial_tag_suffix("done", "<tool_call|>") == 0


class TestToolMarkupStripper:
    def test_passes_plain_text(self):
        s = _ToolMarkupStripper()
        assert s.feed("hello world") == "hello world"
        assert s.flush() == ""

    def test_removes_whole_tool_call_in_one_chunk(self):
        s = _ToolMarkupStripper()
        out = s.feed("before<|tool_call>call:f{a:1}<tool_call|>after")
        assert out == "beforeafter"
        assert s.flush() == ""

    def test_removes_tool_call_split_across_chunks(self):
        s = _ToolMarkupStripper()
        out = "".join(
            [
                s.feed("vis<|tool_"),
                s.feed("call>call:f{a:"),
                s.feed("1}<tool_"),
                s.feed("call|>tail"),
            ]
        )
        assert out == "vistail"
        assert s.flush() == ""

    def test_holds_partial_open_then_resolves_as_literal(self):
        s = _ToolMarkupStripper()
        # "<|too" looks like a partial open tag, held back...
        assert s.feed("x<|too") == "x"
        # ...but the next chunk shows it was literal text.
        assert s.feed("ls are fun") == "<|tools are fun"
        assert s.flush() == ""

    def test_flush_emits_held_partial_when_outside(self):
        s = _ToolMarkupStripper()
        assert s.feed("done<|tool") == "done"
        # Stream ended mid partial open-tag; it was literal.
        assert s.flush() == "<|tool"

    def test_flush_drops_unterminated_tool_call(self):
        s = _ToolMarkupStripper()
        assert s.feed("a<|tool_call>call:f{") == "a"
        # Unterminated tool call (no close) — drop it from display.
        assert s.flush() == ""

    def test_removes_two_consecutive_tool_calls(self):
        s = _ToolMarkupStripper()
        out = s.feed(
            "before<|tool_call>call:f1{}<tool_call|>mid"
            "<|tool_call>call:f2{}<tool_call|>after"
        )
        assert out == "beforemidafter"
        assert s.flush() == ""


class TestThinkingTrackerGemma4:
    def _drain(self, tracker, chunks):
        think, visible = [], []
        started = ended = False
        for c in chunks:
            td, vd, te, ts = tracker.feed(c)
            if td:
                think.append(td)
            if vd:
                visible.append(vd)
            started = started or ts
            ended = ended or te
        return "".join(think), "".join(visible), started, ended

    def test_gemma4_channel_split_single_chunk(self):
        t = ThinkingTracker()
        raw = "<|channel>thought\nreasoning here<channel|>The answer is 4."
        think, visible, started, ended = self._drain(t, [raw])
        assert think == "reasoning here"
        assert visible == "The answer is 4."
        assert started and ended

    def test_gemma4_channel_split_across_chunks(self):
        t = ThinkingTracker()
        chunks = ["<|channel>thought\nrea", "soning<chan", "nel|>visible"]
        think, visible, started, ended = self._drain(t, chunks)
        assert think == "reasoning"
        assert visible == "visible"
        # Event flags must fire correctly even when the delimiters straddle
        # chunk boundaries.
        assert started and ended

    def test_gemma4_tool_call_suppressed_from_visible(self):
        t = ThinkingTracker()
        raw = (
            "<|channel>thought\nI will call it.<channel|>"
            '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
        )
        think, visible, _, _ = self._drain(t, [raw])
        assert think == "I will call it."
        assert visible == ""
        # Raw markup is preserved for the turn-end parse.
        assert "<|tool_call>" in t.accumulated
        assert "<|channel>thought" in t.accumulated

    def test_accumulated_is_raw(self):
        t = ThinkingTracker()
        chunks = ["<|channel>thought\nx<channel|>", "<|tool_call>call:f{}<tool_call|>"]
        for c in chunks:
            t.feed(c)
        assert t.accumulated == "".join(chunks)

    def test_gemma4_thinking_disabled_strips_channel(self):
        t = ThinkingTracker(thinking_disabled=True)
        raw = "<|channel>thought\nhidden reasoning<channel|>visible answer"
        think, visible, started, ended = self._drain(t, [raw])
        assert think == ""
        assert visible == "visible answer"
        assert not started

    def test_gemma4_repetition_strip_truncates_at_channel(self):
        t = ThinkingTracker()
        t.feed("<|channel>thought\nlooping looping looping")
        assert t.in_thinking
        t.strip_on_repetition()
        assert "<|channel>thought" not in t.accumulated
        assert not t.in_thinking

    def test_flush_implicit_thinking_no_close_routes_to_thinking(self):
        # thinking expected (implicit), buffered content, never a close tag →
        # flushed as thinking (the old implicit-mode contract).
        t = ThinkingTracker(template_has_thinking=True)
        td, vd, _te, _ts = t.feed("Still thinking")
        assert td is None and vd is None  # held in the detect buffer
        think_delta, visible_delta, started = t.flush()
        assert think_delta == "Still thinking"
        assert visible_delta is None
        assert started and t.in_thinking

    def test_flush_plain_content_routes_to_visible(self):
        # No thinking expected → plain content streams during feed; flush adds
        # nothing.
        t = ThinkingTracker()
        td, vd, _te, _ts = t.feed("plain answer")
        assert vd == "plain answer"
        assert t.flush() == (None, None, False)

    def test_flush_disabled_surfaces_buffer_as_visible(self):
        # Thinking disabled + implicit + no close → the buffered (would-be
        # thinking) content is surfaced as visible, never as thinking.
        t = ThinkingTracker(thinking_disabled=True, template_has_thinking=True)
        t.feed("buffered content")
        think_delta, visible_delta, started = t.flush()
        assert think_delta is None
        assert visible_delta == "buffered content"
        assert not started
