"""Unit tests for the helpers extracted from ChatSession.send_message (issue #480).

- ``_parse_turn_output``: the parse step (parse_model_output + error fallback)
- ``ChatSession._stream_one_turn``: per-turn generate_chat consume loop +
  ThinkingTracker handling

All tests are hermetic: generate_chat is mocked, no model, no GPU.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from olmlx.chat.config import ChatConfig
from olmlx.chat.session import ChatSession, _parse_turn_output
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
