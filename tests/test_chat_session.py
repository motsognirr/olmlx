"""Tests for olmlx.chat.session."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from olmlx.chat.config import ChatConfig
from olmlx.chat.session import ChatSession
from olmlx.chat.skills import SkillManager


def _make_session(
    *,
    mcp=None,
    skills=None,
    model_name="test:latest",
    thinking=True,
    max_turns=25,
    system_prompt=None,
):
    config = ChatConfig(
        model_name=model_name,
        thinking=thinking,
        max_turns=max_turns,
        system_prompt=system_prompt,
    )
    manager = MagicMock()
    return ChatSession(config=config, manager=manager, mcp=mcp, skills=skills)


class TestChatSessionInit:
    def test_empty_messages(self):
        session = _make_session()
        assert session.messages == []

    def test_system_prompt_in_messages(self):
        session = _make_session(system_prompt="Be helpful.")
        assert len(session.messages) == 1
        assert session.messages[0] == {"role": "system", "content": "Be helpful."}


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """Model returns plain text, no tool calls."""
        session = _make_session()

        async def fake_stream(*args, **kwargs):
            yield {"text": "Hello", "done": False}
            yield {"text": " world", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch("olmlx.chat.session.generate_chat", return_value=fake_stream()):
            events = []
            async for event in session.send_message("Hi"):
                events.append(event)

        # Should have token events and a done event
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 2
        assert token_events[0]["text"] == "Hello"
        assert token_events[1]["text"] == " world"

        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

        # Messages should contain user and assistant
        assert session.messages[-2]["role"] == "user"
        assert session.messages[-2]["content"] == "Hi"
        assert session.messages[-1]["role"] == "assistant"
        assert session.messages[-1]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_thinking_extracted(self):
        """Model output with <think> tags should emit thinking event."""
        session = _make_session()

        async def fake_stream(*args, **kwargs):
            yield {"text": "<think>Let me think</think>The answer is 42", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch("olmlx.chat.session.generate_chat", return_value=fake_stream()):
            events = []
            async for event in session.send_message("What is the answer?"):
                events.append(event)

        thinking_events = [e for e in events if e["type"] == "thinking"]
        assert len(thinking_events) == 1
        assert "Let me think" in thinking_events[0]["text"]

        # Token events should NOT contain raw <think> tags
        token_events = [e for e in events if e["type"] == "token"]
        token_text = "".join(e["text"] for e in token_events)
        assert "<think>" not in token_text
        assert "</think>" not in token_text
        assert "The answer is 42" in token_text

        # Assistant message should have the visible text only
        assert session.messages[-1]["content"] == "The answer is 42"

    @pytest.mark.asyncio
    async def test_thinking_streamed_incrementally(self):
        """Thinking tokens streamed across multiple chunks should be suppressed."""
        session = _make_session()

        async def fake_stream(*args, **kwargs):
            yield {"text": "<think>", "done": False}
            yield {"text": "Let me ", "done": False}
            yield {"text": "think", "done": False}
            yield {"text": "</think>", "done": False}
            yield {"text": "The answer", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch("olmlx.chat.session.generate_chat", return_value=fake_stream()):
            events = []
            async for event in session.send_message("Q"):
                events.append(event)

        token_events = [e for e in events if e["type"] == "token"]
        token_text = "".join(e["text"] for e in token_events)
        assert "<think>" not in token_text
        assert "The answer" in token_text

    @pytest.mark.asyncio
    async def test_tool_call_agent_loop(self):
        """Model calls a tool, result is fed back, model continues."""
        mcp = MagicMock()
        mcp.get_tools_for_chat.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ]
        mcp.call_tool = AsyncMock(return_value="file contents here")

        session = _make_session(mcp=mcp)
        call_count = 0

        async def fake_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: model makes a tool call
                yield {
                    "text": '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": MagicMock()}
            else:
                # Second call: model responds with the result
                yield {"text": "The file contains: file contents here", "done": False}
                yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_generate(),
        ):
            events = []
            async for event in session.send_message("Read /tmp/test.txt"):
                events.append(event)

        # Should have tool_call and tool_result events
        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 1
        assert tool_call_events[0]["name"] == "read_file"

        tool_result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_result_events) == 1
        assert "file contents here" in tool_result_events[0]["result"]

        # MCP should have been called
        mcp.call_tool.assert_awaited_once_with("read_file", {"path": "/tmp/test.txt"})

        # Should have two generate_chat calls
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        """Agent loop stops after max_turns to prevent infinite loops."""
        mcp = MagicMock()
        mcp.get_tools_for_chat.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        mcp.call_tool = AsyncMock(return_value="pong")

        session = _make_session(mcp=mcp, max_turns=2)

        async def fake_generate(*args, **kwargs):
            # Always make a tool call
            yield {
                "text": '<tool_call>{"name": "ping", "arguments": {}}</tool_call>',
                "done": False,
            }
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_generate(),
        ):
            events = []
            async for event in session.send_message("Keep pinging"):
                events.append(event)

        # Should have max_turns_exceeded event
        exceeded = [e for e in events if e["type"] == "max_turns_exceeded"]
        assert len(exceeded) == 1

    @pytest.mark.asyncio
    async def test_tool_call_error_fed_back(self):
        """Tool call error is fed back to the model as a tool result."""
        mcp = MagicMock()
        mcp.get_tools_for_chat.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "fail_tool",
                    "description": "A tool that fails",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        mcp.call_tool = AsyncMock(side_effect=RuntimeError("Connection refused"))

        session = _make_session(mcp=mcp)
        call_count = 0

        async def fake_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield {
                    "text": '<tool_call>{"name": "fail_tool", "arguments": {}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": MagicMock()}
            else:
                yield {"text": "The tool failed, sorry.", "done": False}
                yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_generate(),
        ):
            events = []
            async for event in session.send_message("Use the tool"):
                events.append(event)

        error_events = [e for e in events if e["type"] == "tool_error"]
        assert len(error_events) == 1
        assert "Connection refused" in error_events[0]["error"]

        # Error should be fed back as tool result in messages
        tool_msgs = [m for m in session.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "Error" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_no_mcp_no_tools(self):
        """Without MCP, no tools are passed to generate_chat."""
        session = _make_session(mcp=None)

        async def fake_stream(*args, **kwargs):
            # Verify no tools kwarg
            assert kwargs.get("tools") is None
            yield {"text": "Hi", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_stream(*a, **kw),
        ):
            events = []
            async for event in session.send_message("Hello"):
                events.append(event)

    @pytest.mark.asyncio
    async def test_cache_info_skipped(self):
        """cache_info chunks from generate_chat should be silently skipped."""
        session = _make_session()

        async def fake_stream(*args, **kwargs):
            yield {
                "cache_info": True,
                "cache_read_tokens": 100,
                "cache_creation_tokens": 50,
            }
            yield {"text": "Hello", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch("olmlx.chat.session.generate_chat", return_value=fake_stream()):
            events = []
            async for event in session.send_message("Hi"):
                events.append(event)

        # No cache_info events should leak through
        assert all(e["type"] != "cache_info" for e in events)


class TestClearHistory:
    def test_clear_removes_messages(self):
        session = _make_session()
        session.messages.append({"role": "user", "content": "Hi"})
        session.messages.append({"role": "assistant", "content": "Hello"})
        session.clear_history()
        assert session.messages == []

    def test_clear_preserves_system_prompt(self):
        session = _make_session(system_prompt="Be helpful.")
        session.messages.append({"role": "user", "content": "Hi"})
        session.clear_history()
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "system"
        assert session.messages[0]["content"] == "Be helpful."

    def test_clear_uses_updated_system_prompt(self):
        """After changing config.system_prompt, clear_history uses the new one."""
        session = _make_session(system_prompt="Old prompt.")
        session.config.system_prompt = "New prompt."
        session.clear_history()
        assert len(session.messages) == 1
        assert session.messages[0]["content"] == "New prompt."

    def test_clear_invalidates_prompt_cache(self):
        """clear_history should invalidate the prompt cache for the chat model."""
        session = _make_session(model_name="qwen3:8b")
        session.clear_history()
        session.manager.invalidate_prompt_cache.assert_called_with("qwen3:8b", "chat")


class TestRepetitionOptions:
    @pytest.mark.asyncio
    async def test_passes_repeat_penalty_to_generate_chat(self):
        """ChatSession should pass repeat_penalty and repeat_last_n as options."""
        config = ChatConfig(
            model_name="test:latest",
            repeat_penalty=1.2,
            repeat_last_n=128,
        )
        manager = MagicMock()
        session = ChatSession(config=config, manager=manager)

        captured_kwargs = {}

        async def fake_stream(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield {"text": "Hello", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_stream(*a, **kw),
        ):
            async for _ in session.send_message("Hi"):
                pass

        assert captured_kwargs["options"] == {
            "repeat_penalty": 1.2,
            "repeat_last_n": 128,
        }

    @pytest.mark.asyncio
    async def test_default_repeat_penalty(self):
        """Default config should pass repeat_penalty=1.1, repeat_last_n=64."""
        session = _make_session()

        captured_kwargs = {}

        async def fake_stream(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield {"text": "Hello", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_stream(*a, **kw),
        ):
            async for _ in session.send_message("Hi"):
                pass

        assert captured_kwargs["options"]["repeat_penalty"] == 1.1
        assert captured_kwargs["options"]["repeat_last_n"] == 64


class TestRepetitionDetection:
    @pytest.mark.asyncio
    async def test_stops_on_repetitive_output(self):
        """Generation should stop early when repetitive output is detected."""
        session = _make_session()

        repeated_phrase = "I am repeating myself. "

        async def fake_stream(*args, **kwargs):
            # Emit the same phrase many times to trigger detection
            for _ in range(50):
                yield {"text": repeated_phrase, "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_stream(*a, **kw),
        ):
            events = []
            async for event in session.send_message("Say something"):
                events.append(event)

        token_events = [e for e in events if e["type"] == "token"]
        # Should have stopped well before all 50 chunks were emitted
        assert len(token_events) < 50

        # Should still have a done event
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_no_false_positive_on_normal_output(self):
        """Normal varied output should not trigger repetition detection."""
        session = _make_session()

        async def fake_stream(*args, **kwargs):
            yield {"text": "The quick brown fox ", "done": False}
            yield {"text": "jumps over the lazy dog. ", "done": False}
            yield {"text": "This is a normal response.", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_stream(*a, **kw),
        ):
            events = []
            async for event in session.send_message("Tell me something"):
                events.append(event)

        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 3  # All tokens should come through


class TestSkillIntegration:
    def _make_skills(self, tmp_path):
        (tmp_path / "review.md").write_text(
            "---\nname: code-review\ndescription: Code review guidelines\n---\n\nReview carefully."
        )
        mgr = SkillManager(tmp_path)
        mgr.load()
        return mgr

    def test_use_skill_tool_included(self, tmp_path):
        """When skills are provided, use_skill tool should be available."""
        skills = self._make_skills(tmp_path)
        _make_session(skills=skills)
        # The skill tool definition should exist
        tool_def = skills.get_tool_definition()
        assert tool_def is not None
        assert tool_def["function"]["name"] == "use_skill"

    def test_skill_index_in_system_prompt(self, tmp_path):
        """Skill index should appear in the system prompt."""
        skills = self._make_skills(tmp_path)
        session = _make_session(skills=skills, system_prompt="Be helpful.")
        system_msgs = [m for m in session.messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "code-review" in system_msgs[0]["content"]
        assert "Be helpful." in system_msgs[0]["content"]

    def test_skill_index_in_system_prompt_no_user_prompt(self, tmp_path):
        """Skill index should be the system prompt when no user prompt given."""
        skills = self._make_skills(tmp_path)
        session = _make_session(skills=skills)
        system_msgs = [m for m in session.messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "code-review" in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_use_skill_handled_locally(self, tmp_path):
        """use_skill calls should be handled by SkillManager, not MCP."""
        skills = self._make_skills(tmp_path)
        session = _make_session(skills=skills)
        call_count = 0

        async def fake_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield {
                    "text": '<tool_call>{"name": "use_skill", "arguments": {"name": "code-review"}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": MagicMock()}
            else:
                yield {"text": "I will review the code carefully.", "done": False}
                yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_generate(),
        ):
            events = []
            async for event in session.send_message("Review my code"):
                events.append(event)

        result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(result_events) == 1
        assert "Review carefully." in result_events[0]["result"]

    @pytest.mark.asyncio
    async def test_mcp_and_skills_together(self, tmp_path):
        """Both MCP tools and use_skill should work together."""
        skills = self._make_skills(tmp_path)
        mcp = MagicMock()
        mcp.get_tools_for_chat.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ]
        mcp.call_tool = AsyncMock(return_value="file contents")

        session = _make_session(skills=skills, mcp=mcp)
        call_count = 0

        async def fake_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield {
                    "text": '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": MagicMock()}
            else:
                yield {"text": "Done", "done": False}
                yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_generate(),
        ):
            events = []
            async for event in session.send_message("Read a file"):
                events.append(event)

        # MCP tool should have been called
        mcp.call_tool.assert_awaited_once()

    def test_no_skills_no_system_prompt_injection(self):
        """Without skills, system prompt should be unchanged."""
        session = _make_session(system_prompt="Be helpful.")
        assert len(session.messages) == 1
        assert session.messages[0]["content"] == "Be helpful."

    def test_clear_history_preserves_skill_index(self, tmp_path):
        """After clear_history, skill index should still be in system prompt."""
        skills = self._make_skills(tmp_path)
        session = _make_session(skills=skills, system_prompt="Be helpful.")
        session.messages.append({"role": "user", "content": "Hi"})
        session.clear_history()
        system_msgs = [m for m in session.messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "code-review" in system_msgs[0]["content"]
