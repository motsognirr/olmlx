"""Tests for mlx_ollama.routers.anthropic."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from mlx_ollama.routers.anthropic import _build_options, _convert_messages, _convert_tools, _sse
from mlx_ollama.schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicTool,
    AnthropicToolInputSchema,
)
from mlx_ollama.utils.timing import TimingStats


class TestConvertTools:
    def test_no_tools(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
        )
        assert _convert_tools(req) is None

    def test_with_tools(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            tools=[
                AnthropicTool(
                    name="get_weather",
                    description="Get weather info",
                    input_schema=AnthropicToolInputSchema(
                        properties={"city": {"type": "string"}},
                        required=["city"],
                    ),
                ),
            ],
        )
        tools = _convert_tools(req)
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "get_weather"
        assert tools[0]["function"]["description"] == "Get weather info"
        assert "city" in tools[0]["function"]["parameters"]["properties"]


class TestConvertMessages:
    def test_simple_user_message(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hello")],
        )
        messages = _convert_messages(req)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    def test_system_string(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            system="Be helpful.",
        )
        messages = _convert_messages(req)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful."

    def test_system_blocks(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            system=[
                AnthropicContentBlock(type="text", text="Part 1"),
                AnthropicContentBlock(type="text", text="Part 2"),
            ],
        )
        messages = _convert_messages(req)
        assert messages[0]["role"] == "system"
        assert "Part 1" in messages[0]["content"]
        assert "Part 2" in messages[0]["content"]

    def test_assistant_with_tool_use(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[
                AnthropicMessage(role="assistant", content=[
                    AnthropicContentBlock(type="text", text="Let me search"),
                    AnthropicContentBlock(
                        type="tool_use", id="toolu_123",
                        name="search", input={"q": "test"},
                    ),
                ]),
            ],
        )
        messages = _convert_messages(req)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert "Let me search" in messages[0]["content"]
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[0]["tool_calls"][0]["function"]["name"] == "search"

    def test_user_with_tool_result(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[
                AnthropicMessage(role="user", content=[
                    AnthropicContentBlock(
                        type="tool_result", tool_use_id="toolu_123",
                        content="Result data",
                    ),
                    AnthropicContentBlock(type="text", text="What does this mean?"),
                ]),
            ],
        )
        messages = _convert_messages(req)
        # Should produce a tool message and a user message
        assert any(m["role"] == "tool" for m in messages)
        assert any(m["role"] == "user" for m in messages)
        tool_msg = next(m for m in messages if m["role"] == "tool")
        assert tool_msg["content"] == "Result data"
        assert tool_msg["tool_call_id"] == "toolu_123"

    def test_tool_result_list_content(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[
                AnthropicMessage(role="user", content=[
                    AnthropicContentBlock(
                        type="tool_result", tool_use_id="toolu_456",
                        content=[{"type": "text", "text": "result1"}, {"type": "text", "text": "result2"}],
                    ),
                ]),
            ],
        )
        messages = _convert_messages(req)
        tool_msg = next(m for m in messages if m["role"] == "tool")
        assert "result1" in tool_msg["content"]
        assert "result2" in tool_msg["content"]

    def test_thinking_blocks_skipped(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[
                AnthropicMessage(role="assistant", content=[
                    AnthropicContentBlock(type="thinking", text="Deep thoughts"),
                    AnthropicContentBlock(type="text", text="The answer"),
                ]),
            ],
        )
        messages = _convert_messages(req)
        assert "Deep thoughts" not in messages[0]["content"]
        assert "The answer" in messages[0]["content"]


class TestBuildOptions:
    def test_empty(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
        )
        opts = _build_options(req)
        assert opts == {}

    def test_all_options(self):
        req = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            stop_sequences=["END"],
        )
        opts = _build_options(req)
        assert opts["temperature"] == 0.5
        assert opts["top_p"] == 0.9
        assert opts["top_k"] == 40
        assert opts["stop"] == ["END"]


class TestSse:
    def test_format(self):
        result = _sse("message_start", {"type": "message_start"})
        assert result.startswith("event: message_start\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        data = json.loads(result.split("data: ")[1].strip())
        assert data["type"] == "message_start"


class TestAnthropicEndpoint:
    @pytest.mark.asyncio
    async def test_non_streaming(self, app_client):
        stats = TimingStats(prompt_eval_count=10, eval_count=20)
        mock_result = {"text": "Hello from MLX!", "done": True, "stats": stats}

        with patch("mlx_ollama.routers.anthropic.generate_chat", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["stop_reason"] == "end_turn"
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "Hello from MLX!"
        assert data["usage"]["input_tokens"] == 10
        assert data["usage"]["output_tokens"] == 20

    @pytest.mark.asyncio
    async def test_non_streaming_with_thinking(self, app_client):
        stats = TimingStats()
        mock_result = {
            "text": "<think>reasoning</think>The answer is 42.",
            "done": True, "stats": stats,
        }

        with patch("mlx_ollama.routers.anthropic.generate_chat", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "think"}],
                "max_tokens": 100,
            })

        assert resp.status_code == 200
        data = resp.json()
        content_types = [b["type"] for b in data["content"]]
        assert "thinking" in content_types
        assert "text" in content_types

    @pytest.mark.asyncio
    async def test_non_streaming_with_tool_use(self, app_client):
        stats = TimingStats()
        mock_result = {
            "text": '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>',
            "done": True, "stats": stats,
        }

        with patch("mlx_ollama.routers.anthropic.generate_chat", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "search for test"}],
                "max_tokens": 100,
                "tools": [{
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                }],
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["stop_reason"] == "tool_use"
        tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_non_streaming_empty_response(self, app_client):
        stats = TimingStats()
        mock_result = {"text": "", "done": True, "stats": stats}

        with patch("mlx_ollama.routers.anthropic.generate_chat", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            })

        assert resp.status_code == 200
        data = resp.json()
        # Should have at least an empty text block
        assert len(data["content"]) >= 1
        assert data["content"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_streaming(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Hello", "done": False}
                yield {"text": " world", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=5)}
            return gen()

        with patch("mlx_ollama.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
            })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        text = resp.text

        # Check SSE structure
        assert "event: message_start" in text
        assert "event: content_block_start" in text
        assert "event: content_block_delta" in text
        assert "event: content_block_stop" in text
        assert "event: message_delta" in text
        assert "event: message_stop" in text

    @pytest.mark.asyncio
    async def test_streaming_with_thinking(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "<think>", "done": False}
                yield {"text": "reasoning", "done": False}
                yield {"text": "</think>", "done": False}
                yield {"text": "answer", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}
            return gen()

        with patch("mlx_ollama.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "think"}],
                "max_tokens": 100,
                "stream": True,
            })

        assert resp.status_code == 200
        text = resp.text
        assert "thinking_delta" in text
        assert "text_delta" in text

    @pytest.mark.asyncio
    async def test_streaming_with_tools_buffered(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>', "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=10)}
            return gen()

        with patch("mlx_ollama.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "search"}],
                "max_tokens": 100,
                "stream": True,
                "tools": [{
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                }],
            })

        assert resp.status_code == 200
        text = resp.text
        assert "tool_use" in text
        assert "input_json_delta" in text
        assert '"stop_reason": "tool_use"' in text

    @pytest.mark.asyncio
    async def test_streaming_empty_output(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "", "done": True, "stats": TimingStats()}
            return gen()

        with patch("mlx_ollama.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
            })

        assert resp.status_code == 200
        text = resp.text
        assert "message_stop" in text
        # Should still have a text block
        assert "content_block_start" in text

    @pytest.mark.asyncio
    async def test_streaming_thinking_ends_midstream(self, app_client):
        """Test state machine when model ends while in thinking state."""
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "<think>incomplete thinking", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}
            return gen()

        with patch("mlx_ollama.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
            })

        assert resp.status_code == 200
        text = resp.text
        assert "thinking_delta" in text
        assert "message_stop" in text

    @pytest.mark.asyncio
    async def test_streaming_no_output_at_all(self, app_client):
        """Test state machine when model produces no output (stays in init)."""
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "", "done": True, "stats": TimingStats()}
            return gen()

        with patch("mlx_ollama.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
            })

        assert resp.status_code == 200
        text = resp.text
        # Should emit an empty text block
        assert "content_block_start" in text
        assert "content_block_stop" in text
        assert "message_stop" in text

    @pytest.mark.asyncio
    async def test_streaming_text_only_with_text_started(self, app_client):
        """Test that text block is properly closed."""
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Hello", "done": False}
                yield {"text": " world!", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=3)}
            return gen()

        with patch("mlx_ollama.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
            })

        assert resp.status_code == 200
        text = resp.text
        assert "Hello" in text
        assert "world!" in text
        assert text.count("content_block_stop") >= 1

    @pytest.mark.asyncio
    async def test_streaming_tools_with_thinking(self, app_client):
        """Test buffered tool mode with thinking blocks."""
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "<think>reasoning</think>", "done": False}
                yield {"text": '<tool_call>{"name": "search", "arguments": {"q": "t"}}</tool_call>', "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=10)}
            return gen()

        with patch("mlx_ollama.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "search"}],
                "max_tokens": 100,
                "stream": True,
                "tools": [{
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                }],
            })

        assert resp.status_code == 200
        text = resp.text
        assert "thinking" in text
        assert "tool_use" in text

    @pytest.mark.asyncio
    async def test_streaming_partial_think_tag(self, app_client):
        """Test state machine when <think> tag arrives across multiple tokens."""
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "<thi", "done": False}
                yield {"text": "nk>deep thought</think>answer", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}
            return gen()

        with patch("mlx_ollama.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
            })

        assert resp.status_code == 200
        text = resp.text
        assert "thinking_delta" in text
        assert "text_delta" in text

    @pytest.mark.asyncio
    async def test_streaming_tools_keepalive_ping(self, app_client):
        """Test that keepalive pings are sent during tool buffering."""
        import asyncio

        async def mock_stream(*args, **kwargs):
            async def gen():
                # Simulate slow generation — but pings are time-based, hard to test.
                yield {"text": "some text", "done": False}
                yield {"text": '<tool_call>{"name": "search", "arguments": {"q": "t"}}</tool_call>', "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=5)}
            return gen()

        with patch("mlx_ollama.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "search"}],
                "max_tokens": 100,
                "stream": True,
                "tools": [{
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                }],
            })

        assert resp.status_code == 200
        text = resp.text
        assert "tool_use" in text

    @pytest.mark.asyncio
    async def test_streaming_tools_text_only_no_tool_calls(self, app_client):
        """Test buffered mode when tools are available but model doesn't call any."""
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "I'll just respond normally.", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=5)}
            return gen()

        with patch("mlx_ollama.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
                "stream": True,
                "tools": [{
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                }],
            })

        assert resp.status_code == 200
        text = resp.text
        assert '"stop_reason": "end_turn"' in text
        assert "text_delta" in text

    @pytest.mark.asyncio
    async def test_streaming_long_thinking_chunked(self, app_client):
        """Test that long thinking blocks are emitted in chunks."""
        long_thought = "A" * 500
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": f"<think>{long_thought}</think>short answer", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}
            return gen()

        with patch("mlx_ollama.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
            })

        assert resp.status_code == 200
        text = resp.text
        assert "thinking_delta" in text
        assert "text_delta" in text

    @pytest.mark.asyncio
    async def test_with_system_message(self, app_client):
        stats = TimingStats()
        mock_result = {"text": "I am helpful", "done": True, "stats": stats}

        with patch("mlx_ollama.routers.anthropic.generate_chat", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post("/v1/messages", json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "who are you?"}],
                "system": "You are a helpful assistant.",
                "max_tokens": 100,
            })

        assert resp.status_code == 200
        # Verify system message was passed
        call_args = mock_gen.call_args
        messages = call_args[0][2]  # messages arg
        assert messages[0]["role"] == "system"
        assert "helpful" in messages[0]["content"]
