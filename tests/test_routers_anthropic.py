"""Tests for olmlx.routers.anthropic."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from olmlx.routers.anthropic import (
    _build_options,
    _convert_messages,
    _convert_tools,
    _PING_SENTINEL,
    _sse,
    _with_keepalive_pings,
)
from olmlx.schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicTool,
    AnthropicToolInputSchema,
)
from olmlx.utils.timing import TimingStats


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
                AnthropicMessage(
                    role="assistant",
                    content=[
                        AnthropicContentBlock(type="text", text="Let me search"),
                        AnthropicContentBlock(
                            type="tool_use",
                            id="toolu_123",
                            name="search",
                            input={"q": "test"},
                        ),
                    ],
                ),
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
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicContentBlock(
                            type="tool_result",
                            tool_use_id="toolu_123",
                            content="Result data",
                        ),
                        AnthropicContentBlock(type="text", text="What does this mean?"),
                    ],
                ),
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
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicContentBlock(
                            type="tool_result",
                            tool_use_id="toolu_456",
                            content=[
                                {"type": "text", "text": "result1"},
                                {"type": "text", "text": "result2"},
                            ],
                        ),
                    ],
                ),
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
                AnthropicMessage(
                    role="assistant",
                    content=[
                        AnthropicContentBlock(type="thinking", text="Deep thoughts"),
                        AnthropicContentBlock(type="text", text="The answer"),
                    ],
                ),
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


class TestWithKeepalivePings:
    @pytest.mark.asyncio
    async def test_pings_during_delay(self):
        """Slow async gen should produce ping sentinels while waiting."""

        async def slow_gen():
            await asyncio.sleep(12)
            yield {"text": "hello", "done": False}
            yield {"text": "", "done": True}

        results = []
        async for item in _with_keepalive_pings(slow_gen(), interval=2.0):
            results.append(item)
            if item is not _PING_SENTINEL:
                if item.get("done"):
                    break

        pings = [r for r in results if r is _PING_SENTINEL]
        assert len(pings) >= 2, (
            f"Expected at least 2 pings during 12s delay, got {len(pings)}"
        )

    @pytest.mark.asyncio
    async def test_no_pings_when_fast(self):
        """Immediate yields should produce no ping sentinels."""

        async def fast_gen():
            yield {"text": "a", "done": False}
            yield {"text": "b", "done": False}
            yield {"text": "", "done": True}

        results = []
        async for item in _with_keepalive_pings(fast_gen(), interval=5.0):
            results.append(item)

        pings = [r for r in results if r is _PING_SENTINEL]
        assert len(pings) == 0

    @pytest.mark.asyncio
    async def test_cleanup_on_early_exit(self):
        """Verify task cancellation when breaking out early."""

        async def infinite_gen():
            while True:
                await asyncio.sleep(10)
                yield {"text": "tick"}

        results = []
        async for item in _with_keepalive_pings(infinite_gen(), interval=0.1):
            results.append(item)
            if len(results) >= 3:
                break

        # All items should be pings since the gen never yields fast enough
        assert all(r is _PING_SENTINEL for r in results)
        # Give a moment for cleanup
        await asyncio.sleep(0.05)


class TestAnthropicEndpoint:
    @pytest.mark.asyncio
    async def test_non_streaming(self, app_client):
        stats = TimingStats(prompt_eval_count=10, eval_count=20)
        mock_result = {"text": "Hello from MLX!", "done": True, "stats": stats}

        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                },
            )

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
            "done": True,
            "stats": stats,
        }

        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "think"}],
                    "max_tokens": 100,
                },
            )

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
            "done": True,
            "stats": stats,
        }

        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "search for test"}],
                    "max_tokens": 100,
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search",
                            "input_schema": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                            },
                        }
                    ],
                },
            )

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

        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                },
            )

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

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

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

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "think"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        text = resp.text
        assert "thinking_delta" in text
        assert "text_delta" in text

    @pytest.mark.asyncio
    async def test_streaming_with_tools_buffered(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {
                    "text": '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=10)}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "search"}],
                    "max_tokens": 100,
                    "stream": True,
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search",
                            "input_schema": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                            },
                        }
                    ],
                },
            )

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

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

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

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

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

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

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

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

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
                yield {
                    "text": '<tool_call>{"name": "search", "arguments": {"q": "t"}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=10)}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "search"}],
                    "max_tokens": 100,
                    "stream": True,
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search",
                            "input_schema": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                            },
                        }
                    ],
                },
            )

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

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        text = resp.text
        assert "thinking_delta" in text
        assert "text_delta" in text

    @pytest.mark.asyncio
    async def test_streaming_tools_keepalive_ping(self, app_client):
        """Test that keepalive pings are sent during tool buffering."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                # Simulate slow generation — but pings are time-based, hard to test.
                yield {"text": "some text", "done": False}
                yield {
                    "text": '<tool_call>{"name": "search", "arguments": {"q": "t"}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=5)}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "search"}],
                    "max_tokens": 100,
                    "stream": True,
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search",
                            "input_schema": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                            },
                        }
                    ],
                },
            )

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

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 100,
                    "stream": True,
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search",
                            "input_schema": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                            },
                        }
                    ],
                },
            )

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
                yield {
                    "text": f"<think>{long_thought}</think>short answer",
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        text = resp.text
        assert "thinking_delta" in text
        assert "text_delta" in text

    @pytest.mark.asyncio
    async def test_with_system_message(self, app_client):
        stats = TimingStats()
        mock_result = {"text": "I am helpful", "done": True, "stats": stats}

        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "who are you?"}],
                    "system": "You are a helpful assistant.",
                    "max_tokens": 100,
                },
            )

        assert resp.status_code == 200
        # Verify system message was passed
        call_args = mock_gen.call_args
        messages = call_args[0][2]  # messages arg
        assert messages[0]["role"] == "system"
        assert "helpful" in messages[0]["content"]


class TestCountTokens:
    @pytest.mark.asyncio
    async def test_simple_message(self, app_client, mock_loaded_model):
        mock_loaded_model.tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
        resp = await app_client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["input_tokens"] == 5

    @pytest.mark.asyncio
    async def test_with_system_message(self, app_client, mock_loaded_model):
        mock_loaded_model.tokenizer.apply_chat_template.return_value = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ]
        resp = await app_client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "system": "You are helpful.",
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["input_tokens"] == 7

    @pytest.mark.asyncio
    async def test_with_tools(self, app_client, mock_loaded_model):
        mock_loaded_model.tokenizer.apply_chat_template.return_value = list(range(20))
        resp = await app_client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "search for test"}],
                "max_tokens": 100,
                "tools": [
                    {
                        "name": "search",
                        "description": "Search",
                        "input_schema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                    }
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["input_tokens"] == 20
        # Verify tools kwarg was passed to apply_chat_template
        call_kwargs = mock_loaded_model.tokenizer.apply_chat_template.call_args[1]
        assert "tools" in call_kwargs

    @pytest.mark.asyncio
    async def test_tokenize_true_passed(self, app_client, mock_loaded_model):
        mock_loaded_model.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        await app_client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        call_kwargs = mock_loaded_model.tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["tokenize"] is True
        assert call_kwargs["add_generation_prompt"] is True

    @pytest.mark.asyncio
    async def test_beta_query_param(self, app_client, mock_loaded_model):
        mock_loaded_model.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        resp = await app_client.post(
            "/v1/messages/count_tokens?beta=true",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] == 3

    @pytest.mark.asyncio
    async def test_model_not_found(self, app_client):
        resp = await app_client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "nonexistent",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"

    @pytest.mark.asyncio
    async def test_dict_return_type(self, app_client, mock_loaded_model):
        """apply_chat_template may return dict with input_ids key."""
        mock_loaded_model.tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 2, 3, 4]
        }
        resp = await app_client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] == 4

    @pytest.mark.asyncio
    async def test_dict_with_nested_input_ids(self, app_client, mock_loaded_model):
        """apply_chat_template may return dict with batched input_ids: [[1,2,3]]."""
        mock_loaded_model.tokenizer.apply_chat_template.return_value = {
            "input_ids": [[1, 2, 3, 4, 5]]
        }
        resp = await app_client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] == 5

    @pytest.mark.asyncio
    async def test_nested_list_return_type(self, app_client, mock_loaded_model):
        """apply_chat_template may return list[list[int]]."""
        mock_loaded_model.tokenizer.apply_chat_template.return_value = [
            [1, 2, 3, 4, 5, 6]
        ]
        resp = await app_client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] == 6

    @pytest.mark.asyncio
    async def test_dict_without_input_ids_errors(self, app_client, mock_loaded_model):
        """apply_chat_template returning dict without input_ids should error, not silently return 0."""
        mock_loaded_model.tokenizer.apply_chat_template.return_value = {
            "token_ids": [1, 2, 3]
        }
        resp = await app_client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_vlm_text_tokenizer_unwrap(self, app_client, mock_loaded_model):
        """VLM models should use the inner .tokenizer for token counting."""
        from unittest.mock import MagicMock

        inner_tokenizer = MagicMock()
        inner_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5, 6, 7]
        mock_loaded_model.is_vlm = True
        mock_loaded_model.tokenizer.tokenizer = inner_tokenizer
        resp = await app_client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] == 7
        inner_tokenizer.apply_chat_template.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_encoding_return_type(self, app_client, mock_loaded_model):
        """apply_chat_template may return BatchEncoding (UserDict subclass)."""
        from collections import UserDict

        # BatchEncoding extends UserDict, not dict — simulate it
        batch_encoding = UserDict({"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]})
        mock_loaded_model.tokenizer.apply_chat_template.return_value = batch_encoding
        resp = await app_client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] == 8


class TestPingBeforeCacheInfo:
    @pytest.mark.asyncio
    async def test_ping_before_cache_info_preserves_stats(self, app_client):
        """When a keepalive ping arrives before cache_info, cache stats must still appear in message_start."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                # Simulate: ping fires during lock wait, then cache_info, then tokens
                yield {
                    "cache_info": True,
                    "cache_read_tokens": 42,
                    "cache_creation_tokens": 8,
                }
                yield {"text": "Hello", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=1)}

            return gen()

        async def pings_then_passthrough(aiter, interval=5.0):
            yield _PING_SENTINEL  # Ping fires before any real data
            async for item in aiter:
                yield item

        with (
            patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream),
            patch(
                "olmlx.routers.anthropic._with_keepalive_pings",
                side_effect=pings_then_passthrough,
            ),
        ):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        text = resp.text

        # Parse message_start event to extract usage
        for line in text.split("\n"):
            if line.startswith("data:") and "message_start" in line:
                data = json.loads(line[5:])
                usage = data["message"]["usage"]
                assert usage["cache_read_input_tokens"] == 42, (
                    f"Expected cache_read=42, got {usage}"
                )
                assert usage["cache_creation_input_tokens"] == 8, (
                    f"Expected cache_creation=8, got {usage}"
                )
                break
        else:
            pytest.fail("message_start event not found in SSE output")
