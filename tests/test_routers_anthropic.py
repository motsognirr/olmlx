"""Tests for olmlx.routers.anthropic."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmlx.routers.anthropic import (
    _anthropic_stop_reason,
    _build_options,
    _convert_messages,
    _convert_tools,
    _resolve_anthropic_model,
    _sse,
    _strip_billing_headers,
)

from olmlx.schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicThinkingParam,
    AnthropicTool,
    AnthropicToolInputSchema,
)
from olmlx.utils.timing import TimingStats

MAP_PATCH = "olmlx.routers.anthropic._anthropic_model_map"


def _content_block_start_types(sse_text: str) -> list[str]:
    """Ordered list of content_block types from `content_block_start` SSE events."""
    types: list[str] = []
    for line in sse_text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            payload = json.loads(line[5:])
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "content_block_start":
            types.append(payload["content_block"]["type"])
    return types


def _message_delta_usage(sse_text: str) -> dict:
    """Usage dict from the `message_delta` SSE event (empty if none)."""
    for line in sse_text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            payload = json.loads(line[5:])
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "message_delta":
            return payload.get("usage", {})
    return {}


class TestConvertTools:
    def test_no_tools(self):
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
        )
        assert _convert_tools(req) is None

    def test_with_tools(self):
        req = AnthropicMessagesRequest(
            max_tokens=100,
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


class TestStripBillingHeaders:
    def test_list_system_billing_first_block(self):
        """Billing header as first block is removed, real content preserved."""
        system = [
            AnthropicContentBlock(
                type="text",
                text="x-anthropic-billing-header: cc_version=2.1.37.981; cc_entrypoint=cli; cch=0a829;",
            ),
            AnthropicContentBlock(type="text", text="You are a helpful assistant."),
        ]
        result = _strip_billing_headers(system)
        assert len(result) == 1
        assert result[0].text == "You are a helpful assistant."

    def test_list_system_no_billing_blocks(self):
        """System with no billing blocks is unchanged."""
        system = [
            AnthropicContentBlock(type="text", text="You are helpful."),
            AnthropicContentBlock(type="text", text="Be concise."),
        ]
        result = _strip_billing_headers(system)
        assert len(result) == 2
        assert result[0].text == "You are helpful."
        assert result[1].text == "Be concise."

    def test_list_system_only_billing_blocks(self):
        """System with only billing blocks returns None."""
        system = [
            AnthropicContentBlock(
                type="text",
                text="x-anthropic-billing-header: cc_version=2.1.37; cch=abc;",
            ),
        ]
        result = _strip_billing_headers(system)
        assert result is None

    def test_string_system_starting_with_billing(self):
        """String system starting with billing header has that line stripped."""
        system = (
            "x-anthropic-billing-header: cc_version=2.1.37; cch=abc;\nYou are helpful."
        )
        result = _strip_billing_headers(system)
        assert result == "You are helpful."

    def test_string_system_without_billing(self):
        """String system without billing header is unchanged."""
        system = "You are a helpful assistant."
        result = _strip_billing_headers(system)
        assert result == "You are a helpful assistant."

    def test_string_system_trailing_newline_preserved(self):
        """Trailing whitespace preserved when no billing headers removed."""
        system = "You are helpful.\n"
        result = _strip_billing_headers(system)
        assert result == "You are helpful.\n"

    def test_none_system(self):
        """None system returns None."""
        result = _strip_billing_headers(None)
        assert result is None

    def test_logs_when_stripping(self, caplog):
        """Info log emitted when billing headers are stripped."""
        system = [
            AnthropicContentBlock(
                type="text",
                text="x-anthropic-billing-header: cc_version=2.1.37; cch=abc;",
            ),
            AnthropicContentBlock(type="text", text="Real content."),
        ]
        with caplog.at_level(logging.INFO, logger="olmlx.routers.anthropic"):
            _strip_billing_headers(system)
        assert any("billing" in r.message.lower() for r in caplog.records)


class TestConvertMessages:
    def test_simple_user_message(self):
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hello")],
        )
        messages = _convert_messages(req)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    def test_system_string(self):
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            system="Be helpful.",
        )
        messages = _convert_messages(req)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful."

    def test_system_blocks(self):
        req = AnthropicMessagesRequest(
            max_tokens=100,
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

    def test_system_role_message_folded_into_leading_system(self):
        """A ``system``-role message inside ``messages`` is folded into the
        leading system block instead of appended at a non-zero index.

        The Anthropic API carries system content in a dedicated top-level
        field, but some clients (e.g. Claude Code) also place a ``system``
        turn inside ``messages``.  Appending it verbatim produced a second
        system message at index >= 1, which strict chat templates (Qwen3.6)
        reject with "System message must be at the beginning." — a 500 on
        every ``/v1/messages`` request.  Regression for that failure.
        """
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            system=[AnthropicContentBlock(type="text", text="Top-level system.")],
            messages=[
                AnthropicMessage(role="system", content="Inline system."),
                AnthropicMessage(role="user", content="hello"),
            ],
        )
        messages = _convert_messages(req)
        # Exactly one system message, and it is first.
        system_indices = [i for i, m in enumerate(messages) if m["role"] == "system"]
        assert system_indices == [0]
        assert "Top-level system." in messages[0]["content"]
        assert "Inline system." in messages[0]["content"]
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_system_role_message_without_top_level_system(self):
        """A ``system``-role message in ``messages`` with no top-level system
        still becomes the single leading system message."""
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[
                AnthropicMessage(role="system", content="Inline only."),
                AnthropicMessage(role="user", content="hi"),
            ],
        )
        messages = _convert_messages(req)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Inline only."
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_assistant_with_tool_use(self):
        req = AnthropicMessagesRequest(
            max_tokens=100,
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
            max_tokens=100,
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
            max_tokens=100,
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
            max_tokens=100,
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

    def test_system_blocks_billing_header_stripped(self):
        """Billing header block in system is stripped during convert_messages."""
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            system=[
                AnthropicContentBlock(
                    type="text",
                    text="x-anthropic-billing-header: cc_version=2.1.37; cch=abc;",
                ),
                AnthropicContentBlock(type="text", text="You are helpful."),
            ],
        )
        messages = _convert_messages(req)
        assert messages[0]["role"] == "system"
        assert "billing" not in messages[0]["content"]
        assert "You are helpful." in messages[0]["content"]

    def test_system_only_billing_header_no_system_message(self):
        """System with only billing header produces no system message."""
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            system=[
                AnthropicContentBlock(
                    type="text",
                    text="x-anthropic-billing-header: cc_version=2.1.37; cch=abc;",
                ),
            ],
        )
        messages = _convert_messages(req)
        assert all(m["role"] != "system" for m in messages)

    def test_mixed_text_and_tool_result_preserves_order(self):
        """A user turn ``[text, tool_result]`` must stay user-then-tool, not
        be reordered to tool-then-user (#627)."""
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        AnthropicContentBlock(type="text", text="here is the result"),
                        AnthropicContentBlock(
                            type="tool_result",
                            tool_use_id="tu_1",
                            content="42",
                        ),
                    ],
                ),
            ],
        )
        messages = _convert_messages(req)
        assert [m["role"] for m in messages] == ["user", "tool"]
        assert messages[0]["content"] == "here is the result"
        assert messages[1]["tool_call_id"] == "tu_1"


class TestAnthropicStopReason:
    def test_stop_sequence_hit_maps_to_stop_sequence(self):
        # Engine sets done_reason="stop" only on a stop_sequences hit (#627).
        assert _anthropic_stop_reason("stop", False) == "stop_sequence"

    def test_natural_end_maps_to_end_turn(self):
        assert _anthropic_stop_reason(None, False) == "end_turn"

    def test_length_maps_to_max_tokens(self):
        assert _anthropic_stop_reason("length", False) == "max_tokens"
        assert _anthropic_stop_reason("timeout", False) == "max_tokens"

    def test_tools_take_precedence(self):
        assert _anthropic_stop_reason("stop", True) == "tool_use"


class TestBuildOptions:
    def test_empty(self):
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
        )
        opts = _build_options(req)
        assert opts == {}

    def test_all_options(self):
        req = AnthropicMessagesRequest(
            max_tokens=100,
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
    async def test_stop_sequence_hit_reports_stop_sequence(self, app_client):
        # A client stop_sequences match (engine done_reason="stop") must report
        # stop_reason "stop_sequence", not "end_turn" (#627).
        stats = TimingStats(prompt_eval_count=5, eval_count=3)
        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = {
                "text": "partial",
                "done": True,
                "done_reason": "stop",
                "stats": stats,
            }
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 50,
                    "stop_sequences": ["STOP"],
                },
            )
        assert resp.status_code == 200
        assert resp.json()["stop_reason"] == "stop_sequence"

    @pytest.mark.asyncio
    async def test_panel_model_dispatches_to_panel_coordinator(self, app_client):
        # A synthetic panel advertised in /v1/models must be dispatchable on
        # /v1/messages, not a 400 "model not found" (#627).
        stats = TimingStats(prompt_eval_count=1, eval_count=1)
        with (
            patch(
                "olmlx.routers.anthropic.panel_generate_chat", new_callable=AsyncMock
            ) as mock_panel,
            patch(
                "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
            ) as mock_gen,
            patch("olmlx.engine.registry.ModelRegistry.is_panel", return_value=True),
        ):
            mock_panel.return_value = {
                "text": "panel answer",
                "done": True,
                "stats": stats,
            }
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "my-panel",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 50,
                },
            )
        assert resp.status_code == 200
        assert resp.json()["content"][0]["text"] == "panel answer"
        mock_panel.assert_called_once()
        mock_gen.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_max_tokens_returns_400(self, app_client):
        """max_tokens is required per the Anthropic spec; omitting it must yield
        a 400 in the Anthropic error envelope, not a silent default 200."""
        resp = await app_client.post(
            "/v1/messages",
            json={"model": "qwen3", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"
        assert "max_tokens" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_count_tokens_without_max_tokens_succeeds(
        self, app_client, mock_loaded_model
    ):
        """The real Anthropic count_tokens endpoint takes no max_tokens field, so
        omitting it must still succeed (unlike /v1/messages)."""
        mock_loaded_model.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        resp = await app_client.post(
            "/v1/messages/count_tokens",
            json={"model": "qwen3", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] == 3

    @pytest.mark.asyncio
    async def test_non_streaming_literal_close_think_preserved_when_not_thinking(
        self, app_client
    ):
        """Issue #307 review: a non-thinking model that legitimately mentions
        the literal `</think>` token (e.g. explaining the syntax) must keep
        it in the text block on the non-streaming path, not have its prefix
        silently routed into a thinking block."""
        raw = "Use </think> to close the thought block."
        stats = TimingStats()
        mock_result = {
            "text": raw,
            "done": True,
            "stats": stats,
            "thinking_expected": False,
        }

        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "syntax?"}],
                    "max_tokens": 100,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        content_types = [b["type"] for b in data["content"]]
        assert "thinking" not in content_types
        assert any(b.get("text") == raw for b in data["content"])

    @pytest.mark.asyncio
    async def test_non_streaming_with_thinking(self, app_client):
        stats = TimingStats()
        mock_result = {
            "text": "<think>reasoning</think>The answer is 42.",
            "done": True,
            "stats": stats,
            "thinking_expected": True,
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
        # Issue #309: thinking content must be in `thinking` field, not `text`.
        thinking_block = next(b for b in data["content"] if b["type"] == "thinking")
        assert thinking_block["thinking"] == "reasoning"
        # `text` is omitted entirely on thinking blocks (response_model_exclude_none).
        assert "text" not in thinking_block
        # SDK expects `signature` as a string; emit empty for non-Claude models.
        assert thinking_block["signature"] == ""
        # `signature` must be omitted on non-thinking blocks per Anthropic spec.
        text_block = next(b for b in data["content"] if b["type"] == "text")
        assert "signature" not in text_block
        assert "thinking" not in text_block

    @pytest.mark.asyncio
    async def test_non_streaming_pre_extracted_thinking(self, app_client):
        """Issue #309: thinking can arrive pre-extracted from the engine via
        `result['thinking']` (no `<think>` tags in `text`). The router must
        still route it into the `thinking` field, not `text`."""
        stats = TimingStats()
        mock_result = {
            "text": "The answer is 42.",
            "thinking": "pre-extracted reasoning",
            "done": True,
            "stats": stats,
            "thinking_expected": True,
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
        thinking_block = next(b for b in data["content"] if b["type"] == "thinking")
        assert thinking_block["thinking"] == "pre-extracted reasoning"
        assert "text" not in thinking_block
        assert thinking_block["signature"] == ""

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
    async def test_non_streaming_forwards_cache_stats(self, app_client):
        """Issue #244: non-streaming response must forward cache_creation_tokens
        / cache_read_tokens from the engine result dict into AnthropicUsage,
        symmetric with the streaming path's cache_info events."""
        stats = TimingStats(prompt_eval_count=10, eval_count=20)
        mock_result = {
            "text": "Hello",
            "done": True,
            "stats": stats,
            "cache_read_tokens": 42,
            "cache_creation_tokens": 8,
        }

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
        usage = resp.json()["usage"]
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 20
        assert usage["cache_read_input_tokens"] == 42
        assert usage["cache_creation_input_tokens"] == 8

    @pytest.mark.asyncio
    async def test_non_streaming_cache_stats_default_zero(self, app_client):
        """When the engine omits cache stats (no prompt cache for this path),
        the response still reports them as 0 rather than raising."""
        stats = TimingStats(prompt_eval_count=5, eval_count=3)
        mock_result = {"text": "ok", "done": True, "stats": stats}

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
        usage = resp.json()["usage"]
        assert usage["cache_read_input_tokens"] == 0
        assert usage["cache_creation_input_tokens"] == 0

    @pytest.mark.asyncio
    async def test_non_streaming_cache_stats_none_coerced_to_zero(self, app_client):
        """If the engine sets cache keys to None, the router must coerce to 0
        rather than letting None reach AnthropicUsage's int field (Pydantic
        would raise ValidationError)."""
        stats = TimingStats(prompt_eval_count=5, eval_count=3)
        mock_result = {
            "text": "ok",
            "done": True,
            "stats": stats,
            "cache_read_tokens": None,
            "cache_creation_tokens": None,
        }

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
        usage = resp.json()["usage"]
        assert usage["cache_read_input_tokens"] == 0
        assert usage["cache_creation_input_tokens"] == 0

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
    async def test_streaming_reports_input_tokens(self, app_client):
        """Issue #610: the non-tools streaming path must report the real
        prompt token count in message_delta.usage, not a hardcoded 0."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Hello", "done": False}
                yield {
                    "text": "",
                    "done": True,
                    "stats": TimingStats(prompt_eval_count=17, eval_count=5),
                }

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
        usage = _message_delta_usage(resp.text)
        assert usage["input_tokens"] == 17
        assert usage["output_tokens"] == 5

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
        # Issue #309: SDK populates ThinkingBlock.signature from this delta.
        assert "signature_delta" in text

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
        # Issue #589: no spurious empty text block before the tool block.
        assert _content_block_start_types(text) == ["tool_use"]

    @pytest.mark.asyncio
    async def test_streaming_tool_only_no_spurious_text_block(self, app_client):
        """Issue #589: a tool call with no visible text must not emit a leading
        empty text content block — matching the non-streaming path."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {
                    "text": '<tool_call>{"name": "weather", "arguments": {"city": "NYC"}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=10)}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "Get weather NYC"}],
                    "max_tokens": 300,
                    "stream": True,
                    "tools": [
                        {
                            "name": "weather",
                            "description": "Get weather",
                            "input_schema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        assert _content_block_start_types(resp.text) == ["tool_use"]

    @pytest.mark.asyncio
    async def test_streaming_tool_with_text_preamble_has_two_blocks(self, app_client):
        """A visible text preamble before the tool call still emits a text block
        first (index 0), then the tool block (index 1)."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {
                    "text": 'Here is the answer <tool_call>{"name": "weather", "arguments": {"city": "NYC"}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=10)}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "Get weather NYC"}],
                    "max_tokens": 300,
                    "stream": True,
                    "tools": [
                        {
                            "name": "weather",
                            "description": "Get weather",
                            "input_schema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        assert _content_block_start_types(resp.text) == ["text", "tool_use"]
        assert "input_json_delta" in resp.text

    @pytest.mark.asyncio
    async def test_streaming_thinking_tool_no_visible_text_no_text_block(
        self, app_client
    ):
        """Thinking + tool call with no visible text: thinking then tool_use, no
        spurious empty text block between them."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {
                    "text": '<think>reasoning</think><tool_call>{"name": "weather", "arguments": {"city": "NYC"}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=10)}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "Get weather NYC"}],
                    "max_tokens": 300,
                    "stream": True,
                    "tools": [
                        {
                            "name": "weather",
                            "description": "Get weather",
                            "input_schema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        assert _content_block_start_types(resp.text) == ["thinking", "tool_use"]

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
        # Mid-stream end goes through `_flush_thinking_buffer` — also emits signature_delta.
        assert "signature_delta" in text

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
        # Buffered-tools path emits signature_delta via _emit_content_block.
        assert "signature_delta" in text

    @pytest.mark.asyncio
    async def test_streaming_orphaned_close_think_classified_as_thinking(
        self, app_client
    ):
        """Issue #307: Qwen3.5/3.6 emit thinking without ``<think>`` opener.

        Streaming-without-tools previously fell through the state machine's
        ``init`` branch (which only entered the thinking state when the
        buffer started with ``<think>``) and emitted the entire reasoning
        preamble as ``text_delta``. With the orphan-handling fix the prefix
        before ``</think>`` must be emitted as ``thinking_delta``.
        """
        # Multi-line preamble exceeds any plausible "give-up" buffer.
        preamble = (
            "Thinking Process:\n\n"
            "1. Analyze the Request: The user wants 17 * 23.\n"
            "2. Recall multiplication: 17 * 23 = 17 * (20 + 3) = 340 + 51 = 391.\n"
            "3. Sanity check: 17 * 25 = 425, minus 2*17 = 34, gives 391. OK.\n"
            "4. Format the answer: just the number, no prose.\n"
            "5. Construct Final Response: 391"
        )

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                # Many small chunks, like real generation
                for i in range(0, len(preamble), 16):
                    yield {"text": preamble[i : i + 16], "done": False}
                yield {"text": "\n</think>\n\n", "done": False}
                yield {"text": "391", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "17*23"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        # Reassemble the per-block content.
        thinking_text = ""
        visible_text = ""
        for line in resp.text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "content_block_delta":
                continue
            delta = payload.get("delta", {})
            if delta.get("type") == "thinking_delta":
                thinking_text += delta.get("thinking", "")
            elif delta.get("type") == "text_delta":
                visible_text += delta.get("text", "")

        assert "Thinking Process" in thinking_text
        assert "Sanity check" in thinking_text
        assert "Thinking Process" not in visible_text
        assert "</think>" not in visible_text
        assert "</think>" not in thinking_text
        assert visible_text.strip() == "391"
        # Orphan-close branch must also emit signature_delta before stop.
        assert "signature_delta" in resp.text

    @pytest.mark.asyncio
    async def test_streaming_text_before_standard_think_pair_not_eaten_as_orphan(
        self, app_client
    ):
        """Issue #307 review round 5: when the buffer assembles into
        ``preamble<think>...</think>answer`` (text before the opener — rare
        but real for slow first tokens), the state machine must NOT fire
        the orphan branch on the late `</think>` and swallow the `<think>`
        opener as part of a thinking content block. The opener-before-closer
        order signals a standard `<think>...</think>` pair.

        Since the rebuild on the shared splitter (issue #471), the mid-text
        ``<think>`` pair is detected and routed to a proper thinking block
        — matching the non-streaming path, which extracts the pair via
        ``parse_model_output`` — instead of leaking the literal tags into
        the text delta."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                # Single fat chunk so the state machine sees both tags at once
                yield {
                    "text": "preamble <think>thoughts</think>answer",
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
        thinking_text = ""
        visible_text = ""
        for line in resp.text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "content_block_delta":
                continue
            delta = payload.get("delta", {})
            if delta.get("type") == "thinking_delta":
                thinking_text += delta.get("thinking", "")
            elif delta.get("type") == "text_delta":
                visible_text += delta.get("text", "")

        # Critical: the orphan branch must NOT fire when `<think>` comes
        # first.  The pair is extracted as a thinking block; the
        # surrounding text stays text; no literal tag survives anywhere.
        assert "<think>" not in thinking_text
        assert "preamble" not in thinking_text
        assert thinking_text == "thoughts"
        assert "preamble" in visible_text
        assert "answer" in visible_text
        assert "<think>" not in visible_text
        assert "</think>" not in visible_text

    @pytest.mark.asyncio
    async def test_streaming_orphan_close_at_position_zero_no_empty_thinking_block(
        self, app_client
    ):
        """Issue #307 review round 10: when `</think>` is the very first
        token in the stream (close_idx == 0), the orphan branch must NOT
        emit an empty thinking content block — the non-streaming path
        skips it entirely, and emitting an empty block diverges from that
        behaviour."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                yield {"text": "</think>\n", "done": False}
                yield {"text": "The answer is 42.", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "?"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        # No content_block of type "thinking" should appear at all (the
        # non-streaming path produces none for an empty orphan prefix).
        for line in resp.text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "content_block_start":
                continue
            assert payload["content_block"]["type"] != "thinking", (
                "Empty orphan prefix must not emit a thinking content block"
            )

    @pytest.mark.asyncio
    async def test_streaming_no_think_whitespace_no_spurious_thinking_block(
        self, app_client
    ):
        """Issue #557: with `/no_think`, Qwen3 emits leading whitespace (`\\n\\n`)
        before an orphan `</think>`. The splitter classifies that whitespace as
        thinking content; the streaming path must NOT surface it as a thinking
        content block — the non-streaming path emits only a text block."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                yield {"text": "\n\n</think>", "done": False}
                yield {"text": "Yes.", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "Say: yes. /no_think"}],
                    "max_tokens": 20,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        block_types: list[str] = []
        thinking_text = ""
        text_text = ""
        for line in resp.text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "content_block_start":
                block_types.append(payload["content_block"]["type"])
            elif payload.get("type") == "content_block_delta":
                delta = payload.get("delta", {})
                if delta.get("type") == "thinking_delta":
                    thinking_text += delta.get("thinking", "")
                elif delta.get("type") == "text_delta":
                    text_text += delta.get("text", "")
        assert "thinking" not in block_types, (
            f"spurious thinking block emitted: {block_types}"
        )
        assert thinking_text == ""
        assert text_text == "Yes."

    @pytest.mark.asyncio
    async def test_streaming_overflow_when_close_think_never_arrives(self, app_client):
        """When `thinking_expected=True` but no `</think>` arrives before the
        buffer crosses `INIT_ORPHAN_DETECT_LIMIT`, the state machine must
        give up and emit the buffered content as text_delta rather than
        silently dropping it."""
        from olmlx.engine.inference import INIT_ORPHAN_DETECT_LIMIT

        long_text = "x" * (INIT_ORPHAN_DETECT_LIMIT + 50)

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                # Chunk in moderately-sized pieces.
                for i in range(0, len(long_text), 128):
                    yield {"text": long_text[i : i + 128], "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 8192,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        thinking_text = ""
        text_text = ""
        for line in resp.text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "content_block_delta":
                continue
            delta = payload.get("delta", {})
            if delta.get("type") == "thinking_delta":
                thinking_text += delta.get("thinking", "")
            elif delta.get("type") == "text_delta":
                text_text += delta.get("text", "")

        assert thinking_text == ""
        assert text_text == long_text

    @pytest.mark.asyncio
    async def test_streaming_thinking_expected_but_direct_answer(self, app_client):
        """Issue #307 review: when `thinking_expected=True` and the model
        produces a direct answer (no `<think>` / `</think>` tags at all),
        the state machine must still emit the buffered content as a
        `text_delta` rather than silently swallow it.

        Exercises the full SSE path through `_stream_thinking_state_machine`
        → init-state wait for orphan close → stream end → flush via
        `_flush_thinking_buffer`.
        """

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                # Direct, short answer — no <think> or </think> anywhere.
                yield {"text": "The answer is ", "done": False}
                yield {"text": "42.", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "what is 6*7?"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        thinking_text = ""
        text_text = ""
        for line in resp.text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "content_block_delta":
                continue
            delta = payload.get("delta", {})
            if delta.get("type") == "thinking_delta":
                thinking_text += delta.get("thinking", "")
            elif delta.get("type") == "text_delta":
                text_text += delta.get("text", "")

        assert thinking_text == ""
        assert text_text == "The answer is 42."

    @pytest.mark.asyncio
    async def test_streaming_literal_close_think_preserved_when_not_thinking(
        self, app_client
    ):
        """When `thinking_expected=False`, a literal `</think>` in the
        output (e.g. a non-thinking model explaining the syntax) must be
        kept as text rather than reclassified as a thinking block."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": False}
                yield {
                    "text": "Use </think> to close the thought block.",
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "syntax?"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        thinking_text = ""
        text_text = ""
        for line in resp.text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "content_block_delta":
                continue
            delta = payload.get("delta", {})
            if delta.get("type") == "thinking_delta":
                thinking_text += delta.get("thinking", "")
            elif delta.get("type") == "text_delta":
                text_text += delta.get("text", "")

        assert thinking_text == ""
        assert "</think>" in text_text
        assert text_text == "Use </think> to close the thought block."

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
    async def test_streaming_gemma4_channel_thinking_block(self, app_client):
        """Gemma 4's ``<|channel>thought\\n...<channel|>`` thinking must map
        to an Anthropic thinking block.  Capability gained when the state
        machine was rebuilt on the shared splitter (issue #471) — the old
        bespoke machine only knew the ``<think>`` spelling and leaked the
        channel markup into text deltas."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "<|channel>thought\n", "done": False}
                yield {"text": "pondering deeply", "done": False}
                yield {"text": "<channel|>", "done": False}
                yield {"text": "The answer is 391.", "done": False}
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
        thinking_text = ""
        visible_text = ""
        for line in resp.text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "content_block_delta":
                continue
            delta = payload.get("delta", {})
            if delta.get("type") == "thinking_delta":
                thinking_text += delta.get("thinking", "")
            elif delta.get("type") == "text_delta":
                visible_text += delta.get("text", "")

        assert thinking_text == "pondering deeply"
        assert visible_text == "The answer is 391."
        assert "<|channel>" not in resp.text
        assert "<channel|>" not in resp.text

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

    @pytest.mark.asyncio
    async def test_streaming_error_mid_stream(self, app_client):
        """Error during streaming emits an SSE error event instead of crashing."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "partial", "done": False}
                raise RuntimeError("GPU exploded")

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
        assert "event: error" in text
        error_data = None
        for line in text.split("\n"):
            if line.startswith("data:") and "api_error" in line:
                error_data = json.loads(line[5:])
                break
        assert error_data is not None
        assert error_data["type"] == "error"
        assert error_data["error"]["type"] == "api_error"
        assert "internal server error" in error_data["error"]["message"]


class TestEmptyMessagesRejected:
    @pytest.mark.asyncio
    async def test_messages_rejects_empty_messages_array(self, app_client):
        resp = await app_client.post(
            "/v1/messages",
            json={"model": "qwen3", "messages": [], "max_tokens": 100},
        )
        # The RequestValidationError handler maps Pydantic validation failures to
        # a 400 in the provider error envelope (Anthropic format here).
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"
        assert "messages cannot be empty" in data["error"]["message"]


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

        async def pings_then_passthrough(aiter, interval, ping):
            yield ping  # Ping fires before any real data
            async for item in aiter:
                yield item

        with (
            patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream),
            patch(
                "olmlx.routers.anthropic.with_keepalive_pings",
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

    @pytest.mark.asyncio
    async def test_cache_info_with_none_token_counts(self, app_client):
        """If a cache_info event arrives with explicit None values, the
        streaming path must coerce them to 0 — emitting JSON null would
        violate the Anthropic SSE protocol."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {
                    "cache_info": True,
                    "cache_read_tokens": None,
                    "cache_creation_tokens": None,
                }
                yield {"text": "Hello", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=1)}

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

        for line in text.split("\n"):
            if line.startswith("data:") and "message_start" in line:
                data = json.loads(line[5:])
                usage = data["message"]["usage"]
                assert usage["cache_read_input_tokens"] == 0
                assert usage["cache_creation_input_tokens"] == 0
                break
        else:
            pytest.fail("message_start event not found in SSE output")


class TestResolveAnthropicModel:
    def test_no_mapping_returns_unchanged(self):
        with patch(MAP_PATCH, []):
            assert _resolve_anthropic_model("claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_haiku_maps(self):
        with patch(MAP_PATCH, [("haiku", "qwen3:latest")]):
            assert (
                _resolve_anthropic_model("claude-haiku-4-5-20251001") == "qwen3:latest"
            )

    def test_sonnet_maps(self):
        with patch(MAP_PATCH, [("sonnet", "qwen3-8b:latest")]):
            assert _resolve_anthropic_model("claude-sonnet-4-6") == "qwen3-8b:latest"

    def test_opus_maps(self):
        with patch(MAP_PATCH, [("opus", "qwen3-30b:latest")]):
            assert _resolve_anthropic_model("claude-opus-4-6") == "qwen3-30b:latest"

    def test_case_insensitive(self):
        # Keys are pre-lowered by _build_anthropic_model_map; model name is lowered at runtime
        with patch(MAP_PATCH, [("sonnet", "qwen3:latest")]):
            assert _resolve_anthropic_model("Claude-SONNET-4-6") == "qwen3:latest"

    def test_no_match_falls_through(self):
        with patch(MAP_PATCH, [("haiku", "qwen3:latest")]):
            assert _resolve_anthropic_model("qwen3:latest") == "qwen3:latest"

    def test_longer_key_matches_first(self):
        """More-specific (longer) keys should take priority over shorter ones."""
        # Sorted descending by length: "sonnet" (6) before "son" (3)
        with patch(MAP_PATCH, [("sonnet", "long-match"), ("son", "short-match")]):
            assert _resolve_anthropic_model("claude-sonnet-4-6") == "long-match"

    def test_equal_length_keys_sorted_alphabetically(self):
        """Equal-length keys use alphabetical order for deterministic tie-breaking."""
        from olmlx.routers.anthropic import _build_anthropic_model_map

        with patch("olmlx.routers.anthropic.settings") as mock_settings:
            mock_settings.anthropic_models = {
                "llama": "llama-model",
                "haiku": "haiku-model",
            }
            result = _build_anthropic_model_map()
            # Both 5 chars — alphabetical: haiku before llama
            assert result == [("haiku", "haiku-model"), ("llama", "llama-model")]

    def test_empty_value_skipped(self):
        """_build_anthropic_model_map filters out empty values at build time."""
        from olmlx.routers.anthropic import _build_anthropic_model_map

        with patch("olmlx.routers.anthropic.settings") as mock_settings:
            mock_settings.anthropic_models = {"sonnet": ""}
            assert _build_anthropic_model_map() == []

    def test_empty_key_skipped(self):
        """_build_anthropic_model_map filters out empty keys at build time."""
        from olmlx.routers.anthropic import _build_anthropic_model_map

        with patch("olmlx.routers.anthropic.settings") as mock_settings:
            mock_settings.anthropic_models = {"": "rogue-model"}
            assert _build_anthropic_model_map() == []

    def test_segment_boundary_matching(self):
        """Keys match whole segments (split on - and :), not arbitrary substrings."""
        with patch(MAP_PATCH, [("net", "wrong-model")]):
            # "net" is a substring of "sonnet" but not a segment
            assert _resolve_anthropic_model("claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_segment_match_with_colon(self):
        """Colons are also segment boundaries (e.g. model:tag)."""
        with patch(MAP_PATCH, [("sonnet", "qwen3:latest")]):
            assert _resolve_anthropic_model("claude-sonnet:4-6") == "qwen3:latest"

    def test_whitespace_value_skipped(self):
        """Whitespace-only values should be filtered out at build time."""
        from olmlx.routers.anthropic import _build_anthropic_model_map

        with patch("olmlx.routers.anthropic.settings") as mock_settings:
            mock_settings.anthropic_models = {"sonnet": "   "}
            assert _build_anthropic_model_map() == []


class TestAnthropicModelResolution:
    """Integration tests: resolver applied in endpoints, response echoes original model."""

    @pytest.mark.asyncio
    async def test_non_streaming_resolves_model(self, app_client):
        """generate_chat receives resolved model name, response has original."""
        stats = TimingStats(prompt_eval_count=5, eval_count=10)
        mock_result = {"text": "Hello!", "done": True, "stats": stats}

        with (
            patch(MAP_PATCH, [("sonnet", "qwen3-8b:latest")]),
            patch(
                "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
            ) as mock_gen,
        ):
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        # Response echoes original Claude model name
        assert data["model"] == "claude-sonnet-4-6"
        # generate_chat was called with the resolved local model
        call_args = mock_gen.call_args
        assert call_args[0][1] == "qwen3-8b:latest"

    @pytest.mark.asyncio
    async def test_streaming_resolves_model(self, app_client):
        """Streaming: generate_chat receives resolved name, SSE has original."""

        async def fake_stream(*args, **kwargs):
            yield {"text": "Hi", "done": False}
            yield {"done": True, "stats": TimingStats(eval_count=5)}

        with (
            patch(MAP_PATCH, [("haiku", "qwen3:latest")]),
            patch(
                "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
            ) as mock_gen,
        ):
            mock_gen.return_value = fake_stream()
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        # Check message_start echoes original model name
        for line in resp.text.split("\n"):
            if line.startswith("data:") and "message_start" in line:
                data = json.loads(line[5:])
                assert data["message"]["model"] == "claude-haiku-4-5-20251001"
                break
        else:
            pytest.fail("message_start event not found")
        # generate_chat was called with resolved model
        call_args = mock_gen.call_args
        assert call_args[0][1] == "qwen3:latest"

    @pytest.mark.asyncio
    async def test_count_tokens_resolves_model(self, app_client):
        """count_tokens endpoint resolves Claude model names."""
        manager = app_client._transport.app.state.model_manager  # type: ignore[union-attr]
        with (
            patch(MAP_PATCH, [("sonnet", "qwen3-8b:latest")]),
            patch("olmlx.routers.anthropic.count_chat_tokens", return_value=42),
            patch.object(
                manager, "ensure_loaded", new_callable=AsyncMock
            ) as mock_ensure,
        ):
            mock_lm = MagicMock()
            mock_lm.active_refs = 0
            mock_ensure.return_value = mock_lm
            resp = await app_client.post(
                "/v1/messages/count_tokens",
                json={
                    "model": "claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                },
            )

        assert resp.status_code == 200
        mock_ensure.assert_called_with("qwen3-8b:latest", pin=True)


class TestXCacheIDHeader:
    @pytest.mark.asyncio
    async def test_header_passed_to_generate_chat(self, app_client):
        stats = TimingStats(prompt_eval_count=10, eval_count=5)
        mock_result = {"text": "response", "done": True, "stats": stats}

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
                headers={"X-Cache-ID": "agent-alpha"},
            )

        assert resp.status_code == 200
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs.get("cache_id") == "agent-alpha"

    @pytest.mark.asyncio
    async def test_no_header_uses_default_cache_id(self, app_client):
        stats = TimingStats(prompt_eval_count=10, eval_count=5)
        mock_result = {"text": "response", "done": True, "stats": stats}

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
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs.get("cache_id") == ""


class TestThinkingParamSchema:
    def test_thinking_enabled(self):
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            thinking=AnthropicThinkingParam(type="enabled", budget_tokens=10000),
        )
        assert req.thinking is not None
        assert req.thinking.type == "enabled"
        assert req.thinking.budget_tokens == 10000

    def test_thinking_disabled(self):
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            thinking=AnthropicThinkingParam(type="disabled"),
        )
        assert req.thinking is not None
        assert req.thinking.type == "disabled"
        assert req.thinking.budget_tokens is None

    def test_thinking_missing(self):
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
        )
        assert req.thinking is None

    def test_thinking_from_dict(self):
        """Schema parses thinking from raw dict (as JSON would arrive)."""
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            thinking={"type": "enabled", "budget_tokens": 5000},
        )
        assert req.thinking.type == "enabled"
        assert req.thinking.budget_tokens == 5000

    def test_thinking_adaptive(self):
        """Schema accepts 'adaptive' type (used by Claude Code)."""
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            thinking={"type": "adaptive"},
        )
        assert req.thinking.type == "adaptive"

    def test_thinking_unknown_type_accepted(self):
        """Schema accepts unknown types for forward compatibility."""
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            thinking={"type": "some_future_type"},
        )
        assert req.thinking.type == "some_future_type"

    def test_thinking_extra_fields_accepted(self):
        """Unknown fields in thinking param are accepted for forward compatibility."""
        req = AnthropicMessagesRequest(
            max_tokens=100,
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            thinking={"type": "enabled", "budget_tokens": 5000, "new_field": "value"},
        )
        assert req.thinking.type == "enabled"


class TestThinkingParamRouter:
    @pytest.mark.asyncio
    async def test_thinking_enabled_passes_to_generate(self, app_client):
        stats = TimingStats(prompt_eval_count=10, eval_count=20)
        mock_result = {"text": "Hello!", "done": True, "stats": stats}

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
                    "thinking": {"type": "enabled", "budget_tokens": 10000},
                },
            )

        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs.get("enable_thinking") is True

    @pytest.mark.asyncio
    async def test_thinking_disabled_passes_to_generate(self, app_client):
        stats = TimingStats(prompt_eval_count=10, eval_count=20)
        mock_result = {"text": "Hello!", "done": True, "stats": stats}

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
                    "thinking": {"type": "disabled"},
                },
            )

        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs.get("enable_thinking") is False

    @pytest.mark.asyncio
    async def test_no_thinking_passes_none(self, app_client):
        stats = TimingStats(prompt_eval_count=10, eval_count=20)
        mock_result = {"text": "Hello!", "done": True, "stats": stats}

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
        assert mock_gen.call_args.kwargs.get("enable_thinking") is None

    @pytest.mark.asyncio
    async def test_thinking_adaptive_passes_true(self, app_client):
        """'adaptive' type maps to enable_thinking=True."""
        stats = TimingStats(prompt_eval_count=10, eval_count=20)
        mock_result = {"text": "Hello!", "done": True, "stats": stats}

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
                    "thinking": {"type": "adaptive"},
                },
            )

        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs.get("enable_thinking") is True

    @pytest.mark.asyncio
    async def test_thinking_unknown_type_passes_none(self, app_client):
        """Unknown thinking types map to enable_thinking=None for forward compat."""
        stats = TimingStats(prompt_eval_count=10, eval_count=20)
        mock_result = {"text": "Hello!", "done": True, "stats": stats}

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
                    "thinking": {"type": "some_future_type"},
                },
            )

        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs.get("enable_thinking") is None

    @pytest.mark.asyncio
    async def test_thinking_enabled_with_tools(self, app_client):
        """Thinking enabled + tools should pass enable_thinking=True (not hardcoded off)."""
        stats = TimingStats(prompt_eval_count=10, eval_count=20)
        mock_result = {
            "text": "<think>reasoning</think>I'll search.",
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
                    "thinking": {"type": "enabled", "budget_tokens": 10000},
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
        assert mock_gen.call_args.kwargs.get("enable_thinking") is True

    @pytest.mark.asyncio
    async def test_streaming_thinking_enabled(self, app_client):
        """Streaming: thinking parameter is forwarded to generate_chat."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Hello", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=1)}

            return gen()

        with patch(
            "olmlx.routers.anthropic.generate_chat", side_effect=mock_stream
        ) as mock_gen:
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stream": True,
                    "thinking": {"type": "enabled", "budget_tokens": 8000},
                },
            )

        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs.get("enable_thinking") is True

    @pytest.mark.asyncio
    async def test_count_tokens_passes_thinking(self, app_client, mock_loaded_model):
        """count_tokens endpoint passes enable_thinking to count_chat_tokens."""
        mock_loaded_model.tokenizer.apply_chat_template.return_value = [1, 2, 3]

        with patch(
            "olmlx.routers.anthropic.count_chat_tokens", return_value=3
        ) as mock_count:
            resp = await app_client.post(
                "/v1/messages/count_tokens",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "thinking": {"type": "enabled", "budget_tokens": 5000},
                },
            )

        assert resp.status_code == 200
        assert mock_count.call_args.kwargs.get("enable_thinking") is True


class TestStreamSseEarlyMessageStart:
    """Tests for early message_start emission and ping flow during prefill."""

    @pytest.mark.asyncio
    async def test_cache_stats_in_message_start(self, app_client):
        """message_start should include cache stats from cache_info."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {
                    "cache_info": True,
                    "cache_read_tokens": 100,
                    "cache_creation_tokens": 50,
                }
                yield {"text": "Hello", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=1)}

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
        assert '"cache_read_input_tokens": 100' in text
        assert '"cache_creation_input_tokens": 50' in text

    @pytest.mark.asyncio
    async def test_pings_delivered_between_message_start_and_content(self, app_client):
        """Pings during prefill should appear between message_start and content_block_start."""
        import re

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {
                    "cache_info": True,
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 200,
                }
                # Short delay with fast ping interval to avoid 12s wall-clock wait
                await asyncio.sleep(0.3)
                yield {"text": "Done", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=1)}

            return gen()

        with (
            patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream),
            patch("olmlx.routers.anthropic.KEEPALIVE_PING_INTERVAL", 0.1),
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
        events = re.findall(r"event: (\w+)", text)
        assert events[0] == "message_start"
        content_idx = events.index("content_block_start")
        between = events[1:content_idx]
        assert any(e == "ping" for e in between), (
            f"Expected pings between message_start and content_block_start, got: {between}"
        )


class _CloseCountingStream:
    """Wraps an async generator to count aclose() calls."""

    def __init__(self, agen):
        self._agen = agen
        self.close_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._agen.__anext__()

    async def aclose(self):
        self.close_count += 1
        await self._agen.aclose()


class TestStreamCloseOnce:
    """The raw MLX stream (result) must be closed exactly once by stream_sse's finally block."""

    @pytest.mark.asyncio
    async def test_buffered_with_tools_closes_once(self, app_client):
        """Buffered-with-tools path (has_tools=True): result.aclose called once."""
        tracker = None

        async def mock_stream(*args, **kwargs):
            nonlocal tracker

            async def gen():
                yield {
                    "text": '<tool_call>{"name": "search", "arguments": {"q": "t"}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=5)}

            tracker = _CloseCountingStream(gen())
            return tracker

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
        assert tracker.close_count == 1, (
            f"Expected result.aclose() called once, got {tracker.close_count}"
        )

    @pytest.mark.asyncio
    async def test_buffered_with_tools_closes_once_on_error(self, app_client):
        """Buffered-with-tools path: result.aclose called once even on mid-stream error."""
        tracker = None

        async def mock_stream(*args, **kwargs):
            nonlocal tracker

            async def gen():
                yield {"text": "partial", "done": False}
                raise RuntimeError("GPU exploded")

            tracker = _CloseCountingStream(gen())
            return tracker

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
        assert "event: error" in resp.text
        assert tracker.close_count == 1, (
            f"Expected result.aclose() called once on error, got {tracker.close_count}"
        )

    @pytest.mark.asyncio
    async def test_thinking_state_machine_closes_once_on_error(self, app_client):
        """Thinking state machine path: result.aclose called once even on mid-stream error."""
        tracker = None

        async def mock_stream(*args, **kwargs):
            nonlocal tracker

            async def gen():
                yield {"text": "<think>partial", "done": False}
                raise RuntimeError("GPU exploded")

            tracker = _CloseCountingStream(gen())
            return tracker

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
        assert "event: error" in resp.text
        assert tracker.close_count == 1, (
            f"Expected result.aclose() called once on error, got {tracker.close_count}"
        )

    @pytest.mark.asyncio
    async def test_thinking_state_machine_closes_once(self, app_client):
        """Thinking state machine path (has_tools=False): result.aclose called once."""
        tracker = None

        async def mock_stream(*args, **kwargs):
            nonlocal tracker

            async def gen():
                yield {"text": "<think>reasoning</think>answer", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            tracker = _CloseCountingStream(gen())
            return tracker

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
        assert tracker.close_count == 1, (
            f"Expected result.aclose() called once, got {tracker.close_count}"
        )


class TestAnthropicStreamingCleanup:
    """Tests for streaming error handling and cleanup in the Anthropic router."""

    async def test_streaming_error_emits_error_event(self, app_client):
        """When the model stream raises mid-stream, an SSE error event is emitted."""
        from tests.conftest import make_error_stream
        from tests.integration.conftest import parse_sse_events

        error_stream = make_error_stream(
            [{"text": "partial"}], error_msg="GPU error mid-stream"
        )

        with patch(
            "olmlx.routers.anthropic.generate_chat",
            return_value=error_stream,
        ):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert resp.status_code == 200
        events = parse_sse_events(resp.text)
        # Should contain an error event
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) >= 1
        error_data = error_events[0]["data"]
        assert error_data["type"] == "error"
        assert error_data["error"]["type"] == "api_error"

    async def test_streaming_cleanup_on_error(self, app_client):
        """Verify result.aclose() is called even when the inner generator raises."""
        from tests.conftest import make_error_stream

        error_stream = make_error_stream(
            [{"text": "partial"}], error_msg="GPU error mid-stream"
        )

        with patch(
            "olmlx.routers.anthropic.generate_chat",
            return_value=error_stream,
        ):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert resp.status_code == 200
        # The finally block in stream_sse should call result.aclose()
        error_stream.aclose.assert_awaited_once()


class TestResolveToolNames:
    """Test _resolve_tool_names mapping parsed names to declared tool names."""

    def test_exact_match_unchanged(self):
        from olmlx.engine.tool_parser import resolve_tool_names

        tools = [{"function": {"name": "Bash"}}]
        uses = [{"name": "Bash", "id": "1", "input": {}}]
        resolve_tool_names(uses, tools)
        assert uses[0]["name"] == "Bash"

    def test_case_insensitive_match(self):
        from olmlx.engine.tool_parser import resolve_tool_names

        tools = [{"function": {"name": "Bash"}}]
        uses = [{"name": "bash", "id": "1", "input": {}}]
        resolve_tool_names(uses, tools)
        assert uses[0]["name"] == "Bash"

    def test_colon_prefix_match(self):
        """Gemma 4 generates 'bash:run_command' for tool 'Bash'."""
        from olmlx.engine.tool_parser import resolve_tool_names

        tools = [
            {"function": {"name": "Bash"}},
            {"function": {"name": "Read"}},
        ]
        uses = [{"name": "bash:run_command", "id": "1", "input": {}}]
        resolve_tool_names(uses, tools)
        assert uses[0]["name"] == "Bash"

    def test_no_declared_tools(self):
        from olmlx.engine.tool_parser import resolve_tool_names

        uses = [{"name": "bash", "id": "1", "input": {}}]
        resolve_tool_names(uses, None)
        assert uses[0]["name"] == "bash"

    def test_no_match_unchanged(self):
        from olmlx.engine.tool_parser import resolve_tool_names

        tools = [{"function": {"name": "Read"}}]
        uses = [{"name": "unknown_tool", "id": "1", "input": {}}]
        resolve_tool_names(uses, tools)
        assert uses[0]["name"] == "unknown_tool"

    def test_multiple_tool_uses(self):
        from olmlx.engine.tool_parser import resolve_tool_names

        tools = [
            {"function": {"name": "Bash"}},
            {"function": {"name": "Read"}},
            {"function": {"name": "Glob"}},
        ]
        uses = [
            {"name": "bash:execute", "id": "1", "input": {}},
            {"name": "Read", "id": "2", "input": {}},
            {"name": "glob", "id": "3", "input": {}},
        ]
        resolve_tool_names(uses, tools)
        assert uses[0]["name"] == "Bash"
        assert uses[1]["name"] == "Read"
        assert uses[2]["name"] == "Glob"


class TestStreamBufferedWithTools:
    """Tests for _stream_buffered_with_tools text accumulation."""

    @pytest.mark.asyncio
    async def test_many_chunks_accumulated_correctly(self):
        """500+ chunks should be accumulated into correct full_text."""
        from olmlx.routers.anthropic import _stream_buffered_with_tools

        words = [f"word{i} " for i in range(500)]

        async def fake_result():
            for word in words:
                yield {"text": word}
            yield {"done": True, "stats": MagicMock(eval_count=500)}

        events = []
        async for event in _stream_buffered_with_tools(fake_result()):
            events.append(event)

        # Find the text content block events and reconstruct
        text_deltas = []
        for event in events:
            if isinstance(event, str) and '"text_delta"' in event:
                # Parse SSE event to extract text
                for line in event.split("\n"):
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            text_deltas.append(data["delta"]["text"])

        full_text = "".join(text_deltas)
        expected = "".join(words)
        # parse_model_output may strip trailing whitespace; check content is preserved
        assert full_text == expected or full_text == expected.strip()


# NOTE: TestFlushThinkingBuffer was removed with the bespoke
# ``_flush_thinking_buffer`` helper (issue #471) — the state machine is now
# built on ``routers/thinking_split.py`` and its end-of-stream flush
# behaviours are pinned at the endpoint level (``test_streaming_empty_output``,
# ``test_streaming_thinking_ends_midstream``,
# ``test_streaming_thinking_expected_but_direct_answer``,
# ``test_streaming_no_output_at_all``).


class TestConvertMessagesImages:
    def test_anthropic_converts_base64_image_block(self):
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "QQ==",
                            },
                        },
                    ],
                }
            ],
        )
        msgs = _convert_messages(req)
        user = [m for m in msgs if m["role"] == "user"][0]
        assert user["content"] == "describe"
        assert user["images"] == ["data:image/jpeg;base64,QQ=="]

    def test_anthropic_image_only_block_creates_user_message(self):
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "data": "QQ=="}}
                    ],
                }
            ],
        )
        msgs = _convert_messages(req)
        user = [m for m in msgs if m["role"] == "user"][0]
        assert user["content"] == ""
        assert user["images"] == ["data:image/png;base64,QQ=="]

    def test_anthropic_text_only_block_has_no_images_key(self):
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        )
        msgs = _convert_messages(req)
        user = [m for m in msgs if m["role"] == "user"][0]
        assert user["content"] == "hello"
        assert "images" not in user

    def test_anthropic_url_source_block(self):
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/img.png",
                            },
                        },
                    ],
                }
            ],
        )
        msgs = _convert_messages(req)
        user = [m for m in msgs if m["role"] == "user"][0]
        assert user["images"] == ["https://example.com/img.png"]

    def test_anthropic_preserves_image_order(self):
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "url", "url": "u/a.png"}},
                        {"type": "image", "source": {"type": "url", "url": "u/b.png"}},
                    ],
                }
            ],
        )
        msgs = _convert_messages(req)
        user = [m for m in msgs if m["role"] == "user"][0]
        assert user["images"] == ["u/a.png", "u/b.png"]

    def test_anthropic_malformed_image_source_raises(self):
        """A base64 source missing 'data' surfaces as ValueError (handler -> 422)."""
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "base64"}}],
                }
            ],
        )
        with pytest.raises(ValueError, match="base64"):
            _convert_messages(req)


def test_convert_messages_collects_audio_block():
    from olmlx.routers.anthropic import _convert_messages
    from olmlx.schemas.anthropic import AnthropicMessagesRequest

    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=16,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe"},
                    {
                        "type": "audio",
                        "source": {
                            "type": "base64",
                            "media_type": "audio/wav",
                            "data": "QQ==",
                        },
                    },
                ],
            }
        ],
    )
    msgs = _convert_messages(req)
    user = [m for m in msgs if m["role"] == "user"][0]
    assert user["audio"] == ["data:audio/wav;base64,QQ=="]


class TestToolChoiceHonored:
    """Issue #620: on /v1/messages ``tool_choice`` was accepted and ignored.
    ``{"type":"none"}`` must suppress tool calls; ``"any"``/forced ``"tool"``
    must 400."""

    TOOLS = [
        {
            "name": "search",
            "description": "Search",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        }
    ]
    TOOL_OUTPUT = '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'

    @pytest.mark.asyncio
    async def test_none_suppresses_tool_use(self, app_client):
        mock_result = {"text": self.TOOL_OUTPUT, "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "search"}],
                    "max_tokens": 100,
                    "tools": self.TOOLS,
                    "tool_choice": {"type": "none"},
                },
            )
        assert resp.status_code == 200
        blocks = resp.json()["content"]
        assert not any(b["type"] == "tool_use" for b in blocks)
        # Tools must not be forwarded to the engine when the client forced text.
        assert mock_gen.call_args.kwargs["tools"] is None

    @pytest.mark.asyncio
    async def test_any_is_rejected_with_400(self, app_client):
        resp = await app_client.post(
            "/v1/messages",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "tools": self.TOOLS,
                "tool_choice": {"type": "any"},
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_forced_tool_is_rejected_with_400(self, app_client):
        resp = await app_client.post(
            "/v1/messages",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "tools": self.TOOLS,
                "tool_choice": {"type": "tool", "name": "search"},
            },
        )
        assert resp.status_code == 400
