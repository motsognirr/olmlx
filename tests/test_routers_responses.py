"""Tests for olmlx.routers.responses and its schemas."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from olmlx.schemas.responses import ResponsesRequest
from olmlx.utils.timing import TimingStats
from olmlx.routers.responses import (
    _build_input_messages,
    _convert_tools,
    _grammar_from_text_format,
    _resolve_reasoning,
)


class TestResponsesRequestSchema:
    def test_string_input_accepted(self):
        req = ResponsesRequest(model="qwen3", input="hello")
        assert req.input == "hello"
        assert req.store is True
        assert req.stream is False

    def test_list_input_accepted(self):
        req = ResponsesRequest(
            model="qwen3",
            input=[{"role": "user", "content": "hi"}],
        )
        assert isinstance(req.input, list)

    def test_defaults(self):
        req = ResponsesRequest(model="qwen3", input="x")
        assert req.previous_response_id is None
        assert req.tools is None
        assert req.max_output_tokens is None

    def test_empty_input_rejected(self):
        import pytest

        with pytest.raises(Exception):
            ResponsesRequest(model="qwen3", input="")
        with pytest.raises(Exception):
            ResponsesRequest(model="qwen3", input=[])


class TestTranslation:
    def test_string_input(self):
        msgs = _build_input_messages("hello")
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_message_item_string_content(self):
        msgs = _build_input_messages([{"role": "user", "content": "hi"}])
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_message_item_text_parts(self):
        msgs = _build_input_messages(
            [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
        )
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_message_item_image_part(self):
        msgs = _build_input_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {"type": "input_image", "image_url": "http://x/y.png"},
                    ],
                }
            ]
        )
        assert msgs[0]["content"] == "describe"
        assert msgs[0]["images"] == ["http://x/y.png"]

    def test_function_call_output_item(self):
        msgs = _build_input_messages(
            [{"type": "function_call_output", "call_id": "call_1", "output": "42"}]
        )
        assert msgs == [{"role": "tool", "tool_call_id": "call_1", "content": "42"}]

    def test_function_call_item(self):
        msgs = _build_input_messages(
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "SF"}',
                }
            ]
        )
        assert msgs[0]["role"] == "assistant"
        tc = msgs[0]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["id"] == "call_1"

    def test_unknown_item_type_raises(self):
        with pytest.raises(ValueError):
            _build_input_messages([{"type": "mystery"}])

    def test_convert_function_tool(self):
        tools = _convert_tools(
            [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "d",
                    "parameters": {"type": "object"},
                }
            ]
        )
        assert tools[0]["function"]["name"] == "get_weather"
        assert tools[0]["type"] == "function"

    def test_builtin_tool_rejected(self):
        with pytest.raises(ValueError):
            _convert_tools([{"type": "web_search"}])

    def test_grammar_json_object(self):
        spec = _grammar_from_text_format({"format": {"type": "json_object"}})
        assert spec is not None

    def test_grammar_none_for_text(self):
        assert _grammar_from_text_format({"format": {"type": "text"}}) is None
        assert _grammar_from_text_format(None) is None

    def test_resolve_reasoning_presence(self):
        assert _resolve_reasoning({"effort": "high"}) is True
        assert _resolve_reasoning({"effort": "none"}) is False
        assert _resolve_reasoning(None) is None

    def test_function_call_missing_name_raises(self):
        with pytest.raises(ValueError):
            _build_input_messages([{"type": "function_call", "call_id": "c1"}])

    def test_function_tool_missing_name_raises(self):
        with pytest.raises(ValueError):
            _convert_tools([{"type": "function", "parameters": {}}])


class TestNonStreamingText:
    @pytest.mark.asyncio
    async def test_text_response_shape(self, app_client):
        stats = TimingStats(prompt_eval_count=5, eval_count=3)
        mock_result = {"text": "Hello there.", "done": True, "stats": stats}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "stream": False},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert data["id"].startswith("resp_")
        msg = next(it for it in data["output"] if it["type"] == "message")
        assert msg["role"] == "assistant"
        assert msg["content"][0]["type"] == "output_text"
        assert msg["content"][0]["text"] == "Hello there."
        assert data["usage"]["input_tokens"] == 5
        assert data["usage"]["output_tokens"] == 3
        assert data["usage"]["total_tokens"] == 8

    @pytest.mark.asyncio
    async def test_timeout_marks_incomplete(self, app_client):
        mock_result = {
            "text": "partial",
            "done": True,
            "done_reason": "timeout",
            "stats": TimingStats(),
        }
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi"},
            )
        data = resp.json()
        assert data["status"] == "incomplete"
        assert data["incomplete_details"]["reason"] == "max_output_tokens"


class TestNonStreamingFeatures:
    @pytest.mark.asyncio
    async def test_function_call_item_emitted(self, app_client):
        raw = '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>'
        mock_result = {"text": raw, "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "weather in SF?",
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ],
                },
            )
        data = resp.json()
        fc = next(it for it in data["output"] if it["type"] == "function_call")
        assert fc["name"] == "get_weather"
        assert json.loads(fc["arguments"]) == {"city": "SF"}
        assert fc["call_id"].startswith("call_")

    @pytest.mark.asyncio
    async def test_reasoning_item_from_think(self, app_client):
        mock_result = {
            "text": "<think>step one</think>The answer is 42.",
            "done": True,
            "stats": TimingStats(),
            "thinking_expected": True,
        }
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "q", "reasoning": {"effort": "high"}},
            )
        data = resp.json()
        reasoning = next(it for it in data["output"] if it["type"] == "reasoning")
        assert "step one" in reasoning["content"][0]["text"]
        msg = next(it for it in data["output"] if it["type"] == "message")
        assert msg["content"][0]["text"] == "The answer is 42."
        assert mock_gen.call_args.kwargs["enable_thinking"] is True

    @pytest.mark.asyncio
    async def test_structured_output_threads_grammar(self, app_client):
        mock_result = {"text": "{}", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "q",
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "thing",
                            "schema": {"type": "object"},
                        }
                    },
                },
            )
        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs["grammar_spec"] is not None

    @pytest.mark.asyncio
    async def test_image_input_threads_images(self, app_client):
        mock_result = {"text": "a cat", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "what is this?"},
                                {
                                    "type": "input_image",
                                    "image_url": "http://x/cat.png",
                                },
                            ],
                        }
                    ],
                },
            )
        assert resp.status_code == 200
        sent_messages = mock_gen.call_args.args[2]
        assert sent_messages[0]["images"] == ["http://x/cat.png"]

    @pytest.mark.asyncio
    async def test_builtin_tool_rejected_422(self, app_client):
        resp = await app_client.post(
            "/v1/responses",
            json={"model": "qwen3", "input": "q", "tools": [{"type": "web_search"}]},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_unknown_input_item_422(self, app_client):
        resp = await app_client.post(
            "/v1/responses",
            json={"model": "qwen3", "input": [{"type": "mystery"}]},
        )
        assert resp.status_code == 422
