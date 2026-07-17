"""Tests for olmlx.routers.responses and its schemas."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from olmlx.engine.responses_state import get_store
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

    def test_function_call_missing_arguments_defaults_to_empty_object(self):
        # A missing/empty arguments must become "{}" (valid JSON), not "".
        msgs = _build_input_messages(
            [{"type": "function_call", "call_id": "c1", "name": "f"}]
        )
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "{}"


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


class TestStateContinuation:
    @pytest.fixture(autouse=True)
    def _clear_store(self):
        get_store().clear()
        yield
        get_store().clear()

    @pytest.mark.asyncio
    async def test_response_is_stored(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses", json={"model": "qwen3", "input": "hi"}
            )
        rid = resp.json()["id"]
        assert get_store().get(rid) is not None

    @pytest.mark.asyncio
    async def test_store_false_skips(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "store": False},
            )
        rid = resp.json()["id"]
        assert get_store().get(rid) is None

    @pytest.mark.asyncio
    async def test_continuation_prepends_history_and_threads_cache_id(self, app_client):
        first = {"text": "Blue.", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = first
            r1 = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "favorite color?"},
            )
        rid = r1.json()["id"]

        second = {"text": "Because.", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = second
            await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "why?",
                    "previous_response_id": rid,
                },
            )
            sent_messages = mock_gen.call_args.args[2]
            roles = [m["role"] for m in sent_messages]
            assert roles == ["user", "assistant", "user"]
            assert sent_messages[1]["content"] == "Blue."
            assert sent_messages[-1]["content"] == "why?"
            assert mock_gen.call_args.kwargs["cache_id"] == rid[:256]

    @pytest.mark.asyncio
    async def test_unknown_previous_id_404(self, app_client):
        resp = await app_client.post(
            "/v1/responses",
            json={"model": "qwen3", "input": "x", "previous_response_id": "resp_nope"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_three_turn_chain(self, app_client):
        async def post(json_body):
            mock_result = {"text": "ok", "done": True, "stats": TimingStats()}
            with patch(
                "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
            ) as mock_gen:
                mock_gen.return_value = mock_result
                resp = await app_client.post("/v1/responses", json=json_body)
                return resp.json()["id"], mock_gen

        id1, _ = await post({"model": "qwen3", "input": "one"})
        id2, _ = await post(
            {"model": "qwen3", "input": "two", "previous_response_id": id1}
        )

        mock_result = {"text": "ok", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "three", "previous_response_id": id2},
            )
            sent = mock_gen.call_args.args[2]
            roles = [m["role"] for m in sent]
            assert roles == ["user", "assistant", "user", "assistant", "user"]
            contents = [m.get("content") for m in sent]
            assert contents[0] == "one"
            assert contents[2] == "two"
            assert contents[4] == "three"

    @pytest.mark.asyncio
    async def test_instructions_not_carried_across_turns(self, app_client):
        # Turn 1 with instructions.
        mock_result = {"text": "ok", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            r1 = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "instructions": "Be terse."},
            )
        id1 = r1.json()["id"]

        # Turn 2 continuation with NO instructions -> no system message at all.
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "more", "previous_response_id": id1},
            )
            sent = mock_gen.call_args.args[2]
            assert all(m["role"] != "system" for m in sent), sent

    @pytest.mark.asyncio
    async def test_instructions_replace_not_stack_on_continuation(self, app_client):
        mock_result = {"text": "ok", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            r1 = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "instructions": "First."},
            )
        id1 = r1.json()["id"]

        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "more",
                    "previous_response_id": id1,
                    "instructions": "Second.",
                },
            )
            sent = mock_gen.call_args.args[2]
            systems = [m for m in sent if m["role"] == "system"]
            assert len(systems) == 1
            assert systems[0]["content"] == "Second."


def _parse_sse(body: str) -> list[dict]:
    """Parse an SSE body into a list of {event, data} dicts."""
    events = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        events.append({"event": event, "data": data})
    return events


class TestStreaming:
    @pytest.mark.asyncio
    async def test_text_stream_event_sequence(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Hel", "done": False}
                yield {"text": "lo", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=2)}

            return gen()

        with patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "stream": True},
            )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        types = [e["event"] for e in events]
        assert types[0] == "response.created"
        assert "response.output_text.delta" in types
        assert types[-1] == "response.completed"
        seqs = [e["data"]["sequence_number"] for e in events]
        assert seqs == sorted(seqs)
        text = "".join(
            e["data"]["delta"]
            for e in events
            if e["event"] == "response.output_text.delta"
        )
        assert text == "Hello"
        final = events[-1]["data"]["response"]
        assert final["status"] == "completed"

    @pytest.mark.asyncio
    async def test_tool_call_stream(self, app_client):
        raw = '<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": raw, "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "go",
                    "stream": True,
                    "tools": [
                        {
                            "type": "function",
                            "name": "f",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            )
        events = _parse_sse(resp.text)
        types = [e["event"] for e in events]
        assert "response.function_call_arguments.delta" in types
        assert "response.function_call_arguments.done" in types
        final = events[-1]["data"]["response"]
        fc = next(it for it in final["output"] if it["type"] == "function_call")
        assert fc["name"] == "f"

    @pytest.mark.asyncio
    async def test_stream_stores_response(self, app_client):
        from olmlx.engine.responses_state import get_store

        get_store().clear()

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "hi", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "stream": True},
            )
        events = _parse_sse(resp.text)
        rid = events[-1]["data"]["response"]["id"]
        assert get_store().get(rid) is not None
        get_store().clear()

    @pytest.mark.asyncio
    async def test_stream_timeout_incomplete(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "partial", "done": False}
                yield {
                    "text": "",
                    "done": True,
                    "done_reason": "timeout",
                    "stats": TimingStats(),
                }

            return gen()

        with patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "stream": True},
            )
        events = _parse_sse(resp.text)
        final = events[-1]["data"]["response"]
        assert final["status"] == "incomplete"
        assert final["incomplete_details"]["reason"] == "max_output_tokens"

    @pytest.mark.asyncio
    async def test_response_created_has_zero_output_tokens(self, app_client):
        # response.created must emit BEFORE generation, so it cannot carry the
        # final token count — usage.output_tokens must be 0.
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Hello", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=3)}

            return gen()

        with patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "stream": True},
            )
        events = _parse_sse(resp.text)
        assert events[0]["event"] == "response.created"
        assert events[0]["data"]["response"]["usage"]["output_tokens"] == 0
        # The final event must still carry the real count.
        assert events[-1]["data"]["response"]["usage"]["output_tokens"] == 3

    @pytest.mark.asyncio
    async def test_per_chunk_content_deltas(self, app_client):
        # Visible content arriving across multiple engine chunks must produce
        # one output_text.delta per chunk as it is generated — not a single
        # combined delta after the whole generation is buffered. (A leading
        # <think> block pushes the splitter past its detect phase so each
        # following content chunk streams immediately.)
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                yield {"text": "<think>t</think>Hel", "done": False}
                yield {"text": "lo", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=2)}

            return gen()

        with patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "hi",
                    "stream": True,
                    "reasoning": {"effort": "high"},
                },
            )
        events = _parse_sse(resp.text)
        deltas = [e for e in events if e["event"] == "response.output_text.delta"]
        assert [d["data"]["delta"] for d in deltas] == ["Hel", "lo"]

    @pytest.mark.asyncio
    async def test_thinking_item_closes_before_message_opens(self, app_client):
        # The reasoning item must be opened and closed before the message item
        # is added, so a client renders thinking then the answer in order.
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                yield {"text": "<think>step</think>Answer", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "q",
                    "stream": True,
                    "reasoning": {"effort": "high"},
                },
            )
        events = _parse_sse(resp.text)
        reasoning_done = next(
            i
            for i, e in enumerate(events)
            if e["event"] == "response.output_item.done"
            and e["data"]["item"]["type"] == "reasoning"
        )
        message_added = next(
            i
            for i, e in enumerate(events)
            if e["event"] == "response.output_item.added"
            and e["data"]["item"]["type"] == "message"
        )
        assert reasoning_done < message_added
        # The reasoning item carries the full thinking text.
        reasoning_item = events[reasoning_done]["data"]["item"]
        assert reasoning_item["content"][0]["text"] == "step"

    @pytest.mark.asyncio
    async def test_done_honored_when_combined_with_thinking_expected(self, app_client):
        # A terminal chunk that also carries `thinking_expected` must still be
        # treated as done — its stats/done_reason can't be dropped.
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "hi", "done": False}
                yield {
                    "thinking_expected": True,
                    "done": True,
                    "stats": TimingStats(eval_count=5),
                }

            return gen()

        with patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "stream": True},
            )
        events = _parse_sse(resp.text)
        final = events[-1]["data"]["response"]
        assert final["status"] == "completed"
        assert final["usage"]["output_tokens"] == 5

    @pytest.mark.asyncio
    async def test_reasoning_text_streams_as_deltas(self, app_client):
        # Thinking tokens must stream as response.reasoning_text.delta events as
        # they arrive (issue #547 "Expected"), with a reasoning_text.done
        # carrying the full text, emitted before the message item opens.
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                yield {"text": "<think>The capital is ", "done": False}
                yield {"text": "Paris.</think>Done", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "q",
                    "stream": True,
                    "reasoning": {"effort": "high"},
                },
            )
        events = _parse_sse(resp.text)
        deltas = [e for e in events if e["event"] == "response.reasoning_text.delta"]
        assert len(deltas) >= 2
        assert "".join(d["data"]["delta"] for d in deltas) == "The capital is Paris."
        done = next(e for e in events if e["event"] == "response.reasoning_text.done")
        assert done["data"]["text"] == "The capital is Paris."
        # reasoning_text.done precedes the message item opening.
        done_idx = events.index(done)
        message_added = next(
            i
            for i, e in enumerate(events)
            if e["event"] == "response.output_item.added"
            and e["data"]["item"]["type"] == "message"
        )
        assert done_idx < message_added

    @pytest.mark.asyncio
    async def test_truncated_thinking_flushes_reasoning_item(self, app_client):
        # A generation cut off mid-<think> (no close tag) must still close the
        # reasoning item before response.completed, with no empty message item.
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                yield {"text": "<think>partial thought", "done": False}
                yield {
                    "text": "",
                    "done": True,
                    "done_reason": "length",
                    "stats": TimingStats(),
                }

            return gen()

        with patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "q",
                    "stream": True,
                    "reasoning": {"effort": "high"},
                },
            )
        events = _parse_sse(resp.text)
        types = [e["event"] for e in events]
        assert types[-1] == "response.completed"
        # A reasoning item was opened and closed.
        assert any(
            e["event"] == "response.output_item.done"
            and e["data"]["item"]["type"] == "reasoning"
            for e in events
        )
        # No message item, since there is no visible text.
        assert not any(
            e["event"] == "response.output_item.added"
            and e["data"]["item"]["type"] == "message"
            for e in events
        )
        final = events[-1]["data"]["response"]
        assert final["status"] == "incomplete"


class TestRetrieveDelete:
    @pytest.fixture(autouse=True)
    def _clear_store(self):
        get_store().clear()
        yield
        get_store().clear()

    @pytest.mark.asyncio
    async def test_get_then_delete_lifecycle(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            created = await app_client.post(
                "/v1/responses", json={"model": "qwen3", "input": "hi"}
            )
        rid = created.json()["id"]

        got = await app_client.get(f"/v1/responses/{rid}")
        assert got.status_code == 200
        assert got.json()["id"] == rid

        deleted = await app_client.delete(f"/v1/responses/{rid}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True

        gone = await app_client.get(f"/v1/responses/{rid}")
        assert gone.status_code == 404

    @pytest.mark.asyncio
    async def test_get_unknown_404(self, app_client):
        resp = await app_client.get("/v1/responses/resp_unknown")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_unknown_404(self, app_client):
        resp = await app_client.delete("/v1/responses/resp_unknown")
        assert resp.status_code == 404


class TestToolChoiceHonored:
    """Issue #620: on /v1/responses ``tool_choice`` was accepted and ignored.
    ``"none"`` must suppress tool calls; forced values must 400."""

    TOOLS = [{"type": "function", "name": "f", "parameters": {"type": "object"}}]
    TOOL_OUTPUT = '<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'

    @pytest.mark.asyncio
    async def test_none_suppresses_function_call(self, app_client):
        mock_result = {"text": self.TOOL_OUTPUT, "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "go",
                    "tools": self.TOOLS,
                    "tool_choice": "none",
                },
            )
        assert resp.status_code == 200
        out = resp.json()["output"]
        assert not any(it["type"] == "function_call" for it in out)
        # Tools must not be forwarded to the engine when the client forced text.
        assert mock_gen.call_args.kwargs["tools"] in (None, [])

    @pytest.mark.asyncio
    async def test_required_is_rejected_with_400(self, app_client):
        resp = await app_client.post(
            "/v1/responses",
            json={
                "model": "qwen3",
                "input": "go",
                "tools": self.TOOLS,
                "tool_choice": "required",
            },
        )
        assert resp.status_code == 400


class TestBufferedToolStreamKeepalive:
    """Issue #616: the Responses tools-mode buffered stream must emit keepalive
    pings during generation instead of zero bytes until it finishes."""

    @pytest.mark.asyncio
    async def test_pings_emitted_during_slow_generation(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                await asyncio.sleep(0.05)
                yield {"text": "hi", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with (
            patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream),
            patch("olmlx.routers.responses.KEEPALIVE_PING_INTERVAL", 0.01),
        ):
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "go",
                    "stream": True,
                    "tools": [
                        {
                            "type": "function",
                            "name": "f",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            )
        # SSE comment lines (": ...") are the protocol-legal keepalive.
        assert any(line.startswith(": ") for line in resp.text.splitlines()), resp.text


class TestSDKShapeRegression:
    @pytest.mark.asyncio
    async def test_tool_choice_defaults_to_auto(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses", json={"model": "qwen3", "input": "hi"}
            )
        assert resp.json()["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_tool_choice_explicit_echoed(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "tool_choice": "none"},
            )
        assert resp.json()["tool_choice"] == "none"

    @pytest.mark.asyncio
    async def test_stream_text_events_carry_logprobs(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "hi", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "hi", "stream": True},
            )
        events = _parse_sse(resp.text)
        delta = next(e for e in events if e["event"] == "response.output_text.delta")
        done = next(e for e in events if e["event"] == "response.output_text.done")
        assert delta["data"]["logprobs"] == []
        assert done["data"]["logprobs"] == []

    @pytest.mark.asyncio
    async def test_stream_function_call_done_carries_name(self, app_client):
        raw = '<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": raw, "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.responses.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "go",
                    "stream": True,
                    "tools": [
                        {
                            "type": "function",
                            "name": "f",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            )
        events = _parse_sse(resp.text)
        done = next(
            e for e in events if e["event"] == "response.function_call_arguments.done"
        )
        assert done["data"]["name"] == "f"


class TestResponsesMalformedToolSchemaRejected:
    """/v1/responses shares the fill_missing_required_args path (via
    _convert_tools) and must reject a non-dict ``parameters`` with a clean
    400 at the boundary — before generation — instead of crashing post-parse
    (issue #644). No ``generate_chat`` mock: the request must fail before
    dispatch."""

    @pytest.mark.asyncio
    async def test_non_streaming_returns_400(self, app_client):
        resp = await app_client.post(
            "/v1/responses",
            json={
                "model": "qwen3",
                "input": "hi",
                "tools": [
                    {"type": "function", "name": "foo", "parameters": "not-an-object"}
                ],
            },
        )
        assert resp.status_code == 400
        # OpenAI-shaped error envelope (Responses is under /v1/).
        assert "parameters" in resp.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_streaming_returns_400(self, app_client):
        resp = await app_client.post(
            "/v1/responses",
            json={
                "model": "qwen3",
                "input": "hi",
                "stream": True,
                "tools": [
                    {"type": "function", "name": "foo", "parameters": "not-an-object"}
                ],
            },
        )
        assert resp.status_code == 400
        assert "text/event-stream" not in resp.headers.get("content-type", "")
