"""Tests for olmlx.routers.chat."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from olmlx.routers.thinking_split import (
    flush_split_thinking,
    split_thinking_streaming,
)
from olmlx.utils.timing import TimingStats


class TestSplitThinkingStreaming:
    """Unit tests for the Ollama-side thinking splitter (issue #307)."""

    def test_flush_in_think_phase_treats_buffer_as_thinking(self):
        """If the stream ends with an open `<think>` and no closer, the held
        buffer must be returned as thinking content (not silently lost)."""
        # Walk through chunks ending mid-think.
        state: dict = {}
        thinking, content = split_thinking_streaming("<think>partial reasoning", state)
        # `<think>` consumed; "partial reasoning" is 17 chars in `in_think`
        # phase — the splitter emits all but the last 8 chars and holds the
        # tail in case `</think>` is straddling a chunk boundary.
        assert thinking == "partial r"
        assert content == ""
        assert state["phase"] == "in_think"
        assert state["buffer"] == "easoning"

        # Stream ends — flush must surface the buffered remainder as thinking.
        tail_thinking, tail_content = flush_split_thinking(state)
        assert tail_thinking == "easoning"
        assert tail_content == ""

    def test_flush_resets_thinking_expected(self):
        """flush_split_thinking resets all managed keys, including
        thinking_expected, so a reused state dict starts a fresh stream."""
        state: dict = {"thinking_expected": True, "buffer": "x", "phase": "detect"}
        flush_split_thinking(state)
        assert not state.get("thinking_expected")


class TestChatRouter:
    @pytest.mark.asyncio
    async def test_chat_think_default_none(self, app_client):
        """No `think` field -> enable_thinking=None (engine default)."""
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs["enable_thinking"] is None

    @pytest.mark.asyncio
    async def test_chat_think_false_disables(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "think": False,
                },
            )
        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_chat_think_true_streaming_enables(self, app_client):
        captured = {}

        async def mock_stream(*args, **kwargs):
            captured.update(kwargs)

            async def gen():
                yield {"text": "hi", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "think": True,
                },
            )
        assert resp.status_code == 200
        assert captured["enable_thinking"] is True

    @pytest.mark.asyncio
    async def test_chat_non_streaming_strips_gemma4_channel(self, app_client):
        """Gemma 4 channel-thinking tokens must not leak into message.content (#306)."""
        stats = TimingStats(eval_count=10)
        mock_result = {
            "text": "<|channel>thought\nLet me think about this.\n<channel|>391",
            "done": True,
            "stats": stats,
        }

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "gemma4",
                    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        content = data["message"]["content"]
        assert "<|channel>" not in content
        assert "<channel|>" not in content
        assert "thought" not in content
        assert "391" in content

    @pytest.mark.asyncio
    async def test_chat_streaming_strips_gemma4_channel(self, app_client):
        """Gemma 4 channel tokens must not leak through the streaming path."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "<|channel>thought\n", "done": False}
                yield {"text": "Let me think.", "done": False}
                yield {"text": "<channel|>", "done": False}
                yield {"text": "391", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "gemma4",
                    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().split("\n") if line.strip()]
        joined = "".join(json.loads(line)["message"]["content"] for line in lines)
        assert "<|channel>" not in joined
        assert "<channel|>" not in joined
        assert "Let me think." not in joined
        assert "391" in joined

    @pytest.mark.asyncio
    async def test_chat_streaming_short_response_emits_content_chunk(self, app_client):
        """A short, no-thinking response must be emitted as a non-done chunk.

        Regression: putting the detect-phase buffer in the final done chunk
        is invisible to Ollama clients that ignore done-chunk content.
        """

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "hello", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        lines = [
            json.loads(line) for line in resp.text.strip().split("\n") if line.strip()
        ]
        non_done = [line for line in lines if not line.get("done")]
        joined_non_done = "".join(c["message"]["content"] for c in non_done)
        assert "hello" in joined_non_done

    @pytest.mark.asyncio
    async def test_chat_non_streaming_returns_tool_calls(self, app_client):
        """Parsed tool calls must be returned in message.tool_calls, not stripped."""
        stats = TimingStats(eval_count=10)
        mock_result = {
            "text": '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
            "done": True,
            "stats": stats,
        }

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "weather in Paris"}],
                    "stream": False,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get the weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                },
                            },
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        tool_calls = data["message"].get("tool_calls")
        assert tool_calls, f"Expected tool_calls in response, got {data['message']}"
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[0]["function"]["arguments"] == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_chat_non_streaming_tool_call_with_null_arguments(self, app_client):
        """A tool call whose arguments parse to JSON ``null`` must not crash
        the ``ToolCallFunction`` Pydantic construction.  Regression: the
        Ollama path passed ``tu["input"]`` to ``arguments: dict[str, Any]``
        unguarded, and gpt-oss-style channel parsing can emit ``input=None``
        when the tool-call ``<|message|>`` payload is the literal string
        ``null``."""
        stats = TimingStats(eval_count=10)
        # Emit a gpt-oss harmony block whose message payload is JSON null.
        raw = (
            "<|start|>assistant<|channel|>commentary to=functions.ping"
            "<|message|>null<|call|>"
        )
        mock_result = {
            "text": "",
            "raw_text": raw,
            "done": True,
            "stats": stats,
        }

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "gpt-oss",
                    "messages": [{"role": "user", "content": "ping?"}],
                    "stream": False,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "ping",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        tool_calls = data["message"].get("tool_calls")
        assert tool_calls
        # Null payload should coerce to an empty dict, not propagate as None.
        assert tool_calls[0]["function"]["arguments"] == {}

    @pytest.mark.asyncio
    async def test_chat_streaming_with_tools_text_only_omits_tool_calls_key(
        self, app_client
    ):
        """When tools are declared but the model returns only text, the
        emitted ``message`` must omit the ``tool_calls`` key entirely
        rather than serialising it as ``null``.  Ollama clients that use
        ``"tool_calls" in chunk["message"]`` to detect tool responses
        would otherwise see a false positive on every text-only chunk on
        this path."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "just answering normally", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "x",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        lines = [
            json.loads(line) for line in resp.text.strip().split("\n") if line.strip()
        ]
        for line in lines:
            msg = line.get("message") or {}
            assert "tool_calls" not in msg, (
                f"text-only response leaked tool_calls key: {line}"
            )

    @pytest.mark.asyncio
    async def test_chat_streaming_with_tools_thinking_only_no_empty_content_chunk(
        self, app_client
    ):
        """If the model's entire output is a thinking block, the
        streaming-with-tools path must surface that thinking via
        ``message.thinking`` rather than dropping it.  The pre-done chunk
        must NOT carry an empty ``content`` field (regression from PR
        #313: without the visible/thinking/tool_calls guard, an
        all-thinking response produced a spurious
        ``{"content": ""}`` line before the done chunk)."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {
                    "text": "<think>just thinking out loud</think>",
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "x",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        lines = [
            json.loads(line) for line in resp.text.strip().split("\n") if line.strip()
        ]
        non_done = [line for line in lines if not line.get("done")]
        # Exactly one non-done chunk, carrying the thinking content.
        # ``content`` may be present as an empty string (Ollama protocol
        # keeps the field on every message), but the thinking text is
        # what makes the chunk non-spurious.
        assert len(non_done) == 1, (
            f"expected one non-done chunk for thinking content, got {non_done}"
        )
        msg = non_done[0]["message"]
        assert "just thinking out loud" in msg.get("thinking", ""), (
            f"thinking content missing: {msg}"
        )
        # The done chunk must still close the stream cleanly.
        assert lines and lines[-1].get("done") is True

    @pytest.mark.asyncio
    async def test_chat_streaming_with_tools_uses_raw_text_for_parsing(
        self, app_client
    ):
        """When ``generate_chat`` sets ``raw_text`` on the done chunk (gpt-oss
        models route their channel-formatted output that way), the
        streaming-with-tools path must parse against ``raw_text`` rather than
        the accumulated visible-only ``full_text``.  Without this the
        tool-call markup encoded in the channel block is invisible to
        ``parse_model_output`` and ``tool_calls`` comes back empty."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                # Visible chunks carry no tool-call text; the channel-
                # formatted call lives only in ``raw_text`` on the done
                # chunk, mimicking the gpt-oss code path.
                yield {"text": "Looking up the time...", "done": False}
                yield {
                    "text": "",
                    "raw_text": (
                        "<|start|>assistant<|channel|>commentary "
                        "to=functions.get_time"
                        '<|message|>{"tz": "UTC"}<|call|>'
                    ),
                    "done": True,
                    "stats": TimingStats(),
                }

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "gpt-oss",
                    "messages": [{"role": "user", "content": "what time is it?"}],
                    "stream": True,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"tz": {"type": "string"}},
                                },
                            },
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        lines = [
            json.loads(line) for line in resp.text.strip().split("\n") if line.strip()
        ]
        tool_calls = []
        for line in lines:
            tc = (line.get("message") or {}).get("tool_calls")
            if tc:
                tool_calls.extend(tc)
        assert tool_calls, f"expected raw_text-sourced tool_calls, got {lines}"
        assert tool_calls[0]["function"]["name"] == "get_time"
        assert tool_calls[0]["function"]["arguments"] == {"tz": "UTC"}

    @pytest.mark.asyncio
    async def test_chat_streaming_with_tools_post_parse_error_yields_error_chunk(
        self, app_client
    ):
        """An exception raised after the stream is drained (e.g. in
        ``_build_tool_calls``) must surface as an NDJSON error line, not a
        silent truncation."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "anything", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with (
            patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream),
            patch(
                "olmlx.routers.chat._build_tool_calls",
                side_effect=RuntimeError("boom"),
            ),
        ):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "x",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        lines = [
            json.loads(line) for line in resp.text.strip().split("\n") if line.strip()
        ]
        assert lines, "expected at least one NDJSON line on error"
        last = lines[-1]
        assert "error" in last, f"expected error chunk, got {last}"
        assert last.get("done") is True

    @pytest.mark.asyncio
    async def test_chat_streaming_with_tools_emits_structured_tool_calls(
        self, app_client
    ):
        """Streaming ``/api/chat`` with tools must surface structured
        ``message.tool_calls`` instead of leaking the raw markup."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {
                    "text": '<tool_call>{"name": "get_weather", '
                    '"arguments": {"city": "Paris"}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "weather in Paris"}],
                    "stream": True,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get the weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                },
                            },
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        lines = [
            json.loads(line) for line in resp.text.strip().split("\n") if line.strip()
        ]
        # Collect tool_calls from any chunk.
        tool_calls = []
        for line in lines:
            tc = (line.get("message") or {}).get("tool_calls")
            if tc:
                tool_calls.extend(tc)
            # Markup must not leak in content.
            content = (line.get("message") or {}).get("content") or ""
            assert "<tool_call>" not in content, (
                f"raw markup leaked in streaming response: {line}"
            )
        assert tool_calls, f"Expected structured tool_calls in stream, got {lines}"
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[0]["function"]["arguments"] == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_chat_non_streaming_fills_missing_required_args(self, app_client):
        """If the model omits a required string arg, the router must inject
        an empty string so ``ToolCallFunction`` construction succeeds
        (mirrors the OpenAI router's behavior).
        """
        stats = TimingStats(eval_count=10)
        mock_result = {
            "text": '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>',
            "done": True,
            "stats": stats,
        }

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "list files"}],
                    "stream": False,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "description": "Run a shell command",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "command": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["command", "description"],
                                },
                            },
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        tool_calls = data["message"].get("tool_calls")
        assert tool_calls
        args = tool_calls[0]["function"]["arguments"]
        assert args.get("command") == "ls"
        assert args.get("description") == "", (
            f"Missing required string arg should be filled, got {args}"
        )

    @pytest.mark.asyncio
    async def test_chat_non_streaming(self, app_client):
        stats = TimingStats(eval_count=10)
        mock_result = {"text": "Hello!", "done": True, "stats": stats}

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert data["message"]["content"] == "Hello!"
        assert data["message"]["role"] == "assistant"
        assert data["done_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_chat_streaming(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Hi", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "ndjson" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_chat_with_options(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "options": {"temperature": 0.5, "num_predict": 100},
                },
            )

        assert resp.status_code == 200
        # Verify options were passed
        call_args = mock_gen.call_args
        assert call_args[1]["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_chat_non_streaming_separates_thinking_from_content(self, app_client):
        """Issue #307: ``<think>...</think>`` and Qwen3.5-style orphan
        ``</think>`` thinking must be separated from ``message.content`` and
        surfaced under ``message.thinking`` to match Ollama's API."""
        # Qwen3.5 case: no opening <think>, closing tag present.
        raw = (
            "Thinking Process:\n\n"
            "1. Analyze the Request: The user wants 17 * 23.\n"
            "2. Compute: 17 * 23 = 391.\n"
            "3. Construct Final Response: 391\n"
            "</think>\n\n"
            "391"
        )
        # Engine signals thinking_expected via the non-streaming result dict,
        # mirroring the streaming meta chunk (issue #307 review).
        mock_result = {
            "text": raw,
            "done": True,
            "stats": TimingStats(eval_count=10),
            "thinking_expected": True,
        }

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "17*23"}],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        msg = data["message"]
        assert msg["content"].strip() == "391"
        assert "</think>" not in msg["content"]
        assert "Thinking Process" not in msg["content"]
        assert "Thinking Process" in msg.get("thinking", "")
        assert "</think>" not in msg.get("thinking", "")

    @pytest.mark.asyncio
    async def test_chat_streaming_separates_thinking_from_content(self, app_client):
        """Issue #307: streaming path must put thinking into ``message.thinking``
        and not leak it into ``message.content`` — even when the orphan
        preamble is longer than the conservative non-thinking buffer."""
        preamble = (
            "Thinking Process:\n\n"
            "1. Analyze the Request: The user wants the product of 17 and 23.\n"
            "2. Compute: 17 * 23 = (17 * 20) + (17 * 3) = 340 + 51 = 391.\n"
            "3. Sanity check: 17 * 25 - 2 * 17 = 425 - 34 = 391.\n"
            "4. Format: just the number, no prose.\n"
            "5. Construct Final Response: 391"
        )
        # Must exceed the conservative 200-char limit used when thinking is
        # not expected so the test actually exercises the plumbed path.
        assert len(preamble) > 200

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                for i in range(0, len(preamble), 16):
                    yield {"text": preamble[i : i + 16], "done": False}
                yield {"text": "\n</think>\n\n", "done": False}
                yield {"text": "391", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "17*23"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        content = ""
        thinking = ""
        for line in resp.text.strip().split("\n"):
            if not line.strip():
                continue
            payload = json.loads(line)
            msg = payload.get("message", {})
            content += msg.get("content", "")
            thinking += msg.get("thinking", "") or ""

        assert "Thinking Process" in thinking
        assert "Thinking Process" not in content
        assert "</think>" not in content
        assert "</think>" not in thinking
        assert content.strip() == "391"

    @pytest.mark.asyncio
    async def test_chat_streaming_orphan_preamble_over_limit_leaks_to_content(
        self, app_client
    ):
        """Documents the known limitation flagged in #307 review round 11:
        when an orphan thinking preamble exceeds ``INIT_ORPHAN_DETECT_LIMIT``
        (1024 chars), the streaming router falls through to passthrough
        before ``</think>`` arrives and the preamble + close tag end up
        verbatim in ``message.content``.  Mitigations are documented on
        the constant; this test pins the current behaviour so future
        tuning of the limit is a deliberate choice."""
        from olmlx.engine.inference import INIT_ORPHAN_DETECT_LIMIT

        # 1100 chars of orphan thinking — pushes past the 1024 limit before
        # `</think>` arrives.
        long_preamble = "thought " * ((INIT_ORPHAN_DETECT_LIMIT // 8) + 10)

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                for i in range(0, len(long_preamble), 64):
                    yield {"text": long_preamble[i : i + 64], "done": False}
                yield {"text": "</think>\nanswer", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "?"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        content = ""
        thinking = ""
        for line in resp.text.strip().split("\n"):
            if not line.strip():
                continue
            payload = json.loads(line)
            msg = payload.get("message", {})
            content += msg.get("content", "")
            thinking += msg.get("thinking", "") or ""

        # Current behaviour (known limitation): once the detect buffer is
        # exhausted the preamble is emitted as content and the late
        # `</think>` arrives in passthrough where it is forwarded verbatim.
        # `answer` still makes it through.
        assert "answer" in content
        assert thinking == ""
        assert "</think>" in content

    @pytest.mark.asyncio
    async def test_chat_streaming_short_direct_answer_in_non_done_chunk(
        self, app_client
    ):
        """Issue #307 review round 9: when `thinking_expected=True` and the
        model gives a short direct answer (fits in the orphan-detect buffer
        without any `</think>`), the buffered content must be flushed to a
        non-done chunk before the done marker.  Standard Ollama clients
        only accumulate `message.content` from non-done chunks; putting
        accumulated content in the done frame's `message.content` would
        be silently dropped."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                yield {"text": "The answer is 42.", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "?"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().split("\n") if line.strip()]
        # Walk all lines; non-done chunks should carry the content.
        non_done_content = ""
        done_content = ""
        for line in lines:
            payload = json.loads(line)
            msg = payload.get("message", {})
            if payload.get("done"):
                done_content += msg.get("content", "")
            else:
                non_done_content += msg.get("content", "")
        # All content must be in non-done chunks; the done marker carries
        # an empty content field per the Ollama convention.
        assert non_done_content == "The answer is 42."
        assert done_content == ""

    @pytest.mark.asyncio
    async def test_chat_non_streaming_literal_close_think_preserved_when_not_thinking(
        self, app_client
    ):
        """Issue #307 review: the non-streaming path must agree with the
        streaming path — a literal `</think>` token from a non-thinking
        model must stay in `message.content`, not be silently routed to
        `message.thinking`."""
        raw = "Use </think> to close the thought block."
        mock_result = {
            "text": raw,
            "done": True,
            "stats": TimingStats(eval_count=10),
            "thinking_expected": False,
        }

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "syntax?"}],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        msg = data["message"]
        assert msg["content"] == raw
        assert "thinking" not in msg or msg["thinking"] is None

    @pytest.mark.asyncio
    async def test_chat_streaming_literal_close_think_preserved_when_not_thinking(
        self, app_client
    ):
        """When `thinking_expected=False`, a literal `</think>` in the
        output must stay in `message.content` and not be silently routed
        to `message.thinking`."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": False}
                yield {
                    "text": "Use </think> to close the thought block.",
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "syntax?"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        content = ""
        thinking = ""
        for line in resp.text.strip().split("\n"):
            if not line.strip():
                continue
            payload = json.loads(line)
            msg = payload.get("message", {})
            content += msg.get("content", "")
            thinking += msg.get("thinking", "") or ""

        assert thinking == ""
        assert "</think>" in content
        assert content == "Use </think> to close the thought block."

    @pytest.mark.asyncio
    async def test_chat_streaming_error_mid_stream(self, app_client):
        """Error during streaming emits an NDJSON error line instead of crashing."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "partial", "done": False}
                raise RuntimeError("GPU exploded")

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().split("\n") if line.strip()]
        last_line = json.loads(lines[-1])
        assert "error" in last_line
        assert "internal server error" in last_line["error"]
        assert last_line["done"] is True
        assert last_line["done_reason"] == "error"
        assert last_line["model"] == "qwen3"
        assert "created_at" in last_line


class TestFormatField:
    """Ollama ``format`` field maps to a GrammarSpec (issue #361)."""

    @pytest.mark.asyncio
    async def test_format_json_string(self, app_client):
        from olmlx.engine.grammar import GrammarSpec

        mock_result = {"text": '{"a": 1}', "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "format": "json",
                },
            )
        assert resp.status_code == 200
        spec = mock_gen.call_args.kwargs.get("grammar_spec")
        assert isinstance(spec, GrammarSpec)
        assert spec.kind == "json_object"

    @pytest.mark.asyncio
    async def test_format_schema_dict(self, app_client):
        from olmlx.engine.grammar import GrammarSpec

        mock_result = {"text": '{"x": 1}', "done": True, "stats": TimingStats()}
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "format": schema,
                },
            )
        assert resp.status_code == 200
        spec = mock_gen.call_args.kwargs.get("grammar_spec")
        assert isinstance(spec, GrammarSpec)
        assert spec.kind == "json_schema"
        assert spec.schema == schema

    @pytest.mark.asyncio
    async def test_invalid_format_returns_422_not_500(self, app_client):
        """Regression: ``parse_response_format`` raises ``ValueError`` on
        unrecognised input; FastAPI's default handler maps uncaught
        exceptions to 500. Must be caught and turned into 422 so the
        caller sees a meaningful error (review #384, bug 2)."""
        resp = await app_client.post(
            "/api/chat",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "format": "xml",
            },
        )
        assert resp.status_code == 422
        assert "format" in resp.text.lower() or "xml" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_format_omitted_means_no_grammar(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs.get("grammar_spec") is None


class TestEmptyMessagesRejected:
    @pytest.mark.asyncio
    async def test_api_chat_rejects_empty_messages(self, app_client):
        resp = await app_client.post(
            "/api/chat",
            json={"model": "qwen3", "messages": []},
        )
        assert resp.status_code == 422
        body = resp.text.lower()
        assert "messages" in body
        assert "empty" in body


class TestXCacheIDHeader:
    @pytest.mark.asyncio
    async def test_header_passed_to_generate_chat(self, app_client):
        stats = TimingStats()
        mock_result = {"text": "response", "done": True, "stats": stats}

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
                headers={"X-Cache-ID": "agent-gamma"},
            )

        assert resp.status_code == 200
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs.get("cache_id") == "agent-gamma"

    @pytest.mark.asyncio
    async def test_no_header_uses_default_cache_id(self, app_client):
        stats = TimingStats()
        mock_result = {"text": "response", "done": True, "stats": stats}

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs.get("cache_id") == ""


def test_chat_message_accepts_audio_field_and_dumps_it():
    from olmlx.schemas.chat import Message

    m = Message(role="user", content="hi", audio=["a.wav"])
    dumped = m.model_dump(exclude_none=True)
    assert dumped["audio"] == ["a.wav"]
