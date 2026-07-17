"""Tests for olmlx.routers.streaming_common (issue #471).

The buffer-and-parse plumbing shared by the OpenAI, Anthropic, and Ollama
chat routers when tools force full-output buffering.
"""

import asyncio

import pytest

from olmlx.routers.streaming_common import (
    BufferedModelOutput,
    buffer_stream,
    collect_stream,
    parse_buffered_output,
    validate_declared_tools,
    with_keepalive_pings,
)
from olmlx.utils.timing import TimingStats

PING = "event: ping\ndata: {}\n\n"


class TestValidateDeclaredTools:
    """Boundary validation of declared tool schemas (issue #644)."""

    def test_none_and_empty_are_noops(self):
        validate_declared_tools(None)
        validate_declared_tools([])

    def test_well_formed_tools_pass(self):
        validate_declared_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "foo",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )

    def test_missing_parameters_is_allowed(self):
        # ``parameters`` is optional; absence is not malformed.
        validate_declared_tools([{"type": "function", "function": {"name": "foo"}}])

    def test_non_dict_parameters_raises(self):
        with pytest.raises(ValueError, match=r"tools\[0\]\.function\.parameters"):
            validate_declared_tools(
                [{"type": "function", "function": {"name": "foo", "parameters": "x"}}]
            )

    def test_non_dict_function_raises(self):
        with pytest.raises(ValueError, match=r"tools\[0\]\.function"):
            validate_declared_tools([{"type": "function", "function": "x"}])

    def test_non_dict_tool_raises(self):
        with pytest.raises(ValueError, match=r"tools\[0\]"):
            validate_declared_tools(["not-an-object"])  # type: ignore[list-item]

    def test_index_is_reported(self):
        with pytest.raises(ValueError, match=r"tools\[1\]"):
            validate_declared_tools(
                [
                    {"type": "function", "function": {"name": "ok"}},
                    {"type": "function", "function": {"name": "bad", "parameters": 3}},
                ]
            )

    def _tool(self, params):
        return [{"type": "function", "function": {"name": "foo", "parameters": params}}]

    def test_non_dict_properties_raises(self):
        # Would otherwise crash fill_missing_required_args at ``.items()``.
        with pytest.raises(ValueError, match=r"parameters\.properties"):
            validate_declared_tools(self._tool({"required": ["a"], "properties": "x"}))

    def test_non_list_required_raises(self):
        # Would otherwise crash at ``set(required)`` for a non-iterable.
        with pytest.raises(ValueError, match=r"parameters\.required"):
            validate_declared_tools(self._tool({"required": 3}))

    def test_required_with_non_string_element_raises(self):
        # An unhashable element would crash ``set(required)``.
        with pytest.raises(ValueError, match=r"parameters\.required"):
            validate_declared_tools(self._tool({"required": [{"nested": 1}]}))

    def test_non_dict_required_property_value_raises(self):
        # Would otherwise crash at ``v.get("type")`` for the required prop.
        with pytest.raises(ValueError, match=r"properties\['a'\]"):
            validate_declared_tools(
                self._tool({"required": ["a"], "properties": {"a": "not-an-object"}})
            )

    def test_non_dict_value_on_non_required_property_is_allowed(self):
        # Only *required* property definitions are dereferenced downstream, so
        # a weird value on a non-required property must not be over-rejected.
        validate_declared_tools(
            self._tool({"required": ["a"], "properties": {"a": {}, "b": "whatever"}})
        )

    def test_missing_properties_with_required_is_allowed(self):
        # fill_missing_required_args coalesces a missing ``properties`` to {},
        # yielding no injection — not a crash, so not rejected.
        validate_declared_tools(self._tool({"required": ["a"]}))


async def _agen(chunks):
    for chunk in chunks:
        yield chunk


class TestCollectStream:
    @pytest.mark.asyncio
    async def test_accumulates_text_and_done_fields(self):
        stats = TimingStats(eval_count=7)
        out = await collect_stream(
            _agen(
                [
                    {"text": "Hello", "done": False},
                    {"text": " world", "done": False},
                    {"text": "", "done": True, "done_reason": "stop", "stats": stats},
                ]
            )
        )
        assert isinstance(out, BufferedModelOutput)
        assert out.full_text == "Hello world"
        assert out.done_reason == "stop"
        assert out.stats is stats
        assert out.raw_text == ""
        assert out.parse_text == "Hello world"

    @pytest.mark.asyncio
    async def test_raw_text_from_done_chunk_supersedes_full_text(self):
        """gpt-oss channel output rides on the done chunk's raw_text."""
        out = await collect_stream(
            _agen(
                [
                    {"text": "visible", "done": False},
                    {"text": "", "done": True, "raw_text": "<|channel|>raw"},
                ]
            )
        )
        assert out.full_text == "visible"
        assert out.raw_text == "<|channel|>raw"
        assert out.parse_text == "<|channel|>raw"

    @pytest.mark.asyncio
    async def test_empty_raw_text_falls_back_to_full_text(self):
        out = await collect_stream(
            _agen(
                [
                    {"text": "visible", "done": False},
                    {"text": "", "done": True, "raw_text": ""},
                ]
            )
        )
        assert out.parse_text == "visible"

    @pytest.mark.asyncio
    async def test_thinking_expected_meta_chunk_captured(self):
        out = await collect_stream(
            _agen(
                [
                    {"thinking_expected": True},
                    {"text": "hi", "done": False},
                    {"text": "", "done": True},
                ]
            )
        )
        assert out.thinking_expected is True
        # The meta chunk must not leak into the text.
        assert out.full_text == "hi"

    @pytest.mark.asyncio
    async def test_cache_info_chunks_skipped(self):
        out = await collect_stream(
            _agen(
                [
                    {"cache_info": True, "cache_read_tokens": 5},
                    {"text": "hi", "done": False},
                    {"text": "", "done": True},
                ]
            )
        )
        assert out.full_text == "hi"

    @pytest.mark.asyncio
    async def test_stream_without_done_chunk(self):
        """A stream that ends without a done chunk still aggregates text."""
        out = await collect_stream(_agen([{"text": "partial", "done": False}]))
        assert out.full_text == "partial"
        assert out.done_reason is None
        assert out.stats is None


class TestBufferStream:
    @pytest.mark.asyncio
    async def test_yields_cache_info_then_final_output(self):
        cache_chunk = {"cache_info": True, "cache_read_tokens": 5}
        items = []
        async for item in buffer_stream(
            _agen([cache_chunk, {"text": "hi", "done": False}, {"done": True}])
        ):
            items.append(item)
        assert items[0] is cache_chunk
        assert isinstance(items[-1], BufferedModelOutput)
        assert items[-1].full_text == "hi"

    @pytest.mark.asyncio
    async def test_keepalive_pings_passed_through(self):
        """With a keepalive interval, slow chunks produce ping events."""

        async def slow():
            yield {"text": "a", "done": False}
            await asyncio.sleep(0.5)
            yield {"text": "", "done": True}

        items = []
        async for item in buffer_stream(slow(), keepalive_interval=0.1, ping=PING):
            items.append(item)
        pings = [i for i in items if i == PING]
        assert len(pings) >= 2
        assert isinstance(items[-1], BufferedModelOutput)
        assert items[-1].full_text == "a"

    @pytest.mark.asyncio
    async def test_protocol_violating_chunk_raises(self):
        """A non-dict chunk that is NOT the keepalive ping must surface as an
        error, not be silently forwarded as if it were a ping — only the
        exact ping object passes through (identity, like the pre-#471
        sentinel)."""
        with pytest.raises(AttributeError):
            async for _ in buffer_stream(_agen(["stray string", {"done": True}])):
                pass


class TestWithKeepalivePings:
    @pytest.mark.asyncio
    async def test_pings_during_delay(self):
        async def slow_gen():
            await asyncio.sleep(0.7)
            yield {"text": "hello", "done": False}
            yield {"text": "", "done": True}

        results = []
        async for item in with_keepalive_pings(slow_gen(), 0.1, PING):
            results.append(item)
            if isinstance(item, dict) and item.get("done"):
                break

        pings = [r for r in results if r == PING]
        assert len(pings) >= 2

    @pytest.mark.asyncio
    async def test_no_pings_when_fast(self):
        async def fast_gen():
            yield {"text": "a", "done": False}
            yield {"text": "b", "done": False}
            yield {"text": "", "done": True}

        results = []
        async for item in with_keepalive_pings(fast_gen(), 5.0, PING):
            results.append(item)

        assert all(r != PING for r in results)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_cleanup_on_early_exit(self):
        async def infinite_gen():
            while True:
                await asyncio.sleep(10)
                yield {"text": "tick"}

        results = []
        async for item in with_keepalive_pings(infinite_gen(), 0.05, PING):
            results.append(item)
            if len(results) >= 3:
                break

        assert all(r == PING for r in results)
        await asyncio.sleep(0.05)


class TestParseBufferedOutput:
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    def _out(self, text, thinking_expected=False):
        return BufferedModelOutput(full_text=text, thinking_expected=thinking_expected)

    def test_parses_tool_calls_and_resolves_names(self):
        out = self._out(
            '<tool_call>{"name": "GET_WEATHER", "arguments": {"city": "London"}}</tool_call>'
        )
        thinking, visible, tool_uses = parse_buffered_output(out, self.TOOLS)
        assert visible == ""
        assert len(tool_uses) == 1
        # resolve_tool_names maps case-insensitively onto the declared name
        assert tool_uses[0]["name"] == "get_weather"
        assert tool_uses[0]["input"] == {"city": "London"}

    def test_fill_missing_args_default_on(self):
        out = self._out(
            '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
        )
        _, _, tool_uses = parse_buffered_output(out, self.TOOLS)
        assert tool_uses[0]["input"] == {"city": ""}

    def test_fill_missing_args_off(self):
        out = self._out(
            '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
        )
        _, _, tool_uses = parse_buffered_output(
            out, self.TOOLS, fill_missing_args=False
        )
        assert tool_uses[0]["input"] == {}

    def test_thinking_expected_gates_orphan_close(self):
        thinking, visible, _ = parse_buffered_output(
            self._out("reasoning</think>answer", thinking_expected=True), self.TOOLS
        )
        assert thinking == "reasoning"
        assert visible == "answer"

        thinking, visible, _ = parse_buffered_output(
            self._out("prose </think> more", thinking_expected=False), self.TOOLS
        )
        assert thinking == ""
        assert "</think>" in visible

    def test_has_tools_false_leaves_markup_in_visible(self):
        out = self._out('<tool_call>{"name": "x", "arguments": {}}</tool_call>')
        _, visible, tool_uses = parse_buffered_output(out, None, has_tools=False)
        assert tool_uses == []
        assert "<tool_call>" in visible
