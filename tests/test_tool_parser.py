"""Tests for mlx_ollama.engine.tool_parser."""

import json

from mlx_ollama.engine.tool_parser import (
    _parse_json_call,
    _try_bare_json,
    _try_deepseek,
    _try_llama,
    _try_mistral,
    _try_qwen,
    parse_model_output,
)


class TestParseJsonCall:
    def test_valid_call(self):
        data = {"name": "get_weather", "arguments": {"city": "Tokyo"}}
        result = _parse_json_call(data)
        assert result is not None
        assert result["type"] == "tool_use"
        assert result["name"] == "get_weather"
        assert result["input"] == {"city": "Tokyo"}
        assert result["id"].startswith("toolu_")

    def test_parameters_key(self):
        data = {"name": "search", "parameters": {"q": "test"}}
        result = _parse_json_call(data)
        assert result is not None
        assert result["input"] == {"q": "test"}

    def test_string_arguments(self):
        data = {"name": "func", "arguments": '{"key": "val"}'}
        result = _parse_json_call(data)
        assert result is not None
        assert result["input"] == {"key": "val"}

    def test_invalid_string_arguments(self):
        data = {"name": "func", "arguments": "not json"}
        result = _parse_json_call(data)
        assert result is None

    def test_no_name(self):
        data = {"arguments": {"x": 1}}
        result = _parse_json_call(data)
        assert result is None

    def test_empty_name(self):
        data = {"name": "", "arguments": {}}
        result = _parse_json_call(data)
        assert result is None

    def test_empty_arguments(self):
        data = {"name": "noop"}
        result = _parse_json_call(data)
        assert result is not None
        assert result["input"] == {}


class TestTryQwen:
    def test_single_tool_call(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
        tool_uses, remaining = _try_qwen(text)
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "get_weather"
        assert tool_uses[0]["input"] == {"city": "NYC"}
        assert remaining.strip() == ""

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
        )
        tool_uses, remaining = _try_qwen(text)
        assert len(tool_uses) == 2
        assert tool_uses[0]["name"] == "a"
        assert tool_uses[1]["name"] == "b"

    def test_tool_call_with_surrounding_text(self):
        text = 'Let me help. <tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call> Done.'
        tool_uses, remaining = _try_qwen(text)
        assert len(tool_uses) == 1
        assert "Let me help." in remaining
        assert "Done." in remaining

    def test_xml_style_inside_tool_call(self):
        text = "<tool_call><function=get_weather><parameter=city>NYC</parameter></function></tool_call>"
        tool_uses, remaining = _try_qwen(text)
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "get_weather"
        assert tool_uses[0]["input"] == {"city": "NYC"}

    def test_no_tool_call(self):
        text = "Just a normal response."
        tool_uses, remaining = _try_qwen(text)
        assert len(tool_uses) == 0
        assert remaining == text

    def test_invalid_json_in_tool_call(self):
        text = "<tool_call>not json</tool_call>"
        tool_uses, remaining = _try_qwen(text)
        assert len(tool_uses) == 0  # can't parse as JSON or XML


class TestTryMistral:
    def test_single_tool_call(self):
        text = '[TOOL_CALLS] [{"name": "search", "arguments": {"q": "test"}}]'
        tool_uses, remaining = _try_mistral(text)
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "search"

    def test_multiple_tool_calls(self):
        calls = [
            {"name": "a", "arguments": {"x": 1}},
            {"name": "b", "arguments": {"y": 2}},
        ]
        text = f"[TOOL_CALLS] {json.dumps(calls)}"
        tool_uses, remaining = _try_mistral(text)
        assert len(tool_uses) == 2

    def test_no_match(self):
        tool_uses, remaining = _try_mistral("normal text")
        assert len(tool_uses) == 0
        assert remaining == "normal text"

    def test_invalid_json(self):
        text = "[TOOL_CALLS] not json"
        tool_uses, remaining = _try_mistral(text)
        assert len(tool_uses) == 0


class TestTryLlama:
    def test_single_call(self):
        text = '<|python_tag|>{"name": "get_info", "parameters": {"id": 42}}<|eom_id|>'
        tool_uses, remaining = _try_llama(text)
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "get_info"
        assert tool_uses[0]["input"] == {"id": 42}

    def test_no_eom_tag(self):
        text = '<|python_tag|>{"name": "func", "arguments": {"x": 1}}'
        tool_uses, remaining = _try_llama(text)
        assert len(tool_uses) == 1

    def test_no_match(self):
        tool_uses, remaining = _try_llama("normal text")
        assert len(tool_uses) == 0


class TestTryDeepseek:
    def test_single_call(self):
        text = (
            "<|tool_calls_begin|>"
            "<|tool_call_begin|>function\nget_weather\n"
            '{"city": "Berlin"}'
            "<|tool_call_end|>"
            "<|tool_calls_end|>"
        )
        tool_uses, remaining = _try_deepseek(text)
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "get_weather"
        assert tool_uses[0]["input"] == {"city": "Berlin"}

    def test_no_match(self):
        tool_uses, remaining = _try_deepseek("normal text")
        assert len(tool_uses) == 0

    def test_empty_args(self):
        text = (
            "<|tool_calls_begin|>"
            "<|tool_call_begin|>function\nlist_items\n"
            "<|tool_call_end|>"
            "<|tool_calls_end|>"
        )
        tool_uses, remaining = _try_deepseek(text)
        assert len(tool_uses) == 1
        assert tool_uses[0]["input"] == {}


class TestTryBareJson:
    def test_bare_json_at_start(self):
        text = '{"name": "func", "arguments": {"x": 1}}'
        tool_uses, remaining = _try_bare_json(text)
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "func"

    def test_bare_json_on_own_line(self):
        text = 'Some text\n{"name": "func", "arguments": {"x": 1}}'
        tool_uses, remaining = _try_bare_json(text)
        assert len(tool_uses) == 1

    def test_no_match(self):
        tool_uses, remaining = _try_bare_json("just text")
        assert len(tool_uses) == 0

    def test_invalid_json(self):
        text = '{"name": "func", "arguments": {invalid}}'
        tool_uses, remaining = _try_bare_json(text)
        assert len(tool_uses) == 0


class TestBareJsonMultiline:
    def test_bare_json_multiline(self):
        text = (
            '{\n  "name": "get_weather",\n  "arguments": {\n    "city": "NYC"\n  }\n}'
        )
        tool_uses, remaining = _try_bare_json(text)
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "get_weather"
        assert tool_uses[0]["input"] == {"city": "NYC"}

    def test_bare_json_deeply_nested(self):
        text = '{"name": "create", "arguments": {"config": {"nested": {"deep": true}}}}'
        tool_uses, remaining = _try_bare_json(text)
        assert len(tool_uses) == 1
        assert tool_uses[0]["input"]["config"]["nested"]["deep"] is True

    def test_bare_json_braces_in_string_values(self):
        text = '{"name": "echo", "arguments": {"msg": "a { b } c"}}'
        tool_uses, remaining = _try_bare_json(text)
        assert len(tool_uses) == 1
        assert tool_uses[0]["input"]["msg"] == "a { b } c"

    def test_bare_json_multiple_multiline(self):
        text = (
            '{\n  "name": "a",\n  "arguments": {"x": 1}\n}\n'
            '{\n  "name": "b",\n  "arguments": {"y": 2}\n}'
        )
        tool_uses, remaining = _try_bare_json(text)
        assert len(tool_uses) == 2
        assert tool_uses[0]["name"] == "a"
        assert tool_uses[1]["name"] == "b"

    def test_bare_json_unclosed_brace(self):
        text = '{"name": "func", "arguments": {"x": 1}'
        tool_uses, remaining = _try_bare_json(text)
        assert len(tool_uses) == 0


class TestParseModelOutput:
    def test_plain_text(self):
        thinking, visible, tools = parse_model_output("Hello world", has_tools=False)
        assert thinking == ""
        assert visible == "Hello world"
        assert tools == []

    def test_thinking_block(self):
        text = "<think>Let me reason about this...</think>The answer is 42."
        thinking, visible, tools = parse_model_output(text, has_tools=False)
        assert thinking == "Let me reason about this..."
        assert visible == "The answer is 42."
        assert tools == []

    def test_multiple_thinking_blocks(self):
        text = "<think>First thought</think>Middle<think>Second thought</think>End"
        thinking, visible, tools = parse_model_output(text, has_tools=False)
        assert "First thought" in thinking
        assert "Second thought" in thinking
        assert "End" in visible

    def test_tool_call_with_thinking(self):
        text = (
            "<think>I need to search</think>"
            '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>'
        )
        thinking, visible, tools = parse_model_output(text, has_tools=True)
        assert thinking == "I need to search"
        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    def test_no_tool_parsing_when_has_tools_false(self):
        text = '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>'
        thinking, visible, tools = parse_model_output(text, has_tools=False)
        assert len(tools) == 0
        assert "<tool_call>" in visible

    def test_tool_name_validation(self):
        text = '<tool_call>{"name": "unknown_func", "arguments": {}}</tool_call>'
        thinking, visible, tools = parse_model_output(
            text,
            has_tools=True,
            tool_names={"search", "get_weather"},
        )
        # Still parses, but logs a warning
        assert len(tools) == 1
        assert tools[0]["name"] == "unknown_func"

    def test_qwen_format_priority(self):
        """Qwen format should be tried first and win."""
        text = '<tool_call>{"name": "func_a", "arguments": {"x": 1}}</tool_call>'
        thinking, visible, tools = parse_model_output(text, has_tools=True)
        assert len(tools) == 1
        assert tools[0]["name"] == "func_a"

    def test_empty_text(self):
        thinking, visible, tools = parse_model_output("", has_tools=False)
        assert thinking == ""
        assert visible == ""
        assert tools == []

    def test_whitespace_stripping(self):
        thinking, visible, tools = parse_model_output("  hello  ", has_tools=False)
        assert visible == "hello"
