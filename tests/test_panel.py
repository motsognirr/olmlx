"""Tests for olmlx.engine.panel."""

from olmlx.engine.grammar import GrammarSpec
from olmlx.engine.panel import (
    first_user_text,
    merge_tool_calls,
    route_grammar,
    select_members,
    serialize_tool_calls_qwen,
)
from olmlx.engine.registry import PanelConfig
from olmlx.engine.tool_parser import parse_model_output


def _panel() -> PanelConfig:
    return PanelConfig.from_entry(
        "p:latest",
        {
            "type": "panel",
            "classifier": "c",
            "judge": "j",
            "routes": {
                "code": ["qwen3-coder", "devstral"],
                "default": ["qwen3", "mistral"],
            },
        },
    )


class TestRoutingHelpers:
    def test_first_user_text_string_content(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi"},
        ]
        assert first_user_text(msgs) == "hello world"

    def test_first_user_text_list_content(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
        assert first_user_text(msgs) == "part one\npart two"

    def test_first_user_text_no_user(self):
        assert first_user_text([{"role": "system", "content": "x"}]) == ""

    def test_route_grammar_enumerates_keys(self):
        spec = route_grammar(_panel())
        assert isinstance(spec, GrammarSpec)
        assert spec.kind == "json_schema"
        enum = spec.schema["properties"]["route"]["enum"]
        assert set(enum) == {"code", "default"}

    def test_select_members_known_route(self):
        assert select_members("code", _panel()) == ["qwen3-coder", "devstral"]

    def test_select_members_unknown_route_falls_back_to_default(self):
        assert select_members("nonsense", _panel()) == ["qwen3", "mistral"]


class TestToolCallUnion:
    def test_merge_dedupes_identical_calls(self):
        # Two panelists; one shared identical call, one unique each.
        per_panelist = [
            [
                {"name": "search", "input": {"q": "x"}, "_span": (0, 1)},
                {"name": "read", "input": {"path": "a"}},
            ],
            [
                {"name": "search", "input": {"q": "x"}},
                {"name": "read", "input": {"path": "b"}},
            ],
        ]
        merged = merge_tool_calls(per_panelist)
        # search{q:x} collapses to one; read{a} and read{b} both kept.
        assert merged == [
            {"name": "search", "input": {"q": "x"}},
            {"name": "read", "input": {"path": "a"}},
            {"name": "read", "input": {"path": "b"}},
        ]
        # _span must be stripped.
        assert all("_span" not in tc for tc in merged)

    def test_merge_empty(self):
        assert merge_tool_calls([[], []]) == []

    def test_serialize_round_trips_through_parser(self):
        merged = [
            {"name": "search", "input": {"q": "x"}},
            {"name": "read", "input": {"path": "a"}},
        ]
        text = serialize_tool_calls_qwen(merged)
        _thinking, _visible, tool_uses = parse_model_output(text, has_tools=True)
        reparsed = [{"name": tu["name"], "input": tu["input"]} for tu in tool_uses]
        assert reparsed == merged

    def test_serialize_round_trips_tool_call_tag_in_args(self):
        # An argument value containing the literal closing tag must survive
        # the serialize -> parse_model_output round-trip (the panel emits
        # this text and the routers re-parse it).
        # With two calls, the non-greedy Qwen regex closes early inside the
        # first arg, the first call is silently dropped, and only the second
        # is returned — a real production failure.
        merged = [
            {"name": "search", "input": {"q": "see </tool_call> here"}},
            {"name": "read", "input": {"path": "/tmp/foo"}},
        ]
        text = serialize_tool_calls_qwen(merged)
        _t, _v, tool_uses = parse_model_output(text, has_tools=True)
        reparsed = [{"name": tu["name"], "input": tu["input"]} for tu in tool_uses]
        assert reparsed == merged

    def test_merge_handles_none_input(self):
        merged = merge_tool_calls([[{"name": "f", "input": None}]])
        assert merged == [{"name": "f", "input": {}}]

    def test_merge_no_panelists(self):
        assert merge_tool_calls([]) == []
