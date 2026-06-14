"""Tests for olmlx.engine.panel."""

from olmlx.engine.grammar import GrammarSpec
from olmlx.engine.panel import (
    first_user_text,
    route_grammar,
    select_members,
)
from olmlx.engine.registry import PanelConfig


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
