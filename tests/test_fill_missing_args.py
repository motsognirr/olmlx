"""Tests for ``fill_missing_required_args``."""

from olmlx.engine.tool_parser import fill_missing_required_args


def _make_tool_def(name: str, properties: dict, required: list[str]) -> dict:
    """Build an OpenAI-format tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _make_tool_use(name: str, inputs: dict) -> dict:
    return {"type": "tool_use", "id": "toolu_test", "name": name, "input": inputs}


class TestFillMissingRequiredArgs:
    def test_fills_missing_required_string(self):
        """Missing required string param gets empty string default."""
        tools = [
            _make_tool_def(
                "bash",
                {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                ["command", "description"],
            )
        ]
        tool_uses = [_make_tool_use("bash", {"command": "ls -F"})]

        fill_missing_required_args(tool_uses, tools)

        assert tool_uses[0]["input"]["description"] == ""
        assert tool_uses[0]["input"]["command"] == "ls -F"

    def test_preserves_existing_values(self):
        """Existing values are never overwritten."""
        tools = [
            _make_tool_def(
                "bash",
                {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                ["command", "description"],
            )
        ]
        tool_uses = [
            _make_tool_use("bash", {"command": "ls", "description": "List files"})
        ]

        fill_missing_required_args(tool_uses, tools)

        assert tool_uses[0]["input"]["description"] == "List files"

    def test_fills_none_value_for_required_string(self):
        """Required string param present but None gets filled with empty string."""
        tools = [
            _make_tool_def(
                "bash",
                {"command": {"type": "string"}},
                ["command"],
            )
        ]
        tool_uses = [_make_tool_use("bash", {"command": None})]

        fill_missing_required_args(tool_uses, tools)

        assert tool_uses[0]["input"]["command"] == ""

    def test_skips_non_string_required_fields(self):
        """Non-string required fields are left alone — don't guess defaults."""
        tools = [
            _make_tool_def(
                "search",
                {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                ["query", "limit"],
            )
        ]
        tool_uses = [_make_tool_use("search", {"query": "foo"})]

        fill_missing_required_args(tool_uses, tools)

        assert "limit" not in tool_uses[0]["input"]

    def test_non_string_none_value_left_as_none(self):
        """Required non-string param present as None: warn but don't fill."""
        tools = [
            _make_tool_def(
                "search",
                {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                ["query", "limit"],
            )
        ]
        tool_uses = [_make_tool_use("search", {"query": "foo", "limit": None})]

        fill_missing_required_args(tool_uses, tools)

        assert tool_uses[0]["input"]["limit"] is None

    def test_no_op_when_tools_none(self):
        """None declared_tools is a safe no-op."""
        tool_uses = [_make_tool_use("bash", {"command": "ls"})]
        fill_missing_required_args(tool_uses, None)
        assert tool_uses[0]["input"] == {"command": "ls"}

    def test_input_none_is_promoted_to_empty_dict(self):
        """When a tool_use arrives with ``input=None`` and matches a declared
        tool, ``input`` is replaced with an empty dict — even when no
        required string injection runs.  Documents the deliberate
        ``None → {}`` promotion called out in the function's docstring."""
        tools = [_make_tool_def("noop", {"count": {"type": "integer"}}, ["count"])]
        tool_uses = [{"type": "tool_use", "id": "x", "name": "noop", "input": None}]
        fill_missing_required_args(tool_uses, tools)
        assert tool_uses[0]["input"] == {}

    def test_input_none_normalized_when_tool_has_no_required_params(self):
        """``None → {}`` is unconditional for any tool present in the
        declared list, including ones with zero required params.  Pins the
        contract so callers can always treat ``tu["input"]`` as a dict
        after the call without an ``or {}`` guard."""
        tools = [
            _make_tool_def(
                "noop",
                {"count": {"type": "integer"}},
                [],  # no required params
            )
        ]
        tool_uses = [{"type": "tool_use", "id": "x", "name": "noop", "input": None}]
        fill_missing_required_args(tool_uses, tools)
        assert tool_uses[0]["input"] == {}

    def test_input_none_preserved_for_unknown_tool(self):
        """If the tool is not in the declared list, ``input`` is left
        untouched (None stays None).  We have no schema to normalize
        against, so the caller decides what to do with the unknown call."""
        tools = [_make_tool_def("known", {}, [])]
        tool_uses = [{"type": "tool_use", "id": "x", "name": "unknown", "input": None}]
        fill_missing_required_args(tool_uses, tools)
        assert tool_uses[0]["input"] is None

    def test_no_op_when_tools_empty(self):
        """Empty declared_tools is a safe no-op."""
        tool_uses = [_make_tool_use("bash", {"command": "ls"})]
        fill_missing_required_args(tool_uses, [])
        assert tool_uses[0]["input"] == {"command": "ls"}

    def test_unmatched_tool_name_ignored(self):
        """Tool calls for tools not in declared list are left untouched."""
        tools = [
            _make_tool_def(
                "grep",
                {"pattern": {"type": "string"}},
                ["pattern"],
            )
        ]
        tool_uses = [_make_tool_use("bash", {"command": "ls"})]

        fill_missing_required_args(tool_uses, tools)

        assert tool_uses[0]["input"] == {"command": "ls"}

    def test_multiple_tool_calls(self):
        """Handles multiple tool calls in one response."""
        tools = [
            _make_tool_def(
                "bash",
                {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                ["command", "description"],
            )
        ]
        tool_uses = [
            _make_tool_use("bash", {"command": "ls"}),
            _make_tool_use("bash", {"command": "pwd", "description": "Print dir"}),
        ]

        fill_missing_required_args(tool_uses, tools)

        assert tool_uses[0]["input"]["description"] == ""
        assert tool_uses[1]["input"]["description"] == "Print dir"

    def test_case_insensitive_tool_name_match(self):
        """Tool name matching should be case-insensitive since models vary."""
        tools = [
            _make_tool_def(
                "Bash",
                {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                ["command", "description"],
            )
        ]
        tool_uses = [_make_tool_use("bash", {"command": "ls"})]

        fill_missing_required_args(tool_uses, tools)

        assert tool_uses[0]["input"]["description"] == ""

    def test_case_insensitive_tool_name_match_reverse(self):
        """Model outputs uppercase name when tool was registered lowercase."""
        tools = [
            _make_tool_def(
                "bash",
                {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                ["command", "description"],
            )
        ]
        tool_uses = [_make_tool_use("BASH", {"command": "ls"})]

        fill_missing_required_args(tool_uses, tools)

        assert tool_uses[0]["input"]["description"] == ""

    def test_fills_when_input_is_empty_dict(self):
        """Both required params missing from empty input (exercises tu['input'] = inp reassignment)."""
        tools = [
            _make_tool_def(
                "bash",
                {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                ["command", "description"],
            )
        ]
        tool_uses = [_make_tool_use("bash", {})]

        fill_missing_required_args(tool_uses, tools)

        assert tool_uses[0]["input"] == {"command": "", "description": ""}

    def test_missing_parameters_key_in_tool_def(self):
        """Gracefully handles tool defs without parameters."""
        tools = [{"type": "function", "function": {"name": "noop"}}]
        tool_uses = [_make_tool_use("noop", {})]

        fill_missing_required_args(tool_uses, tools)

        assert tool_uses[0]["input"] == {}
