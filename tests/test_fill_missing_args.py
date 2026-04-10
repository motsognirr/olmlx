"""Tests for _fill_missing_required_args in the OpenAI router."""

from olmlx.routers.openai import _fill_missing_required_args


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

        _fill_missing_required_args(tool_uses, tools)

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

        _fill_missing_required_args(tool_uses, tools)

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

        _fill_missing_required_args(tool_uses, tools)

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

        _fill_missing_required_args(tool_uses, tools)

        assert "limit" not in tool_uses[0]["input"]

    def test_no_op_when_tools_none(self):
        """None declared_tools is a safe no-op."""
        tool_uses = [_make_tool_use("bash", {"command": "ls"})]
        _fill_missing_required_args(tool_uses, None)
        assert tool_uses[0]["input"] == {"command": "ls"}

    def test_no_op_when_tools_empty(self):
        """Empty declared_tools is a safe no-op."""
        tool_uses = [_make_tool_use("bash", {"command": "ls"})]
        _fill_missing_required_args(tool_uses, [])
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

        _fill_missing_required_args(tool_uses, tools)

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

        _fill_missing_required_args(tool_uses, tools)

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

        _fill_missing_required_args(tool_uses, tools)

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

        _fill_missing_required_args(tool_uses, tools)

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

        _fill_missing_required_args(tool_uses, tools)

        assert tool_uses[0]["input"] == {"command": "", "description": ""}

    def test_missing_parameters_key_in_tool_def(self):
        """Gracefully handles tool defs without parameters."""
        tools = [{"type": "function", "function": {"name": "noop"}}]
        tool_uses = [_make_tool_use("noop", {})]

        _fill_missing_required_args(tool_uses, tools)

        assert tool_uses[0]["input"] == {}
