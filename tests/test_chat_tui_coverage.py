"""Regression coverage for olmlx.chat.tui rendering helpers.

These tests exercise the pure formatting/state methods of ChatTUI and the
StreamContext state machine with a captured StringIO console (no live
terminal, no model, no network). They assert on real rendered output and
branch outcomes to catch formatting/state regressions.
"""

from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

from olmlx.chat.tool_safety import (
    ToolPolicy,
    ToolSafetyConfig,
    ToolSafetyPolicy,
)
from olmlx.chat.tui import ChatTUI, StreamContext


def make_tui(truncation=None):
    """ChatTUI with a captured console (force_terminal so styles render)."""
    tui = ChatTUI(tool_result_truncation=truncation)
    tui.console = Console(file=StringIO(), force_terminal=True, width=200)
    return tui


def out(tui):
    return tui.console.file.getvalue()


class TestWelcomeAndTools:
    def test_welcome_lists_each_tool_name_and_description(self):
        tui = make_tui()
        tools = [
            {"function": {"name": "read_file", "description": "Read a file"}},
            {"function": {"name": "search", "description": "Search the web"}},
        ]
        tui.display_welcome("qwen3:8b", tools)
        text = out(tui)
        assert "Model: qwen3:8b" in text
        assert "Tools: 2 available" in text
        assert "read_file" in text and "Read a file" in text
        assert "search" in text and "Search the web" in text
        assert "Commands:" in text

    def test_welcome_tool_missing_fields_uses_placeholder(self):
        tui = make_tui()
        tui.display_welcome("m", [{"function": {}}])
        text = out(tui)
        # name defaults to "?" when absent
        assert "?" in text

    def test_welcome_no_tools_mentions_mcp_config(self):
        tui = make_tui()
        tui.display_welcome("m", [])
        text = out(tui)
        assert "Tools: none" in text
        assert "mcp.json" in text

    def test_display_tools_lists_entries(self):
        tui = make_tui()
        tools = [
            {"function": {"name": "ls", "description": "list dir"}},
            {"function": {"name": "cat", "description": "print file"}},
        ]
        tui.display_tools(tools)
        text = out(tui)
        assert "ls" in text and "list dir" in text
        assert "cat" in text and "print file" in text
        assert "available tools" in text

    def test_display_tools_empty_short_circuits(self):
        tui = make_tui()
        tui.display_tools([])
        assert "No tools available" in out(tui)

    def test_stream_response_returns_streamcontext(self):
        tui = make_tui()
        ctx = tui.stream_response("seed")
        assert isinstance(ctx, StreamContext)
        assert ctx.get_text() == "seed"


class TestPanelHelpers:
    def test_display_tool_call_renders_name_and_json_args(self):
        tui = make_tui()
        tui.display_tool_call("read_file", {"path": "/tmp/x", "n": 3})
        text = out(tui)
        assert "read_file" in text
        assert "tool call" in text
        # arguments are JSON-formatted
        assert '"path"' in text
        assert '"/tmp/x"' in text

    def test_display_tool_call_empty_args_shows_braces(self):
        tui = make_tui()
        tui.display_tool_call("noop", {})
        text = out(tui)
        assert "noop" in text
        assert "{}" in text

    def test_display_tool_result_below_limit_not_truncated(self):
        tui = make_tui(truncation=50)
        tui.display_tool_result("grep", "short output")
        text = out(tui)
        assert "short output" in text
        assert "truncated" not in text
        assert "tool result: grep" in text

    def test_display_tool_result_over_limit_is_truncated(self):
        tui = make_tui(truncation=10)
        tui.display_tool_result("grep", "x" * 40)
        text = out(tui)
        assert "... (truncated)" in text

    def test_default_truncation_is_2000(self):
        tui = make_tui()  # no override
        assert tui._tool_result_truncation == 2000
        # exactly-at-limit content is not truncated
        tui.display_tool_result("t", "y" * 2000)
        assert "truncated" not in out(tui)

    def test_display_tool_error_renders_message(self):
        tui = make_tui()
        tui.display_tool_error("write_file", "permission denied")
        text = out(tui)
        assert "permission denied" in text
        assert "tool error: write_file" in text

    def test_display_error_panel(self):
        tui = make_tui()
        tui.display_error("boom")
        assert "boom" in out(tui)
        assert "error" in out(tui)

    def test_display_memory_truncated(self):
        tui = make_tui()
        tui.display_memory_truncated("dropped 3 old messages")
        text = out(tui)
        assert "dropped 3 old messages" in text
        assert "memory" in text

    def test_display_repetition_detected(self):
        tui = make_tui()
        tui.display_repetition_detected()
        assert "Repetitive output detected" in out(tui)

    def test_display_tool_failures_exceeded(self):
        tui = make_tui()
        tui.display_tool_failures_exceeded("too many failures")
        text = out(tui)
        assert "too many failures" in text
        assert "tool failures exceeded" in text

    def test_display_model_load_error(self):
        tui = make_tui()
        tui.display_model_load_error("could not load weights")
        assert "could not load weights" in out(tui)

    def test_display_tool_auto_judging(self):
        tui = make_tui()
        tui.display_tool_auto_judging("rm")
        assert "Judging rm" in out(tui)


class TestToolDenied:
    def test_denied_by_user(self):
        tui = make_tui()
        tui.display_tool_denied("rm", reason="user")
        text = out(tui)
        assert "denied by user" in text
        assert "tool denied" in text

    def test_denied_by_auto(self):
        tui = make_tui()
        tui.display_tool_denied("rm", reason="auto")
        assert "denied by safety check" in out(tui)

    def test_denied_default_policy(self):
        tui = make_tui()
        tui.display_tool_denied("rm")  # default reason -> "policy"
        assert "blocked by safety policy" in out(tui)


class TestConfirmToolCall:
    def test_always_allow_short_circuits(self):
        tui = make_tui()
        tui._always_allow.add("rm")
        # No console.input call should be needed; result is True.
        assert tui.confirm_tool_call("rm", {}) is True

    def test_yes_answer_approves(self):
        tui = make_tui()
        tui.console.input = MagicMock(return_value="y")
        assert tui.confirm_tool_call("ls", {}) is True

    def test_no_answer_rejects(self):
        tui = make_tui()
        tui.console.input = MagicMock(return_value="n")
        assert tui.confirm_tool_call("ls", {}) is False

    def test_always_answer_caches_tool(self):
        tui = make_tui()
        tui.console.input = MagicMock(return_value="a")
        assert tui.confirm_tool_call("ls", {}) is True
        assert "ls" in tui._always_allow
        # subsequent call returns True without prompting again
        tui.console.input = MagicMock(side_effect=AssertionError("should not prompt"))
        assert tui.confirm_tool_call("ls", {}) is True

    def test_eof_during_confirm_rejects(self):
        tui = make_tui()
        tui.console.input = MagicMock(side_effect=EOFError)
        assert tui.confirm_tool_call("ls", {}) is False

    def test_keyboard_interrupt_during_confirm_rejects(self):
        tui = make_tui()
        tui.console.input = MagicMock(side_effect=KeyboardInterrupt)
        assert tui.confirm_tool_call("ls", {}) is False


class TestAskQuestion:
    def test_with_options_returns_stripped_choice(self):
        tui = make_tui()
        tui.console.input = MagicMock(return_value="  yes  ")
        ans = tui.ask_question("Header", "Pick one", options=["yes", "no"])
        assert ans == "yes"
        text = out(tui)
        assert "Header" in text
        assert "Pick one" in text
        assert "yes" in text and "no" in text

    def test_without_options_returns_stripped_answer(self):
        tui = make_tui()
        tui.console.input = MagicMock(return_value="  hello ")
        ans = tui.ask_question("Header", "Type something")
        assert ans == "hello"

    def test_eof_returns_none(self):
        tui = make_tui()
        tui.console.input = MagicMock(side_effect=EOFError)
        assert tui.ask_question("H", "Q", options=["a"]) is None

    def test_keyboard_interrupt_returns_none(self):
        tui = make_tui()
        tui.console.input = MagicMock(side_effect=KeyboardInterrupt)
        assert tui.ask_question("H", "Q") is None


class TestGetUserInput:
    def test_single_line(self):
        tui = make_tui()
        tui.console.input = MagicMock(return_value="hello")
        assert tui.get_user_input() == "hello"

    def test_line_continuation_joins_with_newline(self):
        tui = make_tui()
        tui.console.input = MagicMock(side_effect=["first\\", "second"])
        assert tui.get_user_input() == "first\nsecond"

    def test_eof_returns_none(self):
        tui = make_tui()
        tui.console.input = MagicMock(side_effect=EOFError)
        assert tui.get_user_input() is None


class TestSafetyPolicyDisplay:
    def test_confirm_default_no_overrides(self):
        policy = ToolSafetyPolicy(ToolSafetyConfig(default_policy=ToolPolicy.CONFIRM))
        tui = make_tui()
        tui.display_safety_policy(policy)
        text = out(tui)
        assert "Default policy: confirm" in text
        assert "no per-tool overrides" in text
        assert "Local tools" in text

    def test_auto_default_without_judge_shows_not_configured(self):
        policy = ToolSafetyPolicy(ToolSafetyConfig(default_policy=ToolPolicy.AUTO))
        tui = make_tui()
        tui.display_safety_policy(policy)
        text = out(tui)
        assert "Default policy: auto" in text
        assert "NOT CONFIGURED" in text

    async def _judge(self, name, args, ctx):  # pragma: no cover - not called
        return True

    def test_auto_default_with_judge_shows_available(self):
        policy = ToolSafetyPolicy(
            ToolSafetyConfig(default_policy=ToolPolicy.AUTO),
            llm_judge=self._judge,
        )
        tui = make_tui()
        tui.display_safety_policy(policy)
        text = out(tui)
        assert "LLM judge: available" in text

    def test_per_tool_overrides_sorted(self):
        config = ToolSafetyConfig(
            default_policy=ToolPolicy.CONFIRM,
            tool_policies={"zsh": ToolPolicy.DENY, "ls": ToolPolicy.ALLOW},
        )
        tui = make_tui()
        tui.display_safety_policy(ToolSafetyPolicy(config))
        text = out(tui)
        # both overrides listed; "ls" should appear before "zsh" (sorted)
        assert "ls" in text and "zsh" in text
        assert text.index("ls") < text.index("zsh")
        assert "allow" in text and "deny" in text


class TestStreamContextState:
    def test_initial_text_written_on_enter(self):
        console = Console(file=StringIO(), force_terminal=True)
        ctx = StreamContext(console, initial_text="seed ")
        with ctx:
            ctx.update("more")
        assert ctx.get_text() == "seed more"

    def test_is_active_lifecycle(self):
        console = Console(file=StringIO(), force_terminal=True)
        ctx = StreamContext(console)
        assert ctx.is_active is False
        with ctx:
            assert ctx.is_active is True
        assert ctx.is_active is False

    def test_finish_is_idempotent(self):
        console = Console(file=StringIO(), force_terminal=True)
        ctx = StreamContext(console)
        with ctx:
            ctx.update("x")
        # already finished by __exit__; calling again must be a no-op
        ctx.finish()
        ctx.finish()
        assert ctx.is_active is False

    def test_thinking_then_response_separated(self):
        console = Console(file=StringIO(), force_terminal=True)
        ctx = StreamContext(console)
        with ctx:
            ctx.start_thinking()
            ctx.update("reasoning")
            ctx.end_thinking()
            ctx.update("answer")
        assert ctx.get_text() == "answer"
        assert ctx.get_thinking_text() == "reasoning"

    def test_start_thinking_idempotent_state(self):
        console = Console(file=StringIO(), force_terminal=True)
        ctx = StreamContext(console)
        with ctx:
            ctx.start_thinking()
            assert ctx._in_thinking is True
            ctx.start_thinking()  # second call should be a no-op
            assert ctx._in_thinking is True
            ctx.end_thinking()
            assert ctx._in_thinking is False
            ctx.end_thinking()  # no-op when already off
            assert ctx._in_thinking is False

    def test_write_ansi_emits_codes_when_tty(self, monkeypatch):
        import olmlx.chat.tui as tui_mod

        writes: list[str] = []
        monkeypatch.setattr(tui_mod.sys.stdout, "write", writes.append)
        monkeypatch.setattr(tui_mod.sys.stdout, "flush", lambda: None)
        console = Console(file=StringIO(), force_terminal=True)
        ctx = StreamContext(console)
        ctx._is_tty = True  # force the tty branch in _write_ansi
        ctx.start_thinking()  # writes _ITALIC_ON via _write_ansi
        assert StreamContext._ITALIC_ON in writes

    def test_write_ansi_suppressed_when_not_tty(self, monkeypatch):
        import olmlx.chat.tui as tui_mod

        writes: list[str] = []
        monkeypatch.setattr(tui_mod.sys.stdout, "write", writes.append)
        monkeypatch.setattr(tui_mod.sys.stdout, "flush", lambda: None)
        console = Console(file=StringIO(), force_terminal=True)
        ctx = StreamContext(console)
        ctx._is_tty = False  # non-tty: ANSI codes must be suppressed
        ctx.start_thinking()
        assert StreamContext._ITALIC_ON not in writes

    def test_finish_while_in_thinking_resets_flag(self):
        console = Console(file=StringIO(), force_terminal=True)
        ctx = StreamContext(console)
        with ctx:
            ctx.start_thinking()
            ctx.update("partial")
            # exit context manager while still in thinking mode
        assert ctx._in_thinking is False
        assert ctx.is_active is False
        assert ctx.get_thinking_text() == "partial"
