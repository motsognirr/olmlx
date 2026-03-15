"""Tests for olmlx.chat.tui."""

from io import StringIO

from rich.console import Console

from olmlx.chat.tui import ChatTUI, StreamContext


class TestChatTUI:
    def test_display_welcome_with_tools(self, capsys):
        tui = ChatTUI()
        tui.console = Console(file=StringIO(), force_terminal=True)
        tools = [
            {
                "type": "function",
                "function": {"name": "read_file", "description": "Read a file"},
            }
        ]
        tui.display_welcome("qwen3:8b", tools)
        output = tui.console.file.getvalue()
        assert "qwen3:8b" in output
        assert "read_file" in output

    def test_display_welcome_no_tools(self):
        tui = ChatTUI()
        tui.console = Console(file=StringIO(), force_terminal=True)
        tui.display_welcome("qwen3:8b", [])
        output = tui.console.file.getvalue()
        assert "qwen3:8b" in output
        assert "none" in output

    def test_display_error(self):
        tui = ChatTUI()
        tui.console = Console(file=StringIO(), force_terminal=True)
        tui.display_error("Something went wrong")
        output = tui.console.file.getvalue()
        assert "Something went wrong" in output

    def test_display_tools(self):
        tui = ChatTUI()
        tui.console = Console(file=StringIO(), force_terminal=True)
        tools = [
            {
                "type": "function",
                "function": {"name": "search", "description": "Search stuff"},
            }
        ]
        tui.display_tools(tools)
        output = tui.console.file.getvalue()
        assert "search" in output
        assert "Search stuff" in output

    def test_display_tools_empty(self):
        tui = ChatTUI()
        tui.console = Console(file=StringIO(), force_terminal=True)
        tui.display_tools([])
        output = tui.console.file.getvalue()
        assert "No tools" in output


class TestStreamContext:
    def test_accumulates_text(self):
        console = Console(file=StringIO(), force_terminal=True)
        ctx = StreamContext(console)
        with ctx:
            ctx.update("Hello")
            ctx.update(" world")
        assert ctx.get_text() == "Hello world"
