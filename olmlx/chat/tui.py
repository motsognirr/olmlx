"""Rich-based terminal UI for chat."""

import json
import logging
import sys

from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)


class ChatTUI:
    """Terminal UI using Rich for markdown rendering and panels."""

    def __init__(self):
        self.console = Console()

    def display_welcome(self, model_name: str, tools: list[dict]) -> None:
        """Show welcome panel with model and tool info."""
        lines = [f"Model: {model_name}"]
        if tools:
            lines.append(f"Tools: {len(tools)} available")
            for tool in tools:
                name = tool.get("function", {}).get("name", "?")
                desc = tool.get("function", {}).get("description", "")
                lines.append(f"  - {name}: {desc}")
        else:
            lines.append("Tools: none (use --mcp-config or ~/.olmlx/mcp.json)")
        lines.append("")
        lines.append(
            "Commands: /exit, /clear, /tools, /system <prompt>, "
            "/model <name>, /model thinking on|off"
        )
        self.console.print(
            Panel("\n".join(lines), title="olmlx chat", border_style="blue")
        )

    def get_user_input(self) -> str | None:
        """Prompt for user input. Returns None on exit (Ctrl+D/Ctrl+C)."""
        try:
            lines = []
            prompt = "[bold green]> [/bold green]"
            while True:
                if not lines:
                    line = self.console.input(prompt)
                else:
                    line = self.console.input("[dim]... [/dim]")

                if line.endswith("\\"):
                    lines.append(line[:-1])
                else:
                    lines.append(line)
                    break

            return "\n".join(lines)
        except (EOFError, KeyboardInterrupt):
            return None

    def stream_response(self, initial_text: str = "") -> "StreamContext":
        """Return a context manager for streaming response display."""
        return StreamContext(self.console, initial_text)

    def display_tool_call(self, name: str, arguments: dict) -> None:
        """Show tool invocation panel."""
        args_str = json.dumps(arguments, indent=2) if arguments else "{}"
        self.console.print(
            Panel(
                f"[bold]{name}[/bold]\n{args_str}",
                title="tool call",
                border_style="yellow",
            )
        )

    def display_tool_result(self, name: str, result: str) -> None:
        """Show tool response panel."""
        # Truncate long results
        display = result if len(result) <= 2000 else result[:2000] + "\n... (truncated)"
        self.console.print(
            Panel(display, title=f"tool result: {name}", border_style="green")
        )

    def display_tool_error(self, name: str, error: str) -> None:
        """Show tool error panel."""
        self.console.print(
            Panel(
                f"[red]{error}[/red]",
                title=f"tool error: {name}",
                border_style="red",
            )
        )

    def display_error(self, message: str) -> None:
        """Show red error panel."""
        self.console.print(
            Panel(f"[red]{message}[/red]", title="error", border_style="red")
        )

    def display_tools(self, tools: list[dict]) -> None:
        """Show available tools."""
        if not tools:
            self.console.print("[dim]No tools available[/dim]")
            return
        lines = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "?")
            desc = func.get("description", "")
            lines.append(f"[bold]{name}[/bold]: {desc}")
        self.console.print(
            Panel("\n".join(lines), title="available tools", border_style="blue")
        )


class StreamContext:
    """Context manager for streaming display.

    Writes tokens directly to stdout during streaming so output scrolls
    naturally (Rich.Live truncates at terminal height, hiding long output).
    Thinking tokens are displayed in italic+dim style, response tokens normally.
    """

    # ANSI escape codes for italic+dim and reset
    _ITALIC_ON = "\033[3m\033[2m"
    _ITALIC_OFF = "\033[23m\033[22m"

    def __init__(self, console: Console, initial_text: str = ""):
        self.console = console
        self._chunks: list[str] = [initial_text] if initial_text else []
        self._thinking_chunks: list[str] = []
        self._in_thinking = False
        self._started = False
        self._is_tty = sys.stdout.isatty()

    def __enter__(self):
        if self._chunks:
            sys.stdout.write("".join(self._chunks))
            sys.stdout.flush()
        self._started = True
        return self

    def __exit__(self, *args):
        if self._started:
            if self._in_thinking:
                self._write_ansi(self._ITALIC_OFF)
                self._in_thinking = False
            # End the streaming line
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._started = False

    def _write_ansi(self, code: str) -> None:
        """Write ANSI escape code only if stdout is a terminal."""
        if self._is_tty:
            sys.stdout.write(code)
            sys.stdout.flush()

    def start_thinking(self) -> None:
        """Enter thinking mode — subsequent tokens display in italic."""
        if not self._in_thinking:
            self._write_ansi(self._ITALIC_ON)
            self._in_thinking = True

    def end_thinking(self) -> None:
        """Exit thinking mode — subsequent tokens display normally."""
        if self._in_thinking:
            self._write_ansi(self._ITALIC_OFF + "\n")
            self._in_thinking = False

    def update(self, token: str) -> None:
        """Append a token and write it directly to stdout."""
        if self._in_thinking:
            self._thinking_chunks.append(token)
        else:
            self._chunks.append(token)
        sys.stdout.write(token)
        sys.stdout.flush()

    def get_text(self) -> str:
        """Return the accumulated response text (excluding thinking)."""
        return "".join(self._chunks)

    def get_thinking_text(self) -> str:
        """Return the accumulated thinking text."""
        return "".join(self._thinking_chunks)
