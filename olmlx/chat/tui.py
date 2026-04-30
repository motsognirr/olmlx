"""Rich-based terminal UI for chat."""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from olmlx.chat.tool_safety import ToolSafetyPolicy

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
            "Commands: /exit, /clear, /tools, /safety, /system <prompt>, "
            "/model <name>, /model thinking on|off, /mode auto|confirm"
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

    def display_memory_truncated(self, message: str) -> None:
        """Show warning that chat history was truncated due to memory."""
        self.console.print(
            Panel(f"[yellow]{message}[/yellow]", title="memory", border_style="yellow")
        )

    def confirm_tool_call(self, name: str, arguments: dict) -> bool:
        """Prompt user to approve a tool call. Returns True if approved."""
        self.display_tool_call(name, arguments)
        try:
            answer = self.console.input("[yellow]Allow? [y/N] [/yellow]")
            return answer.strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def ask_question(
        self,
        header: str,
        question: str,
        options: list | None = None,
        multiple: bool = False,
    ) -> str | None:
        """Prompt user with a question. Returns the answer or None on EOF."""
        try:
            if options:
                opts_str = ", ".join(f"'{o}'" for o in options)
                self.console.print(f"[bold]{header}[/bold]")
                self.console.print(question)
                self.console.print(f"[dim]Options: {opts_str}[/dim]")
                answer = self.console.input("[bold cyan]Your choice: [/bold cyan]")
                return answer.strip()
            else:
                self.console.print(f"[bold]{header}[/bold]")
                self.console.print(question)
                answer = self.console.input("[bold cyan]> [/bold cyan]")
                return answer.strip()
        except (EOFError, KeyboardInterrupt):
            return None

    def display_tool_auto_judging(self, name: str) -> None:
        """Show that a tool call is being auto-judged by the LLM."""
        self.console.print(f"[dim]Judging {name}...[/dim]")

    def display_tool_denied(self, name: str, reason: str = "policy") -> None:
        """Show that a tool call was blocked."""
        if reason == "user":
            msg = f"{name} — denied by user"
        elif reason == "auto":
            msg = f"{name} — denied by safety check"
        else:
            msg = f"{name} — blocked by safety policy"
        self.console.print(
            Panel(f"[dim]{msg}[/dim]", title="tool denied", border_style="dim")
        )

    def display_safety_policy(self, policy: "ToolSafetyPolicy") -> None:
        """Show current tool safety policy."""
        default = policy.config.default_policy.value
        lines = [f"Default policy: {default}"]
        if default == "auto":
            has_judge = policy.llm_judge is not None
            lines.append(
                f"  LLM judge: {'available' if has_judge else 'NOT CONFIGURED'}"
            )
        if policy.config.tool_policies:
            for name, pol in sorted(policy.config.tool_policies.items()):
                lines.append(f"  {name}: {pol.value}")
        else:
            lines.append("  (no per-tool overrides)")
        lines.append("Local tools (skills, builtins): always allowed")
        self.console.print(
            Panel("\n".join(lines), title="tool safety", border_style="blue")
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
        self._needs_newline = False
        self._is_tty = sys.stdout.isatty()

    def __enter__(self):
        if self._chunks:
            sys.stdout.write("".join(self._chunks))
            sys.stdout.flush()
        self._needs_newline = True
        return self

    def __exit__(self, *args):
        self.finish()

    @property
    def is_active(self) -> bool:
        """Whether the stream context is currently started."""
        return self._needs_newline

    def finish(self) -> None:
        """End streaming output. Idempotent — safe to call multiple times."""
        if self._needs_newline:
            if self._in_thinking:
                self._write_ansi(self._ITALIC_OFF)
                self._in_thinking = False
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._needs_newline = False

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
        self._needs_newline = True
        sys.stdout.write(token)
        sys.stdout.flush()

    def get_text(self) -> str:
        """Return the accumulated response text (excluding thinking)."""
        return "".join(self._chunks)

    def get_thinking_text(self) -> str:
        """Return the accumulated thinking text."""
        return "".join(self._thinking_chunks)
