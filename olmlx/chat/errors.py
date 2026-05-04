"""Shared error types for the chat subsystem."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolError:
    """Unified error type for tool execution failures.

    Replaces the previous ad-hoc patterns:
    - ``builtin_tools.py`` — error strings ``"Error: ..."``
    - ``mcp_client.py`` — raised exceptions (ValueError, RuntimeError)
    - ``session.py`` — TypedDict ``_ToolErrorEvent`` constructed manually

    ``is_user_error`` distinguishes user-input problems (bad path,
    invalid arguments) from system errors (connection refused, timeout)
    so callers can format or filter differently.

    ``tool_name`` is carried for logging and for programmatic callers
    (MCP, tests); the session layer uses the dispatch-time name as the
    authoritative source in emitted events.
    """

    message: str
    tool_name: str
    is_user_error: bool = False

    def __str__(self) -> str:
        return self.message
