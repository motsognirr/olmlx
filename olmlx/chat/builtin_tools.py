"""Built-in tools for the chat client — file system, shell, research, planning."""

import asyncio
import glob as glob_module
import html.parser
import logging
import re
import subprocess
import urllib.request
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from olmlx.chat.config import ChatConfig

logger = logging.getLogger(__name__)

# Maximum results for glob
_GLOB_MAX_RESULTS = 500
# Maximum output for grep
_GREP_MAX_BYTES = 50_000
# Default bash timeout
_BASH_DEFAULT_TIMEOUT = 120
# Maximum characters for web_fetch output
_WEB_FETCH_MAX_CHARS = 10_000


def _web_search_impl(query: str, max_results: int = 5) -> list[dict]:
    """Run a web search via duckduckgo-search. Raises ImportError if not installed."""
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Extract visible text from HTML, stripping tags."""

    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._pieces.append(data)

    def get_text(self) -> str:
        return " ".join(self._pieces)


def _strip_html(text: str) -> str:
    """Strip HTML tags and return visible text."""
    parser = _HTMLTextExtractor()
    parser.feed(text)
    return parser.get_text()


# -- Tool handler functions --


async def _handle_read_file(args: dict) -> str:
    path = args.get("path", "")
    offset = args.get("offset", 1)
    limit = args.get("limit")

    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError as exc:
        return f"Error reading file: {exc}"

    # Apply offset (1-based)
    start = max(offset - 1, 0)
    if limit is not None:
        end = start + limit
        selected = lines[start:end]
    else:
        selected = lines[start:]

    numbered = []
    for i, line in enumerate(selected, start=start + 1):
        numbered.append(f"{i}\t{line.rstrip()}")
    return "\n".join(numbered)


async def _handle_write_file(args: dict) -> str:
    path = Path(args.get("path", ""))
    content = args.get("content", "")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    except OSError as exc:
        return f"Error writing file: {exc}"

    return f"Wrote {len(content)} bytes to {path}"


async def _handle_edit_file(args: dict) -> str:
    path = Path(args.get("path", ""))
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")

    try:
        content = path.read_text()
    except OSError as exc:
        return f"Error reading file: {exc}"

    count = content.count(old_text)
    if count == 0:
        return "Error: old_text not found in file."
    if count > 1:
        return f"Error: old_text found {count} times (multiple matches). Provide more context to make it unique."

    new_content = content.replace(old_text, new_text, 1)
    try:
        path.write_text(new_content)
    except OSError as exc:
        return f"Error writing file: {exc}"

    return "Applied edit successfully."


async def _handle_glob(args: dict) -> str:
    pattern = args.get("pattern", "")
    path = args.get("path", ".")

    matches = sorted(glob_module.glob(pattern, root_dir=path, recursive=True))
    if not matches:
        return "No matches found."

    if len(matches) > _GLOB_MAX_RESULTS:
        matches = matches[:_GLOB_MAX_RESULTS]
        return "\n".join(matches) + f"\n... truncated at {_GLOB_MAX_RESULTS} results"

    return "\n".join(matches)


async def _handle_grep(args: dict) -> str:
    pattern = args.get("pattern", "")
    path = args.get("path", ".")

    # Try rg first, fall back to grep
    try:
        rg = subprocess.run(
            ["rg", "-n", "--no-heading", pattern, path],
            capture_output=True, text=True, timeout=30,
        )
        if rg.returncode == 0:
            output = rg.stdout
        elif rg.returncode == 1:
            return "No matches found."
        else:
            raise FileNotFoundError("rg failed")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # rg not available, fall back to grep
        try:
            grep = subprocess.run(
                ["grep", "-rn", pattern, path],
                capture_output=True, text=True, timeout=30,
            )
            output = grep.stdout
            if not output:
                return "No matches found."
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return "Error: search tools (rg, grep) not available or timed out."

    if len(output) > _GREP_MAX_BYTES:
        output = output[:_GREP_MAX_BYTES] + "\n... truncated"

    return output


async def _handle_bash(args: dict) -> str:
    command = args.get("command", "")
    timeout = args.get("timeout", _BASH_DEFAULT_TIMEOUT)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return f"Command timed out after {timeout}s."
    except OSError as exc:
        return f"Error running command: {exc}"

    parts = []
    if stdout:
        parts.append(stdout.decode(errors="replace"))
    if stderr:
        parts.append(f"STDERR:\n{stderr.decode(errors='replace')}")
    if proc.returncode != 0:
        parts.append(f"Exit code: {proc.returncode}")

    return "\n".join(parts) if parts else "(no output)"


async def _handle_create_plan(args: dict, plans_dir: Path) -> str:
    content = args.get("content", "")
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plans_dir / "plan.md"
    plan_path.write_text(content)
    return f"Plan created at {plan_path}"


async def _handle_update_plan(args: dict, plans_dir: Path) -> str:
    content = args.get("content", "")
    plan_path = plans_dir / "plan.md"
    if not plan_path.exists():
        return "Error: No plan found. Use create_plan first."
    plan_path.write_text(content)
    return f"Plan updated at {plan_path}"


async def _handle_read_plan(args: dict, plans_dir: Path) -> str:
    plan_path = plans_dir / "plan.md"
    if not plan_path.exists():
        return "No plan found. Use create_plan to create one."
    return plan_path.read_text()


async def _handle_web_fetch(args: dict) -> str:
    url = args.get("url", "")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            raw = resp.read().decode(charset, errors="replace")
    except Exception as exc:
        return f"Error fetching URL: {exc}"

    text = _strip_html(raw)
    if len(text) > _WEB_FETCH_MAX_CHARS:
        text = text[:_WEB_FETCH_MAX_CHARS] + "\n... (truncated)"
    return text


async def _handle_web_search(args: dict) -> str:
    query = args.get("query", "")
    max_results = args.get("max_results", 5)

    try:
        results = _web_search_impl(query, max_results=max_results)
    except ImportError:
        return "Error: duckduckgo-search is not installed. Install it with: pip install duckduckgo-search"
    except Exception as exc:
        return f"Error performing search: {exc}"

    if not results:
        return "No results found."

    lines = []
    for r in results:
        lines.append(f"**{r.get('title', '')}**")
        lines.append(r.get("href", ""))
        lines.append(r.get("body", ""))
        lines.append("")
    return "\n".join(lines)


# -- Tool definitions (OpenAI function-calling format) --

_TOOL_DEFS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file content with optional line offset and limit. Returns lines with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"},
                    "offset": {"type": "integer", "description": "Starting line number (1-based, default: 1)"},
                    "limit": {"type": "integer", "description": "Maximum number of lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Find and replace text in a file. old_text must match exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_text": {"type": "string", "description": "Text to find (must be unique in file)"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern. Supports ** for recursive matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
                    "path": {"type": "string", "description": "Root directory to search in (default: current dir)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file content with regex. Returns matching lines with file:line format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "File or directory to search (default: current dir)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and return stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Requires optional duckduckgo-search package.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum results (default: 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return its text content (HTML tags stripped). Truncated to 10k chars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_plan",
            "description": "Write a markdown plan to the plans directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Plan content in markdown"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": "Overwrite an existing plan file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Updated plan content"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_plan",
            "description": "Read the current plan content.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# Mapping of tool name -> handler function
_SIMPLE_HANDLERS: dict[str, Callable] = {
    "read_file": _handle_read_file,
    "write_file": _handle_write_file,
    "edit_file": _handle_edit_file,
    "glob": _handle_glob,
    "grep": _handle_grep,
    "bash": _handle_bash,
    "web_fetch": _handle_web_fetch,
    "web_search": _handle_web_search,
}

_PLAN_HANDLERS: dict[str, Callable] = {
    "create_plan": _handle_create_plan,
    "update_plan": _handle_update_plan,
    "read_plan": _handle_read_plan,
}


class BuiltinToolManager:
    """Manages built-in tools for the chat client."""

    def __init__(self, config: ChatConfig):
        self._config = config

    @property
    def tool_names(self) -> set[str]:
        return {d["function"]["name"] for d in _TOOL_DEFS}

    def get_tool_definitions(self) -> list[dict]:
        return list(_TOOL_DEFS)

    async def call_tool(self, name: str, arguments: dict) -> str:
        if name in _SIMPLE_HANDLERS:
            return await _SIMPLE_HANDLERS[name](arguments)
        if name in _PLAN_HANDLERS:
            return await _PLAN_HANDLERS[name](arguments, self._config.plans_dir)
        raise ValueError(f"Unknown built-in tool: {name!r}")
