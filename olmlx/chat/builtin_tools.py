"""Built-in tools for the chat client — file system, shell, research, planning."""

import asyncio
import glob as glob_module
import http.client
import ipaddress
import logging
import os
import signal
import socket
import ssl
import urllib.parse
import urllib.request
import urllib.error
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

from olmlx.chat.config import ChatConfig
from olmlx.chat.errors import ToolError

logger = logging.getLogger(__name__)

# Maximum file size for read_file (10 MB)
_READ_FILE_MAX_BYTES = 10 * 1024 * 1024
# Maximum results for glob
_GLOB_MAX_RESULTS = 500
# Maximum output for grep
_GREP_MAX_BYTES = 50_000
# Default bash timeout
_BASH_DEFAULT_TIMEOUT = 120
# Maximum output for bash
_BASH_MAX_BYTES = 100_000
# Maximum characters for web_fetch output
_WEB_FETCH_MAX_CHARS = 10_000


def _web_search_impl(query: str, max_results: int = 5) -> list[dict]:
    """Run a web search via duckduckgo-search. Raises ImportError if not installed."""
    from duckduckgo_search import DDGS  # type: ignore[import-not-found]

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


class _HTMLTextExtractor(HTMLParser):
    """Extract visible text from HTML, stripping tags."""

    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = max(0, self._skip - 1)

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


def _resolve_path(path: str, base_dir: Path | None = None) -> Path:
    """Resolve and validate path to prevent path traversal attacks.

    Handles both relative and absolute paths. For absolute paths, returns
    resolved directly. For relative paths, resolves relative to base_dir
    (defaults to cwd) and prevents traversal via '..'.

    Args:
        path: User-provided file path (absolute or relative)
        base_dir: Allowed base directory for relative paths (defaults to cwd)

    Returns:
        Resolved absolute Path

    Raises:
        ValueError: If path attempts to escape base_dir via traversal
    """
    input_path = Path(path).expanduser()

    if input_path.is_absolute():
        return input_path.resolve()

    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = base_dir.resolve()

    resolved = (base_dir / path).resolve()

    try:
        resolved.relative_to(base_dir)
    except ValueError:
        raise ValueError(f"Path {path!r} is outside allowed directory {base_dir}")
    return resolved


# -- Tool handler functions --


async def _handle_read_file(args: dict) -> str | ToolError:
    path = args.get("path", "")
    try:
        safe_path = _resolve_path(path)
    except ValueError as exc:
        return ToolError(message=str(exc), tool_name="read_file", is_user_error=True)
    offset = args.get("offset", 1)
    limit = args.get("limit")
    start = max(offset - 1, 0)

    def _read() -> list[str]:
        with open(safe_path, errors="replace") as f:
            # Check file size after opening to avoid TOCTOU
            f.seek(0, 2)
            size = f.tell()
            if size > _READ_FILE_MAX_BYTES:
                raise ValueError(
                    f"file is {size} bytes (limit is {_READ_FILE_MAX_BYTES}). "
                    "Use offset/limit for large files."
                )
            f.seek(0)
            # Skip lines before offset
            for _ in range(start):
                if not f.readline():
                    break
            if limit is not None:
                lines = []
                for _ in range(limit):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
                return lines
            return f.readlines()

    try:
        selected = await asyncio.to_thread(_read)
    except (OSError, ValueError) as exc:
        return ToolError(
            message=f"Error reading file: {exc}",
            tool_name="read_file",
            is_user_error=isinstance(exc, ValueError),
        )

    numbered = []
    for i, line in enumerate(selected, start=start + 1):
        numbered.append(f"{i}\t{line.rstrip()}")
    return "\n".join(numbered)


async def _handle_write_file(args: dict) -> str | ToolError:
    path = args.get("path", "")
    content = args.get("content", "")

    try:
        safe_path = _resolve_path(path)
    except ValueError as exc:
        return ToolError(message=str(exc), tool_name="write_file", is_user_error=True)

    def _write():
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")

    try:
        await asyncio.to_thread(_write)
    except OSError as exc:
        return ToolError(
            message=f"Error writing file: {exc}",
            tool_name="write_file",
            is_user_error=False,
        )

    return f"Wrote {len(content.encode())} bytes to {safe_path}"


async def _handle_edit_file(args: dict) -> str | ToolError:
    path = args.get("path", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")

    try:
        safe_path = _resolve_path(path)
    except ValueError as exc:
        return ToolError(message=str(exc), tool_name="edit_file", is_user_error=True)

    def _edit() -> str | ToolError:
        try:
            content = safe_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolError(
                message=f"Error: {exc}",
                tool_name="edit_file",
                is_user_error=False,
            )
        count = content.count(old_text)
        if count == 0:
            return ToolError(
                message="old_text not found in file.",
                tool_name="edit_file",
                is_user_error=True,
            )
        if count > 1:
            return ToolError(
                message=f"old_text found {count} times (multiple matches). Provide more context to make it unique.",
                tool_name="edit_file",
                is_user_error=True,
            )
        try:
            new_content = content.replace(old_text, new_text, 1)
            safe_path.write_text(new_content, encoding="utf-8")
        except OSError as exc:
            return ToolError(
                message=f"Error: {exc}",
                tool_name="edit_file",
                is_user_error=False,
            )
        return "Applied edit successfully."

    return await asyncio.to_thread(_edit)


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


async def _handle_grep(args: dict) -> str | ToolError:
    pattern = args.get("pattern", "")
    path = args.get("path", ".")

    # Max lines to return from search to bound output at the source
    max_count = "1000"

    async def _run_search(cmd: list[str]) -> tuple[str, str, int | None]:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=30,
            )
        except BaseException:
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
            raise
        return (
            stdout_bytes.decode(errors="replace"),
            stderr_bytes.decode(errors="replace"),
            proc.returncode,
        )

    # Try rg first, fall back to grep
    try:
        stdout, stderr, rc = await _run_search(
            ["rg", "-n", "--no-heading", "-m", max_count, pattern, path]
        )
        if rc == 0:
            output = stdout
        elif rc == 1:
            return "No matches found."
        else:
            return ToolError(
                message=stderr.strip() or "rg returned exit code " + str(rc),
                tool_name="grep",
                is_user_error=True,  # exit code 2+ is typically invalid regex
            )
    except FileNotFoundError:
        # rg not installed, fall back to grep
        try:
            stdout, stderr, rc = await _run_search(
                ["grep", "-rn", "-m", max_count, pattern, path]
            )
            if rc == 0:
                output = stdout
            elif rc == 1:
                return "No matches found."
            else:
                return ToolError(
                    message=stderr.strip() or "grep returned exit code " + str(rc),
                    tool_name="grep",
                    is_user_error=False,  # grep exit code 2+ is system errors (perm, I/O)
                )
        except FileNotFoundError:
            return ToolError(
                message="search tools (rg, grep) not available.",
                tool_name="grep",
                is_user_error=False,
            )
        except asyncio.TimeoutError:
            return ToolError(
                message="search timed out.",
                tool_name="grep",
                is_user_error=False,
            )
    except asyncio.TimeoutError:
        return ToolError(
            message="search timed out.",
            tool_name="grep",
            is_user_error=False,
        )

    if len(output) > _GREP_MAX_BYTES:
        output = output[:_GREP_MAX_BYTES] + "\n... truncated"

    return output


async def _handle_read_directory(args: dict) -> str | ToolError:
    path = args.get("path", ".")

    try:
        safe_path = _resolve_path(path)
    except ValueError as exc:
        return ToolError(
            message=str(exc), tool_name="read_directory", is_user_error=True
        )

    def _list_dir() -> list[str] | ToolError:
        if not safe_path.is_dir():
            return ToolError(
                message=f"not a directory: {safe_path}",
                tool_name="read_directory",
                is_user_error=True,
            )
        lines = []
        for entry in sorted(safe_path.iterdir()):
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                size = entry.stat().st_size
                lines.append(f"{entry.name} ({size} bytes)")
        return lines

    try:
        entries = await asyncio.to_thread(_list_dir)
    except OSError as exc:
        return ToolError(
            message=f"Error listing directory: {exc}",
            tool_name="read_directory",
            is_user_error=False,
        )

    if isinstance(entries, ToolError):
        return entries

    return "\n".join(entries)


async def _handle_question(args: dict) -> str:
    import json

    payload = {
        "header": args.get("header", ""),
        "question": args.get("question", ""),
        "multiple": args.get("multiple", False),
        "options": args.get("options"),
    }
    return "__question__:" + json.dumps(payload)


async def _handle_bash(args: dict) -> str | ToolError:
    command = args.get("command", "")
    timeout = args.get("timeout", _BASH_DEFAULT_TIMEOUT)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        return ToolError(
            message=f"Error running command: {exc}",
            tool_name="bash",
            is_user_error=isinstance(exc, FileNotFoundError),
        )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        return ToolError(
            message=f"Command timed out after {timeout}s.",
            tool_name="bash",
            is_user_error=False,
        )
    except OSError as exc:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        return ToolError(
            message=f"Error running command: {exc}",
            tool_name="bash",
            is_user_error=False,
        )
    except BaseException:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        raise

    parts = []
    if stdout:
        text = stdout.decode(errors="replace")
        if len(text) > _BASH_MAX_BYTES:
            text = text[:_BASH_MAX_BYTES] + "\n... (truncated)"
        parts.append(text)
    if stderr:
        text = stderr.decode(errors="replace")
        if len(text) > _BASH_MAX_BYTES:
            text = text[:_BASH_MAX_BYTES] + "\n... (truncated)"
        parts.append(f"STDERR:\n{text}")
    if proc.returncode != 0:
        parts.append(f"Exit code: {proc.returncode}")

    return "\n".join(parts) if parts else "(no output)"


async def _handle_todo_write(args: dict) -> str:
    todos = args.get("todos", [])
    if not todos:
        return "Todo list cleared."

    lines = []
    for i, item in enumerate(todos, 1):
        content = item.get("content", "")
        priority = item.get("priority", "medium")
        status = item.get("status", "pending")
        lines.append(f"[{status}] [{priority}] {i}. {content}")

    return "\n".join(lines)


async def _handle_create_plan(args: dict, plans_dir: Path) -> str | ToolError:
    content = args.get("content", "")

    def _create():
        plans_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plans_dir / "plan.md"
        plan_path.write_text(content, encoding="utf-8")
        return f"Plan created at {plan_path}"

    try:
        return await asyncio.to_thread(_create)
    except OSError as exc:
        return ToolError(
            message=f"Error writing plan: {exc}",
            tool_name="create_plan",
            is_user_error=False,
        )


async def _handle_update_plan(args: dict, plans_dir: Path) -> str | ToolError:
    content = args.get("content", "")
    plan_path = plans_dir / "plan.md"

    def _update() -> str | ToolError:
        if not plan_path.exists():
            return ToolError(
                message="No plan found. Use create_plan first.",
                tool_name="update_plan",
                is_user_error=True,
            )
        try:
            plan_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolError(
                message=f"Error writing plan: {exc}",
                tool_name="update_plan",
                is_user_error=False,
            )
        return f"Plan updated at {plan_path}"

    return await asyncio.to_thread(_update)


async def _handle_read_plan(args: dict, plans_dir: Path) -> str | ToolError:
    plan_path = plans_dir / "plan.md"

    def _read() -> str | ToolError:
        if not plan_path.exists():
            return ToolError(
                message="No plan found. Use create_plan to create one.",
                tool_name="read_plan",
                is_user_error=True,
            )
        return plan_path.read_text(encoding="utf-8", errors="replace")

    try:
        return await asyncio.to_thread(_read)
    except OSError as exc:
        return ToolError(
            message=f"Error reading plan: {exc}",
            tool_name="read_plan",
            is_user_error=False,
        )


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv4Address):
        return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_site_local
    )


async def _handle_web_fetch(args: dict) -> str | ToolError:
    url = args.get("url", "")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ToolError(
            message=f"unsupported URL scheme {parsed.scheme!r}. Only http and https are allowed.",
            tool_name="web_fetch",
            is_user_error=True,
        )

    if not parsed.hostname:
        return ToolError(
            message="invalid URL: missing hostname.",
            tool_name="web_fetch",
            is_user_error=True,
        )

    class _SafeHTTPConnection(http.client.HTTPConnection):
        def connect(self):
            super().connect()
            sock_ip = self.sock.getpeername()[0]
            try:
                addr = ipaddress.ip_address(sock_ip)
            except ValueError:
                raise urllib.error.URLError(f"Invalid IP address: {sock_ip}")
            if _is_private_ip(addr):
                raise urllib.error.URLError(
                    f"Connection to private IP {sock_ip} blocked"
                )

    class _SafeHTTPSConnection(_SafeHTTPConnection, http.client.HTTPSConnection):
        pass

    class _SafeHTTPHandler(urllib.request.HTTPHandler):
        def http_open(self, req, *args, **kwargs):
            return self.do_open(_SafeHTTPConnection, req, *args, **kwargs)

    class _SafeHTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req, *args, **kwargs):
            ctx = ssl.create_default_context()
            return self.do_open(
                lambda host, **kw: _SafeHTTPSConnection(host, context=ctx, **kw),
                req,
                *args,
                **kwargs,
            )

    class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            p = urllib.parse.urlparse(newurl)
            if p.scheme not in ("http", "https"):
                raise urllib.error.URLError(
                    f"Redirect to non-HTTP(S) URL blocked: {newurl}"
                )
            if p.hostname:
                try:
                    results = socket.getaddrinfo(
                        p.hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
                    )
                    for family, _, _, _, sockaddr in results:
                        ip = sockaddr[0]
                        addr = ipaddress.ip_address(ip)
                        if _is_private_ip(addr):
                            raise urllib.error.URLError(
                                f"Redirect to private IP blocked: {ip}"
                            )
                except socket.gaierror as exc:
                    raise urllib.error.URLError(
                        f"Failed to resolve redirect hostname {p.hostname}: {exc}"
                    )
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    def _fetch() -> str:
        opener = urllib.request.build_opener(
            _SafeHTTPHandler, _SafeHTTPSHandler, _SafeRedirectHandler
        )
        with opener.open(url, timeout=30) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read(_WEB_FETCH_MAX_CHARS * 10).decode(
                charset, errors="replace"
            )

    try:
        raw = await asyncio.to_thread(_fetch)
    except Exception as exc:
        return ToolError(
            message=f"Error fetching URL: {exc}",
            tool_name="web_fetch",
            is_user_error=isinstance(exc, ValueError),
        )

    text = _strip_html(raw)
    if len(text) > _WEB_FETCH_MAX_CHARS:
        text = text[:_WEB_FETCH_MAX_CHARS] + "\n... (truncated)"
    return text


async def _handle_web_search(args: dict) -> str | ToolError:
    query = args.get("query", "")
    max_results = args.get("max_results", 5)

    try:
        results = await asyncio.to_thread(_web_search_impl, query, max_results)
    except ImportError:
        return ToolError(
            message="duckduckgo-search is not installed. Install it with: pip install duckduckgo-search",
            tool_name="web_search",
            is_user_error=False,
        )
    except Exception as exc:
        return ToolError(
            message=f"Error performing search: {exc}",
            tool_name="web_search",
            is_user_error=False,
        )

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
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Starting line number (1-based, default: 1)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read",
                    },
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
                    "old_text": {
                        "type": "string",
                        "description": "Text to find (must be unique in file)",
                    },
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
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.py')",
                    },
                    "path": {
                        "type": "string",
                        "description": "Root directory to search in (default: current dir)",
                    },
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
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search (default: current dir)",
                    },
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
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 120)",
                    },
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
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default: 5)",
                    },
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
                    "content": {
                        "type": "string",
                        "description": "Plan content in markdown",
                    },
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
                    "content": {
                        "type": "string",
                        "description": "Updated plan content",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_directory",
            "description": "List files and directories in a path (ls equivalent).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative directory path (default: current directory)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "question",
            "description": "Ask the user a question and return their answer. Use for disambiguation, choosing between options, or gathering required input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "header": {
                        "type": "string",
                        "description": "Short label for the question (max 30 chars)",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user",
                    },
                    "multiple": {
                        "type": "boolean",
                        "description": "Allow selecting multiple options (default: false)",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Available choices",
                    },
                },
                "required": ["header", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TodoWrite",
            "description": "Create and manage a task list for the current session. Tracks todo items by content, priority, and status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Brief description of the task",
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                    "description": "Priority level",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "in_progress",
                                        "completed",
                                        "cancelled",
                                        "pending",
                                    ],
                                    "description": "Current status",
                                },
                            },
                        },
                        "description": "List of todo items to set",
                    },
                },
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
    "read_directory": _handle_read_directory,
    "web_fetch": _handle_web_fetch,
    "web_search": _handle_web_search,
}

_PLAN_HANDLERS: dict[str, Callable] = {
    "create_plan": _handle_create_plan,
    "update_plan": _handle_update_plan,
    "read_plan": _handle_read_plan,
}

_TODO_HANDLERS: dict[str, Callable] = {
    "TodoWrite": _handle_todo_write,
}

_QUESTION_HANDLERS: dict[str, Callable] = {
    "question": _handle_question,
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

    async def call_tool(self, name: str, arguments: dict) -> str | ToolError:
        if name in _SIMPLE_HANDLERS:
            return await _SIMPLE_HANDLERS[name](arguments)
        if name in _PLAN_HANDLERS:
            return await _PLAN_HANDLERS[name](arguments, self._config.plans_dir)
        if name in _TODO_HANDLERS:
            return await _TODO_HANDLERS[name](arguments)
        if name in _QUESTION_HANDLERS:
            return await _QUESTION_HANDLERS[name](arguments)
        return ToolError(
            message=f"Unknown built-in tool: {name!r}",
            tool_name=name,
            is_user_error=True,
        )
