"""Regression coverage for olmlx.chat.builtin_tools.

Targets currently-uncovered branches: path-traversal guards, file-size limit,
read_directory, question/TodoWrite handlers, glob truncation, grep error/
fallback paths, web_fetch SSRF guards (private-IP / missing host / redirect),
and the _is_private_ip / _resolve_path helpers. All tests are hermetic: only
tmp_path, in-process subprocesses (echo/sleep), and mocked network openers.
"""

import ipaddress
from unittest.mock import MagicMock, patch

import pytest

from olmlx.chat.builtin_tools import (
    BuiltinToolManager,
    _GLOB_MAX_RESULTS,
    _is_private_ip,
    _resolve_path,
)
from olmlx.chat.config import ChatConfig
from olmlx.chat.errors import ToolError


@pytest.fixture
def config(tmp_path):
    return ChatConfig(model_name="test:latest", plans_dir=tmp_path / "plans")


@pytest.fixture
def manager(config):
    return BuiltinToolManager(config)


# -- _resolve_path / traversal guards --


class TestResolvePath:
    def test_relative_traversal_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="outside allowed directory"):
            _resolve_path("../escape.txt", base_dir=tmp_path)

    def test_relative_resolves_under_base(self, tmp_path):
        resolved = _resolve_path("sub/file.txt", base_dir=tmp_path)
        assert resolved == (tmp_path / "sub" / "file.txt").resolve()

    def test_absolute_path_passthrough(self, tmp_path):
        target = tmp_path / "abs.txt"
        assert _resolve_path(str(target)) == target.resolve()

    def test_relative_default_base_is_cwd(self, tmp_path, monkeypatch):
        # base_dir=None branch resolves against cwd.
        monkeypatch.chdir(tmp_path)
        resolved = _resolve_path("inside.txt")
        assert resolved == (tmp_path / "inside.txt").resolve()

    @pytest.mark.asyncio
    async def test_read_file_traversal_returns_user_error(
        self, manager, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        result = await manager.call_tool("read_file", {"path": "../../../etc/passwd"})
        assert isinstance(result, ToolError)
        assert result.tool_name == "read_file"
        assert result.is_user_error is True
        assert "outside allowed directory" in result.message


# -- read_file edge cases --


class TestReadFileEdges:
    @pytest.mark.asyncio
    async def test_offset_beyond_eof_returns_empty(self, manager, tmp_path):
        f = tmp_path / "short.txt"
        f.write_text("only one line\n")
        result = await manager.call_tool("read_file", {"path": str(f), "offset": 50})
        assert result == ""

    @pytest.mark.asyncio
    async def test_limit_stops_at_eof(self, manager, tmp_path):
        f = tmp_path / "two.txt"
        f.write_text("a\nb\n")
        # limit exceeds available lines -> readline returns "" -> break path.
        result = await manager.call_tool("read_file", {"path": str(f), "limit": 10})
        assert "1\ta" in result
        assert "2\tb" in result
        # Only two numbered lines emitted.
        assert result.count("\t") == 2

    @pytest.mark.asyncio
    async def test_file_too_large_is_user_error(self, manager, tmp_path):
        from olmlx.chat import builtin_tools

        f = tmp_path / "big.txt"
        f.write_text("x\n")
        # Shrink the limit instead of writing 10 MB to keep the test fast/hermetic.
        with patch.object(builtin_tools, "_READ_FILE_MAX_BYTES", 1):
            result = await manager.call_tool("read_file", {"path": str(f)})
        assert isinstance(result, ToolError)
        assert result.is_user_error is True
        assert "limit is" in result.message


# -- write_file error path --


class TestWriteFileErrors:
    @pytest.mark.asyncio
    async def test_write_oserror_is_system_error(self, manager, tmp_path):
        # Make the parent a file so mkdir/write raises OSError.
        clash = tmp_path / "clash"
        clash.write_text("i am a file")
        target = clash / "child.txt"
        result = await manager.call_tool(
            "write_file", {"path": str(target), "content": "data"}
        )
        assert isinstance(result, ToolError)
        assert result.tool_name == "write_file"
        assert result.is_user_error is False
        assert "Error writing file" in result.message

    @pytest.mark.asyncio
    async def test_write_byte_count_uses_utf8_encoding(self, manager, tmp_path):
        f = tmp_path / "u.txt"
        # "é" is 2 bytes in UTF-8 -> reported byte count must be 2, not 1 char.
        result = await manager.call_tool("write_file", {"path": str(f), "content": "é"})
        assert "Wrote 2 bytes" in result


# -- edit_file system-error path --


class TestEditFileErrors:
    @pytest.mark.asyncio
    async def test_edit_read_oserror_is_system_error(self, manager, tmp_path):
        # Path points at a directory -> read_text raises OSError (IsADirectoryError).
        d = tmp_path / "adir"
        d.mkdir()
        result = await manager.call_tool(
            "edit_file", {"path": str(d), "old_text": "x", "new_text": "y"}
        )
        assert isinstance(result, ToolError)
        assert result.tool_name == "edit_file"
        assert result.is_user_error is False


# -- glob truncation --


class TestGlobTruncation:
    @pytest.mark.asyncio
    async def test_glob_truncates_at_max(self, manager, tmp_path):
        for i in range(_GLOB_MAX_RESULTS + 5):
            (tmp_path / f"f{i:04d}.txt").write_text("")
        result = await manager.call_tool(
            "glob", {"pattern": "*.txt", "path": str(tmp_path)}
        )
        lines = result.splitlines()
        assert lines[-1] == f"... truncated at {_GLOB_MAX_RESULTS} results"
        # _GLOB_MAX_RESULTS match lines + 1 truncation notice.
        assert len(lines) == _GLOB_MAX_RESULTS + 1

    @pytest.mark.asyncio
    async def test_glob_default_path_dot(self, manager, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "only.md").write_text("")
        # No "path" arg -> default "." branch.
        result = await manager.call_tool("glob", {"pattern": "*.md"})
        assert "only.md" in result


# -- grep error / fallback paths (mock the subprocess search) --


class TestGrepBranches:
    @pytest.mark.asyncio
    async def test_grep_rg_error_returns_system_error(self, manager):
        async def fake_create(*cmd, **kwargs):
            proc = MagicMock()

            async def communicate():
                return (b"", b"rg: bad regex")

            proc.communicate = communicate
            proc.returncode = 2
            return proc

        with patch(
            "olmlx.chat.builtin_tools.asyncio.create_subprocess_exec",
            side_effect=fake_create,
        ):
            result = await manager.call_tool("grep", {"pattern": "(", "path": "."})
        assert isinstance(result, ToolError)
        assert result.tool_name == "grep"
        assert result.is_user_error is False
        assert "bad regex" in result.message

    @pytest.mark.asyncio
    async def test_grep_falls_back_to_grep_when_rg_missing(self, manager):
        calls = []

        async def fake_create(*cmd, **kwargs):
            calls.append(cmd[0])
            if cmd[0] == "rg":
                raise FileNotFoundError("rg")
            proc = MagicMock()

            async def communicate():
                return (b"path:1:hit\n", b"")

            proc.communicate = communicate
            proc.returncode = 0
            return proc

        with patch(
            "olmlx.chat.builtin_tools.asyncio.create_subprocess_exec",
            side_effect=fake_create,
        ):
            result = await manager.call_tool("grep", {"pattern": "hit", "path": "."})
        assert "hit" in result
        assert "rg" in calls and "grep" in calls

    @pytest.mark.asyncio
    async def test_grep_no_tools_available(self, manager):
        async def fake_create(*cmd, **kwargs):
            raise FileNotFoundError(cmd[0])

        with patch(
            "olmlx.chat.builtin_tools.asyncio.create_subprocess_exec",
            side_effect=fake_create,
        ):
            result = await manager.call_tool("grep", {"pattern": "x", "path": "."})
        assert isinstance(result, ToolError)
        assert "not available" in result.message

    @pytest.mark.asyncio
    async def test_grep_output_truncated(self, manager):
        from olmlx.chat import builtin_tools

        async def fake_create(*cmd, **kwargs):
            proc = MagicMock()

            async def communicate():
                return (b"y" * 5000, b"")

            proc.communicate = communicate
            proc.returncode = 0
            return proc

        with patch.object(builtin_tools, "_GREP_MAX_BYTES", 100):
            with patch(
                "olmlx.chat.builtin_tools.asyncio.create_subprocess_exec",
                side_effect=fake_create,
            ):
                result = await manager.call_tool("grep", {"pattern": "y", "path": "."})
        assert result.endswith("... truncated")
        assert len(result) <= 100 + len("\n... truncated")


# -- read_directory --


class TestReadDirectory:
    @pytest.mark.asyncio
    async def test_lists_files_and_dirs(self, manager, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("hello")  # 5 bytes
        result = await manager.call_tool("read_directory", {"path": str(tmp_path)})
        assert "sub/" in result
        assert "a.txt (5 bytes)" in result

    @pytest.mark.asyncio
    async def test_not_a_directory_is_user_error(self, manager, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = await manager.call_tool("read_directory", {"path": str(f)})
        assert isinstance(result, ToolError)
        assert result.tool_name == "read_directory"
        assert result.is_user_error is True
        assert "not a directory" in result.message

    @pytest.mark.asyncio
    async def test_traversal_rejected(self, manager, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = await manager.call_tool("read_directory", {"path": "../../.."})
        assert isinstance(result, ToolError)
        assert result.is_user_error is True
        assert "outside allowed directory" in result.message


# -- question + TodoWrite handlers --


class TestQuestionHandler:
    @pytest.mark.asyncio
    async def test_question_serializes_payload(self, manager):
        import json

        result = await manager.call_tool(
            "question",
            {
                "header": "Pick",
                "question": "Which one?",
                "multiple": True,
                "options": ["a", "b"],
            },
        )
        assert result.startswith("__question__:")
        payload = json.loads(result[len("__question__:") :])
        assert payload == {
            "header": "Pick",
            "question": "Which one?",
            "multiple": True,
            "options": ["a", "b"],
        }

    @pytest.mark.asyncio
    async def test_question_defaults(self, manager):
        import json

        result = await manager.call_tool("question", {})
        payload = json.loads(result[len("__question__:") :])
        assert payload["multiple"] is False
        assert payload["options"] is None


class TestTodoWrite:
    @pytest.mark.asyncio
    async def test_empty_clears_list(self, manager):
        result = await manager.call_tool("TodoWrite", {"todos": []})
        assert result == "Todo list cleared."

    @pytest.mark.asyncio
    async def test_formats_items_with_defaults(self, manager):
        result = await manager.call_tool(
            "TodoWrite",
            {
                "todos": [
                    {"content": "first", "priority": "high", "status": "completed"},
                    {"content": "second"},  # defaults: medium / pending
                ]
            },
        )
        lines = result.splitlines()
        assert lines[0] == "[completed] [high] 1. first"
        assert lines[1] == "[pending] [medium] 2. second"


# -- create_plan / update_plan / read_plan error paths --


class TestPlanErrors:
    @pytest.mark.asyncio
    async def test_create_plan_oserror(self, manager, config, tmp_path):
        # Make plans_dir collide with an existing file so mkdir raises.
        config.plans_dir.parent.mkdir(parents=True, exist_ok=True)
        blocker = config.plans_dir
        blocker.write_text("not a dir")
        result = await manager.call_tool("create_plan", {"content": "# Plan"})
        assert isinstance(result, ToolError)
        assert result.tool_name == "create_plan"
        assert result.is_user_error is False


# -- web_fetch SSRF guards --


class TestWebFetchGuards:
    @pytest.mark.asyncio
    async def test_missing_hostname_user_error(self, manager):
        result = await manager.call_tool("web_fetch", {"url": "http:///nohost"})
        assert isinstance(result, ToolError)
        assert "missing hostname" in result.message
        assert result.is_user_error is True

    @pytest.mark.asyncio
    async def test_fetch_failure_wrapped_as_tool_error(self, manager):
        # build_opener returns an opener whose open() raises -> generic except path.
        opener = MagicMock()
        opener.open.side_effect = OSError("connection refused")
        with patch("urllib.request.build_opener", return_value=opener):
            result = await manager.call_tool(
                "web_fetch", {"url": "https://example.com"}
            )
        assert isinstance(result, ToolError)
        assert result.tool_name == "web_fetch"
        assert "connection refused" in result.message
        # OSError is not ValueError -> system error.
        assert result.is_user_error is False


# -- _is_private_ip helper (the SSRF allowlist core) --


class TestIsPrivateIP:
    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",  # loopback
            "10.0.0.1",  # private
            "192.168.1.1",  # private
            "169.254.0.1",  # link-local
            "224.0.0.1",  # multicast
        ],
    )
    def test_private_ipv4_blocked(self, ip):
        assert _is_private_ip(ipaddress.ip_address(ip)) is True

    def test_public_ipv4_allowed(self):
        assert _is_private_ip(ipaddress.ip_address("8.8.8.8")) is False

    @pytest.mark.parametrize(
        "ip",
        [
            "::1",  # loopback
            "fc00::1",  # unique-local (private)
            "fe80::1",  # link-local
            "ff02::1",  # multicast
            "fec0::1",  # site-local
        ],
    )
    def test_private_ipv6_blocked(self, ip):
        assert _is_private_ip(ipaddress.ip_address(ip)) is True

    def test_public_ipv6_allowed(self):
        assert _is_private_ip(ipaddress.ip_address("2001:4860:4860::8888")) is False


# -- bash error/empty-output branches (no real model, just /bin/sh) --


class TestBashBranches:
    @pytest.mark.asyncio
    async def test_bash_no_output(self, manager):
        result = await manager.call_tool("bash", {"command": "true"})
        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_bash_spawn_oserror(self, manager):
        with patch(
            "olmlx.chat.builtin_tools.asyncio.create_subprocess_shell",
            side_effect=OSError("no shell"),
        ):
            result = await manager.call_tool("bash", {"command": "echo hi"})
        assert isinstance(result, ToolError)
        assert result.tool_name == "bash"
        assert "Error running command" in result.message
        assert result.is_user_error is False
