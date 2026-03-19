"""Tests for olmlx.chat.builtin_tools."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olmlx.chat.builtin_tools import BuiltinToolManager
from olmlx.chat.config import ChatConfig


@pytest.fixture
def config(tmp_path):
    return ChatConfig(
        model_name="test:latest",
        plans_dir=tmp_path / "plans",
    )


@pytest.fixture
def manager(config):
    return BuiltinToolManager(config)


class TestBuiltinToolManagerSkeleton:
    def test_get_tool_definitions_returns_list(self, manager):
        defs = manager.get_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) > 0
        for d in defs:
            assert d["type"] == "function"
            assert "name" in d["function"]
            assert "description" in d["function"]
            assert "parameters" in d["function"]

    def test_tool_names_property(self, manager):
        names = manager.tool_names
        assert isinstance(names, set)
        expected = {
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "bash",
            "web_search",
            "web_fetch",
            "create_plan",
            "update_plan",
            "read_plan",
        }
        assert names == expected

    @pytest.mark.asyncio
    async def test_call_unknown_tool_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown built-in tool"):
            await manager.call_tool("nonexistent", {})


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_file_basic(self, manager, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("line one\nline two\nline three\n")
        result = await manager.call_tool("read_file", {"path": str(f)})
        assert "1\tline one" in result
        assert "2\tline two" in result
        assert "3\tline three" in result

    @pytest.mark.asyncio
    async def test_read_file_offset_limit(self, manager, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("\n".join(f"line {i}" for i in range(1, 11)))
        result = await manager.call_tool(
            "read_file",
            {
                "path": str(f),
                "offset": 3,
                "limit": 2,
            },
        )
        assert "3\tline 3" in result
        assert "4\tline 4" in result
        assert "5\tline 5" not in result
        assert "2\tline 2" not in result

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, manager, tmp_path):
        result = await manager.call_tool(
            "read_file", {"path": str(tmp_path / "nope.txt")}
        )
        assert "Error" in result or "error" in result


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_file_creates(self, manager, tmp_path):
        f = tmp_path / "new.txt"
        result = await manager.call_tool(
            "write_file",
            {
                "path": str(f),
                "content": "hello world",
            },
        )
        assert f.read_text() == "hello world"
        assert "Wrote" in result

    @pytest.mark.asyncio
    async def test_write_file_creates_parent_dirs(self, manager, tmp_path):
        f = tmp_path / "sub" / "deep" / "file.txt"
        await manager.call_tool(
            "write_file",
            {
                "path": str(f),
                "content": "nested",
            },
        )
        assert f.read_text() == "nested"


class TestEditFile:
    @pytest.mark.asyncio
    async def test_edit_file_single_match(self, manager, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def hello():\n    return 'world'\n")
        result = await manager.call_tool(
            "edit_file",
            {
                "path": str(f),
                "old_text": "return 'world'",
                "new_text": "return 'universe'",
            },
        )
        assert "universe" in f.read_text()
        assert "Applied" in result or "applied" in result.lower()

    @pytest.mark.asyncio
    async def test_edit_file_no_match(self, manager, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def hello():\n    return 'world'\n")
        result = await manager.call_tool(
            "edit_file",
            {
                "path": str(f),
                "old_text": "not found text",
                "new_text": "replacement",
            },
        )
        assert "not found" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_edit_file_multiple_matches(self, manager, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("foo\nfoo\n")
        result = await manager.call_tool(
            "edit_file",
            {
                "path": str(f),
                "old_text": "foo",
                "new_text": "bar",
            },
        )
        assert "multiple" in result.lower() or "2" in result

    @pytest.mark.asyncio
    async def test_edit_file_not_found(self, manager, tmp_path):
        result = await manager.call_tool(
            "edit_file",
            {
                "path": str(tmp_path / "nope.py"),
                "old_text": "x",
                "new_text": "y",
            },
        )
        assert "error" in result.lower()


class TestGlob:
    @pytest.mark.asyncio
    async def test_glob_finds_files(self, manager, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = await manager.call_tool(
            "glob",
            {
                "pattern": "*.py",
                "path": str(tmp_path),
            },
        )
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    @pytest.mark.asyncio
    async def test_glob_recursive(self, manager, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("")
        result = await manager.call_tool(
            "glob",
            {
                "pattern": "**/*.py",
                "path": str(tmp_path),
            },
        )
        assert "nested.py" in result

    @pytest.mark.asyncio
    async def test_glob_no_matches(self, manager, tmp_path):
        result = await manager.call_tool(
            "glob",
            {
                "pattern": "*.xyz",
                "path": str(tmp_path),
            },
        )
        assert "no matches" in result.lower() or result.strip() == ""


class TestGrep:
    @pytest.mark.asyncio
    async def test_grep_finds_matches(self, manager, tmp_path):
        (tmp_path / "file.py").write_text("def hello():\n    return 42\n")
        result = await manager.call_tool(
            "grep",
            {
                "pattern": "hello",
                "path": str(tmp_path),
            },
        )
        assert "hello" in result
        assert "file.py" in result

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, manager, tmp_path):
        (tmp_path / "file.py").write_text("def hello():\n    return 42\n")
        result = await manager.call_tool(
            "grep",
            {
                "pattern": "nonexistent",
                "path": str(tmp_path),
            },
        )
        assert "no matches" in result.lower() or result.strip() == ""


class TestBash:
    @pytest.mark.asyncio
    async def test_bash_stdout(self, manager):
        result = await manager.call_tool("bash", {"command": "echo hello"})
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_bash_stderr_and_exit_code(self, manager):
        result = await manager.call_tool(
            "bash",
            {
                "command": "echo err >&2; exit 1",
            },
        )
        assert "err" in result
        assert (
            "exit code: 1" in result.lower()
            or "exit_code: 1" in result.lower()
            or "exit code 1" in result.lower()
        )

    @pytest.mark.asyncio
    async def test_bash_timeout(self, manager):
        result = await manager.call_tool(
            "bash",
            {
                "command": "sleep 10",
                "timeout": 1,
            },
        )
        assert "timeout" in result.lower() or "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_bash_output_truncated(self, manager):
        # Generate output larger than _BASH_MAX_BYTES (100KB)
        # python3 writes all at once, avoiding pipe buffer issues
        result = await manager.call_tool(
            "bash",
            {
                "command": "python3 -c \"import sys; sys.stdout.write('x' * 200_000); sys.stdout.flush()\"",
                "timeout": 10,
            },
        )
        assert "truncated" in result.lower()
        assert len(result) <= 110_000


class TestPlanTools:
    @pytest.mark.asyncio
    async def test_create_and_read_plan(self, manager, config):
        await manager.call_tool("create_plan", {"content": "# My Plan\n- Step 1\n"})
        result = await manager.call_tool("read_plan", {})
        assert "My Plan" in result
        assert "Step 1" in result

    @pytest.mark.asyncio
    async def test_update_plan(self, manager, config):
        await manager.call_tool("create_plan", {"content": "# V1"})
        await manager.call_tool("update_plan", {"content": "# V2"})
        result = await manager.call_tool("read_plan", {})
        assert "V2" in result
        assert "V1" not in result

    @pytest.mark.asyncio
    async def test_read_plan_no_plan(self, manager):
        result = await manager.call_tool("read_plan", {})
        assert "no plan" in result.lower() or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_update_plan_no_plan(self, manager):
        result = await manager.call_tool("update_plan", {"content": "# V2"})
        assert (
            "no plan" in result.lower()
            or "not found" in result.lower()
            or "error" in result.lower()
        )


class TestBashSubprocessCleanup:
    """Issue #82: subprocess must be killed on cancellation, not leaked."""

    @pytest.mark.asyncio
    async def test_bash_kills_process_group_on_cancellation(self, manager):
        """The entire process group (start_new_session=True) must be killed."""
        import os

        # Use a unique marker so pgrep doesn't match other sleep commands
        marker = f"olmlx_test_{os.getpid()}"
        task = asyncio.create_task(
            manager.call_tool(
                "bash", {"command": f"sleep 60 # {marker}", "timeout": 30}
            )
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Poll for process cleanup instead of fixed sleep
        for _ in range(20):
            await asyncio.sleep(0.025)
            proc = await asyncio.create_subprocess_exec(
                "pgrep",
                "-f",
                marker,
                stdout=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode == 1:
                break
        else:
            pytest.fail(
                f"Orphan process with marker {marker} still running after 500ms"
            )


class TestWebFetchSchemeValidation:
    @pytest.mark.asyncio
    async def test_rejects_file_scheme(self, manager):
        result = await manager.call_tool("web_fetch", {"url": "file:///etc/passwd"})
        assert "unsupported" in result.lower() or "error" in result.lower()
        assert "http" in result.lower()

    @pytest.mark.asyncio
    async def test_rejects_ftp_scheme(self, manager):
        result = await manager.call_tool("web_fetch", {"url": "ftp://example.com/file"})
        assert "unsupported" in result.lower() or "error" in result.lower()


class TestHTMLExtractorNested:
    def test_nested_skip_tags(self):
        from olmlx.chat.builtin_tools import _strip_html

        html = "<noscript><style>css</style>hidden text</noscript>visible"
        result = _strip_html(html)
        assert "hidden text" not in result
        assert "visible" in result


class TestWebFetch:
    def _mock_opener(self, html_bytes):
        """Create a mock opener that returns the given HTML bytes."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = html_bytes
        mock_resp.headers.get_content_charset.return_value = "utf-8"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        return mock_opener

    @pytest.mark.asyncio
    async def test_web_fetch_strips_html(self, manager):
        html = "<html><body><h1>Title</h1><p>Content here</p></body></html>"
        with patch(
            "urllib.request.build_opener", return_value=self._mock_opener(html.encode())
        ):
            result = await manager.call_tool(
                "web_fetch", {"url": "https://example.com"}
            )
        assert "Title" in result
        assert "Content here" in result
        assert "<h1>" not in result

    @pytest.mark.asyncio
    async def test_web_fetch_truncates(self, manager):
        long_text = "x" * 20000
        html = f"<html><body>{long_text}</body></html>"
        with patch(
            "urllib.request.build_opener", return_value=self._mock_opener(html.encode())
        ):
            result = await manager.call_tool(
                "web_fetch", {"url": "https://example.com"}
            )
        assert len(result) <= 11000  # 10k + some overhead for truncation message


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_web_search_success(self, manager):
        mock_results = [
            {
                "title": "Result 1",
                "href": "https://example.com/1",
                "body": "First result",
            },
            {
                "title": "Result 2",
                "href": "https://example.com/2",
                "body": "Second result",
            },
        ]
        with patch("olmlx.chat.builtin_tools._web_search_impl") as mock_search:
            mock_search.return_value = mock_results
            result = await manager.call_tool("web_search", {"query": "test query"})
        assert "Result 1" in result
        assert "Result 2" in result

    @pytest.mark.asyncio
    async def test_web_search_missing_dependency(self, manager):
        with patch(
            "olmlx.chat.builtin_tools._web_search_impl",
            side_effect=ImportError("No module named 'duckduckgo_search'"),
        ):
            result = await manager.call_tool("web_search", {"query": "test"})
        assert "duckduckgo" in result.lower() or "install" in result.lower()


class TestSessionIntegration:
    """Test that built-in tools integrate correctly with ChatSession."""

    @pytest.mark.asyncio
    async def test_builtin_tools_in_tool_list(self, tmp_path):
        """Built-in tools should be included in tools passed to generate_chat."""
        from unittest.mock import MagicMock, patch
        from olmlx.chat.session import ChatSession

        config = ChatConfig(model_name="test:latest", plans_dir=tmp_path / "plans")
        mgr = MagicMock()
        builtin = BuiltinToolManager(config)
        session = ChatSession(config=config, manager=mgr, builtin=builtin)

        captured_kwargs = {}

        async def fake_stream(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield {"text": "Hello", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_stream(*a, **kw),
        ):
            async for _ in session.send_message("Hi"):
                pass

        tools = captured_kwargs.get("tools")
        assert tools is not None
        tool_names = {t["function"]["name"] for t in tools}
        assert "read_file" in tool_names
        assert "bash" in tool_names
        assert "web_search" in tool_names

    @pytest.mark.asyncio
    async def test_builtin_tool_routed_correctly(self, tmp_path):
        """Built-in tool calls should be routed to BuiltinToolManager, not MCP."""
        from unittest.mock import MagicMock, patch
        from olmlx.chat.session import ChatSession

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello from file\n")

        config = ChatConfig(model_name="test:latest", plans_dir=tmp_path / "plans")
        mgr = MagicMock()
        builtin = BuiltinToolManager(config)
        session = ChatSession(config=config, manager=mgr, builtin=builtin)
        call_count = 0

        async def fake_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield {
                    "text": f'{{"name": "read_file", "arguments": {{"path": "{test_file}"}}}}'.replace(
                        "{",
                        '<tool_call>{"name": "read_file", "arguments": {"path": "'
                        + str(test_file)
                        + '"}}</tool_call>',
                    )[:0]
                    + f'<tool_call>{{"name": "read_file", "arguments": {{"path": "{test_file}"}}}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": MagicMock()}
            else:
                yield {"text": "The file says hello", "done": False}
                yield {"text": "", "done": True, "stats": MagicMock()}

        with patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: fake_generate(),
        ):
            events = []
            async for event in session.send_message("Read the file"):
                events.append(event)

        result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(result_events) == 1
        assert "hello from file" in result_events[0]["result"]


class TestChatConfigFields:
    def test_builtin_tools_enabled_default(self):
        config = ChatConfig(model_name="test")
        assert config.builtin_tools_enabled is True

    def test_plans_dir_default(self):
        config = ChatConfig(model_name="test")
        assert config.plans_dir == Path.home() / ".olmlx" / "plans"

    def test_plans_dir_custom(self, tmp_path):
        config = ChatConfig(model_name="test", plans_dir=tmp_path / "my_plans")
        assert config.plans_dir == tmp_path / "my_plans"
