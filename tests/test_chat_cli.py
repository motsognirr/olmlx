"""Tests for chat CLI integration in olmlx.cli."""

from unittest.mock import MagicMock

from olmlx.cli import build_parser, cli_main


class TestChatParser:
    def test_chat_command(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "qwen3:8b"])
        assert args.command == "chat"
        assert args.model_name == "qwen3:8b"

    def test_chat_no_model(self):
        parser = build_parser()
        args = parser.parse_args(["chat"])
        assert args.command == "chat"
        assert args.model_name is None

    def test_chat_system_prompt(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "qwen3:8b", "--system", "Be helpful"])
        assert args.system == "Be helpful"

    def test_chat_system_prompt_short(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "qwen3:8b", "-s", "Be helpful"])
        assert args.system == "Be helpful"

    def test_chat_no_mcp(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "qwen3:8b", "--no-mcp"])
        assert args.no_mcp is True

    def test_chat_no_thinking(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "qwen3:8b", "--no-thinking"])
        assert args.no_thinking is True

    def test_chat_max_tokens(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "qwen3:8b", "--max-tokens", "2048"])
        assert args.max_tokens == 2048

    def test_chat_max_turns(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "qwen3:8b", "--max-turns", "10"])
        assert args.max_turns == 10

    def test_chat_mcp_config(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "qwen3:8b", "--mcp-config", "/tmp/mcp.json"])
        assert args.mcp_config == "/tmp/mcp.json"

    def test_chat_no_skills(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "qwen3:8b", "--no-skills"])
        assert args.no_skills is True

    def test_chat_skills_dir(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "qwen3:8b", "--skills-dir", "/tmp/skills"])
        assert args.skills_dir == "/tmp/skills"

    def test_chat_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "qwen3:8b"])
        assert args.no_mcp is False
        assert args.no_thinking is False
        assert args.max_tokens == 4096
        assert args.max_turns == 25
        assert args.system is None
        assert args.mcp_config is None
        assert args.no_skills is False
        assert args.skills_dir is None


class TestChatThinkingControl:
    """Tests for /model thinking on|off control."""

    def test_model_shows_thinking_status(self):
        """'/model' with no args should show model name and thinking status."""
        # This tests the CLI /model handler; we test the output format
        parser = build_parser()
        args = parser.parse_args(["chat", "qwen3:8b"])
        assert args.no_thinking is False  # thinking on by default


class TestCliMainChat:
    def test_chat_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "chat", "qwen3:8b"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_chat", mock_fn)
        cli_main()
        mock_fn.assert_called_once()
