"""Tests for olmlx.chat.config."""

import json

from olmlx.chat.config import ChatConfig, load_mcp_config


class TestChatConfig:
    def test_defaults(self):
        cfg = ChatConfig(model_name="qwen3:8b")
        assert cfg.model_name == "qwen3:8b"
        assert cfg.system_prompt is None
        assert cfg.max_tokens == 4096
        assert cfg.max_turns == 25
        assert cfg.thinking is True
        assert cfg.mcp_enabled is True
        assert cfg.mcp_config_path.name == "mcp.json"

    def test_custom_values(self, tmp_path):
        cfg = ChatConfig(
            model_name="llama3:8b",
            system_prompt="You are helpful.",
            max_tokens=2048,
            max_turns=10,
            thinking=False,
            mcp_enabled=False,
            mcp_config_path=tmp_path / "custom.json",
        )
        assert cfg.model_name == "llama3:8b"
        assert cfg.system_prompt == "You are helpful."
        assert cfg.max_tokens == 2048
        assert cfg.max_turns == 10
        assert cfg.thinking is False
        assert cfg.mcp_enabled is False
        assert cfg.mcp_config_path == tmp_path / "custom.json"


class TestLoadMcpConfig:
    def test_loads_stdio_server(self, tmp_path):
        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                }
            }
        }
        config_path = tmp_path / "mcp.json"
        config_path.write_text(json.dumps(config))
        result = load_mcp_config(config_path)
        assert "filesystem" in result
        assert result["filesystem"]["command"] == "npx"
        assert result["filesystem"]["transport"] == "stdio"

    def test_loads_sse_server(self, tmp_path):
        config = {"mcpServers": {"remote": {"url": "http://localhost:8080/sse"}}}
        config_path = tmp_path / "mcp.json"
        config_path.write_text(json.dumps(config))
        result = load_mcp_config(config_path)
        assert "remote" in result
        assert result["remote"]["url"] == "http://localhost:8080/sse"
        assert result["remote"]["transport"] == "sse"

    def test_loads_mixed_servers(self, tmp_path):
        config = {
            "mcpServers": {
                "fs": {"command": "node", "args": ["server.js"]},
                "api": {"url": "http://localhost:9000/sse"},
            }
        }
        config_path = tmp_path / "mcp.json"
        config_path.write_text(json.dumps(config))
        result = load_mcp_config(config_path)
        assert result["fs"]["transport"] == "stdio"
        assert result["api"]["transport"] == "sse"

    def test_missing_file_returns_empty(self, tmp_path):
        result = load_mcp_config(tmp_path / "nonexistent.json")
        assert result == {}

    def test_malformed_json_returns_empty(self, tmp_path):
        config_path = tmp_path / "mcp.json"
        config_path.write_text("{bad json")
        result = load_mcp_config(config_path)
        assert result == {}

    def test_no_mcpServers_key_returns_empty(self, tmp_path):
        config_path = tmp_path / "mcp.json"
        config_path.write_text(json.dumps({"other": "data"}))
        result = load_mcp_config(config_path)
        assert result == {}

    def test_skips_entries_without_command_or_url(self, tmp_path):
        config = {
            "mcpServers": {
                "valid": {"command": "node", "args": []},
                "invalid": {"description": "no command or url"},
            }
        }
        config_path = tmp_path / "mcp.json"
        config_path.write_text(json.dumps(config))
        result = load_mcp_config(config_path)
        assert "valid" in result
        assert "invalid" not in result

    def test_env_passed_through(self, tmp_path):
        config = {
            "mcpServers": {
                "server": {
                    "command": "node",
                    "args": ["server.js"],
                    "env": {"API_KEY": "secret"},
                }
            }
        }
        config_path = tmp_path / "mcp.json"
        config_path.write_text(json.dumps(config))
        result = load_mcp_config(config_path)
        assert result["server"]["env"] == {"API_KEY": "secret"}
