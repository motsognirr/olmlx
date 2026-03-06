"""Tests for mlx_ollama.cli."""

import json
import plistlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlx_ollama.cli import (
    DEFAULT_MODELS,
    PLIST_LABEL,
    _build_plist,
    build_parser,
    cli_main,
    cmd_serve,
    cmd_service_install,
    cmd_service_status,
    cmd_service_uninstall,
    ensure_config,
)


class TestEnsureConfig:
    def test_creates_dir_and_models_json(self, tmp_path, monkeypatch):
        config_path = tmp_path / "subdir" / "models.json"
        monkeypatch.setattr("mlx_ollama.cli.settings.models_config", config_path)
        ensure_config()
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data == DEFAULT_MODELS

    def test_does_not_overwrite_existing(self, tmp_path, monkeypatch):
        config_path = tmp_path / "models.json"
        existing = {"custom:latest": "some/model"}
        config_path.write_text(json.dumps(existing))
        monkeypatch.setattr("mlx_ollama.cli.settings.models_config", config_path)
        ensure_config()
        data = json.loads(config_path.read_text())
        assert data == existing


class TestBuildPlist:
    def test_plist_structure(self, monkeypatch):
        monkeypatch.setattr(
            "mlx_ollama.cli.shutil.which", lambda _: "/usr/local/bin/mlx-ollama"
        )
        plist = _build_plist()
        assert plist["Label"] == PLIST_LABEL
        assert plist["ProgramArguments"] == ["/usr/local/bin/mlx-ollama"]
        assert plist["RunAtLoad"] is True
        assert plist["KeepAlive"] is True
        assert "StandardOutPath" in plist
        assert "StandardErrorPath" in plist

    def test_plist_fallback_to_python(self, monkeypatch):
        monkeypatch.setattr("mlx_ollama.cli.shutil.which", lambda _: None)
        monkeypatch.setattr("mlx_ollama.cli.sys.executable", "/usr/bin/python3")
        plist = _build_plist()
        assert plist["ProgramArguments"] == ["/usr/bin/python3", "-m", "mlx_ollama"]

    def test_plist_forwards_env_vars(self, monkeypatch):
        monkeypatch.setattr(
            "mlx_ollama.cli.shutil.which", lambda _: "/usr/local/bin/mlx-ollama"
        )
        monkeypatch.setenv("MLX_OLLAMA_PORT", "9999")
        plist = _build_plist()
        assert plist["EnvironmentVariables"]["MLX_OLLAMA_PORT"] == "9999"


class TestServiceInstall:
    def test_install_creates_plist_and_loads(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "com.dpalmqvist.mlx-ollama.plist"
        monkeypatch.setattr("mlx_ollama.cli.PLIST_PATH", plist_path)
        monkeypatch.setattr(
            "mlx_ollama.cli.settings.models_config", tmp_path / "models.json"
        )
        monkeypatch.setattr(
            "mlx_ollama.cli.shutil.which", lambda _: "/usr/local/bin/mlx-ollama"
        )
        mock_run = MagicMock()
        monkeypatch.setattr("mlx_ollama.cli.subprocess.run", mock_run)
        cmd_service_install(None)
        assert plist_path.exists()
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
        assert plist["Label"] == PLIST_LABEL
        mock_run.assert_called_once_with(
            ["launchctl", "load", str(plist_path)],
            check=True,
        )


class TestServiceUninstall:
    def test_uninstall_removes_plist(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "com.dpalmqvist.mlx-ollama.plist"
        plist_path.write_text("dummy")
        monkeypatch.setattr("mlx_ollama.cli.PLIST_PATH", plist_path)
        mock_run = MagicMock()
        monkeypatch.setattr("mlx_ollama.cli.subprocess.run", mock_run)
        cmd_service_uninstall(None)
        assert not plist_path.exists()
        mock_run.assert_called_once_with(
            ["launchctl", "unload", str(plist_path)],
            check=False,
        )

    def test_uninstall_no_plist(self, tmp_path, monkeypatch, capsys):
        plist_path = tmp_path / "com.dpalmqvist.mlx-ollama.plist"
        monkeypatch.setattr("mlx_ollama.cli.PLIST_PATH", plist_path)
        mock_run = MagicMock()
        monkeypatch.setattr("mlx_ollama.cli.subprocess.run", mock_run)
        cmd_service_uninstall(None)
        mock_run.assert_not_called()
        assert "No plist found" in capsys.readouterr().out


class TestServiceStatus:
    def test_status_loaded(self, monkeypatch, capsys):
        mock_result = MagicMock(
            returncode=0, stdout="PID\tStatus\tLabel\n123\t0\tcom.dpalmqvist.mlx-ollama"
        )
        monkeypatch.setattr(
            "mlx_ollama.cli.subprocess.run", lambda *a, **kw: mock_result
        )
        cmd_service_status(None)
        out = capsys.readouterr().out
        assert "is loaded" in out

    def test_status_not_loaded(self, monkeypatch, capsys):
        mock_result = MagicMock(returncode=1, stdout="")
        monkeypatch.setattr(
            "mlx_ollama.cli.subprocess.run", lambda *a, **kw: mock_result
        )
        cmd_service_status(None)
        out = capsys.readouterr().out
        assert "is not loaded" in out


class TestBuildParser:
    def test_default_command(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_serve_command(self):
        parser = build_parser()
        args = parser.parse_args(["serve"])
        assert args.command == "serve"

    def test_service_install(self):
        parser = build_parser()
        args = parser.parse_args(["service", "install"])
        assert args.command == "service"
        assert args.service_command == "install"

    def test_service_uninstall(self):
        parser = build_parser()
        args = parser.parse_args(["service", "uninstall"])
        assert args.command == "service"
        assert args.service_command == "uninstall"

    def test_service_status(self):
        parser = build_parser()
        args = parser.parse_args(["service", "status"])
        assert args.command == "service"
        assert args.service_command == "status"


class TestCliMain:
    def test_default_calls_serve(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["mlx-ollama"])
        mock_serve = MagicMock()
        monkeypatch.setattr("mlx_ollama.cli.cmd_serve", mock_serve)
        cli_main()
        mock_serve.assert_called_once()

    def test_serve_calls_serve(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["mlx-ollama", "serve"])
        mock_serve = MagicMock()
        monkeypatch.setattr("mlx_ollama.cli.cmd_serve", mock_serve)
        cli_main()
        mock_serve.assert_called_once()

    def test_service_install_calls_install(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["mlx-ollama", "service", "install"])
        mock_install = MagicMock()
        monkeypatch.setattr("mlx_ollama.cli.cmd_service_install", mock_install)
        cli_main()
        mock_install.assert_called_once()
