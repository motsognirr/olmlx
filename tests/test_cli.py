"""Tests for olmlx.cli."""

import json
import plistlib
from unittest.mock import MagicMock

import pytest

from olmlx.cli import (
    DEFAULT_MODELS,
    PLIST_LABEL,
    _build_plist,
    _create_store,
    build_parser,
    cli_main,
    cmd_config_show,
    cmd_models_delete,
    cmd_models_list,
    cmd_models_pull,
    cmd_models_show,
    cmd_service_install,
    cmd_service_status,
    cmd_service_uninstall,
    ensure_config,
)
from olmlx.models.manifest import ModelManifest


class TestEnsureConfig:
    def test_creates_dir_and_models_json(self, tmp_path, monkeypatch):
        config_path = tmp_path / "subdir" / "models.json"
        monkeypatch.setattr("olmlx.cli.settings.models_config", config_path)
        ensure_config()
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data == DEFAULT_MODELS

    def test_does_not_overwrite_existing(self, tmp_path, monkeypatch):
        config_path = tmp_path / "models.json"
        existing = {"custom:latest": "some/model"}
        config_path.write_text(json.dumps(existing))
        monkeypatch.setattr("olmlx.cli.settings.models_config", config_path)
        ensure_config()
        data = json.loads(config_path.read_text())
        assert data == existing


class TestBuildPlist:
    def test_plist_structure(self, monkeypatch):
        monkeypatch.setattr("olmlx.cli.shutil.which", lambda _: "/usr/local/bin/olmlx")
        plist = _build_plist()
        assert plist["Label"] == PLIST_LABEL
        assert plist["ProgramArguments"] == ["/usr/local/bin/olmlx"]
        assert plist["RunAtLoad"] is True
        assert plist["KeepAlive"] is True
        assert "StandardOutPath" in plist
        assert "StandardErrorPath" in plist

    def test_plist_fallback_to_python(self, monkeypatch):
        monkeypatch.setattr("olmlx.cli.shutil.which", lambda _: None)
        monkeypatch.setattr("olmlx.cli.sys.executable", "/usr/bin/python3")
        plist = _build_plist()
        assert plist["ProgramArguments"] == ["/usr/bin/python3", "-m", "olmlx"]

    def test_plist_forwards_env_vars(self, monkeypatch):
        monkeypatch.setattr("olmlx.cli.shutil.which", lambda _: "/usr/local/bin/olmlx")
        monkeypatch.setenv("OLMLX_PORT", "9999")
        plist = _build_plist()
        assert plist["EnvironmentVariables"]["OLMLX_PORT"] == "9999"


class TestServiceInstall:
    def test_install_creates_plist_and_loads(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "com.dpalmqvist.olmlx.plist"
        monkeypatch.setattr("olmlx.cli.PLIST_PATH", plist_path)
        monkeypatch.setattr(
            "olmlx.cli.settings.models_config", tmp_path / "models.json"
        )
        monkeypatch.setattr("olmlx.cli.shutil.which", lambda _: "/usr/local/bin/olmlx")
        mock_run = MagicMock()
        monkeypatch.setattr("olmlx.cli.subprocess.run", mock_run)
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
        plist_path = tmp_path / "com.dpalmqvist.olmlx.plist"
        plist_path.write_text("dummy")
        monkeypatch.setattr("olmlx.cli.PLIST_PATH", plist_path)
        mock_run = MagicMock()
        monkeypatch.setattr("olmlx.cli.subprocess.run", mock_run)
        cmd_service_uninstall(None)
        assert not plist_path.exists()
        mock_run.assert_called_once_with(
            ["launchctl", "unload", str(plist_path)],
            check=False,
        )

    def test_uninstall_no_plist(self, tmp_path, monkeypatch, capsys):
        plist_path = tmp_path / "com.dpalmqvist.olmlx.plist"
        monkeypatch.setattr("olmlx.cli.PLIST_PATH", plist_path)
        mock_run = MagicMock()
        monkeypatch.setattr("olmlx.cli.subprocess.run", mock_run)
        cmd_service_uninstall(None)
        mock_run.assert_not_called()
        assert "No plist found" in capsys.readouterr().out


class TestServiceStatus:
    def test_status_loaded(self, monkeypatch, capsys):
        mock_result = MagicMock(
            returncode=0, stdout="PID\tStatus\tLabel\n123\t0\tcom.dpalmqvist.olmlx"
        )
        monkeypatch.setattr("olmlx.cli.subprocess.run", lambda *a, **kw: mock_result)
        cmd_service_status(None)
        out = capsys.readouterr().out
        assert "is loaded" in out

    def test_status_not_loaded(self, monkeypatch, capsys):
        mock_result = MagicMock(returncode=1, stdout="")
        monkeypatch.setattr("olmlx.cli.subprocess.run", lambda *a, **kw: mock_result)
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
        monkeypatch.setattr("sys.argv", ["olmlx"])
        mock_serve = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_serve", mock_serve)
        cli_main()
        mock_serve.assert_called_once()

    def test_serve_calls_serve(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "serve"])
        mock_serve = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_serve", mock_serve)
        cli_main()
        mock_serve.assert_called_once()

    def test_service_install_calls_install(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "service", "install"])
        mock_install = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_service_install", mock_install)
        cli_main()
        mock_install.assert_called_once()

    def test_models_list_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "models", "list"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_models_list", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_models_pull_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "models", "pull", "qwen2.5:3b"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_models_pull", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_models_show_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "models", "show", "qwen2.5:3b"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_models_show", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_models_delete_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "models", "delete", "qwen2.5:3b"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_models_delete", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_config_show_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "config", "show"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_config_show", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_models_no_subcommand_shows_help(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "models"])
        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        assert exc_info.value.code == 0

    def test_config_no_subcommand_shows_help(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "config"])
        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        assert exc_info.value.code == 0


class TestCreateStore:
    def test_calls_ensure_config(self, tmp_path, monkeypatch):
        """_create_store must call ensure_config so fresh installs get models.json."""
        config_path = tmp_path / "models.json"
        monkeypatch.setattr("olmlx.cli.settings.models_config", config_path)
        monkeypatch.setattr("olmlx.cli.settings.models_dir", tmp_path / "models")
        _create_store()
        # ensure_config should have created models.json
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data == DEFAULT_MODELS

    def test_malformed_config_raises(self, tmp_path, monkeypatch):
        """Malformed models.json should raise, not sys.exit."""
        config_path = tmp_path / "models.json"
        config_path.write_text("{bad json")
        monkeypatch.setattr("olmlx.cli.settings.models_config", config_path)
        with pytest.raises(Exception):
            _create_store()

    def test_ensure_config_permission_error_raises(self, tmp_path, monkeypatch):
        """Permission error in ensure_config should propagate."""
        monkeypatch.setattr(
            "olmlx.cli.ensure_config",
            lambda: (_ for _ in ()).throw(PermissionError("Permission denied")),
        )
        with pytest.raises(PermissionError):
            _create_store()


@pytest.fixture
def mock_store():
    """Create a mock ModelStore for CLI tests."""
    store = MagicMock()
    return store


@pytest.fixture
def _patch_store(monkeypatch, mock_store):
    """Patch _create_store to return mock_store."""
    monkeypatch.setattr("olmlx.cli._create_store", lambda: mock_store)


class TestCreateStoreErrorHandlingInCommands:
    """Commands should catch _create_store errors and exit cleanly."""

    def test_list_handles_store_error(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "olmlx.cli._create_store",
            lambda: (_ for _ in ()).throw(Exception("bad config")),
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_list(None)
        assert exc_info.value.code == 1
        assert "bad config" in capsys.readouterr().err

    def test_show_handles_store_error(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "olmlx.cli._create_store",
            lambda: (_ for _ in ()).throw(Exception("bad config")),
        )
        args = MagicMock(model_name="test")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_show(args)
        assert exc_info.value.code == 1
        assert "bad config" in capsys.readouterr().err

    def test_pull_handles_store_error(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "olmlx.cli._create_store",
            lambda: (_ for _ in ()).throw(Exception("bad config")),
        )
        args = MagicMock(model_name="test")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_pull(args)
        assert exc_info.value.code == 1
        assert "bad config" in capsys.readouterr().err

    def test_delete_handles_store_error(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "olmlx.cli._create_store",
            lambda: (_ for _ in ()).throw(Exception("bad config")),
        )
        args = MagicMock(model_name="test", yes=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_delete(args)
        assert exc_info.value.code == 1
        assert "bad config" in capsys.readouterr().err


class TestModelsListCmd:
    def test_lists_models(self, capsys, mock_store, _patch_store):
        mock_store.list_local.return_value = [
            ModelManifest(
                name="qwen2.5:3b",
                hf_path="mlx-community/Qwen2.5-3B-Instruct-4bit",
                size=2_000_000_000,
                parameter_size="3B",
                quantization_level="4-bit",
            ),
            ModelManifest(
                name="llama3.2:latest",
                hf_path="mlx-community/Llama-3.2-3B-Instruct-4bit",
                size=1_500_000_000,
            ),
        ]
        cmd_models_list(None)
        out = capsys.readouterr().out
        assert "qwen2.5:3b" in out
        assert "llama3.2:latest" in out
        assert "2.0 GB" in out
        assert "1.5 GB" in out

    def test_truncates_long_names(self, capsys, mock_store, _patch_store):
        long_name = "a" * 40
        mock_store.list_local.return_value = [
            ModelManifest(
                name=long_name,
                hf_path="some/model",
                size=1_000_000,
                parameter_size="x" * 15,
                quantization_level="y" * 15,
            ),
        ]
        cmd_models_list(None)
        out = capsys.readouterr().out
        lines = out.strip().split("\n")
        # The data line (after header + separator) should not contain the full 40-char name
        data_line = lines[2]
        assert long_name not in data_line
        assert "x" * 15 not in data_line
        assert "y" * 15 not in data_line

    def test_exact_column_width_not_truncated(self, capsys, mock_store, _patch_store):
        """A name exactly at column width (30) should not be truncated."""
        name_30 = "a" * 30
        mock_store.list_local.return_value = [
            ModelManifest(
                name=name_30,
                hf_path="some/model",
                size=1_000_000,
                parameter_size="1234567890",
                quantization_level="1234567890",
            ),
        ]
        cmd_models_list(None)
        out = capsys.readouterr().out
        data_line = out.strip().split("\n")[2]
        assert name_30 in data_line
        assert "1234567890" in data_line

    def test_handles_none_string_fields(self, capsys, mock_store, _patch_store):
        """Manifests with None for string fields should not crash."""
        m = ModelManifest(
            name="test:latest",
            hf_path="some/model",
            size=1_000_000,
        )
        m.parameter_size = None
        m.quantization_level = None
        mock_store.list_local.return_value = [m]
        cmd_models_list(None)
        out = capsys.readouterr().out
        assert "test:latest" in out

    def test_handles_none_name(self, capsys, mock_store, _patch_store):
        """Manifests with None name should not crash the sort."""
        m = ModelManifest(name="valid:latest", hf_path="some/model", size=100)
        m2 = ModelManifest(name="other:latest", hf_path="some/model2", size=200)
        m2.name = None
        mock_store.list_local.return_value = [m, m2]
        cmd_models_list(None)
        out = capsys.readouterr().out
        assert "valid:latest" in out

    def test_lists_no_models(self, capsys, mock_store, _patch_store):
        mock_store.list_local.return_value = []
        cmd_models_list(None)
        out = capsys.readouterr().out
        assert "No models" in out

    def test_handles_corrupt_manifest(self, capsys, mock_store, _patch_store):
        """Corrupt manifest in list_local should produce clean error, not traceback."""
        mock_store.list_local.side_effect = ValueError("Field 'size' should be int")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_list(None)
        assert exc_info.value.code == 1
        assert "size" in capsys.readouterr().err


class TestModelsShowCmd:
    def test_show_model(self, capsys, mock_store, _patch_store):
        mock_store.show.return_value = ModelManifest(
            name="qwen2.5:3b",
            hf_path="mlx-community/Qwen2.5-3B-Instruct-4bit",
            size=2_000_000_000,
            family="qwen2",
            parameter_size="3B",
            quantization_level="4-bit",
            format="mlx",
        )
        args = MagicMock(model_name="qwen2.5:3b")
        cmd_models_show(args)
        out = capsys.readouterr().out
        assert "qwen2.5:3b" in out
        assert "mlx-community/Qwen2.5-3B-Instruct-4bit" in out
        assert "qwen2" in out

    def test_show_handles_corrupt_manifest(self, capsys, mock_store, _patch_store):
        """Corrupt manifest should produce clean error, not traceback."""
        mock_store.show.side_effect = ValueError("Field 'size' should be int")
        args = MagicMock(model_name="qwen2.5:3b")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_show(args)
        assert exc_info.value.code == 1
        assert "size" in capsys.readouterr().err

    def test_show_not_found_exits_nonzero(self, capsys, mock_store, _patch_store):
        mock_store.show.return_value = None
        args = MagicMock(model_name="nonexistent")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_show(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()
        assert "not found" not in captured.out.lower()


class TestModelsPullCmd:
    def test_pull_model(self, capsys, mock_store, _patch_store):
        async def fake_pull(name):
            yield {"status": "pulling manifest"}
            yield {"status": "downloading mlx-community/Qwen2.5-3B-Instruct-4bit"}
            yield {"status": "success"}

        mock_store.pull = fake_pull
        args = MagicMock(model_name="qwen2.5:3b")
        cmd_models_pull(args)
        out = capsys.readouterr().out
        assert "pulling manifest" in out
        assert "success" in out

    def test_pull_model_not_found_exits_nonzero(self, capsys, mock_store, _patch_store):
        async def fake_pull(name):
            if False:
                yield {}
            raise ValueError(f"Model '{name}' not found in config")

        mock_store.pull = fake_pull
        args = MagicMock(model_name="nonexistent")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_pull(args)
        assert exc_info.value.code == 1
        assert "not found" in capsys.readouterr().err.lower()

    def test_pull_handles_os_error_exits_nonzero(
        self, capsys, mock_store, _patch_store
    ):
        async def fake_pull(name):
            yield {"status": "pulling manifest"}
            raise OSError("Disk full")

        mock_store.pull = fake_pull
        args = MagicMock(model_name="qwen2.5:3b")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_pull(args)
        assert exc_info.value.code == 1
        assert "Disk full" in capsys.readouterr().err

    def test_pull_handles_generic_exception_exits_nonzero(
        self, capsys, mock_store, _patch_store
    ):
        async def fake_pull(name):
            if False:
                yield {}
            raise RuntimeError("Something unexpected")

        mock_store.pull = fake_pull
        args = MagicMock(model_name="qwen2.5:3b")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_pull(args)
        assert exc_info.value.code == 1
        assert "Something unexpected" in capsys.readouterr().err

    def test_pull_error_goes_to_stderr(self, mock_store, _patch_store, capsys):
        """Error output should go to stderr, not stdout."""

        async def fake_pull(name):
            if False:
                yield {}
            raise ValueError("Model not found")

        mock_store.pull = fake_pull
        args = MagicMock(model_name="nonexistent")
        with pytest.raises(SystemExit):
            cmd_models_pull(args)
        captured = capsys.readouterr()
        assert "Model not found" in captured.err
        assert "Model not found" not in captured.out

    def test_pull_keyboard_interrupt_exits_130(self, capsys, mock_store, _patch_store):
        """Ctrl+C during pull should exit with code 130, not crash."""

        async def fake_pull(name):
            yield {"status": "pulling manifest"}
            raise KeyboardInterrupt

        mock_store.pull = fake_pull
        args = MagicMock(model_name="qwen2.5:3b")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_pull(args)
        assert exc_info.value.code == 130

    def test_pull_flushes_output(self, mock_store, _patch_store, monkeypatch):
        """Status lines should be flushed immediately for piped output."""
        flush_calls = []
        original_print = print

        def tracking_print(*args, **kwargs):
            if kwargs.get("flush"):
                flush_calls.append(True)
            original_print(*args, **kwargs)

        monkeypatch.setattr("builtins.print", tracking_print)

        async def fake_pull(name):
            yield {"status": "pulling manifest"}
            yield {"status": "success"}

        mock_store.pull = fake_pull
        args = MagicMock(model_name="qwen2.5:3b")
        cmd_models_pull(args)
        assert len(flush_calls) == 2


class TestModelsDeleteCmd:
    def test_delete_model_with_yes_flag(self, capsys, mock_store, _patch_store):
        mock_store.delete.return_value = True
        args = MagicMock(model_name="qwen2.5:3b", yes=True)
        cmd_models_delete(args)
        out = capsys.readouterr().out
        assert "deleted" in out.lower()

    def test_delete_not_found_exits_nonzero(self, capsys, mock_store, _patch_store):
        mock_store.delete.return_value = False
        args = MagicMock(model_name="nonexistent", yes=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_delete(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()
        assert "not found" not in captured.out.lower()

    def test_delete_prompts_for_confirmation(
        self, capsys, monkeypatch, mock_store, _patch_store
    ):
        mock_store.delete.return_value = True
        monkeypatch.setattr("builtins.input", lambda _: "y")
        args = MagicMock(model_name="qwen2.5:3b", yes=False)
        cmd_models_delete(args)
        out = capsys.readouterr().out
        assert "deleted" in out.lower()

    def test_delete_aborts_on_no(self, capsys, monkeypatch, mock_store, _patch_store):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        args = MagicMock(model_name="qwen2.5:3b", yes=False)
        cmd_models_delete(args)
        out = capsys.readouterr().out
        assert "aborted" in out.lower()
        mock_store.delete.assert_not_called()

    def test_delete_aborts_on_eof(self, capsys, monkeypatch, mock_store, _patch_store):
        """Closed stdin (piped/CI) should abort cleanly, not crash."""

        def raise_eof(_):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        args = MagicMock(model_name="qwen2.5:3b", yes=False)
        cmd_models_delete(args)
        out = capsys.readouterr().out
        assert "aborted" in out.lower()
        mock_store.delete.assert_not_called()

    def test_delete_handles_os_error(self, capsys, mock_store, _patch_store):
        """OSError from shutil.rmtree should produce clean error, not traceback."""
        mock_store.delete.side_effect = OSError("Permission denied")
        args = MagicMock(model_name="qwen2.5:3b", yes=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_delete(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "permission denied" in captured.err.lower()

    def test_delete_handles_non_os_error(self, capsys, mock_store, _patch_store):
        """Non-OSError from store.delete should also produce clean error."""
        mock_store.delete.side_effect = ValueError("Invalid model dir")
        args = MagicMock(model_name="qwen2.5:3b", yes=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_delete(args)
        assert exc_info.value.code == 1
        assert "Invalid model dir" in capsys.readouterr().err


class TestConfigShowCmd:
    def test_shows_config(self, capsys):
        cmd_config_show(None)
        out = capsys.readouterr().out
        assert "host" in out.lower()
        assert "port" in out.lower()
        assert "models dir" in out.lower()


class TestBuildParserModels:
    def test_models_list(self):
        parser = build_parser()
        args = parser.parse_args(["models", "list"])
        assert args.command == "models"
        assert args.models_command == "list"

    def test_models_pull(self):
        parser = build_parser()
        args = parser.parse_args(["models", "pull", "qwen2.5:3b"])
        assert args.command == "models"
        assert args.models_command == "pull"
        assert args.model_name == "qwen2.5:3b"

    def test_models_show(self):
        parser = build_parser()
        args = parser.parse_args(["models", "show", "qwen2.5:3b"])
        assert args.command == "models"
        assert args.models_command == "show"
        assert args.model_name == "qwen2.5:3b"

    def test_models_delete(self):
        parser = build_parser()
        args = parser.parse_args(["models", "delete", "qwen2.5:3b"])
        assert args.command == "models"
        assert args.models_command == "delete"
        assert args.model_name == "qwen2.5:3b"
        assert args.yes is False

    def test_models_delete_yes_flag(self):
        parser = build_parser()
        args = parser.parse_args(["models", "delete", "--yes", "qwen2.5:3b"])
        assert args.yes is True

    def test_models_delete_y_flag(self):
        parser = build_parser()
        args = parser.parse_args(["models", "delete", "-y", "qwen2.5:3b"])
        assert args.yes is True

    def test_config_show(self):
        parser = build_parser()
        args = parser.parse_args(["config", "show"])
        assert args.command == "config"
        assert args.config_command == "show"
