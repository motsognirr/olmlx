from __future__ import annotations

import pytest

import olmlx.cli as cli
from olmlx.cli.parser import build_parser


class TestParser:
    def test_calibrate_defaults(self):
        args = build_parser().parse_args(["reap", "calibrate", "some-model"])
        assert args.command == "reap" and args.reap_command == "calibrate"
        assert args.model == "some-model"
        assert args.sources == "english,code,chinese"
        assert args.samples_per_source == 256 and args.max_tokens == 512
        assert args.output is None

    def test_plan_modes(self):
        args = build_parser().parse_args(
            [
                "reap",
                "plan",
                "m",
                "--mode",
                "graded",
                "--keep-fraction",
                "0.75",
                "--high-fraction",
                "0.25",
            ]
        )
        assert args.mode == "graded" and args.keep_fraction == 0.75
        assert args.high_bits == 8 and args.low_bits == 4

    def test_apply_requires_plan(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["reap", "apply", "m"])

    def test_report_flags(self):
        args = build_parser().parse_args(
            ["reap", "report", "m", "--skip-ppl", "--keep", "64"]
        )
        assert args.skip_ppl is True and args.keep == 64


class TestDispatch:
    def test_handlers_registered(self):
        for sub in ("calibrate", "plan", "apply", "report"):
            assert ("reap", sub) in cli._COMMAND_HANDLERS
            name = cli._COMMAND_HANDLERS[("reap", sub)]
            assert callable(getattr(cli, name))

    def test_cli_main_routes(self, monkeypatch):
        called = {}
        monkeypatch.setattr(
            "olmlx.cli.cmd_reap_plan", lambda args: called.setdefault("args", args)
        )
        monkeypatch.setattr("sys.argv", ["olmlx", "reap", "plan", "m", "--keep", "8"])
        cli.cli_main()
        assert called["args"].model == "m"


class TestPlanHandlerValidation:
    def test_uniform_requires_keep(self, monkeypatch):
        import olmlx.cli.reap_cmd as reap_cmd

        args = build_parser().parse_args(["reap", "plan", "m", "--mode", "uniform"])
        # validation must fire BEFORE any resolve/download
        monkeypatch.setattr(
            reap_cmd,
            "_resolve_and_download",
            lambda *a, **k: pytest.fail("resolved before validation"),
        )
        with pytest.raises(SystemExit):
            reap_cmd.cmd_reap_plan(args)
