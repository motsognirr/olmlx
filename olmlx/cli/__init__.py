"""CLI for olmlx with serve, service, models, and config subcommands.

Split into per-subcommand modules; this package namespace re-exports every
name the old single-module ``olmlx/cli.py`` exposed. Command handlers are
resolved via ``globals()`` in ``_resolve_handler`` so test monkeypatching
(``monkeypatch.setattr("olmlx.cli.cmd_serve", mock)``) keeps working.

Rebindable module state (``_cli_distributed_*``, ``_atexit_registered``,
``_signal_handlers_installed``) is deliberately NOT re-exported here — a
static re-import would go stale when the owning module rebinds it; import
those from ``olmlx.cli.serve`` / ``olmlx.cli.distributed_launch``.
"""

import argparse
import sys
from collections.abc import Callable
from typing import Any

from olmlx.cli.config_cmd import (  # noqa: F401
    DEFAULT_MODELS,
    ensure_config,
    _configure_logging,
)
from olmlx.cli.serve import (  # noqa: F401
    cmd_serve,
    _DEPRECATED_SPECULATIVE_ENV_VARS,
    _LEGACY_SPECULATIVE_FORWARD,
    _legacy_speculative_values_in_dotenv,
    _forward_legacy_speculative_env,
    _DEPRECATED_DFLASH_ENV_VARS,
    _surface_legacy_dflash_env,
    _surface_legacy_speculative_env,
    _legacy_kv_cache_quant_in_dotenv,
    _surface_legacy_kv_cache_quant_env,
    _warn_kv_cache_quant_incompatibilities,
    _DISTRIBUTED_LEGACY_ENV_MAP,
    _surface_legacy_distributed_env,
    _apply_serve_overrides,
    _audit_per_model_flash_in_distributed,
    _models_with_promoted_keys_in_experimental,
    _load_registry_for_audit,
    _audit_speculative_config,
    cmd_config_show,
)
from olmlx.cli.distributed_launch import (  # noqa: F401
    _VALID_HOSTNAME_RE,
    _worker_procs,
    _worker_log_fhs,
    _install_signal_handlers,
    _cleanup_workers,
    _pre_shard_and_distribute,
    validate_remote_python,
    _launch_distributed_workers,
    _find_executable,
)
from olmlx.cli.service import (  # noqa: F401
    PLIST_LABEL,
    PLIST_PATH,
    _SECRET_ENV_SUFFIXES,
    _SECRET_ENV_SUBSTRINGS,
    _is_secret_env_key,
    _build_plist,
    cmd_service_install,
    cmd_service_uninstall,
    cmd_service_status,
)
from olmlx.cli.models_cmd import (  # noqa: F401
    _create_store,
    _resolve_and_download,
    _format_size,
    cmd_models_list,
    cmd_models_search,
    cmd_models_show,
    cmd_models_pull,
    cmd_models_delete,
)
from olmlx.cli.chat_cmd import (  # noqa: F401
    _VOICE_FLAGS,
    _build_chat_arg_parser_voice_defaults,
    _check_voice_deps,
    cmd_chat,
)
from olmlx.cli.bench_cmd import (  # noqa: F401
    cmd_bench_run,
    cmd_bench_compare,
    cmd_bench_list,
    _positive_int,
    _non_empty_str,
    cmd_bench_leaderboard,
)
from olmlx.cli.prepare_cmd import (  # noqa: F401
    _flash_progress,
    cmd_spectral_prepare,
    cmd_shard_prepare,
    cmd_flash_prepare,
    _cmd_flash_moe_prepare,
    _cmd_flash_dense_prepare,
    cmd_dflash_precompute,
    cmd_dflash_prepare,
    cmd_eagle_prepare,
    cmd_flash_info,
    _show_flash_moe_info,
    _show_flash_dense_info,
)
from olmlx.cli.reap_cmd import (  # noqa: F401
    cmd_reap_apply,
    cmd_reap_calibrate,
    cmd_reap_plan,
    cmd_reap_report,
)
from olmlx.cli.parser import (  # noqa: F401
    build_parser,
)

# Registry: (command, subcommand) → handler name.
# Handler names are resolved via globals() at call time so test
# monkeypatching (``monkeypatch.setattr("olmlx.cli.cmd_serve", mock)``)
# works.  _validate_command_handlers() catches typos at import time.
# Subcommand=None for commands that take no subcommand (serve, chat).
#
# IMPORTANT: every cmd_* handler referenced below must be defined
# above this point in the file (import-time globals() resolution).
_COMMAND_HANDLERS: dict[tuple[str, str | None], str] = {
    ("serve", None): "cmd_serve",
    ("chat", None): "cmd_chat",
    ("service", "install"): "cmd_service_install",
    ("service", "uninstall"): "cmd_service_uninstall",
    ("service", "status"): "cmd_service_status",
    ("models", "list"): "cmd_models_list",
    ("models", "pull"): "cmd_models_pull",
    ("models", "show"): "cmd_models_show",
    ("models", "delete"): "cmd_models_delete",
    ("models", "search"): "cmd_models_search",
    ("flash", "prepare"): "cmd_flash_prepare",
    ("flash", "info"): "cmd_flash_info",
    ("dflash", "prepare"): "cmd_dflash_prepare",
    ("dflash", "precompute"): "cmd_dflash_precompute",
    ("eagle", "prepare"): "cmd_eagle_prepare",
    ("spectral", "prepare"): "cmd_spectral_prepare",
    ("shard", "prepare"): "cmd_shard_prepare",
    ("bench", "run"): "cmd_bench_run",
    ("bench", "compare"): "cmd_bench_compare",
    ("bench", "list"): "cmd_bench_list",
    ("bench", "leaderboard"): "cmd_bench_leaderboard",
    ("reap", "calibrate"): "cmd_reap_calibrate",
    ("reap", "plan"): "cmd_reap_plan",
    ("reap", "apply"): "cmd_reap_apply",
    ("reap", "report"): "cmd_reap_report",
    ("config", "show"): "cmd_config_show",
}


def _validate_command_handlers() -> None:
    """Verify every handler name in the registry refers to a callable.

    Called at module load to catch typos before runtime dispatch.
    """
    for (cmd, sub), name in _COMMAND_HANDLERS.items():
        handler = globals().get(name)
        if handler is None:
            raise NameError(
                f"Handler {name!r} (registered for ({cmd!r}, {sub!r})) "
                f"not found in module globals"
            )
        if not callable(handler):
            raise TypeError(f"Handler {name!r} for ({cmd!r}, {sub!r}) is not callable")


_validate_command_handlers()


def _resolve_handler(cmd: str, sub_name: str | None) -> Callable[..., Any] | None:
    """Look up a handler by (command, subcommand) in the registry.

    Resolves via ``globals()`` so that test monkeypatching
    (``monkeypatch.setattr("olmlx.cli.cmd_serve", mock_fn)``) works.
    """
    name = _COMMAND_HANDLERS.get((cmd, sub_name))
    if name is None:
        return None
    handler = globals().get(name)
    if handler is None:
        raise NameError(
            f"Handler {name!r} (registered for ({cmd!r}, {sub_name!r})) "
            f"not found in module globals"
        )
    return handler


def cli_main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        # Bare invocation: derive serve-subparser defaults from the
        # parser itself rather than hardcoding the flag list. New serve
        # flags wire through automatically; any top-level flags already
        # on ``args`` win because ``hasattr`` short-circuits the copy.
        # Invariant: top-level parser flag names must not overlap with
        # serve-only flag names — otherwise this loop would suppress the
        # serve default for the colliding name. The root parser only
        # declares the ``command`` dest today, so this holds.
        serve_defaults = vars(parser.parse_args(["serve"]))
        for _name, _default in serve_defaults.items():
            if not hasattr(args, _name):
                setattr(args, _name, _default)

    cmd = args.command or "serve"  # default
    sub_name = getattr(args, f"{cmd}_command", None)

    handler = _resolve_handler(cmd, sub_name)
    if handler:
        # ``ModelsConfigError`` surfaces from ``registry.load()`` /
        # ``_save_mappings_locked()`` when ``models.json`` is unreadable
        # or corrupt — refusing to clobber the file. Convert to a clean
        # exit with the error text instead of dumping a traceback.
        from olmlx.engine.registry import ModelsConfigError

        try:
            return handler(args)
        except ModelsConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    # Unknown or missing subcommand. Print the parent command's help to
    # stderr (not stdout) and exit non-zero so scripts / CI don't read the
    # no-op as success — ``parser.parse_args([cmd, "--help"])`` would print
    # to stdout and exit 0 (#635).
    subparser = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparser = action.choices.get(cmd)
            break
    (subparser or parser).print_help(sys.stderr)
    sys.exit(2)
