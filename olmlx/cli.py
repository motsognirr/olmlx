"""CLI for olmlx with serve, service, models, and config subcommands."""

import argparse
import asyncio
import json
import logging
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from olmlx.config import settings

PLIST_LABEL = "com.dpalmqvist.olmlx"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_LABEL}.plist"

DEFAULT_MODELS = {
    "llama3.2:latest": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistral:7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "qwen2.5:3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "gemma2:2b": "mlx-community/gemma-2-2b-it-4bit",
}


def ensure_config():
    """Create ~/.olmlx/ and seed models.json if missing."""
    config_dir = settings.models_config.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    if not settings.models_config.exists():
        with open(settings.models_config, "w") as f:
            json.dump(DEFAULT_MODELS, f, indent=2)
        print(f"Created {settings.models_config} with example models")


def cmd_serve(_args):
    """Start the olmlx server."""
    import uvicorn

    ensure_config()
    _configure_logging()
    uvicorn.run(
        "olmlx.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


def _find_executable() -> str:
    """Find the olmlx executable path."""
    exe = shutil.which("olmlx")
    if exe:
        return exe
    # Fallback: use the current Python interpreter with -m
    return sys.executable


def _build_plist() -> dict:
    """Build a launchd plist dict for the olmlx service."""
    exe = _find_executable()
    if exe == sys.executable:
        program_args = [exe, "-m", "olmlx"]
    else:
        program_args = [exe]

    env_vars = {}
    # Forward OLMLX_ env vars if set
    for key, value in os.environ.items():
        if key.startswith("OLMLX_"):
            env_vars[key] = value
    # Ensure PATH includes common tool locations
    env_vars["PATH"] = os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin")

    plist = {
        "Label": PLIST_LABEL,
        "ProgramArguments": program_args,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(Path.home() / ".olmlx" / "olmlx.log"),
        "StandardErrorPath": str(Path.home() / ".olmlx" / "olmlx.log"),
        "EnvironmentVariables": env_vars,
    }
    return plist


def cmd_service_install(_args):
    """Install and load the launchd service."""
    ensure_config()
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    plist = _build_plist()
    with open(PLIST_PATH, "wb") as f:
        plistlib.dump(plist, f)
    print(f"Wrote {PLIST_PATH}")

    try:
        subprocess.run(
            ["launchctl", "load", str(PLIST_PATH)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else "(no output)"
        print(
            f"Plist was written to {PLIST_PATH} but the service could not be loaded.\n"
            f"launchctl stderr: {stderr}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Service {PLIST_LABEL} loaded")


def cmd_service_uninstall(_args):
    """Unload and remove the launchd service."""
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], check=False)
        PLIST_PATH.unlink()
        print(f"Service {PLIST_LABEL} unloaded and plist removed")
    else:
        print(f"No plist found at {PLIST_PATH}")


def cmd_service_status(_args):
    """Show the status of the launchd service."""
    result = subprocess.run(
        ["launchctl", "list", PLIST_LABEL],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"Service {PLIST_LABEL} is loaded")
        print(result.stdout.strip())
    else:
        print(f"Service {PLIST_LABEL} is not loaded")


def _create_store():
    """Create a ModelStore instance for CLI use.

    Raises on failure — callers are responsible for catching and exiting.
    """
    from olmlx.engine.registry import ModelRegistry
    from olmlx.models.store import ModelStore

    ensure_config()
    registry = ModelRegistry()
    registry.load()
    return ModelStore(registry)


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.1f} GB"
    elif size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f} MB"
    elif size_bytes >= 1e3:
        return f"{size_bytes / 1e3:.1f} KB"
    return f"{size_bytes} B"


def cmd_models_list(_args):
    """List locally downloaded models."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        models = store.list_local()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if not models:
        print("No models downloaded.")
        return
    # Header
    print(f"{'NAME':<30} {'SIZE':<12} {'PARAMS':<10} {'QUANT':<10} {'HF PATH'}")
    print("-" * 90)
    for m in sorted(models, key=lambda x: x.name or ""):
        name = (m.name or "")[:30]
        params = (m.parameter_size or "")[:10]
        quant = (m.quantization_level or "")[:10]
        print(
            f"{name:<30} {_format_size(m.size):<12} "
            f"{params:<10} {quant:<10} {m.hf_path}"
        )


def cmd_models_show(args):
    """Show details for a specific model."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        manifest = store.show(args.model_name)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if manifest is None:
        print(f"Model '{args.model_name}' not found locally.", file=sys.stderr)
        sys.exit(1)
    print(f"Name:           {manifest.name}")
    print(f"HF Path:        {manifest.hf_path}")
    print(f"Size:           {_format_size(manifest.size)}")
    print(f"Format:         {manifest.format}")
    print(f"Family:         {manifest.family}")
    print(f"Parameters:     {manifest.parameter_size}")
    print(f"Quantization:   {manifest.quantization_level}")
    print(f"Modified:       {manifest.modified_at}")
    print(f"Digest:         {manifest.digest}")


def cmd_models_pull(args):
    """Pull/download a model."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    async def _pull():
        async for status in store.pull(args.model_name):
            if msg := status.get("status", ""):
                print(msg, flush=True)

    try:
        asyncio.run(_pull())
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


def cmd_models_delete(args):
    """Delete a locally downloaded model."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if not args.yes:
        try:
            confirm = input(f"Delete model '{args.model_name}'? [y/N] ")
        except EOFError:
            print("Aborted.")
            return
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return
    try:
        deleted = store.delete(args.model_name)
    except Exception as e:
        print(f"Error deleting model '{args.model_name}': {e}", file=sys.stderr)
        sys.exit(1)
    if deleted:
        print(f"Model '{args.model_name}' deleted.")
    else:
        print(f"Model '{args.model_name}' not found locally.", file=sys.stderr)
        sys.exit(1)


def _configure_logging():
    """Configure logging from settings."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def cmd_chat(args):
    """Start an interactive chat session."""
    from olmlx.chat.config import ChatConfig, load_mcp_config, load_tool_safety_config
    from olmlx.chat.mcp_client import MCPClientManager
    from olmlx.chat.session import ChatSession
    from olmlx.chat.tool_safety import ToolSafetyPolicy
    from olmlx.chat.tui import ChatTUI
    from olmlx.engine.model_manager import ModelManager

    ensure_config()
    _configure_logging()

    model_name = args.model_name
    if model_name is None:
        print("Error: model name required. Usage: olmlx chat <model>", file=sys.stderr)
        sys.exit(1)

    chat_kwargs = dict(
        model_name=model_name,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        thinking=not args.no_thinking,
        mcp_enabled=not args.no_mcp,
        repeat_penalty=args.repeat_penalty,
        repeat_last_n=args.repeat_last_n,
        skills_enabled=not args.no_skills,
        builtin_tools_enabled=not args.no_builtin_tools,
    )
    if args.mcp_config:
        chat_kwargs["mcp_config_path"] = Path(args.mcp_config)
    if args.skills_dir:
        chat_kwargs["skills_dir"] = Path(args.skills_dir)
    config = ChatConfig(**chat_kwargs)

    async def _run_chat():
        from olmlx.chat.builtin_tools import BuiltinToolManager
        from olmlx.chat.skills import SkillManager

        store = _create_store()
        manager = ModelManager(store.registry, store)

        tui = ChatTUI()
        mcp = None
        skills = None
        builtin = None

        try:
            tui.console.print(f"[dim]Loading {model_name}...[/dim]")
            await manager.ensure_loaded(model_name, keep_alive="-1")

            if config.mcp_enabled:
                mcp_cfg = load_mcp_config(config.mcp_config_path)
                if mcp_cfg:
                    mcp = MCPClientManager()
                    await mcp.connect_all(mcp_cfg)

            if config.skills_enabled:
                skills = SkillManager(config.skills_dir)
                skills.load()
                if skills.list_skills():
                    tui.console.print(
                        f"[dim]Loaded {len(skills.list_skills())} skill(s)[/dim]"
                    )

            if config.builtin_tools_enabled:
                builtin = BuiltinToolManager(config)

            # Load tool safety policy
            safety_config = load_tool_safety_config(config.mcp_config_path)
            active_stream_ctx = None

            async def confirm_decider(name: str, args: dict) -> bool:
                nonlocal active_stream_ctx
                if active_stream_ctx and active_stream_ctx.is_active:
                    active_stream_ctx.finish()
                return await asyncio.to_thread(tui.confirm_tool_call, name, args)

            policy = ToolSafetyPolicy(safety_config, decider=confirm_decider)

            session = ChatSession(
                config=config,
                manager=manager,
                mcp=mcp,
                skills=skills,
                builtin=builtin,
                tool_safety=policy,
            )
            tools = mcp.get_tools_for_chat() if mcp else []
            if builtin:
                tools = tools + builtin.get_tool_definitions()
            tui.display_welcome(model_name, tools)

            while True:
                user_input = tui.get_user_input()
                if user_input is None:
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    cmd_parts = user_input.split(None, 1)
                    command = cmd_parts[0].lower()
                    arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                    if command in ("/exit", "/quit"):
                        break
                    elif command == "/clear":
                        session.clear_history()
                        tui.console.print("[dim]History cleared[/dim]")
                    elif command == "/tools":
                        tui.display_tools(tools)
                    elif command == "/skills":
                        if skills and skills.list_skills():
                            tui.console.print("[bold]Skills:[/bold]")
                            for s in skills.list_skills():
                                desc = f" — {s.description}" if s.description else ""
                                tui.console.print(f"  [cyan]{s.name}[/cyan]{desc}")
                        else:
                            tui.console.print("[dim]No skills loaded[/dim]")
                    elif command == "/safety":
                        tui.display_safety_policy(policy)
                    elif command == "/system":
                        if arg:
                            config.system_prompt = arg
                            session.clear_history()
                            tui.console.print(
                                "[dim]System prompt set. History cleared.[/dim]"
                            )
                        else:
                            current = config.system_prompt or "(none)"
                            tui.console.print(f"[dim]System prompt: {current}[/dim]")
                    elif command == "/model":
                        model_parts = arg.split(None, 1)
                        if model_parts and model_parts[0] == "thinking":
                            # /model thinking [on|off]
                            if len(model_parts) == 2 and model_parts[1] in (
                                "on",
                                "off",
                            ):
                                config.thinking = model_parts[1] == "on"
                                tui.console.print(
                                    f"[dim]Thinking: {'on' if config.thinking else 'off'}[/dim]"
                                )
                            else:
                                thinking_str = "on" if config.thinking else "off"
                                tui.console.print(
                                    f"[dim]Thinking: {thinking_str}. Use: /model thinking on|off[/dim]"
                                )
                        elif arg:
                            tui.console.print(f"[dim]Loading {arg}...[/dim]")
                            try:
                                await manager.ensure_loaded(arg, keep_alive="-1")
                                config.model_name = arg
                                session.clear_history()
                                tui.console.print(
                                    f"[dim]Switched to {arg}. History cleared.[/dim]"
                                )
                            except Exception as exc:
                                tui.display_error(str(exc))
                        else:
                            thinking_str = "on" if config.thinking else "off"
                            tui.console.print(
                                f"[dim]Current model: {config.model_name} | thinking: {thinking_str}[/dim]"
                            )
                    else:
                        tui.display_error(f"Unknown command: {command}")
                    continue

                # Collect events while streaming tokens
                pending_events = []
                confirmed_tool_ids: set[str] = set()
                stream_ctx = tui.stream_response()
                active_stream_ctx = stream_ctx
                try:
                    with stream_ctx:
                        async for event in session.send_message(user_input):
                            if event["type"] == "thinking_start":
                                stream_ctx.start_thinking()
                            elif event["type"] == "thinking_end":
                                stream_ctx.end_thinking()
                            elif event["type"] == "thinking_token":
                                stream_ctx.update(event["text"])
                            elif event["type"] == "token":
                                stream_ctx.update(event["text"])
                            elif event["type"] == "tool_approved":
                                # Track confirmed IDs to avoid duplicate display
                                confirmed_tool_ids.add(event["id"])
                            else:
                                pending_events.append(event)
                finally:
                    active_stream_ctx = None

                # Display collected events
                for event in pending_events:
                    if event["type"] == "tool_call":
                        # Skip display for confirmed tools — already shown
                        # by confirm_tool_call during the prompt
                        if event["id"] not in confirmed_tool_ids:
                            tui.display_tool_call(event["name"], event["arguments"])
                    elif event["type"] == "tool_result":
                        tui.display_tool_result(event["name"], event["result"])
                    elif event["type"] == "tool_error":
                        tui.display_tool_error(event["name"], event["error"])
                    elif event["type"] == "tool_denied":
                        # Only show panel for policy-denied tools; user-denied
                        # tools were already shown at the confirm prompt
                        if event.get("reason") != "user":
                            tui.display_tool_denied(event["name"])
                    elif event["type"] == "tool_confirmation_needed":
                        pass  # handled inline by decider callback
                    elif event["type"] == "max_turns_exceeded":
                        tui.display_error("Max tool turns reached")

        except MemoryError as exc:
            tui.display_error(str(exc))
            sys.exit(1)
        except ValueError as exc:
            tui.display_error(str(exc))
            sys.exit(1)
        finally:
            if mcp is not None:
                await mcp.disconnect_all()
            await manager.stop()

    try:
        asyncio.run(_run_chat())
    except KeyboardInterrupt:
        print("\nBye!", file=sys.stderr)


def cmd_config_show(_args):
    """Show current configuration."""
    print(f"Host:                   {settings.host}")
    print(f"Port:                   {settings.port}")
    print(f"Models dir:             {settings.models_dir}")
    print(f"Models config:          {settings.models_config}")
    print(f"Default keep-alive:     {settings.default_keep_alive}")
    print(f"Max loaded models:      {settings.max_loaded_models}")
    print(f"Memory limit fraction:  {settings.memory_limit_fraction}")
    print(f"Log level:              {settings.log_level}")
    print(f"Prompt cache:           {settings.prompt_cache}")
    print(f"Prompt cache max tokens: {settings.prompt_cache_max_tokens}")
    print(f"CORS origins:           {settings.cors_origins}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="olmlx",
        description="Ollama-compatible API server using Apple MLX",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="Start the server (default)")

    svc = sub.add_parser("service", help="Manage the launchd service")
    svc_sub = svc.add_subparsers(dest="service_command")
    svc_sub.add_parser("install", help="Install and start the launchd service")
    svc_sub.add_parser("uninstall", help="Stop and remove the launchd service")
    svc_sub.add_parser("status", help="Show service status")

    mdl = sub.add_parser("models", help="Manage local models")
    mdl_sub = mdl.add_subparsers(dest="models_command")
    mdl_sub.add_parser("list", help="List locally downloaded models")
    pull_p = mdl_sub.add_parser("pull", help="Pull/download a model")
    pull_p.add_argument("model_name", help="Model name or HF path")
    show_p = mdl_sub.add_parser("show", help="Show model details")
    show_p.add_argument("model_name", help="Model name or HF path")
    del_p = mdl_sub.add_parser("delete", help="Delete a local model")
    del_p.add_argument("model_name", help="Model name or HF path")
    del_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    chat_p = sub.add_parser("chat", help="Interactive chat")
    chat_p.add_argument("model_name", nargs="?", help="Model name or HF path")
    chat_p.add_argument("--system", "-s", help="System prompt")
    chat_p.add_argument(
        "--mcp-config", help="MCP config path (default: ~/.olmlx/mcp.json)"
    )
    chat_p.add_argument(
        "--no-mcp", action="store_true", default=False, help="Disable MCP"
    )
    chat_p.add_argument(
        "--no-thinking", action="store_true", default=False, help="Disable thinking"
    )
    chat_p.add_argument("--max-tokens", type=int, default=4096)
    chat_p.add_argument("--max-turns", type=int, default=25)
    chat_p.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (1.0 = disabled, default: 1.1)",
    )
    chat_p.add_argument(
        "--repeat-last-n",
        type=int,
        default=64,
        help="Context window for repetition penalty (default: 64)",
    )
    chat_p.add_argument(
        "--no-skills", action="store_true", default=False, help="Disable skills"
    )
    chat_p.add_argument(
        "--no-builtin-tools",
        action="store_true",
        default=False,
        help="Disable built-in tools",
    )
    chat_p.add_argument(
        "--skills-dir", help="Skills directory (default: ~/.olmlx/skills)"
    )

    cfg = sub.add_parser("config", help="Show configuration")
    cfg_sub = cfg.add_subparsers(dest="config_command")
    cfg_sub.add_parser("show", help="Show current configuration")

    return parser


def cli_main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "serve":
        cmd_serve(args)
    elif args.command == "service":
        if args.service_command == "install":
            cmd_service_install(args)
        elif args.service_command == "uninstall":
            cmd_service_uninstall(args)
        elif args.service_command == "status":
            cmd_service_status(args)
        else:
            parser.parse_args(["service", "--help"])
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "models":
        if args.models_command == "list":
            cmd_models_list(args)
        elif args.models_command == "pull":
            cmd_models_pull(args)
        elif args.models_command == "show":
            cmd_models_show(args)
        elif args.models_command == "delete":
            cmd_models_delete(args)
        else:
            parser.parse_args(["models", "--help"])
    elif args.command == "config":
        if args.config_command == "show":
            cmd_config_show(args)
        else:
            parser.parse_args(["config", "--help"])
