"""CLI for mlx-ollama with serve and service subcommands."""

import argparse
import json
import logging
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from mlx_ollama.config import settings

PLIST_LABEL = "com.dpalmqvist.mlx-ollama"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_LABEL}.plist"

DEFAULT_MODELS = {
    "llama3.2:latest": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistral:7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "qwen2.5:3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "gemma2:2b": "mlx-community/gemma-2-2b-it-4bit",
}


def ensure_config():
    """Create ~/.mlx_ollama/ and seed models.json if missing."""
    config_dir = settings.models_config.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    if not settings.models_config.exists():
        with open(settings.models_config, "w") as f:
            json.dump(DEFAULT_MODELS, f, indent=2)
        print(f"Created {settings.models_config} with example models")


def cmd_serve(_args):
    """Start the mlx-ollama server."""
    import uvicorn

    ensure_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(
        "mlx_ollama.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


def _find_executable() -> str:
    """Find the mlx-ollama executable path."""
    exe = shutil.which("mlx-ollama")
    if exe:
        return exe
    # Fallback: use the current Python interpreter with -m
    return sys.executable


def _build_plist() -> dict:
    """Build a launchd plist dict for the mlx-ollama service."""
    exe = _find_executable()
    if exe == sys.executable:
        program_args = [exe, "-m", "mlx_ollama"]
    else:
        program_args = [exe]

    env_vars = {}
    # Forward MLX_OLLAMA_ env vars if set
    for key, value in os.environ.items():
        if key.startswith("MLX_OLLAMA_"):
            env_vars[key] = value
    # Ensure PATH includes common tool locations
    env_vars["PATH"] = os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin")

    plist = {
        "Label": PLIST_LABEL,
        "ProgramArguments": program_args,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(Path.home() / ".mlx_ollama" / "mlx-ollama.log"),
        "StandardErrorPath": str(Path.home() / ".mlx_ollama" / "mlx-ollama.log"),
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

    subprocess.run(["launchctl", "load", str(PLIST_PATH)], check=True)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-ollama",
        description="Ollama-compatible API server using Apple MLX",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="Start the server (default)")

    svc = sub.add_parser("service", help="Manage the launchd service")
    svc_sub = svc.add_subparsers(dest="service_command")
    svc_sub.add_parser("install", help="Install and start the launchd service")
    svc_sub.add_parser("uninstall", help="Stop and remove the launchd service")
    svc_sub.add_parser("status", help="Show service status")

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
