"""launchd service install/uninstall/status subcommands."""

import logging
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


from olmlx.cli.config_cmd import ensure_config
from olmlx.cli.distributed_launch import _find_executable

logger = logging.getLogger(__name__)

PLIST_LABEL = "com.dpalmqvist.olmlx"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_LABEL}.plist"


# Heuristics for spotting credential-bearing env vars so they are never
# persisted to the cleartext plist. Suffix matching (not substring) for the
# generic markers so the LLM-token config keys this codebase is full of
# (OLMLX_SPECULATIVE_TOKENS, OLMLX_PROMPT_CACHE_MAX_TOKENS, …) are NOT mistaken
# for credentials — only an actual trailing _TOKEN/_KEY/_PASSWORD counts.
_SECRET_ENV_SUFFIXES = (
    "_SECRET",
    "_TOKEN",
    "_KEY",
    "_APIKEY",
    "_PASSWORD",
    "_PASSWD",
    "_CREDENTIAL",
    "_CREDENTIALS",
)
# Substrings that are unambiguous credentials regardless of position.
_SECRET_ENV_SUBSTRINGS = ("SECRET", "PASSWORD", "CREDENTIAL")


def _is_secret_env_key(key: str) -> bool:
    """True if an env var name looks like it carries a credential."""
    upper = key.upper()
    return upper.endswith(_SECRET_ENV_SUFFIXES) or any(
        marker in upper for marker in _SECRET_ENV_SUBSTRINGS
    )


def _build_plist() -> dict:
    """Build a launchd plist dict for the olmlx service."""
    exe = _find_executable()
    if exe == sys.executable:
        program_args = [exe, "-m", "olmlx"]
    else:
        program_args = [exe]

    env_vars = {}
    # Forward OLMLX_ env vars if set, but never persist secrets into the
    # cleartext launchd plist — ~/Library/LaunchAgents/com.olmlx.plist is
    # readable by any local process and is not a safe credential store (#454).
    for key, value in os.environ.items():
        if key.startswith("OLMLX_") and not _is_secret_env_key(key):
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
