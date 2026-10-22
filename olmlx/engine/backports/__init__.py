"""mlx-lm architectures backported to the pinned mlx-lm release line.

olmlx pins mlx-lm below the next minor (#468), so architectures that land
upstream after the pinned release are vendored here and registered under
their upstream module name (``mlx_lm.models.<model_type>``). That keeps
mlx-lm's own ``_get_classes`` lookup and olmlx's ``_detect_model_kind``
``find_spec`` probe working unchanged. A module mlx-lm already ships always
wins, so a pin bump retires the backport without code changes.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys

logger = logging.getLogger(__name__)

# upstream model_type -> vendored module
_BACKPORTS = {
    "flint": "olmlx.engine.backports.flint",
}

_installed = False


def install_mlx_lm_backports() -> None:
    """Register vendored architectures that the installed mlx-lm lacks. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True
    for model_type, module_name in _BACKPORTS.items():
        target = f"mlx_lm.models.{model_type}"
        if target in sys.modules or importlib.util.find_spec(target) is not None:
            continue
        sys.modules[target] = importlib.import_module(module_name)
        logger.debug("Registered backported mlx-lm architecture %r", model_type)
