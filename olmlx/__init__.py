__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Defer heavy mlx_lm top-level imports.
#
# mlx_lm/__init__.py imports generate.py which imports transformers.
# transformers v5 scans ~2000 .py files at import time, taking minutes.
# By pre-seeding sys.modules with a lightweight stub, Python skips
# mlx_lm/__init__.py when we import submodules like mlx_lm.models.cache.
# The real mlx_lm package is loaded on demand via ensure_mlx_lm().
# ---------------------------------------------------------------------------
import importlib
import importlib.util
import sys
import types

if "mlx_lm" not in sys.modules:
    _mlx_lm_stub = types.ModuleType("mlx_lm")
    _spec = importlib.util.find_spec("mlx_lm")
    if _spec is not None and _spec.submodule_search_locations:
        _mlx_lm_stub.__path__ = list(_spec.submodule_search_locations)
        _mlx_lm_stub.__package__ = "mlx_lm"
        _mlx_lm_stub.__spec__ = _spec
        _mlx_lm_stub.__mlx_stub__ = True  # marker so we can detect it
        sys.modules["mlx_lm"] = _mlx_lm_stub
    del _spec


def ensure_mlx_lm():
    """Force-load the real mlx_lm package (triggers transformers import).

    Call this before using mlx_lm.load(), mlx_lm.generate(), etc.
    Safe to call multiple times — only the first call does work.
    """
    mod = sys.modules.get("mlx_lm")
    if mod is not None and getattr(mod, "__mlx_stub__", False):
        # Remove stub so importlib loads the real package
        del sys.modules["mlx_lm"]
        importlib.import_module("mlx_lm")
