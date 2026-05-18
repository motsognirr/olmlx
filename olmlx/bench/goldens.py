"""Golden output storage for regression-vs-baseline quality checks.

Goldens are stored at ``<bench_dir>/goldens/<safe_model>/<prompt_name>.txt``
where ``safe_model`` uses the same sanitizer as the on-disk model store
so filesystem naming stays consistent across the project.

Typical flow:
1. Capture: ``olmlx bench capture-goldens --model M`` runs the baseline
   scenario and writes each prompt's ``output_text`` as a golden.
2. Compare: subsequent ``olmlx bench run`` invocations auto-assign the
   ``regression_snapshot`` grader to prompts that don't carry their own
   grader, reading the matching golden for diff-based scoring.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

GOLDENS_SUBDIR = "goldens"

# Sanitize model names and prompt names for filesystem use. Intentionally
# duplicates the rule used by ``olmlx.models.store`` (replace anything
# outside ``[A-Za-z0-9_.-]`` with ``_``) rather than importing a private
# helper across module boundaries — golden file naming is a local concern
# that should not break if the model-store sanitizer ever evolves.
_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_.-]")


def _sanitize(name: str) -> str:
    s = _SANITIZE_RE.sub("_", name)
    # `.` and `..` are filesystem-special — `bench_dir / "goldens" / ".."`
    # resolves to `bench_dir`, letting a model identifier of ".." escape
    # the goldens tree. Map both to a safe placeholder. An empty input
    # (or one of only-disallowed chars that all became "_") still ends up
    # as a single "_" via the final guard.
    if s in (".", ".."):
        s = s.replace(".", "_")
    return s or "_"


def goldens_dir(bench_dir: Path, model: str) -> Path:
    """Directory where goldens for ``model`` live under ``bench_dir``."""
    return bench_dir / GOLDENS_SUBDIR / _sanitize(model)


def golden_path(bench_dir: Path, model: str, prompt_name: str) -> Path:
    """Absolute path to a single prompt's golden file."""
    return goldens_dir(bench_dir, model) / f"{_sanitize(prompt_name)}.txt"


def load_golden(bench_dir: Path, model: str, prompt_name: str) -> str | None:
    """Return the stored golden output, or ``None`` if none exists.

    Returns ``None`` for both "no golden captured" and "golden exists but
    unreadable" so callers can use a single missing-data check, but logs
    a warning on the latter — otherwise a permission or I/O error would
    silently look identical to "needs capturing" and the next capture
    run would overwrite a golden that is actually intact on disk but
    just temporarily inaccessible.
    """
    path = golden_path(bench_dir, model, prompt_name)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("golden exists but unreadable: %s (%s)", path, exc)
        return None


def save_golden(
    bench_dir: Path,
    model: str,
    prompt_name: str,
    output_text: str,
    *,
    force: bool = False,
) -> Path:
    """Write a golden for ``prompt_name``. Refuses to overwrite unless ``force``.

    Returns the path written.
    """
    path = golden_path(bench_dir, model, prompt_name)
    if path.exists() and not force:
        raise FileExistsError(
            f"Golden already exists at {path}; pass force=True to overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output_text, encoding="utf-8")
    return path
