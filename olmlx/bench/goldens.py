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

from pathlib import Path

from olmlx.models.store import _safe_dir_name

GOLDENS_SUBDIR = "goldens"


def goldens_dir(bench_dir: Path, model: str) -> Path:
    """Directory where goldens for ``model`` live under ``bench_dir``."""
    return bench_dir / GOLDENS_SUBDIR / _safe_dir_name(model)


def golden_path(bench_dir: Path, model: str, prompt_name: str) -> Path:
    """Absolute path to a single prompt's golden file."""
    return goldens_dir(bench_dir, model) / f"{_safe_dir_name(prompt_name)}.txt"


def load_golden(bench_dir: Path, model: str, prompt_name: str) -> str | None:
    """Return the stored golden output, or ``None`` if none exists."""
    path = golden_path(bench_dir, model, prompt_name)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
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
