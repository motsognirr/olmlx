"""AWQ / GPTQ → MLX auto-conversion utilities.

Imported by ``olmlx.models.store`` during ``pull()``; no heavy dependencies
at module level so the server starts without mlx_lm being fully configured.
``mlx_lm.convert`` is imported lazily inside ``convert_to_mlx`` so test
suites that mock ``mlx_lm`` at a higher level can still import this module
freely.
"""

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def converting_marker(dst: Path) -> Path:
    """Sibling marker flagging an in-progress or failed conversion to *dst*.

    Kept OUTSIDE *dst* deliberately: ``mlx_lm.convert`` refuses to write to an
    ``mlx_path`` that already exists, so a marker placed *inside* *dst* would
    make the directory exist and abort every conversion.
    """
    return dst.parent / (dst.name + ".converting")


def detect_format(model_dir: Path) -> str | None:
    """Detect whether *model_dir* contains AWQ or GPTQ weights.

    Returns ``"awq"``, ``"gptq"``, or ``None`` for MLX-native / plain FP16.

    Detection strategy (in priority order):
    1. ``config.json`` → ``quantization_config`` field. The canonical HF
       discriminator is ``quant_method`` (written by AutoAWQ / auto_gptq /
       transformers); ``quant_type`` is accepted as a fallback alias.
    2. Presence of ``quantize_config.json`` — the auto_gptq canonical GPTQ
       marker file.

    MLX-native models carry a top-level ``quantization`` key in
    ``config.json`` (not ``quantization_config``) and are returned as
    ``None``.
    """
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            quant_config = cfg.get("quantization_config")
            if isinstance(quant_config, dict):
                # quant_method is the canonical HF key; quant_type is a fallback alias.
                marker = (
                    quant_config.get("quant_method")
                    or quant_config.get("quant_type")
                    or ""
                ).lower()
                if marker == "awq":
                    return "awq"
                if marker == "gptq":
                    return "gptq"
        except Exception:
            # Corrupt/unreadable config.json — fall through to the
            # quantize_config.json check, but leave a breadcrumb: an AWQ model
            # (no quantize_config.json fallback) would otherwise be silently
            # treated as MLX-native.
            logger.warning(
                "Could not parse %s for AWQ/GPTQ detection", config_path, exc_info=True
            )

    if (model_dir / "quantize_config.json").exists():
        return "gptq"

    return None


def convert_to_mlx(src: Path, dst: Path, bits: int, group_size: int) -> None:
    """Convert an AWQ/GPTQ model at *src* to MLX int4/int8 at *dst*.

    Places a ``.converting`` marker *beside* *dst* before starting (see
    ``converting_marker``); removes it on success and writes
    ``conversion_source.json`` with provenance.  On failure the marker is left
    in place so ``_is_valid_mlx_dir()`` in ``store.py`` returns ``False`` — a
    subsequent re-pull can retry (and clears any partial output first).

    ``mlx_lm.convert`` is imported here (not at module level) so the import
    chain stays lightweight when the function is not called.
    """
    import mlx_lm

    dst.parent.mkdir(parents=True, exist_ok=True)
    marker = converting_marker(dst)
    marker.touch()

    # mlx_lm.convert refuses a pre-existing mlx_path; clear any partial output
    # left by an earlier failed attempt before re-converting.
    if dst.exists():
        shutil.rmtree(dst)

    try:
        mlx_lm.convert(
            hf_path=str(src),
            mlx_path=str(dst),
            quantize=True,
            q_bits=bits,
            q_group_size=group_size,
        )
    except Exception:
        # Intentionally leave the marker so _is_valid_mlx_dir stays False and a
        # subsequent re-pull retries (clearing the partial dst above).
        raise

    fmt = detect_format(src)
    (dst / "conversion_source.json").write_text(
        json.dumps({"original_hf_path": str(src), "format": fmt})
    )
    marker.unlink(missing_ok=True)
