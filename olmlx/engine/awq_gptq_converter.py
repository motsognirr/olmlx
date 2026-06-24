"""AWQ / GPTQ → MLX auto-conversion utilities.

Imported by ``olmlx.models.store`` during ``pull()``; no heavy dependencies
at module level so the server starts without mlx_lm being fully configured.
``mlx_lm.convert`` is imported lazily inside ``convert_to_mlx`` so test
suites that mock ``mlx_lm`` at a higher level can still import this module
freely.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_format(model_dir: Path) -> str | None:
    """Detect whether *model_dir* contains AWQ or GPTQ weights.

    Returns ``"awq"``, ``"gptq"``, or ``None`` for MLX-native / plain FP16.

    Detection strategy (in priority order):
    1. ``config.json`` → ``quantization_config.quant_type`` field (HF
       standard, used by both AWQ and GPTQ transformers checkpoints).
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
                quant_type = quant_config.get("quant_type", "").lower()
                if quant_type == "awq":
                    return "awq"
                if quant_type == "gptq":
                    return "gptq"
        except Exception:
            pass

    if (model_dir / "quantize_config.json").exists():
        return "gptq"

    return None


def convert_to_mlx(src: Path, dst: Path, bits: int, group_size: int) -> None:
    """Convert an AWQ/GPTQ model at *src* to MLX int4/int8 at *dst*.

    Places a ``.converting`` marker in *dst* before starting; removes it on
    success and writes ``conversion_source.json`` with provenance.  On
    failure the marker is left in place so ``_is_valid_mlx_dir()`` in
    ``store.py`` returns ``False`` — a subsequent re-pull can retry.

    ``mlx_lm.convert`` is imported here (not at module level) so the import
    chain stays lightweight when the function is not called.
    """
    import mlx_lm

    dst.mkdir(parents=True, exist_ok=True)
    marker = dst / ".converting"
    marker.touch()

    try:
        mlx_lm.convert(
            model=str(src),
            mlx_path=str(dst),
            quantize=True,
            q_bits=bits,
            q_group_size=group_size,
        )
    except Exception:
        # Intentionally leave .converting so _is_valid_mlx_dir stays False.
        raise

    marker.unlink(missing_ok=True)

    fmt = detect_format(src)
    (dst / "conversion_source.json").write_text(
        json.dumps({"original_hf_path": str(src), "format": fmt})
    )
