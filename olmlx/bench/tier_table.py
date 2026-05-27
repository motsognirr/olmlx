"""Static tier assignment for the May 2026 extended benchmark.

See `docs/superpowers/specs/2026-05-24-extended-bench-design.md` for the
tiering rationale. Keys are HuggingFace paths (the value of ``hf_path`` in
``~/.olmlx/models.json``), not Ollama-style ``:latest`` aliases.
"""

from __future__ import annotations

from enum import Enum


class Tier(Enum):
    EXTENDED = "extended"
    CORE_ONLY = "core_only"


EXTENDED: frozenset[str] = frozenset(
    {
        "mlx-community/Qwen3-Coder-Next-4bit",
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        "mlx-community/Qwen3.6-35B-A3B-6bit",
        "mlx-community/Qwen3.6-27B-4bit",
        "mlx-community/Nemotron-Cascade-2-30B-A3B-4bit",
        "mlx-community/gemma-4-31B-it-OptiQ-4bit",
        "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit",
        "mlx-community/Qwen3-8B-4bit",
        "lmstudio-community/Devstral-Small-2505-MLX-6bit",
        "mlx-community/Qwen3-4B-4bit",
        "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        "prism-ml/Ternary-Bonsai-8B-mlx-2bit",
        "mlx-community/gemma-4-e2b-it-OptiQ-4bit",
    }
)

CORE_ONLY: frozenset[str] = frozenset(
    {
        "mlx-community/gpt-oss-120b-MXFP4-Q4",
        "mlx-community/MiniMax-M2.7-5bit",
        "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit",
        "mlx-community/Step-3.5-Flash-6bit",
        "unsloth/Qwen3.6-27B-MLX-8bit",
        "mlx-community/Qwen3.5-27B-4bit",
        "clowncar/generalist",
        "mlx-community/Qwen3-0.6B-4bit",
        "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    }
)


# Models that get the ablation reruns; one anchor per knob.
ABLATION_ANCHORS: dict[str, list[str]] = {
    "turboquant": ["mlx-community/Qwen3-Coder-Next-4bit"],
    "speculative": ["mlx-community/Qwen3.6-35B-A3B-4bit"],
}


def tier_for(hf_path: str) -> Tier | None:
    """Return the tier for an HF path, or ``None`` if unknown."""
    if hf_path in EXTENDED:
        return Tier.EXTENDED
    if hf_path in CORE_ONLY:
        return Tier.CORE_ONLY
    return None
