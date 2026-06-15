"""GPT-5.4-mini judge for Stage-3: scores convention-adherence + coherence."""

from __future__ import annotations

import json
import re
from typing import Any

_SYSTEM = (
    "You are a strict code-review judge for the olmlx project (an Apple-MLX, "
    "Ollama-compatible inference server). Score a model's answer to an olmlx "
    "prompt on two axes, each an integer 1-5:\n"
    "- convention_adherence: uses olmlx idioms, respects documented invariants, "
    "matches the project's style and type-annotation rules.\n"
    "- coherence: fluent, on-topic, internally consistent; no degradation.\n"
    "Reply with ONLY a JSON object: "
    '{"convention_adherence": <1-5>, "coherence": <1-5>, "rationale": "<one sentence>"}.'
)

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _clamp(v: Any) -> int:
    return max(1, min(5, int(v)))


class ProxyEvalJudge:
    """Wraps a ``.generate(system, user) -> str`` generator with the rubric."""

    def __init__(self, generator: Any):
        self._gen = generator

    def score(self, *, prompt: str, completion: str) -> tuple[int, int, str]:
        user = (
            f"PROMPT:\n{prompt}\n\n"
            f"MODEL ANSWER:\n{completion}\n\n"
            "Score the answer per the rubric and reply with only the JSON object."
        )
        raw = self._gen.generate(_SYSTEM, user)
        obj = self._parse(raw)
        return (
            _clamp(obj["convention_adherence"]),
            _clamp(obj["coherence"]),
            str(obj.get("rationale", "")),
        )

    @staticmethod
    def _parse(raw: str) -> dict[str, Any]:
        for pat in (_FENCE, _OBJ):
            m = pat.search(raw)
            if m:
                try:
                    return json.loads(m.group(1) if pat is _FENCE else m.group(0))
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"judge returned unparseable response: {raw[:200]!r}")
