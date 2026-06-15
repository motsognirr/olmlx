"""Expansion of extracted units into instruction->response pairs via a Generator."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from olmlx.proxy_tuning_pipeline.schema import ChatExample, ExtractionUnit

# Clamp the grounding so a long unit can't blow the generator's context budget.
_MAX_USER_CHARS = 12000

_SYSTEM_PROMPT = (
    "You are generating supervised fine-tuning data about the olmlx codebase "
    "(an Ollama-compatible MLX inference server) and adjacent MLX / inference "
    "optimization topics. You MUST ground every answer strictly in the SOURCE "
    "provided by the user — do not invent APIs, file names, or behavior that "
    "is not in the SOURCE; if the SOURCE does not support a detail, omit it. "
    "Produce diverse, natural instruction/response pairs that teach olmlx's "
    "conventions and idioms. Reply with ONLY a JSON object of the form "
    '{"pairs": [{"instruction": "...", "response": "..."}]} and nothing else.'
)


class Generator(Protocol):
    """Anything that maps (system, user) prompts to a single text completion."""

    def generate(self, system: str, user: str) -> str: ...


def build_messages(unit: ExtractionUnit, n_pairs: int) -> tuple[str, str]:
    """Build (system, user) prompts asking for `n_pairs` grounded pairs."""
    source = unit.source_context[:_MAX_USER_CHARS]
    user = (
        f"Generate {n_pairs} diverse instruction/response pairs that teach "
        f"{unit.instruction_hint}. Vary the task type (explain / implement / "
        f"review / convert) and phrasing. Ground every answer in this SOURCE "
        f"(provenance: {unit.provenance}):\n\n--- SOURCE ---\n{source}\n--- END SOURCE ---"
    )
    return _SYSTEM_PROMPT, user


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` span, or None.

    A greedy ``\\{.*\\}`` regex over-grabs when the model wraps the JSON in
    prose or emits a second object — it would span to the *last* ``}`` and
    fail to parse, silently dropping good data. Walk braces instead so prose
    and nested braces in values are handled correctly.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_pairs(text: str) -> list[tuple[str, str]]:
    """Parse a generator reply into (instruction, response) pairs; tolerant.

    Strips ```json fences, extracts the first balanced JSON object (ignoring
    surrounding prose), and drops pairs missing a non-empty instruction or
    response.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    obj = _extract_json_object(text)
    if obj is None:
        return []
    try:
        data = json.loads(obj)
    except json.JSONDecodeError:
        return []
    out: list[tuple[str, str]] = []
    for item in data.get("pairs", []):
        if not isinstance(item, dict):
            continue
        instr = str(item.get("instruction", "")).strip()
        resp = str(item.get("response", "")).strip()
        if instr and resp:
            out.append((instr, resp))
    return out


logger = logging.getLogger(__name__)


def expand_units(
    units: list[ExtractionUnit],
    generator: Generator,
    n_per_unit: int,
) -> list[ChatExample]:
    """Expand each unit into ChatExamples via the generator (one call per unit)."""
    examples: list[ChatExample] = []
    failures = 0
    for i, unit in enumerate(units):
        system, user = build_messages(unit, n_per_unit)
        try:
            reply = generator.generate(system, user)
        except Exception:  # noqa: BLE001 — one bad unit must not abort the run
            failures += 1
            logger.warning("generation failed for %s", unit.provenance, exc_info=True)
            continue
        for instr, resp in parse_pairs(reply):
            examples.append(
                ChatExample(
                    kind=unit.kind,
                    provenance=unit.provenance,
                    user=instr,
                    assistant=resp,
                )
            )
        if (i + 1) % 100 == 0:
            logger.info(
                "expanded %d/%d units, %d pairs so far",
                i + 1,
                len(units),
                len(examples),
            )
    # Surface systemic failure: a run where every call failed (bad key, bad
    # model, exhausted quota) would otherwise silently produce an empty dataset
    # only noticed at training time.
    if failures:
        logger.warning(
            "expand_units: %d/%d units failed generation", failures, len(units)
        )
    if units and not examples:
        logger.error("expand_units: produced 0 examples from %d units", len(units))
    return examples


DEFAULT_MODEL = "gpt-5.4-mini"


class OpenAIGenerator:
    """`Generator` backed by OpenAI chat completions (GPT-5.4-mini by default).

    `client` is injectable for testing; when None it is lazily constructed via
    ``openai.OpenAI()`` (reads ``OPENAI_API_KEY`` from the environment).
    """

    def __init__(self, client: Any = None, model: str = DEFAULT_MODEL):
        self._client = client
        self._model = model

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def generate(self, system: str, user: str) -> str:
        # TODO: add retry/backoff on 429 / transient 5xx before the full run —
        # a single rate-limit error currently drops one unit's pairs.
        resp = self._ensure_client().chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
