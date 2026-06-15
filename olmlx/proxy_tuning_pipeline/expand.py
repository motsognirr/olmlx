"""Expansion of extracted units into instruction->response pairs via a Generator."""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Protocol

import concurrent.futures
from pathlib import Path

from olmlx.proxy_tuning_pipeline.schema import (
    ChatExample,
    ExtractionUnit,
    read_jsonl,
)

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
    prose or emits a second object. Walk braces instead — and track JSON
    string/escape state so a literal ``{`` or ``}`` *inside a quoted value*
    (very common here: the responses contain olmlx code snippets) doesn't
    miscount the depth and drop otherwise-valid data.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
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


def _examples_for_unit(
    unit: ExtractionUnit, generator: Generator, n_per_unit: int
) -> list[ChatExample]:
    """One generate call -> parsed ChatExamples for a single unit. May raise."""
    system, user = build_messages(unit, n_per_unit)
    reply = generator.generate(system, user)
    return [
        ChatExample(
            kind=unit.kind, provenance=unit.provenance, user=instr, assistant=resp
        )
        for instr, resp in parse_pairs(reply)
    ]


def expand_units(
    units: list[ExtractionUnit],
    generator: Generator,
    n_per_unit: int,
) -> list[ChatExample]:
    """Expand each unit into ChatExamples via the generator (one call per unit).

    In-memory, sequential. ``expand_units_checkpointed`` is the crash-safe,
    concurrent variant used for the full run.
    """
    examples: list[ChatExample] = []
    failures = 0
    for i, unit in enumerate(units):
        try:
            examples.extend(_examples_for_unit(unit, generator, n_per_unit))
        except Exception:  # noqa: BLE001 — one bad unit must not abort the run
            failures += 1
            logger.warning("generation failed for %s", unit.provenance, exc_info=True)
            continue
        if (i + 1) % 100 == 0:
            logger.info(
                "expanded %d/%d units, %d pairs so far",
                i + 1,
                len(units),
                len(examples),
            )
    if failures:
        logger.warning(
            "expand_units: %d/%d units failed generation", failures, len(units)
        )
    if units and not examples:
        logger.error("expand_units: produced 0 examples from %d units", len(units))
    return examples


def load_done_provenances(raw_path: str | Path) -> set[str]:
    """Provenances already present in the checkpoint file (for resume)."""
    p = Path(raw_path)
    if not p.exists():
        return set()
    return {row["provenance"] for row in read_jsonl(p)}


def load_examples(raw_path: str | Path) -> list[ChatExample]:
    """Read the checkpoint file back into ChatExamples (empty if absent)."""
    p = Path(raw_path)
    if not p.exists():
        return []
    return [ChatExample.from_dict(row) for row in read_jsonl(p)]


def expand_units_checkpointed(
    units: list[ExtractionUnit],
    generator: Generator,
    n_per_unit: int,
    raw_path: str | Path,
    concurrency: int = 8,
    *,
    done: set[str] | None = None,
) -> int:
    """Expand units concurrently, appending each unit's pairs to ``raw_path``.

    Crash-safe and resumable: every unit's pairs are flushed to ``raw_path`` the
    moment that unit finishes, so an interrupt loses at most the in-flight units.
    Units whose provenance is in ``done`` are skipped (resume). Returns the
    number of pairs written this run.

    Generator calls run in a bounded thread pool (the OpenAI client releases the
    GIL during network I/O); the single main thread is the only writer, so
    ``raw_path`` needs no locking.
    """
    done = done or set()
    todo = [u for u in units if u.provenance not in done]
    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    failures = 0
    with (
        raw_path.open("a") as f,
        concurrent.futures.ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool,
    ):
        fut_to_unit = {
            pool.submit(_examples_for_unit, u, generator, n_per_unit): u for u in todo
        }
        for i, fut in enumerate(concurrent.futures.as_completed(fut_to_unit), 1):
            unit = fut_to_unit[fut]
            try:
                examples = fut.result()
            except Exception:  # noqa: BLE001 — one bad unit must not abort the run
                failures += 1
                logger.warning(
                    "generation failed for %s", unit.provenance, exc_info=True
                )
                continue
            for e in examples:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
                written += 1
            f.flush()  # checkpoint durability: survive a crash on the next unit
            if i % 100 == 0:
                logger.info(
                    "expanded %d/%d units, %d pairs written", i, len(todo), written
                )
    if failures:
        logger.warning("expand: %d/%d units failed generation", failures, len(todo))
    return written


DEFAULT_MODEL = "gpt-5.4-mini"


class OpenAIGenerator:
    """`Generator` backed by OpenAI chat completions (GPT-5.4-mini by default).

    `client` is injectable for testing; when None it is lazily constructed via
    ``openai.OpenAI()`` (reads ``OPENAI_API_KEY`` from the environment).
    """

    def __init__(
        self, client: Any = None, model: str = DEFAULT_MODEL, max_retries: int = 6
    ):
        self._client = client
        self._model = model
        self._max_retries = max_retries
        self._client_lock = threading.Lock()

    def _ensure_client(self) -> Any:
        # Double-checked lock: under concurrency the first batch of worker
        # threads would otherwise each construct a client.
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    from openai import OpenAI

                    # The SDK auto-retries 429 / transient 5xx with exponential
                    # backoff; raise the cap above the default 2 for a long run.
                    self._client = OpenAI(max_retries=self._max_retries)
        return self._client

    def generate(self, system: str, user: str) -> str:
        resp = self._ensure_client().chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
