"""Data types + JSONL IO for the proxy-tuning data pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ExtractionUnit:
    """A grounded source chunk + a seed describing what to generate from it."""

    kind: str  # "function" | "invariant" | "doc" | "test" | "commit"
    provenance: str  # "olmlx/foo.py:10" or "git:<sha>"
    instruction_hint: str  # short seed for the generator
    source_context: str  # grounding text the generator must answer from

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "provenance": self.provenance,
            "instruction_hint": self.instruction_hint,
            "source_context": self.source_context,
        }


@dataclass(frozen=True)
class ChatExample:
    """One instruction->response training pair plus provenance metadata."""

    kind: str
    provenance: str
    user: str
    assistant: str

    def to_chat_row(self) -> dict[str, Any]:
        """mlx-lm chat format — only this shape goes into train.jsonl."""
        return {
            "messages": [
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ]
        }

    def to_dict(self) -> dict[str, Any]:
        """Full row (with kind + provenance) for the checkpoint ``raw.jsonl``."""
        return {
            "kind": self.kind,
            "provenance": self.provenance,
            "user": self.user,
            "assistant": self.assistant,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChatExample":
        """Inverse of :meth:`to_dict` for reading back ``raw.jsonl``."""
        return cls(
            kind=d["kind"],
            provenance=d["provenance"],
            user=d["user"],
            assistant=d["assistant"],
        )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as JSON Lines; return the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file into a list of dicts (blank lines skipped)."""
    out: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
