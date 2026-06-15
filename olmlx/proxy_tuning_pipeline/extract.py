"""Mechanical extraction of grounded units from the olmlx repo (no LLM)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator

from olmlx.proxy_tuning_pipeline.schema import ExtractionUnit

# Token shapes to redact before any text leaves the repo for generation.
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),  # OpenAI-style keys
    re.compile(r"\bhf_[A-Za-z0-9]{16,}"),  # HuggingFace tokens
    re.compile(r"\b[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\s*=\s*\S+"),
]


def strip_secrets(text: str) -> str:
    """Redact secret-shaped substrings; preserve everything else verbatim."""
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


# Cap grounding size so a huge function/file doesn't blow the generator context.
_MAX_SOURCE_CHARS = 6000


def _iter_py_files(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        yield p


def extract_functions(root: str | Path) -> Iterator[ExtractionUnit]:
    """Yield one unit per top-level/class function with its source + docstring."""
    root = Path(root)
    for path in _iter_py_files(root):
        try:
            text = path.read_text()
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            try:
                segment = ast.get_source_segment(text, node) or ""
            except Exception:  # noqa: BLE001
                segment = ""
            if not segment.strip():
                continue
            rel = path.relative_to(root) if path.is_relative_to(root) else path
            yield ExtractionUnit(
                kind="function",
                provenance=f"{rel}:{node.lineno}",
                instruction_hint=f"the `{node.name}` function and its olmlx conventions",
                source_context=strip_secrets(segment[:_MAX_SOURCE_CHARS]),
            )
