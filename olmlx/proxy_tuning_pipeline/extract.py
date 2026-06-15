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


def extract_invariants(claude_md_path: str | Path) -> Iterator[ExtractionUnit]:
    """Yield one unit per `**Title** — ...` paragraph under the invariants section."""
    path = Path(claude_md_path)
    text = path.read_text()
    # Isolate the "## Non-Obvious Invariants" section up to the next "## " heading.
    m = re.search(r"^##\s+Non-Obvious Invariants\s*$", text, flags=re.MULTILINE)
    if not m:
        return
    rest = text[m.end():]
    nxt = re.search(r"^##\s+", rest, flags=re.MULTILINE)
    section = rest[: nxt.start()] if nxt else rest
    for para in re.split(r"\n\s*\n", section):
        para = para.strip()
        title_m = re.match(r"\*\*(.+?)\*\*", para)
        if not title_m:
            continue
        yield ExtractionUnit(
            kind="invariant",
            provenance=f"{path.name}#non-obvious-invariants",
            instruction_hint=f"the olmlx invariant: {title_m.group(1)}",
            source_context=strip_secrets(para[:_MAX_SOURCE_CHARS]),
        )


def extract_docs(docs_dir: str | Path) -> Iterator[ExtractionUnit]:
    """Yield one unit per markdown section (## or # heading + its body)."""
    docs_dir = Path(docs_dir)
    for path in sorted(docs_dir.rglob("*.md")):
        text = path.read_text()
        # Split on headings, keeping the heading with its following body.
        parts = re.split(r"(?m)^(#{1,3}\s+.*)$", text)
        # parts = [pre, heading1, body1, heading2, body2, ...]
        for i in range(1, len(parts), 2):
            heading = parts[i].lstrip("#").strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if not body:
                continue
            rel = path.relative_to(docs_dir) if path.is_relative_to(docs_dir) else path
            yield ExtractionUnit(
                kind="doc",
                provenance=f"docs/{rel}#{heading[:40]}",
                instruction_hint=f"the olmlx documentation section: {heading}",
                source_context=strip_secrets((parts[i] + "\n" + body)[:_MAX_SOURCE_CHARS]),
            )
