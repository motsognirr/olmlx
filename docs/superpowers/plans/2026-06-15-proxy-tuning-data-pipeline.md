# Proxy-Tuning Data Pipeline (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline data pipeline that turns the olmlx repo into a curated ~10k-example chat-format SFT dataset (`train.jsonl` / `valid.jsonl`) for training the proxy-tuning expert M⁺.

**Architecture:** A three-phase offline pipeline in a new importable package `olmlx/proxy_tuning_pipeline/`: (1) **mechanical extraction** mines grounded `(source, seed)` units from functions, CLAUDE.md invariants, docs, tests, and git commits — pure stdlib, deterministic; (2) **expansion** sends each unit to a `Generator` (OpenAI GPT-5.4-mini by default, dependency-injected so tests use a fake) to produce diverse instruction→response pairs; (3) **curation** dedupes, filters, and splits into mlx-lm chat JSONL. A thin CLI wires the phases with intermediate artifacts under `data/proxy_tuning/` (gitignored).

**Tech Stack:** Python 3.11 (stdlib `ast`, `subprocess`, `json`, `re`, `hashlib`, `random`), `openai>=1.66` (already a dependency; lazy/injected), pytest.

**Scope:** This plan is **Stage 1 only** of `docs/superpowers/specs/2026-06-15-olmlx-proxy-tuning-pair-design.md`. Stages 2 (training) and 3 (eval/serve) are separate plans, each blocked on this stage's artifacts.

**Key design choices (locked in this plan):**
- Package `olmlx/proxy_tuning_pipeline/` (importable; never imported by the running server, so it adds no runtime coupling).
- `Generator` is a `Protocol`; `expand_units` takes one via dependency injection. Tests use `FakeGenerator`; the concrete `OpenAIGenerator` accepts an injectable client so its test needs no network and no real `openai` object.
- Generator returns JSON `{"pairs": [{"instruction": ..., "response": ...}]}`; the driver parses + validates.
- Dedup is **normalized-exact** (lowercase + whitespace-collapsed hash) — no embedding/ML deps (YAGNI). Near-dup-by-embedding is a documented future option, not built here.
- Output is mlx-lm chat format: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `olmlx/proxy_tuning_pipeline/__init__.py` | Package marker | Create |
| `olmlx/proxy_tuning_pipeline/schema.py` | `ExtractionUnit`, `ChatExample` dataclasses + JSONL read/write | Create |
| `olmlx/proxy_tuning_pipeline/extract.py` | `strip_secrets` + 5 extractors + `extract_repo` aggregator + `dedupe_units` | Create |
| `olmlx/proxy_tuning_pipeline/expand.py` | `Generator` protocol, `build_messages`, `parse_pairs`, `expand_units`, `OpenAIGenerator` | Create |
| `olmlx/proxy_tuning_pipeline/curate.py` | `quality_filter`, `dedupe_examples`, `split_train_valid` | Create |
| `olmlx/proxy_tuning_pipeline/cli.py` | `main()` — wire extract → expand → curate → write artifacts | Create |
| `tests/test_proxy_tuning_pipeline.py` | All unit + end-to-end tests for the pipeline | Create |
| `.gitignore` | Ignore `data/proxy_tuning/` outputs | Modify |

---

## Task 1: Package + schema + JSONL IO

**Files:**
- Create: `olmlx/proxy_tuning_pipeline/__init__.py`
- Create: `olmlx/proxy_tuning_pipeline/schema.py`
- Modify: `.gitignore`
- Test: `tests/test_proxy_tuning_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_proxy_tuning_pipeline.py`:

```python
"""Tests for the proxy-tuning data pipeline (olmlx/proxy_tuning_pipeline)."""

from __future__ import annotations

from olmlx.proxy_tuning_pipeline.schema import (
    ChatExample,
    ExtractionUnit,
    read_jsonl,
    write_jsonl,
)


def test_jsonl_round_trip(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = tmp_path / "out.jsonl"
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_extraction_unit_to_dict():
    u = ExtractionUnit(
        kind="function",
        provenance="olmlx/foo.py:10",
        instruction_hint="explain this",
        source_context="def f(): ...",
    )
    assert u.to_dict() == {
        "kind": "function",
        "provenance": "olmlx/foo.py:10",
        "instruction_hint": "explain this",
        "source_context": "def f(): ...",
    }


def test_chat_example_to_jsonl_row_is_mlx_chat_format():
    ex = ChatExample(
        kind="function",
        provenance="olmlx/foo.py:10",
        user="How does f work?",
        assistant="It returns nothing.",
    )
    # mlx-lm chat format: only the `messages` key reaches train.jsonl.
    assert ex.to_chat_row() == {
        "messages": [
            {"role": "user", "content": "How does f work?"},
            {"role": "assistant", "content": "It returns nothing."},
        ]
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.proxy_tuning_pipeline'`

- [ ] **Step 3: Write minimal implementation**

Create `olmlx/proxy_tuning_pipeline/__init__.py`:

```python
"""Offline data/training tooling for proxy-tuning (Stage 1: data pipeline).

NOT imported by the running olmlx server — this package builds the SFT dataset
used to train the proxy-tuning expert M+. See
docs/superpowers/specs/2026-06-15-olmlx-proxy-tuning-pair-design.md.
"""
```

Create `olmlx/proxy_tuning_pipeline/schema.py`:

```python
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
```

Append to `.gitignore` (add a new line at the end):

```
# Proxy-tuning generated datasets (large, regenerable)
data/proxy_tuning/
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/proxy_tuning_pipeline/__init__.py olmlx/proxy_tuning_pipeline/schema.py tests/test_proxy_tuning_pipeline.py .gitignore
git commit -m "feat(proxy-tuning-data): schema + JSONL IO for the data pipeline"
```

---

## Task 2: `strip_secrets`

**Files:**
- Create: `olmlx/proxy_tuning_pipeline/extract.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
from olmlx.proxy_tuning_pipeline.extract import strip_secrets


def test_strip_secrets_redacts_known_token_shapes():
    text = (
        "key sk-abc123def456ghi789jkl012mno345pqr and "
        "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345 plus "
        "OLMLX_SECRET=topsecretvalue123 done"
    )
    out = strip_secrets(text)
    assert "sk-abc123" not in out
    assert "hf_AbCdEf" not in out
    assert "topsecretvalue123" not in out
    assert "[REDACTED]" in out
    # Non-secret text is preserved.
    assert out.startswith("key ") and out.endswith(" done")


def test_strip_secrets_leaves_ordinary_text_untouched():
    text = "def generate_chat(model, messages): return run(model, messages)"
    assert strip_secrets(text) == text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k strip_secrets -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.proxy_tuning_pipeline.extract'`

- [ ] **Step 3: Write minimal implementation**

Create `olmlx/proxy_tuning_pipeline/extract.py`:

```python
"""Mechanical extraction of grounded units from the olmlx repo (no LLM)."""

from __future__ import annotations

import re

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k strip_secrets -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/proxy_tuning_pipeline/extract.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): strip_secrets redaction"
```

---

## Task 3: Function/docstring extractor

**Files:**
- Modify: `olmlx/proxy_tuning_pipeline/extract.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
from olmlx.proxy_tuning_pipeline.extract import extract_functions


def test_extract_functions_yields_unit_per_function(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text(
        'def add(a, b):\n'
        '    """Return the sum of a and b."""\n'
        '    return a + b\n'
        '\n'
        'def _private():\n'
        '    return 1\n'
    )
    units = list(extract_functions(tmp_path))
    by_name = {u.provenance: u for u in units}
    # Both functions captured; provenance is file:lineno.
    assert any(p.endswith("mod.py:1") for p in by_name)
    add_unit = next(u for u in units if "def add" in u.source_context)
    assert add_unit.kind == "function"
    assert "Return the sum" in add_unit.source_context
    assert "add" in add_unit.instruction_hint


def test_extract_functions_skips_non_python_and_pycache(tmp_path):
    (tmp_path / "note.txt").write_text("def not_python(): pass")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "x.py").write_text("def cached(): pass")
    assert list(extract_functions(tmp_path)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k extract_functions -v`
Expected: FAIL with `ImportError: cannot import name 'extract_functions'`

- [ ] **Step 3: Write minimal implementation**

Add to `olmlx/proxy_tuning_pipeline/extract.py` (append imports + function):

```python
import ast
from pathlib import Path
from typing import Iterator

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k extract_functions -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/proxy_tuning_pipeline/extract.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): function/docstring extractor"
```

---

## Task 4: CLAUDE.md invariants extractor

**Files:**
- Modify: `olmlx/proxy_tuning_pipeline/extract.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

**Context:** In `CLAUDE.md`, the `## Non-Obvious Invariants` section contains one invariant per paragraph, each starting with a bold title then an em-dash: `**Metal stream hazard** — All non-speculative inference ...`. Paragraphs are blank-line separated. The section ends at the next `## ` heading.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
from olmlx.proxy_tuning_pipeline.extract import extract_invariants


def test_extract_invariants_parses_bold_lead_paragraphs(tmp_path):
    md = tmp_path / "CLAUDE.md"
    md.write_text(
        "# Title\n\n"
        "## Non-Obvious Invariants\n\n"
        "**Metal stream hazard** — All inference must run on one stream.\n\n"
        "**MTP concat order** — embed first, opposite of EAGLE.\n\n"
        "## Development\n\n"
        "not an invariant\n"
    )
    units = list(extract_invariants(md))
    titles = [u.instruction_hint for u in units]
    assert any("Metal stream hazard" in t for t in titles)
    assert any("MTP concat order" in t for t in titles)
    assert all(u.kind == "invariant" for u in units)
    # The trailing "## Development" content is not captured.
    assert not any("not an invariant" in u.source_context for u in units)
    assert len(units) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k extract_invariants -v`
Expected: FAIL with `ImportError: cannot import name 'extract_invariants'`

- [ ] **Step 3: Write minimal implementation**

Add to `olmlx/proxy_tuning_pipeline/extract.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k extract_invariants -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/proxy_tuning_pipeline/extract.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): CLAUDE.md invariants extractor"
```

---

## Task 5: Docs/specs extractor

**Files:**
- Modify: `olmlx/proxy_tuning_pipeline/extract.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
from olmlx.proxy_tuning_pipeline.extract import extract_docs


def test_extract_docs_chunks_by_section(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text(
        "# Intro\n\nWelcome to olmlx.\n\n"
        "## Speculative\n\nDraft then verify.\n\n"
        "## Flash\n\nSSD-backed MoE.\n"
    )
    (d / "nested" / "b.md").parent.mkdir()
    (d / "nested" / "b.md").write_text("# Nested\n\nbody here\n")
    units = list(extract_docs(d))
    headings = [u.instruction_hint for u in units]
    assert any("Speculative" in h for h in headings)
    assert any("Flash" in h for h in headings)
    assert any("Nested" in h for h in headings)
    assert all(u.kind == "doc" for u in units)
    spec = next(u for u in units if "Speculative" in u.instruction_hint)
    assert "Draft then verify" in spec.source_context
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k extract_docs -v`
Expected: FAIL with `ImportError: cannot import name 'extract_docs'`

- [ ] **Step 3: Write minimal implementation**

Add to `olmlx/proxy_tuning_pipeline/extract.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k extract_docs -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/proxy_tuning_pipeline/extract.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): docs/specs section extractor"
```

---

## Task 6: Tests extractor

**Files:**
- Modify: `olmlx/proxy_tuning_pipeline/extract.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
from olmlx.proxy_tuning_pipeline.extract import extract_tests


def test_extract_tests_yields_unit_per_test_function(tmp_path):
    t = tmp_path / "tests"
    t.mkdir()
    (t / "test_thing.py").write_text(
        'def test_adds():\n'
        '    """Addition works."""\n'
        '    assert 1 + 1 == 2\n'
        '\n'
        'def helper():\n'
        '    return 0\n'
    )
    units = list(extract_tests(t))
    assert len(units) == 1
    u = units[0]
    assert u.kind == "test"
    assert "test_adds" in u.instruction_hint
    assert "assert 1 + 1 == 2" in u.source_context
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k extract_tests -v`
Expected: FAIL with `ImportError: cannot import name 'extract_tests'`

- [ ] **Step 3: Write minimal implementation**

Add to `olmlx/proxy_tuning_pipeline/extract.py`:

```python
def extract_tests(tests_dir: str | Path) -> Iterator[ExtractionUnit]:
    """Yield one unit per `def test_*` function (the behavior it enforces)."""
    tests_dir = Path(tests_dir)
    for path in sorted(tests_dir.rglob("test_*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text()
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            segment = ast.get_source_segment(text, node) or ""
            if not segment.strip():
                continue
            rel = path.relative_to(tests_dir) if path.is_relative_to(tests_dir) else path
            yield ExtractionUnit(
                kind="test",
                provenance=f"tests/{rel}:{node.lineno}",
                instruction_hint=f"the behavior enforced by `{node.name}`",
                source_context=strip_secrets(segment[:_MAX_SOURCE_CHARS]),
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k extract_tests -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/proxy_tuning_pipeline/extract.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): test-function extractor"
```

---

## Task 7: Git-commit extractor

**Files:**
- Modify: `olmlx/proxy_tuning_pipeline/extract.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
import subprocess

from olmlx.proxy_tuning_pipeline.extract import extract_commits


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": __import__("os").environ.get("PATH", ""),
        },
    )


def test_extract_commits_yields_message_and_diff(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "f.py").write_text("x = 1\n")
    _git(repo, "add", "f.py")
    _git(repo, "commit", "-m", "feat: add x constant")
    units = list(extract_commits(repo, limit=10))
    assert len(units) == 1
    u = units[0]
    assert u.kind == "commit"
    assert "feat: add x constant" in u.source_context
    assert "x = 1" in u.source_context  # diff body included
    assert u.provenance.startswith("git:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k extract_commits -v`
Expected: FAIL with `ImportError: cannot import name 'extract_commits'`

- [ ] **Step 3: Write minimal implementation**

Add to `olmlx/proxy_tuning_pipeline/extract.py` (append `import subprocess` near the top imports):

```python
import subprocess


def extract_commits(root: str | Path, limit: int = 400) -> Iterator[ExtractionUnit]:
    """Yield one unit per commit: subject + body + diff (capped). Newest first."""
    root = Path(root)
    try:
        shas = subprocess.run(
            ["git", "-C", str(root), "log", f"-n{limit}", "--format=%H"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    for sha in shas:
        show = subprocess.run(
            ["git", "-C", str(root), "show", "--format=%s%n%n%b", "--stat", "-p", sha],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if not show.strip():
            continue
        yield ExtractionUnit(
            kind="commit",
            provenance=f"git:{sha[:12]}",
            instruction_hint="implementing this change the olmlx way (message -> diff)",
            source_context=strip_secrets(show[:_MAX_SOURCE_CHARS]),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k extract_commits -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/proxy_tuning_pipeline/extract.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): git-commit extractor"
```

---

## Task 8: `extract_repo` aggregator + `dedupe_units`

**Files:**
- Modify: `olmlx/proxy_tuning_pipeline/extract.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
from olmlx.proxy_tuning_pipeline.extract import dedupe_units, extract_repo


def test_dedupe_units_drops_identical_source_context():
    u1 = ExtractionUnit("function", "a.py:1", "h", "def f(): pass")
    u2 = ExtractionUnit("function", "b.py:9", "h2", "def f(): pass")  # same source
    u3 = ExtractionUnit("function", "c.py:1", "h", "def g(): pass")
    out = dedupe_units([u1, u2, u3])
    assert len(out) == 2
    assert out[0] is u1 and out[1] is u3  # first occurrence kept, order preserved


def test_extract_repo_composes_all_extractors(tmp_path):
    repo = tmp_path / "repo"
    (repo / "olmlx").mkdir(parents=True)
    (repo / "olmlx" / "m.py").write_text('def f():\n    """doc."""\n    return 1\n')
    (repo / "CLAUDE.md").write_text(
        "## Non-Obvious Invariants\n\n**Inv one** — a rule.\n"
    )
    (repo / "docs").mkdir()
    (repo / "docs" / "x.md").write_text("## Sec\n\nbody.\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_m.py").write_text("def test_f():\n    assert True\n")
    units = extract_repo(repo)
    kinds = {u.kind for u in units}
    # Commit extractor finds nothing here (no git repo) — that's fine.
    assert {"function", "invariant", "doc", "test"} <= kinds
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k "dedupe_units or extract_repo" -v`
Expected: FAIL with `ImportError: cannot import name 'dedupe_units'`

- [ ] **Step 3: Write minimal implementation**

Add to `olmlx/proxy_tuning_pipeline/extract.py`:

```python
def dedupe_units(units: list[ExtractionUnit]) -> list[ExtractionUnit]:
    """Drop units whose `source_context` was already seen (order-preserving)."""
    seen: set[str] = set()
    out: list[ExtractionUnit] = []
    for u in units:
        key = " ".join(u.source_context.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


def extract_repo(root: str | Path) -> list[ExtractionUnit]:
    """Run every extractor over the repo and return deduped units.

    Each extractor is best-effort: a missing CLAUDE.md / docs / tests / git
    history simply contributes nothing rather than failing the whole run.
    """
    root = Path(root)
    units: list[ExtractionUnit] = []
    units += list(extract_functions(root / "olmlx" if (root / "olmlx").is_dir() else root))
    claude = root / "CLAUDE.md"
    if claude.is_file():
        units += list(extract_invariants(claude))
    docs = root / "docs"
    if docs.is_dir():
        units += list(extract_docs(docs))
    tests = root / "tests"
    if tests.is_dir():
        units += list(extract_tests(tests))
    units += list(extract_commits(root))
    return dedupe_units(units)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k "dedupe_units or extract_repo" -v`
Expected: 2 passed

- [ ] **Step 5: Run ruff, then commit**

Run: `uv run ruff check olmlx/proxy_tuning_pipeline/ && uv run ruff format olmlx/proxy_tuning_pipeline/`
Expected: no errors

```bash
git add olmlx/proxy_tuning_pipeline/extract.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): extract_repo aggregator + unit dedup"
```

---

## Task 9: `Generator` protocol + `build_messages` + `parse_pairs`

**Files:**
- Create: `olmlx/proxy_tuning_pipeline/expand.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
import pytest

from olmlx.proxy_tuning_pipeline.expand import build_messages, parse_pairs


def test_build_messages_grounds_in_source_and_requests_json():
    u = ExtractionUnit("function", "a.py:1", "the `f` function", "def f(): return 1")
    system, user = build_messages(u, n_pairs=3)
    assert "ground" in system.lower()  # grounding discipline present
    assert "JSON" in system or "json" in system
    assert "def f(): return 1" in user  # source context embedded
    assert "3" in user  # requested count
    assert "the `f` function" in user  # instruction hint embedded


def test_build_messages_truncates_oversized_source():
    u = ExtractionUnit("doc", "d.md#s", "a doc", "x" * 99999)
    _system, user = build_messages(u, n_pairs=2)
    assert len(user) < 20000  # clamped well below the raw 99999


def test_parse_pairs_extracts_valid_pairs():
    text = '{"pairs": [{"instruction": "Q1", "response": "A1"}, {"instruction": "Q2", "response": "A2"}]}'
    assert parse_pairs(text) == [("Q1", "A1"), ("Q2", "A2")]


def test_parse_pairs_tolerates_code_fences_and_drops_incomplete():
    text = (
        "```json\n"
        '{"pairs": [{"instruction": "Q", "response": "A"}, '
        '{"instruction": "", "response": "x"}, '
        '{"instruction": "only-q"}]}\n'
        "```"
    )
    assert parse_pairs(text) == [("Q", "A")]


def test_parse_pairs_returns_empty_on_garbage():
    assert parse_pairs("not json at all") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k "build_messages or parse_pairs" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.proxy_tuning_pipeline.expand'`

- [ ] **Step 3: Write minimal implementation**

Create `olmlx/proxy_tuning_pipeline/expand.py`:

```python
"""Expansion of extracted units into instruction->response pairs via a Generator."""

from __future__ import annotations

import json
import re
from typing import Protocol

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


def parse_pairs(text: str) -> list[tuple[str, str]]:
    """Parse a generator reply into (instruction, response) pairs; tolerant.

    Strips ```json fences, ignores anything around the JSON object, and drops
    pairs missing a non-empty instruction or response.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k "build_messages or parse_pairs" -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/proxy_tuning_pipeline/expand.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): Generator protocol + prompt build + pair parse"
```

---

## Task 10: `expand_units` driver (with injected generator)

**Files:**
- Modify: `olmlx/proxy_tuning_pipeline/expand.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
from olmlx.proxy_tuning_pipeline.expand import expand_units


class _FakeGenerator:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def test_expand_units_produces_chat_examples_with_provenance():
    units = [
        ExtractionUnit("function", "a.py:1", "fn a", "def a(): pass"),
        ExtractionUnit("doc", "d.md#s", "doc s", "body"),
    ]
    gen = _FakeGenerator(
        '{"pairs": [{"instruction": "Q1", "response": "A1"}, '
        '{"instruction": "Q2", "response": "A2"}]}'
    )
    examples = expand_units(units, gen, n_per_unit=2)
    assert gen.calls == 2  # one generation call per unit
    assert len(examples) == 4  # 2 pairs * 2 units
    assert all(isinstance(e, ChatExample) for e in examples)
    first = examples[0]
    assert first.user == "Q1" and first.assistant == "A1"
    assert first.provenance == "a.py:1" and first.kind == "function"


def test_expand_units_skips_units_that_yield_no_pairs():
    units = [ExtractionUnit("doc", "d.md#s", "doc", "body")]
    gen = _FakeGenerator("garbage, not json")
    assert expand_units(units, gen, n_per_unit=3) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k expand_units -v`
Expected: FAIL with `ImportError: cannot import name 'expand_units'`

- [ ] **Step 3: Write minimal implementation**

Add to `olmlx/proxy_tuning_pipeline/expand.py`:

```python
import logging

logger = logging.getLogger(__name__)


def expand_units(
    units: list[ExtractionUnit],
    generator: Generator,
    n_per_unit: int,
) -> list[ChatExample]:
    """Expand each unit into ChatExamples via the generator (one call per unit)."""
    examples: list[ChatExample] = []
    for i, unit in enumerate(units):
        system, user = build_messages(unit, n_per_unit)
        try:
            reply = generator.generate(system, user)
        except Exception:  # noqa: BLE001 — one bad unit must not abort the run
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
            logger.info("expanded %d/%d units, %d pairs so far", i + 1, len(units), len(examples))
    return examples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k expand_units -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/proxy_tuning_pipeline/expand.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): expand_units driver"
```

---

## Task 11: `OpenAIGenerator` (injectable client)

**Files:**
- Modify: `olmlx/proxy_tuning_pipeline/expand.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

**Context:** `openai>=1.66` is already a project dependency. The real call is `client.chat.completions.create(model=..., messages=[...]).choices[0].message.content`. The class accepts an injectable `client` so the test needs no network and no real `openai` object; the real client is lazily constructed only when `client is None`. `DEFAULT_MODEL = "gpt-5.4-mini"` is the user-chosen generator and is overridable.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
from olmlx.proxy_tuning_pipeline.expand import DEFAULT_MODEL, OpenAIGenerator


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeOpenAIClient:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None

        class _Completions:
            def create(inner, **kwargs):
                self.last_kwargs = kwargs
                return _FakeCompletion(self._content)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_openai_generator_calls_chat_completions():
    client = _FakeOpenAIClient('{"pairs": []}')
    gen = OpenAIGenerator(client=client)
    out = gen.generate("sys", "usr")
    assert out == '{"pairs": []}'
    assert client.last_kwargs["model"] == DEFAULT_MODEL
    assert client.last_kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_openai_generator_honors_model_override():
    client = _FakeOpenAIClient("x")
    OpenAIGenerator(client=client, model="custom-model").generate("s", "u")
    assert client.last_kwargs["model"] == "custom-model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k openai_generator -v`
Expected: FAIL with `ImportError: cannot import name 'OpenAIGenerator'`

- [ ] **Step 3: Write minimal implementation**

Add to `olmlx/proxy_tuning_pipeline/expand.py`:

```python
from typing import Any

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
        resp = self._ensure_client().chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k openai_generator -v`
Expected: 2 passed

- [ ] **Step 5: Run ruff, then commit**

Run: `uv run ruff check olmlx/proxy_tuning_pipeline/ && uv run ruff format olmlx/proxy_tuning_pipeline/`
Expected: no errors

```bash
git add olmlx/proxy_tuning_pipeline/expand.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): OpenAIGenerator (GPT-5.4-mini)"
```

---

## Task 12: Curation — `quality_filter` + `dedupe_examples`

**Files:**
- Create: `olmlx/proxy_tuning_pipeline/curate.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
from olmlx.proxy_tuning_pipeline.curate import dedupe_examples, quality_filter


def _ex(user, assistant, kind="function", prov="a.py:1"):
    return ChatExample(kind=kind, provenance=prov, user=user, assistant=assistant)


def test_quality_filter_drops_too_short_and_too_long():
    good = _ex("How does generate_chat route output?", "It calls parse_model_output ...")
    too_short_q = _ex("hi", "a real-enough answer here please")
    too_short_a = _ex("A perfectly reasonable question?", "no")
    too_long = _ex("Q?", "x" * 100_000)
    out = quality_filter([good, too_short_q, too_short_a, too_long])
    assert out == [good]


def test_dedupe_examples_drops_normalized_duplicates():
    a = _ex("How  does   F work?", "Answer A")
    b = _ex("how does f work?", "answer a")  # same after normalization
    c = _ex("Different question?", "Different answer")
    out = dedupe_examples([a, b, c])
    assert out == [a, c]  # first kept, order preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k "quality_filter or dedupe_examples" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.proxy_tuning_pipeline.curate'`

- [ ] **Step 3: Write minimal implementation**

Create `olmlx/proxy_tuning_pipeline/curate.py`:

```python
"""Curation: quality filtering, dedup, and train/valid split."""

from __future__ import annotations

import random

from olmlx.proxy_tuning_pipeline.schema import ChatExample

_MIN_USER_CHARS = 8
_MIN_ASSISTANT_CHARS = 12
_MAX_ASSISTANT_CHARS = 20_000


def quality_filter(examples: list[ChatExample]) -> list[ChatExample]:
    """Drop pairs that are too short to teach or implausibly long (likely junk)."""
    out: list[ChatExample] = []
    for e in examples:
        if len(e.user.strip()) < _MIN_USER_CHARS:
            continue
        if not (_MIN_ASSISTANT_CHARS <= len(e.assistant.strip()) <= _MAX_ASSISTANT_CHARS):
            continue
        out.append(e)
    return out


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def dedupe_examples(examples: list[ChatExample]) -> list[ChatExample]:
    """Drop normalized-duplicate (user, assistant) pairs; order-preserving."""
    seen: set[tuple[str, str]] = set()
    out: list[ChatExample] = []
    for e in examples:
        key = (_norm(e.user), _norm(e.assistant))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k "quality_filter or dedupe_examples" -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/proxy_tuning_pipeline/curate.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): quality filter + example dedup"
```

---

## Task 13: Curation — `split_train_valid`

**Files:**
- Modify: `olmlx/proxy_tuning_pipeline/curate.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
from olmlx.proxy_tuning_pipeline.curate import split_train_valid


def test_split_train_valid_ratio_and_determinism():
    examples = [_ex(f"Q{i}?", f"answer number {i}") for i in range(100)]
    train1, valid1 = split_train_valid(examples, valid_frac=0.1, seed=42)
    assert len(valid1) == 10 and len(train1) == 90
    # No overlap; union is the whole set.
    assert {id(e) for e in train1}.isdisjoint({id(e) for e in valid1})
    assert len(train1) + len(valid1) == 100
    # Deterministic for a fixed seed.
    train2, valid2 = split_train_valid(examples, valid_frac=0.1, seed=42)
    assert [e.user for e in valid1] == [e.user for e in valid2]


def test_split_train_valid_guarantees_at_least_one_valid():
    examples = [_ex("Q?", "a real answer here")]
    train, valid = split_train_valid(examples, valid_frac=0.1, seed=0)
    assert len(valid) == 1 and len(train) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k split_train_valid -v`
Expected: FAIL with `ImportError: cannot import name 'split_train_valid'`

- [ ] **Step 3: Write minimal implementation**

Add to `olmlx/proxy_tuning_pipeline/curate.py`:

```python
def split_train_valid(
    examples: list[ChatExample],
    valid_frac: float,
    seed: int,
) -> tuple[list[ChatExample], list[ChatExample]]:
    """Shuffle deterministically and split off a validation fraction.

    Guarantees at least one validation example when the input is non-empty so
    ``mlx_lm.lora`` always has a ``valid.jsonl`` to evaluate against.
    """
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * valid_frac)) if shuffled else 0
    valid = shuffled[:n_valid]
    train = shuffled[n_valid:]
    return train, valid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k split_train_valid -v`
Expected: 2 passed

- [ ] **Step 5: Run ruff, then commit**

Run: `uv run ruff check olmlx/proxy_tuning_pipeline/ && uv run ruff format olmlx/proxy_tuning_pipeline/`
Expected: no errors

```bash
git add olmlx/proxy_tuning_pipeline/curate.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): deterministic train/valid split"
```

---

## Task 14: CLI wiring (`cli.py`) + end-to-end test

**Files:**
- Create: `olmlx/proxy_tuning_pipeline/cli.py`
- Test: `tests/test_proxy_tuning_pipeline.py`

**Context:** `run_pipeline` is the testable core (takes an injected `Generator`, writes artifacts, returns a stats dict). `main()` is the thin argv wrapper that constructs the real `OpenAIGenerator`. The end-to-end test drives `run_pipeline` with a `_FakeGenerator` over a fixture repo — no network. Stats include `generated` vs `kept` so truncation is never silent (per spec §5c).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_pipeline.py`:

```python
from olmlx.proxy_tuning_pipeline.cli import run_pipeline


def test_run_pipeline_end_to_end(tmp_path):
    # Minimal fixture repo.
    repo = tmp_path / "repo"
    (repo / "olmlx").mkdir(parents=True)
    (repo / "olmlx" / "m.py").write_text('def f():\n    """doc."""\n    return 1\n')
    (repo / "CLAUDE.md").write_text(
        "## Non-Obvious Invariants\n\n**Inv one** — a real rule about streams.\n"
    )
    out_dir = tmp_path / "out"

    gen = _FakeGenerator(
        '{"pairs": [{"instruction": "Explain this clearly?", '
        '"response": "A grounded, sufficiently long answer about olmlx."}]}'
    )
    stats = run_pipeline(
        repo_root=repo,
        out_dir=out_dir,
        generator=gen,
        n_per_unit=1,
        valid_frac=0.5,
        seed=7,
    )

    # Artifacts written in mlx-lm chat format.
    train = read_jsonl(out_dir / "train.jsonl")
    valid = read_jsonl(out_dir / "valid.jsonl")
    assert train and valid
    assert set(train[0].keys()) == {"messages"}
    roles = [m["role"] for m in train[0]["messages"]]
    assert roles == ["user", "assistant"]

    # Stats report generated-vs-kept (no silent truncation).
    assert stats["units"] >= 2  # function + invariant
    assert stats["generated"] >= 2
    assert stats["kept"] == stats["train"] + stats["valid"]
    assert stats["train"] == len(train) and stats["valid"] == len(valid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k run_pipeline -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.proxy_tuning_pipeline.cli'`

- [ ] **Step 3: Write minimal implementation**

Create `olmlx/proxy_tuning_pipeline/cli.py`:

```python
"""CLI + orchestration for the proxy-tuning data pipeline.

Usage:
    OPENAI_API_KEY=... uv run python -m olmlx.proxy_tuning_pipeline.cli \
        --repo . --out data/proxy_tuning --n-per-unit 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from olmlx.proxy_tuning_pipeline.curate import (
    dedupe_examples,
    quality_filter,
    split_train_valid,
)
from olmlx.proxy_tuning_pipeline.expand import (
    DEFAULT_MODEL,
    Generator,
    OpenAIGenerator,
    expand_units,
)
from olmlx.proxy_tuning_pipeline.extract import extract_repo
from olmlx.proxy_tuning_pipeline.schema import write_jsonl

logger = logging.getLogger(__name__)


def run_pipeline(
    repo_root: str | Path,
    out_dir: str | Path,
    generator: Generator,
    n_per_unit: int,
    valid_frac: float,
    seed: int,
) -> dict[str, Any]:
    """Extract -> expand -> curate -> write. Returns a stats dict."""
    out_dir = Path(out_dir)
    units = extract_repo(repo_root)
    logger.info("extracted %d units", len(units))

    examples = expand_units(units, generator, n_per_unit=n_per_unit)
    generated = len(examples)

    examples = dedupe_examples(quality_filter(examples))
    kept = len(examples)

    train, valid = split_train_valid(examples, valid_frac=valid_frac, seed=seed)
    write_jsonl(out_dir / "train.jsonl", (e.to_chat_row() for e in train))
    write_jsonl(out_dir / "valid.jsonl", (e.to_chat_row() for e in valid))

    stats = {
        "units": len(units),
        "generated": generated,
        "kept": kept,
        "dropped": generated - kept,
        "train": len(train),
        "valid": len(valid),
    }
    logger.info("pipeline stats: %s", stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build proxy-tuning SFT data from the olmlx repo.")
    ap.add_argument("--repo", default=".", help="repo root to mine")
    ap.add_argument("--out", default="data/proxy_tuning", help="output dir for train/valid jsonl")
    ap.add_argument("--n-per-unit", type=int, default=4, help="pairs requested per source unit")
    ap.add_argument("--valid-frac", type=float, default=0.08, help="validation fraction")
    ap.add_argument("--seed", type=int, default=0, help="split shuffle seed")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI generator model id")
    args = ap.parse_args(argv)

    stats = run_pipeline(
        repo_root=args.repo,
        out_dir=args.out,
        generator=OpenAIGenerator(model=args.model),
        n_per_unit=args.n_per_unit,
        valid_frac=args.valid_frac,
        seed=args.seed,
    )
    print(f"Done. {stats}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -k run_pipeline -v`
Expected: 1 passed

- [ ] **Step 5: Run the full pipeline test suite + ruff, then commit**

Run: `uv run pytest tests/test_proxy_tuning_pipeline.py -v && uv run ruff check olmlx/proxy_tuning_pipeline/ tests/test_proxy_tuning_pipeline.py && uv run ruff format olmlx/proxy_tuning_pipeline/ tests/test_proxy_tuning_pipeline.py`
Expected: all pass; ruff clean

```bash
git add olmlx/proxy_tuning_pipeline/cli.py tests/test_proxy_tuning_pipeline.py
git commit -m "feat(proxy-tuning-data): CLI wiring + end-to-end pipeline"
```

---

## Task 15: Real dry run on the olmlx repo (manual validation)

**Files:** none (operational validation — needs an `OPENAI_API_KEY`)

> This is the only step that calls the real GPT-5.4-mini. It validates the whole pipeline end-to-end on the actual repo at a tiny scale before the full ~10k run. **Cost is a few cents at this scale.**

- [ ] **Step 1: Tiny real run (a handful of units)**

Add a temporary `--limit-units` guard if you want, or just run with `--n-per-unit 1` and Ctrl-C after ~20 units. Simplest: run the extractor-only count first to confirm volume:

Run: `uv run python -c "from olmlx.proxy_tuning_pipeline.extract import extract_repo; print(len(extract_repo('.')))"`
Expected: a few thousand units (functions + tests dominate).

- [ ] **Step 2: Small real generation pass**

Run:
```bash
OPENAI_API_KEY=$OPENAI_API_KEY uv run python -m olmlx.proxy_tuning_pipeline.cli \
  --repo . --out /tmp/ptdata_smoke --n-per-unit 1 --valid-frac 0.1
```
(Interrupt after a minute if you only want a smoke sample, or let it complete for a real partial set.)
Expected: logs show `extracted N units` and a final `pipeline stats: {...}`; `/tmp/ptdata_smoke/train.jsonl` and `valid.jsonl` exist.

- [ ] **Step 3: Eyeball the output quality**

Run: `head -3 /tmp/ptdata_smoke/train.jsonl`
Expected: valid mlx-lm chat rows; responses are **grounded in real olmlx code/conventions**, not hallucinated APIs. If responses invent non-existent olmlx APIs, tighten the grounding language in `_SYSTEM_PROMPT` (expand.py) and re-run before the full pass.

- [ ] **Step 4: No commit** — validation only. If Step 3 surfaces a grounding/quality issue, fix `_SYSTEM_PROMPT` or filters with a failing test first, then re-run.

---

## Self-Review

**Spec coverage (against `2026-06-15-...-design.md` §5):**
- ✅ §5a mechanical extraction — all five sources: functions (Task 3), invariants (Task 4), docs (Task 5), tests (Task 6), commits (Task 7); aggregated + deduped (Task 8). Secret-stripping (Task 2). Provenance on every unit. Context clamped (`_MAX_SOURCE_CHARS` / `_MAX_USER_CHARS`).
- ✅ §5b GPT-5.4-mini expansion — `Generator` protocol + `OpenAIGenerator` (Tasks 9, 11); grounding discipline in `_SYSTEM_PROMPT`; JSON pairs parsed (Task 9); one call per unit (Task 10). OpenAI SDK confirmed already a dep.
- ✅ §5c curate — dedup (Task 12), quality/length filters (Task 12), train/valid split (Task 13), JSONL output in mlx-lm chat format (Task 1/14), generated-vs-kept stats logged (Task 14, no silent truncation).
- ✅ §5 testing — extractors unit-tested on fixtures; OpenAI mocked via injected client; curation unit-tested; end-to-end with `_FakeGenerator` (Task 14); real dry run (Task 15).
- ✅ Artifacts: `data/proxy_tuning/train.jsonl` + `valid.jsonl`, gitignored (Task 1). Target ~10k is reached by setting `--n-per-unit` against the unit count from Task 15 Step 1 (e.g. ~3–4k units × `--n-per-unit 3` → ~10k before curation).

**Placeholder scan:** No TBD/TODO. `DEFAULT_MODEL = "gpt-5.4-mini"` is a concrete configurable default (overridable via `--model`), not a placeholder. Every code step shows complete code.

**Type consistency:** `ExtractionUnit(kind, provenance, instruction_hint, source_context)` and `ChatExample(kind, provenance, user, assistant)` are used identically across Tasks 1–14. `Generator.generate(system, user) -> str`, `build_messages(unit, n_pairs) -> (system, user)`, `parse_pairs(text) -> list[tuple[str,str]]`, `expand_units(units, generator, n_per_unit)`, `run_pipeline(repo_root, out_dir, generator, n_per_unit, valid_frac, seed)` are consistent between definition and call sites (Task 14 `main` → `run_pipeline`; tests).
