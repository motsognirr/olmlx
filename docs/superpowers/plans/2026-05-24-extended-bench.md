# Extended speed + quality benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an extended speed + quality benchmark across all 23 configured olmlx models, producing a tiered report (Core on all, Extended on 13, Ablation on 2) with matplotlib charts and an analysis markdown.

**Architecture:** Reuses the existing `olmlx/bench/` worker + grader stack. Six new files: four loaders/builders in `olmlx/bench/`, two driver scripts under `scripts/`. Dataset fetching uses the HuggingFace `datasets` library; downloaded payloads are cached under `~/.olmlx/bench-cache/` so tests run offline against pre-populated cache fixtures.

**Tech Stack:** Python 3.11+, existing olmlx bench stack, HuggingFace `datasets` (new dev dep), `matplotlib` (new dev dep), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-05-24-extended-bench-design.md`

---

## File structure

| File | Responsibility |
|---|---|
| `olmlx/bench/extended_suites.py` | Dataset loaders + subset selection + RULER generator. Returns `list[BenchPrompt]` per suite. |
| `olmlx/bench/ifeval_grader.py` | Vendored IFEval verifiable-constraint checks; registers `"ifeval"` into `quality.GRADERS`. |
| `olmlx/bench/tier_table.py` | Static maps of model HF path → tier (`extended` / `core_only`) + ablation anchors. |
| `olmlx/bench/extended_runner.py` | Per-model orchestration: warmup triage, suite assembly, per-row JSON writer. |
| `scripts/run_extended_bench.py` | Thin CLI around the runner; argparse, model-list discovery, fan-out across models. |
| `scripts/build_extended_report.py` | Pure post-processor: reads `raw/`, renders matplotlib PNG charts + the README. |

Tests mirror the source tree under `tests/`.

---

## Task 1: Add dev deps + dataset cache scaffolding + subset helper

**Files:**
- Modify: `pyproject.toml` (dev deps + new optional extra `bench-extended`)
- Create: `olmlx/bench/extended_suites.py`
- Create: `tests/test_extended_suites_subset.py`

- [ ] **Step 1: Add dependencies**

Modify `pyproject.toml` `[dependency-groups].dev` block — append two entries (keep alphabetic order if already sorted, otherwise just append):

```toml
dev = [
    "datasets>=3.0",
    "httpx>=0.28.1",
    "matplotlib>=3.9",
    "pytest>=9.0.2",
    "pytest-asyncio>=0.24",
    "pytest-cov>=7.0.0",
    "pyright>=1.1.380",
    "ruff>=0.9",
]
```

Then run: `uv sync --no-editable`
Expected: succeeds, lockfile updated.

- [ ] **Step 2: Write failing tests for the subset helper and cache dir**

Create `tests/test_extended_suites_subset.py`:

```python
"""Tests for olmlx.bench.extended_suites: deterministic subset selection + cache dir."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmlx.bench.extended_suites import bench_cache_dir, select_subset


class TestSelectSubset:
    def test_returns_n_items_when_n_lt_len(self):
        items = list(range(100))
        out = select_subset(items, 10)
        assert len(out) == 10

    def test_returns_all_items_when_n_ge_len(self):
        items = list(range(5))
        out = select_subset(items, 10)
        assert out == items

    def test_is_deterministic(self):
        items = list(range(100))
        a = select_subset(items, 10)
        b = select_subset(items, 10)
        assert a == b

    def test_spreads_evenly(self):
        # 50 from 164: indices ~0, 3, 6, 10, 13, ... — first and last present.
        items = list(range(164))
        out = select_subset(items, 50)
        assert out[0] == 0
        # Last element of the selection should be near (but not necessarily
        # exactly) the last element of the source.
        assert out[-1] >= 160

    def test_stratified_balances_strata(self):
        # 12 items in 3 strata of 4 each; pick 6 stratified → 2 per stratum.
        items = [(i, f"stratum-{i % 3}") for i in range(12)]
        out = select_subset(items, 6, key=lambda x: x[1])
        counts: dict[str, int] = {}
        for _, s in out:
            counts[s] = counts.get(s, 0) + 1
        assert counts == {"stratum-0": 2, "stratum-1": 2, "stratum-2": 2}


class TestBenchCacheDir:
    def test_default_path(self):
        d = bench_cache_dir()
        assert d == Path("~/.olmlx/bench-cache").expanduser()

    def test_creates_if_missing(self, tmp_path, monkeypatch):
        target = tmp_path / "bc"
        monkeypatch.setenv("OLMLX_BENCH_CACHE_DIR", str(target))
        d = bench_cache_dir()
        assert d == target
        assert d.is_dir()
```

Run: `uv run pytest tests/test_extended_suites_subset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.bench.extended_suites'`.

- [ ] **Step 3: Implement the scaffolding**

Create `olmlx/bench/extended_suites.py`:

```python
"""Extended quality suites.

Loaders fetch canonical datasets (HumanEval+, MBPP+, GSM8K, MATH-500,
MMLU-Pro, GPQA-Diamond, IFEval) once, cache them under
``~/.olmlx/bench-cache/`` (or ``OLMLX_BENCH_CACHE_DIR`` if set), and return
``list[BenchPrompt]`` ready for the existing bench worker.

Subset selection is deterministic — re-runs on a different machine grade the
same prompts. The selected ID list is recorded by the runner alongside
results so a third-party reviewer can verify which prompts contributed to
each score.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def bench_cache_dir() -> Path:
    """Directory for downloaded benchmark datasets."""
    env = os.environ.get("OLMLX_BENCH_CACHE_DIR")
    base = Path(env).expanduser() if env else Path("~/.olmlx/bench-cache").expanduser()
    base.mkdir(parents=True, exist_ok=True)
    return base


def select_subset(
    items: Sequence[T],
    n: int,
    *,
    key: Callable[[T], str] | None = None,
) -> list[T]:
    """Deterministically pick ``n`` items, spread evenly across ``items``.

    With ``key`` set, balances counts across the strata returned by ``key``
    (e.g. MMLU-Pro category labels), splitting ``n`` as evenly as possible
    and applying the same even-spread rule within each stratum.

    Picks indices ``round(i * len(items) / n)`` for i in 0..n-1, so first
    and last items are always included.
    """
    if n >= len(items):
        return list(items)
    if key is None:
        return [items[round(i * len(items) / n)] for i in range(n)]

    # Stratified: bucket then pick floor/ceil per bucket so totals sum to n.
    buckets: dict[str, list[T]] = {}
    for item in items:
        buckets.setdefault(key(item), []).append(item)
    labels = sorted(buckets)
    base = n // len(labels)
    remainder = n - base * len(labels)
    out: list[T] = []
    # First ``remainder`` strata get one extra item (alphabetic by label, so
    # which strata get the bonus is deterministic).
    for i, label in enumerate(labels):
        take = base + (1 if i < remainder else 0)
        bucket = buckets[label]
        if take >= len(bucket):
            out.extend(bucket)
        else:
            out.extend(bucket[round(j * len(bucket) / take)] for j in range(take))
    return out
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/test_extended_suites_subset.py -v`
Expected: PASS (all 6 tests).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check olmlx/bench/extended_suites.py tests/test_extended_suites_subset.py
uv run ruff format olmlx/bench/extended_suites.py tests/test_extended_suites_subset.py
git add pyproject.toml uv.lock olmlx/bench/extended_suites.py tests/test_extended_suites_subset.py
git commit -m "feat(bench): add deterministic subset helper + cache dir for extended suites"
```

---

## Task 2: HumanEval+ and MBPP+ loaders

**Files:**
- Modify: `olmlx/bench/extended_suites.py`
- Create: `tests/test_extended_suites_coding.py`
- Create: `tests/fixtures/bench_cache/humanevalplus_sample.json` (small fixture so tests are offline)
- Create: `tests/fixtures/bench_cache/mbppplus_sample.json`

- [ ] **Step 1: Add fixture files**

Create `tests/fixtures/bench_cache/humanevalplus_sample.json`:

```json
[
  {
    "task_id": "HumanEval/0",
    "prompt": "def has_close_elements(numbers, threshold):\n    \"\"\"...\"\"\"\n",
    "entry_point": "has_close_elements",
    "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n"
  },
  {
    "task_id": "HumanEval/1",
    "prompt": "def separate_paren_groups(s):\n    \"\"\"...\"\"\"\n",
    "entry_point": "separate_paren_groups",
    "test": "def check(candidate):\n    assert candidate('( )') == ['()']\n"
  }
]
```

Create `tests/fixtures/bench_cache/mbppplus_sample.json`:

```json
[
  {
    "task_id": "Mbpp/2",
    "prompt": "Write a function that returns the n-th smallest item.",
    "code": "def kth_smallest(arr, k):\n    return sorted(arr)[k-1]\n",
    "test_list": ["assert kth_smallest([1,2,3], 1) == 1"]
  }
]
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_extended_suites_coding.py`:

```python
"""Tests for olmlx.bench.extended_suites: coding suite loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from olmlx.bench.extended_suites import load_humaneval_plus, load_mbpp_plus


@pytest.fixture
def coding_cache(tmp_path, monkeypatch):
    """Populate a fake bench cache from tests/fixtures/bench_cache/."""
    monkeypatch.setenv("OLMLX_BENCH_CACHE_DIR", str(tmp_path))
    fixtures = Path(__file__).parent / "fixtures" / "bench_cache"
    (tmp_path / "humanevalplus.json").write_bytes(
        (fixtures / "humanevalplus_sample.json").read_bytes()
    )
    (tmp_path / "mbppplus.json").write_bytes(
        (fixtures / "mbppplus_sample.json").read_bytes()
    )
    return tmp_path


class TestHumanEvalPlus:
    def test_loads_from_cache(self, coding_cache):
        prompts = load_humaneval_plus(n=None)
        assert len(prompts) == 2

    def test_subset_size(self, coding_cache):
        prompts = load_humaneval_plus(n=1)
        assert len(prompts) == 1

    def test_each_prompt_has_code_exec_grader(self, coding_cache):
        for p in load_humaneval_plus(n=None):
            assert p.grader == "code_exec"
            assert p.expected["entry_point"]
            assert "def check" in p.expected["tests"]

    def test_max_tokens_4096(self, coding_cache):
        for p in load_humaneval_plus(n=None):
            assert p.max_tokens == 4096

    def test_category(self, coding_cache):
        for p in load_humaneval_plus(n=None):
            assert p.category == "humaneval-plus"


class TestMbppPlus:
    def test_loads_from_cache(self, coding_cache):
        prompts = load_mbpp_plus(n=None)
        assert len(prompts) == 1

    def test_each_prompt_has_code_exec_grader(self, coding_cache):
        for p in load_mbpp_plus(n=None):
            assert p.grader == "code_exec"
            assert "def check" in p.expected["tests"]
```

Run: `uv run pytest tests/test_extended_suites_coding.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_humaneval_plus'`.

- [ ] **Step 3: Implement the loaders**

Append to `olmlx/bench/extended_suites.py`:

```python
import json

from olmlx.bench.prompts import BenchPrompt

_HUMANEVAL_PLUS_FILE = "humanevalplus.json"
_MBPP_PLUS_FILE = "mbppplus.json"


def _fetch_humaneval_plus_to(path: Path) -> None:
    """Download HumanEval+ to ``path`` if not cached.

    Defers the ``datasets`` import so cached runs don't pay the import cost.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("evalplus/humanevalplus", split="test")
    records = [
        {
            "task_id": r["task_id"],
            "prompt": r["prompt"],
            "entry_point": r["entry_point"],
            "test": r["test"],
        }
        for r in ds
    ]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_humaneval_plus(n: int | None = 50) -> list[BenchPrompt]:
    """Load HumanEval+ as bench prompts. ``n=None`` returns all 164."""
    cache_path = bench_cache_dir() / _HUMANEVAL_PLUS_FILE
    if not cache_path.exists():
        _fetch_humaneval_plus_to(cache_path)
    records = json.loads(cache_path.read_text(encoding="utf-8"))
    if n is not None:
        records = select_subset(records, n)
    out: list[BenchPrompt] = []
    for r in records:
        # Reuse the same user-facing instruction shape as the bundled
        # mini-suite so verbose models behave identically.
        content = (
            "Complete the following Python function. Respond with a single "
            "fenced ```python``` code block containing the full function "
            "definition (you may include the signature and docstring). Do "
            "not include explanations.\n\n"
            f"{r['prompt']}"
        )
        out.append(
            BenchPrompt(
                name=f"humaneval-plus-{r['task_id'].split('/')[-1]}",
                category="humaneval-plus",
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
                grader="code_exec",
                expected={
                    "prompt": r["prompt"],
                    "tests": r["test"],
                    "entry_point": r["entry_point"],
                },
            )
        )
    return out


def _fetch_mbpp_plus_to(path: Path) -> None:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("evalplus/mbppplus", split="test")
    records = []
    for r in ds:
        # MBPP+ rows hold a list of test assertions; wrap them in a check()
        # function so we can grade with the existing code_exec grader (which
        # expects ``def check(candidate)`` semantics like HumanEval).
        asserts = "\n    ".join(r["test_list"])
        # Heuristic: extract the function name from the first assert (e.g.
        # ``assert kth_smallest(...) == ...`` → ``kth_smallest``). MBPP+ has
        # this consistently as ``assert <name>(``.
        first = r["test_list"][0]
        entry = first.split("assert", 1)[1].strip().split("(", 1)[0].strip()
        check_src = (
            f"def check(candidate):\n"
            f"    {entry} = candidate\n"
            f"    {asserts}\n"
        )
        records.append(
            {
                "task_id": r["task_id"],
                "prompt": r["prompt"],
                "entry_point": entry,
                "test": check_src,
            }
        )
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_mbpp_plus(n: int | None = 50) -> list[BenchPrompt]:
    """Load MBPP+ as bench prompts. ``n=None`` returns all."""
    cache_path = bench_cache_dir() / _MBPP_PLUS_FILE
    if not cache_path.exists():
        _fetch_mbpp_plus_to(cache_path)
    records = json.loads(cache_path.read_text(encoding="utf-8"))
    if n is not None:
        records = select_subset(records, n)
    out: list[BenchPrompt] = []
    for r in records:
        content = (
            "Solve this Python task. Respond with a single fenced ```python``` "
            "code block containing the full function definition. Do not include "
            f"explanations.\n\n{r['prompt']}\n\n"
            f"Your function must be named ``{r['entry_point']}``."
        )
        out.append(
            BenchPrompt(
                name=f"mbpp-plus-{r['task_id'].split('/')[-1]}",
                category="mbpp-plus",
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
                grader="code_exec",
                expected={
                    "prompt": "",  # MBPP+ has no shared prompt prefix
                    "tests": r["test"],
                    "entry_point": r["entry_point"],
                },
            )
        )
    return out
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/test_extended_suites_coding.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check olmlx/bench/extended_suites.py tests/test_extended_suites_coding.py
uv run ruff format olmlx/bench/extended_suites.py tests/test_extended_suites_coding.py
git add olmlx/bench/extended_suites.py tests/test_extended_suites_coding.py tests/fixtures/
git commit -m "feat(bench): add HumanEval+ and MBPP+ loaders for extended suite"
```

---

## Task 3: GSM8K (full) and MATH-500 loaders

**Files:**
- Modify: `olmlx/bench/extended_suites.py`
- Create: `tests/test_extended_suites_math.py`
- Create: `tests/fixtures/bench_cache/gsm8k_sample.json`
- Create: `tests/fixtures/bench_cache/math500_sample.json`

- [ ] **Step 1: Add fixtures**

`tests/fixtures/bench_cache/gsm8k_sample.json`:

```json
[
  {"question": "Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4. She sells the rest at $2 each. Daily earnings?", "answer": "She eats 3 and bakes 4, so 16-7=9 left. 9*2=18.\n#### 18"},
  {"question": "What is 12 + 4?", "answer": "12+4=16.\n#### 16"},
  {"question": "A train goes 60 mph for 2 hours. Distance?", "answer": "60*2=120.\n#### 120"}
]
```

`tests/fixtures/bench_cache/math500_sample.json`:

```json
[
  {"problem": "Find $a$ such that $a^2 = 16$.", "answer": "4", "level": "Level 1"},
  {"problem": "Simplify $\\frac{6}{8}$.", "answer": "\\frac{3}{4}", "level": "Level 1"}
]
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_extended_suites_math.py`:

```python
"""Tests for olmlx.bench.extended_suites: math suite loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmlx.bench.extended_suites import load_gsm8k, load_math500


@pytest.fixture
def math_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OLMLX_BENCH_CACHE_DIR", str(tmp_path))
    fixtures = Path(__file__).parent / "fixtures" / "bench_cache"
    (tmp_path / "gsm8k.json").write_bytes(
        (fixtures / "gsm8k_sample.json").read_bytes()
    )
    (tmp_path / "math500.json").write_bytes(
        (fixtures / "math500_sample.json").read_bytes()
    )
    return tmp_path


class TestGsm8k:
    def test_loads_from_cache(self, math_cache):
        prompts = load_gsm8k(n=None)
        assert len(prompts) == 3

    def test_extracts_integer_answer(self, math_cache):
        prompts = load_gsm8k(n=None)
        assert prompts[0].expected["answer"] == 18
        assert prompts[1].expected["answer"] == 16

    def test_grader_is_numeric(self, math_cache):
        for p in load_gsm8k(n=None):
            assert p.grader == "numeric"

    def test_max_tokens_4096(self, math_cache):
        for p in load_gsm8k(n=None):
            assert p.max_tokens == 4096


class TestMath500:
    def test_loads_from_cache(self, math_cache):
        prompts = load_math500(n=None)
        assert len(prompts) == 2

    def test_grader_is_regex_match_for_boxed(self, math_cache):
        prompts = load_math500(n=None)
        # MATH uses boxed answers like \boxed{4}; grader should regex-extract.
        assert prompts[0].grader == "regex_match"
        # Answers are kept as strings since MATH includes fractions, sqrt, etc.
        assert prompts[0].expected["answer"] == "4"
```

Run: `uv run pytest tests/test_extended_suites_math.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_gsm8k'`.

- [ ] **Step 3: Implement loaders**

Append to `olmlx/bench/extended_suites.py`:

```python
import re

_GSM8K_FILE = "gsm8k.json"
_MATH500_FILE = "math500.json"
_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+)")


def _fetch_gsm8k_to(path: Path) -> None:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("openai/gsm8k", "main", split="test")
    records = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _parse_gsm8k_answer(answer_text: str) -> int:
    m = _GSM8K_ANSWER_RE.search(answer_text)
    if m is None:
        raise ValueError(f"no #### marker in: {answer_text!r}")
    return int(m.group(1).replace(",", ""))


def load_gsm8k(n: int | None = 70) -> list[BenchPrompt]:
    """Load GSM8K test split. ``n=None`` returns all 1319."""
    cache_path = bench_cache_dir() / _GSM8K_FILE
    if not cache_path.exists():
        _fetch_gsm8k_to(cache_path)
    records = json.loads(cache_path.read_text(encoding="utf-8"))
    if n is not None:
        # Stratify by answer-length bucket so the subset doesn't bias toward
        # short or long problems.
        def _bucket(r: dict) -> str:
            n_chars = len(r["answer"])
            if n_chars < 200:
                return "short"
            if n_chars < 500:
                return "medium"
            return "long"
        records = select_subset(records, n, key=_bucket)
    out: list[BenchPrompt] = []
    for i, r in enumerate(records):
        ans = _parse_gsm8k_answer(r["answer"])
        content = (
            f"{r['question']}\n\n"
            "Think step by step, then write the final answer on its own line "
            "as '#### <number>'."
        )
        out.append(
            BenchPrompt(
                name=f"gsm8k-{i:04d}",
                category="gsm8k",
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
                grader="numeric",
                expected={"answer": ans, "tol": 0.0},
            )
        )
    return out


def _fetch_math500_to(path: Path) -> None:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    records = [
        {
            "problem": r["problem"],
            "answer": r["answer"],
            "level": r.get("level", "Level ?"),
        }
        for r in ds
    ]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_math500(n: int | None = 50) -> list[BenchPrompt]:
    """Load MATH-500 (HuggingFaceH4 curated subset of MATH test split)."""
    cache_path = bench_cache_dir() / _MATH500_FILE
    if not cache_path.exists():
        _fetch_math500_to(cache_path)
    records = json.loads(cache_path.read_text(encoding="utf-8"))
    if n is not None:
        records = select_subset(records, n, key=lambda r: r.get("level", "?"))
    out: list[BenchPrompt] = []
    for i, r in enumerate(records):
        content = (
            f"{r['problem']}\n\n"
            "Solve step by step. End your answer with \\boxed{...}."
        )
        out.append(
            BenchPrompt(
                name=f"math500-{i:04d}",
                category=f"math500-{r.get('level', '?').lower().replace(' ', '-')}",
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
                grader="regex_match",
                expected={
                    "pattern": r"\\boxed\{([^{}]+)\}",
                    "group": 1,
                    "answer": r["answer"],
                },
            )
        )
    return out
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/test_extended_suites_math.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check olmlx/bench/extended_suites.py tests/test_extended_suites_math.py
uv run ruff format olmlx/bench/extended_suites.py tests/test_extended_suites_math.py
git add olmlx/bench/extended_suites.py tests/test_extended_suites_math.py tests/fixtures/
git commit -m "feat(bench): add GSM8K and MATH-500 loaders for extended suite"
```

---

## Task 4: MMLU-Pro, GPQA-Diamond, IFEval loaders + RULER generator

**Files:**
- Modify: `olmlx/bench/extended_suites.py`
- Create: `tests/test_extended_suites_knowledge.py`
- Create: `tests/fixtures/bench_cache/mmlu_pro_sample.json`
- Create: `tests/fixtures/bench_cache/gpqa_diamond_sample.json`
- Create: `tests/fixtures/bench_cache/ifeval_sample.json`

- [ ] **Step 1: Add fixtures**

`tests/fixtures/bench_cache/mmlu_pro_sample.json`:

```json
[
  {"question": "What is 2+2?", "options": ["3", "4", "5", "6"], "answer": "B", "category": "math"},
  {"question": "Capital of France?", "options": ["London", "Berlin", "Paris", "Madrid"], "answer": "C", "category": "geography"}
]
```

`tests/fixtures/bench_cache/gpqa_diamond_sample.json`:

```json
[
  {
    "question": "What is the spin of an electron?",
    "correct": "1/2",
    "incorrect": ["1", "0", "3/2"],
    "domain": "Physics"
  }
]
```

`tests/fixtures/bench_cache/ifeval_sample.json`:

```json
[
  {
    "key": 1,
    "prompt": "Write a one-sentence reply containing the word 'banana'.",
    "instruction_id_list": ["keywords:existence"],
    "kwargs": [{"keywords": ["banana"]}]
  }
]
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_extended_suites_knowledge.py`:

```python
"""Tests for MMLU-Pro, GPQA-Diamond, IFEval loaders + RULER generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmlx.bench.extended_suites import (
    load_gpqa_diamond,
    load_ifeval,
    load_mmlu_pro,
    make_ruler_niah,
)


@pytest.fixture
def knowledge_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OLMLX_BENCH_CACHE_DIR", str(tmp_path))
    fixtures = Path(__file__).parent / "fixtures" / "bench_cache"
    for name in ("mmlu_pro.json", "gpqa_diamond.json", "ifeval.json"):
        src = fixtures / name.replace(".json", "_sample.json")
        (tmp_path / name).write_bytes(src.read_bytes())
    return tmp_path


class TestMmluPro:
    def test_loads(self, knowledge_cache):
        prompts = load_mmlu_pro(n=None)
        assert len(prompts) == 2

    def test_regex_grader(self, knowledge_cache):
        for p in load_mmlu_pro(n=None):
            assert p.grader == "regex_match"
            assert p.expected["answer"] in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")

    def test_max_tokens_1024(self, knowledge_cache):
        for p in load_mmlu_pro(n=None):
            assert p.max_tokens == 1024


class TestGpqaDiamond:
    def test_loads(self, knowledge_cache):
        prompts = load_gpqa_diamond(n=None)
        assert len(prompts) == 1

    def test_answer_is_letter(self, knowledge_cache):
        for p in load_gpqa_diamond(n=None):
            assert p.expected["answer"] in ("A", "B", "C", "D")


class TestIfeval:
    def test_loads(self, knowledge_cache):
        prompts = load_ifeval(n=None)
        assert len(prompts) == 1

    def test_grader_is_ifeval(self, knowledge_cache):
        for p in load_ifeval(n=None):
            assert p.grader == "ifeval"
            assert p.expected["instruction_id_list"] == ["keywords:existence"]


class TestRulerNiah:
    def test_generates_n_prompts(self):
        prompts = make_ruler_niah(context_tokens=4096, n=5)
        assert len(prompts) == 5

    def test_grader_is_contains_with_needle(self):
        prompts = make_ruler_niah(context_tokens=4096, n=3)
        for p in prompts:
            assert p.grader == "contains"
            assert len(p.expected["substrings"]) == 1

    def test_is_deterministic(self):
        a = make_ruler_niah(context_tokens=4096, n=5)
        b = make_ruler_niah(context_tokens=4096, n=5)
        assert [p.name for p in a] == [p.name for p in b]
        assert [p.expected for p in a] == [p.expected for p in b]
```

Run: `uv run pytest tests/test_extended_suites_knowledge.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_mmlu_pro'`.

- [ ] **Step 3: Implement loaders**

Append to `olmlx/bench/extended_suites.py`:

```python
import random
import string

_MMLU_PRO_FILE = "mmlu_pro.json"
_GPQA_FILE = "gpqa_diamond.json"
_IFEVAL_FILE = "ifeval.json"


def _fetch_mmlu_pro_to(path: Path) -> None:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    records = []
    for r in ds:
        # MMLU-Pro answers come as the option index (0-based); convert to
        # letter for the grader. Some categories have 10 options (A-J).
        letter = string.ascii_uppercase[r["answer_index"]]
        records.append(
            {
                "question": r["question"],
                "options": r["options"],
                "answer": letter,
                "category": r["category"],
            }
        )
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_mmlu_pro(n: int | None = 50) -> list[BenchPrompt]:
    cache_path = bench_cache_dir() / _MMLU_PRO_FILE
    if not cache_path.exists():
        _fetch_mmlu_pro_to(cache_path)
    records = json.loads(cache_path.read_text(encoding="utf-8"))
    if n is not None:
        records = select_subset(records, n, key=lambda r: r["category"])
    out: list[BenchPrompt] = []
    for i, r in enumerate(records):
        letters = string.ascii_uppercase[: len(r["options"])]
        option_lines = "\n".join(
            f"{letters[j]}) {opt}" for j, opt in enumerate(r["options"])
        )
        content = (
            f"Question: {r['question']}\n{option_lines}\n\n"
            "Respond with a single line at the end: 'Answer: <letter>'."
        )
        out.append(
            BenchPrompt(
                name=f"mmlu-pro-{i:04d}",
                category=f"mmlu-pro-{r['category']}",
                messages=[{"role": "user", "content": content}],
                max_tokens=1024,
                grader="regex_match",
                expected={
                    "pattern": r"(?i)answer[:\s]*([A-J])",
                    "group": 1,
                    "answer": r["answer"],
                },
            )
        )
    return out


def _fetch_gpqa_diamond_to(path: Path) -> None:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset(
        "Idavidrein/gpqa", "gpqa_diamond", split="train"
    )  # GPQA only ships a "train" split
    records = []
    for r in ds:
        records.append(
            {
                "question": r["Question"],
                "correct": r["Correct Answer"],
                "incorrect": [
                    r["Incorrect Answer 1"],
                    r["Incorrect Answer 2"],
                    r["Incorrect Answer 3"],
                ],
                "domain": r.get("High-level domain", "?"),
            }
        )
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_gpqa_diamond(n: int | None = 60) -> list[BenchPrompt]:
    cache_path = bench_cache_dir() / _GPQA_FILE
    if not cache_path.exists():
        _fetch_gpqa_diamond_to(cache_path)
    records = json.loads(cache_path.read_text(encoding="utf-8"))
    if n is not None:
        records = select_subset(records, n, key=lambda r: r.get("domain", "?"))
    rng = random.Random(42)  # deterministic answer-position shuffling
    out: list[BenchPrompt] = []
    for i, r in enumerate(records):
        options = [r["correct"], *r["incorrect"]]
        # Shuffle so the correct answer isn't always option A.
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [options[j] for j in order]
        correct_idx = order.index(0)
        letter = "ABCD"[correct_idx]
        option_lines = "\n".join(
            f"{'ABCD'[j]}) {opt}" for j, opt in enumerate(shuffled)
        )
        content = (
            f"Question: {r['question']}\n{option_lines}\n\n"
            "Respond with a single line at the end: 'Answer: <letter>'."
        )
        out.append(
            BenchPrompt(
                name=f"gpqa-{i:04d}",
                category=f"gpqa-{r.get('domain', '?').lower().replace(' ', '-')}",
                messages=[{"role": "user", "content": content}],
                max_tokens=1024,
                grader="regex_match",
                expected={
                    "pattern": r"(?i)answer[:\s]*([A-D])",
                    "group": 1,
                    "answer": letter,
                },
            )
        )
    return out


def _fetch_ifeval_to(path: Path) -> None:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("google/IFEval", split="train")
    records = []
    for r in ds:
        records.append(
            {
                "key": r["key"],
                "prompt": r["prompt"],
                "instruction_id_list": r["instruction_id_list"],
                "kwargs": r["kwargs"],
            }
        )
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


# IFEval constraint families we vendor; rubric-graded ones are excluded.
_VERIFIABLE_IFEVAL_PREFIXES = (
    "keywords:",
    "language:",
    "length_constraints:",
    "detectable_format:",
    "detectable_content:",
    "punctuation:",
    "startend:",
    "change_case:",
    "combination:",
)


def _is_verifiable(instruction_ids: list[str]) -> bool:
    return all(
        any(iid.startswith(p) for p in _VERIFIABLE_IFEVAL_PREFIXES)
        for iid in instruction_ids
    )


def load_ifeval(n: int | None = 50) -> list[BenchPrompt]:
    cache_path = bench_cache_dir() / _IFEVAL_FILE
    if not cache_path.exists():
        _fetch_ifeval_to(cache_path)
    records = json.loads(cache_path.read_text(encoding="utf-8"))
    # Filter to verifiable-constraint subset only.
    records = [r for r in records if _is_verifiable(r["instruction_id_list"])]
    if n is not None:
        # Stratify by the first instruction-id family.
        def _family(r: dict) -> str:
            return r["instruction_id_list"][0].split(":")[0]
        records = select_subset(records, n, key=_family)
    out: list[BenchPrompt] = []
    for r in records:
        out.append(
            BenchPrompt(
                name=f"ifeval-{r['key']:04d}",
                category="ifeval-" + r["instruction_id_list"][0].split(":")[0],
                messages=[{"role": "user", "content": r["prompt"]}],
                max_tokens=1024,
                grader="ifeval",
                expected={
                    "instruction_id_list": r["instruction_id_list"],
                    "kwargs": r["kwargs"],
                },
            )
        )
    return out


_RULER_CONTEXT_PARAGRAPH = (
    "The migratory patterns of Arctic terns span the entire globe. "
    "Researchers tag individuals to study route choice and longevity. "
)


def make_ruler_niah(
    context_tokens: int, n: int = 10, seed: int = 42
) -> list[BenchPrompt]:
    """Generate single-needle-in-haystack prompts at a given context size.

    Each prompt embeds a unique 6-digit "key" inside a long block of filler
    text and asks the model to retrieve it. ``context_tokens`` is approximate
    (4 chars ≈ 1 token under typical BPE tokenizers).
    """
    rng = random.Random(seed)
    target_chars = context_tokens * 4
    filler = (_RULER_CONTEXT_PARAGRAPH * (target_chars // len(_RULER_CONTEXT_PARAGRAPH) + 1))[
        :target_chars
    ]
    out: list[BenchPrompt] = []
    for i in range(n):
        key = f"{rng.randint(100000, 999999)}"
        # Place the needle at a deterministic position spread across the haystack.
        pos = round(len(filler) * (i + 1) / (n + 1))
        needle = f"The magic key is {key}. Remember it. "
        haystack = filler[:pos] + needle + filler[pos:][: target_chars - pos - len(needle)]
        content = (
            f"{haystack}\n\n"
            "What is the magic key mentioned earlier? Respond with just the "
            "6-digit number."
        )
        out.append(
            BenchPrompt(
                name=f"ruler-niah-{context_tokens}-{i:03d}",
                category=f"ruler-niah-{context_tokens}",
                messages=[{"role": "user", "content": content}],
                max_tokens=1024,
                grader="contains",
                expected={"substrings": [key], "all": True, "ignore_case": False},
            )
        )
    return out
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/test_extended_suites_knowledge.py -v`
Expected: PASS (12 tests).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check olmlx/bench/extended_suites.py tests/test_extended_suites_knowledge.py
uv run ruff format olmlx/bench/extended_suites.py tests/test_extended_suites_knowledge.py
git add olmlx/bench/extended_suites.py tests/test_extended_suites_knowledge.py tests/fixtures/
git commit -m "feat(bench): add MMLU-Pro, GPQA-Diamond, IFEval loaders + RULER NIAH generator"
```

---

## Task 5: IFEval grader

**Files:**
- Create: `olmlx/bench/ifeval_grader.py`
- Modify: `olmlx/bench/quality.py` (register the new grader)
- Create: `tests/test_ifeval_grader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ifeval_grader.py`:

```python
"""Tests for the vendored IFEval verifiable-constraint grader."""

from __future__ import annotations

from olmlx.bench.quality import grade


class TestKeywordsExistence:
    def test_pass(self):
        result = grade(
            "ifeval",
            "I had a banana for breakfast.",
            {"instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["banana"]}]},
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_fail(self):
        result = grade(
            "ifeval",
            "I had an apple for breakfast.",
            {"instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["banana"]}]},
        )
        assert result.passed is False


class TestLengthConstraints:
    def test_min_words_pass(self):
        result = grade(
            "ifeval",
            "one two three four five",
            {
                "instruction_id_list": ["length_constraints:number_words"],
                "kwargs": [{"num_words": 5, "relation": "at least"}],
            },
        )
        assert result.passed is True

    def test_max_words_fail(self):
        result = grade(
            "ifeval",
            "one two three four five six seven",
            {
                "instruction_id_list": ["length_constraints:number_words"],
                "kwargs": [{"num_words": 5, "relation": "at most"}],
            },
        )
        assert result.passed is False


class TestPunctuation:
    def test_no_commas_pass(self):
        result = grade(
            "ifeval",
            "Hello world",
            {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
        )
        assert result.passed is True

    def test_no_commas_fail(self):
        result = grade(
            "ifeval",
            "Hello, world",
            {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
        )
        assert result.passed is False


class TestStartEnd:
    def test_end_checker_pass(self):
        result = grade(
            "ifeval",
            "My reply ends here. END",
            {
                "instruction_id_list": ["startend:end_checker"],
                "kwargs": [{"end_phrase": "END"}],
            },
        )
        assert result.passed is True


class TestMultipleConstraints:
    def test_all_must_pass(self):
        result = grade(
            "ifeval",
            "Hello banana world",
            {
                "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
                "kwargs": [{"keywords": ["banana"]}, {}],
            },
        )
        assert result.passed is True

    def test_one_fail_means_overall_fail(self):
        result = grade(
            "ifeval",
            "Hello, banana world",
            {
                "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
                "kwargs": [{"keywords": ["banana"]}, {}],
            },
        )
        assert result.passed is False
        assert result.score == 0.5


class TestUnknownConstraint:
    def test_returns_ungraded(self):
        result = grade(
            "ifeval",
            "anything",
            {"instruction_id_list": ["totally:unknown"], "kwargs": [{}]},
        )
        # Unknown constraints leave the prompt ungraded (passed=None) so they
        # don't silently inflate failure counts.
        assert result.passed is None
```

Run: `uv run pytest tests/test_ifeval_grader.py -v`
Expected: FAIL with `unknown grader 'ifeval'`.

- [ ] **Step 2: Implement the grader**

Create `olmlx/bench/ifeval_grader.py`:

```python
"""Vendored IFEval verifiable-constraint checks.

Subset of the checks from the IFEval reference implementation
(https://github.com/google-research/google-research/tree/master/instruction_following_eval,
Apache-2.0). We only vendor the prefixes listed in
``_VERIFIABLE_IFEVAL_PREFIXES`` in extended_suites.py — those that admit a
purely-rule-based pass/fail decision. Rubric-graded constraints are not
supported and any unknown constraint returns ``passed=None``.

The grader returns a fractional ``score`` (fraction of constraints in the
list that passed) plus a boolean ``passed`` requiring all constraints to
pass. Detail field lists per-constraint outcomes for debugging.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from olmlx.bench.quality import QualityResult

ConstraintCheck = Callable[[str, dict[str, Any]], bool]


def _keywords_existence(output: str, kwargs: dict[str, Any]) -> bool:
    keywords = kwargs.get("keywords", [])
    out_low = output.casefold()
    return all(k.casefold() in out_low for k in keywords)


def _keywords_forbidden_words(output: str, kwargs: dict[str, Any]) -> bool:
    forbidden = kwargs.get("forbidden_words", [])
    out_low = output.casefold()
    return all(k.casefold() not in out_low for k in forbidden)


def _keywords_frequency(output: str, kwargs: dict[str, Any]) -> bool:
    keyword = kwargs.get("keyword", "")
    relation = kwargs.get("relation", "at least")
    target = int(kwargs.get("frequency", 0))
    actual = len(re.findall(re.escape(keyword), output, flags=re.IGNORECASE))
    return _compare(actual, target, relation)


def _length_constraints_number_words(output: str, kwargs: dict[str, Any]) -> bool:
    n_words = len(re.findall(r"\b\w+\b", output))
    return _compare(n_words, int(kwargs["num_words"]), kwargs.get("relation", "at least"))


def _length_constraints_number_sentences(output: str, kwargs: dict[str, Any]) -> bool:
    sentences = [s for s in re.split(r"[.!?]+", output) if s.strip()]
    return _compare(
        len(sentences), int(kwargs["num_sentences"]), kwargs.get("relation", "at least")
    )


def _length_constraints_nth_paragraph_first_word(
    output: str, kwargs: dict[str, Any]
) -> bool:
    paras = [p for p in output.split("\n\n") if p.strip()]
    n = int(kwargs.get("nth_paragraph", 1))
    if n < 1 or n > len(paras):
        return False
    first = paras[n - 1].strip().split(None, 1)[0].strip(".,;:!?\"'")
    return first.casefold() == str(kwargs.get("first_word", "")).casefold()


def _length_constraints_number_paragraphs(
    output: str, kwargs: dict[str, Any]
) -> bool:
    paras = [p for p in output.split("\n\n") if p.strip()]
    return _compare(
        len(paras), int(kwargs["num_paragraphs"]), kwargs.get("relation", "at least")
    )


def _punctuation_no_comma(output: str, kwargs: dict[str, Any]) -> bool:
    return "," not in output


def _detectable_content_postscript(output: str, kwargs: dict[str, Any]) -> bool:
    marker = kwargs.get("postscript_marker", "P.S.")
    return marker in output


def _detectable_content_number_placeholders(
    output: str, kwargs: dict[str, Any]
) -> bool:
    target = int(kwargs.get("num_placeholders", 0))
    actual = len(re.findall(r"\[[^\[\]\n]+\]", output))
    return actual >= target


def _detectable_format_number_bullet_lists(
    output: str, kwargs: dict[str, Any]
) -> bool:
    target = int(kwargs.get("num_bullets", 0))
    actual = len(re.findall(r"(?m)^\s*[-*]\s+", output))
    return actual == target


def _detectable_format_json_format(output: str, kwargs: dict[str, Any]) -> bool:
    import json
    s = output.strip()
    if s.startswith("```"):
        # Strip code fence.
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def _detectable_format_title(output: str, kwargs: dict[str, Any]) -> bool:
    return bool(re.search(r"<<[^<>]+>>", output))


def _detectable_format_constrained_response(
    output: str, kwargs: dict[str, Any]
) -> bool:
    allowed = ["My answer is yes.", "My answer is no.", "My answer is maybe."]
    return any(opt in output for opt in allowed)


def _startend_end_checker(output: str, kwargs: dict[str, Any]) -> bool:
    end_phrase = kwargs.get("end_phrase", "")
    return output.rstrip().endswith(end_phrase)


def _startend_quotation(output: str, kwargs: dict[str, Any]) -> bool:
    s = output.strip()
    return s.startswith('"') and s.endswith('"')


def _change_case_english_capital(output: str, kwargs: dict[str, Any]) -> bool:
    letters = [c for c in output if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _change_case_english_lowercase(output: str, kwargs: dict[str, Any]) -> bool:
    letters = [c for c in output if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters)


def _change_case_capital_word_frequency(
    output: str, kwargs: dict[str, Any]
) -> bool:
    target = int(kwargs.get("capital_frequency", 0))
    relation = kwargs.get("capital_relation", "at least")
    actual = len(re.findall(r"\b[A-Z]{2,}\b", output))
    return _compare(actual, target, relation)


def _combination_two_responses(output: str, kwargs: dict[str, Any]) -> bool:
    return "******" in output


def _combination_repeat_prompt(output: str, kwargs: dict[str, Any]) -> bool:
    prompt_to_repeat = kwargs.get("prompt_to_repeat", "")
    return prompt_to_repeat in output


def _language_response_language(output: str, kwargs: dict[str, Any]) -> bool:
    # We don't ship a language detector; treat as ungraded by returning False
    # so an incorrect-by-construction reply is never accidentally marked True.
    # Plan-stage decision: rather than report a wrong pass on language
    # constraints, we let these consistently fail. Future work could vendor a
    # cheap detector (langid) if this constraint family becomes important.
    return False


def _compare(actual: int, target: int, relation: str) -> bool:
    if relation == "at least":
        return actual >= target
    if relation == "at most":
        return actual <= target
    if relation == "less than":
        return actual < target
    if relation == "more than":
        return actual > target
    if relation == "equal to":
        return actual == target
    return False


CONSTRAINT_CHECKS: dict[str, ConstraintCheck] = {
    "keywords:existence": _keywords_existence,
    "keywords:forbidden_words": _keywords_forbidden_words,
    "keywords:frequency": _keywords_frequency,
    "length_constraints:number_words": _length_constraints_number_words,
    "length_constraints:number_sentences": _length_constraints_number_sentences,
    "length_constraints:nth_paragraph_first_word": _length_constraints_nth_paragraph_first_word,
    "length_constraints:number_paragraphs": _length_constraints_number_paragraphs,
    "punctuation:no_comma": _punctuation_no_comma,
    "detectable_content:postscript": _detectable_content_postscript,
    "detectable_content:number_placeholders": _detectable_content_number_placeholders,
    "detectable_format:number_bullet_lists": _detectable_format_number_bullet_lists,
    "detectable_format:json_format": _detectable_format_json_format,
    "detectable_format:title": _detectable_format_title,
    "detectable_format:constrained_response": _detectable_format_constrained_response,
    "startend:end_checker": _startend_end_checker,
    "startend:quotation": _startend_quotation,
    "change_case:english_capital": _change_case_english_capital,
    "change_case:english_lowercase": _change_case_english_lowercase,
    "change_case:capital_word_frequency": _change_case_capital_word_frequency,
    "combination:two_responses": _combination_two_responses,
    "combination:repeat_prompt": _combination_repeat_prompt,
    "language:response_language": _language_response_language,
}


def grade_ifeval(output: str, expected: dict[str, Any]) -> QualityResult:
    ids = expected.get("instruction_id_list", [])
    kwargs_list = expected.get("kwargs", [])
    if not ids:
        return QualityResult(
            grader="ifeval",
            passed=None,
            score=None,
            detail="no instruction_id_list",
        )
    # If any constraint id is unknown, the whole prompt is ungraded (so a
    # vendor gap doesn't silently fail prompts that are actually fine).
    unknown = [iid for iid in ids if iid not in CONSTRAINT_CHECKS]
    if unknown:
        return QualityResult(
            grader="ifeval",
            passed=None,
            score=None,
            detail=f"unknown constraint(s): {','.join(unknown)}",
        )
    outcomes: list[tuple[str, bool]] = []
    for iid, kw in zip(ids, kwargs_list, strict=True):
        check = CONSTRAINT_CHECKS[iid]
        outcomes.append((iid, bool(check(output, kw))))
    n_passed = sum(1 for _, ok in outcomes if ok)
    total = len(outcomes)
    detail = "; ".join(f"{iid}={'ok' if ok else 'fail'}" for iid, ok in outcomes)
    return QualityResult(
        grader="ifeval",
        passed=(n_passed == total),
        score=n_passed / total,
        detail=detail,
    )
```

- [ ] **Step 3: Register the grader**

Modify `olmlx/bench/quality.py` — the existing `GRADERS` dict at the bottom of the file. Append a registration:

```python
# Add at the very bottom of the file, after GRADERS is defined:
from olmlx.bench.ifeval_grader import grade_ifeval  # noqa: E402

GRADERS["ifeval"] = grade_ifeval
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/test_ifeval_grader.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check olmlx/bench/ifeval_grader.py olmlx/bench/quality.py tests/test_ifeval_grader.py
uv run ruff format olmlx/bench/ifeval_grader.py olmlx/bench/quality.py tests/test_ifeval_grader.py
git add olmlx/bench/ifeval_grader.py olmlx/bench/quality.py tests/test_ifeval_grader.py
git commit -m "feat(bench): add vendored IFEval verifiable-constraint grader"
```

---

## Task 6: Model tier table

**Files:**
- Create: `olmlx/bench/tier_table.py`
- Create: `tests/test_tier_table.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tier_table.py`:

```python
"""Tests for the model tier table."""

from __future__ import annotations

from olmlx.bench.tier_table import (
    ABLATION_ANCHORS,
    CORE_ONLY,
    EXTENDED,
    Tier,
    tier_for,
)


class TestTierAssignment:
    def test_total_count_is_23(self):
        assert len(EXTENDED) + len(CORE_ONLY) == 23

    def test_no_overlap(self):
        assert EXTENDED.isdisjoint(CORE_ONLY)

    def test_extended_count_is_13(self):
        assert len(EXTENDED) == 13

    def test_core_only_count_is_10(self):
        assert len(CORE_ONLY) == 10

    def test_known_extended_member(self):
        assert "mlx-community/Qwen3-Coder-Next-4bit" in EXTENDED

    def test_known_core_only_member(self):
        assert "mlx-community/gpt-oss-120b-MXFP4-Q4" in CORE_ONLY

    def test_pure_drafts_are_core_only(self):
        for hf in (
            "mlx-community/Qwen3-0.6B-4bit",
            "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        ):
            assert hf in CORE_ONLY


class TestTierFor:
    def test_extended(self):
        assert tier_for("mlx-community/Qwen3-Coder-Next-4bit") == Tier.EXTENDED

    def test_core_only(self):
        assert tier_for("mlx-community/gpt-oss-120b-MXFP4-Q4") == Tier.CORE_ONLY

    def test_unknown(self):
        assert tier_for("some/unknown-model") is None


class TestAblationAnchors:
    def test_has_turboquant_anchor(self):
        assert "mlx-community/Qwen3-Coder-Next-4bit" in ABLATION_ANCHORS["turboquant"]

    def test_has_speculative_anchor(self):
        assert "mlx-community/Qwen3.6-35B-A3B-4bit" in ABLATION_ANCHORS["speculative"]
```

Run: `uv run pytest tests/test_tier_table.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: Implement the table**

Create `olmlx/bench/tier_table.py`:

```python
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
```

- [ ] **Step 3: Verify tests pass**

Run: `uv run pytest tests/test_tier_table.py -v`
Expected: PASS (11 tests).

- [ ] **Step 4: Lint + commit**

```bash
uv run ruff check olmlx/bench/tier_table.py tests/test_tier_table.py
uv run ruff format olmlx/bench/tier_table.py tests/test_tier_table.py
git add olmlx/bench/tier_table.py tests/test_tier_table.py
git commit -m "feat(bench): add extended-bench tier table for 23 configured models"
```

---

## Task 7: Extended runner with triage rule

**Files:**
- Create: `olmlx/bench/extended_runner.py`
- Create: `tests/test_extended_runner.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_extended_runner.py`:

```python
"""Tests for olmlx.bench.extended_runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmlx.bench.extended_runner import (
    SuiteAssignment,
    apply_runtime_triage,
    assemble_core_suite,
    assemble_extended_suite,
    composite_score,
)


@pytest.fixture
def cached_datasets(tmp_path, monkeypatch):
    """Pre-populate cache so the loaders are offline."""
    monkeypatch.setenv("OLMLX_BENCH_CACHE_DIR", str(tmp_path))
    fixtures = Path(__file__).parent / "fixtures" / "bench_cache"
    mapping = {
        "humanevalplus.json": "humanevalplus_sample.json",
        "mbppplus.json": "mbppplus_sample.json",
        "gsm8k.json": "gsm8k_sample.json",
        "math500.json": "math500_sample.json",
        "mmlu_pro.json": "mmlu_pro_sample.json",
        "gpqa_diamond.json": "gpqa_diamond_sample.json",
        "ifeval.json": "ifeval_sample.json",
    }
    for dest, src in mapping.items():
        (tmp_path / dest).write_bytes((fixtures / src).read_bytes())
    return tmp_path


class TestAssembleSuites:
    def test_core_has_three_suites(self, cached_datasets):
        suite = assemble_core_suite()
        # Fixtures are tiny; we just check structure.
        categories = {p.category for p in suite}
        assert "humaneval-plus" in categories
        assert "gsm8k" in categories
        # GPQA category is suffixed with domain ("gpqa-physics").
        assert any(c.startswith("gpqa") for c in categories)

    def test_extended_has_six_more_suites(self, cached_datasets):
        suite = assemble_extended_suite()
        categories = {p.category for p in suite}
        assert "mbpp-plus" in categories
        assert any(c.startswith("math500") for c in categories)
        assert any(c.startswith("mmlu-pro") for c in categories)
        assert any(c.startswith("ifeval") for c in categories)
        assert any(c.startswith("ruler-niah") for c in categories)


class TestRuntimeTriage:
    def test_keep_core_if_budget_sufficient(self):
        # 100 tok/s × 3600s = 360k tokens >> core requirement (180 × 500 = 90k).
        decision = apply_runtime_triage(observed_tok_per_s=100, remaining_seconds=3600)
        assert decision == SuiteAssignment.FULL_CORE

    def test_drop_gpqa_when_tight(self):
        # 5 tok/s × 1200s = 6k tokens — far below 90k. Must drop the heaviest.
        decision = apply_runtime_triage(observed_tok_per_s=5, remaining_seconds=1200)
        assert decision == SuiteAssignment.CORE_MINUS_GPQA

    def test_he_plus_only_when_critical(self):
        decision = apply_runtime_triage(observed_tok_per_s=1, remaining_seconds=600)
        assert decision == SuiteAssignment.HE_PLUS_ONLY


class TestCompositeScore:
    def test_unweighted_mean_of_suite_pass_rates(self):
        # Three suites with pass rates 1.0, 0.5, 0.0 → composite 0.5.
        per_suite = {"humaneval-plus": 1.0, "gsm8k": 0.5, "gpqa": 0.0}
        assert composite_score(per_suite) == pytest.approx(0.5)

    def test_handles_empty(self):
        assert composite_score({}) == 0.0
```

Run: `uv run pytest tests/test_extended_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: Implement the runner**

Create `olmlx/bench/extended_runner.py`:

```python
"""Per-model orchestration for the extended benchmark.

Drives a single olmlx serve process per model (cold load → warmup → core →
optionally extended → optionally ablation → unload), with a runtime triage
rule that shrinks the suite for models too slow to finish Core within the
remaining wall-clock budget. The HTTP requests reuse the existing bench
worker pattern (subprocess-per-prompt against ``http://localhost:11434``).

Per-row JSON is written to ``<output_dir>/raw/<safe-model>.json`` with the
full per-prompt grading detail so the report builder can re-render without
touching the model again.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from olmlx.bench.extended_suites import (
    load_gpqa_diamond,
    load_gsm8k,
    load_humaneval_plus,
    load_ifeval,
    load_math500,
    load_mbpp_plus,
    load_mmlu_pro,
    make_ruler_niah,
)
from olmlx.bench.prompts import BenchPrompt

logger = logging.getLogger(__name__)


CORE_HUMANEVAL_PLUS = 50
CORE_GSM8K = 70
CORE_GPQA = 60

EXT_HUMANEVAL_PLUS = 164  # full set
EXT_MBPP_PLUS = 50
EXT_MATH500 = 50
EXT_MMLU_PRO = 50
EXT_IFEVAL = 50
EXT_RULER_4K = 10
EXT_RULER_8K = 10

# Approximate average output tokens per prompt; used by the triage rule's
# budget math. Picked from the May 2026 report's observation that reasoning
# prompts averaged 400-800 generated tokens.
_AVG_TOKENS_PER_PROMPT = 500


class SuiteAssignment(Enum):
    FULL_CORE = "full_core"
    CORE_MINUS_GPQA = "core_minus_gpqa"
    HE_PLUS_ONLY = "he_plus_only"


@dataclass
class PromptResult:
    name: str
    category: str
    suite: str
    passed: bool | None
    score: float | None
    detail: str
    output_text_clip: str  # first 500 chars only, to keep JSON small


@dataclass
class ModelRunResult:
    model: str
    tier: str
    assignment: SuiteAssignment
    warmup_tok_per_s: float
    per_suite_pass_rate: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0
    prompts: list[PromptResult] = field(default_factory=list)
    speed: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["assignment"] = self.assignment.value
        return d


def assemble_core_suite() -> list[BenchPrompt]:
    """Build the Core suite (~180 prompts, runs on all models)."""
    return [
        *load_humaneval_plus(n=CORE_HUMANEVAL_PLUS),
        *load_gsm8k(n=CORE_GSM8K),
        *load_gpqa_diamond(n=CORE_GPQA),
    ]


def assemble_extended_suite() -> list[BenchPrompt]:
    """Build the Extended suite (~250 prompts, runs on 13 user-facing models)."""
    return [
        *load_humaneval_plus(n=EXT_HUMANEVAL_PLUS),
        *load_mbpp_plus(n=EXT_MBPP_PLUS),
        *load_math500(n=EXT_MATH500),
        *load_mmlu_pro(n=EXT_MMLU_PRO),
        *load_ifeval(n=EXT_IFEVAL),
        *make_ruler_niah(context_tokens=4096, n=EXT_RULER_4K),
        *make_ruler_niah(context_tokens=8192, n=EXT_RULER_8K, seed=43),
    ]


def apply_runtime_triage(
    observed_tok_per_s: float, remaining_seconds: float
) -> SuiteAssignment:
    """Decide which slice of Core a slow model can finish.

    Budget math: each prompt averages ``_AVG_TOKENS_PER_PROMPT`` output
    tokens. Full core = 180 prompts; core-minus-GPQA = 120 prompts;
    HE+ only = 50 prompts.
    """
    budget_tokens = observed_tok_per_s * remaining_seconds
    full_need = 180 * _AVG_TOKENS_PER_PROMPT
    minus_gpqa_need = 120 * _AVG_TOKENS_PER_PROMPT
    if budget_tokens >= full_need:
        return SuiteAssignment.FULL_CORE
    if budget_tokens >= minus_gpqa_need:
        return SuiteAssignment.CORE_MINUS_GPQA
    return SuiteAssignment.HE_PLUS_ONLY


def composite_score(per_suite_pass_rate: dict[str, float]) -> float:
    """Unweighted mean of per-suite pass rates. Each suite contributes equally."""
    if not per_suite_pass_rate:
        return 0.0
    return sum(per_suite_pass_rate.values()) / len(per_suite_pass_rate)


_SUITE_FROM_CATEGORY = (
    ("humaneval-plus", "humaneval-plus"),
    ("mbpp-plus", "mbpp-plus"),
    ("gsm8k", "gsm8k"),
    ("math500", "math500"),
    ("mmlu-pro", "mmlu-pro"),
    ("gpqa", "gpqa"),
    ("ifeval", "ifeval"),
    ("ruler-niah", "ruler-niah"),
)


def suite_of(category: str) -> str:
    """Map a prompt category to the headline suite name."""
    for prefix, suite in _SUITE_FROM_CATEGORY:
        if category.startswith(prefix):
            return suite
    return category


def safe_model_name(hf_path: str) -> str:
    """Filesystem-safe model name; mirrors the goldens sanitizer."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", hf_path)


def aggregate_per_suite(results: list[PromptResult]) -> dict[str, float]:
    """Group prompts by suite, compute pass rate ignoring ungraded (passed=None)."""
    buckets: dict[str, list[bool]] = {}
    for r in results:
        if r.passed is None:
            continue
        buckets.setdefault(r.suite, []).append(r.passed)
    return {s: sum(passes) / len(passes) for s, passes in buckets.items() if passes}


def write_result(output_dir: Path, result: ModelRunResult) -> Path:
    """Write per-model JSON to ``<output_dir>/raw/<safe-name>.json``."""
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{safe_model_name(result.model)}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path
```

- [ ] **Step 3: Verify tests pass**

Run: `uv run pytest tests/test_extended_runner.py -v`
Expected: PASS (8 tests).

- [ ] **Step 4: Lint + commit**

```bash
uv run ruff check olmlx/bench/extended_runner.py tests/test_extended_runner.py
uv run ruff format olmlx/bench/extended_runner.py tests/test_extended_runner.py
git add olmlx/bench/extended_runner.py tests/test_extended_runner.py
git commit -m "feat(bench): add extended-runner suite assembly + triage + scoring"
```

---

## Task 8: CLI orchestrator script

**Files:**
- Create: `scripts/run_extended_bench.py`
- Modify: `olmlx/bench/extended_runner.py` (add the `run_model` driver fn that hits HTTP)
- Create: `tests/test_run_extended_bench_cli.py`

- [ ] **Step 1: Add the HTTP driver to extended_runner.py**

Append to `olmlx/bench/extended_runner.py`:

```python
import asyncio

import httpx

from olmlx.bench.quality import grade


async def _drive_prompt(
    client: httpx.AsyncClient,
    model: str,
    prompt: BenchPrompt,
    suite: str,
) -> PromptResult:
    """Issue one chat request to a running olmlx server, grade the output."""
    body = {
        "model": model,
        "messages": prompt.messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "top_p": 1.0,
            "num_predict": prompt.max_tokens,
        },
    }
    expected = dict(prompt.expected)
    if prompt.grader == "code_exec":
        expected["_enabled"] = True
    try:
        resp = await client.post("/api/chat", json=body, timeout=600.0)
        resp.raise_for_status()
        output = resp.json().get("message", {}).get("content", "")
    except (httpx.HTTPError, ValueError) as exc:
        return PromptResult(
            name=prompt.name,
            category=prompt.category,
            suite=suite,
            passed=False,
            score=0.0,
            detail=f"transport error: {exc!r}",
            output_text_clip="",
        )
    grade_result = grade(prompt.grader or "exact_match", output, expected)
    return PromptResult(
        name=prompt.name,
        category=prompt.category,
        suite=suite,
        passed=grade_result.passed,
        score=grade_result.score,
        detail=grade_result.detail,
        output_text_clip=output[:500],
    )


async def _warmup(
    client: httpx.AsyncClient, model: str
) -> float:
    """Issue a small warmup prompt to load the model and measure tok/s."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42, "num_predict": 32},
    }
    t0 = time.monotonic()
    resp = await client.post("/api/chat", json=body, timeout=900.0)
    elapsed = time.monotonic() - t0
    resp.raise_for_status()
    data = resp.json()
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0) or 1
    if eval_count > 0 and eval_duration_ns > 0:
        return eval_count / (eval_duration_ns / 1e9)
    return max(eval_count, 1) / max(elapsed, 0.001)


async def run_model(
    model: str,
    tier: str,
    base_url: str,
    output_dir: Path,
    remaining_seconds: float,
) -> ModelRunResult:
    """Cold-load a model, warm it up, run the assigned suites, write JSON."""
    t_start = time.monotonic()
    async with httpx.AsyncClient(base_url=base_url) as client:
        warmup_tps = await _warmup(client, model)
        assignment = apply_runtime_triage(warmup_tps, remaining_seconds)
        core = assemble_core_suite()
        if assignment == SuiteAssignment.HE_PLUS_ONLY:
            core = [p for p in core if p.category == "humaneval-plus"]
        elif assignment == SuiteAssignment.CORE_MINUS_GPQA:
            core = [p for p in core if not p.category.startswith("gpqa")]
        suite_prompts = list(core)
        if tier == "extended":
            suite_prompts.extend(assemble_extended_suite())
        results: list[PromptResult] = []
        for p in suite_prompts:
            result = await _drive_prompt(client, model, p, suite_of(p.category))
            results.append(result)
            # Persist every prompt as it lands, so a mid-run crash leaves
            # partial progress on disk rather than nothing.
            partial = ModelRunResult(
                model=model,
                tier=tier,
                assignment=assignment,
                warmup_tok_per_s=warmup_tps,
                prompts=results,
            )
            write_result(output_dir, partial)
    per_suite = aggregate_per_suite(results)
    final = ModelRunResult(
        model=model,
        tier=tier,
        assignment=assignment,
        warmup_tok_per_s=warmup_tps,
        per_suite_pass_rate=per_suite,
        composite=composite_score(per_suite),
        prompts=results,
        elapsed_seconds=time.monotonic() - t_start,
    )
    write_result(output_dir, final)
    return final


def run_model_sync(*args: Any, **kwargs: Any) -> ModelRunResult:
    return asyncio.run(run_model(*args, **kwargs))
```

- [ ] **Step 2: Write failing CLI test**

Create `tests/test_run_extended_bench_cli.py`:

```python
"""Smoke test the CLI orchestrator's argparse + tier-dispatch logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Import the script as a module — pytest will find it via the rootdir.
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import run_extended_bench  # type: ignore[import-not-found]


class TestArgParser:
    def test_default_output_dir(self):
        args = run_extended_bench.parse_args(["--models-config", "x.json"])
        assert args.output.name == "extended-2026-05"

    def test_only_flag(self):
        args = run_extended_bench.parse_args(
            ["--models-config", "x.json", "--only", "foo", "--only", "bar"]
        )
        assert args.only == ["foo", "bar"]


class TestSelectModels:
    def test_filters_to_only_list(self, tmp_path):
        config = {
            "alpha:latest": {"hf_path": "org/alpha"},
            "beta:latest": {"hf_path": "org/beta"},
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        selection = run_extended_bench.select_models(
            config_path, only=["org/alpha"]
        )
        assert selection == [("org/alpha", "alpha:latest")]

    def test_returns_all_when_only_empty(self, tmp_path):
        config = {
            "alpha:latest": {"hf_path": "org/alpha"},
            "beta:latest": "org/beta",  # short form
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        selection = run_extended_bench.select_models(config_path, only=[])
        hfs = {hf for hf, _ in selection}
        assert hfs == {"org/alpha", "org/beta"}
```

Run: `uv run pytest tests/test_run_extended_bench_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_extended_bench'`.

- [ ] **Step 3: Implement the CLI**

Create `scripts/run_extended_bench.py`:

```python
"""Orchestrate the extended benchmark across all configured models.

Per model: launches ``olmlx serve`` if not already running, drives the
assigned Core/Extended suites via the existing HTTP API, writes per-row
JSON to ``<output_dir>/raw/<safe-name>.json``. Server lifecycle is shared
across models (one serve process is reused; each model loads on first
request and unloads via keep-alive expiry).

This is a thin CLI — most logic lives in olmlx.bench.extended_runner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from olmlx.bench.extended_runner import run_model
from olmlx.bench.tier_table import Tier, tier_for

logger = logging.getLogger("run_extended_bench")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the extended benchmark.")
    parser.add_argument(
        "--models-config",
        type=Path,
        default=Path("~/.olmlx/models.json").expanduser(),
        help="Path to the models.json config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/extended-2026-05"),
        help="Output directory for raw/ + README.md",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="HF path of a single model to run (repeatable). Default: all.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Base URL of a running olmlx serve",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=48 * 3600,
        help="Total wall-clock budget; per-model remaining is computed from this",
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Launch `olmlx serve` as a subprocess and kill it on exit",
    )
    parser.add_argument(
        "--enable-code-exec",
        action="store_true",
        help="Required to grade HumanEval+/MBPP+ (sandboxed code execution)",
    )
    return parser.parse_args(argv)


def select_models(
    config_path: Path, only: list[str]
) -> list[tuple[str, str]]:
    """Return [(hf_path, alias)] pairs from the models.json config."""
    config: dict[str, Any] = json.loads(config_path.read_text())
    pairs: list[tuple[str, str]] = []
    for alias, entry in config.items():
        if isinstance(entry, str):
            hf = entry
        else:
            hf = entry.get("hf_path", "")
        if not hf:
            continue
        if only and hf not in only:
            continue
        pairs.append((hf, alias))
    return pairs


def _maybe_start_server(start: bool) -> subprocess.Popen | None:
    if not start:
        return None
    logger.info("launching olmlx serve")
    proc = subprocess.Popen(
        ["uv", "run", "olmlx", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    # Give the server a chance to bind before we start hitting it.
    time.sleep(8)
    return proc


async def _run_all(
    pairs: list[tuple[str, str]],
    base_url: str,
    output: Path,
    total_budget_s: float,
) -> None:
    start = time.monotonic()
    for hf, _alias in pairs:
        tier = tier_for(hf)
        if tier is None:
            logger.warning("skipping unknown model %s", hf)
            continue
        elapsed = time.monotonic() - start
        remaining = max(60.0, total_budget_s - elapsed)
        logger.info(
            "running %s tier=%s remaining=%.0fs", hf, tier.value, remaining
        )
        try:
            await run_model(
                model=hf,
                tier=tier.value,
                base_url=base_url,
                output_dir=output,
                remaining_seconds=remaining,
            )
        except Exception as exc:  # don't let one model's failure abort the run
            logger.exception("model %s failed: %s", hf, exc)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    if args.enable_code_exec:
        os.environ["OLMLX_BENCH_CODE_EXEC"] = "1"
    pairs = select_models(args.models_config, args.only)
    if not pairs:
        logger.error("no models selected")
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    server = _maybe_start_server(args.start_server)
    try:
        asyncio.run(_run_all(pairs, args.base_url, args.output, args.budget_seconds))
    finally:
        if server is not None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/test_run_extended_bench_cli.py tests/test_extended_runner.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check scripts/run_extended_bench.py olmlx/bench/extended_runner.py tests/test_run_extended_bench_cli.py
uv run ruff format scripts/run_extended_bench.py olmlx/bench/extended_runner.py tests/test_run_extended_bench_cli.py
git add scripts/run_extended_bench.py olmlx/bench/extended_runner.py tests/test_run_extended_bench_cli.py
git commit -m "feat(bench): add CLI orchestrator + HTTP driver for extended bench"
```

---

## Task 9: Chart builders (matplotlib)

**Files:**
- Create: `scripts/build_extended_report.py` (charts only — markdown comes in Task 10)
- Create: `tests/test_build_extended_report_charts.py`
- Create: `tests/fixtures/extended_run/raw/qwen-tiny.json` (synthetic per-model result for testing)

- [ ] **Step 1: Add the fixture**

Create `tests/fixtures/extended_run/raw/qwen-tiny.json`:

```json
{
  "model": "org/qwen-tiny",
  "tier": "extended",
  "assignment": "full_core",
  "warmup_tok_per_s": 95.0,
  "per_suite_pass_rate": {
    "humaneval-plus": 0.80,
    "gsm8k": 0.90,
    "gpqa": 0.55,
    "mbpp-plus": 0.74,
    "math500": 0.60,
    "mmlu-pro": 0.72,
    "ifeval": 0.85,
    "ruler-niah": 1.0
  },
  "composite": 0.77,
  "prompts": [],
  "speed": {"decode_tok_per_s_p50": 95.0, "ttft_p50_ms": 220.0},
  "elapsed_seconds": 1234.5
}
```

Create a second fixture `tests/fixtures/extended_run/raw/big-moe.json` (core-only model):

```json
{
  "model": "org/big-moe",
  "tier": "core_only",
  "assignment": "core_minus_gpqa",
  "warmup_tok_per_s": 7.5,
  "per_suite_pass_rate": {"humaneval-plus": 0.90, "gsm8k": 0.95},
  "composite": 0.925,
  "prompts": [],
  "speed": {"decode_tok_per_s_p50": 7.5, "ttft_p50_ms": 850.0},
  "elapsed_seconds": 8765.4
}
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_build_extended_report_charts.py`:

```python
"""Tests for the chart builders in scripts/build_extended_report.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_extended_report  # type: ignore[import-not-found]


@pytest.fixture
def run_dir(tmp_path):
    """Copy the synthetic run fixture into tmp_path."""
    raw = tmp_path / "raw"
    raw.mkdir()
    fixtures = Path(__file__).parent / "fixtures" / "extended_run" / "raw"
    for src in fixtures.glob("*.json"):
        (raw / src.name).write_bytes(src.read_bytes())
    return tmp_path


class TestLoadResults:
    def test_loads_all_json(self, run_dir):
        results = build_extended_report.load_results(run_dir)
        assert len(results) == 2
        models = {r["model"] for r in results}
        assert models == {"org/qwen-tiny", "org/big-moe"}


class TestFrontierChart:
    def test_writes_png(self, run_dir):
        out = run_dir / "charts" / "frontier.png"
        build_extended_report.render_frontier_chart(
            build_extended_report.load_results(run_dir), out
        )
        assert out.exists()
        assert out.stat().st_size > 0


class TestSuiteHeatmap:
    def test_writes_png(self, run_dir):
        out = run_dir / "charts" / "suite_heatmap.png"
        build_extended_report.render_suite_heatmap(
            build_extended_report.load_results(run_dir), out
        )
        assert out.exists()


class TestQuantPairsChart:
    def test_handles_empty_pairs(self, run_dir):
        # Synthetic fixtures don't include matched quant pairs — render should
        # produce a placeholder PNG noting "no matched pairs".
        out = run_dir / "charts" / "quant_pairs.png"
        build_extended_report.render_quant_pairs_chart(
            build_extended_report.load_results(run_dir), out
        )
        assert out.exists()
```

Run: `uv run pytest tests/test_build_extended_report_charts.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the chart code**

Create `scripts/build_extended_report.py`:

```python
"""Render the extended-bench report from per-model raw JSON.

Pure post-processor: reads ``<output>/raw/*.json``, draws matplotlib charts
into ``<output>/charts/``, and writes ``<output>/README.md``. Idempotent;
can be re-run without re-benching any model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless rendering for CI / scripted use
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger("build_extended_report")

# Stable suite ordering across the report.
SUITES_CORE = ("humaneval-plus", "gsm8k", "gpqa")
SUITES_EXT = ("mbpp-plus", "math500", "mmlu-pro", "ifeval", "ruler-niah")
SUITES_ALL = SUITES_CORE + SUITES_EXT

# Matched quant pairs reported in the comparison chart.
QUANT_PAIRS = [
    (
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        "mlx-community/Qwen3.6-35B-A3B-6bit",
        "Qwen3.6-35B-A3B",
    ),
    (
        "mlx-community/Qwen3.6-27B-4bit",
        "unsloth/Qwen3.6-27B-MLX-8bit",
        "Qwen3.6-27B",
    ),
]


def load_results(output_dir: Path) -> list[dict[str, Any]]:
    raw = output_dir / "raw"
    results: list[dict[str, Any]] = []
    for path in sorted(raw.glob("*.json")):
        if path.parent.name == "ablation":
            continue
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def _short_name(hf_path: str) -> str:
    return hf_path.rsplit("/", 1)[-1]


def render_frontier_chart(results: list[dict[str, Any]], out_path: Path) -> None:
    """Scatter: decode tok/s vs composite quality, colored by tier."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for tier, color in (("extended", "tab:blue"), ("core_only", "tab:orange")):
        xs = [r.get("warmup_tok_per_s", 0) for r in results if r["tier"] == tier]
        ys = [r.get("composite", 0) for r in results if r["tier"] == tier]
        labels = [_short_name(r["model"]) for r in results if r["tier"] == tier]
        ax.scatter(xs, ys, c=color, label=tier, s=80, alpha=0.75)
        for x, y, lbl in zip(xs, ys, labels, strict=True):
            ax.annotate(lbl, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Warmup decode tok/s (proxy for production throughput)")
    ax.set_ylabel("Composite quality (unweighted mean of suite pass rates)")
    ax.set_title("Speed-quality frontier")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_suite_heatmap(results: list[dict[str, Any]], out_path: Path) -> None:
    """Heatmap of per-suite pass rates: 23 rows × suite columns."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = sorted(results, key=lambda r: (r["tier"], -r.get("composite", 0)))
    suites = list(SUITES_ALL)
    grid = []
    labels = []
    for r in results:
        labels.append(_short_name(r["model"]))
        per_suite = r.get("per_suite_pass_rate", {})
        grid.append([per_suite.get(s, float("nan")) for s in suites])
    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.35)))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(suites)))
    ax.set_xticklabels(suites, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v == v:  # not NaN
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
    fig.colorbar(im, ax=ax, label="pass rate")
    ax.set_title("Per-suite pass rate by model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_quant_pairs_chart(results: list[dict[str, Any]], out_path: Path) -> None:
    """Grouped bars: per-suite pass rate for matched 4-bit vs higher-bit pairs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_model = {r["model"]: r for r in results}
    pairs_found = [(a, b, n) for a, b, n in QUANT_PAIRS if a in by_model and b in by_model]
    if not pairs_found:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(
            0.5,
            0.5,
            "No matched quant pairs in this run",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return
    # One bar group per (pair, suite); two bars per group for the two quants.
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_w = 0.35
    x_base = 0
    xticks = []
    xtick_labels = []
    for a, b, name in pairs_found:
        a_rates = by_model[a].get("per_suite_pass_rate", {})
        b_rates = by_model[b].get("per_suite_pass_rate", {})
        common = [s for s in SUITES_ALL if s in a_rates and s in b_rates]
        for j, suite in enumerate(common):
            ax.bar(x_base + j - bar_w / 2, a_rates[suite], width=bar_w, color="tab:blue")
            ax.bar(x_base + j + bar_w / 2, b_rates[suite], width=bar_w, color="tab:orange")
            xticks.append(x_base + j)
            xtick_labels.append(f"{name}\n{suite}")
        x_base += len(common) + 1
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("pass rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Matched-pair quant comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/test_build_extended_report_charts.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check scripts/build_extended_report.py tests/test_build_extended_report_charts.py
uv run ruff format scripts/build_extended_report.py tests/test_build_extended_report_charts.py
git add scripts/build_extended_report.py tests/test_build_extended_report_charts.py tests/fixtures/
git commit -m "feat(bench): add matplotlib chart builders for extended report"
```

---

## Task 10: Markdown report renderer

**Files:**
- Modify: `scripts/build_extended_report.py`
- Create: `tests/test_build_extended_report_markdown.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_build_extended_report_markdown.py`:

```python
"""Tests for the markdown report renderer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_extended_report  # type: ignore[import-not-found]


def _fixture_run(tmp_path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    fixtures = Path(__file__).parent / "fixtures" / "extended_run" / "raw"
    for src in fixtures.glob("*.json"):
        (raw / src.name).write_bytes(src.read_bytes())
    return tmp_path


class TestHeadlineTable:
    def test_contains_all_models(self, tmp_path):
        run = _fixture_run(tmp_path)
        results = build_extended_report.load_results(run)
        md = build_extended_report.render_headline_table(results)
        assert "qwen-tiny" in md
        assert "big-moe" in md

    def test_em_dash_for_missing_extended_cells(self, tmp_path):
        # big-moe is core_only, so its extended suite columns should be em-dashes.
        run = _fixture_run(tmp_path)
        results = build_extended_report.load_results(run)
        md = build_extended_report.render_headline_table(results)
        # Find the big-moe row.
        big_row = next(line for line in md.splitlines() if "big-moe" in line)
        # The row should contain the em-dash placeholder.
        assert "—" in big_row


class TestBuildReport:
    def test_writes_readme_and_charts(self, tmp_path):
        run = _fixture_run(tmp_path)
        build_extended_report.build_report(run)
        assert (run / "README.md").exists()
        assert (run / "charts" / "frontier.png").exists()
        assert (run / "charts" / "suite_heatmap.png").exists()
        assert (run / "charts" / "quant_pairs.png").exists()
        text = (run / "README.md").read_text()
        assert "# Extended benchmark" in text
        assert "## Methodology" in text
        assert "## Findings" in text
        assert "## Future research directions" in text
```

Run: `uv run pytest tests/test_build_extended_report_markdown.py -v`
Expected: FAIL with `AttributeError: module 'build_extended_report' has no attribute 'render_headline_table'`.

- [ ] **Step 2: Implement the markdown renderer**

Append to `scripts/build_extended_report.py`:

```python
def render_headline_table(results: list[dict[str, Any]]) -> str:
    """Render the 23-row headline table."""
    results = sorted(results, key=lambda r: (r["tier"], -r.get("composite", 0)))
    header = (
        "| Model | Tier | tok/s | Composite | "
        + " | ".join(SUITES_ALL)
        + " |"
    )
    sep = "|" + "|".join(["---"] * (len(SUITES_ALL) + 4)) + "|"
    lines = [header, sep]
    for r in results:
        per_suite = r.get("per_suite_pass_rate", {})
        cells = []
        for s in SUITES_ALL:
            if s in per_suite:
                cells.append(f"{per_suite[s]:.2f}")
            elif s in SUITES_CORE:
                cells.append("0.00")
            else:
                cells.append("—")
        lines.append(
            f"| {_short_name(r['model'])} | {r['tier']} | "
            f"{r.get('warmup_tok_per_s', 0):.1f} | "
            f"{r.get('composite', 0):.2f} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def render_findings(results: list[dict[str, Any]]) -> str:
    """Auto-generate findings bullets from the result set."""
    lines: list[str] = []
    extended = [r for r in results if r["tier"] == "extended"]
    if extended:
        top = max(extended, key=lambda r: r.get("composite", 0))
        lines.append(
            f"- **Top extended-tier composite:** "
            f"`{_short_name(top['model'])}` at {top.get('composite', 0):.2f}."
        )
        fastest = max(extended, key=lambda r: r.get("warmup_tok_per_s", 0))
        lines.append(
            f"- **Fastest extended row:** `{_short_name(fastest['model'])}` "
            f"at {fastest.get('warmup_tok_per_s', 0):.1f} tok/s "
            f"(composite {fastest.get('composite', 0):.2f})."
        )
    saturated = []
    for s in SUITES_ALL:
        rates = [
            r["per_suite_pass_rate"][s]
            for r in results
            if s in r.get("per_suite_pass_rate", {})
        ]
        if rates and min(rates) >= 0.95:
            saturated.append(s)
    if saturated:
        lines.append(
            "- **Saturated suites this run:** "
            + ", ".join(f"`{s}`" for s in saturated)
            + ". Retire or replace before the next report."
        )
    if not lines:
        lines.append("- No models produced grade-able results in this run.")
    return "\n".join(lines)


def render_future_research(results: list[dict[str, Any]]) -> str:
    """Heuristic future-research bullets derived from the run."""
    bullets: list[str] = []
    has_ext = any(r["tier"] == "extended" for r in results)
    if has_ext:
        bullets.append(
            "- Compare spectral-calibrated KV-quant vs TurboQuant on a target "
            "that has on-disk calibration data, using the Core suite as the "
            "common ground."
        )
        bullets.append(
            "- Run a same-version draft for the A3B target if/when one is "
            "released — the May 2026 report's cross-version draft hit 55% "
            "acceptance but still ran net-slower."
        )
    if any(
        r.get("per_suite_pass_rate", {}).get("ruler-niah", 0) < 0.5 for r in results
    ):
        bullets.append(
            "- RULER pass rate drops below 50% on at least one row; sweep KV "
            "quant bit width (off / TQ-4 / TQ-2 / spectral-4) at 4k/8k/16k "
            "context to map the degradation curve."
        )
    if any(
        r.get("per_suite_pass_rate", {}).get("humaneval-plus", 0) >= 0.95
        for r in results
    ):
        bullets.append(
            "- HumanEval+ is saturating at the top — swap to LiveCodeBench "
            "for the coding deep-dive in the next report."
        )
    if not bullets:
        bullets.append(
            "- No clear research direction surfaced by this run; consider a "
            "longer follow-up at full-split sizes."
        )
    return "\n".join(bullets)


_METHODOLOGY_BLOCK = """
- **Serving:** each model loaded via `olmlx serve` and hit through `/api/chat`
  at `temperature=0`, `seed=42`, `top_p=1.0`. Per-model `options` from
  `models.json` are overridden so models are compared at the same sampling
  distribution.
- **Token caps:** 4096 for HumanEval+/MBPP+/GSM8K/MATH-500 (avoids
  `<think>`-block truncation on verbose reasoners); 1024 for MMLU-Pro / GPQA /
  IFEval / RULER.
- **Composite score:** unweighted mean of per-suite pass rates. Each suite
  contributes equally regardless of prompt count.
- **Suite subset selection:** deterministic spread by source-ID order (or
  stratified by category/level/instruction-family); the selected ID list is
  recorded in each `raw/<model>.json` for reproducibility.
- **Runtime triage:** when a slow model can't finish Core within its
  remaining budget, GPQA is dropped first; HumanEval+ only is the last-resort
  slice. The assignment is recorded per row.
""".strip()


def build_report(output_dir: Path) -> None:
    """Render charts + README into output_dir."""
    results = load_results(output_dir)
    charts_dir = output_dir / "charts"
    render_frontier_chart(results, charts_dir / "frontier.png")
    render_suite_heatmap(results, charts_dir / "suite_heatmap.png")
    render_quant_pairs_chart(results, charts_dir / "quant_pairs.png")
    md = f"""# Extended benchmark — 23 configured models

Speed + quality comparison across the models in `~/.olmlx/models.json`.
See `docs/superpowers/specs/2026-05-24-extended-bench-design.md` for the
design that drove this run.

## Methodology

{_METHODOLOGY_BLOCK}

## Headline

![Speed-quality frontier](charts/frontier.png)

{render_headline_table(results)}

## Per-suite pass rates

![Suite heatmap](charts/suite_heatmap.png)

## Matched quant pairs

![Quant pairs](charts/quant_pairs.png)

## Findings

{render_findings(results)}

## Future research directions

{render_future_research(results)}

## Caveats

- HumanEval+ / MBPP+ run with sandboxed `code_exec`; acceptable for the
  single-user local olmlx tool per `CLAUDE.md`.
- KV-quant noise is length-dependent and not load-bearing at this set size;
  would be at full-split sizes.
- Flash-MoE rows reflect SSD I/O variance — re-run on a quiet machine for
  best comparability.
- IFEval covers the verifiable-constraint subset only; rubric-graded
  constraints are excluded.

## Reproducing

```bash
python scripts/run_extended_bench.py \\
    --models-config ~/.olmlx/models.json \\
    --output {output_dir} \\
    --enable-code-exec --start-server

python scripts/build_extended_report.py {output_dir}
```
"""
    (output_dir / "README.md").write_text(md, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Render extended bench report")
    parser.add_argument("output", type=Path, help="Run directory containing raw/")
    args = parser.parse_args(argv)
    build_report(args.output)
    logger.info("wrote %s", args.output / "README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Verify tests pass**

Run: `uv run pytest tests/test_build_extended_report_markdown.py tests/test_build_extended_report_charts.py -v`
Expected: PASS (8 tests total).

- [ ] **Step 4: Lint + commit**

```bash
uv run ruff check scripts/build_extended_report.py tests/test_build_extended_report_markdown.py
uv run ruff format scripts/build_extended_report.py tests/test_build_extended_report_markdown.py
git add scripts/build_extended_report.py tests/test_build_extended_report_markdown.py
git commit -m "feat(bench): add markdown report renderer for extended bench"
```

---

## Task 11: End-to-end smoke test

**Files:**
- Create: `tests/integration/test_extended_bench_smoke.py`

- [ ] **Step 1: Write the smoke test (skipped by default, opt-in via env)**

Create `tests/integration/test_extended_bench_smoke.py`:

```python
"""End-to-end smoke test for the extended bench.

Skipped unless `OLMLX_RUN_E2E_BENCH=1` and an `olmlx serve` is reachable at
the configured base URL. Verifies the full pipeline (load → grade → write
JSON → render report) on a single tiny draft model with a 3-prompt suite.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest

# Skip if e2e is not opted in OR if olmlx isn't reachable.
pytestmark = pytest.mark.skipif(
    os.environ.get("OLMLX_RUN_E2E_BENCH") != "1",
    reason="set OLMLX_RUN_E2E_BENCH=1 to run; needs a live olmlx serve",
)


@pytest.fixture
def is_server_up() -> bool:
    try:
        httpx.get("http://localhost:11434/api/version", timeout=2.0)
        return True
    except httpx.HTTPError:
        return False


@pytest.mark.asyncio
async def test_smoke_run_and_report(tmp_path, is_server_up):
    if not is_server_up:
        pytest.skip("olmlx serve not reachable at localhost:11434")
    from olmlx.bench.extended_runner import run_model

    # Use a tiny draft model so the smoke completes in <2 minutes.
    result = await run_model(
        model="mlx-community/Qwen3-0.6B-4bit",
        tier="core_only",  # core-only keeps the suite small
        base_url="http://localhost:11434",
        output_dir=tmp_path,
        remaining_seconds=600.0,  # short — triage will drop to HE_PLUS_ONLY or smaller
    )
    assert result.composite >= 0.0
    raw_path = tmp_path / "raw" / "mlx-community_Qwen3-0.6B-4bit.json"
    assert raw_path.exists()
    data = json.loads(raw_path.read_text())
    assert data["model"] == "mlx-community/Qwen3-0.6B-4bit"

    # Build the report — should produce all charts + README.
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    import build_extended_report  # type: ignore[import-not-found]

    build_extended_report.build_report(tmp_path)
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "charts" / "frontier.png").exists()
```

- [ ] **Step 2: Verify the test is collected but skipped by default**

Run: `uv run pytest tests/integration/test_extended_bench_smoke.py -v`
Expected: SKIPPED (1 skipped) — the env-var guard fires.

- [ ] **Step 3: Lint + commit**

```bash
uv run ruff check tests/integration/test_extended_bench_smoke.py
uv run ruff format tests/integration/test_extended_bench_smoke.py
git add tests/integration/test_extended_bench_smoke.py
git commit -m "test(bench): add opt-in end-to-end smoke test for extended bench"
```

---

## Task 12: Full-suite verification + final cleanup

**Files:**
- Modify (if needed): any file with lint / type issues uncovered

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/test_extended_suites_subset.py tests/test_extended_suites_coding.py tests/test_extended_suites_math.py tests/test_extended_suites_knowledge.py tests/test_ifeval_grader.py tests/test_tier_table.py tests/test_extended_runner.py tests/test_run_extended_bench_cli.py tests/test_build_extended_report_charts.py tests/test_build_extended_report_markdown.py -v`

Expected: PASS (~60+ tests). Address any failure before moving on.

- [ ] **Step 2: Run ruff over the whole new surface**

```bash
uv run ruff check olmlx/bench/extended_suites.py olmlx/bench/ifeval_grader.py olmlx/bench/tier_table.py olmlx/bench/extended_runner.py scripts/run_extended_bench.py scripts/build_extended_report.py
uv run ruff format --check olmlx/bench/extended_suites.py olmlx/bench/ifeval_grader.py olmlx/bench/tier_table.py olmlx/bench/extended_runner.py scripts/run_extended_bench.py scripts/build_extended_report.py
```

Expected: no errors. Fix any reported issue and re-run.

- [ ] **Step 3: Run pyright over the new modules**

```bash
uv run pyright olmlx/bench/extended_suites.py olmlx/bench/ifeval_grader.py olmlx/bench/tier_table.py olmlx/bench/extended_runner.py
```

Expected: 0 errors, 0 warnings on the new files. (Pre-existing repo-wide issues unrelated to this work are not a blocker.)

- [ ] **Step 4: Commit any final fixes (skip if Steps 1-3 reported no issues)**

```bash
# Only run if there are uncommitted changes from Steps 2-3:
git status --short
# If non-empty:
git add -p   # review and stage only intentional changes
git commit -m "chore(bench): final lint / type cleanup for extended-bench modules"
```

---

## Deferred / incremental (not in this plan, intentionally)

These spec items are intentionally deferred so this plan delivers a complete-but-minimal harness before polish:

- **Per-model speed-suite integration.** The runner captures `warmup_tok_per_s` (enough for the frontier scatter) but not the full 7-prompt TTFT/throughput suite. The user can run `olmlx bench run --model <hf> --scenarios <appropriate>` separately and the report's speed columns can be enriched in a follow-up by reading the existing bench result JSON.
- **`ablation_delta.png` and `ruler_position.png` charts.** Listed in the spec but require ablation-run data and per-prompt position metadata respectively. The chart builders in Task 9 can be extended once those data sources land; the report builder gracefully omits them when absent.
- **Detailed deep-dive report sections** (Coding / Math / Knowledge / Steerability / Quant-pairs analysis prose). Task 10 ships a complete-but-minimal report (Methodology, Headline + heatmap, Findings, Future research, Caveats, Reproducing). The deeper analysis sections need real run data to be meaningful; once a run completes, the operator can extend `build_report` with per-suite sections informed by the actual numbers.
- **2-model ablation driver script.** The spec calls for `scripts/run_ablation.py`. It can be added once `run_model` from Task 7 is stable — it's a small wrapper that calls `run_model` with patched env vars per ablation cell. Marked as follow-up because the ablation cells aren't on the wall-clock critical path for the headline report.

## Out of scope (per spec)

- Vision-language quality benchmarking
- Multi-turn / agentic eval (no SWE-bench, no τ-bench)
- Full-split GSM8K (1319) / full MMLU (14k)
- Quality × full feature-flag matrix (replaced by 2-model ablation; see spec)
- Distributed-inference scenarios
- Actually executing the run on real models (this plan delivers the harness — the user kicks the run themselves over the weekend with `python scripts/run_extended_bench.py --start-server --enable-code-exec`)

## Future work surfaced during planning

- The 2-model ablation (TurboQuant / speculative) is mentioned in the spec but its driver is *not* implemented as a separate script. Once Task 7's `run_model` is in place, a small follow-up plan can add `scripts/run_ablation.py` that re-invokes `run_model` with patched env vars. Out of scope here because the ablation can be performed by running `OLMLX_KV_CACHE_QUANT=spectral:4 python scripts/run_extended_bench.py --only ... --output docs/benchmarks/extended-2026-05/ablation/...` and the report builder can ingest those subdirectories in a later iteration.
- The IFEval `language:response_language` constraint is stubbed to always-fail; a follow-up could vendor `langid` to actually grade it.
