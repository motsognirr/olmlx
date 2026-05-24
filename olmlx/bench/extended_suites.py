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

import json
import os
import random
import re
import string
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from olmlx.bench.prompts import BenchPrompt

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

    Picks indices ``round(i * len(items) / n)`` for i in 0..n-1, so the
    first item is always included; the last item is included when ``n`` is
    large relative to ``len(items)``.

    When ``key`` is provided and all strata can satisfy their allocation,
    returns exactly ``n`` items.  When some strata are smaller than their
    allocation, surplus is redistributed to other strata so the total still
    equals ``n``.
    """
    if n >= len(items):
        return list(items)
    if key is None:
        return [items[round(i * len(items) / n)] for i in range(n)]

    buckets: dict[str, list[T]] = {}
    for item in items:
        buckets.setdefault(key(item), []).append(item)
    labels = sorted(buckets)
    base = n // len(labels)
    remainder = n - base * len(labels)
    # Initial allocation per bucket.
    allotments: dict[str, int] = {
        label: base + (1 if i < remainder else 0) for i, label in enumerate(labels)
    }

    # Iteratively redistribute slack from exhausted buckets to non-exhausted ones.
    while True:
        slack = 0
        can_absorb: list[str] = []
        for label in labels:
            allot = allotments[label]
            bucket_len = len(buckets[label])
            if allot >= bucket_len:
                slack += allot - bucket_len
                allotments[label] = bucket_len  # cap at bucket size
            else:
                can_absorb.append(label)
        if slack == 0 or not can_absorb:
            break
        # Distribute slack evenly across absorbing buckets (priority to earlier labels).
        per = slack // len(can_absorb)
        extra = slack - per * len(can_absorb)
        for i, label in enumerate(can_absorb):
            allotments[label] += per + (1 if i < extra else 0)

    out: list[T] = []
    for label in labels:
        take = allotments[label]
        bucket = buckets[label]
        if take >= len(bucket):
            out.extend(bucket)
        else:
            out.extend(bucket[round(j * len(bucket) / take)] for j in range(take))
    return out


# ---------------------------------------------------------------------------
# HumanEval+ loader
# ---------------------------------------------------------------------------

_HUMANEVAL_PLUS_FILE = "humanevalplus.json"


def _fetch_humaneval_plus_to(path: Path) -> None:
    """Download HumanEval+ to ``path`` if not cached.

    Defers the ``datasets`` import so cached runs don't pay the import cost.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("evalplus/humanevalplus", split="test")
    records: list[dict[str, Any]] = [
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
    """Load HumanEval+ as bench prompts. ``n=None`` returns all available records."""
    cache_path = bench_cache_dir() / _HUMANEVAL_PLUS_FILE
    if not cache_path.exists():
        _fetch_humaneval_plus_to(cache_path)
    records: list[dict[str, Any]] = json.loads(cache_path.read_text(encoding="utf-8"))
    if n is not None:
        records = select_subset(records, n)
    result: list[BenchPrompt] = []
    for r in records:
        content = (
            "Complete the following Python function. Respond with a single "
            "fenced ```python``` code block containing the full function "
            "definition (you may include the signature and docstring). Do "
            "not include explanations.\n\n"
            f"{r['prompt']}"
        )
        result.append(
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
    return result


# ---------------------------------------------------------------------------
# MBPP+ loader
# ---------------------------------------------------------------------------

_MBPP_PLUS_FILE = "mbppplus.json"


def _mbpp_test_list_to_check(test_list: list[str]) -> tuple[str, str]:
    """Convert a list of assert strings into a ``def check(candidate):`` block.

    Returns ``(check_src, entry_point)`` so callers don't need to re-extract
    the function name themselves.
    """
    # Extract the function name from the first assert: ``assert fn_name(...)``
    first = test_list[0].strip()
    call_part = first.split("assert", 1)[1].strip()
    entry = call_part.split("(", 1)[0].strip()
    indented = "\n    ".join(s.strip() for s in test_list)
    return f"def check(candidate):\n    {entry} = candidate\n    {indented}\n", entry


def _fetch_mbpp_plus_to(path: Path) -> None:
    """Download MBPP+ to ``path`` if not cached."""
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("evalplus/mbppplus", split="test")
    records: list[dict[str, Any]] = [
        {
            "task_id": r["task_id"],
            "prompt": r["prompt"],
            "test_list": r["test_list"],
        }
        for r in ds
    ]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_mbpp_plus(n: int | None = 50) -> list[BenchPrompt]:
    """Load MBPP+ as bench prompts. ``n=None`` returns all."""
    cache_path = bench_cache_dir() / _MBPP_PLUS_FILE
    if not cache_path.exists():
        _fetch_mbpp_plus_to(cache_path)
    records: list[dict[str, Any]] = json.loads(cache_path.read_text(encoding="utf-8"))
    if n is not None:
        records = select_subset(records, n)
    result: list[BenchPrompt] = []
    for r in records:
        test_list: list[str] = r["test_list"]
        check_src, entry_point = _mbpp_test_list_to_check(test_list)
        content = (
            "Solve this Python task. Respond with a single fenced ```python``` "
            "code block containing the full function definition. Do not include "
            f"explanations.\n\n{r['prompt']}\n\n"
            f"Your function must be named ``{entry_point}``."
        )
        result.append(
            BenchPrompt(
                name=f"mbpp-plus-{r['task_id'].split('/')[-1]}",
                category="mbpp-plus",
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
                grader="code_exec",
                expected={
                    "prompt": "",
                    "tests": check_src,
                    "entry_point": entry_point,
                },
            )
        )
    return result


# ---------------------------------------------------------------------------
# GSM8K loader
# ---------------------------------------------------------------------------

_GSM8K_FILE = "gsm8k.json"
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


# ---------------------------------------------------------------------------
# MATH-500 loader
# ---------------------------------------------------------------------------

_MATH500_FILE = "math500.json"


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
            r"Solve step by step. End your answer with \boxed{...}."
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


# ---------------------------------------------------------------------------
# MMLU-Pro loader
# ---------------------------------------------------------------------------

_MMLU_PRO_FILE = "mmlu_pro.json"


def _fetch_mmlu_pro_to(path: Path) -> None:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    records = []
    for r in ds:
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
    """Load MMLU-Pro as bench prompts. ``n=None`` returns all available records."""
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


# ---------------------------------------------------------------------------
# GPQA-Diamond loader
# ---------------------------------------------------------------------------

_GPQA_FILE = "gpqa_diamond.json"


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
    """Load GPQA-Diamond as bench prompts. ``n=None`` returns all available records."""
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


# ---------------------------------------------------------------------------
# IFEval loader (verifiable-constraint subset)
# ---------------------------------------------------------------------------

_IFEVAL_FILE = "ifeval.json"

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


def load_ifeval(n: int | None = 50) -> list[BenchPrompt]:
    """Load IFEval (verifiable-constraint subset). ``n=None`` returns all."""
    cache_path = bench_cache_dir() / _IFEVAL_FILE
    if not cache_path.exists():
        _fetch_ifeval_to(cache_path)
    records = json.loads(cache_path.read_text(encoding="utf-8"))
    records = [r for r in records if _is_verifiable(r["instruction_id_list"])]
    if n is not None:

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


# ---------------------------------------------------------------------------
# RULER NIAH generator
# ---------------------------------------------------------------------------

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
    filler = (
        _RULER_CONTEXT_PARAGRAPH * (target_chars // len(_RULER_CONTEXT_PARAGRAPH) + 1)
    )[:target_chars]
    out: list[BenchPrompt] = []
    for i in range(n):
        key = f"{rng.randint(100000, 999999)}"
        pos = round(len(filler) * (i + 1) / (n + 1))
        needle = f"The magic key is {key}. Remember it. "
        haystack = (
            filler[:pos] + needle + filler[pos:][: target_chars - pos - len(needle)]
        )
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
