"""Quality evaluation for bench prompts.

Graders take an ``output_text`` and a grader-specific ``expected`` payload
and return a ``QualityResult``. They're pure functions, invoked in the
parent process after worker HTTP calls return — the worker only produces
``output_text`` and never runs graders itself.

The ``code_exec`` grader runs model-generated Python in a ``subprocess``
with ``resource`` limits. It's disabled by default and must be opted in
via ``--enable-code-exec`` / ``OLMLX_BENCH_CODE_EXEC=1``. This is not a
hard sandbox — acceptable only because olmlx is a single-user local tool
evaluating a model the user already chose to run locally.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    grader: str
    passed: bool | None  # None = ungraded (no expected, no golden, opted out)
    score: float | None  # grader-specific, typically 0.0–1.0
    detail: str
    reference: str | None = None

    def to_dict(self) -> dict:
        return {
            "grader": self.grader,
            "passed": self.passed,
            "score": self.score,
            "detail": self.detail,
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, d: dict) -> QualityResult:
        return cls(
            grader=d["grader"],
            passed=d.get("passed"),
            score=d.get("score"),
            detail=d.get("detail", ""),
            reference=d.get("reference"),
        )


Grader = Callable[[str, dict[str, Any]], QualityResult]


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _clip(s: str, n: int = 200) -> str:
    return s if len(s) <= n else s[:n] + "…"


def grade_exact_match(output: str, expected: dict[str, Any]) -> QualityResult:
    answer = str(expected.get("answer", ""))
    if expected.get("normalize", True):
        passed = _normalize_ws(output).casefold() == _normalize_ws(answer).casefold()
    else:
        passed = output == answer
    return QualityResult(
        grader="exact_match",
        passed=passed,
        score=1.0 if passed else 0.0,
        detail="match" if passed else "mismatch",
        reference=_clip(answer),
    )


def grade_contains(output: str, expected: dict[str, Any]) -> QualityResult:
    raw = expected.get("substrings", [])
    substrings = [str(s) for s in raw]
    if not substrings:
        return QualityResult(
            grader="contains",
            passed=None,
            score=None,
            detail="no substrings specified",
        )
    ignore_case = expected.get("ignore_case", True)
    out = output.casefold() if ignore_case else output
    hits = [s for s in substrings if (s.casefold() if ignore_case else s) in out]
    require_all = expected.get("all", True)
    passed = len(hits) == len(substrings) if require_all else len(hits) > 0
    score = len(hits) / len(substrings)
    detail = f"{len(hits)}/{len(substrings)} substrings matched"
    return QualityResult(
        grader="contains",
        passed=passed,
        score=score,
        detail=detail,
        reference=_clip("; ".join(substrings)),
    )


def grade_regex_match(output: str, expected: dict[str, Any]) -> QualityResult:
    pattern = expected.get("pattern")
    if not pattern:
        return QualityResult(
            grader="regex_match",
            passed=None,
            score=None,
            detail="no pattern specified",
        )
    # Default to group 0 (the whole match) so a pattern without an
    # explicit capture group still produces a useful comparison. Callers
    # that need a substring extraction pass `"group": 1` (or higher)
    # explicitly — MMLU does.
    group = int(expected.get("group", 0))
    answer = str(expected.get("answer", ""))
    try:
        m = re.search(pattern, output, flags=re.DOTALL)
    except re.error as exc:
        return QualityResult(
            grader="regex_match",
            passed=False,
            score=0.0,
            detail=f"invalid pattern: {exc}",
            reference=_clip(answer),
        )
    if m is None:
        return QualityResult(
            grader="regex_match",
            passed=False,
            score=0.0,
            detail="no regex match",
            reference=_clip(answer),
        )
    try:
        extracted = m.group(group)
    except IndexError:
        return QualityResult(
            grader="regex_match",
            passed=False,
            score=0.0,
            detail=f"group {group} missing",
            reference=_clip(answer),
        )
    if extracted is None:
        return QualityResult(
            grader="regex_match",
            passed=False,
            score=0.0,
            detail=f"group {group} did not participate in match",
            reference=_clip(answer),
        )
    passed = extracted.strip().casefold() == answer.strip().casefold()
    return QualityResult(
        grader="regex_match",
        passed=passed,
        score=1.0 if passed else 0.0,
        detail=f"extracted={extracted!r}",
        reference=_clip(answer),
    )


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Allow comma thousands separators so "#### 1,234" parses as 1234. Commas
# are stripped before float() because Python's float() rejects them.
_GSM8K_FINAL_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _extract_number(text: str) -> float | None:
    m = _GSM8K_FINAL_RE.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None
    # Also tolerate "answer: 42", "= 42", "\\boxed{42}", "42."
    boxed = re.search(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", text)
    if boxed:
        try:
            return float(boxed.group(1))
        except ValueError:
            pass
    matches = _NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    # `_NUMBER_RE` only matches strings of the form `-?\d+(?:\.\d+)?`,
    # which `float()` always accepts — no ValueError fallback needed.
    return float(matches[-1])


def grade_numeric(output: str, expected: dict[str, Any]) -> QualityResult:
    try:
        answer = float(expected["answer"])
    except (KeyError, TypeError, ValueError):
        return QualityResult(
            grader="numeric",
            passed=None,
            score=None,
            detail="no numeric answer specified",
        )
    tol = float(expected.get("tol", 0.0))
    extracted = _extract_number(output)
    if extracted is None:
        return QualityResult(
            grader="numeric",
            passed=False,
            score=0.0,
            detail="no number found in output",
            reference=str(answer),
        )
    passed = abs(extracted - answer) <= tol
    return QualityResult(
        grader="numeric",
        passed=passed,
        score=1.0 if passed else 0.0,
        detail=f"extracted={extracted}",
        reference=str(answer),
    )


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_completion(output: str) -> str:
    m = _CODE_BLOCK_RE.search(output)
    if m:
        return m.group(1)
    # Fallback: strip leading <think> blocks, return rest.
    cleaned = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
    return cleaned.strip()


def _code_exec_disabled(reason: str) -> QualityResult:
    return QualityResult(
        grader="code_exec",
        passed=None,
        score=None,
        detail=reason,
    )


def _code_exec_preexec() -> None:  # pragma: no cover — POSIX child
    # Locally import everything the child needs. preexec_fn runs in the
    # forked-but-not-yet-exec'd child, so it inherits the parent's
    # module namespace — but a local import keeps the function
    # self-contained and obvious to read.
    import resource
    import sys

    cpu = 5
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except (ValueError, OSError):
        pass
    # Cap output file size at 64 MB — generated code that tries to fill
    # the disk gets SIGXFSZ instead of running unbounded. 64 MB is
    # comfortably above any legitimate test output.
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    except (ValueError, OSError):
        pass
    # RLIMIT_AS limits the *virtual* address space, not resident memory.
    # On macOS, Python's allocator + linked dylibs can pre-map several GB
    # of address space at startup, so a tight RLIMIT_AS may kill the
    # child before its first statement. Apply only on Linux, where
    # startup VA usage is small enough for the limit to be meaningful;
    # RLIMIT_CPU + subprocess.run(timeout=10) still bound runaway code
    # cross-platform.
    if sys.platform.startswith("linux"):
        mem = 512 * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except (ValueError, OSError):
            pass
        # Defense against fork bombs in graded code. (0, 0) bars the
        # child from creating any more processes. Linux-only: macOS
        # does not honor RLIMIT_NPROC the same way and the python
        # subprocess we just spawned already exists, so this caps the
        # *graded code* itself, not the wrapper.
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except (ValueError, OSError):
            pass


def grade_code_exec(output: str, expected: dict[str, Any]) -> QualityResult:
    if not expected.get("_enabled", False):
        return _code_exec_disabled("code_exec disabled (pass --enable-code-exec)")
    if os.name != "posix":
        return _code_exec_disabled("code_exec requires POSIX (rlimits unavailable)")

    prompt = expected.get("prompt", "")
    tests = expected.get("tests", "")
    entry_point = expected.get("entry_point", "")
    if not tests or not entry_point:
        return QualityResult(
            grader="code_exec",
            passed=None,
            score=None,
            detail="missing tests or entry_point",
        )

    completion = _extract_completion(output)
    script = f"{prompt}\n{completion}\n{tests}\ncheck({entry_point})\n"

    ref = f"entry={entry_point}"
    # Capture the temp path *before* writing so a write failure still
    # unlinks the empty file in the finally below.
    script_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            script_path = f.name
            f.write(script)

        try:
            proc = subprocess.run(
                # -I isolates from user environment (no PYTHONPATH, no user
                # site-packages, no script-dir on sys.path). We deliberately
                # do NOT add -S — that would skip site.py and break imports
                # of any third-party package from the active venv (numpy,
                # which HumanEval+/MBPP+ tests routinely use). -I alone is
                # restrictive enough for a single-user local tool.
                [sys.executable, "-I", script_path],
                capture_output=True,
                text=True,
                timeout=10,
                env={"PATH": os.environ.get("PATH", "")},
                preexec_fn=_code_exec_preexec,
            )
        except subprocess.TimeoutExpired:
            return QualityResult(
                grader="code_exec",
                passed=False,
                score=0.0,
                detail="timeout",
                reference=ref,
            )
        if proc.returncode == 0:
            return QualityResult(
                grader="code_exec",
                passed=True,
                score=1.0,
                detail="tests passed",
                reference=ref,
            )
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        last = err[-1] if err else f"exit={proc.returncode}"
        return QualityResult(
            grader="code_exec",
            passed=False,
            score=0.0,
            detail=_clip(last, 200),
            reference=ref,
        )
    finally:
        if script_path is not None:
            try:
                Path(script_path).unlink()
            except OSError:
                pass


def _similarity(a: str, b: str) -> float:
    """Character-level similarity in [0, 1]. 1.0 means identical strings.

    Uses ``difflib.SequenceMatcher`` (C-accelerated, longest-common-subsequence
    based) instead of pure-Python edit distance — for a 2k-char model output
    vs a same-size golden, the difference is roughly 100× per call, and that
    cost dominated a batch bench run.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def grade_regression_snapshot(output: str, expected: dict[str, Any]) -> QualityResult:
    ref_path = expected.get("reference_path")
    if not ref_path:
        return QualityResult(
            grader="regression_snapshot",
            passed=None,
            score=None,
            detail="no golden captured",
        )
    try:
        reference = Path(ref_path).read_text(encoding="utf-8")
    except OSError as exc:
        # The golden was specified but is unreadable (deleted, perms,
        # corrupted). This is *not* a model failure — return passed=None,
        # the same sentinel as "no golden captured". The `detail`
        # discriminates the two cases for aggregators; the warning makes
        # sure the issue surfaces in operator logs instead of going
        # silently into the ungraded bucket.
        logger.warning("regression_snapshot golden unreadable: %s (%s)", ref_path, exc)
        return QualityResult(
            grader="regression_snapshot",
            passed=None,
            score=None,
            detail=f"cannot read golden: {exc}",
            reference=None,
        )
    mode = expected.get("mode", "similarity")
    threshold = float(expected.get("threshold", 0.95))
    if mode == "exact":
        passed = _normalize_ws(output) == _normalize_ws(reference)
        score = 1.0 if passed else 0.0
        detail = "exact match" if passed else "exact mismatch"
    else:
        score = _similarity(_normalize_ws(output), _normalize_ws(reference))
        passed = score >= threshold
        detail = f"similarity={score:.3f} threshold={threshold:.3f}"
    return QualityResult(
        grader="regression_snapshot",
        passed=passed,
        score=score,
        detail=detail,
        reference=_clip(reference),
    )


GRADERS: dict[str, Grader] = {
    "exact_match": grade_exact_match,
    "contains": grade_contains,
    "regex_match": grade_regex_match,
    "numeric": grade_numeric,
    "code_exec": grade_code_exec,
    "regression_snapshot": grade_regression_snapshot,
}

# Register graders defined in sibling modules. Bottom-of-file import is
# intentional — keeps the GRADERS table extensible without circular issues
# at top-of-module import time.
from olmlx.bench.ifeval_grader import grade_ifeval  # noqa: E402

GRADERS["ifeval"] = grade_ifeval


def grade(grader_name: str, output: str, expected: dict[str, Any]) -> QualityResult:
    grader_fn = GRADERS.get(grader_name)
    if grader_fn is None:
        return QualityResult(
            grader=grader_name,
            passed=None,
            score=None,
            detail=f"unknown grader {grader_name!r}",
        )
    try:
        return grader_fn(output, expected)
    except Exception as exc:  # defensive: a bad grader must not crash the run
        # passed=None (not False) so a grader bug does not silently inflate
        # the failure count and drag the reported pass rate down — same
        # sentinel used for "no expected", "disabled", "unknown grader".
        logger.exception("Grader %s raised", grader_name)
        return QualityResult(
            grader=grader_name,
            passed=None,
            score=None,
            detail=f"grader raised: {exc!r}",
        )
