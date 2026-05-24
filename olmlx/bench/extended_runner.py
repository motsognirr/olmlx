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


# ---------------------------------------------------------------------------
# HTTP driver — Task 8 additions
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

import httpx  # noqa: E402

from olmlx.bench.quality import grade  # noqa: E402


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


async def _warmup(client: httpx.AsyncClient, model: str) -> float:
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
