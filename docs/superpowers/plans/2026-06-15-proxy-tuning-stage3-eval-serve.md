# Proxy-Tuning Stage 3 — Evaluate & Serve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an eval harness that sweeps the proxy-tuning α over a held-out olmlx prompt set, LLM-judges base-vs-steered completions on convention-adherence + coherence, and gates a ship/no-ship decision; then (on ship) register the trained pair for serving.

**Architecture:** A standalone harness in `olmlx/proxy_tuning_pipeline/eval.py` that loads the dense base (Qwen3-8B-4bit) + M⁺ + M⁻ **once**, then for each α∈{0, 0.5, 1.0, 1.5} rebinds a `ProxyTuningDecoder` and runs a minimal greedy `prefill`/`step` driver over every prompt (α=0 is the base baseline). Completions are scored by GPT-5.4-mini (reusing Stage 1's `OpenAIGenerator`), aggregated per α, and run through a concrete ship gate. The harness is decomposed into small pure units (driver, judge, aggregation, gate) that are TDD'd with fakes; the real α-sweep run, the 32B spot-check, and serve-registration are an operator runbook.

**Tech Stack:** MLX + `mlx_lm` (`load`, `ProxyTuningDecoder`), the OpenAI SDK (`gpt-5.4-mini`, `OPENAI_API_KEY`), pytest with injected fakes (no Metal/network in unit tests).

**Grounded facts (verified during planning):**
- α is **global-only**, baked into `ProxyTuningDecoder.__init__(base, expert, antiexpert, *, alpha=1.0)` at construction (`olmlx/engine/proxy_tuning.py:135`). The spec forbids changing the decode mode, so the sweep rebinds `_alpha` / rebuilds the decoder over **one** model load.
- The decoder protocol: `prefill(prompt: mx.array(1,seq)) -> int` (self-`reset()`s first, `spec_decoder_base.py:220`) and `step() -> (list[int], int)` returning one token (`proxy_tuning.py:239`). Greedy argmax → **deterministic, reproducible** output. Runs on the **default stream** (no `mx.stream` wrapper — `proxy_tuning.py:208-210`); the driver must not wrap it in a stream.
- `mlx_lm.load(path) -> (model, TokenizerWrapper)`; on a tokenizer **instance**: `apply_chat_template(messages, add_generation_prompt=True) -> list[int]`, `eos_token_ids -> set[int]` (e.g. `{151643}`), `decode(list[int]) -> str`.
- Stage 1's judge client: `OpenAIGenerator(client=None, model="gpt-5.4-mini", max_retries=6)` with `.generate(system, user) -> str`, lazy double-checked-lock client reading `OPENAI_API_KEY` (`olmlx/proxy_tuning_pipeline/expand.py:244-281`). Reused as-is for the judge.
- Stage 2 artifacts (present): `~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-4bit` (M⁺), `~/.olmlx/proxy_tuning/qwen3-1.7b-base-4bit` (M⁻). Steered base available: `~/.olmlx/models/mlx-community_Qwen3-8B-4bit`.
- Pre-flight gate already exists: `olmlx.proxy_tuning_pipeline.verify.assert_serveable_pair(anti_expert_dir, expert_dir, base_vocab_size)`.

**Scope:** Stage 3 only of `docs/superpowers/specs/2026-06-15-olmlx-proxy-tuning-pair-design.md` §7 (see "Resolved decisions" there). Blocked on Stage 2 artifacts (complete).

**Execution note:** Tasks 1–5 are TDD code (agent-implementable). Tasks 6–8 are an **operator runbook** — the real α-sweep run (Metal + the trained pair), the 32B spot-check, and serve-registration on the live `olmlx-1`/:11436 checkout. The operator runs 6–8 and pastes results back, as in Stages 1–2.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `olmlx/proxy_tuning_pipeline/eval_schema.py` | `EvalPrompt`, `EvalScore`, `AlphaSummary`, `ShipDecision` dataclasses + JSONL prompt loader | Create |
| `olmlx/proxy_tuning_pipeline/eval_prompts.jsonl` | ~50 held-out olmlx-style prompts (3 categories), authored fresh | Create (artifact, committed) |
| `olmlx/proxy_tuning_pipeline/eval_driver.py` | `generate_one()` — greedy `prefill`/`step` driver over one decoder + prompt | Create |
| `olmlx/proxy_tuning_pipeline/eval_judge.py` | `ProxyEvalJudge` — rubric prompt assembly + GPT-5.4-mini call + JSON parse | Create |
| `olmlx/proxy_tuning_pipeline/eval.py` | `aggregate()`, `ship_decision()`, `run_eval()` orchestrator + a `main()` CLI (own `python -m ...eval` entry point, mirroring `verify.py`) | Create |
| `tests/test_proxy_tuning_eval.py` | unit tests for schema/loader, driver, judge, aggregation, gate, orchestration (all with fakes) | Create |

> **CLI convention:** this package's Stage-2 `verify.py` exposes its own `main()` run as `python -m olmlx.proxy_tuning_pipeline.verify` — it does **not** add subcommands to `cli.py` (which is a single-command Stage-1 data-pipeline entry point). The eval CLI follows the same pattern: a `main()` in `eval.py`, run as `python -m olmlx.proxy_tuning_pipeline.eval`. `cli.py` is **not** modified.

---

## Task 1: Eval schema + prompt loader (TDD)

**Files:**
- Create: `olmlx/proxy_tuning_pipeline/eval_schema.py`
- Test: `tests/test_proxy_tuning_eval.py`

**Context:** The harness consumes a JSONL prompt set and produces structured scores. Define the dataclasses once here so every later unit shares them. An `EvalPrompt` carries an `id`, a `category` (one of the three rubric kinds), and `messages` (mlx-lm chat shape, user-only — the model generates the assistant turn).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_proxy_tuning_eval.py`:

```python
"""Tests for the Stage-3 proxy-tuning eval harness."""

from __future__ import annotations

import json

import pytest

from olmlx.proxy_tuning_pipeline.eval_schema import (
    EvalPrompt,
    load_eval_prompts,
)


def test_load_eval_prompts_parses_jsonl(tmp_path):
    p = tmp_path / "prompts.jsonl"
    rows = [
        {"id": "inv-1", "category": "explain_invariant",
         "messages": [{"role": "user", "content": "Explain the Metal stream invariant."}]},
        {"id": "impl-1", "category": "implement_convention",
         "messages": [{"role": "user", "content": "Add a config flag following olmlx conventions."}]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    prompts = load_eval_prompts(str(p))

    assert [pr.id for pr in prompts] == ["inv-1", "impl-1"]
    assert isinstance(prompts[0], EvalPrompt)
    assert prompts[0].category == "explain_invariant"
    assert prompts[0].messages[0]["role"] == "user"


def test_load_eval_prompts_rejects_bad_category(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text(json.dumps(
        {"id": "x", "category": "nonsense",
         "messages": [{"role": "user", "content": "hi"}]}) + "\n")
    with pytest.raises(ValueError, match="category"):
        load_eval_prompts(str(p))


def test_load_eval_prompts_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "prompts.jsonl"
    row = {"id": "dup", "category": "convention_qa",
           "messages": [{"role": "user", "content": "hi"}]}
    p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_prompts(str(p))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.proxy_tuning_pipeline.eval_schema'`

- [ ] **Step 3: Write minimal implementation**

Create `olmlx/proxy_tuning_pipeline/eval_schema.py`:

```python
"""Dataclasses + JSONL loader for the Stage-3 proxy-tuning eval harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# The three rubric categories the held-out prompt set spans.
VALID_CATEGORIES = frozenset(
    {"explain_invariant", "implement_convention", "convention_qa"}
)


@dataclass(frozen=True)
class EvalPrompt:
    id: str
    category: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class EvalScore:
    prompt_id: str
    category: str
    alpha: float
    convention: int  # 1-5
    coherence: int  # 1-5
    rationale: str
    output: str


@dataclass(frozen=True)
class AlphaSummary:
    alpha: float
    n: int
    mean_convention: float
    mean_coherence: float


@dataclass(frozen=True)
class ShipDecision:
    ship: bool
    best_alpha: float
    base_convention: float
    best_convention: float
    base_coherence: float
    best_coherence: float
    conv_margin: float
    coh_drop: float
    reason: str
    per_alpha: list[AlphaSummary] = field(default_factory=list)


def load_eval_prompts(path: str) -> list[EvalPrompt]:
    """Load + validate the held-out prompt JSONL.

    Raises ValueError on an unknown category or a duplicate id — both would
    silently skew the aggregate, so fail loud.
    """
    prompts: list[EvalPrompt] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            cat = row["category"]
            if cat not in VALID_CATEGORIES:
                raise ValueError(
                    f"prompt {row.get('id')!r} has unknown category {cat!r}; "
                    f"expected one of {sorted(VALID_CATEGORIES)}"
                )
            pid = row["id"]
            if pid in seen:
                raise ValueError(f"duplicate prompt id {pid!r}")
            seen.add(pid)
            prompts.append(
                EvalPrompt(id=pid, category=cat, messages=row["messages"])
            )
    return prompts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_eval.py -v`
Expected: 3 passed

- [ ] **Step 5: Run ruff, then commit**

Run: `uv run ruff check olmlx/proxy_tuning_pipeline/eval_schema.py tests/test_proxy_tuning_eval.py && uv run ruff format olmlx/proxy_tuning_pipeline/eval_schema.py tests/test_proxy_tuning_eval.py`

```bash
git add olmlx/proxy_tuning_pipeline/eval_schema.py tests/test_proxy_tuning_eval.py
git commit -m "feat(proxy-tuning): Stage-3 eval schema + prompt loader"
```

---

## Task 2: Author the held-out prompt set

**Files:** Create: `olmlx/proxy_tuning_pipeline/eval_prompts.jsonl`

**Context:** ~50 prompts, authored **fresh** — grounded in olmlx's domain (CLAUDE.md invariants, `docs/` specs, code conventions) but **never copied from the training data** (they test generalization). Spread across the three categories (~17 each): `explain_invariant` (ask to explain a documented invariant / why it holds), `implement_convention` ("implement X following olmlx conventions"), `convention_qa` (Q&A on idioms: type-annotation rules, streaming patterns, error-message policy). Each is a single user turn; the model writes the assistant turn.

- [ ] **Step 1: Write the seed prompt set**

Create `olmlx/proxy_tuning_pipeline/eval_prompts.jsonl` with these concrete starters (one JSON object per line), then **extend to ~50 in the same shape/spread during execution**:

```jsonl
{"id": "inv-metal-stream", "category": "explain_invariant", "messages": [{"role": "user", "content": "In an Apple-MLX inference server, why must all non-speculative prefill and decode run on a single dedicated generation stream? What breaks if you mix streams on a Qwen3 hybrid?"}]}
{"id": "inv-rotating-prefill", "category": "explain_invariant", "messages": [{"role": "user", "content": "Why must a pure-SWA (RotatingKVCache, no ArraysCache) model be prefilled in a single forward call? What goes wrong if you split at a message boundary?"}]}
{"id": "inv-grammar-tokenizer", "category": "explain_invariant", "messages": [{"role": "user", "content": "A grammar cache keyed by tokenizer id() can serve a wrong-vocabulary grammar after CPython recycles the address. How would you make the cache safe?"}]}
{"id": "impl-config-flag", "category": "implement_convention", "messages": [{"role": "user", "content": "Add a new boolean setting to a pydantic-settings Settings class that uses an OLMLX_ env prefix, with a sensible default and a one-line docstring. Show the field."}]}
{"id": "impl-streaming-keepalive", "category": "implement_convention", "messages": [{"role": "user", "content": "Implement an async generator that yields server-sent JSON chunks for a chat stream and emits a keepalive every few seconds while the model is still thinking. Follow idiomatic FastAPI streaming."}]}
{"id": "impl-typed-dict", "category": "implement_convention", "messages": [{"role": "user", "content": "Define a TypedDict for a tool-call message where some keys are required and some optional, following the project rule of total=True for required and total=False for optional."}]}
{"id": "qa-typeddict-vs-dict", "category": "convention_qa", "messages": [{"role": "user", "content": "When should a Pydantic v2 model field use dict[str, Any] instead of a TypedDict for free-shape JSON, and why?"}]}
{"id": "qa-asynciterator-annotation", "category": "convention_qa", "messages": [{"role": "user", "content": "An async generator yields both str and dict values. What return annotation should it carry?"}]}
{"id": "qa-error-detail-policy", "category": "convention_qa", "messages": [{"role": "user", "content": "For a single-user localhost inference server, what is a reasonable policy on including internal details (paths, model names) in error messages, and why?"}]}
{"id": "qa-aexit-return-type", "category": "convention_qa", "messages": [{"role": "user", "content": "Why should an async context manager's __aexit__ be annotated to return bool | None rather than None?"}]}
{"id": "impl-tdd-bugfix", "category": "implement_convention", "messages": [{"role": "user", "content": "Describe the workflow for fixing a bug in a TDD codebase: what do you write first, and why?"}]}
{"id": "qa-deepcopy-mlx-dtype", "category": "convention_qa", "messages": [{"role": "user", "content": "Why might a custom KV-cache __deepcopy__ need to share mx.Dtype by reference and eager-eval private dequant buffers before a snapshot crosses threads?"}]}
```

> **Authoring rules for the remaining ~38:** keep each grounded in a real olmlx concept (scan CLAUDE.md "Non-Obvious Invariants", `docs/` specs, and the type-annotation rules) but phrased as a *fresh* question, never lifted verbatim from `data/proxy_tuning/`. Keep ~17 per category. Avoid prompts whose answer requires a precise API name the 1.7B can't supply (proxy-tuning transfers conventions, not facts).

- [ ] **Step 2: Validate the file loads through Task 1's loader**

Run:
```bash
uv run python -c "from olmlx.proxy_tuning_pipeline.eval_schema import load_eval_prompts as L; ps=L('olmlx/proxy_tuning_pipeline/eval_prompts.jsonl'); import collections; c=collections.Counter(p.category for p in ps); print('count:', len(ps)); print('by category:', dict(c))"
```
Expected: `count:` ~50 and roughly balanced across the three categories. No ValueError.

- [ ] **Step 3: Commit** (user reviews before the Task-6 run)

```bash
git add olmlx/proxy_tuning_pipeline/eval_prompts.jsonl
git commit -m "feat(proxy-tuning): held-out Stage-3 eval prompt set"
```

---

## Task 3: Greedy decode driver (TDD)

**Files:**
- Create: `olmlx/proxy_tuning_pipeline/eval_driver.py`
- Test: `tests/test_proxy_tuning_eval.py` (append)

**Context:** `generate_one()` drives one `ProxyTuningDecoder` over one prompt: chat-template the messages → token ids → `mx.array([ids])` → `prefill()` (returns first token) → loop `step()` until an EOS id or `max_tokens` → `decode()`. It takes the decoder and tokenizer as parameters so it's testable with fakes (no Metal). The function never wraps calls in `mx.stream` — the decoder runs on the default stream by contract.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_eval.py`:

```python
from olmlx.proxy_tuning_pipeline.eval_driver import generate_one


class _FakeTok:
    eos_token_ids = {9}

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return [1, 2, 3]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class _FakeDecoder:
    """prefill returns first; step replays a scripted token list."""

    def __init__(self, first, rest):
        self._first = first
        self._rest = list(rest)
        self.alpha = None

    def prefill(self, prompt):
        return self._first

    def step(self):
        return [self._rest.pop(0)], 0


def test_generate_one_stops_on_eos():
    dec = _FakeDecoder(first=5, rest=[6, 7, 9, 8])  # 9 == eos
    out = generate_one(dec, _FakeTok(), [{"role": "user", "content": "hi"}], max_tokens=20)
    assert out == "5 6 7"  # eos (9) excluded, 8 never reached


def test_generate_one_respects_max_tokens():
    dec = _FakeDecoder(first=5, rest=[6, 7, 8, 8, 8])
    out = generate_one(dec, _FakeTok(), [{"role": "user", "content": "hi"}], max_tokens=3)
    assert out == "5 6 7"  # first + 2 steps == 3 tokens


def test_generate_one_empty_on_immediate_eos():
    dec = _FakeDecoder(first=9, rest=[])  # first token is eos
    out = generate_one(dec, _FakeTok(), [{"role": "user", "content": "hi"}], max_tokens=20)
    assert out == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_eval.py -k generate_one -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.proxy_tuning_pipeline.eval_driver'`

- [ ] **Step 3: Write minimal implementation**

Create `olmlx/proxy_tuning_pipeline/eval_driver.py`:

```python
"""Greedy decode driver for the Stage-3 eval (one decoder, one prompt)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def generate_one(
    decoder: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 256,
) -> str:
    """Run the proxy-tuning decoder greedily over one chat prompt.

    Calls run on the default stream (the decoder's contract) — no mx.stream
    wrapper. ``max_tokens`` counts the first token plus subsequent steps.
    """
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    eos = set(tokenizer.eos_token_ids)

    first = decoder.prefill(mx.array([ids]))
    out: list[int] = []
    if first in eos:
        return tokenizer.decode(out)
    out.append(int(first))

    for _ in range(max_tokens - 1):
        toks, _ = decoder.step()
        tok = int(toks[0])
        if tok in eos:
            break
        out.append(tok)

    return tokenizer.decode(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_eval.py -k generate_one -v`
Expected: 3 passed

- [ ] **Step 5: Run ruff, then commit**

Run: `uv run ruff check olmlx/proxy_tuning_pipeline/eval_driver.py tests/test_proxy_tuning_eval.py && uv run ruff format olmlx/proxy_tuning_pipeline/eval_driver.py tests/test_proxy_tuning_eval.py`

```bash
git add olmlx/proxy_tuning_pipeline/eval_driver.py tests/test_proxy_tuning_eval.py
git commit -m "feat(proxy-tuning): Stage-3 greedy decode driver"
```

---

## Task 4: LLM judge (TDD)

**Files:**
- Create: `olmlx/proxy_tuning_pipeline/eval_judge.py`
- Test: `tests/test_proxy_tuning_eval.py` (append)

**Context:** `ProxyEvalJudge` assembles a rubric prompt (the user prompt + the model completion), calls a text generator, and parses `{"convention_adherence": 1-5, "coherence": 1-5, "rationale": "..."}`. It depends on an object with `.generate(system, user) -> str` — Stage 1's `OpenAIGenerator` satisfies this, so the judge takes one in its constructor (inject a fake in tests; default to a real `OpenAIGenerator` in `run_eval`). Scores are clamped to 1–5; a parse failure raises so a bad judge response can't silently score 0.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_eval.py`:

```python
from olmlx.proxy_tuning_pipeline.eval_judge import ProxyEvalJudge


class _FakeGen:
    def __init__(self, reply):
        self.reply = reply
        self.last_system = None
        self.last_user = None

    def generate(self, system, user):
        self.last_system = system
        self.last_user = user
        return self.reply


def test_judge_parses_scores_and_builds_prompt():
    gen = _FakeGen('{"convention_adherence": 4, "coherence": 5, "rationale": "good"}')
    judge = ProxyEvalJudge(gen)
    score = judge.score(
        prompt="Explain the invariant.",
        completion="The invariant is ...",
    )
    assert score == (4, 5, "good")
    # completion + prompt both reach the model
    assert "Explain the invariant." in gen.last_user
    assert "The invariant is ..." in gen.last_user


def test_judge_clamps_out_of_range():
    gen = _FakeGen('{"convention_adherence": 9, "coherence": 0, "rationale": "x"}')
    judge = ProxyEvalJudge(gen)
    conv, coh, _ = judge.score(prompt="p", completion="c")
    assert (conv, coh) == (5, 1)


def test_judge_tolerates_fenced_json():
    gen = _FakeGen('```json\n{"convention_adherence": 3, "coherence": 3, "rationale": "ok"}\n```')
    judge = ProxyEvalJudge(gen)
    assert judge.score(prompt="p", completion="c") == (3, 3, "ok")


def test_judge_raises_on_unparseable():
    judge = ProxyEvalJudge(_FakeGen("not json at all"))
    with pytest.raises(ValueError, match="judge"):
        judge.score(prompt="p", completion="c")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_eval.py -k judge -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.proxy_tuning_pipeline.eval_judge'`

- [ ] **Step 3: Write minimal implementation**

Create `olmlx/proxy_tuning_pipeline/eval_judge.py`:

```python
"""GPT-5.4-mini judge for Stage-3: scores convention-adherence + coherence."""

from __future__ import annotations

import json
import re
from typing import Any

_SYSTEM = (
    "You are a strict code-review judge for the olmlx project (an Apple-MLX, "
    "Ollama-compatible inference server). Score a model's answer to an olmlx "
    "prompt on two axes, each an integer 1-5:\n"
    "- convention_adherence: uses olmlx idioms, respects documented invariants, "
    "matches the project's style and type-annotation rules.\n"
    "- coherence: fluent, on-topic, internally consistent; no degradation.\n"
    "Reply with ONLY a JSON object: "
    '{"convention_adherence": <1-5>, "coherence": <1-5>, "rationale": "<one sentence>"}.'
)

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _clamp(v: Any) -> int:
    return max(1, min(5, int(v)))


class ProxyEvalJudge:
    """Wraps a ``.generate(system, user) -> str`` generator with the rubric."""

    def __init__(self, generator: Any):
        self._gen = generator

    def score(self, *, prompt: str, completion: str) -> tuple[int, int, str]:
        user = (
            f"PROMPT:\n{prompt}\n\n"
            f"MODEL ANSWER:\n{completion}\n\n"
            "Score the answer per the rubric and reply with only the JSON object."
        )
        raw = self._gen.generate(_SYSTEM, user)
        obj = self._parse(raw)
        return (
            _clamp(obj["convention_adherence"]),
            _clamp(obj["coherence"]),
            str(obj.get("rationale", "")),
        )

    @staticmethod
    def _parse(raw: str) -> dict[str, Any]:
        for pat in (_FENCE, _OBJ):
            m = pat.search(raw)
            if m:
                try:
                    return json.loads(m.group(1) if pat is _FENCE else m.group(0))
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"judge returned unparseable response: {raw[:200]!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_eval.py -k judge -v`
Expected: 4 passed

- [ ] **Step 5: Run ruff, then commit**

Run: `uv run ruff check olmlx/proxy_tuning_pipeline/eval_judge.py tests/test_proxy_tuning_eval.py && uv run ruff format olmlx/proxy_tuning_pipeline/eval_judge.py tests/test_proxy_tuning_eval.py`

```bash
git add olmlx/proxy_tuning_pipeline/eval_judge.py tests/test_proxy_tuning_eval.py
git commit -m "feat(proxy-tuning): Stage-3 GPT-5.4-mini judge"
```

---

## Task 5: Aggregation, ship gate, orchestration + CLI (TDD)

**Files:**
- Create: `olmlx/proxy_tuning_pipeline/eval.py`
- Test: `tests/test_proxy_tuning_eval.py` (append)

**Context:** `aggregate()` reduces per-prompt `EvalScore`s to one `AlphaSummary` per α. `ship_decision()` applies the concrete gate: best-α (the α>0 with the highest mean convention) ships iff `best_convention >= base_convention + conv_margin` (default 0.5) AND `best_coherence >= base_coherence - coh_drop` (default 0.2), where base is α=0. `run_eval()` ties it together: load the 3 models **once**, pre-flight `assert_serveable_pair`, build the decoder, sweep α (rebinding `_alpha`), generate + judge each prompt, aggregate, decide, write JSON. Model loading, the decoder, and the judge are injected (defaults are the real ones) so the orchestrator is unit-testable with fakes.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_eval.py`:

```python
from olmlx.proxy_tuning_pipeline.eval import aggregate, ship_decision, run_eval
from olmlx.proxy_tuning_pipeline.eval_schema import EvalScore, EvalPrompt


def _score(alpha, conv, coh, pid="p"):
    return EvalScore(pid, "convention_qa", alpha, conv, coh, "r", "out")


def test_aggregate_means_per_alpha():
    scores = [_score(0.0, 2, 4), _score(0.0, 4, 4), _score(1.0, 5, 4), _score(1.0, 3, 4)]
    summaries = {s.alpha: s for s in aggregate(scores)}
    assert summaries[0.0].mean_convention == 3.0
    assert summaries[1.0].mean_convention == 4.0
    assert summaries[0.0].n == 2


def test_ship_decision_ships_on_clear_lift():
    scores = [_score(0.0, 3, 5), _score(1.0, 4, 5)]  # +1.0 conv, equal coherence
    d = ship_decision(aggregate(scores))
    assert d.ship is True
    assert d.best_alpha == 1.0


def test_ship_decision_blocks_on_coherence_drop():
    # +0.5 conv but coherence falls 5 -> 4.0 (drop 1.0 > 0.2 allowed)
    scores = [_score(0.0, 3, 5), _score(1.0, 3.5 if False else 4, 4)]
    d = ship_decision(aggregate(scores))
    assert d.ship is False
    assert "coherence" in d.reason


def test_ship_decision_blocks_on_insufficient_margin():
    scores = [_score(0.0, 3, 5), _score(1.0, 3, 5)]  # no lift
    d = ship_decision(aggregate(scores))
    assert d.ship is False


def test_run_eval_orchestration_with_fakes(tmp_path, monkeypatch):
    prompts = [EvalPrompt("a", "convention_qa", [{"role": "user", "content": "hi"}])]

    # Fake decoder whose output we don't inspect; fake driver via monkeypatch.
    class _Dec:
        def __init__(self):
            self._alpha = None

    def fake_loader(base, expert, anti):
        return ("BASE", "EXP", "ANTI", "TOK")

    def fake_decoder_factory(base, expert, anti, alpha):
        d = _Dec()
        d._alpha = alpha
        return d

    def fake_generate(decoder, tokenizer, messages, *, max_tokens):
        return f"out@{decoder._alpha}"

    class _Judge:
        def score(self, *, prompt, completion):
            # steered (alpha>0) scores higher to force a ship
            alpha = float(completion.split("@")[1])
            return (5 if alpha > 0 else 3, 5, "r")

    monkeypatch.setattr("olmlx.proxy_tuning_pipeline.eval.generate_one", fake_generate)

    out_path = tmp_path / "results.json"
    report = run_eval(
        base_dir="b", expert_dir="e", antiexpert_dir="a",
        prompts=prompts, alphas=[0.0, 1.0],
        judge=_Judge(), out_path=str(out_path),
        loader=fake_loader, decoder_factory=fake_decoder_factory,
        max_tokens=8, preflight=lambda *a, **k: None,
    )
    assert report.ship is True
    assert out_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_eval.py -k "aggregate or ship_decision or run_eval" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.proxy_tuning_pipeline.eval'`

- [ ] **Step 3: Write minimal implementation**

Create `olmlx/proxy_tuning_pipeline/eval.py`:

```python
"""Stage-3 eval orchestration: aggregate, ship gate, run_eval."""

from __future__ import annotations

import json
from typing import Any, Callable

from olmlx.proxy_tuning_pipeline.eval_driver import generate_one
from olmlx.proxy_tuning_pipeline.eval_schema import (
    AlphaSummary,
    EvalPrompt,
    EvalScore,
    ShipDecision,
)


def aggregate(scores: list[EvalScore]) -> list[AlphaSummary]:
    by_alpha: dict[float, list[EvalScore]] = {}
    for s in scores:
        by_alpha.setdefault(s.alpha, []).append(s)
    out: list[AlphaSummary] = []
    for alpha in sorted(by_alpha):
        group = by_alpha[alpha]
        n = len(group)
        out.append(
            AlphaSummary(
                alpha=alpha,
                n=n,
                mean_convention=sum(s.convention for s in group) / n,
                mean_coherence=sum(s.coherence for s in group) / n,
            )
        )
    return out


def ship_decision(
    summaries: list[AlphaSummary],
    *,
    conv_margin: float = 0.5,
    coh_drop: float = 0.2,
) -> ShipDecision:
    by_alpha = {s.alpha: s for s in summaries}
    base = by_alpha[0.0]
    steered = [s for s in summaries if s.alpha > 0.0]
    best = max(steered, key=lambda s: s.mean_convention)

    conv_ok = best.mean_convention >= base.mean_convention + conv_margin
    coh_ok = best.mean_coherence >= base.mean_coherence - coh_drop
    ship = conv_ok and coh_ok
    if ship:
        reason = (
            f"best α={best.alpha}: convention {best.mean_convention:.2f} ≥ base "
            f"{base.mean_convention:.2f}+{conv_margin}, coherence held."
        )
    elif not conv_ok:
        reason = (
            f"insufficient convention lift at best α={best.alpha}: "
            f"{best.mean_convention:.2f} vs base {base.mean_convention:.2f} "
            f"(need +{conv_margin})."
        )
    else:
        reason = (
            f"coherence dropped at best α={best.alpha}: "
            f"{best.mean_coherence:.2f} vs base {base.mean_coherence:.2f} "
            f"(max drop {coh_drop})."
        )
    return ShipDecision(
        ship=ship,
        best_alpha=best.alpha,
        base_convention=base.mean_convention,
        best_convention=best.mean_convention,
        base_coherence=base.mean_coherence,
        best_coherence=best.mean_coherence,
        conv_margin=conv_margin,
        coh_drop=coh_drop,
        reason=reason,
        per_alpha=summaries,
    )


def _default_loader(base_dir: str, expert_dir: str, antiexpert_dir: str):
    from mlx_lm import load

    base, tok = load(base_dir)
    expert, _ = load(expert_dir)
    anti, _ = load(antiexpert_dir)
    return base, expert, anti, tok


def _default_decoder_factory(base, expert, anti, alpha):
    from olmlx.engine.proxy_tuning import ProxyTuningDecoder

    return ProxyTuningDecoder(base, expert, anti, alpha=alpha)


def _default_preflight(antiexpert_dir: str, expert_dir: str) -> None:
    from olmlx.proxy_tuning_pipeline.verify import assert_serveable_pair

    # Steered base config vocab_size (Qwen3 dense = 151936).
    assert_serveable_pair(antiexpert_dir, expert_dir, 151936)


def run_eval(
    *,
    base_dir: str,
    expert_dir: str,
    antiexpert_dir: str,
    prompts: list[EvalPrompt],
    alphas: list[float],
    judge: Any,
    out_path: str,
    max_tokens: int = 256,
    loader: Callable[..., Any] = _default_loader,
    decoder_factory: Callable[..., Any] = _default_decoder_factory,
    preflight: Callable[..., None] = _default_preflight,
) -> ShipDecision:
    """Load once, sweep α, generate + judge each prompt, aggregate, decide.

    The base+expert+anti-expert load once; per α a fresh decoder is built over
    the same model objects (cheap — caches are per-decoder). Calls run on the
    default stream (decoder contract).
    """
    preflight(antiexpert_dir, expert_dir)
    base, expert, anti, tokenizer = loader(base_dir, expert_dir, antiexpert_dir)

    scores: list[EvalScore] = []
    for alpha in alphas:
        decoder = decoder_factory(base, expert, anti, alpha)
        for p in prompts:
            completion = generate_one(
                decoder, tokenizer, p.messages, max_tokens=max_tokens
            )
            conv, coh, rationale = judge.score(
                prompt=p.messages[-1]["content"], completion=completion
            )
            scores.append(
                EvalScore(p.id, p.category, alpha, conv, coh, rationale, completion)
            )

    summaries = aggregate(scores)
    decision = ship_decision(summaries)

    with open(out_path, "w") as f:
        json.dump(
            {
                "ship": decision.ship,
                "best_alpha": decision.best_alpha,
                "reason": decision.reason,
                "per_alpha": [vars(s) for s in summaries],
                "scores": [vars(s) for s in scores],
            },
            f,
            indent=2,
        )
    return decision
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_eval.py -k "aggregate or ship_decision or run_eval" -v`
Expected: 5 passed

- [ ] **Step 5: Add the `main()` CLI to `eval.py`**

Append to `olmlx/proxy_tuning_pipeline/eval.py` (its own entry point, run as `python -m olmlx.proxy_tuning_pipeline.eval`, mirroring `verify.py` — `cli.py` is not touched):

```python
def main(argv: list[str] | None = None) -> None:
    import argparse

    from olmlx.proxy_tuning_pipeline.eval_judge import ProxyEvalJudge
    from olmlx.proxy_tuning_pipeline.eval_schema import load_eval_prompts
    from olmlx.proxy_tuning_pipeline.expand import OpenAIGenerator

    ap = argparse.ArgumentParser(
        description="Stage-3 proxy-tuning α-sweep eval + ship gate."
    )
    ap.add_argument("--base", required=True, help="dense Qwen3 base model dir")
    ap.add_argument("--expert", required=True, help="M+ (4-bit) dir")
    ap.add_argument("--antiexpert", required=True, help="M- (4-bit) dir")
    ap.add_argument("--prompts", required=True, help="eval_prompts.jsonl path")
    ap.add_argument("--alphas", default="0,0.5,1.0,1.5")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args(argv)

    prompts = load_eval_prompts(args.prompts)
    judge = ProxyEvalJudge(OpenAIGenerator())
    decision = run_eval(
        base_dir=args.base,
        expert_dir=args.expert,
        antiexpert_dir=args.antiexpert,
        prompts=prompts,
        alphas=[float(a) for a in args.alphas.split(",")],
        judge=judge,
        out_path=args.out,
        max_tokens=args.max_tokens,
    )
    print(f"\n=== {'SHIP' if decision.ship else 'NO-SHIP'} ===")
    print(decision.reason)
    for s in decision.per_alpha:
        print(
            f"  α={s.alpha:>4}: convention {s.mean_convention:.2f}  "
            f"coherence {s.mean_coherence:.2f}  (n={s.n})"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify the CLI wires up (no run)**

Run: `uv run python -m olmlx.proxy_tuning_pipeline.eval --help`
Expected: prints usage with `--base/--expert/--antiexpert/--prompts/--alphas/--max-tokens/--out`.

- [ ] **Step 7: Run ruff + the full eval test file, then commit**

Run: `uv run ruff check olmlx/proxy_tuning_pipeline/eval.py tests/test_proxy_tuning_eval.py && uv run ruff format olmlx/proxy_tuning_pipeline/eval.py tests/test_proxy_tuning_eval.py && uv run pytest tests/test_proxy_tuning_eval.py -q`
Expected: all pass.

```bash
git add olmlx/proxy_tuning_pipeline/eval.py tests/test_proxy_tuning_eval.py
git commit -m "feat(proxy-tuning): Stage-3 eval aggregation, ship gate, run_eval + CLI"
```

---

## Task 6: Run the α-sweep eval (operator runbook)

**Files:** none (real run on 8B + the trained pair; writes `eval_results.json`)

> Real Metal activity: loads Qwen3-8B-4bit + M⁺ + M⁻ once and sweeps α over the ~50 prompts, judged by GPT-5.4-mini. Needs `OPENAI_API_KEY` in the environment. Greedy decode → deterministic; re-runs reproduce.

- [ ] **Step 1: Confirm the prompt set (user review)** — read `olmlx/proxy_tuning_pipeline/eval_prompts.jsonl`; edit any prompt that drifts from a real olmlx concept or that demands a precise API name.

- [ ] **Step 2: Run the sweep**

Run:
```bash
export OPENAI_API_KEY=...   # if not already set
uv run python -m olmlx.proxy_tuning_pipeline.eval \
  --base ~/.olmlx/models/mlx-community_Qwen3-8B-4bit \
  --expert ~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-4bit \
  --antiexpert ~/.olmlx/proxy_tuning/qwen3-1.7b-base-4bit \
  --prompts olmlx/proxy_tuning_pipeline/eval_prompts.jsonl \
  --alphas 0,0.5,1.0,1.5 \
  --out ~/.olmlx/proxy_tuning/eval_results.json 2>&1 | tee ~/.olmlx/proxy_tuning/eval.log
```
Expected: a per-α table and a `=== SHIP ===` / `=== NO-SHIP ===` verdict. `eval_results.json` written with per-prompt scores.

- [ ] **Step 3: Read the verdict.** If **SHIP**: note `best_alpha` for Task 8. If **NO-SHIP**: per spec §9, the delta isn't transferring — options are more data diversity (re-run Stage 1/2), a higher α, a larger M± pair, or accepting proxy-tuning's limits for this content. Do **not** register a null result.

- [ ] **Step 4: Manual spot-check** — open `eval_results.json` and read a few base (α=0) vs best-α completions to confirm the judge's scores match your read (the spec requires a manual spot-check alongside the LLM judge).

- [ ] **Step 5: No commit** — `eval_results.json` is an artifact under `~/.olmlx/proxy_tuning/`.

---

## Task 7: 32B spot-check (operator runbook, on ship)

**Files:** none

> The full sweep runs on 8B for cost; the spec asks for a best-α spot-check on a 32B base to confirm the lift holds at the larger scale (decode overhead from the 1.7B pair is ~+11% on 32B).

- [ ] **Step 1: If a dense Qwen3-32B-4bit is available**, run a single-α eval at `best_alpha` on a handful of prompts:

```bash
uv run python -m olmlx.proxy_tuning_pipeline.eval \
  --base <qwen3-32b-4bit-dir> \
  --expert ~/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-4bit \
  --antiexpert ~/.olmlx/proxy_tuning/qwen3-1.7b-base-4bit \
  --prompts olmlx/proxy_tuning_pipeline/eval_prompts.jsonl \
  --alphas 0,<best_alpha> \
  --out ~/.olmlx/proxy_tuning/eval_results_32b.json
```
Expected: the convention lift at `best_alpha` over base holds on 32B (need not match the 8B margin exactly; a clear positive direction is the bar for the spot-check).

- [ ] **Step 2: If no 32B base is available**, record that the spot-check was skipped (no silent omission) and proceed on the 8B result.

- [ ] **Step 3: No commit** — artifact only.

---

## Task 8: Register the pair for serving (operator runbook, on ship)

**Files:** the **live** `olmlx-1` checkout's `.env` + `~/.olmlx/models.json` (NOT this repo — see project memory: the running server is the `olmlx-1` checkout on :11436).

> α is global-only in v1, so registration is global env + a per-model `speculative_strategy` flag on the chosen dense base.

- [ ] **Step 1: Set the proxy env in the live server's `.env`:**

```bash
OLMLX_SPECULATIVE_PROXY_EXPERT_MODEL=/Users/daniel/.olmlx/proxy_tuning/qwen3-1.7b-olmlx-expert-4bit
OLMLX_SPECULATIVE_PROXY_ANTIEXPERT_MODEL=/Users/daniel/.olmlx/proxy_tuning/qwen3-1.7b-base-4bit
OLMLX_SPECULATIVE_PROXY_ALPHA=<best_alpha>
```

- [ ] **Step 2: Mark the dense base as proxy-tuning in `~/.olmlx/models.json`** — on the `mlx-community/Qwen3-8B-4bit` entry, set `"speculative": true, "speculative_strategy": "proxy_tuning"` (mirror the existing speculative entries' shape).

- [ ] **Step 3: Restart the live server** and confirm load: the log shows the proxy-tuning decoder loading expert + anti-expert inline and `check_vocab_identity` passing (no vocab-mismatch error).

- [ ] **Step 4: Smoke a steered completion** over HTTP against :11436 with an olmlx-convention prompt; confirm coherent, on-topic output (sanity, not the gate — the gate was Task 6).

- [ ] **Step 5: No commit** — deployment config lives on the live checkout, not in this repo.

---

## Self-Review

**Spec coverage (against §7 + Resolved decisions):**
- ✅ Held-out prompt set (~50, fresh, 3 categories) — Task 2, loaded/validated by Task 1.
- ✅ α-sweep via **load once, rebind α** on a dense 8B base — Task 5 `run_eval` + Task 3 driver; Task 6 runs it.
- ✅ GPT-5.4-mini judge reusing `OpenAIGenerator`, rubric = convention-adherence + coherence — Task 4.
- ✅ Concrete ship gate (+0.5 convention, coherence ≥ base−0.2) — Task 5 `ship_decision`.
- ✅ Manual spot-check + 32B spot-check — Task 6 Step 4, Task 7.
- ✅ Serve registration (global env + `models.json` flag) on the live checkout — Task 8.
- ✅ Unit-test scaffolding with mocked judge + mocked decoder/generation — Tasks 1,3,4,5; real run is manual (Task 6).
- ✅ Pre-flight reuse of `assert_serveable_pair` — Task 5 `_default_preflight`.

**Placeholder scan:** No TBD/TODO. Task 2 ships 12 concrete prompts + explicit authoring rules to reach ~50 (the one expand-during-execution step, bounded by category counts and grounding rules). Every command is concrete and runnable.

**Type/name consistency:** `EvalPrompt/EvalScore/AlphaSummary/ShipDecision` defined in Task 1 are used unchanged in Tasks 3–5. `generate_one(decoder, tokenizer, messages, *, max_tokens)`, `ProxyEvalJudge.score(*, prompt, completion) -> (int,int,str)`, `aggregate(list[EvalScore]) -> list[AlphaSummary]`, `ship_decision(list[AlphaSummary]) -> ShipDecision`, and `run_eval(...) -> ShipDecision` are consistent across definition, tests, and the CLI handler. The driver/judge/loader injection points in `run_eval` match the fakes in the Task-5 test.

**Execution-mode note:** Tasks 1–5 are TDD code (agent-runnable). Tasks 6–8 are an operator runbook — real Metal + the trained pair, GPT-5.4-mini calls, and live-server config; the operator runs them and pastes results back, as in Stages 1–2.
