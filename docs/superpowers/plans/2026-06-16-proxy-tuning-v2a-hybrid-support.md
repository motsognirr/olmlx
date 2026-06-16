# Proxy-Tuning v2a — Hybrid Decode-Path Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document and lock in (via a regression test) that `ProxyTuningDecoder` correctly steers a **hybrid GatedDeltaNet (linear-attention) base** — no algorithmic change needed.

**Architecture:** Verification-first. The correctness hypothesis was **already proven empirically during planning** (see below), so v2a is just: lift the "dense-only" documentation and add a Metal-gated regression test that guards the behavior. The §3.2 single-chunk-prefill contingency from the spec is **dropped** — the long-prompt smoke matched, so it's unneeded (YAGNI).

**Tech Stack:** MLX, `mlx_lm` (`load`, `make_prompt_cache`), `olmlx.engine.proxy_tuning.ProxyTuningDecoder`, pytest with `@pytest.mark.real_model` + Metal gating.

**Verification evidence (run during planning, 2026-06-16):** Loaded `Qwen3.5-0.8B-MLX-4bit` (qwen3_5, hybrid GDN) and passed it as base=expert=anti-expert to `ProxyTuningDecoder` (identical trio → `M⁺−M⁻=0` → combined logits == base). Compared 40-token greedy proxy output to a single-call-prefill reference greedy loop:
- SHORT prompt (25 tok, 1-chunk prefill): **exact match**.
- LONG prompt (3624 tok, **multi-chunk** prefill — exercises GDN recurrent-state threading across chunks): **exact match**.
Conclusion: the decoder + chunked prefill handle GDN state correctly; no decoder code change required.

**Scope note:** the 0.8B is hybrid-attention but **non-MoE**. v2a verifies GDN-attention handling; MoE remains transparent-by-design (the decoder takes only final-position logits) and is exercised later in v2c (Coder-Next) and the eval in v2b (Qwen3.5-27B).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `olmlx/engine/proxy_tuning.py` | Drop "dense-only" claims from module + class docstrings; state hybrid GDN/MoE is supported and why GDN capture is unneeded | Modify |
| `CLAUDE.md` | Update the proxy-tuning invariant paragraph (dense **and** hybrid supported) | Modify |
| `tests/test_proxy_tuning_hybrid.py` | Metal-gated, `real_model` regression test: identical-trio correctness on Qwen3.5-0.8B, short + long prompt | Create |

---

## Task 1: Lift the dense-only documentation

**Files:**
- Modify: `olmlx/engine/proxy_tuning.py` (module docstring ~lines 20-24; class docstring ~lines 123-125)
- Modify: `CLAUDE.md` (proxy-tuning invariant paragraph)

No tests (documentation only). Verify by re-reading + ruff.

- [ ] **Step 1: Update the module docstring in `olmlx/engine/proxy_tuning.py`**

Replace this block:
```python
v1 targets **dense** model families only (Qwen2.5/Qwen3 dense, Llama 3.x).
Hybrid linear-attention / GDN families are out of scope — this decoder installs
no ``GDNStateCapture`` and runs every forward on the default stream (the same
stream the speculative path decodes on).
```
with:
```python
Supports **dense and hybrid GatedDeltaNet (linear-attention) / MoE** bases
(Qwen2.5/Qwen3 dense, Qwen3-Next/Qwen3.5 hybrid, Llama 3.x). Unlike the
speculative decoders, proxy-tuning needs **no** ``GDNStateCapture``: that
machinery exists only to roll back rejected draft tokens, and proxy-tuning
never speculates (no draft -> verify -> accept, no cache trimming — every model
advances one token per step over the same committed sequence). Per-token
forward + ``make_prompt_cache`` maintains GDN recurrent state correctly, and
everything runs on the default stream, so the GDN cross-stream hazard does not
apply. MoE routing is internal to the forward; only final-position logits are
used. All three models must still share one exact tokenizer/vocabulary.
```

- [ ] **Step 2: Update the class docstring in `olmlx/engine/proxy_tuning.py`**

Replace:
```python
    Installs **no** layer hooks, **no** draft bind, and **no** GDN capture
    (dense-only v1), so the base-class ``reset()`` teardown is effectively a
    no-op beyond clearing the caches in :meth:`_reset_state`.
```
with:
```python
    Installs **no** layer hooks, **no** draft bind, and **no** GDN capture
    (proxy-tuning never rejects tokens, so rollback capture is unneeded — this
    is why hybrid GDN/MoE bases work without it), so the base-class ``reset()``
    teardown is effectively a no-op beyond clearing the caches in
    :meth:`_reset_state`.
```

- [ ] **Step 3: Update the proxy-tuning invariant in `CLAUDE.md`**

Find the sentence ending the proxy-tuning invariant paragraph:
```
v1 is **dense-only** (no GDN capture installed); grammar disables it per-request like every other speculative strategy.
```
Replace with:
```
Supports **dense and hybrid GDN/MoE** bases: it installs no GDN capture because that only undoes rejected draft tokens and proxy-tuning never speculates (no draft→verify→accept, no trimming), so per-token forward + `make_prompt_cache` keeps GDN recurrent state correct on the default stream; verified on a hybrid Qwen3.5 base (`tests/test_proxy_tuning_hybrid.py`). Grammar disables it per-request like every other speculative strategy.
```

- [ ] **Step 4: Verify + commit**

Run: `uv run ruff check olmlx/engine/proxy_tuning.py` (expect clean) and re-read the two docstrings to confirm they read correctly.

```bash
git add olmlx/engine/proxy_tuning.py CLAUDE.md
git commit -m "docs(proxy-tuning): support hybrid GDN/MoE bases (no rollback => no GDN capture)"
```

---

## Task 2: Metal-gated regression test for hybrid base support

**Files:**
- Create: `tests/test_proxy_tuning_hybrid.py`

**Context:** GDN state correctness can't be checked with fakes — it needs a real hybrid model. This test mirrors the planning smoke and follows the repo's `@pytest.mark.real_model` convention (conftest blocks real loads without it; CI deselects `-m "not real_model"`). It also skips when Metal is unavailable or the model can't be fetched, so collection never touches the network and non-Metal envs skip cleanly.

- [ ] **Step 1: Write the test**

Create `tests/test_proxy_tuning_hybrid.py`:

```python
"""Regression test: proxy-tuning steers a hybrid GatedDeltaNet base correctly.

The decoder is dense+hybrid capable (proxy-tuning never rejects tokens, so it
needs no GDN rollback capture). With an identical base/expert/anti-expert trio,
M+ - M- == 0, so the combined logits equal the base's — the proxy output must
therefore match a plain single-call-prefill greedy generation of the same
hybrid model, even on a >2048-token prompt that forces multi-chunk prefill
(the path that exercises GDN recurrent-state threading across chunks).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

_HYBRID = "mlx-community/Qwen3.5-0.8B-MLX-4bit"  # qwen3_5, hybrid GDN, non-MoE


def _model_dir() -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(_HYBRID))
    except Exception:
        return None


def _reference_greedy(model, ids, n):
    """Single-call-prefill greedy decode — the known-correct GDN baseline."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)[0, -1, :]
    out = []
    tok = int(mx.argmax(logits).item())
    for _ in range(n):
        out.append(tok)
        logits = model(mx.array([[tok]]), cache=cache)[0, -1, :]
        tok = int(mx.argmax(logits).item())
    return out


def _proxy_greedy(model, ids, n):
    from olmlx.engine.proxy_tuning import ProxyTuningDecoder

    dec = ProxyTuningDecoder(model, model, model, alpha=1.0)
    first = dec.prefill(mx.array([ids]))
    out = [first]
    for _ in range(n - 1):
        toks, _ = dec.step()
        out.append(toks[0])
    return out


@pytest.mark.real_model
@pytest.mark.parametrize(
    "kind, n_repeat",
    [("short_single_chunk", 0), ("long_multi_chunk", 400)],
)
def test_proxy_tuning_matches_reference_on_hybrid_base(kind, n_repeat):
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    path = _model_dir()
    if path is None:
        pytest.skip(f"{_HYBRID} not downloadable")

    from mlx_lm import load

    model, tok = load(str(path))

    if n_repeat:
        content = (
            "The transformer architecture processes tokens through attention "
            "layers. " * n_repeat
        ) + " Summarize the above in one sentence. /no_think"
    else:
        content = "Explain what a KV cache is in two sentences. /no_think"
    ids = tok.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True
    )
    if n_repeat:
        assert len(ids) > 2048, "long prompt must exceed the 2048-token prefill chunk"

    ref = _reference_greedy(model, ids, 40)
    proxy = _proxy_greedy(model, ids, 40)
    assert proxy == ref, (
        f"[{kind}] proxy output diverged from single-call-prefill reference on a "
        f"hybrid GDN base — GDN state likely mishandled. "
        f"ref={tok.decode(ref)[:120]!r} proxy={tok.decode(proxy)[:120]!r}"
    )
```

- [ ] **Step 2: Run the test (Metal machine)**

Run: `uv run pytest tests/test_proxy_tuning_hybrid.py -v -m real_model`
Expected: 2 passed (`short_single_chunk`, `long_multi_chunk`). On a non-Metal box or without the model, both skip.

- [ ] **Step 3: Confirm CI-path collection is clean (no real load without Metal/marker)**

Run: `uv run pytest tests/test_proxy_tuning_hybrid.py -m "not real_model" -q`
Expected: 0 selected / all deselected (the marker keeps it out of the CI default set; conftest's `block_real_model_loads` is satisfied because the test carries the marker).

- [ ] **Step 4: Ruff + commit**

Run: `uv run ruff check tests/test_proxy_tuning_hybrid.py && uv run ruff format tests/test_proxy_tuning_hybrid.py`

```bash
git add tests/test_proxy_tuning_hybrid.py
git commit -m "test(proxy-tuning): Metal-gated regression for hybrid GDN base support"
```

---

## Self-Review

**Spec coverage (against §3 of the v2 spec):**
- ✅ §3.1 correctness smoke — executed during planning (evidence in the header); short + long prompt both matched.
- ✅ §3.2 contingency — correctly **dropped** (long-prompt smoke matched → single-chunk prefill unneeded).
- ✅ §3.3 lift constraint + docs — Task 1 (proxy_tuning.py module + class docstrings, CLAUDE.md).
- ✅ §3.4 regression test — Task 2 (Metal-gated, `real_model`, identical-trio, short + long).
- ✅ §3.6 no v1 regressions — `tests/ -k proxy_tuning` stays green (unchanged decoder); run it after Task 2.

**Placeholder scan:** none — both tasks have concrete edits/code and exact commands.

**Type/name consistency:** `ProxyTuningDecoder(base, expert, antiexpert, *, alpha=1.0)` and `prefill(mx.array)->int` / `step()->(list,int)` match the real signatures used in the planning smoke. Test model id `mlx-community/Qwen3.5-0.8B-MLX-4bit` matches the verified local artifact.

**Execution-mode note:** v2a is small (docs + one Metal-gated test) and the core verification is already done, so inline execution is reasonable; subagent-driven also fine.
