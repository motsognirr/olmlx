# Proxy-Tuning Decode Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a decode-time logit-arithmetic "proxy-tuning" mode (Liu et al. 2024) to olmlx — steer a large base model `M` with a small tuned expert `M⁺` and small untuned anti-expert `M⁻`, combining per step as `logits = base + α·(expert − antiexpert)`, then greedy-sampling.

**Architecture:** Implement `ProxyTuningDecoder` as a fresh `SpecDecoderBase` subclass holding three `(model, cache)` pairs with no draft-bind, no GDN capture, and no cache trimming. Register it as a **new `speculative_strategy` value `"proxy_tuning"`** so it inherits the entire existing wiring: `lm.is_speculative` gating, the streaming/non-streaming dispatch in `inference.py`, the `speculative_stream_generate`/`async_speculative_stream` bridge (which accepts any `prefill()→int` + `step()→(list[int],int)` protocol), the grammar mutual-exclusion gate, and the greedy sampling-param drop. The expert/anti-expert are loaded **inline by the loader** (exactly like today's speculative draft model — held by the decoder, run in the decoder's own loop on `default_stream`), so they never enter `ModelManager._loaded` and need no `max_loaded_models` bump or `ensure_loaded` pinning.

**Tech Stack:** Python 3, MLX (`mlx.core`, `mlx.nn`), mlx-lm (`make_prompt_cache`), pydantic-settings (config), pytest (TDD).

**Scope notes / deliberate v1 boundaries:**
- **Dense models only** (Qwen2.5/Qwen3 dense, Llama 3.x). Hybrid/GDN families (Qwen3.5/Next, Gemma 3) are out — the shared `GDNStateCapture` raises on mixed GDN classes, and dense targets avoid the Metal-stream recurrent-state hazard entirely. We install **no** GDN capture.
- **Global-settings / env-driven config** for the two extra model paths and α (`OLMLX_SPECULATIVE_PROXY_EXPERT_MODEL`, `OLMLX_SPECULATIVE_PROXY_ANTIEXPERT_MODEL`, `OLMLX_SPECULATIVE_PROXY_ALPHA`). Per-model `models.json` overrides of these three fields are a documented follow-up (the strategy itself is already selectable per-model via `speculative_strategy`).
- **Grammar/JSON-mode disables proxy-tuning** for that request, identical to every other speculative strategy today (the existing gate handles this with zero new code). "Grammar applied after combination" is a future enhancement.
- **No CLI subcommand** — proxy-tuning needs no training step; it's configured via env/`models.json` like `pld`/`classic`.

**Why this differs from issue #523's "tightened scope":** the issue recommended panel-style `ensure_loaded(pin=True)` + `max_loaded_models ≥ 3`. Code audit showed loaders run *inside* `_load_model` under the manager lock (cannot re-enter `ensure_loaded`), and that the speculative draft already proves inline-loaded auxiliary models run correctly in the decoder's own `default_stream` loop. Inline loading is therefore simpler and strictly correct — adopted here.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `olmlx/engine/proxy_tuning.py` | `combine_proxy_logits()`, `check_vocab_identity()`, `ProxyTuningDecoder(SpecDecoderBase)` | **Create** |
| `olmlx/config.py` | Add `"proxy_tuning"` strategy + three `Settings` fields + a model-validator | Modify |
| `olmlx/engine/registry.py` | Add `"proxy_tuning"` to strategy literals + flash-moe incompat; extend `SpeculativeConfig`; populate in `resolved_speculative()` | Modify |
| `olmlx/engine/speculative_loaders.py` | `_load_proxy_tuning_decoder()` | Modify |
| `olmlx/engine/model_manager.py` | Add `proxy_tuning` branch to the strategy dispatch ladder | Modify |
| `CLAUDE.md` | Document the new decode mode + invariants | Modify |
| `tests/test_proxy_tuning.py` | Decoder + combine + vocab-guard unit tests | **Create** |
| `tests/test_proxy_tuning_config.py` | Config/registry resolution + loader-guard tests | **Create** |

---

## Phase 1 — Decoder, combiner, vocab guard (standalone, fully unit-testable)

### Task 1: `combine_proxy_logits` pure function

**Files:**
- Create: `olmlx/engine/proxy_tuning.py`
- Test: `tests/test_proxy_tuning.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_proxy_tuning.py`:

```python
"""Tests for proxy-tuning decode mode (engine/proxy_tuning.py)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from olmlx.engine.proxy_tuning import (
    ProxyTuningDecoder,
    check_vocab_identity,
    combine_proxy_logits,
)


def test_combine_proxy_logits_basic():
    base = mx.array([1.0, 2.0, 3.0])
    expert = mx.array([0.0, 5.0, 0.0])
    antiexpert = mx.array([0.0, 1.0, 0.0])
    # base + alpha*(expert - antiexpert) with alpha=1.0
    out = combine_proxy_logits(base, expert, antiexpert, 1.0)
    assert out.tolist() == [1.0, 6.0, 3.0]


def test_combine_proxy_logits_alpha_scales_delta():
    base = mx.array([0.0, 0.0])
    expert = mx.array([0.0, 4.0])
    antiexpert = mx.array([0.0, 0.0])
    out = combine_proxy_logits(base, expert, antiexpert, 0.5)
    assert out.tolist() == [0.0, 2.0]


def test_combine_proxy_logits_alpha_zero_is_base():
    base = mx.array([7.0, -3.0])
    expert = mx.array([100.0, 100.0])
    antiexpert = mx.array([-100.0, -100.0])
    out = combine_proxy_logits(base, expert, antiexpert, 0.0)
    assert out.tolist() == [7.0, -3.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning.py::test_combine_proxy_logits_basic -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.engine.proxy_tuning'`

- [ ] **Step 3: Write minimal implementation**

Create `olmlx/engine/proxy_tuning.py`:

```python
"""Proxy-Tuning decode mode (Liu et al. 2024, *Tuning Language Models by Proxy*).

Steers a large base model ``M`` at decode time — without touching its weights
— using a small tuned expert ``M⁺`` and small untuned anti-expert ``M⁻``
(``M⁺`` is ``M⁻`` fine-tuned). Each decode step combines per-token logits as::

    logits = base + alpha * (expert - antiexpert)

then samples greedily. The learned "tuning delta" from the cheap small models
is transplanted onto the big model.

Unlike speculative decoding (which is *exactness-preserving*), proxy-tuning
*deliberately alters* the output distribution, so it cannot reuse the
draft -> verify -> accept logic. It is registered as a sibling
``speculative_strategy`` ("proxy_tuning") only to reuse the mechanical
plumbing (lifecycle, stream bridge, dispatch); the algorithm is its own.

Hard constraint: all three models must share one exact tokenizer / vocabulary.

v1 targets **dense** model families only (Qwen2.5/Qwen3 dense, Llama 3.x).
Hybrid linear-attention / GDN families are out of scope — this decoder installs
no ``GDNStateCapture`` and runs every forward on the default stream (the same
stream the speculative path decodes on).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import mlx.core as mx

from olmlx.engine.spec_decoder_base import SpecDecoderBase

logger = logging.getLogger(__name__)


def combine_proxy_logits(
    base: mx.array,
    expert: mx.array,
    antiexpert: mx.array,
    alpha: float,
) -> mx.array:
    """Proxy-tuning logit arithmetic: ``base + alpha * (expert - antiexpert)``.

    All three arrays are ``(vocab,)`` last-position logits from the three
    models forwarded over the same token. ``alpha`` scales the tuning delta;
    ``alpha == 0`` reduces to the unsteered base distribution.
    """
    return base + alpha * (expert - antiexpert)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning.py -k combine -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/proxy_tuning.py tests/test_proxy_tuning.py
git commit -m "feat(proxy-tuning): combine_proxy_logits logit arithmetic"
```

---

### Task 2: `check_vocab_identity` guard

**Files:**
- Modify: `olmlx/engine/proxy_tuning.py`
- Test: `tests/test_proxy_tuning.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning.py`:

```python
class _FakeTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def test_check_vocab_identity_passes_on_match():
    tok_a = _FakeTokenizer({"a": 0, "b": 1, "c": 2})
    tok_b = _FakeTokenizer({"a": 0, "b": 1, "c": 2})
    # No raise == pass.
    check_vocab_identity(tok_a, tok_b, reference_label="base", other_label="expert")


def test_check_vocab_identity_raises_on_token_mapping_diff():
    # Same SIZE, different mapping — the exact case vocab_size-only checks miss.
    tok_a = _FakeTokenizer({"a": 0, "b": 1, "c": 2})
    tok_b = _FakeTokenizer({"a": 0, "b": 2, "c": 1})
    with pytest.raises(ValueError, match="vocab"):
        check_vocab_identity(
            tok_a, tok_b, reference_label="base", other_label="expert"
        )


def test_check_vocab_identity_warns_when_unavailable(caplog):
    class _NoVocab:
        pass

    # Missing get_vocab() -> warn-and-return (loader's vocab_size check is the floor).
    with caplog.at_level("WARNING"):
        check_vocab_identity(
            _NoVocab(), _NoVocab(), reference_label="base", other_label="expert"
        )
    assert any("vocab" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning.py -k vocab_identity -v`
Expected: FAIL with `ImportError: cannot import name 'check_vocab_identity'`

- [ ] **Step 3: Write minimal implementation**

Add to `olmlx/engine/proxy_tuning.py` (after `combine_proxy_logits`):

```python
def _safe_get_vocab(tokenizer: Any) -> dict[str, int] | None:
    """Return ``tokenizer.get_vocab()`` as a dict, or ``None`` if unavailable.

    mlx-lm's ``TokenizerWrapper`` forwards attribute access to the underlying
    HuggingFace tokenizer, so ``get_vocab()`` works on the wrapper too. Any
    failure (no method, raises) yields ``None`` so the caller can fall back to
    the loader's ``vocab_size`` check rather than hard-failing on an exotic
    tokenizer.
    """
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        return None
    try:
        vocab = get_vocab()
    except Exception:  # noqa: BLE001 — any failure -> fall back to size check
        return None
    return vocab if isinstance(vocab, dict) else None


def check_vocab_identity(
    reference_tokenizer: Any,
    other_tokenizer: Any,
    *,
    reference_label: str,
    other_label: str,
) -> None:
    """Raise ``ValueError`` if two tokenizers map tokens differently.

    Proxy-tuning adds logits across models position-by-position; a token id
    that means different things in two models silently corrupts the output.
    This is stricter than a ``vocab_size`` comparison (two vocabularies can
    match on size yet differ in mapping). When either tokenizer does not expose
    ``get_vocab()``, this warns and returns — the loader's ``vocab_size`` guard
    is the hard floor in that case.
    """
    ref_vocab = _safe_get_vocab(reference_tokenizer)
    other_vocab = _safe_get_vocab(other_tokenizer)
    if ref_vocab is None or other_vocab is None:
        logger.warning(
            "Could not verify tokenizer/vocab identity for proxy-tuning "
            "(%s or %s tokenizer has no usable get_vocab()); relying on the "
            "vocab_size check only. A token-mapping mismatch would corrupt "
            "the combined logits.",
            reference_label,
            other_label,
        )
        return
    if ref_vocab != other_vocab:
        raise ValueError(
            f"Proxy-tuning requires identical vocabularies: the {other_label} "
            f"tokenizer's token->id mapping differs from the {reference_label} "
            f"tokenizer's (sizes {len(other_vocab)} vs {len(ref_vocab)}). All "
            f"three models (base, expert, anti-expert) must share one exact "
            f"tokenizer — use models from the same family and tokenizer revision."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning.py -k vocab_identity -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/proxy_tuning.py tests/test_proxy_tuning.py
git commit -m "feat(proxy-tuning): tokenizer-identity vocab guard"
```

---

### Task 3: `ProxyTuningDecoder` — construction + reset

**Files:**
- Modify: `olmlx/engine/proxy_tuning.py`
- Test: `tests/test_proxy_tuning.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning.py`:

```python
class _StubModel(nn.Module):
    """Returns a fixed last-position logit vector on every forward.

    ``make_cache`` returns ``[]`` so ``make_prompt_cache(model)`` produces an
    empty cache the stub ignores — keeps the test off real attention/KV code.
    The decoder only reads ``out[0, -1, :]``, so broadcasting the fixed vector
    across all positions is sufficient and deterministic.
    """

    def __init__(self, vocab_size: int, logit_vec: mx.array):
        super().__init__()
        self._vocab_size = vocab_size
        self._logit_vec = logit_vec
        self.calls = 0

    def make_cache(self) -> list:
        return []

    def __call__(self, tokens: mx.array, cache: Any = None) -> mx.array:
        self.calls += 1
        seq = tokens.shape[1]
        return mx.broadcast_to(
            self._logit_vec.reshape(1, 1, -1), (1, seq, self._vocab_size)
        )


def _make_decoder(vocab=4, base=None, expert=None, anti=None, alpha=1.0):
    base = base if base is not None else mx.zeros((vocab,))
    expert = expert if expert is not None else mx.zeros((vocab,))
    anti = anti if anti is not None else mx.zeros((vocab,))
    return ProxyTuningDecoder(
        base_model=_StubModel(vocab, base),
        expert_model=_StubModel(vocab, expert),
        antiexpert_model=_StubModel(vocab, anti),
        alpha=alpha,
    )


def test_decoder_construction_sets_target_and_alpha():
    dec = _make_decoder(alpha=2.0)
    assert dec._alpha == 2.0
    # _target must be the base model (base teardown reference; never patched).
    assert dec._target is dec._base
    assert dec._patched is False
    assert dec._bound is False
    assert dec._capture is None


def test_reset_clears_caches_and_pending():
    dec = _make_decoder()
    dec._base_cache = ["x"]
    dec._expert_cache = ["y"]
    dec._antiexpert_cache = ["z"]
    dec._pending_token = 3
    dec.reset()
    assert dec._base_cache is None
    assert dec._expert_cache is None
    assert dec._antiexpert_cache is None
    assert dec._pending_token is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning.py -k "construction or reset_clears" -v`
Expected: FAIL with `ImportError: cannot import name 'ProxyTuningDecoder'`

- [ ] **Step 3: Write minimal implementation**

Add to `olmlx/engine/proxy_tuning.py` (after the guard functions):

```python
class ProxyTuningDecoder(SpecDecoderBase):
    """Decode-time logit-arithmetic steering with three models.

    Holds three independent ``(model, KV-cache)`` pairs — base, expert,
    anti-expert. Each decode step forwards the single pending token through all
    three, combines their last-position logits via :func:`combine_proxy_logits`,
    and greedily argmaxes the result. Every model advances over the same
    committed token sequence, so the caches stay aligned with no trimming.

    Installs **no** layer hooks, **no** draft bind, and **no** GDN capture
    (dense-only v1), so the base-class ``reset()`` teardown is effectively a
    no-op beyond clearing the caches in :meth:`_reset_state`.

    Implements the speculative decoder protocol (``prefill() -> int``,
    ``step() -> (list[int], int)``) so it slots into ``speculative_stream``
    unchanged. ``num_draft`` is always 0 (proxy-tuning does not speculate; one
    token per step at ``base + 2*small`` forward cost).

    Not thread-safe: one decoder instance serves one request at a time.
    """

    def __init__(
        self,
        base_model: Any,
        expert_model: Any,
        antiexpert_model: Any,
        *,
        alpha: float = 1.0,
    ):
        super().__init__()
        self._base = base_model
        self._expert = expert_model
        self._antiexpert = antiexpert_model
        # Base teardown reference for the inherited reset() path. We never call
        # _install_layer_hooks/_bind_draft, so _patched/_bound stay False and
        # reset() never actually touches _target — but set it for correctness.
        self._target = base_model
        self._alpha = float(alpha)

        # Per-request state (populated by prefill, cleared by _reset_state).
        self._base_cache: list | None = None
        self._expert_cache: list | None = None
        self._antiexpert_cache: list | None = None
        self._pending_token: int | None = None

    def _reset_state(self) -> None:
        self._base_cache = None
        self._expert_cache = None
        self._antiexpert_cache = None
        self._pending_token = None

    def _stats_extra(self) -> dict[str, Any]:
        return {"alpha": self._alpha}

    def _prefill_impl(
        self,
        prompt: mx.array,
        *,
        segmented: Any = None,
        cancel_event: threading.Event | None = None,
    ) -> int:
        raise NotImplementedError  # implemented in Task 4

    def _step_impl(self) -> tuple[list[int], int]:
        raise NotImplementedError  # implemented in Task 5
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning.py -k "construction or reset_clears" -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/proxy_tuning.py tests/test_proxy_tuning.py
git commit -m "feat(proxy-tuning): ProxyTuningDecoder scaffold (init + reset)"
```

---

### Task 4: `ProxyTuningDecoder._prefill_impl`

**Files:**
- Modify: `olmlx/engine/proxy_tuning.py`
- Test: `tests/test_proxy_tuning.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning.py`:

```python
def test_prefill_returns_combined_argmax():
    # base favors idx 0, expert favors idx 2, antiexpert favors idx 1.
    # combined = base + (expert - antiexpert):
    #   idx0: 3 + (0-0) = 3
    #   idx1: 0 + (0-5) = -5
    #   idx2: 0 + (5-0) = 5   <- argmax
    base = mx.array([3.0, 0.0, 0.0])
    expert = mx.array([0.0, 0.0, 5.0])
    anti = mx.array([0.0, 5.0, 0.0])
    dec = _make_decoder(vocab=3, base=base, expert=expert, anti=anti, alpha=1.0)
    prompt = mx.array([[7, 8, 9]])  # any 3-token prompt
    first = dec.prefill(prompt)
    assert first == 2
    assert dec._pending_token == 2
    # All three caches must be populated.
    assert dec._base_cache is not None
    assert dec._expert_cache is not None
    assert dec._antiexpert_cache is not None


def test_prefill_single_token_prompt():
    base = mx.array([0.0, 9.0])
    dec = _make_decoder(vocab=2, base=base)
    first = dec.prefill(mx.array([[5]]))
    assert first == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning.py -k prefill -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

In `olmlx/engine/proxy_tuning.py`, add these imports near the top (below the existing imports):

```python
from mlx_lm.models.cache import make_prompt_cache

from olmlx.engine.speculative import _eval_cache, _prefill_last_logit
```

Then replace the `_prefill_impl` body:

```python
    def _prefill_impl(
        self,
        prompt: mx.array,
        *,
        segmented: Any = None,
        cancel_event: threading.Event | None = None,
    ) -> int:
        """Prefill all three models over ``prompt`` and return the first token.

        Each model gets its own fresh KV cache. ``_prefill_last_logit`` (reused
        from the speculative module) sub-chunks the prefix so a long prompt
        cannot OOM Metal, and returns the model's final-position ``(vocab,)``
        logit. The three logits are combined and argmaxed for the first token.
        ``segmented`` is accepted and ignored — proxy-tuning has no cross-request
        snapshot store in v1.

        Runs on the default stream (no ``mx.stream`` wrapper), the same stream
        ``step()`` decodes on — consistent with the speculative path's
        single-stream invariant.
        """
        self._base_cache = make_prompt_cache(self._base)
        self._expert_cache = make_prompt_cache(self._expert)
        self._antiexpert_cache = make_prompt_cache(self._antiexpert)

        base_logit = _prefill_last_logit(
            self._base, prompt, self._base_cache, cancel_event=cancel_event
        )
        expert_logit = _prefill_last_logit(
            self._expert, prompt, self._expert_cache, cancel_event=cancel_event
        )
        antiexpert_logit = _prefill_last_logit(
            self._antiexpert, prompt, self._antiexpert_cache, cancel_event=cancel_event
        )
        combined = combine_proxy_logits(
            base_logit, expert_logit, antiexpert_logit, self._alpha
        )
        # Materialize the combined logit and every cache's final-token write
        # before decode begins — keeps the per-model lazy graphs from chaining
        # across step() boundaries.
        first_token = int(mx.argmax(combined).item())
        _eval_cache(self._base_cache)
        _eval_cache(self._expert_cache)
        _eval_cache(self._antiexpert_cache)

        self._pending_token = first_token
        return first_token
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning.py -k prefill -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/proxy_tuning.py tests/test_proxy_tuning.py
git commit -m "feat(proxy-tuning): _prefill_impl with three-model logit combine"
```

---

### Task 5: `ProxyTuningDecoder._step_impl`

**Files:**
- Modify: `olmlx/engine/proxy_tuning.py`
- Test: `tests/test_proxy_tuning.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning.py`:

```python
def test_step_returns_combined_argmax():
    base = mx.array([3.0, 0.0, 0.0])
    expert = mx.array([0.0, 0.0, 5.0])
    anti = mx.array([0.0, 5.0, 0.0])
    dec = _make_decoder(vocab=3, base=base, expert=expert, anti=anti, alpha=1.0)
    dec.prefill(mx.array([[7, 8, 9]]))
    accepted, num_draft = dec.step()
    assert accepted == [2]
    assert num_draft == 0
    assert dec._pending_token == 2
    assert dec._stats_steps == 1


def test_step_before_prefill_raises():
    dec = _make_decoder()
    with pytest.raises(RuntimeError, match="prefill"):
        dec.step()


def test_alpha_changes_winner():
    # With alpha=0 the base wins (idx0); with alpha=1 the delta flips it to idx2.
    base = mx.array([1.0, 0.0, 0.0])
    expert = mx.array([0.0, 0.0, 10.0])
    anti = mx.array([0.0, 0.0, 0.0])
    dec0 = _make_decoder(vocab=3, base=base, expert=expert, anti=anti, alpha=0.0)
    assert dec0.prefill(mx.array([[1, 2]])) == 0
    dec1 = _make_decoder(vocab=3, base=base, expert=expert, anti=anti, alpha=1.0)
    assert dec1.prefill(mx.array([[1, 2]])) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning.py -k "step or alpha_changes" -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

In `olmlx/engine/proxy_tuning.py`, add `_logits` to the speculative import:

```python
from olmlx.engine.speculative import _eval_cache, _logits, _prefill_last_logit
```

Then replace the `_step_impl` body:

```python
    def _step_impl(self) -> tuple[list[int], int]:
        """One proxy-tuning decode step: forward the pending token through all
        three models, combine logits, greedy-argmax the next token.

        Returns ``([next_token], 0)`` — proxy-tuning emits exactly one token per
        step and never speculates (``num_draft == 0``). Must be called after
        :meth:`prefill`.
        """
        if self._base_cache is None or self._pending_token is None:
            raise RuntimeError(
                "ProxyTuningDecoder.step() called before prefill(); "
                "call prefill(prompt) first"
            )

        tok = mx.array([[self._pending_token]])
        base_logit = _logits(self._base(tok, cache=self._base_cache))[0, -1, :]
        expert_logit = _logits(self._expert(tok, cache=self._expert_cache))[0, -1, :]
        antiexpert_logit = _logits(
            self._antiexpert(tok, cache=self._antiexpert_cache)
        )[0, -1, :]

        combined = combine_proxy_logits(
            base_logit, expert_logit, antiexpert_logit, self._alpha
        )
        next_token = int(mx.argmax(combined).item())

        self._stats_steps += 1
        self._pending_token = next_token
        return [next_token], 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning.py -v`
Expected: all tests pass (combine + vocab + construction + reset + prefill + step + alpha)

- [ ] **Step 5: Run ruff, then commit**

Run: `uv run ruff check olmlx/engine/proxy_tuning.py tests/test_proxy_tuning.py && uv run ruff format olmlx/engine/proxy_tuning.py tests/test_proxy_tuning.py`
Expected: no errors

```bash
git add olmlx/engine/proxy_tuning.py tests/test_proxy_tuning.py
git commit -m "feat(proxy-tuning): _step_impl greedy combined decode"
```

---

## Phase 2 — Config + registry plumbing

### Task 6: Register `"proxy_tuning"` strategy + `Settings` fields

**Files:**
- Modify: `olmlx/config.py:256-258` (strategy Literal), add fields after `:276`, add validator near `:447`
- Test: `tests/test_proxy_tuning_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_proxy_tuning_config.py`:

```python
"""Config + registry resolution tests for proxy-tuning."""

from __future__ import annotations

import pytest

from olmlx.config import Settings
from olmlx.engine.registry import (
    _VALID_SPECULATIVE_STRATEGIES,
    ModelConfig,
)


def test_proxy_tuning_is_valid_strategy():
    assert "proxy_tuning" in _VALID_SPECULATIVE_STRATEGIES


def test_settings_accepts_proxy_strategy_and_models():
    s = Settings(
        speculative=True,
        speculative_strategy="proxy_tuning",
        speculative_proxy_expert_model="org/expert",
        speculative_proxy_antiexpert_model="org/anti",
        speculative_proxy_alpha=1.5,
    )
    assert s.speculative_strategy == "proxy_tuning"
    assert s.speculative_proxy_expert_model == "org/expert"
    assert s.speculative_proxy_antiexpert_model == "org/anti"
    assert s.speculative_proxy_alpha == 1.5


def test_proxy_strategy_requires_both_models():
    with pytest.raises(ValueError, match="proxy"):
        Settings(
            speculative=True,
            speculative_strategy="proxy_tuning",
            speculative_proxy_expert_model="org/expert",
            # antiexpert missing
        )


def test_proxy_alpha_default_is_one():
    s = Settings()
    assert s.speculative_proxy_alpha == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_config.py::test_proxy_tuning_is_valid_strategy -v`
Expected: FAIL — `assert 'proxy_tuning' in frozenset({'classic', ...})` is False

- [ ] **Step 3a: Edit the strategy Literal in `config.py`**

In `olmlx/config.py`, change lines 256-258:

```python
    speculative_strategy: Literal[
        "classic", "dflash", "eagle", "pld", "self_speculative"
    ] = "classic"
```

to:

```python
    speculative_strategy: Literal[
        "classic", "dflash", "eagle", "pld", "self_speculative", "proxy_tuning"
    ] = "classic"
```

- [ ] **Step 3b: Add the three proxy fields**

In `olmlx/config.py`, immediately after the `speculative_layers_skip` field (line 276), insert:

```python
    #: Proxy-tuning (engine/proxy_tuning.py) model paths + steering strength.
    #: Only read when ``speculative_strategy == "proxy_tuning"``. The expert
    #: (``M+``, tuned) and anti-expert (``M-``, untuned) are small models that
    #: must share the base model's exact tokenizer/vocabulary. ``alpha`` scales
    #: the tuning delta ``(expert - antiexpert)``; 1.0 is the paper's default.
    speculative_proxy_expert_model: Annotated[str, Field(min_length=1)] | None = None
    speculative_proxy_antiexpert_model: (
        Annotated[str, Field(min_length=1)] | None
    ) = None
    speculative_proxy_alpha: float = 1.0
```

- [ ] **Step 3c: Add the model-validator**

In `olmlx/config.py`, after the `validate_tree_speculative` method (ends at line 467), insert:

```python
    @model_validator(mode="after")
    def validate_proxy_tuning(self) -> "Settings":
        if self.speculative_strategy != "proxy_tuning":
            return self
        if not self.speculative:
            # Scope the requirement to an actually-enabled proxy-tuning config,
            # mirroring how PLD/tree validators avoid rejecting inert settings.
            return self
        missing = [
            name
            for name, val in (
                ("speculative_proxy_expert_model", self.speculative_proxy_expert_model),
                (
                    "speculative_proxy_antiexpert_model",
                    self.speculative_proxy_antiexpert_model,
                ),
            )
            if not val
        ]
        if missing:
            raise ValueError(
                "speculative_strategy='proxy_tuning' requires "
                + " and ".join(missing)
                + " to be set (OLMLX_SPECULATIVE_PROXY_EXPERT_MODEL / "
                "OLMLX_SPECULATIVE_PROXY_ANTIEXPERT_MODEL)."
            )
        import math

        if not math.isfinite(self.speculative_proxy_alpha):
            raise ValueError(
                "speculative_proxy_alpha must be a finite number, got "
                f"{self.speculative_proxy_alpha!r}"
            )
        return self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_config.py -v`
Expected: `test_proxy_tuning_is_valid_strategy` still fails (registry not yet edited); the four `Settings`-only tests pass. If `test_proxy_tuning_is_valid_strategy` is the only failure, proceed — it's fixed in Task 7.

- [ ] **Step 5: Commit**

```bash
git add olmlx/config.py tests/test_proxy_tuning_config.py
git commit -m "feat(proxy-tuning): config Settings fields + validation"
```

---

### Task 7: Registry — strategy literals, `SpeculativeConfig`, `resolved_speculative()`, flash-moe incompat

**Files:**
- Modify: `olmlx/engine/registry.py:19-23` (literals + valid set), `:30-32` (flash-moe incompat), `:285-302` (`SpeculativeConfig`), `:616-714` (`resolved_speculative`)
- Test: `tests/test_proxy_tuning_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_config.py`:

```python
from olmlx.engine.registry import _FLASH_MOE_INCOMPATIBLE_STRATEGIES


def test_resolved_speculative_carries_proxy_fields(monkeypatch):
    from olmlx import config as config_mod

    monkeypatch.setattr(config_mod.settings, "speculative", True, raising=False)
    monkeypatch.setattr(
        config_mod.settings, "speculative_strategy", "proxy_tuning", raising=False
    )
    monkeypatch.setattr(
        config_mod.settings,
        "speculative_proxy_expert_model",
        "org/expert",
        raising=False,
    )
    monkeypatch.setattr(
        config_mod.settings,
        "speculative_proxy_antiexpert_model",
        "org/anti",
        raising=False,
    )
    monkeypatch.setattr(
        config_mod.settings, "speculative_proxy_alpha", 1.25, raising=False
    )

    mc = ModelConfig(hf_path="org/base")
    resolved = mc.resolved_speculative()
    assert resolved.strategy == "proxy_tuning"
    assert resolved.proxy_expert_model == "org/expert"
    assert resolved.proxy_antiexpert_model == "org/anti"
    assert resolved.proxy_alpha == 1.25


def test_proxy_tuning_incompatible_with_flash_moe():
    assert "proxy_tuning" in _FLASH_MOE_INCOMPATIBLE_STRATEGIES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_config.py -k "resolved_speculative_carries or incompatible_with_flash" -v`
Expected: FAIL — `AttributeError: 'SpeculativeConfig' object has no attribute 'proxy_expert_model'` and the incompat assertion fails.

- [ ] **Step 3a: Add `"proxy_tuning"` to the registry literals**

In `olmlx/engine/registry.py`, change lines 19-24:

```python
SpeculativeStrategy = Literal[
    "classic", "dflash", "eagle", "pld", "self_speculative", "mtp"
]
_VALID_SPECULATIVE_STRATEGIES: frozenset[str] = frozenset(
    ("classic", "dflash", "eagle", "pld", "self_speculative", "mtp")
)
```

to:

```python
SpeculativeStrategy = Literal[
    "classic", "dflash", "eagle", "pld", "self_speculative", "mtp", "proxy_tuning"
]
_VALID_SPECULATIVE_STRATEGIES: frozenset[str] = frozenset(
    ("classic", "dflash", "eagle", "pld", "self_speculative", "mtp", "proxy_tuning")
)
```

- [ ] **Step 3b: Add `"proxy_tuning"` to the flash-moe incompat set**

In `olmlx/engine/registry.py`, change lines 30-32:

```python
_FLASH_MOE_INCOMPATIBLE_STRATEGIES: frozenset[str] = frozenset(
    ("dflash", "eagle", "mtp")
)
```

to:

```python
_FLASH_MOE_INCOMPATIBLE_STRATEGIES: frozenset[str] = frozenset(
    ("dflash", "eagle", "mtp", "proxy_tuning")
)
```

- [ ] **Step 3c: Extend the `SpeculativeConfig` NamedTuple**

In `olmlx/engine/registry.py`, after the `layers_skip` field of `SpeculativeConfig` (line 302), add three fields:

```python
    #: Proxy-tuning model paths + steering strength. Populated only when
    #: ``strategy == "proxy_tuning"``; otherwise left at their defaults.
    proxy_expert_model: str | None = None
    proxy_antiexpert_model: str | None = None
    proxy_alpha: float = 1.0
```

- [ ] **Step 3d: Populate them in `resolved_speculative()`**

In `olmlx/engine/registry.py`, inside `resolved_speculative()`: just before the `if not enabled:` block (line 689), add the resolution from global settings:

```python
        proxy_expert_model = settings.speculative_proxy_expert_model
        proxy_antiexpert_model = settings.speculative_proxy_antiexpert_model
        proxy_alpha = settings.speculative_proxy_alpha
```

Then add the three keys to **both** `SpeculativeConfig(...)` returns in this method (the `if not enabled:` return at line 690-699 and the final return at line 705-714). For each, append:

```python
                proxy_expert_model=proxy_expert_model,
                proxy_antiexpert_model=proxy_antiexpert_model,
                proxy_alpha=proxy_alpha,
```

(Match the existing indentation/trailing-comma style of each return.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_config.py -v`
Expected: all tests pass (including `test_proxy_tuning_is_valid_strategy` from Task 6)

- [ ] **Step 5: Run ruff, then commit**

Run: `uv run ruff check olmlx/config.py olmlx/engine/registry.py tests/test_proxy_tuning_config.py && uv run ruff format olmlx/config.py olmlx/engine/registry.py tests/test_proxy_tuning_config.py`
Expected: no errors

```bash
git add olmlx/engine/registry.py tests/test_proxy_tuning_config.py
git commit -m "feat(proxy-tuning): registry strategy + SpeculativeConfig fields"
```

---

## Phase 3 — Loader + model-manager dispatch

### Task 8: `_load_proxy_tuning_decoder` loader

**Files:**
- Modify: `olmlx/engine/speculative_loaders.py` (add method after `_load_self_speculative_decoder`, line 766)
- Test: `tests/test_proxy_tuning_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_proxy_tuning_config.py`:

```python
from olmlx.engine.registry import SpeculativeConfig
from olmlx.engine.speculative_loaders import SpeculativeLoaderMixin


class _LoaderHarness(SpeculativeLoaderMixin):
    """Minimal carrier so we can call mixin methods without a full ModelManager."""

    store = None


def test_load_proxy_tuning_requires_expert_and_antiexpert():
    harness = _LoaderHarness()
    cfg = SpeculativeConfig(
        enabled=True,
        draft_model=None,
        num_tokens=None,
        strategy="proxy_tuning",
        proxy_expert_model=None,  # missing
        proxy_antiexpert_model="org/anti",
    )
    with pytest.raises(ValueError, match="proxy"):
        harness._load_proxy_tuning_decoder(object(), object(), cfg)


def test_load_proxy_tuning_rejects_disabled_config():
    harness = _LoaderHarness()
    cfg = SpeculativeConfig(
        enabled=False,
        draft_model=None,
        num_tokens=None,
        strategy="proxy_tuning",
    )
    with pytest.raises(RuntimeError, match="enabled=False"):
        harness._load_proxy_tuning_decoder(object(), object(), cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proxy_tuning_config.py -k load_proxy_tuning -v`
Expected: FAIL with `AttributeError: ... has no attribute '_load_proxy_tuning_decoder'`

- [ ] **Step 3: Write the loader**

In `olmlx/engine/speculative_loaders.py`, add this method to `SpeculativeLoaderMixin` after `_load_self_speculative_decoder` (after line 766):

```python
    def _load_proxy_tuning_decoder(
        self,
        target_model: Any,
        target_tokenizer: Any,
        spec_config: SpeculativeConfig,
    ) -> Any:
        """Load expert + anti-expert and build a ProxyTuningDecoder.

        The base model is the already-loaded ``target_model``; the small expert
        (``M+``) and anti-expert (``M-``) are loaded inline here via mlx-lm —
        the same pattern ``_load_speculative_decoder`` uses for the draft model.
        They are held by the returned decoder (not registered in the model
        manager), so they coexist with the base for the decoder's lifetime
        without a ``max_loaded_models`` bump.

        All three models must share one exact tokenizer/vocabulary: we hard-fail
        on a ``vocab_size`` mismatch and additionally verify token-mapping
        identity via the tokenizers when available.
        """
        from olmlx.engine.proxy_tuning import ProxyTuningDecoder, check_vocab_identity

        if not spec_config.enabled:
            raise RuntimeError(
                "_load_proxy_tuning_decoder called with spec_config.enabled=False"
            )
        expert_path = spec_config.proxy_expert_model
        antiexpert_path = spec_config.proxy_antiexpert_model
        if not expert_path or not antiexpert_path:
            raise ValueError(
                "speculative_strategy='proxy_tuning' requires both "
                "speculative_proxy_expert_model and "
                "speculative_proxy_antiexpert_model to be set "
                "(OLMLX_SPECULATIVE_PROXY_EXPERT_MODEL / "
                "OLMLX_SPECULATIVE_PROXY_ANTIEXPERT_MODEL)."
            )

        import mlx_lm

        # Imported at call time to avoid the circular import (this module is
        # imported by model_manager to build ModelManager).
        from olmlx.engine.model_manager import _load_with_model_type_fallback

        logger.info(
            "Loading proxy-tuning expert %s and anti-expert %s",
            expert_path,
            antiexpert_path,
        )
        expert_load_path = self._resolve_draft_path(expert_path)
        antiexpert_load_path = self._resolve_draft_path(antiexpert_path)
        expert_model, expert_tokenizer = _load_with_model_type_fallback(
            mlx_lm, expert_load_path, lazy=False
        )
        antiexpert_model, antiexpert_tokenizer = _load_with_model_type_fallback(
            mlx_lm, antiexpert_load_path, lazy=False
        )

        # Hard floor: integer vocab_size must match across all three models.
        self._check_vocab_match(target_model, expert_model)
        self._check_vocab_match(target_model, antiexpert_model)
        # Stronger check: token->id mapping identity (catches same-size,
        # different-mapping vocabularies that the size check misses).
        check_vocab_identity(
            target_tokenizer,
            expert_tokenizer,
            reference_label="base",
            other_label="expert",
        )
        check_vocab_identity(
            target_tokenizer,
            antiexpert_tokenizer,
            reference_label="base",
            other_label="anti-expert",
        )

        return ProxyTuningDecoder(
            base_model=target_model,
            expert_model=expert_model,
            antiexpert_model=antiexpert_model,
            alpha=spec_config.proxy_alpha,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proxy_tuning_config.py -k load_proxy_tuning -v`
Expected: 2 passed

- [ ] **Step 5: Run ruff, then commit**

Run: `uv run ruff check olmlx/engine/speculative_loaders.py tests/test_proxy_tuning_config.py && uv run ruff format olmlx/engine/speculative_loaders.py tests/test_proxy_tuning_config.py`
Expected: no errors

```bash
git add olmlx/engine/speculative_loaders.py tests/test_proxy_tuning_config.py
git commit -m "feat(proxy-tuning): _load_proxy_tuning_decoder inline loader"
```

---

### Task 9: Wire `proxy_tuning` into the model-manager strategy dispatch

**Files:**
- Modify: `olmlx/engine/model_manager.py:3099-3114` (primary text-model dispatch ladder)

- [ ] **Step 1: Read the dispatch ladder to confirm `tokenizer` is in scope**

Run: `uv run python -c "import re,sys; s=open('olmlx/engine/model_manager.py').read().splitlines(); print(chr(10).join(f'{i+1}: {l}' for i,l in enumerate(s[3090:3120])))"`
Expected: shows the `if spec_config.strategy == "dflash":` ... ladder and a `return model, tokenizer, is_vlm, caps, decoder` line — confirming `model` and `tokenizer` locals exist here.

- [ ] **Step 2: Add the `proxy_tuning` branch**

In `olmlx/engine/model_manager.py`, in the dispatch ladder at lines 3099-3114, add an `elif` branch before the final `else:` (the classic loader). Change:

```python
            elif spec_config.strategy == "self_speculative":
                decoder = self._load_self_speculative_decoder(model, spec_config)
            else:
                decoder = self._load_speculative_decoder(model, hf_path, spec_config)
```

to:

```python
            elif spec_config.strategy == "self_speculative":
                decoder = self._load_self_speculative_decoder(model, spec_config)
            elif spec_config.strategy == "proxy_tuning":
                decoder = self._load_proxy_tuning_decoder(
                    model, tokenizer, spec_config
                )
            else:
                decoder = self._load_speculative_decoder(model, hf_path, spec_config)
```

- [ ] **Step 3: Verify the strategy is reachable (no real models needed)**

Run: `uv run pytest tests/test_proxy_tuning.py tests/test_proxy_tuning_config.py -v`
Expected: all pass (the dispatch edit doesn't break existing import paths)

Run: `uv run python -c "import olmlx.engine.model_manager"`
Expected: imports cleanly (no syntax error in the edited ladder)

- [ ] **Step 4: Run the speculative regression suite**

Run: `uv run pytest tests/ -k "speculative or spec_decoder or registry" -q`
Expected: no new failures introduced (compare against a pre-change run if unsure)

- [ ] **Step 5: Run ruff, then commit**

Run: `uv run ruff check olmlx/engine/model_manager.py && uv run ruff format olmlx/engine/model_manager.py`
Expected: no errors

```bash
git add olmlx/engine/model_manager.py
git commit -m "feat(proxy-tuning): dispatch proxy_tuning strategy in model manager"
```

---

## Phase 4 — Docs + end-to-end smoke validation

### Task 10: Document the decode mode in `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (Project Structure block + a new Non-Obvious Invariant)

- [ ] **Step 1: Add the module to the Project Structure tree**

In `CLAUDE.md`, under the `engine/` listing, add a line after the `speculative.py` entry:

```
│   ├── proxy_tuning.py # Decode-time logit arithmetic (base + α·(expert−antiexpert))
```

- [ ] **Step 2: Add a Non-Obvious Invariant**

In `CLAUDE.md`, in the "Non-Obvious Invariants" section, add:

```markdown
**Proxy-tuning is a non-exactness-preserving speculative strategy** — `engine/proxy_tuning.py`'s `ProxyTuningDecoder` registers as `speculative_strategy="proxy_tuning"` to reuse the speculative lifecycle/stream/dispatch plumbing, but it *alters* the output distribution (`base + α·(expert − antiexpert)`), so it deliberately has no draft→verify→accept and no cache trimming — every model advances over the same committed tokens. Expert/anti-expert are loaded **inline by the loader** (held by the decoder, not in `ModelManager._loaded`), so no `max_loaded_models` bump is needed — unlike the panel coordinator. All three models must share one exact tokenizer/vocabulary (`check_vocab_identity` enforces token-mapping equality, stricter than `_check_vocab_match`'s size-only test). v1 is **dense-only** (no GDN capture installed); grammar disables it per-request like every other speculative strategy.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(proxy-tuning): document decode mode + invariants in CLAUDE.md"
```

---

### Task 11: End-to-end smoke validation on a real Qwen3 triple

**Files:** none (manual validation — requires a dense Qwen3 base + a small expert/anti-expert pair sharing the Qwen tokenizer)

> This step needs real model weights and a Mac with Metal. It is the only validation that exercises the full stack (loader → decoder → stream bridge → router). If you lack a tuned expert/anti-expert pair, use any two same-family Qwen dense checkpoints (e.g. an instruct-tuned small model as "expert" and its base as "anti-expert") just to prove the pipeline runs end-to-end; output quality is not the goal here.

- [ ] **Step 1: Configure the strategy via env**

Run (set paths to real local/HF Qwen dense models sharing vocab 151936):

```bash
export OLMLX_SPECULATIVE=true
export OLMLX_SPECULATIVE_STRATEGY=proxy_tuning
export OLMLX_SPECULATIVE_PROXY_EXPERT_MODEL=<org/qwen-small-instruct>
export OLMLX_SPECULATIVE_PROXY_ANTIEXPERT_MODEL=<org/qwen-small-base>
export OLMLX_SPECULATIVE_PROXY_ALPHA=1.0
```

- [ ] **Step 2: Start the server and confirm clean load**

Run: `uv run olmlx` (in a background terminal)
Expected: logs show "Loading proxy-tuning expert ... and anti-expert ..." and no vocab/Metal errors; server binds `http://localhost:11434`.

- [ ] **Step 3: Issue a non-streaming completion against the base model**

Run:

```bash
curl -s http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<base-model-name>","messages":[{"role":"user","content":"Say hello in one short sentence."}],"stream":false}'
```

Expected: a coherent single-sentence reply (not a pretraining-data dump, not empty). Confirms prefill + step + combine + the stream bridge all work end-to-end.

- [ ] **Step 4: Issue a streaming completion**

Run the same curl with `"stream":true`.
Expected: tokens stream incrementally and the response terminates cleanly on EOS.

- [ ] **Step 5: Confirm grammar mutual-exclusion path is intact**

Run a request with a JSON schema / `format: json`. Expected: the server logs "Speculative decoding disabled for this request: grammar-constrained decoding is not yet plumbed through the speculative path" and returns a valid grammar-constrained response (proves proxy-tuning cleanly yields to the existing gate).

- [ ] **Step 6: Final full targeted-suite run + ruff**

Run: `uv run pytest tests/test_proxy_tuning.py tests/test_proxy_tuning_config.py -v && uv run ruff check olmlx/ tests/`
Expected: all proxy-tuning tests pass; ruff clean. (Per project memory, the *full* suite intermittently SIGABRTs locally — trust CI for the whole-suite signal.)

- [ ] **Step 7: No commit** — this task is validation only. If Step 3/4 surface a bug, return to the relevant phase, add a failing regression test reproducing it, then fix.

---

## Self-Review

**Spec coverage** (against issue #523's tightened scope):
- ✅ Vocab-identity guard via `get_vocab()` equality (Task 2), with the loader's `vocab_size` check as the hard floor (Task 8).
- ✅ `ProxyTuningDecoder(SpecDecoderBase)` — fresh subclass, 3 caches, no hooks/bind/GDN/trim (Tasks 3-5).
- ✅ Reuses prefill machinery (`_prefill_last_logit`) on the default stream (Task 4).
- ✅ Config + registry knobs mirroring `speculative_*` (Tasks 6-7).
- ✅ Mutual-exclusion + dispatch: achieved *by construction* (registered as a speculative strategy → inherits both `inference.py` gate sites + the bridge), plus flash-moe incompat (Task 7) and dense-only / grammar-disabled documented limits.
- ✅ TDD throughout; dense-only v1 target validated end-to-end on Qwen3 (Task 11).
- ⚠️ **Deliberately descoped** (documented in the header): per-model `models.json` overrides of the three proxy fields (`from_entry`/`to_entry`/`__post_init__` boilerplate), and a CLI subcommand. Both are follow-ups; the strategy is still selectable per-model and fully drivable via env.
- ⚠️ **Corrected from issue:** no `max_loaded_models ≥ 3` preflight and no `ensure_loaded(pin=True)` — inline loading makes them unnecessary. No `inference.py` edits — the strategy registration inherits the dispatch.

**Placeholder scan:** the two `raise NotImplementedError` stubs in Task 3 are intentional and replaced verbatim in Tasks 4-5; no TODO/TBD/"add error handling" placeholders remain.

**Type consistency:** `ProxyTuningDecoder(base_model, expert_model, antiexpert_model, *, alpha)`, `combine_proxy_logits(base, expert, antiexpert, alpha)`, `check_vocab_identity(reference_tokenizer, other_tokenizer, *, reference_label, other_label)`, and `SpeculativeConfig.proxy_expert_model/proxy_antiexpert_model/proxy_alpha` use the same names across Tasks 1-9. `_load_proxy_tuning_decoder(target_model, target_tokenizer, spec_config)` matches its call site in Task 9.
