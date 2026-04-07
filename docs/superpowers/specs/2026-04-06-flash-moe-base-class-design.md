# Flash-MoE Base Class Refactoring

**Date:** 2026-04-06
**Issue:** #174

## Problem

Four MoE replacement classes in `flash_moe_model.py` (`_FlashMoEDeepSeek`, `_FlashMoEGptOss`, `_FlashMoEQwen3Next`, `_FlashMoEMiniMax`) duplicate:

1. Sharding group check (identical 5-line block in every `__init__`)
2. `self._flash_moe = flash_moe` storage
3. `__call__` tail: `self._flash_moe(x, inds, scores).astype(x.dtype)` + optional shared expert addition

Each class has genuinely different routing logic (gate styles, scoring functions, top-k selection) and different shared expert handling.

## Design

### Template method with `_route` + `_combine`

Introduce `_FlashMoEBase(nn.Module)` that owns the shared logic:

```python
class _FlashMoEBase(nn.Module):
    def __init__(self, original_moe, flash_moe: FlashMoE):
        super().__init__()
        if getattr(original_moe, "sharding_group", None) is not None:
            raise NotImplementedError(
                "Flash-MoE does not support distributed tensor parallelism. "
                "Each rank loads all needed experts, so all_sum would produce "
                "incorrect results. Disable distributed or Flash-MoE."
            )
        self._flash_moe = flash_moe

    def _route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Return (inds, scores) for expert selection."""
        raise NotImplementedError

    def _combine(self, x: mx.array, y: mx.array) -> mx.array:
        """Combine expert output with input (e.g. add shared experts). Default: identity."""
        return y

    def __call__(self, x):
        inds, scores = self._route(x)
        y = self._flash_moe(x, inds, scores).astype(x.dtype)
        return self._combine(x, y)
```

### Subclasses

Each subclass inherits `_FlashMoEBase` and implements only its unique logic:

**`_FlashMoEDeepSeek`** — DeepSeek-V3 / Kimi-K2.5 style:
- `__init__`: copies `gate`, optionally `shared_experts`
- `_route`: `return self.gate(x)` (gate returns `(inds, scores)` directly)
- `_combine`: adds `self.shared_experts(x)` if present

**`_FlashMoEGptOss`** — gpt-oss style:
- `__init__`: copies `router`, `num_experts_per_tok`
- `_route`: router logits -> argpartition -> softmax -> `(inds, scores)`
- `_combine`: default (identity)

**`_FlashMoEQwen3Next`** — Qwen3-Next style:
- `__init__`: copies `gate`, `top_k`, `norm_topk_prob`, `shared_expert`, `shared_expert_gate`
- `_route`: softmax(gate(x)) -> argpartition -> optional normalization -> `(inds, scores)`
- `_combine`: adds `sigmoid(shared_expert_gate(x)) * shared_expert(x)`

**`_FlashMoEMiniMax`** — MiniMax style:
- `__init__`: copies `gate`, `num_experts_per_tok`, `e_score_correction_bias`, optionally `shared_experts`
- `_route`: sigmoid(gate(x)) + correction bias -> argpartition -> renormalize -> `(inds, scores)`
- `_combine`: adds `self.shared_experts(x)` if present

### Scope

**Changed:**
- `olmlx/engine/flash/flash_moe_model.py`: Add `_FlashMoEBase`, refactor four subclasses
- `tests/test_flash_moe_model.py`: Add base class tests, verify isinstance relationships

**Unchanged:**
- `FlashMoE` (dispatch class)
- `FlashMoeModelWrapper`
- `_replace_moe_layers` detection logic
- All public APIs

### Testing

- All existing tests must pass unchanged (behavioral equivalence)
- New tests:
  - `_FlashMoEBase` raises `NotImplementedError` for sharding group
  - Each subclass is an instance of `_FlashMoEBase`
