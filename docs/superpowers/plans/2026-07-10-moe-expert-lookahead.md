# Flash-MoE Trained Expert Lookahead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hide Flash-MoE SSD expert-load latency behind compute via trained per-layer-pair lookahead predictors, and replace LRU eviction with predicted-need scoring.

**Architecture:** A `MoeLookaheadBank` of low-rank heads (hidden state at MoE layer L → expert scores at the next MoE layer) drives a `MoePrefetcher` (structural sibling of the dense `Prefetcher` in `olmlx/engine/flash/prefetch.py`) that overlaps prediction + SSD reads with compute. The same score vectors feed a `ScoredLayerCache` so eviction victims are lowest-predicted-need experts. Heads are trained offline by a new `olmlx flash train-moe-lookahead` CLI command from router traces recorded during a Flash-MoE-wrapped forward pass. Correctness is never affected: the synchronous `load_experts` path is unchanged; prediction only warms the cache and picks eviction victims.

**Tech Stack:** Python 3.12, MLX (`mlx.core`/`mlx.nn`), pydantic-settings, pytest, numpy.

**Spec:** `docs/superpowers/specs/2026-07-10-moe-expert-lookahead-design.md`

## Global Constraints

- TDD: write the failing test first, watch it fail, then implement (project CLAUDE.md).
- Run tests with `uv run pytest <path> -v`. Full suite: `uv run pytest`.
- Concurrency invariant (from the dense prefetcher, `prefetch.py`): `mx.eval` is not safe for concurrent calls. The prediction thread is the ONLY background thread that calls `mx.eval`; the main thread materializes the hidden state BEFORE handing it to the prediction thread, and must not `mx.eval` between `submit()` and the next `wait()`.
- Teardown invariant (CLAUDE.md): prefetcher must close before the weight store. `ModelManager._close_model_resources` already closes `getattr(lm.model, "prefetcher", None)` before `lm.weight_store` (model_manager.py:1054-1071) — exposing `wrapper.prefetcher` is sufficient; do not add manager close logic.
- Prediction failures must degrade to the synchronous path — never raise out of the forward pass.
- Type annotations per CLAUDE.md (e.g. `dict[str, Any]` for freeform JSON, `from __future__ import annotations` in new modules).
- Commit after every task with a conventional-commit message.

---

### Task 1: `ScoredLayerCache` in `_ssd_base.py`

**Files:**
- Modify: `olmlx/engine/flash/_ssd_base.py` (append after `LayerLruCache`, line 239)
- Test: `tests/test_flash_ssd_base.py` (append a new test class)

**Interfaces:**
- Consumes: `LayerLruCache[K, V]` (existing, `_ssd_base.py:184`).
- Produces: `ScoredLayerCache(LayerLruCache[K, V])` with three new methods:
  - `set_scores(layer_idx: int, scores: dict[K, float]) -> None` — replaces the layer's score map.
  - `clear_scores(layer_idx: int) -> None` — removes the layer's score map (staleness guard).
  - `protect(layer_idx: int, keys: set[K]) -> None` — replaces the layer's protected set (in-flight request experts).
  - `put()` overflow victim: lowest score among non-protected keys (missing keys score 0.0); plain LRU-oldest non-protected when no scores are set; LRU-oldest overall when every key is protected. Ties resolve to LRU-oldest (`min` returns the first, and iteration order is LRU order).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_flash_ssd_base.py`:

```python
class TestScoredLayerCache:
    def _cache(self, max_per_layer=3):
        from olmlx.engine.flash._ssd_base import ScoredLayerCache

        return ScoredLayerCache(max_per_layer=max_per_layer)

    def test_behaves_as_lru_without_scores(self):
        cache = self._cache(max_per_layer=2)
        cache.put(0, "a", 1)
        cache.put(0, "b", 2)
        cache.put(0, "c", 3)  # evicts "a" (LRU-oldest)
        assert cache.get(0, "a") is None
        assert cache.get(0, "b") == 2
        assert cache.get(0, "c") == 3

    def test_evicts_lowest_scored(self):
        cache = self._cache(max_per_layer=2)
        cache.put(0, "a", 1)
        cache.put(0, "b", 2)
        # "a" is LRU-oldest but has the higher predicted need
        cache.set_scores(0, {"a": 0.9, "b": 0.1})
        cache.put(0, "c", 3)  # evicts "b" (lowest score), not "a"
        assert cache.get(0, "b") is None
        assert cache.get(0, "a") == 1
        assert cache.get(0, "c") == 3

    def test_missing_score_treated_as_zero(self):
        cache = self._cache(max_per_layer=2)
        cache.put(0, "a", 1)
        cache.put(0, "b", 2)
        cache.set_scores(0, {"a": 0.5})  # "b" unscored -> 0.0
        cache.put(0, "c", 3)  # evicts "b"
        assert cache.get(0, "b") is None
        assert cache.get(0, "a") == 1

    def test_protected_keys_survive_scored_eviction(self):
        cache = self._cache(max_per_layer=2)
        cache.put(0, "a", 1)
        cache.put(0, "b", 2)
        cache.set_scores(0, {"a": 0.0, "b": 0.9})
        cache.protect(0, {"a"})
        cache.put(0, "c", 3)  # "a" protected -> evicts "b" despite high score
        assert cache.get(0, "a") == 1
        assert cache.get(0, "b") is None

    def test_all_protected_falls_back_to_lru_oldest(self):
        cache = self._cache(max_per_layer=2)
        cache.put(0, "a", 1)
        cache.put(0, "b", 2)
        cache.protect(0, {"a", "b", "c"})
        cache.put(0, "c", 3)  # cannot grow unbounded: evicts LRU-oldest "a"
        assert cache.get(0, "a") is None

    def test_clear_scores_restores_lru(self):
        cache = self._cache(max_per_layer=2)
        cache.put(0, "a", 1)
        cache.put(0, "b", 2)
        cache.set_scores(0, {"a": 0.9, "b": 0.1})
        cache.clear_scores(0)
        cache.put(0, "c", 3)  # back to LRU: evicts "a"
        assert cache.get(0, "a") is None
        assert cache.get(0, "b") == 2

    def test_scores_are_per_layer(self):
        cache = self._cache(max_per_layer=2)
        cache.put(1, "a", 1)
        cache.put(1, "b", 2)
        cache.set_scores(0, {"a": 0.9, "b": 0.1})  # different layer
        cache.put(1, "c", 3)  # layer 1 has no scores -> LRU evicts "a"
        assert cache.get(1, "a") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_flash_ssd_base.py::TestScoredLayerCache -v`
Expected: FAIL with `ImportError: cannot import name 'ScoredLayerCache'`

- [ ] **Step 3: Implement `ScoredLayerCache`**

Append to `olmlx/engine/flash/_ssd_base.py`:

```python
class ScoredLayerCache(LayerLruCache[K, V]):
    """LRU cache whose eviction victim can be steered by predicted-need scores.

    Used by the MoE store: a lookahead predictor pushes per-expert scores via
    ``set_scores`` before its prefetch I/O lands; on overflow the victim is
    the lowest-scored non-protected key instead of the LRU-oldest. Behaves as
    a plain ``LayerLruCache`` for layers without scores, so it can back the
    store unconditionally.

    Scores are consume-once: the store clears them when the layer's forward
    pass loads its experts, so a prediction from a previous token can never
    steer a later eviction.
    """

    def __init__(self, max_per_layer: int):
        super().__init__(max_per_layer)
        # Guarded by the parent class's self._lock.
        self._scores: dict[int, dict[K, float]] = {}
        self._protected: dict[int, set[K]] = {}

    def set_scores(self, layer_idx: int, scores: dict[K, float]) -> None:
        """Replace the predicted-need scores for *layer_idx*."""
        with self._lock:
            self._scores[layer_idx] = scores

    def clear_scores(self, layer_idx: int) -> None:
        """Drop scores for *layer_idx* (consume-once staleness guard)."""
        with self._lock:
            self._scores.pop(layer_idx, None)

    def protect(self, layer_idx: int, keys: set[K]) -> None:
        """Replace the eviction-protected set for *layer_idx* (in-flight experts)."""
        with self._lock:
            self._protected[layer_idx] = keys

    def put(self, layer_idx: int, key: K, value: V) -> None:
        if self._max <= 0:
            return
        with self._lock:
            layer_cache = self._cache.setdefault(layer_idx, OrderedDict())
            layer_cache[key] = value
            layer_cache.move_to_end(key)
            while len(layer_cache) > self._max:
                victim = self._pick_victim(layer_idx, layer_cache)
                del layer_cache[victim]

    def _pick_victim(self, layer_idx: int, layer_cache: "OrderedDict[K, V]") -> K:
        """Choose the eviction victim. Caller holds self._lock."""
        protected = self._protected.get(layer_idx, set())
        candidates = [k for k in layer_cache if k not in protected]
        if not candidates:
            # Everything protected (request set >= budget): the cache must not
            # grow unbounded, so fall back to plain LRU-oldest.
            return next(iter(layer_cache))
        scores = self._scores.get(layer_idx)
        if not scores:
            return candidates[0]  # LRU-oldest non-protected
        # min() keeps the first (LRU-oldest) key on score ties.
        return min(candidates, key=lambda k: scores.get(k, 0.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_flash_ssd_base.py -v`
Expected: all PASS (new class and pre-existing tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/flash/_ssd_base.py tests/test_flash_ssd_base.py
git commit -m "feat(flash): ScoredLayerCache with predicted-need eviction"
```

---

### Task 2: `MoeLookaheadBank` predictor

**Files:**
- Create: `olmlx/engine/flash/moe_predictor.py`
- Test: `tests/test_moe_lookahead_predictor.py` (new)

**Interfaces:**
- Consumes: `SparsityPredictor` from `olmlx/engine/flash/predictor.py` (architecture: `Linear(H, rank)` → ReLU → `Linear(rank, out)` → sigmoid).
- Produces (used by Tasks 4, 5, 7):
  - `MoeLookaheadBank(moe_layer_indices: list[int], hidden_size: int, num_experts: int, rank: int = 128, num_experts_per_tok: int = 8)`
  - `bank.heads: list[SparsityPredictor]` — `len(moe_layer_indices) - 1` heads; head *i* maps hidden at `moe_layer_indices[i]` → expert scores at `moe_layer_indices[i+1]`.
  - `bank.next_moe_layer(layer_idx: int) -> int | None`
  - `bank.predict_next(layer_idx: int, hidden_state: mx.array, *, margin: float = 1.5) -> tuple[list[int], np.ndarray] | None` — `(sorted top-m expert indices, full float32 score vector of shape (num_experts,))`; `None` when `layer_idx` has no successor head. `m = min(num_experts, ceil(margin * num_experts_per_tok))`.
  - `bank.save(path: Path) -> None` / `MoeLookaheadBank.load(path: Path) -> MoeLookaheadBank` — heads as `head_XX.npz` plus a `moe_lookahead.json` sidecar (`hidden_size`, `num_experts`, `num_experts_per_tok`, `rank`, `moe_layer_indices`).
  - Module constant `SIDECAR_NAME = "moe_lookahead.json"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_moe_lookahead_predictor.py`:

```python
"""Tests for olmlx.engine.flash.moe_predictor — expert lookahead heads."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.flash.moe_predictor import SIDECAR_NAME, MoeLookaheadBank


class TestMoeLookaheadBank:
    def test_head_count_and_pair_map(self):
        # Non-contiguous MoE layers (dense layers at 0 and 3)
        bank = MoeLookaheadBank(
            [1, 2, 4, 5], hidden_size=16, num_experts=8, rank=4,
            num_experts_per_tok=2,
        )
        assert len(bank.heads) == 3
        assert bank.next_moe_layer(1) == 2
        assert bank.next_moe_layer(2) == 4  # skips dense layer 3
        assert bank.next_moe_layer(4) == 5
        assert bank.next_moe_layer(5) is None  # last MoE layer
        assert bank.next_moe_layer(0) is None  # not an MoE layer

    def test_predict_next_shape_and_count(self):
        bank = MoeLookaheadBank(
            [0, 1], hidden_size=16, num_experts=8, rank=4, num_experts_per_tok=2
        )
        hidden = mx.random.normal((1, 3, 16))  # (B, L, H)
        result = bank.predict_next(0, hidden, margin=1.5)
        assert result is not None
        indices, scores = result
        assert len(indices) == 3  # ceil(1.5 * 2)
        assert indices == sorted(indices)
        assert all(0 <= i < 8 for i in indices)
        assert scores.shape == (8,)
        assert scores.dtype == np.float32

    def test_predict_next_margin_capped_at_num_experts(self):
        bank = MoeLookaheadBank(
            [0, 1], hidden_size=16, num_experts=4, rank=4, num_experts_per_tok=4
        )
        result = bank.predict_next(0, mx.random.normal((1, 1, 16)), margin=2.0)
        assert result is not None
        indices, _ = result
        assert len(indices) == 4  # capped

    def test_predict_next_returns_none_for_last_layer(self):
        bank = MoeLookaheadBank(
            [0, 1], hidden_size=16, num_experts=8, rank=4, num_experts_per_tok=2
        )
        assert bank.predict_next(1, mx.random.normal((1, 1, 16))) is None

    def test_save_load_round_trip(self, tmp_path):
        bank = MoeLookaheadBank(
            [1, 2, 4], hidden_size=16, num_experts=8, rank=4, num_experts_per_tok=2
        )
        out = tmp_path / "moe_lookahead"
        bank.save(out)

        sidecar = json.loads((out / SIDECAR_NAME).read_text())
        assert sidecar["hidden_size"] == 16
        assert sidecar["num_experts"] == 8
        assert sidecar["moe_layer_indices"] == [1, 2, 4]

        loaded = MoeLookaheadBank.load(out)
        assert loaded.moe_layer_indices == [1, 2, 4]
        assert len(loaded.heads) == 2

        hidden = mx.random.normal((1, 2, 16))
        orig = bank.predict_next(1, hidden)
        rt = loaded.predict_next(1, hidden)
        assert orig is not None and rt is not None
        np.testing.assert_allclose(orig[1], rt[1], rtol=1e-5)

    def test_load_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MoeLookaheadBank.load(tmp_path / "nonexistent")

    def test_requires_two_moe_layers(self):
        with pytest.raises(ValueError):
            MoeLookaheadBank([3], hidden_size=16, num_experts=8, rank=4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_moe_lookahead_predictor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.engine.flash.moe_predictor'`

- [ ] **Step 3: Implement `moe_predictor.py`**

Create `olmlx/engine/flash/moe_predictor.py`:

```python
"""Expert lookahead predictors for Flash-MoE.

Per-MoE-layer-pair low-rank heads that predict which experts the NEXT MoE
layer's router will select, given the hidden state entering the current MoE
layer. Drives speculative expert prefetch and predicted-need cache eviction.

Same architecture family as the dense-path ``SparsityPredictor``
(``predictor.py``), with ``num_experts`` outputs instead of
``intermediate_size``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from olmlx.engine.flash.predictor import SparsityPredictor

SIDECAR_NAME = "moe_lookahead.json"


class MoeLookaheadBank:
    """Per-MoE-layer-pair expert lookahead heads.

    Head ``i`` predicts the expert set of ``moe_layer_indices[i + 1]`` from
    the hidden state entering ``moe_layer_indices[i]``. Pairs follow
    *consecutive MoE layers*, not consecutive layer indices — interleaved
    dense layers just add I/O lead time.
    """

    def __init__(
        self,
        moe_layer_indices: list[int],
        hidden_size: int,
        num_experts: int,
        rank: int = 128,
        num_experts_per_tok: int = 8,
    ):
        indices = sorted(moe_layer_indices)
        if len(indices) < 2:
            raise ValueError(
                f"Need at least 2 MoE layers for lookahead, got {indices}"
            )
        self.moe_layer_indices = indices
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.rank = rank
        self.heads = [
            SparsityPredictor(hidden_size, num_experts, rank)
            for _ in range(len(indices) - 1)
        ]
        self._pair_for_layer = {indices[i]: i for i in range(len(indices) - 1)}
        self._next_for_layer = {
            indices[i]: indices[i + 1] for i in range(len(indices) - 1)
        }

    def next_moe_layer(self, layer_idx: int) -> int | None:
        """The MoE layer whose experts head(layer_idx) predicts, or None."""
        return self._next_for_layer.get(layer_idx)

    def predict_next(
        self,
        layer_idx: int,
        hidden_state: mx.array,
        *,
        margin: float = 1.5,
    ) -> tuple[list[int], np.ndarray] | None:
        """Predict the next MoE layer's expert set from *layer_idx*'s input.

        Returns ``(sorted top-m expert indices, full score vector)`` where
        ``m = min(num_experts, ceil(margin * num_experts_per_tok))``, or
        ``None`` if *layer_idx* has no successor head. Calls ``mx.eval`` —
        only safe on the prediction thread (or when no prediction is in
        flight).
        """
        pair_idx = self._pair_for_layer.get(layer_idx)
        if pair_idx is None:
            return None
        flat = hidden_state.reshape(-1, hidden_state.shape[-1])
        scores = self.heads[pair_idx](flat).mean(axis=0)
        mx.eval(scores)
        scores_np = np.array(scores, dtype=np.float32)
        m = min(self.num_experts, math.ceil(margin * self.num_experts_per_tok))
        top_m = np.argpartition(-scores_np, m - 1)[:m] if m < len(scores_np) else np.arange(len(scores_np))
        return sorted(int(i) for i in top_m), scores_np

    def save(self, path: Path) -> None:
        """Save heads + sidecar to a directory."""
        path.mkdir(parents=True, exist_ok=True)
        for i, head in enumerate(self.heads):
            mx.savez(
                str(path / f"head_{i:02d}.npz"),
                **{
                    f"pair_{i}.down.weight": head.down.weight,
                    f"pair_{i}.up.weight": head.up.weight,
                },
            )
        (path / SIDECAR_NAME).write_text(
            json.dumps(
                {
                    "hidden_size": self.hidden_size,
                    "num_experts": self.num_experts,
                    "num_experts_per_tok": self.num_experts_per_tok,
                    "rank": self.rank,
                    "moe_layer_indices": self.moe_layer_indices,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> MoeLookaheadBank:
        """Load a bank from a directory written by :meth:`save`."""
        sidecar_path = path / SIDECAR_NAME
        if not sidecar_path.exists():
            raise FileNotFoundError(f"No {SIDECAR_NAME} in {path}")
        meta = json.loads(sidecar_path.read_text())
        bank = cls(
            meta["moe_layer_indices"],
            hidden_size=meta["hidden_size"],
            num_experts=meta["num_experts"],
            rank=meta["rank"],
            num_experts_per_tok=meta["num_experts_per_tok"],
        )
        for i, head in enumerate(bank.heads):
            weights = dict(mx.load(str(path / f"head_{i:02d}.npz")))  # pyright: ignore[reportCallIssue]
            head.down.weight = weights[f"pair_{i}.down.weight"]
            head.up.weight = weights[f"pair_{i}.up.weight"]
        return bank
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_moe_lookahead_predictor.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/flash/moe_predictor.py tests/test_moe_lookahead_predictor.py
git commit -m "feat(flash): MoeLookaheadBank expert lookahead heads"
```

---

### Task 3: Store — `prefetch_experts`, scored cache, protection

**Files:**
- Modify: `olmlx/engine/flash/moe_weight_store.py`
- Test: `tests/test_flash_moe_weight_store.py` (append a new test class)

**Interfaces:**
- Consumes: `ScoredLayerCache` (Task 1).
- Produces (used by Task 4):
  - `FlashMoeWeightStore.prefetch_experts(layer_idx: int, expert_indices: list[int]) -> tuple[int, int]` — `(already_cached, fetched)` counts; blocks until reads land; read failures are logged and skipped, never raised.
  - `FlashMoeWeightStore.set_layer_scores(layer_idx: int, scores: dict[int, float]) -> None`.
  - `_load_experts_impl` now calls `self._cache.protect(layer_idx, set(expert_indices))` before loading, and in its `finally` block both clears the protection (`protect(layer_idx, set())`) and clears the scores (`clear_scores(layer_idx)`). Protection scoped to the load is load-bearing: if it outlived the load, the next prefetch insert for the layer would find every cached key protected and fall back to LRU-oldest, defeating scored eviction.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_flash_moe_weight_store.py`:

```python
class TestPrefetchExperts:
    @pytest.fixture()
    def store_setup(self, tmp_path):
        hidden, inter, experts = 64, 32, 8
        model_dir = _make_synthetic_moe_weights(hidden, inter, experts, 2, 1, tmp_path)
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        bundle_moe_experts(model_dir, output_dir)

        from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

        store = FlashMoeWeightStore(output_dir, num_io_threads=4, cache_budget_experts=4)
        yield store, experts
        store.close()

    def test_prefetch_warms_cache(self, store_setup):
        store, _ = store_setup
        cached, fetched = store.prefetch_experts(1, [0, 2])
        assert (cached, fetched) == (0, 2)
        # Subsequent load is a pure cache hit
        store.load_experts(1, [0, 2])
        snap = store.stats.snapshot()
        assert snap["cache_hits"] == 2
        assert snap["cache_misses"] == 0

    def test_prefetch_counts_already_cached(self, store_setup):
        store, _ = store_setup
        store.prefetch_experts(1, [0, 2])
        cached, fetched = store.prefetch_experts(1, [0, 2, 3])
        assert (cached, fetched) == (2, 1)

    def test_prefetch_empty_is_noop(self, store_setup):
        store, _ = store_setup
        assert store.prefetch_experts(1, []) == (0, 0)

    def test_scored_eviction_keeps_predicted_experts(self, store_setup):
        store, _ = store_setup  # budget = 4
        # Fill cache for layer 1 with experts 0-3
        store.load_experts(1, [0, 1, 2, 3])
        # Predictor says 0 and 1 are needed next, 2 and 3 are not
        store.set_layer_scores(1, {0: 0.9, 1: 0.8, 2: 0.1, 3: 0.05})
        # Prefetch two more -> two evictions, victims must be 3 then 2
        store.prefetch_experts(1, [4, 5])
        store.set_layer_scores(1, {0: 0.9, 1: 0.8, 4: 0.5, 5: 0.5})
        store.load_experts(1, [0, 1])
        snap = store.stats.snapshot()
        assert snap["cache_misses"] == 4  # only the initial fill missed
        assert snap["cache_hits"] == 2  # 0 and 1 survived eviction

    def test_load_clears_scores(self, store_setup):
        store, _ = store_setup
        store.set_layer_scores(1, {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9})
        store.load_experts(1, [0])  # consume -> clears scores for layer 1
        # With scores cleared, filling past budget uses plain LRU again
        store.load_experts(1, [1, 2, 3, 4])
        # LRU-oldest (0) evicted despite its old high score
        _, missing = store._cache.get_cached_indices(1, [0])
        assert missing == [0]

    def test_in_flight_request_protected(self, store_setup):
        store, _ = store_setup  # budget = 4
        # Request more experts than the budget in one call: the request's own
        # experts must not evict each other into a miss for the return value.
        loaded = store.load_experts(1, [0, 1, 2, 3, 4, 5])
        assert loaded.up_weight.shape[0] == 6  # all six stacked
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_flash_moe_weight_store.py::TestPrefetchExperts -v`
Expected: FAIL with `AttributeError: 'FlashMoeWeightStore' object has no attribute 'prefetch_experts'`

- [ ] **Step 3: Implement store changes**

In `olmlx/engine/flash/moe_weight_store.py`:

1. Change the import from `_ssd_base` (line 15-20) to include `ScoredLayerCache`:

```python
from olmlx.engine.flash._ssd_base import (
    LayerLruCache,
    ScoredLayerCache,
    close_fds,
    full_pread,
    open_fds,
)
```

2. Change the cache alias (line 117) — keep `LayerLruCache` imported for the type bound reference but switch the store's cache:

```python
_ExpertCache = ScoredLayerCache[int, dict[str, Any]]
```

and in `__init__` (line 131):

```python
self._cache: _ExpertCache = ScoredLayerCache(max_per_layer=cache_budget_experts)
```

(If `LayerLruCache` becomes unused after this, drop it from the import.)

3. Add the two new public methods after `_read_expert` (after line 196):

```python
    def set_layer_scores(self, layer_idx: int, scores: dict[int, float]) -> None:
        """Push predicted-need scores for *layer_idx* (consumed by eviction)."""
        self._cache.set_scores(layer_idx, scores)

    def prefetch_experts(
        self, layer_idx: int, expert_indices: list[int]
    ) -> tuple[int, int]:
        """Warm the cache with *expert_indices* for *layer_idx*.

        Blocks until all reads land. Read failures are logged and skipped —
        the synchronous ``load_experts`` path re-reads and reports them.
        Returns ``(already_cached, fetched)`` counts.
        """
        if not expert_indices:
            return (0, 0)
        cached, missing = self._cache.get_cached_indices(layer_idx, expert_indices)
        if not missing:
            return (len(cached), 0)
        future_to_idx = {
            self._executor.submit(self._read_expert, layer_idx, idx): idx
            for idx in missing
        }
        fetched = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                self._cache.put(layer_idx, idx, future.result())
                fetched += 1
            except Exception:
                logger.warning(
                    "Prefetch read failed for layer %d expert %d",
                    layer_idx,
                    idx,
                    exc_info=True,
                )
        return (len(cached), fetched)
```

Add `import logging` and `logger = logging.getLogger(__name__)` at module top if not present (check — the module currently has no logger).

4. In `_load_experts_impl` (line 338), protect the in-flight request right after the layout lookup (line 345):

```python
        layout = self._layouts[layer_idx]
        # Protect the in-flight request set from scored eviction, so a
        # low-scored expert of THIS batch is not evicted mid-load by a
        # concurrent prefetch insert for the same layer.
        self._cache.protect(layer_idx, set(expert_indices))
```

and extend the existing `finally` (line 383-384) to consume scores:

```python
        finally:
            self.stats.record(hits, misses, failures)
            # Scores are consume-once: this layer's forward has now used
            # (or bypassed) them; a stale prediction must not steer later
            # evictions.
            self._cache.clear_scores(layer_idx)
            # Protection is scoped to THIS load. If it lingered, the next
            # prefetch insert for this layer would find every cached key
            # protected and evict LRU-oldest, defeating scored eviction.
            self._cache.protect(layer_idx, set())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_flash_moe_weight_store.py tests/test_moe_weight_store_coverage.py -v`
Expected: all PASS (new class and all pre-existing store tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/flash/moe_weight_store.py tests/test_flash_moe_weight_store.py
git commit -m "feat(flash): expert prefetch + scored eviction in FlashMoeWeightStore"
```

---

### Task 4: `MoePrefetcher`

**Files:**
- Create: `olmlx/engine/flash/moe_prefetch.py`
- Test: `tests/test_moe_prefetch.py` (new)

**Interfaces:**
- Consumes: `MoeLookaheadBank` (Task 2), `FlashMoeWeightStore.prefetch_experts` / `.set_layer_scores` (Task 3), `PrefetchStats` and `_LayerPrefetchState` from `olmlx/engine/flash/prefetch.py`.
- Produces (used by Task 5):
  - `MoePrefetcher(bank: MoeLookaheadBank, weight_store: FlashMoeWeightStore, *, margin: float = 1.5, max_positions: int = 8, scored_eviction: bool = True, io_threads: int = 8)`
  - `submit(layer_idx: int, hidden_state: mx.array) -> None` — non-blocking; no-op when `layer_idx` has no successor head, when positions exceed `max_positions`, or when a prefetch for the successor is already pending.
  - `wait(layer_idx: int) -> None` — blocks until any pending prefetch targeting `layer_idx` completes.
  - `close() -> None` — drains prediction executor then I/O executor; logs a stats summary.
  - `stats: PrefetchStats`.

Concurrency contract (copied from the dense `Prefetcher`, `prefetch.py:107-154`): `submit` registers the `_pending` entry and materializes `hidden_state` on the calling thread BEFORE enqueueing prediction; the single prediction thread is the only background `mx.eval` caller; `wait` pops and blocks on the entry's `done` event. There is no `submit_bulk` in v1, so the dense path's `_predict_in_flight` guard is not needed.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_moe_prefetch.py`:

```python
"""Tests for olmlx.engine.flash.moe_prefetch — MoE expert prefetcher."""

from __future__ import annotations

import mlx.core as mx
import pytest

from tests.test_flash_moe_bundler import _make_synthetic_moe_weights


HIDDEN, INTER, EXPERTS = 64, 32, 8


@pytest.fixture()
def store(tmp_path):
    # 2 MoE layers (indices 1, 2 — layer 0 is dense) over 8 experts
    model_dir = _make_synthetic_moe_weights(HIDDEN, INTER, EXPERTS, 2, 1, tmp_path)
    output_dir = tmp_path / "flash_moe"

    from olmlx.engine.flash.moe_bundler import bundle_moe_experts

    bundle_moe_experts(model_dir, output_dir)

    from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

    s = FlashMoeWeightStore(output_dir, num_io_threads=4, cache_budget_experts=8)
    yield s
    s.close()


@pytest.fixture()
def bank():
    from olmlx.engine.flash.moe_predictor import MoeLookaheadBank

    return MoeLookaheadBank(
        [1, 2], hidden_size=HIDDEN, num_experts=EXPERTS, rank=4,
        num_experts_per_tok=2,
    )


def _make_prefetcher(bank, store, **kwargs):
    from olmlx.engine.flash.moe_prefetch import MoePrefetcher

    return MoePrefetcher(bank, store, **kwargs)


class TestMoePrefetcher:
    def test_submit_then_wait_warms_cache(self, bank, store):
        pf = _make_prefetcher(bank, store)
        try:
            hidden = mx.random.normal((1, 1, HIDDEN))
            pf.submit(1, hidden)
            pf.wait(2)
            assert pf.stats.submitted == 1
            # ceil(1.5 * 2) = 3 experts predicted; all landed in the cache
            assert pf.stats.cache_misses == 3
            cached, missing = store._cache.get_cached_indices(2, list(range(EXPERTS)))
            assert len(cached) == 3
        finally:
            pf.close()

    def test_wait_without_submit_returns(self, bank, store):
        pf = _make_prefetcher(bank, store)
        try:
            pf.wait(2)  # no pending entry — must not block or raise
        finally:
            pf.close()

    def test_submit_last_moe_layer_is_noop(self, bank, store):
        pf = _make_prefetcher(bank, store)
        try:
            pf.submit(2, mx.random.normal((1, 1, HIDDEN)))  # layer 2 has no successor
            pf.wait(2)
            assert pf.stats.submitted == 0
        finally:
            pf.close()

    def test_prefill_guard_skips_large_batches(self, bank, store):
        pf = _make_prefetcher(bank, store, max_positions=4)
        try:
            pf.submit(1, mx.random.normal((1, 5, HIDDEN)))  # 5 positions > 4
            pf.wait(2)
            assert pf.stats.submitted == 0
        finally:
            pf.close()

    def test_duplicate_submit_is_noop(self, bank, store):
        pf = _make_prefetcher(bank, store)
        try:
            hidden = mx.random.normal((1, 1, HIDDEN))
            pf.submit(1, hidden)
            pf.submit(1, hidden)  # second submit while first pending: no-op
            pf.wait(2)
            assert pf.stats.submitted <= 2  # never double-registers pending
        finally:
            pf.close()

    def test_prediction_failure_unblocks_wait(self, bank, store):
        pf = _make_prefetcher(bank, store)
        try:
            def _boom(*a, **k):
                raise RuntimeError("synthetic predictor failure")

            bank.predict_next = _boom
            pf.submit(1, mx.random.normal((1, 1, HIDDEN)))
            pf.wait(2)  # must not hang
            assert pf.stats.failures == 1
        finally:
            pf.close()

    def test_scored_eviction_pushes_scores(self, bank, store):
        pf = _make_prefetcher(bank, store, scored_eviction=True)
        try:
            pf.submit(1, mx.random.normal((1, 1, HIDDEN)))
            pf.wait(2)
            with store._cache._lock:
                assert 2 in store._cache._scores
                assert len(store._cache._scores[2]) == EXPERTS
        finally:
            pf.close()

    def test_scored_eviction_disabled_pushes_nothing(self, bank, store):
        pf = _make_prefetcher(bank, store, scored_eviction=False)
        try:
            pf.submit(1, mx.random.normal((1, 1, HIDDEN)))
            pf.wait(2)
            with store._cache._lock:
                assert 2 not in store._cache._scores
        finally:
            pf.close()

    def test_close_is_idempotent(self, bank, store):
        pf = _make_prefetcher(bank, store)
        pf.close()
        pf.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_moe_prefetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.engine.flash.moe_prefetch'`

- [ ] **Step 3: Implement `moe_prefetch.py`**

Create `olmlx/engine/flash/moe_prefetch.py`:

```python
"""Speculative expert prefetcher for Flash-MoE.

While MoE layer L computes, a trained lookahead head predicts which experts
the NEXT MoE layer will route to and starts background SSD reads, so the next
layer's ``load_experts`` finds them cached. The same prediction's score
vector steers cache eviction (``ScoredLayerCache``) when enabled.

Concurrency contract (identical to the dense ``Prefetcher`` in
``prefetch.py``): ``mx.eval`` is not safe for concurrent calls, so the single
prediction thread is the ONLY background thread that evaluates arrays; the
calling thread materializes the hidden state before enqueueing, and must not
call ``mx.eval`` between ``submit()`` and the next ``wait()``. Prediction can
never affect correctness — a misprediction or failure only costs latency.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx

from olmlx.engine.flash.moe_predictor import MoeLookaheadBank
from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore
from olmlx.engine.flash.prefetch import PrefetchStats, _LayerPrefetchState

logger = logging.getLogger(__name__)


class MoePrefetcher:
    """Orchestrates background expert prefetching from SSD."""

    def __init__(
        self,
        bank: MoeLookaheadBank,
        weight_store: FlashMoeWeightStore,
        *,
        margin: float = 1.5,
        max_positions: int = 8,
        scored_eviction: bool = True,
        io_threads: int = 8,
    ):
        self._bank = bank
        self._weight_store = weight_store
        self._margin = margin
        self._max_positions = max_positions
        self._scored_eviction = scored_eviction
        self._io_executor = ThreadPoolExecutor(
            max_workers=io_threads, thread_name_prefix="moe-prefetch-io"
        )
        self._predict_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="moe-prefetch-predict"
        )
        self._lock = threading.Lock()
        self._pending: dict[int, _LayerPrefetchState] = {}
        self.stats = PrefetchStats()

    def submit(self, layer_idx: int, hidden_state: mx.array) -> None:
        """Predict the next MoE layer's experts and start background I/O.

        No-op when *layer_idx* has no successor head, the position count
        exceeds the prefill guard, or a prefetch for the successor is already
        pending. The hidden state is materialized on the calling thread
        (``mx.eval`` is not safe for concurrent calls).
        """
        next_layer = self._bank.next_moe_layer(layer_idx)
        if next_layer is None:
            return
        positions = hidden_state.size // hidden_state.shape[-1]
        if positions > self._max_positions:
            return  # prefill: most experts activate anyway, skip the work

        with self._lock:
            if next_layer in self._pending:
                return  # already in flight

        mx.eval(hidden_state)

        state = _LayerPrefetchState()
        with self._lock:
            self._pending[next_layer] = state
        try:
            self._predict_executor.submit(
                self._do_predict_and_io, layer_idx, next_layer, hidden_state, state
            )
        except RuntimeError:  # executor shut down
            with self._lock:
                self._pending.pop(next_layer, None)
            state.done.set()

    def wait(self, layer_idx: int) -> None:
        """Block until any pending prefetch targeting *layer_idx* completes."""
        with self._lock:
            state = self._pending.pop(layer_idx, None)
        if state is None:
            return
        state.done.wait()

    def _do_predict_and_io(
        self,
        layer_idx: int,
        next_layer: int,
        hidden_state: mx.array,
        state: _LayerPrefetchState,
    ) -> None:
        """Run on the prediction thread: predict, push scores, enqueue I/O."""
        try:
            result = self._bank.predict_next(
                layer_idx, hidden_state, margin=self._margin
            )
        except Exception:
            logger.warning(
                "Expert prediction failed for layer %d", next_layer, exc_info=True
            )
            with self._lock:
                self.stats.failures += 1
            result = None

        if result is None:
            state.done.set()
            return

        indices, scores = result
        if self._scored_eviction:
            # Push BEFORE the I/O lands so prefetch inserts already evict by
            # predicted need. The store clears these when next_layer's
            # forward consumes them.
            self._weight_store.set_layer_scores(
                next_layer, {i: float(scores[i]) for i in range(len(scores))}
            )

        with self._lock:
            self.stats.submitted += 1

        def _do_prefetch() -> None:
            try:
                cached, fetched = self._weight_store.prefetch_experts(
                    next_layer, indices
                )
                with self._lock:
                    self.stats.cache_hits += cached
                    self.stats.cache_misses += fetched
            except Exception:
                logger.warning(
                    "Expert prefetch I/O failed for layer %d",
                    next_layer,
                    exc_info=True,
                )
                with self._lock:
                    self.stats.failures += 1
            finally:
                state.done.set()

        try:
            self._io_executor.submit(_do_prefetch)
        except RuntimeError:  # executor shut down
            state.done.set()
            with self._lock:
                self.stats.submitted -= 1

    def close(self) -> None:
        """Drain the prediction pool, then the I/O pool; log a summary.

        Prediction executor first — it submits work into the I/O executor.
        Idempotent: ThreadPoolExecutor.shutdown tolerates repeated calls.
        """
        self._predict_executor.shutdown(wait=True)
        self._io_executor.shutdown(wait=True)
        logger.info(
            "MoE prefetch stats: submitted=%d prefetched=%d already_cached=%d "
            "failures=%d",
            self.stats.submitted,
            self.stats.cache_misses,
            self.stats.cache_hits,
            self.stats.failures,
        )
```

Note: `PrefetchStats.cache_hits`/`cache_misses` are reused with MoE meaning "already cached" / "fetched from SSD" — the log line labels them accordingly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_moe_prefetch.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/flash/moe_prefetch.py tests/test_moe_prefetch.py
git commit -m "feat(flash): MoePrefetcher — background expert prediction + SSD prefetch"
```

---

### Task 5: Forward-pass integration (`FlashMoE` + wrapper + `wrap_flash_moe`)

**Files:**
- Modify: `olmlx/engine/flash/flash_moe.py` (constructor + `__call__`)
- Modify: `olmlx/engine/flash/flash_moe_model.py` (`FlashMoeModelWrapper.__init__`, `_replace_moe_layers`, `wrap_flash_moe`, new `_maybe_create_prefetcher`)
- Test: `tests/test_flash_moe_model.py` (append a new test class)

**Interfaces:**
- Consumes: `MoePrefetcher` (Task 4), `MoeLookaheadBank.load` + `SIDECAR_NAME` (Task 2).
- Produces (used by Task 6):
  - `FlashMoE.__init__(..., prefetcher: MoePrefetcher | None = None)` (new trailing keyword param).
  - `FlashMoeModelWrapper.__init__(..., prefetcher: MoePrefetcher | None = None)`; sets `self.prefetcher` (read by `ModelManager._close_model_resources` via `getattr(lm.model, "prefetcher", None)` — no manager change needed).
  - `wrap_flash_moe(model, flash_moe_dir, *, io_threads, cache_budget_experts, prefetch: bool = False, lookahead_margin: float = 1.5, prefetch_max_positions: int = 8, scored_eviction: bool = True) -> tuple[Any, Any]` — prefetch defaults OFF so the offline calibration loader (`load_flash_moe_model`) is unaffected.
  - `_maybe_create_prefetcher(flash_moe_dir: Path, moe_config: dict[str, Any], store: FlashMoeWeightStore, *, margin: float, max_positions: int, scored_eviction: bool) -> MoePrefetcher | None` — returns `None` (with a log line) when `moe_lookahead/` is missing, unreadable, or its sidecar disagrees with `flash_moe_config.json` on `hidden_size`, `num_experts`, or `moe_layer_indices`.

`FlashMoE.__call__` ordering (the load-bearing part — comments in code must state it):
1. `wait(self.layer_idx)` FIRST — before this layer's `mx.eval(inds)`, so the main thread never evals concurrently with a prediction targeting this layer.
2. `mx.eval(inds)` + build `unique_experts` (existing code).
3. `submit(self.layer_idx, x)` — after the main-thread eval completes, before the blocking `load_experts`, so prediction + next-layer I/O overlap this layer's load and compute.
4. `load_experts` + gather_mm (existing code, unchanged).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_flash_moe_model.py`:

```python
class TestMoePrefetchIntegration:
    HIDDEN, INTER, EXPERTS = 64, 32, 8

    def _bundle(self, tmp_path):
        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        model_dir = _make_synthetic_moe_weights(
            self.HIDDEN, self.INTER, self.EXPERTS, 2, 1, tmp_path
        )
        output_dir = tmp_path / "flash_moe"
        bundle_moe_experts(model_dir, output_dir)
        return output_dir

    def _store(self, output_dir):
        from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

        return FlashMoeWeightStore(
            output_dir, num_io_threads=4, cache_budget_experts=8
        )

    def _bank(self):
        from olmlx.engine.flash.moe_predictor import MoeLookaheadBank

        return MoeLookaheadBank(
            [1, 2], hidden_size=self.HIDDEN, num_experts=self.EXPERTS, rank=4,
            num_experts_per_tok=2,
        )

    def test_flash_moe_output_identical_with_prefetcher(self, tmp_path):
        """Prefetch must never change outputs — only warm the cache."""
        import mlx.core as mx

        from olmlx.engine.flash.flash_moe import FlashMoE
        from olmlx.engine.flash.moe_prefetch import MoePrefetcher

        output_dir = self._bundle(tmp_path)
        x = mx.random.normal((1, 2, self.HIDDEN))
        inds = mx.array([[[0, 3], [1, 2]]], dtype=mx.uint32)
        scores = mx.softmax(mx.random.normal((1, 2, 2)), axis=-1)

        def _run(prefetcher, store):
            moe = FlashMoE(
                layer_idx=1,
                hidden_size=self.HIDDEN,
                intermediate_size=self.INTER,
                num_experts=self.EXPERTS,
                num_experts_per_tok=2,
                weight_store=store,
                prefetcher=prefetcher,
            )
            out = moe(x, inds, scores)
            mx.eval(out)
            return out

        store_a = self._store(output_dir)
        baseline = _run(None, store_a)
        store_a.close()

        store_b = self._store(output_dir)
        pf = MoePrefetcher(self._bank(), store_b)
        try:
            with_prefetch = _run(pf, store_b)
        finally:
            pf.close()
            store_b.close()

        import numpy as np

        np.testing.assert_allclose(
            np.array(baseline), np.array(with_prefetch), rtol=1e-5
        )

    def test_flash_moe_submits_after_eval_and_waits_first(self, tmp_path):
        """FlashMoE calls wait(layer) before eval and submit(layer) before load."""
        import mlx.core as mx

        from olmlx.engine.flash.flash_moe import FlashMoE

        output_dir = self._bundle(tmp_path)
        store = self._store(output_dir)
        calls: list[tuple[str, int]] = []

        class _SpyPrefetcher:
            def wait(self, layer_idx):
                calls.append(("wait", layer_idx))

            def submit(self, layer_idx, hidden_state):
                calls.append(("submit", layer_idx))

        try:
            moe = FlashMoE(
                layer_idx=1,
                hidden_size=self.HIDDEN,
                intermediate_size=self.INTER,
                num_experts=self.EXPERTS,
                num_experts_per_tok=2,
                weight_store=store,
                prefetcher=_SpyPrefetcher(),
            )
            x = mx.random.normal((1, 1, self.HIDDEN))
            inds = mx.array([[[0, 1]]], dtype=mx.uint32)
            scores = mx.array([[[0.5, 0.5]]])
            mx.eval(moe(x, inds, scores))
        finally:
            store.close()

        assert calls == [("wait", 1), ("submit", 1)]

    def test_maybe_create_prefetcher_missing_dir_returns_none(self, tmp_path):
        from olmlx.engine.flash.flash_moe_model import _maybe_create_prefetcher

        output_dir = self._bundle(tmp_path)
        store = self._store(output_dir)
        try:
            moe_config = {
                "hidden_size": self.HIDDEN,
                "num_experts": self.EXPERTS,
                "num_experts_per_tok": 2,
                "moe_layer_indices": [1, 2],
            }
            assert (
                _maybe_create_prefetcher(
                    output_dir, moe_config, store,
                    margin=1.5, max_positions=8, scored_eviction=True,
                )
                is None
            )
        finally:
            store.close()

    def test_maybe_create_prefetcher_loads_valid_bank(self, tmp_path):
        from olmlx.engine.flash.flash_moe_model import _maybe_create_prefetcher

        output_dir = self._bundle(tmp_path)
        self._bank().save(output_dir / "moe_lookahead")
        store = self._store(output_dir)
        pf = None
        try:
            moe_config = {
                "hidden_size": self.HIDDEN,
                "num_experts": self.EXPERTS,
                "num_experts_per_tok": 2,
                "moe_layer_indices": [1, 2],
            }
            pf = _maybe_create_prefetcher(
                output_dir, moe_config, store,
                margin=1.5, max_positions=8, scored_eviction=True,
            )
            assert pf is not None
        finally:
            if pf is not None:
                pf.close()
            store.close()

    def test_maybe_create_prefetcher_stale_sidecar_returns_none(self, tmp_path):
        """A bank trained against a different bundle must be rejected."""
        from olmlx.engine.flash.flash_moe_model import _maybe_create_prefetcher

        output_dir = self._bundle(tmp_path)
        self._bank().save(output_dir / "moe_lookahead")
        store = self._store(output_dir)
        try:
            moe_config = {
                "hidden_size": self.HIDDEN,
                "num_experts": self.EXPERTS + 8,  # re-bundled differently
                "num_experts_per_tok": 2,
                "moe_layer_indices": [1, 2],
            }
            assert (
                _maybe_create_prefetcher(
                    output_dir, moe_config, store,
                    margin=1.5, max_positions=8, scored_eviction=True,
                )
                is None
            )
        finally:
            store.close()
```

(`_make_synthetic_moe_weights` is already imported at the top of `tests/test_flash_moe_model.py` from `tests.test_flash_moe_bundler` — verify, and add the import if that file imports it differently.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_flash_moe_model.py::TestMoePrefetchIntegration -v`
Expected: FAIL — `FlashMoE.__init__` has no `prefetcher` kwarg; `_maybe_create_prefetcher` does not exist.

- [ ] **Step 3: Implement `flash_moe.py` changes**

In `olmlx/engine/flash/flash_moe.py`, add the import (top of file, guarded to avoid a cycle — `moe_prefetch` imports `moe_weight_store`, not `flash_moe`, so a plain import is safe; use TYPE_CHECKING to keep it annotation-only):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmlx.engine.flash.moe_prefetch import MoePrefetcher
```

Constructor: add trailing keyword param and attribute (after `activation` at line 31-40):

```python
        activation: nn.Module | None = None,
        prefetcher: "MoePrefetcher | None" = None,
    ):
        ...
        self._activation = activation
        self.prefetcher = prefetcher
```

`__call__`: replace the opening of the method (lines 73-83) with:

```python
        # Wait for any pending prefetch targeting THIS layer (submitted by
        # the previous MoE layer) BEFORE the mx.eval below — the prediction
        # thread owns the only background mx.eval, and it must have finished
        # before the main thread evaluates.
        if self.prefetcher is not None:
            self.prefetcher.wait(self.layer_idx)

        # Collect unique expert indices for the SSD read list (Python-side, one eval per layer).
        # Invariant: ``unique_experts`` must contain every value in ``inds`` so the remap LUT
        # has a valid entry for each routed token. ``mx.take`` would silently return the
        # sentinel 0xFFFFFFFF otherwise — keep these two derivations from the same ``inds``.
        mx.eval(inds)
        flat_inds = inds.reshape(-1).tolist()
        unique_experts = sorted(set(flat_inds))

        # Kick off prediction + SSD I/O for the NEXT MoE layer before this
        # layer's blocking load, so they overlap with the load, the expert
        # matmuls, and the intervening dense/attention work. Must come AFTER
        # the mx.eval above (concurrent mx.eval is unsafe) and the main
        # thread must not eval again until the next wait().
        if self.prefetcher is not None:
            self.prefetcher.submit(self.layer_idx, x)

        # Load experts from SSD (or RAM cache); LoadedExperts includes a device-side
        # remap LUT that maps global expert idx -> local stack position.
        loaded = self.weight_store.load_experts(self.layer_idx, unique_experts)
```

- [ ] **Step 4: Implement `flash_moe_model.py` changes**

1. `FlashMoeModelWrapper.__init__` gains a `prefetcher` keyword and stores it (used by manager teardown via `getattr(lm.model, "prefetcher", None)`); it is threaded into `_replace_moe_layers`:

```python
    def __init__(
        self,
        model: nn.Module,
        weight_store: FlashMoeWeightStore,
        moe_layer_indices: list[int],
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        prefetcher: "MoePrefetcher | None" = None,
    ):
        super().__init__()
        self._model = model
        self._weight_store = weight_store
        # Surfaced for ModelManager._close_model_resources, which closes
        # ``lm.model.prefetcher`` BEFORE the weight store (prefetch tasks
        # submit into the store's I/O pool). Mirrors FlashModelWrapper.
        self.prefetcher = prefetcher
        _replace_moe_layers(
            model,
            weight_store,
            moe_layer_indices,
            hidden_size,
            intermediate_size,
            num_experts,
            num_experts_per_tok,
            prefetcher=prefetcher,
        )
```

(Check the actual current attribute assignments at the top of `__init__` — keep `self._model` / `self._weight_store` exactly as they are today; only add the `prefetcher` line and the kwarg.) Add the TYPE_CHECKING import as in `flash_moe.py`.

2. `_replace_moe_layers` signature gains `prefetcher: "MoePrefetcher | None" = None` (trailing keyword) and passes it to every `FlashMoE(...)` construction:

```python
        flash_moe = FlashMoE(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            weight_store=weight_store,
            activation=activation,
            prefetcher=prefetcher,
        )
```

3. New module-level helper (place before `wrap_flash_moe`):

```python
def _maybe_create_prefetcher(
    flash_moe_dir: Path,
    moe_config: dict[str, Any],
    store: FlashMoeWeightStore,
    *,
    margin: float,
    max_positions: int,
    scored_eviction: bool,
) -> "MoePrefetcher | None":
    """Load the trained lookahead bank and build a prefetcher, or None.

    Never raises: a missing/corrupt/stale ``moe_lookahead/`` directory is an
    optional accelerator, not a load failure. A sidecar that disagrees with
    the bundle (re-bundled model, wrong architecture) is rejected — serving a
    wrong-shaped predictor would prefetch garbage every token.
    """
    from olmlx.engine.flash.moe_predictor import MoeLookaheadBank
    from olmlx.engine.flash.moe_prefetch import MoePrefetcher

    lookahead_dir = flash_moe_dir / "moe_lookahead"
    if not lookahead_dir.exists():
        logger.debug("No moe_lookahead/ in %s — prefetch disabled", flash_moe_dir)
        return None
    try:
        bank = MoeLookaheadBank.load(lookahead_dir)
    except Exception:
        logger.warning(
            "Failed to load MoE lookahead bank from %s — prefetch disabled",
            lookahead_dir,
            exc_info=True,
        )
        return None

    expected = {
        "hidden_size": moe_config["hidden_size"],
        "num_experts": moe_config["num_experts"],
        "moe_layer_indices": sorted(moe_config["moe_layer_indices"]),
    }
    actual = {
        "hidden_size": bank.hidden_size,
        "num_experts": bank.num_experts,
        "moe_layer_indices": bank.moe_layer_indices,
    }
    if expected != actual:
        logger.warning(
            "MoE lookahead bank at %s does not match the bundle "
            "(expected %s, got %s) — prefetch disabled; retrain with "
            "`olmlx flash train-moe-lookahead`",
            lookahead_dir,
            expected,
            actual,
        )
        return None

    logger.info(
        "MoE expert prefetch enabled (margin=%.2f, max_positions=%d, "
        "scored_eviction=%s)",
        margin,
        max_positions,
        scored_eviction,
    )
    return MoePrefetcher(
        bank,
        store,
        margin=margin,
        max_positions=max_positions,
        scored_eviction=scored_eviction,
    )
```

Add `from pathlib import Path` and `from typing import Any` to the module imports if missing (both are already imported at the top of `flash_moe_model.py`).

4. `wrap_flash_moe` gains the four keyword params (defaults OFF so the calibration loader is unaffected) and closes the prefetcher on failure:

```python
def wrap_flash_moe(
    model: Any,
    flash_moe_dir: Path | str,
    *,
    io_threads: int,
    cache_budget_experts: int,
    prefetch: bool = False,
    lookahead_margin: float = 1.5,
    prefetch_max_positions: int = 8,
    scored_eviction: bool = True,
) -> tuple[Any, Any]:
```

Inside, after the store is created and before the `try`:

```python
    prefetcher = None
    if prefetch:
        prefetcher = _maybe_create_prefetcher(
            flash_moe_dir,
            moe_config,
            store,
            margin=lookahead_margin,
            max_positions=prefetch_max_positions,
            scored_eviction=scored_eviction,
        )
    try:
        wrapped = FlashMoeModelWrapper(
            model,
            store,
            moe_layer_indices=moe_config["moe_layer_indices"],
            hidden_size=moe_config["hidden_size"],
            intermediate_size=moe_config["intermediate_size"],
            num_experts=moe_config["num_experts"],
            num_experts_per_tok=moe_config["num_experts_per_tok"],
            prefetcher=prefetcher,
        )
        # Materialize only non-expert weights.
        mx.eval(wrapped.parameters())
    except Exception:
        if prefetcher is not None:
            prefetcher.close()
        store.close()
        raise
    return wrapped, store
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_flash_moe_model.py tests/test_flash_moe.py tests/test_flash_moe_integration.py -v`
Expected: all PASS (new class and all pre-existing Flash-MoE tests — the new kwargs are optional and default-off).

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/flash/flash_moe.py olmlx/engine/flash/flash_moe_model.py tests/test_flash_moe_model.py
git commit -m "feat(flash): wire MoePrefetcher into FlashMoE forward and wrap_flash_moe"
```

---

### Task 6: Config plumbing (Settings → FlashMoeConfig → ModelManager)

**Files:**
- Modify: `olmlx/config.py` (Settings fields near line 431-433; `FlashMoeConfig` dataclass at line 840)
- Modify: `olmlx/engine/registry.py` (`resolved_flash_moe`, line 901)
- Modify: `olmlx/engine/model_manager.py` (`_load_flash_moe_model`, line 3259)
- Test: `tests/test_moe_prefetch.py` (append a config test class)

**Interfaces:**
- Consumes: `wrap_flash_moe` keyword params (Task 5).
- Produces: four new `Settings` fields (env: `OLMLX_FLASH_MOE_PREFETCH`, `OLMLX_FLASH_MOE_LOOKAHEAD_MARGIN`, `OLMLX_FLASH_MOE_PREFETCH_MAX_POSITIONS`, `OLMLX_FLASH_MOE_SCORED_EVICTION`) and four new `FlashMoeConfig` fields with matching defaults. Global-only (no per-model registry overrides — YAGNI; the existing three per-model fields stay as they are).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_moe_prefetch.py`:

```python
class TestPrefetchConfig:
    def test_flash_moe_config_prefetch_defaults(self):
        from olmlx.config import FlashMoeConfig

        cfg = FlashMoeConfig(enabled=True, cache_budget_experts=48, io_threads=32)
        assert cfg.prefetch is True
        assert cfg.lookahead_margin == 1.5
        assert cfg.prefetch_max_positions == 8
        assert cfg.scored_eviction is True

    def test_settings_prefetch_defaults(self, monkeypatch):
        from olmlx.config import Settings

        for var in (
            "OLMLX_FLASH_MOE_PREFETCH",
            "OLMLX_FLASH_MOE_LOOKAHEAD_MARGIN",
            "OLMLX_FLASH_MOE_PREFETCH_MAX_POSITIONS",
            "OLMLX_FLASH_MOE_SCORED_EVICTION",
        ):
            monkeypatch.delenv(var, raising=False)
        s = Settings(_env_file=None)
        assert s.flash_moe_prefetch is True
        assert s.flash_moe_lookahead_margin == 1.5
        assert s.flash_moe_prefetch_max_positions == 8
        assert s.flash_moe_scored_eviction is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_moe_prefetch.py::TestPrefetchConfig -v`
Expected: FAIL with `TypeError` / `AttributeError` (fields don't exist).

- [ ] **Step 3: Implement config changes**

1. `olmlx/config.py` — after `flash_moe_io_threads` (line 433), add:

```python
    # MoE expert prefetch (requires a trained lookahead bank —
    # ``olmlx flash train-moe-lookahead``; silently off without one).
    flash_moe_prefetch: bool = True
    flash_moe_lookahead_margin: Annotated[float, Field(ge=1.0)] = 1.5
    flash_moe_prefetch_max_positions: Annotated[int, Field(gt=0)] = 8
    flash_moe_scored_eviction: bool = True
```

2. `olmlx/config.py` — `FlashMoeConfig` dataclass (line 840), add defaulted fields:

```python
@dataclass
class FlashMoeConfig:
    """Resolved Flash-MoE configuration: per-model overrides global Settings."""

    enabled: bool
    cache_budget_experts: int
    io_threads: int
    # Prefetch knobs are global-only (no per-model registry override).
    prefetch: bool = True
    lookahead_margin: float = 1.5
    prefetch_max_positions: int = 8
    scored_eviction: bool = True
```

3. `olmlx/engine/registry.py` — `resolved_flash_moe` (line 905-919), add to the constructor call:

```python
            prefetch=settings.flash_moe_prefetch,
            lookahead_margin=settings.flash_moe_lookahead_margin,
            prefetch_max_positions=settings.flash_moe_prefetch_max_positions,
            scored_eviction=settings.flash_moe_scored_eviction,
```

4. `olmlx/engine/model_manager.py` — `_load_flash_moe_model`'s `wrap_flash_moe` call (line 3259-3264):

```python
        wrapped, _ = wrap_flash_moe(
            model,
            flash_moe_dir,
            io_threads=flash_moe_config.io_threads,
            cache_budget_experts=flash_moe_config.cache_budget_experts,
            prefetch=flash_moe_config.prefetch,
            lookahead_margin=flash_moe_config.lookahead_margin,
            prefetch_max_positions=flash_moe_config.prefetch_max_positions,
            scored_eviction=flash_moe_config.scored_eviction,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_moe_prefetch.py tests/test_config.py -v`
Expected: all PASS (new tests plus no regressions in the existing config suite)

- [ ] **Step 5: Commit**

```bash
git add olmlx/config.py olmlx/engine/registry.py olmlx/engine/model_manager.py tests/test_moe_prefetch.py
git commit -m "feat(flash): OLMLX_FLASH_MOE_PREFETCH settings plumbed to wrap_flash_moe"
```

---

### Task 7: Offline training pipeline (`moe_lookahead_train.py`)

**Files:**
- Create: `olmlx/engine/flash/moe_lookahead_train.py`
- Test: `tests/test_moe_lookahead_train.py` (new)

**Interfaces:**
- Consumes: `MoeLookaheadBank` (Task 2), `_train_single_predictor` from `olmlx/engine/flash/prepare.py:489` (signature: `(pred, inputs, targets, epochs, lr, pos_weight_multiplier=1.0, epoch_callback=None)`), `load_flash_moe_model` from `flash_moe_model.py:307`, `_get_c4_calibration_data(num_samples)` / `_get_calibration_data(num_samples)` from `prepare.py:124/176`.
- Produces (used by Task 8):
  - `build_multi_hot(inds: np.ndarray, num_experts: int) -> np.ndarray` — `(P, K)` int indices → `(P, num_experts)` float32 multi-hot.
  - `recall_at_m(pred_scores: np.ndarray, true_inds: np.ndarray, m: int) -> float` — mean fraction of true experts found in the top-m predictions.
  - `record_moe_router_traces(model: Any, tokenizer: Any, texts: list[str], moe_layer_indices: list[int], *, max_positions_per_layer: int = 4096, max_tokens_per_text: int = 512) -> dict[int, tuple[np.ndarray, np.ndarray]]` — per MoE layer: `(hidden (P, H) float32, router top-k inds (P, K) int32)`, positions aligned across layers. Layers whose module lacks `_route` (Gemma4-style) are skipped with a warning.
  - `train_from_traces(traces, moe_layer_indices, hidden_size, num_experts, *, num_experts_per_tok, rank=128, epochs=5, lr=1e-3, holdout_fraction=0.1, eval_margin=1.5, progress_callback=None) -> tuple[MoeLookaheadBank, dict[str, float]]` — returns the trained bank and per-pair holdout `recall@m` keyed `"L→M"`.
  - `train_moe_lookahead(model_path: str, flash_moe_dir: Path, *, rank=128, epochs=5, lr=1e-3, num_samples=32, calibration_dataset: str | None = None, max_positions_per_layer=4096, holdout_fraction=0.1, io_threads=16, cache_budget_experts=48, progress_callback=None) -> Path` — orchestrator: load Flash-MoE-wrapped model, record, train, save to `<flash_moe_dir>/moe_lookahead`, close the store, return the path.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_moe_lookahead_train.py`:

```python
"""Tests for olmlx.engine.flash.moe_lookahead_train."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from olmlx.engine.flash.moe_lookahead_train import (
    build_multi_hot,
    recall_at_m,
    record_moe_router_traces,
    train_from_traces,
)


class TestBuildMultiHot:
    def test_basic(self):
        inds = np.array([[0, 2], [1, 3]])
        out = build_multi_hot(inds, num_experts=4)
        expected = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32
        )
        np.testing.assert_array_equal(out, expected)
        assert out.dtype == np.float32


class TestRecallAtM:
    def test_perfect_prediction(self):
        scores = np.array([[0.9, 0.8, 0.1, 0.0], [0.0, 0.1, 0.8, 0.9]])
        true_inds = np.array([[0, 1], [2, 3]])
        assert recall_at_m(scores, true_inds, m=2) == 1.0

    def test_half_recall(self):
        scores = np.array([[0.9, 0.0, 0.1, 0.8]])  # top-2 = {0, 3}
        true_inds = np.array([[0, 1]])  # only 0 found
        assert recall_at_m(scores, true_inds, m=2) == 0.5


class TestTrainFromTraces:
    def test_learns_deterministic_mapping(self):
        """A learnable hidden->expert rule must reach high holdout recall."""
        rng = np.random.default_rng(42)
        num_experts, hidden_size, positions = 4, 16, 400
        hid = rng.standard_normal((positions, hidden_size)).astype(np.float32)
        # Rule: expert pair determined by the sign of feature 0
        next_inds = np.where(
            hid[:, :1] > 0,
            np.array([[0, 1]]),
            np.array([[2, 3]]),
        ).astype(np.int32)
        traces = {
            1: (hid, next_inds),  # inds at layer 1 unused for pair 1->2 input
            2: (hid, next_inds),  # targets for the 1->2 pair
        }
        bank, recalls = train_from_traces(
            traces,
            [1, 2],
            hidden_size,
            num_experts,
            num_experts_per_tok=2,
            rank=8,
            epochs=200,
            holdout_fraction=0.1,
        )
        assert "1→2" in recalls
        assert recalls["1→2"] > 0.9  # trivially learnable rule

    def test_missing_layer_trace_skipped(self):
        hid = np.zeros((10, 16), dtype=np.float32)
        inds = np.zeros((10, 2), dtype=np.int32)
        bank, recalls = train_from_traces(
            {1: (hid, inds)},  # layer 4 trace missing
            [1, 4],
            16,
            8,
            num_experts_per_tok=2,
            rank=4,
            epochs=1,
        )
        assert recalls == {}


class _FakeRoutedMoE(nn.Module):
    """Stands in for a _FlashMoEBase replacement: has _route, returns x."""

    def __init__(self, num_experts: int, k: int):
        super().__init__()
        self._num_experts = num_experts
        self._k = k

    def _route(self, x: mx.array):
        flat = x.reshape(-1, x.shape[-1])
        logits = mx.zeros((flat.shape[0], self._num_experts))
        inds = mx.argpartition(logits, kth=-self._k, axis=-1)[..., -self._k :]
        return inds.reshape(*x.shape[:-1], self._k), None

    def __call__(self, x: mx.array) -> mx.array:
        return x


class _FakeLayer(nn.Module):
    def __init__(self, moe: nn.Module | None):
        super().__init__()
        if moe is not None:
            self.mlp = moe

    def __call__(self, x: mx.array, **kwargs) -> mx.array:
        return getattr(self, "mlp", lambda v: v)(x) + 0.1


class _FakeModel(nn.Module):
    def __init__(self, hidden: int, num_experts: int, k: int):
        super().__init__()
        self._embed = nn.Embedding(32, hidden)
        self.layers = [
            _FakeLayer(None),  # dense layer 0
            _FakeLayer(_FakeRoutedMoE(num_experts, k)),  # MoE layer 1
            _FakeLayer(_FakeRoutedMoE(num_experts, k)),  # MoE layer 2
        ]

    def __call__(self, inputs: mx.array) -> mx.array:
        h = self._embed(inputs)
        for layer in self.layers:
            h = layer(h)
        return h


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 for c in text][:16]


class TestRecordTraces:
    def test_records_hidden_and_inds_per_moe_layer(self):
        model = _FakeModel(hidden=8, num_experts=4, k=2)
        traces = record_moe_router_traces(
            model,
            _FakeTokenizer(),
            ["hello world", "foo bar"],
            [1, 2],
            max_positions_per_layer=100,
        )
        assert set(traces.keys()) == {1, 2}
        hid, inds = traces[1]
        assert hid.shape[1] == 8
        assert inds.shape[1] == 2
        assert hid.shape[0] == inds.shape[0]
        # Both texts' positions recorded, aligned across layers
        assert traces[1][0].shape[0] == traces[2][0].shape[0]

    def test_position_cap_respected(self):
        model = _FakeModel(hidden=8, num_experts=4, k=2)
        traces = record_moe_router_traces(
            model,
            _FakeTokenizer(),
            ["a much longer text that produces many tokens"] * 4,
            [1, 2],
            max_positions_per_layer=5,
        )
        assert traces[1][0].shape[0] <= 5

    def test_layer_without_route_skipped(self):
        model = _FakeModel(hidden=8, num_experts=4, k=2)
        del model.layers[2].mlp  # layer 2 has no MoE module at all
        traces = record_moe_router_traces(
            model, _FakeTokenizer(), ["hello"], [1, 2]
        )
        assert 1 in traces
        assert 2 not in traces
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_moe_lookahead_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.engine.flash.moe_lookahead_train'`

- [ ] **Step 3: Implement `moe_lookahead_train.py`**

Create `olmlx/engine/flash/moe_lookahead_train.py`:

```python
"""Offline training for Flash-MoE expert lookahead heads.

Records (hidden state at MoE layer L, router top-k at the next MoE layer)
traces by running the Flash-MoE-wrapped model over calibration texts, then
trains one low-rank head per consecutive MoE-layer pair with recall-biased
BCE. Router targets come free from the resident gates — no labels beyond the
forward pass itself.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from olmlx.engine.flash.moe_predictor import MoeLookaheadBank
from olmlx.engine.flash.prepare import _train_single_predictor

logger = logging.getLogger(__name__)


def build_multi_hot(inds: np.ndarray, num_experts: int) -> np.ndarray:
    """(P, K) integer expert indices -> (P, num_experts) float32 multi-hot."""
    out = np.zeros((inds.shape[0], num_experts), dtype=np.float32)
    out[np.arange(inds.shape[0])[:, None], inds] = 1.0
    return out


def recall_at_m(pred_scores: np.ndarray, true_inds: np.ndarray, m: int) -> float:
    """Mean fraction of true experts found in the top-m predicted experts."""
    m = min(m, pred_scores.shape[1])
    top_m = np.argpartition(-pred_scores, m - 1, axis=1)[:, :m]
    hits = 0
    for row_top, row_true in zip(top_m, true_inds):
        hits += len(set(row_top.tolist()) & set(row_true.tolist()))
    return hits / true_inds.size


def record_moe_router_traces(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    moe_layer_indices: list[int],
    *,
    max_positions_per_layer: int = 4096,
    max_tokens_per_text: int = 512,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Record per-MoE-layer (input hidden, router top-k) over *texts*.

    Temporarily wraps each MoE layer's replacement module (which must expose
    ``_route``) with a recorder, runs full forward passes, and restores the
    original modules. Positions are aligned across layers (each forward
    contributes the same positions everywhere), so pair training can zip
    layer L's hiddens with layer M's indices.

    Modules without ``_route`` (Gemma4-style pre-routed experts, or dense
    layers) are skipped with a warning.
    """
    import mlx.nn as nn

    class _Recorder(nn.Module):
        def __init__(self, inner: Any, hidden_sink: list, inds_sink: list):
            super().__init__()
            # Bypass nn.Module attribute registration for the inner module —
            # the recorder must not appear to own its parameters.
            object.__setattr__(self, "_inner", inner)
            object.__setattr__(self, "_hidden_sink", hidden_sink)
            object.__setattr__(self, "_inds_sink", inds_sink)

        def __call__(self, x: mx.array) -> mx.array:
            recorded = sum(a.shape[0] for a in self._hidden_sink)
            if recorded < max_positions_per_layer:
                flat = x.reshape(-1, x.shape[-1])
                inds, _ = self._inner._route(x)
                flat_inds = inds.reshape(-1, inds.shape[-1])
                mx.eval(flat, flat_inds)
                budget = max_positions_per_layer - recorded
                self._hidden_sink.append(
                    np.array(flat.astype(mx.float32))[:budget]
                )
                self._inds_sink.append(np.array(flat_inds)[:budget])
            return self._inner(x)

    sinks: dict[int, tuple[list, list]] = {}
    originals: dict[int, tuple[Any, str]] = {}
    layers = model.layers

    for layer_idx in sorted(moe_layer_indices):
        layer = layers[layer_idx]
        # The Flash-MoE replacement sits where the original MoE module was.
        attr = None
        for candidate in ("mlp", "block_sparse_moe", "mixer", "experts"):
            mod = getattr(layer, candidate, None)
            if mod is not None:
                attr = candidate
                break
        mod = getattr(layer, attr, None) if attr else None
        if mod is None or not hasattr(mod, "_route"):
            logger.warning(
                "MoE layer %d has no _route-style module — skipping trace "
                "recording (Gemma4-style pre-routed layers are unsupported)",
                layer_idx,
            )
            continue
        hidden_sink: list = []
        inds_sink: list = []
        sinks[layer_idx] = (hidden_sink, inds_sink)
        originals[layer_idx] = (mod, attr)
        setattr(layer, attr, _Recorder(mod, hidden_sink, inds_sink))

    try:
        for text in texts:
            tokens = tokenizer.encode(text)[:max_tokens_per_text]
            if not tokens:
                continue
            out = model(mx.array([tokens]))
            mx.eval(out)
    finally:
        for layer_idx, (mod, attr) in originals.items():
            setattr(layers[layer_idx], attr, mod)

    traces: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer_idx, (hidden_sink, inds_sink) in sinks.items():
        if not hidden_sink:
            logger.warning("No positions recorded for MoE layer %d", layer_idx)
            continue
        traces[layer_idx] = (
            np.concatenate(hidden_sink, axis=0),
            np.concatenate(inds_sink, axis=0).astype(np.int32),
        )
    return traces


def train_from_traces(
    traces: dict[int, tuple[np.ndarray, np.ndarray]],
    moe_layer_indices: list[int],
    hidden_size: int,
    num_experts: int,
    *,
    num_experts_per_tok: int,
    rank: int = 128,
    epochs: int = 5,
    lr: float = 1e-3,
    holdout_fraction: float = 0.1,
    eval_margin: float = 1.5,
    progress_callback: Callable[[str, float], None] | None = None,
) -> tuple[MoeLookaheadBank, dict[str, float]]:
    """Train per-pair heads from recorded traces.

    Returns ``(bank, recalls)`` where ``recalls`` maps ``"L→M"`` to holdout
    recall@m (m = ceil(eval_margin * num_experts_per_tok)). Pairs with a
    missing trace are skipped (absent from ``recalls``).
    """
    bank = MoeLookaheadBank(
        moe_layer_indices,
        hidden_size,
        num_experts,
        rank=rank,
        num_experts_per_tok=num_experts_per_tok,
    )
    indices = bank.moe_layer_indices
    recalls: dict[str, float] = {}
    num_pairs = len(indices) - 1

    for pair_idx in range(num_pairs):
        src, dst = indices[pair_idx], indices[pair_idx + 1]
        if src not in traces or dst not in traces:
            logger.warning("No trace pair for MoE layers %d→%d, skipping", src, dst)
            continue
        hid = traces[src][0]
        next_inds = traces[dst][1]
        n = min(len(hid), len(next_inds))
        hid, next_inds = hid[:n], next_inds[:n]

        n_hold = int(n * holdout_fraction) if n > 10 else 0
        n_train = n - n_hold

        def _on_epoch(epoch: int, _p=pair_idx) -> None:
            if progress_callback:
                progress_callback(
                    f"Training pair {_p + 1}/{num_pairs} epoch {epoch + 1}/{epochs}",
                    (_p * epochs + epoch + 1) / (num_pairs * epochs),
                )

        # 2x recall bias, same as the dense lookahead: a false negative is a
        # synchronous SSD miss on the critical path; a false positive is one
        # wasted read.
        _train_single_predictor(
            bank.heads[pair_idx],
            mx.array(hid[:n_train]),
            mx.array(build_multi_hot(next_inds[:n_train], num_experts)),
            epochs=epochs,
            lr=lr,
            pos_weight_multiplier=2.0,
            epoch_callback=_on_epoch,
        )

        if n_hold:
            scores = bank.heads[pair_idx](mx.array(hid[n_train:]))
            mx.eval(scores)
            m = min(num_experts, math.ceil(eval_margin * num_experts_per_tok))
            recalls[f"{src}→{dst}"] = recall_at_m(
                np.array(scores, dtype=np.float32), next_inds[n_train:], m
            )

    return bank, recalls


def train_moe_lookahead(
    model_path: str,
    flash_moe_dir: Path,
    *,
    rank: int = 128,
    epochs: int = 5,
    lr: float = 1e-3,
    num_samples: int = 32,
    calibration_dataset: str | None = None,
    max_positions_per_layer: int = 4096,
    holdout_fraction: float = 0.1,
    io_threads: int = 16,
    cache_budget_experts: int = 48,
    progress_callback: Callable[[str, float], None] | None = None,
) -> Path:
    """End-to-end: record traces on the Flash-MoE model, train, save.

    Saves to ``<flash_moe_dir>/moe_lookahead`` and returns that path. Prints
    per-pair holdout recall via the logger; the CLI surfaces it to the user.
    """
    import json

    from olmlx.engine.flash.flash_moe_model import load_flash_moe_model
    from olmlx.engine.flash.prepare import (
        _get_c4_calibration_data,
        _get_calibration_data,
    )

    moe_config = json.loads((flash_moe_dir / "flash_moe_config.json").read_text())
    moe_layer_indices = moe_config["moe_layer_indices"]
    if len(moe_layer_indices) < 2:
        raise ValueError(
            f"Need at least 2 MoE layers for lookahead, got {moe_layer_indices}"
        )

    if calibration_dataset == "synthetic":
        texts = _get_calibration_data(num_samples)
    else:
        texts = _get_c4_calibration_data(num_samples)

    if progress_callback:
        progress_callback("Loading model (Flash-MoE, lazy)", 0.0)

    model, tokenizer, store = load_flash_moe_model(
        model_path,
        flash_moe_dir,
        cache_budget_experts=cache_budget_experts,
        io_threads=io_threads,
    )
    try:
        if progress_callback:
            progress_callback("Recording router traces", 0.05)
        traces = record_moe_router_traces(
            model,
            tokenizer,
            texts,
            moe_layer_indices,
            max_positions_per_layer=max_positions_per_layer,
        )
    finally:
        store.close()

    if not traces:
        raise RuntimeError(
            "No router traces recorded — model has no _route-style MoE layers"
        )

    bank, recalls = train_from_traces(
        traces,
        moe_layer_indices,
        hidden_size=moe_config["hidden_size"],
        num_experts=moe_config["num_experts"],
        num_experts_per_tok=moe_config["num_experts_per_tok"],
        rank=rank,
        epochs=epochs,
        lr=lr,
        holdout_fraction=holdout_fraction,
        progress_callback=lambda desc, frac: (
            progress_callback(desc, 0.1 + frac * 0.85) if progress_callback else None
        ),
    )

    out_dir = flash_moe_dir / "moe_lookahead"
    bank.save(out_dir)
    for pair, recall in recalls.items():
        logger.info("Holdout recall@m for pair %s: %.3f", pair, recall)
    if progress_callback:
        progress_callback("Done", 1.0)
    return out_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_moe_lookahead_train.py -v`
Expected: all PASS. (The deterministic-mapping test trains 200 epochs on a rank-8 head over 360 samples — runs in seconds.)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/flash/moe_lookahead_train.py tests/test_moe_lookahead_train.py
git commit -m "feat(flash): offline trace recording + training for MoE expert lookahead"
```

---

### Task 8: CLI — `olmlx flash train-moe-lookahead` + `--train-lookahead` on prepare

**Files:**
- Modify: `olmlx/cli.py` — new command function near `cmd_flash_prepare` (line 2680), new subparser near the `flash` parser block (line 3368-3417), new dispatch entry in the command map (line 3930-3931), `--train-lookahead` chaining in `_cmd_flash_moe_prepare` (line 2702).
- Test: `tests/test_flash_cli.py` (append)

**Interfaces:**
- Consumes: `train_moe_lookahead` (Task 7).
- Produces: `("flash", "train-moe-lookahead")` entry in `_COMMAND_HANDLERS` (cli.py:3919) → `cmd_flash_train_moe_lookahead(args)`; args: `model` (positional), `--rank` (default 128), `--epochs` (default 5), `--samples` (default 32), `--calibration-dataset` (default None → C4). `olmlx flash prepare --train-lookahead` chains training after MoE bundling. Note: `_COMMAND_HANDLERS` resolves handler names via `globals()` at import time (`_validate_command_handlers`), so `cmd_flash_train_moe_lookahead` must be defined above the dict — placing it after `_cmd_flash_dense_prepare` (line ~2760) satisfies this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_flash_cli.py` (it already imports `build_parser` from `olmlx.cli` at the top):

```python
class TestTrainMoeLookaheadCli:
    def test_parser_accepts_train_moe_lookahead(self):
        from olmlx.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["flash", "train-moe-lookahead", "my-model", "--rank", "64",
             "--epochs", "3", "--samples", "8"]
        )
        assert args.flash_command == "train-moe-lookahead"
        assert args.model == "my-model"
        assert args.rank == 64
        assert args.epochs == 3
        assert args.samples == 8

    def test_dispatch_map_has_entry(self):
        import olmlx.cli as cli

        # The (command, subcommand) -> handler-name registry (cli.py:3919)
        assert ("flash", "train-moe-lookahead") in cli._COMMAND_HANDLERS
        assert hasattr(cli, cli._COMMAND_HANDLERS[("flash", "train-moe-lookahead")])

    def test_prepare_parser_accepts_train_lookahead_flag(self):
        from olmlx.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["flash", "prepare", "my-model", "--train-lookahead"])
        assert args.train_lookahead is True
```

(`build_parser` is the real parser factory at `olmlx/cli.py:3113`, already imported by `tests/test_flash_cli.py`; `_COMMAND_HANDLERS` is the real dispatch registry at `olmlx/cli.py:3919`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_flash_cli.py -k TrainMoeLookahead -v`
Expected: FAIL (unknown subcommand / missing dispatch entry / unknown `--train-lookahead` flag).

- [ ] **Step 3: Implement CLI changes**

1. New command function, placed directly after `_cmd_flash_dense_prepare` in `olmlx/cli.py`:

```python
def cmd_flash_train_moe_lookahead(args):
    """Train expert lookahead predictors for Flash-MoE prefetch."""
    from pathlib import Path

    _configure_logging()
    store = _create_store()
    _resolved = store.registry.resolve(args.model)
    hf_path = _resolved.hf_path if _resolved is not None else args.model
    local_dir = store.ensure_downloaded(hf_path)
    model_path = str(local_dir)

    flash_moe_dir = Path(model_path) / "flash_moe"
    if not (flash_moe_dir / "flash_moe_layout.json").exists():
        print(f"No Flash-MoE bundle at {flash_moe_dir}")
        print("Run `olmlx flash prepare <model>` first.")
        raise SystemExit(1)

    from olmlx.engine.flash.moe_lookahead_train import train_moe_lookahead

    print(f"Training MoE expert lookahead for {args.model}...")
    print(f"  Rank: {args.rank}  Epochs: {args.epochs}  Samples: {args.samples}")
    print()

    out_dir = train_moe_lookahead(
        model_path,
        flash_moe_dir,
        rank=args.rank,
        epochs=args.epochs,
        num_samples=args.samples,
        calibration_dataset=args.calibration_dataset,
        progress_callback=_flash_progress,
    )

    print("\nMoE lookahead training complete!")
    print(f"  Output: {out_dir}")
    print("\nExpert prefetch activates automatically on the next model load")
    print("(disable with OLMLX_FLASH_MOE_PREFETCH=false).")
```

2. Subparser, added inside the flash parser block (after the `info_p` block at line 3414-3417):

```python
    tml_p = flash_sub.add_parser(
        "train-moe-lookahead",
        help="Train expert lookahead predictors for Flash-MoE prefetch",
    )
    tml_p.add_argument("model", help="Model name or HF path (must be flash-prepared)")
    tml_p.add_argument(
        "--rank", type=int, default=128, help="Predictor rank (default: 128)"
    )
    tml_p.add_argument(
        "--epochs", type=int, default=5, help="Training epochs (default: 5)"
    )
    tml_p.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Calibration texts to trace (default: 32; each forward pass "
        "streams experts from SSD, so tracing is slow)",
    )
    tml_p.add_argument(
        "--calibration-dataset",
        type=str,
        default=None,
        help="Calibration dataset: 'c4' (default) or 'synthetic'",
    )
```

3. Dispatch entry, next to `("flash", "prepare")` in `_COMMAND_HANDLERS` (line 3930):

```python
    ("flash", "train-moe-lookahead"): "cmd_flash_train_moe_lookahead",
```

(`_validate_command_handlers()` runs at import time and will fail loudly on a typo or a handler defined below the dict.)

4. `--train-lookahead` chaining. Add to the `prepare_p` parser block:

```python
    prepare_p.add_argument(
        "--train-lookahead",
        action="store_true",
        help="MoE models: also train expert lookahead predictors after bundling",
    )
```

and extend `_cmd_flash_moe_prepare` (line 2702) — after the existing completion prints:

```python
    if getattr(args, "train_lookahead", False):
        from olmlx.engine.flash.moe_lookahead_train import train_moe_lookahead

        print("\nTraining expert lookahead predictors...")
        lookahead_dir = train_moe_lookahead(
            model_path,
            output_dir,
            progress_callback=_flash_progress,
        )
        print(f"  Lookahead predictors: {lookahead_dir}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_flash_cli.py -v`
Expected: all PASS

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest`
Expected: PASS (no regressions anywhere — particularly `tests/test_flash_moe*.py`, `tests/test_flash_prefetch.py`, `tests/test_flash_cli.py`).

- [ ] **Step 6: Commit**

```bash
git add olmlx/cli.py tests/test_flash_cli.py
git commit -m "feat(cli): flash train-moe-lookahead command + prepare --train-lookahead"
```

---

## Post-plan verification (manual, SSD-bound model required)

Not a task — a checklist for the human/agent with a real Flash-MoE model available:

1. `olmlx flash train-moe-lookahead <model>` — note per-pair holdout recall@m.
2. Serve with prefetch on vs `OLMLX_FLASH_MOE_PREFETCH=false`; compare `ExpertCacheStats.hit_rate()` (log line on unload) and decode tok/s.
3. Confirm identical outputs (temperature 0) with prefetch on vs off.
4. Lower `OLMLX_FLASH_MOE_CACHE_BUDGET_EXPERTS` stepwise; confirm hit-rate holds better than LRU-only at the same budget.
