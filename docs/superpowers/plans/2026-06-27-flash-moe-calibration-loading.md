# Flash-MoE-backed KV-quant calibration loading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let spectral/shard KV-quant calibration run on models too large to fully load (e.g. `Qwen3-235B-A22B-4bit` on 64 GB) by loading them through the Flash-MoE path when a `flash_moe/` bundle exists.

**Architecture:** Add one free helper `load_flash_moe_model` in `flash_moe_model.py` that reproduces `model_manager._load_flash_moe_model`'s recipe (lazy load → weight store → `FlashMoeModelWrapper` → eval non-expert params). The shared calibration loader `_load_calibration_model` probes for a bundle and calls the helper (else keeps today's full load), returning the store so `calibrate_model` / `calibrate_model_shard` can close it in a `finally`.

**Tech Stack:** Python, MLX (`mlx.core`, `mlx_lm`), pytest. Spec: `docs/superpowers/specs/2026-06-27-flash-moe-calibration-loading-design.md`.

## Global Constraints

- Calibration expert budget constant: `_CALIBRATION_CACHE_BUDGET_EXPERTS = 8`.
- Calibration io-threads constant: `_CALIBRATION_IO_THREADS = 32`.
- Bundle probe path: `Path(model_path) / "flash_moe" / "flash_moe_layout.json"`.
- If a `flash_moe/` bundle is present but loading fails, **raise** — never fall back to the full `mlx_lm.load(lazy=False)` (would OOM on large models).
- Calibration produces identical K/V whether loaded full or via Flash-MoE (Flash-MoE replaces only the MoE FFN, not attention).
- Run `uv run ruff check` + `uv run ruff format` before any commit that touches Python.
- Do NOT modify `model_manager._load_flash_moe_model` in this change.

---

### Task 1: `load_flash_moe_model` shared helper

**Files:**
- Modify: `olmlx/engine/flash/flash_moe_model.py` (add free function after `FlashMoeModelWrapper`, near line 257)
- Test: `tests/test_flash_moe_model.py`

**Interfaces:**
- Produces: `load_flash_moe_model(load_path: str, flash_moe_dir: Path | str, *, cache_budget_experts: int, io_threads: int) -> tuple[Any, Any, Any]` returning `(wrapped_model, tokenizer, store)`. `wrapped_model` is a `FlashMoeModelWrapper`; `store` is a `FlashMoeWeightStore` the caller must close.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_flash_moe_model.py` (the `_make_synthetic_moe_weights` import already exists at top of file; `_MockModel` is defined in this file):

```python
def test_load_flash_moe_model_builds_wrapper_and_returns_store(tmp_path):
    from unittest.mock import MagicMock, patch

    from olmlx.engine.flash.moe_bundler import bundle_moe_experts
    from olmlx.engine.flash.flash_moe_model import (
        FlashMoeModelWrapper,
        _FlashMoEBase,
        load_flash_moe_model,
    )

    hidden, inter, experts = 64, 32, 8
    num_dense, num_moe, ntok = 1, 2, 2
    model_dir = _make_synthetic_moe_weights(
        hidden, inter, experts, num_moe, num_dense, tmp_path
    )
    flash_dir = tmp_path / "flash_moe"
    bundle_moe_experts(model_dir, flash_dir)

    synth = _MockModel(hidden, inter, experts, ntok, num_dense, num_moe)
    tok = MagicMock()

    with patch(
        "olmlx.engine.flash.prepare.load_model_with_strict_fallback",
        return_value=(synth, tok),
    ) as load_mock:
        model, tokenizer, store = load_flash_moe_model(
            str(model_dir), flash_dir, cache_budget_experts=4, io_threads=4
        )
    try:
        # lazy load is mandatory so experts are never materialized
        assert load_mock.call_args.kwargs["lazy"] is True
        assert isinstance(model, FlashMoeModelWrapper)
        assert tokenizer is tok
        for i in (1, 2):
            assert isinstance(model.layers[i].mlp, _FlashMoEBase)
        assert hasattr(store, "close")
    finally:
        store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_flash_moe_model.py::test_load_flash_moe_model_builds_wrapper_and_returns_store -v`
Expected: FAIL with `ImportError: cannot import name 'load_flash_moe_model'`.

- [ ] **Step 3: Write minimal implementation**

In `olmlx/engine/flash/flash_moe_model.py`, add after the `FlashMoeModelWrapper` class (after line 256, before `_find_moe_module`):

```python
def load_flash_moe_model(
    load_path: str,
    flash_moe_dir: "Path | str",
    *,
    cache_budget_experts: int,
    io_threads: int,
) -> tuple[Any, Any, Any]:
    """Load a model in Flash-MoE mode for offline use (e.g. KV-quant calibration).

    Mirrors ``model_manager._load_flash_moe_model``: lazy-load so routed expert
    weights are never materialized, wrap with ``FlashMoeModelWrapper`` (experts
    streamed from the SSD bundle), and eval only the non-expert params. Returns
    ``(wrapped_model, tokenizer, store)``; the caller owns the store and must
    close it. Raises if the bundle is present but cannot be loaded.
    """
    import json
    from pathlib import Path

    from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore
    from olmlx.engine.flash.prepare import load_model_with_strict_fallback

    flash_moe_dir = Path(flash_moe_dir)
    model, tokenizer = load_model_with_strict_fallback(load_path, lazy=True)
    moe_config = json.loads((flash_moe_dir / "flash_moe_config.json").read_text())

    store = FlashMoeWeightStore(
        flash_moe_dir,
        num_io_threads=io_threads,
        cache_budget_experts=cache_budget_experts,
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
        )
        mx.eval(wrapped.parameters())
    except Exception:
        store.close()
        raise
    return wrapped, tokenizer, store
```

Note: `mx`, `Any`, and `FlashMoeModelWrapper` are already in module scope; `Path` is only needed as a runtime value here so it is imported function-locally (the `"Path | str"` annotation is a string, evaluated lazily).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_flash_moe_model.py::test_load_flash_moe_model_builds_wrapper_and_returns_store -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check olmlx/engine/flash/flash_moe_model.py tests/test_flash_moe_model.py
uv run ruff format olmlx/engine/flash/flash_moe_model.py tests/test_flash_moe_model.py
git add olmlx/engine/flash/flash_moe_model.py tests/test_flash_moe_model.py
git commit -m "feat(flash-moe): add load_flash_moe_model offline helper"
```

---

### Task 2: `_load_calibration_model` Flash-MoE branch + store in return tuple

**Files:**
- Modify: `olmlx/engine/spectralquant_calibrate.py` (constants near the other `_SPECTRAL_DEFAULT_*`; `_load_calibration_model` at line 323)
- Test: `tests/test_spectralquant_calibrate_coverage.py`

**Interfaces:**
- Consumes: `load_flash_moe_model` (Task 1).
- Produces: `_load_calibration_model(model_path: str)` now returns a **7-tuple** `(model, tokenizer, inner, head_dim, n_kv_heads, num_layers, store)` where `store` is a `FlashMoeWeightStore` or `None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_spectralquant_calibrate_coverage.py` (module imported as `sc`; `_FakeBackbone` is defined in this file):

```python
def test_load_calibration_model_uses_flash_when_bundle_present(tmp_path):
    from unittest.mock import MagicMock, patch

    fdir = tmp_path / "flash_moe"
    fdir.mkdir()
    (fdir / "flash_moe_layout.json").write_text("{}")

    sentinel_model = MagicMock()
    sentinel_model._backbone = _FakeBackbone(2, 2, 8)
    sentinel_tok = MagicMock()
    sentinel_store = MagicMock()

    with patch(
        "olmlx.engine.flash.flash_moe_model.load_flash_moe_model",
        return_value=(sentinel_model, sentinel_tok, sentinel_store),
    ) as load_flash, patch(
        "olmlx.engine.flash.prepare._get_backbone",
        return_value=sentinel_model._backbone,
    ), patch(
        "olmlx.engine.turboquant_cache._detect_head_dim", return_value=8
    ):
        result = sc._load_calibration_model(str(tmp_path))

    load_flash.assert_called_once()
    assert result[0] is sentinel_model
    assert result[6] is sentinel_store


def test_load_calibration_model_full_load_when_no_bundle(tmp_path):
    from unittest.mock import MagicMock, patch

    model = MagicMock()
    backbone = _FakeBackbone(2, 2, 8)
    tok = MagicMock()

    with patch(
        "olmlx.engine.flash.prepare.load_model_with_strict_fallback",
        return_value=(model, tok),
    ) as full_load, patch(
        "olmlx.engine.flash.prepare._get_backbone", return_value=backbone
    ), patch(
        "olmlx.engine.turboquant_cache._detect_head_dim", return_value=8
    ):
        result = sc._load_calibration_model(str(tmp_path))

    full_load.assert_called_once_with(str(tmp_path), lazy=False)
    assert result[6] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_spectralquant_calibrate_coverage.py::test_load_calibration_model_uses_flash_when_bundle_present tests/test_spectralquant_calibrate_coverage.py::test_load_calibration_model_full_load_when_no_bundle -v`
Expected: FAIL — `test_..._uses_flash...` errors (no flash branch / 6-tuple has no index 6); `test_..._full_load...` fails on `result[6]` (IndexError).

- [ ] **Step 3: Write minimal implementation**

In `olmlx/engine/spectralquant_calibrate.py`, near the existing `_SPECTRAL_DEFAULT_*` constants (top of file), add:

```python
#: Conservative expert LRU budget during calibration. Calibration is
#: latency-tolerant and the budget only affects *which* experts are resident,
#: never the K/V output, so we minimize footprint to leave room for the
#: K/V-collection buffers. Bump if calibration is too slow.
_CALIBRATION_CACHE_BUDGET_EXPERTS = 8
_CALIBRATION_IO_THREADS = 32
```

Replace the body of `_load_calibration_model` (the `try/except ValueError` load block at line 336-344) so the load chooses Flash-MoE when a bundle exists, and extend the return. The function currently starts:

```python
    from olmlx.engine.flash.prepare import (
        _get_backbone,
        load_model_with_strict_fallback,
    )
    from olmlx.engine.turboquant_cache import _detect_head_dim

    try:
        model, tokenizer = load_model_with_strict_fallback(model_path, lazy=False)
    except ValueError:
        import mlx_vlm

        model, processor = mlx_vlm.load(model_path, lazy=False)
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
```

Change it to:

```python
    from pathlib import Path

    from olmlx.engine.flash.prepare import (
        _get_backbone,
        load_model_with_strict_fallback,
    )
    from olmlx.engine.turboquant_cache import _detect_head_dim

    flash_moe_dir = Path(model_path) / "flash_moe"
    store = None
    if (flash_moe_dir / "flash_moe_layout.json").exists():
        from olmlx.engine.flash.flash_moe_model import load_flash_moe_model

        # Bundle present: commit to the Flash-MoE path. Do NOT fall back to a
        # full load on failure — that would OOM on the large models this path
        # exists to support.
        model, tokenizer, store = load_flash_moe_model(
            model_path,
            flash_moe_dir,
            cache_budget_experts=_CALIBRATION_CACHE_BUDGET_EXPERTS,
            io_threads=_CALIBRATION_IO_THREADS,
        )
    else:
        try:
            model, tokenizer = load_model_with_strict_fallback(
                model_path, lazy=False
            )
        except ValueError:
            import mlx_vlm

            model, processor = mlx_vlm.load(model_path, lazy=False)
            tokenizer = (
                processor.tokenizer
                if hasattr(processor, "tokenizer")
                else processor
            )
```

Then change the final `return` of the function from:

```python
    return model, tokenizer, inner, head_dim, n_kv_heads, num_layers
```

to:

```python
    return model, tokenizer, inner, head_dim, n_kv_heads, num_layers, store
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_spectralquant_calibrate_coverage.py::test_load_calibration_model_uses_flash_when_bundle_present tests/test_spectralquant_calibrate_coverage.py::test_load_calibration_model_full_load_when_no_bundle -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check olmlx/engine/spectralquant_calibrate.py tests/test_spectralquant_calibrate_coverage.py
uv run ruff format olmlx/engine/spectralquant_calibrate.py tests/test_spectralquant_calibrate_coverage.py
git add olmlx/engine/spectralquant_calibrate.py tests/test_spectralquant_calibrate_coverage.py
git commit -m "feat(calibration): load via Flash-MoE when a bundle exists"
```

---

### Task 3: Unpack + close the store in `calibrate_model` and `calibrate_model_shard`

**Files:**
- Modify: `olmlx/engine/spectralquant_calibrate.py` (`calibrate_model`, the `_load_calibration_model` unpack at line 513 + collection block)
- Modify: `olmlx/engine/shardquant_calibrate.py` (`calibrate_model_shard`, unpack at line 195 + collection block)
- Test: `tests/test_spectralquant_calibrate_coverage.py`

**Interfaces:**
- Consumes: the 7-tuple from `_load_calibration_model` (Task 2).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_spectralquant_calibrate_coverage.py` (reuses `_FakeModel`, `_FakeBackbone`, `_patch_helpers`, `KVCache` already in the file):

```python
def test_calibrate_model_closes_store(tmp_path):
    from unittest.mock import MagicMock, patch

    head_dim, num_layers, n_kv_heads = 8, 2, 2
    backbone = _FakeBackbone(num_layers, n_kv_heads, head_dim)
    model = _FakeModel(backbone, head_dim)
    tokenizer = MagicMock()
    texts = ["one two three four five six seven eight nine ten"] * 2
    spy_store = MagicMock()

    def cache_factory(_owner):
        return [KVCache() for _ in range(num_layers)]

    patches = _patch_helpers(model, tokenizer, head_dim, texts, cache_factory)
    # Replace the loader so it returns our spy store as the 7th element.
    patches.append(
        patch.object(
            sc,
            "_load_calibration_model",
            return_value=(
                model, tokenizer, backbone, head_dim, n_kv_heads, num_layers, spy_store,
            ),
        )
    )
    for p in patches:
        p.start()
    try:
        sc.calibrate_model(
            "fake/model", output_dir=tmp_path / "spectral", num_samples=2
        )
    finally:
        for p in patches:
            p.stop()

    spy_store.close.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_spectralquant_calibrate_coverage.py::test_calibrate_model_closes_store -v`
Expected: FAIL — either a ValueError unpacking the 7-tuple into 6 names, or `spy_store.close` never called.

- [ ] **Step 3: Write minimal implementation**

In `olmlx/engine/spectralquant_calibrate.py`, change the unpack in `calibrate_model` (line ~513) from:

```python
    model, tokenizer, inner, head_dim, n_kv_heads, num_layers = _load_calibration_model(
        model_path
    )
```

to:

```python
    (
        model,
        tokenizer,
        inner,
        head_dim,
        n_kv_heads,
        num_layers,
        store,
    ) = _load_calibration_model(model_path)
```

Then wrap the KV-collection call in try/finally so the store is always closed. Change:

```python
    kv_collectors = collect_kv_vectors(
        model,
        tokenizer,
        inner,
        num_layers=num_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        texts=texts,
        max_tokens_per_head=max_tokens_per_head,
        progress_callback=progress_callback,
    )
```

to:

```python
    try:
        kv_collectors = collect_kv_vectors(
            model,
            tokenizer,
            inner,
            num_layers=num_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            texts=texts,
            max_tokens_per_head=max_tokens_per_head,
            progress_callback=progress_callback,
        )
    finally:
        if store is not None:
            store.close()
```

Apply the identical two edits to `olmlx/engine/shardquant_calibrate.py` in `calibrate_model_shard`: unpack the 7-tuple at line ~195 (add `store` as the 7th name) and wrap its `collect_kv_vectors(...)` call in the same `try/finally: if store is not None: store.close()`.

- [ ] **Step 4: Run test to verify it passes (and no regressions)**

Run: `uv run pytest tests/test_spectralquant_calibrate_coverage.py tests/test_shardquant_calibrate.py -v`
Expected: PASS (new `test_calibrate_model_closes_store` passes; all pre-existing calibrate-model tests still pass because `store` is `None` on the mocked full-load path and `if store is not None` skips the close).

- [ ] **Step 5: Commit**

```bash
uv run ruff check olmlx/engine/spectralquant_calibrate.py olmlx/engine/shardquant_calibrate.py tests/test_spectralquant_calibrate_coverage.py
uv run ruff format olmlx/engine/spectralquant_calibrate.py olmlx/engine/shardquant_calibrate.py tests/test_spectralquant_calibrate_coverage.py
git add olmlx/engine/spectralquant_calibrate.py olmlx/engine/shardquant_calibrate.py tests/test_spectralquant_calibrate_coverage.py
git commit -m "feat(calibration): close Flash-MoE store after KV collection"
```

---

### Task 4: Regression test — wrapper forwards `cache` (K/V collection mechanism)

**Files:**
- Test: `tests/test_flash_moe_model.py`

**Interfaces:**
- Consumes: `load_flash_moe_model` (Task 1), `FlashMoeModelWrapper`.

This proves the claim that attention K/V is unaffected by expert offloading: calibration collects K/V by passing a `cache=` into the model forward, and the wrapper must forward it to the inner model untouched.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_flash_moe_model.py`:

```python
def test_flash_moe_wrapper_forwards_cache_to_inner_model(tmp_path):
    from unittest.mock import MagicMock, patch

    import mlx.core as mx

    from olmlx.engine.flash.moe_bundler import bundle_moe_experts
    from olmlx.engine.flash.flash_moe_model import load_flash_moe_model

    hidden, inter, experts = 64, 32, 8
    num_dense, num_moe, ntok = 1, 2, 2
    model_dir = _make_synthetic_moe_weights(
        hidden, inter, experts, num_moe, num_dense, tmp_path
    )
    flash_dir = tmp_path / "flash_moe"
    bundle_moe_experts(model_dir, flash_dir)

    synth = _MockModel(hidden, inter, experts, ntok, num_dense, num_moe)
    with patch(
        "olmlx.engine.flash.prepare.load_model_with_strict_fallback",
        return_value=(synth, MagicMock()),
    ):
        wrapped, _, store = load_flash_moe_model(
            str(model_dir), flash_dir, cache_budget_experts=4, io_threads=4
        )
    try:
        spy = MagicMock(return_value=mx.zeros((1, ntok, hidden)))
        wrapped._model = spy  # replace inner so we observe the forward call
        sentinel_cache = [object(), object()]
        wrapped(mx.zeros((1, ntok, hidden)), cache=sentinel_cache)
        spy.assert_called_once()
        assert spy.call_args.kwargs["cache"] is sentinel_cache
    finally:
        store.close()
```

- [ ] **Step 2: Run test to verify it fails (or passes immediately)**

Run: `uv run pytest tests/test_flash_moe_model.py::test_flash_moe_wrapper_forwards_cache_to_inner_model -v`
Expected: PASS (the forwarding already exists in `FlashMoeModelWrapper.__call__`). This is a regression guard, not a behavior change — if it fails, the wrapper forward was altered and calibration would break.

- [ ] **Step 3: (no implementation needed)**

The behavior under test already exists. If Step 2 fails, stop and investigate `FlashMoeModelWrapper.__call__`.

- [ ] **Step 4: Run the full affected suites**

Run: `uv run pytest tests/test_flash_moe_model.py tests/test_spectralquant_calibrate_coverage.py tests/test_shardquant_calibrate.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check tests/test_flash_moe_model.py
uv run ruff format tests/test_flash_moe_model.py
git add tests/test_flash_moe_model.py
git commit -m "test(flash-moe): guard wrapper cache forwarding for calibration"
```

---

### Task 5: Live acceptance run + switch model to `spectral:2`

**Files:**
- Modify: `~/.olmlx/models.json` (the `mlx-community/Qwen3-235B-A22B-4bit:latest` entry)

This task is **manual / operational** — it exercises the real model, which CI cannot. Run it on the 64 GB machine with the prepared `flash_moe/` bundle present.

- [ ] **Step 1: Run spectral calibration through the new Flash-MoE path**

```bash
cd /Users/daniel/devel/olmlx
uv run olmlx spectral prepare mlx-community/Qwen3-235B-A22B-4bit --avg-bits 2
```

Watch in another shell while it runs:
```bash
df -h /Volumes/External | tail -1      # volume must stay mounted
memory_pressure | grep "free percentage"
```
Expected: completes without a Metal OOM abort; prints `Spectral calibration complete!` and an output dir under the model directory. If it OOMs, lower `_CALIBRATION_CACHE_BUDGET_EXPERTS` (Task 2) toward 4 and re-run.

- [ ] **Step 2: Switch the model to spectral:2**

Edit `~/.olmlx/models.json`, in the `mlx-community/Qwen3-235B-A22B-4bit:latest` entry change:
```json
"kv_cache_quant": "turboquant:4",
```
to:
```json
"kv_cache_quant": "spectral:2",
```
Keep `"flash_moe_cache_budget_experts": 24`. Validate:
```bash
python3 -c "import json; json.load(open('$HOME/.olmlx/models.json')); print('valid')"
```

- [ ] **Step 3: Verify end-to-end**

Restart the server (`uv run olmlx` from the serving checkout) and send a long-context request; confirm it loads `spectral:2` (server log: `Created Spectral KV cache` or equivalent) and answers without OOM. No commit — `~/.olmlx/models.json` is machine-local, not in the repo.

---

## Self-Review

**Spec coverage:**
- Component 1 (helper) → Task 1. ✅
- Component 2 (loader branch) → Task 2. ✅
- Memory-safety constants (budget 8 / io 32) → Task 2 Global Constraints + constants. ✅
- Store lifecycle (return + finally close, both pipelines) → Task 2 (return) + Task 3 (close in both `calibrate_model` and `calibrate_model_shard`). ✅
- Error handling (raise, don't fall back) → Task 2 implementation + Global Constraints. ✅
- Tests 1–4 from spec → Task 1 (helper build), Task 2 (loader selection ×2), Task 3 (store close), Task 4 (cache forwarding). ✅
- Acceptance run + models.json flip → Task 5. ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✅

**Type consistency:** `load_flash_moe_model(load_path, flash_moe_dir, *, cache_budget_experts, io_threads) -> (model, tokenizer, store)` is defined in Task 1 and consumed with the same keywords in Task 2. `_load_calibration_model` returns the 7-tuple `(model, tokenizer, inner, head_dim, n_kv_heads, num_layers, store)` in Task 2 and is unpacked with 7 names in Task 3 (both pipelines). Constant names `_CALIBRATION_CACHE_BUDGET_EXPERTS` / `_CALIBRATION_IO_THREADS` match between definition and use. ✅
