# Lift VLM-path Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable prompt caching and grammar-constrained decoding on the VLM (image) inference path, give non-streaming VLM correct token counts, and document the speculative-on-VLM decision — closing #429.

**Architecture:** mlx_vlm 0.4.4 natively supports `prompt_cache`, `logits_processors`, and a `PromptCacheState` cross-turn reuse mechanism. We lift olmlx's stale gates, add a tiny per-`cache_id` LRU of `PromptCacheState` objects (`VlmPromptCacheStore`, modeled on the speculative `_SpecCacheStore`), delete the image-dropping `input_ids` hack, and switch the non-streaming VLM path from `mlx_vlm.generate` to draining `mlx_vlm.stream_generate`.

**Tech Stack:** Python 3.11, MLX, mlx_vlm 0.4.4, FastAPI, pydantic-settings, pytest.

**Spec:** `docs/superpowers/specs/2026-06-07-vlm-gating-lift-design.md`

---

## File Structure

- `olmlx/config.py` — new `vlm_prompt_cache_slots` setting.
- `olmlx/engine/prompt_cache/vlm_state.py` — **new**: `VlmPromptCacheStore` (LRU of `cache_id → PromptCacheState`).
- `olmlx/engine/model_manager.py` — new `LoadedModel.vlm_prompt_cache_store` field; create it for VLMs; clear on unload.
- `olmlx/engine/inference.py` — lift grammar gate; extend `_resolve_model_vocab_size` for VLM; `_setup_vlm_prompt_cache` helper; remove `input_ids` hack; VLM cache branch in `_stream_completion`/`_full_completion`; `use_prompt_cache` gate; non-streaming VLM drain.
- `tests/engine/test_vlm_prompt_cache_store.py` — **new**: unit tests for the store (mocked, no model).
- `tests/engine/test_grammar_vlm_vocab.py` — **new**: unit test for VLM vocab resolution (mocked).
- `tests/live/test_vlm_cache_grammar.py` — **new**: real_model live tests (grammar, cache reuse, token counts, image-drop regression).
- `CLAUDE.md` — rewrite the VLM-gating design note.

---

## Task 1: Config knob `vlm_prompt_cache_slots`

**Files:**
- Modify: `olmlx/config.py` (after the `prompt_cache: bool = True` field, ~line 114)
- Test: `tests/engine/test_vlm_prompt_cache_store.py` (created in Task 2 covers the default indirectly; no separate config test needed)

- [ ] **Step 1: Add the setting**

In `olmlx/config.py`, immediately after the `prompt_cache: bool = True` line (~line 114), add:

```python
    # Number of per-cache_id VLM KV-cache slots retained for cross-turn image-prefix
    # reuse (mlx_vlm PromptCacheState). 0 disables VLM prompt caching entirely.
    # In-memory only (no disk spill / radix / KV-quant); 2 slots bound the memory.
    vlm_prompt_cache_slots: int = 2
```

- [ ] **Step 2: Verify it loads**

Run: `uv run python -c "from olmlx.config import settings; print(settings.vlm_prompt_cache_slots)"`
Expected: `2`

- [ ] **Step 3: Verify env override**

Run: `OLMLX_VLM_PROMPT_CACHE_SLOTS=0 uv run python -c "from olmlx.config import settings; print(settings.vlm_prompt_cache_slots)"`
Expected: `0`

- [ ] **Step 4: Commit**

```bash
git add olmlx/config.py
git commit -m "feat(config): add OLMLX_VLM_PROMPT_CACHE_SLOTS (#429)"
```

---

## Task 2: `VlmPromptCacheStore`

A tiny LRU keyed by `cache_id`, holding `mlx_vlm.generate.PromptCacheState` objects. Modeled on `_SpecCacheStore` (capacity, `enabled()`, kill switch).

**Files:**
- Create: `olmlx/engine/prompt_cache/vlm_state.py`
- Test: `tests/engine/test_vlm_prompt_cache_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/engine/test_vlm_prompt_cache_store.py`:

```python
"""Unit tests for VlmPromptCacheStore (#429). No real model required."""

from olmlx.engine.prompt_cache.vlm_state import VlmPromptCacheStore


class _FakeState:
    """Stand-in for mlx_vlm.generate.PromptCacheState (only identity matters)."""

    def __init__(self, tag):
        self.tag = tag


def test_disabled_when_capacity_zero():
    store = VlmPromptCacheStore(capacity=0)
    assert store.enabled() is False
    store.insert("a", _FakeState("a"))
    assert store.get("a") is None  # insert is a no-op when disabled


def test_insert_and_get():
    store = VlmPromptCacheStore(capacity=2)
    s = _FakeState("a")
    store.insert("a", s)
    assert store.get("a") is s
    assert store.get("missing") is None


def test_lru_eviction_past_capacity():
    store = VlmPromptCacheStore(capacity=2)
    store.insert("a", _FakeState("a"))
    store.insert("b", _FakeState("b"))
    store.insert("c", _FakeState("c"))  # evicts "a" (least-recently-used)
    assert store.get("a") is None
    assert store.get("b").tag == "b"
    assert store.get("c").tag == "c"


def test_get_promotes_to_mru():
    store = VlmPromptCacheStore(capacity=2)
    store.insert("a", _FakeState("a"))
    store.insert("b", _FakeState("b"))
    assert store.get("a").tag == "a"  # promote "a" -> MRU
    store.insert("c", _FakeState("c"))  # evicts "b" (now LRU), not "a"
    assert store.get("b") is None
    assert store.get("a").tag == "a"
    assert store.get("c").tag == "c"


def test_insert_same_id_replaces_without_growing():
    store = VlmPromptCacheStore(capacity=2)
    store.insert("a", _FakeState("first"))
    store.insert("a", _FakeState("second"))
    assert store.get("a").tag == "second"
    # Only one slot consumed: inserting two more should evict by capacity=2,
    # so "a" survives alongside the most recent.
    store.insert("b", _FakeState("b"))
    assert store.get("a").tag == "second"
    assert store.get("b").tag == "b"


def test_clear():
    store = VlmPromptCacheStore(capacity=2)
    store.insert("a", _FakeState("a"))
    store.clear()
    assert store.get("a") is None


def test_metrics_counters():
    store = VlmPromptCacheStore(capacity=2)
    assert store.metrics() == {
        "vlm_cache_hits": 0,
        "vlm_cache_misses": 0,
        "vlm_cache_tokens_reused": 0,
    }
    store.note_miss()
    store.note_hit(reused_tokens=10)
    store.note_hit(reused_tokens=5)
    assert store.metrics() == {
        "vlm_cache_hits": 2,
        "vlm_cache_misses": 1,
        "vlm_cache_tokens_reused": 15,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_vlm_prompt_cache_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'olmlx.engine.prompt_cache.vlm_state'`

- [ ] **Step 3: Write the implementation**

Create `olmlx/engine/prompt_cache/vlm_state.py`:

```python
"""Per-cache_id LRU of mlx_vlm PromptCacheState objects for VLM prompt caching.

mlx_vlm's ``stream_generate`` owns the hard parts of cross-turn KV reuse:
prefix matching against a stored token sequence, trimming the cache to the
common prefix, detecting whether the new tokens still contain image
placeholders (so vision features are only recomputed when needed), and
updating the state in place after generation.  This store just bounds how
many ``PromptCacheState`` lineages are retained, keyed by the request's
``cache_id``.

Mirrors the speculative ``_SpecCacheStore`` ergonomics: all inference is
serialized under the inference lock, so no internal locking is needed;
``capacity == 0`` is the kill switch (``OLMLX_VLM_PROMPT_CACHE_SLOTS=0``).

v1 limits (documented in CLAUDE.md): in-memory only — no radix-takeover, no
disk spill, no KV-quant.  At the small default slot count, keying on
``cache_id`` (rather than a longest-prefix scan) is sufficient.
"""

from __future__ import annotations

from typing import Any


class VlmPromptCacheStore:
    def __init__(self, capacity: int) -> None:
        self._capacity = max(int(capacity), 0)
        # Insertion-ordered dict as an LRU: first key is least-recently-used.
        self._entries: dict[str, Any] = {}
        # Cumulative reuse counters surfaced on /api/ps (acceptance criterion 1).
        self._hits = 0
        self._misses = 0
        self._tokens_reused = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def enabled(self) -> bool:
        return self._capacity > 0

    def clear(self) -> None:
        """Drop retained states. Counters are cumulative and NOT reset here, so
        /api/ps reuse totals survive memory-pressure flushes."""
        self._entries.clear()

    def note_hit(self, reused_tokens: int) -> None:
        self._hits += 1
        self._tokens_reused += max(int(reused_tokens), 0)

    def note_miss(self) -> None:
        self._misses += 1

    def metrics(self) -> dict[str, int]:
        return {
            "vlm_cache_hits": self._hits,
            "vlm_cache_misses": self._misses,
            "vlm_cache_tokens_reused": self._tokens_reused,
        }

    def get(self, cache_id: str) -> Any | None:
        """Return the PromptCacheState for ``cache_id`` and promote it to
        most-recently-used, or ``None`` on miss / when disabled."""
        if not self.enabled():
            return None
        state = self._entries.pop(cache_id, None)
        if state is None:
            return None
        self._entries[cache_id] = state  # re-insert at MRU end
        return state

    def insert(self, cache_id: str, state: Any) -> None:
        """Store ``state`` under ``cache_id`` as most-recently-used, evicting
        the least-recently-used entries past ``capacity``. No-op when disabled."""
        if not self.enabled():
            return
        self._entries.pop(cache_id, None)  # refresh position if already present
        self._entries[cache_id] = state
        while len(self._entries) > self._capacity:
            # Pop the oldest (least-recently-used) entry.
            oldest = next(iter(self._entries))
            del self._entries[oldest]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_vlm_prompt_cache_store.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/prompt_cache/vlm_state.py tests/engine/test_vlm_prompt_cache_store.py
git commit -m "feat(prompt_cache): VlmPromptCacheStore LRU for VLM KV reuse (#429)"
```

---

## Task 3: Wire the store onto `LoadedModel`

**Files:**
- Modify: `olmlx/engine/model_manager.py` — `LoadedModel` dataclass (~line 490) and `__post_init__` (~line 533); unload clear (~line 773).

- [ ] **Step 1: Add the field**

In `olmlx/engine/model_manager.py`, in the `LoadedModel` dataclass, immediately after the `prompt_cache_store` field (~line 490), add:

```python
    # Per-cache_id LRU of mlx_vlm PromptCacheState objects for cross-turn
    # image-prefix KV reuse. Only populated for VLMs (None otherwise). #429.
    vlm_prompt_cache_store: Any = None
```

(`Any` is already imported in this module — it is used throughout, e.g. the `model: Any` field.)

- [ ] **Step 2: Create it for VLMs in `__post_init__`**

In `__post_init__` (~line 533), after the existing `if self.prompt_cache_store is None:` block that ends at the `PromptCacheStore(...)` constructor (~line 552), add:

```python
        if self.is_vlm and self.vlm_prompt_cache_store is None:
            from olmlx.engine.prompt_cache.vlm_state import VlmPromptCacheStore

            self.vlm_prompt_cache_store = VlmPromptCacheStore(
                capacity=settings.vlm_prompt_cache_slots
            )
```

- [ ] **Step 3: Clear it on unload**

Find the unload path that clears `prompt_cache_store` (~line 773):

```python
        if lm.prompt_cache_store is not None:
            ...
                lm.prompt_cache_store.clear()
            ...
                lm.prompt_cache_store = None
```

Immediately after that block, add:

```python
        if getattr(lm, "vlm_prompt_cache_store", None) is not None:
            lm.vlm_prompt_cache_store.clear()
            lm.vlm_prompt_cache_store = None
```

- [ ] **Step 4: Verify the field is wired (mocked import smoke test)**

Run:
```bash
uv run python -c "
from olmlx.engine.model_manager import LoadedModel
lm = LoadedModel(name='x', hf_path='x', model=object(), tokenizer=object(), is_vlm=True)
print('store:', type(lm.vlm_prompt_cache_store).__name__, 'cap:', lm.vlm_prompt_cache_store.capacity)
lm2 = LoadedModel(name='y', hf_path='y', model=object(), tokenizer=object(), is_vlm=False)
print('non-vlm store:', lm2.vlm_prompt_cache_store)
"
```
Expected:
```
store: VlmPromptCacheStore cap: 2
non-vlm store: None
```

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/model_manager.py
git commit -m "feat(model_manager): own VlmPromptCacheStore per VLM (#429)"
```

---

## Task 4: Grammar on VLM — lift gate + VLM vocab resolution

**Files:**
- Modify: `olmlx/engine/inference.py` — `_resolve_model_vocab_size` (~line 76–114) and `_install_grammar_processor` (~line 138).
- Test: `tests/engine/test_grammar_vlm_vocab.py`

- [ ] **Step 1: Write the failing test**

Create `tests/engine/test_grammar_vlm_vocab.py`:

```python
"""VLM vocab resolution for grammar bitmask sizing (#429)."""

import types

from olmlx.engine.inference import _resolve_model_vocab_size


class _Weight:
    def __init__(self, rows):
        self.shape = (rows, 4096)


class _Head:
    def __init__(self, rows):
        self.weight = _Weight(rows)


def _fake_lm(model):
    # Minimal stand-in: _resolve_model_vocab_size only reads lm.model.
    return types.SimpleNamespace(model=model)


def test_resolves_vocab_under_language_model_for_vlm():
    # VLM layout: lm_head lives under model.language_model, not model/model.model.
    language_model = types.SimpleNamespace(lm_head=_Head(151936))
    vlm = types.SimpleNamespace(language_model=language_model)
    assert _resolve_model_vocab_size(_fake_lm(vlm)) == 151936


def test_text_model_vocab_still_resolves():
    text = types.SimpleNamespace(lm_head=_Head(32000), model=None)
    assert _resolve_model_vocab_size(_fake_lm(text)) == 32000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_grammar_vlm_vocab.py -q`
Expected: FAIL on `test_resolves_vocab_under_language_model_for_vlm` (returns `None`, not `151936`).

- [ ] **Step 3: Extend `_resolve_model_vocab_size`**

In `olmlx/engine/inference.py`, the current loop owner tuple is:

```python
    for attr in ("lm_head", "embed_tokens"):
        for owner in (model, getattr(model, "model", None)):
```

Replace the `for owner in (...)` line with one that also descends into the VLM language tower:

```python
    for attr in ("lm_head", "embed_tokens"):
        language_model = getattr(model, "language_model", None)
        for owner in (
            model,
            getattr(model, "model", None),
            language_model,
            getattr(language_model, "model", None),
        ):
```

- [ ] **Step 4: Lift the VLM gate in `_install_grammar_processor`**

Delete this block (~line 138):

```python
    if lm.is_vlm:
        logger.warning(
            "Grammar-constrained decoding requested but model is a VLM "
            "(mlx-vlm does not accept logits_processors); ignoring "
            "constraint for this request"
        )
        return False
```

Update the function docstring (~lines 126–135): remove the sentence claiming VLM models are unsupported, replacing it with:

```python
    """Build and install a grammar logits processor on *gen_kwargs*.

    Returns ``True`` when grammar is active for the request. Works for both
    text and VLM models — mlx_vlm's ``generate_step`` accepts
    ``logits_processors`` and olmlx forwards ``gen_kwargs`` to it (#429).
    Distributed mode is still rejected: workers don't receive the processor
    over the sideband and would diverge from rank-0. Tool-use requests are
    rejected: the JSON grammar masks the format-specific tool-call tokens
    (``<tool_call>``, ``[TOOL_CALLS]``, ``<function=...>``, …) so the model
    could never emit a tool call. Constraining tool *arguments* is the
    deferred Anthropic case (issue #361).
    """
```

- [ ] **Step 5: Run the vocab test to verify it passes**

Run: `uv run pytest tests/engine/test_grammar_vlm_vocab.py -q`
Expected: PASS (2 passed)

- [ ] **Step 6: Run the existing grammar suite to check for regressions**

Run: `uv run pytest -q -k grammar -m "not real_model"`
Expected: PASS (no new failures)

- [ ] **Step 7: Commit**

```bash
git add olmlx/engine/inference.py tests/engine/test_grammar_vlm_vocab.py
git commit -m "feat(grammar): support VLM via logits_processors + language_model vocab (#429)"
```

---

## Task 5: `_setup_vlm_prompt_cache` helper + remove the `input_ids` hack

**Files:**
- Modify: `olmlx/engine/inference.py` — add helper near `_setup_prompt_cache` (~line 2862); remove VLM `input_ids` branches inside `_setup_prompt_cache` (~lines 3065–3066, 3091–3092).

- [ ] **Step 1: Add the `_setup_vlm_prompt_cache` helper**

In `olmlx/engine/inference.py`, immediately **before** `async def _setup_prompt_cache(` (~line 2862), add:

```python
def _setup_vlm_prompt_cache(
    lm: LoadedModel,
    prompt_tokens: list[int] | None,
    gen_kwargs: dict,
    *,
    cache_id: str,
) -> tuple[int, int]:
    """Attach an mlx_vlm ``PromptCacheState`` to ``gen_kwargs`` for VLM KV reuse.

    Returns ``(cache_read_tokens, cache_creation_tokens)`` for metrics. mlx_vlm
    owns prefix matching / cache trimming / image-in-prefix detection and
    updates the state in place after generation; here we only fetch-or-create
    the per-cache_id state and report an *estimate* of the reused prefix.

    The estimate uses ``lm.text_tokenizer`` tokenization, which can differ
    slightly from mlx_vlm's internal ``input_ids`` (image placeholder
    expansion), so the counts are approximate — the actual reuse is whatever
    mlx_vlm's own ``find_prefix_length`` decides at generate time.
    """
    from mlx_vlm.generate import PromptCacheState

    store = getattr(lm, "vlm_prompt_cache_store", None)
    if store is None or not store.enabled() or prompt_tokens is None:
        return 0, len(prompt_tokens) if prompt_tokens is not None else 0

    # Memory pressure: drop the VLM store and fall back to a fresh state.
    if memory_utils.is_memory_pressure_high(settings.memory_limit_fraction):
        logger.warning("Memory pressure high, clearing VLM prompt cache")
        store.clear()
        mx.clear_cache()
        _safe_sync()

    state = store.get(cache_id)
    if state is not None:
        read = state.find_prefix_length(prompt_tokens)
        # A full-prefix re-request reuses everything but the seed; clamp so we
        # never claim more reuse than there are new tokens to process.
        read = min(read, max(len(prompt_tokens) - 1, 0))
        store.note_hit(reused_tokens=read)
    else:
        state = PromptCacheState()
        store.insert(cache_id, state)
        read = 0
        store.note_miss()

    gen_kwargs["prompt_cache_state"] = state
    creation = len(prompt_tokens) - read
    logger.info(
        "VLM prompt cache: ~%d prefix tokens reusable, ~%d new (cache_id=%s)",
        read,
        creation,
        cache_id,
    )
    return read, creation
```

(`memory_utils`, `mx`, `_safe_sync`, `settings`, `logger` are all already imported/defined in this module.)

- [ ] **Step 2: Remove the image-dropping `input_ids` hack (cache-hit branch)**

In `_setup_prompt_cache`, find the cache-hit assignment (~lines 3064–3068):

```python
            gen_kwargs["prompt_cache"] = working_cache
            if lm.is_vlm:
                gen_kwargs["input_ids"] = mx.array([suffix_tokens])
            else:
                result.prompt = suffix_tokens
```

Replace with (VLM no longer reaches this function — see Task 6 gating — so the branch is dead and the hack must go):

```python
            gen_kwargs["prompt_cache"] = working_cache
            result.prompt = suffix_tokens
```

- [ ] **Step 3: Remove the image-dropping `input_ids` hack (fresh branch)**

Find the fresh-cache assignment (~lines 3091–3094):

```python
        if lm.is_vlm:
            gen_kwargs["input_ids"] = mx.array([prompt_tokens])
        else:
            result.prompt = prompt_tokens
```

Replace with:

```python
        result.prompt = prompt_tokens
```

- [ ] **Step 4: Run the existing prompt-cache suite for regressions**

Run: `uv run pytest -q -k "prompt_cache or cache" -m "not real_model"`
Expected: PASS (no new failures; the text path is unchanged).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/inference.py
git commit -m "feat(inference): _setup_vlm_prompt_cache; drop image-dropping input_ids hack (#429)"
```

---

## Task 6: Wire the VLM cache branch + open the gate

**Files:**
- Modify: `olmlx/engine/inference.py` — `use_prompt_cache` gate (~line 4506); VLM branch in `_stream_completion` cache setup (~line 3489); VLM branch in `_full_completion` cache setup (~line 3952).

- [ ] **Step 1: Open the `use_prompt_cache` gate for VLM**

In `generate_chat`, change the gate (~lines 4506–4512) from:

```python
        use_prompt_cache = (
            effective_prompt_cache
            and make_prompt_cache is not None
            and not lm.is_distributed
            and not lm.is_speculative
            and (stream or not lm.is_vlm)
        )
```

to:

```python
        # VLM now caches on both stream and non-stream via the VLM store path
        # (mlx_vlm PromptCacheState); its KV reuse is independent of the text
        # path's checkpoint/radix machinery. A VLM with its store disabled
        # (vlm_prompt_cache_slots=0) is excluded so prompt_tokens stays None.
        vlm_cache_ok = (
            lm.is_vlm
            and getattr(lm, "vlm_prompt_cache_store", None) is not None
            and lm.vlm_prompt_cache_store.enabled()
        )
        use_prompt_cache = (
            effective_prompt_cache
            and make_prompt_cache is not None
            and not lm.is_distributed
            and not lm.is_speculative
            and (not lm.is_vlm or vlm_cache_ok)
        )
```

- [ ] **Step 2: Branch cache setup in `_stream_completion`**

`_stream_completion` funnels both branches through a `_CacheSetupResult` and then
uniformly reads `cs.prompt` / `cs.cache_read_tokens` / `cs.cache_creation_tokens` /
`cs.full_prompt_tokens` / `cs.cache_setup_done`. So the VLM branch just builds a
`_CacheSetupResult`. Find (~lines 3488–3501):

```python
        # Cache setup — must happen after lock to prevent concurrent cache corruption
        if use_prompt_cache:
            cs = await _setup_prompt_cache(
                lm,
                prompt,
                gen_kwargs,
                prompt_tokens=prompt_tokens,
                cache_id=cache_id,
                messages=messages,
                tokenizer=tokenizer,
                template_kwargs=template_kwargs,
            )
        else:
            cs = _CacheSetupResult(prompt=prompt)
```

Replace with:

```python
        # Cache setup — must happen after lock to prevent concurrent cache corruption
        if use_prompt_cache and lm.is_vlm:
            # VLM: attach an mlx_vlm PromptCacheState. ``prompt`` stays the full
            # str — mlx_vlm tokenizes it and reuses the KV prefix internally.
            read, creation = _setup_vlm_prompt_cache(
                lm, prompt_tokens, gen_kwargs, cache_id=cache_id
            )
            cs = _CacheSetupResult(
                prompt=prompt,
                cache_read_tokens=read,
                cache_creation_tokens=creation,
                full_prompt_tokens=prompt_tokens,
                cache_setup_done=True,
            )
        elif use_prompt_cache:
            cs = await _setup_prompt_cache(
                lm,
                prompt,
                gen_kwargs,
                prompt_tokens=prompt_tokens,
                cache_id=cache_id,
                messages=messages,
                tokenizer=tokenizer,
                template_kwargs=template_kwargs,
            )
        else:
            cs = _CacheSetupResult(prompt=prompt)
```

The unconditional `_kv_cache_preflight_check` that follows (~line 3508) is already
VLM-safe (its `not lm.is_vlm` guards leave a str prompt unchanged, and it only pops
`prompt_cache`/`input_ids`, never `prompt_cache_state`) — leave it as-is.

- [ ] **Step 3: Branch cache setup in `_full_completion`**

In `_full_completion`, find (~lines 3952–3967):

```python
                if use_prompt_cache:
                    cs = await _setup_prompt_cache(
                        lm,
                        prompt,
                        gen_kwargs,
                        prompt_tokens=prompt_tokens,
                        cache_id=cache_id,
                        messages=messages,
                        tokenizer=tokenizer,
                        template_kwargs=template_kwargs,
                    )
                    prompt = cs.prompt
                    cache_read_tokens = cs.cache_read_tokens
                    cache_creation_tokens = cs.cache_creation_tokens
                    full_prompt_tokens = cs.full_prompt_tokens
                    cache_setup_done = cs.cache_setup_done
```

Replace with:

```python
                if use_prompt_cache and lm.is_vlm:
                    # VLM: attach an mlx_vlm PromptCacheState. Leave ``prompt``
                    # as the full str — mlx_vlm tokenizes it and reuses the KV
                    # prefix internally (no suffix-only trimming on this path).
                    cache_read_tokens, cache_creation_tokens = (
                        _setup_vlm_prompt_cache(
                            lm, prompt_tokens, gen_kwargs, cache_id=cache_id
                        )
                    )
                    full_prompt_tokens = prompt_tokens
                    cache_setup_done = True
                elif use_prompt_cache:
                    cs = await _setup_prompt_cache(
                        lm,
                        prompt,
                        gen_kwargs,
                        prompt_tokens=prompt_tokens,
                        cache_id=cache_id,
                        messages=messages,
                        tokenizer=tokenizer,
                        template_kwargs=template_kwargs,
                    )
                    prompt = cs.prompt
                    cache_read_tokens = cs.cache_read_tokens
                    cache_creation_tokens = cs.cache_creation_tokens
                    full_prompt_tokens = cs.full_prompt_tokens
                    cache_setup_done = cs.cache_setup_done
```

- [ ] **Step 4: Leave the `_full_completion` preflight as-is**

No change. The `_kv_cache_preflight_check` call inside `if use_prompt_cache:`
(~line 3969) is VLM-safe: for a VLM the `not lm.is_vlm` guards leave `pf.prompt`
equal to the input str, and it only pops `prompt_cache`/`input_ids` from
`gen_kwargs` (never the VLM `prompt_cache_state`). Running it for VLM additionally
bounds VLM KV memory via `cache_creation_tokens`, which is desirable. Confirm by
reading the call site — it stays unchanged.

- [ ] **Step 5: Surface VLM reuse on `/api/ps`**

In `olmlx/routers/status.py`, find where `cache_metrics` is populated (~lines 67–70):

```python
        cache_metrics: dict[str, int] = {}
        cache_store = getattr(lm, "prompt_cache_store", None)
        if cache_store is not None and hasattr(cache_store, "metrics"):
            cache_metrics = cache_store.metrics.to_dict()
```

Immediately after that block, add (the text store exposes `metrics` as an object
with `.to_dict()`; the VLM store exposes `metrics()` returning a dict — merge it):

```python
        vlm_store = getattr(lm, "vlm_prompt_cache_store", None)
        if vlm_store is not None:
            cache_metrics = {**cache_metrics, **vlm_store.metrics()}
```

- [ ] **Step 6: Run the cache + inference suites for regressions**

Run: `uv run pytest -q -k "cache or inference or completion or status" -m "not real_model"`
Expected: PASS (text path unchanged; VLM branches only execute under real_model).

- [ ] **Step 7: Commit**

```bash
git add olmlx/engine/inference.py olmlx/routers/status.py
git commit -m "feat(inference): route VLM prompt caching through PromptCacheState (#429)"
```

---

## Task 7: Non-streaming VLM → drain `stream_generate`

Switching from `mlx_vlm.generate` (returns a bare `str`, discards counts) to draining `mlx_vlm.stream_generate` threads `prompt_cache_state` + `logits_processors` and yields a `GenerationResult` with token counts. The downstream unpacking at `_full_completion_inner` already handles a `(GenerationResult, full_text)` tuple.

> **Discrepancy to resolve here:** the comment at `inference.py:4102-4104` claims non-streaming VLM token counts "stay 0", but the committed live test `test_ollama_vlm_tools_with_image_produces_tool_call` asserts `prompt_eval_count > 200`. The token-count live test in Task 8 establishes the actual pre-change behavior; this change makes the counts correct regardless.

**Files:**
- Modify: `olmlx/engine/inference.py` — both VLM `mlx_vlm.generate(...)` call sites in `_full_completion_inner` (~lines 4097–4115 and ~4148–4161).

- [ ] **Step 1: Replace the `lm.is_vlm and images` branch**

Find (~lines 4097–4115):

```python
        if lm.is_vlm and images:
            if lm.is_speculative:
                logger.debug("speculative decoding skipped: request includes images")
            import mlx_vlm

            # mlx_vlm.generate returns a plain str; prompt/generation token
            # counts are not exposed, so stats.prompt_eval_count /
            # stats.eval_count stay 0 on this path.
            result = mlx_vlm.generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                image=images,
                max_tokens=max_tokens,
                **gen_kwargs,
            )
            from mlx_vlm.generate import (
                generation_stream,
            )  # used by mx.synchronize below
```

Replace with:

```python
        if lm.is_vlm and images:
            if lm.is_speculative:
                logger.debug("speculative decoding skipped: request includes images")
            import mlx_vlm
            from mlx_vlm.generate import (
                generation_stream,
            )  # used by mx.synchronize below

            # Drain stream_generate (not generate): it forwards prompt_cache_state
            # + logits_processors and yields GenerationResult with real token
            # counts. Return (last_result, full_text) so the downstream tuple
            # unpacking captures prompt/generation token counts (#429).
            result = None
            text_parts = []
            for response in mlx_vlm.stream_generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                image=images,
                max_tokens=max_tokens,
                **gen_kwargs,
            ):
                text_parts.append(response.text)
                result = response
            if result is not None:
                result = (result, "".join(text_parts))
```

- [ ] **Step 2: Replace the `elif lm.is_vlm` (no-images) branch**

Find (~lines 4148–4161):

```python
        elif lm.is_vlm:
            import mlx_vlm

            result = mlx_vlm.generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                image=images,
                max_tokens=max_tokens,
                **gen_kwargs,
            )
            from mlx_vlm.generate import (
                generation_stream,
            )  # used by mx.synchronize below
```

Replace with:

```python
        elif lm.is_vlm:
            import mlx_vlm
            from mlx_vlm.generate import (
                generation_stream,
            )  # used by mx.synchronize below

            result = None
            text_parts = []
            for response in mlx_vlm.stream_generate(
                lm.model,
                lm.tokenizer,
                prompt=prompt,
                image=images,
                max_tokens=max_tokens,
                **gen_kwargs,
            ):
                text_parts.append(response.text)
                result = response
            if result is not None:
                result = (result, "".join(text_parts))
```

- [ ] **Step 3: Static check (the tuple path is already handled downstream)**

Confirm `_full_completion_inner`'s post-generation block (~lines 4220–4230) unpacks a `(gen_result, full_text)` tuple and reads `result.prompt_tokens` / `result.generation_tokens`. It does — no change needed. Run a syntax/import smoke:

Run: `uv run python -c "import olmlx.engine.inference"`
Expected: no error.

- [ ] **Step 4: Run the inference suite for regressions**

Run: `uv run pytest -q -k "inference or completion or vlm" -m "not real_model"`
Expected: PASS (mocked VLM paths, if any, still work; real VLM exercised in Task 8).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/inference.py
git commit -m "feat(inference): non-streaming VLM drains stream_generate for counts + cache (#429)"
```

---

## Task 8: Live (real_model) tests

Real Gemma 4 VLM, in-process app over httpx (mirrors `tests/live/test_vlm_tools_images.py`). Covers all #429 acceptance criteria. Skipped in CI (`-m "not real_model"`) and when the model is not downloaded.

**Files:**
- Create: `tests/live/test_vlm_cache_grammar.py`

- [ ] **Step 1: Write the live tests**

Create `tests/live/test_vlm_cache_grammar.py`:

```python
"""Live VLM prompt-cache + grammar + token-count tests (#429).

Loads a REAL Gemma 4 VLM and verifies the lifted gating:
  * grammar (response_format json_schema) applies with an image present,
  * a multi-turn vision chat reuses the image-prefix KV (cache_read > 0) with
    output identical to the uncached (slots=0) path,
  * non-streaming VLM reports non-zero token counts,
  * a fresh image request is not silently dropped (image reaches the model).

Lives OUTSIDE tests/integration/ to dodge that package's autouse MLX mock.
Run on a machine where the model is downloaded.
"""

import base64
import io
import json

import pytest

from olmlx.config import settings

VLM_MODEL = "mlx-community/gemma-4-26B-A4B-it-4bit"


def _model_present() -> bool:
    from olmlx.models.store import _safe_dir_name

    return (settings.models_dir / _safe_dir_name(VLM_MODEL) / "config.json").exists()


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not _model_present(),
        reason=f"{VLM_MODEL} not downloaded in {settings.models_dir}",
    ),
]


def _png_data_uri_with_number(text: str = "42") -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (160, 96), "white")
    ImageDraw.Draw(img).text((50, 30), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


async def _make_client(tmp_path, monkeypatch):
    """Build a real app + ModelManager over the real model store. Caller sets
    OLMLX_VLM_PROMPT_CACHE_SLOTS via monkeypatch.setenv BEFORE calling this so
    the loaded model's VlmPromptCacheStore picks up the chosen capacity."""
    models_config = tmp_path / "models.json"
    models_config.write_text(json.dumps({}))
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text("{}")
    monkeypatch.setattr("olmlx.config.settings.models_config", models_config)
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_config)

    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.registry import ModelRegistry
    from olmlx.models.store import ModelStore

    registry = ModelRegistry()
    registry._aliases_path = aliases_path
    registry.load()
    store = ModelStore(registry)
    manager = ModelManager(registry, store)
    manager.start_expiry_checker()

    from olmlx.app import create_app

    app = create_app()
    app.state.registry = registry
    app.state.model_manager = manager
    app.state.model_store = store

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, manager


@pytest.fixture
async def live_client(tmp_path, monkeypatch):
    """Default client with VLM caching enabled (slots=2)."""
    monkeypatch.setattr("olmlx.config.settings.vlm_prompt_cache_slots", 2)
    client, manager = await _make_client(tmp_path, monkeypatch)
    async with client:
        yield client
    await manager.stop()


def _image_message(text: str, number: str = "42"):
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": _png_data_uri_with_number(number)}},
        ],
    }


async def test_grammar_json_schema_on_image_request(live_client):
    """Acceptance criterion 2: schema-valid JSON on an image request."""
    resp = await live_client.post(
        "/v1/chat/completions",
        json={
            "model": VLM_MODEL,
            "messages": [_image_message("Read the number in the image.")],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "number",
                    "schema": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            },
            "max_tokens": 64,
            "temperature": 0,
        },
        timeout=600,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)  # must be valid JSON conforming to the schema
    assert isinstance(parsed["value"], int), parsed


async def test_non_streaming_vlm_reports_token_counts(live_client):
    """Non-streaming VLM must report non-zero prompt/eval counts (#429)."""
    resp = await live_client.post(
        "/v1/chat/completions",
        json={
            "model": VLM_MODEL,
            "messages": [_image_message("Describe the image briefly.")],
            "max_tokens": 32,
            "temperature": 0,
            "stream": False,
        },
        timeout=600,
    )
    assert resp.status_code == 200, resp.text
    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] > 200, usage  # image placeholder expands large
    assert usage["completion_tokens"] > 0, usage


def _growing_vision_turns(n_assistant_turns):
    """A vision conversation that grows by appending assistant+user turns, so
    later turns share the (image-bearing) leading prefix with earlier ones."""
    img = _png_data_uri_with_number("42")
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What number is in the image?"},
                {"type": "image_url", "image_url": {"url": img}},
            ],
        }
    ]
    followups = ["Say it again.", "And once more."]
    for i in range(n_assistant_turns):
        msgs.append({"role": "assistant", "content": "42"})
        msgs.append({"role": "user", "content": followups[i]})
    return msgs


async def _ask(client, messages, cache_id):
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": VLM_MODEL,
            "messages": messages,
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
        },
        # cache_id is supplied via the x-cache-id header (olmlx/routers/openai.py).
        headers={"x-cache-id": cache_id},
        timeout=600,
    )
    assert r.status_code == 200, r.text
    return r.json()


async def test_multi_turn_vision_reuses_image_prefix_kv(live_client):
    """Acceptance criterion 1 (reuse half): a 3-turn vision chat under one
    cache_id reuses the image-prefix KV — observable via /api/ps."""
    out = []
    for turn in range(3):
        body = await _ask(live_client, _growing_vision_turns(turn), "vision-reuse")
        out.append(body["choices"][0]["message"]["content"])

    ps = await live_client.get("/api/ps")
    assert ps.status_code == 200, ps.text
    models = ps.json()["models"]
    me = next(m for m in models if m["model"] == VLM_MODEL or m["name"] == VLM_MODEL)
    metrics = me["cache_metrics"]
    # Turns 2 and 3 extend the cached prefix → real reuse.
    assert metrics.get("vlm_cache_hits", 0) >= 2, metrics
    assert metrics.get("vlm_cache_tokens_reused", 0) > 0, metrics


async def test_cached_and_uncached_outputs_match(tmp_path, monkeypatch):
    """Acceptance criterion 1 (correctness half): greedy output with caching ON
    equals output with caching OFF (slots=0). Two independently-built clients so
    each model's VlmPromptCacheStore is created with the intended capacity."""
    # Caching ON.
    monkeypatch.setattr("olmlx.config.settings.vlm_prompt_cache_slots", 2)
    client_on, manager_on = await _make_client(tmp_path, monkeypatch)
    async with client_on:
        cached = [
            (await _ask(client_on, _growing_vision_turns(t), "match-on"))[
                "choices"
            ][0]["message"]["content"]
            for t in range(3)
        ]
    await manager_on.stop()

    # Caching OFF (slots=0): store disabled, fresh prefill every turn.
    monkeypatch.setattr("olmlx.config.settings.vlm_prompt_cache_slots", 0)
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    client_off, manager_off = await _make_client(off_dir, monkeypatch)
    async with client_off:
        uncached = [
            (await _ask(client_off, _growing_vision_turns(t), "match-off"))[
                "choices"
            ][0]["message"]["content"]
            for t in range(3)
        ]
    await manager_off.stop()

    assert cached == uncached, (cached, uncached)
```

- [ ] **Step 2: Run the new live tests (on a machine with the model)**

Run: `uv run pytest tests/live/test_vlm_cache_grammar.py -v -m real_model`
Expected: PASS (or SKIP if the model isn't downloaded). Resolve the implementer notes until green.

- [ ] **Step 3: Confirm CI-safe collection (no real model)**

Run: `uv run pytest tests/live/test_vlm_cache_grammar.py -q -m "not real_model"`
Expected: deselected/skipped, no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/live/test_vlm_cache_grammar.py
git commit -m "test(vlm): live cache + grammar + token-count coverage (#429)"
```

---

## Task 9: Documentation — CLAUDE.md + manual

**Files:**
- Modify: `CLAUDE.md` — the VLM-gating design note (the "VLM tools + images (#428)" bullet and the prompt-caching/grammar/speculative gating notes).

- [ ] **Step 1: Update the grammar (structured outputs) note**

In `CLAUDE.md`, find the **Structured outputs** bullet that ends with "VLM, distributed, and speculative paths skip with a warning". Update it to state that VLM is now supported:

```
- **Structured outputs** (`engine/grammar.py`): … Compiled grammars cached per (tokenizer-id, spec-hash). **VLM now supported** (#429): mlx_vlm's `generate_step` accepts `logits_processors`, which olmlx forwards via `gen_kwargs`; `_resolve_model_vocab_size` descends into `model.language_model` to size the bitmask. Distributed and (non-VLM) speculative paths still skip with a warning — `logits_processors` isn't threaded through them.
```

- [ ] **Step 2: Update the prompt-caching note**

Find the **Prompt caching** bullet ("non-streaming VLM gated off (`mlx_vlm.generate` doesn't accept `prompt_cache`)") and replace that clause with the new VLM design:

```
- **VLM prompt caching** (#429): both streaming and non-streaming VLM requests reuse the image-prefix KV across turns via mlx_vlm's native `PromptCacheState`, held in a per-`cache_id` LRU (`VlmPromptCacheStore`, `engine/prompt_cache/vlm_state.py`, `OLMLX_VLM_PROMPT_CACHE_SLOTS`, default 2, 0=off). mlx_vlm owns prefix matching / cache trimming / image-in-prefix detection and updates the state post-generation; olmlx only fetch-or-creates the state and reports approximate read/creation counts. Non-streaming VLM drains `mlx_vlm.stream_generate` (not `generate`) so `prompt_cache_state` + `logits_processors` thread through and token counts are real. **v1 limits:** in-memory only — no radix-takeover, no disk spill, no KV-quant for VLM; distributed VLM out of scope. The old text-path `input_ids` VLM hack (which silently dropped images on fresh requests) is removed.
```

- [ ] **Step 3: Add the speculative-on-VLM decision (acceptance criterion 3)**

Find the **Speculative decoding** section and add to its scope/limits text:

```
- **Speculative + VLM (images): out of scope** (#429). Drafts are text-only and the classic/eagle/mtp decoders don't thread image features through the target forward, so verifying a text draft against a VLM target isn't wired. Image requests skip speculative decoding (logged at debug); revisit as PLD-over-text-suffix if needed.
```

- [ ] **Step 4: Verify CLAUDE.md is coherent**

Run: `uv run python -c "open('CLAUDE.md').read()"` (smoke) and re-read the three edited bullets for internal consistency (no remaining "mlx_vlm.generate doesn't accept prompt_cache" / "VLM ... skip" claims for grammar or cache).

Run: `grep -n "mlx_vlm.generate doesn't accept\|VLM, distributed, and speculative paths skip" CLAUDE.md`
Expected: no matches (both stale claims removed).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude-md): VLM prompt cache + grammar supported; speculative out of scope (#429)"
```

---

## Final verification

- [ ] **Run the full non-real-model suite**

Run: `uv run ruff check olmlx tests && uv run ruff format --check olmlx tests && uv run pytest -q -m "not real_model"`
Expected: lint clean, format clean, tests PASS (modulo the known full-suite SIGABRT flake — re-run targeted suites if it trips).

- [ ] **Run the VLM live suite (machine with the model)**

Run: `uv run pytest tests/live/ -v -m real_model`
Expected: PASS — `test_vlm_tools_images.py` (existing, must still pass) + `test_vlm_cache_grammar.py` (new).

- [ ] **Cross-thread lazy-graph watch (#284 family):** the `PromptCacheState` cache
  persists in the store across requests and is mutated on the generation worker
  thread. mlx_vlm `mx.eval`s the cache during/after generation, so it should be
  materialized before the next cross-thread reuse. If the multi-turn live test
  produces a Metal stream error or corrupted output on turn ≥2, eager-eval the cache
  on store insert/update — add `mx.eval([c.state for c in state.cache])` in
  `_setup_vlm_prompt_cache` after a hit (the same remedy as
  `snapshot_cache_for_persistence`). Only add this if the symptom appears; don't
  pre-emptively pessimize.

- [ ] **Confirm all #429 acceptance criteria are met:**
  - [ ] 3-turn vision chat reuses image-prefix KV, output identical to uncached path (Task 8).
  - [ ] `response_format` json_schema produces schema-valid JSON on an image request (Task 8).
  - [ ] Speculative-with-images decision documented in CLAUDE.md (Task 9).
  - [ ] Non-streaming VLM reports non-zero token counts (Task 8).
