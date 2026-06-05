# MTP Draft-Head Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `mtp` speculative-decoding strategy that loads Qwen3.6's pretrained, shipped multi-token-prediction head (`model_type: qwen3_5_mtp`) as the draft for a matching Qwen3.6 target (27B dense + 35B-A3B MoE).

**Architecture:** A new `olmlx/engine/mtp/` module with `MTPDraftModel` (one full-attention Qwen3.6 layer reusing mlx-lm's `qwen3_5`/`qwen3_next` blocks, plus the MTP front-end: `pre_fc_norm_hidden`/`pre_fc_norm_embedding` → `fc` projection of `concat[hidden; embed]`, and a final `norm`; `embed_tokens`/`lm_head` borrowed from the target) and `MTPDecoder` (prefill/step/reset, structurally a sibling of `EagleDecoder`, composing the already-shared `_patch_model` hidden-capture, `GDNStateCapture` rejection-rollback, and `verify_draft_greedy`). No training step — the head is pretrained. No changes to `olmlx/engine/eagle/*`.

**Tech Stack:** Python, MLX, mlx-lm (`qwen3_5`, `qwen3_next`, `cache`, `base`), pytest.

**Reference (read before starting):**
- Spec: `docs/superpowers/specs/2026-06-04-mtp-draft-support-design.md`
- EAGLE draft model: `olmlx/engine/eagle/draft_model.py` (the `bind()`/`_find_embed`/`_find_lm_head` pattern to copy; **note MTP differs**: it chains the *pre-final-norm* hidden, not `norm(x)`)
- EAGLE decoder: `olmlx/engine/eagle/decoder.py` (prefill/step/reset to mirror almost verbatim)
- mlx-lm blocks: `.venv/lib/python3.11/site-packages/mlx_lm/models/qwen3_5.py` (`TextModelArgs`, `DecoderLayer`) and `qwen3_next.py` (`Qwen3NextAttention`, `Qwen3NextMLP`, `Qwen3NextSparseMoeBlock`)

**Key facts already established (do not re-derive):**
- MTP head weights: `fc.{weight,scales,biases}` (logical `2*hidden → hidden`), `pre_fc_norm_hidden.weight`, `pre_fc_norm_embedding.weight`, `layers.0.{input_layernorm,post_attention_layernorm,self_attn.*,mlp.*}`, `norm.weight`. **No** `embed_tokens`/`lm_head` (borrowed). `block_size: 3`. Quantized (group_size 64, bits 4).
- The MTP layer is **full-attention** (has `q_norm`/`k_norm` + output gate) — EAGLE's `_Attention` lacks these, so reuse mlx-lm's `qwen3_next` blocks, not EAGLE's.
- **Norm weights are already standard form** (means ~0.83–2.28, all positive). **Do NOT apply the mlx-lm `+1.0` zero-centered shift.**
- Capture point = output of the target's **last** layer (pre `model.norm`), which is exactly what `_patch_model` with `target_layer_id = num_layers-1` captures.
- 35B-A3B MTP head's `layers.0.mlp` is MoE (`gate` + `shared_expert` + routed experts).
- flash_moe + MTP is mutually exclusive (mirror eagle/dflash) — reject at load.

**Branch:** `feat/mtp-speculative` (already created; spec already committed).

---

## File Structure

- Create: `olmlx/engine/mtp/__init__.py` — module marker + public exports
- Create: `olmlx/engine/mtp/draft_model.py` — `MTPConfig`, `MTPDraftModel`, `load_mtp_draft(path, ...)`
- Create: `olmlx/engine/mtp/decoder.py` — `MTPDecoder`
- Modify: `olmlx/engine/registry.py:17-20` — add `"mtp"` to strategy literal + valid set
- Modify: `olmlx/engine/model_manager.py` — `_load_mtp_decoder` + dispatch branch + `_FLASH_MOE_INCOMPATIBLE_STRATEGIES` constant (extends the two existing flash_moe guards)
- Modify: `olmlx/utils/metrics.py:172` — add `"MTPDecoder": "mtp"` to `_STRATEGY_BY_CLASS`
- Create: `tests/test_mtp_config.py`
- Create: `tests/test_mtp_draft_model.py`
- Create: `tests/test_mtp_loader.py`
- Create: `tests/test_mtp_decoder.py`
- Create: `tests/test_mtp_integration.py` — live, gated on model presence
- Modify: `CLAUDE.md` — document the `mtp` strategy

---

## Task 1: Register the `mtp` strategy + module scaffold

**Files:**
- Modify: `olmlx/engine/registry.py:17-20`
- Create: `olmlx/engine/mtp/__init__.py`
- Test: `tests/test_mtp_config.py` (registry portion)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mtp_config.py
from olmlx.engine.registry import _VALID_SPECULATIVE_STRATEGIES


def test_mtp_is_a_valid_strategy():
    assert "mtp" in _VALID_SPECULATIVE_STRATEGIES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mtp_config.py::test_mtp_is_a_valid_strategy -v`
Expected: FAIL — `"mtp"` not in the frozenset.

- [ ] **Step 3: Edit the registry**

In `olmlx/engine/registry.py`, change the strategy literal and valid set (currently lines 17-20):

```python
SpeculativeStrategy = Literal[
    "classic", "dflash", "eagle", "pld", "self_speculative", "mtp"
]
_VALID_SPECULATIVE_STRATEGIES: frozenset[str] = frozenset(
    ("classic", "dflash", "eagle", "pld", "self_speculative", "mtp")
)
```

- [ ] **Step 4: Create the module marker**

```python
# olmlx/engine/mtp/__init__.py
"""MTP (multi-token-prediction) speculative draft head.

Loads Qwen3.6's pretrained ``qwen3_5_mtp`` head as the speculative draft
for a matching Qwen3.6 target. See
``docs/superpowers/specs/2026-06-04-mtp-draft-support-design.md``.
"""
```

Tests are flat under `tests/` (e.g. `tests/test_eagle.py`) — no package `__init__.py` is needed. Cross-test imports like `from tests.test_mtp_config import _DENSE_CFG` resolve under pytest's rootdir; confirm by running the test in Step 5.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_mtp_config.py::test_mtp_is_a_valid_strategy -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/registry.py olmlx/engine/mtp/__init__.py tests/
git commit -m "feat(mtp): register mtp speculative strategy + module scaffold"
```

---

## Task 2: `MTPConfig` — parse a `qwen3_5_mtp` config

**Files:**
- Create: `olmlx/engine/mtp/draft_model.py`
- Test: `tests/test_mtp_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mtp_config.py (append)
from olmlx.engine.mtp.draft_model import MTPConfig

_DENSE_CFG = {
    "block_size": 3,
    "model_type": "qwen3_5_mtp",
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "text_config": {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "full_attention_interval": 4,
        "tie_word_embeddings": False,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    },
}

_MOE_CFG = {
    "block_size": 3,
    "model_type": "qwen3_5_mtp",
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "text_config": {
        **_DENSE_CFG["text_config"],
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "shared_expert_intermediate_size": 768,
        "norm_topk_prob": True,
    },
}


def test_mtp_config_parses_dense():
    cfg = MTPConfig.from_dict(_DENSE_CFG)
    assert cfg.block_size == 3
    assert cfg.hidden_size == 5120
    assert cfg.head_dim == 256
    assert cfg.num_key_value_heads == 4
    assert cfg.vocab_size == 248320
    assert cfg.num_experts == 0
    assert cfg.quant_group_size == 64 and cfg.quant_bits == 4


def test_mtp_config_parses_moe():
    cfg = MTPConfig.from_dict(_MOE_CFG)
    assert cfg.num_experts == 128
    assert cfg.num_experts_per_tok == 8
    assert cfg.is_moe is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mtp_config.py -v`
Expected: FAIL — `MTPConfig` not defined.

- [ ] **Step 3: Implement `MTPConfig`**

```python
# olmlx/engine/mtp/draft_model.py
"""MTP draft head: one full-attention Qwen3.6 layer + MTP front-end.

The head consumes ``(token_{i+1}, h_i)`` where ``h_i`` is the target's
last-layer (pre-``model.norm``) hidden, and produces ``(logits, h_new)``.
``h_new`` is the layer output BEFORE the head's own ``norm`` — it is fed
back as ``h_prev`` for the next autoregressive draft step (DeepSeek/Qwen
MTP convention). ``norm`` is applied only to compute logits via the
target's borrowed ``lm_head``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer as _Qwen35DecoderLayer
from mlx_lm.models.qwen3_5 import TextModelArgs as _Qwen35TextArgs


@dataclass
class MTPConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    full_attention_interval: int
    block_size: int
    rope_parameters: dict[str, Any] | None = None
    tie_word_embeddings: bool = False
    # MoE (0 => dense)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    # Quantization (None => not quantized)
    quant_group_size: int | None = None
    quant_bits: int | None = None

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "MTPConfig":
        text = config.get("text_config", config)
        quant = config.get("quantization") or config.get("quantization_config") or {}
        return cls(
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text.get(
                "head_dim", text["hidden_size"] // text["num_attention_heads"]
            ),
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            vocab_size=text["vocab_size"],
            max_position_embeddings=text.get("max_position_embeddings", 262144),
            full_attention_interval=text.get("full_attention_interval", 4),
            block_size=config.get("block_size", 1),
            rope_parameters=text.get("rope_parameters"),
            tie_word_embeddings=text.get("tie_word_embeddings", False),
            num_experts=text.get("num_experts", 0),
            num_experts_per_tok=text.get("num_experts_per_tok", 0),
            moe_intermediate_size=text.get("moe_intermediate_size", 0),
            shared_expert_intermediate_size=text.get(
                "shared_expert_intermediate_size", 0
            ),
            norm_topk_prob=text.get("norm_topk_prob", True),
            decoder_sparse_step=text.get("decoder_sparse_step", 1),
            quant_group_size=quant.get("group_size"),
            quant_bits=quant.get("bits"),
        )

    def to_qwen35_text_args(self) -> _Qwen35TextArgs:
        """Build the mlx-lm ``TextModelArgs`` the reused ``DecoderLayer``
        expects. ``num_hidden_layers`` is forced to ``full_attention_interval``
        so layer_idx ``full_attention_interval - 1`` is a FULL-attention
        layer (``(idx+1) % interval == 0``)."""
        return _Qwen35TextArgs(
            model_type="qwen3_5_text",
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.full_attention_interval,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rms_norm_eps=self.rms_norm_eps,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            full_attention_interval=self.full_attention_interval,
            tie_word_embeddings=self.tie_word_embeddings,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_intermediate_size=self.moe_intermediate_size,
            shared_expert_intermediate_size=self.shared_expert_intermediate_size,
            norm_topk_prob=self.norm_topk_prob,
            decoder_sparse_step=self.decoder_sparse_step,
            rope_parameters=self.rope_parameters
            or {
                "type": "default",
                "mrope_section": [11, 11, 10],
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mtp_config.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/mtp/draft_model.py tests/test_mtp_config.py
git commit -m "feat(mtp): MTPConfig parser for qwen3_5_mtp configs"
```

---

## Task 3: `MTPDraftModel` — build + forward (dense)

**Files:**
- Modify: `olmlx/engine/mtp/draft_model.py`
- Test: `tests/test_mtp_draft_model.py`

The single layer is reused from mlx-lm: instantiate `_Qwen35DecoderLayer(args, layer_idx=full_attention_interval-1)` so it is the full-attention variant. Its submodule names are `self_attn.*`, `input_layernorm`, `post_attention_layernorm`, `mlp.*` — exactly the MTP weight keys under `layers.0.`. Borrow `embed_tokens`/`lm_head` exactly like EAGLE (copy `_find_embed`/`_find_lm_head`/`bind`/`unbind`/`bind_via_modules` from `eagle/draft_model.py` verbatim).

**`__call__` differs from EAGLE in two places (the design's correctness crux):**
1. Front-end: `x = fc(concat([pre_fc_norm_hidden(h_prev), pre_fc_norm_embedding(emb)], axis=-1))` — two separate norms, not EAGLE's single `input_proj`.
2. Chained hidden: return `h_new = x` (pre-`norm`), and `logits = lm_head(norm(x))`. EAGLE returns `norm(x)` for both.

Make the concat order configurable via a module attribute `self.concat_hidden_first: bool = True` (default `[hidden, embed]`) so the Task 8 probe can flip it without an interface change.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mtp_draft_model.py
import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.mtp.draft_model import MTPConfig, MTPDraftModel
from tests.test_mtp_config import _DENSE_CFG


def _tiny_cfg():
    # Shrink the dense config so the test builds instantly.
    cfg = MTPConfig.from_dict(_DENSE_CFG)
    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 32
    cfg.vocab_size = 512
    return cfg


def test_mtp_draft_forward_shapes():
    cfg = _tiny_cfg()
    draft = MTPDraftModel(cfg)
    embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    draft.bind_via_modules(embed, lm_head)
    mx.eval(draft.parameters())

    tok = mx.array([[5]], dtype=mx.int32)
    h_prev = mx.zeros((1, 1, cfg.hidden_size))
    cache = draft.make_cache()
    logits, h_new = draft(tok, h_prev, cache=cache)
    assert logits.shape == (1, 1, cfg.vocab_size)
    assert h_new.shape == (1, 1, cfg.hidden_size)


def test_mtp_draft_requires_bind():
    cfg = _tiny_cfg()
    draft = MTPDraftModel(cfg)
    try:
        draft(mx.array([[1]], dtype=mx.int32), mx.zeros((1, 1, cfg.hidden_size)))
        assert False, "expected RuntimeError without bind()"
    except RuntimeError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mtp_draft_model.py -v`
Expected: FAIL — `MTPDraftModel` not defined.

- [ ] **Step 3: Implement `MTPDraftModel`**

Append to `olmlx/engine/mtp/draft_model.py`:

```python
class MTPDraftModel(nn.Module):
    """Single-layer MTP draft head conditioned on target hidden states.

    Forward: ``(token_ids, h_prev, cache=None, compute_logits=True) ->
    (logits|None, h_new)``. ``embed_tokens``/``lm_head`` are borrowed from
    the target via ``bind()`` (kept out of the parameter tree via
    ``object.__setattr__``, same as EAGLE).
    """

    def __init__(self, args: MTPConfig):
        super().__init__()
        self.args = args
        self.concat_hidden_first = True  # [hidden, embed]; Task 8 may flip
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.fc = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        # One full-attention Qwen3.6 layer. layer_idx = interval-1 forces
        # the non-linear (full attention) branch.
        text_args = args.to_qwen35_text_args()
        self.layers = [
            _Qwen35DecoderLayer(text_args, layer_idx=args.full_attention_interval - 1)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        object.__setattr__(self, "embed_tokens", None)
        object.__setattr__(self, "lm_head", None)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    # --- bind/unbind: copied verbatim from EagleDraftModel ---
    def bind(self, target_model: Any) -> None:
        embed = self._find_embed(target_model)
        if embed is None:
            raise AttributeError(
                f"Cannot find embed_tokens on target {type(target_model).__name__}"
            )
        lm_head = self._find_lm_head(target_model, embed)
        if lm_head is None:
            raise AttributeError(
                f"Cannot find lm_head on target {type(target_model).__name__}"
            )
        object.__setattr__(self, "embed_tokens", embed)
        object.__setattr__(self, "lm_head", lm_head)

    @staticmethod
    def _find_embed(target: Any) -> nn.Module | None:
        for path in (
            ("embed_tokens",),
            ("model", "embed_tokens"),
            ("language_model", "model", "embed_tokens"),
            ("language_model", "embed_tokens"),
        ):
            obj: Any = target
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        return None

    @staticmethod
    def _find_lm_head(
        target: Any, embed: nn.Module
    ) -> nn.Module | Callable[..., Any] | None:
        for path in (
            ("lm_head",),
            ("language_model", "lm_head"),
            ("model", "lm_head"),
            ("language_model", "model", "lm_head"),
        ):
            obj: Any = target
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        as_linear = getattr(embed, "as_linear", None)
        if callable(as_linear):
            return as_linear
        return None

    def bind_via_modules(self, embed_tokens: nn.Module, lm_head: nn.Module) -> None:
        object.__setattr__(self, "embed_tokens", embed_tokens)
        object.__setattr__(self, "lm_head", lm_head)

    def unbind(self) -> None:
        object.__setattr__(self, "embed_tokens", None)
        object.__setattr__(self, "lm_head", None)

    def __call__(
        self,
        token_ids: mx.array,
        h_prev: mx.array,
        cache: list[KVCache] | None = None,
        compute_logits: bool = True,
    ) -> tuple[mx.array | None, mx.array]:
        if self.embed_tokens is None:
            raise RuntimeError("MTPDraftModel.__call__ requires bind() first.")
        if compute_logits and self.lm_head is None:
            raise RuntimeError(
                "MTPDraftModel.__call__(compute_logits=True) requires bind()."
            )
        emb = self.embed_tokens(token_ids)
        h = self.pre_fc_norm_hidden(h_prev)
        e = self.pre_fc_norm_embedding(emb)
        parts = [h, e] if self.concat_hidden_first else [e, h]
        x = self.fc(mx.concatenate(parts, axis=-1))

        L = x.shape[1]
        mask = None
        if L > 1:
            mask = create_causal_mask(L, offset=cache[0].offset if cache else 0)
        layer_cache = cache[0] if cache is not None else None
        x = self.layers[0](x, mask=mask, cache=layer_cache)

        h_new = x  # pre-norm; chained for the next draft step
        if compute_logits:
            lm_head = self.lm_head
            if lm_head is None:
                raise RuntimeError("MTPDraftModel internal invariant: lm_head None.")
            return lm_head(self.norm(x)), h_new
        return None, h_new
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mtp_draft_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/mtp/draft_model.py tests/test_mtp_draft_model.py
git commit -m "feat(mtp): MTPDraftModel (dense) build + forward"
```

---

## Task 4: `load_mtp_draft` — quantize + strict weight load (dense)

**Files:**
- Modify: `olmlx/engine/mtp/draft_model.py`
- Test: `tests/test_mtp_loader.py`

The shipped head is quantized. Pattern: build `MTPDraftModel`, `nn.quantize(model, group_size, bits)` when the config declares quantization, then `model.load_weights(list(weights.items()), strict=True)`. Do **not** shift norms (already standard form). The borrowed embed/lm_head are not part of the draft, so they are absent from the head's weight file — `strict=True` over the draft's own params must leave **no** missing/unexpected keys.

- [ ] **Step 1: Write the failing test (live, gated on the head being present)**

```python
# tests/test_mtp_loader.py
import json
from pathlib import Path

import pytest

from olmlx.engine.mtp.draft_model import MTPConfig, MTPDraftModel, load_mtp_draft

_HEAD = "mlx-community/Qwen3.6-27B-MTP-4bit"


def _head_dir() -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(_HEAD))
    except Exception:
        return None


@pytest.mark.skipif(_head_dir() is None, reason="MTP head not downloadable")
def test_load_mtp_draft_strict_no_leftover_keys():
    path = _head_dir()
    cfg = MTPConfig.from_dict(json.loads((path / "config.json").read_text()))
    # load_mtp_draft must succeed with strict weight loading (raises on any
    # missing/unexpected key in the draft's own parameter tree).
    draft = load_mtp_draft(path, cfg)
    assert isinstance(draft, MTPDraftModel)
    assert cfg.num_experts == 0  # 27B head is dense
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mtp_loader.py -v`
Expected: FAIL — `load_mtp_draft` not defined (or SKIP if offline; ensure the head is downloaded first via `uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3.6-27B-MTP-4bit')"`).

- [ ] **Step 3: Implement `load_mtp_draft`**

Append to `olmlx/engine/mtp/draft_model.py`:

```python
def load_mtp_draft(path: Any, cfg: MTPConfig) -> MTPDraftModel:
    """Build an ``MTPDraftModel`` and load the shipped (quantized) weights.

    Quantizes the module to match the head's stored layout before loading.
    Loads with ``strict=True`` over the draft's own parameters: the head
    file contains no ``embed_tokens``/``lm_head`` (those are borrowed), and
    the draft does not register them, so there must be zero missing/unexpected
    keys. Norm weights are loaded as-is (already standard form — NO +1.0 shift).
    """
    import glob
    import os

    path = str(path)
    draft = MTPDraftModel(cfg)
    if cfg.quant_bits is not None and cfg.quant_group_size is not None:
        nn.quantize(draft, group_size=cfg.quant_group_size, bits=cfg.quant_bits)

    weights: dict[str, mx.array] = {}
    for wf in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        weights.update(mx.load(wf))

    draft.load_weights(list(weights.items()), strict=True)
    mx.eval(draft.parameters())
    return draft
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mtp_loader.py -v`
Expected: PASS. If it fails on a key mismatch, print `set(weights) ^ set(dict(tree_flatten(draft.parameters())))` to see which keys differ; the likely culprit is a quantize-predicate gap (a Linear that wasn't quantized) — fix by matching the head's quantization config, not by relaxing `strict`.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/mtp/draft_model.py tests/test_mtp_loader.py
git commit -m "feat(mtp): load_mtp_draft quantize + strict weight load (dense)"
```

---

## Task 5: MoE MLP variant (35B-A3B head)

**Files:**
- Modify: `olmlx/engine/mtp/draft_model.py` (only if needed)
- Test: `tests/test_mtp_loader.py` (append)

`_Qwen35DecoderLayer` already builds `SparseMoeBlock(args)` when `args.num_experts > 0` (qwen3_5.py:223-226), so the dense build path should also produce the correct MoE layer for the 35B config — no model code change expected. The risk is quantization of the MoE expert `SwitchLinear` weights: plain `nn.quantize` may not cover them the way the head stored them. If the strict load fails, reuse mlx-lm's MoE quant predicate (the `TextModel.quant_predicate` property in qwen3_5.py:333).

- [ ] **Step 1: Write the failing test (live, gated)**

```python
# tests/test_mtp_loader.py (append)
_MOE_HEAD = "mlx-community/Qwen3.6-35B-A3B-MTP-4bit"


def _moe_head_dir() -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(_MOE_HEAD))
    except Exception:
        return None


@pytest.mark.skipif(_moe_head_dir() is None, reason="MoE MTP head not downloadable")
def test_load_mtp_draft_moe_strict():
    path = _moe_head_dir()
    cfg = MTPConfig.from_dict(json.loads((path / "config.json").read_text()))
    assert cfg.is_moe
    draft = load_mtp_draft(path, cfg)
    assert isinstance(draft, MTPDraftModel)
```

- [ ] **Step 2: Run test to verify it fails (or passes immediately)**

Run: `uv run pytest tests/test_mtp_loader.py::test_load_mtp_draft_moe_strict -v`
Expected: PASS if `nn.quantize` already covers the MoE experts; FAIL on a key/shape mismatch otherwise.

- [ ] **Step 3: If it failed, switch to mlx-lm's MoE quant predicate**

In `load_mtp_draft`, replace the unconditional `nn.quantize(draft, ...)` with a predicate that mirrors mlx-lm for MoE heads:

```python
    if cfg.quant_bits is not None and cfg.quant_group_size is not None:
        if cfg.is_moe:
            # SwitchLinear experts need the same per-path predicate mlx-lm
            # uses (qwen3_5.py TextModel.quant_predicate); group_size must
            # divide the expert intermediate dim.
            def _predicate(path: str, module: nn.Module) -> bool:
                return hasattr(module, "to_quantized")

            nn.quantize(
                draft,
                group_size=cfg.quant_group_size,
                bits=cfg.quant_bits,
                class_predicate=_predicate,
            )
        else:
            nn.quantize(draft, group_size=cfg.quant_group_size, bits=cfg.quant_bits)
```

If the mismatch is a `group_size` divisibility error on a small expert dim, read the head's `quantization` block in `config.json` for any per-tensor overrides and honor them (the dense head used a single flat group_size; the MoE head may not).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mtp_loader.py::test_load_mtp_draft_moe_strict -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/mtp/draft_model.py tests/test_mtp_loader.py
git commit -m "feat(mtp): support MoE MTP head (35B-A3B) weight load"
```

---

## Task 6: `MTPDecoder` — prefill/step/reset

**Files:**
- Create: `olmlx/engine/mtp/decoder.py`
- Test: `tests/test_mtp_decoder.py`

Mirror `EagleDecoder` almost verbatim (copy the imports, `__init__`, `close`/`reset`/`__del__`, `stats_summary`, `prefill`, `step`/`_step_impl`). **Three differences only:**
1. Type the draft as `MTPDraftModel` (import from `olmlx.engine.mtp.draft_model`).
2. The draft `__call__` already returns the pre-norm chained hidden, so the seed/rotation logic is unchanged — `self._seed_hidden` is the target's captured layer output (pre `model.norm`), which matches the head's expected input space.
3. `stats_summary()` keeps the same keys; the strategy label for metrics is `"mtp"` (see Task 7 — the metrics layer reads the decoder type).

Everything else (the GDN capture path, two-pass prefill, `verify_draft_greedy`, trim arithmetic `trim = (block_size+1) - num_accepted`, `rollback_single(accepted=num_accepted-1)`) is identical to EAGLE because the verify/cache mechanics are draft-agnostic.

- [ ] **Step 1: Write the failing test (synthetic target — real prefill/step coverage, no large model)**

`test_eagle.py` already exercises `EagleDecoder` against a tiny `_SyntheticTarget` (a module with `.model.embed_tokens`, `.model.layers`, `.model.norm`, `.lm_head`, and a trim-able `KVCache` per layer). Reuse that exact harness for the MTP decoder — the verify/cache mechanics are draft-agnostic, so a synthetic target validates prefill→step→stats→reset without loading the 27B.

```python
# tests/test_mtp_decoder.py
import inspect

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.mtp.decoder import MTPDecoder
from olmlx.engine.mtp.draft_model import MTPConfig, MTPDraftModel

# Copy _SimpleAttn, _SimpleLayer, _Inner, _SyntheticTarget verbatim from
# tests/test_eagle.py (lines ~284-347). They have no EAGLE dependency.
from tests.test_eagle import _SyntheticTarget  # if importable; else copy them in


def _make_mtp_decoder(vocab=64, hidden=32, num_layers=3, block_size=2):
    target = _SyntheticTarget(vocab=vocab, hidden=hidden, num_layers=num_layers)
    cfg = MTPConfig(
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=hidden // 2,
        rms_norm_eps=1e-6,
        vocab_size=vocab,
        max_position_embeddings=512,
        full_attention_interval=4,
        block_size=block_size,
    )
    draft = MTPDraftModel(cfg)
    mx.eval(draft.parameters())
    decoder = MTPDecoder(target, draft, block_size=block_size)
    return decoder, target, draft


def test_mtp_decoder_protocol_surface():
    for name in ("prefill", "step", "reset", "stats_summary", "close"):
        assert callable(getattr(MTPDecoder, name)), name
    assert "cancel_event" in inspect.signature(MTPDecoder.prefill).parameters


def test_mtp_decoder_block_size_validation():
    target = _SyntheticTarget()
    cfg = MTPConfig(
        hidden_size=32, intermediate_size=64, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, rms_norm_eps=1e-6, vocab_size=64,
        max_position_embeddings=512, full_attention_interval=4, block_size=2,
    )
    try:
        MTPDecoder(target, MTPDraftModel(cfg), block_size=0)
        assert False, "expected ValueError for block_size=0"
    except ValueError:
        pass


def test_mtp_prefill_then_step_produces_accepted_tokens():
    decoder, _, _ = _make_mtp_decoder()
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    first = decoder.prefill(prompt)
    assert isinstance(first, int)
    accepted, num_drafts = decoder.step()
    assert len(accepted) == num_drafts + 1  # accepted drafts + 1 target token
    stats = decoder.stats_summary()
    assert stats["steps"] == 1 and stats["block_size"] == 2


def test_mtp_step_before_prefill_raises():
    decoder, _, _ = _make_mtp_decoder()
    try:
        decoder.step()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
```

If `from tests.test_eagle import _SyntheticTarget` doesn't resolve under the rootdir, copy the four synthetic classes (`_SimpleAttn`, `_SimpleLayer`, `_Inner`, `_SyntheticTarget`) into `tests/test_mtp_decoder.py` directly — they are EAGLE-independent.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mtp_decoder.py -v`
Expected: FAIL — `MTPDecoder` not defined.

- [ ] **Step 3: Implement `MTPDecoder`**

Create `olmlx/engine/mtp/decoder.py` by copying `olmlx/engine/eagle/decoder.py` wholesale and applying these edits:
- Module docstring: replace "EAGLE" with "MTP"; note the chained hidden is pre-final-norm.
- Imports: replace `from olmlx.engine.eagle.draft_model import EagleDraftModel` with `from olmlx.engine.mtp.draft_model import MTPDraftModel`.
- Class name `EagleDecoder` → `MTPDecoder`; type hint `draft_model: EagleDraftModel` → `MTPDraftModel`.
- Keep `block_size` default at `3` (the head's shipped `block_size`) instead of `4`.
- No other logic changes.

(The copy is intentional per the approved design — "dedicated MTPDecoder composing shared helpers" — so EAGLE stays untouched and the MTP hidden-space handling lives here. The shared helpers `_patch_model`, `GDNStateCapture`, `verify_draft_greedy` are imported, not duplicated.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mtp_decoder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/mtp/decoder.py tests/test_mtp_decoder.py
git commit -m "feat(mtp): MTPDecoder prefill/step/reset (mirrors EagleDecoder)"
```

---

## Task 7: `_load_mtp_decoder` + dispatch + flash_moe guard + metrics label

**Files:**
- Modify: `olmlx/engine/model_manager.py` — add `_load_mtp_decoder` (near `_load_eagle_decoder` ~line 2630); add the `mtp` dispatch branch (~line 3515); add `"mtp"` to the flash_moe-incompatibility guards (~lines 3330, 3461)
- Modify: `olmlx/utils/metrics.py:172` — add `"MTPDecoder": "mtp"` to `_STRATEGY_BY_CLASS`
- Test: `tests/test_mtp_loader.py` (append)

**How flash_moe exclusivity is already enforced:** the load flow rejects `dflash`/`eagle` under flash_moe at two sites (currently `if spec_config.strategy in ("dflash", "eagle"): raise ...` around lines 3330 and 3461), then the non-flash_moe dispatch (3515) builds the decoder. We extend both guard tuples to include `"mtp"` via a shared constant (DRY), and add the `mtp` dispatch branch.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mtp_loader.py (append)
def test_mtp_in_flash_moe_incompatible_set():
    """MTP must be rejected under flash_moe, like eagle/dflash."""
    from olmlx.engine.model_manager import _FLASH_MOE_INCOMPATIBLE_STRATEGIES

    assert "mtp" in _FLASH_MOE_INCOMPATIBLE_STRATEGIES
    assert {"dflash", "eagle"} <= _FLASH_MOE_INCOMPATIBLE_STRATEGIES


def test_mtp_loader_rejects_wrong_model_type(tmp_path, monkeypatch):
    """_load_mtp_decoder rejects a draft repo that isn't a qwen3_5_mtp head."""
    import json

    from olmlx.engine import model_manager as mm

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    mgr = mm.ModelManager.__new__(mm.ModelManager)
    monkeypatch.setattr(
        mgr, "_resolve_draft_path", lambda p: str(tmp_path), raising=False
    )

    class _Cfg:
        enabled = True
        draft_model = "some/not-an-mtp-head"
        num_tokens = None

    import pytest

    with pytest.raises(ValueError, match="qwen3_5_mtp"):
        mgr._load_mtp_decoder(object(), _Cfg())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mtp_loader.py::test_mtp_in_flash_moe_incompatible_set tests/test_mtp_loader.py::test_mtp_loader_rejects_wrong_model_type -v`
Expected: FAIL — `_FLASH_MOE_INCOMPATIBLE_STRATEGIES` / `_load_mtp_decoder` not defined.

- [ ] **Step 3: Add the shared constant + extend the two guards**

Near the top of `olmlx/engine/model_manager.py` (module level, after imports), add:

```python
# Speculative strategies that consume target hidden states / run a
# feature-conditioned verify forward and therefore can't compose with
# flash_moe's per-token expert offload. Rejected at load.
_FLASH_MOE_INCOMPATIBLE_STRATEGIES: frozenset[str] = frozenset(
    ("dflash", "eagle", "mtp")
)
```

At the two existing guard sites (currently `if spec_enabled and spec_config.strategy in ("dflash", "eagle"):` ~lines 3330 and 3461), replace the inline tuple with the constant:

```python
                if spec_enabled and spec_config.strategy in _FLASH_MOE_INCOMPATIBLE_STRATEGIES:
```

Keep each site's existing `raise`/error message text (it already says the strategy "is incompatible/not supported" under flash_moe — it interpolates `spec_config.strategy`, so it now reads correctly for `mtp` too).

- [ ] **Step 4: Add the dispatch branch + `_load_mtp_decoder`**

In the non-flash_moe dispatch chain (~line 3515), add before the final `else`:

```python
            elif spec_config.strategy == "mtp":
                decoder = self._load_mtp_decoder(model, spec_config)
```

Add the method near `_load_eagle_decoder`. No per-method flash_moe check is needed (Step 3's upstream guard covers it; `model` here is already the non-flash_moe model):

```python
    def _load_mtp_decoder(
        self, target_model: Any, spec_config: "SpeculativeConfig"
    ) -> Any:
        """Load Qwen3.6's native MTP head (``qwen3_5_mtp``) as the draft.

        Pretrained/shipped — no training step. flash_moe exclusivity is
        enforced upstream via ``_FLASH_MOE_INCOMPATIBLE_STRATEGIES``.
        """
        import json

        from olmlx.engine.mtp.decoder import MTPDecoder
        from olmlx.engine.mtp.draft_model import MTPConfig, load_mtp_draft

        if not spec_config.enabled:
            raise RuntimeError(
                "_load_mtp_decoder called with spec_config.enabled=False"
            )
        if not spec_config.draft_model:
            raise ValueError(
                "speculative_strategy='mtp' requires speculative_draft_model "
                "to point at the MTP head repo (e.g. "
                "mlx-community/Qwen3.6-27B-MTP-4bit)."
            )
        load_path = self._resolve_draft_path(spec_config.draft_model)
        cfg_dict = json.loads((Path(load_path) / "config.json").read_text())
        if cfg_dict.get("model_type") != "qwen3_5_mtp":
            raise ValueError(
                f"Expected an MTP head (model_type 'qwen3_5_mtp'); got "
                f"'{cfg_dict.get('model_type')}' at {spec_config.draft_model}."
            )
        cfg = MTPConfig.from_dict(cfg_dict)
        draft = load_mtp_draft(load_path, cfg)
        # Reuse the existing vocab guard. ``MTPDraftModel.args`` is an
        # ``MTPConfig`` exposing ``vocab_size``; the helper no-ops with a
        # warning when the target reports ``None`` (same as the classic path).
        self._check_vocab_match(target_model, draft)
        block_size = (
            spec_config.num_tokens
            if spec_config.num_tokens is not None
            else cfg.block_size
        )
        return MTPDecoder(target_model, draft, block_size=block_size)
```

`Path` is already imported in `model_manager.py` (used at `_flash_moe_dir`); no new import needed.

- [ ] **Step 5: Add the metrics strategy label**

In `olmlx/utils/metrics.py`, add the entry to `_STRATEGY_BY_CLASS` (line 172) so `_strategy_label`/metrics report `mtp`:

```python
_STRATEGY_BY_CLASS = {
    "SpeculativeDecoder": "classic",
    "SpeculativeFlashDecoder": "classic",
    "PromptLookupDecoder": "pld",
    "DFlashDecoder": "dflash",
    "EagleDecoder": "eagle",
    "SelfSpeculativeDecoder": "self",
    "MTPDecoder": "mtp",
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_mtp_loader.py tests/test_mtp_decoder.py -v`
Expected: PASS (config, draft, loader-dense, decoder, flash_moe-set, wrong-model-type). MoE + integration may SKIP if models absent.

- [ ] **Step 7: Commit**

```bash
git add olmlx/engine/model_manager.py olmlx/utils/metrics.py tests/test_mtp_loader.py
git commit -m "feat(mtp): _load_mtp_decoder dispatch, flash_moe exclusivity, metrics label"
```

---

## Task 8: Live integration + hidden-source probe (27B) — the acceptance gate

**Files:**
- Create: `tests/test_mtp_integration.py`
- Possibly modify: `olmlx/engine/mtp/draft_model.py` (flip `concat_hidden_first` if the probe says so)

This is where "generates correctly" is proven: output **token-identical** to non-speculative greedy **and acceptance > 0.66**. The decoder is exactness-preserving, so the real signal is acceptance. If acceptance is ~0, the wiring is wrong on one of the two remaining unknowns: **concat order** (`concat_hidden_first`) and the **chained-hidden space** (we default to pre-`norm`). The probe resolves them.

- [ ] **Step 1: Write the live integration test**

```python
# tests/test_mtp_integration.py
import os

import pytest

TARGET = "unsloth/Qwen3.6-27B-MLX-8bit"
HEAD = "mlx-community/Qwen3.6-27B-MTP-4bit"

pytestmark = pytest.mark.skipif(
    os.environ.get("OLMLX_RUN_MTP_INTEGRATION") != "1",
    reason="set OLMLX_RUN_MTP_INTEGRATION=1 to run the heavy MTP acceptance test",
)


def _generate(strategy, draft, max_tokens=120):
    """Run a fixed greedy prompt through ModelManager + generate_chat with
    the given speculative config; return (text, acceptance_rate)."""
    # Build a ModelManager with an in-memory models.json entry for TARGET
    # configured with speculative_strategy=strategy, speculative_draft_model=draft,
    # enable_thinking=False, temperature=0. Mirror the setup in
    # tests/ that already drives generate_chat against a real model
    # (grep for an existing speculative integration test to copy the harness).
    # Return the generated text and the decoder.stats_summary()['acceptance_rate'].
    raise NotImplementedError("wire to the existing generate_chat test harness")


def test_mtp_matches_greedy_and_beats_4b():
    baseline, _ = _generate(strategy=None, draft=None)         # non-spec greedy
    mtp_text, accept = _generate(strategy="mtp", draft=HEAD)
    assert mtp_text == baseline, "MTP output must equal non-speculative greedy"
    assert accept > 0.66, f"acceptance {accept:.3f} must beat the 4B classic draft"
```

Wire `_generate` to the existing real-model speculative harness — grep `tests/` for a test that already loads a model through `ModelManager` and calls `generate_chat`/the speculative stream with a per-model `speculative_*` config, and copy that setup. Use `temperature=0` and a fixed prompt (e.g. "Explain in 3 sentences why the sky is blue, then list two other colors the sky can appear.") so the greedy comparison is deterministic.

- [ ] **Step 2: Download models and run**

```bash
uv run python -c "from huggingface_hub import snapshot_download as d; d('mlx-community/Qwen3.6-27B-MTP-4bit')"
OLMLX_RUN_MTP_INTEGRATION=1 uv run pytest tests/test_mtp_integration.py::test_mtp_matches_greedy_and_beats_4b -v -s
```

Expected (first run): output equals greedy (exactness holds regardless of wiring). Acceptance is the discriminator.

- [ ] **Step 3: If acceptance ≤ 0.66, run the wiring probe**

Toggle the two unknowns and re-measure acceptance (4 combos), keeping the highest:

1. `concat_hidden_first = True`, chain pre-norm (default)
2. `concat_hidden_first = False`, chain pre-norm
3. chain post-norm: in `MTPDraftModel.__call__`, set `h_new = self.norm(x)` and `logits = lm_head(h_new)` (EAGLE-style), with `concat_hidden_first = True`
4. post-norm + `concat_hidden_first = False`

The correct combination jumps to ~0.7–0.85; the others sit near random (<0.1). Lock in the winner by setting the default in code (and, if post-norm wins, capture the post-`model.norm` target hidden instead — but note `_patch_model` captures a layer output, so post-norm chaining would also require normalizing the seed hidden with the target's `model.norm`; the pre-norm default avoids that and is the expected winner per the DeepSeek/Qwen MTP convention). Add a one-line comment recording the empirical result.

- [ ] **Step 4: Re-run to confirm the gate passes**

Run: `OLMLX_RUN_MTP_INTEGRATION=1 uv run pytest tests/test_mtp_integration.py -v -s`
Expected: PASS — output == greedy and acceptance > 0.66. Record the measured acceptance in the commit message.

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/mtp/draft_model.py tests/test_mtp_integration.py
git commit -m "test(mtp): 27B acceptance gate (output==greedy, accept=<MEASURED>)"
```

---

## Task 9: Live integration (35B-A3B MoE, flash_moe off)

**Files:**
- Modify: `tests/test_mtp_integration.py` (append)

- [ ] **Step 1: Write the live test**

```python
# tests/test_mtp_integration.py (append)
MOE_TARGET = "mlx-community/Qwen3.6-35B-A3B-4bit"   # flash_moe OFF for MTP
MOE_HEAD = "mlx-community/Qwen3.6-35B-A3B-MTP-4bit"


def test_mtp_moe_matches_greedy():
    baseline, _ = _generate(strategy=None, draft=None, target=MOE_TARGET)
    mtp_text, accept = _generate(strategy="mtp", draft=MOE_HEAD, target=MOE_TARGET)
    assert mtp_text == baseline
    assert accept > 0.5, f"MoE MTP acceptance {accept:.3f} should be clearly non-trivial"
```

Extend `_generate` to accept a `target` argument (default `TARGET`). Ensure the 35B target is configured with **flash_moe off** (do not set `flash_moe` in its models.json entry).

- [ ] **Step 2: Download + run**

```bash
uv run python -c "from huggingface_hub import snapshot_download as d; [d(x) for x in ('mlx-community/Qwen3.6-35B-A3B-4bit','mlx-community/Qwen3.6-35B-A3B-MTP-4bit')]"
OLMLX_RUN_MTP_INTEGRATION=1 uv run pytest tests/test_mtp_integration.py::test_mtp_moe_matches_greedy -v -s
```

Expected: PASS — output == greedy; acceptance clearly non-trivial. (The 35B at 4-bit ~18 GB fits in 64 GB with the small MoE head.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_mtp_integration.py
git commit -m "test(mtp): 35B-A3B MoE acceptance (flash_moe off)"
```

---

## Task 10: Documentation

**Files:**
- Modify: `CLAUDE.md` (the speculative-decoding section)

- [ ] **Step 1: Add an `mtp` bullet to the speculative-decoding strategy list**

Under the `OLMLX_SPECULATIVE_STRATEGY` strategies in `CLAUDE.md`, add:

```markdown
  - `mtp`: Qwen3.6's **native** multi-token-prediction head (`model_type:
    qwen3_5_mtp`) as the draft — pretrained and shipped (no `prepare` step).
    One full-attention Qwen3.6 layer with an MTP front-end
    (`pre_fc_norm_hidden`/`pre_fc_norm_embedding` → `fc(concat[h; e])`),
    final `norm`, borrowing the target's `embed_tokens`/`lm_head` via
    `bind()`. `MTPDecoder` composes the EAGLE-shared `_patch_model`
    hidden-capture + `GDNStateCapture` rollback + `verify_draft_greedy`;
    chains the **pre-final-norm** hidden (DeepSeek/Qwen MTP convention).
    Heads: `mlx-community/Qwen3.6-27B-MTP-4bit` (dense),
    `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` (MoE). Measured 27B acceptance
    <MEASURED> (vs 0.66 for the Qwen3.5-4B classic draft). **Incompatible
    with flash_moe** (same as eagle/dflash) — rejected at load; run the 35B
    target with flash_moe off.
```

Replace `<MEASURED>` with the Task 8 number.

- [ ] **Step 2: Run the full test suite (no regressions)**

Run: `uv run pytest tests/test_mtp_*.py tests/test_eagle.py tests/test_speculative.py tests/test_registry.py -q` and `uv run ruff check olmlx/engine/mtp olmlx/engine/registry.py olmlx/engine/model_manager.py olmlx/utils/metrics.py && uv run ruff format --check olmlx/engine/mtp`
Expected: PASS / clean (per the "run ruff before pushing" memory).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(mtp): document the mtp speculative strategy"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** scope (27B + 35B MoE) → Tasks 4/5/8/9; success bar (output==greedy & accept>0.66) → Task 8; dedicated MTPDecoder + shared helpers → Task 6; flash_moe exclusivity → Task 7; no-training → loader only; registry/loader wiring → Tasks 1/7; hidden-source risk → Task 8 probe.
- **No norm shift:** established empirically (norms already standard form) — Task 4 explicitly does not shift.
- **Type consistency:** `MTPConfig`/`MTPDraftModel`/`load_mtp_draft`/`MTPDecoder`/`_load_mtp_decoder` names are used identically across tasks; `__call__` returns `(logits|None, h_new)` everywhere; `block_size` default 3 in both draft config and decoder.
- **flash_moe exclusivity** is now concrete: a `_FLASH_MOE_INCOMPATIBLE_STRATEGIES` constant added to the two existing guard sites (model_manager.py ~3330, ~3461), not a new ad-hoc helper. Metrics label is a one-line dict entry in `metrics.py:172`.
- **Decoder coverage without the 27B:** Task 6 drives `MTPDecoder` against `test_eagle.py`'s synthetic target — real prefill/step/reset/stats coverage; the heavy model is reserved for the Task 8/9 acceptance gates.
- **One genuine runtime-determined value remains:** `<MEASURED>` (the 27B acceptance number) is filled in after Task 8 runs — it is a measured result, not undefined logic. The Task 8 probe is concrete (four enumerated wiring combinations + the pre-norm default reasoned from the DeepSeek/Qwen MTP convention).
- **Task 8 harness pointer:** `_generate` must be wired to whatever existing test loads a real model through `ModelManager` + `generate_chat` with a per-model `speculative_*` config — copy that setup. This is the only "follow the existing pattern" lookup left, and it is inherent (the repo's real-model harness, not new logic).
