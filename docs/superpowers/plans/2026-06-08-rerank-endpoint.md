# `/v1/rerank` Cross-Encoder Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Cohere-v2-compatible `POST /v1/rerank` endpoint backed by a native MLX XLM-RoBERTa cross-encoder, serving `bge-reranker-v2-m3` and `jina-reranker-v2-base-multilingual`.

**Architecture:** A new `reranker` model kind in `ModelManager`. A pure-MLX `XLMRobertaCrossEncoder` (absolute-position, post-LN BERT attention, first-token classification head) loads from HF safetensors via one of two weight-remap paths (standard HF layout for bge, fused-QKV flash layout for jina), picked by checkpoint-key inspection. A `generate_rerank` engine function tokenizes `(query, doc)` pairs with the model's `transformers` tokenizer (silent doc-side truncation), batch-scores them, sigmoids, sorts, and applies `top_n`. A thin router exposes `/v1/rerank` (+ `/rerank` alias).

**Tech Stack:** MLX / `mlx.nn`, `transformers.AutoTokenizer`, FastAPI, Pydantic v2, pytest. Reference spec: `docs/superpowers/specs/2026-06-08-rerank-endpoint-design.md`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `olmlx/engine/rerank/__init__.py` | Package marker, re-export `XLMRobertaCrossEncoder`, `RerankerConfig`, `load_cross_encoder` |
| `olmlx/engine/rerank/config.py` | `RerankerConfig` dataclass parsed from HF `config.json` |
| `olmlx/engine/rerank/model.py` | `XLMRobertaCrossEncoder` nn.Module (embeddings, encoder layers, classification head) |
| `olmlx/engine/rerank/weights.py` | Key-layout detection + two remap functions (standard / flash), `load_cross_encoder(path)` |
| `olmlx/schemas/rerank.py` | `RerankRequest`, `RerankResult`, `RerankResponse` |
| `olmlx/routers/rerank.py` | `POST /v1/rerank` + `POST /rerank` handlers |
| `olmlx/engine/inference.py` | Add `generate_rerank(...)` |
| `olmlx/engine/model_manager.py` | `_detect_model_kind` reranker branch; `_load_model` reranker branch; `LoadedModel.is_reranker`; guards |
| `olmlx/app.py` | Register the router |
| `CLAUDE.md` | Document endpoint + reranker kind |
| `tests/test_rerank_model.py` | Unit: encoder math, position offset, both remaps |
| `tests/test_rerank_schemas.py` | Unit: request/response validation |
| `tests/test_rerank_engine.py` | Unit: `generate_rerank` (mocked model) — tokenize/truncate/sort/top_n |
| `tests/test_routers_rerank.py` | Router contract (mocked engine) |
| `tests/live/test_rerank_real.py` | `real_model` live tests for bge + jina |

---

## Task 1: Reranker config dataclass

**Files:**
- Create: `olmlx/engine/rerank/__init__.py`
- Create: `olmlx/engine/rerank/config.py`
- Test: `tests/test_rerank_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rerank_model.py
from olmlx.engine.rerank.config import RerankerConfig


def test_rerankerconfig_from_dict_bge():
    raw = {
        "architectures": ["XLMRobertaForSequenceClassification"],
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_position_embeddings": 8194,
        "vocab_size": 250002,
        "type_vocab_size": 1,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 1,
        "hidden_act": "gelu",
        "id2label": {"0": "LABEL_0"},
    }
    cfg = RerankerConfig.from_dict(raw)
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 24
    assert cfg.num_attention_heads == 16
    assert cfg.intermediate_size == 4096
    assert cfg.max_position_embeddings == 8194
    assert cfg.vocab_size == 250002
    assert cfg.type_vocab_size == 1
    assert cfg.pad_token_id == 1
    assert cfg.num_labels == 1
    assert cfg.head_dim == 64


def test_rerankerconfig_num_labels_from_num_labels_key():
    raw = {
        "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
        "intermediate_size": 3072, "max_position_embeddings": 1026,
        "vocab_size": 250002, "type_vocab_size": 1, "layer_norm_eps": 1e-5,
        "pad_token_id": 1, "num_labels": 1,
    }
    cfg = RerankerConfig.from_dict(raw)
    assert cfg.num_labels == 1
    assert cfg.head_dim == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rerank_model.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'olmlx.engine.rerank'`

- [ ] **Step 3: Write minimal implementation**

```python
# olmlx/engine/rerank/__init__.py
"""Native MLX cross-encoder reranker (XLM-RoBERTa family). Issue #369."""

from olmlx.engine.rerank.config import RerankerConfig

__all__ = ["RerankerConfig"]
```

```python
# olmlx/engine/rerank/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RerankerConfig:
    """Sizing parameters parsed from an XLM-RoBERTa cross-encoder config.json."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    vocab_size: int
    type_vocab_size: int
    layer_norm_eps: float
    pad_token_id: int
    num_labels: int
    hidden_act: str = "gelu"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RerankerConfig:
        num_labels = raw.get("num_labels")
        if num_labels is None:
            id2label = raw.get("id2label") or {"0": "LABEL_0"}
            num_labels = len(id2label)
        return cls(
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            intermediate_size=int(raw["intermediate_size"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            vocab_size=int(raw["vocab_size"]),
            type_vocab_size=int(raw.get("type_vocab_size", 1)),
            layer_norm_eps=float(raw.get("layer_norm_eps", 1e-5)),
            pad_token_id=int(raw.get("pad_token_id", 1)),
            num_labels=int(num_labels),
            hidden_act=str(raw.get("hidden_act", "gelu")),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rerank_model.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/rerank/__init__.py olmlx/engine/rerank/config.py tests/test_rerank_model.py
git commit -m "feat(rerank): RerankerConfig parsed from HF config.json (#369)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Encoder module — embeddings + RoBERTa position offset

**Files:**
- Create: `olmlx/engine/rerank/model.py`
- Test: `tests/test_rerank_model.py`

This task builds only the embedding stage and the position-id helper so we can pin the
RoBERTa offset behavior before the full forward exists.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_rerank_model.py
import mlx.core as mx
from olmlx.engine.rerank.model import roberta_position_ids


def test_roberta_position_ids_offset_no_padding():
    # pad_token_id=1; a fully-real sequence -> positions start at 2
    input_ids = mx.array([[5, 6, 7, 8]])
    pos = roberta_position_ids(input_ids, pad_token_id=1)
    assert pos.tolist() == [[2, 3, 4, 5]]


def test_roberta_position_ids_offset_with_padding():
    # trailing pad tokens (id=1) keep position == pad_token_id (1)
    input_ids = mx.array([[5, 6, 1, 1]])
    pos = roberta_position_ids(input_ids, pad_token_id=1)
    assert pos.tolist() == [[2, 3, 1, 1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rerank_model.py::test_roberta_position_ids_offset_no_padding -q`
Expected: FAIL — `ImportError: cannot import name 'roberta_position_ids'`

- [ ] **Step 3: Write minimal implementation**

```python
# olmlx/engine/rerank/model.py
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.rerank.config import RerankerConfig


def roberta_position_ids(input_ids: mx.array, pad_token_id: int) -> mx.array:
    """Position ids with the RoBERTa offset.

    Mirrors transformers' ``create_position_ids_from_input_ids``:
    ``position_ids = cumsum(mask) * mask + pad_token_id`` where
    ``mask = (input_ids != pad_token_id)``. The first real token therefore
    gets position ``pad_token_id + 1`` (== 2 for XLM-RoBERTa), which is why
    the position-embedding table is sized ``max_seq + pad_token_id + 1``.
    """
    mask = (input_ids != pad_token_id).astype(mx.int32)
    incremental = mx.cumsum(mask, axis=1) * mask
    return incremental + pad_token_id


class XLMRobertaEmbeddings(nn.Module):
    def __init__(self, config: RerankerConfig):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array) -> mx.array:
        pos_ids = roberta_position_ids(input_ids, self.pad_token_id)
        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(pos_ids)
        # token_type_ids are all zero (type_vocab_size == 1); add row-0 bias.
        token_type = self.token_type_embeddings(mx.zeros_like(input_ids))
        return self.LayerNorm(words + positions + token_type)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rerank_model.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/rerank/model.py tests/test_rerank_model.py
git commit -m "feat(rerank): XLM-RoBERTa embeddings + RoBERTa position offset (#369)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Encoder layer + full forward + classification head

**Files:**
- Modify: `olmlx/engine/rerank/model.py`
- Test: `tests/test_rerank_model.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_rerank_model.py
from olmlx.engine.rerank.model import XLMRobertaCrossEncoder


def _tiny_config() -> RerankerConfig:
    return RerankerConfig(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=32, vocab_size=50,
        type_vocab_size=1, layer_norm_eps=1e-5, pad_token_id=1, num_labels=1,
    )


def test_cross_encoder_forward_shape():
    cfg = _tiny_config()
    model = XLMRobertaCrossEncoder(cfg)
    input_ids = mx.array([[5, 6, 7, 2], [5, 6, 1, 1]])  # 2nd row padded
    attention_mask = mx.array([[1, 1, 1, 1], [1, 1, 0, 0]])
    logits = model(input_ids, attention_mask)
    mx.eval(logits)
    assert logits.shape == (2, 1)


def test_cross_encoder_padding_invariance():
    # Scoring a sequence must not change when extra pad tokens are appended,
    # because the attention mask zeroes them out.
    cfg = _tiny_config()
    model = XLMRobertaCrossEncoder(cfg)
    short_ids = mx.array([[5, 6, 7, 2]])
    short_mask = mx.array([[1, 1, 1, 1]])
    padded_ids = mx.array([[5, 6, 7, 2, 1, 1]])
    padded_mask = mx.array([[1, 1, 1, 1, 0, 0]])
    a = model(short_ids, short_mask)
    b = model(padded_ids, padded_mask)
    mx.eval(a, b)
    assert abs(float(a[0, 0]) - float(b[0, 0])) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rerank_model.py::test_cross_encoder_forward_shape -q`
Expected: FAIL — `ImportError: cannot import name 'XLMRobertaCrossEncoder'`

- [ ] **Step 3: Write minimal implementation**

Append to `olmlx/engine/rerank/model.py`:

```python
class XLMRobertaSelfAttention(nn.Module):
    def __init__(self, config: RerankerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        h = config.hidden_size
        self.query = nn.Linear(h, h)
        self.key = nn.Linear(h, h)
        self.value = nn.Linear(h, h)

    def __call__(self, x: mx.array, additive_mask: mx.array) -> mx.array:
        b, s, _ = x.shape
        q = self.query(x).reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        scores = scores + additive_mask  # [b, 1, 1, s] broadcast
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v  # [b, heads, s, head_dim]
        return out.transpose(0, 2, 1, 3).reshape(b, s, -1)


class XLMRobertaLayer(nn.Module):
    def __init__(self, config: RerankerConfig):
        super().__init__()
        h = config.hidden_size
        self.attention_self = XLMRobertaSelfAttention(config)
        self.attention_output_dense = nn.Linear(h, h)
        self.attention_output_norm = nn.LayerNorm(h, eps=config.layer_norm_eps)
        self.intermediate_dense = nn.Linear(h, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, h)
        self.output_norm = nn.LayerNorm(h, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, additive_mask: mx.array) -> mx.array:
        attn = self.attention_self(x, additive_mask)
        x = self.attention_output_norm(self.attention_output_dense(attn) + x)
        inter = nn.gelu(self.intermediate_dense(x))
        x = self.output_norm(self.output_dense(inter) + x)
        return x


class XLMRobertaClassificationHead(nn.Module):
    def __init__(self, config: RerankerConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(self, features: mx.array) -> mx.array:
        x = features[:, 0, :]  # first token (<s> / CLS)
        x = mx.tanh(self.dense(x))
        return self.out_proj(x)


class XLMRobertaCrossEncoder(nn.Module):
    def __init__(self, config: RerankerConfig):
        super().__init__()
        self.config = config
        self.embeddings = XLMRobertaEmbeddings(config)
        self.layers = [XLMRobertaLayer(config) for _ in range(config.num_hidden_layers)]
        self.classifier = XLMRobertaClassificationHead(config)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        # additive mask: keep -> 0, pad -> -inf, shaped [b, 1, 1, s]
        additive = (1.0 - attention_mask.astype(mx.float32))[:, None, None, :] * -1e9
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, additive)
        return self.classifier(x)
```

Note on attribute naming: this module uses **flat** attribute names
(`attention_self`, `attention_output_dense`, …) so the weight-remap functions in Task 4
can target them directly without nested container modules. The remap is the single source
of truth mapping HF/flash checkpoint keys to these names.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rerank_model.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/rerank/model.py tests/test_rerank_model.py
git commit -m "feat(rerank): XLM-RoBERTa encoder layers + classification head (#369)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Weight loading — layout detection + two remaps

**Files:**
- Create: `olmlx/engine/rerank/weights.py`
- Modify: `olmlx/engine/rerank/__init__.py`
- Test: `tests/test_rerank_model.py`

`detect_layout(keys)` returns `"flash"` when any key contains `mixer.Wqkv` or `emb_ln`,
else `"standard"`. `remap_standard` / `remap_flash` translate a HF state dict (numpy
arrays) into a flat `{our_attr_path: mx.array}` dict matching the module in Task 3.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_rerank_model.py
import numpy as np
from olmlx.engine.rerank.weights import detect_layout, remap_standard, remap_flash


def test_detect_layout():
    assert detect_layout(["roberta.encoder.layer.0.attention.self.query.weight"]) == "standard"
    assert detect_layout(["roberta.encoder.layers.0.mixer.Wqkv.weight"]) == "flash"
    assert detect_layout(["roberta.emb_ln.weight"]) == "flash"


def test_remap_standard_keys():
    cfg = _tiny_config()
    h = cfg.hidden_size
    sd = {
        "roberta.embeddings.word_embeddings.weight": np.zeros((cfg.vocab_size, h), np.float32),
        "roberta.embeddings.position_embeddings.weight": np.zeros((cfg.max_position_embeddings, h), np.float32),
        "roberta.embeddings.token_type_embeddings.weight": np.zeros((cfg.type_vocab_size, h), np.float32),
        "roberta.embeddings.LayerNorm.weight": np.ones((h,), np.float32),
        "roberta.embeddings.LayerNorm.bias": np.zeros((h,), np.float32),
        "classifier.dense.weight": np.zeros((h, h), np.float32),
        "classifier.dense.bias": np.zeros((h,), np.float32),
        "classifier.out_proj.weight": np.zeros((1, h), np.float32),
        "classifier.out_proj.bias": np.zeros((1,), np.float32),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"roberta.encoder.layer.{i}."
        for proj in ("query", "key", "value"):
            sd[f"{p}attention.self.{proj}.weight"] = np.zeros((h, h), np.float32)
            sd[f"{p}attention.self.{proj}.bias"] = np.zeros((h,), np.float32)
        sd[f"{p}attention.output.dense.weight"] = np.zeros((h, h), np.float32)
        sd[f"{p}attention.output.dense.bias"] = np.zeros((h,), np.float32)
        sd[f"{p}attention.output.LayerNorm.weight"] = np.ones((h,), np.float32)
        sd[f"{p}attention.output.LayerNorm.bias"] = np.zeros((h,), np.float32)
        sd[f"{p}intermediate.dense.weight"] = np.zeros((cfg.intermediate_size, h), np.float32)
        sd[f"{p}intermediate.dense.bias"] = np.zeros((cfg.intermediate_size,), np.float32)
        sd[f"{p}output.dense.weight"] = np.zeros((h, cfg.intermediate_size), np.float32)
        sd[f"{p}output.dense.bias"] = np.zeros((h,), np.float32)
        sd[f"{p}output.LayerNorm.weight"] = np.ones((h,), np.float32)
        sd[f"{p}output.LayerNorm.bias"] = np.zeros((h,), np.float32)
    flat = remap_standard(sd, cfg)
    model = XLMRobertaCrossEncoder(cfg)
    model.load_weights(list(flat.items()))  # raises if any key/shape mismatches
    mx.eval(model.parameters())


def test_remap_flash_splits_fused_qkv():
    cfg = _tiny_config()
    h = cfg.hidden_size
    # Fused QKV: rows [q | k | v] each h.
    wqkv = np.arange(3 * h * h, dtype=np.float32).reshape(3 * h, h)
    bqkv = np.arange(3 * h, dtype=np.float32)
    sd = {
        "roberta.embeddings.word_embeddings.weight": np.zeros((cfg.vocab_size, h), np.float32),
        "roberta.embeddings.position_embeddings.weight": np.zeros((cfg.max_position_embeddings, h), np.float32),
        "roberta.embeddings.token_type_embeddings.weight": np.zeros((cfg.type_vocab_size, h), np.float32),
        "roberta.emb_ln.weight": np.ones((h,), np.float32),
        "roberta.emb_ln.bias": np.zeros((h,), np.float32),
        "classifier.dense.weight": np.zeros((h, h), np.float32),
        "classifier.dense.bias": np.zeros((h,), np.float32),
        "classifier.out_proj.weight": np.zeros((1, h), np.float32),
        "classifier.out_proj.bias": np.zeros((1,), np.float32),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"roberta.encoder.layers.{i}."
        sd[f"{p}mixer.Wqkv.weight"] = wqkv.copy()
        sd[f"{p}mixer.Wqkv.bias"] = bqkv.copy()
        sd[f"{p}mixer.out_proj.weight"] = np.zeros((h, h), np.float32)
        sd[f"{p}mixer.out_proj.bias"] = np.zeros((h,), np.float32)
        sd[f"{p}norm1.weight"] = np.ones((h,), np.float32)
        sd[f"{p}norm1.bias"] = np.zeros((h,), np.float32)
        sd[f"{p}mlp.fc1.weight"] = np.zeros((cfg.intermediate_size, h), np.float32)
        sd[f"{p}mlp.fc1.bias"] = np.zeros((cfg.intermediate_size,), np.float32)
        sd[f"{p}mlp.fc2.weight"] = np.zeros((h, cfg.intermediate_size), np.float32)
        sd[f"{p}mlp.fc2.bias"] = np.zeros((h,), np.float32)
        sd[f"{p}norm2.weight"] = np.ones((h,), np.float32)
        sd[f"{p}norm2.bias"] = np.zeros((h,), np.float32)
    flat = remap_flash(sd, cfg)
    # Q slice is the first h rows of the fused matrix.
    assert np.allclose(np.array(flat["layers.0.attention_self.query.weight"]), wqkv[:h])
    assert np.allclose(np.array(flat["layers.0.attention_self.key.weight"]), wqkv[h:2 * h])
    assert np.allclose(np.array(flat["layers.0.attention_self.value.weight"]), wqkv[2 * h:])
    model = XLMRobertaCrossEncoder(cfg)
    model.load_weights(list(flat.items()))
    mx.eval(model.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rerank_model.py::test_detect_layout -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'olmlx.engine.rerank.weights'`

- [ ] **Step 3: Write minimal implementation**

```python
# olmlx/engine/rerank/weights.py
from __future__ import annotations

import glob
import json
import os
from typing import Any

import mlx.core as mx
import numpy as np

from olmlx.engine.rerank.config import RerankerConfig
from olmlx.engine.rerank.model import XLMRobertaCrossEncoder


def detect_layout(keys: list[str]) -> str:
    for k in keys:
        if "mixer.Wqkv" in k or "emb_ln" in k:
            return "flash"
    return "standard"


def _emb_and_head(sd: dict[str, Any], emb_ln_prefix: str) -> dict[str, np.ndarray]:
    e = "roberta.embeddings."
    return {
        "embeddings.word_embeddings.weight": sd[f"{e}word_embeddings.weight"],
        "embeddings.position_embeddings.weight": sd[f"{e}position_embeddings.weight"],
        "embeddings.token_type_embeddings.weight": sd[f"{e}token_type_embeddings.weight"],
        "embeddings.LayerNorm.weight": sd[f"{emb_ln_prefix}.weight"],
        "embeddings.LayerNorm.bias": sd[f"{emb_ln_prefix}.bias"],
        "classifier.dense.weight": sd["classifier.dense.weight"],
        "classifier.dense.bias": sd["classifier.dense.bias"],
        "classifier.out_proj.weight": sd["classifier.out_proj.weight"],
        "classifier.out_proj.bias": sd["classifier.out_proj.bias"],
    }


def remap_standard(sd: dict[str, Any], cfg: RerankerConfig) -> dict[str, mx.array]:
    out = _emb_and_head(sd, "roberta.embeddings.LayerNorm")
    for i in range(cfg.num_hidden_layers):
        p = f"roberta.encoder.layer.{i}."
        q = f"layers.{i}."
        for proj in ("query", "key", "value"):
            out[f"{q}attention_self.{proj}.weight"] = sd[f"{p}attention.self.{proj}.weight"]
            out[f"{q}attention_self.{proj}.bias"] = sd[f"{p}attention.self.{proj}.bias"]
        out[f"{q}attention_output_dense.weight"] = sd[f"{p}attention.output.dense.weight"]
        out[f"{q}attention_output_dense.bias"] = sd[f"{p}attention.output.dense.bias"]
        out[f"{q}attention_output_norm.weight"] = sd[f"{p}attention.output.LayerNorm.weight"]
        out[f"{q}attention_output_norm.bias"] = sd[f"{p}attention.output.LayerNorm.bias"]
        out[f"{q}intermediate_dense.weight"] = sd[f"{p}intermediate.dense.weight"]
        out[f"{q}intermediate_dense.bias"] = sd[f"{p}intermediate.dense.bias"]
        out[f"{q}output_dense.weight"] = sd[f"{p}output.dense.weight"]
        out[f"{q}output_dense.bias"] = sd[f"{p}output.dense.bias"]
        out[f"{q}output_norm.weight"] = sd[f"{p}output.LayerNorm.weight"]
        out[f"{q}output_norm.bias"] = sd[f"{p}output.LayerNorm.bias"]
    return {k: mx.array(np.asarray(v)) for k, v in out.items()}


def remap_flash(sd: dict[str, Any], cfg: RerankerConfig) -> dict[str, mx.array]:
    h = cfg.hidden_size
    out = _emb_and_head(sd, "roberta.emb_ln")
    for i in range(cfg.num_hidden_layers):
        p = f"roberta.encoder.layers.{i}."
        q = f"layers.{i}."
        wqkv = np.asarray(sd[f"{p}mixer.Wqkv.weight"])
        bqkv = np.asarray(sd[f"{p}mixer.Wqkv.bias"])
        out[f"{q}attention_self.query.weight"] = wqkv[:h]
        out[f"{q}attention_self.key.weight"] = wqkv[h : 2 * h]
        out[f"{q}attention_self.value.weight"] = wqkv[2 * h :]
        out[f"{q}attention_self.query.bias"] = bqkv[:h]
        out[f"{q}attention_self.key.bias"] = bqkv[h : 2 * h]
        out[f"{q}attention_self.value.bias"] = bqkv[2 * h :]
        out[f"{q}attention_output_dense.weight"] = sd[f"{p}mixer.out_proj.weight"]
        out[f"{q}attention_output_dense.bias"] = sd[f"{p}mixer.out_proj.bias"]
        out[f"{q}attention_output_norm.weight"] = sd[f"{p}norm1.weight"]
        out[f"{q}attention_output_norm.bias"] = sd[f"{p}norm1.bias"]
        out[f"{q}intermediate_dense.weight"] = sd[f"{p}mlp.fc1.weight"]
        out[f"{q}intermediate_dense.bias"] = sd[f"{p}mlp.fc1.bias"]
        out[f"{q}output_dense.weight"] = sd[f"{p}mlp.fc2.weight"]
        out[f"{q}output_dense.bias"] = sd[f"{p}mlp.fc2.bias"]
        out[f"{q}output_norm.weight"] = sd[f"{p}norm2.weight"]
        out[f"{q}output_norm.bias"] = sd[f"{p}norm2.bias"]
    return {k: mx.array(np.asarray(v)) for k, v in out.items()}


def _load_state_dict(path: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors weights in {path}")
    sd: dict[str, np.ndarray] = {}
    for f in files:
        sd.update(mx.load(f))  # mx.load returns {name: mx.array}
    return sd


def load_cross_encoder(path: str) -> XLMRobertaCrossEncoder:
    """Build an XLMRobertaCrossEncoder from a local model directory."""
    with open(os.path.join(path, "config.json")) as fh:
        cfg = RerankerConfig.from_dict(json.load(fh))
    sd = _load_state_dict(path)
    layout = detect_layout(list(sd.keys()))
    flat = remap_flash(sd, cfg) if layout == "flash" else remap_standard(sd, cfg)
    model = XLMRobertaCrossEncoder(cfg)
    model.load_weights(list(flat.items()))
    model.eval()
    mx.eval(model.parameters())
    return model
```

Add to `olmlx/engine/rerank/__init__.py`:

```python
from olmlx.engine.rerank.model import XLMRobertaCrossEncoder
from olmlx.engine.rerank.weights import detect_layout, load_cross_encoder

__all__ = [
    "RerankerConfig",
    "XLMRobertaCrossEncoder",
    "detect_layout",
    "load_cross_encoder",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rerank_model.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/rerank/ tests/test_rerank_model.py
git commit -m "feat(rerank): weight loader with standard + flash (fused-QKV) remaps (#369)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Request/response schemas

**Files:**
- Create: `olmlx/schemas/rerank.py`
- Test: `tests/test_rerank_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rerank_schemas.py
import pytest
from pydantic import ValidationError

from olmlx.schemas.rerank import RerankRequest, RerankResponse, RerankResult


def test_rerank_request_defaults():
    req = RerankRequest(model="bge-reranker", query="q", documents=["a", "b"])
    assert req.top_n is None
    assert req.max_tokens_per_doc == 4096
    assert req.return_documents is False


def test_rerank_request_rejects_empty_documents():
    with pytest.raises(ValidationError):
        RerankRequest(model="m", query="q", documents=[])


def test_rerank_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        RerankRequest(model="m", query="  ", documents=["a"])


def test_rerank_request_rejects_nonpositive_top_n():
    with pytest.raises(ValidationError):
        RerankRequest(model="m", query="q", documents=["a"], top_n=0)


def test_rerank_response_shape():
    resp = RerankResponse(
        id="rerank-xyz",
        results=[RerankResult(index=1, relevance_score=0.9)],
        meta={"api_version": {"version": "2"}},
    )
    dumped = resp.model_dump()
    assert dumped["results"][0]["index"] == 1
    assert dumped["results"][0]["relevance_score"] == 0.9
    assert "document" not in dumped["results"][0] or dumped["results"][0]["document"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rerank_schemas.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'olmlx.schemas.rerank'`

- [ ] **Step 3: Write minimal implementation**

```python
# olmlx/schemas/rerank.py
from __future__ import annotations

from pydantic import BaseModel, field_validator

from olmlx.schemas.common import ModelName, validate_non_empty_text_input


class RerankRequest(BaseModel):
    model: ModelName
    query: str
    documents: list[str]
    top_n: int | None = None
    max_tokens_per_doc: int = 4096
    return_documents: bool = False
    keep_alive: int | str | None = None

    @field_validator("query")
    @classmethod
    def _query_non_empty(cls, v: str) -> str:
        return validate_non_empty_text_input(v, "query")

    @field_validator("documents")
    @classmethod
    def _documents_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("documents must be a non-empty list")
        return v

    @field_validator("top_n")
    @classmethod
    def _top_n_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("top_n must be a positive integer")
        return v


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str | None = None


class RerankResponse(BaseModel):
    id: str
    results: list[RerankResult]
    meta: dict = {}
```

Confirm `validate_non_empty_text_input` accepts a plain `str` (it is used for the
embeddings `prompt` field). If its signature differs, adapt the call.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rerank_schemas.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/schemas/rerank.py tests/test_rerank_schemas.py
git commit -m "feat(rerank): Cohere-v2 request/response schemas (#369)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `LoadedModel.is_reranker` + kind detection

**Files:**
- Modify: `olmlx/engine/model_manager.py`
- Test: `tests/test_rerank_manager.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rerank_manager.py
import json

from olmlx.engine.model_manager import ModelManager


def _write_config(tmp_path, store_dir, hf_path, config):
    d = store_dir / hf_path
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))
    return d


def test_detect_model_kind_reranker(tmp_path, monkeypatch):
    mgr = ModelManager()
    cfg = {
        "architectures": ["XLMRobertaForSequenceClassification"],
        "model_type": "xlm-roberta",
        "hidden_size": 1024, "num_hidden_layers": 24,
        "num_attention_heads": 16, "intermediate_size": 4096,
        "max_position_embeddings": 8194, "vocab_size": 250002,
        "type_vocab_size": 1, "pad_token_id": 1,
    }

    # Force _detect_model_kind to read this config (monkeypatch the loader's
    # config fetch). Simplest: patch _read_config_for_kind if present, else
    # write into a fake store. Use the store hook the manager already exposes.
    monkeypatch.setattr(
        mgr, "_load_config_json", lambda hf: cfg, raising=False
    )
    # Fallback: if the manager has no such helper, this test documents the
    # expected return given the architecture marker; adapt to the real seam
    # discovered while implementing (see Step 3).
    assert _kind_for_config(cfg) == "reranker"


def _kind_for_config(cfg):
    from olmlx.engine.model_manager import _is_cross_encoder_config

    return "reranker" if _is_cross_encoder_config(cfg) else "other"
```

Note: the exact monkeypatch seam depends on how `_detect_model_kind` reads config (it
reads `config.json` from the local store or HF hub — see `model_manager.py:1683`). The
testable unit extracted here is a pure helper `_is_cross_encoder_config(config: dict)`,
which `_detect_model_kind` calls. Test the helper directly; the branch wiring is covered
by the live test in Task 10.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rerank_manager.py -q`
Expected: FAIL — `ImportError: cannot import name '_is_cross_encoder_config'`

- [ ] **Step 3: Write minimal implementation**

In `olmlx/engine/model_manager.py`, add a module-level helper near the other detection
code:

```python
def _is_cross_encoder_config(config: dict) -> bool:
    """True for an XLM-RoBERTa-family sequence-classification reranker."""
    archs = config.get("architectures") or []
    return any(
        isinstance(a, str) and a.endswith("ForSequenceClassification") for a in archs
    )
```

In `_detect_model_kind`, after the whisper check and before the VLM/text resolution
(around `model_manager.py:1718`, right after the `if not model_type: return "unknown"`
guard — but the reranker check must run even when `model_type` is set, so place it
immediately after the whisper block at ~line 1716):

```python
        if _is_cross_encoder_config(config):
            return "reranker"
```

Add the field to `LoadedModel` (after `is_whisper`, `model_manager.py:479`):

```python
    is_reranker: bool = False
```

At the `LoadedModel(...)` construction site (`model_manager.py:1456`), add:

```python
                        is_reranker=(_model_kind == "reranker"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rerank_manager.py -q`
Expected: PASS

- [ ] **Step 5: Run the broader manager suite for regressions**

Run: `uv run pytest tests/test_model_manager*.py -q`
Expected: PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add olmlx/engine/model_manager.py tests/test_rerank_manager.py
git commit -m "feat(rerank): detect reranker kind + LoadedModel.is_reranker (#369)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: `_load_model` reranker branch + LLM-only-path guards

**Files:**
- Modify: `olmlx/engine/model_manager.py`
- Test: covered by Task 10 live test (loading requires real weights); add a guard unit test

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_rerank_manager.py
from olmlx.engine.model_manager import LoadedModel


def test_is_reranker_guards_skip_llm_paths():
    # A reranker LoadedModel must report is_reranker and must not be treated as
    # an embedding/LLM target by the kv-quant / prompt-cache guards.
    lm = LoadedModel(name="bge", hf_path="bge", model=object(), tokenizer=object(),
                     is_reranker=True)
    assert lm.is_reranker is True
    # Mirror the is_whisper guard contract: rerankers carry no KV cache.
    assert lm.kv_cache_quant is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rerank_manager.py::test_is_reranker_guards_skip_llm_paths -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'is_reranker'`
(if Task 6's field was not yet present) OR PASS the constructor but the test asserts the
guard contract — if it fails on the kv_cache_quant assertion, proceed to Step 3.

- [ ] **Step 3: Write minimal implementation**

In `_load_model`, in the kind-dispatch block (after the `kind == "vlm"` branch, near
`model_manager.py:3512`), add a reranker branch:

```python
        if kind == "reranker":
            from transformers import AutoTokenizer

            from olmlx.engine.rerank import load_cross_encoder

            model = load_cross_encoder(load_path)
            tokenizer = AutoTokenizer.from_pretrained(load_path)
            # No chat template, no speculative, no KV cache for a cross-encoder.
            return model, tokenizer, False, TemplateCaps(), None
```

In `ensure_loaded`, where `_model_kind` gates whisper KV-quant (`model_manager.py:1100`),
extend the guard so rerankers also skip KV-cache quant / spectral calibration:

```python
                _model_kind = self._detect_model_kind(hf_path)
                if _model_kind in ("whisper", "reranker"):
                    kv_cache_quant = None
```

Audit the LLM-only paths that branch on `is_whisper` and add `or lm.is_reranker` where a
cross-encoder must be excluded (it has no KV cache and no generation):
`model_manager.py:809`, `:824`, `:2100`. For each, change `if lm.is_whisper:` /
`if not lm.is_whisper:` to include `is_reranker` with the same polarity as whisper.
Verify by reading each site — the intent is "skip prompt-cache / KV-cache / generation
bookkeeping for non-LLM models".

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rerank_manager.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/model_manager.py tests/test_rerank_manager.py
git commit -m "feat(rerank): _load_model reranker branch + is_reranker guards (#369)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: `generate_rerank` engine function

**Files:**
- Modify: `olmlx/engine/inference.py`
- Test: `tests/test_rerank_engine.py`

`generate_rerank` mirrors `generate_embeddings` (inference lock, pinned ref, trailing
`mx.synchronize()`), but scores `(query, doc)` pairs and returns a Cohere-shaped dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rerank_engine.py
import mlx.core as mx
import pytest

from olmlx.engine.inference import _build_rerank_results, _score_pairs


class _FakeTokenizer:
    model_max_length = 512

    def __call__(self, query, documents, truncation, max_length, padding,
                 return_tensors=None):
        # Return numpy-like dict; one row per document.
        import numpy as np
        n = len(documents)
        ids = np.ones((n, 4), dtype=np.int64) * 5
        mask = np.ones((n, 4), dtype=np.int64)
        return {"input_ids": ids, "attention_mask": mask}


class _FakeModel:
    def __init__(self, scores):
        self._scores = scores
        self.calls = 0

    def __call__(self, input_ids, attention_mask):
        b = input_ids.shape[0]
        out = self._scores[self.calls : self.calls + b]
        self.calls += b
        return mx.array(out).reshape(b, 1)


def test_score_pairs_sigmoid_and_order():
    # Raw logits -> sigmoid; higher logit -> higher score.
    model = _FakeModel([2.0, -2.0, 0.0])
    scores = _score_pairs(
        model, _FakeTokenizer(), "q", ["a", "b", "c"],
        max_tokens_per_doc=256, batch_size=8,
    )
    assert len(scores) == 3
    assert scores[0] > scores[2] > scores[1]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_build_rerank_results_top_n_and_sort():
    results = _build_rerank_results(
        scores=[0.1, 0.9, 0.5], documents=["a", "b", "c"],
        top_n=2, return_documents=False,
    )
    assert [r["index"] for r in results] == [1, 2]
    assert results[0]["relevance_score"] == 0.9
    assert "document" not in results[0] or results[0]["document"] is None


def test_build_rerank_results_return_documents():
    results = _build_rerank_results(
        scores=[0.1, 0.9], documents=["a", "b"], top_n=None,
        return_documents=True,
    )
    assert results[0]["document"] == "b"


def test_build_rerank_results_top_n_clamped():
    results = _build_rerank_results(
        scores=[0.1, 0.9], documents=["a", "b"], top_n=10,
        return_documents=False,
    )
    assert len(results) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rerank_engine.py -q`
Expected: FAIL — `ImportError: cannot import name '_score_pairs'`

- [ ] **Step 3: Write minimal implementation**

Add to `olmlx/engine/inference.py` (near `generate_embeddings`):

```python
def _score_pairs(
    model,
    tokenizer,
    query: str,
    documents: list[str],
    *,
    max_tokens_per_doc: int,
    batch_size: int = 32,
) -> list[float]:
    """Tokenize (query, doc) pairs, batch-score, sigmoid -> [0,1] scores."""
    import numpy as np

    model_max = int(getattr(tokenizer, "model_max_length", 512) or 512)
    # transformers sometimes reports a sentinel (very large) model_max_length.
    if model_max > 100_000:
        model_max = 512
    max_len = min(max_tokens_per_doc, model_max)

    scores: list[float] = []
    for start in range(0, len(documents), batch_size):
        chunk = documents[start : start + batch_size]
        enc = tokenizer(
            [query] * len(chunk),
            chunk,
            truncation="only_second",
            max_length=max_len,
            padding=True,
            return_tensors="np",
        )
        input_ids = mx.array(np.asarray(enc["input_ids"]).astype(np.int32))
        attn = mx.array(np.asarray(enc["attention_mask"]).astype(np.int32))
        logits = model(input_ids, attn)  # [b, 1]
        probs = mx.sigmoid(logits[:, 0])
        mx.eval(probs)
        scores.extend(float(x) for x in probs.tolist())
    return scores


def _build_rerank_results(
    *,
    scores: list[float],
    documents: list[str],
    top_n: int | None,
    return_documents: bool,
) -> list[dict]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if top_n is not None:
        order = order[: min(top_n, len(order))]
    results: list[dict] = []
    for idx in order:
        item: dict = {"index": idx, "relevance_score": scores[idx]}
        if return_documents:
            item["document"] = documents[idx]
        results.append(item)
    return results


async def generate_rerank(
    manager: ModelManager,
    model_name: str,
    query: str,
    documents: list[str],
    *,
    top_n: int | None = None,
    max_tokens_per_doc: int = 4096,
    return_documents: bool = False,
    keep_alive: int | str | None = None,
) -> dict:
    """Score documents against a query with a cross-encoder reranker (#369)."""
    lm = await manager.ensure_loaded(model_name, keep_alive, pin=True)
    try:
        if not getattr(lm, "is_reranker", False):
            raise ValueError(
                f"Model '{model_name}' is not a reranker. /v1/rerank requires "
                "an XLM-RoBERTa cross-encoder (e.g. bge-reranker-v2-m3)."
            )
        async with _inference_locked(
            lm.inference_queue_timeout, sync_mode=lm.sync_mode
        ):
            with (
                _tracing.span(
                    "inference",
                    model=lm.name,
                    surface=surface_var.get(),
                    strategy="none",
                ),
                _inference_ref(lm, keep_alive=keep_alive, adopt=True),
            ):
                scores = _score_pairs(
                    lm.model,
                    lm.text_tokenizer,
                    query,
                    documents,
                    max_tokens_per_doc=max_tokens_per_doc,
                )
                results = _build_rerank_results(
                    scores=scores,
                    documents=documents,
                    top_n=top_n,
                    return_documents=return_documents,
                )
                # Same sync_mode="none" barrier rationale as generate_embeddings.
                try:
                    mx.synchronize()
                except Exception:
                    logger.warning(
                        "rerank post-compute sync failed — next inference may crash",
                        exc_info=True,
                    )
                return {"results": results}
    finally:
        lm.release_ref()
```

Confirm `lm.text_tokenizer` returns the `transformers` tokenizer for a reranker (it is set
from the `_load_model` return). If reranker tokenizers surface under a different attribute,
use `lm.tokenizer`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rerank_engine.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/inference.py tests/test_rerank_engine.py
git commit -m "feat(rerank): generate_rerank engine function (#369)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Router + app registration

**Files:**
- Create: `olmlx/routers/rerank.py`
- Modify: `olmlx/app.py:22-35` (imports), `:399` (registration)
- Test: `tests/test_routers_rerank.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_routers_rerank.py
from unittest.mock import AsyncMock, patch

import pytest


class TestRerankRouter:
    @pytest.mark.asyncio
    async def test_rerank_v1(self, app_client):
        with patch(
            "olmlx.routers.rerank.generate_rerank", new_callable=AsyncMock
        ) as mock:
            mock.return_value = {
                "results": [
                    {"index": 1, "relevance_score": 0.92},
                    {"index": 0, "relevance_score": 0.10},
                ]
            }
            resp = await app_client.post(
                "/v1/rerank",
                json={
                    "model": "bge-reranker",
                    "query": "what is mlx",
                    "documents": ["unrelated", "mlx is an array framework"],
                    "top_n": 2,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["index"] == 1
        assert data["results"][0]["relevance_score"] == 0.92
        assert data["id"].startswith("rerank-")
        assert data["meta"]["api_version"]["version"] == "2"

    @pytest.mark.asyncio
    async def test_rerank_alias(self, app_client):
        with patch(
            "olmlx.routers.rerank.generate_rerank", new_callable=AsyncMock
        ) as mock:
            mock.return_value = {"results": []}
            resp = await app_client.post(
                "/rerank",
                json={"model": "m", "query": "q", "documents": ["a"]},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_rerank_rejects_empty_documents(self, app_client):
        resp = await app_client.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": []},
        )
        assert resp.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_routers_rerank.py -q`
Expected: FAIL — 404 (route not registered) / import error

- [ ] **Step 3: Write minimal implementation**

```python
# olmlx/routers/rerank.py
import uuid

from fastapi import APIRouter, Request

from olmlx.engine.inference import generate_rerank
from olmlx.schemas.rerank import RerankRequest, RerankResponse, RerankResult

router = APIRouter()


async def _rerank(req: RerankRequest, request: Request) -> RerankResponse:
    manager = request.app.state.model_manager
    out = await generate_rerank(
        manager,
        req.model,
        req.query,
        req.documents,
        top_n=req.top_n,
        max_tokens_per_doc=req.max_tokens_per_doc,
        return_documents=req.return_documents,
        keep_alive=req.keep_alive,
    )
    return RerankResponse(
        id=f"rerank-{uuid.uuid4().hex}",
        results=[RerankResult(**r) for r in out["results"]],
        meta={"api_version": {"version": "2"}},
    )


@router.post("/v1/rerank", response_model=RerankResponse)
async def rerank_v1(req: RerankRequest, request: Request) -> RerankResponse:
    return await _rerank(req, request)


@router.post("/rerank", response_model=RerankResponse)
async def rerank_alias(req: RerankRequest, request: Request) -> RerankResponse:
    return await _rerank(req, request)
```

In `olmlx/app.py`, add `rerank` to the routers import block (`app.py:22`) and register it
alongside the others (after `embed.router`, `app.py:399`):

```python
    app.include_router(rerank.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_routers_rerank.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the app-wiring test**

Run: `uv run pytest tests/test_app.py -q`
Expected: PASS (router registration doesn't break app construction)

- [ ] **Step 6: Commit**

```bash
git add olmlx/routers/rerank.py olmlx/app.py tests/test_routers_rerank.py
git commit -m "feat(rerank): /v1/rerank router (+ /rerank alias) (#369)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Live tests against real models (bge + jina)

**Files:**
- Create: `tests/live/test_rerank_real.py`
- Reference: `tests/live/` patterns and the `real_model` marker (`tests/integration/conftest.py:147-161`)

These download real weights. Mark `real_model` and `slow`; keep them outside
`tests/integration/` so the autouse MLX mock doesn't apply (per CLAUDE.md).

- [ ] **Step 1: Write the discovery test (asserts real checkpoint key layout)**

```python
# tests/live/test_rerank_real.py
import pytest

pytestmark = [pytest.mark.real_model, pytest.mark.slow]

BGE = "BAAI/bge-reranker-v2-m3"
JINA = "jinaai/jina-reranker-v2-base-multilingual"


def _local_dir(repo: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo)


@pytest.mark.parametrize("repo,expected", [(BGE, "standard"), (JINA, "flash")])
def test_checkpoint_layout_matches_expectation(repo, expected):
    import glob
    import os

    import mlx.core as mx

    from olmlx.engine.rerank.weights import detect_layout

    path = _local_dir(repo)
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    keys: list[str] = []
    for f in files:
        keys.extend(mx.load(f).keys())
    assert detect_layout(keys) == expected, (
        f"{repo}: detected {detect_layout(keys)} (sample keys: {keys[:8]})"
    )
```

- [ ] **Step 2: Run the discovery test**

Run: `uv run pytest tests/live/test_rerank_real.py -m real_model -q`
Expected: PASS. **If jina's keys differ from the assumed flash layout**, the assertion
prints sample keys — update `detect_layout` / `remap_flash` in Task 4 to the real strings,
re-run, then continue.

- [ ] **Step 3: Write the ranking + reference-parity test**

```python
# add to tests/live/test_rerank_real.py
@pytest.mark.parametrize("repo", [BGE, JINA])
def test_load_and_rank(repo):
    import mlx.core as mx
    import numpy as np
    from transformers import AutoTokenizer

    from olmlx.engine.inference import _build_rerank_results, _score_pairs
    from olmlx.engine.rerank.weights import load_cross_encoder

    path = _local_dir(repo)
    model = load_cross_encoder(path)
    tok = AutoTokenizer.from_pretrained(path)

    query = "What is the capital of France?"
    docs = [
        "The capital of France is Paris.",
        "Bananas are a good source of potassium.",
    ]
    scores = _score_pairs(model, tok, query, docs, max_tokens_per_doc=512)
    results = _build_rerank_results(
        scores=scores, documents=docs, top_n=None, return_documents=False
    )
    assert results[0]["index"] == 0  # the relevant doc ranks first
    assert scores[0] > scores[1]


def test_batch_of_100_documents():
    from transformers import AutoTokenizer

    from olmlx.engine.inference import _score_pairs
    from olmlx.engine.rerank.weights import load_cross_encoder

    path = _local_dir(BGE)
    model = load_cross_encoder(path)
    tok = AutoTokenizer.from_pretrained(path)
    docs = [f"document number {i} about various topics" for i in range(100)]
    scores = _score_pairs(model, tok, "find the relevant document", docs,
                          max_tokens_per_doc=256)
    assert len(scores) == 100
```

- [ ] **Step 4: Add a transformers reference-parity check (bge only)**

```python
# add to tests/live/test_rerank_real.py
def test_parity_against_transformers_reference():
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from olmlx.engine.inference import _score_pairs
    from olmlx.engine.rerank.weights import load_cross_encoder

    pytest.importorskip("torch")
    path = _local_dir(BGE)
    tok = AutoTokenizer.from_pretrained(path)
    query = "What is MLX?"
    docs = ["MLX is an array framework for Apple silicon.", "I like sandwiches."]

    ours = _score_pairs(load_cross_encoder(path), tok, query, docs,
                        max_tokens_per_doc=512)

    ref_model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    with torch.no_grad():
        enc = tok([query, query], docs, padding=True, truncation="only_second",
                  max_length=512, return_tensors="pt")
        ref_logits = ref_model(**enc).logits.squeeze(-1)
        ref = torch.sigmoid(ref_logits).tolist()

    assert np.allclose(np.array(ours), np.array(ref), atol=1e-2), (ours, ref)
```

`torch`/`transformers` reference model is `importorskip`-guarded so the suite skips
cleanly where torch isn't installed.

- [ ] **Step 5: Run the live suite**

Run: `uv run pytest tests/live/test_rerank_real.py -m real_model -q`
Expected: PASS (or SKIP for the torch-parity test if torch absent). Investigate any
ranking failure before proceeding.

- [ ] **Step 6: Commit**

```bash
git add tests/live/test_rerank_real.py
git commit -m "test(rerank): live bge + jina ranking, layout, and parity checks (#369)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: Documentation + final verification

**Files:**
- Modify: `CLAUDE.md`
- Reference: full test suite

- [ ] **Step 1: Document the endpoint in CLAUDE.md**

Add under "Project Structure" (routers list): a line for `routers/rerank.py`. Add a bullet
under "Key Design Decisions":

```markdown
- **Rerank endpoint** (`routers/rerank.py`, #369): `POST /v1/rerank` (+ `/rerank` alias),
  Cohere-v2 shape. Cross-encoder rerankers are a first-class `ModelManager` kind
  (`reranker`): `_detect_model_kind` matches an `architectures` entry ending in
  `ForSequenceClassification`; `LoadedModel.is_reranker` guards the LLM-only prompt-cache /
  KV-quant / generation paths (like `is_whisper`). The encoder is a native MLX
  `XLMRobertaCrossEncoder` (`engine/rerank/`): absolute-position, post-LN BERT attention,
  first-token classification head, config-driven sizing. Two checkpoint layouts load into
  the same module via key-inspection remaps — standard HF (`attention.self.{query,key,
  value}`, bge) and flash (`mixer.Wqkv` fused QKV + `emb_ln`/`norm1`/`norm2`, jina).
  `generate_rerank` tokenizes `(query, doc)` pairs (`transformers` tokenizer,
  `truncation="only_second"` — silent doc-side truncation), batch-scores, `sigmoid` →
  `[0,1]`, sorts, applies `top_n`. No KV cache, speculative, distributed, or streaming for
  rerankers. Live coverage: `tests/live/test_rerank_real.py` (real_model, bge + jina).
```

- [ ] **Step 2: Run the full focused test set**

Run:
```bash
uv run pytest tests/test_rerank_model.py tests/test_rerank_schemas.py \
  tests/test_rerank_engine.py tests/test_routers_rerank.py \
  tests/test_rerank_manager.py tests/test_app.py -q
```
Expected: PASS (all)

- [ ] **Step 3: Lint**

Run: `uv run ruff check olmlx/engine/rerank olmlx/routers/rerank.py olmlx/schemas/rerank.py olmlx/engine/inference.py olmlx/engine/model_manager.py tests/test_rerank_*.py tests/live/test_rerank_real.py && uv run ruff format --check olmlx/engine/rerank olmlx/routers/rerank.py olmlx/schemas/rerank.py`
Expected: no errors (run `ruff format` to fix formatting if the check fails)

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(rerank): document /v1/rerank endpoint and reranker kind (#369)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5: Push and open PR** (only when the user asks — per project git policy)

```bash
git push -u origin feat/rerank-endpoint
gh pr create --fill --base main
```

---

## Self-Review notes (for the implementer)

- **Spec coverage**: kind detection (T6), native MLX encoder (T2/T3), two weight layouts
  (T4), loader + guards (T6/T7), `generate_rerank` truncation/sort/top_n (T8), schemas
  (T5), router + alias (T9), tests incl. live bge+jina + parity (T10), docs (T11). All
  spec sections map to a task.
- **Uncertain seam — jina flash key strings**: the exact `mixer.Wqkv`/`norm1`/`emb_ln`
  strings are inferred from flash-attn conventions. Task 10 Step 1/2 is a hard
  validation gate that prints the real keys and forces a Task-4 fix before the rest of
  the live suite runs. Do not skip it.
- **Uncertain seam — `_detect_model_kind` config-read path**: Task 6 extracts a pure
  helper (`_is_cross_encoder_config`) to keep the unit test independent of the
  store/hub config-fetch plumbing; wire the helper into the real `_detect_model_kind`
  branch and rely on Task 10 for end-to-end coverage.
- **`text_tokenizer` vs `tokenizer`**: confirmed steps in T8 to verify the attribute the
  reranker tokenizer lands on; adapt if needed.
- **GELU variant**: bge/jina use `hidden_act: "gelu"` (exact erf GELU). `nn.gelu` is the
  exact form — correct. (If a future model specifies `gelu_new`/`gelu_pytorch_tanh`, branch
  on `cfg.hidden_act`.)
