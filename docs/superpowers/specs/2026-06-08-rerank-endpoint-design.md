# `/v1/rerank` — Cross-encoder reranking endpoint

**Issue:** #369
**Date:** 2026-06-08
**Status:** Approved, ready for implementation planning

## Why

The standard RAG pipeline is embed → retrieve top-k → **rerank** → LLM. olmlx already
serves embeddings and manages model lifecycle; cross-encoder rerankers slot in as a
natural pairing. Without rerank, callers either skip it (retrieval-quality drop) or
run a second tool.

## Decisions (locked)

| Decision | Choice |
|----------|--------|
| Model backend | **Native MLX XLM-RoBERTa** — no new runtime deps, pure MLX/Metal |
| Model coverage | **Generic config-driven** cross-encoder (absolute *and* rotary position embeddings) to cover bge + jina + future XLM-RoBERTa-family rerankers |
| API shape | **Cohere v2** rerank contract (`/v1/rerank`), drop-in for the Cohere SDK via `base_url` |
| Long-document handling | **Silent truncation** (Cohere default) — truncate document side, preserve full query |

## Acceptance criteria (from issue)

- `bge-reranker-v2-m3` and `jina-reranker-v2-base-multilingual` load and serve.
- Compatible with the Cohere SDK request shape.
- A batch of 100 documents scores on M3 in reasonable time.

## Architecture

### 1. New model kind: `reranker`

Extend `ModelManager._detect_model_kind` to return `"reranker"` when `config.json`'s
`architectures` entry ends in `ForSequenceClassification` (XLM-RoBERTa cross-encoder
family). Checked **before** the generic text path and after whisper/VLM detection.
Generic discriminator rather than a hardcoded model allowlist, per the generic
config-driven decision.

### 2. Native MLX encoder — `olmlx/engine/rerank/model.py`

`XLMRobertaCrossEncoder(nn.Module)`, fully config-driven:

- **Embeddings**: word + position + token_type, followed by LayerNorm. The config's
  `position_embedding_type` selects **absolute** (bge) vs **rotary** (jina) — the single
  branch that covers both target architectures.
- **Encoder**: `num_hidden_layers` post-LN transformer blocks (self-attention + GELU
  FFN), sized from config (`hidden_size`, `num_attention_heads`, `intermediate_size`).
- **Classification head**: project the first token's hidden state to a single relevance
  logit (`num_labels == 1`).
- **Weights**: load from HF safetensors via a key-remap table (HF `xlm-roberta.*` naming
  → our module names). Forward signature: `(input_ids, attention_mask) -> [batch, 1]`
  logits.

### 3. Loader + `LoadedModel`

- Add `is_reranker: bool = False` to `LoadedModel` (mirrors `is_whisper`).
- New `kind == "reranker"` branch in `_load_model`: build the encoder from config, load
  weights, return `(model, tokenizer, is_vlm=False, TemplateCaps(), decoder=None)` and
  set `is_reranker=True` at `LoadedModel` construction.
- Tokenizer loaded via `transformers.AutoTokenizer` (transformers + sentencepiece already
  present). XLM-RoBERTa uses a SentencePiece tokenizer.
- `is_reranker` guards the LLM-only paths (prompt cache, KV-quant, speculative,
  embeddings) exactly as `is_whisper` does today.
- `ensure_loaded` / keep-alive / LRU eviction / memory check all work unchanged — the
  reranker is a managed model like any other.

### 4. Engine: `generate_rerank(...)` — `olmlx/engine/inference.py`

```
async def generate_rerank(
    manager, model_name, query, documents,
    *, top_n=None, max_tokens_per_doc=4096, return_documents=False,
    keep_alive=None,
) -> dict
```

Runs under `_inference_locked` + a tracing `inference` span (strategy `"none"`), pinned
ref, mirroring `generate_embeddings`:

1. Reject non-reranker models with a clear `ValueError` (→ HTTP 400), like
   `generate_transcription` does for non-whisper.
2. For each `(query, doc)` pair, encode with
   `tokenizer(query, doc, truncation="only_second",
   max_length=min(max_tokens_per_doc, model_max))` → silent doc-side truncation,
   full query preserved.
3. **Micro-batch**: pad to the longest sequence in the batch, build the attention mask,
   run the encoder forward in bounded batches (cap peak activation memory), collect
   logits, apply `sigmoid` → `relevance_score ∈ [0, 1]`.
4. Sort by score descending, apply `top_n` (clamp to `len(documents)`), build the
   results list. Include `document` text only when `return_documents=True`.
5. Trailing `mx.synchronize()` barrier — same `sync_mode="none"` lock-boundary concern
   documented in `generate_embeddings`.

Edge cases: empty `documents` → empty `results`; `top_n` larger than the list → clamp;
`top_n <= 0` → 422 at the schema layer.

### 5. Schemas — `olmlx/schemas/rerank.py`

- `RerankRequest`: `model: ModelName`, `query: str`, `documents: list[str]`,
  `top_n: int | None = None`, `max_tokens_per_doc: int = 4096`,
  `return_documents: bool = False`, `keep_alive: int | str | None = None`.
  Validators: non-empty `query`, non-empty `documents`, `top_n` positive when present.
- `RerankResult`: `index: int`, `relevance_score: float`, `document: str | None = None`.
- `RerankResponse`: `id: str`, `results: list[RerankResult]`,
  `meta: dict` (`{"api_version": {"version": "2"}}`; billed-units omitted for a local tool).

### 6. Router — `olmlx/routers/rerank.py`

`POST /v1/rerank` → `generate_rerank` → `RerankResponse`. Plus a `POST /rerank` alias for
Jina/TEI-style clients (cheap, same handler). Registered in `app.py` alongside the other
routers.

### 7. Tests (TDD — write failing first)

- **Unit (model)**: encoder forward parity against a tiny saved fixture; weight-remap
  correctness; absolute vs rotary position-embedding branch.
- **Unit (engine)**: pair tokenization + `only_second` truncation; sigmoid scoring; sort +
  `top_n` clamp; empty-documents → empty results; non-reranker model → ValueError.
- **Router**: Cohere v2 response shape; `return_documents` toggle; 422 on malformed
  request (empty documents, non-positive `top_n`).
- **Live (`real_model`, outside `tests/integration/` to dodge the autouse MLX mock)**:
  `bge-reranker-v2-m3` ranks a known-relevant document above an irrelevant one; smoke-test
  scoring a batch of 100 documents.

## Out of scope (v1)

- Document objects / `rank_fields` (string documents only).
- Multi-label classification heads (`num_labels > 1`).
- Streaming, distributed, and flash paths for rerankers.
- Billed-units accounting in `meta`.

## Files

- new `olmlx/engine/rerank/__init__.py`, `olmlx/engine/rerank/model.py`
- new `olmlx/schemas/rerank.py`
- new `olmlx/routers/rerank.py`
- `olmlx/engine/inference.py` — add `generate_rerank`
- `olmlx/engine/model_manager.py` — `_detect_model_kind` reranker branch, `_load_model`
  reranker branch, `LoadedModel.is_reranker`, `is_reranker` guards on LLM-only paths
- `olmlx/app.py` — register the router
- `CLAUDE.md` — document the endpoint + reranker model kind under Key Design Decisions
- tests under `tests/` (+ a live test outside `tests/integration/`)
