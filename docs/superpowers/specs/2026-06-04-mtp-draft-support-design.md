# MTP draft-head speculative decoding — design

**Date:** 2026-06-04
**Status:** Approved (brainstorm), pending implementation plan
**Branch:** `feat/mtp-speculative`

## Summary

Add a new `mtp` speculative-decoding strategy that loads Qwen3.6's **pretrained,
shipped** multi-token-prediction (MTP) head (`model_type: qwen3_5_mtp`) as the
draft for a matching Qwen3.6 target. The MTP head is the model's *native*,
trained-in-distribution draft, so it should reach higher acceptance than any
cross-family standalone draft.

Unlike the existing `dflash` / `eagle` strategies, **there is no training step** —
the head ships as a small (~260 MB at 4-bit) HuggingFace repo and is loaded at
inference time only.

### Targets
- **Qwen3.6-27B (dense)** — primary. `unsloth/Qwen3.6-27B-MLX-8bit` target +
  `mlx-community/Qwen3.6-27B-MTP-4bit` head.
- **Qwen3.6-35B-A3B (MoE)** — `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` head; the
  MTP layer itself is MoE (router `gate` + `shared_expert` + routed experts).

### Success criteria (hard pass/fail)
1. Speculative output is **token-identical** to non-speculative greedy decoding
   (guaranteed by greedy verification; asserted as a regression guard).
2. **Acceptance rate > 0.66** — strictly above the `mlx-community/Qwen3.5-4B-OptiQ-4bit`
   classic draft measured on this target (0.659, 3.64 tok/step). The native head
   is expected to reach ~0.7–0.8+. This is the real signal that the head is wired
   correctly: a mis-wired head still emits correct text but collapses to ~0%
   acceptance (cf. the existing dflash ~2% / eagle ~6% results).

## Background: the MTP head structure

Inspected from `mlx-community/Qwen3.6-27B-MTP-4bit` weights/config:

```
config.json:
  model_type: qwen3_5_mtp
  block_size: 3                 # drafts up to 3 tokens per verify
  text_config: { ...full Qwen3.6 text config: hidden 5120, head_dim 256,
                 num_attention_heads 24, num_key_value_heads 4,
                 attn_output_gate true, partial_rotary_factor 0.25,
                 mrope_section [11,11,10], full_attention_interval 4,
                 mtp_num_hidden_layers 1, vocab_size 248320,
                 tie_word_embeddings false }
```

Weight tensors (27B, dense):
```
pre_fc_norm_hidden.weight      (5120,)        RMSNorm on target hidden
pre_fc_norm_embedding.weight   (5120,)        RMSNorm on next-token embedding
fc.{weight,scales,biases}      logical 10240 -> 5120   concat[h; e] projection
layers.0.input_layernorm / post_attention_layernorm
layers.0.self_attn.{q,k,v,o}_proj (+ q_norm, k_norm)   full attention, output gate
layers.0.mlp.{gate,up,down}_proj                        DENSE MLP (27B)
norm.weight                    (5120,)        final norm before lm_head
# NB: no embed_tokens, no lm_head -> borrowed from the target
```

35B-A3B head differs only in `layers.0.mlp`, which is the Qwen3.6 **sparse MoE
block**: `mlp.gate` (router), `mlp.shared_expert.*`, and routed experts.

### Forward (per the DeepSeek/Qwen-MTP family)
```
h'  = pre_fc_norm_hidden(h_prev)              # h_prev: target hidden for position i
e'  = pre_fc_norm_embedding(embed(token_{i+1}))
x   = fc(concat([h', e'], axis=-1))           # 10240 -> 5120
x   = DecoderLayer(x, cache=draft_cache)      # one full-attention Qwen3.6 layer
logits = lm_head(norm(x))                      # lm_head borrowed from target
h_new  = x                                     # chained hidden for the next draft step
```
The exact **hidden-state source** (raw last-layer residual vs post-`model.norm`)
and the space of the chained `h_new` are the one open correctness risk — see
Risks below.

## Architecture

### New module: `olmlx/engine/mtp/`
- `__init__.py`
- `draft_model.py` — `MTPConfig`, `MTPDraftModel`
- `decoder.py` — `MTPDecoder`

No changes to `olmlx/engine/eagle/*`. The shared, already-extracted helpers are
reused as-is.

### `MTPDraftModel` (`draft_model.py`)

Reuses **mlx-lm's own building blocks** so numerics match the target exactly:
- Attention + RoPE: the Qwen3.6 full-attention module
  (`mlx_lm.models.qwen3_next.Qwen3NextAttention` / `qwen3_5` decoder layer) —
  partial rotary 0.25, mrope, q/k RMSNorm, output gate, GQA 24/4, `head_dim` 256.
- MLP: **dense** (`Qwen3NextMLP`) when the head config has no MoE fields;
  **sparse MoE** (the `qwen3_5_moe` sparse block) when it does. This is the only
  real branch in the draft model. Routing is reused, not reimplemented.
- Norms: `nn.RMSNorm(hidden, eps=rms_norm_eps)` for `pre_fc_norm_hidden`,
  `pre_fc_norm_embedding`, `layers.0.*_layernorm`, and final `norm`.
- `fc`: `nn.Linear(2*hidden, hidden, bias=False)` (quantized on load).

Interface (matches the EAGLE draft so it slots into the same drafting loop):
- `bind(target_model)` — borrows `embed_tokens` and `lm_head` via
  `object.__setattr__` (does not register them as draft params). Reuses EAGLE's
  `_find_embed` / `_find_lm_head` fallback chain (handles VLM-wrapped targets).
- `make_cache()` — single-layer KV cache for `layers.0` (standard `KVCache`;
  the MTP layer is full-attention, not linear).
- `__call__(token_ids, h_prev, cache=None, compute_logits=True) -> (logits|None, h_new)`.

`MTPConfig` is parsed from the head's `config.json` `text_config` (+ top-level
`block_size`): hidden_size, intermediate_size, num_attention_heads,
num_key_value_heads, head_dim, attn_output_gate, partial_rotary_factor,
rope_parameters, rms_norm_eps, vocab_size, plus MoE fields when present
(num_experts, num_experts_per_tok, moe_intermediate_size,
shared_expert_intermediate_size, norm_topk_prob, decoder_sparse_step) and
`block_size`.

### `MTPDecoder` (`decoder.py`)

Implements the shared decoder protocol and **composes** the existing shared
helpers — structurally a sibling of `EagleDecoder`, but free to set the MTP
hidden-capture point without touching EAGLE:
- `prefill(prompt, cancel_event=None) -> int` — `reset()`; `_patch_model(target,
  [target_layer_id], hidden_storage)`; `draft.bind(target)`; build target +
  draft caches; decide trim regime via `can_trim_prompt_cache`, installing
  `GDNStateCapture.for_model(target)` for the non-trimmable (GatedDeltaNet)
  case; two-pass target prefill (prefix + single token) honoring `cancel_event`
  at chunk boundaries; capture target hidden at position -1; greedy-sample the
  first token; store `(_seed_token, _seed_hidden)`.
- `step() -> (accepted_tokens, num_accepted_draft)` — autoregressive draft chain
  of `block_size` steps (seed from target hidden, then feed draft `h_new`);
  single target verify forward over `[seed, *draft_tokens]`;
  `verify_draft_greedy`; cache trim by `(block_size+1) - num_accepted`
  (`trim_prompt_cache` when trimmable, else `GDNStateCapture.rollback_single`);
  always trim the draft cache; rotate seed to the last accepted token + its
  target hidden.
- `reset()` / `close()` — unpatch target, unbind draft, close GDN capture, drop
  caches/state. `__del__ -> reset()` safety net.
- `stats_summary()` — `steps`, `proposed`, `accepted_draft`, `acceptance_rate`,
  `avg_tokens_per_step`, `block_size`; strategy label `"mtp"` for metrics/logs.

### Loader + registry wiring

`olmlx/engine/registry.py`:
- Add `"mtp"` to the `SpeculativeStrategy` literal and
  `_VALID_SPECULATIVE_STRATEGIES`.
- No new config fields. Reuses `speculative_strategy="mtp"`,
  `speculative_draft_model=<mtp head repo>`, `speculative_tokens` (→ `block_size`,
  default from the head config when unset).

`olmlx/engine/model_manager.py`:
- Add a `spec_config.strategy == "mtp"` branch in the strategy dispatch →
  `_load_mtp_decoder(model, spec_config)`.
- `_load_mtp_decoder`: `_resolve_draft_path`; read head `config.json`; build
  `MTPConfig`; instantiate `MTPDraftModel`; `load_weights(..., strict=False)`
  with an explicit leftover/missing-key assertion in tests; vocab + hidden-size
  cross-check against the target; resolve `block_size` from `spec_config.num_tokens`
  or the head config; **raise a clear error if the target has flash_moe enabled**;
  return `MTPDecoder(target, draft, block_size, target_layer_id)`.

### flash_moe interaction

Consistent with the existing eagle/dflash limitation, **MTP + flash_moe is
mutually exclusive**. `_load_mtp_decoder` raises a descriptive error when the
target's resolved config enables flash_moe. The 35B-A3B target (~18 GB at 4-bit)
is validated with flash_moe **off** (fits in 64 GB). Documented in CLAUDE.md
beside the current eagle/dflash + flash_moe note.

## Testing

TDD — failing tests first.

**Unit (no large model required):**
- `MTPConfig` parses a real `qwen3_5_mtp` config (dense and MoE variants).
- `MTPDraftModel` builds and loads the shipped weights `strict=False` with **no
  missing and no unexpected keys** (the wiring guard).
- `bind()` borrows embed/lm_head; dense-vs-MoE MLP is selected from config.
- Registry accepts `strategy="mtp"`; `resolved_speculative()` threads it through.
- `_load_mtp_decoder` raises on flash_moe-enabled targets.

**Integration / acceptance (gated on model presence, run live):**
- 27B: load `unsloth/Qwen3.6-27B-MLX-8bit` + `mlx-community/Qwen3.6-27B-MTP-4bit`,
  greedy-generate; assert output **== non-speculative greedy** and
  **acceptance > 0.66**.
- 35B-A3B (flash_moe off): same assertions with the MoE head.

Measured acceptance numbers reported back, as done for the Qwen3.5-4B draft.

## Out of scope
- `olmlx mtp prepare` / any training (the head is pretrained).
- Distributed inference.
- MTP + flash_moe (mutually exclusive, by design).
- Multi-layer MTP (`mtp_num_hidden_layers > 1`); config-driven but only depth 1
  is validated (all shipped heads are depth 1).

## Risks & open questions
1. **Hidden-state source (primary correctness risk).** Whether the head consumes
   the raw last-layer residual or the post-`model.norm` hidden, and the space of
   the chained `h_new`, must match the pretrained head. A wrong choice yields
   correct text but ~0% acceptance. Resolution: pin against the HF `transformers`
   Qwen3.6 MTP modeling code and the mlx conversion script **before** building the
   loop; verify with a single-forward acceptance probe. The dedicated `MTPDecoder`
   exists precisely so this can be set without EAGLE regression risk.
2. **mrope for text-only.** The target uses interleaved mrope; for text-only
   inference this must reduce to standard positions consistently between the
   borrowed embeddings, the target, and the MTP layer. Reuse mlx-lm's RoPE path
   rather than rolling our own.
3. **MoE head numerics.** Reuse the `qwen3_5_moe` sparse block verbatim;
   reimplementing routing risks subtle acceptance loss.
4. **Quantized `fc`/borrowed-head dtype.** Ensure the quantized `fc` and the
   borrowed (possibly 8-bit target) `lm_head` compose without an eval-time dtype
   mismatch.
