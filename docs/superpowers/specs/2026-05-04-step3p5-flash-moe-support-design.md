# Step-3.5-Flash Flash-MoE Support

## Problem

`olmlx flash prepare mlx-community/Step-3.5-Flash-6bit` falls through to dense Flash prep instead of Flash-MoE prep, then dies on `trust_remote_code` for the custom Step3p5 architecture.

Root cause: Step-3.5 declares its MoE topology with non-standard config keys that the Flash-MoE pipeline doesn't recognize.

## Gaps

Compared to existing Flash-MoE-supported models (DeepSeek, Qwen3, gpt-oss, MiniMax, Gemma4, Kimi-K2.5):

1. **Expert count key.** Step-3.5 uses `moe_num_experts: 288`. `is_moe_model()` and `prepare_moe_for_flash()` only check `n_routed_experts` / `num_local_experts` / `num_experts`.

2. **MoE layer set.** Step-3.5 specifies MoE layers via `moe_layers_enum: "3,4,...,44"` (explicit list). `_detect_moe_layers()` only knows `first_k_dense_replace` + `moe_layer_freq` / `decoder_sparse_step` and the Nemotron-H `hybrid_override_pattern`.

3. **Shared expert attribute name.** Step3p5MoE has a `share_expert` (singular) attribute alongside `switch_mlp`. The runtime `_FlashMoEDeepSeek` variant — which is selected for Step-3.5 because the gate is a custom module returning `(inds, scores)` — only probes `shared_experts` (plural). Without a fix, the shared-expert contribution is silently dropped, producing wrong outputs.

## Out of scope

- **MTP layers.** Step-3.5 has `num_nextn_predict_layers: 3`. mlx-lm's `step3p5.sanitize()` (lines 422-428) drops `.mtp.*` weights and any layer index ≥ `num_hidden_layers`, so the bundler never sees them. No change needed.
- **`moe_layer_offset` / `moe_every_n_layer`.** Step-3.5 also sets these, but `moe_layers_enum` is authoritative and present, so we use it directly. Add fallback only when a future model needs it.

## Changes

### `engine/flash/moe_prepare.py`

`is_moe_model()` and the expert-count read in `prepare_moe_for_flash()`: add `moe_num_experts` to the OR-chain alongside the three existing aliases.

### `engine/flash/moe_bundler.py`

`_detect_moe_layers()`: if `text_config.get("moe_layers_enum")` is a non-empty string, parse it as a comma-separated list of int layer indices, sort, return. This branch sits above the existing freq/offset logic and below the `hybrid_override_pattern` branch.

### `engine/flash/flash_moe_model.py`

`_FlashMoEDeepSeek.__init__`: probe `shared_experts` first, then fall back to `share_expert`. Store under `self.shared_experts` (single attribute name) so `_combine` is unchanged.

## Tests

Unit tests using synthetic config dicts (no model files required):

- `is_moe_model()` returns True for a config with only `moe_num_experts`.
- `_detect_moe_layers()` returns the parsed list for a config with `moe_layers_enum` and ignores any conflicting `moe_layer_freq` / `first_k_dense_replace` settings on the same dict.
- `_detect_moe_layers()` still works on existing DeepSeek- and Qwen3-style configs (regression).

Skip an end-to-end prep/serve test for now — the real model is too large for CI. Manual verification: run `olmlx flash prepare mlx-community/Step-3.5-Flash-6bit` then `olmlx serve` and exercise a chat completion.

## Risks

- The serving path may surface additional Step3p5-specific issues not visible from prep alone (e.g., Step3p5MoEGate behavior differences from DeepSeek). Plan: ship the prep + runtime changes, then iterate on serving issues as they appear rather than guessing now.
- `share_expert` could be named differently in some Step3p5 fork. Acceptable — fix when seen.
