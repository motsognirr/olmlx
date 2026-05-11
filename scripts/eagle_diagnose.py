"""Diagnose EAGLE draft vs target token-by-token agreement.

Loads the trained EAGLE draft and the target, runs the target greedily
on a fixed prompt for N continuation tokens, captures the target's
hidden states at the configured layer, then teacher-forces the draft
through the same sequence under the EAGLE pairing
``(h_{t-1}, token_t) -> token_{t+1}`` and reports per-position
agreement with the target's actual greedy choice.

This is the same pairing used at inference time, so a high agreement
rate here means the draft will accept well during real speculative
decoding. A low rate isolates the mismatch from the verify/rollback
machinery and tells us the issue is in the conditioning itself.

Usage:
    uv run python scripts/eagle_diagnose.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn  # noqa: F401  (used implicitly by mlx_lm.load)
from mlx_lm import load as mlx_lm_load

from olmlx.engine.dflash.decoder import _get_layers, _patch_model, _unpatch_model
from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

_DEFAULT_TARGET = Path.home() / ".olmlx/models/mlx-community_Qwen3.5-27B-4bit"

_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
_parser.add_argument(
    "--target",
    type=Path,
    default=_DEFAULT_TARGET,
    help="Path to the local target model directory (default: %(default)s)",
)
_parser.add_argument(
    "--draft",
    type=Path,
    default=None,
    help="Path to the EAGLE draft directory (default: <target>/eagle)",
)
_parser.add_argument(
    "--prompt",
    type=str,
    default="Here is a brief explanation of how photosynthesis works:\n\n",
    help="Prompt for the diagnostic. Default is a benign coding-prose seed.",
)
_parser.add_argument(
    "--n-continue",
    type=int,
    default=64,
    help="Number of greedy continuation tokens to measure (default: %(default)s)",
)
_args = _parser.parse_args()
TARGET_PATH = _args.target
DRAFT_PATH = _args.draft if _args.draft is not None else TARGET_PATH / "eagle"
PROMPT = _args.prompt
N_CONTINUE = _args.n_continue

print(f"Target: {TARGET_PATH}")
print(f"Draft:  {DRAFT_PATH}")
print(f"Prompt: {PROMPT!r}")
print(f"Continuation length: {N_CONTINUE}\n")

target, tokenizer = mlx_lm_load(str(TARGET_PATH))
target.eval()

# Load draft + EAGLE config.
draft_cfg = json.loads((DRAFT_PATH / "config.json").read_text())
target_layer_id = draft_cfg["eagle_config"].get("target_layer_id")
print(f"EAGLE target_layer_id from config: {target_layer_id}")

eagle_cfg = EagleConfig(
    hidden_size=draft_cfg["hidden_size"],
    num_hidden_layers=draft_cfg["num_hidden_layers"],
    num_attention_heads=draft_cfg["num_attention_heads"],
    num_key_value_heads=draft_cfg["num_key_value_heads"],
    head_dim=draft_cfg["head_dim"],
    intermediate_size=draft_cfg["intermediate_size"],
    vocab_size=draft_cfg["vocab_size"],
    rms_norm_eps=draft_cfg["rms_norm_eps"],
    rope_theta=draft_cfg["rope_theta"],
    max_position_embeddings=draft_cfg["max_position_embeddings"],
    block_size=draft_cfg["eagle_config"]["block_size"],
    rope_scaling=draft_cfg.get("rope_scaling"),
)
draft = EagleDraftModel(eagle_cfg)
weight_files = sorted(DRAFT_PATH.glob("model*.safetensors"))
weights: list[tuple[str, mx.array]] = []
for wf in weight_files:
    weights.extend(mx.load(str(wf)).items())
draft.load_weights(weights, strict=False)
draft.bind(target)

# Hook the configured target layer to capture hiddens.
#
# Fallback for older checkpoints without ``target_layer_id``: use the
# *target*'s deepest layer, not the draft's layer count. Earlier
# revisions of this script used ``eagle_cfg.num_hidden_layers - 1``
# which is the draft layer count (typically 1) — that hooks target
# layer 0 instead of layer 63, reproducing the exact conditioning
# mismatch the training fix was designed to prevent.
storage: list[mx.array | None] = [None]
hook_layer = (
    target_layer_id if target_layer_id is not None else (len(_get_layers(target)) - 1)
)
print(f"Hooking target layer index: {hook_layer}\n")
_patch_model(target, [hook_layer], storage)

try:
    # Tokenise prompt with chat template (same path bench uses).
    from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: F401

    messages = [{"role": "user", "content": PROMPT}]
    template_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = mx.array(
        tokenizer.encode(template_text, add_special_tokens=False), dtype=mx.int32
    ).reshape(1, -1)
    print(f"Tokenised prompt length: {prompt_ids.shape[1]}")

    # ---- Step 1: greedy roll out the target for N_CONTINUE tokens.
    from mlx_lm.models.cache import make_prompt_cache

    target_cache = make_prompt_cache(target)
    out = target(prompt_ids, cache=target_cache)
    # Mirrors ``_logits`` in the production decoder: mlx-lm targets
    # return a raw ``mx.array``, mlx-vlm targets return a dataclass with
    # a ``logits`` attribute. ``out["logits"]`` would TypeError on the
    # dataclass case (latent here because we typically run against
    # mlx-lm Qwen3.5 — but the diagnostic should work for any target
    # the production loader accepts).
    logits = getattr(out, "logits", out)
    h_at_last = storage[0]
    if h_at_last is None:
        print("ERROR: layer hook did not populate hidden after prompt forward.")
        sys.exit(1)

    target_tokens: list[int] = []
    target_hiddens: list[mx.array] = [h_at_last[:, -1:, :]]  # h_{P-1}

    last_logits = logits[:, -1:, :]
    next_tok = int(mx.argmax(last_logits, axis=-1).item())
    for _ in range(N_CONTINUE):
        target_tokens.append(next_tok)
        tok_in = mx.array([[next_tok]], dtype=mx.int32)
        out = target(tok_in, cache=target_cache)
        logits = getattr(out, "logits", out)
        target_hiddens.append(storage[0][:, -1:, :])
        last_logits = logits[:, -1:, :]
        next_tok = int(mx.argmax(last_logits, axis=-1).item())

    print(f"Target greedy continuation (decoded): {tokenizer.decode(target_tokens)!r}")
    print(f"Token IDs: {target_tokens}\n")

    # ---- Step 2: teacher-force the draft under EAGLE pairing.
    # At position t (0..N-2), draft input = (h_{t-1}, token_t),
    # label = token_{t+1}.
    # h_{t-1} = target_hiddens[t]   (target_hiddens[0] is h_{P-1})
    # token_t = target_tokens[t]
    # token_{t+1} = target_tokens[t+1]
    matches: list[bool] = []
    print("Per-position diagnostic (top-1 draft vs target greedy choice):")
    print(
        f"{'pos':>3} {'h_idx':>5} {'token_t':>8} -> "
        f"{'draft_top1':>10} | {'target':>6} | match"
    )
    for t in range(len(target_tokens) - 1):
        h_prev = target_hiddens[t]
        token_t = target_tokens[t]
        label = target_tokens[t + 1]

        tok_in = mx.array([[token_t]], dtype=mx.int32)
        # Fresh draft cache for each position so the comparison is a
        # pure single-step prediction (matches the very first draft
        # call of each speculative step in the decoder).
        draft_cache = draft.make_cache()
        logits, _h_new = draft(token_ids=tok_in, h_prev=h_prev, cache=draft_cache)
        draft_top1 = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        ok = draft_top1 == label
        matches.append(ok)
        if t < 16 or not ok:
            print(
                f"{t:>3} {t:>5} {token_t:>8} -> "
                f"{draft_top1:>10} | {label:>6} | {'OK' if ok else 'MISS'}"
            )

    n = len(matches)
    accept = sum(matches) / n if n else 0.0
    print(f"\nAgreement: {sum(matches)}/{n} = {accept:.1%}")
finally:
    _unpatch_model(target)
    draft.unbind()
