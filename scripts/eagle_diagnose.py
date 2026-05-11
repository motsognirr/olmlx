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

SCOPE: each position is measured with a *fresh* draft cache (cold-
cache top-1 agreement). The real decoder's draft cache holds the
accepted prefix from all prior steps, and a draft whose accuracy
degrades as that cache fills would look healthy here. On
Qwen3.5-27B-4bit the diagnostic reports ~24% top-1 while real bench
acceptance is ~6% — the gap is exactly this warm-cache compounding-
error effect. Don't read the diagnostic's number as predicting bench
acceptance; read it as the *upper bound* on per-step agreement.

Usage:
    uv run python scripts/eagle_diagnose.py --target <path>
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--target",
        type=Path,
        required=True,
        help=(
            "Path to the local target model directory (e.g. "
            "``~/.olmlx/models/mlx-community_Qwen3.5-27B-4bit``). Required — "
            "no default so the script doesn't silently fail on a missing "
            "hardcoded path when run on a different machine."
        ),
    )
    p.add_argument(
        "--draft",
        type=Path,
        default=None,
        help="Path to the EAGLE draft directory (default: <target>/eagle)",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Here is a brief explanation of how photosynthesis works:\n\n",
        help="Prompt for the diagnostic. Default is a benign coding-prose seed.",
    )
    p.add_argument(
        "--n-continue",
        type=int,
        default=64,
        help="Number of greedy continuation tokens to measure (default: %(default)s)",
    )
    return p


def main() -> int:
    """All side-effecting work happens here so the module is import-safe.

    Previously the script ran ``parse_args()`` + the entire diagnostic
    body at module top level. Any accidental ``import`` of this script
    (pytest collection, tooling that walks the source tree, an IDE's
    static analyser) would consume ``sys.argv`` at import time and
    immediately bail with argparse's help output. Wrapping everything
    in ``main()`` keeps the import side-effect-free.
    """
    args = _build_parser().parse_args()
    target_path: Path = args.target
    draft_path: Path = args.draft if args.draft is not None else target_path / "eagle"
    prompt: str = args.prompt
    n_continue: int = args.n_continue

    print(f"Target: {target_path}")
    print(f"Draft:  {draft_path}")
    print(f"Prompt: {prompt!r}")
    print(f"Continuation length: {n_continue}\n")

    target, tokenizer = mlx_lm_load(str(target_path))
    target.eval()

    # Load draft + EAGLE config.
    draft_cfg = json.loads((draft_path / "config.json").read_text())
    # Defensive: ``draft_cfg["eagle_config"]`` would raise ``KeyError``
    # if the operator accidentally pointed ``--draft`` at a DFlash
    # directory (or anything else without an ``eagle_config`` block).
    # Surface the type mismatch up front rather than crash mid-script.
    # Mirrors the ``isinstance(eagle_cfg, dict)`` check in
    # ``_load_eagle_decoder``.
    eagle_block = draft_cfg.get("eagle_config")
    if not isinstance(eagle_block, dict):
        print(
            f"ERROR: {draft_path / 'config.json'} has no 'eagle_config' "
            "block — this looks like a DFlash draft or an unrelated "
            "checkpoint. The diagnostic only works against EAGLE drafts."
        )
        return 1
    target_layer_id = eagle_block.get("target_layer_id")
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
        block_size=eagle_block["block_size"],
        rope_scaling=draft_cfg.get("rope_scaling"),
    )
    draft = EagleDraftModel(eagle_cfg)
    weight_files = sorted(draft_path.glob("model*.safetensors"))
    # Defensive: ``load_weights([], strict=False)`` silently succeeds
    # with all-random weights — the diagnostic would then report
    # near-zero agreement with no hint why. Mirrors the
    # ``FileNotFoundError`` in ``_load_eagle_decoder``.
    if not weight_files:
        print(
            f"ERROR: no model-*.safetensors weight files found in "
            f"{draft_path}. The draft directory must contain at least "
            "one weight file produced by `olmlx eagle prepare`."
        )
        return 1
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
        target_layer_id
        if target_layer_id is not None
        else (len(_get_layers(target)) - 1)
    )
    print(f"Hooking target layer index: {hook_layer}\n")
    _patch_model(target, [hook_layer], storage)

    try:
        # Tokenise prompt with chat template (same path bench uses).
        from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: F401

        messages = [{"role": "user", "content": prompt}]
        template_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = mx.array(
            tokenizer.encode(template_text, add_special_tokens=False), dtype=mx.int32
        ).reshape(1, -1)
        print(f"Tokenised prompt length: {prompt_ids.shape[1]}")

        # ---- Step 1: greedy roll out the target for n_continue tokens.
        from mlx_lm.models.cache import make_prompt_cache

        target_cache = make_prompt_cache(target)
        out = target(prompt_ids, cache=target_cache)
        # Mirrors ``_logits`` in the production decoder: mlx-lm targets
        # return a raw ``mx.array``, mlx-vlm targets return a dataclass
        # with a ``logits`` attribute. ``out["logits"]`` would TypeError
        # on the dataclass case (latent here because we typically run
        # against mlx-lm Qwen3.5 — but the diagnostic should work for
        # any target the production loader accepts).
        logits = getattr(out, "logits", out)
        h_at_last = storage[0]
        if h_at_last is None:
            print("ERROR: layer hook did not populate hidden after prompt forward.")
            return 1

        target_tokens: list[int] = []
        target_hiddens: list[mx.array] = [h_at_last[:, -1:, :]]  # h_{P-1}

        last_logits = logits[:, -1:, :]
        next_tok = int(mx.argmax(last_logits, axis=-1).item())
        for _ in range(n_continue):
            target_tokens.append(next_tok)
            tok_in = mx.array([[next_tok]], dtype=mx.int32)
            out = target(tok_in, cache=target_cache)
            logits = getattr(out, "logits", out)
            # Eagerly materialise the appended slice. Without this, each
            # iteration appends a lazy slice node referencing the live
            # ``storage[0]`` array; the whole chain would re-evaluate in
            # one shot when the teacher-forcing loop reaches its first
            # ``.item()`` call below — at ``--n-continue 64`` that's a
            # 64-step graph compile each time. Bound the graph per step.
            h_t = storage[0][:, -1:, :]
            mx.eval(h_t)
            target_hiddens.append(h_t)
            last_logits = logits[:, -1:, :]
            next_tok = int(mx.argmax(last_logits, axis=-1).item())

        print(
            f"Target greedy continuation (decoded): {tokenizer.decode(target_tokens)!r}"
        )
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
            # Fresh draft cache for each position so the comparison is
            # a pure single-step prediction. This matches the *first*
            # draft call of each speculative verify step in the real
            # decoder — when the draft cache holds only what survived
            # the previous step's trim, which for step 1 is empty.
            #
            # SCOPE: this measures cold-cache top-1 agreement. The
            # real decoder feeds its own predicted hidden into
            # subsequent positions of the same verify, and a draft
            # whose accuracy degrades as the draft KV cache fills
            # would look fine in this diagnostic. The actual bench
            # numbers (~6% acceptance vs 24% teacher-forced top-1
            # here) capture that degradation. For a fuller picture,
            # see ``stats_summary()`` from a real ``EagleDecoder``
            # run.
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
        return 0
    finally:
        _unpatch_model(target)
        draft.unbind()


if __name__ == "__main__":
    sys.exit(main())
