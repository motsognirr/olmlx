"""Gate a trained M-/M+ pair: do they share one tokenizer + match the base vocab?

Proxy-tuning combines logits across M (base), M- (anti-expert), M+ (expert)
token-by-token, so all three must share one exact tokenizer. Run this before
registering the pair for serving.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable

from olmlx.engine.proxy_tuning import check_vocab_identity


def _load_tokenizer(path: str) -> Any:
    from mlx_lm.utils import load_tokenizer

    # Tokenizer-only load: fetches just the tokenizer/config files, never the
    # multi-GB model weights (we only need get_vocab() for the identity check).
    # Using load() here would fully materialize each model — and the base would
    # then be loaded a second time by the eval pipeline's own loader.
    return load_tokenizer(path, {"trust_remote_code": True})


def _load_config_vocab_size(path: str) -> int:
    """Return ``config.json``'s ``vocab_size`` (the model's logits dimension).

    This is the padded embedding/logits width (Qwen3 = 151936), which is what
    proxy-tuning's per-token logit arithmetic needs to line up across M/M-/M+ —
    distinct from ``len(tokenizer.get_vocab())`` (Qwen3 = 151669, the count of
    actual token entries).

    Hybrid/VLM-style configs (e.g. ``qwen3_5``, ``qwen3_next``) nest
    ``vocab_size`` under ``text_config`` and leave the top level unset, so fall
    back there before failing.
    """
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    vocab_size = config.get("vocab_size")
    if vocab_size is None:
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            vocab_size = text_config.get("vocab_size")
    if vocab_size is None:
        raise ValueError(
            f"config.json at {path} has no vocab_size (checked top level and "
            f"text_config) — cannot verify the proxy-tuning pair's logits width."
        )
    return int(vocab_size)


def assert_serveable_pair(
    anti_expert_dir: str,
    expert_dir: str,
    base_vocab_size: int,
    *,
    base_dir: str | None = None,
    loader: Callable[[str], Any] = _load_tokenizer,
    config_loader: Callable[[str], int] = _load_config_vocab_size,
) -> None:
    """Raise ValueError unless M-/M+ share a token->id mapping and match the base.

    ``base_vocab_size`` is the steered model's logits dimension — its
    ``config.json`` ``vocab_size`` (Qwen3 = 151936) — which must match the
    pair's so the combined logits align position-by-position.

    When ``base_dir`` is given, the base's tokenizer is also checked for
    token-mapping identity against the pair. A width match alone is too weak:
    two model families can share a ``vocab_size`` yet map tokens differently, so
    a same-width base from another family would otherwise be silently accepted
    and corrupt the per-token logit arithmetic at generation time.
    """
    tok_anti = loader(anti_expert_dir)
    tok_expert = loader(expert_dir)
    # M- <-> M+ token-mapping identity (reuses the engine's tested guard).
    check_vocab_identity(
        tok_anti,
        tok_expert,
        reference_label="anti-expert (M-)",
        other_label="expert (M+)",
    )
    # M <-> M+ token-mapping identity. M- and M+ are already proven identical
    # above, so matching the base against M+ transitively covers all three.
    if base_dir is not None:
        check_vocab_identity(
            loader(base_dir),
            tok_expert,
            reference_label="base (M)",
            other_label="expert (M+)",
        )
    # Each model's logits width must equal the steered base's, or the per-token
    # logit arithmetic (base + alpha*(expert - anti-expert)) is misaligned. When
    # base_dir is supplied, width-check the base too: token-mapping identity does
    # not pin the padded width (a same-mapping base can still declare a different
    # config vocab_size), and base_vocab_size may come from a separate source
    # than base_dir (e.g. the CLI's --base-vocab vs --base-dir).
    width_checks = [
        ("anti-expert (M-)", anti_expert_dir),
        ("expert (M+)", expert_dir),
    ]
    if base_dir is not None:
        width_checks.append(("base (M)", base_dir))
    for label, model_dir in width_checks:
        vocab_size = config_loader(model_dir)
        if vocab_size != base_vocab_size:
            raise ValueError(
                f"{label} config vocab_size ({vocab_size}) does not match the "
                f"steered base vocabulary ({base_vocab_size}). All three models "
                f"must share one logits dimension — confirm the base, M-, and "
                f"M+ are the same family."
            )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Verify an M-/M+ proxy-tuning pair is serveable."
    )
    ap.add_argument("anti_expert", help="M- (untuned base) model directory")
    ap.add_argument("expert", help="M+ (fine-tuned) model directory")
    ap.add_argument(
        "--base-vocab",
        type=int,
        default=151936,
        help="steered base vocab size (Qwen3 dense = 151936)",
    )
    ap.add_argument(
        "--base-dir",
        default=None,
        help="steered base (M) model dir; when given, its tokenizer is also "
        "checked for token-mapping identity against the M-/M+ pair",
    )
    args = ap.parse_args(argv)
    assert_serveable_pair(
        args.anti_expert,
        args.expert,
        args.base_vocab,
        base_dir=args.base_dir,
    )
    if args.base_dir is not None:
        print("OK: base, M-, and M+ share one tokenizer and one vocabulary width.")
    else:
        print(
            f"OK: M-/M+ share one tokenizer and match the base vocab width "
            f"({args.base_vocab}). Base tokenizer NOT checked — pass --base-dir "
            f"to verify the base shares the pair's token mapping."
        )


if __name__ == "__main__":
    main()
