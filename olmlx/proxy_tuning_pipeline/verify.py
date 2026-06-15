"""Gate a trained M-/M+ pair: do they share one tokenizer + match the base vocab?

Proxy-tuning combines logits across M (base), M- (anti-expert), M+ (expert)
token-by-token, so all three must share one exact tokenizer. Run this before
registering the pair for serving.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

from olmlx.engine.proxy_tuning import check_vocab_identity


def _load_tokenizer(path: str) -> Any:
    from mlx_lm import load

    _model, tokenizer = load(path)
    return tokenizer


def assert_serveable_pair(
    anti_expert_dir: str,
    expert_dir: str,
    base_vocab_size: int,
    *,
    loader: Callable[[str], Any] = _load_tokenizer,
) -> None:
    """Raise ValueError unless M-/M+ share a token->id mapping and match the base.

    ``base_vocab_size`` is the steered model's vocabulary size (Qwen3 = 151936).
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
    vocab = tok_anti.get_vocab()
    if len(vocab) != base_vocab_size:
        raise ValueError(
            f"M-/M+ vocab size ({len(vocab)}) does not match the steered base "
            f"vocabulary ({base_vocab_size}). All three models must share one "
            f"tokenizer — confirm the base, M-, and M+ are the same family."
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
    args = ap.parse_args(argv)
    assert_serveable_pair(args.anti_expert, args.expert, args.base_vocab)
    print("OK: M-/M+ share one tokenizer and match the base vocabulary.")


if __name__ == "__main__":
    main()
