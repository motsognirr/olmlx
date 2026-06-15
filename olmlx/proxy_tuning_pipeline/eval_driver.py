"""Greedy decode driver for the Stage-3 eval (one decoder, one prompt)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def generate_one(
    decoder: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 256,
) -> str:
    """Run the proxy-tuning decoder greedily over one chat prompt.

    Calls run on the default stream (the decoder's contract) — no mx.stream
    wrapper. ``max_tokens`` counts the first token plus subsequent steps.
    """
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    eos = set(tokenizer.eos_token_ids)

    first = decoder.prefill(mx.array([ids]))
    out: list[int] = []
    if first in eos:
        return tokenizer.decode(out)
    out.append(int(first))

    for _ in range(max_tokens - 1):
        toks, _ = decoder.step()
        tok = int(toks[0])
        if tok in eos:
            break
        out.append(tok)

    return tokenizer.decode(out)
