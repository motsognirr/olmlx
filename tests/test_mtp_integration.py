"""Live MTP acceptance + exactness gate (27B dense head).

Heavy (loads the 27B target) — gated behind OLMLX_RUN_MTP_INTEGRATION=1 so the
default suite stays fast. Validates two things against the real shipped
``mlx-community/Qwen3.6-27B-MTP-4bit`` head:

1. Speculative decoding is exactness-preserving: the MTP-drafted greedy
   sequence is token-identical to plain target greedy.
2. Acceptance is well above the broken-wiring floor (~0.006), proving the
   draft front-end (concat order + post-norm chain) is wired correctly. The
   native head reaches ~0.85 on code and ~0.54 on prose; we assert > 0.5 on a
   code prompt as a robust regression guard (see the design doc for the
   prose/code spread).
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OLMLX_RUN_MTP_INTEGRATION") != "1",
    reason="set OLMLX_RUN_MTP_INTEGRATION=1 to run the heavy MTP acceptance test",
)

TARGET = "unsloth/Qwen3.6-27B-MLX-8bit"
HEAD = "mlx-community/Qwen3.6-27B-MTP-4bit"
PROMPT = "Write a Python function that returns the n-th Fibonacci number iteratively."
N_TOKENS = 96


def _setup():
    import json
    from pathlib import Path

    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    from olmlx.engine.mtp.decoder import MTPDecoder
    from olmlx.engine.mtp.draft_model import MTPConfig, load_mtp_draft

    model, tok = load(TARGET)
    hd = Path(snapshot_download(HEAD))
    cfg = MTPConfig.from_dict(json.loads((hd / "config.json").read_text()))
    draft = load_mtp_draft(hd, cfg)
    ids = mx.array([tok.encode(PROMPT)], dtype=mx.int32)
    return model, draft, cfg, ids, mx, MTPDecoder


def _greedy(model, ids, n, mx):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    out = model(ids, cache=cache)
    tokens = []
    cur = int(mx.argmax(out[0, -1]).item())
    for _ in range(n):
        tokens.append(cur)
        nxt = model(mx.array([[cur]], dtype=mx.int32), cache=cache)
        cur = int(mx.argmax(nxt[0, -1]).item())
    return tokens


def test_mtp_27b_exactness_and_acceptance():
    model, draft, cfg, ids, mx, MTPDecoder = _setup()

    # Plain greedy reference.
    greedy = _greedy(model, ids, N_TOKENS, mx)

    # Speculative greedy via the MTP decoder.
    dec = MTPDecoder(model, draft, block_size=cfg.block_size)
    first = dec.prefill(ids)
    spec = [first]
    while len(spec) < N_TOKENS + 1:
        accepted, _ = dec.step()
        spec.extend(accepted)
    stats = dec.stats_summary()
    dec.reset()

    # Exactness: the speculative sequence must match plain greedy.
    assert spec[:N_TOKENS] == greedy[:N_TOKENS], "MTP output diverged from greedy"

    # Acceptance well above the broken-wiring floor.
    assert stats["acceptance_rate"] > 0.5, (
        f"acceptance {stats['acceptance_rate']:.3f} too low — draft wiring "
        "(concat order / post-norm chain) likely regressed"
    )
