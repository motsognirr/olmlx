"""Regression test: proxy-tuning steers a hybrid GatedDeltaNet base correctly.

The decoder is dense+hybrid capable (proxy-tuning never rejects tokens, so it
needs no GDN rollback capture). With an identical base/expert/anti-expert trio,
M+ - M- == 0, so the combined logits equal the base's — the proxy output must
therefore match a plain single-call-prefill greedy generation of the same
hybrid model, even on a >2048-token prompt that forces multi-chunk prefill
(the path that exercises GDN recurrent-state threading across chunks).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

_HYBRID = "mlx-community/Qwen3.5-0.8B-MLX-4bit"  # qwen3_5, hybrid GDN, non-MoE


def _model_dir() -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(_HYBRID))
    except Exception:
        return None


def _reference_greedy(model, ids, n):
    """Single-call-prefill greedy decode — the known-correct GDN baseline."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)[0, -1, :]
    out = []
    tok = int(mx.argmax(logits).item())
    for _ in range(n):
        out.append(tok)
        logits = model(mx.array([[tok]]), cache=cache)[0, -1, :]
        tok = int(mx.argmax(logits).item())
    return out


def _proxy_greedy(model, ids, n):
    from olmlx.engine.proxy_tuning import ProxyTuningDecoder

    dec = ProxyTuningDecoder(model, model, model, alpha=1.0)
    first = dec.prefill(mx.array([ids]))
    out = [first]
    for _ in range(n - 1):
        toks, _ = dec.step()
        out.append(toks[0])
    return out


@pytest.mark.real_model
@pytest.mark.parametrize(
    "kind, n_repeat",
    [("short_single_chunk", 0), ("long_multi_chunk", 400)],
)
def test_proxy_tuning_matches_reference_on_hybrid_base(kind, n_repeat):
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    path = _model_dir()
    if path is None:
        pytest.skip(f"{_HYBRID} not downloadable")

    from mlx_lm import load

    model, tok = load(str(path))

    if n_repeat:
        content = (
            "The transformer architecture processes tokens through attention "
            "layers. " * n_repeat
        ) + " Summarize the above in one sentence. /no_think"
    else:
        content = "Explain what a KV cache is in two sentences. /no_think"
    ids = tok.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True
    )
    if n_repeat:
        assert len(ids) > 2048, "long prompt must exceed the 2048-token prefill chunk"

    ref = _reference_greedy(model, ids, 40)
    proxy = _proxy_greedy(model, ids, 40)
    assert proxy == ref, (
        f"[{kind}] proxy output diverged from single-call-prefill reference on a "
        f"hybrid GDN base — GDN state likely mishandled. "
        f"ref={tok.decode(ref)[:120]!r} proxy={tok.decode(proxy)[:120]!r}"
    )
