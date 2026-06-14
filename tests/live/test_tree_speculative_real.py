"""Live greedy-parity canary for tree speculative verification (#501).

Greedy speculative decoding is exactness-preserving: its output must equal
the target model's own greedy decode, *modulo argmax tie-breaks*. When two
tokens share the exact same top logit, the L=1 single-token forward (pure
greedy / draft) and the L=N verify forward may resolve the tie to different
token ids; that is benign nondeterminism, not corruption, and it affects the
linear path too (verified 2026-06: on this prompt the linear path diverges
from pure greedy at an exact 0.0-margin tie while the tree path matches it).

So this canary asserts both the tree and the linear path match the target's
pure-greedy decode, tolerating divergences that occur only at a tie position
(top1-top2 logit margin below TIE_EPS). A divergence at a clear-margin
position is a real correctness bug and fails loudly.

This is also the regression guard for the bit-rot that left the tree path
crashing on current mlx (no real-model test exercised it before #501 — the
_tree_call signature + mask-dtype fixes).

Lives outside tests/integration/ to dodge its autouse MLX mock.
real_model; skipped in CI (`-m "not real_model"`) and when the model
isn't downloaded.

Run: uv run pytest tests/live/test_tree_speculative_real.py -m real_model -v
"""

import pytest

from olmlx.config import settings

TARGET = "mlx-community/Qwen3-0.6B-4bit"
DRAFT = "mlx-community/Qwen3-0.6B-4bit"
N_TOKENS = 48
# Tokens whose target top1-top2 logit gap is below this are exact/near ties;
# a path may legitimately break them differently from pure greedy.
TIE_EPS = 1e-2


def _model_dir(model: str):
    from olmlx.models.store import _safe_dir_name

    return settings.models_dir / _safe_dir_name(model)


def _present(model: str) -> bool:
    return (_model_dir(model) / "config.json").exists()


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not (_present(TARGET) and _present(DRAFT)),
        reason=f"{TARGET}/{DRAFT} not downloaded in {settings.models_dir}",
    ),
]


@pytest.fixture(scope="module")
def models():
    from mlx_lm import load

    target, tokenizer = load(str(_model_dir(TARGET)))
    draft, _ = load(str(_model_dir(DRAFT)))
    return target, draft, tokenizer


def _prompt_ids(tokenizer, content: str):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _spec_sequence(draft, target, prompt_ids, *, tree_width):
    import mlx.core as mx

    from olmlx.engine.speculative import SpeculativeDecoder

    dec = SpeculativeDecoder(
        draft,
        target,
        num_speculative_tokens=4,
        tree_width=tree_width,
        tree_max_nodes=12,
    )
    out = [dec.prefill(mx.array([prompt_ids]))]
    while len(out) < N_TOKENS:
        accepted, _ = dec.step()
        out.extend(accepted)
    dec.close()
    return out[:N_TOKENS]


def _pure_greedy(target, prompt_ids):
    """Reference: token-by-token greedy target decode plus, per step, the
    top1-top2 logit margin used to recognise tie positions."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(target)
    logits = target(mx.array([prompt_ids]), cache=cache)[:, -1, :]
    tokens = [int(mx.argmax(logits, axis=-1).item())]
    margins = [float("inf")]  # the prefill step's own token (margin unused)
    for _ in range(N_TOKENS - 1):
        logits = target(mx.array([[tokens[-1]]]), cache=cache)[:, -1, :]
        row = logits[0]
        top2 = mx.sort(row)[-2:]
        margins.append(float((top2[1] - top2[0]).item()))
        tokens.append(int(mx.argmax(row).item()))
    return tokens, margins


def _assert_greedy_modulo_ties(label, seq, greedy, margins):
    """seq must equal the pure-greedy decode up to the first divergence, and
    any first divergence must land on a tie position (margin < TIE_EPS)."""
    for i, (a, b) in enumerate(zip(seq, greedy)):
        if a == b:
            continue
        # First divergence: acceptable only if greedy[i] was a tie.
        assert margins[i] < TIE_EPS, (
            f"{label} diverges from pure greedy at index {i} "
            f"(got {a}, greedy {b}) where the target logit margin is "
            f"{margins[i]:.4f} >= {TIE_EPS} — not a tie, real corruption"
        )
        return  # contexts diverge after a legit tie-break; stop comparing
    # No divergence at all.


def test_tree_and_linear_match_pure_greedy(models):
    target, draft, tokenizer = models
    ids = _prompt_ids(
        tokenizer,
        "List the first five prime numbers, comma-separated.",
    )
    greedy, margins = _pure_greedy(target, ids)
    linear = _spec_sequence(draft, target, ids, tree_width=1)
    tree = _spec_sequence(draft, target, ids, tree_width=3)

    _assert_greedy_modulo_ties("linear", linear, greedy, margins)
    _assert_greedy_modulo_ties("tree", tree, greedy, margins)
