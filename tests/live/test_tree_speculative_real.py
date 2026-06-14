"""Live greedy-parity canary for tree speculative verification (#501).

Greedy speculative decoding is exactness-preserving: every token it emits
must be the target model's own greedy argmax given the true preceding
context — *modulo argmax tie-breaks*. When two tokens share (within bf16
noise) the top logit, a batched verify forward and a single-token forward
may resolve the tie to different ids; that is benign nondeterminism, not
corruption, and it affects the linear path too.

This canary teacher-forces each path's full output back through the target
in one clean forward and asserts every position is the target's argmax,
tolerating only positions whose top1-top2 logit margin is below TIE_EPS.
Teacher-forcing checks the WHOLE sequence (not just up to a first
divergence), so a real corruption that lands after a benign tie-break is
still caught; any clear-margin mismatch fails loudly.

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
# A position whose target top1-top2 logit gap is below this is an exact/near
# tie that a path may legitimately break differently from a single clean
# forward. Kept small: a real corruption surfaces as a clearly-higher-logit
# alternative (margin well above this), so it is still caught.
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


def _assert_target_greedy(label, seq, target, prompt_ids):
    """Teacher-force ``seq`` through the target in one forward and assert every
    token is the target's greedy argmax given its true preceding context,
    excusing only positions where the target logits are a tie (< TIE_EPS).

    Unlike a reference-walk that bails at the first divergence, this checks
    every position, so corruption after a benign tie-break is still caught.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    # full = prompt + seq[:-1]; logit at index (len(prompt)-1 + i) predicts
    # seq[i] from the real context prompt + seq[:i].
    full = list(prompt_ids) + list(seq[:-1])
    cache = make_prompt_cache(target)
    logits = target(mx.array([full]), cache=cache)[0]
    base = len(prompt_ids) - 1

    for i, tok in enumerate(seq):
        row = logits[base + i]
        pred = int(mx.argmax(row).item())
        if pred == tok:
            continue
        top2 = mx.sort(row)[-2:]
        margin = float((top2[1] - top2[0]).item())
        assert margin < TIE_EPS, (
            f"{label} token {i} = {tok} but target greedy = {pred} "
            f"(top1-top2 margin {margin:.4f} >= {TIE_EPS}) — not a tie, "
            f"real corruption"
        )


def test_tree_and_linear_are_target_greedy(models):
    target, draft, tokenizer = models
    ids = _prompt_ids(
        tokenizer,
        "List the first five prime numbers, comma-separated.",
    )
    linear = _spec_sequence(draft, target, ids, tree_width=1)
    tree = _spec_sequence(draft, target, ids, tree_width=3)

    _assert_target_greedy("linear", linear, target, ids)
    _assert_target_greedy("tree", tree, target, ids)
