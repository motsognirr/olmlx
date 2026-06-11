"""Live end-to-end shard-quant test: calibrate a tiny model, generate (#377).

Lives outside tests/integration/ to dodge its autouse MLX mock. real_model;
skipped in CI (`-m "not real_model"`) and when the model isn't downloaded.

Run: uv run pytest tests/live/test_shard_quant_real.py -m real_model -v
"""

import pytest

from olmlx.config import settings

MODEL = "mlx-community/Qwen3-0.6B-4bit"


def _model_dir():
    from olmlx.models.store import _safe_dir_name

    return settings.models_dir / _safe_dir_name(MODEL)


def _model_present() -> bool:
    return (_model_dir() / "config.json").exists()


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not _model_present(),
        reason=f"{MODEL} not downloaded in {settings.models_dir}",
    ),
]


@pytest.fixture(scope="module")
def calibrated_model_path(tmp_path_factory):
    from olmlx.engine.shardquant_calibrate import calibrate_model_shard

    model_path = _model_dir()
    out = tmp_path_factory.mktemp("shard-calib")
    calibrate_model_shard(
        model_path=str(model_path),
        output_dir=out,
        num_samples=8,
        calibration_dataset="synthetic",
        bits=4,
        max_tokens_per_head=1024,
    )
    return model_path, out


def test_generation_stays_coherent_with_shard_cache(calibrated_model_path):
    from mlx_lm import load, stream_generate

    from olmlx.engine.shardquant_cache import ShardKVCache, make_shard_cache

    model_path, calib_dir = calibrated_model_path
    model, tokenizer = load(str(model_path))
    cache = make_shard_cache(model, calib_dir, bits=4)
    assert any(isinstance(c, ShardKVCache) for c in cache)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from one to ten in words."}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    text = ""
    for resp in stream_generate(
        model, tokenizer, prompt, max_tokens=96, prompt_cache=cache
    ):
        text += resp.text
    lowered = text.lower()
    # Loose coherence check: the model should produce several number words.
    hits = sum(w in lowered for w in ["one", "two", "three", "four", "five"])
    assert hits >= 3, f"incoherent output under shard quant: {text!r}"


def test_fused_decode_matches_tier1_generation(calibrated_model_path):
    """Greedy generation with the fused decode path (#377 Tier 2) must track
    the Tier-1 path on a real model (same packed state, same math)."""
    from mlx_lm import load, stream_generate

    from olmlx.engine.shardquant_cache import make_shard_cache

    model_path, calib_dir = calibrated_model_path
    model, tokenizer = load(str(model_path))
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "List the days of the week."}],
        add_generation_prompt=True,
        enable_thinking=False,
    )

    texts = {}
    for fused in (False, True):
        cache = make_shard_cache(model, calib_dir, bits=4, fused=fused)
        out = ""
        for resp in stream_generate(
            model, tokenizer, prompt, max_tokens=64, prompt_cache=cache
        ):
            out += resp.text
        texts[fused] = out

    # fp32 fused softmax vs mx.fast sdpa can diverge on argmax near-ties;
    # require the long common prefix that exact-parity math produces.
    a, b = texts[False], texts[True]
    common = sum(1 for x, y in zip(a, b) if x == y)
    assert common >= int(0.8 * min(len(a), len(b))), (a, b)
    assert "monday" in b.lower() or "tuesday" in b.lower(), b
