"""Live batched-decode canary for the BatchGenerator integration
(docs/batching-plan.md Phase 0).

Greedy batch_generate over identical prompts must produce identical,
coherent rows under ``safe_rope_patch``. Without the patch, the mlx
0.31.x ``mx.fast.rope`` B>1/L==1 bug silently corrupts every row but 0;
this is the regression gate for both the patch and its eventual removal.

Lives outside tests/integration/ to dodge its autouse MLX mock.
real_model; skipped in CI (`-m "not real_model"`) and when the model
isn't downloaded.

Run: uv run pytest tests/live/test_batching_real.py -m real_model -v
"""

import pytest

from olmlx.config import settings

MODEL = "mlx-community/Qwen3-4B-4bit"


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
def model_and_tokenizer():
    from mlx_lm import load

    return load(str(_model_dir()))


def _chat_tokens(tokenizer, content: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


def test_batched_rows_match_single_stream_reference(model_and_tokenizer):
    """B=4 identical greedy prompts: all rows identical to each other and
    to the unbatched generate() output."""
    from mlx_lm import batch_generate, generate

    from olmlx.engine.ropefix import safe_rope_patch

    model, tokenizer = model_and_tokenizer
    prompt = _chat_tokens(tokenizer, "Count from one to five in words.")
    max_tokens = 32

    reference = generate(model, tokenizer, prompt, max_tokens=max_tokens)

    with safe_rope_patch():
        result = batch_generate(
            model, tokenizer, [list(prompt) for _ in range(4)], max_tokens=max_tokens
        )

    assert len(result.texts) == 4
    # Rows must agree with each other — the rope bug corrupts rows >= 1
    # while leaving row 0 intact, so any corruption breaks this.
    assert all(t == result.texts[0] for t in result.texts), result.texts
    # And with the single-stream reference (greedy; same kernel family).
    assert result.texts[0] == reference, (result.texts[0], reference)


async def test_stream_completion_batched_concurrent_parity(
    model_and_tokenizer, monkeypatch
):
    """Engine-level Phase 1 gate: with OLMLX_BATCHING on, three concurrent
    `_stream_completion` calls ride one batch and each produces exactly the
    exclusive path's greedy output."""
    import asyncio

    from olmlx.config import settings as cfg
    from olmlx.engine.inference import _stream_completion
    from olmlx.engine.model_manager import LoadedModel
    from olmlx.utils.timing import TimingStats

    model, tokenizer = model_and_tokenizer
    lm = LoadedModel(
        name="batch-live-test",
        hf_path=str(_model_dir()),
        model=model,
        tokenizer=tokenizer,
    )
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Name three primary colors, briefly."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    async def run_once():
        stats = TimingStats()
        text = ""
        async for chunk in _stream_completion(lm, prompt_text, 48, {}, stats):
            if not chunk.get("done"):
                text += chunk["text"]
        assert stats.eval_count > 0
        assert stats.prompt_eval_count > 0
        return text

    try:
        monkeypatch.setattr(cfg, "batching", False)
        reference = await run_once()
        assert reference.strip()

        monkeypatch.setattr(cfg, "batching", True)
        outputs = await asyncio.gather(*(run_once() for _ in range(3)))
        # The batched path actually engaged...
        assert lm.batch_scheduler is not None
        # ...and matches the exclusive greedy output exactly.
        assert outputs == [reference] * 3, (outputs, reference)
    finally:
        if lm.batch_scheduler is not None:
            lm.batch_scheduler.close()


async def test_concurrent_batched_beats_serial_wall_clock(
    model_and_tokenizer, monkeypatch
):
    """The throughput win: 3 concurrent requests batched should finish in
    well under 3× a single request's time (decode is bandwidth-bound, so
    batched rows ride one weight read). Generous threshold to stay robust
    on a loaded machine."""
    import asyncio
    import time

    from olmlx.config import settings as cfg
    from olmlx.engine.inference import _stream_completion
    from olmlx.engine.model_manager import LoadedModel
    from olmlx.utils.timing import TimingStats

    model, tokenizer = model_and_tokenizer
    lm = LoadedModel(
        name="batch-bench-test",
        hf_path=str(_model_dir()),
        model=model,
        tokenizer=tokenizer,
    )
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write a short paragraph about rivers."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    async def run_once():
        stats = TimingStats()
        async for chunk in _stream_completion(lm, prompt_text, 96, {}, stats):
            pass
        return stats

    try:
        monkeypatch.setattr(cfg, "batching", True)
        await run_once()  # warmup (wired limit, kernel caches)

        t0 = time.monotonic()
        for _ in range(3):
            await run_once()
        serial = time.monotonic() - t0

        t0 = time.monotonic()
        await asyncio.gather(*(run_once() for _ in range(3)))
        concurrent = time.monotonic() - t0

        # Batched-concurrent must beat serial clearly; near-linear scaling
        # would be ~0.33×, we accept anything under 0.7×.
        assert concurrent < serial * 0.7, (
            f"concurrent {concurrent:.2f}s vs serial {serial:.2f}s"
        )
    finally:
        if lm.batch_scheduler is not None:
            lm.batch_scheduler.close()


def test_mixed_length_batch_stays_coherent(model_and_tokenizer):
    """Different-length prompts (exercises left-padding) each produce
    coherent, on-topic output."""
    from mlx_lm import batch_generate

    from olmlx.engine.ropefix import safe_rope_patch

    model, tokenizer = model_and_tokenizer
    prompts = [
        _chat_tokens(tokenizer, "What color is the sky on a clear day? One word."),
        _chat_tokens(
            tokenizer,
            "Please answer briefly and exactly: what is two plus two? "
            "Reply with the number word only.",
        ),
    ]

    with safe_rope_patch():
        result = batch_generate(
            model, tokenizer, [list(p) for p in prompts], max_tokens=24
        )

    assert "blue" in result.texts[0].lower(), result.texts[0]
    assert "four" in result.texts[1].lower(), result.texts[1]
