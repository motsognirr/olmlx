"""Tests for segment-aware tokenization and the segmented-prefill drive."""

import asyncio

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from olmlx.engine.inference import (
    _drive_segmented_prefill,
    _store_prompt_cache_after_generation,
    tokenize_segmented_chat,
)
from olmlx.engine.model_manager import LoadedModel
from olmlx.engine.prompt_cache.checkpoint import Segment, SegmentedPrompt
from olmlx.engine.prompt_cache.store import PromptCacheStore


class _FakeTokenizer:
    """Minimal tokenizer stub: tokens are 1 per character, role-tagged via
    the chat template adding a fixed-length wrapper per message."""

    bos_token_id = None
    eos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        # Each message expands to [role_marker, *content_chars, end_marker].
        # We encode role as 1=system, 2=user, 3=assistant. End marker is 9.
        ROLE = {"system": 1, "user": 2, "assistant": 3}
        out = []
        for m in messages:
            out.append(ROLE[m["role"]])
            out.extend(ord(c) for c in m["content"])
            out.append(9)
        return out


class _DummyModel:
    """Records call-by-call which tokens were fed in. Cache pretends to grow."""

    def __init__(self):
        self.calls: list[list[int]] = []

    def __call__(self, tokens, cache=None):
        # mlx-lm's contract: model(input_tokens_batched[None], cache=cache).
        # tokens is a 2D mx.array, batch dim first.
        self.calls.append(list(tokens.flatten().tolist()))
        # Grow each layer's "state" by len(tokens) — using KVCache primitives.
        S = tokens.shape[-1]
        keys = mx.zeros((1, 1, S, 4))
        values = mx.zeros((1, 1, S, 4))
        for layer in cache:
            layer.update_and_fetch(keys, values)
        # Return a dummy logits tensor of the right shape.
        return mx.zeros((1, S, 32))


def test_drive_runs_one_model_call_per_uncovered_segment():
    model = _DummyModel()
    store = PromptCacheStore(max_slots=8)
    sp = SegmentedPrompt(
        segments=[
            Segment(tokens=[1, 2, 3], role="system"),
            Segment(tokens=[4, 5], role="user"),
        ]
    )
    cache = [KVCache()]
    suffix = _drive_segmented_prefill(
        model=model, segmented=sp, cache=cache, store=store, eager_eval=False
    )
    assert len(model.calls) == 2, "one call per segment when starting cold"
    # First call processes the system segment, second processes the user segment.
    assert model.calls[0] == [1, 2, 3]
    assert model.calls[1] == [4]  # [4,5] minus the trailing token reserved for decode
    # Both boundaries should have been snapshotted into the store.
    assert store.fetch_nearest([1, 2, 3, 4, 5, 99]) is not None
    # The drive returns the suffix to be fed to stream_generate: the last token only.
    assert suffix == [5]


def test_drive_skips_segments_below_already_covered():
    model = _DummyModel()
    store = PromptCacheStore(max_slots=8)
    sp = SegmentedPrompt(
        segments=[
            Segment(tokens=[1, 2, 3], role="system"),
            Segment(tokens=[4, 5], role="user"),
        ]
    )
    cache = [KVCache()]
    # Pretend the cache is already warmed to position 3 (end of system).
    # Drive should NOT re-process [1,2,3]; only [4,5].
    suffix = _drive_segmented_prefill(
        model=model,
        segmented=sp,
        cache=cache,
        store=store,
        eager_eval=False,
        already_covered_tokens=3,
    )
    assert len(model.calls) == 1
    assert model.calls[0] == [4]  # [4,5] minus the trailing token reserved for decode
    assert suffix == [5]


def test_tokenize_segmented_chat_returns_one_segment_per_message():
    tok = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
    ]
    sp = tokenize_segmented_chat(tok, messages)
    assert isinstance(sp, SegmentedPrompt)
    assert len(sp.segments) == 2
    assert sp.segments[0].role == "system"
    assert sp.segments[1].role == "user"
    # System segment tokens = [1, 'A', 'B', 9] = [1, 65, 66, 9]
    assert sp.segments[0].tokens == [1, 65, 66, 9]
    # User segment tokens = [2, 'C', 'D', 9]
    assert sp.segments[1].tokens == [2, 67, 68, 9]


def test_tokenize_segmented_chat_flatten_matches_full_apply():
    """The concatenation of per-segment tokens must equal a full template
    application on all messages at once. If this fails the segment boundaries
    are off and snapshots would be misaligned."""
    tok = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
        {"role": "assistant", "content": "EF"},
        {"role": "user", "content": "GH"},
    ]
    sp = tokenize_segmented_chat(tok, messages)
    full = tok.apply_chat_template(messages)
    assert sp.flatten() == full


def test_setup_prompt_cache_drives_segments_when_lm_uses_checkpoint_path(monkeypatch):
    """When lm.uses_checkpoint_persistence is True and messages+tokenizer
    are provided, _setup_prompt_cache routes through the checkpoint path
    and the returned suffix is the one-token tail."""
    from olmlx.engine import inference as inference_mod
    from olmlx.engine.inference import _setup_prompt_cache

    tok = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
    ]
    full_tokens = tok.apply_chat_template(messages)
    model = _DummyModel()
    lm = LoadedModel(
        name="x",
        hf_path="y",
        model=model,
        tokenizer=tok,
        prompt_cache_store=PromptCacheStore(max_slots=4),
        uses_checkpoint_persistence=True,
        supports_cache_persistence=True,
    )
    # Patch _make_prompt_cache_for_lm to return a plain [KVCache()] for the
    # dummy model (avoids needing a real mlx-lm model object).
    monkeypatch.setattr(
        inference_mod, "_make_prompt_cache_for_lm", lambda m: [KVCache()]
    )
    gen_kwargs: dict = {}
    cs = asyncio.run(
        _setup_prompt_cache(
            lm,
            "ignored",
            gen_kwargs,
            prompt_tokens=full_tokens,
            cache_id="test",
            messages=messages,
            tokenizer=tok,
        )
    )
    # Cold start: drive ran both segments, returned single-token suffix.
    assert cs.cache_setup_done is True
    assert isinstance(cs.prompt, list) and len(cs.prompt) == 1
    assert cs.prompt[0] == full_tokens[-1]
    assert gen_kwargs["prompt_cache"] is not None
    assert cs.cache_read_tokens == 0
    assert cs.cache_creation_tokens == len(full_tokens)


def test_drive_does_not_snapshot_last_segment_boundary():
    """The last segment's KV depth is N-1 (the trailing token is reserved
    for stream_generate); storing a checkpoint with tokens length N would
    misalign future warm-starts (KV state would lag the claimed depth by 1)."""
    model = _DummyModel()
    store = PromptCacheStore(max_slots=8)
    sp = SegmentedPrompt(
        segments=[
            Segment(tokens=[1, 2, 3], role="system"),
            Segment(tokens=[4, 5], role="user"),
        ]
    )
    cache = [KVCache()]
    _drive_segmented_prefill(
        model=model, segmented=sp, cache=cache, store=store, eager_eval=False
    )
    # System boundary (3 tokens) is snapshotted; last boundary (5 tokens) is not.
    assert store.fetch_nearest([1, 2, 3, 99]) is not None, (
        "system-boundary checkpoint must be present"
    )
    # A query whose prefix exactly matches the full flat tokens must NOT
    # find a 5-token checkpoint — that would mean the bug is back.
    hit = store.fetch_nearest([1, 2, 3, 4, 5, 99])
    assert hit is not None
    state, _ = hit
    assert len(state.tokens) == 3, (
        f"expected 3-token (system) checkpoint as deepest hit, got "
        f"{len(state.tokens)}-token checkpoint — last-segment snapshot bug"
    )


def test_store_prompt_cache_after_generation_is_noop_for_checkpoint_path():
    """With uses_checkpoint_persistence=True, the post-generation store
    must not write to the cache store."""
    store = PromptCacheStore(max_slots=4)
    lm = LoadedModel(
        name="x",
        hf_path="y",
        model=None,
        tokenizer=None,
        prompt_cache_store=store,
        uses_checkpoint_persistence=True,
    )
    # Build a minimal call — only the early-return path is exercised, so
    # most args can be no-ops.  The real signature uses gen_kwargs (not
    # prompt_cache directly).
    asyncio.run(
        _store_prompt_cache_after_generation(
            lm=lm,
            gen_kwargs={"prompt_cache": [KVCache()]},
            full_prompt_tokens=[1, 2, 3],
            generated_tokens=[4],
            eval_count=1,
            cache_id="test",
        )
    )
    assert len(store) == 0
