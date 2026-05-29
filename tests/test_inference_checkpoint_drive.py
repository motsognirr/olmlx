"""Tests for segment-aware tokenization and the segmented-prefill drive."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from olmlx.engine.inference import _drive_segmented_prefill, tokenize_segmented_chat
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
