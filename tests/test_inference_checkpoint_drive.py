"""Tests for segment-aware tokenization and the segmented-prefill drive."""

import asyncio

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

import pytest

from olmlx.engine.inference import (
    _drive_segmented_prefill,
    _message_boundary_token_ids,
    _store_prompt_cache_after_generation,
    tokenize_segmented_chat,
)
from olmlx.engine.model_manager import LoadedModel
from olmlx.engine.prompt_cache.checkpoint import Segment, SegmentedPrompt
from olmlx.engine.prompt_cache.store import PromptCacheStore


class _FakeTokenizer:
    """Minimal tokenizer stub: tokens are 1 per character, role-tagged via
    the chat template adding a fixed-length wrapper per message.

    Token 9 is the end-of-message marker (``eos_token_id``), so
    ``_message_boundary_token_ids`` returns ``{9}`` and the EOM-boundary
    segmentation strategy correctly detects one boundary per message.
    """

    bos_token_id = None
    eos_token_id = 9

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


def test_drive_two_segment_cold_start():
    """Two segments, cold start: one chunk per side of the single interior
    boundary, one snapshot at that boundary. With only one interior
    boundary this case is identical under both the per-segment and the
    deepest-only chunking strategies."""
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
        model=model, segmented=sp, cache=cache, store=store
    )
    # Two chunks split at the single interior boundary (depth 3): the
    # system tokens, then the user segment minus its reserved trailing
    # token.
    assert model.calls == [[1, 2, 3], [4]]
    # System boundary (depth 3) is snapshotted; the final boundary
    # (depth 5 = len(flat)) is always skipped — KV depth there is 4
    # because the trailing token is reserved for stream_generate.
    assert len(store) == 1
    hit = store.fetch_nearest([1, 2, 3, 4, 5, 99])
    assert hit is not None
    state, _ = hit
    assert len(state.tokens) == 3
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
        already_covered_tokens=3,
    )
    assert len(model.calls) == 1
    assert model.calls[0] == [4]  # [4,5] minus the trailing token reserved for decode
    assert suffix == [5]


def test_tokenize_segmented_chat_accepts_extra_eom_boundaries():
    """For templates like gpt-oss Harmony that inject a baseline
    system/developer message, the EOM-boundary count exceeds the input
    message count by 1+. The function must accept extras and treat them
    as preamble segments rather than falling back to a single segment."""

    class _HarmonyLikeTokenizer:
        eos_token_id = 9
        unk_token_id = None

        def convert_tokens_to_ids(self, tok_str):
            return None

        def apply_chat_template(self, messages, **kwargs):
            # Template emits: <PREAMBLE_ROLE 7> ... <EOM 9>, then the
            # caller's messages each as <ROLE_TAG> ... <EOM 9>.
            ROLE = {"system": 1, "user": 2, "assistant": 3}
            out = [7, 100, 101, 9]  # preamble emitted by the template
            for m in messages:
                out.append(ROLE[m["role"]])
                out.extend(ord(c) for c in m["content"])
                out.append(9)
            return out

    tok = _HarmonyLikeTokenizer()
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
    ]
    full = tok.apply_chat_template(messages)
    sp = tokenize_segmented_chat(tok, messages, full_tokens=list(full))
    # 3 EOMs in `full`: one for the template-injected preamble plus one
    # per caller message. Result: 3 segments.
    assert len(sp.segments) == 3, (
        f"expected 3 segments (1 preamble + 2 caller), got {len(sp.segments)}"
    )
    # The flatten() must still equal the input.
    assert sp.flatten() == list(full)
    # Preamble role takes the first caller-supplied role.
    assert sp.segments[0].role == "system"
    assert sp.segments[1].role == "system"  # caller's first message
    assert sp.segments[2].role == "user"  # caller's second message


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
    _drive_segmented_prefill(model=model, segmented=sp, cache=cache, store=store)
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


def test_message_boundary_token_ids_returns_eos_token_id():
    """_message_boundary_token_ids should return {eos_token_id} when set."""
    tok = _FakeTokenizer()
    assert _message_boundary_token_ids(tok) == {9}


def test_message_boundary_token_ids_returns_empty_when_none():
    """_message_boundary_token_ids should return empty set when eos_token_id is None."""

    class _NoEos:
        eos_token_id = None

    assert _message_boundary_token_ids(_NoEos()) == set()


def test_message_boundary_token_ids_picks_up_harmony_end_token():
    """For gpt-oss / Harmony format, eos_token_id is <|return|> (end of
    generation, never in input prompts). The actual per-message marker
    is <|end|>. The helper must include any known per-template marker
    that the tokenizer can resolve to a real (non-UNK) token id."""

    class _HarmonyTokenizer:
        eos_token_id = 200002  # <|return|>
        unk_token_id = None

        def convert_tokens_to_ids(self, tok_str):
            mapping = {
                "<|end|>": 200007,
                "<end_of_turn>": None,  # not present in this template
            }
            return mapping.get(tok_str)

    ids = _message_boundary_token_ids(_HarmonyTokenizer())
    assert ids == {200002, 200007}, (
        f"expected eos + <|end|>, got {ids} — Harmony-format models won't "
        f"see message boundaries without the extra lookup"
    )


def test_message_boundary_token_ids_skips_unk_lookups():
    """Tokenizers return unk_token_id for unknown strings — these must
    NOT pollute the boundary set."""

    class _UnkReturningTokenizer:
        eos_token_id = 9
        unk_token_id = 0

        def convert_tokens_to_ids(self, _):
            return self.unk_token_id

    ids = _message_boundary_token_ids(_UnkReturningTokenizer())
    assert ids == {9}, f"unk leaked into boundary set: {ids}"


def test_tokenize_segmented_chat_falls_back_when_no_eos():
    """Without eos_token_id, falls back to a single segment."""

    class _NoEosTok:
        eos_token_id = None

        def apply_chat_template(self, messages, **kwargs):
            out = []
            for m in messages:
                out.extend(ord(c) for c in m["content"])
            return out

    tok = _NoEosTok()
    messages = [
        {"role": "system", "content": "Hi"},
        {"role": "user", "content": "Bye"},
    ]
    sp = tokenize_segmented_chat(tok, messages)
    assert len(sp.segments) == 1
    assert sp.segments[0].role == "user"
    assert sp.flatten() == tok.apply_chat_template(messages)


def test_tokenize_segmented_chat_handles_batchencoding_dict_return():
    """Some tokenizers return a BatchEncoding mapping with input_ids,
    not a flat list. The helper must extract input_ids transparently."""

    class _DictReturnTok:
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            # Encode the same way as _FakeTokenizer but wrap in a dict
            # to mimic a BatchEncoding-like return.
            ROLE = {"system": 1, "user": 2, "assistant": 3}
            out = []
            for m in messages:
                out.append(ROLE[m["role"]])
                out.extend(ord(c) for c in m["content"])
                out.append(9)
            return {"input_ids": out, "attention_mask": [1] * len(out)}

    tok = _DictReturnTok()
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
    ]
    sp = tokenize_segmented_chat(tok, messages)
    assert len(sp.segments) == 2
    assert sp.flatten() == [1, 65, 66, 9, 2, 67, 68, 9]


def test_tokenize_segmented_chat_handles_nested_list_return():
    """Some tokenizers return a batch-of-1 nested list. Unwrap it."""

    class _NestedListTok:
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            ROLE = {"system": 1, "user": 2}
            out = []
            for m in messages:
                out.append(ROLE[m["role"]])
                out.extend(ord(c) for c in m["content"])
                out.append(9)
            return [out]  # batch dim of 1

    tok = _NestedListTok()
    messages = [{"role": "system", "content": "X"}, {"role": "user", "content": "Y"}]
    sp = tokenize_segmented_chat(tok, messages)
    assert len(sp.segments) == 2
    assert sp.flatten() == [1, ord("X"), 9, 2, ord("Y"), 9]


def test_tokenize_segmented_chat_full_tokens_bypasses_apply_chat_template():
    """When full_tokens is provided, the function MUST use those tokens
    directly and not call apply_chat_template — the caller's tokenization
    is authoritative.

    This regression test guards against the BOS-handling mismatch where
    apply_chat_template(tokenize=True) and tokenize_for_cache produce
    different leading-token sequences on tokenizers like Llama 3's,
    which would silently disable the checkpoint path for every request.
    """

    class _BosMismatchTok:
        """Returns a token list with one extra leading token when called
        directly — simulates the apply_chat_template-vs-tokenize_for_cache
        BOS mismatch."""

        eos_token_id = 9
        applied = False

        def apply_chat_template(self, messages, **kwargs):
            type(self).applied = True
            ROLE = {"system": 1, "user": 2}
            out = [999]  # leading BOS that the caller's tokenization lacks
            for m in messages:
                out.append(ROLE[m["role"]])
                out.extend(ord(c) for c in m["content"])
                out.append(9)
            return out

    tok = _BosMismatchTok()
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
    ]
    # Caller passes its own tokenization WITHOUT the leading 999.
    caller_tokens = [1, 65, 66, 9, 2, 67, 68, 9]
    sp = tokenize_segmented_chat(tok, messages, full_tokens=caller_tokens)
    assert tok.applied is False, (
        "apply_chat_template must not be called when full_tokens is provided"
    )
    assert sp.flatten() == caller_tokens, (
        "segments must reconstruct exactly the caller's tokens"
    )
    assert len(sp.segments) == 2


def test_tokenize_segmented_chat_coerces_arraylike_to_list_of_int():
    """Tokenizers that return numpy-style array-likes (anything with
    int elements) must NOT silently fall back to an empty segment."""

    class _IntArray:
        """Minimal array-like: iterable + indexable + len, returns ints."""

        def __init__(self, data):
            self._data = data

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, i):
            return self._data[i]

    class _ArrayLikeTok:
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            ROLE = {"system": 1, "user": 2}
            out = []
            for m in messages:
                out.append(ROLE[m["role"]])
                out.extend(ord(c) for c in m["content"])
                out.append(9)
            return _IntArray(out)

    tok = _ArrayLikeTok()
    messages = [{"role": "system", "content": "AB"}, {"role": "user", "content": "CD"}]
    sp = tokenize_segmented_chat(tok, messages)
    assert len(sp.segments) == 2, (
        "array-like tokenizer output must be coerced, not dropped"
    )
    assert sp.flatten() == [1, 65, 66, 9, 2, 67, 68, 9]


def test_tokenize_segmented_chat_returns_empty_segment_for_unrecognised_shape():
    """Anything we can't unpack to list[int] becomes an empty single
    segment so the caller's prompt_tokens != segmented.flatten() check
    catches it cleanly."""

    class _StringReturnTok:
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            return "not a token list"

    tok = _StringReturnTok()
    sp = tokenize_segmented_chat(tok, [{"role": "user", "content": "hi"}])
    assert len(sp.segments) == 1
    assert sp.segments[0].tokens == []


@pytest.mark.slow
def test_tokenize_segmented_chat_real_qwen3_5_template():
    """Real-template test: Qwen3.5 chat template should produce one
    segment per message, with token boundaries at <|im_end|> positions."""
    try:
        from mlx_lm import load

        _, tok = load("mlx-community/Qwen3.5-0.8B-MLX-4bit")
    except Exception as e:
        pytest.skip(f"model not available: {e}")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello."},
    ]
    sp = tokenize_segmented_chat(
        tok, messages, tokenize=True, add_generation_prompt=True
    )
    assert len(sp.segments) == 2
    assert sp.segments[0].role == "system"
    assert sp.segments[1].role == "user"
    # Flatten must match full tokenization.
    full = list(
        tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    )
    assert sp.flatten() == full


def test_setup_prompt_cache_single_segment_request_still_uses_existing_checkpoint(
    monkeypatch,
):
    """Regression for aider Finding on aabc3aa: a single-message request
    that is a strict extension of a previously-stored checkpoint must
    still warm-start. The previous code short-circuited on the
    single-segment guard before consulting fetch_nearest, so
    single-message requests got zero benefit from existing checkpoints.
    """
    from olmlx.engine.inference import _setup_prompt_cache
    from olmlx.engine.model_manager import LoadedModel
    from olmlx.engine.prompt_cache.checkpoint import (
        snapshot_cache_for_persistence,
    )
    from olmlx.engine.prompt_cache.state import CachedPromptState
    from olmlx.engine.prompt_cache.store import PromptCacheStore

    tok = _FakeTokenizer()
    # Single-message request: just one user turn — yields 1 segment under
    # the EOM split.
    messages = [{"role": "user", "content": "EXTRA"}]
    full_tokens = tok.apply_chat_template(messages)
    # Pre-populate the store with a checkpoint that is a strict prefix
    # of `full_tokens`. (In real usage this came from an earlier multi-
    # segment request that shared the prefix.)
    store = PromptCacheStore(max_slots=4)
    prefix_len = 3
    stored_cache = [KVCache()]
    stored_cache[0].update_and_fetch(
        mx.zeros((1, 1, prefix_len, 4)), mx.zeros((1, 1, prefix_len, 4))
    )
    store.insert_checkpoint(
        CachedPromptState(
            tokens=full_tokens[:prefix_len],
            cache=snapshot_cache_for_persistence(stored_cache, eager_eval=False),
            cache_type="system",
            is_checkpoint=True,
        )
    )

    model = _DummyModel()
    lm = LoadedModel(
        name="x",
        hf_path="y",
        model=model,
        tokenizer=tok,
        prompt_cache_store=store,
        uses_checkpoint_persistence=True,
        supports_cache_persistence=True,
    )
    monkeypatch.setattr(
        "olmlx.engine.inference._make_prompt_cache_for_lm",
        lambda lm: [KVCache()],
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
    assert cs.cache_setup_done is True
    assert cs.cache_read_tokens == prefix_len, (
        f"expected cache_read={prefix_len} (warm-start), got "
        f"{cs.cache_read_tokens} — single-segment request didn't take "
        f"the hit"
    )
    # Drive should have run on the un-covered tokens only.
    assert len(model.calls) == 1
    assert model.calls[0] == full_tokens[prefix_len : len(full_tokens) - 1], (
        "drive should process only tokens between hit_depth and the "
        "reserved trailing token"
    )


class _MixedRotatingArraysModel:
    """Dummy model for Qwen3-Next-style mixed RotatingKVCache + ArraysCache
    layouts (#396).  Advances each layer per its real update API so the
    checkpoint snapshot captures meaningful state.
    """

    def __init__(self):
        self.calls: list[list[int]] = []

    def __call__(self, tokens, cache=None):
        self.calls.append(list(tokens.flatten().tolist()))
        S = tokens.shape[-1]
        for layer in cache:
            if isinstance(layer, RotatingKVCache):
                keys = mx.zeros((1, 1, S, 4))
                values = mx.zeros((1, 1, S, 4))
                layer.update_and_fetch(keys, values)
            elif isinstance(layer, ArraysCache):
                # Simulate SSM state evolution: overwrite each slot with a
                # tensor whose values encode how many tokens have flowed
                # through.  The exact values don't matter for the round-trip
                # check — what matters is that deepcopy + continued mutation
                # works without crashing.
                for i in range(len(layer.cache)):
                    prev = layer.cache[i]
                    base = mx.ones((1, 4, 8)) * float(S)
                    layer.cache[i] = base if prev is None else prev + base
        return mx.zeros((1, S, 32))


def test_drive_handles_mixed_rotating_arrays_layout():
    """Regression for #396 (Qwen3-Next mixed layout).

    The checkpoint path must drive prefill correctly over a layer list
    that mixes RotatingKVCache (SWA) and ArraysCache (Gated-DeltaNet SSM)
    in alternating order.  ``snapshot_cache_for_persistence`` deepcopies
    each layer independently, so the joint snapshot survives a continued
    forward pass on the snapshot without disturbing the original.
    """
    from olmlx.engine.prompt_cache.checkpoint import (
        snapshot_cache_for_persistence,
    )

    model = _MixedRotatingArraysModel()
    store = PromptCacheStore(max_slots=8)
    sp = SegmentedPrompt(
        segments=[
            Segment(tokens=[1, 2, 3, 4], role="system"),
            Segment(tokens=[5, 6], role="user"),
        ]
    )
    cache = [
        RotatingKVCache(max_size=32, keep=2),
        ArraysCache(2),
        RotatingKVCache(max_size=32, keep=2),
    ]
    suffix = _drive_segmented_prefill(
        model=model, segmented=sp, cache=cache, store=store
    )
    # Two segments: one model call per uncovered segment, last token
    # reserved for stream_generate's decode init.
    assert model.calls == [[1, 2, 3, 4], [5]]
    assert suffix == [6]
    # System-boundary checkpoint should be in the store; final-boundary
    # snapshot is deliberately skipped (see test above).
    hit = store.fetch_nearest([1, 2, 3, 4, 5, 6, 99])
    assert hit is not None
    state, _ = hit
    assert len(state.tokens) == 4, "deepest hit should be the system boundary"
    # Round-trip: warm-start by deepcopying the stored snapshot and feeding
    # one more token.  This is what _setup_via_checkpoint_path does for a
    # request whose prompt strictly extends the stored prefix.
    warm = snapshot_cache_for_persistence(state.cache, eager_eval=True)
    assert [type(layer).__name__ for layer in warm] == [
        "RotatingKVCache",
        "ArraysCache",
        "RotatingKVCache",
    ]
    # Rotating layers carry the snapshot's offset; arrays slots are populated.
    rotating_layers = [layer for layer in warm if isinstance(layer, RotatingKVCache)]
    assert all(layer.offset == 4 for layer in rotating_layers)
    arrays_layer = next(layer for layer in warm if isinstance(layer, ArraysCache))
    assert all(slot is not None for slot in arrays_layer.cache)
    # Drive the uncovered tail through the warmed cache.
    warm_model = _MixedRotatingArraysModel()
    suffix2 = _drive_segmented_prefill(
        model=warm_model,
        segmented=sp,
        cache=warm,
        store=PromptCacheStore(max_slots=4),
        already_covered_tokens=4,
    )
    # Only the user segment's prefill tokens get fed; the trailing token
    # is reserved.
    assert warm_model.calls == [[5]]
    assert suffix2 == [6]
    assert all(layer.offset == 5 for layer in rotating_layers)


def test_drive_uses_two_chunks_and_one_snapshot_for_multi_segment_request():
    """Regression for chunking-induced GDN drift (issue: Qwen3.6 MoE
    repetition on 2nd request).

    For a multi-turn prompt the drive must:
      - feed the uncovered prefix up to the deepest interior message
        boundary in ONE model call,
      - take ONE snapshot there,
      - feed the rest (up to the reserved trailing token) in ONE more
        model call.

    Per-segment chunking is what was causing the bug: mlx-lm's
    ``gated_delta_kernel`` is not exactly chunking-invariant, and the
    error compounded with conversation depth until MoE routing
    thresholds were crossed.
    """
    model = _DummyModel()
    store = PromptCacheStore(max_slots=8)
    # 4 segments — sys (covered by an earlier hit), user1, asst1, user2.
    sp = SegmentedPrompt(
        segments=[
            Segment(tokens=[1, 2, 3], role="system"),
            Segment(tokens=[4, 5, 6], role="user"),
            Segment(tokens=[7, 8, 9, 10], role="assistant"),
            Segment(tokens=[11, 12, 13], role="user"),
        ]
    )
    cache = [KVCache()]
    cache[0].update_and_fetch(mx.zeros((1, 1, 3, 4)), mx.zeros((1, 1, 3, 4)))
    suffix = _drive_segmented_prefill(
        model=model,
        segmented=sp,
        cache=cache,
        store=store,
        already_covered_tokens=3,
    )
    # Deepest interior boundary > 3 and < 13 is the assistant boundary (10).
    # Chunk 1: tokens [4..10) = [4,5,6,7,8,9,10].
    # Chunk 2: tokens [10..12) = [11,12]  (13 reserved for stream_generate).
    assert model.calls == [[4, 5, 6, 7, 8, 9, 10], [11, 12]], (
        "drive must chunk the uncovered tail at the deepest interior "
        "boundary only, not per segment"
    )
    assert suffix == [13]
    # Only ONE snapshot, taken at the assistant boundary (depth 10).
    assert len(store) == 1, (
        f"expected exactly one snapshot at the deepest interior boundary, "
        f"got {len(store)}"
    )
    hit = store.fetch_nearest([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99])
    assert hit is not None
    state, _ = hit
    assert len(state.tokens) == 10, (
        f"snapshot should be at the assistant boundary (depth 10), "
        f"got {len(state.tokens)}"
    )
    assert state.cache_type == "assistant"


def test_drive_no_snapshot_when_only_final_segment_remains():
    """Warm-start where everything but the final segment is already covered
    yields a single chunk and zero snapshots — there is no usable interior
    boundary strictly less than ``len(flat)``."""
    model = _DummyModel()
    store = PromptCacheStore(max_slots=8)
    sp = SegmentedPrompt(
        segments=[
            Segment(tokens=[1, 2, 3], role="system"),
            Segment(tokens=[4, 5, 6], role="user"),
            Segment(tokens=[7, 8, 9], role="assistant"),
            Segment(tokens=[10, 11, 12], role="user"),
        ]
    )
    cache = [KVCache()]
    cache[0].update_and_fetch(mx.zeros((1, 1, 9, 4)), mx.zeros((1, 1, 9, 4)))
    suffix = _drive_segmented_prefill(
        model=model,
        segmented=sp,
        cache=cache,
        store=store,
        already_covered_tokens=9,  # end of assistant
    )
    # Single chunk = [10, 11] (12 reserved). No snapshot.
    assert model.calls == [[10, 11]]
    assert suffix == [12]
    assert len(store) == 0
