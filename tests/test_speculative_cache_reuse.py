"""Cross-request KV cache reuse for speculative decoding (issue #421).

A long agentic session re-prefills the entire growing conversation every
turn because prompt caching is gated off under speculative. These tests
cover the decoder-owned persistent cache store (`_SpecCacheStore`) and the
segmented-prefix reuse logic in `SpeculativeDecoder.prefill` /
`PromptLookupDecoder.prefill` that lets each turn prefill only the new suffix.
"""

from __future__ import annotations

import mlx.core as mx

from olmlx.engine.prompt_cache.checkpoint import Segment, SegmentedPrompt
from olmlx.engine.speculative import (
    PromptLookupDecoder,
    SpeculativeDecoder,
    _common_prefix_len,
    _spec_reuse_decision,
    _SpecCacheStore,
)
from tests.test_flash_speculative import MockModel


class TestCommonPrefixLen:
    def test_empty_inputs(self):
        assert _common_prefix_len([], [1, 2, 3]) == 0
        assert _common_prefix_len([1, 2, 3], []) == 0

    def test_full_prefix(self):
        assert _common_prefix_len([1, 2, 3], [1, 2, 3, 4, 5]) == 3

    def test_partial_prefix(self):
        assert _common_prefix_len([1, 2, 9, 4], [1, 2, 3, 4]) == 2

    def test_no_common(self):
        assert _common_prefix_len([9, 8], [1, 2]) == 0


class TestSpecCacheStore:
    def test_disabled_when_capacity_zero(self):
        store = _SpecCacheStore(capacity=0)
        assert not store.enabled()
        store.insert([1, 2, 3], payload="A")
        assert store.find([1, 2, 3]) is None

    def test_find_empty_store_returns_none(self):
        store = _SpecCacheStore(capacity=2)
        assert store.enabled()
        assert store.find([1, 2, 3]) is None

    def test_longest_prefix_lookup(self):
        store = _SpecCacheStore(capacity=4)
        store.insert([1, 2, 3], payload="short")
        store.insert([1, 2, 3, 4, 5], payload="long")
        hit = store.find([1, 2, 3, 4, 5, 6, 7])
        assert hit is not None
        entry, common = hit
        assert entry.payload == "long"
        assert common == 5

    def test_find_requires_nonzero_common_prefix(self):
        store = _SpecCacheStore(capacity=2)
        store.insert([1, 2, 3], payload="A")
        assert store.find([9, 9, 9]) is None

    def test_lru_eviction_at_capacity(self):
        store = _SpecCacheStore(capacity=2)
        store.insert([1, 1], payload="A")
        store.insert([2, 2], payload="B")
        store.insert([3, 3], payload="C")  # evicts oldest ("A")
        assert store.find([1, 1]) is None
        assert store.find([2, 2]) is not None
        assert store.find([3, 3]) is not None

    def test_find_promotes_to_mru(self):
        store = _SpecCacheStore(capacity=2)
        store.insert([1, 1], payload="A")
        store.insert([2, 2], payload="B")
        # Touch "A" so it becomes most-recently-used; inserting "C" must
        # then evict "B", not "A".
        assert store.find([1, 1]) is not None
        store.insert([3, 3], payload="C")
        assert store.find([1, 1]) is not None
        assert store.find([2, 2]) is None
        assert store.find([3, 3]) is not None

    def test_insert_identical_tokens_refreshes(self):
        store = _SpecCacheStore(capacity=2)
        store.insert([1, 2, 3], payload="old")
        store.insert([1, 2, 3], payload="new")
        hit = store.find([1, 2, 3])
        assert hit is not None
        entry, _ = hit
        assert entry.payload == "new"
        # Must not have consumed two slots for the same lineage.
        store.insert([4, 4], payload="other")
        assert store.find([1, 2, 3]) is not None
        assert store.find([4, 4]) is not None


class TestReuseDecision:
    """Pure decision logic: trimmable trims to the common prefix; a
    non-trimmable (hybrid ArraysCache) cache reuses only on a strict full
    prefix match, and never on an exact match it cannot back up from."""

    def test_trimmable_partial_branch_trims_to_common(self):
        # entry covers 8 tokens, but the new prompt diverges at 3.
        assert _spec_reuse_decision(True, [0] * 8, common=3, prompt_len=10) == (True, 3)

    def test_trimmable_strict_extension(self):
        assert _spec_reuse_decision(True, [0] * 5, common=5, prompt_len=10) == (True, 5)

    def test_trimmable_exact_match_backs_up_one(self):
        # Exact match: can't seed the first logit without a token to forward,
        # so reuse all but the last position.
        assert _spec_reuse_decision(True, [0] * 10, common=10, prompt_len=10) == (
            True,
            9,
        )

    def test_trimmable_single_token_prompt_is_useless(self):
        assert _spec_reuse_decision(True, [0] * 5, common=1, prompt_len=1) == (False, 0)

    def test_hybrid_strict_full_prefix_reuses(self):
        assert _spec_reuse_decision(False, [0] * 5, common=5, prompt_len=10) == (
            True,
            5,
        )

    def test_hybrid_partial_match_discarded(self):
        # Common prefix shorter than the stored entry → can't continue a
        # non-trimmable cache from there; fall back to fresh prefill.
        assert _spec_reuse_decision(False, [0] * 8, common=3, prompt_len=10) == (
            False,
            0,
        )

    def test_hybrid_exact_match_discarded(self):
        # Full match of the whole prompt: non-trimmable can't back up one
        # position to re-derive the seeding logit.
        assert _spec_reuse_decision(False, [0] * 10, common=10, prompt_len=10) == (
            False,
            0,
        )


def _segmented(*segment_token_lists: list[int]) -> SegmentedPrompt:
    return SegmentedPrompt(
        segments=[
            Segment(tokens=list(toks), role="user") for toks in segment_token_lists
        ]
    )


def _decoder(cache_slots: int) -> SpeculativeDecoder:
    target = MockModel(64, 16)
    draft = MockModel(64, 16)
    return SpeculativeDecoder(
        draft_model=draft,
        target_model=target,
        num_speculative_tokens=2,
        cache_slots=cache_slots,
    )


class TestSpeculativeDecoderReuse:
    def test_first_turn_stores_snapshot_at_deepest_interior_boundary(self):
        dec = _decoder(cache_slots=2)
        # 3 segments → boundaries 3, 5, 7. Deepest interior boundary (< 7) is 5.
        seg = _segmented([1, 2, 3], [4, 5], [6, 7])
        flat = seg.flatten()
        dec.prefill(mx.array([flat]), segmented=seg)

        assert dec._last_reused_tokens == 0
        hit = dec._cache_store.find(flat)
        assert hit is not None
        entry, common = hit
        assert entry.tokens == [1, 2, 3, 4, 5]  # deepest interior boundary
        assert common == 5

    def test_strict_extension_reuses_only_the_suffix(self):
        dec = _decoder(cache_slots=2)
        seg1 = _segmented([1, 2, 3], [4, 5], [6, 7])
        seg2 = _segmented([1, 2, 3], [4, 5], [6, 7, 8], [9, 10])
        dec.prefill(mx.array([seg1.flatten()]), segmented=seg1)
        dec.prefill(mx.array([seg2.flatten()]), segmented=seg2)
        # Reused up to turn-1's stored boundary (5 tokens).
        assert dec._last_reused_tokens == 5

    def test_reuse_is_token_identical_to_fresh_prefill(self):
        dec = _decoder(cache_slots=2)
        seg1 = _segmented([1, 2, 3], [4, 5], [6, 7])
        seg2 = _segmented([1, 2, 3], [4, 5], [6, 7, 8], [9, 10])

        # Fresh baseline on the same models with an empty store.
        dec._cache_store.clear()
        tok_fresh = dec.prefill(mx.array([seg2.flatten()]), segmented=seg2)
        assert dec._last_reused_tokens == 0

        # Prime turn 1, then reuse on turn 2.
        dec._cache_store.clear()
        dec.prefill(mx.array([seg1.flatten()]), segmented=seg1)
        tok_reuse = dec.prefill(mx.array([seg2.flatten()]), segmented=seg2)
        assert dec._last_reused_tokens == 5
        assert tok_reuse == tok_fresh

    def test_step_mutations_do_not_corrupt_stored_snapshot(self):
        dec = _decoder(cache_slots=2)
        seg = _segmented([1, 2, 3], [4, 5], [6, 7])
        dec.prefill(mx.array([seg.flatten()]), segmented=seg)
        hit = dec._cache_store.find(seg.flatten())
        assert hit is not None
        entry, _ = hit
        target_snap = entry.payload[0]
        stored_offset = target_snap[0].offset  # KV depth of the stored snapshot

        for _ in range(3):
            dec.step()

        # The working caches grew during decode; the stored snapshot must not.
        assert target_snap[0].offset == stored_offset == 5

    def test_disabled_store_falls_back_to_fresh(self):
        dec = _decoder(cache_slots=0)
        seg = _segmented([1, 2, 3], [4, 5], [6, 7])
        flat = seg.flatten()
        first = dec.prefill(mx.array([flat]), segmented=seg)
        assert isinstance(first, int)
        assert dec._last_reused_tokens == 0
        assert not dec._cache_store.enabled()
        assert dec._cache_store.find(flat) is None


def _pld(cache_slots: int) -> PromptLookupDecoder:
    return PromptLookupDecoder(
        target_model=MockModel(64, 16),
        num_speculative_tokens=3,
        max_ngram_size=2,
        min_ngram_size=1,
        lookup_window=8192,
        cache_slots=cache_slots,
    )


class TestPromptLookupDecoderReuse:
    def test_first_turn_stores_target_snapshot(self):
        dec = _pld(cache_slots=2)
        seg = _segmented([1, 2, 3], [4, 5], [6, 7])
        flat = seg.flatten()
        dec.prefill(mx.array([flat]), segmented=seg)
        assert dec._last_reused_tokens == 0
        hit = dec._cache_store.find(flat)
        assert hit is not None
        entry, common = hit
        assert entry.tokens == [1, 2, 3, 4, 5]
        assert common == 5

    def test_strict_extension_reuses_only_suffix(self):
        dec = _pld(cache_slots=2)
        seg1 = _segmented([1, 2, 3], [4, 5], [6, 7])
        seg2 = _segmented([1, 2, 3], [4, 5], [6, 7, 8], [9, 10])
        dec.prefill(mx.array([seg1.flatten()]), segmented=seg1)
        dec.prefill(mx.array([seg2.flatten()]), segmented=seg2)
        assert dec._last_reused_tokens == 5

    def test_reuse_is_token_identical_to_fresh(self):
        dec = _pld(cache_slots=2)
        seg1 = _segmented([1, 2, 3], [4, 5], [6, 7])
        seg2 = _segmented([1, 2, 3], [4, 5], [6, 7, 8], [9, 10])

        dec._cache_store.clear()
        tok_fresh = dec.prefill(mx.array([seg2.flatten()]), segmented=seg2)
        assert dec._last_reused_tokens == 0

        dec._cache_store.clear()
        dec.prefill(mx.array([seg1.flatten()]), segmented=seg1)
        tok_reuse = dec.prefill(mx.array([seg2.flatten()]), segmented=seg2)
        assert dec._last_reused_tokens == 5
        assert tok_reuse == tok_fresh

    def test_reuse_seeds_full_lookup_history(self):
        # The n-gram lookup table must hold the FULL prompt after a reuse
        # prefill, not just the freshly-prefilled suffix — otherwise PLD
        # can't match against the reused prefix.
        dec = _pld(cache_slots=2)
        seg1 = _segmented([1, 2, 3], [4, 5], [6, 7])
        seg2 = _segmented([1, 2, 3], [4, 5], [6, 7, 8], [9, 10])
        dec.prefill(mx.array([seg1.flatten()]), segmented=seg1)
        dec.prefill(mx.array([seg2.flatten()]), segmented=seg2)
        assert dec._last_reused_tokens == 5
        assert dec._tokens == seg2.flatten()

    def test_disabled_store_falls_back_to_fresh(self):
        dec = _pld(cache_slots=0)
        seg = _segmented([1, 2, 3], [4, 5], [6, 7])
        flat = seg.flatten()
        first = dec.prefill(mx.array([flat]), segmented=seg)
        assert isinstance(first, int)
        assert dec._last_reused_tokens == 0
        assert dec._cache_store.find(flat) is None


class TestStreamPlumbing:
    """``speculative_stream_generate`` must thread ``segmented`` into
    ``prefill`` so the reuse path activates end-to-end."""

    def _drain(self, decoder, tokens, segmented):
        import threading

        from olmlx.engine.speculative_stream import speculative_stream_generate

        cancel = threading.Event()
        gen = speculative_stream_generate(
            decoder,
            tokens,
            max_tokens=2,
            cancel_event=cancel,
            eos_token_id=None,
            tokenizer=None,
            segmented=segmented,
        )
        return list(gen)

    def test_segmented_reaches_prefill_and_drives_reuse(self):
        dec = _decoder(cache_slots=2)
        seg1 = _segmented([1, 2, 3], [4, 5], [6, 7])
        seg2 = _segmented([1, 2, 3], [4, 5], [6, 7, 8], [9, 10])

        out1 = self._drain(dec, seg1.flatten(), seg1)
        assert out1 and dec._last_reused_tokens == 0

        out2 = self._drain(dec, seg2.flatten(), seg2)
        assert out2 and dec._last_reused_tokens == 5

    def test_no_segmented_keeps_fresh_prefill(self):
        # Without ``segmented``, the stream must not pass the kwarg and the
        # decoder stays on the legacy fresh-prefill path.
        dec = _decoder(cache_slots=2)
        seg = _segmented([1, 2, 3], [4, 5], [6, 7])
        self._drain(dec, seg.flatten(), None)
        assert dec._last_reused_tokens == 0
        assert dec._cache_store.find(seg.flatten()) is None
