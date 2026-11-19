"""Cross-request KV cache reuse for speculative decoding (issue #421).

A long agentic session re-prefills the entire growing conversation every
turn because prompt caching is gated off under speculative. These tests
cover the decoder-owned persistent cache store (`_SpecCacheStore`) and the
segmented-prefix reuse logic in `SpeculativeDecoder.prefill` /
`PromptLookupDecoder.prefill` that lets each turn prefill only the new suffix.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from olmlx.engine.prompt_cache.checkpoint import Segment, SegmentedPrompt
from olmlx.engine.speculative import (
    PromptLookupDecoder,
    SpeculativeDecoder,
    _common_prefix_len,
    _spec_reuse_decision,
    _spec_snapshot_points,
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

    # --- #781: lookups and eviction for end-of-final-message snapshots ---

    def test_full_prefix_lookup_skips_longer_partial_overlaps(self):
        # An extend-only (hybrid) cache can't use an entry that diverges from
        # the prompt, however long the overlap: only a strict prefix counts.
        store = _SpecCacheStore(capacity=4)
        store.insert([1, 2, 3], payload="system", preamble=True)
        store.insert([1, 2, 3, 4, 5], payload="other conversation")
        hit = store.find([1, 2, 3, 4, 9, 9], full_prefix_only=True)
        assert hit is not None
        entry, common = hit
        assert entry.payload == "system"
        assert common == 3

    def test_full_prefix_lookup_ignores_an_exact_match(self):
        # An exact match leaves no token to forward for the seeding logit.
        store = _SpecCacheStore(capacity=2)
        store.insert([1, 2, 3], payload="A")
        assert store.find([1, 2, 3], full_prefix_only=True) is None

    def test_eviction_drops_a_superseded_tip_first(self):
        # A conversation's previous snapshot is a prefix of its new one; the
        # next turn extends the new one, so the old one goes first, not the
        # LRU entry (another conversation's tip).
        store = _SpecCacheStore(capacity=2)
        store.insert([7, 7], payload="other conversation")
        store.insert([1, 2], payload="turn 1")
        store.insert([1, 2, 3, 4], payload="turn 2")
        assert [e.payload for e in store._entries] == [
            "other conversation",
            "turn 2",
        ]

    def test_eviction_keeps_a_preamble_its_tip_extends(self):
        # The system-prompt snapshot is a prefix of its own conversation's
        # tip by design; it is not superseded by it.
        store = _SpecCacheStore(capacity=1)
        store.insert([1, 2], payload="system", preamble=True)
        store.insert([1, 2, 3], payload="turn 1")
        store.insert([1, 2, 3, 4], payload="turn 2")
        assert [e.payload for e in store._entries] == ["system", "turn 2"]

    def test_a_preamble_does_not_take_a_tip_slot(self):
        # Two interleaved conversations keep both tips and the shared system
        # prompt; capacity counts tips only.
        store = _SpecCacheStore(capacity=2)
        store.insert([1, 2], payload="system", preamble=True)
        store.insert([1, 2, 3], payload="conversation A")
        store.insert([1, 2, 4], payload="conversation B")
        assert [e.payload for e in store._entries] == [
            "system",
            "conversation A",
            "conversation B",
        ]

    def test_a_new_preamble_survives_a_store_full_of_tips(self):
        store = _SpecCacheStore(capacity=2)
        store.insert([5, 5], payload="tip X")
        store.insert([6, 6], payload="tip Y")
        store.insert([1, 2], payload="system", preamble=True)
        assert [e.payload for e in store._entries] == ["tip X", "tip Y", "system"]

    def test_back_to_back_conversations_keep_the_preamble(self):
        # One conversation per task: each new one reuses the system prompt
        # and the finished conversations' tips age out instead.
        store = _SpecCacheStore(capacity=2)
        store.insert([1, 2], payload="system", preamble=True)
        for task in (3, 4, 5, 6):
            hit = store.find([1, 2, task, 0], full_prefix_only=True)
            assert hit is not None and hit[0].payload == "system"
            store.insert([1, 2, task], payload=f"task {task}")
        assert {e.payload for e in store._entries} == {"system", "task 5", "task 6"}

    def test_only_the_newest_preamble_is_kept(self):
        store = _SpecCacheStore(capacity=2)
        store.insert([1, 2], payload="system A", preamble=True)
        store.insert([3, 4], payload="system B", preamble=True)
        assert [e.payload for e in store._entries] == ["system B"]

    def test_a_tip_with_the_preamble_tokens_keeps_it_a_preamble(self):
        # Same tokens, same KV state: one entry, and it stays in the
        # preamble slot instead of becoming an evictable tip.
        store = _SpecCacheStore(capacity=1)
        store.insert([1, 2], payload="system", preamble=True)
        store.insert([1, 2], payload="system as a tip")
        store.insert([1, 2, 3], payload="turn 1")
        store.insert([1, 2, 3, 4], payload="turn 2")
        assert [(e.payload, e.preamble) for e in store._entries] == [
            ("system as a tip", True),
            ("turn 2", False),
        ]

    def test_a_preamble_does_not_supersede_a_tip(self):
        # A system-prompt snapshot is no conversation's continuation, even if
        # a tip happens to be a prefix of it; the LRU tip goes instead.
        store = _SpecCacheStore(capacity=2)
        store.insert([5, 5], payload="tip B")
        store.insert([1, 2], payload="tip A")
        store.insert([1, 2, 3], payload="system", preamble=True)
        store.insert([7, 7], payload="tip C")
        assert {e.payload for e in store._entries} == {"tip A", "system", "tip C"}

    def test_a_new_snapshot_is_never_superseded_by_an_older_one(self):
        # A rewound conversation's new tip is a prefix of its old, deeper
        # tip; an extend-only cache can only continue the new one.
        store = _SpecCacheStore(capacity=2)
        store.insert([5, 5], payload="other conversation")
        store.insert([1, 2, 3, 4], payload="before rewind")
        store.insert([1, 2], payload="after rewind")
        assert [e.payload for e in store._entries] == [
            "before rewind",
            "after rewind",
        ]

    def test_a_lookup_does_not_unsupersede_an_older_snapshot(self):
        # A lookup that promotes turn 1 above turn 2 (a request that branched
        # at turn 1 and stored nothing, e.g. cancelled) must not make turn 2,
        # its deeper successor, the eviction victim.
        store = _SpecCacheStore(capacity=2)
        store.insert([1, 2], payload="turn 1")
        store.insert([1, 2, 3], payload="turn 2")
        hit = store.find([1, 2, 9], full_prefix_only=True)
        assert hit is not None and hit[0].payload == "turn 1"
        store.insert([7, 7], payload="other conversation")
        assert [e.payload for e in store._entries] == [
            "turn 2",
            "other conversation",
        ]


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


# --- #781: snapshot at the end of the final message --------------------------

_GEN = [8, 9]  # the generation prompt; the next turn replaces it with the reply
_SYSTEM = [1, 2, 3]
_U1 = [4, 5]
_A1 = [8, 9, 20, 21]  # rendered reply: starts where the generation prompt did
_U2 = [22, 23]
_A2 = [8, 9, 24]
_U3 = [25]


def _turn(*messages: list[int]) -> SegmentedPrompt:
    """A chat prompt with one segment per message and the generation prompt
    appended to the last one, as ``tokenize_segmented_chat`` builds it."""
    segs = [Segment(tokens=list(m), role="user") for m in messages]
    segs[-1] = Segment(tokens=list(messages[-1]) + _GEN, role="user")
    return SegmentedPrompt(segments=segs, trailing_tokens=len(_GEN))


_TURN1 = _turn(_SYSTEM, _U1)  # final message ends at 5
_TURN2 = _turn(_SYSTEM, _U1, _A1, _U2)  # ... at 11
_TURN3 = _turn(_SYSTEM, _U1, _A1, _U2, _A2, _U3)


class _ExtendOnlyCache(KVCache):
    """A KVCache the reuse logic classifies as non-trimmable (it goes by class
    name), standing in for a hybrid GatedDeltaNet cache."""


class _ExtendOnlyModel(MockModel):
    def make_cache(self):
        return [_ExtendOnlyCache() for _ in self.layers]


def _extend_only_decoder(cache_slots: int = 2) -> SpeculativeDecoder:
    return SpeculativeDecoder(
        draft_model=_ExtendOnlyModel(64, 16),
        target_model=_ExtendOnlyModel(64, 16),
        num_speculative_tokens=2,
        cache_slots=cache_slots,
    )


def _prefill(dec, seg: SegmentedPrompt) -> int:
    return dec.prefill(mx.array([seg.flatten()]), segmented=seg)


def _stored(dec) -> list[tuple[list[int], bool]]:
    return [(e.tokens, e.preamble) for e in dec._cache_store._entries]


class TestFinalMessageEndSnapshot:
    def test_only_the_first_segment_end_is_flagged_as_the_preamble(self, monkeypatch):
        # The flag follows the offset, not the position: an extra interior
        # point must be stored as a tip, not replace the system prompt.
        seg = _turn(_SYSTEM, _U1, _A1, _U2)
        monkeypatch.setattr(
            SegmentedPrompt,
            "snapshot_points",
            lambda self, covered, include_first_segment=True: [
                (3, "system"),
                (5, "user"),
                (11, "user"),
            ],
        )
        assert _spec_snapshot_points(seg, seg.flatten(), 0, extend_only=True) == [
            (3, True),
            (5, False),
            (11, False),
        ]

    def test_classic_snapshots_the_end_of_the_final_message(self):
        dec = _decoder(cache_slots=2)
        _prefill(dec, _TURN1)
        # A trimmable cache serves the system prompt from any deeper
        # snapshot, so it keeps no separate one.
        assert _stored(dec) == [([1, 2, 3, 4, 5], False)]

    def test_classic_next_turn_prefills_only_the_reply_and_new_message(self):
        dec = _decoder(cache_slots=2)
        _prefill(dec, _TURN1)
        _prefill(dec, _TURN2)
        assert dec._last_reused_tokens == 5
        _prefill(dec, _TURN3)
        assert dec._last_reused_tokens == 11

    def test_classic_reuse_is_token_identical_to_fresh_prefill(self):
        dec = _decoder(cache_slots=2)
        fresh = _prefill(dec, _TURN2)
        dec._cache_store.clear()
        _prefill(dec, _TURN1)
        reused = _prefill(dec, _TURN2)
        assert dec._last_reused_tokens == 5
        assert reused == fresh

    def test_classic_single_message_prompt_is_snapshotted(self):
        dec = _decoder(cache_slots=2)
        first = _turn([1, 2, 3, 4])
        _prefill(dec, first)
        assert _stored(dec) == [([1, 2, 3, 4], False)]
        _prefill(dec, _turn([1, 2, 3, 4], [8, 9, 20], [21]))
        assert dec._last_reused_tokens == 4

    def test_pld_snapshots_the_end_of_the_final_message(self):
        dec = _pld(cache_slots=2)
        _prefill(dec, _TURN1)
        assert _stored(dec) == [([1, 2, 3, 4, 5], False)]
        _prefill(dec, _TURN2)
        assert dec._last_reused_tokens == 5
        assert dec._tokens == _TURN2.flatten()

    def test_extend_only_cold_start_also_keeps_the_system_prompt(self):
        dec = _extend_only_decoder()
        _prefill(dec, _TURN1)
        assert _stored(dec) == [([1, 2, 3], True), ([1, 2, 3, 4, 5], False)]

    def test_extend_only_new_conversation_reuses_the_system_prompt(self):
        # The other conversation's tip overlaps the new prompt further
        # ([1, 2, 3, 4]) but diverges from it, so it is unusable.
        dec = _extend_only_decoder()
        _prefill(dec, _TURN1)
        _prefill(dec, _turn(_SYSTEM, [4, 6]))
        assert dec._last_reused_tokens == 3

    def test_extend_only_agent_loop_keeps_the_system_prompt(self):
        dec = _extend_only_decoder(cache_slots=2)
        _prefill(dec, _TURN1)
        _prefill(dec, _TURN2)
        assert dec._last_reused_tokens == 5
        _prefill(dec, _TURN3)
        assert dec._last_reused_tokens == 11
        # Turn 1's tip is superseded and gone; turns 2 and 3 fill the two tip
        # slots, beside the system prompt.
        assert _stored(dec) == [
            ([1, 2, 3], True),
            (_TURN2.flatten()[:11], False),
            (_TURN3.flatten()[: _TURN3.final_message_end()], False),
        ]
        _prefill(dec, _turn(_SYSTEM, [7]))
        assert dec._last_reused_tokens == 3

    def test_extend_only_two_interleaved_conversations_both_reuse(self):
        dec = _extend_only_decoder(cache_slots=2)
        b1 = _turn(_SYSTEM, [6])
        b2 = _turn(_SYSTEM, [6], [8, 9, 30], [31])
        _prefill(dec, _TURN1)
        _prefill(dec, b1)
        _prefill(dec, _TURN2)
        assert dec._last_reused_tokens == 5
        _prefill(dec, b2)
        assert dec._last_reused_tokens == 4
        _prefill(dec, _TURN3)
        assert dec._last_reused_tokens == 11

    def test_pld_extend_only_cold_start_also_keeps_the_system_prompt(self):
        dec = PromptLookupDecoder(
            target_model=_ExtendOnlyModel(64, 16),
            num_speculative_tokens=3,
            max_ngram_size=2,
            min_ngram_size=1,
            lookup_window=8192,
            cache_slots=2,
        )
        _prefill(dec, _TURN1)
        assert _stored(dec) == [([1, 2, 3], True), ([1, 2, 3, 4, 5], False)]
        _prefill(dec, _turn(_SYSTEM, [4, 6]))
        assert dec._last_reused_tokens == 3
