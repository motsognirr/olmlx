"""Tests for the consolidated stop-sequence scan (batching plan Phase 2).

One incremental scanner (StopScanner) shared by the exclusive and batched
streaming paths, and one whole-text helper (truncate_at_stop) for the
non-streaming path — replacing three divergent inline copies.
"""

from olmlx.engine.stop_sequences import StopScanner, truncate_at_stop


class TestStopScanner:
    def test_no_stop_sequences_passes_through(self):
        s = StopScanner(None)
        assert s.feed("hello") == ("hello", False)
        assert s.feed(" world") == (" world", False)
        s2 = StopScanner([])
        assert s2.feed("hello") == ("hello", False)

    def test_match_within_one_piece_truncates(self):
        s = StopScanner(["STOP"])
        piece, hit = s.feed("abcSTOPdef")
        assert (piece, hit) == ("abc", True)

    def test_match_spanning_piece_boundary(self):
        s = StopScanner(["STOP"])
        assert s.feed("abcST") == ("abcST", False)
        piece, hit = s.feed("OPdef")
        # The match starts in already-emitted text; nothing new to emit.
        assert (piece, hit) == ("", True)

    def test_match_at_piece_start(self):
        s = StopScanner(["STOP"])
        assert s.feed("abc") == ("abc", False)
        piece, hit = s.feed("STOP")
        assert (piece, hit) == ("", True)

    def test_earliest_of_multiple_sequences_wins(self):
        s = StopScanner(["zz", "b"])
        piece, hit = s.feed("abzz")
        assert (piece, hit) == ("a", True)

    def test_stop_hit_attribute_latches(self):
        s = StopScanner(["X"])
        s.feed("aX")
        assert s.stop_hit is True

    def test_empty_stop_strings_ignored(self):
        s = StopScanner([""])
        assert s.feed("abc") == ("abc", False)

    def test_partial_then_no_match_keeps_streaming(self):
        s = StopScanner(["STOP"])
        assert s.feed("ST") == ("ST", False)
        assert s.feed("ART") == ("ART", False)
        assert s.feed(" more") == (" more", False)

    def test_old_text_not_rescanned(self):
        # A match fully inside previously emitted text is the caller's
        # responsibility (it would have stopped there); the bounded scan
        # only finds matches ending in the new piece.
        s = StopScanner(["ab"])
        piece, hit = s.feed("ab")
        assert (piece, hit) == ("", True)


class TestTruncateAtStop:
    def test_no_sequences(self):
        assert truncate_at_stop("abc", None) == ("abc", False)
        assert truncate_at_stop("abc", []) == ("abc", False)

    def test_no_match(self):
        assert truncate_at_stop("abc", ["X"]) == ("abc", False)

    def test_truncates_at_earliest(self):
        assert truncate_at_stop("a STOP b END", ["END", "STOP"]) == ("a ", True)

    def test_empty_stop_ignored(self):
        assert truncate_at_stop("abc", [""]) == ("abc", False)


class TestStopScannerThinkingAware:
    """Issue #588: stop sequences must not fire inside a <think> block."""

    def test_stop_inside_think_does_not_fire(self):
        s = StopScanner(["three"], thinking_aware=True)
        assert s.feed("<think>") == ("<think>", False)
        assert s.feed("one two three four") == ("one two three four", False)
        assert not s.stop_hit

    def test_stop_in_visible_content_fires_after_think(self):
        s = StopScanner(["three"], thinking_aware=True)
        s.feed("<think>one two three</think>\n")
        piece, hit = s.feed("one two three four five")
        assert hit and piece == "one two "

    def test_stop_before_think_fires_normally(self):
        s = StopScanner(["stop"], thinking_aware=True)
        piece, hit = s.feed("prestop<think>thinking</think>\ncontent")
        assert hit and piece == "pre"

    def test_no_think_block_identical_to_non_aware(self):
        s = StopScanner(["STOP"], thinking_aware=True)
        assert s.feed("abc") == ("abc", False)
        piece, hit = s.feed("STOPdef")
        assert hit and piece == ""

    def test_stop_spanning_piece_boundary_after_think(self):
        s = StopScanner(["STOP"], thinking_aware=True)
        s.feed("<think>thinking</think>\nprefix")
        assert s.feed("ST") == ("ST", False)
        piece, hit = s.feed("OP")
        assert hit and piece == ""

    def test_gemma_channel_close_tag_respected(self):
        s = StopScanner(["three"], thinking_aware=True)
        s.feed("<|channel>thought\none two three<channel|>")
        piece, hit = s.feed("one two three four")
        assert hit and piece == "one two "

    def test_think_open_and_close_in_one_piece(self):
        s = StopScanner(["three"], thinking_aware=True)
        # Open and close tag in the same fed piece, stop only inside think.
        piece, hit = s.feed("<think>one two three</think>\nvisible")
        assert not hit and piece == "<think>one two three</think>\nvisible"
        piece, hit = s.feed("a three b")
        assert hit and piece == "a "


class TestTruncateAtStopThinkingAware:
    """Issue #588: whole-text variant must skip the thinking block."""

    def test_stop_inside_think_not_matched(self):
        text = "<think>one two three four</think>\nvisible"
        result, hit = truncate_at_stop(text, ["three"], thinking_aware=True)
        assert not hit and result == text

    def test_stop_in_visible_part_fires(self):
        text = "<think>thinking</think>\none two three four"
        result, hit = truncate_at_stop(text, ["three"], thinking_aware=True)
        assert hit and result == "<think>thinking</think>\none two "

    def test_thinking_aware_no_think_block_normal(self):
        result, hit = truncate_at_stop("one two three", ["three"], thinking_aware=True)
        assert hit and result == "one two "

    def test_gemma_channel_thinking_skipped(self):
        text = "<|channel>thought\none two three<channel|>one two three four"
        result, hit = truncate_at_stop(text, ["three"], thinking_aware=True)
        assert hit and result == "<|channel>thought\none two three<channel|>one two "


def test_thinking_pairs_in_sync_with_thinking_split():
    """Guard against drift: the stop-scanner's thinking tag pairs must match the
    canonical set in routers/thinking_split.py (issue #588 risk note)."""
    from olmlx.engine.stop_sequences import _THINKING_PAIRS as STOP_PAIRS
    from olmlx.routers.thinking_split import _THINKING_PAIRS as SPLIT_PAIRS

    assert STOP_PAIRS == SPLIT_PAIRS


class TestThinkingAwareMultipleBlocks:
    """Issue #588 review: every thinking block is skipped, not just the first."""

    def test_scanner_stop_in_second_think_block_skipped(self):
        s = StopScanner(["three"], thinking_aware=True)
        s.feed("<think>a</think>\nout1 <think>one two three</think>\n")
        piece, hit = s.feed("out2 three end")
        assert hit and piece == "out2 "

    def test_scanner_no_stop_when_only_in_think_blocks(self):
        s = StopScanner(["three"], thinking_aware=True)
        s.feed("<think>three</think>\nout1 ")
        piece, hit = s.feed("<think>three again</think>\nout2")
        assert not hit and not s.stop_hit

    def test_truncate_stop_in_second_think_block_skipped(self):
        text = "<think>a</think>\nout1 <think>one two three</think>\nout2"
        result, hit = truncate_at_stop(text, ["three"], thinking_aware=True)
        assert not hit and result == text

    def test_truncate_stop_after_second_think_fires(self):
        text = "<think>a</think>\nout1 <think>b</think>\nthe answer is three!"
        result, hit = truncate_at_stop(text, ["three"], thinking_aware=True)
        assert (
            hit and result == "<think>a</think>\nout1 <think>b</think>\nthe answer is "
        )
