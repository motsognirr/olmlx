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
