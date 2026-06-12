"""Stop-sequence scanning shared by the completion paths.

Three call sites previously carried their own copy of this logic
(exclusive streaming, batched streaming, non-streaming whole-text);
``StopScanner`` is the one incremental implementation and
``truncate_at_stop`` the whole-text variant (batching plan Phase 2,
PR #507 review follow-up).
"""

from __future__ import annotations


class StopScanner:
    """Incremental stop-sequence scanner for streaming decode loops.

    Feed each decoded text piece; get back the emittable portion of that
    piece, truncated at the earliest stop-sequence match, plus whether a
    stop matched. Matches may span piece boundaries (text accumulates
    internally), but the search start is bounded so each call scans
    O(len(piece)) — not O(generation length).
    """

    def __init__(self, stop_sequences: list[str] | None):
        self._stops = [s for s in (stop_sequences or []) if s]
        self._text = ""
        self.stop_hit = False

    def feed(self, piece: str) -> tuple[str, bool]:
        """Append *piece*; return (emittable_text, stop_hit).

        On a match, ``emittable_text`` is the part of *piece* before the
        match (empty when the match started in already-emitted text).
        """
        if not self._stops:
            return piece, False
        prev_len = len(self._text)
        self._text += piece
        stop_idx = -1
        for stop_seq in self._stops:
            # A match ending in the new piece must start within
            # len(stop_seq)-1 of the boundary; anything earlier was
            # caught (and truncated at) by a previous call.
            idx = self._text.find(stop_seq, max(0, prev_len - len(stop_seq) + 1))
            if idx != -1 and (stop_idx == -1 or idx < stop_idx):
                stop_idx = idx
        if stop_idx == -1:
            return piece, False
        self.stop_hit = True
        return (self._text[prev_len:stop_idx] if prev_len < stop_idx else ""), True


def truncate_at_stop(
    text: str, stop_sequences: list[str] | None
) -> tuple[str, bool]:
    """Whole-text variant for the non-streaming path.

    Returns (text truncated at the earliest match, whether one matched).
    """
    earliest = -1
    for stop_seq in stop_sequences or []:
        if not stop_seq:
            continue
        idx = text.find(stop_seq)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
    if earliest == -1:
        return text, False
    return text[:earliest], True
