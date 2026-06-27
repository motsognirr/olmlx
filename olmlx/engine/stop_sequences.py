"""Stop-sequence scanning shared by the completion paths.

Three call sites previously carried their own copy of this logic
(exclusive streaming, batched streaming, non-streaming whole-text);
``StopScanner`` is the one incremental implementation and
``truncate_at_stop`` the whole-text variant (batching plan Phase 2,
PR #507 review follow-up).

``thinking_aware=True`` makes both variants ignore stop sequences that
fall inside a ``<think>...</think>`` block, so a stop string that appears
only in the model's reasoning does not halt generation before any visible
text is produced (issue #588).
"""

from __future__ import annotations

# Open/close tag pairs that delimit a hidden "thinking" region. Must stay in
# sync with ``_THINKING_PAIRS`` in ``routers/thinking_split.py`` — adding a new
# thinking format there without mirroring it here would let stop sequences fire
# inside that format's thinking block. Cannot import from thinking_split.py:
# it imports INIT_ORPHAN_DETECT_LIMIT from inference.py, which imports this
# module — a circular import.
_THINKING_PAIRS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<|channel>thought\n", "<channel|>"),
)


def _find_think_span(text: str) -> tuple[int, int]:
    """Span ``[open_start, visible_resume)`` of the first thinking block.

    ``visible_resume`` is the offset where visible text resumes after the close
    tag (leading newlines skipped, mirroring ``thinking_split``). Returns
    ``(open_start, len(text))`` for an unterminated block (everything after the
    open tag is still hidden) and ``(-1, -1)`` when no open tag is present.
    """
    best_idx, best_open, best_close = -1, "", ""
    for open_tag, close_tag in _THINKING_PAIRS:
        i = text.find(open_tag)
        if i != -1 and (best_idx == -1 or i < best_idx):
            best_idx, best_open, best_close = i, open_tag, close_tag
    if best_idx == -1:
        return -1, -1
    j = text.find(best_close, best_idx + len(best_open))
    if j == -1:
        return best_idx, len(text)
    resume = j + len(best_close)
    while resume < len(text) and text[resume] == "\n":
        resume += 1
    return best_idx, resume


class StopScanner:
    """Incremental stop-sequence scanner for streaming decode loops.

    Feed each decoded text piece; get back the emittable portion of that
    piece, truncated at the earliest stop-sequence match, plus whether a
    stop matched. Matches may span piece boundaries (text accumulates
    internally), but the search start is bounded so each call scans
    O(len(piece)) — not O(generation length).

    When ``thinking_aware=True`` a stop match is honoured only if it lies in
    visible text (before a ``<think>`` open tag or after its ``</think>``
    close); matches inside the thinking block are skipped (issue #588).
    """

    def __init__(self, stop_sequences: list[str] | None, thinking_aware: bool = False):
        self._stops = [s for s in (stop_sequences or []) if s]
        self._text = ""
        self.stop_hit = False
        self._thinking_aware = thinking_aware
        # Thinking-phase state (only used when thinking_aware).
        self._phase = "detect"  # "detect" | "in_think" | "passthrough"
        self._expected_close = ""
        self._open_start = -1  # offset of the open tag, or -1 if none seen
        self._open_end = 0  # offset just past the open tag
        self._visible_resume = -1  # offset where visible text resumes, or -1

    def feed(self, piece: str) -> tuple[str, bool]:
        """Append *piece*; return (emittable_text, stop_hit).

        On a match, ``emittable_text`` is the part of *piece* before the
        match (empty when the match started in already-emitted text).
        """
        if not self._stops:
            return piece, False
        prev_len = len(self._text)
        self._text += piece
        if self._thinking_aware:
            self._advance_thinking_state(prev_len)
        stop_idx = -1
        for stop_seq in self._stops:
            # A match ending in the new piece must start within
            # len(stop_seq)-1 of the boundary; anything earlier was
            # caught (and truncated at) by a previous call.
            start = max(0, prev_len - len(stop_seq) + 1)
            idx = self._text.find(stop_seq, start)
            if self._thinking_aware:
                # Skip matches that fall inside the thinking block.
                while idx != -1 and not self._is_visible(idx):
                    idx = self._text.find(stop_seq, idx + 1)
            if idx != -1 and (stop_idx == -1 or idx < stop_idx):
                stop_idx = idx
        if stop_idx == -1:
            return piece, False
        self.stop_hit = True
        return (self._text[prev_len:stop_idx] if prev_len < stop_idx else ""), True

    def _is_visible(self, idx: int) -> bool:
        """Whether a match starting at *idx* lies in visible (non-thinking) text."""
        if self._open_start == -1:
            return True  # no thinking block seen → all visible
        if idx < self._open_start:
            return True  # before the thinking block
        if self._visible_resume != -1 and idx >= self._visible_resume:
            return True  # after the thinking block closed
        return False  # inside the (open) thinking block

    def _advance_thinking_state(self, prev_len: int) -> None:
        """Advance detect→in_think→passthrough over newly-appended text.

        Bounded to the boundary window so each call is O(len(piece)): an open or
        close tag that completes in the new piece must start within
        ``len(tag)-1`` of ``prev_len``; one entirely in a prior piece was
        resolved on an earlier call.
        """
        if self._phase == "passthrough":
            return
        if self._phase == "detect":
            best_idx, best_open, best_close = -1, "", ""
            for open_tag, close_tag in _THINKING_PAIRS:
                start = max(0, prev_len - len(open_tag) + 1)
                i = self._text.find(open_tag, start)
                if i != -1 and (best_idx == -1 or i < best_idx):
                    best_idx, best_open, best_close = i, open_tag, close_tag
            if best_idx == -1:
                return
            self._open_start = best_idx
            self._open_end = best_idx + len(best_open)
            self._expected_close = best_close
            self._phase = "in_think"
            # Fall through: the close tag may be in the same piece.
        if self._phase == "in_think":
            start = max(self._open_end, prev_len - len(self._expected_close) + 1)
            j = self._text.find(self._expected_close, start)
            if j != -1:
                resume = j + len(self._expected_close)
                while resume < len(self._text) and self._text[resume] == "\n":
                    resume += 1
                self._visible_resume = resume
                self._phase = "passthrough"


def truncate_at_stop(
    text: str, stop_sequences: list[str] | None, thinking_aware: bool = False
) -> tuple[str, bool]:
    """Whole-text variant for the non-streaming path.

    Returns (text truncated at the earliest match, whether one matched). When
    ``thinking_aware=True``, matches inside the first ``<think>...</think>``
    block are ignored (issue #588).
    """
    skip_start, skip_end = (-1, -1)
    if thinking_aware:
        skip_start, skip_end = _find_think_span(text)
    earliest = -1
    for stop_seq in stop_sequences or []:
        if not stop_seq:
            continue
        idx = text.find(stop_seq)
        while idx != -1 and skip_start != -1 and skip_start <= idx < skip_end:
            idx = text.find(stop_seq, idx + 1)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
    if earliest == -1:
        return text, False
    return text[:earliest], True
