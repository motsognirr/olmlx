"""Stop-sequence scanning shared by the completion paths.

Three call sites previously carried their own copy of this logic
(exclusive streaming, batched streaming, non-streaming whole-text);
``StopScanner`` is the one incremental implementation and
``truncate_at_stop`` the whole-text variant (batching plan Phase 2,
PR #507 review follow-up).

``thinking_aware=True`` makes both variants ignore stop sequences that
fall inside a ``<think>...</think>`` block, so a stop string that appears
only in the model's reasoning does not halt generation before any visible
text is produced (issue #588). Every thinking block in the output is
skipped — matching ``thinking_split.py``, which re-enters the thinking
state on each open tag — not just the first.
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

_MAX_OPEN_TAG_LEN = max(len(open_tag) for open_tag, _ in _THINKING_PAIRS)


def _find_think_spans(text: str) -> list[tuple[int, int]]:
    """Every hidden thinking span ``[open_start, visible_resume)`` in *text*.

    Mirrors ``thinking_split.py``: each ``<think>...</think>`` pair is hidden,
    and ``visible_resume`` skips leading newlines after the close tag. An
    unterminated final block hides everything to the end of *text*.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    n = len(text)
    while pos < n:
        best_idx, best_open, best_close = -1, "", ""
        for open_tag, close_tag in _THINKING_PAIRS:
            i = text.find(open_tag, pos)
            if i != -1 and (best_idx == -1 or i < best_idx):
                best_idx, best_open, best_close = i, open_tag, close_tag
        if best_idx == -1:
            break
        j = text.find(best_close, best_idx + len(best_open))
        if j == -1:
            spans.append((best_idx, n))
            break
        resume = j + len(best_close)
        while resume < n and text[resume] == "\n":
            resume += 1
        spans.append((best_idx, resume))
        pos = resume
    return spans


class StopScanner:
    """Incremental stop-sequence scanner for streaming decode loops.

    Feed each decoded text piece; get back the emittable portion of that
    piece, truncated at the earliest stop-sequence match, plus whether a
    stop matched. Matches may span piece boundaries (text accumulates
    internally), but the search start is bounded so each call scans
    O(len(piece)) — not O(generation length).

    When ``thinking_aware=True`` a stop match is honoured only if it lies in
    visible text (outside every ``<think>...</think>`` block); matches inside a
    thinking block are skipped (issue #588).
    """

    def __init__(self, stop_sequences: list[str] | None, thinking_aware: bool = False):
        self._stops = [s for s in (stop_sequences or []) if s]
        self._text = ""
        self.stop_hit = False
        self._thinking_aware = thinking_aware
        # Thinking-phase state (only used when thinking_aware). The phase machine
        # cycles detect → in_think → detect so every block is tracked, not just
        # the first.
        self._phase = "detect"  # "detect" | "in_think"
        self._expected_close = ""
        self._cur_open = -1  # start of the currently-open block, or -1
        self._cur_open_end = 0  # offset just past the current open tag
        self._spans: list[tuple[int, int]] = []  # closed (start, visible_resume)
        self._detect_from = 0  # lower bound for the next open-tag search

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
                # Skip matches that fall inside a thinking block.
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
        for start, resume in self._spans:
            if start <= idx < resume:
                return False  # inside a closed thinking block
        if self._cur_open != -1 and idx >= self._cur_open:
            return False  # inside the currently-open thinking block
        return True

    def _advance_thinking_state(self, prev_len: int) -> None:
        """Advance the detect↔in_think phase machine over newly-appended text.

        Bounded to the boundary window so each call is O(len(piece)): a tag that
        completes in the new piece must start within ``len(tag)-1`` of the
        boundary; one entirely in a prior piece was resolved on an earlier call.
        Loops so multiple blocks opening/closing within one piece are all seen.
        """
        while True:
            if self._phase == "detect":
                start = max(self._detect_from, prev_len - _MAX_OPEN_TAG_LEN + 1, 0)
                best_idx, best_open, best_close = -1, "", ""
                for open_tag, close_tag in _THINKING_PAIRS:
                    i = self._text.find(open_tag, start)
                    if i != -1 and (best_idx == -1 or i < best_idx):
                        best_idx, best_open, best_close = i, open_tag, close_tag
                if best_idx == -1:
                    # No open tag in range; remember how far we scanned so the
                    # next call doesn't rescan from the start.
                    self._detect_from = max(
                        self._detect_from, len(self._text) - _MAX_OPEN_TAG_LEN + 1
                    )
                    return
                self._cur_open = best_idx
                self._cur_open_end = best_idx + len(best_open)
                self._expected_close = best_close
                self._phase = "in_think"
                # Loop: the close tag may be in the same piece.
            else:  # in_think
                start = max(
                    self._cur_open_end, prev_len - len(self._expected_close) + 1
                )
                j = self._text.find(self._expected_close, start)
                if j == -1:
                    return
                resume = j + len(self._expected_close)
                while resume < len(self._text) and self._text[resume] == "\n":
                    resume += 1
                self._spans.append((self._cur_open, resume))
                self._cur_open = -1
                self._phase = "detect"
                self._detect_from = resume
                # Loop: a further thinking block may follow.


def truncate_at_stop(
    text: str, stop_sequences: list[str] | None, thinking_aware: bool = False
) -> tuple[str, bool]:
    """Whole-text variant for the non-streaming path.

    Returns (text truncated at the earliest match, whether one matched). When
    ``thinking_aware=True``, matches inside any ``<think>...</think>`` block are
    ignored (issue #588).
    """
    spans = _find_think_spans(text) if thinking_aware else []

    def _hidden(idx: int) -> bool:
        return any(start <= idx < resume for start, resume in spans)

    earliest = -1
    for stop_seq in stop_sequences or []:
        if not stop_seq:
            continue
        idx = text.find(stop_seq)
        while idx != -1 and _hidden(idx):
            idx = text.find(stop_seq, idx + 1)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
    if earliest == -1:
        return text, False
    return text[:earliest], True
