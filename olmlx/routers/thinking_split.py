"""Streaming thinking-block splitter shared by the chat and generate routers.

Routes a model's thinking text into a separate output channel so the Ollama
API can populate ``message.thinking`` / the generate ``thinking`` field
(issue #307).  Recognizes the Qwen-style ``<think>...</think>`` and Gemma 4's
``<|channel>thought\\n...<channel|>`` formats (issue #306).
"""

from olmlx.engine.inference import INIT_ORPHAN_DETECT_LIMIT

# Tag pairs the streaming splitter recognizes.  Adding a third entry
# requires only updating this table — the state machine below is
# tag-pair-aware.
_THINKING_PAIRS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<|channel>thought\n", "<channel|>"),
)


def _find_earliest_open(buf: str) -> tuple[int, str, str]:
    """Earliest open-tag occurrence in *buf*.

    Returns ``(idx, open_tag, paired_close)`` or ``(-1, "", "")`` if none.
    """
    best_idx = -1
    best_open = ""
    best_close = ""
    for open_tag, close_tag in _THINKING_PAIRS:
        idx = buf.find(open_tag)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx, best_open, best_close = idx, open_tag, close_tag
    return best_idx, best_open, best_close


def _find_earliest_close(buf: str) -> tuple[int, str]:
    """Earliest close-tag occurrence (any pair) in *buf*.

    Returns ``(idx, close_tag)`` or ``(-1, "")`` if none.
    """
    best_idx = -1
    best_close = ""
    for _, close_tag in _THINKING_PAIRS:
        idx = buf.find(close_tag)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx, best_close = idx, close_tag
    return best_idx, best_close


def _longest_open_tag_suffix(buf: str) -> int:
    """Largest ``k`` such that ``buf[-k:]`` is a prefix of some open tag.

    Used in passthrough phase to hold back bytes that might be the start
    of a tag straddling a chunk boundary.
    """
    longest = 0
    for open_tag, _ in _THINKING_PAIRS:
        for i in range(min(len(open_tag), len(buf)), longest, -1):
            if open_tag.startswith(buf[-i:]):
                longest = i
                break
    return longest


def split_thinking_streaming(text: str, state: dict) -> tuple[str, str]:
    """Split a streaming token into ``(thinking_chunk, content_chunk)``.

    State keys:

    - ``phase`` – ``"detect"``, ``"in_think"``, ``"passthrough"``.
    - ``buffer`` – accumulated text waiting to be resolved.
    - ``expected_close`` – when in ``in_think``, the close tag paired
      with the open tag we entered through; ensures cross-format
      mentions inside thinking content can't end the block early.
    - ``thinking_expected`` – when True, the detect phase tolerates a
      longer orphan-thinking preamble before giving up
      (``INIT_ORPHAN_DETECT_LIMIT``) and the orphan-close heuristic
      fires.  Off-by-default keeps non-thinking models from
      misclassifying a literal ``</think>``-in-prose as thinking.
    """
    buf = state.get("buffer", "") + text
    thinking_parts: list[str] = []
    content_parts: list[str] = []
    phase = state.get("phase", "detect")
    expected_close = state.get("expected_close", "")
    thinking_expected = bool(state.get("thinking_expected"))
    detect_limit = INIT_ORPHAN_DETECT_LIMIT if thinking_expected else 200

    while buf:
        if phase == "detect":
            open_idx, open_tag, paired_close = _find_earliest_open(buf)
            close_idx, close_tag = _find_earliest_close(buf)

            if (
                close_idx != -1
                and (open_idx == -1 or close_idx < open_idx)
                and thinking_expected
            ):
                # Orphan close: prefix is thinking.  Gated on
                # ``thinking_expected`` so a non-thinking model that
                # legitimately mentions the literal token isn't routed
                # to ``message.thinking``.
                thinking_parts.append(buf[:close_idx])
                buf = buf[close_idx + len(close_tag) :].lstrip("\n")
                phase = "passthrough"
            elif open_idx != -1:
                content_parts.append(buf[:open_idx])
                buf = buf[open_idx + len(open_tag) :]
                expected_close = paired_close
                phase = "in_think"
            else:
                if len(buf) > detect_limit:
                    content_parts.append(buf)
                    buf = ""
                    phase = "passthrough"
                break

        elif phase == "in_think":
            end = buf.find(expected_close)
            if end == -1:
                # Hold back up to ``len(expected_close)`` trailing chars so
                # a close tag straddling a chunk boundary still matches on
                # the next chunk; emit the rest as thinking.
                hold = len(expected_close)
                if len(buf) > hold:
                    thinking_parts.append(buf[:-hold])
                    buf = buf[-hold:]
                break
            thinking_parts.append(buf[:end])
            buf = buf[end + len(expected_close) :].lstrip("\n")
            expected_close = ""
            phase = "passthrough"

        else:  # passthrough
            open_idx, open_tag, paired_close = _find_earliest_open(buf)
            if open_idx == -1:
                longest_partial = _longest_open_tag_suffix(buf)
                if longest_partial:
                    content_parts.append(buf[:-longest_partial])
                    buf = buf[-longest_partial:]
                else:
                    content_parts.append(buf)
                    buf = ""
                break
            content_parts.append(buf[:open_idx])
            buf = buf[open_idx + len(open_tag) :]
            expected_close = paired_close
            phase = "in_think"

    state["buffer"] = buf
    state["phase"] = phase
    state["expected_close"] = expected_close
    return "".join(thinking_parts), "".join(content_parts)


def flush_split_thinking(state: dict) -> tuple[str, str]:
    """Flush remaining buffer at stream end.

    If still in ``detect`` (no tag ever seen), treat as content.  In
    ``in_think`` (open tag without close), treat as thinking so the
    response isn't truncated.

    Resets managed state keys to their initial values so a caller that
    reuses the dict starts fresh — matches ``flush_thinking_buffer`` in
    ``utils/streaming.py``.
    """
    buf = state.get("buffer", "")
    phase = state.get("phase", "detect")
    state["buffer"] = ""
    state["phase"] = "detect"
    state["expected_close"] = ""
    if not buf:
        return "", ""
    if phase == "in_think":
        return buf, ""
    return "", buf
