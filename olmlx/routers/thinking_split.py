"""Streaming thinking-block splitter shared by all chat-surface routers.

The single state machine for the ``<think>`` grammar (issue #471).  Routes a
model's thinking text into a separate output channel so the Ollama API can
populate ``message.thinking`` / the generate ``thinking`` field (issue #307),
the OpenAI router can strip it (:func:`strip_thinking_streaming`), and the
Anthropic router can emit ``thinking`` content blocks.  Recognizes the
Qwen-style ``<think>...</think>`` and Gemma 4's
``<|channel>thought\\n...<channel|>`` formats (issue #306).

The core is :func:`split_thinking_parts`, which preserves the interleaving
order of channels within a chunk (the Anthropic content-block emitter needs
ordering); :func:`split_thinking_streaming` joins each channel for callers
that keep two independent channels per chunk.
"""

import logging

from olmlx.engine.inference import INIT_ORPHAN_DETECT_LIMIT

logger = logging.getLogger(__name__)

# Tag pairs the streaming splitter recognizes.  Adding a third entry
# requires only updating this table — the state machine below is
# tag-pair-aware.
_THINKING_PAIRS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<|channel>thought\n", "<channel|>"),
)

# Maximum buffer size in the ``detect`` phase before giving up on finding a
# (possibly orphaned) close tag and transitioning to ``passthrough``.  This
# only bounds the ``thinking_expected`` orphan-detection window: when thinking
# is not expected the detect phase flushes immediately (issue #659), so a
# non-thinking model never buffers up to this limit.  Callers that know
# thinking is incoming raise the window via ``state["thinking_expected"]`` (or
# an explicit ``state["detect_limit"]``) so a longer orphaned-``</think>``
# preamble (that a chat template pre-opened in the prompt) is still caught.
_STREAM_DETECT_LIMIT = 200


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

    Used to hold back bytes that might be the start of a tag straddling a
    chunk boundary — in the passthrough phase and at the
    detect→passthrough transition.
    """
    longest = 0
    for open_tag, _ in _THINKING_PAIRS:
        for i in range(min(len(open_tag), len(buf)), longest, -1):
            if open_tag.startswith(buf[-i:]):
                longest = i
                break
    return longest


def split_thinking_parts(text: str, state: dict) -> list[tuple[str, str]]:
    """Split a streaming token into ordered ``(channel, text)`` parts.

    ``channel`` is ``"thinking"`` or ``"content"``; parts preserve the
    interleaving order within the chunk (e.g. ``a<think>b</think>c`` yields
    content ``a``, thinking ``b``, content ``c``).  Empty fragments are
    never emitted.

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
    - ``detect_limit`` – explicit override for the detect-phase window;
      defaults from ``thinking_expected`` as above.
    """
    buf = state.get("buffer", "") + text
    parts: list[tuple[str, str]] = []
    phase = state.get("phase", "detect")
    expected_close = state.get("expected_close", "")
    thinking_expected = bool(state.get("thinking_expected"))
    detect_limit = int(
        state.get(
            "detect_limit",
            INIT_ORPHAN_DETECT_LIMIT if thinking_expected else _STREAM_DETECT_LIMIT,
        )
    )

    def emit(channel: str, fragment: str) -> None:
        if fragment:
            parts.append((channel, fragment))

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
                # to the thinking channel.  Debug-logged because this
                # branch also fires on the pathological literal-token-in-
                # prose case when thinking *was* expected; raise the level
                # if missing content is reported.
                logger.debug(
                    "split_thinking_parts: orphaned %r at byte %d", close_tag, close_idx
                )
                emit("thinking", buf[:close_idx])
                buf = buf[close_idx + len(close_tag) :].lstrip("\n")
                phase = "passthrough"
            elif open_idx != -1:
                emit("content", buf[:open_idx])
                buf = buf[open_idx + len(open_tag) :]
                expected_close = paired_close
                phase = "in_think"
            else:
                # With no open tag pending, the detect phase only needs to
                # keep accumulating while the orphan-close heuristic is armed
                # (``thinking_expected``): the whole preamble before a possible
                # orphaned ``</think>`` is thinking and must be withheld until
                # that close tag appears (or the detect window fills).  When
                # thinking is *not* expected the orphan-close branch above can
                # never fire, so detect behaves exactly like passthrough — hold
                # back only a trailing partial open-tag and flush the rest
                # immediately, instead of buffering up to ``detect_limit``
                # before releasing a single byte (issue #659).
                if not thinking_expected or len(buf) > detect_limit:
                    # Hold back any partial open-tag suffix so a tag
                    # straddling the detect→passthrough transition is still
                    # recognized on the next chunk.  Close-tag partials are
                    # deliberately not retained: a non-thinking response's
                    # later close tag is literal text.
                    partial = _longest_open_tag_suffix(buf)
                    emit("content", buf[:-partial] if partial else buf)
                    buf = buf[-partial:] if partial else ""
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
                    emit("thinking", buf[:-hold])
                    buf = buf[-hold:]
                break
            emit("thinking", buf[:end])
            buf = buf[end + len(expected_close) :].lstrip("\n")
            expected_close = ""
            phase = "passthrough"

        else:  # passthrough
            open_idx, open_tag, paired_close = _find_earliest_open(buf)
            if open_idx == -1:
                longest_partial = _longest_open_tag_suffix(buf)
                if longest_partial:
                    emit("content", buf[:-longest_partial])
                    buf = buf[-longest_partial:]
                else:
                    emit("content", buf)
                    buf = ""
                break
            emit("content", buf[:open_idx])
            buf = buf[open_idx + len(open_tag) :]
            expected_close = paired_close
            phase = "in_think"

    state["buffer"] = buf
    state["phase"] = phase
    state["expected_close"] = expected_close
    return parts


def split_thinking_streaming(text: str, state: dict) -> tuple[str, str]:
    """Split a streaming token into ``(thinking_chunk, content_chunk)``.

    Joining wrapper over :func:`split_thinking_parts` for callers with two
    independent output channels per chunk (Ollama ``message.thinking`` /
    ``message.content``); interleaving order within the chunk is collapsed.
    """
    parts = split_thinking_parts(text, state)
    thinking = "".join(frag for channel, frag in parts if channel == "thinking")
    content = "".join(frag for channel, frag in parts if channel == "content")
    return thinking, content


def flush_split_thinking(state: dict) -> tuple[str, str]:
    """Flush remaining buffer at stream end.

    If still in ``detect`` (no tag ever seen), treat as content.  In
    ``in_think`` (open tag without close), treat as thinking so the
    response isn't truncated.  ``passthrough`` may hold a partial
    open-tag suffix — real visible bytes once no more chunks are coming.

    Resets managed state keys to their initial values so a caller that
    reuses the dict starts fresh.
    """
    buf = state.get("buffer", "")
    phase = state.get("phase", "detect")
    state["buffer"] = ""
    state["phase"] = "detect"
    state["expected_close"] = ""
    state["thinking_expected"] = False
    if not buf:
        return "", ""
    if phase == "in_think":
        return buf, ""
    return "", buf


def strip_thinking_streaming(text: str, state: dict) -> str:
    """Strip thinking blocks from streaming text chunks.

    The OpenAI surface has no thinking channel, so this is
    :func:`split_thinking_streaming` with the thinking output discarded.
    Same *state* contract.
    """
    return split_thinking_streaming(text, state)[1]


def flush_thinking_buffer(state: dict) -> str:
    """Flush any remaining buffer when the stream ends, content channel only.

    In ``detect`` the buffer holds the entire pre-tag output; in
    ``passthrough`` it may hold a partial open-tag suffix withheld in case a
    tag straddled a chunk boundary.  Both are real visible bytes once no
    more chunks are coming.  ``in_think`` state is thinking content and
    stays dropped.
    """
    return flush_split_thinking(state)[1]
