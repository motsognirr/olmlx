"""Logits processors and decoding-output filters for inference.

Extracted from ``engine/inference.py`` (#454). Holds the request-scoped
logits-processor builders (grammar-constrained decoding, OpenAI
frequency/presence penalties) plus the gpt-oss channel filter that strips
``<|channel|>`` structural tokens from the decoded stream. ``inference.py``
re-imports these names so existing call sites and tests are unchanged.
"""

import dataclasses
import logging
from typing import TYPE_CHECKING

import mlx.core as mx

from olmlx.engine.grammar import (
    GrammarSpec,
    make_processor as _make_grammar_processor,
    unwrap_mlx_tokenizer as _unwrap_mlx_tokenizer,
)

if TYPE_CHECKING:
    from olmlx.engine.model_manager import LoadedModel

logger = logging.getLogger(__name__)


def _resolve_model_vocab_size(lm: "LoadedModel") -> int | None:
    """Return the model's lm_head vocab dimension, or None if undiscoverable.

    Used by grammar-constrained decoding to size the token bitmask. The
    model's lm_head dim can differ from ``tokenizer.vocab_size`` (Phi-3,
    Llama-3.2-Vision, …) — xgrammar needs the model's number for the mask
    to align with the actual logits tensor.

    Fallback order matters: prefer ``lm_head.weight.shape[0]`` (the actual
    output dimension) over ``embed_tokens.weight.shape[0]`` (input
    dimension). For tied embeddings the two are equal; for untied or
    expanded lm_head the output is larger, and a bitmask sized to the
    input would truncate the tail of the logit tensor and let
    out-of-grammar tokens through.
    """
    model = lm.model
    # mlx-lm convention: model.args.vocab_size is set by the loader.
    args = getattr(model, "args", None)
    vs = getattr(args, "vocab_size", None) if args is not None else None
    if isinstance(vs, int) and vs > 0:
        return vs
    # Prefer the lm_head output dimension over the embed_tokens input dim
    # AT EVERY nesting depth. Some models nest lm_head under
    # ``model.model`` while exposing ``embed_tokens`` at the top level;
    # iterating attr-first avoids returning the top-level embed_tokens
    # when a deeper lm_head exists.
    for attr in ("lm_head", "embed_tokens"):
        language_model = getattr(model, "language_model", None)
        for owner in (
            model,
            getattr(model, "model", None),
            language_model,
            getattr(language_model, "model", None),
        ):
            if owner is None:
                continue
            layer = getattr(owner, attr, None)
            if layer is not None and hasattr(layer, "weight"):
                try:
                    shape = layer.weight.shape  # type: ignore[attr-defined]
                    if shape:
                        return int(shape[0])
                except Exception:
                    pass
    return None


def _install_grammar_processor(
    lm: "LoadedModel",
    gen_kwargs: dict,
    grammar_spec: GrammarSpec | None,
    *,
    has_tools: bool = False,
) -> bool:
    """Build and install a grammar logits processor on *gen_kwargs*.

    Returns ``True`` when grammar is active for the request. Works for both
    text and VLM models — mlx_vlm's ``generate_step`` accepts
    ``logits_processors`` and olmlx forwards ``gen_kwargs`` to it (#429).
    Distributed mode is still rejected: workers don't receive the processor
    over the sideband and would diverge from rank-0. Tool-use requests are
    rejected: the JSON grammar masks the format-specific tool-call tokens
    (``<tool_call>``, ``[TOOL_CALLS]``, ``<function=...>``, …) so the model
    could never emit a tool call. Constraining tool *arguments* is the
    deferred Anthropic case (issue #361).
    """
    if grammar_spec is None:
        return False
    if lm.is_distributed:
        logger.warning(
            "Grammar-constrained decoding requested but model is running "
            "in distributed mode; ignoring constraint for this request"
        )
        return False
    if has_tools:
        logger.warning(
            "Grammar-constrained decoding requested alongside tools; "
            "the JSON grammar would mask tool-call tokens, breaking "
            "tool use. Ignoring grammar constraint for this request "
            "(constraining tool arguments specifically is a follow-up)"
        )
        return False
    vocab_size = _resolve_model_vocab_size(lm)
    if vocab_size is None:
        logger.warning(
            "Grammar-constrained decoding requested but model vocab_size "
            "could not be resolved; ignoring constraint for this request"
        )
        return False
    # xgrammar's ``TokenizerInfo.from_huggingface`` does a strict isinstance
    # check against ``PreTrainedTokenizerBase`` and rejects mlx-lm's
    # ``TokenizerWrapper``. Peel the wrapper. HF fast tokenizers also
    # expose ``_tokenizer`` (holding the Rust core) but ``unwrap_mlx_tokenizer``
    # only peels when the outer class name is ``TokenizerWrapper``.
    hf_tokenizer = _unwrap_mlx_tokenizer(lm.text_tokenizer)
    processor = _make_grammar_processor(hf_tokenizer, vocab_size, grammar_spec)
    existing = gen_kwargs.get("logits_processors", [])
    gen_kwargs["logits_processors"] = list(existing) + [processor]
    logger.info(
        "Grammar-constrained decoding active: kind=%s vocab_size=%d",
        grammar_spec.kind,
        vocab_size,
    )
    return True


def _make_frequency_penalty_processor(frequency_penalty: float):
    """Create a logits processor that applies OpenAI-style frequency penalty.

    Positive values penalize new tokens based on their existing frequency
    in the text so far, decreasing the model's likelihood to repeat the
    same line verbatim.

    Uses an incremental frequency dict (O(1) per step) to avoid O(n²)
    rebuilds for long generations.  The dict is seeded from the initial
    token list on first call then incremented by one per step.
    """
    freq: dict[int, int] = {}
    _initialised = False

    def processor(tokens: list[int], logits: mx.array) -> mx.array:
        nonlocal freq, _initialised
        if not tokens or frequency_penalty == 0:
            return logits
        vocab_size = logits.shape[-1]
        if not _initialised:
            for tid in tokens:
                if 0 <= tid < vocab_size:
                    freq[tid] = freq.get(tid, 0) + 1
            _initialised = True
        else:
            new_tid = tokens[-1]
            if 0 <= new_tid < vocab_size:
                freq[new_tid] = freq.get(new_tid, 0) + 1
        for tid, count in freq.items():
            logits[..., tid] -= frequency_penalty * count
        return logits

    return processor


def _make_presence_penalty_processor(presence_penalty: float):
    """Create a logits processor that applies OpenAI-style presence penalty.

    Positive values penalize new tokens based on whether they appear
    in the text so far, increasing the model's likelihood to talk about
    new topics.

    Uses an incremental seen set (O(1) per step) to avoid O(n²)
    set-builds for long generations.  The set is seeded from the
    initial token list on first call then incremented by one per step.
    """
    seen: set[int] = set()
    _initialised = False

    def processor(tokens: list[int], logits: mx.array) -> mx.array:
        nonlocal seen, _initialised
        if not tokens or presence_penalty == 0:
            return logits
        vocab_size = logits.shape[-1]
        if not _initialised:
            for tid in tokens:
                if 0 <= tid < vocab_size and tid not in seen:
                    seen.add(tid)
                    logits[..., tid] -= presence_penalty
            _initialised = True
        else:
            new_tid = tokens[-1]
            if 0 <= new_tid < vocab_size and new_tid not in seen:
                seen.add(new_tid)
                logits[..., new_tid] -= presence_penalty
        return logits

    return processor


# gpt-oss special tokens used by the streaming filter
_GPT_OSS_STRUCTURAL_TOKENS = frozenset(
    {
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|call|>",
        "<|return|>",
    }
)


class _GptOssChannelFilter:
    """Stateful filter for gpt-oss channel tokens.

    Call ``feed(text)`` for each token; it returns the text to send to the
    client (``""`` for nothing). Final-channel tokens pass through verbatim.
    Analysis-channel text is buffered and flushed as one ``<think>...</think>``
    block when the final channel starts, so the routers' shared thinking
    splitter routes it to ``message.thinking`` / a ``thinking`` block (#713).
    After the stream ends, yield ``get_fallback_texts()``: the analysis text
    promoted to visible content when no final channel was produced (the same
    rule ``_parse_gpt_oss_channels`` applies to non-streaming output), or a
    trailing ``<think>`` block for analysis that arrived after the final one.

    This is a class (not an async generator) so the caller can iterate the raw
    stream for prompt-cache token accumulation while only yielding filtered text.
    """

    _INIT = "init"
    _AFTER_START = "after_start"
    _EXPECT_CHANNEL = "expect_channel"
    _IN_BLOCK = "in_block"
    _CONTENT = "content"

    def __init__(self):
        self._state = self._INIT
        self._channel = None
        self._saw_any_channel = False
        self._saw_final = False
        # One entry per analysis block, each a list of token texts.
        self._analysis_blocks: list[list[str]] = []
        self._full_text_parts: list[str] = []

    def _flush_analysis(self) -> str:
        """Drain buffered analysis blocks as a single ``<think>`` block.

        Blocks are stripped and newline-joined like ``_parse_gpt_oss_channels``.
        """
        thinking = "\n".join(
            text for block in self._analysis_blocks if (text := "".join(block).strip())
        )
        self._analysis_blocks = []
        return f"<think>{thinking}</think>" if thinking else ""

    def feed(self, text: str) -> str:
        """Process one token's text and return the text to yield (may be empty)."""
        self._full_text_parts.append(text)

        if text == "<|start|>":
            self._state = self._AFTER_START
            self._saw_any_channel = True
            return ""

        if text == "<|channel|>":
            self._state = self._EXPECT_CHANNEL
            self._saw_any_channel = True
            return ""

        if self._state == self._AFTER_START:
            return ""

        if self._state == self._EXPECT_CHANNEL:
            self._channel = text.strip()
            self._state = self._IN_BLOCK
            if self._channel == "analysis":
                self._analysis_blocks.append([])
            elif self._channel == "final":
                self._saw_final = True
                return self._flush_analysis()
            return ""

        if text == "<|message|>" and self._state == self._IN_BLOCK:
            self._state = self._CONTENT
            return ""

        if text in ("<|end|>", "<|call|>", "<|return|>"):
            self._state = self._INIT
            self._channel = None
            return ""

        if self._state == self._CONTENT and self._channel == "final":
            return text

        if self._state == self._CONTENT and self._channel == "analysis":
            self._analysis_blocks[-1].append(text)
            return ""

        if (
            self._state == self._INIT
            and not self._saw_any_channel
            and text not in _GPT_OSS_STRUCTURAL_TOKENS
        ):
            return text

        return ""

    def get_fallback_texts(self) -> list[str]:
        """Return what's left to yield once the stream has ended.

        Without a final channel the buffered analysis texts are promoted to
        visible content; after one, leftover analysis is a ``<think>`` block.
        """
        if not self._saw_final:
            return [text for block in self._analysis_blocks for text in block]
        flushed = self._flush_analysis()
        return [flushed] if flushed else []

    def get_full_text(self) -> str:
        """Return the complete raw text accumulated during streaming."""
        return "".join(self._full_text_parts)


async def _gpt_oss_filter(token_stream):
    """Async generator wrapper for backward compatibility with tests."""
    filt = _GptOssChannelFilter()
    last = None
    async for token in token_stream:
        last = token
        if out := filt.feed(token.text):
            yield dataclasses.replace(token, text=out)
    if last is not None:
        for text in filt.get_fallback_texts():
            yield dataclasses.replace(last, text=text)


# ---------------------------------------------------------------------------
# Thinking budget (Anthropic ``thinking.budget_tokens``, issue #716)
# ---------------------------------------------------------------------------

# Vocab fallback for tokenizers that aren't mlx-lm ``TokenizerWrapper``s (the
# wrapper exposes ``think_start_tokens``/``think_end_tokens`` directly).
_THINK_MARKER_FALLBACKS = (("<think>", "</think>"),)


def _resolve_think_markers(
    tokenizer,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return ``(think_start_ids, think_end_ids)`` for *tokenizer*, or None.

    Prefers mlx-lm's ``TokenizerWrapper`` detection (single-token
    ``<think>``/``</think>``, multi-token Gemma-4 ``<|channel>thought`` /
    ``<channel|>``), falling back to a vocab lookup. The ``isinstance``
    checks keep a MagicMock tokenizer from masquerading as having markers.
    """
    start = getattr(tokenizer, "think_start_tokens", None)
    end = getattr(tokenizer, "think_end_tokens", None)
    if isinstance(start, (tuple, list)) and isinstance(end, (tuple, list)):
        if start and end:
            # Third-party contract: degrade to "no markers" on a shape
            # mismatch rather than failing every budget-carrying request.
            try:
                return tuple(int(t) for t in start), tuple(int(t) for t in end)
            except (TypeError, ValueError):
                pass
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if get_vocab is None:
        return None
    try:
        vocab = get_vocab()
    except Exception:
        return None
    if not isinstance(vocab, dict):
        return None
    for s, e in _THINK_MARKER_FALLBACKS:
        if s in vocab and e in vocab:
            return (int(vocab[s]),), (int(vocab[e]),)
    return None


def _thinking_open_after(
    generated: list[int],
    start_seq: tuple[int, ...],
    end_seq: tuple[int, ...],
    initially_open: bool,
) -> bool:
    """Replay think open/close markers over *generated*; True if still open."""
    state = initially_open
    i, n = 0, len(generated)
    while i < n:
        if tuple(generated[i : i + len(end_seq)]) == end_seq:
            state = False
            i += len(end_seq)
        elif tuple(generated[i : i + len(start_seq)]) == start_seq:
            state = True
            i += len(start_seq)
        else:
            i += 1
    return state


def _partial_marker_suffix(generated: list[int], seq: tuple[int, ...]) -> int:
    """Length of the longest *proper* prefix of *seq* ending *generated* (0 if none)."""
    for n in range(min(len(seq) - 1, len(generated)), 0, -1):
        if tuple(generated[-n:]) == seq[:n]:
            return n
    return 0


def _generated_tail(tokens, k: int) -> list[int]:
    """Last *k* entries of a processor token history as a Python list.

    ``generate_step`` passes an ``mx.array``; the batched path passes a
    per-sequence buffer. Either way the last *k* entries are exactly the
    tokens generated so far (the history begins at the prompt tail).
    """
    if k <= 0:
        return []
    tail = tokens[-k:]
    if hasattr(tail, "tolist"):
        tail = tail.tolist()
    return [int(t) for t in tail]


def _force_token(logits: mx.array, token_id: int) -> mx.array:
    """A logits row that can only sample *token_id*.

    Built fresh (not masked from *logits*) so an earlier processor that
    already sent *token_id* to ``-inf`` can't yield an all-``-inf`` row (NaN
    logsumexp). ``mx.depends`` keeps the model forward in the graph: without
    it the step's forward — and its KV-cache / recurrent-state writes — would
    not be evaluated by the step's ``mx.eval``, leaving a lazy graph bound to
    the generation worker's stream that could be stored and later evaluated
    from a different thread (the #499 thread-local-stream hazard).
    """
    vocab = logits.shape[-1]
    forced = mx.where(
        mx.arange(vocab) == token_id,
        mx.array(0.0, dtype=logits.dtype),
        mx.array(-mx.inf, dtype=logits.dtype),
    )
    forced = mx.broadcast_to(forced, logits.shape)
    return mx.depends(forced, logits)


def _make_thinking_budget_processor(
    budget: int,
    start_seq: tuple[int, ...],
    end_seq: tuple[int, ...],
    *,
    initially_open: bool,
):
    """Logits processor that ends an open think block after *budget* tokens.

    Relies on the processor contract shared by mlx-lm's ``generate_step``,
    mlx-vlm's ``generate_step`` and the batched ``GenerationBatch``: exactly
    one call per sampled token, with the history's last ``k`` entries being
    the ``k`` tokens generated so far. Below the budget it is a pure
    pass-through (no host sync — the decode pipeline is untouched). At call
    ``budget`` it syncs the generated tokens *once* and replays the think
    markers; if a think block is still open it forces *end_seq* one token per
    call, then goes inert so the model writes its answer. If thinking already
    closed (or never opened), it goes inert immediately.
    """
    calls = 0
    forcing: int | None = None
    done = False

    def thinking_budget_processor(tokens, logits: mx.array) -> mx.array:
        nonlocal calls, forcing, done
        k = calls
        calls += 1
        if done:
            return logits
        if forcing is None:
            if k < budget:
                return logits
            generated = _generated_tail(tokens, k)
            if not _thinking_open_after(generated, start_seq, end_seq, initially_open):
                # A multi-token opener (Gemma-4 ``<|channel>thought``) may be
                # straddling the boundary: stay armed and re-check next call
                # rather than going inert before it completes.
                if _partial_marker_suffix(generated, start_seq):
                    return logits
                done = True
                return logits
            # Finish a multi-token close the model already began instead of
            # restarting it (which would emit a duplicated, malformed marker).
            forcing = _partial_marker_suffix(generated, end_seq)
        token_id = end_seq[forcing]
        forcing += 1
        if forcing >= len(end_seq):
            done = True
        if token_id >= logits.shape[-1]:
            done = True
            return logits
        return _force_token(logits, token_id)

    return thinking_budget_processor


def _install_thinking_budget_processor(
    lm: "LoadedModel",
    gen_kwargs: dict,
    thinking_budget: int | None,
    *,
    thinking_expected: bool,
    prompt: str | list | None,
    grammar_active: bool,
) -> bool:
    """Install the #716 thinking-budget processor on *gen_kwargs*.

    Returns True when installed. Not enforced (logged) for distributed models
    (callables can't cross the worker broadcast), speculative models (their
    decoders ignore ``logits_processors``, like sampling/penalties), grammar
    requests (forcing a close token the grammar rejects would desync its
    matcher), and tokenizers with no detectable think markers (e.g. gpt-oss
    Harmony channels).
    """
    if thinking_budget is None:
        return False
    if thinking_budget < 0:
        logger.warning("Ignoring negative thinking budget %d", thinking_budget)
        return False
    reason = None
    if lm.is_distributed is True:
        reason = "distributed mode"
    elif lm.is_speculative is True:
        reason = "speculative decoding (decoders ignore logits processors)"
    elif grammar_active:
        reason = "grammar-constrained decoding"
    if reason is not None:
        # info, not warning: Anthropic clients send a budget on every request,
        # so a per-request warning for a model kind that can never enforce it
        # is noise the operator cannot act on.
        logger.info("Thinking budget %d not enforced: %s", thinking_budget, reason)
        return False
    tokenizer = lm.text_tokenizer
    markers = _resolve_think_markers(tokenizer)
    if markers is None:
        logger.info(
            "Thinking budget %d not enforced: tokenizer has no think markers",
            thinking_budget,
        )
        return False
    start_seq, end_seq = markers
    # Only force a close on positive evidence of an open block: the prompt
    # ending in the opener (Qwen3.5/DeepSeek-R1-style templates, with or
    # without an enable_thinking switch) or the model emitting it (replayed
    # by the processor). ``thinking_expected`` is only a request-level flag —
    # trusting it would force ``</think>`` into a direct answer from a model
    # that skipped thinking, and the routers' orphan-close handling (#307)
    # would then reclassify that answer as thinking. It is the fallback only
    # when there is no string prompt to inspect.
    initially_open = bool(thinking_expected)
    if isinstance(prompt, str):
        try:
            start_text = tokenizer.decode(list(start_seq))
        except Exception:
            start_text = None
        if isinstance(start_text, str) and start_text:
            initially_open = prompt.rstrip().endswith(start_text)
    processor = _make_thinking_budget_processor(
        thinking_budget, start_seq, end_seq, initially_open=initially_open
    )
    existing = gen_kwargs.get("logits_processors", [])
    gen_kwargs["logits_processors"] = list(existing) + [processor]
    logger.info(
        "Thinking budget active: %d tokens (open at start=%s)",
        thinking_budget,
        initially_open,
    )
    return True
