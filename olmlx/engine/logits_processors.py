"""Logits processors and decoding-output filters for inference.

Extracted from ``engine/inference.py`` (#454). Holds the request-scoped
logits-processor builders (grammar-constrained decoding, OpenAI
frequency/presence penalties) plus the gpt-oss channel filter that strips
``<|channel|>`` structural tokens from the decoded stream. ``inference.py``
re-imports these names so existing call sites and tests are unchanged.
"""

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

    Call ``should_yield(text)`` for each token. Returns True if the token's text
    should be sent to the client. After the stream ends, call
    ``get_fallback_texts()`` — if non-empty, yield those as fallback (the model
    produced analysis but no final channel).

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
        self._analysis_texts: list[str] = []
        self._full_text_parts: list[str] = []

    def should_yield(self, text: str) -> bool:
        """Process one token's text and return whether it should be yielded."""
        self._full_text_parts.append(text)

        if text == "<|start|>":
            self._state = self._AFTER_START
            self._saw_any_channel = True
            return False

        if text == "<|channel|>":
            self._state = self._EXPECT_CHANNEL
            self._saw_any_channel = True
            return False

        if self._state == self._AFTER_START:
            return False

        if self._state == self._EXPECT_CHANNEL:
            self._channel = text.strip()
            self._state = self._IN_BLOCK
            if self._channel == "final":
                self._saw_final = True
            return False

        if text == "<|message|>" and self._state == self._IN_BLOCK:
            self._state = self._CONTENT
            return False

        if text in ("<|end|>", "<|call|>", "<|return|>"):
            self._state = self._INIT
            self._channel = None
            return False

        if self._state == self._CONTENT and self._channel == "final":
            return True

        if (
            self._state == self._CONTENT
            and self._channel == "analysis"
            and not self._saw_final
        ):
            self._analysis_texts.append(text)
            return False

        if (
            self._state == self._INIT
            and not self._saw_any_channel
            and text not in _GPT_OSS_STRUCTURAL_TOKENS
        ):
            return True

        return False

    def get_fallback_texts(self) -> list[str]:
        """Return buffered analysis texts if no final channel was seen."""
        if not self._saw_final and self._analysis_texts:
            return self._analysis_texts
        return []

    def get_full_text(self) -> str:
        """Return the complete raw text accumulated during streaming."""
        return "".join(self._full_text_parts)


async def _gpt_oss_filter(token_stream):
    """Async generator wrapper for backward compatibility with tests."""
    filt = _GptOssChannelFilter()
    buffered = []
    async for token in token_stream:
        if filt.should_yield(token.text):
            yield token
        else:
            buffered.append(token)
    for text in filt.get_fallback_texts():
        # Find matching token from buffer
        for tok in buffered:
            if tok.text == text:
                yield tok
                buffered.remove(tok)
                break
