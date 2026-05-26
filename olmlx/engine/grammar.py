"""Grammar-constrained decoding via xgrammar (issue #361).

Exposes a tiny surface:

- ``GrammarSpec`` — what to constrain (``"json_object"`` or a JSON Schema).
- ``compile_for_tokenizer(tokenizer, vocab_size, spec)`` — returns a
  ``CompiledGrammar``; cached per (tokenizer-id, spec-hash) so repeat
  requests with the same schema pay the ~7 ms compile only once.
- ``GrammarLogitsProcessor`` — mlx-lm-compatible callable
  ``(tokens, logits) -> logits`` that masks disallowed tokens. Advances the
  matcher by reading newly appended tokens out of the running ``tokens``
  list, so no separate "post-sample" hook is required.

Speculative decoding is **not** plumbed through here; callers must
disable speculative when ``grammar_spec`` is set on a request. See
``inference.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx
import numpy as np

try:
    import xgrammar as xgr
except ImportError:  # pragma: no cover
    xgr = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrammarSpec:
    """Describes a grammar constraint for a single request.

    ``kind="json_object"`` allows any well-formed JSON value (xgrammar's
    builtin JSON grammar). ``kind="json_schema"`` requires ``schema`` to be
    a JSON-Schema dict.
    """

    kind: Literal["json_object", "json_schema"]
    schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind == "json_schema" and self.schema is None:
            raise ValueError("schema is required for kind='json_schema'")
        if self.kind == "json_object" and self.schema is not None:
            raise ValueError("schema must be None for kind='json_object'")

    def cache_key(self) -> str:
        if self.kind == "json_object":
            return "json_object"
        # Sort keys so equivalent schemas hash to the same key.
        payload = json.dumps(self.schema, sort_keys=True, separators=(",", ":"))
        return "json_schema:" + hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Compile cache
# ---------------------------------------------------------------------------


# Module-level cache keyed on (tokenizer id, spec cache_key). Tokenizer ID
# is ``id(tokenizer)`` — safe because mlx-lm holds a stable reference per
# loaded model, and entries are dropped on model unload (the tokenizer is
# garbage-collected, so the id is freed; stale entries are not an issue in
# practice for this single-user server because there is only one active
# model at a time). The cache is keyed by spec so repeat requests with the
# same schema reuse the ~7 ms compile.
_compile_cache: dict[tuple[int, str], Any] = {}
_compile_cache_lock = threading.Lock()

# A separate cache for the per-tokenizer ``GrammarCompiler`` itself, since
# constructing it walks the full vocab once. Keyed on ``id(tokenizer)``.
_compiler_cache: dict[int, Any] = {}
_compiler_cache_lock = threading.Lock()


def _get_compiler(tokenizer: Any, vocab_size: int) -> Any:
    """Return a memoized ``GrammarCompiler`` for *tokenizer*."""
    if xgr is None:
        raise RuntimeError(
            "xgrammar is not installed — grammar-constrained decoding unavailable"
        )
    key = id(tokenizer)
    with _compiler_cache_lock:
        compiler = _compiler_cache.get(key)
        if compiler is not None:
            return compiler
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        compiler = xgr.GrammarCompiler(info, max_threads=4)
        _compiler_cache[key] = compiler
        return compiler


def compile_for_tokenizer(tokenizer: Any, vocab_size: int, spec: GrammarSpec) -> Any:
    """Compile *spec* against *tokenizer*; cached.

    *vocab_size* is the model's lm_head vocab dimension, which may differ
    from ``tokenizer.vocab_size`` (Phi-3, Llama-3.2-Vision, etc.).
    """
    if xgr is None:
        raise RuntimeError(
            "xgrammar is not installed — grammar-constrained decoding unavailable"
        )
    cache_key = (id(tokenizer), spec.cache_key())
    with _compile_cache_lock:
        cached = _compile_cache.get(cache_key)
        if cached is not None:
            return cached

    compiler = _get_compiler(tokenizer, vocab_size)
    if spec.kind == "json_object":
        compiled = compiler.compile_builtin_json_grammar()
    else:
        assert spec.schema is not None  # post_init enforces
        compiled = compiler.compile_json_schema(spec.schema)

    with _compile_cache_lock:
        _compile_cache[cache_key] = compiled
    return compiled


# ---------------------------------------------------------------------------
# Logits processor
# ---------------------------------------------------------------------------


class GrammarLogitsProcessor:
    """mlx-lm-compatible logits processor that enforces *compiled_grammar*.

    The matcher is advanced by inspecting ``tokens`` on each call — the
    list grows by one per generation step, and we accept whichever tokens
    are new since the previous call. This mirrors the bookkeeping in the
    existing ``_make_frequency_penalty_processor``.

    The first call's ``tokens`` argument may include the prompt
    (mlx-lm passes the full token list). We treat all tokens present on
    the first call as the prefix and do not feed them to the matcher;
    only tokens generated *after* the first call are accepted.
    """

    def __init__(self, compiled_grammar: Any, vocab_size: int) -> None:
        if xgr is None:
            raise RuntimeError(
                "xgrammar is not installed — grammar-constrained decoding unavailable"
            )
        self._matcher = xgr.GrammarMatcher(compiled_grammar)
        self._vocab_size = vocab_size
        # Pre-allocate the int32 packed bitmask buffer (~19 KB for vocab=151k).
        self._bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        # Cached neg-inf scalar for masking; re-cast per call to match
        # the logits dtype.
        self._last_token_count: int | None = None

    # The signature must match the existing penalty processors so it can
    # be inserted into the same ``logits_processors`` list.
    def __call__(self, tokens: list[int], logits: mx.array) -> mx.array:
        if self._last_token_count is None:
            # First call: ``tokens`` is the prompt. Do not feed to matcher.
            self._last_token_count = len(tokens)
        else:
            # Accept any tokens generated since the previous call. In the
            # common case there's exactly one new token per step, but be
            # defensive against multi-token jumps in case a caller batches.
            for tid in tokens[self._last_token_count :]:
                # ``accept_token`` returns False if the token is not in
                # the accepted set; this would indicate either (a) a
                # tokens-list reset we don't expect, or (b) speculative
                # decoding bypassing the mask. Log and skip-advance so
                # one bad token doesn't permanently lock the matcher.
                if not self._matcher.accept_token(int(tid)):
                    logger.warning(
                        "GrammarLogitsProcessor: rejected accept_token(%d); "
                        "skipping advance — output past this point may not "
                        "conform to schema",
                        int(tid),
                    )
            self._last_token_count = len(tokens)

        if self._matcher.is_terminated():
            # Grammar completed — nothing more to mask; mlx-lm will emit
            # the EOS-equivalent on its own loop. Returning logits
            # unchanged is safe because the matcher already accepted the
            # terminating token(s).
            return logits

        # Fill bitmask. Cheap (~0.15 ms for vocab=151k).
        self._matcher.fill_next_token_bitmask(self._bitmask)
        # ``self._bitmask`` is a torch int32 tensor of shape
        # (1, ceil(vocab_size/32)). Convert to an MLX bool mask of shape
        # (vocab_size,) and apply.
        bitmask_np = self._bitmask.numpy()  # zero-copy view of CPU tensor
        # unpackbits is little-endian per byte; xgrammar packs little-endian
        # within each int32, so a uint8 little-bitorder view matches.
        unpacked = np.unpackbits(bitmask_np.view(np.uint8), bitorder="little").astype(
            bool
        )
        # Trim to the model's vocab_size (the packed array is rounded up
        # to a multiple of 32).
        allowed = unpacked[: self._vocab_size]
        allowed_mx = mx.array(allowed)
        # mlx-lm logits have shape (1, vocab_size) in the streaming path
        # but can also be (vocab_size,) in other paths; broadcast naturally.
        neg_inf = mx.array(-mx.inf, dtype=logits.dtype)
        return mx.where(allowed_mx, logits, neg_inf)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_processor(
    tokenizer: Any, vocab_size: int, spec: GrammarSpec
) -> GrammarLogitsProcessor:
    """Compile *spec* (cached) and return a fresh per-request processor."""
    compiled = compile_for_tokenizer(tokenizer, vocab_size, spec)
    return GrammarLogitsProcessor(compiled, vocab_size)


def parse_response_format(value: Any) -> GrammarSpec | None:
    """Normalize router-level input into a ``GrammarSpec`` or ``None``.

    Accepted shapes:

    - ``None`` / empty / ``"text"`` → ``None`` (no constraint)
    - ``"json"`` or ``"json_object"`` → ``json_object`` mode
    - a dict that looks like a JSON Schema (has ``type`` / ``properties`` /
      ``$schema``) → ``json_schema`` mode with that schema
    - an OpenAI-style ``{"type": "json_object"}`` or
      ``{"type": "json_schema", "json_schema": {"schema": {...}}}`` →
      mapped accordingly

    Anything else is rejected with ``ValueError``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "text"):
            return None
        if s in ("json", "json_object"):
            return GrammarSpec(kind="json_object")
        raise ValueError(f"unsupported grammar format string: {value!r}")
    if isinstance(value, dict):
        # OpenAI shape.
        if value.get("type") == "json_object":
            return GrammarSpec(kind="json_object")
        if value.get("type") == "json_schema":
            js = value.get("json_schema") or {}
            schema = js.get("schema")
            if schema is None:
                raise ValueError("response_format.json_schema.schema is required")
            return GrammarSpec(kind="json_schema", schema=schema)
        # Ollama shape: format is itself the schema dict.
        if any(k in value for k in ("type", "properties", "$schema", "anyOf", "oneOf")):
            return GrammarSpec(kind="json_schema", schema=value)
        raise ValueError(
            "unrecognized grammar format dict (need OpenAI response_format shape "
            "or a JSON Schema with 'type'/'properties'/'$schema')"
        )
    raise ValueError(f"unsupported grammar format type: {type(value).__name__}")


def clear_caches() -> None:
    """Drop compile/compiler caches. Call on model unload."""
    with _compile_cache_lock:
        _compile_cache.clear()
    with _compiler_cache_lock:
        _compiler_cache.clear()
