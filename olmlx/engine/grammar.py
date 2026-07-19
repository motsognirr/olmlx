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
import re
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx
import numpy as np

try:
    import xgrammar as xgr
except ImportError:  # pragma: no cover
    xgr = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


_OPENAI_RESPONSE_FORMAT_TYPES = frozenset({"text", "json_object", "json_schema"})

_JSON_SCHEMA_TYPES = frozenset(
    {"object", "array", "string", "number", "integer", "boolean", "null"}
)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrammarSpec:
    """Describes a grammar constraint for a single request.

    ``kind="json_object"`` constrains output to a JSON object (``{}``),
    matching the OpenAI spec guarantee. ``kind="json_schema"`` requires
    ``schema`` to be a JSON-Schema dict.

    Note: ``frozen=True`` would normally generate ``__hash__``, but the
    ``schema`` field is a dict and not hashable, so attempting to hash a
    ``GrammarSpec`` would raise ``TypeError`` at use site instead of
    failing fast. Explicitly disabling hashing keeps that contract clear;
    callers wanting a deduplication key should use ``cache_key()``.
    """

    kind: Literal["json_object", "json_schema"]
    schema: dict[str, Any] | None = None

    __hash__ = None  # type: ignore[assignment]

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
# is ``id(tokenizer)``, and CPython freely recycles addresses, so every
# entry also carries a ``weakref.ref`` to the owning tokenizer; lookups
# validate that the referent is *the same live object* before serving.
# Without that check, an entry surviving a partially-failed unload
# (issue #464 — ``unload()`` absorbs ``_close_loaded_model`` failures)
# could collide with a later tokenizer's id and return a stale
# ``CompiledGrammar`` built for a different vocabulary, producing
# wrong-token masks. ``drop_for_tokenizer`` (wired into
# ``ModelManager._close_loaded_model``) remains the deterministic
# cleanup path; dead entries it missed are additionally swept on insert
# so they cannot accumulate.
_compile_cache: dict[tuple[int, str], tuple[weakref.ref[Any], Any]] = {}
_compile_cache_lock = threading.Lock()

# A separate cache for the per-tokenizer ``GrammarCompiler`` itself, since
# constructing it walks the full vocab once. Keyed on ``id(tokenizer)``,
# same (weakref, value) entry shape and validation as ``_compile_cache``.
_compiler_cache: dict[int, tuple[weakref.ref[Any], Any]] = {}
_compiler_cache_lock = threading.Lock()


# Classes already warned about in _make_ref, to keep the log at one line
# per offending tokenizer type rather than one per request. Guarded by
# its own lock: _make_ref is called under both cache locks, so neither
# alone makes the check-then-add atomic.
_unrefable_warned_classes: set[type] = set()
_unrefable_warned_lock = threading.Lock()


def _make_ref(tokenizer: Any) -> weakref.ref[Any] | None:
    """A weakref to *tokenizer*, or ``None`` if it isn't weakref-able.

    HF tokenizers are plain Python classes and always weakref-able; the
    fallback only matters for exotic stand-ins, which then simply skip
    caching (correctness over speed). That silently turns every
    structured-output request into a full recompile (grammar + the
    vocab-walking GrammarCompiler), so warn once per class to make the
    degradation diagnosable.
    """
    try:
        return weakref.ref(tokenizer)
    except TypeError:
        cls = type(tokenizer)
        with _unrefable_warned_lock:
            if cls in _unrefable_warned_classes:
                return None
            _unrefable_warned_classes.add(cls)
        logger.warning(
            "tokenizer type %s is not weakref-able; grammar compile "
            "caching is disabled for it — every structured-output "
            "request recompiles the grammar",
            cls.__name__,
        )
        return None


def _live_value(entry: tuple[weakref.ref[Any], Any] | None, tokenizer: Any) -> Any:
    """*entry*'s cached value if it is owned by this live *tokenizer*.

    ``None`` for a missing entry, a dead referent, or an entry owned by
    a different tokenizer whose ``id()`` was recycled (issue #464) —
    serving the latter would apply a grammar compiled for a different
    vocabulary.
    """
    if entry is not None and entry[0]() is tokenizer:
        return entry[1]
    return None


def _sweep_dead_entries_locked(cache: dict[Any, tuple[weakref.ref[Any], Any]]) -> None:
    """Remove entries whose owning tokenizer has been GC'd.

    Caller must hold the cache's lock. Bounds leakage when
    ``drop_for_tokenizer`` was skipped by a failed unload — dead entries
    can never be *served* (identity validation) but would otherwise
    linger forever.
    """
    dead = [k for k, (ref, _) in cache.items() if ref() is None]
    for k in dead:
        del cache[k]


def _get_compiler(tokenizer: Any, vocab_size: int) -> Any:
    """Return a memoized ``GrammarCompiler`` for *tokenizer*."""
    if xgr is None:
        raise RuntimeError(
            "xgrammar is not installed — grammar-constrained decoding unavailable"
        )
    key = id(tokenizer)
    with _compiler_cache_lock:
        cached = _live_value(_compiler_cache.get(key), tokenizer)
        if cached is not None:
            return cached
    # Build OUTSIDE the lock: ``from_huggingface`` + ``GrammarCompiler`` walk
    # the full vocabulary (the heaviest step) and would otherwise serialize
    # every concurrent structured-output request — including ones for
    # *unrelated* tokenizers — behind this single mutex (#634). The
    # duplicate-build race is harmless: compilers for the same tokenizer are
    # equivalent, so the loser of the race just discards its result — the same
    # losers-discard reconciliation ``_compile_cache`` uses.
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(info, max_threads=4)
    with _compiler_cache_lock:
        cached = _live_value(_compiler_cache.get(key), tokenizer)
        if cached is not None:
            return cached
        _sweep_dead_entries_locked(_compiler_cache)
        ref = _make_ref(tokenizer)
        if ref is not None:
            _compiler_cache[key] = (ref, compiler)
        return compiler


def compile_for_tokenizer(tokenizer: Any, vocab_size: int, spec: GrammarSpec) -> Any:
    """Compile *spec* against *tokenizer*; cached.

    *vocab_size* is the model's lm_head vocab dimension, which may differ
    from ``tokenizer.vocab_size`` (Phi-3, Llama-3.2-Vision, etc.).

    Uses double-checked locking so the ~7 ms compile (and the much
    heavier first-time GrammarCompiler construction, which walks the
    full vocabulary) does not serialise all concurrent requests behind
    a single mutex. The duplicate-compile race in the gap is harmless:
    compiled grammars are immutable and equivalent, so the loser of the
    race just discards its result.
    """
    if xgr is None:
        raise RuntimeError(
            "xgrammar is not installed — grammar-constrained decoding unavailable"
        )
    cache_key = (id(tokenizer), spec.cache_key())
    with _compile_cache_lock:
        # _live_value serves only entries owned by *this live tokenizer*
        # — an id-keyed entry may belong to a dead tokenizer whose
        # address was recycled (issue #464).
        cached = _live_value(_compile_cache.get(cache_key), tokenizer)
        if cached is not None:
            return cached
    # Compile outside the lock — see docstring.
    compiler = _get_compiler(tokenizer, vocab_size)
    if spec.kind == "json_object":
        compiled = compiler.compile_json_schema({"type": "object"})
    else:
        assert spec.schema is not None  # post_init enforces
        compiled = compiler.compile_json_schema(spec.schema)
    with _compile_cache_lock:
        # A concurrent compile may have stored already; prefer the
        # existing entry so callers see object identity stability.
        existing = _live_value(_compile_cache.get(cache_key), tokenizer)
        if existing is not None:
            return existing
        _sweep_dead_entries_locked(_compile_cache)
        ref = _make_ref(tokenizer)
        if ref is not None:
            _compile_cache[cache_key] = (ref, compiled)
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
        self._last_token_count: int | None = None
        # Cache the neg-inf scalar per logits dtype so the hot path avoids
        # a small Metal allocation every step.
        self._neg_inf_cache: dict[Any, mx.array] = {}

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
        # numpy.unpackbits returns uint8 (0/1); mx.where treats non-zero
        # as true, so the .astype(bool) round-trip in the previous
        # version was an extra ~150 KB per-step allocation for no
        # behavioural change. (np.unpackbits does not yet accept ``out=``,
        # so the unpack itself still allocates — would-be reviewer fix
        # for that is blocked on the numpy API.)
        unpacked = np.unpackbits(bitmask_np.view(np.uint8), bitorder="little")
        # Trim to the model's vocab_size (the packed array is rounded up
        # to a multiple of 32).
        allowed_mx = mx.array(unpacked[: self._vocab_size])
        # mlx-lm logits have shape (1, vocab_size) in the streaming path
        # but can also be (vocab_size,) in other paths; broadcast naturally.
        neg_inf = self._neg_inf_cache.get(logits.dtype)
        if neg_inf is None:
            neg_inf = mx.array(-mx.inf, dtype=logits.dtype)
            self._neg_inf_cache[logits.dtype] = neg_inf
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


_XGRAMMAR_TIMESTAMP_RE = re.compile(r"^\s*\[[^\]]*\]\s*")
# An absolute source-file location (``/a/b/file.cc:123``, optional trailing
# ``:``) anywhere in the message. Kept independent of the leading-timestamp
# strip so a change to xgrammar's message layout can't leak a build path.
_XGRAMMAR_SRC_LOCATION_RE = re.compile(r"/\S+\.(?:cc|cpp|cxx|hpp|h|c|py):\d+:?\s*")


def _sanitize_xgrammar_error(msg: str) -> str:
    """Reduce an xgrammar error to its actionable tail without leaking the
    internal C++ build path/line to the client.

    xgrammar's canonical layout is ``[HH:MM:SS] /abs/path/file.cc:NNN: <detail>``,
    but rather than depend on that exact shape we strip the leading timestamp
    bracket *and* any absolute source-file location wherever it appears. If
    nothing meaningful survives, fall back to a generic string — never the raw
    message, which could itself be a bare path.
    """
    cleaned = _XGRAMMAR_TIMESTAMP_RE.sub("", msg.strip())
    cleaned = _XGRAMMAR_SRC_LOCATION_RE.sub("", cleaned).strip()
    return cleaned or "schema compilation failed"


def _reject_invalid_json_schema(schema: Any) -> None:
    """Validate *schema* eagerly so a malformed JSON Schema surfaces as a
    clean client error at request-parse time instead of crashing xgrammar's
    compiler mid-generation with a leaked-C++-path 500 (issue #645).

    ``Grammar.from_json_schema`` runs the exact same schema→grammar
    conversion the compiler does (``json_schema_converter``) but needs no
    tokenizer, so it is a cheap, faithful pre-check. Skipped when xgrammar is
    unavailable — grammar decoding is already inert in that case.
    """
    if xgr is None:
        return
    try:
        xgr.Grammar.from_json_schema(schema)
    except RuntimeError as exc:
        # xgrammar raises RuntimeError (incl. InvalidJSONError subclasses)
        # for unsupported types / malformed schemas — all client errors.
        raise ValueError(
            "invalid JSON schema in response_format: "
            + _sanitize_xgrammar_error(str(exc))
        ) from exc


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
        # OpenAI shape — dispatch only on the values we explicitly support.
        # An unrecognised ``type`` (e.g. ``{"type": "text"}`` or a future
        # OpenAI extension) must NOT silently fall through to the Ollama
        # schema branch via ``"type" in value``, or xgrammar would try to
        # compile a non-JSON-Schema dict.
        openai_type = value.get("type")
        if openai_type in _OPENAI_RESPONSE_FORMAT_TYPES:
            if openai_type == "text":
                return None
            if openai_type == "json_object":
                return GrammarSpec(kind="json_object")
            # json_schema
            js = value.get("json_schema") or {}
            schema = js.get("schema")
            if schema is None:
                raise ValueError("response_format.json_schema.schema is required")
            _reject_invalid_json_schema(schema)
            return GrammarSpec(kind="json_schema", schema=schema)
        # Ollama shape: ``format`` is itself a JSON Schema dict. Accept
        # only schemas with explicit JSON-Schema markers OR a ``type``
        # field whose value is a real JSON-Schema type (so an OpenAI-style
        # ``{"type": "image_url"}`` doesn't sneak through here).
        if any(k in value for k in ("properties", "$schema", "anyOf", "oneOf")):
            _reject_invalid_json_schema(value)
            return GrammarSpec(kind="json_schema", schema=value)
        if value.get("type") in _JSON_SCHEMA_TYPES:
            _reject_invalid_json_schema(value)
            return GrammarSpec(kind="json_schema", schema=value)
        raise ValueError(
            "unrecognized grammar format dict (need OpenAI response_format shape "
            "with type in {text,json_object,json_schema}, or a JSON Schema with "
            "'properties'/'$schema'/'anyOf'/'oneOf' or a JSON-Schema 'type')"
        )
    raise ValueError(f"unsupported grammar format type: {type(value).__name__}")


def clear_caches() -> None:
    """Drop compile/compiler caches. Call on model unload."""
    with _compile_cache_lock:
        _compile_cache.clear()
    with _compiler_cache_lock:
        _compiler_cache.clear()


def unwrap_mlx_tokenizer(tokenizer: Any) -> Any:
    """Peel mlx-lm's ``TokenizerWrapper`` down to the underlying HF
    tokenizer (xgrammar's strict isinstance check rejects the wrapper).

    Only peels when the outer class is named ``TokenizerWrapper`` —
    HF fast tokenizers also expose a ``_tokenizer`` attribute (holding
    the Rust ``tokenizers.Tokenizer`` core), and peeling there would
    hand xgrammar an unsupported type AND make the id-keyed cache
    look up the wrong object.
    """
    if type(tokenizer).__name__ == "TokenizerWrapper":
        inner = getattr(tokenizer, "_tokenizer", None)
        if inner is not None:
            return inner
    return tokenizer


def drop_for_tokenizer(tokenizer: Any) -> None:
    """Drop cache entries keyed on *tokenizer*'s id.

    Must be called *while the tokenizer object is still alive* — the
    cache key is ``id(tokenizer)`` and CPython can recycle freed
    addresses for new allocations. Called from
    ``ModelManager._close_loaded_model`` (first, so no earlier close
    failure can skip it — issue #464) before the LoadedModel's tokenizer
    reference is nulled.

    This is the deterministic cleanup path; correctness no longer
    depends on it. Entries carry a weakref to their owning tokenizer
    and lookups validate referent identity, so an entry that survives a
    skipped/failed drop can never be served to a colliding tokenizer
    and is swept on the next cache insert.
    """
    tid = id(unwrap_mlx_tokenizer(tokenizer))
    with _compile_cache_lock:
        stale = [k for k in _compile_cache if k[0] == tid]
        for k in stale:
            del _compile_cache[k]
    with _compiler_cache_lock:
        _compiler_cache.pop(tid, None)
