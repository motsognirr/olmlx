"""Unit tests for ``olmlx/engine/grammar.py`` (issue #361).

Uses the gpt2 tokenizer because it's small (~50k vocab) and almost
always present in the local HF cache. The actual matcher / mask
behaviour is verified end-to-end against xgrammar; only the routing
helpers (``parse_response_format``) are pure-Python and unit-testable
without a tokenizer.
"""

from __future__ import annotations

import pytest

import mlx.core as mx

from olmlx.engine.grammar import (
    GrammarSpec,
    clear_caches,
    compile_for_tokenizer,
    drop_for_tokenizer,
    make_processor,
    parse_response_format,
)


# ---------------------------------------------------------------------------
# Spec + router parsing — pure-Python, no tokenizer required
# ---------------------------------------------------------------------------


class TestParseResponseFormat:
    def test_none_passes_through(self):
        assert parse_response_format(None) is None

    @pytest.mark.parametrize("value", ["", "text"])
    def test_empty_string_is_none(self, value):
        assert parse_response_format(value) is None

    @pytest.mark.parametrize("value", ["json", "json_object", "JSON", " json "])
    def test_json_strings_map_to_json_object(self, value):
        spec = parse_response_format(value)
        assert spec is not None
        assert spec.kind == "json_object"
        assert spec.schema is None

    def test_openai_json_object_dict(self):
        spec = parse_response_format({"type": "json_object"})
        assert spec is not None
        assert spec.kind == "json_object"
        assert spec.schema is None

    def test_openai_text_type_is_none(self):
        """``response_format={"type": "text"}`` is the unconstrained default."""
        assert parse_response_format({"type": "text"}) is None

    def test_unknown_openai_type_raises_not_silently_falls_through(self):
        """Regression: an unrecognised ``type`` value used to slip into
        the Ollama schema branch via ``"type" in value`` and try to
        compile a non-Schema dict (review #384, bug 2)."""
        with pytest.raises(ValueError, match="unrecognized grammar format dict"):
            parse_response_format({"type": "image_url"})

    def test_bare_schema_with_valid_jsonschema_type(self):
        """Ollama accepts ``format: {"type": "string"}`` (or any other
        JSON-Schema type at the top level)."""
        for t in ("object", "array", "string", "integer", "number", "boolean", "null"):
            spec = parse_response_format({"type": t})
            assert spec is not None, t
            assert spec.kind == "json_schema"
            assert spec.schema == {"type": t}

    def test_openai_json_schema_dict(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        spec = parse_response_format(
            {
                "type": "json_schema",
                "json_schema": {"name": "foo", "schema": schema},
            }
        )
        assert spec is not None
        assert spec.kind == "json_schema"
        assert spec.schema == schema

    def test_openai_json_schema_missing_schema_raises(self):
        with pytest.raises(ValueError, match="schema is required"):
            parse_response_format(
                {"type": "json_schema", "json_schema": {"name": "foo"}}
            )

    def test_bare_schema_dict_is_json_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        spec = parse_response_format(schema)
        assert spec is not None
        assert spec.kind == "json_schema"
        assert spec.schema == schema

    def test_bare_anyof_schema_dict(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        spec = parse_response_format(schema)
        assert spec is not None
        assert spec.kind == "json_schema"
        assert spec.schema == schema

    def test_unrecognised_dict_raises(self):
        with pytest.raises(ValueError, match="unrecognized grammar format dict"):
            parse_response_format({"random": "key"})

    def test_unsupported_string_raises(self):
        with pytest.raises(ValueError, match="unsupported grammar format string"):
            parse_response_format("xml")

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="unsupported grammar format type"):
            parse_response_format(42)


class TestGrammarSpec:
    def test_not_hashable(self):
        """The schema field is a dict; the frozen dataclass would
        otherwise generate a __hash__ that raises TypeError at use site
        instead of failing fast (review #384, nit)."""
        spec = GrammarSpec("json_schema", schema={"type": "object"})
        with pytest.raises(TypeError):
            hash(spec)

    def test_json_object_post_init_rejects_schema(self):
        with pytest.raises(ValueError, match="schema must be None"):
            GrammarSpec(kind="json_object", schema={"type": "string"})

    def test_json_schema_post_init_requires_schema(self):
        with pytest.raises(ValueError, match="schema is required"):
            GrammarSpec(kind="json_schema", schema=None)

    def test_cache_key_stable_for_json_object(self):
        assert GrammarSpec("json_object").cache_key() == "json_object"

    def test_cache_key_independent_of_key_order(self):
        # JSON-Schema serialization sorts keys, so two semantically-identical
        # schemas produce the same cache_key.
        a = GrammarSpec("json_schema", schema={"a": 1, "b": 2})
        b = GrammarSpec("json_schema", schema={"b": 2, "a": 1})
        assert a.cache_key() == b.cache_key()

    def test_cache_key_differs_by_schema(self):
        a = GrammarSpec("json_schema", schema={"type": "string"})
        b = GrammarSpec("json_schema", schema={"type": "integer"})
        assert a.cache_key() != b.cache_key()


# ---------------------------------------------------------------------------
# Compile + processor — require a real tokenizer (gpt2, cached locally)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    """Module-scoped gpt2 tokenizer; falls back to a skip if unavailable."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception as exc:
        pytest.skip(f"gpt2 tokenizer not available: {exc}")
    return tok


@pytest.fixture(autouse=True)
def _reset_grammar_caches():
    clear_caches()
    yield
    clear_caches()


class TestCompileCache:
    def test_compile_same_spec_returns_cached(self, gpt2_tokenizer):
        spec = GrammarSpec("json_object")
        first = compile_for_tokenizer(gpt2_tokenizer, gpt2_tokenizer.vocab_size, spec)
        second = compile_for_tokenizer(gpt2_tokenizer, gpt2_tokenizer.vocab_size, spec)
        # Same compiled object is reused (not just equivalent).
        assert first is second

    def test_compile_different_spec_returns_different(self, gpt2_tokenizer):
        s1 = GrammarSpec("json_object")
        s2 = GrammarSpec(
            "json_schema",
            schema={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
        )
        c1 = compile_for_tokenizer(gpt2_tokenizer, gpt2_tokenizer.vocab_size, s1)
        c2 = compile_for_tokenizer(gpt2_tokenizer, gpt2_tokenizer.vocab_size, s2)
        assert c1 is not c2

    def test_drop_for_tokenizer_removes_entries(self, gpt2_tokenizer):
        """Wire-up safety: ``drop_for_tokenizer`` must remove the cache
        entries for the *specific* tokenizer being unloaded so that a
        future tokenizer landing on the same ``id()`` cannot pick up a
        stale ``CompiledGrammar`` (review #384, bug 1).

        Asserts cache state directly: object identity isn't a reliable
        signal because xgrammar's own ``GrammarCompiler`` can intern
        compiled-grammar objects.
        """
        from olmlx.engine import grammar as _grammar

        spec = GrammarSpec("json_object")
        compile_for_tokenizer(gpt2_tokenizer, gpt2_tokenizer.vocab_size, spec)
        tid = id(gpt2_tokenizer)
        assert any(k[0] == tid for k in _grammar._compile_cache)
        assert tid in _grammar._compiler_cache
        drop_for_tokenizer(gpt2_tokenizer)
        assert not any(k[0] == tid for k in _grammar._compile_cache)
        assert tid not in _grammar._compiler_cache


class TestStaleEntryProtection:
    """CPython recycles ``id()`` addresses, so an entry surviving a
    partially-failed unload could collide with a later tokenizer's id and
    serve a ``CompiledGrammar`` built for a different vocabulary
    (issue #464). Cache entries therefore carry a weakref to the owning
    tokenizer and lookups validate referent identity before serving.
    """

    def test_stale_compile_entry_for_recycled_id_not_served(self, gpt2_tokenizer):
        """An entry left behind by a *different* tokenizer that happens to
        share the lookup id must be ignored (recompiled), not served."""
        import weakref

        from olmlx.engine import grammar as _grammar

        spec = GrammarSpec("json_object")
        other_tokenizer = type("FakeTok", (), {})()  # weakref-able stand-in
        sentinel = object()
        # Simulate the id-collision: a stale entry owned by another
        # tokenizer sits under *this* tokenizer's id.
        key = (id(gpt2_tokenizer), spec.cache_key())
        _grammar._compile_cache[key] = (weakref.ref(other_tokenizer), sentinel)

        result = compile_for_tokenizer(gpt2_tokenizer, gpt2_tokenizer.vocab_size, spec)

        assert result is not sentinel
        # The stale entry was replaced by one owned by the live tokenizer.
        ref, value = _grammar._compile_cache[key]
        assert ref() is gpt2_tokenizer
        assert value is result

    def test_stale_compiler_entry_for_recycled_id_not_used(self, gpt2_tokenizer):
        """Same hazard for the per-tokenizer ``GrammarCompiler`` cache."""
        import weakref

        from olmlx.engine import grammar as _grammar

        other_tokenizer = type("FakeTok", (), {})()
        bogus_compiler = object()  # would blow up if actually used
        _grammar._compiler_cache[id(gpt2_tokenizer)] = (
            weakref.ref(other_tokenizer),
            bogus_compiler,
        )

        spec = GrammarSpec("json_object")
        compiled = compile_for_tokenizer(
            gpt2_tokenizer, gpt2_tokenizer.vocab_size, spec
        )

        assert compiled is not None
        ref, compiler = _grammar._compiler_cache[id(gpt2_tokenizer)]
        assert ref() is gpt2_tokenizer
        assert compiler is not bogus_compiler

    def test_dead_entry_for_recycled_id_not_served(self, gpt2_tokenizer):
        """An entry whose owning tokenizer has been GC'd must never be
        served, even when its key collides with a live tokenizer's id."""
        import gc
        import weakref

        from olmlx.engine import grammar as _grammar

        spec = GrammarSpec("json_object")
        dead_tok = type("FakeTok", (), {})()
        dead_ref = weakref.ref(dead_tok)
        del dead_tok
        gc.collect()
        assert dead_ref() is None
        sentinel = object()
        key = (id(gpt2_tokenizer), spec.cache_key())
        _grammar._compile_cache[key] = (dead_ref, sentinel)

        result = compile_for_tokenizer(gpt2_tokenizer, gpt2_tokenizer.vocab_size, spec)
        assert result is not sentinel
        # A fresh, correctly-owned entry replaced the dead one.
        ref, value = _grammar._compile_cache[key]
        assert ref() is gpt2_tokenizer
        assert value is result

    def test_dead_entries_swept_on_insert(self, gpt2_tokenizer):
        """Entries owned by GC'd tokenizers are purged on the next insert,
        so a failed ``drop_for_tokenizer`` can't leak entries forever."""
        import gc
        import weakref

        from olmlx.engine import grammar as _grammar

        dead_tok = type("FakeTok", (), {})()
        dead_ref = weakref.ref(dead_tok)
        del dead_tok
        gc.collect()
        assert dead_ref() is None
        stale_key = (123456789, "json_object")
        _grammar._compile_cache[stale_key] = (dead_ref, object())
        _grammar._compiler_cache[987654321] = (dead_ref, object())

        spec = GrammarSpec("json_object")
        compile_for_tokenizer(gpt2_tokenizer, gpt2_tokenizer.vocab_size, spec)

        assert stale_key not in _grammar._compile_cache
        assert 987654321 not in _grammar._compiler_cache


class TestLogitsProcessor:
    def test_first_call_treats_tokens_as_prompt(self, gpt2_tokenizer):
        spec = GrammarSpec("json_object")
        proc = make_processor(gpt2_tokenizer, gpt2_tokenizer.vocab_size, spec)
        logits = mx.zeros((1, gpt2_tokenizer.vocab_size))
        # Prompt tokens — should not be fed to the matcher.
        out = proc([1, 2, 3], logits)
        # Some tokens must be masked to -inf (the JSON grammar starts with
        # whitespace or `{`/`[`/`"`/digit/`-`/`t`/`f`/`n`); ordinary words
        # should be -inf.
        out_np = (out == -mx.inf).sum().item()
        assert out_np > 0, "expected at least one token to be masked"

    def test_advances_on_new_tokens(self, gpt2_tokenizer):
        spec = GrammarSpec("json_object")
        proc = make_processor(gpt2_tokenizer, gpt2_tokenizer.vocab_size, spec)
        logits = mx.zeros((1, gpt2_tokenizer.vocab_size))

        # Find an "opening brace" token id. xgrammar's mask reveals it.
        proc([1, 2, 3], logits)  # initial prompt call
        # Pull the bitmask directly off the processor to find an allowed token.
        proc._matcher.fill_next_token_bitmask(proc._bitmask)
        import numpy as np

        unpacked = np.unpackbits(
            proc._bitmask.numpy().view(np.uint8), bitorder="little"
        ).astype(bool)
        allowed_ids = np.where(unpacked[: gpt2_tokenizer.vocab_size])[0]
        assert len(allowed_ids) > 0
        # Pick a token whose decoded text is "{" if available, otherwise
        # the first allowed token. We don't actually care which token —
        # the test asserts that ``accept_token`` advances state.
        chosen = int(allowed_ids[0])
        # The processor's "next call" should accept ``chosen`` into the matcher.
        out2 = proc([1, 2, 3, chosen], logits)
        assert out2.shape == logits.shape
        # The internal counter must have advanced.
        assert proc._last_token_count == 4

    def test_dtype_preserved(self, gpt2_tokenizer):
        spec = GrammarSpec("json_object")
        proc = make_processor(gpt2_tokenizer, gpt2_tokenizer.vocab_size, spec)
        logits = mx.zeros((1, gpt2_tokenizer.vocab_size), dtype=mx.float16)
        out = proc([1, 2, 3], logits)
        assert out.dtype == mx.float16

    def test_terminated_matcher_returns_logits_unmodified(self, gpt2_tokenizer):
        """When the matcher is fully terminated, the processor should
        return logits unchanged. We simulate by constructing a processor
        and monkey-patching ``is_terminated`` to True after the first call."""
        spec = GrammarSpec("json_object")
        proc = make_processor(gpt2_tokenizer, gpt2_tokenizer.vocab_size, spec)
        logits = mx.ones((1, gpt2_tokenizer.vocab_size))
        proc([1, 2, 3], logits)  # initial
        # Force termination.
        original = proc._matcher.is_terminated
        proc._matcher.is_terminated = lambda: True  # type: ignore[method-assign]
        try:
            out = proc([1, 2, 3, 99], logits)
            # No masking → all ones preserved.
            assert (out == 1.0).all().item()
        finally:
            proc._matcher.is_terminated = original  # type: ignore[method-assign]
