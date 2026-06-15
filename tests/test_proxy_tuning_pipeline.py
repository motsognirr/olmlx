"""Tests for the proxy-tuning data pipeline (olmlx/proxy_tuning_pipeline)."""

from __future__ import annotations

import subprocess

from olmlx.proxy_tuning_pipeline.cli import run_pipeline
from olmlx.proxy_tuning_pipeline.curate import (
    dedupe_examples,
    quality_filter,
    split_train_valid,
)
from olmlx.proxy_tuning_pipeline.expand import (
    DEFAULT_MODEL,
    OpenAIGenerator,
    build_messages,
    expand_units,
    parse_pairs,
)
from olmlx.proxy_tuning_pipeline.extract import (
    dedupe_units,
    extract_commits,
    extract_docs,
    extract_functions,
    extract_invariants,
    extract_repo,
    extract_tests,
    strip_secrets,
)
from olmlx.proxy_tuning_pipeline.schema import (
    ChatExample,
    ExtractionUnit,
    read_jsonl,
    write_jsonl,
)


def test_jsonl_round_trip(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = tmp_path / "out.jsonl"
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_extraction_unit_to_dict():
    u = ExtractionUnit(
        kind="function",
        provenance="olmlx/foo.py:10",
        instruction_hint="explain this",
        source_context="def f(): ...",
    )
    assert u.to_dict() == {
        "kind": "function",
        "provenance": "olmlx/foo.py:10",
        "instruction_hint": "explain this",
        "source_context": "def f(): ...",
    }


def test_chat_example_to_jsonl_row_is_mlx_chat_format():
    ex = ChatExample(
        kind="function",
        provenance="olmlx/foo.py:10",
        user="How does f work?",
        assistant="It returns nothing.",
    )
    # mlx-lm chat format: only the `messages` key reaches train.jsonl.
    assert ex.to_chat_row() == {
        "messages": [
            {"role": "user", "content": "How does f work?"},
            {"role": "assistant", "content": "It returns nothing."},
        ]
    }


def test_strip_secrets_redacts_known_token_shapes():
    text = (
        "key sk-abc123def456ghi789jkl012mno345pqr and "
        "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345 plus "
        "OLMLX_SECRET=topsecretvalue123 done"
    )
    out = strip_secrets(text)
    assert "sk-abc123" not in out
    assert "hf_AbCdEf" not in out
    assert "topsecretvalue123" not in out
    assert "[REDACTED]" in out
    # Non-secret text is preserved.
    assert out.startswith("key ") and out.endswith(" done")


def test_strip_secrets_leaves_ordinary_text_untouched():
    text = "def generate_chat(model, messages): return run(model, messages)"
    assert strip_secrets(text) == text


def test_extract_functions_yields_unit_per_function(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text(
        "def add(a, b):\n"
        '    """Return the sum of a and b."""\n'
        "    return a + b\n"
        "\n"
        "def _private():\n"
        "    return 1\n"
    )
    units = list(extract_functions(tmp_path))
    by_name = {u.provenance: u for u in units}
    # Both functions captured; provenance is file:lineno.
    assert any(p.endswith("mod.py:1") for p in by_name)
    add_unit = next(u for u in units if "def add" in u.source_context)
    assert add_unit.kind == "function"
    assert "Return the sum" in add_unit.source_context
    assert "add" in add_unit.instruction_hint


def test_extract_functions_skips_non_python_and_pycache(tmp_path):
    (tmp_path / "note.txt").write_text("def not_python(): pass")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "x.py").write_text("def cached(): pass")
    assert list(extract_functions(tmp_path)) == []


def test_extract_invariants_parses_bold_lead_paragraphs(tmp_path):
    md = tmp_path / "CLAUDE.md"
    md.write_text(
        "# Title\n\n"
        "## Non-Obvious Invariants\n\n"
        "**Metal stream hazard** — All inference must run on one stream.\n\n"
        "**MTP concat order** — embed first, opposite of EAGLE.\n\n"
        "## Development\n\n"
        "not an invariant\n"
    )
    units = list(extract_invariants(md))
    titles = [u.instruction_hint for u in units]
    assert any("Metal stream hazard" in t for t in titles)
    assert any("MTP concat order" in t for t in titles)
    assert all(u.kind == "invariant" for u in units)
    # The trailing "## Development" content is not captured.
    assert not any("not an invariant" in u.source_context for u in units)
    assert len(units) == 2


def test_extract_docs_chunks_by_section(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text(
        "# Intro\n\nWelcome to olmlx.\n\n"
        "## Speculative\n\nDraft then verify.\n\n"
        "## Flash\n\nSSD-backed MoE.\n"
    )
    (d / "nested" / "b.md").parent.mkdir()
    (d / "nested" / "b.md").write_text("# Nested\n\nbody here\n")
    units = list(extract_docs(d))
    headings = [u.instruction_hint for u in units]
    assert any("Speculative" in h for h in headings)
    assert any("Flash" in h for h in headings)
    assert any("Nested" in h for h in headings)
    assert all(u.kind == "doc" for u in units)
    spec = next(u for u in units if "Speculative" in u.instruction_hint)
    assert "Draft then verify" in spec.source_context


def test_extract_tests_yields_unit_per_test_function(tmp_path):
    t = tmp_path / "tests"
    t.mkdir()
    (t / "test_thing.py").write_text(
        "def test_adds():\n"
        '    """Addition works."""\n'
        "    assert 1 + 1 == 2\n"
        "\n"
        "def helper():\n"
        "    return 0\n"
    )
    units = list(extract_tests(t))
    assert len(units) == 1
    u = units[0]
    assert u.kind == "test"
    assert "test_adds" in u.instruction_hint
    assert "assert 1 + 1 == 2" in u.source_context


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": __import__("os").environ.get("PATH", ""),
        },
    )


def test_extract_commits_yields_message_and_diff(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "f.py").write_text("x = 1\n")
    _git(repo, "add", "f.py")
    _git(repo, "commit", "-m", "feat: add x constant")
    units = list(extract_commits(repo, limit=10))
    assert len(units) == 1
    u = units[0]
    assert u.kind == "commit"
    assert "feat: add x constant" in u.source_context
    assert "x = 1" in u.source_context  # diff body included
    assert u.provenance.startswith("git:")


def test_dedupe_units_drops_identical_source_context():
    u1 = ExtractionUnit("function", "a.py:1", "h", "def f(): pass")
    u2 = ExtractionUnit("function", "b.py:9", "h2", "def f(): pass")  # same source
    u3 = ExtractionUnit("function", "c.py:1", "h", "def g(): pass")
    out = dedupe_units([u1, u2, u3])
    assert len(out) == 2
    assert out[0] is u1 and out[1] is u3  # first occurrence kept, order preserved


def test_extract_repo_composes_all_extractors(tmp_path):
    repo = tmp_path / "repo"
    (repo / "olmlx").mkdir(parents=True)
    (repo / "olmlx" / "m.py").write_text('def f():\n    """doc."""\n    return 1\n')
    (repo / "CLAUDE.md").write_text(
        "## Non-Obvious Invariants\n\n**Inv one** — a rule.\n"
    )
    (repo / "docs").mkdir()
    (repo / "docs" / "x.md").write_text("## Sec\n\nbody.\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_m.py").write_text("def test_f():\n    assert True\n")
    units = extract_repo(repo)
    kinds = {u.kind for u in units}
    # Commit extractor finds nothing here (no git repo) — that's fine.
    assert {"function", "invariant", "doc", "test"} <= kinds


# ---------------------------------------------------------------------------
# Task 9: Generator protocol + build_messages + parse_pairs
# ---------------------------------------------------------------------------


def test_build_messages_grounds_in_source_and_requests_json():
    u = ExtractionUnit("function", "a.py:1", "the `f` function", "def f(): return 1")
    system, user = build_messages(u, n_pairs=3)
    assert "ground" in system.lower()  # grounding discipline present
    assert "JSON" in system or "json" in system
    assert "def f(): return 1" in user  # source context embedded
    assert "3" in user  # requested count
    assert "the `f` function" in user  # instruction hint embedded


def test_build_messages_truncates_oversized_source():
    u = ExtractionUnit("doc", "d.md#s", "a doc", "x" * 99999)
    _system, user = build_messages(u, n_pairs=2)
    assert len(user) < 20000  # clamped well below the raw 99999


def test_parse_pairs_extracts_valid_pairs():
    text = '{"pairs": [{"instruction": "Q1", "response": "A1"}, {"instruction": "Q2", "response": "A2"}]}'
    assert parse_pairs(text) == [("Q1", "A1"), ("Q2", "A2")]


def test_parse_pairs_tolerates_code_fences_and_drops_incomplete():
    text = (
        "```json\n"
        '{"pairs": [{"instruction": "Q", "response": "A"}, '
        '{"instruction": "", "response": "x"}, '
        '{"instruction": "only-q"}]}\n'
        "```"
    )
    assert parse_pairs(text) == [("Q", "A")]


def test_parse_pairs_returns_empty_on_garbage():
    assert parse_pairs("not json at all") == []


def test_parse_pairs_extracts_first_object_from_prose_wrapped_reply():
    # Model wraps the JSON in prose and emits a trailing object — the balanced
    # brace walk must grab only the first complete object, not span to the last }.
    text = (
        "Here is the JSON you asked for:\n"
        '{"pairs": [{"instruction": "Q", "response": "A with {braces} inside"}]}\n\n'
        'Also note: {"unrelated": true}'
    )
    assert parse_pairs(text) == [("Q", "A with {braces} inside")]


# ---------------------------------------------------------------------------
# Task 10: expand_units driver
# ---------------------------------------------------------------------------


class _FakeGenerator:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def test_expand_units_produces_chat_examples_with_provenance():
    units = [
        ExtractionUnit("function", "a.py:1", "fn a", "def a(): pass"),
        ExtractionUnit("doc", "d.md#s", "doc s", "body"),
    ]
    gen = _FakeGenerator(
        '{"pairs": [{"instruction": "Q1", "response": "A1"}, '
        '{"instruction": "Q2", "response": "A2"}]}'
    )
    examples = expand_units(units, gen, n_per_unit=2)
    assert gen.calls == 2  # one generation call per unit
    assert len(examples) == 4  # 2 pairs * 2 units
    assert all(isinstance(e, ChatExample) for e in examples)
    first = examples[0]
    assert first.user == "Q1" and first.assistant == "A1"
    assert first.provenance == "a.py:1" and first.kind == "function"


def test_expand_units_skips_units_that_yield_no_pairs():
    units = [ExtractionUnit("doc", "d.md#s", "doc", "body")]
    gen = _FakeGenerator("garbage, not json")
    assert expand_units(units, gen, n_per_unit=3) == []
    assert gen.calls == 1  # the generator was still invoked


def test_expand_units_continues_after_generator_exception():
    class _ExplodingGenerator:
        def generate(self, system: str, user: str) -> str:
            raise RuntimeError("boom")

    units = [
        ExtractionUnit("doc", "a", "h", "b"),
        ExtractionUnit("doc", "c", "h", "d"),
    ]
    # A raising generator must not abort the run; all-failed -> empty result.
    assert expand_units(units, _ExplodingGenerator(), n_per_unit=2) == []


# ---------------------------------------------------------------------------
# Task 11: OpenAIGenerator (injectable client)
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeOpenAIClient:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None

        class _Completions:
            def create(inner, **kwargs):
                self.last_kwargs = kwargs
                return _FakeCompletion(self._content)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_openai_generator_calls_chat_completions():
    client = _FakeOpenAIClient('{"pairs": []}')
    gen = OpenAIGenerator(client=client)
    out = gen.generate("sys", "usr")
    assert out == '{"pairs": []}'
    assert client.last_kwargs["model"] == DEFAULT_MODEL
    assert client.last_kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_openai_generator_honors_model_override():
    client = _FakeOpenAIClient("x")
    OpenAIGenerator(client=client, model="custom-model").generate("s", "u")
    assert client.last_kwargs["model"] == "custom-model"


# ---------------------------------------------------------------------------
# Task 12: Curation — quality_filter + dedupe_examples
# ---------------------------------------------------------------------------


def _ex(user, assistant, kind="function", prov="a.py:1"):
    return ChatExample(kind=kind, provenance=prov, user=user, assistant=assistant)


def test_quality_filter_drops_too_short_and_too_long():
    good = _ex(
        "How does generate_chat route output?", "It calls parse_model_output ..."
    )
    too_short_q = _ex("hi", "a real-enough answer here please")
    too_short_a = _ex("A perfectly reasonable question?", "no")
    # Long-enough user so this exercises the assistant MAX-length guard, not
    # the user MIN-length guard.
    too_long = _ex("How does this work in practice?", "x" * 100_000)
    out = quality_filter([good, too_short_q, too_short_a, too_long])
    assert out == [good]


def test_dedupe_examples_drops_normalized_duplicates():
    a = _ex("How  does   F work?", "Answer A")
    b = _ex("how does f work?", "answer a")  # same after normalization
    c = _ex("Different question?", "Different answer")
    out = dedupe_examples([a, b, c])
    assert out == [a, c]  # first kept, order preserved


# ---------------------------------------------------------------------------
# Task 13: Curation — split_train_valid
# ---------------------------------------------------------------------------


def test_split_train_valid_ratio_and_determinism():
    examples = [_ex(f"Q{i}?", f"answer number {i}") for i in range(100)]
    train1, valid1 = split_train_valid(examples, valid_frac=0.1, seed=42)
    assert len(valid1) == 10 and len(train1) == 90
    # No overlap; union is the whole set.
    assert {id(e) for e in train1}.isdisjoint({id(e) for e in valid1})
    assert len(train1) + len(valid1) == 100
    # Deterministic for a fixed seed.
    train2, valid2 = split_train_valid(examples, valid_frac=0.1, seed=42)
    assert [e.user for e in valid1] == [e.user for e in valid2]


def test_split_train_valid_guarantees_at_least_one_valid():
    examples = [_ex("Q?", "a real answer here")]
    train, valid = split_train_valid(examples, valid_frac=0.1, seed=0)
    assert len(valid) == 1 and len(train) == 0


# ---------------------------------------------------------------------------
# Task 14: CLI wiring + end-to-end test
# ---------------------------------------------------------------------------


def test_run_pipeline_end_to_end(tmp_path):
    # Minimal fixture repo.
    repo = tmp_path / "repo"
    (repo / "olmlx").mkdir(parents=True)
    (repo / "olmlx" / "m.py").write_text('def f():\n    """doc."""\n    return 1\n')
    (repo / "CLAUDE.md").write_text(
        "## Non-Obvious Invariants\n\n**Inv one** — a real rule about streams.\n"
    )
    out_dir = tmp_path / "out"

    # Use a call-varying generator so each unit produces distinct (user, assistant)
    # pairs that survive dedup and both reach the train/valid split.
    class _VaryingGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, system: str, user: str) -> str:
            self.calls += 1
            return (
                f'{{"pairs": [{{"instruction": "Explain unit {self.calls} clearly?", '
                f'"response": "A grounded answer about olmlx unit {self.calls} detail."}}]}}'
            )

    gen = _VaryingGenerator()
    stats = run_pipeline(
        repo_root=repo,
        out_dir=out_dir,
        generator=gen,
        n_per_unit=1,
        valid_frac=0.5,
        seed=7,
    )

    # Artifacts written in mlx-lm chat format.
    train = read_jsonl(out_dir / "train.jsonl")
    valid = read_jsonl(out_dir / "valid.jsonl")
    assert train and valid
    assert set(train[0].keys()) == {"messages"}
    roles = [m["role"] for m in train[0]["messages"]]
    assert roles == ["user", "assistant"]

    # Stats report generated-vs-kept (no silent truncation).
    assert stats["units"] >= 2  # function + invariant
    assert stats["generated"] >= 2
    assert stats["kept"] == stats["train"] + stats["valid"]
    assert stats["train"] == len(train) and stats["valid"] == len(valid)


def test_extract_functions_skips_unparseable_and_non_utf8(tmp_path):
    # A syntactically broken file and a non-UTF8 file must be skipped, not crash
    # extraction over the real repo.
    (tmp_path / "broken.py").write_text("def f(:\n    pass\n")  # SyntaxError
    (tmp_path / "binary.py").write_bytes(b"\xff\xfe def g(): pass")  # not UTF-8
    (tmp_path / "ok.py").write_text("def good():\n    return 1\n")
    units = list(extract_functions(tmp_path))
    names = [u.instruction_hint for u in units]
    assert any("good" in n for n in names)
    assert len(units) == 1  # only the valid file contributed
