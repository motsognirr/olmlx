"""Tests for the proxy-tuning data pipeline (olmlx/proxy_tuning_pipeline)."""

from __future__ import annotations

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


from olmlx.proxy_tuning_pipeline.extract import strip_secrets


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


from olmlx.proxy_tuning_pipeline.extract import extract_functions


def test_extract_functions_yields_unit_per_function(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text(
        'def add(a, b):\n'
        '    """Return the sum of a and b."""\n'
        '    return a + b\n'
        '\n'
        'def _private():\n'
        '    return 1\n'
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


from olmlx.proxy_tuning_pipeline.extract import extract_invariants


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


from olmlx.proxy_tuning_pipeline.extract import extract_docs


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


from olmlx.proxy_tuning_pipeline.extract import extract_tests


def test_extract_tests_yields_unit_per_test_function(tmp_path):
    t = tmp_path / "tests"
    t.mkdir()
    (t / "test_thing.py").write_text(
        'def test_adds():\n'
        '    """Addition works."""\n'
        '    assert 1 + 1 == 2\n'
        '\n'
        'def helper():\n'
        '    return 0\n'
    )
    units = list(extract_tests(t))
    assert len(units) == 1
    u = units[0]
    assert u.kind == "test"
    assert "test_adds" in u.instruction_hint
    assert "assert 1 + 1 == 2" in u.source_context


import subprocess

from olmlx.proxy_tuning_pipeline.extract import extract_commits


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
