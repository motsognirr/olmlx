"""Tests for the Stage-3 proxy-tuning eval harness."""

from __future__ import annotations

import json

import pytest

from olmlx.proxy_tuning_pipeline.eval_schema import (
    EvalPrompt,
    load_eval_prompts,
)


def test_load_eval_prompts_parses_jsonl(tmp_path):
    p = tmp_path / "prompts.jsonl"
    rows = [
        {
            "id": "inv-1",
            "category": "explain_invariant",
            "messages": [
                {"role": "user", "content": "Explain the Metal stream invariant."}
            ],
        },
        {
            "id": "impl-1",
            "category": "implement_convention",
            "messages": [
                {
                    "role": "user",
                    "content": "Add a config flag following olmlx conventions.",
                }
            ],
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    prompts = load_eval_prompts(str(p))

    assert [pr.id for pr in prompts] == ["inv-1", "impl-1"]
    assert isinstance(prompts[0], EvalPrompt)
    assert prompts[0].category == "explain_invariant"
    assert prompts[0].messages[0]["role"] == "user"


def test_load_eval_prompts_rejects_bad_category(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        json.dumps(
            {
                "id": "x",
                "category": "nonsense",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="category"):
        load_eval_prompts(str(p))


def test_load_eval_prompts_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "prompts.jsonl"
    row = {
        "id": "dup",
        "category": "convention_qa",
        "messages": [{"role": "user", "content": "hi"}],
    }
    p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_prompts(str(p))


def test_load_eval_prompts_rejects_missing_field(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        json.dumps({"id": "x", "category": "convention_qa"}) + "\n"
    )  # no messages
    with pytest.raises(ValueError, match="missing required field"):
        load_eval_prompts(str(p))


def test_load_eval_prompts_skips_blank_lines(tmp_path):
    p = tmp_path / "prompts.jsonl"
    row = {
        "id": "a",
        "category": "convention_qa",
        "messages": [{"role": "user", "content": "hi"}],
    }
    p.write_text("\n" + json.dumps(row) + "\n\n")
    prompts = load_eval_prompts(str(p))
    assert [pr.id for pr in prompts] == ["a"]
