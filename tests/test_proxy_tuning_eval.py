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


# ---------------------------------------------------------------------------
# Stage-3: generate_one tests
# ---------------------------------------------------------------------------

from olmlx.proxy_tuning_pipeline.eval_driver import generate_one  # noqa: E402


class _FakeTok:
    eos_token_ids = {9}

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return [1, 2, 3]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class _FakeDecoder:
    """prefill returns first; step replays a scripted token list."""

    def __init__(self, first, rest):
        self._first = first
        self._rest = list(rest)
        self.alpha = None

    def prefill(self, prompt):
        return self._first

    def step(self):
        return [self._rest.pop(0)], 0


def test_generate_one_stops_on_eos():
    dec = _FakeDecoder(first=5, rest=[6, 7, 9, 8])  # 9 == eos
    out = generate_one(
        dec, _FakeTok(), [{"role": "user", "content": "hi"}], max_tokens=20
    )
    assert out == "5 6 7"  # eos (9) excluded, 8 never reached


def test_generate_one_respects_max_tokens():
    dec = _FakeDecoder(first=5, rest=[6, 7, 8, 8, 8])
    out = generate_one(
        dec, _FakeTok(), [{"role": "user", "content": "hi"}], max_tokens=3
    )
    assert out == "5 6 7"  # first + 2 steps == 3 tokens


def test_generate_one_empty_on_immediate_eos():
    dec = _FakeDecoder(first=9, rest=[])  # first token is eos
    out = generate_one(
        dec, _FakeTok(), [{"role": "user", "content": "hi"}], max_tokens=20
    )
    assert out == ""
