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


# ---------------------------------------------------------------------------
# Stage-3: ProxyEvalJudge tests
# ---------------------------------------------------------------------------

from olmlx.proxy_tuning_pipeline.eval_judge import ProxyEvalJudge  # noqa: E402


class _FakeGen:
    def __init__(self, reply):
        self.reply = reply
        self.last_system = None
        self.last_user = None

    def generate(self, system, user):
        self.last_system = system
        self.last_user = user
        return self.reply


def test_judge_parses_scores_and_builds_prompt():
    gen = _FakeGen('{"convention_adherence": 4, "coherence": 5, "rationale": "good"}')
    judge = ProxyEvalJudge(gen)
    score = judge.score(
        prompt="Explain the invariant.",
        completion="The invariant is ...",
    )
    assert score == (4, 5, "good")
    assert "Explain the invariant." in gen.last_user
    assert "The invariant is ..." in gen.last_user


def test_judge_clamps_out_of_range():
    gen = _FakeGen('{"convention_adherence": 9, "coherence": 0, "rationale": "x"}')
    judge = ProxyEvalJudge(gen)
    conv, coh, _ = judge.score(prompt="p", completion="c")
    assert (conv, coh) == (5, 1)


def test_judge_tolerates_fenced_json():
    gen = _FakeGen(
        '```json\n{"convention_adherence": 3, "coherence": 3, "rationale": "ok"}\n```'
    )
    judge = ProxyEvalJudge(gen)
    assert judge.score(prompt="p", completion="c") == (3, 3, "ok")


def test_judge_raises_on_unparseable():
    judge = ProxyEvalJudge(_FakeGen("not json at all"))
    with pytest.raises(ValueError, match="judge"):
        judge.score(prompt="p", completion="c")


from olmlx.proxy_tuning_pipeline.eval import aggregate, ship_decision, run_eval  # noqa: E402
from olmlx.proxy_tuning_pipeline.eval_schema import EvalScore  # noqa: E402


def _score(alpha, conv, coh, pid="p"):
    return EvalScore(pid, "convention_qa", alpha, conv, coh, "r", "out")


def test_aggregate_means_per_alpha():
    scores = [
        _score(0.0, 2, 4),
        _score(0.0, 4, 4),
        _score(1.0, 5, 4),
        _score(1.0, 3, 4),
    ]
    summaries = {s.alpha: s for s in aggregate(scores)}
    assert summaries[0.0].mean_convention == 3.0
    assert summaries[1.0].mean_convention == 4.0
    assert summaries[0.0].n == 2


def test_ship_decision_ships_on_clear_lift():
    scores = [_score(0.0, 3, 5), _score(1.0, 4, 5)]  # +1.0 conv, equal coherence
    d = ship_decision(aggregate(scores))
    assert d.ship is True
    assert d.best_alpha == 1.0


def test_ship_decision_blocks_on_coherence_drop():
    # +1.0 conv but coherence falls 5 -> 4.0 (drop 1.0 > 0.2 allowed)
    scores = [_score(0.0, 3, 5), _score(1.0, 4, 4)]
    d = ship_decision(aggregate(scores))
    assert d.ship is False
    assert "coherence" in d.reason


def test_ship_decision_blocks_on_insufficient_margin():
    scores = [_score(0.0, 3, 5), _score(1.0, 3, 5)]  # no lift
    d = ship_decision(aggregate(scores))
    assert d.ship is False


def test_run_eval_orchestration_with_fakes(tmp_path, monkeypatch):
    prompts = [EvalPrompt("a", "convention_qa", [{"role": "user", "content": "hi"}])]

    class _Dec:
        def __init__(self):
            self._alpha = None

    def fake_loader(base, expert, anti):
        return ("BASE", "EXP", "ANTI", "TOK")

    def fake_decoder_factory(base, expert, anti, alpha):
        d = _Dec()
        d._alpha = alpha
        return d

    def fake_generate(decoder, tokenizer, messages, *, max_tokens):
        return f"out@{decoder._alpha}"

    class _Judge:
        def score(self, *, prompt, completion):
            alpha = float(completion.split("@")[1])
            return (5 if alpha > 0 else 3, 5, "r")

    monkeypatch.setattr("olmlx.proxy_tuning_pipeline.eval.generate_one", fake_generate)

    out_path = tmp_path / "results.json"
    report = run_eval(
        base_dir="b",
        expert_dir="e",
        antiexpert_dir="a",
        prompts=prompts,
        alphas=[0.0, 1.0],
        judge=_Judge(),
        out_path=str(out_path),
        loader=fake_loader,
        decoder_factory=fake_decoder_factory,
        max_tokens=8,
        preflight=lambda *a, **k: None,
    )
    assert report.ship is True
    assert out_path.exists()
