"""Tests for olmlx.engine.panel."""

import pytest

from olmlx.engine import panel as panel_mod
from olmlx.engine.grammar import GrammarSpec
from olmlx.engine.panel import (
    build_judge_messages,
    first_user_text,
    merge_tool_calls,
    route_grammar,
    select_members,
    serialize_tool_calls_qwen,
)
from olmlx.engine.registry import PanelConfig
from olmlx.engine.tool_parser import parse_model_output


def _panel() -> PanelConfig:
    return PanelConfig.from_entry(
        "p:latest",
        {
            "type": "panel",
            "classifier": "c",
            "judge": "j",
            "routes": {
                "code": ["qwen3-coder", "devstral"],
                "default": ["qwen3", "mistral"],
            },
        },
    )


class TestRoutingHelpers:
    def test_first_user_text_string_content(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi"},
        ]
        assert first_user_text(msgs) == "hello world"

    def test_first_user_text_list_content(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
        assert first_user_text(msgs) == "part one\npart two"

    def test_first_user_text_no_user(self):
        assert first_user_text([{"role": "system", "content": "x"}]) == ""

    def test_route_grammar_enumerates_keys(self):
        spec = route_grammar(_panel())
        assert isinstance(spec, GrammarSpec)
        assert spec.kind == "json_schema"
        enum = spec.schema["properties"]["route"]["enum"]
        assert set(enum) == {"code", "default"}

    def test_select_members_known_route(self):
        assert select_members("code", _panel()) == ["qwen3-coder", "devstral"]

    def test_select_members_unknown_route_falls_back_to_default(self):
        assert select_members("nonsense", _panel()) == ["qwen3", "mistral"]


class TestToolCallUnion:
    def test_merge_dedupes_identical_calls(self):
        # Two panelists; one shared identical call, one unique each.
        per_panelist = [
            [
                {"name": "search", "input": {"q": "x"}, "_span": (0, 1)},
                {"name": "read", "input": {"path": "a"}},
            ],
            [
                {"name": "search", "input": {"q": "x"}},
                {"name": "read", "input": {"path": "b"}},
            ],
        ]
        merged = merge_tool_calls(per_panelist)
        # search{q:x} collapses to one; read{a} and read{b} both kept.
        assert merged == [
            {"name": "search", "input": {"q": "x"}},
            {"name": "read", "input": {"path": "a"}},
            {"name": "read", "input": {"path": "b"}},
        ]
        # _span must be stripped.
        assert all("_span" not in tc for tc in merged)

    def test_merge_empty(self):
        assert merge_tool_calls([[], []]) == []

    def test_serialize_round_trips_through_parser(self):
        merged = [
            {"name": "search", "input": {"q": "x"}},
            {"name": "read", "input": {"path": "a"}},
        ]
        text = serialize_tool_calls_qwen(merged)
        _thinking, _visible, tool_uses = parse_model_output(text, has_tools=True)
        reparsed = [{"name": tu["name"], "input": tu["input"]} for tu in tool_uses]
        assert reparsed == merged

    def test_serialize_round_trips_tool_call_tag_in_args(self):
        # An argument value containing the literal closing tag must survive
        # the serialize -> parse_model_output round-trip (the panel emits
        # this text and the routers re-parse it).
        # With two calls, the non-greedy Qwen regex closes early inside the
        # first arg, the first call is silently dropped, and only the second
        # is returned — a real production failure.
        merged = [
            {"name": "search", "input": {"q": "see </tool_call> here"}},
            {"name": "read", "input": {"path": "/tmp/foo"}},
        ]
        text = serialize_tool_calls_qwen(merged)
        _t, _v, tool_uses = parse_model_output(text, has_tools=True)
        reparsed = [{"name": tu["name"], "input": tu["input"]} for tu in tool_uses]
        assert reparsed == merged

    def test_merge_handles_none_input(self):
        merged = merge_tool_calls([[{"name": "f", "input": None}]])
        assert merged == [{"name": "f", "input": {}}]

    def test_merge_no_panelists(self):
        assert merge_tool_calls([]) == []


class TestJudgePrompt:
    def test_appends_candidates_as_final_user_turn(self):
        original = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        out = build_judge_messages(original, ["qwen3", "mistral"], ["four", "4"])
        # Original messages preserved as a prefix (not mutated).
        assert out[:2] == original
        assert original[-1]["content"] == "What is 2+2?"  # input untouched
        judge_turn = out[-1]
        assert judge_turn["role"] == "user"
        assert "qwen3" in judge_turn["content"]
        assert "mistral" in judge_turn["content"]
        assert "four" in judge_turn["content"]
        assert "4" in judge_turn["content"]

    def test_handles_empty_candidate_answer(self):
        out = build_judge_messages([{"role": "user", "content": "q"}], ["m1"], [""])
        assert "m1" in out[-1]["content"]


def _make_panel():
    from olmlx.engine.registry import PanelConfig

    return PanelConfig.from_entry(
        "p:latest",
        {
            "type": "panel",
            "classifier": "c",
            "judge": "j",
            "routes": {"code": ["qa", "qb"], "default": ["da", "db"]},
        },
    )


def _fake_generate_chat_factory(responses: dict):
    """Return an async generate_chat stub keyed by model name -> text."""

    async def _fake(
        manager,
        model_name,
        messages,
        options=None,
        tools=None,
        stream=False,
        keep_alive=None,
        max_tokens=512,
        cache_id="",
        enable_thinking=None,
        grammar_spec=None,
    ):
        text = responses[model_name]
        return {"text": text, "done": True, "stats": None}

    return _fake


class TestClassify:
    @pytest.mark.asyncio
    async def test_classify_returns_route_members(self, monkeypatch):
        monkeypatch.setattr(
            panel_mod,
            "generate_chat",
            _fake_generate_chat_factory({"c": '{"route": "code"}'}),
        )
        members = await panel_mod.classify(
            manager=None, panel=_make_panel(), user_text="write a function"
        )
        assert members == ["qa", "qb"]

    @pytest.mark.asyncio
    async def test_classify_bad_json_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(
            panel_mod,
            "generate_chat",
            _fake_generate_chat_factory({"c": "not json at all"}),
        )
        members = await panel_mod.classify(
            manager=None, panel=_make_panel(), user_text="hi"
        )
        assert members == ["da", "db"]


class TestRunPanel:
    @pytest.mark.asyncio
    async def test_returns_union_when_any_panelist_wants_tools(self, monkeypatch):
        # da proposes a tool call (Qwen format), db answers in prose.
        responses = {
            "c": '{"route": "default"}',
            "da": '<tool_call>\n{"name": "search", "arguments": {"q": "x"}}\n</tool_call>',
            "db": "I think the answer is 42.",
        }
        monkeypatch.setattr(
            panel_mod, "generate_chat", _fake_generate_chat_factory(responses)
        )
        answers, merged = await panel_mod._run_panel(
            manager=None,
            panel=_make_panel(),
            messages=[{"role": "user", "content": "find x"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            options=None,
            keep_alive=None,
            max_tokens=128,
            enable_thinking=None,
        )
        assert merged == [{"name": "search", "input": {"q": "x"}}]

    @pytest.mark.asyncio
    async def test_returns_answers_when_no_tools_requested(self, monkeypatch):
        responses = {
            "c": '{"route": "default"}',
            "da": "Answer A",
            "db": "Answer B",
        }
        monkeypatch.setattr(
            panel_mod, "generate_chat", _fake_generate_chat_factory(responses)
        )
        answers, merged = await panel_mod._run_panel(
            manager=None,
            panel=_make_panel(),
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            options=None,
            keep_alive=None,
            max_tokens=128,
            enable_thinking=None,
        )
        assert merged == []
        assert answers == (["da", "db"], ["Answer A", "Answer B"])


class TestPanelGenerateChatNonStream:
    @pytest.mark.asyncio
    async def test_tool_turn_returns_qwen_blocks(self, monkeypatch):
        responses = {
            "c": '{"route": "default"}',
            "da": '<tool_call>\n{"name": "search", "arguments": {"q": "x"}}\n</tool_call>',
            "db": "prose",
        }
        monkeypatch.setattr(
            panel_mod, "generate_chat", _fake_generate_chat_factory(responses)
        )
        monkeypatch.setattr(
            panel_mod, "_resolve_panel", lambda manager, name: _make_panel()
        )
        result = await panel_mod.panel_generate_chat(
            manager=None,
            model_name="p:latest",
            messages=[{"role": "user", "content": "find x"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            stream=False,
        )
        # Router parses raw_text -> tool calls.
        _t, _v, tool_uses = parse_model_output(result["raw_text"], has_tools=True)
        assert [tu["name"] for tu in tool_uses] == ["search"]
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_final_turn_returns_judge_answer(self, monkeypatch):
        responses = {
            "c": '{"route": "default"}',
            "da": "Answer A",
            "db": "Answer B",
            "j": "Reconciled final answer.",
        }
        monkeypatch.setattr(
            panel_mod, "generate_chat", _fake_generate_chat_factory(responses)
        )
        monkeypatch.setattr(
            panel_mod, "_resolve_panel", lambda manager, name: _make_panel()
        )
        result = await panel_mod.panel_generate_chat(
            manager=None,
            model_name="p:latest",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            stream=False,
        )
        assert result["text"] == "Reconciled final answer."


def _fake_generate_chat_streaming_factory(responses: dict, stream_models: set):
    """generate_chat stub: dict for non-stream models, async-gen for streamed ones."""

    async def _fake(
        manager,
        model_name,
        messages,
        options=None,
        tools=None,
        stream=False,
        keep_alive=None,
        max_tokens=512,
        cache_id="",
        enable_thinking=None,
        grammar_spec=None,
    ):
        text = responses[model_name]
        if stream and model_name in stream_models:

            async def _gen():
                yield {"text": text}
                yield {"done": True, "done_reason": "stop"}

            return _gen()
        return {"text": text, "done": True, "stats": None}

    return _fake


async def _drain(agen) -> list[dict]:
    return [chunk async for chunk in agen]


class TestPanelGenerateChatStream:
    @pytest.mark.asyncio
    async def test_stream_tool_turn_emits_qwen_text_then_done(self, monkeypatch):
        responses = {
            "c": '{"route": "default"}',
            "da": '<tool_call>\n{"name": "search", "arguments": {"q": "x"}}\n</tool_call>',
            "db": "prose",
        }
        monkeypatch.setattr(
            panel_mod,
            "generate_chat",
            _fake_generate_chat_streaming_factory(responses, stream_models=set()),
        )
        monkeypatch.setattr(panel_mod, "_resolve_panel", lambda m, n: _make_panel())
        agen = await panel_mod.panel_generate_chat(
            manager=None,
            model_name="p:latest",
            messages=[{"role": "user", "content": "find x"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            stream=True,
        )
        chunks = await _drain(agen)
        full = "".join(c.get("text", "") for c in chunks)
        _t, _v, tool_uses = parse_model_output(full, has_tools=True)
        assert [tu["name"] for tu in tool_uses] == ["search"]
        assert chunks[-1].get("done") is True

    @pytest.mark.asyncio
    async def test_stream_final_turn_proxies_judge_stream(self, monkeypatch):
        responses = {
            "c": '{"route": "default"}',
            "da": "A",
            "db": "B",
            "j": "Final synthesized answer.",
        }
        monkeypatch.setattr(
            panel_mod,
            "generate_chat",
            _fake_generate_chat_streaming_factory(responses, stream_models={"j"}),
        )
        monkeypatch.setattr(panel_mod, "_resolve_panel", lambda m, n: _make_panel())
        agen = await panel_mod.panel_generate_chat(
            manager=None,
            model_name="p:latest",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            stream=True,
        )
        chunks = await _drain(agen)
        full = "".join(c.get("text", "") for c in chunks)
        assert full == "Final synthesized answer."
        assert chunks[-1].get("done") is True
