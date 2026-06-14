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
    def test_flattens_request_and_candidates(self):
        original = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        out = build_judge_messages(original, ["qwen3", "mistral"], ["four", "4"])
        # Flattened to a short fixed prompt (system + user), not a conversation
        # replay; original is not mutated.
        assert all(m["role"] in ("system", "user") for m in out)
        assert original[-1]["content"] == "What is 2+2?"
        blob = " ".join(m["content"] for m in out)
        assert "What is 2+2?" in blob  # request
        assert "qwen3" in blob and "mistral" in blob  # candidate labels
        assert "four" in blob and "4" in blob  # candidate answers

    def test_does_not_replay_agentic_turns(self):
        # The whole point of the fix: a tool-heavy conversation must NOT be
        # replayed as assistant(tool_calls)/tool turns, which primes the judge
        # to keep calling tools. It must be flattened into a clean prompt that
        # still carries the tool results.
        original = [
            {"role": "user", "content": "read the config"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "config.py"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "PREFIX=OLMLX_"},
        ]
        out = build_judge_messages(original, ["m1"], ["the prefix is OLMLX_"])
        # No assistant tool_calls and no tool-role messages leak through.
        assert not any("tool_calls" in m for m in out)
        assert not any(m["role"] == "tool" for m in out)
        blob = " ".join(m["content"] for m in out)
        assert "read the config" in blob  # request preserved
        assert "PREFIX=OLMLX_" in blob  # tool RESULT preserved
        assert "read_file" in blob  # which call produced it
        assert "the prefix is OLMLX_" in blob  # candidate

    def test_handles_empty_candidate_answer(self):
        out = build_judge_messages([{"role": "user", "content": "q"}], ["m1"], [""])
        blob = " ".join(m["content"] for m in out)
        assert "q" in blob and "m1" in blob


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
        reasoning_effort=None,
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

    @pytest.mark.asyncio
    async def test_classify_disables_thinking(self, monkeypatch):
        # Routing is a constrained JSON task: the classifier must never be
        # asked to think, regardless of model/request defaults.
        captured = {}

        async def _capture(manager, model_name, messages, **kwargs):
            captured.update(kwargs)
            return {"text": '{"route": "code"}', "done": True, "stats": None}

        monkeypatch.setattr(panel_mod, "generate_chat", _capture)
        await panel_mod.classify(
            manager=None, panel=_make_panel(), user_text="write a function"
        )
        assert captured["enable_thinking"] is False


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


_TOOL = '<tool_call>\n{"name": "search", "arguments": {"q": "x"}}\n</tool_call>'


def _panel_with_stop(cond):
    from olmlx.engine.registry import PanelConfig

    return PanelConfig.from_entry(
        "p:latest",
        {
            "type": "panel",
            "classifier": "c",
            "judge": "j",
            "routes": {"default": ["da", "db", "dc"]},
            "stop_condition": cond,
        },
    )


class TestStopConditions:
    async def _run(self, monkeypatch, cond, responses):
        monkeypatch.setattr(
            panel_mod, "generate_chat", _fake_generate_chat_factory(responses)
        )
        (_members, _answers), merged = await panel_mod._run_panel(
            manager=None,
            panel=_panel_with_stop(cond),
            messages=[{"role": "user", "content": "q"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            options=None,
            keep_alive=None,
            max_tokens=128,
            enable_thinking=None,
        )
        return merged

    @pytest.mark.asyncio
    async def test_all_emits_on_any_tool(self, monkeypatch):
        # 1 of 3 wants tools -> "all" still emits the union.
        responses = {"c": '{"route": "default"}', "da": _TOOL, "db": "x", "dc": "y"}
        merged = await self._run(monkeypatch, "all", responses)
        assert [t["name"] for t in merged] == ["search"]

    @pytest.mark.asyncio
    async def test_majority_finalizes_when_majority_ready(self, monkeypatch):
        # 1 of 3 wants tools, 2 ready -> majority finalizes (no tools emitted).
        responses = {"c": '{"route": "default"}', "da": _TOOL, "db": "x", "dc": "y"}
        merged = await self._run(monkeypatch, "majority", responses)
        assert merged == []

    @pytest.mark.asyncio
    async def test_majority_gathers_when_majority_want_tools(self, monkeypatch):
        # 2 of 3 want tools, 1 ready -> majority gathers (emit union).
        responses = {"c": '{"route": "default"}', "da": _TOOL, "db": _TOOL, "dc": "y"}
        merged = await self._run(monkeypatch, "majority", responses)
        assert [t["name"] for t in merged] == ["search"]

    @pytest.mark.asyncio
    async def test_judge_gather_emits_tools(self, monkeypatch):
        responses = {
            "c": '{"route": "default"}',
            "da": _TOOL,
            "db": "x",
            "dc": "y",
            "j": '{"action": "gather"}',
        }
        merged = await self._run(monkeypatch, "judge", responses)
        assert [t["name"] for t in merged] == ["search"]

    @pytest.mark.asyncio
    async def test_judge_answer_finalizes(self, monkeypatch):
        responses = {
            "c": '{"route": "default"}',
            "da": _TOOL,
            "db": "x",
            "dc": "y",
            "j": '{"action": "answer"}',
        }
        merged = await self._run(monkeypatch, "judge", responses)
        assert merged == []

    @pytest.mark.asyncio
    async def test_judge_bad_json_finalizes(self, monkeypatch):
        # Unparseable judge decision defaults to finalize (curb runaway loops).
        responses = {
            "c": '{"route": "default"}',
            "da": _TOOL,
            "db": "x",
            "dc": "y",
            "j": "not json",
        }
        merged = await self._run(monkeypatch, "judge", responses)
        assert merged == []


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
    async def test_judge_synthesis_uses_low_reasoning_effort(self, monkeypatch):
        # The judge call must request reasoning_effort="low" so channel-format
        # reasoners (gpt-oss) stay terse instead of leaking analysis.
        captured = {}

        async def _capture(manager, model_name, messages, **kwargs):
            if model_name == "j":  # judge synthesis
                captured.update(kwargs)
            return {"text": "final", "done": True, "stats": None}

        monkeypatch.setattr(panel_mod, "generate_chat", _capture)
        monkeypatch.setattr(panel_mod, "_resolve_panel", lambda m, n: _make_panel())
        await panel_mod.panel_generate_chat(
            manager=None,
            model_name="p:latest",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            stream=False,
        )
        assert captured.get("reasoning_effort") == "low"

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
        reasoning_effort=None,
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
        # Done chunk carries stats, matching the non-streaming tool turn so the
        # Ollama timing-metadata path stays symmetric.
        assert chunks[-1].get("stats") is not None

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


class TestRouterDispatch:
    def test_openai_router_imports_panel_dispatch(self):
        from olmlx.routers import openai as openai_router

        assert hasattr(openai_router, "panel_generate_chat")

    def test_ollama_router_imports_panel_dispatch(self):
        from olmlx.routers import chat as chat_router

        assert hasattr(chat_router, "panel_generate_chat")


# ---------------------------------------------------------------------------
# Gap 1 — Continuation-history re-routing stays stable
# ---------------------------------------------------------------------------


class TestContinuationRouting:
    def test_first_user_text_stable_with_tool_history(self):
        # A continuation history: system first, then the original user turn,
        # then an assistant tool-call turn and a tool result.  first_user_text
        # must still return the ORIGINAL user message (routing key is stable).
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "original task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "tool result", "tool_call_id": "1"},
        ]
        assert first_user_text(messages) == "original task"

    @pytest.mark.asyncio
    async def test_run_panel_routes_same_on_continuation(self, monkeypatch):
        # Same classifier output -> same members, regardless of added history.
        responses = {"c": '{"route": "code"}', "qa": "A", "qb": "B"}
        monkeypatch.setattr(
            panel_mod, "generate_chat", _fake_generate_chat_factory(responses)
        )
        base = [{"role": "user", "content": "write code"}]
        continuation = base + [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "result", "tool_call_id": "1"},
        ]
        (members_a, _), _ = await panel_mod._run_panel(
            None, _make_panel(), base, None, None, None, 128, None
        )
        (members_b, _), _ = await panel_mod._run_panel(
            None, _make_panel(), continuation, None, None, None, 128, None
        )
        assert members_a == members_b == ["qa", "qb"]


# ---------------------------------------------------------------------------
# Gap 2 — Failure propagation
# ---------------------------------------------------------------------------


class TestFailurePropagation:
    @staticmethod
    def _raising_factory(responses: dict, raise_on: str):
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
            reasoning_effort=None,
            grammar_spec=None,
        ):
            if model_name == raise_on:
                raise RuntimeError(f"boom in {model_name}")
            return {"text": responses[model_name], "done": True, "stats": None}

        return _fake

    @pytest.mark.asyncio
    async def test_panelist_failure_propagates_nonstream(self, monkeypatch):
        responses = {"c": '{"route": "default"}', "da": "A", "db": "B"}
        monkeypatch.setattr(
            panel_mod,
            "generate_chat",
            self._raising_factory(responses, raise_on="da"),
        )
        monkeypatch.setattr(panel_mod, "_resolve_panel", lambda m, n: _make_panel())
        with pytest.raises(RuntimeError, match="boom in da"):
            await panel_mod.panel_generate_chat(
                manager=None,
                model_name="p:latest",
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                stream=False,
            )

    @pytest.mark.asyncio
    async def test_classifier_failure_propagates_nonstream(self, monkeypatch):
        responses = {"c": "unused"}
        monkeypatch.setattr(
            panel_mod,
            "generate_chat",
            self._raising_factory(responses, raise_on="c"),
        )
        monkeypatch.setattr(panel_mod, "_resolve_panel", lambda m, n: _make_panel())
        with pytest.raises(RuntimeError, match="boom in c"):
            await panel_mod.panel_generate_chat(
                manager=None,
                model_name="p:latest",
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                stream=False,
            )

    @pytest.mark.asyncio
    async def test_panelist_failure_propagates_stream(self, monkeypatch):
        responses = {"c": '{"route": "default"}', "da": "A", "db": "B"}
        monkeypatch.setattr(
            panel_mod,
            "generate_chat",
            self._raising_factory(responses, raise_on="da"),
        )
        monkeypatch.setattr(panel_mod, "_resolve_panel", lambda m, n: _make_panel())
        agen = await panel_mod.panel_generate_chat(
            manager=None,
            model_name="p:latest",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            stream=True,
        )
        with pytest.raises(RuntimeError, match="boom in da"):
            await _drain(agen)


# ---------------------------------------------------------------------------
# Gap 4 — Router actually dispatches to the coordinator (behavioral)
# ---------------------------------------------------------------------------


class TestRouterDispatchBehavioral:
    @pytest.mark.asyncio
    async def test_panel_model_dispatches_to_panel_generate_chat(self, monkeypatch):
        """Non-streaming request for a panel model calls panel_generate_chat,
        NOT generate_chat."""
        import json as _json

        from httpx import ASGITransport, AsyncClient
        from olmlx.app import create_app
        from olmlx.engine.registry import ModelRegistry
        from olmlx.utils.timing import TimingStats

        # Build a registry that exposes a panel model.
        import tempfile
        import pathlib

        config = {
            "classifier-m": "org/small",
            "judge-m": "org/judge",
            "panelist-m": "org/panelist",
            "my-panel": {
                "type": "panel",
                "classifier": "classifier-m",
                "judge": "judge-m",
                "routes": {"default": ["panelist-m"]},
            },
        }
        with tempfile.TemporaryDirectory() as td:
            cfg_path = pathlib.Path(td) / "models.json"
            cfg_path.write_text(_json.dumps(config))
            monkeypatch.setattr(
                "olmlx.engine.registry.settings.models_config", cfg_path
            )
            reg = ModelRegistry()
            reg.load()

        assert reg.is_panel("my-panel") is True

        panel_called: list[str] = []
        generate_called: list[str] = []

        async def fake_panel_generate_chat(
            manager, model_name, messages, options=None, **kwargs
        ):
            panel_called.append(model_name)
            return {
                "text": "panel answer",
                "done": True,
                "stats": TimingStats(),
            }

        async def fake_generate_chat(
            manager, model_name, messages, options=None, **kwargs
        ):
            generate_called.append(model_name)
            return {
                "text": "plain answer",
                "done": True,
                "stats": TimingStats(),
            }

        monkeypatch.setattr(
            "olmlx.routers.openai.panel_generate_chat", fake_panel_generate_chat
        )
        monkeypatch.setattr("olmlx.routers.openai.generate_chat", fake_generate_chat)

        app = create_app()
        app.state.registry = reg
        # model_manager can be a minimal stub — dispatch happens before any
        # actual inference.
        from unittest.mock import MagicMock

        app.state.model_manager = MagicMock()
        app.state.model_store = MagicMock()

        transport = ASGITransport(app=app, raise_app_exceptions=True)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "my-panel",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        assert panel_called == ["my-panel"], (
            f"panel_generate_chat was not called; panel_called={panel_called}"
        )
        assert generate_called == [], (
            f"generate_chat was called unexpectedly: {generate_called}"
        )


def _registry_with_panel(monkeypatch):
    import json as _json
    import pathlib
    import tempfile

    from olmlx.engine.registry import ModelRegistry

    config = {
        "small": "org/small",
        "judge-m": "org/judge",
        "panelist-m": "org/panelist",
        "list-panel": {
            "type": "panel",
            "classifier": "small",
            "judge": "judge-m",
            "routes": {"default": ["panelist-m"]},
        },
    }
    with tempfile.TemporaryDirectory() as td:
        cfg = pathlib.Path(td) / "models.json"
        cfg.write_text(_json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", cfg)
        reg = ModelRegistry()
        reg.load()
    return reg


class TestPanelModelListing:
    @pytest.mark.asyncio
    async def test_openai_v1_models_includes_panel(self, monkeypatch):
        from unittest.mock import MagicMock

        from httpx import ASGITransport, AsyncClient
        from olmlx.app import create_app

        app = create_app()
        app.state.registry = _registry_with_panel(monkeypatch)
        app.state.model_manager = MagicMock()
        app.state.model_store = MagicMock()

        transport = ASGITransport(app=app, raise_app_exceptions=True)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/models")
        assert resp.status_code == 200
        ids = {m["id"] for m in resp.json()["data"]}
        assert "list-panel:latest" in ids

    @pytest.mark.asyncio
    async def test_ollama_api_tags_includes_panel(self, monkeypatch):
        from unittest.mock import MagicMock

        from httpx import ASGITransport, AsyncClient
        from olmlx.app import create_app

        app = create_app()
        app.state.registry = _registry_with_panel(monkeypatch)
        app.state.model_manager = MagicMock()
        store = MagicMock()
        store.list_local.return_value = []
        app.state.model_store = store

        transport = ASGITransport(app=app, raise_app_exceptions=True)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/tags")
        assert resp.status_code == 200
        entries = {m["name"]: m for m in resp.json()["models"]}
        assert "list-panel:latest" in entries
        assert entries["list-panel:latest"]["details"]["family"] == "panel"
