"""Regression coverage for olmlx.chat.session agent-loop branches.

Focuses on currently-uncovered paths with generate_chat mocked:
- model load failure rollback
- generation option passthrough (temperature/top_p/top_k)
- thinking_expected meta-chunk capture
- parse_model_output failure handling
- memory check + history truncation
- sequential tool execution
- the question builtin-tool feedback path
- deferred non-Exception BaseException re-raise from parallel gather

All tests are hermetic: no network, no real model, no GPU, no real FS.
asyncio_mode=auto so async tests need no decorator.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from olmlx.chat.config import ChatConfig
from olmlx.chat.session import ChatSession
from olmlx.engine.template_caps import TemplateCaps


def _make_session(
    *,
    mcp=None,
    skills=None,
    builtin=None,
    tool_safety=None,
    model_name="test:latest",
    thinking=True,
    max_turns=25,
    max_tokens=4096,
    system_prompt=None,
    template_has_thinking=False,
    sequential=False,
    temperature=None,
    top_p=None,
    top_k=None,
):
    config = ChatConfig(
        model_name=model_name,
        thinking=thinking,
        max_turns=max_turns,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        sequential_tool_execution=sequential,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    manager = MagicMock()
    loaded_model = MagicMock()
    loaded_model.template_caps = TemplateCaps(
        supports_enable_thinking=template_has_thinking,
        has_thinking_tags=template_has_thinking,
    )
    loaded_model.kv_cache_quant = None
    manager.ensure_loaded = AsyncMock(return_value=loaded_model)
    session = ChatSession(
        config=config,
        manager=manager,
        mcp=mcp,
        skills=skills,
        builtin=builtin,
        tool_safety=tool_safety,
    )
    return session, manager, loaded_model


def _stream(*chunks):
    """Build a zero-arg async generator factory yielding the given chunks."""

    async def gen(*args, **kwargs):
        for c in chunks:
            yield c

    return gen


def _patch_no_memory_check():
    """Disable the memory check so it never truncates (returns 0 system bytes)."""
    return patch(
        "olmlx.chat.session.memory_utils.get_system_memory_bytes", return_value=0
    )


# --------------------------------------------------------------------------
# Model-load failure path (lines 950-957)
# --------------------------------------------------------------------------


async def test_model_load_failure_rolls_back_user_message():
    session, manager, _ = _make_session()
    manager.ensure_loaded = AsyncMock(side_effect=RuntimeError("disk full"))

    events = []
    async for ev in session.send_message("Hello"):
        events.append(ev)

    types = [e["type"] for e in events]
    assert "model_load_error" in types
    assert types[-1] == "done"
    err = next(e for e in events if e["type"] == "model_load_error")
    assert "disk full" in err["error"]
    # The user message appended at the top must have been popped on failure.
    assert session.messages == []


# --------------------------------------------------------------------------
# Generation option passthrough (lines 940-944)
# --------------------------------------------------------------------------


async def test_sampling_options_passed_through():
    session, _, _ = _make_session(temperature=0.7, top_p=0.9, top_k=40)
    captured = {}

    async def gen(*args, **kwargs):
        captured.update(kwargs)
        yield {"text": "hi", "done": False}
        yield {"text": "", "done": True}

    with (
        _patch_no_memory_check(),
        patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: gen(*a, **kw),
        ),
    ):
        async for _ in session.send_message("Q"):
            pass

    opts = captured["options"]
    assert opts["temperature"] == 0.7
    assert opts["top_p"] == 0.9
    assert opts["top_k"] == 40


async def test_sampling_options_omitted_when_none():
    session, _, _ = _make_session()
    captured = {}

    async def gen(*args, **kwargs):
        captured.update(kwargs)
        yield {"text": "hi", "done": False}
        yield {"text": "", "done": True}

    with (
        _patch_no_memory_check(),
        patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: gen(*a, **kw),
        ),
    ):
        async for _ in session.send_message("Q"):
            pass

    opts = captured["options"]
    assert "temperature" not in opts
    assert "top_p" not in opts
    assert "top_k" not in opts


# --------------------------------------------------------------------------
# thinking_expected meta chunk (lines 1007-1009)
# --------------------------------------------------------------------------


async def test_thinking_expected_meta_chunk_consumed_not_emitted():
    """A {'thinking_expected': ...} chunk is consumed and not surfaced as text."""
    session, _, _ = _make_session()

    stream = _stream(
        {"thinking_expected": True},
        {"text": "Answer", "done": False},
        {"text": "", "done": True},
    )

    with (
        _patch_no_memory_check(),
        patch("olmlx.chat.session.generate_chat", return_value=stream()),
    ):
        events = []
        async for ev in session.send_message("Q"):
            events.append(ev)

    # The meta chunk must not have produced a token; only "Answer" is visible.
    token_text = "".join(e["text"] for e in events if e["type"] == "token")
    assert token_text == "Answer"
    assert all("thinking_expected" not in e for e in events)


# --------------------------------------------------------------------------
# parse_model_output failure (lines 1061-1075)
# --------------------------------------------------------------------------


async def test_parse_error_drops_tools_and_emits_tool_error():
    mcp = MagicMock()
    mcp.get_tools_for_chat.return_value = [
        {"function": {"name": "x"}, "type": "function"}
    ]
    session, _, _ = _make_session(mcp=mcp)

    stream = _stream(
        {"text": "<think>t</think>visible body", "done": False},
        {"text": "", "done": True},
    )

    with (
        _patch_no_memory_check(),
        patch("olmlx.chat.session.generate_chat", return_value=stream()),
        patch(
            "olmlx.chat.session.parse_model_output",
            side_effect=ValueError("bad tokens"),
        ),
    ):
        events = []
        async for ev in session.send_message("Q"):
            events.append(ev)

    parse_errs = [
        e for e in events if e["type"] == "tool_error" and e["name"] == "(parse error)"
    ]
    assert len(parse_errs) == 1
    assert "bad tokens" in parse_errs[0]["error"]
    # Fallback visible text strips thinking and becomes the assistant message.
    assert session.messages[-1]["role"] == "assistant"
    assert session.messages[-1]["content"] == "visible body"
    # No tool calls remained after the parse failure -> loop ends, done emitted.
    assert events[-1]["type"] == "done"


# --------------------------------------------------------------------------
# Memory check + truncation (lines 388-496, 967-976)
# --------------------------------------------------------------------------


async def test_memory_truncation_emits_event_and_drops_messages():
    """When KV estimate exceeds the limit, history is truncated and reported."""
    session, manager, lm = _make_session(model_name="m:1")

    # Seed a long history: system + many user/assistant pairs.
    session.messages.append({"role": "system", "content": "sys"})
    for i in range(12):
        session.messages.append({"role": "user", "content": f"u{i}"})
        session.messages.append({"role": "assistant", "content": f"a{i}"})
    before = len(session.messages)

    stream = _stream(
        {"text": "ok", "done": False},
        {"text": "", "done": True},
    )

    with (
        patch(
            "olmlx.chat.session.memory_utils.get_system_memory_bytes",
            return_value=100,
        ),
        patch(
            "olmlx.chat.session.memory_utils.get_metal_memory",
            return_value=0,
        ),
        # settings.memory_limit_fraction * 100 -> some positive limit.
        patch(
            "olmlx.chat.session.apply_chat_template_text",
            return_value="prompt",
        ),
        patch(
            "olmlx.chat.session.tokenize_for_cache",
            return_value=list(range(50)),
        ),
        # First (pre-truncate) estimate huge, post-truncate small enough.
        patch(
            "olmlx.chat.session.estimate_kv_cache_bytes",
            side_effect=[10**12, 1, 1],
        ),
        patch("olmlx.chat.session.generate_chat", return_value=stream()),
    ):
        events = []
        async for ev in session.send_message("new question"):
            events.append(ev)

    trunc = [e for e in events if e["type"] == "memory_truncated"]
    assert len(trunc) == 1
    # The new user message + assistant reply were appended during the turn,
    # but the conversation must be shorter than before + 2 (history removed).
    assert len(session.messages) < before + 2
    manager.invalidate_prompt_cache.assert_called_with("m:1", "chat")


async def test_memory_check_noop_when_system_memory_unknown():
    """get_system_memory_bytes()<=0 short-circuits; no truncation, no estimate."""
    session, _, lm = _make_session()
    session.messages.append({"role": "user", "content": "old"})

    estimate = MagicMock()
    stream = _stream({"text": "hi", "done": False}, {"text": "", "done": True})

    with (
        patch(
            "olmlx.chat.session.memory_utils.get_system_memory_bytes",
            return_value=0,
        ),
        patch("olmlx.chat.session.estimate_kv_cache_bytes", estimate),
        patch("olmlx.chat.session.generate_chat", return_value=stream()),
    ):
        events = []
        async for ev in session.send_message("Q"):
            events.append(ev)

    assert not any(e["type"] == "memory_truncated" for e in events)
    estimate.assert_not_called()


async def test_memory_check_under_limit_does_not_truncate():
    session, _, lm = _make_session()
    session.messages.append({"role": "user", "content": "old"})
    stream = _stream({"text": "hi", "done": False}, {"text": "", "done": True})

    with (
        patch(
            "olmlx.chat.session.memory_utils.get_system_memory_bytes",
            return_value=10**9,
        ),
        patch(
            "olmlx.chat.session.memory_utils.get_metal_memory",
            return_value=0,
        ),
        patch(
            "olmlx.chat.session.apply_chat_template_text",
            return_value="prompt",
        ),
        patch(
            "olmlx.chat.session.tokenize_for_cache",
            return_value=[1, 2, 3],
        ),
        patch(
            "olmlx.chat.session.estimate_kv_cache_bytes",
            return_value=1,
        ),
        patch("olmlx.chat.session.generate_chat", return_value=stream()),
    ):
        events = []
        async for ev in session.send_message("Q"):
            events.append(ev)

    assert not any(e["type"] == "memory_truncated" for e in events)


async def test_memory_check_exception_is_swallowed():
    """If the memory check raises, send_message proceeds anyway."""
    session, _, _ = _make_session()
    stream = _stream({"text": "hi", "done": False}, {"text": "", "done": True})

    with (
        patch.object(
            ChatSession,
            "_check_memory_and_truncate",
            side_effect=RuntimeError("boom"),
        ),
        patch("olmlx.chat.session.generate_chat", return_value=stream()),
    ):
        events = []
        async for ev in session.send_message("Q"):
            events.append(ev)

    # Generation still happened despite the failed memory check.
    assert "".join(e["text"] for e in events if e["type"] == "token") == "hi"
    assert events[-1]["type"] == "done"


async def test_emergency_second_pass_truncation():
    """First-pass truncation still over limit -> emergency 2-message keep."""
    session, manager, lm = _make_session(model_name="m:2", max_tokens=10)
    session.messages.append({"role": "system", "content": "sys"})
    for i in range(10):
        session.messages.append({"role": "user", "content": f"u{i}"})
        session.messages.append({"role": "assistant", "content": f"a{i}"})
    original_len = len(session.messages)

    with (
        patch(
            "olmlx.chat.session.memory_utils.get_system_memory_bytes",
            return_value=100,
        ),
        patch(
            "olmlx.chat.session.memory_utils.get_metal_memory",
            return_value=0,
        ),
        patch(
            "olmlx.chat.session.apply_chat_template_text",
            return_value="prompt",
        ),
        patch(
            "olmlx.chat.session.tokenize_for_cache",
            return_value=list(range(50)),
        ),
        # initial, after-first-pass, after-emergency-pass — all over limit.
        patch(
            "olmlx.chat.session.estimate_kv_cache_bytes",
            return_value=10**12,
        ),
    ):
        truncated = await session._check_memory_and_truncate(lm, 10)

    assert truncated is True
    assert len(session.messages) < original_len
    # Emergency pass keeps only a small tail (system + ~last turn), well below
    # the first-pass target of _MIN_MESSAGES_AFTER_TRUNCATE (6) + system.
    assert len(session.messages) <= 6
    # System message preserved at the front.
    assert session.messages[0]["role"] == "system"
    manager.invalidate_prompt_cache.assert_called_with("m:2", "chat")


async def test_sequential_execution_generic_exception_yields_error():
    """A regular Exception in the sequential path is captured per-tool."""
    mcp = MagicMock()
    mcp.get_tools_for_chat.return_value = [
        {"function": {"name": "t1"}, "type": "function"}
    ]
    session, _, _ = _make_session(mcp=mcp, sequential=True)

    async def fake_exec(tu):
        raise RuntimeError("kaboom")

    tus = [{"name": "t1", "input": {}, "id": "1"}]
    with patch.object(session, "_exec_tool", side_effect=fake_exec):
        events = []
        async for ev in session._execute_tool_calls(tus):
            events.append(ev)

    err = [e for e in events if e["type"] == "tool_error"]
    assert len(err) == 1
    assert "kaboom" in err[0]["error"]
    assert "Error calling t1" in session.messages[-1]["content"]


async def test_estimate_conversation_tokens_returns_zero_on_error():
    """_estimate_conversation_tokens swallows tokenizer errors and returns 0."""
    session, _, lm = _make_session()
    session.messages.append({"role": "user", "content": "hi"})

    with patch(
        "olmlx.chat.session.apply_chat_template_text",
        side_effect=RuntimeError("tokenizer exploded"),
    ):
        result = await session._estimate_conversation_tokens(lm)

    assert result == 0


# --------------------------------------------------------------------------
# Sequential tool execution (lines 750-773)
# --------------------------------------------------------------------------


async def test_sequential_tool_execution_runs_in_order():
    mcp = MagicMock()
    mcp.get_tools_for_chat.return_value = [
        {"function": {"name": "tool_a"}, "type": "function"},
        {"function": {"name": "tool_b"}, "type": "function"},
    ]
    order = []

    async def call_tool(name, args, timeout=30.0):
        order.append(name)
        return f"result-{name}"

    mcp.call_tool = AsyncMock(side_effect=call_tool)
    session, _, _ = _make_session(mcp=mcp, sequential=True)

    tus = [
        {"name": "tool_a", "input": {}, "id": "a"},
        {"name": "tool_b", "input": {}, "id": "b"},
    ]
    events = []
    async for ev in session._execute_tool_calls(tus):
        events.append(ev)

    assert order == ["tool_a", "tool_b"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert {r["name"] for r in results} == {"tool_a", "tool_b"}
    # Messages appended in original call order.
    assert [m["tool_call_id"] for m in session.messages] == ["a", "b"]


async def test_sequential_execution_cancelled_pads_and_raises():
    """CancelledError mid-sequence pads remaining slots and re-raises."""
    mcp = MagicMock()
    mcp.get_tools_for_chat.return_value = [
        {"function": {"name": "t1"}, "type": "function"},
        {"function": {"name": "t2"}, "type": "function"},
    ]

    async def call_tool(name, args, timeout=30.0):
        if name == "t1":
            raise asyncio.CancelledError("stop")
        return "ok"

    mcp.call_tool = AsyncMock(side_effect=call_tool)
    session, _, _ = _make_session(mcp=mcp, sequential=True)

    tus = [
        {"name": "t1", "input": {}, "id": "1"},
        {"name": "t2", "input": {}, "id": "2"},
    ]

    events = []
    with pytest.raises(asyncio.CancelledError):
        async for ev in session._execute_tool_calls(tus):
            events.append(ev)

    # t1 yields a tool_error (task cancelled); history records it.
    err = [e for e in events if e["type"] == "tool_error"]
    assert any(e["name"] == "t1" for e in err)
    tool_msgs = [m for m in session.messages if m["role"] == "tool"]
    assert any("cancelled" in m["content"] for m in tool_msgs)


# --------------------------------------------------------------------------
# Deferred non-Exception BaseException in parallel gather (lines 895-906)
# --------------------------------------------------------------------------


async def test_parallel_base_exception_reraised():
    """A non-Exception BaseException from _exec_tool is re-raised after events."""
    mcp = MagicMock()
    mcp.get_tools_for_chat.return_value = [
        {"function": {"name": "boom"}, "type": "function"}
    ]
    session, _, _ = _make_session(mcp=mcp, sequential=False)

    class Boom(BaseException):
        pass

    async def fake_exec(tu):
        raise Boom("fatal")

    tus = [{"name": "boom", "input": {}, "id": "z"}]
    with patch.object(session, "_exec_tool", side_effect=fake_exec):
        events = []
        with pytest.raises(Boom):
            async for ev in session._execute_tool_calls(tus):
                events.append(ev)

    # The tool_error event was still surfaced before the re-raise.
    assert any(e["type"] == "tool_error" and e["name"] == "boom" for e in events)


# --------------------------------------------------------------------------
# question builtin-tool feedback path (lines 855-874)
# --------------------------------------------------------------------------


async def test_question_tool_payload_yields_question_event():
    """The 'question' builtin's __question__: payload surfaces a question event.

    Regression test: the source must slice content[len("__question__:"):] so
    the leading ':' is stripped before json.loads. An off-by-one slice that
    kept the ':' silently swallowed the event (json.loads raised), so the
    question never reached the UI.
    """
    import json as _json

    builtin = MagicMock()
    builtin.tool_names = {"question"}
    builtin.get_tool_definitions.return_value = [
        {"function": {"name": "question"}, "type": "function"}
    ]
    payload = {
        "header": "Pick one",
        "question": "Which?",
        "options": ["a", "b"],
        "multiple": False,
    }
    builtin.call_tool = AsyncMock(return_value="__question__:" + _json.dumps(payload))

    session, _, _ = _make_session(builtin=builtin)
    tus = [{"name": "question", "input": {}, "id": "q1"}]

    events = []
    async for ev in session._execute_tool_calls(tus):
        events.append(ev)

    q_events = [e for e in events if e["type"] == "question"]
    assert len(q_events) == 1
    assert q_events[0]["header"] == "Pick one"
    assert q_events[0]["question"] == "Which?"
    assert q_events[0]["options"] == ["a", "b"]
    assert q_events[0]["multiple"] is False
    assert q_events[0]["id"] == "q1"


async def test_question_tool_message_also_appended_to_history():
    """The raw tool-result message is recorded in history alongside the event.

    Emitting the question event does not consume the result: after the
    try/except the original __question__: message is still appended so the
    conversation transcript stays complete.
    """
    import json as _json

    builtin = MagicMock()
    builtin.tool_names = {"question"}
    builtin.get_tool_definitions.return_value = [
        {"function": {"name": "question"}, "type": "function"}
    ]
    content = "__question__:" + _json.dumps({"header": "h", "question": "q"})
    builtin.call_tool = AsyncMock(return_value=content)

    session, _, _ = _make_session(builtin=builtin)
    tus = [{"name": "question", "input": {}, "id": "q1"}]

    async for _ in session._execute_tool_calls(tus):
        pass

    tool_msgs = [m for m in session.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == content


# --------------------------------------------------------------------------
# _exec_tool no-handler branch (lines 518-523)
# --------------------------------------------------------------------------


async def test_exec_tool_no_handler_returns_user_error():
    """No MCP/builtin/skill handler -> ToolError(is_user_error=True) event."""
    session, _, _ = _make_session()  # no mcp, no builtin, no skills
    tus = [{"name": "ghost_tool", "input": {}, "id": "g"}]

    events = []
    async for ev in session._execute_tool_calls(tus):
        events.append(ev)

    err = [e for e in events if e["type"] == "tool_error"]
    assert len(err) == 1
    assert err[0]["name"] == "ghost_tool"
    assert err[0]["is_user_error"] is True
    assert "No handler" in session.messages[-1]["content"]


# --------------------------------------------------------------------------
# ToolError returned (not raised) from a tool (lines 525-546)
# --------------------------------------------------------------------------


async def test_tool_returning_toolerror_becomes_tool_error_event():
    from olmlx.chat.errors import ToolError

    mcp = MagicMock()
    mcp.get_tools_for_chat.return_value = [
        {"function": {"name": "flaky"}, "type": "function"}
    ]
    mcp.call_tool = AsyncMock(
        return_value=ToolError(
            message="bad path", tool_name="flaky", is_user_error=True
        )
    )
    session, _, _ = _make_session(mcp=mcp)
    tus = [{"name": "flaky", "input": {"path": "/x"}, "id": "f"}]

    events = []
    async for ev in session._execute_tool_calls(tus):
        events.append(ev)

    err = [e for e in events if e["type"] == "tool_error"]
    assert len(err) == 1
    assert err[0]["error"] == "bad path"
    assert err[0]["is_user_error"] is True
    assert session.messages[-1]["content"] == "bad path"


# --------------------------------------------------------------------------
# Repetition strip in thinking block (lines 263-272, 1044-1051)
# --------------------------------------------------------------------------


async def test_repetition_during_thinking_emits_thinking_end():
    """Repetition detected mid-<think> strips the block and closes thinking."""
    session, _, _ = _make_session(thinking=True)
    phrase = "<think>" + ("loop loop loop loop " * 1)

    async def gen(*args, **kwargs):
        # Open a think block then repeat a phrase enough to trip detection.
        yield {"text": "<think>", "done": False}
        for _ in range(60):
            yield {"text": "repeat phrase here. ", "done": False}
        yield {"text": "", "done": True}

    with (
        _patch_no_memory_check(),
        patch(
            "olmlx.chat.session.generate_chat",
            side_effect=lambda *a, **kw: gen(*a, **kw),
        ),
    ):
        events = []
        async for ev in session.send_message("Q"):
            events.append(ev)

    types = [e["type"] for e in events]
    assert "repetition_detected" in types
    assert "thinking_end" in types
    assert types[-1] == "done"
    _ = phrase  # silence linters; documents intent
