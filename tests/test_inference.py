"""Tests for olmlx.engine.inference (non-GPU parts)."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import olmlx.engine.inference as _inf_mod
from olmlx.engine.inference import (
    _acquire_inference_lock,
    _apply_seed,
    _build_generate_kwargs,
    estimate_kv_cache_bytes,
    _extract_images,
    _inference_lock,
    _inference_locked,
    _inject_tools_into_system,
    _normalize_tool_calls_in_messages,
    apply_chat_template_text,
    _lock_boundary_sync,
    _safe_sync,
    _schedule_deferred_inference_cleanup,
    generate_chat,
    generate_completion,
    generate_embeddings,
    ServerBusyError,
)
from olmlx.engine.template_caps import TemplateCaps
from olmlx.utils.streaming import CancellableStream, StreamToken
from olmlx.utils.timing import TimingStats


class TestBuildGenerateKwargs:
    def test_empty_options(self):
        assert _build_generate_kwargs(None) == {}
        assert _build_generate_kwargs({}) == {}

    def test_temperature_returns_sampler(self):
        """Text model: temperature should produce a sampler callable, not a temp key."""
        result = _build_generate_kwargs({"temperature": 0.7})
        assert "temp" not in result
        assert "temperature" not in result
        assert callable(result["sampler"])

    def test_sampling_params_folded_into_sampler(self):
        """Text model: top_p, top_k, min_p all folded into sampler callable."""
        result = _build_generate_kwargs(
            {"temperature": 0.5, "top_p": 0.9, "top_k": 40, "min_p": 0.05}
        )
        assert callable(result["sampler"])
        for key in ("temp", "top_p", "top_k", "min_p", "temperature"):
            assert key not in result

    def test_repetition_penalty_returns_logits_processors(self):
        """Text model: repetition_penalty produces logits_processors list."""
        result = _build_generate_kwargs({"repeat_penalty": 1.1, "repeat_last_n": 64})
        assert "repetition_penalty" not in result
        assert "repetition_context_size" not in result
        assert isinstance(result["logits_processors"], list)

    def test_temperature_vlm_unchanged(self):
        """VLM: sampling params passed directly (not via sampler)."""
        result = _build_generate_kwargs({"temperature": 0.7}, is_vlm=True)
        assert result == {"temperature": 0.7}

    def test_vlm_all_params_direct(self):
        """VLM: all sampling params passed as direct kwargs."""
        result = _build_generate_kwargs({"temperature": 0.5, "top_p": 0.9}, is_vlm=True)
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9
        assert "sampler" not in result

    def test_all_mappings(self):
        opts = {
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "seed": 42,
            "num_predict": 100,
            "repeat_penalty": 1.1,
            "repeat_last_n": 64,
            "min_p": 0.05,
        }
        result = _build_generate_kwargs(opts)
        # Sampling params folded into sampler
        assert callable(result["sampler"])
        for key in ("temp", "top_p", "top_k", "min_p"):
            assert key not in result
        # Penalty params folded into logits_processors
        assert isinstance(result["logits_processors"], list)
        for key in ("repetition_penalty", "repetition_context_size"):
            assert key not in result
        # max_tokens still direct
        assert result["max_tokens"] == 100
        # seed in kwargs for _apply_seed to consume before generation
        assert result["seed"] == 42

    def test_stop_warns_and_ignored_for_text_model(self, caplog):
        """stop is silently ignored with a warning (not rejected) for text models."""
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.inference"):
            result = _build_generate_kwargs({"stop": [".", "\n"]})
        assert "stop" not in result
        assert any("stop" in r.message for r in caplog.records)

    def test_stop_vlm_ignored(self):
        result = _build_generate_kwargs({"stop": [".", "\n"]}, is_vlm=True)
        assert "stop" not in result

    def test_frequency_presence_penalty_dropped_with_warning(self, caplog):
        """frequency_penalty/presence_penalty dropped with warning."""
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.inference"):
            result = _build_generate_kwargs(
                {"frequency_penalty": 0.5, "presence_penalty": 0.3}
            )
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result
        assert any("frequency_penalty" in r.message for r in caplog.records)
        assert any("presence_penalty" in r.message for r in caplog.records)

    def test_zero_penalty_warns(self, caplog):
        """Even 0.0 values should warn when explicitly set."""
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.inference"):
            result = _build_generate_kwargs(
                {"frequency_penalty": 0.0, "presence_penalty": 0.0}
            )
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result
        assert any("frequency_penalty" in r.message for r in caplog.records)
        assert any("presence_penalty" in r.message for r in caplog.records)

    def test_unknown_options_ignored(self):
        result = _build_generate_kwargs({"unknown_key": 99})
        assert result == {}

    def test_no_sampler_without_sampling_params(self):
        """No sampler created when only non-sampling params present."""
        result = _build_generate_kwargs({"num_predict": 100})
        assert "sampler" not in result
        assert result["max_tokens"] == 100

    def test_make_sampler_none_raises(self, monkeypatch):
        """When make_sampler is None (mlx-lm not installed), should raise RuntimeError."""
        monkeypatch.setattr(_inf_mod, "make_sampler", None)
        with pytest.raises(RuntimeError, match="mlx-lm is not installed"):
            _build_generate_kwargs({"temperature": 0.7})

    def test_make_logits_processors_none_raises(self, monkeypatch):
        """When make_logits_processors is None (mlx-lm not installed), should raise RuntimeError."""
        monkeypatch.setattr(_inf_mod, "make_logits_processors", None)
        with pytest.raises(RuntimeError, match="mlx-lm is not installed"):
            _build_generate_kwargs({"repeat_penalty": 1.1})

    def test_seed_in_text_kwargs_for_apply_seed(self):
        """Text model: seed must be in kwargs so _apply_seed can read it."""
        result = _build_generate_kwargs({"seed": 42})
        assert result["seed"] == 42

    def test_vlm_seed_in_kwargs(self):
        """VLM: seed should be in kwargs (mlx-vlm accepts it directly)."""
        result = _build_generate_kwargs({"seed": 42}, is_vlm=True)
        assert result["seed"] == 42

    def test_repeat_last_n_without_penalty_warns(self, caplog):
        """repeat_last_n alone is a no-op — should warn and not build processors."""
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.inference"):
            result = _build_generate_kwargs({"repeat_last_n": 64})
        assert "logits_processors" not in result
        assert any("repeat_last_n" in r.message for r in caplog.records)

    def test_sampling_without_temperature_no_sampler(self):
        """top_k/top_p/min_p without temperature should not build a sampler.

        make_sampler defaults temp=0.0 (greedy), which makes other sampling
        params irrelevant — a silent regression from old behavior.
        """
        result = _build_generate_kwargs({"top_k": 40})
        assert "sampler" not in result


class TestApplySeed:
    def test_apply_seed_consume_pops(self):
        """_apply_seed(consume=True) should pop seed — text models must not forward it."""
        kwargs = {"seed": 42, "max_tokens": 100}
        with patch("olmlx.engine.inference.mx") as mock_mx:
            _apply_seed(kwargs, consume=True)
        assert "seed" not in kwargs
        mock_mx.random.seed.assert_called_once_with(42)

    def test_apply_seed_no_consume_keeps(self):
        """_apply_seed(consume=False) should keep seed — VLMs forward it to mlx-vlm."""
        kwargs = {"seed": 42, "max_tokens": 100}
        with patch("olmlx.engine.inference.mx") as mock_mx:
            _apply_seed(kwargs, consume=False)
        assert "seed" in kwargs
        mock_mx.random.seed.assert_called_once_with(42)

    def test_apply_seed_no_seed(self):
        """_apply_seed with no seed key should be a no-op."""
        kwargs = {"max_tokens": 100}
        with patch("olmlx.engine.inference.mx") as mock_mx:
            _apply_seed(kwargs)
        mock_mx.random.seed.assert_not_called()


class TestExtractImages:
    def test_no_images(self):
        messages = [{"role": "user", "content": "hi"}]
        assert _extract_images(messages) is None

    def test_with_images(self):
        messages = [
            {"role": "user", "content": "describe", "images": ["img1.jpg", "img2.png"]},
        ]
        result = _extract_images(messages)
        assert result == ["img1.jpg", "img2.png"]

    def test_multiple_messages(self):
        messages = [
            {"role": "user", "content": "first", "images": ["a.jpg"]},
            {"role": "user", "content": "second", "images": ["b.jpg"]},
        ]
        result = _extract_images(messages)
        assert result == ["a.jpg", "b.jpg"]

    def test_empty_images_list(self):
        messages = [{"role": "user", "content": "hi", "images": []}]
        assert _extract_images(messages) is None


class TestInjectToolsIntoSystem:
    def test_with_existing_system(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        tools = [
            {
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {},
                }
            }
        ]
        result = _inject_tools_into_system(messages, tools)
        assert result[0]["role"] == "system"
        assert "search" in result[0]["content"]
        assert "You are helpful." in result[0]["content"]
        # Original should not be modified
        assert "search" not in messages[0]["content"]

    def test_without_system(self):
        messages = [{"role": "user", "content": "hi"}]
        tools = [
            {
                "function": {
                    "name": "func",
                    "description": "A function",
                    "parameters": {},
                }
            }
        ]
        result = _inject_tools_into_system(messages, tools)
        assert result[0]["role"] == "system"
        assert "func" in result[0]["content"]
        assert result[1] == messages[0]

    def test_tool_without_function_key(self):
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"name": "direct_tool", "description": "Direct", "parameters": {}}]
        result = _inject_tools_into_system(messages, tools)
        assert "direct_tool" in result[0]["content"]


class TestAddNativeToolHint:
    def test_appends_hint_when_function_pattern_present(self):
        from olmlx.engine.inference import _add_native_tool_hint

        messages = [
            {
                "role": "system",
                "content": "Use this format: <function=Name>...</function>",
            },
            {"role": "user", "content": "hi"},
        ]
        result = _add_native_tool_hint(messages)
        assert "native tool call format" in result[0]["content"]
        # Original not modified
        assert "native" not in messages[0]["content"]

    def test_appends_hint_when_tool_calls_pattern_present(self):
        from olmlx.engine.inference import _add_native_tool_hint

        messages = [
            {"role": "system", "content": "Mistral format: [TOOL_CALLS] [...]"},
        ]
        result = _add_native_tool_hint(messages)
        assert "native tool call format" in result[0]["content"]

    def test_appends_hint_when_python_tag_pattern_present(self):
        from olmlx.engine.inference import _add_native_tool_hint

        messages = [
            {"role": "system", "content": "Use <|python_tag|>{...}"},
        ]
        result = _add_native_tool_hint(messages)
        assert "native tool call format" in result[0]["content"]

    def test_no_hint_for_qwen_native_tool_call(self):
        """`<tool_call>` is Qwen's native format — must not trigger the hint.

        For Qwen models, client instructions describing `<tool_call>` match
        the model's actual native format.  Adding the hint would tell the
        model to "disregard" instructions that are correct for it.
        """
        from olmlx.engine.inference import _add_native_tool_hint

        messages = [
            {"role": "system", "content": "Use <tool_call>{...}</tool_call>"},
        ]
        result = _add_native_tool_hint(messages)
        assert "native tool call format" not in result[0]["content"]

    def test_no_hint_when_pattern_is_in_template(self):
        """Skip patterns that the model's own chat template uses natively.

        Mistral's template literally contains `[TOOL_CALLS]`, so client
        instructions referencing that pattern match the native format and
        should not trigger the override.
        """
        from olmlx.engine.inference import _add_native_tool_hint

        messages = [
            {"role": "system", "content": "Use [TOOL_CALLS] [...]"},
        ]
        # Mistral-style template — contains [TOOL_CALLS] natively
        template_text = "{% if tools %}[TOOL_CALLS] {{ tools }}{% endif %}"
        result = _add_native_tool_hint(messages, template_text)
        assert "native tool call format" not in result[0]["content"]

    def test_hint_fires_when_pattern_not_in_template(self):
        """Pattern in system message but NOT in template → hint fires."""
        from olmlx.engine.inference import _add_native_tool_hint

        messages = [
            {"role": "system", "content": "Use <function=Name>...</function>"},
        ]
        # Gemma 4-style template — uses <|tool_call> tokens, not <function=
        template_text = "{% if tools %}<|tool_call>{{ tools }}<tool_call|>{% endif %}"
        result = _add_native_tool_hint(messages, template_text)
        assert "native tool call format" in result[0]["content"]

    def test_no_hint_when_no_conflict_pattern(self):
        """Plain system prompts without conflicting format instructions are untouched."""
        from olmlx.engine.inference import _add_native_tool_hint

        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "hi"},
        ]
        result = _add_native_tool_hint(messages)
        assert result[0]["content"] == "You are a helpful coding assistant."

    def test_no_system_message_is_noop(self):
        from olmlx.engine.inference import _add_native_tool_hint

        messages = [{"role": "user", "content": "<function=foo>bar</function>"}]
        result = _add_native_tool_hint(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "native tool call format" not in result[0]["content"]

    def test_not_duplicated(self):
        from olmlx.engine.inference import _add_native_tool_hint

        messages = [
            {"role": "system", "content": "Use <function=Name>...</function>"},
        ]
        result = _add_native_tool_hint(messages)
        result2 = _add_native_tool_hint(result)
        assert result2[0]["content"].count("native tool call format") == 1

    def test_non_string_content_is_noop(self):
        from olmlx.engine.inference import _add_native_tool_hint

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "<function=foo>"}],
            },
        ]
        result = _add_native_tool_hint(messages)
        assert result[0]["content"] == [{"type": "text", "text": "<function=foo>"}]


class TestConvertToolMessagesToResponses:
    def test_no_tool_messages_unchanged(self):
        """Messages without role=tool pass through unchanged."""
        from olmlx.engine.inference import _convert_tool_messages_to_responses

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _convert_tool_messages_to_responses(messages)
        assert result == messages

    def test_tool_message_merged_into_preceding_assistant(self):
        """role=tool messages become tool_responses on the preceding assistant message."""
        from olmlx.engine.inference import _convert_tool_messages_to_responses

        messages = [
            {"role": "user", "content": "search for cats"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "search", "arguments": {"q": "cats"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "found 3 cats"},
            {"role": "user", "content": "thanks"},
        ]
        result = _convert_tool_messages_to_responses(messages)
        # role=tool message should be gone
        assert not any(m.get("role") == "tool" for m in result)
        # tool_responses should be on the assistant message (index 1)
        assert result[1]["role"] == "assistant"
        assert result[1]["tool_responses"] == [
            {"name": "search", "response": "found 3 cats"}
        ]
        # Intermediate assistant gets newline content to close the model turn
        assert result[1]["content"] == "\n"
        # user message still present
        assert result[2] == {"role": "user", "content": "thanks"}

    def test_multiple_tool_results_merged_into_assistant(self):
        """Multiple consecutive tool messages merged into preceding assistant."""
        from olmlx.engine.inference import _convert_tool_messages_to_responses

        messages = [
            {"role": "user", "content": "do things"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "read", "arguments": {"f": "a.py"}},
                    },
                    {
                        "id": "tc2",
                        "type": "function",
                        "function": {"name": "write", "arguments": {"f": "b.py"}},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "contents of a"},
            {"role": "tool", "tool_call_id": "tc2", "content": "ok"},
        ]
        result = _convert_tool_messages_to_responses(messages)
        # Both tool responses merged into the assistant message
        assert result[1]["role"] == "assistant"
        assert len(result[1]["tool_responses"]) == 2
        assert result[1]["tool_responses"][0]["name"] == "read"
        assert result[1]["tool_responses"][1]["name"] == "write"
        # Only 2 messages remain (user + assistant)
        assert len(result) == 2

    def test_tool_name_resolved_from_preceding_assistant(self):
        """Tool name is resolved from the assistant's tool_calls by matching tool_call_id."""
        from olmlx.engine.inference import _convert_tool_messages_to_responses

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": {"cmd": "ls"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "file1\nfile2"},
        ]
        result = _convert_tool_messages_to_responses(messages)
        tool_resp_msg = [m for m in result if "tool_responses" in m]
        assert tool_resp_msg[0]["tool_responses"][0]["name"] == "Bash"

    def test_last_assistant_keeps_empty_content(self):
        """Last assistant with tool_responses keeps empty content for model continuation."""
        from olmlx.engine.inference import _convert_tool_messages_to_responses

        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": {}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "done"},
        ]
        result = _convert_tool_messages_to_responses(messages)
        # Last message — content stays empty so model turn stays open
        assert result[1]["content"] == ""

    def test_intermediate_assistant_gets_newline_content(self):
        """Intermediate assistant with tool_responses gets newline to close model turn."""
        from olmlx.engine.inference import _convert_tool_messages_to_responses

        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": {}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "file1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc2",
                        "type": "function",
                        "function": {"name": "Read", "arguments": {}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc2", "content": "contents"},
        ]
        result = _convert_tool_messages_to_responses(messages)
        # First assistant (intermediate) gets newline to close turn
        assert result[1]["content"] == "\n"
        assert result[1]["tool_responses"][0]["name"] == "Bash"
        # Second assistant (last) keeps empty content
        assert result[2]["content"] == ""
        assert result[2]["tool_responses"][0]["name"] == "Read"

    def test_original_messages_not_modified(self):
        """Conversion does not mutate the input list."""
        from olmlx.engine.inference import _convert_tool_messages_to_responses

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "foo", "arguments": {}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "bar"},
        ]
        original_len = len(messages)
        _convert_tool_messages_to_responses(messages)
        assert len(messages) == original_len
        assert messages[1]["role"] == "tool"


class TestApplyChatTemplateText:
    def test_basic(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted prompt")
        messages = [{"role": "user", "content": "hi"}]
        result = apply_chat_template_text(tokenizer, messages)
        assert result == "formatted prompt"
        tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def test_with_tools_supported(self):
        """Default (enable_thinking=None) + tools → enable_thinking=False (backward compat)."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt with tools")
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "f"}}]
        caps = TemplateCaps(supports_tools=True, supports_enable_thinking=True)
        apply_chat_template_text(tokenizer, messages, tools, caps)
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["enable_thinking"] is False

    def test_with_tools_not_supported(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "description": "desc", "parameters": {}},
            }
        ]
        caps = TemplateCaps(supports_tools=False)
        apply_chat_template_text(tokenizer, messages, tools, caps)
        # Should inject tools into system message instead
        call_args = tokenizer.apply_chat_template.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert "f" in call_args[0]["content"]

    def test_fallback_on_template_error(self):
        tokenizer = MagicMock()
        call_count = [0]

        def side_effect(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and "tools" in kwargs:
                raise TypeError("tools not supported")
            return "fallback prompt"

        tokenizer.apply_chat_template = side_effect
        messages = [{"role": "user", "content": "hi"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "description": "d", "parameters": {}},
            }
        ]
        caps = TemplateCaps(supports_tools=True)
        result = apply_chat_template_text(tokenizer, messages, tools, caps)
        assert result == "fallback prompt"
        assert call_count[0] == 2

    def test_fallback_preserves_enable_thinking(self):
        """When tools kwarg fails and falls back to injection, enable_thinking is preserved."""
        tokenizer = MagicMock()
        call_count = [0]
        retry_kwargs = {}

        def side_effect(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and "tools" in kwargs:
                raise TypeError("tools not supported")
            retry_kwargs.update(kwargs)
            return "fallback prompt"

        tokenizer.apply_chat_template = side_effect
        messages = [{"role": "user", "content": "hi"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "description": "d", "parameters": {}},
            }
        ]
        caps = TemplateCaps(supports_tools=True, supports_enable_thinking=True)
        result = apply_chat_template_text(
            tokenizer, messages, tools, caps, enable_thinking=False
        )
        assert result == "fallback prompt"
        assert retry_kwargs.get("enable_thinking") is False
        assert "tools" not in retry_kwargs

    def test_enable_thinking_true(self):
        """Explicit enable_thinking=True → passed through to template."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        caps = TemplateCaps(supports_tools=False, supports_enable_thinking=True)
        apply_chat_template_text(tokenizer, messages, caps=caps, enable_thinking=True)
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["enable_thinking"] is True

    def test_enable_thinking_false(self):
        """Explicit enable_thinking=False → passed through to template."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        caps = TemplateCaps(supports_tools=False, supports_enable_thinking=True)
        apply_chat_template_text(tokenizer, messages, caps=caps, enable_thinking=False)
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["enable_thinking"] is False

    def test_enable_thinking_none_no_tools_defaults_true(self):
        """enable_thinking=None + no tools → defaults to True."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        caps = TemplateCaps(supports_tools=False, supports_enable_thinking=True)
        apply_chat_template_text(tokenizer, messages, caps=caps)
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["enable_thinking"] is True

    def test_enable_thinking_true_with_tools(self):
        """Explicit enable_thinking=True + tools → both tools and enable_thinking=True passed."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "f"}}]
        caps = TemplateCaps(supports_tools=True, supports_enable_thinking=True)
        apply_chat_template_text(tokenizer, messages, tools, caps, enable_thinking=True)
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["enable_thinking"] is True

    def test_enable_thinking_not_supported(self):
        """caps.supports_enable_thinking=False → enable_thinking kwarg not passed."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        caps = TemplateCaps(supports_tools=False, supports_enable_thinking=False)
        apply_chat_template_text(tokenizer, messages, caps=caps, enable_thinking=True)
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert "enable_thinking" not in call_kwargs

    def test_no_caps_defaults(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="result")
        apply_chat_template_text(tokenizer, [], None, caps=None)
        tokenizer.apply_chat_template.assert_called_once()

    def test_template_fails_completely(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(side_effect=RuntimeError("broken"))
        with pytest.raises(RuntimeError, match="Chat template failed"):
            apply_chat_template_text(tokenizer, [], None)

    def test_tools_kwarg_fails_then_injection_fails(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(
            side_effect=TypeError("all calls fail")
        )
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "description": "d", "parameters": {}},
            }
        ]
        caps = TemplateCaps(supports_tools=True)
        with pytest.raises(
            RuntimeError, match="Chat template failed even without tools"
        ):
            apply_chat_template_text(tokenizer, [], tools, caps)


class TestApplyChatTemplateVlm:
    def test_vlm_tools_with_images_warns(self, caplog):
        """When tools and images are both provided, a warning must be logged."""
        from olmlx.engine.inference import _apply_chat_template_vlm

        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "prompt"
        mock_processor.tokenizer = mock_tok
        mock_model = MagicMock()

        tools = [{"type": "function", "function": {"name": "test"}}]

        with caplog.at_level(logging.WARNING, logger="olmlx.engine.inference"):
            _apply_chat_template_vlm(
                mock_processor,
                mock_model,
                [{"role": "user", "content": "describe"}],
                images=["img.jpg"],
                tools=tools,
            )

        assert any("image" in r.message.lower() for r in caplog.records), (
            f"Expected warning about images, got: {[r.message for r in caplog.records]}"
        )

    def test_vlm_template(self):
        from olmlx.engine.inference import _apply_chat_template_vlm

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt"
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.config = {"model_type": "qwen2_vl"}

        with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
            result = _apply_chat_template_vlm(
                mock_processor,
                mock_model,
                [{"role": "user", "content": "describe"}],
                images=["img.jpg"],
            )

        assert result == "vlm prompt"
        mock_mlx_vlm.apply_chat_template.assert_called_once()

    def test_vlm_template_no_config(self):
        from olmlx.engine.inference import _apply_chat_template_vlm

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "result"
        mock_processor = MagicMock()
        mock_model = MagicMock(spec=[])  # no config attr

        with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
            result = _apply_chat_template_vlm(
                mock_processor,
                mock_model,
                [{"role": "user", "content": "hi"}],
            )
        assert result == "result"


class TestNormalizeToolCallsInMessages:
    """Normalise OpenAI-format tool_calls to flat format for chat templates."""

    def test_openai_format_converted(self):
        """OpenAI format produces union dict with both flat and nested keys."""
        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "sunny"},
        ]
        result = _normalize_tool_calls_in_messages(messages)
        tc = result[1]["tool_calls"][0]
        # Flat keys (Qwen)
        assert tc["name"] == "get_weather"
        assert tc["arguments"] == {"city": "London"}
        # Nested keys (Gemma)
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == {"city": "London"}
        # Preserved metadata
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        # Non-assistant messages unchanged
        assert result[0] == messages[0]
        assert result[2] == messages[2]

    def test_already_flat_format_gets_function_key(self):
        """Qwen-style {name, arguments: dict} gets function key added."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"name": "search", "arguments": {"q": "test"}},
                ],
            },
        ]
        result = _normalize_tool_calls_in_messages(messages)
        tc = result[0]["tool_calls"][0]
        assert tc["name"] == "search"
        assert tc["arguments"] == {"q": "test"}
        assert tc["function"]["name"] == "search"

    def test_no_tool_calls_unchanged(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _normalize_tool_calls_in_messages(messages)
        assert result == messages

    def test_invalid_json_arguments(self):
        """Malformed JSON arguments fall back to empty dict."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "foo", "arguments": "not json"},
                    }
                ],
            },
        ]
        result = _normalize_tool_calls_in_messages(messages)
        assert result[0]["tool_calls"][0]["arguments"] == {}


class TestGemma4TemplateOrdering:
    """Document Gemma 4 template ordering: system prompt before tool declarations.

    Gemma 4's chat template places the system prompt content *before* native
    ``<|tool>`` declarations.  When the system prompt is long and contains its
    own natural-language tool-usage instructions (e.g. Claude Code's system
    prompt), the model may follow those instructions instead of producing
    native ``<|tool_call>`` output.

    Qwen 3.5's template avoids this by placing explicit tool-calling format
    instructions (with examples) *before* the system prompt.
    """

    GEMMA4_TEMPLATE = (
        "/Users/daniel/.olmlx/models/"
        "mlx-community_gemma-4-26b-a4b-it-4bit/chat_template.jinja"
    )
    QWEN35_TEMPLATE = (
        "/Users/daniel/.olmlx/models/mlx-community_Qwen3.5-27B-4bit/chat_template.jinja"
    )

    @staticmethod
    def _render(template_path, messages, tools=None, enable_thinking=None):
        """Render a Jinja2 chat template from file."""
        import os

        from jinja2 import BaseLoader, Environment

        if not os.path.exists(template_path):
            pytest.skip(f"Template not found: {template_path}")

        with open(template_path) as f:
            source = f.read()

        env = Environment(loader=BaseLoader())
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(
            ValueError(msg)
        )
        tmpl = env.from_string(source)

        kwargs = {
            "messages": messages,
            "add_generation_prompt": True,
            "bos_token": "<bos>",
        }
        if tools:
            kwargs["tools"] = tools
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return tmpl.render(**kwargs)

    def test_gemma4_system_before_tools(self):
        """Gemma 4 places system content before <|tool> declarations."""
        messages = [
            {"role": "system", "content": "SYSTEM_MARKER"},
            {"role": "user", "content": "hello"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run a command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command",
                            }
                        },
                        "required": ["command"],
                    },
                },
            }
        ]
        prompt = self._render(self.GEMMA4_TEMPLATE, messages, tools=tools)
        sys_pos = prompt.find("SYSTEM_MARKER")
        tool_pos = prompt.find("<|tool>")
        assert sys_pos != -1, "System content not found in prompt"
        assert tool_pos != -1, "Tool declaration not found in prompt"
        assert sys_pos < tool_pos, (
            f"Expected system content (pos {sys_pos}) before tool declarations "
            f"(pos {tool_pos}), but tools came first"
        )

    def test_qwen35_tools_before_system(self):
        """Qwen 3.5 places tool instructions before system content."""
        messages = [
            {"role": "system", "content": "SYSTEM_MARKER"},
            {"role": "user", "content": "hello"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run a command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command",
                            }
                        },
                        "required": ["command"],
                    },
                },
            }
        ]
        prompt = self._render(self.QWEN35_TEMPLATE, messages, tools=tools)
        sys_pos = prompt.find("SYSTEM_MARKER")
        tool_pos = prompt.find("# Tools")
        assert sys_pos != -1, "System content not found in prompt"
        assert tool_pos != -1, "Tool instructions not found in prompt"
        assert tool_pos < sys_pos, (
            f"Expected tool instructions (pos {tool_pos}) before system content "
            f"(pos {sys_pos}), but system came first"
        )

    def test_gemma4_thinking_disabled_by_default(self):
        """When enable_thinking is not passed, Gemma 4 forces empty thinking."""
        messages = [
            {"role": "user", "content": "hello"},
        ]
        prompt = self._render(self.GEMMA4_TEMPLATE, messages)
        assert "<|channel>thought\n<channel|>" in prompt, (
            "Expected empty thinking block when enable_thinking is not set"
        )

    def test_gemma4_thinking_enabled(self):
        """When enable_thinking=True, Gemma 4 allows the model to think."""
        messages = [
            {"role": "user", "content": "hello"},
        ]
        prompt = self._render(self.GEMMA4_TEMPLATE, messages, enable_thinking=True)
        assert "<|think|>" in prompt, (
            "Expected <|think|> token when enable_thinking=True"
        )
        assert "<|channel>thought\n<channel|>" not in prompt, (
            "Should NOT force empty thinking when enable_thinking=True"
        )


class TestGenerateCompletion:
    @pytest.mark.asyncio
    async def test_non_streaming(self, mock_manager):
        mock_mx = MagicMock()

        mock_mx.core = mock_mx

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate.return_value = "Generated output"

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                with patch(
                    "olmlx.engine.inference.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_thread:
                    mock_thread.return_value = "Generated output"
                    result = await generate_completion(
                        mock_manager,
                        "qwen3",
                        "Hello",
                        stream=False,
                    )

        assert result["text"] == "Generated output"
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_apply_chat_template(self, mock_manager):
        """When apply_chat_template=True, prompt is wrapped as a user message
        and passed through the chat template before generation."""
        mock_mx = MagicMock()
        mock_mx.core = mock_mx
        mock_mlx_lm = MagicMock()

        lm = mock_manager._loaded["qwen3:latest"]
        lm.text_tokenizer.apply_chat_template.return_value = (
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        )

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                with patch(
                    "olmlx.engine.inference.asyncio.to_thread",
                    new_callable=AsyncMock,
                    return_value="Generated output",
                ):
                    result = await generate_completion(
                        mock_manager,
                        "qwen3",
                        "Hello",
                        stream=False,
                        apply_chat_template=True,
                    )

        assert result["text"] == "Generated output"
        # Verify the chat template was applied with correct messages and caps
        lm.text_tokenizer.apply_chat_template.assert_called_once()
        call_args = lm.text_tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert messages == [{"role": "user", "content": "Hello"}]
        # Should pass enable_thinking=False for /api/generate (no thinking extraction)
        assert call_args[1]["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_apply_chat_template_with_system(self, mock_manager):
        """System prompt becomes a proper system role message, not part of user content."""
        mock_mx = MagicMock()
        mock_mx.core = mock_mx
        mock_mlx_lm = MagicMock()
        lm = mock_manager._loaded["qwen3:latest"]
        lm.text_tokenizer.apply_chat_template.return_value = "templated"

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                with patch(
                    "olmlx.engine.inference.asyncio.to_thread",
                    new_callable=AsyncMock,
                    return_value="output",
                ):
                    await generate_completion(
                        mock_manager,
                        "qwen3",
                        "Hello",
                        stream=False,
                        apply_chat_template=True,
                        system="You are helpful",
                    )

        call_args = lm.text_tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert messages == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

    @pytest.mark.asyncio
    async def test_apply_chat_template_fallback_on_error(self, mock_manager):
        """If the chat template fails (e.g. base model), fall back to raw prompt."""
        mock_mx = MagicMock()
        mock_mx.core = mock_mx
        mock_mlx_lm = MagicMock()
        lm = mock_manager._loaded["qwen3:latest"]
        lm.text_tokenizer.apply_chat_template.side_effect = RuntimeError(
            "No chat template"
        )

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                with patch(
                    "olmlx.engine.inference.asyncio.to_thread",
                    new_callable=AsyncMock,
                    return_value="output",
                ):
                    result = await generate_completion(
                        mock_manager,
                        "qwen3",
                        "Hello",
                        stream=False,
                        apply_chat_template=True,
                    )

        # Should succeed with the raw prompt, not raise
        assert result["text"] == "output"

    @pytest.mark.asyncio
    async def test_apply_chat_template_fallback_preserves_system(self, mock_manager):
        """When template fails, system prompt is prepended as plain text."""
        mock_mx = MagicMock()
        mock_mx.core = mock_mx
        mock_mlx_lm = MagicMock()
        lm = mock_manager._loaded["qwen3:latest"]
        lm.text_tokenizer.apply_chat_template.side_effect = RuntimeError(
            "No chat template"
        )

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                # Patch _full_completion to capture the prompt it receives
                with patch(
                    "olmlx.engine.inference._full_completion",
                    new_callable=AsyncMock,
                ) as mock_full:
                    mock_full.return_value = {"text": "output", "done": True}
                    await generate_completion(
                        mock_manager,
                        "qwen3",
                        "Hello",
                        stream=False,
                        apply_chat_template=True,
                        system="You are helpful",
                    )

        # _full_completion receives (lm, prompt, max_tokens, gen_kwargs, stats, images)
        prompt_arg = mock_full.call_args[0][1]
        assert prompt_arg == "You are helpful\n\nHello"

    @pytest.mark.asyncio
    async def test_apply_chat_template_vlm_uses_vlm_template(self, mock_manager):
        """VLM models apply chat template via _apply_chat_template_vlm."""
        mock_mx = MagicMock()
        mock_mx.core = mock_mx
        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm formatted prompt"

        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                with patch(
                    "olmlx.engine.inference._full_completion",
                    new_callable=AsyncMock,
                ) as mock_full:
                    mock_full.return_value = {"text": "Hi", "done": True}
                    await generate_completion(
                        mock_manager,
                        "qwen3",
                        "Hello",
                        stream=False,
                        apply_chat_template=True,
                    )

        # Text template should NOT be applied for VLM models
        lm.text_tokenizer.apply_chat_template.assert_not_called()
        # VLM template should be applied — prompt should be formatted
        mock_mlx_vlm.apply_chat_template.assert_called_once()
        prompt_arg = mock_full.call_args[0][1]
        assert prompt_arg == "vlm formatted prompt"

    @pytest.mark.asyncio
    async def test_apply_chat_template_vlm_with_system(self, mock_manager):
        """VLM models include system message in chat template."""
        mock_mx = MagicMock()
        mock_mx.core = mock_mx
        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm with system"

        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                with patch(
                    "olmlx.engine.inference._full_completion",
                    new_callable=AsyncMock,
                ) as mock_full:
                    mock_full.return_value = {"text": "Hi", "done": True}
                    await generate_completion(
                        mock_manager,
                        "qwen3",
                        "Hello",
                        stream=False,
                        apply_chat_template=True,
                        system="You are helpful",
                    )

        # Check messages passed to apply_chat_template include system
        call_args = mock_mlx_vlm.apply_chat_template.call_args
        messages = call_args[0][2]  # 3rd positional arg
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "Hello"}
        prompt_arg = mock_full.call_args[0][1]
        assert prompt_arg == "vlm with system"

    @pytest.mark.asyncio
    async def test_streaming(self, mock_manager):
        mock_mx = MagicMock()

        tokens = [
            StreamToken(
                text="Hello",
                token=1,
                prompt_tokens=5,
                generation_tokens=1,
                prompt_tps=100.0,
                generation_tps=50.0,
            ),
            StreamToken(
                text=" world",
                token=2,
                prompt_tokens=5,
                generation_tokens=2,
                prompt_tps=100.0,
                generation_tps=50.0,
            ),
        ]

        mock_stream = MagicMock(spec=CancellableStream)
        mock_stream.drain_and_join = AsyncMock()
        mock_stream._thread = None  # No thread — normal completion path

        # Make it a proper async iterator
        token_iter = iter(tokens)

        async def anext_impl():
            try:
                return next(token_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.async_mlx_stream", return_value=mock_stream
            ):
                gen = await generate_completion(
                    mock_manager,
                    "qwen3",
                    "Hello",
                    stream=True,
                )
                chunks = []
                async for chunk in gen:
                    chunks.append(chunk)

        # last chunk is the done signal
        assert chunks[-1]["done"] is True
        assert any(c.get("text") == "Hello" for c in chunks if not c.get("done"))
        # prompt_eval_duration should be derived from prompt_tps so the
        # Ollama-compatible response surfaces a non-zero prefill time.
        # 5 prompt tokens at 100 tok/s → 50 ms = 50_000_000 ns
        done_stats = chunks[-1]["stats"]
        assert done_stats.prompt_eval_count == 5
        assert done_stats.prompt_eval_duration == 50_000_000


class TestGenerateChat:
    @pytest.mark.asyncio
    async def test_non_streaming(self, mock_manager):
        mock_mx = MagicMock()

        mock_manager._loaded["qwen3:latest"].tokenizer.apply_chat_template = MagicMock(
            return_value="formatted prompt"
        )

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
            ) as mock_thread:
                mock_thread.return_value = "Chat response"
                result = await generate_chat(
                    mock_manager,
                    "qwen3",
                    [{"role": "user", "content": "hi"}],
                    stream=False,
                )

        assert result["text"] == "Chat response"
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_with_tools(self, mock_manager):
        mock_mx = MagicMock()

        mock_manager._loaded["qwen3:latest"].tokenizer.apply_chat_template = MagicMock(
            return_value="formatted prompt with tools"
        )

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
            ) as mock_thread:
                mock_thread.return_value = "tool response"
                result = await generate_chat(
                    mock_manager,
                    "qwen3",
                    [{"role": "user", "content": "search"}],
                    tools=[{"type": "function", "function": {"name": "search"}}],
                    stream=False,
                )

        assert result["text"] == "tool response"


class TestGenerateChatEnableThinking:
    @pytest.mark.asyncio
    async def test_enable_thinking_passed_to_template(self, mock_manager):
        """generate_chat(enable_thinking=True) passes it to template application."""
        mock_mx = MagicMock()

        mock_manager._loaded["qwen3:latest"].tokenizer.apply_chat_template = MagicMock(
            return_value="formatted prompt"
        )

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
            ) as mock_thread:
                mock_thread.return_value = "response"
                await generate_chat(
                    mock_manager,
                    "qwen3",
                    [{"role": "user", "content": "hi"}],
                    stream=False,
                    enable_thinking=True,
                )

        call_kwargs = mock_manager._loaded[
            "qwen3:latest"
        ].tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["enable_thinking"] is True

    @pytest.mark.asyncio
    async def test_enable_thinking_false_passed_to_template(self, mock_manager):
        """generate_chat(enable_thinking=False) passes it to template application."""
        mock_mx = MagicMock()

        mock_manager._loaded["qwen3:latest"].tokenizer.apply_chat_template = MagicMock(
            return_value="formatted prompt"
        )

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
            ) as mock_thread:
                mock_thread.return_value = "response"
                await generate_chat(
                    mock_manager,
                    "qwen3",
                    [{"role": "user", "content": "hi"}],
                    stream=False,
                    enable_thinking=False,
                )

        call_kwargs = mock_manager._loaded[
            "qwen3:latest"
        ].tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["enable_thinking"] is False


class TestFullCompletionInner:
    @pytest.mark.asyncio
    async def test_result_with_text_attr(self, mock_manager):
        """Test when result has .text attribute (GenerationResult dataclass)."""
        from olmlx.engine.inference import _full_completion_inner

        lm = mock_manager._loaded["qwen3:latest"]

        mock_result = MagicMock()
        mock_result.text = "result text"

        with patch(
            "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread:
            mock_thread.return_value = mock_result
            result = await _full_completion_inner(
                lm,
                "prompt",
                100,
                {},
                TimingStats(),
            )

        assert result["text"] == "result text"

    @pytest.mark.asyncio
    async def test_result_as_string(self, mock_manager):
        from olmlx.engine.inference import _full_completion_inner

        lm = mock_manager._loaded["qwen3:latest"]

        with patch(
            "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread:
            mock_thread.return_value = "plain string"
            result = await _full_completion_inner(
                lm,
                "prompt",
                100,
                {},
                TimingStats(),
            )

        assert result["text"] == "plain string"

    @pytest.mark.asyncio
    async def test_result_other_type(self, mock_manager):
        from olmlx.engine.inference import _full_completion_inner

        lm = mock_manager._loaded["qwen3:latest"]

        with patch(
            "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread:
            mock_thread.return_value = 42  # unusual type
            result = await _full_completion_inner(
                lm,
                "prompt",
                100,
                {},
                TimingStats(),
            )

        assert result["text"] == "42"

    @pytest.mark.asyncio
    async def test_vlm_completion(self, mock_manager):
        from olmlx.engine.inference import _full_completion_inner

        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True

        mock_mlx_vlm = MagicMock()
        with patch(
            "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread:
            mock_thread.return_value = "vlm output"
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                result = await _full_completion_inner(
                    lm,
                    "prompt",
                    100,
                    {},
                    TimingStats(),
                    images=["img.jpg"],
                )

        assert result["text"] == "vlm output"


class TestGenerateChatVlm:
    @pytest.mark.asyncio
    async def test_vlm_always_uses_vlm_template(self, mock_manager):
        """VLM models always use VLM template path, even with enable_thinking."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True
        lm.template_caps = TemplateCaps(
            supports_tools=True, supports_enable_thinking=True
        )

        mock_mx = MagicMock()
        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt"

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                with patch(
                    "olmlx.engine.inference.apply_chat_template_text",
                ) as mock_text_tpl:
                    with patch(
                        "olmlx.engine.inference.asyncio.to_thread",
                        new_callable=AsyncMock,
                    ) as mock_thread:
                        mock_thread.return_value = "response"
                        await generate_chat(
                            mock_manager,
                            "qwen3",
                            [{"role": "user", "content": "describe"}],
                            stream=False,
                            enable_thinking=False,
                        )

        # VLM should use VLM template, not text template
        mock_text_tpl.assert_not_called()
        mock_mlx_vlm.apply_chat_template.assert_called_once()

    @pytest.mark.asyncio
    async def test_vlm_enable_thinking_warns_when_unsupported(
        self, mock_manager, caplog
    ):
        """VLM path logs warning when enable_thinking is set but template doesn't support it."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True
        lm.template_caps = TemplateCaps(
            supports_tools=False, supports_enable_thinking=False
        )

        mock_mx = MagicMock()

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt"

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                with patch(
                    "olmlx.engine.inference.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_thread:
                    mock_thread.return_value = "vlm response"
                    with caplog.at_level(
                        logging.DEBUG, logger="olmlx.engine.inference"
                    ):
                        await generate_chat(
                            mock_manager,
                            "qwen3",
                            [{"role": "user", "content": "describe"}],
                            stream=False,
                            enable_thinking=True,
                        )

        assert any("ignored for VLM" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_vlm_without_tools(self, mock_manager):
        """Test chat with VLM model (no tools)."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True

        mock_mx = MagicMock()

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt"

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                with patch(
                    "olmlx.engine.inference.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_thread:
                    mock_thread.return_value = "vlm response"
                    result = await generate_chat(
                        mock_manager,
                        "qwen3",
                        [
                            {
                                "role": "user",
                                "content": "describe",
                                "images": ["img.jpg"],
                            }
                        ],
                        stream=False,
                    )

        assert result["text"] == "vlm response"

    @pytest.mark.asyncio
    async def test_vlm_with_tools_native_support(self, mock_manager):
        """VLM+tools uses tokenizer directly for native tool template (bypasses mlx_vlm)."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True
        lm.template_caps = TemplateCaps(
            supports_tools=True, supports_enable_thinking=True
        )

        mock_mx = MagicMock()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {},
                },
            }
        ]

        # Mock the tokenizer's apply_chat_template (called directly, not via mlx_vlm)
        lm.tokenizer.tokenizer = MagicMock()
        lm.tokenizer.tokenizer.apply_chat_template = MagicMock(
            return_value="direct tokenizer prompt with tools"
        )

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as mock_thread:
                mock_thread.return_value = "tool response"
                result = await generate_chat(
                    mock_manager,
                    "qwen3",
                    [{"role": "user", "content": "search"}],
                    tools=tools,
                    stream=False,
                )

        # Should call tokenizer.apply_chat_template directly with tools
        tpl_call = lm.tokenizer.tokenizer.apply_chat_template
        tpl_call.assert_called_once()
        call_kwargs = tpl_call.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tokenize"] is False
        assert call_kwargs["add_generation_prompt"] is True
        assert result["text"] == "tool response"

    @pytest.mark.asyncio
    async def test_vlm_with_tools_no_native_support_injects(self, mock_manager):
        """VLM+tools injects into system message when caps.supports_tools is False."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True
        lm.template_caps = TemplateCaps(
            supports_tools=False, supports_enable_thinking=False
        )

        mock_mx = MagicMock()
        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt with injected tools"

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {},
                },
            }
        ]

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                with patch(
                    "olmlx.engine.inference.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_thread:
                    mock_thread.return_value = "tool response"
                    result = await generate_chat(
                        mock_manager,
                        "qwen3",
                        [{"role": "user", "content": "search"}],
                        tools=tools,
                        stream=False,
                    )

        # Should have called mlx_vlm.apply_chat_template with injected tools in system msg
        vlm_call_args = mock_mlx_vlm.apply_chat_template.call_args
        messages_passed = vlm_call_args[0][2]  # 3rd positional arg = messages
        assert messages_passed[0]["role"] == "system"
        assert "search" in messages_passed[0]["content"]
        assert result["text"] == "tool response"

    @pytest.mark.asyncio
    async def test_vlm_tool_messages_converted_to_tool_responses(self, mock_manager):
        """VLM with uses_tool_responses converts role=tool to tool_responses format."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True
        lm.template_caps = TemplateCaps(
            supports_tools=True,
            supports_enable_thinking=True,
            uses_tool_responses=True,
        )

        mock_mx = MagicMock()
        tools = [
            {
                "type": "function",
                "function": {"name": "Bash", "description": "Run", "parameters": {}},
            }
        ]

        lm.tokenizer.tokenizer = MagicMock()
        lm.tokenizer.tokenizer.apply_chat_template = MagicMock(return_value="prompt")

        messages = [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": {"cmd": "ls"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "file1\nfile2"},
            {"role": "user", "content": "thanks"},
        ]

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.asyncio.to_thread",
                new_callable=AsyncMock,
            ) as mock_thread:
                mock_thread.return_value = "done"
                await generate_chat(
                    mock_manager,
                    "qwen3",
                    messages,
                    tools=tools,
                    stream=False,
                )

        # Verify the messages passed to apply_chat_template have no role=tool
        tpl_call = lm.tokenizer.tokenizer.apply_chat_template
        tpl_call.assert_called_once()
        passed_messages = tpl_call.call_args[0][0]
        assert not any(m.get("role") == "tool" for m in passed_messages)
        # tool_responses should be merged into the assistant message
        tool_resp_msgs = [m for m in passed_messages if "tool_responses" in m]
        assert len(tool_resp_msgs) == 1
        assert tool_resp_msgs[0]["role"] == "assistant"
        assert tool_resp_msgs[0]["tool_responses"][0]["name"] == "Bash"
        assert tool_resp_msgs[0]["tool_responses"][0]["response"] == "file1\nfile2"


class TestGenerateEmbeddings:
    def _setup_tokenizer(self, mock_manager):
        """Set tokenizer.encode to return real ints so mx.array works."""
        tok = mock_manager._loaded["qwen3:latest"].tokenizer
        # Remove .tokenizer sub-attr to avoid double-unwrap
        if hasattr(tok, "tokenizer"):
            del tok.tokenizer
        tok.encode = MagicMock(return_value=[1, 2, 3])
        return tok

    @pytest.mark.asyncio
    async def test_with_embed_layer_3d(self, mock_manager):
        import mlx.core as mx

        self._setup_tokenizer(mock_manager)

        model = mock_manager._loaded["qwen3:latest"].model
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock(return_value=mx.zeros((1, 3, 4)))

        result = await generate_embeddings(mock_manager, "qwen3", ["hello"])
        assert len(result) == 1
        assert len(result[0]) == 4

    @pytest.mark.asyncio
    async def test_without_embed_layer_2d(self, mock_manager):
        import mlx.core as mx

        self._setup_tokenizer(mock_manager)

        model = mock_manager._loaded["qwen3:latest"].model
        model.model = MagicMock(spec=[])  # no embed_tokens attribute
        model.return_value = mx.zeros((3, 4))

        result = await generate_embeddings(mock_manager, "qwen3", ["hello"])
        assert len(result) == 1
        assert len(result[0]) == 4

    @pytest.mark.asyncio
    async def test_1d_embedding(self, mock_manager):
        import mlx.core as mx

        self._setup_tokenizer(mock_manager)

        model = mock_manager._loaded["qwen3:latest"].model
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock(return_value=mx.zeros((4,)))

        result = await generate_embeddings(mock_manager, "qwen3", ["hello"])
        assert len(result) == 1
        assert len(result[0]) == 4

    @pytest.mark.asyncio
    async def test_multiple_texts(self, mock_manager):
        import mlx.core as mx

        self._setup_tokenizer(mock_manager)

        model = mock_manager._loaded["qwen3:latest"].model
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock(return_value=mx.zeros((1, 3, 4)))

        result = await generate_embeddings(mock_manager, "qwen3", ["hello", "world"])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_sync_mode_none_still_syncs_metal(self, mock_manager):
        """With lm.sync_mode='none', the lock-boundary sync is skipped, but
        the inline load-bearing mx.synchronize() at the tail of
        generate_embeddings must still run — it's the only Metal barrier
        before the inference lock is released. Guards against a future
        refactor that wraps that call in a `sync_mode != "none"` check.
        """
        import mlx.core as mx

        self._setup_tokenizer(mock_manager)

        lm = mock_manager._loaded["qwen3:latest"]
        lm.sync_mode = "none"
        model = lm.model
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock(return_value=mx.zeros((1, 3, 4)))

        with patch.object(_inf_mod.mx, "synchronize") as mock_sync:
            result = await generate_embeddings(mock_manager, "qwen3", ["hello"])
        assert len(result) == 1
        # Under sync_mode="none" the lock-boundary _lock_boundary_sync() calls
        # are no-ops (they return before calling mx.synchronize), so any
        # synchronize() we observe here must come from the inline
        # load-bearing fallback at the tail of generate_embeddings. Use
        # >= 1 rather than == N — the meaningful invariant is "at least one
        # Metal barrier fired", not an exact count that future defensive
        # syncs added elsewhere in the call chain would silently break.
        assert mock_sync.call_count >= 1


class TestStreamCancellationHoldsLock:
    @pytest.mark.asyncio
    async def test_lock_held_during_drain(self, mock_manager):
        """Lock should stay held until drain_and_join completes."""
        mock_mx = MagicMock()

        tokens = [
            StreamToken(
                text="a",
                token=1,
                prompt_tokens=5,
                generation_tokens=1,
                prompt_tps=100.0,
                generation_tps=50.0,
            ),
        ]

        lock_held_during_drain = None

        mock_stream = MagicMock(spec=CancellableStream)
        mock_stream._thread = None  # No thread — normal completion path
        token_iter = iter(tokens)

        async def anext_impl():
            try:
                return next(token_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        async def spy_drain():
            nonlocal lock_held_during_drain
            lock_held_during_drain = _inference_lock.locked()

        mock_stream.drain_and_join = spy_drain

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.async_mlx_stream", return_value=mock_stream
            ):
                gen = await generate_completion(
                    mock_manager,
                    "qwen3",
                    "Hello",
                    stream=True,
                )
                chunks = []
                async for chunk in gen:
                    chunks.append(chunk)

        assert lock_held_during_drain is True


class TestGenerateEmbeddingsAcquiresLock:
    @pytest.mark.asyncio
    async def test_embeddings_blocks_when_lock_held(self, mock_manager):
        """generate_embeddings should wait for the inference lock."""
        import mlx.core as mx

        tok = mock_manager._loaded["qwen3:latest"].tokenizer
        if hasattr(tok, "tokenizer"):
            del tok.tokenizer
        tok.encode = MagicMock(return_value=[1, 2, 3])

        model = mock_manager._loaded["qwen3:latest"].model
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock(return_value=mx.zeros((1, 3, 4)))

        order = []

        async with _inference_lock:
            # Start embeddings in background — it should block
            task = asyncio.create_task(
                generate_embeddings(mock_manager, "qwen3", ["hello"])
            )
            await asyncio.sleep(0.05)
            assert not task.done(), "embeddings should be blocked by lock"
            order.append("lock_released")

        # Now it should complete
        result = await task
        order.append("embeddings_done")
        assert order == ["lock_released", "embeddings_done"]
        assert len(result) == 1


class TestSafeSync:
    def test_success(self):
        """_safe_sync() should sync default stream and any resolved generation streams."""
        mock_stream = MagicMock()
        with (
            patch("olmlx.engine.inference.mx") as mock_mx,
            patch("olmlx.engine.inference._generation_streams", [mock_stream]),
        ):
            _safe_sync()
            # Default stream + 1 generation stream
            assert mock_mx.synchronize.call_count == 2
            mock_mx.synchronize.assert_any_call(mock_stream)

    def test_suppresses_exception(self):
        """_safe_sync() should suppress exceptions from mx.synchronize()."""
        with patch("olmlx.engine.inference.mx") as mock_mx:
            mock_mx.synchronize.side_effect = RuntimeError("Metal error")
            _safe_sync()  # should not raise


class TestLockBoundarySync:
    def test_full_matches_safe_sync(self):
        """'full' mode: sync default + every generation stream."""
        mock_stream = MagicMock()
        with (
            patch("olmlx.engine.inference.mx") as mock_mx,
            patch("olmlx.engine.inference._generation_streams", [mock_stream]),
        ):
            _lock_boundary_sync("full")
            assert mock_mx.synchronize.call_count == 2
            mock_mx.synchronize.assert_any_call(mock_stream)

    def test_minimal_skips_generation_streams(self):
        """'minimal' mode: sync default stream only, skip generation streams."""
        mock_stream = MagicMock()
        with (
            patch("olmlx.engine.inference.mx") as mock_mx,
            patch("olmlx.engine.inference._generation_streams", [mock_stream]),
        ):
            _lock_boundary_sync("minimal")
            assert mock_mx.synchronize.call_count == 1
            mock_mx.synchronize.assert_called_once_with()

    def test_none_skips_all_sync(self):
        """'none' mode: skip sync entirely at lock boundaries."""
        mock_stream = MagicMock()
        with (
            patch("olmlx.engine.inference.mx") as mock_mx,
            patch("olmlx.engine.inference._generation_streams", [mock_stream]),
        ):
            _lock_boundary_sync("none")
            assert mock_mx.synchronize.call_count == 0

    def test_null_mode_falls_back_to_global_setting(self):
        """mode=None (Python None sentinel, not the "none" SyncMode string)
        should resolve to settings.sync_mode. The two are opposite: None
        inherits the global, "none" skips all sync."""
        mock_stream = MagicMock()
        with (
            patch("olmlx.engine.inference.mx") as mock_mx,
            patch("olmlx.engine.inference._generation_streams", [mock_stream]),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.sync_mode = "none"
            _lock_boundary_sync(None)
            assert mock_mx.synchronize.call_count == 0

            mock_mx.reset_mock()
            mock_settings.sync_mode = "minimal"
            _lock_boundary_sync(None)
            assert mock_mx.synchronize.call_count == 1

    def test_suppresses_exceptions(self):
        """Exceptions from mx.synchronize must not propagate — covers both the
        default-stream and generation-stream suppression paths."""
        mock_stream = MagicMock()
        with (
            patch("olmlx.engine.inference.mx") as mock_mx,
            patch("olmlx.engine.inference._generation_streams", [mock_stream]),
        ):
            mock_mx.synchronize.side_effect = RuntimeError("Metal error")
            _lock_boundary_sync("full")  # exercises default + generation stream loop
            _lock_boundary_sync("minimal")  # exercises default only
            # Exactly: 1 default + 1 generation (full) + 1 default (minimal) = 3
            assert mock_mx.synchronize.call_count == 3

    def test_unknown_mode_raises(self):
        """Unknown modes must raise ValueError instead of silently falling
        through to a default — prevents silent drift if a new mode is added
        to SyncMode without updating this helper."""
        with patch("olmlx.engine.inference.mx"):
            with pytest.raises(ValueError, match="Unknown sync_mode"):
                _lock_boundary_sync("sometimes")  # type: ignore[arg-type]


class TestInferenceLocked:
    @pytest.mark.asyncio
    async def test_acquires_and_releases_lock(self):
        """_inference_locked() should acquire lock on entry and release on exit."""
        with patch("olmlx.engine.inference.mx"):
            assert not _inference_lock.locked()
            async with _inference_locked():
                assert _inference_lock.locked()
            assert not _inference_lock.locked()

    @pytest.mark.asyncio
    async def test_releases_lock_on_exception(self):
        """_inference_locked() should release lock even if body raises."""
        with patch("olmlx.engine.inference.mx"):
            with pytest.raises(ValueError):
                async with _inference_locked():
                    raise ValueError("test error")
            assert not _inference_lock.locked()

    @pytest.mark.asyncio
    async def test_syncs_on_entry_and_exit(self):
        """_inference_locked() should call mx.synchronize on entry and exit."""
        with patch("olmlx.engine.inference.mx") as mock_mx:
            async with _inference_locked():
                pass
            # Called at least twice: entry sync + exit sync
            assert mock_mx.synchronize.call_count >= 2

    @pytest.mark.asyncio
    async def test_sync_mode_none_skips_lock_boundary_sync(self):
        """sync_mode='none' should skip both entry and exit lock-boundary syncs."""
        with patch("olmlx.engine.inference.mx") as mock_mx:
            async with _inference_locked(sync_mode="none"):
                pass
            assert mock_mx.synchronize.call_count == 0

    @pytest.mark.asyncio
    async def test_sync_mode_minimal_syncs_default_only(self):
        """sync_mode='minimal' should sync default stream only on entry+exit."""
        mock_stream = MagicMock()
        with (
            patch("olmlx.engine.inference.mx") as mock_mx,
            patch("olmlx.engine.inference._generation_streams", [mock_stream]),
        ):
            async with _inference_locked(sync_mode="minimal"):
                pass
            # 1 default sync on entry + 1 default sync on exit, no generation streams
            assert mock_mx.synchronize.call_count == 2
            for call in mock_mx.synchronize.call_args_list:
                assert call.args == ()

    @pytest.mark.asyncio
    async def test_lock_released_if_entry_sync_raises(self):
        """If entry _lock_boundary_sync raises, the inference lock must be released."""
        assert not _inference_lock.locked()
        with patch(
            "olmlx.engine.inference._lock_boundary_sync",
            side_effect=ValueError("boom"),
        ):
            with pytest.raises(ValueError, match="boom"):
                async with _inference_locked():
                    pass
        assert not _inference_lock.locked()

    @pytest.mark.asyncio
    async def test_exit_sync_raise_is_suppressed_and_safe_sync_runs(self):
        """If exit _lock_boundary_sync raises (unknown mode), the error must
        be caught, _safe_sync() must run as a fail-safe fallback, and the
        lock must be released. No exception should propagate from an
        otherwise-successful body."""
        assert not _inference_lock.locked()
        calls = {"n": 0}

        def side_effect(*_a, **_kw):
            calls["n"] += 1
            if calls["n"] == 2:  # only raise on exit sync
                raise ValueError("exit boom")

        with (
            patch(
                "olmlx.engine.inference._lock_boundary_sync", side_effect=side_effect
            ),
            patch("olmlx.engine.inference._safe_sync") as mock_safe_sync,
        ):
            # No exception should escape — the exit-sync raise is caught.
            async with _inference_locked():
                pass
        assert not _inference_lock.locked()
        mock_safe_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_exit_sync_raise_does_not_mask_body_exception(self):
        """If both the body and exit _lock_boundary_sync raise, the body's
        exception must propagate (not the sync ValueError). Python's
        default finally-semantics would replace the body exception with
        the sync one — this test guards against that regression."""
        assert not _inference_lock.locked()
        calls = {"n": 0}

        def side_effect(*_a, **_kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ValueError("exit boom")

        class BodyError(RuntimeError):
            pass

        with (
            patch(
                "olmlx.engine.inference._lock_boundary_sync", side_effect=side_effect
            ),
            patch("olmlx.engine.inference._safe_sync"),
        ):
            with pytest.raises(BodyError, match="body failed"):
                async with _inference_locked():
                    raise BodyError("body failed")
        assert not _inference_lock.locked()


class TestInferenceLockedWaitsDeferredCleanup:
    @pytest.mark.asyncio
    async def test_waits_for_deferred_cleanup_pre_lock(self):
        """_inference_locked() should wait for deferred cleanup instead of raising."""
        completed = False

        async def cleanup():
            nonlocal completed
            await asyncio.sleep(0.05)
            completed = True

        _inf_mod._deferred_cleanup_tasks[asyncio.get_running_loop()] = (
            asyncio.create_task(cleanup())
        )
        with patch("olmlx.engine.inference.mx"):
            async with _inference_locked():
                pass
        assert completed
        _inf_mod._deferred_cleanup_tasks.pop(asyncio.get_running_loop(), None)

    @pytest.mark.asyncio
    async def test_waits_for_deferred_cleanup_post_lock(self):
        """_inference_locked() should wait for deferred cleanup created during lock acquisition (TOCTOU)."""
        cleanup_done = asyncio.Event()

        async def cleanup():
            await asyncio.sleep(0.05)
            cleanup_done.set()

        # Wrap _acquire_inference_lock to inject a deferred task after lock
        # is acquired but before the post-lock _await_deferred_cleanup runs.
        original_acquire = _inf_mod._acquire_inference_lock

        async def acquire_then_inject(timeout_override=None):
            await original_acquire(timeout_override)
            _inf_mod._deferred_cleanup_tasks[asyncio.get_running_loop()] = (
                asyncio.create_task(cleanup())
            )

        with patch("olmlx.engine.inference.mx"):
            _inf_mod._deferred_cleanup_tasks.pop(asyncio.get_running_loop(), None)
            with patch(
                "olmlx.engine.inference._acquire_inference_lock",
                side_effect=acquire_then_inject,
            ):
                async with _inference_locked():
                    pass
            assert cleanup_done.is_set()
        _inf_mod._deferred_cleanup_tasks.pop(asyncio.get_running_loop(), None)


class TestAcquireInferenceLockTimeoutOverride:
    @pytest.mark.asyncio
    async def test_timeout_override_used_instead_of_global(self):
        """timeout_override should take precedence over settings.inference_queue_timeout."""
        fresh_lock = asyncio.Lock()
        _inf_mod._deferred_cleanup_tasks.pop(asyncio.get_running_loop(), None)
        with patch("olmlx.engine.inference._inference_lock", fresh_lock):
            # Hold the lock so acquire will block
            await fresh_lock.acquire()
            # Use a very short timeout override
            with pytest.raises(ServerBusyError, match="timeout after 0.05s"):
                await _acquire_inference_lock(timeout_override=0.05)
            fresh_lock.release()

    @pytest.mark.asyncio
    async def test_timeout_override_none_falls_through_to_global(self):
        """When timeout_override is None, global setting is used."""
        fresh_lock = asyncio.Lock()
        _inf_mod._deferred_cleanup_tasks.pop(asyncio.get_running_loop(), None)
        with (
            patch("olmlx.engine.inference._inference_lock", fresh_lock),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.inference_queue_timeout = 0.05
            await fresh_lock.acquire()
            with pytest.raises(ServerBusyError, match="timeout after 0.05s"):
                await _acquire_inference_lock(timeout_override=None)
            fresh_lock.release()

    @pytest.mark.asyncio
    async def test_inference_locked_passes_timeout_override(self):
        """_inference_locked() should pass timeout_override to _acquire_inference_lock."""
        fresh_lock = asyncio.Lock()
        _inf_mod._deferred_cleanup_tasks.pop(asyncio.get_running_loop(), None)
        with (
            patch("olmlx.engine.inference._inference_lock", fresh_lock),
            patch("olmlx.engine.inference.mx"),
        ):
            await fresh_lock.acquire()
            with pytest.raises(ServerBusyError):
                async with _inference_locked(timeout_override=0.05):
                    pass
            fresh_lock.release()


class TestQueueDepth:
    @pytest.mark.asyncio
    async def test_queue_depth_incremented_during_lock_wait(self):
        """Queue depth should be logged when requests wait for the lock."""
        # Use a fresh lock to avoid event loop binding issues
        fresh_lock = asyncio.Lock()
        _inf_mod._queue_depth = 0
        _inf_mod._deferred_cleanup_tasks.pop(asyncio.get_running_loop(), None)
        with (
            patch("olmlx.engine.inference.mx"),
            patch("olmlx.engine.inference._inference_lock", fresh_lock),
        ):
            # Hold the lock
            await fresh_lock.acquire()

            async def second_request():
                async with _inference_locked():
                    pass

            task = asyncio.create_task(second_request())
            await asyncio.sleep(0.05)
            # Release lock so second request can proceed
            fresh_lock.release()
            await task

        assert _inf_mod._queue_depth == 0  # should be back to 0


class TestInferenceQueueTimeout:
    @pytest.mark.asyncio
    async def test_lock_acquisition_times_out(self):
        """_inference_locked() should raise ServerBusyError when queue timeout expires."""
        from olmlx.engine.inference import ServerBusyError

        fresh_lock = asyncio.Lock()
        _inf_mod._queue_depth = 0
        _inf_mod._deferred_cleanup_tasks.pop(asyncio.get_running_loop(), None)
        with (
            patch("olmlx.engine.inference.mx"),
            patch("olmlx.engine.inference._inference_lock", fresh_lock),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.inference_queue_timeout = 0.05
            # Hold the lock
            await fresh_lock.acquire()
            with pytest.raises(ServerBusyError, match="queue timeout"):
                async with _inference_locked():
                    pass
            fresh_lock.release()


class TestStreamCompletionQueueTimeout:
    @pytest.mark.asyncio
    async def test_streaming_lock_acquisition_times_out(self, mock_manager):
        """_stream_completion() should raise ServerBusyError when queue timeout expires."""
        from olmlx.engine.inference import ServerBusyError, _stream_completion

        fresh_lock = asyncio.Lock()
        _inf_mod._queue_depth = 0
        _inf_mod._deferred_cleanup_tasks.pop(asyncio.get_running_loop(), None)
        with (
            patch("olmlx.engine.inference.mx"),
            patch("olmlx.engine.inference._inference_lock", fresh_lock),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.inference_queue_timeout = 0.05
            await fresh_lock.acquire()
            lm = mock_manager._loaded["qwen3:latest"]
            with pytest.raises(ServerBusyError, match="queue timeout"):
                gen = _stream_completion(lm, "prompt", 100, {}, TimingStats())
                async for _ in gen:
                    pass
            fresh_lock.release()


class TestStreamCompletionFallbackJoinLogging:
    @pytest.mark.asyncio
    async def test_fallback_join_defers_cleanup_when_thread_alive(
        self, mock_manager, caplog
    ):
        """When shield is interrupted and fallback join times out, should defer cleanup."""
        mock_mx = MagicMock()

        # Create a mock stream that simulates a stuck thread
        mock_stream = MagicMock(spec=CancellableStream)
        mock_thread = MagicMock()
        # Thread is alive during cleanup, then exits when deferred task polls
        alive_calls = [True, True, True, False]
        mock_thread.is_alive.side_effect = lambda: (
            alive_calls.pop(0) if alive_calls else False
        )
        mock_thread.join = MagicMock()  # join returns but thread still "alive"
        mock_stream._thread = mock_thread

        token_iter = iter(
            [
                StreamToken(
                    text="a",
                    token=1,
                    prompt_tokens=5,
                    generation_tokens=1,
                    prompt_tps=100.0,
                    generation_tps=50.0,
                ),
            ]
        )

        async def anext_impl():
            try:
                return next(token_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        # Make drain_and_join raise CancelledError to trigger fallback path
        async def drain_raises():
            raise asyncio.CancelledError()

        mock_stream.drain_and_join = drain_raises

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ):
                with caplog.at_level(logging.WARNING, logger="olmlx.engine.inference"):
                    gen = await generate_completion(
                        mock_manager,
                        "qwen3",
                        "Hello",
                        stream=True,
                    )
                    chunks = []
                    async for chunk in gen:
                        chunks.append(chunk)

        assert any(
            "deferring Metal sync" in record.message for record in caplog.records
        )
        # Wait for deferred cleanup task to complete and release the lock
        task = _inf_mod._deferred_cleanup_tasks.get(asyncio.get_running_loop())
        assert task is not None, "expected a running cleanup task"
        await task
        assert not _inference_lock.locked(), (
            "_inference_lock must be released by deferred cleanup task"
        )


class TestDeferredInferenceCleanup:
    @pytest.mark.asyncio
    async def test_safe_sync_called_after_thread_exits(self):
        """_safe_sync should be called once the thread exits."""
        mock_stream = MagicMock()
        mock_thread = MagicMock()
        # Thread alive on first check (while loop), dead on re-check and finally guard
        mock_thread.is_alive.side_effect = [True, False, False]
        mock_thread.join = MagicMock()
        mock_stream._thread = mock_thread

        # Manually acquire the lock to simulate _stream_completion holding it
        await _inference_lock.acquire()

        with patch("olmlx.engine.inference._safe_sync") as mock_safe_sync:
            await _schedule_deferred_inference_cleanup(mock_stream)
            task = _inf_mod._deferred_cleanup_tasks.get(asyncio.get_running_loop())
            assert task is not None, "expected a running cleanup task"
            await task

        # _safe_sync should have been called (after thread exited)
        mock_safe_sync.assert_called_once()
        assert not _inference_lock.locked()

    @pytest.mark.asyncio
    async def test_lock_released_after_thread_exits(self):
        """The deferred task must release the lock once the thread exits."""
        mock_stream = MagicMock()
        mock_thread = MagicMock()
        # Thread alive on first check, dead on re-check and finally guard
        mock_thread.is_alive.side_effect = [True, False, False]
        mock_thread.join = MagicMock()
        mock_stream._thread = mock_thread

        await _inference_lock.acquire()
        assert _inference_lock.locked()

        with patch("olmlx.engine.inference._safe_sync"):
            await _schedule_deferred_inference_cleanup(mock_stream)
            # Task is running — lock should still be held initially
            assert _inference_lock.locked()
            # Wait for deferred task to complete
            task = _inf_mod._deferred_cleanup_tasks.get(asyncio.get_running_loop())
            assert task is not None, "expected a running cleanup task"
            await task

        assert not _inference_lock.locked()


class TestAwaitDeferredCleanup:
    @pytest.mark.asyncio
    async def test_returns_immediately_when_no_task(self):
        """Should return immediately when no deferred cleanup task exists."""
        from olmlx.engine.inference import _await_deferred_cleanup

        _inf_mod._deferred_cleanup_tasks.pop(asyncio.get_running_loop(), None)
        await _await_deferred_cleanup()  # should not raise

    @pytest.mark.asyncio
    async def test_returns_immediately_when_task_done(self):
        """Should return immediately when deferred cleanup task is already done."""
        from olmlx.engine.inference import _await_deferred_cleanup

        done_task = asyncio.create_task(asyncio.sleep(0))
        await done_task  # let it finish
        _inf_mod._deferred_cleanup_tasks[asyncio.get_running_loop()] = done_task
        await _await_deferred_cleanup()  # should not raise
        _inf_mod._deferred_cleanup_tasks.pop(asyncio.get_running_loop(), None)

    @pytest.mark.asyncio
    async def test_waits_for_running_task(self):
        """Should wait for a running deferred cleanup task to complete."""
        from olmlx.engine.inference import _await_deferred_cleanup

        completed = False

        async def slow_cleanup():
            nonlocal completed
            await asyncio.sleep(0.05)
            completed = True

        _inf_mod._deferred_cleanup_tasks[asyncio.get_running_loop()] = (
            asyncio.create_task(slow_cleanup())
        )
        await _await_deferred_cleanup()
        assert completed
        _inf_mod._deferred_cleanup_tasks.pop(asyncio.get_running_loop(), None)

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):
        """Should raise ServerBusyError if cleanup doesn't complete within timeout."""
        import contextlib

        from olmlx.engine.inference import (
            ServerBusyError,
            _await_deferred_cleanup,
        )

        async def stuck_cleanup():
            await asyncio.sleep(999)

        loop = asyncio.get_running_loop()
        stuck_task = asyncio.create_task(stuck_cleanup())
        _inf_mod._deferred_cleanup_tasks[loop] = stuck_task
        try:
            with patch("olmlx.engine.inference._DEFERRED_WAIT_TIMEOUT", 0.05):
                with pytest.raises(ServerBusyError, match="did not complete"):
                    await _await_deferred_cleanup()
        finally:
            stuck_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stuck_task
            _inf_mod._deferred_cleanup_tasks.pop(loop, None)

    @pytest.mark.asyncio
    async def test_does_not_cancel_cleanup_task(self):
        """asyncio.shield should prevent wait_for from cancelling the cleanup task."""
        import contextlib

        from olmlx.engine.inference import _await_deferred_cleanup

        task_was_cancelled = False

        async def slow_cleanup():
            nonlocal task_was_cancelled
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                task_was_cancelled = True
                raise

        loop = asyncio.get_running_loop()
        slow_task = asyncio.create_task(slow_cleanup())
        _inf_mod._deferred_cleanup_tasks[loop] = slow_task
        try:
            with patch("olmlx.engine.inference._DEFERRED_WAIT_TIMEOUT", 0.05):
                with pytest.raises(Exception):
                    await _await_deferred_cleanup()
            # The task should NOT have been cancelled by shield
            assert not task_was_cancelled
        finally:
            slow_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await slow_task
            _inf_mod._deferred_cleanup_tasks.pop(loop, None)


class TestEstimateKvCacheBytes:
    """Tests for estimate_kv_cache_bytes()."""

    def _make_model_args(
        self,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        head_dim=None,
    ):
        args = MagicMock(spec=[])
        args.num_hidden_layers = num_hidden_layers
        args.num_attention_heads = num_attention_heads
        args.num_key_value_heads = num_key_value_heads
        args.hidden_size = hidden_size
        if head_dim is not None:
            args.head_dim = head_dim
        return args

    def _make_model(self, **kwargs):
        model = MagicMock(spec=[])
        model.args = self._make_model_args(**kwargs)
        return model

    def test_basic_estimate(self):
        """KV cache bytes = layers * 2 * kv_heads * head_dim * tokens * 2 * MEMORY_SAFETY_FACTOR."""
        model = self._make_model(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
        )
        # head_dim = 4096 / 32 = 128
        # raw = 32 * 2 * 8 * 128 * 1000 * 2 = 131_072_000
        # expected = int(131_072_000 * 1.3) = 170_393_600
        result = estimate_kv_cache_bytes(model, 1000)
        assert result == int(131_072_000 * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_zero_tokens(self):
        model = self._make_model()
        assert estimate_kv_cache_bytes(model, 0) == 0

    def test_gqa_fallback_to_mha(self):
        """When num_key_value_heads is missing, fall back to num_attention_heads."""
        model = MagicMock()
        model.args = MagicMock(spec=[])
        model.args.num_hidden_layers = 32
        model.args.num_attention_heads = 32
        model.args.hidden_size = 4096
        # Delete num_key_value_heads so hasattr returns False
        del model.args.num_key_value_heads
        # head_dim = 128, kv_heads = 32 (fallback)
        # raw = 32 * 2 * 32 * 128 * 100 * 2 = 52_428_800
        result = estimate_kv_cache_bytes(model, 100)
        assert result == int(52_428_800 * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_large_prompt(self):
        """Test with a 22k token prompt (the crash scenario)."""
        model = self._make_model(
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            hidden_size=3584,
        )
        # head_dim = 3584 / 28 = 128
        # raw = 28 * 2 * 4 * 128 * 22000 * 2 = 1_261_568_000 (~1.2 GB)
        result = estimate_kv_cache_bytes(model, 22000)
        assert result == int(1_261_568_000 * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_explicit_head_dim(self):
        """When model.args.head_dim exists, use it instead of hidden_size // num_heads."""
        model = self._make_model(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            head_dim=256,
        )
        # raw = 32 * 2 * 8 * 256 * 100 * 2 = 26_214_400
        result = estimate_kv_cache_bytes(model, 100)
        assert result == int(26_214_400 * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_vlm_fallback_to_language_model_args(self):
        """Falls back to model.language_model.args for VLM models."""
        model = MagicMock(spec=[])
        model.language_model = MagicMock(spec=[])
        model.language_model.args = self._make_model_args(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
        )
        result = estimate_kv_cache_bytes(model, 1000)
        assert result == int(131_072_000 * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_vlm_introspects_language_model_layers(self):
        """VLM layer introspection uses language_model.model.layers, not model.model."""
        model = MagicMock(spec=[])
        model.language_model = MagicMock(spec=[])
        model.language_model.args = self._make_model_args(
            num_hidden_layers=32,
            num_attention_heads=32,
            hidden_size=4096,
        )
        # Remove num_key_value_heads from args to force introspection
        del model.language_model.args.num_key_value_heads

        # language_model.model.layers has the real attention layers
        layers = []
        for _ in range(32):
            layer = MagicMock()
            layer.self_attn = MagicMock()
            layer.self_attn.n_kv_heads = 4
            layers.append(layer)
        model.language_model.model = MagicMock()
        model.language_model.model.layers = layers

        # Should use introspected kv_heads=4, not fallback to num_attention_heads=32
        result = estimate_kv_cache_bytes(model, 1000)
        expected_raw = 32 * 2 * 4 * 128 * 1000 * 2
        assert result == int(expected_raw * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_vlm_fallback_to_config(self):
        """Falls back to model.config or model.language_model.config when args missing."""
        model = MagicMock(spec=[])
        model.language_model = MagicMock(spec=[])
        model.language_model.config = self._make_model_args(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
        )
        result = estimate_kv_cache_bytes(model, 1000)
        assert result == int(131_072_000 * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_vlm_text_config_wrapper_prefers_language_model_args(self):
        """When model.args is a text_config wrapper (Qwen3.5 MoE pattern), prefer language_model.args.

        Qwen3_5_MoE's top-level ModelArgs only has ``model_type`` and ``text_config``;
        the real attention config lives at ``model.language_model.args``. The estimator
        must not blindly use the wrapper, which would AttributeError on num_attention_heads.
        """
        model = MagicMock(spec=[])
        # Wrapper args (mimics Qwen3_5_MoE.ModelArgs): has model_type + text_config,
        # but NOT num_attention_heads / num_hidden_layers.
        wrapper = MagicMock(spec=[])
        wrapper.model_type = "qwen3_5_moe"
        wrapper.text_config = {"num_hidden_layers": 64}
        model.args = wrapper

        model.language_model = MagicMock(spec=[])
        model.language_model.args = self._make_model_args(
            num_hidden_layers=64,
            num_attention_heads=32,
            num_key_value_heads=4,
            hidden_size=5120,
            head_dim=256,
        )
        # raw = 64 * 2 * 4 * 256 * 1000 * 2 = 262_144_000
        result = estimate_kv_cache_bytes(model, 1000)
        assert result == int(262_144_000 * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_raises_when_no_args_found(self):
        """Raises AttributeError when model has no discoverable args or config."""
        model = MagicMock(spec=[])
        with pytest.raises(AttributeError, match="no 'args'"):
            estimate_kv_cache_bytes(model, 1000)

    def test_wrapper_without_language_model_raises(self):
        """Wrapper args + no language_model → explicit error (not opaque later crash)."""
        model = MagicMock(spec=[])
        wrapper = MagicMock(spec=[])
        wrapper.text_config = {"num_hidden_layers": 64}
        model.args = wrapper
        with pytest.raises(AttributeError, match="text_config wrapper"):
            estimate_kv_cache_bytes(model, 1000)

    def test_wrapper_with_empty_language_model_raises(self):
        """Wrapper args + language_model without args/config → explicit error."""
        model = MagicMock(spec=[])
        wrapper = MagicMock(spec=[])
        wrapper.text_config = {"num_hidden_layers": 64}
        model.args = wrapper
        model.language_model = MagicMock(spec=[])
        with pytest.raises(AttributeError, match="text_config wrapper"):
            estimate_kv_cache_bytes(model, 1000)

    def test_nas_model_with_per_layer_variable_attention(self):
        """NAS models (nemotron-nas) have per-layer variable attention.

        Only layers with actual attention contribute to KV cache, and
        the KV head count comes from each layer's n_kv_heads, not from
        num_attention_heads (which would be a huge overestimate).
        """
        # Simulate nemotron-nas: 80 layers total, 49 with attention (8 kv_heads),
        # 31 with no_op (self_attn=None). No num_key_value_heads in args.
        args = MagicMock(spec=[])
        args.num_hidden_layers = 80
        args.num_attention_heads = 64
        args.hidden_size = 8192
        # num_key_value_heads is NOT on args (NAS model)
        # head_dim is NOT on args

        model = MagicMock(spec=[])
        model.args = args

        # Build layers: 49 with attention, 31 without
        layers = []
        for i in range(80):
            layer = MagicMock()
            if i % 80 < 49:  # first 49 layers have attention
                attn = MagicMock()
                attn.n_kv_heads = 8
                layer.self_attn = attn
            else:
                layer.self_attn = None
            layers.append(layer)
        model.model = MagicMock()
        model.model.layers = layers

        # head_dim = 8192 / 64 = 128
        # Correct: 49 layers * 2 * 8 kv_heads * 128 head_dim * 1000 tokens * 2 bytes
        # = 49 * 2 * 8 * 128 * 1000 * 2 = 200_704_000
        result = estimate_kv_cache_bytes(model, 1000)
        expected_raw = 49 * 2 * 8 * 128 * 1000 * 2
        assert result == int(expected_raw * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_nas_model_no_layers_introspection_fallback(self):
        """When model has no introspectable layers, fall back to args-based estimate."""
        args = MagicMock(spec=[])
        args.num_hidden_layers = 80
        args.num_attention_heads = 64
        args.hidden_size = 8192
        # No num_key_value_heads — falls back to num_attention_heads

        model = MagicMock(spec=[])
        model.args = args

        # No model.model.layers available
        del model.model

        # Falls back to: 80 * 2 * 64 * 128 * 1000 * 2
        result = estimate_kv_cache_bytes(model, 1000)
        expected_raw = 80 * 2 * 64 * 128 * 1000 * 2
        assert result == int(expected_raw * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_nas_model_wrong_attn_attr_name_falls_back_to_args(self):
        """If layers exist but use a different attr name than self_attn, fall back to args."""
        args = MagicMock(spec=[])
        args.num_hidden_layers = 32
        args.num_attention_heads = 32
        args.num_key_value_heads = 8
        args.hidden_size = 4096

        model = MagicMock(spec=[])
        model.args = args

        layers = []
        for _ in range(32):
            layer = MagicMock(spec=["attention"])  # has "attention", not "self_attn"
            layer.attention = MagicMock()
            layer.attention.n_kv_heads = 8
            layers.append(layer)
        model.model = MagicMock()
        model.model.layers = layers

        # Should fall back to args-based estimate, not return 0
        result = estimate_kv_cache_bytes(model, 1000)
        expected_raw = 32 * 2 * 8 * 128 * 1000 * 2
        assert result == int(expected_raw * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_mla_model_uses_compressed_kv_dimensions(self):
        """MLA models (DeepSeek V3) store compressed KV: kv_lora_rank + qk_rope_head_dim per layer.

        The KV cache should NOT use num_key_value_heads * head_dim (which is the
        uncompressed attention dimension), but instead kv_lora_rank + qk_rope_head_dim.
        """
        args = MagicMock(spec=[])
        args.num_hidden_layers = 61
        args.num_attention_heads = 128
        args.num_key_value_heads = 128
        args.hidden_size = 7168
        args.kv_lora_rank = 512
        args.qk_rope_head_dim = 64
        model = MagicMock(spec=[])
        model.args = args

        result = estimate_kv_cache_bytes(model, 1000)

        # MLA cache per layer per token: (kv_lora_rank + qk_rope_head_dim) * 2 bytes
        # = (512 + 64) * 2 = 1152 bytes  (keys=compressed_kv, values=k_pe, 1 "head" each)
        # Total raw = 61 * (512 + 64) * 2 * 1000 * 2 = 140_608_000
        # Note: factor of 2 for K+V cache entries (compressed_kv stored as keys,
        # k_pe stored as values — both have 1 effective head).
        expected_raw = 61 * 2 * (512 + 64) * 1000 * 2
        assert result == int(expected_raw * _inf_mod.MEMORY_SAFETY_FACTOR)

        # Verify it's dramatically less than the naive MHA estimate
        naive_head_dim = 7168 // 128  # = 56
        naive_raw = 61 * 2 * 128 * naive_head_dim * 1000 * 2
        assert result < naive_raw  # MLA should be ~25x smaller

    def test_gemma4_hybrid_attention_with_sliding_window(self):
        """Gemma 4-style hybrid model: sliding-window + full attention with per-layer head_dim.

        Gemma 4 31b has:
          - 60 layers total
          - 50 sliding-window layers (n_kv=16, head_dim=256, capped at 1024 tokens)
          - 10 full-attention layers (n_kv=4, head_dim=512, uncapped)

        Naive estimation (uniform layers, no window cap) overestimates by ~8x
        for long prompts and triggers spurious MemoryError 503s.
        """
        args = MagicMock(spec=[])
        args.num_hidden_layers = 60
        args.num_attention_heads = 32
        args.num_key_value_heads = 16
        args.hidden_size = 5376
        args.head_dim = 256
        args.global_head_dim = 512
        args.sliding_window = 1024

        model = MagicMock(spec=[])
        model.args = args

        layers = []
        for i in range(60):
            layer = MagicMock()
            attn = MagicMock()
            # Mimic mlx-vlm gemma4: full-attention layers have larger head_dim
            # but fewer kv_heads.  Layer types alternate (5 sliding, 1 full).
            if i % 6 == 5:
                attn.n_kv_heads = 4
                attn.head_dim = 512
                attn.is_sliding = False
            else:
                attn.n_kv_heads = 16
                attn.head_dim = 256
                attn.is_sliding = True
            layer.self_attn = attn
            layers.append(layer)
        model.model = MagicMock()
        model.model.layers = layers

        num_tokens = 33181
        # Expected per-layer:
        #   sliding: 2 * 16 * 256 * min(33181, 1024) * 2 = 16,777,216 bytes
        #   full:    2 * 4 * 512 * 33181 * 2 = 271,773,696 bytes
        #   sum: 50 * 16,777,216 + 10 * 271,773,696 = 838,860,800 + 2,717,736,960
        #      = 3,556,597,760 bytes raw (≈3.31 GB)
        sliding_per_layer = 2 * 16 * 256 * 1024 * 2
        full_per_layer = 2 * 4 * 512 * num_tokens * 2
        expected_raw = 50 * sliding_per_layer + 10 * full_per_layer

        result = estimate_kv_cache_bytes(model, num_tokens)
        assert result == int(expected_raw * _inf_mod.MEMORY_SAFETY_FACTOR)

        # Sanity check: result should fit in 16 GB.  The naive uniform-layer
        # formula (60 * 2 * 16 * 256 * 33181 * 2 * 1.3) ≈ 38 GB would not.
        assert result < 16 * 1024**3, f"Expected ~4.3 GB, got {result / 1024**3:.1f} GB"

    def test_per_layer_sliding_window_override(self):
        """Per-layer self_attn.sliding_window_size takes precedence over args.sliding_window.

        Defensive test: today no shipping model exposes heterogeneous
        per-layer windows, but the introspection should honour them if a
        future model does.
        """
        args = MagicMock(spec=[])
        args.num_hidden_layers = 4
        args.num_attention_heads = 8
        args.num_key_value_heads = 8
        args.hidden_size = 1024
        args.head_dim = 128
        args.sliding_window = 4096  # global default

        model = MagicMock(spec=[])
        model.args = args

        layers = []
        for _ in range(4):
            layer = MagicMock()
            attn = MagicMock()
            attn.n_kv_heads = 8
            attn.head_dim = 128
            attn.is_sliding = True
            # Per-layer window — overrides the global args.sliding_window
            attn.sliding_window_size = 512
            layer.self_attn = attn
            layers.append(layer)
        model.model = MagicMock()
        model.model.layers = layers

        # 8000 tokens, but per-layer window is 512 (not 4096)
        result = estimate_kv_cache_bytes(model, 8000)
        expected_raw = 4 * 2 * 8 * 128 * 512 * 2
        assert result == int(expected_raw * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_sliding_window_cap_short_prompt(self):
        """When num_tokens < sliding_window, sliding layers should NOT be capped down."""
        args = MagicMock(spec=[])
        args.num_hidden_layers = 4
        args.num_attention_heads = 8
        args.num_key_value_heads = 8
        args.hidden_size = 1024
        args.head_dim = 128
        args.sliding_window = 1024

        model = MagicMock(spec=[])
        model.args = args

        layers = []
        for _ in range(4):
            layer = MagicMock()
            attn = MagicMock()
            attn.n_kv_heads = 8
            attn.head_dim = 128
            attn.is_sliding = True
            layer.self_attn = attn
            layers.append(layer)
        model.model = MagicMock()
        model.model.layers = layers

        # 500 < 1024, so sliding layers use full prompt length
        result = estimate_kv_cache_bytes(model, 500)
        expected_raw = 4 * 2 * 8 * 128 * 500 * 2
        assert result == int(expected_raw * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_turboquant_4bit_reduces_estimate(self):
        """TurboQuant 4-bit KV cache should reduce the memory estimate.

        Normal: head_dim * 2 bytes per head per K or V entry
        TurboQuant 4-bit: head_dim/2 bytes (packed indices) + 4 bytes (norm)
        For head_dim=128: 256 bytes → 68 bytes ≈ 3.76x reduction.
        """
        model = self._make_model(
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            hidden_size=8192,
        )
        fp16_result = estimate_kv_cache_bytes(model, 37000)
        tq4_result = estimate_kv_cache_bytes(
            model, 37000, kv_cache_quant="turboquant:4"
        )
        # TurboQuant 4-bit should be significantly smaller
        assert tq4_result < fp16_result
        # Per-element: fp16 = 256 bytes, tq4 = 68 bytes → ratio ≈ 3.76x
        assert fp16_result / tq4_result == pytest.approx(256 / 68, rel=0.01)

    def test_turboquant_2bit_reduces_estimate(self):
        """TurboQuant 2-bit should compress even more than 4-bit."""
        model = self._make_model(
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            hidden_size=8192,
        )
        fp16_result = estimate_kv_cache_bytes(model, 37000)
        tq2_result = estimate_kv_cache_bytes(
            model, 37000, kv_cache_quant="turboquant:2"
        )
        # Per-element: fp16 = 256 bytes, tq2 = 36 bytes → ratio ≈ 7.1x
        assert tq2_result < fp16_result
        assert fp16_result / tq2_result == pytest.approx(256 / 36, rel=0.01)

    def test_turboquant_none_unchanged(self):
        """Passing kv_cache_quant=None should give the same result as no arg."""
        model = self._make_model()
        assert estimate_kv_cache_bytes(model, 1000) == estimate_kv_cache_bytes(
            model, 1000, kv_cache_quant=None
        )

    def test_qwen3_next_hybrid_linear_and_full_attention(self):
        """Qwen3-Next: only full-attention layers have KV cache, linear layers are skipped.

        Qwen3-Next has 48 layers with full_attention_interval=4, meaning every
        4th layer (12 total) uses full attention with KVCache, while the other
        36 use Gated Delta Net (linear attention) with fixed-size ArraysCache.

        The estimator must skip linear layers (no self_attn) and correctly read
        num_key_value_heads (not n_kv_heads) from the full-attention layers.
        Without this, the estimate is 4x too high and triggers spurious 503s.
        """
        args = MagicMock(spec=[])
        args.num_hidden_layers = 48
        args.num_attention_heads = 16
        args.num_key_value_heads = 2
        args.hidden_size = 2048
        args.head_dim = 256

        model = MagicMock(spec=[])
        model.args = args

        layers = []
        for i in range(48):
            layer = MagicMock()
            if (i + 1) % 4 == 0:
                # Full-attention layer — uses num_key_value_heads (not n_kv_heads)
                attn = MagicMock(spec=[])
                attn.num_key_value_heads = 2
                attn.head_dim = 256
                attn.is_sliding = False
                layer.self_attn = attn
            else:
                # Linear attention layer — no self_attn
                layer.self_attn = None
            layers.append(layer)
        model.model = MagicMock()
        model.model.layers = layers

        num_tokens = 86245
        # Only 12 full-attention layers contribute to growing KV cache
        expected_raw = 12 * 2 * 2 * 256 * num_tokens * 2
        result = estimate_kv_cache_bytes(model, num_tokens)
        assert result == int(expected_raw * _inf_mod.MEMORY_SAFETY_FACTOR)

        # Verify: ~2.6 GB, NOT ~10.3 GB
        assert result < 4 * 1024**3, f"Expected ~2.6 GB, got {result / 1024**3:.1f} GB"

        # The naive fallback (all 48 layers) would give ~10.3 GB
        naive_raw = 48 * 2 * 2 * 256 * num_tokens * 2
        naive = int(naive_raw * _inf_mod.MEMORY_SAFETY_FACTOR)
        assert naive > 10 * 1024**3  # confirm the old estimate was wrong
        assert result < naive / 3  # introspection should be at least 3x lower


class TestKvCachePreflightCheck:
    """Tests for the pre-flight KV cache memory check in _stream_completion."""

    @pytest.fixture
    def mock_lm(self):
        lm = MagicMock()
        lm.is_vlm = False
        lm.is_distributed = False
        lm.speculative_decoder = None
        type(lm).is_speculative = property(
            lambda self: self.speculative_decoder is not None
        )
        lm.model = MagicMock()
        lm.model.args = MagicMock()
        lm.model.args.num_hidden_layers = 32
        lm.model.args.num_attention_heads = 32
        lm.model.args.num_key_value_heads = 8
        lm.model.args.hidden_size = 4096
        lm.tokenizer = MagicMock()
        lm.text_tokenizer = MagicMock()
        lm.prompt_cache_store = MagicMock()
        lm.prompt_cache_store.get.return_value = None
        lm.prompt_cache_store.evict_all_to_disk = MagicMock()
        lm.prompt_cache_store.async_get = AsyncMock(return_value=None)
        lm.prompt_cache_store.async_set = AsyncMock(return_value=None)
        lm.prompt_cache_store.async_evict_all_to_disk = AsyncMock()
        lm.active_refs = 0
        lm.inference_timeout = None
        lm.inference_queue_timeout = None
        lm.sync_mode = None
        return lm

    @pytest.mark.asyncio
    async def test_rejects_when_kv_cache_exceeds_memory(self, mock_lm):
        """_stream_completion should raise MemoryError when estimated KV cache
        would push Metal memory over the limit."""
        stats = TimingStats()

        # Simulate: current memory is 10GB, limit is 12GB, KV cache needs 3GB
        # 10GB + 3GB > 12GB → should reject
        total_mem = 24 * 1024**3  # 24GB system
        current_metal = 10 * 1024**3  # 10GB used
        kv_estimate = 3 * 1024**3  # 3GB KV cache

        # Pre-acquire the real lock so the finally block can release it
        await _inference_lock.acquire()
        try:
            with (
                patch(
                    "olmlx.utils.memory.get_system_memory_bytes", return_value=total_mem
                ),
                patch(
                    "olmlx.utils.memory.get_metal_memory", return_value=current_metal
                ),
                patch("olmlx.engine.inference.settings") as mock_settings,
                patch("olmlx.engine.inference.mx"),
                patch(
                    "olmlx.engine.inference.estimate_kv_cache_bytes",
                    return_value=kv_estimate,
                ),
                patch("olmlx.engine.inference._safe_sync"),
                patch(
                    "olmlx.engine.inference._await_deferred_cleanup",
                    new_callable=AsyncMock,
                ),
                patch(
                    "olmlx.engine.inference._acquire_inference_lock",
                    new_callable=AsyncMock,
                ),
            ):
                mock_settings.memory_limit_fraction = 0.5  # 12GB limit
                mock_settings.prompt_cache = False
                mock_settings.inference_timeout = None
                mock_settings.sync_mode = "full"

                gen = _inf_mod._stream_completion(
                    mock_lm, list(range(22000)), 512, {}, stats
                )
                with pytest.raises(MemoryError, match="prompt too long"):
                    async for _ in gen:
                        pass
        finally:
            # Ensure lock is released even if test fails
            if _inference_lock.locked():
                _inference_lock.release()

    @pytest.mark.asyncio
    async def test_allows_when_kv_cache_fits(self, mock_lm):
        """Should proceed normally when KV cache fits in memory."""
        stats = TimingStats()

        total_mem = 24 * 1024**3
        current_metal = 5 * 1024**3  # 5GB used
        kv_estimate = 1 * 1024**3  # 1GB KV cache — fits within 12GB limit

        # Create a proper async iterable mock with one token
        final_token = StreamToken(
            text="hi",
            token=1,
            prompt_tokens=100,
            generation_tokens=1,
            prompt_tps=100.0,
            generation_tps=50.0,
        )
        mock_stream = AsyncMock()
        mock_stream.__aiter__ = MagicMock(return_value=mock_stream)
        mock_stream.__anext__ = AsyncMock(side_effect=[final_token, StopAsyncIteration])
        mock_stream._thread = None
        mock_stream.drain_and_join = AsyncMock()
        mock_stream.cancel = MagicMock()

        # Pre-acquire the real lock so the finally block can release it
        await _inference_lock.acquire()
        try:
            with (
                patch(
                    "olmlx.utils.memory.get_system_memory_bytes", return_value=total_mem
                ),
                patch(
                    "olmlx.utils.memory.get_metal_memory", return_value=current_metal
                ),
                patch("olmlx.engine.inference.settings") as mock_settings,
                patch("olmlx.engine.inference.mx"),
                patch(
                    "olmlx.engine.inference.estimate_kv_cache_bytes",
                    return_value=kv_estimate,
                ),
                patch("olmlx.engine.inference._safe_sync"),
                patch(
                    "olmlx.engine.inference._await_deferred_cleanup",
                    new_callable=AsyncMock,
                ),
                patch(
                    "olmlx.engine.inference._acquire_inference_lock",
                    new_callable=AsyncMock,
                ),
                patch(
                    "olmlx.engine.inference.async_mlx_stream", return_value=mock_stream
                ),
                patch("olmlx.engine.inference.parse_keep_alive", return_value=300),
            ):
                mock_settings.memory_limit_fraction = 0.5  # 12GB limit
                mock_settings.prompt_cache = False
                mock_settings.default_keep_alive = "5m"
                mock_settings.inference_timeout = None
                mock_settings.sync_mode = "full"

                gen = _inf_mod._stream_completion(
                    mock_lm, list(range(100)), 512, {}, stats
                )
                # Should not raise — just exhaust the generator
                chunks = []
                async for chunk in gen:
                    chunks.append(chunk)
                # Generator completed without MemoryError
        finally:
            if _inference_lock.locked():
                _inference_lock.release()


class TestPrefillMemoryCallback:
    """Tests for memory checking during prefill."""

    def test_prefill_callback_returns_false_on_memory_exceeded(self):
        """The prefill progress callback should return False when memory exceeds limit."""
        from olmlx.utils.streaming import _make_prefill_progress

        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        # memory_limit = 12GB, current memory at 13GB → over limit
        mock_mx = MagicMock()
        mock_mx.get_active_memory.return_value = 10 * 1024**3
        mock_mx.get_cache_memory.return_value = 3 * 1024**3

        callback = _make_prefill_progress(
            cancel_event, memory_limit=12 * 1024**3, mx_module=mock_mx
        )
        # At 50% progress, should check memory and return False
        result = callback(0.5)
        assert result is False

    def test_prefill_callback_returns_true_when_memory_ok(self):
        """The prefill progress callback should return True when memory is fine."""
        from olmlx.utils.streaming import _make_prefill_progress

        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        mock_mx = MagicMock()
        mock_mx.get_active_memory.return_value = 5 * 1024**3
        mock_mx.get_cache_memory.return_value = 1 * 1024**3

        callback = _make_prefill_progress(
            cancel_event, memory_limit=12 * 1024**3, mx_module=mock_mx
        )
        result = callback(0.5)
        assert result is True

    def test_prefill_callback_respects_cancel_event(self):
        """Cancel event should still take priority."""
        from olmlx.utils.streaming import _make_prefill_progress

        cancel_event = MagicMock()
        cancel_event.is_set.return_value = True

        callback = _make_prefill_progress(cancel_event, memory_limit=12 * 1024**3)
        result = callback(0.5)
        assert result is False

    def test_prefill_callback_no_limit(self):
        """With no memory limit, should only check cancel event."""
        from olmlx.utils.streaming import _make_prefill_progress

        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        callback = _make_prefill_progress(cancel_event, memory_limit=0)
        result = callback(0.5)
        assert result is True


class TestSetupPromptCache:
    """Tests for the extracted _setup_prompt_cache helper."""

    @pytest.fixture
    def mock_lm(self):
        lm = MagicMock()
        lm.is_vlm = False
        lm.kv_cache_quant = None
        lm.model = MagicMock()
        lm.prompt_cache_store = MagicMock()
        lm.prompt_cache_store.async_get = AsyncMock(return_value=None)
        lm.prompt_cache_store.async_set = AsyncMock(return_value=None)
        lm.prompt_cache_store.async_evict_all_to_disk = AsyncMock()
        return lm

    @pytest.mark.asyncio
    async def test_cache_disabled_returns_defaults(self, mock_lm):
        """When prompt_tokens is None, returns default result."""
        from olmlx.engine.inference import _setup_prompt_cache

        gen_kwargs = {}
        result = await _setup_prompt_cache(
            mock_lm, "hello", gen_kwargs, prompt_tokens=None, cache_id="test"
        )
        assert result.cache_read_tokens == 0
        assert result.cache_creation_tokens == 0
        assert result.full_prompt_tokens is None
        assert result.cache_setup_done is False
        assert result.prompt == "hello"
        assert "prompt_cache" not in gen_kwargs

    @pytest.mark.asyncio
    async def test_cache_miss_creates_fresh(self, mock_lm):
        """Cache miss creates a fresh cache and sets cache_creation_tokens."""
        from olmlx.engine.inference import _setup_prompt_cache

        mock_cache = MagicMock()
        gen_kwargs = {}
        prompt_tokens = [1, 2, 3, 4, 5]

        with (
            patch("olmlx.engine.inference.make_prompt_cache", return_value=mock_cache),
            patch("olmlx.utils.memory.is_memory_pressure_high", return_value=False),
            patch("olmlx.engine.inference._find_common_prefix", return_value=0),
            patch(
                "olmlx.engine.inference._get_model_for_cache", return_value=MagicMock()
            ),
        ):
            result = await _setup_prompt_cache(
                mock_lm,
                "hello",
                gen_kwargs,
                prompt_tokens=prompt_tokens,
                cache_id="test",
            )

        assert result.cache_setup_done is True
        assert result.cache_creation_tokens == 5
        assert result.cache_read_tokens == 0
        assert result.prompt == prompt_tokens
        assert gen_kwargs["prompt_cache"] is mock_cache

    @pytest.mark.asyncio
    async def test_cache_hit_sets_read_tokens(self, mock_lm):
        """Cache hit with prefix match sets cache_read_tokens and mutates gen_kwargs."""
        from olmlx.engine.inference import _setup_prompt_cache
        from olmlx.engine.model_manager import CachedPromptState

        cached_state = CachedPromptState(tokens=[1, 2, 3], cache=MagicMock())
        mock_lm.prompt_cache_store.async_get = AsyncMock(return_value=cached_state)

        gen_kwargs = {}
        prompt_tokens = [1, 2, 3, 4, 5]

        with (
            patch("olmlx.utils.memory.is_memory_pressure_high", return_value=False),
            patch("olmlx.engine.inference._find_common_prefix", return_value=3),
            patch("olmlx.engine.inference.trim_prompt_cache"),
        ):
            result = await _setup_prompt_cache(
                mock_lm,
                "hello",
                gen_kwargs,
                prompt_tokens=prompt_tokens,
                cache_id="test",
            )

        assert result.cache_setup_done is True
        # suffix_start = min(prefix_len=3, len(prompt_tokens)-1=4) = 3
        assert result.cache_read_tokens == 3
        assert result.cache_creation_tokens == 2  # suffix tokens [4, 5]
        assert "prompt_cache" in gen_kwargs
        mock_lm.prompt_cache_store.remove.assert_called_with("test")

    @pytest.mark.asyncio
    async def test_memory_pressure_evicts_caches(self, mock_lm):
        """High memory pressure triggers evict_all_to_disk."""
        from olmlx.engine.inference import _setup_prompt_cache

        gen_kwargs = {}
        prompt_tokens = [1, 2, 3]

        with (
            patch("olmlx.utils.memory.is_memory_pressure_high", return_value=True),
            patch("olmlx.engine.inference._safe_sync"),
            patch("olmlx.engine.inference.mx"),
        ):
            result = await _setup_prompt_cache(
                mock_lm,
                "hello",
                gen_kwargs,
                prompt_tokens=prompt_tokens,
                cache_id="test",
            )

        mock_lm.prompt_cache_store.async_evict_all_to_disk.assert_awaited_once()
        # After eviction, memory still high → no cache setup
        assert result.cache_setup_done is False

    @pytest.mark.asyncio
    async def test_vlm_uses_input_ids(self, mock_lm):
        """VLM path sets input_ids in gen_kwargs instead of replacing prompt."""
        from olmlx.engine.inference import _setup_prompt_cache

        mock_lm.is_vlm = True
        mock_cache = MagicMock()
        gen_kwargs = {}
        prompt_tokens = [1, 2, 3]

        with (
            patch("olmlx.engine.inference.make_prompt_cache", return_value=mock_cache),
            patch("olmlx.utils.memory.is_memory_pressure_high", return_value=False),
            patch("olmlx.engine.inference._find_common_prefix", return_value=0),
            patch(
                "olmlx.engine.inference._get_model_for_cache", return_value=MagicMock()
            ),
        ):
            result = await _setup_prompt_cache(
                mock_lm,
                "hello",
                gen_kwargs,
                prompt_tokens=prompt_tokens,
                cache_id="test",
            )

        assert "input_ids" in gen_kwargs
        assert result.prompt == "hello"  # prompt unchanged for VLM


class TestKvCachePreflightCheckHelper:
    """Tests for the extracted _kv_cache_preflight_check helper."""

    @pytest.fixture
    def mock_lm(self):
        lm = MagicMock()
        lm.is_vlm = False
        lm.kv_cache_quant = None
        lm.model = MagicMock()
        lm.text_tokenizer = MagicMock()
        lm.prompt_cache_store = MagicMock()
        lm.prompt_cache_store.async_set = AsyncMock(return_value=None)
        lm.prompt_cache_store.async_evict_all_to_disk = AsyncMock()
        return lm

    @pytest.mark.asyncio
    async def test_allows_when_fits(self, mock_lm):
        from olmlx.engine.inference import _kv_cache_preflight_check

        gen_kwargs = {}
        with (
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=24 * 1024**3,
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=5 * 1024**3,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch(
                "olmlx.engine.inference.estimate_kv_cache_bytes",
                return_value=1 * 1024**3,
            ),
        ):
            mock_settings.memory_limit_fraction = 0.5
            result = await _kv_cache_preflight_check(
                mock_lm,
                [1, 2, 3],
                100,
                gen_kwargs,
                cache_read_tokens=0,
                cache_creation_tokens=3,
                full_prompt_tokens=None,
                cache_id="test",
            )

        assert result.memory_limit == 12 * 1024**3
        assert result.prompt == [1, 2, 3]  # unchanged

    @pytest.mark.asyncio
    async def test_rejects_when_exceeds_memory(self, mock_lm):
        from olmlx.engine.inference import _kv_cache_preflight_check

        gen_kwargs = {}
        with (
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=24 * 1024**3,
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=10 * 1024**3,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch(
                "olmlx.engine.inference.estimate_kv_cache_bytes",
                return_value=3 * 1024**3,
            ),
            patch("olmlx.engine.inference._safe_sync"),
            patch("olmlx.engine.inference.mx"),
        ):
            mock_settings.memory_limit_fraction = 0.5
            with pytest.raises(MemoryError, match="prompt too long"):
                await _kv_cache_preflight_check(
                    mock_lm,
                    [1, 2, 3],
                    100,
                    gen_kwargs,
                    cache_read_tokens=0,
                    cache_creation_tokens=3,
                    full_prompt_tokens=None,
                    cache_id="test",
                )

    @pytest.mark.asyncio
    async def test_skips_when_zero_prefill(self, mock_lm):
        from olmlx.engine.inference import _kv_cache_preflight_check

        gen_kwargs = {}
        with (
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=24 * 1024**3,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.memory_limit_fraction = 0.5
            result = await _kv_cache_preflight_check(
                mock_lm,
                "hello",
                100,
                gen_kwargs,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                full_prompt_tokens=None,
                cache_id="test",
            )

        # cache_creation_tokens=0 and no tokenizable prompt → num_prefill_tokens=0 → skip check
        assert result.memory_limit == 12 * 1024**3

    @pytest.mark.asyncio
    async def test_handles_estimation_error(self, mock_lm):
        from olmlx.engine.inference import _kv_cache_preflight_check

        gen_kwargs = {}
        with (
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=24 * 1024**3,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch(
                "olmlx.engine.inference.estimate_kv_cache_bytes",
                side_effect=ValueError("bad model"),
            ),
        ):
            mock_settings.memory_limit_fraction = 0.5
            # Should not raise — estimation error is caught and logged
            result = await _kv_cache_preflight_check(
                mock_lm,
                [1, 2, 3],
                100,
                gen_kwargs,
                cache_read_tokens=0,
                cache_creation_tokens=3,
                full_prompt_tokens=None,
                cache_id="test",
            )
        assert result.memory_limit > 0


class TestStorePromptCacheAfterGeneration:
    """Tests for the extracted _store_prompt_cache_after_generation helper."""

    @pytest.fixture
    def mock_lm(self):
        lm = MagicMock()
        lm.prompt_cache_store = MagicMock()
        lm.prompt_cache_store.async_set = AsyncMock(return_value=None)
        return lm

    @pytest.mark.asyncio
    async def test_noop_when_no_cache(self, mock_lm):
        from olmlx.engine.inference import _store_prompt_cache_after_generation

        gen_kwargs = {}  # no prompt_cache key
        await _store_prompt_cache_after_generation(
            mock_lm, gen_kwargs, [1, 2, 3], [4, 5], 2, "test"
        )
        mock_lm.prompt_cache_store.async_set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_when_no_full_prompt_tokens(self, mock_lm):
        from olmlx.engine.inference import _store_prompt_cache_after_generation

        gen_kwargs = {"prompt_cache": MagicMock()}
        await _store_prompt_cache_after_generation(
            mock_lm, gen_kwargs, None, [4, 5], 2, "test"
        )
        mock_lm.prompt_cache_store.async_set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stores_without_trimming(self, mock_lm):
        from olmlx.engine.inference import _store_prompt_cache_after_generation

        cache = MagicMock()
        gen_kwargs = {"prompt_cache": cache}
        with patch("olmlx.engine.inference.settings") as mock_settings:
            mock_settings.prompt_cache_max_tokens = None
            mock_settings.memory_limit_fraction = 0.9
            await _store_prompt_cache_after_generation(
                mock_lm, gen_kwargs, [1, 2, 3], [4, 5], 2, "test"
            )

        mock_lm.prompt_cache_store.async_set.assert_awaited_once()
        call_args = mock_lm.prompt_cache_store.async_set.call_args
        assert call_args[0][0] == "test"
        stored = call_args[0][1]
        assert stored.tokens == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_trims_when_over_max(self, mock_lm):
        from olmlx.engine.inference import _store_prompt_cache_after_generation

        cache = MagicMock()
        gen_kwargs = {"prompt_cache": cache}
        # trim_prompt_cache must return the amount it was asked to trim
        # so that the trimmed != trim_amount check passes.
        # prompt=3 + eval=2 = 5 total > max 4 → trim_amount = 1
        with (
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                side_effect=lambda _cache, amount: amount,
            ),
        ):
            mock_settings.prompt_cache_max_tokens = 4
            mock_settings.memory_limit_fraction = 0.9
            await _store_prompt_cache_after_generation(
                mock_lm, gen_kwargs, [1, 2, 3], [4, 5], 2, "test"
            )

        mock_lm.prompt_cache_store.async_set.assert_awaited_once()
        call_args = mock_lm.prompt_cache_store.async_set.call_args
        stored = call_args[0][1]
        assert len(stored.tokens) == 4

    @pytest.mark.asyncio
    async def test_trim_failure_invalidates(self, mock_lm):
        from olmlx.engine.inference import _store_prompt_cache_after_generation

        cache = MagicMock()
        gen_kwargs = {"prompt_cache": cache}
        with (
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                side_effect=RuntimeError("trim broke"),
            ),
            patch("olmlx.engine.inference.mx"),
        ):
            mock_settings.prompt_cache_max_tokens = 4
            mock_settings.memory_limit_fraction = 0.9
            await _store_prompt_cache_after_generation(
                mock_lm, gen_kwargs, [1, 2, 3], [4, 5], 2, "test"
            )

        # Cache should be invalidated (removed) on trim failure
        mock_lm.prompt_cache_store.remove.assert_called_with("test")
        mock_lm.prompt_cache_store.async_set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_trimmable_stores_without_trim(self, mock_lm):
        """When supports_cache_trim=False and total > max, store as-is."""
        from olmlx.engine.inference import _store_prompt_cache_after_generation

        mock_lm.supports_cache_trim = False
        cache = MagicMock()
        gen_kwargs = {"prompt_cache": cache}
        with patch("olmlx.engine.inference.settings") as mock_settings:
            mock_settings.prompt_cache_max_tokens = 4
            mock_settings.memory_limit_fraction = 0.9
            await _store_prompt_cache_after_generation(
                mock_lm, gen_kwargs, [1, 2, 3], [4, 5], 2, "test"
            )

        # Cache stored as-is (3 prompt + 2 gen = 5, exceeding limit of 4)
        mock_lm.prompt_cache_store.async_set.assert_awaited_once()
        call_args = mock_lm.prompt_cache_store.async_set.call_args
        assert call_args[0][0] == "test"
        stored = call_args[0][1]
        assert stored.tokens == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_non_trimmable_misaligned_eval_count(self, mock_lm):
        """When supports_cache_trim=False and eval_count != len(generated_tokens),
        skip storage entirely — stale generation entries in the ring buffer
        can't be trimmed out and would corrupt the next turn."""
        from olmlx.engine.inference import _store_prompt_cache_after_generation

        mock_lm.supports_cache_trim = False
        cache = MagicMock()
        gen_kwargs = {"prompt_cache": cache}
        # eval_count=7 != len(generated_tokens)=2: None-ID tokens produced
        with patch("olmlx.engine.inference.settings") as mock_settings:
            mock_settings.prompt_cache_max_tokens = 4
            mock_settings.memory_limit_fraction = 0.9
            await _store_prompt_cache_after_generation(
                mock_lm, gen_kwargs, [1, 2, 3], [4, 5], 7, "test"
            )

        # Cache must NOT be stored — stale ring-buffer entries would be unsafe.
        # The previous entry from a prior turn must be cleaned up since the
        # mutable ring buffer was shared and is now corrupted.
        mock_lm.prompt_cache_store.async_set.assert_not_awaited()
        mock_lm.prompt_cache_store.remove.assert_called_once_with("test")

    @pytest.mark.asyncio
    async def test_non_trimmable_within_limit_stores_as_is(self, mock_lm):
        """When supports_cache_trim=False and total <= max, store without trimming."""
        from olmlx.engine.inference import _store_prompt_cache_after_generation

        mock_lm.supports_cache_trim = False
        cache = MagicMock()
        gen_kwargs = {"prompt_cache": cache}
        with patch("olmlx.engine.inference.settings") as mock_settings:
            mock_settings.prompt_cache_max_tokens = 32768  # well above 5
            mock_settings.memory_limit_fraction = 0.9
            await _store_prompt_cache_after_generation(
                mock_lm, gen_kwargs, [1, 2, 3], [4, 5], 2, "test"
            )

        mock_lm.prompt_cache_store.async_set.assert_awaited_once()
        call_args = mock_lm.prompt_cache_store.async_set.call_args
        assert call_args[0][0] == "test"
        stored = call_args[0][1]
        assert stored.tokens == [1, 2, 3, 4, 5]


class TestInferenceTimeout:
    @pytest.mark.asyncio
    async def test_streaming_stops_after_inference_timeout(self, mock_manager):
        """Streaming generation should stop and return done_reason=timeout."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.inference_timeout = 0.1  # 100ms timeout

        mock_mx = MagicMock()

        # Create a slow token stream: each token takes 60ms
        tokens = [
            StreamToken(
                text=f"tok{i}",
                token=i,
                prompt_tokens=5,
                generation_tokens=i + 1,
                prompt_tps=100.0,
                generation_tps=50.0,
            )
            for i in range(20)  # would take 1.2s total, well past 100ms
        ]

        mock_stream = MagicMock(spec=CancellableStream)
        mock_stream.drain_and_join = AsyncMock()
        mock_stream._thread = None

        token_iter = iter(tokens)

        async def anext_impl():
            await asyncio.sleep(0.06)  # 60ms per token
            try:
                return next(token_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.async_mlx_stream", return_value=mock_stream
            ):
                gen = await generate_completion(
                    mock_manager,
                    "qwen3",
                    "Hello",
                    stream=True,
                )
                chunks = []
                async for chunk in gen:
                    chunks.append(chunk)

        done_chunk = chunks[-1]
        assert done_chunk["done"] is True
        assert done_chunk.get("done_reason") == "timeout"
        # Should have produced some tokens but not all 20
        text_chunks = [c for c in chunks if not c.get("done")]
        assert 0 < len(text_chunks) < 20

    @pytest.mark.asyncio
    async def test_streaming_no_timeout_when_none(self, mock_manager):
        """No timeout when inference_timeout is None — all tokens produced."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.inference_timeout = None

        mock_mx = MagicMock()

        tokens = [
            StreamToken(
                text=f"tok{i}",
                token=i,
                prompt_tokens=5,
                generation_tokens=i + 1,
                prompt_tps=100.0,
                generation_tps=50.0,
            )
            for i in range(3)
        ]

        mock_stream = MagicMock(spec=CancellableStream)
        mock_stream.drain_and_join = AsyncMock()
        mock_stream._thread = None

        token_iter = iter(tokens)

        async def anext_impl():
            try:
                return next(token_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch("olmlx.engine.inference.async_mlx_stream", return_value=mock_stream),
        ):
            mock_settings.inference_queue_timeout = None
            mock_settings.inference_timeout = None
            mock_settings.prompt_cache = False
            mock_settings.memory_limit_fraction = 0.9
            mock_settings.sync_mode = "full"
            gen = await generate_completion(
                mock_manager,
                "qwen3",
                "Hello",
                stream=True,
            )
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)

        done_chunk = chunks[-1]
        assert done_chunk["done"] is True
        assert "done_reason" not in done_chunk
        text_chunks = [c for c in chunks if not c.get("done")]
        assert len(text_chunks) == 3

    @pytest.mark.asyncio
    async def test_global_inference_timeout_used_when_model_none(self, mock_manager):
        """Global settings.inference_timeout is used when model has no override."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.inference_timeout = None  # No per-model override

        mock_mx = MagicMock()

        tokens = [
            StreamToken(
                text=f"tok{i}",
                token=i,
                prompt_tokens=5,
                generation_tokens=i + 1,
                prompt_tps=100.0,
                generation_tps=50.0,
            )
            for i in range(20)
        ]

        mock_stream = MagicMock(spec=CancellableStream)
        mock_stream.drain_and_join = AsyncMock()
        mock_stream._thread = None

        token_iter = iter(tokens)

        async def anext_impl():
            await asyncio.sleep(0.06)
            try:
                return next(token_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch("olmlx.engine.inference.async_mlx_stream", return_value=mock_stream),
        ):
            mock_settings.inference_queue_timeout = None
            mock_settings.inference_timeout = 0.1  # Global 100ms timeout
            mock_settings.prompt_cache = False
            mock_settings.memory_limit_fraction = 0.9
            mock_settings.sync_mode = "full"
            gen = await generate_completion(
                mock_manager,
                "qwen3",
                "Hello",
                stream=True,
            )
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)

        done_chunk = chunks[-1]
        assert done_chunk["done"] is True
        assert done_chunk.get("done_reason") == "timeout"
        text_chunks = [c for c in chunks if not c.get("done")]
        assert 0 < len(text_chunks) < 20


class TestStreamCompletionLockLeakOnSyncFailure:
    """_stream_completion must release the inference lock even if a
    _lock_boundary_sync call raises. Mirrors the _inference_locked lock-leak
    regression tests."""

    def _mock_stream(self, n_tokens: int = 3):
        mock_stream = MagicMock(spec=CancellableStream)
        mock_stream.drain_and_join = AsyncMock()
        mock_stream._thread = None
        tokens = [
            StreamToken(
                text=f"tok{i}",
                token=i,
                prompt_tokens=5,
                generation_tokens=i + 1,
                prompt_tps=100.0,
                generation_tps=50.0,
            )
            for i in range(n_tokens)
        ]
        token_iter = iter(tokens)

        async def anext_impl():
            try:
                return next(token_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()
        return mock_stream

    @pytest.mark.asyncio
    async def test_lock_released_if_entry_sync_raises(self, mock_manager):
        """If _lock_boundary_sync raises on stream entry, the inference lock
        must not leak."""
        assert not _inference_lock.locked()
        with (
            patch("olmlx.engine.inference.mx"),
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=self._mock_stream(),
            ),
            patch(
                "olmlx.engine.inference._lock_boundary_sync",
                side_effect=ValueError("entry boom"),
            ),
        ):
            mock_settings.inference_queue_timeout = None
            mock_settings.inference_timeout = None
            mock_settings.prompt_cache = False
            mock_settings.memory_limit_fraction = 0.9
            mock_settings.sync_mode = "full"
            gen = await generate_completion(mock_manager, "qwen3", "Hello", stream=True)
            with pytest.raises(ValueError, match="entry boom"):
                async for _ in gen:
                    pass
        assert not _inference_lock.locked()

    @pytest.mark.asyncio
    async def test_exit_sync_raise_is_suppressed_and_safe_sync_runs(self, mock_manager):
        """If exit _lock_boundary_sync raises during normal cleanup, the
        streaming path must catch it, run _safe_sync() as a fail-safe
        fallback, and release the lock. No exception should propagate
        from a successful stream body."""
        assert not _inference_lock.locked()
        call_count = {"n": 0}

        def side_effect(*_a, **_kw):
            call_count["n"] += 1
            # Let entry sync pass; raise only on exit sync (second call).
            if call_count["n"] >= 2:
                raise ValueError("exit boom")

        with (
            patch("olmlx.engine.inference.mx"),
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=self._mock_stream(),
            ),
            patch(
                "olmlx.engine.inference._lock_boundary_sync", side_effect=side_effect
            ),
            patch("olmlx.engine.inference._safe_sync") as mock_safe_sync,
        ):
            mock_settings.inference_queue_timeout = None
            mock_settings.inference_timeout = None
            mock_settings.prompt_cache = False
            mock_settings.memory_limit_fraction = 0.9
            mock_settings.sync_mode = "full"
            gen = await generate_completion(mock_manager, "qwen3", "Hello", stream=True)
            # No exception should escape — the exit-sync raise is caught.
            async for _ in gen:
                pass
        assert not _inference_lock.locked()
        # Entry + exit = exactly 2 lock_boundary_sync calls in the success path.
        assert call_count["n"] == 2
        mock_safe_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_mode_none_skips_lock_boundary_sync(self, mock_manager):
        """With lm.sync_mode='none', the streaming path must not call
        mx.synchronize at either lock-boundary site. Independent of the
        equivalent _inference_locked test because _stream_completion
        manages its own lock entry/exit.
        """
        lm = mock_manager._loaded["qwen3:latest"]
        lm.sync_mode = "none"
        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=self._mock_stream(),
            ),
        ):
            mock_settings.inference_queue_timeout = None
            mock_settings.inference_timeout = None
            mock_settings.prompt_cache = False
            mock_settings.memory_limit_fraction = 0.9
            mock_settings.sync_mode = "full"  # per-model "none" wins over global "full"
            gen = await generate_completion(mock_manager, "qwen3", "Hello", stream=True)
            async for _ in gen:
                pass
        assert not _inference_lock.locked()
        assert mock_mx.synchronize.call_count == 0
