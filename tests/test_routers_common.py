"""Tests for routers/common.py shared utilities."""

import json

from olmlx.routers.common import (
    build_inference_options,
    format_error,
    resolve_openai_think,
    resolve_think_flag,
)


class TestFormatError:
    def test_json_structure(self):
        result = json.loads(format_error("test-model"))
        assert result["model"] == "test-model"
        assert "created_at" in result
        assert result["done"] is True
        assert result["done_reason"] == "error"
        assert "error" in result

    def test_model_propagation(self):
        result = json.loads(format_error("my-fancy-model"))
        assert result["model"] == "my-fancy-model"

    def test_ends_with_newline(self):
        assert format_error("m").endswith("\n")


class TestBuildInferenceOptions:
    def test_all_none_returns_empty(self):
        assert build_inference_options() == {}

    def test_all_fields_populated(self):
        opts = build_inference_options(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            seed=42,
            stop=["###"],
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        assert opts == {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "seed": 42,
            "stop": ["###"],
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
        }

    def test_stop_string_coerced_to_list(self):
        opts = build_inference_options(stop="END")
        assert opts == {"stop": ["END"]}

    def test_stop_list_preserved(self):
        opts = build_inference_options(stop=["a", "b"])
        assert opts == {"stop": ["a", "b"]}

    def test_empty_stop_list_omitted(self):
        # Empty list should not emit stop key — mlx-lm treats absence and
        # empty list differently in some paths; preserve legacy behavior.
        assert build_inference_options(stop=[]) == {}

    def test_empty_stop_string_omitted(self):
        assert build_inference_options(stop="") == {}

    def test_zero_penalties_preserved(self):
        # Regression: earlier truthiness check dropped 0.0 penalties.
        opts = build_inference_options(
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        assert opts == {"frequency_penalty": 0.0, "presence_penalty": 0.0}

    def test_zero_temperature_preserved(self):
        # temperature=0.0 is meaningful (greedy) and must be emitted.
        opts = build_inference_options(temperature=0.0)
        assert opts == {"temperature": 0.0}

    def test_zero_seed_preserved(self):
        opts = build_inference_options(seed=0)
        assert opts == {"seed": 0}

    def test_none_fields_omitted(self):
        opts = build_inference_options(temperature=0.5, top_p=None, top_k=None)
        assert opts == {"temperature": 0.5}


class TestResolveThinkFlag:
    def test_none_returns_none(self):
        # Omitted -> engine default applies.
        assert resolve_think_flag(None) is None

    def test_true_passthrough(self):
        assert resolve_think_flag(True) is True

    def test_false_passthrough(self):
        assert resolve_think_flag(False) is False

    def test_string_level_maps_to_true(self):
        # gpt-oss thinking levels collapse to on (engine is bool-only).
        assert resolve_think_flag("low") is True
        assert resolve_think_flag("medium") is True
        assert resolve_think_flag("high") is True

    def test_arbitrary_nonempty_string_is_on(self):
        assert resolve_think_flag("yes") is True

    def test_stringified_bool_parsed_not_inverted(self):
        # A weakly-typed client sending the JSON string "false"/"true" must not
        # get the opposite of its intent (silent footgun).
        assert resolve_think_flag("false") is False
        assert resolve_think_flag("False") is False
        assert resolve_think_flag("true") is True
        assert resolve_think_flag("TRUE") is True

    def test_empty_string_is_off(self):
        assert resolve_think_flag("") is False


class TestResolveOpenAIThink:
    def test_both_none_returns_none(self):
        assert resolve_openai_think(None, None) is None

    def test_reasoning_effort_present_returns_true(self):
        assert resolve_openai_think("high", None) is True

    def test_empty_reasoning_effort_does_not_enable(self):
        # An empty string is not a real effort value; truthiness, not is-not-None.
        assert resolve_openai_think("", None) is None

    def test_disable_word_reasoning_effort_does_not_enable(self):
        # Out-of-spec but likely-mistake values meaning "off" must not enable.
        assert resolve_openai_think("none", None) is None
        assert resolve_openai_think("off", None) is None
        assert resolve_openai_think("disabled", None) is None
        assert resolve_openai_think("NONE", None) is None

    def test_chat_template_kwargs_enable_thinking_true(self):
        assert resolve_openai_think(None, {"enable_thinking": True}) is True

    def test_chat_template_kwargs_enable_thinking_false(self):
        # The clean OFF switch.
        assert resolve_openai_think(None, {"enable_thinking": False}) is False

    def test_chat_template_kwargs_overrides_reasoning_effort(self):
        # Explicit enable_thinking is authoritative even when effort is set.
        assert resolve_openai_think("high", {"enable_thinking": False}) is False

    def test_chat_template_kwargs_without_enable_thinking_falls_through(self):
        # Unrelated kwargs don't force a decision; effort presence still wins.
        assert resolve_openai_think("low", {"foo": "bar"}) is True
        assert resolve_openai_think(None, {"foo": "bar"}) is None

    def test_chat_template_kwargs_enable_thinking_coerced(self):
        # Truthy/falsey values are coerced to bool.
        assert resolve_openai_think(None, {"enable_thinking": 0}) is False
        assert resolve_openai_think(None, {"enable_thinking": 1}) is True
