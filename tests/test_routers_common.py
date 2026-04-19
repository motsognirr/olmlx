"""Tests for routers/common.py shared utilities."""

import json

from olmlx.routers.common import build_inference_options, format_error


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
