"""Tests for panel-type entries in olmlx.engine.registry."""

import pytest

from olmlx.engine.registry import PanelConfig


class TestPanelConfig:
    def test_from_entry_parses_fields(self):
        entry = {
            "type": "panel",
            "classifier": "qwen3-0.6b",
            "judge": "gpt-oss-20b",
            "routes": {
                "code": ["qwen3-coder", "devstral"],
                "default": ["qwen3", "mistral"],
            },
        }
        pc = PanelConfig.from_entry("my-panel:latest", entry)
        assert pc.name == "my-panel:latest"
        assert pc.classifier == "qwen3-0.6b"
        assert pc.judge == "gpt-oss-20b"
        assert pc.routes["code"] == ["qwen3-coder", "devstral"]

    def test_from_entry_requires_default_route(self):
        entry = {
            "type": "panel",
            "classifier": "c",
            "judge": "j",
            "routes": {"code": ["a"]},
        }
        with pytest.raises(ValueError, match="default"):
            PanelConfig.from_entry("p:latest", entry)

    def test_from_entry_requires_classifier_and_judge(self):
        entry = {"type": "panel", "routes": {"default": ["a"]}}
        with pytest.raises(ValueError, match="classifier"):
            PanelConfig.from_entry("p:latest", entry)

    def test_from_entry_rejects_non_string_member(self):
        entry = {
            "type": "panel",
            "classifier": "c",
            "judge": "j",
            "routes": {"default": ["ok", 42]},
        }
        with pytest.raises(ValueError, match="default"):
            PanelConfig.from_entry("p:latest", entry)

    def test_from_entry_rejects_empty_member_list(self):
        entry = {
            "type": "panel",
            "classifier": "c",
            "judge": "j",
            "routes": {"default": []},
        }
        with pytest.raises(ValueError, match="default"):
            PanelConfig.from_entry("p:latest", entry)

    def test_all_member_names_unions_routes(self):
        pc = PanelConfig.from_entry(
            "p:latest",
            {
                "type": "panel",
                "classifier": "c",
                "judge": "j",
                "routes": {"code": ["a", "b"], "default": ["b", "c2"]},
            },
        )
        assert pc.all_member_names() == {"a", "b", "c2"}
