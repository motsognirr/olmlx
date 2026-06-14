"""Tests for panel-type entries in olmlx.engine.registry."""

import json

import pytest

from olmlx.engine.registry import ModelRegistry, PanelConfig


class TestPanelStopCondition:
    def _entry(self, **extra):
        return {
            "type": "panel",
            "classifier": "c",
            "judge": "j",
            "routes": {"default": ["a"]},
            **extra,
        }

    def test_defaults_to_all(self):
        pc = PanelConfig.from_entry("p:latest", self._entry())
        assert pc.stop_condition == "all"

    def test_accepts_majority_and_judge(self):
        for cond in ("all", "majority", "judge"):
            pc = PanelConfig.from_entry("p:latest", self._entry(stop_condition=cond))
            assert pc.stop_condition == cond

    def test_rejects_unknown_stop_condition(self):
        with pytest.raises(ValueError, match="stop_condition"):
            PanelConfig.from_entry("p:latest", self._entry(stop_condition="bogus"))


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


def _load_registry(tmp_path, monkeypatch, config: dict) -> ModelRegistry:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", path)
    reg = ModelRegistry()
    reg.load()
    return reg


class TestRegistryPanelLoading:
    def test_panel_entry_loaded_and_resolvable(self, tmp_path, monkeypatch):
        reg = _load_registry(
            tmp_path,
            monkeypatch,
            {
                "qwen3": "Qwen/Qwen3-8B-MLX",
                "small": "org/small",
                "judgem": "org/judge",
                "my-panel": {
                    "type": "panel",
                    "classifier": "small",
                    "judge": "judgem",
                    "routes": {"default": ["qwen3"]},
                },
            },
        )
        assert reg.is_panel("my-panel") is True
        assert reg.is_panel("my-panel:latest") is True
        assert reg.is_panel("qwen3") is False
        pc = reg.resolve_panel("my-panel")
        assert pc.judge == "judgem"
        # A panel name is NOT a normal model.
        assert reg.resolve("my-panel") is None
        # ...but it IS listed for model-listing surfaces, separate from models.
        panels = reg.list_panels()
        assert "my-panel:latest" in panels
        assert panels["my-panel:latest"].judge == "judgem"
        assert "my-panel:latest" not in reg.list_models()

    def test_panel_with_missing_member_is_dropped(self, tmp_path, monkeypatch):
        reg = _load_registry(
            tmp_path,
            monkeypatch,
            {
                "small": "org/small",
                "judgem": "org/judge",
                "bad-panel": {
                    "type": "panel",
                    "classifier": "small",
                    "judge": "judgem",
                    "routes": {"default": ["does-not-exist"]},
                },
            },
        )
        assert reg.is_panel("bad-panel") is False

    # ---------------------------------------------------------------------------
    # Gap 3 — Registry panel validation: judge-in-panel warns; missing judge drops
    # ---------------------------------------------------------------------------

    def test_judge_in_panel_warns_but_keeps(self, tmp_path, monkeypatch, caplog):
        import logging

        config = {
            "qwen3": "Qwen/Qwen3-8B-MLX",
            "small": "org/small",
            "panel-x": {
                "type": "panel",
                "classifier": "small",
                "judge": "qwen3",
                "routes": {"default": ["qwen3"]},  # judge is also a member
            },
        }
        with caplog.at_level(logging.WARNING):
            reg = _load_registry(tmp_path, monkeypatch, config)
        assert reg.is_panel("panel-x") is True  # kept, not dropped
        # The warning from _validate_panels mentions self-preference bias.
        assert any(
            "self-preference" in r.message or "self-preference bias" in r.message
            for r in caplog.records
        ), (
            f"Expected self-preference warning, got: {[r.message for r in caplog.records]}"
        )

    def test_missing_judge_drops_panel(self, tmp_path, monkeypatch):
        config = {
            "small": "org/small",
            "panel-y": {
                "type": "panel",
                "classifier": "small",
                "judge": "no-such-model",
                "routes": {"default": ["small"]},
            },
        }
        reg = _load_registry(tmp_path, monkeypatch, config)
        assert reg.is_panel("panel-y") is False
