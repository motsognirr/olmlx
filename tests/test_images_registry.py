"""models.json ``"type": "image"`` marker for text-to-image models (#723).

Image models are *declared*, never sniffed: mflux checkpoints use a
diffusers layout with no top-level ``config.json``, and mflux's own
``ModelConfig.from_name`` is a loose substring matcher that resolves plain
``Qwen/Qwen3-*`` text models to a Qwen-Image base.
"""

import json

import pytest

from olmlx.engine.registry import ModelConfig, ModelRegistry


class TestModelConfigImageType:
    def test_type_image_parsed(self):
        mc = ModelConfig.from_entry({"type": "image", "hf_path": "Qwen/Qwen-Image-2.1"})
        assert mc.type == "image"
        assert mc.is_image is True

    def test_default_is_not_image(self):
        mc = ModelConfig.from_entry("Qwen/Qwen3-8B")
        assert mc.type is None
        assert mc.is_image is False

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="type"):
            ModelConfig.from_entry({"type": "video", "hf_path": "a/b"})

    def test_image_quantize_parsed(self):
        mc = ModelConfig.from_entry(
            {"type": "image", "hf_path": "Qwen/Qwen-Image-2.1", "image_quantize": 8}
        )
        assert mc.image_quantize == 8

    @pytest.mark.parametrize("bad", [2, 7, 16, "8", True, 4.0])
    def test_invalid_image_quantize_rejected(self, bad):
        with pytest.raises(ValueError, match="image_quantize"):
            ModelConfig.from_entry(
                {
                    "type": "image",
                    "hf_path": "Qwen/Qwen-Image-2.1",
                    "image_quantize": bad,
                }
            )

    def test_image_quantize_requires_image_type(self):
        with pytest.raises(ValueError, match="image_quantize"):
            ModelConfig.from_entry({"hf_path": "a/b", "image_quantize": 8})

    def test_round_trip(self):
        entry = {"hf_path": "Qwen/Qwen-Image-2.1", "type": "image", "image_quantize": 4}
        mc = ModelConfig.from_entry(entry)
        assert mc.to_entry() == entry
        assert "type" not in mc._extra


class TestRegistryImageEntries:
    @pytest.fixture
    def reg(self, tmp_path, monkeypatch):
        cfg = {
            "qwen-image:2.1": {"type": "image", "hf_path": "Qwen/Qwen-Image-2.1"},
            "qwen3:32b": "Qwen/Qwen3-32B-4bit",
        }
        path = tmp_path / "models.json"
        path.write_text(json.dumps(cfg))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", path)
        r = ModelRegistry()
        r.load()
        return r

    def test_image_entry_resolves(self, reg):
        mc = reg.resolve("qwen-image:2.1")
        assert mc is not None and mc.is_image

    def test_text_entry_is_not_image(self, reg):
        mc = reg.resolve("qwen3:32b")
        assert mc is not None and not mc.is_image
