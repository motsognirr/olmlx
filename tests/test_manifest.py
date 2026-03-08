"""Tests for olmlx.models.manifest."""

import json

from olmlx.models.manifest import ModelManifest


class TestModelManifest:
    def test_defaults(self):
        m = ModelManifest(name="test:latest", hf_path="test/model")
        assert m.size == 0
        assert m.format == "mlx"
        assert m.digest == ""
        assert m.family == ""

    def test_to_dict(self):
        m = ModelManifest(name="test:latest", hf_path="test/model", size=1000)
        d = m.to_dict()
        assert d["name"] == "test:latest"
        assert d["hf_path"] == "test/model"
        assert d["size"] == 1000

    def test_save_and_load(self, tmp_path):
        original = ModelManifest(
            name="qwen3:latest",
            hf_path="Qwen/Qwen3-8B-MLX",
            size=5000,
            modified_at="2024-01-01T00:00:00Z",
            digest="sha256:abc123",
            format="mlx",
            family="qwen",
            parameter_size="8B",
            quantization_level="4-bit",
        )
        path = tmp_path / "manifest.json"
        original.save(path)

        loaded = ModelManifest.load(path)
        assert loaded.name == original.name
        assert loaded.hf_path == original.hf_path
        assert loaded.size == original.size
        assert loaded.family == original.family

    def test_save_creates_parents(self, tmp_path):
        m = ModelManifest(name="test:latest", hf_path="test/model")
        path = tmp_path / "subdir" / "deep" / "manifest.json"
        m.save(path)
        assert path.exists()

    def test_compute_digest(self):
        digest = ModelManifest.compute_digest("qwen3:latest")
        assert digest.startswith("sha256:")
        assert len(digest) == len("sha256:") + 12

    def test_compute_digest_deterministic(self):
        d1 = ModelManifest.compute_digest("test")
        d2 = ModelManifest.compute_digest("test")
        assert d1 == d2

    def test_compute_digest_different(self):
        d1 = ModelManifest.compute_digest("model_a")
        d2 = ModelManifest.compute_digest("model_b")
        assert d1 != d2

    def test_load_ignores_extra_fields(self, tmp_path):
        path = tmp_path / "manifest.json"
        data = {
            "name": "test:latest",
            "hf_path": "test/model",
            "extra_field": "should be ignored",
        }
        path.write_text(json.dumps(data))
        m = ModelManifest.load(path)
        assert m.name == "test:latest"

    def test_load_coerces_null_strings(self, tmp_path):
        """Null values for str fields should be coerced to empty strings."""
        path = tmp_path / "manifest.json"
        data = {
            "name": "test:latest",
            "hf_path": "test/model",
            "parameter_size": None,
            "quantization_level": None,
            "family": None,
            "format": None,
        }
        path.write_text(json.dumps(data))
        m = ModelManifest.load(path)
        assert m.parameter_size == ""
        assert m.quantization_level == ""
        assert m.family == ""
        assert m.format == "mlx"

    def test_load_raises_on_null_required_fields(self, tmp_path):
        """Null values for required str fields (name, hf_path) should raise ValueError."""
        import pytest

        path = tmp_path / "manifest.json"
        data = {
            "name": None,
            "hf_path": "test/model",
        }
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="name"):
            ModelManifest.load(path)

        data = {
            "name": "test:latest",
            "hf_path": None,
        }
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="hf_path"):
            ModelManifest.load(path)

    def test_load_raises_on_missing_required_fields(self, tmp_path):
        """Missing required fields (name, hf_path) should raise ValueError."""
        import pytest

        path = tmp_path / "manifest.json"
        # Missing 'name' entirely
        data = {"hf_path": "test/model"}
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="name"):
            ModelManifest.load(path)

        # Missing 'hf_path' entirely
        data = {"name": "test:latest"}
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="hf_path"):
            ModelManifest.load(path)

    def test_load_coerces_null_int_fields(self, tmp_path):
        """Null values for int fields (e.g. size) should be coerced to their default."""
        path = tmp_path / "manifest.json"
        data = {
            "name": "test:latest",
            "hf_path": "test/model",
            "size": None,
        }
        path.write_text(json.dumps(data))
        m = ModelManifest.load(path)
        assert m.size == 0
