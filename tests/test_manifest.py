"""Tests for mlx_ollama.models.manifest."""

import json

from mlx_ollama.models.manifest import ModelManifest


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
