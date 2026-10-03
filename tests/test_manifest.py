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

    def test_load_raises_on_type_mismatch(self, tmp_path):
        """Non-null wrong types should raise ValueError."""
        import pytest

        path = tmp_path / "manifest.json"
        # size should be int, not str
        data = {"name": "test:latest", "hf_path": "test/model", "size": "2gb"}
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="size"):
            ModelManifest.load(path)

        # name should be str, not int
        data = {"name": 123, "hf_path": "test/model"}
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="name"):
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

    def test_load_missing_estimator_version_defaults_to_zero(self, tmp_path):
        """Pre-#702 manifests carry no ``estimator_version``; they must load
        with version 0 so the store flags them as stale (#702)."""
        path = tmp_path / "manifest.json"
        data = {
            "name": "test:latest",
            "hf_path": "test/model",
            "parameter_size": "77M",
        }
        path.write_text(json.dumps(data))
        m = ModelManifest.load(path)
        assert m.estimator_version == 0
        assert m.parameter_size == "77M"

    def test_save_is_atomic_on_write_failure(self, tmp_path, monkeypatch):
        """A crash mid-write must leave the previous manifest intact rather
        than a truncated file (list_local/show now write on read, #702)."""
        path = tmp_path / "manifest.json"
        ModelManifest(name="old:latest", hf_path="a/b", parameter_size="77M").save(path)
        before = path.read_bytes()

        real_dump = json.dump

        def _partial_dump(obj, f, **kw):
            f.write('{"name": "trunc')
            raise OSError("disk full")

        monkeypatch.setattr(json, "dump", _partial_dump)
        try:
            ModelManifest(name="new:latest", hf_path="a/b").save(path)
        except OSError:
            pass
        monkeypatch.setattr(json, "dump", real_dump)

        assert path.read_bytes() == before
        assert ModelManifest.load(path).parameter_size == "77M"
        # No stray temp files left behind.
        assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]

    def test_save_new_file_in_public_dir_is_world_readable(self, tmp_path):
        d = tmp_path / "public"
        d.mkdir()
        d.chmod(0o755)
        path = d / "manifest.json"
        ModelManifest(name="a:latest", hf_path="a/b").save(path)
        assert path.stat().st_mode & 0o777 == 0o644

    def test_save_preserves_existing_file_mode(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text("{}")
        path.chmod(0o640)
        ModelManifest(name="a:latest", hf_path="a/b").save(path)
        assert path.stat().st_mode & 0o777 == 0o640

    def test_save_new_file_honours_directory_mode(self, tmp_path):
        """A private (0700) store dir gets private (0600) manifests, as the old
        umask-honouring open(path, "w") would have produced."""
        d = tmp_path / "private"
        d.mkdir(mode=0o700)
        d.chmod(0o700)
        path = d / "manifest.json"
        ModelManifest(name="a:latest", hf_path="a/b").save(path)
        assert path.stat().st_mode & 0o777 == 0o600

    def test_save_closes_fd_when_chmod_fails(self, tmp_path, monkeypatch):
        import os
        import tempfile

        import pytest

        fds = []
        real_mkstemp = tempfile.mkstemp

        def _rec(*a, **kw):
            fd, name = real_mkstemp(*a, **kw)
            fds.append(fd)
            return fd, name

        def _eperm(*a, **kw):
            raise PermissionError("EPERM")

        monkeypatch.setattr(tempfile, "mkstemp", _rec)
        monkeypatch.setattr(os, "fchmod", _eperm)
        monkeypatch.setattr(os, "chmod", _eperm)
        with pytest.raises(PermissionError):
            ModelManifest(name="a:latest", hf_path="a/b").save(
                tmp_path / "manifest.json"
            )
        assert fds
        with pytest.raises(OSError):
            os.fstat(fds[0])
        assert list(tmp_path.iterdir()) == []

    def test_save_closes_fd_when_fdopen_fails(self, tmp_path, monkeypatch):
        import os
        import tempfile

        import pytest

        fds = []
        real_mkstemp = tempfile.mkstemp

        def _rec(*a, **kw):
            fd, name = real_mkstemp(*a, **kw)
            fds.append(fd)
            return fd, name

        def _fail(*a, **kw):
            raise OSError("fdopen failed")

        monkeypatch.setattr(tempfile, "mkstemp", _rec)
        monkeypatch.setattr(os, "fdopen", _fail)
        with pytest.raises(OSError):
            ModelManifest(name="a:latest", hf_path="a/b").save(
                tmp_path / "manifest.json"
            )
        with pytest.raises(OSError):
            os.fstat(fds[0])
        assert list(tmp_path.iterdir()) == []
