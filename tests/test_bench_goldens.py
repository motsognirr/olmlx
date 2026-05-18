"""Tests for olmlx.bench.goldens."""

from __future__ import annotations

import pytest

from olmlx.bench.goldens import (
    golden_path,
    goldens_dir,
    load_golden,
    save_golden,
)


class TestPaths:
    def test_goldens_dir_sanitizes_model(self, tmp_path):
        d = goldens_dir(tmp_path, "mlx-community/Qwen3-8B")
        assert d.parent == tmp_path / "goldens"
        # '/' is not allowed by _safe_dir_name — must be replaced.
        assert "/" not in d.name
        assert "Qwen3-8B" in d.name

    def test_golden_path_includes_prompt_name(self, tmp_path):
        p = golden_path(tmp_path, "m", "factual")
        assert p.name == "factual.txt"
        assert p.parent == goldens_dir(tmp_path, "m")


class TestRoundTrip:
    def test_save_then_load(self, tmp_path):
        path = save_golden(tmp_path, "m", "factual", "Paris.")
        assert path.exists()
        assert load_golden(tmp_path, "m", "factual") == "Paris."

    def test_save_creates_parent_dirs(self, tmp_path):
        save_golden(tmp_path, "mlx-community/Qwen3-8B", "factual", "x")
        d = goldens_dir(tmp_path, "mlx-community/Qwen3-8B")
        assert d.exists()

    def test_save_refuses_overwrite_by_default(self, tmp_path):
        save_golden(tmp_path, "m", "p", "v1")
        with pytest.raises(FileExistsError):
            save_golden(tmp_path, "m", "p", "v2")

    def test_save_force_overwrites(self, tmp_path):
        save_golden(tmp_path, "m", "p", "v1")
        save_golden(tmp_path, "m", "p", "v2", force=True)
        assert load_golden(tmp_path, "m", "p") == "v2"

    def test_load_missing_returns_none(self, tmp_path):
        assert load_golden(tmp_path, "m", "absent") is None

    def test_prompt_name_with_unsafe_chars(self, tmp_path):
        # Prompt names are arbitrary strings; sanitize when used as a filename.
        save_golden(tmp_path, "m", "multi/turn name", "x")
        assert load_golden(tmp_path, "m", "multi/turn name") == "x"
