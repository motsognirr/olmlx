"""Tests for the unmarked-real-model guard (#470).

CI deselects ``-m "not real_model"`` and the integration suite autouse-mocks
MLX, so a test that loads a real model *without* the marker either downloads
gigabytes in CI or silently passes against a mock (a false positive — the
exact gap that let the VLM image-drop bug through, #429). The autouse guard
in ``tests/conftest.py`` patches the real loading/downloading entry points
to fail the offending test with an actionable message.
"""

import huggingface_hub
import mlx_lm
import mlx_vlm
import pytest


class TestRealModelGuard:
    def test_mlx_lm_load_blocked_without_marker(self, tmp_path):
        with pytest.raises(pytest.fail.Exception, match="real_model"):
            mlx_lm.load(str(tmp_path / "no-such-model"))

    def test_mlx_vlm_load_blocked_without_marker(self, tmp_path):
        with pytest.raises(pytest.fail.Exception, match="real_model"):
            mlx_vlm.load(str(tmp_path / "no-such-model"))

    def test_snapshot_download_blocked_without_marker(self):
        # Nonexistent repo id: if the guard ever regresses, this fails
        # fast on a 404 instead of downloading a real model.
        with pytest.raises(pytest.fail.Exception, match="real_model"):
            huggingface_hub.snapshot_download("olmlx-tests/no-such-repo")

    def test_hf_hub_download_raises_offline_error_without_marker(self):
        # Metadata fetches (config.json, chat templates) have designed
        # except-Exception fallbacks at every production call site, so
        # the guard raises the offline-mode error instead of failing
        # the test — fallback tests stay meaningful without a real
        # network 404 per test.
        from huggingface_hub.errors import LocalEntryNotFoundError

        with pytest.raises(LocalEntryNotFoundError, match="real-model guard"):
            huggingface_hub.hf_hub_download("olmlx-tests/no-such-repo", "config.json")

    def test_model_info_raises_offline_error_without_marker(self):
        from huggingface_hub.errors import LocalEntryNotFoundError

        with pytest.raises(LocalEntryNotFoundError, match="real-model guard"):
            huggingface_hub.model_info("olmlx-tests/no-such-repo")

    def test_guard_is_installed_for_unmarked_tests(self):
        # The patched callables are tagged so this test (and humans
        # debugging a confusing failure) can tell guard from real.
        assert getattr(mlx_lm.load, "_olmlx_real_model_guard", False)
        assert getattr(mlx_vlm.load, "_olmlx_real_model_guard", False)
        assert getattr(
            huggingface_hub.snapshot_download, "_olmlx_real_model_guard", False
        )

    @pytest.mark.real_model
    def test_marked_test_bypasses_guard(self):
        # Deselected in CI; validates locally that the marker restores
        # the real entry points.
        assert not getattr(mlx_lm.load, "_olmlx_real_model_guard", False)
        assert not getattr(mlx_vlm.load, "_olmlx_real_model_guard", False)
        assert not getattr(
            huggingface_hub.snapshot_download, "_olmlx_real_model_guard", False
        )
