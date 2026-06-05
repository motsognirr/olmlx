import json
from pathlib import Path

import pytest

from olmlx.engine.mtp.draft_model import MTPConfig, MTPDraftModel, load_mtp_draft

_HEAD = "mlx-community/Qwen3.6-27B-MTP-4bit"


def _head_dir() -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(_HEAD))
    except Exception:
        return None


@pytest.mark.skipif(_head_dir() is None, reason="MTP head not downloadable")
def test_load_mtp_draft_strict_no_leftover_keys():
    path = _head_dir()
    assert path is not None  # guarded by skipif above; narrows for the type checker
    cfg = MTPConfig.from_dict(json.loads((path / "config.json").read_text()))
    draft = load_mtp_draft(path, cfg)
    assert isinstance(draft, MTPDraftModel)
    assert cfg.num_experts == 0  # 27B head is dense


_MOE_HEAD = "mlx-community/Qwen3.6-35B-A3B-MTP-4bit"


def _moe_head_dir() -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(_MOE_HEAD))
    except Exception:
        return None


@pytest.mark.skipif(_moe_head_dir() is None, reason="MoE MTP head not downloadable")
def test_load_mtp_draft_moe_strict():
    path = _moe_head_dir()
    assert path is not None
    cfg = MTPConfig.from_dict(json.loads((path / "config.json").read_text()))
    assert cfg.is_moe
    draft = load_mtp_draft(path, cfg)
    assert isinstance(draft, MTPDraftModel)


def test_mtp_in_flash_moe_incompatible_set():
    """MTP must be rejected under flash_moe, like eagle/dflash."""
    from olmlx.engine.model_manager import _FLASH_MOE_INCOMPATIBLE_STRATEGIES

    assert "mtp" in _FLASH_MOE_INCOMPATIBLE_STRATEGIES
    assert {"dflash", "eagle"} <= _FLASH_MOE_INCOMPATIBLE_STRATEGIES


def test_mtp_loader_rejects_wrong_model_type(tmp_path, monkeypatch):
    """_load_mtp_decoder rejects a draft repo that isn't a qwen3_5_mtp head."""
    import json

    from olmlx.engine import model_manager as mm

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    mgr = mm.ModelManager.__new__(mm.ModelManager)
    monkeypatch.setattr(
        mgr, "_resolve_draft_path", lambda p: str(tmp_path), raising=False
    )

    class _Cfg:
        enabled = True
        draft_model = "some/not-an-mtp-head"
        num_tokens = None

    with pytest.raises(ValueError, match="qwen3_5_mtp"):
        mgr._load_mtp_decoder(object(), _Cfg())
