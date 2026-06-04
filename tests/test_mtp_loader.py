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
