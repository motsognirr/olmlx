"""Live mflux text-to-image test against a real model (#723).

Loads on one thread and generates on another — the cross-thread shape the
server uses (``asyncio.to_thread`` load vs. generation) and that mflux's
single-threaded CLI never exercises. Skips cleanly when mflux (the [image]
extra) or the weights are not already present (no forced multi-GB pull).
"""

import threading

import pytest

pytestmark = [pytest.mark.real_model]

REPO = "Qwen/Qwen-Image-2.1"


def _cached_or_skip(repo: str) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(repo, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip(f"{repo} not downloaded; skipping live image test")


def _on_thread(fn):
    out: dict = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            out["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if "error" in out:
        raise out["error"]
    return out["value"]


def test_generate_cross_thread(tmp_path, monkeypatch):
    import json

    pytest.importorskip("mflux")
    _cached_or_skip(REPO)

    from olmlx.engine import image_gen
    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.registry import ModelRegistry

    cfg = tmp_path / "models.json"
    cfg.write_text(
        json.dumps(
            {"qwen-image:2.1": {"type": "image", "hf_path": REPO, "image_quantize": 8}}
        )
    )
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", cfg)
    registry = ModelRegistry()
    registry.load()
    mgr = ModelManager.__new__(ModelManager)
    mgr.registry = registry
    mgr.store = None

    model, *_ = _on_thread(lambda: mgr._load_model_image(REPO))
    image = _on_thread(
        lambda: image_gen.generate_image(
            model,
            "a red apple on a white table",
            seed=0,
            width=256,
            height=256,
            steps=2,
        )
    )
    assert image.size == (256, 256)
    assert len(image_gen.encode_image(image, "png")) > 100
