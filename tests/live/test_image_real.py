"""Live mflux text-to-image test against a real model (#723).

Drives the real load path — the olmlx ModelStore directory under
``OLMLX_MODELS_DIR`` handed to mflux — loading on one thread and generating on
another, the cross-thread shape the server uses (``asyncio.to_thread`` load
vs. generation) and that mflux's single-threaded CLI never exercises. Skips
cleanly when mflux (the [image] extra) is missing or the model is not already
in the store (no forced multi-GB pull). Runs with ``HF_HUB_OFFLINE=1`` so any
attempt to reach the Hub / HF cache instead of the store fails loudly.
"""

import json
import threading

import pytest

pytestmark = [pytest.mark.real_model]

REPO = "Qwen/Qwen-Image-2.1"


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


def test_generate_cross_thread_from_store(tmp_path, monkeypatch):
    pytest.importorskip("mflux")

    from olmlx.engine import image_gen
    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.registry import ModelRegistry
    from olmlx.models.store import ModelStore

    cfg = tmp_path / "models.json"
    cfg.write_text(
        json.dumps(
            {"qwen-image:2.1": {"type": "image", "hf_path": REPO, "image_quantize": 8}}
        )
    )
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", cfg)
    registry = ModelRegistry()
    registry.load()
    store = ModelStore(registry)  # real OLMLX_MODELS_DIR
    if not store.is_downloaded(REPO):
        pytest.skip(f"{REPO} not in the model store; `olmlx models pull` it first")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    mgr = ModelManager(registry, store)
    mc = registry.resolve("qwen-image:2.1")
    model, *_ = _on_thread(lambda: mgr._load_model(REPO, image_config=mc))
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
