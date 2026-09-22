"""Image generation engine: mflux adapter + inference.generate_image (#723)."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmlx.engine import image_gen
from olmlx.engine.image_gen import ImageGenerationCancelled
from olmlx.engine.model_manager import LoadedModel


class _FakeRegistry:
    """Mirrors mflux's CallbackRegistry.register / in_loop list."""

    def __init__(self):
        self.in_loop = []

    def register(self, cb):
        if hasattr(cb, "call_in_loop"):
            self.in_loop.append(cb)


class _FakeMfluxModel:
    """Runs a fake denoise loop that invokes in-loop callbacks per step."""

    def __init__(self, steps=3, on_step=None):
        self.callbacks = _FakeRegistry()
        self.prompt_cache = {}
        self.steps = steps
        self.on_step = on_step
        self.calls = []
        self.steps_run = 0

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        self.prompt_cache[kwargs["prompt"]] = "embeds"
        for t in range(self.steps):
            if self.on_step is not None:
                self.on_step(t)
            self.steps_run += 1
            for cb in self.callbacks.in_loop:
                cb.call_in_loop(
                    t=t,
                    seed=kwargs["seed"],
                    prompt=kwargs["prompt"],
                    latents=None,
                    config=None,
                    time_steps=None,
                )
        result = MagicMock()
        result.image = "PIL-IMAGE"
        return result


class TestAdapterGenerate:
    def test_unset_knobs_are_omitted(self):
        # Omitting steps/guidance lets each variant apply its own default
        # (QwenImage: 4 / 4.0, QwenImage21: 40 / 1.0).
        m = _FakeMfluxModel()
        img = image_gen.generate_image(m, "a cat", seed=1, width=512, height=256)
        assert img == "PIL-IMAGE"
        assert m.calls == [{"seed": 1, "prompt": "a cat", "width": 512, "height": 256}]

    def test_knobs_forwarded(self):
        m = _FakeMfluxModel()
        image_gen.generate_image(
            m,
            "a cat",
            seed=1,
            width=512,
            height=512,
            steps=8,
            guidance=3.5,
            negative_prompt="blurry",
        )
        call = m.calls[0]
        assert call["num_inference_steps"] == 8
        assert call["guidance"] == 3.5
        assert call["negative_prompt"] == "blurry"

    def test_cancel_mid_loop_raises_and_unregisters(self):
        ev = threading.Event()
        m = _FakeMfluxModel(steps=10, on_step=lambda t: ev.set() if t == 2 else None)
        with pytest.raises(ImageGenerationCancelled):
            image_gen.generate_image(
                m, "p", seed=0, width=64, height=64, cancel_event=ev
            )
        assert m.steps_run == 3  # stopped right after the step that set it
        assert m.callbacks.in_loop == []  # callback removed even on failure

    def test_cancel_before_start(self):
        ev = threading.Event()
        ev.set()
        m = _FakeMfluxModel()
        with pytest.raises(ImageGenerationCancelled):
            image_gen.generate_image(
                m, "p", seed=0, width=64, height=64, cancel_event=ev
            )
        assert m.calls == []

    def test_callback_removed_after_success(self):
        m = _FakeMfluxModel()
        image_gen.generate_image(
            m, "p", seed=0, width=64, height=64, cancel_event=threading.Event()
        )
        assert m.callbacks.in_loop == []

    def test_prompt_cache_cleared(self):
        m = _FakeMfluxModel()
        image_gen.generate_image(m, "p", seed=0, width=64, height=64)
        assert m.prompt_cache == {}


def _image_lm(model=None):
    return LoadedModel(
        name="qwen-image:2.1",
        hf_path="Qwen/Qwen-Image-2.1",
        model=model or _FakeMfluxModel(),
        tokenizer=None,
        is_image=True,
    )


def _manager(lm):
    """Mock manager whose ensure_loaded(pin=True) takes a ref like the real one."""

    async def _ensure_loaded(name, keep_alive=None, *, pin=False):
        if pin:
            lm.acquire_ref()
        return lm

    mgr = MagicMock()
    mgr.ensure_loaded = AsyncMock(side_effect=_ensure_loaded)
    return mgr


class TestInferenceGenerateImage:
    @pytest.mark.asyncio
    async def test_rejects_non_image_model(self):
        from olmlx.engine.inference import generate_image

        lm = LoadedModel(name="qwen3", hf_path="q", model=MagicMock(), tokenizer=None)
        lm.release_ref = MagicMock()
        with pytest.raises(ValueError, match="not an image model"):
            await generate_image(_manager(lm), "qwen3", "a cat", width=64, height=64)
        lm.release_ref.assert_called_once()

    @pytest.mark.asyncio
    async def test_generates_with_seed(self):
        from olmlx.engine.inference import generate_image

        lm = _image_lm()
        mgr = _manager(lm)
        out = await generate_image(
            mgr, "qwen-image:2.1", "a cat", width=64, height=64, seed=42, steps=2
        )
        assert out["image"] == "PIL-IMAGE"
        assert out["seed"] == 42
        assert lm.model.calls[0]["seed"] == 42
        assert lm.model.calls[0]["num_inference_steps"] == 2
        mgr.ensure_loaded.assert_awaited_once()
        assert mgr.ensure_loaded.call_args.kwargs.get("pin") is True
        assert lm.active_refs == 0

    @pytest.mark.asyncio
    async def test_random_seed_when_unset(self):
        from olmlx.engine.inference import generate_image

        lm = _image_lm()
        out = await generate_image(_manager(lm), "m", "a cat", width=64, height=64)
        assert isinstance(out["seed"], int)
        assert lm.model.calls[0]["seed"] == out["seed"]

    @pytest.mark.asyncio
    async def test_runs_off_the_event_loop(self):
        from olmlx.engine.inference import generate_image

        loop_thread = threading.get_ident()
        seen = []
        lm = _image_lm(
            _FakeMfluxModel(on_step=lambda t: seen.append(threading.get_ident()))
        )
        await generate_image(_manager(lm), "m", "p", width=64, height=64)
        assert seen and all(tid != loop_thread for tid in seen)

    @pytest.mark.asyncio
    async def test_external_cancel_event_stops_generation(self):
        from olmlx.engine.inference import generate_image

        ev = threading.Event()
        lm = _image_lm(_FakeMfluxModel(steps=50, on_step=lambda t: ev.set()))
        with pytest.raises(ImageGenerationCancelled):
            await generate_image(
                _manager(lm), "m", "p", width=64, height=64, cancel_event=ev
            )
        assert lm.model.steps_run == 1
        assert lm.active_refs == 0

    @pytest.mark.asyncio
    async def test_task_cancel_stops_worker_before_releasing_lock(self):
        # Cancelling the request coroutine must signal the worker AND wait for
        # it to stop before the inference lock is released — otherwise the
        # next request would run on Metal concurrently with the dying loop.
        from olmlx.engine import inference

        started = threading.Event()
        finished = threading.Event()

        def on_step(t):
            started.set()
            # Slow steps: without the cancel signal this would run ~5s.
            threading.Event().wait(0.05)

        model = _FakeMfluxModel(steps=100, on_step=on_step)
        orig = model.generate_image

        def gen(**kw):
            try:
                return orig(**kw)
            finally:
                finished.set()

        model.generate_image = gen
        lm = _image_lm(model)
        task = asyncio.create_task(
            inference.generate_image(_manager(lm), "m", "p", width=64, height=64)
        )
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The coroutine only finished after the worker did.
        assert finished.is_set()
        assert model.steps_run < 100
        assert not inference._get_inference_lock().locked()

    @pytest.mark.asyncio
    async def test_backend_failure_is_runtime_error(self):
        from olmlx.engine.inference import ImageGenerationError, generate_image

        model = _FakeMfluxModel(
            on_step=lambda t: (_ for _ in ()).throw(ValueError("shape"))
        )
        lm = _image_lm(model)
        with pytest.raises(ImageGenerationError) as ei:
            await generate_image(_manager(lm), "m", "p", width=64, height=64)
        # A backend crash is never the client's fault: RuntimeError -> 500,
        # not the ValueError -> 400 mapping (same contract as TTS, #703).
        assert isinstance(ei.value, RuntimeError)
        assert not isinstance(ei.value, ValueError)


class TestEncode:
    @pytest.mark.parametrize(
        "fmt,magic", [("png", b"\x89PNG"), ("jpeg", b"\xff\xd8"), ("webp", b"RIFF")]
    )
    def test_encode_formats(self, fmt, magic):
        from PIL import Image

        img = Image.new("RGB", (8, 8), (255, 0, 0))
        data = image_gen.encode_image(img, fmt)
        assert data.startswith(magic)

    def test_encode_rgba_to_jpeg(self):
        from PIL import Image

        img = Image.new("RGBA", (8, 8))
        assert image_gen.encode_image(img, "jpeg").startswith(b"\xff\xd8")


def test_patch_target_exists():
    # Router tests patch this name.
    with patch("olmlx.routers.images.generate_image"):
        pass
