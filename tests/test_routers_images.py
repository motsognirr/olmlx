"""/v1/images/generations router (#723)."""

import base64
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from olmlx.engine.image_gen import ImageGenerationCancelled


def _fake_out(seed=7):
    return {"image": Image.new("RGB", (16, 16), (0, 128, 255)), "seed": seed}


class TestImagesRouter:
    @pytest.mark.asyncio
    async def test_generates_b64_png(self, app_client):
        with patch(
            "olmlx.routers.images.generate_image", new_callable=AsyncMock
        ) as mock:
            mock.return_value = _fake_out(seed=123)
            resp = await app_client.post(
                "/v1/images/generations",
                json={
                    "model": "qwen-image:2.1",
                    "prompt": "a lighthouse at dusk",
                    "size": "512x768",
                    "seed": 123,
                    "steps": 8,
                    "guidance": 2.5,
                    "negative_prompt": "blurry",
                    "quality": "hd",  # OpenAI field: accepted and ignored
                },
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert isinstance(body["created"], int)
        assert body["output_format"] == "png"
        assert body["size"] == "512x768"
        assert len(body["data"]) == 1
        assert body["data"][0]["seed"] == 123
        assert base64.b64decode(body["data"][0]["b64_json"]).startswith(b"\x89PNG")
        args, kwargs = mock.call_args
        assert args[1:] == ("qwen-image:2.1", "a lighthouse at dusk")
        assert kwargs["width"] == 512 and kwargs["height"] == 768
        assert kwargs["seed"] == 123
        assert kwargs["steps"] == 8
        assert kwargs["guidance"] == 2.5
        assert kwargs["negative_prompt"] == "blurry"
        assert kwargs["cancel_event"] is not None

    @pytest.mark.asyncio
    async def test_jpeg_output(self, app_client):
        with patch(
            "olmlx.routers.images.generate_image", new_callable=AsyncMock
        ) as mock:
            mock.return_value = _fake_out()
            resp = await app_client.post(
                "/v1/images/generations",
                json={"model": "m", "prompt": "p", "output_format": "jpeg"},
            )
        assert resp.status_code == 200
        raw = base64.b64decode(resp.json()["data"][0]["b64_json"])
        assert raw.startswith(b"\xff\xd8")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            {"prompt": "   "},
            {"n": 2},
            {"response_format": "url"},
            {"size": "1024"},
            {"size": "1000x1024"},  # not a multiple of 16
            {"size": "32x32"},
            {"steps": 0},
            {"guidance": -1},
            {"seed": -1},
            {"output_format": "gif"},
        ],
    )
    async def test_rejects_invalid_requests(self, app_client, payload):
        body = {"model": "m", "prompt": "p", **payload}
        with patch(
            "olmlx.routers.images.generate_image", new_callable=AsyncMock
        ) as mock:
            resp = await app_client.post("/v1/images/generations", json=body)
        assert resp.status_code == 400, resp.text
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_size_over_limit(self, app_client, monkeypatch):
        monkeypatch.setattr("olmlx.routers.images.settings.image_max_dimension", 1024)
        with patch(
            "olmlx.routers.images.generate_image", new_callable=AsyncMock
        ) as mock:
            resp = await app_client.post(
                "/v1/images/generations",
                json={"model": "m", "prompt": "p", "size": "2048x1024"},
            )
        assert resp.status_code == 400
        assert "OLMLX_IMAGE_MAX_DIMENSION" in resp.text
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_image_model_is_400(self, app_client):
        with patch(
            "olmlx.routers.images.generate_image",
            new_callable=AsyncMock,
            side_effect=ValueError("Model 'qwen3' is not an image model."),
        ):
            resp = await app_client.post(
                "/v1/images/generations", json={"model": "qwen3", "prompt": "p"}
            )
        assert resp.status_code == 400
        assert "not an image model" in resp.text

    @pytest.mark.asyncio
    async def test_backend_failure_is_500(self, app_client):
        from olmlx.engine.inference import ImageGenerationError

        with patch(
            "olmlx.routers.images.generate_image",
            new_callable=AsyncMock,
            side_effect=ImageGenerationError("ValueError: shape mismatch"),
        ):
            resp = await app_client.post(
                "/v1/images/generations", json={"model": "m", "prompt": "p"}
            )
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_cancelled_generation_returns_499(self, app_client):
        with patch(
            "olmlx.routers.images.generate_image",
            new_callable=AsyncMock,
            side_effect=ImageGenerationCancelled("cancelled"),
        ):
            resp = await app_client.post(
                "/v1/images/generations", json={"model": "m", "prompt": "p"}
            )
        assert resp.status_code == 499

    @pytest.mark.asyncio
    async def test_disconnect_sets_cancel_event(self):
        # The watcher flips the cancel event the worker polls each step.
        import asyncio
        import threading
        from unittest.mock import MagicMock

        from olmlx.routers.images import _watch_disconnect

        request = MagicMock()
        request.receive = AsyncMock(
            side_effect=[{"type": "http.request"}, {"type": "http.disconnect"}]
        )
        cancel = threading.Event()
        await asyncio.wait_for(_watch_disconnect(request, cancel), 2)
        assert cancel.is_set()


class TestImageModelListing:
    @pytest.mark.asyncio
    async def test_tags_marks_image_family(self, app_client, registry):
        from olmlx.engine.registry import ModelConfig

        registry._mappings["qwen-image:2.1"] = ModelConfig.from_entry(
            {"type": "image", "hf_path": "Qwen/Qwen-Image-2.1"}
        )
        resp = await app_client.get("/api/tags")
        assert resp.status_code == 200
        by_name = {m["name"]: m for m in resp.json()["models"]}
        assert by_name["qwen-image:2.1"]["details"]["family"] == "image"
        assert by_name["qwen3:latest"]["details"].get("family") != "image"

    @pytest.mark.asyncio
    async def test_v1_models_lists_image_model(self, app_client, registry):
        from olmlx.engine.registry import ModelConfig

        registry._mappings["qwen-image:2.1"] = ModelConfig.from_entry(
            {"type": "image", "hf_path": "Qwen/Qwen-Image-2.1"}
        )
        resp = await app_client.get("/v1/models")
        assert "qwen-image:2.1" in {m["id"] for m in resp.json()["data"]}


class TestDisconnectThroughRealStack:
    @pytest.mark.asyncio
    async def test_client_disconnect_sets_cancel_event(self):
        """End-to-end over a real socket + the app's real middleware stack.

        ``request.is_disconnected()`` never reports a disconnect behind the
        app's ``BaseHTTPMiddleware``s (it receives under a pre-cancelled scope,
        which drops the message), so a mocked-request test can't catch a
        regression here — this drives uvicorn and an actually-closed client.
        """
        import asyncio
        import socket
        import threading

        import httpx
        import uvicorn

        from olmlx.app import create_app

        fired = threading.Event()

        async def slow_generate(manager, model, prompt, *, cancel_event, **kw):
            for _ in range(200):
                if cancel_event.is_set():
                    fired.set()
                    raise ImageGenerationCancelled("cancelled")
                await asyncio.sleep(0.05)
            raise AssertionError("cancel never fired")

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        app = create_app()
        app.state.model_manager = object()
        server = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=port, log_level="error", lifespan="off"
            )
        )
        with patch("olmlx.routers.images.generate_image", slow_generate):
            serve = asyncio.create_task(server.serve())
            try:
                while not server.started:
                    await asyncio.sleep(0.02)
                async with httpx.AsyncClient() as client:
                    with pytest.raises(httpx.TimeoutException):
                        await client.post(
                            f"http://127.0.0.1:{port}/v1/images/generations",
                            json={"model": "m", "prompt": "p"},
                            timeout=0.5,
                        )
                for _ in range(100):
                    if fired.is_set():
                        break
                    await asyncio.sleep(0.05)
                assert fired.is_set(), "client disconnect did not cancel generation"
            finally:
                server.should_exit = True
                await serve


class TestPromptLimit:
    @pytest.mark.asyncio
    async def test_oversized_prompt_is_413(self, app_client, monkeypatch):
        monkeypatch.setattr("olmlx.routers.images.settings.image_max_prompt_chars", 10)
        with patch(
            "olmlx.routers.images.generate_image", new_callable=AsyncMock
        ) as mock:
            resp = await app_client.post(
                "/v1/images/generations", json={"model": "m", "prompt": "x" * 11}
            )
        assert resp.status_code == 413
        assert "OLMLX_IMAGE_MAX_PROMPT_CHARS" in resp.text
        mock.assert_not_called()


class TestNegativePromptLimit:
    @pytest.mark.asyncio
    async def test_oversized_negative_prompt_is_413(self, app_client, monkeypatch):
        monkeypatch.setattr("olmlx.routers.images.settings.image_max_prompt_chars", 10)
        with patch(
            "olmlx.routers.images.generate_image", new_callable=AsyncMock
        ) as mock:
            resp = await app_client.post(
                "/v1/images/generations",
                json={"model": "m", "prompt": "p", "negative_prompt": "x" * 11},
            )
        assert resp.status_code == 413
        mock.assert_not_called()


class TestHandlerCancellationPropagates:
    @pytest.mark.asyncio
    async def test_cancel_during_watcher_cleanup_is_not_swallowed(self):
        """A cancel of the handler that lands while the finally awaits the
        watcher must propagate, not be suppressed as if it were the watcher's
        own CancelledError (which would carry on to encode + respond)."""
        import asyncio
        import threading
        from unittest.mock import MagicMock

        from olmlx.routers import images as images_router
        from olmlx.schemas.images import ImageGenerationRequest

        in_cleanup = asyncio.Event()

        async def slow_to_stop_watcher(request, cancel: threading.Event):
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                in_cleanup.set()
                await asyncio.sleep(0.5)  # slow shutdown: widens the window
                raise

        async def quick_generate(*a, **kw):
            await asyncio.sleep(0.05)  # let the watcher task start
            return {"image": object(), "seed": 1}

        request = MagicMock()
        request.app.state.model_manager = object()
        encode = MagicMock(return_value=b"x")
        with (
            patch.object(images_router, "generate_image", quick_generate),
            patch.object(images_router, "_watch_disconnect", slow_to_stop_watcher),
            patch.object(images_router, "encode_image", encode),
        ):
            task = asyncio.create_task(
                images_router.images_generations(
                    ImageGenerationRequest(model="m", prompt="p"), request
                )
            )
            await asyncio.wait_for(in_cleanup.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        encode.assert_not_called()
