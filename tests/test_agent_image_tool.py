"""Tests for the agent ``generate_image`` tool (issue #725)."""

import asyncio
import threading

import pytest
from PIL import Image

from olmlx.chat.config import ChatConfig
from olmlx.chat.errors import ToolError
from olmlx.config import Settings
from olmlx.engine.agent.orchestrator import AgentContext
from olmlx.engine.agent.service import AgentService
from olmlx.engine.agent.store import AgentStore
from olmlx.engine.agent.tools import AgentImageTool, AgentToolManager
from olmlx.engine.image_gen import ImageGenerationCancelled
from olmlx.engine.registry import ModelConfig


@pytest.fixture
def store(tmp_path):
    s = AgentStore(tmp_path / "agent.db")
    yield s
    s.close()


@pytest.fixture
def context(store):
    return AgentContext(run_id="abcdef1234567890", store=store)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


class FakeGenerator:
    """Records calls and returns a small real PIL image."""

    def __init__(self, *, seed=42, exc=None):
        self.calls: list[dict] = []
        self.seed = seed
        self.exc = exc

    async def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.exc is not None:
            raise self.exc
        w, h = kwargs["width"], kwargs["height"]
        return {"image": Image.new("RGB", (w, h), (255, 0, 0)), "seed": self.seed}


def _tools(context, workspace, gen, **over):
    over.setdefault("max_dimension", 2048)
    over.setdefault("max_prompt_chars", 8192)
    return AgentToolManager(
        ChatConfig(model_name="m", write_root=workspace),
        context,
        image_tool=AgentImageTool(generate=gen, **over),
    )


class TestToolOffering:
    def test_not_offered_without_image_tool(self, context):
        tools = AgentToolManager(ChatConfig(model_name="m"), context)
        assert "generate_image" not in tools.tool_names
        names = {d["function"]["name"] for d in tools.get_tool_definitions()}
        assert "generate_image" not in names

    async def test_call_without_image_tool_is_not_dispatched(self, context):
        tools = AgentToolManager(ChatConfig(model_name="m"), context)
        result = await tools.call_tool("generate_image", {"prompt": "a cat"})
        # Falls through to the builtin manager, which doesn't know it.
        assert isinstance(result, ToolError) or "unknown" in str(result).lower()

    def test_offered_with_image_tool(self, context, workspace):
        tools = _tools(context, workspace, FakeGenerator())
        assert "generate_image" in tools.tool_names
        names = {d["function"]["name"] for d in tools.get_tool_definitions()}
        assert "generate_image" in names


class TestGenerateImage:
    async def test_writes_png_in_workspace_and_returns_path(self, context, workspace):
        gen = FakeGenerator(seed=7)
        tools = _tools(context, workspace, gen)
        result = await tools.call_tool(
            "generate_image", {"prompt": "a red square", "width": 64, "height": 128}
        )
        assert isinstance(result, str)
        expected = (workspace / "images" / "abcdef12-7.png").resolve()
        assert expected.exists()
        assert str(expected) in result
        assert "seed 7" in result
        with Image.open(expected) as im:
            assert im.format == "PNG"
            assert im.size == (64, 128)
        # No base64 payload leaks into the model's context.
        assert len(result) < 500
        assert gen.calls[0]["prompt"] == "a red square"
        assert gen.calls[0]["width"] == 64
        assert gen.calls[0]["height"] == 128
        assert isinstance(gen.calls[0]["cancel_event"], threading.Event)

    async def test_defaults_to_1024_square(self, context, workspace):
        gen = FakeGenerator()
        tools = _tools(context, workspace, gen)
        await tools.call_tool("generate_image", {"prompt": "x"})
        assert gen.calls[0]["width"] == 1024
        assert gen.calls[0]["height"] == 1024

    async def test_passes_optional_knobs(self, context, workspace):
        gen = FakeGenerator()
        tools = _tools(context, workspace, gen)
        await tools.call_tool(
            "generate_image",
            {
                "prompt": "x",
                "width": 64,
                "height": 64,
                "seed": 3,
                "steps": 4,
                "negative_prompt": "blurry",
            },
        )
        call = gen.calls[0]
        assert call["seed"] == 3
        assert call["steps"] == 4
        assert call["negative_prompt"] == "blurry"

    async def test_filename_honored_and_format_from_suffix(self, context, workspace):
        tools = _tools(context, workspace, FakeGenerator())
        result = await tools.call_tool(
            "generate_image",
            {"prompt": "x", "width": 64, "height": 64, "filename": "art/logo.jpg"},
        )
        assert isinstance(result, str)
        out = workspace / "art" / "logo.jpg"
        with Image.open(out) as im:
            assert im.format == "JPEG"

    async def test_filename_without_suffix_gets_png(self, context, workspace):
        tools = _tools(context, workspace, FakeGenerator())
        await tools.call_tool(
            "generate_image",
            {"prompt": "x", "width": 64, "height": 64, "filename": "logo"},
        )
        assert (workspace / "logo.png").exists()

    async def test_unsupported_suffix_rejected_before_generation(
        self, context, workspace
    ):
        gen = FakeGenerator()
        tools = _tools(context, workspace, gen)
        result = await tools.call_tool(
            "generate_image", {"prompt": "x", "filename": "logo.gif"}
        )
        assert isinstance(result, ToolError)
        assert gen.calls == []

    async def test_filename_outside_workspace_rejected(
        self, context, workspace, tmp_path
    ):
        gen = FakeGenerator()
        tools = _tools(context, workspace, gen)
        for bad in ("../escape.png", str(tmp_path / "abs.png")):
            result = await tools.call_tool(
                "generate_image", {"prompt": "x", "filename": bad}
            )
            assert isinstance(result, ToolError), bad
        assert gen.calls == []
        assert not (tmp_path / "escape.png").exists()
        assert not (tmp_path / "abs.png").exists()

    async def test_existing_file_not_overwritten(self, context, workspace):
        gen = FakeGenerator()
        (workspace / "keep.png").write_bytes(b"original")
        tools = _tools(context, workspace, gen)
        result = await tools.call_tool(
            "generate_image", {"prompt": "x", "filename": "keep.png"}
        )
        assert isinstance(result, ToolError)
        assert (workspace / "keep.png").read_bytes() == b"original"
        assert gen.calls == []

    async def test_default_name_collision_gets_suffix(self, context, workspace):
        tools = _tools(context, workspace, FakeGenerator(seed=7))
        args = {"prompt": "x", "width": 64, "height": 64}
        await tools.call_tool("generate_image", args)
        result = await tools.call_tool("generate_image", args)
        assert isinstance(result, str)
        files = sorted(p.name for p in (workspace / "images").iterdir())
        assert files == ["abcdef12-7-1.png", "abcdef12-7.png"]

    async def test_no_write_root_uses_cwd(self, context, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tools = AgentToolManager(
            ChatConfig(model_name="m"),
            context,
            image_tool=AgentImageTool(
                generate=FakeGenerator(seed=1), max_dimension=2048, max_prompt_chars=10
            ),
        )
        await tools.call_tool(
            "generate_image", {"prompt": "x", "width": 64, "height": 64}
        )
        assert (tmp_path / "images" / "abcdef12-1.png").exists()


class TestValidation:
    @pytest.mark.parametrize(
        "args",
        [
            {"prompt": ""},
            {"prompt": "   "},
            {},
            {"prompt": "x" * 11},
            {"prompt": "x", "width": 100},
            {"prompt": "x", "height": 32},
            {"prompt": "x", "width": 4096},
            {"prompt": "x", "width": "big"},
            {"prompt": "x", "steps": 0},
            {"prompt": "x", "steps": 500},
            {"prompt": "x", "seed": -1},
            {"prompt": "x", "negative_prompt": "y" * 11},
            {"prompt": "x", "width": float("inf")},
            {"prompt": "x", "seed": float("inf")},
            {"prompt": "x", "width": 64.5},
            {"prompt": "x", "width": True},
            {"prompt": "x", "filename": "renders/"},
            {"prompt": "x", "filename": ".."},
            {"prompt": 5},
        ],
    )
    async def test_invalid_args_rejected_before_generation(
        self, context, workspace, args
    ):
        gen = FakeGenerator()
        tools = _tools(context, workspace, gen, max_prompt_chars=10)
        result = await tools.call_tool("generate_image", args)
        assert isinstance(result, ToolError)
        assert result.is_user_error is True
        assert gen.calls == []


class TestDefaults:
    async def test_default_size_respects_small_max_dimension(self, context, workspace):
        gen = FakeGenerator()
        tools = _tools(context, workspace, gen, max_dimension=770)
        result = await tools.call_tool("generate_image", {"prompt": "x"})
        assert isinstance(result, str), result
        assert gen.calls[0]["width"] == 768
        assert gen.calls[0]["height"] == 768


class TestSandbox:
    async def test_symlinked_default_dir_cannot_escape(
        self, context, workspace, tmp_path
    ):
        outside = tmp_path / "outside"
        outside.mkdir()
        (workspace / "images").symlink_to(outside)
        tools = _tools(context, workspace, FakeGenerator())
        result = await tools.call_tool(
            "generate_image", {"prompt": "x", "width": 64, "height": 64}
        )
        assert isinstance(result, ToolError)
        assert list(outside.iterdir()) == []

    async def test_symlink_planted_during_generation_cannot_escape(
        self, context, workspace, tmp_path
    ):
        outside = tmp_path / "outside"
        outside.mkdir()

        async def gen(prompt, **kwargs):
            # Passed the pre-generation check; now redirect the target dir.
            (workspace / "art").symlink_to(outside)
            return {"image": Image.new("RGB", (64, 64)), "seed": 1}

        tools = AgentToolManager(
            ChatConfig(model_name="m", write_root=workspace),
            context,
            image_tool=AgentImageTool(
                generate=gen, max_dimension=2048, max_prompt_chars=100
            ),
        )
        result = await tools.call_tool(
            "generate_image", {"prompt": "x", "filename": "art/logo.png"}
        )
        assert isinstance(result, ToolError)
        assert list(outside.iterdir()) == []

    async def test_images_path_is_a_file(self, context, workspace):
        (workspace / "images").write_text("not a dir")
        tools = _tools(context, workspace, FakeGenerator())
        result = await tools.call_tool(
            "generate_image", {"prompt": "x", "width": 64, "height": 64}
        )
        assert isinstance(result, ToolError)
        assert "None" not in result.message


class TestErrors:
    async def test_backend_value_error_is_not_a_user_error(self, context, workspace):
        # Args are validated up front, so a ValueError from generate_image is
        # configuration (text model, missing extra) the model can't fix.
        gen = FakeGenerator(exc=ValueError("Model 'm' is not an image model."))
        tools = _tools(context, workspace, gen)
        result = await tools.call_tool("generate_image", {"prompt": "x"})
        assert isinstance(result, ToolError)
        assert result.is_user_error is False
        assert "not an image model" in result.message

    async def test_backend_error_is_tool_error(self, context, workspace):
        from olmlx.engine.inference import ImageGenerationError

        gen = FakeGenerator(exc=ImageGenerationError("boom"))
        tools = _tools(context, workspace, gen)
        result = await tools.call_tool("generate_image", {"prompt": "x"})
        assert isinstance(result, ToolError)
        assert result.is_user_error is False
        assert "boom" in result.message
        assert not (workspace / "images").exists() or not any(
            (workspace / "images").iterdir()
        )


class TestSaveErrors:
    async def test_nul_in_filename_is_tool_error(self, context, workspace):
        gen = FakeGenerator()
        tools = _tools(context, workspace, gen)
        result = await tools.call_tool(
            "generate_image", {"prompt": "x", "filename": "a\x00b.png"}
        )
        assert isinstance(result, ToolError)
        assert result.is_user_error is True
        assert gen.calls == []

    async def test_encode_value_error_is_tool_error(
        self, context, workspace, monkeypatch
    ):
        def bad_encode(image, fmt):
            raise ValueError("cannot encode")

        monkeypatch.setattr("olmlx.engine.image_gen.encode_image", bad_encode)
        tools = _tools(context, workspace, FakeGenerator())
        result = await tools.call_tool(
            "generate_image", {"prompt": "x", "width": 64, "height": 64}
        )
        assert isinstance(result, ToolError)
        assert "cannot encode" in result.message


class TestCancellation:
    async def test_run_cancel_sets_generation_cancel_event(self, context, workspace):
        started = asyncio.Event()

        async def gen(prompt, **kwargs):
            started.set()
            ev: threading.Event = kwargs["cancel_event"]
            # Mimic generate_image: a worker polls the threading.Event.
            while not ev.is_set():
                await asyncio.sleep(0.01)
            raise ImageGenerationCancelled()

        tools = AgentToolManager(
            ChatConfig(model_name="m", write_root=workspace),
            context,
            image_tool=AgentImageTool(
                generate=gen, max_dimension=2048, max_prompt_chars=100
            ),
        )
        task = asyncio.create_task(tools.call_tool("generate_image", {"prompt": "x"}))
        await asyncio.wait_for(started.wait(), 1)
        context.cancel_event.set()
        result = await asyncio.wait_for(task, 1)
        assert isinstance(result, ToolError)
        assert "cancel" in result.message.lower()

    async def test_run_cancel_aborts_call_blocked_before_denoise(
        self, context, workspace
    ):
        """A cancel while generate_image is loading the model / queued on the
        inference lock (never polling the per-step event) must still abort
        promptly by cancelling the task."""
        started = asyncio.Event()
        task_cancelled = asyncio.Event()

        async def gen(prompt, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()  # blocked, never polls
            except asyncio.CancelledError:
                task_cancelled.set()
                raise

        tools = AgentToolManager(
            ChatConfig(model_name="m", write_root=workspace),
            context,
            image_tool=AgentImageTool(
                generate=gen, max_dimension=2048, max_prompt_chars=100
            ),
        )
        call = asyncio.create_task(tools.call_tool("generate_image", {"prompt": "x"}))
        await asyncio.wait_for(started.wait(), 1)
        context.cancel_event.set()
        result = await asyncio.wait_for(call, 1)
        assert isinstance(result, ToolError)
        assert "cancel" in result.message.lower()
        assert task_cancelled.is_set()

    async def test_tool_call_cancel_propagates_to_generation(self, context, workspace):
        started = asyncio.Event()
        task_cancelled = asyncio.Event()

        async def gen(prompt, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                assert kwargs["cancel_event"].is_set()
                task_cancelled.set()
                raise

        tools = AgentToolManager(
            ChatConfig(model_name="m", write_root=workspace),
            context,
            image_tool=AgentImageTool(
                generate=gen, max_dimension=2048, max_prompt_chars=100
            ),
        )
        call = asyncio.create_task(tools.call_tool("generate_image", {"prompt": "x"}))
        await asyncio.wait_for(started.wait(), 1)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert task_cancelled.is_set()

    async def test_wallclock_budget_aborts_generation(self, context, workspace):
        task_cancelled = asyncio.Event()

        async def gen(prompt, **kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                task_cancelled.set()
                raise

        context.time_remaining = lambda: 0.05
        tools = AgentToolManager(
            ChatConfig(model_name="m", write_root=workspace),
            context,
            image_tool=AgentImageTool(
                generate=gen, max_dimension=2048, max_prompt_chars=100
            ),
        )
        result = await asyncio.wait_for(
            tools.call_tool("generate_image", {"prompt": "x"}), 1
        )
        assert isinstance(result, ToolError)
        assert "budget" in result.message
        assert task_cancelled.is_set()

    async def test_exhausted_budget_does_not_generate(self, context, workspace):
        gen = FakeGenerator()
        context.time_remaining = lambda: 0.0
        tools = _tools(context, workspace, gen)
        result = await tools.call_tool("generate_image", {"prompt": "x"})
        assert isinstance(result, ToolError)
        assert gen.calls == []

    async def test_already_cancelled_run_does_not_generate(self, context, workspace):
        gen = FakeGenerator()
        tools = _tools(context, workspace, gen)
        context.cancel_event.set()
        result = await tools.call_tool("generate_image", {"prompt": "x"})
        assert isinstance(result, ToolError)
        assert gen.calls == []


class _FakeRegistry:
    def __init__(self, entries):
        self._entries = entries

    def resolve(self, name):
        return self._entries.get(name)


def _manager(**entries):
    from types import SimpleNamespace

    return SimpleNamespace(registry=_FakeRegistry(entries))


_IMAGE_ENTRY = ModelConfig.from_entry(
    {"type": "image", "hf_path": "Qwen/Qwen-Image-2.1"}
)
_TEXT_ENTRY = ModelConfig.from_entry({"hf_path": "Qwen/Qwen3-8B"})


class TestServiceWiring:
    def _service(self, store, tmp_path, manager=None, **over):
        over.setdefault("agent_skills_dir", tmp_path / "skills")
        if manager is None:
            manager = _manager(**{"qwen-image": _IMAGE_ENTRY})
        return AgentService(
            store=store,
            manager_getter=lambda: manager,
            settings=Settings(**over),
        )

    def _session(self, store, tmp_path, manager=None, **over):
        svc = self._service(store, tmp_path, manager, **over)
        run = {"model": "m", "goal": "g"}
        return svc._default_session(run, AgentContext(run_id="r1", store=store))

    def test_not_offered_by_default(self, store, tmp_path):
        sess = self._session(store, tmp_path)
        assert "generate_image" not in sess.builtin.tool_names

    def test_offered_when_image_model_configured(self, store, tmp_path):
        sess = self._session(
            store, tmp_path, agent_image_model="qwen-image", max_loaded_models=2
        )
        assert "generate_image" in sess.builtin.tool_names

    @pytest.mark.parametrize("name", ["missing", "qwen3"])
    def test_not_offered_for_undeclared_or_text_model(
        self, store, tmp_path, caplog, name
    ):
        manager = _manager(**{"qwen-image": _IMAGE_ENTRY, "qwen3": _TEXT_ENTRY})
        with caplog.at_level("WARNING", logger="olmlx.engine.agent.service"):
            sess = self._session(
                store,
                tmp_path,
                manager,
                agent_image_model=name,
                max_loaded_models=2,
            )
        assert "generate_image" not in sess.builtin.tool_names
        assert any("agent_image_model" in r.getMessage() for r in caplog.records)

    def test_registry_error_disables_tool(self, store, tmp_path):
        from types import SimpleNamespace

        def boom(name):
            raise ValueError("invalid model name")

        manager = SimpleNamespace(registry=SimpleNamespace(resolve=boom))
        sess = self._session(
            store, tmp_path, manager, agent_image_model="bad!", max_loaded_models=2
        )
        assert "generate_image" not in sess.builtin.tool_names

    def test_not_offered_when_file_writes_denied(self, store, tmp_path):
        sess = self._session(
            store,
            tmp_path,
            agent_image_model="qwen-image",
            agent_file_write_policy="deny",
            max_loaded_models=2,
        )
        assert "generate_image" not in sess.builtin.tool_names

    def test_warns_when_model_slots_too_few(self, store, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="olmlx.engine.agent.service"):
            self._session(
                store, tmp_path, agent_image_model="qwen-image", max_loaded_models=1
            )
        assert any("max_loaded_models" in r.getMessage() for r in caplog.records)

    async def test_generator_routes_through_inference_generate_image(
        self, store, tmp_path, monkeypatch
    ):
        calls = []
        manager = _manager(**{"qwen-image": _IMAGE_ENTRY})

        async def fake_generate_image(manager, model_name, prompt, **kwargs):
            calls.append((manager, model_name, prompt, kwargs))
            return {"image": Image.new("RGB", (64, 64)), "seed": 5}

        monkeypatch.setattr(
            "olmlx.engine.inference.generate_image", fake_generate_image
        )
        ws = tmp_path / "ws"
        sess = self._session(
            store,
            tmp_path,
            manager,
            agent_image_model="qwen-image",
            agent_workspace_dir=ws,
            max_loaded_models=2,
        )
        result = await sess.builtin.call_tool(
            "generate_image", {"prompt": "hi", "width": 64, "height": 64}
        )
        assert isinstance(result, str), result
        assert calls[0][0] is manager
        assert calls[0][1] == "qwen-image"
        assert calls[0][2] == "hi"
        assert (ws / "images" / "r1-5.png").exists()
