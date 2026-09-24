"""Agent control tools layered over the chat builtin tools (issue #446+).

``AgentToolManager`` subclasses ``BuiltinToolManager`` so the wrapped
``ChatSession`` sees the agent's control tools transparently — ``tool_names``,
``get_tool_definitions``, and ``call_tool`` all fall through to the agent tools
first, then to the inherited file/shell/web/plan tools.

Phase 1 adds only ``finish`` (the self-judged success terminator). Later phases
register ``remember`` / ``recall`` (Phase 2), ``create_skill`` (Phase 3), and
``delegate`` (Phase 4) by extending ``_agent_handlers`` / ``_agent_defs``.
``generate_image`` (#725) is offered only when an ``AgentImageTool`` is passed
(``OLMLX_AGENT_IMAGE_MODEL`` set).

``finish`` itself does no control-flow magic: the orchestrator detects it from
the ``tool_call`` event ``ChatSession`` emits, so the handler only needs to
return a confirmation string (and record the summary on the context for
bookkeeping).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from olmlx.chat.builtin_tools import BuiltinToolManager, _resolve_path
from olmlx.chat.config import ChatConfig
from olmlx.chat.errors import ToolError
from olmlx.engine.image_gen import IMAGE_FORMATS

if TYPE_CHECKING:
    from olmlx.engine.agent.orchestrator import AgentContext

logger = logging.getLogger(__name__)


_FINISH_DEF = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Call this when the goal is fully complete to end the autonomous "
            "run. Provide a short summary of what was accomplished. This is the "
            "only clean way to stop — otherwise the run continues."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short summary of what was accomplished.",
                },
            },
            "required": ["summary"],
        },
    },
}

_REMEMBER_DEF = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Save a durable note to long-term memory (survives restart and "
            "context truncation). Use for decisions, facts learned, and "
            "progress worth recalling later."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The note to remember.",
                },
            },
            "required": ["text"],
        },
    },
}

_RECALL_DEF = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Search long-term memory for notes relevant to a query. Returns "
            "the matching notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search memory for.",
                },
            },
            "required": ["query"],
        },
    },
}

_DELEGATE_DEF = {
    "type": "function",
    "function": {
        "name": "delegate",
        "description": (
            "Delegate a focused sub-task to a child agent that works on it "
            "independently and returns its result. Use to decompose a large "
            "goal. Children run one at a time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The sub-task goal for the child agent.",
                },
            },
            "required": ["goal"],
        },
    },
}

_CREATE_SKILL_DEF = {
    "type": "function",
    "function": {
        "name": "create_skill",
        "description": (
            "Author a reusable skill (markdown instructions) after solving a "
            "non-trivial task, so future runs can load it on demand. Use a "
            "short kebab/snake-case name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name (letters, digits, '-' or '_').",
                },
                "description": {
                    "type": "string",
                    "description": "One-line summary of when to use the skill.",
                },
                "body": {
                    "type": "string",
                    "description": "The skill's full markdown instructions.",
                },
            },
            "required": ["name", "description", "body"],
        },
    },
}

_GENERATE_IMAGE_DEF = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "Generate an image from a text prompt and save it to a file in the "
            "workspace. Returns the saved file's path (not the image data). "
            "Slow: expect minutes per image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description of the image.",
                },
                "filename": {
                    "type": "string",
                    "description": (
                        "Optional output path inside the workspace; the "
                        "extension (.png, .jpg, .webp) picks the format. "
                        "Existing files are never overwritten. Defaults to "
                        "images/<run>-<seed>.png."
                    ),
                },
                "width": {
                    "type": "integer",
                    "description": "Width in pixels (multiple of 16). Default 1024.",
                },
                "height": {
                    "type": "integer",
                    "description": "Height in pixels (multiple of 16). Default 1024.",
                },
                "seed": {
                    "type": "integer",
                    "description": "Optional seed for reproducibility.",
                },
                "steps": {
                    "type": "integer",
                    "description": "Optional diffusion steps (model default if unset).",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Optional description of what to avoid.",
                },
            },
            "required": ["prompt"],
        },
    },
}

#: Output-file suffix -> ``image_gen.encode_image`` format.
_IMAGE_SUFFIX_FORMATS = {f".{fmt}": fmt for fmt in IMAGE_FORMATS} | {".jpg": "jpeg"}
_IMAGE_DEFAULT_SIZE = 1024


@dataclass(frozen=True)
class AgentImageTool:
    """Wiring for the ``generate_image`` tool (issue #725).

    ``generate`` has ``inference.generate_image``'s signature minus the
    manager/model (bound by the service), so every call goes through the
    inference lock and Metal-stream handling — never mflux directly. Limits
    mirror the ``/v1/images/generations`` router's settings.
    """

    generate: Callable[..., Awaitable[dict]]
    max_dimension: int
    max_prompt_chars: int


class _ImageArgError(ValueError):
    pass


def _int_arg(arguments: dict, key: str) -> int | None:
    """Coerce an optional integer tool argument; ranges are checked by the
    ``ImageGenerationRequest`` schema."""
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise _ImageArgError(f"{key!r} must be an integer")
    try:
        as_int = int(value)
    except (TypeError, ValueError, OverflowError):
        raise _ImageArgError(f"{key!r} must be an integer") from None
    if isinstance(value, float) and value != as_int:
        raise _ImageArgError(f"{key!r} must be an integer")
    return as_int


def _image_filename(filename: Any) -> tuple[str | None, str]:
    """Validate an explicit ``filename`` (string checks only, no disk I/O).

    Returns ``(None, "png")`` when none was given — the default name needs
    the seed, known only after generation.
    """
    if filename is None or (isinstance(filename, str) and not filename.strip()):
        return None, "png"
    if not isinstance(filename, str) or "\x00" in filename:
        raise _ImageArgError("'filename' must be a plain path string")
    if filename.endswith(("/", os.sep)) or Path(filename).name in ("", ".", ".."):
        raise _ImageArgError("'filename' must name a file, not a directory")
    suffix = Path(filename).suffix.lower()
    if not suffix:
        return filename + ".png", "png"
    fmt = _IMAGE_SUFFIX_FORMATS.get(suffix)
    if fmt is None:
        raise _ImageArgError(
            f"unsupported image extension {suffix!r}; "
            f"use one of {', '.join(sorted(_IMAGE_SUFFIX_FORMATS))}"
        )
    return filename, fmt


def _confined(name: str, root: Path) -> Path:
    try:
        return _resolve_path(name, confine_root=root)
    except ValueError as exc:
        raise _ImageArgError(str(exc)) from None


def _check_image_target(name: str, root: Path) -> None:
    """Early (pre-generation) check so a bad path doesn't waste minutes of
    generation. Re-checked at save time — the disk can change meanwhile."""
    path = _confined(name, root)
    if path.exists():
        raise _ImageArgError(f"{path} already exists; choose a different filename")


def _check_image_dir(name: str, root: Path) -> None:
    """Early check for the default output dir (a symlink escaping the
    workspace, or a non-directory), before minutes of generation."""
    path = _confined(name, root)
    if path.exists() and not path.is_dir():
        raise _ImageArgError(f"{path} exists and is not a directory")


#: Backstop for waiting on an aborted generation. generate_image bounds its own
#: worker drain (``_IMAGE_DRAIN_TIMEOUT``); this only guarantees the tool call
#: returns even if that contract breaks. None = drain timeout + 30s.
_ABORT_DRAIN_TIMEOUT: float | None = None


async def _abort_generation(
    gen: "asyncio.Future[dict]", cancel: threading.Event
) -> None:
    cancel.set()
    gen.cancel()
    timeout = _ABORT_DRAIN_TIMEOUT
    if timeout is None:
        from olmlx.engine.inference import _IMAGE_DRAIN_TIMEOUT

        timeout = _IMAGE_DRAIN_TIMEOUT + 30.0
    await asyncio.wait({gen}, timeout=timeout)
    if not gen.done():
        logger.warning(
            "generate_image did not stop within %.0fs of being aborted; abandoning it",
            timeout,
        )
    # Consume the outcome (now or whenever it lands) so it is never logged as
    # an unretrieved exception; the abort reason is what gets reported.
    gen.add_done_callback(lambda f: f.cancelled() or f.exception())


class _NameTaken(Exception):
    """The target file already exists (distinct from a failing ``mkdir``,
    which also raises ``FileExistsError`` when a parent is a regular file)."""


def _write_new(name: str, root: Path, data: bytes) -> Path:
    """Resolve *name* inside *root* (following symlinks, like ``write_file``)
    and write *data* to a new file; never overwrites (``"xb"``)."""
    path = _confined(name, root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        raise NotADirectoryError(
            f"{path.parent} exists and is not a directory"
        ) from None
    # Re-resolve after mkdir: a symlink planted during generation must not
    # redirect the write outside the workspace.
    path = _confined(name, root)
    try:
        with open(path, "xb") as f:
            f.write(data)
    except FileExistsError:
        raise _NameTaken(str(path)) from None
    return path


#: Inherited builtin tools that make no sense for a headless autonomous run.
#: ``question`` blocks on a human answer that never comes — it would return the
#: raw ``__question__:`` sentinel and stall the loop — so it is dropped from the
#: agent's toolset (the system prompt also tells the model not to ask).
_EXCLUDED_BUILTINS = frozenset({"question"})


class AgentToolManager(BuiltinToolManager):
    """Builtin tools plus the agent's control tools, bound to an AgentContext."""

    def __init__(
        self,
        config: ChatConfig,
        context: "AgentContext",
        skills: Any = None,
        image_tool: AgentImageTool | None = None,
    ):
        super().__init__(config)
        self._context = context
        self._image_tool = image_tool
        # The live SkillManager (when the agent rides a ChatSession), so a
        # ``create_skill`` mid-run is immediately usable via ``use_skill``
        # (#636). None in tests / bare tool-manager use.
        self._skills = skills
        self._agent_defs: list[dict] = [
            _FINISH_DEF,
            _REMEMBER_DEF,
            _RECALL_DEF,
            _CREATE_SKILL_DEF,
            _DELEGATE_DEF,
        ]
        if image_tool is not None:
            self._agent_defs.append(_GENERATE_IMAGE_DEF)

    @property
    def tool_names(self) -> set[str]:
        names = super().tool_names - _EXCLUDED_BUILTINS
        return names | {d["function"]["name"] for d in self._agent_defs}

    def get_tool_definitions(self) -> list[dict]:
        inherited = [
            d
            for d in super().get_tool_definitions()
            if d["function"]["name"] not in _EXCLUDED_BUILTINS
        ]
        return inherited + list(self._agent_defs)

    async def call_tool(self, name: str, arguments: dict) -> str | ToolError:
        if name == "finish":
            return self._handle_finish(arguments)
        if name == "remember":
            return await self._handle_remember(arguments)
        if name == "recall":
            return await self._handle_recall(arguments)
        if name == "create_skill":
            return await self._handle_create_skill(arguments)
        if name == "delegate":
            return await self._handle_delegate(arguments)
        if name == "generate_image" and self._image_tool is not None:
            return await self._handle_generate_image(self._image_tool, arguments)
        return await super().call_tool(name, arguments)

    def _handle_finish(self, arguments: dict) -> str:
        summary = str(arguments.get("summary", "")).strip()
        self._context.finish_summary = summary
        self._context.finished = True
        return f"Run marked complete. Summary recorded: {summary or '(none)'}"

    async def _handle_remember(self, arguments: dict) -> str | ToolError:
        if self._context.memory is None:
            return ToolError(
                message="Memory is not available for this run.",
                tool_name="remember",
                is_user_error=False,
            )
        text = str(arguments.get("text", "")).strip()
        if not text:
            return ToolError(
                message="remember requires non-empty 'text'.",
                tool_name="remember",
                is_user_error=True,
            )
        await self._context.memory.record(text)
        return "Saved to memory."

    async def _handle_recall(self, arguments: dict) -> str | ToolError:
        if self._context.memory is None:
            return ToolError(
                message="Memory is not available for this run.",
                tool_name="recall",
                is_user_error=False,
            )
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolError(
                message="recall requires a non-empty 'query'.",
                tool_name="recall",
                is_user_error=True,
            )
        results = await self._context.memory.recall(query)
        if not results:
            return "No relevant memories found."
        return "\n".join(f"- {r}" for r in results)

    async def _handle_create_skill(self, arguments: dict) -> str | ToolError:
        from olmlx.chat.skills import write_skill_file

        name = str(arguments.get("name", "")).strip()
        description = str(arguments.get("description", "")).strip()
        body = str(arguments.get("body", ""))
        try:
            if self._skills is not None:
                # Route through the live SkillManager so the new skill is
                # registered in-memory and an immediate ``use_skill`` finds it,
                # not just written to disk for the next run to reload (#636).
                # This is also SkillManager.create_skill's only production
                # caller (was otherwise dead code).
                skill = await asyncio.to_thread(
                    self._skills.create_skill, name, description, body
                )
                path = skill.path
            else:
                path = await asyncio.to_thread(
                    write_skill_file,
                    self._config.skills_dir,
                    name,
                    description,
                    body,
                )
        except ValueError as exc:
            return ToolError(
                message=f"Invalid skill: {exc}",
                tool_name="create_skill",
                is_user_error=True,
            )
        await self._context.store.upsert_skill(
            name, description, body.strip(), source_run=self._context.run_id
        )
        return f"Created skill {name!r} at {path}."

    async def _handle_delegate(self, arguments: dict) -> str | ToolError:
        from olmlx.engine.agent.delegate import DelegateError

        runner = self._context.delegate_runner
        if runner is None:
            return ToolError(
                message="Delegation is not available for this run.",
                tool_name="delegate",
                is_user_error=False,
            )
        goal = str(arguments.get("goal", "")).strip()
        try:
            result = await runner.delegate(parent_id=self._context.run_id, goal=goal)
        except DelegateError as exc:
            return ToolError(message=str(exc), tool_name="delegate", is_user_error=True)
        status = result.get("status")
        if status == "finished":
            return f"Subagent finished. Result: {result.get('result') or '(none)'}"
        # Failure / cancellation surfaces to the parent as a tool error so the
        # parent model can decide whether to continue or finish.
        return ToolError(
            message=(f"Subagent {status}: {result.get('error') or 'no result'}"),
            tool_name="delegate",
            is_user_error=False,
        )

    def _validate_image_args(self, tool: AgentImageTool, arguments: dict) -> dict:
        from pydantic import ValidationError

        from olmlx.schemas.images import ImageGenerationRequest

        # Never default above the operator's cap (image_max_dimension may be
        # < 1024); keep it a multiple of 16 like the schema requires.
        default = min(_IMAGE_DEFAULT_SIZE, tool.max_dimension // 16 * 16)
        width = _int_arg(arguments, "width")
        height = _int_arg(arguments, "height")
        width = default if width is None else width
        height = default if height is None else height
        if width > tool.max_dimension or height > tool.max_dimension:
            raise _ImageArgError(
                f"'width'/'height' must be at most {tool.max_dimension}"
            )
        # The router's schema is the single source of truth for prompt,
        # size (>=64, multiple of 16), seed and steps ranges.
        try:
            req = ImageGenerationRequest(
                model="agent",
                prompt=arguments.get("prompt"),  # type: ignore[arg-type]
                size=f"{width}x{height}",
                seed=_int_arg(arguments, "seed"),
                steps=_int_arg(arguments, "steps"),
                negative_prompt=arguments.get("negative_prompt"),
            )
        except ValidationError as exc:
            err = exc.errors()[0]
            loc = ".".join(str(p) for p in err["loc"]) or "arguments"
            raise _ImageArgError(f"invalid {loc!r}: {err['msg']}") from None
        for key, text in (
            ("prompt", req.prompt),
            ("negative_prompt", req.negative_prompt),
        ):
            if text is not None and len(text) > tool.max_prompt_chars:
                raise _ImageArgError(
                    f"{key!r} exceeds {tool.max_prompt_chars} characters"
                )
        return {
            "prompt": req.prompt,
            "width": width,
            "height": height,
            "seed": req.seed,
            "steps": req.steps,
            "negative_prompt": req.negative_prompt or None,
        }

    def _workspace(self) -> Path:
        return self._config.write_root or Path.cwd()

    async def _handle_generate_image(
        self, tool: AgentImageTool, arguments: dict
    ) -> str | ToolError:
        from olmlx.engine.image_gen import ImageGenerationCancelled, encode_image

        def _err(message: str, *, user: bool) -> ToolError:
            return ToolError(
                message=message, tool_name="generate_image", is_user_error=user
            )

        workspace = self._workspace()
        try:
            params = self._validate_image_args(tool, arguments)
            name, fmt = _image_filename(arguments.get("filename"))
            # Disk syscalls stay off the event loop (agent I/O invariant).
            if name is not None:
                await asyncio.to_thread(_check_image_target, name, workspace)
            else:
                await asyncio.to_thread(_check_image_dir, "images", workspace)
        except _ImageArgError as exc:
            return _err(str(exc), user=True)
        if self._context.cancel_event.is_set():
            return _err("Run is cancelled; image not generated.", user=False)
        remaining = (
            self._context.time_remaining()
            if self._context.time_remaining is not None
            else None
        )
        if remaining is not None and remaining <= 0:
            return _err("Run's wallclock budget is exhausted.", user=False)

        # generate_image polls this threading.Event per diffusion step. A run
        # cancel or budget expiry also cancels the task itself, so a cancel
        # while loading the model / queued on the inference lock takes effect
        # immediately; generate_image drains its worker before returning.
        cancel = threading.Event()
        gen = asyncio.ensure_future(tool.generate(**params, cancel_event=cancel))
        cancelled = asyncio.ensure_future(self._context.cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {gen, cancelled},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if gen not in done:
                await _abort_generation(gen, cancel)
                if cancelled in done:
                    return _err("Image generation was cancelled.", user=False)
                return _err(
                    "Image generation exceeded the run's wallclock budget "
                    "and was aborted.",
                    user=False,
                )
            out = gen.result()
        except ImageGenerationCancelled:
            return _err("Image generation was cancelled.", user=False)
        except ValueError as exc:
            # Arguments were validated above, so this is configuration (not
            # an image model, missing [image] extra) — nothing the model can fix.
            return _err(f"Image generation unavailable: {exc}", user=False)
        except Exception as exc:
            logger.warning("generate_image failed", exc_info=True)
            return _err(f"Image generation failed: {exc}", user=False)
        finally:
            cancelled.cancel()
            if not gen.done():
                # This tool call itself was cancelled: stop the denoise and
                # wait for the worker so it never outlives the call.
                await _abort_generation(gen, cancel)

        seed = out["seed"]
        stem = f"images/{self._context.run_id[:8]}-{seed}"

        def _save() -> Path:
            data = encode_image(out["image"], fmt)
            if name is not None:
                return _write_new(name, workspace, data)
            n = 0
            while True:
                candidate = f"{stem}.png" if n == 0 else f"{stem}-{n}.png"
                try:
                    return _write_new(candidate, workspace, data)
                except _NameTaken:
                    n += 1

        try:
            path = await asyncio.to_thread(_save)
        except _ImageArgError as exc:
            return _err(str(exc), user=True)
        except _NameTaken as exc:
            return _err(f"{exc} already exists; choose a different filename", user=True)
        except (OSError, ValueError) as exc:
            return _err(f"Error saving image: {exc}", user=False)
        return (
            f"Saved {params['width']}x{params['height']} image to {path} (seed {seed})."
        )
