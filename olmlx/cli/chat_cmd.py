"""The chat subcommand (terminal chat, voice)."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from olmlx.config import (
    settings,
    warn_legacy_flash_env as _warn_legacy_flash_env,
)

from olmlx.cli.config_cmd import _configure_logging, ensure_config
from olmlx.cli.models_cmd import _create_store
from olmlx.cli.serve import (
    _surface_legacy_kv_cache_quant_env,
    _surface_legacy_speculative_env,
    _warn_kv_cache_quant_incompatibilities,
)

logger = logging.getLogger(__name__)


_VOICE_FLAGS = ("--voice", "--stt-model", "--tts-model", "--voice-name")


def _build_chat_arg_parser_voice_defaults() -> tuple[str, ...]:
    """Return the voice flag strings the chat subparser registers (test seam)."""
    return _VOICE_FLAGS


def _check_voice_deps() -> None:
    """Exit with an install hint if voice deps are unavailable."""
    import importlib.util

    def _absent(name: str) -> bool:
        mod = sys.modules.get(name, "x")
        if mod is None:
            # Explicitly stubbed-out as missing.
            return True
        if mod != "x":
            # Already imported as a real module object -> present.
            return False
        try:
            return importlib.util.find_spec(name) is None
        except Exception:
            return True

    # The [voice] extra carries sounddevice (PortAudio) and, via the
    # self-referential olmlx[audio] requirement, the Kokoro TTS stack —
    # all moved out of core deps in #469. Gate on the TTS transitives too:
    # a hand-installed mlx-audio (pip install, no extra) lacks misaki /
    # en_core_web_sm and would otherwise fail at the first TTS call.
    missing = [
        pkg
        for pkg, mod in (
            ("sounddevice", "sounddevice"),
            ("mlx-audio", "mlx_audio"),
            ("misaki", "misaki"),
            ("en-core-web-sm", "en_core_web_sm"),
        )
        if _absent(mod)
    ]
    if missing:
        print(
            f"--voice needs the {' and '.join(repr(p) for p in missing)} "
            "package(s). Install with: uv sync --extra voice",
            file=sys.stderr,
        )
        raise SystemExit(1)


def cmd_chat(args):
    """Start an interactive chat session."""
    from olmlx.chat.config import ChatConfig, load_mcp_config, load_tool_safety_config
    from olmlx.chat.mcp_client import MCPClientManager
    from olmlx.chat.session import ChatSession
    from olmlx.chat.tool_safety import ToolPolicy, ToolSafetyPolicy
    from olmlx.chat.tui import ChatTUI
    from olmlx.engine.model_manager import ModelManager

    ensure_config()
    _configure_logging()
    # ``olmlx chat`` reads ``settings.speculative*`` via ModelManager
    # too, so honour the deprecation window here. Without this a user
    # who only runs chat would silently lose forwarding even though
    # ``serve`` handles it correctly.
    _surface_legacy_speculative_env()
    _surface_legacy_kv_cache_quant_env()
    _warn_legacy_flash_env()
    _warn_kv_cache_quant_incompatibilities()

    model_name = args.model_name
    if model_name is None:
        print("Error: model name required. Usage: olmlx chat <model>", file=sys.stderr)
        sys.exit(1)

    chat_kwargs: dict[str, Any] = dict(
        model_name=model_name,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        thinking=not args.no_thinking,
        mcp_enabled=not args.no_mcp,
        repeat_penalty=args.repeat_penalty,
        repeat_last_n=args.repeat_last_n,
        skills_enabled=not args.no_skills,
        builtin_tools_enabled=not args.no_builtin_tools,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tool_timeout=args.tool_timeout,
        mcp_connect_retries=args.mcp_connect_retries,
        local_tool_safety=args.local_tool_safety,
        tool_result_truncation=args.tool_result_truncation,
        max_consecutive_tool_failures=args.max_consecutive_tool_failures,
    )
    # Filter out None for nullable args so ChatConfig defaults apply.
    # Boolean flags (store_true) are never None — only filter numeric args.
    for _key in (
        "temperature",
        "top_p",
        "top_k",
        "tool_timeout",
        "mcp_connect_retries",
        "tool_result_truncation",
        "max_consecutive_tool_failures",
    ):
        if chat_kwargs[_key] is None:
            del chat_kwargs[_key]
    # Boolean store_true flags default to False. Drop them when False so
    # ChatConfig's default wins — if the default ever flips to True, the CLI
    # won't silently override it back to False.
    for _key in ("local_tool_safety",):
        if not chat_kwargs.get(_key):
            chat_kwargs.pop(_key, None)  # ChatConfig default wins
    if args.mcp_config:
        chat_kwargs["mcp_config_path"] = Path(args.mcp_config)
    if args.skills_dir:
        chat_kwargs["skills_dir"] = Path(args.skills_dir)
    config = ChatConfig(**chat_kwargs)

    async def _run_chat():
        from olmlx.chat.builtin_tools import BuiltinToolManager
        from olmlx.chat.skills import SkillManager

        store = _create_store()
        manager = ModelManager(store.registry, store)

        tui = ChatTUI(tool_result_truncation=config.tool_result_truncation)
        mcp = None
        skills = None
        builtin = None

        try:
            tui.console.print(f"[dim]Loading {model_name}...[/dim]")
            await manager.ensure_loaded(model_name, keep_alive="-1")

            if config.mcp_enabled:
                mcp_cfg = load_mcp_config(config.mcp_config_path)
                if mcp_cfg:
                    mcp = MCPClientManager()
                    await mcp.connect_all(
                        mcp_cfg, max_attempts=config.mcp_connect_retries
                    )

            if config.skills_enabled:
                skills = SkillManager(config.skills_dir)
                skills.load()
                if skills.list_skills():
                    tui.console.print(
                        f"[dim]Loaded {len(skills.list_skills())} skill(s)[/dim]"
                    )

            if config.builtin_tools_enabled:
                builtin = BuiltinToolManager(config)

            # Load tool safety policy
            safety_config = load_tool_safety_config(config.mcp_config_path)
            active_stream_ctx = None

            async def confirm_decider(name: str, args: dict) -> bool:
                nonlocal active_stream_ctx
                if active_stream_ctx and active_stream_ctx.is_active:
                    active_stream_ctx.finish()
                return await asyncio.to_thread(tui.confirm_tool_call, name, args)

            def _build_judge():
                from olmlx.chat.llm_judge import SafeJudge

                return SafeJudge(
                    manager,
                    model_name=lambda: (
                        safety_config.judge_model
                        if safety_config.judge_model
                        else config.model_name
                    ),
                )

            llm_judge = None
            uses_auto = safety_config.default_policy == ToolPolicy.AUTO or any(
                p == ToolPolicy.AUTO for p in safety_config.tool_policies.values()
            )
            if uses_auto:
                llm_judge = _build_judge()
                if safety_config.judge_model:
                    tui.console.print(
                        f"[dim]LLM judge using separate model: "
                        f"{safety_config.judge_model}[/dim]"
                    )

            policy = ToolSafetyPolicy(
                safety_config,
                decider=confirm_decider,
                llm_judge=llm_judge,
            )

            session = ChatSession(
                config=config,
                manager=manager,
                mcp=mcp,
                skills=skills,
                builtin=builtin,
                tool_safety=policy,
            )
            tools = mcp.get_tools_for_chat() if mcp else []
            if builtin:
                tools = tools + builtin.get_tool_definitions()
            tui.display_welcome(model_name, tools)

            voice_io = None
            if getattr(args, "voice", False):
                _check_voice_deps()
                from olmlx.chat.voice.io import VoiceIO

                stt_model = args.stt_model or settings.chat_stt_model
                tts_model = args.tts_model or settings.chat_tts_model
                voice_name = args.voice_name or settings.chat_tts_voice
                tui.console.print(f"[dim]Loading STT {stt_model}...[/dim]")
                await manager.ensure_loaded(stt_model, keep_alive="-1")
                tui.console.print(f"[dim]Loading TTS {tts_model}...[/dim]")
                await manager.ensure_loaded(tts_model, keep_alive="-1")
                voice_io = VoiceIO(
                    manager=manager,
                    stt_model=stt_model,
                    tts_model=tts_model,
                    voice=voice_name,
                )
                tui.console.print(
                    "[dim]Voice on. Press Enter at the prompt to talk; "
                    "type to send text.[/dim]"
                )

            _pending_answer: str | None = None

            while True:
                if _pending_answer is not None:
                    user_input = _pending_answer
                    _pending_answer = None
                else:
                    user_input = tui.get_user_input()
                    if user_input is None:
                        break

                # Empty line in voice mode => push-to-talk.
                if voice_io is not None and not user_input.strip():
                    try:
                        user_input = await voice_io.listen()
                    except RuntimeError as exc:  # device/dep failure
                        tui.display_error(str(exc))
                        continue
                    if not user_input:
                        continue
                    tui.console.print(f"[dim]heard:[/dim] {user_input}")

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    cmd_parts = user_input.split(None, 1)
                    command = cmd_parts[0].lower()
                    arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                    if command in ("/exit", "/quit"):
                        break
                    elif command == "/clear":
                        session.clear_history()
                        tui.console.print("[dim]History cleared[/dim]")
                    elif command == "/tools":
                        tui.display_tools(tools)
                    elif command == "/skills":
                        if skills and skills.list_skills():
                            tui.console.print("[bold]Skills:[/bold]")
                            for s in skills.list_skills():
                                desc = f" — {s.description}" if s.description else ""
                                tui.console.print(f"  [cyan]{s.name}[/cyan]{desc}")
                        else:
                            tui.console.print("[dim]No skills loaded[/dim]")
                    elif command == "/safety":
                        tui.display_safety_policy(policy)
                    elif command == "/mode":
                        if arg == "auto":
                            new_default = ToolPolicy.AUTO
                        elif arg == "confirm":
                            new_default = ToolPolicy.CONFIRM
                        else:
                            tui.display_error("Usage: /mode auto|confirm")
                            continue
                        safety_config.default_policy = new_default
                        if new_default == ToolPolicy.CONFIRM:
                            # Clear per-tool AUTO overrides so tools that
                            # were auto-judged are now confirmed manually.
                            # ALLOW and DENY overrides are intentionally
                            # preserved — the user explicitly configured
                            # those and switching to confirm mode shouldn't
                            # undo that policy.
                            cleared = [
                                name
                                for name, pol in list(
                                    safety_config.tool_policies.items()
                                )
                                if pol == ToolPolicy.AUTO
                            ]
                            for name in cleared:
                                del safety_config.tool_policies[name]
                            if cleared:
                                tui.console.print(
                                    f"[dim]Cleared AUTO override(s): "
                                    f"{', '.join(cleared)}[/dim]"
                                )
                        if new_default == ToolPolicy.AUTO and llm_judge is None:
                            llm_judge = _build_judge()
                            policy.llm_judge = llm_judge
                            tui.console.print("[dim]LLM judge initialised[/dim]")
                        tui.console.print(
                            f"[dim]Default policy: {new_default.value}[/dim]"
                        )
                    elif command == "/system":
                        if arg:
                            config.system_prompt = arg
                            session.clear_history()
                            tui.console.print(
                                "[dim]System prompt set. History cleared.[/dim]"
                            )
                        else:
                            current = config.system_prompt or "(none)"
                            tui.console.print(f"[dim]System prompt: {current}[/dim]")
                    elif command == "/model":
                        model_parts = arg.split(None, 1)
                        if model_parts and model_parts[0] == "thinking":
                            if len(model_parts) == 2 and model_parts[1] in (
                                "on",
                                "off",
                            ):
                                config.thinking = model_parts[1] == "on"
                                tui.console.print(
                                    f"[dim]Thinking: {'on' if config.thinking else 'off'}[/dim]"
                                )
                            else:
                                thinking_str = "on" if config.thinking else "off"
                                tui.console.print(
                                    f"[dim]Thinking: {thinking_str}. Use: /model thinking on|off[/dim]"
                                )
                        elif arg:
                            tui.console.print(f"[dim]Loading {arg}...[/dim]")
                            try:
                                await manager.ensure_loaded(arg, keep_alive="-1")
                                config.model_name = arg
                                session.clear_history()
                                tui.console.print(
                                    f"[dim]Switched to {arg}. History cleared.[/dim]"
                                )
                            except Exception as exc:
                                tui.display_error(str(exc))
                        else:
                            thinking_str = "on" if config.thinking else "off"
                            tui.console.print(
                                f"[dim]Current model: {config.model_name} | thinking: {thinking_str}[/dim]"
                            )
                    else:
                        tui.display_error(f"Unknown command: {command}")
                    continue

                # Collect events while streaming tokens
                pending_events = []
                confirmed_tool_ids: set[str] = set()
                spoken_parts: list[str] = []
                stream_ctx = tui.stream_response()
                active_stream_ctx = stream_ctx
                try:
                    with stream_ctx:
                        async for event in session.send_message(user_input):
                            if event["type"] == "thinking_start":
                                stream_ctx.start_thinking()
                            elif event["type"] == "thinking_end":
                                stream_ctx.end_thinking()
                            elif event["type"] == "thinking_token":
                                stream_ctx.update(event["text"])
                            elif event["type"] == "token":
                                stream_ctx.update(event["text"])
                                if voice_io is not None:
                                    spoken_parts.append(event["text"])
                            elif event["type"] == "tool_approved":
                                # Track confirmed IDs to avoid duplicate display
                                confirmed_tool_ids.add(event["id"])
                            else:
                                pending_events.append(event)
                    # Track if question was asked for post-processing
                    question_asked = any(
                        e.get("type") == "question" for e in pending_events
                    )
                    if question_asked:
                        # Find the question event and ask user
                        for event in pending_events:
                            if event.get("type") == "question":
                                answer = tui.ask_question(
                                    event.get("header", ""),
                                    event.get("question", ""),
                                    options=event.get("options"),
                                    multiple=event.get("multiple", False),
                                )
                                if answer is not None:
                                    _pending_answer = answer
                                    break
                finally:
                    active_stream_ctx = None

                # Display collected events
                for event in pending_events:
                    if event["type"] == "tool_call":
                        # Skip display for confirmed tools — already shown
                        # by confirm_tool_call during the prompt
                        if event["id"] not in confirmed_tool_ids:
                            tui.display_tool_call(event["name"], event["arguments"])
                    elif event["type"] == "tool_result":
                        tui.display_tool_result(event["name"], event["result"])
                    elif event["type"] == "tool_error":
                        tui.display_tool_error(event["name"], event["error"])
                    elif event["type"] == "tool_denied":
                        # Only show panel for policy-denied or auto-denied
                        # tools; user-denied tools were already shown
                        # at the confirm prompt
                        if event.get("reason") != "user":
                            tui.display_tool_denied(
                                event["name"], reason=event.get("reason", "policy")
                            )
                    elif event["type"] == "tool_auto_judging":
                        tui.display_tool_auto_judging(event["name"])
                    elif event["type"] == "tool_confirmation_needed":
                        pass  # handled inline by decider callback
                    elif event["type"] == "max_turns_exceeded":
                        tui.display_error("Max tool turns reached")
                    elif event["type"] == "tool_failures_exceeded":
                        tui.display_tool_failures_exceeded(event["message"])
                    elif event["type"] == "memory_truncated":
                        tui.display_memory_truncated(event["message"])
                    elif event["type"] == "repetition_detected":
                        tui.display_repetition_detected()
                    elif event["type"] == "model_load_error":
                        tui.display_model_load_error(event["error"])
                        break

                if voice_io is not None and spoken_parts:
                    try:
                        await voice_io.speak("".join(spoken_parts))
                    except RuntimeError as exc:
                        tui.display_error(str(exc))

        except MemoryError as exc:
            tui.display_error(str(exc))
            sys.exit(1)
        except ValueError as exc:
            tui.display_error(str(exc))
            sys.exit(1)
        finally:
            if mcp is not None:
                await mcp.disconnect_all()
            await manager.stop()

    try:
        asyncio.run(_run_chat())
    except KeyboardInterrupt:
        print("\nBye!", file=sys.stderr)
