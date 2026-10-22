"""Direct and interactive agent CLI command."""

import asyncio
import importlib
import json
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, cast

import typer
from rich.console import Console

from nanobot import __logo__
from nanobot.agent.memory_recovery_startup_admission import (
    admitted_startup_mutation,
    require_startup_receipt_admission,
)
from nanobot.cli.log_control import (
    clear_compatibility_log_file,
    configure_agent_runtime_logs,
)
from nanobot.cli.runtime_config import (
    _load_runtime_config,
    _migrate_cron_store,
    _model_display,
    _print_agent_start_error,
)
from nanobot.config.paths import get_default_sessions_root

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.tools.mcp import MCPProvider
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.agent.turn_delivery import TurnDeliveryFactory
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import Config
    from nanobot.cron.service import CronService
    from nanobot.session.manager import SessionManager

console = Console()

_CLASSIC_DEPENDENCIES = {
    "AgentLoop": ("nanobot.agent.loop", "AgentLoop"),
    "StreamRenderer": ("nanobot.cli.stream", "StreamRenderer"),
    "consume_restart_notice_from_env": (
        "nanobot.utils.restart",
        "consume_restart_notice_from_env",
    ),
    "is_default_workspace": ("nanobot.config.paths", "is_default_workspace"),
    "sync_workspace_templates": ("nanobot.utils.helpers", "sync_workspace_templates"),
}


def __getattr__(name: str) -> Any:
    """Preserve patchable classic-agent symbols without loading them for the TUI."""
    dependency = _CLASSIC_DEPENDENCIES.get(name)
    if dependency is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = dependency
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def _classic_dependency(name: str) -> Any:
    if name in globals():
        return globals()[name]
    return __getattr__(name)


def _warn_deprecated_config_keys(config_path: str | None) -> None:
    """Retain the fork's agent-CLI hints for ignored or relocated config keys."""
    from nanobot.config.loader import get_config_path

    path = Path(config_path).expanduser().resolve() if config_path else get_config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    agents = cast(dict[object, object], raw).get("agents")
    defaults = cast(dict[object, object], agents).get("defaults") if isinstance(agents, dict) else None
    if not isinstance(defaults, dict):
        return
    if "memoryWindow" in defaults:
        console.print(
            "[dim]Hint: `memoryWindow` in your config is no longer used "
            "and can be safely removed.[/dim]"
        )
    for key in ("contextWindowTokens", "maxInputTokens"):
        if key in defaults:
            console.print(f"[dim]Hint: `{key}` has moved to `maxTokens.input`.[/dim]")


def agent(
    message: str | None = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    markdown: bool = typer.Option(
        True,
        "--markdown/--no-markdown",
        help="Render assistant output as Markdown",
    ),
    logs: bool = typer.Option(
        False,
        "--logs/--no-logs",
        help="Show nanobot runtime logs during chat",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Write runtime logs to this file (rotated at 20 MB, 7 days kept)",
    ),
    classic: bool = typer.Option(
        True,
        "--classic",
        "--no-tui",
        help="Use the classic Python prompt instead of the native terminal UI",
    ),
    theme: str = typer.Option(
        "auto",
        "--theme",
        help="Terminal UI appearance: auto, dark, or light",
    ),
):
    """Chat in the terminal or send one message non-interactively."""
    runtime_config = _load_runtime_config(config, workspace)
    _warn_deprecated_config_keys(config)
    sessions_root = (
        runtime_config.runtime_data_dir / "sessions"
        if runtime_config.runtime_data_dir is not None
        else get_default_sessions_root()
    )
    require_startup_receipt_admission(runtime_config.workspace_path, sessions_root=sessions_root)
    theme = theme.strip().lower()
    if theme not in {"auto", "dark", "light"}:
        raise typer.BadParameter("must be auto, dark, or light", param_hint="--theme")
    native_tui = message is None and not classic
    if native_tui:
        from nanobot.cli.tui_launcher import TuiSessionError, TuiUnavailableError, launch_tui
        from nanobot.config.loader import get_config_path

        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise typer.BadParameter(
                "the native TUI requires an interactive terminal; use --message for "
                "one-shot input or --classic for the legacy prompt",
                param_hint="terminal",
            )
        if not markdown:
            raise typer.BadParameter("--no-markdown requires --classic", param_hint="--no-markdown")
        if logs:
            raise typer.BadParameter("--logs requires --classic", param_hint="--logs")
        try:
            exit_code = launch_tui(
                runtime_config,
                config_path=get_config_path().resolve(strict=False),
                workspace_override=workspace,
                session_id=session_id,
                theme=theme,
            )
        except TuiSessionError as exc:
            raise typer.BadParameter(str(exc), param_hint="--session") from exc
        except TuiUnavailableError as exc:
            console.print(f"[red]Native TUI unavailable: {exc}[/red]")
            console.print("[dim]Use `nanobot agent --classic` only if you want the old prompt.[/dim]")
            raise typer.Exit(1) from exc
        else:
            if exit_code:
                raise typer.Exit(exit_code)
            return

    from nanobot.bus.outbound_events import (
        StreamDeltaEvent,
        StreamedResponseEvent,
        StreamEndEvent,
    )
    from nanobot.cli import terminal as cli_terminal
    from nanobot.cli.stream import ThinkingSpinner
    from nanobot.providers.factory import make_provider, validate_provider_setup
    from nanobot.utils.helpers import sanitize_surrogates as _sanitize_surrogates
    from nanobot.utils.restart import (
        format_restart_completed_message,
        should_show_cli_restart_notice,
    )

    agent_loop_class: "type[AgentLoop]" = _classic_dependency("AgentLoop")
    stream_renderer_class = _classic_dependency("StreamRenderer")
    consume_restart_notice_from_env = _classic_dependency("consume_restart_notice_from_env")
    is_default_workspace = _classic_dependency("is_default_workspace")
    sync_workspace_templates = _classic_dependency("sync_workspace_templates")

    session_id = session_id or "cli:direct"

    try:
        # Reject local setup errors before acquiring durable runtime resources.
        # Provider clients themselves are still created by the owned graph below.
        from nanobot.runtime_composition import runtime_preflight

        runtime_preflight(runtime_config, sessions_root=sessions_root, startup_receipt_admitted=True)
        validate_provider_setup(runtime_config)
        from nanobot.runtime_composition import RuntimeComposition, RuntimeFactories

        defaults = RuntimeFactories.defaults()

        def build_agent(
            config: "Config",
            bus: "MessageBus",
            sessions: "SessionManager",
            tools: "ToolRegistry",
            mcp: "MCPProvider",
            cron: "CronService",
            delivery: "TurnDeliveryFactory",
        ) -> "AgentLoop":
            value = agent_loop_class.from_config(
                config,
                bus,
                tool_registry=tools,
                provider=make_provider(config),
                session_manager=sessions,
                cron_service=cron,
                turn_delivery_factory=delivery,
                enable_legacy_auto_compact=False,
            )
            value.bus = bus
            value.sessions = sessions
            value.turn_delivery_factory = delivery
            return value

        composition = RuntimeComposition.define(
            runtime_config,
            sessions_root=sessions_root,
            factories=RuntimeFactories(
                defaults.bus,
                defaults.tools,
                defaults.mcp,
                defaults.cron,
                build_agent,
            ),
            startup_receipt_admitted=True,
        )
    except ValueError as exc:
        clear_compatibility_log_file()
        _print_agent_start_error(exc)
        raise typer.Exit(1) from exc
    except BaseException:
        clear_compatibility_log_file()
        raise
    def _prepare_acquired_runtime() -> None:
        with admitted_startup_mutation(
            runtime_config.workspace_path,
            sessions_root=sessions_root,
        ):
            sync_workspace_templates(runtime_config.workspace_path)
            if is_default_workspace(runtime_config.workspace_path):
                _migrate_cron_store(runtime_config)
        configure_agent_runtime_logs(show_console=logs, log_file=log_file)
        if logs:
            console.print(
                "[dim]Runtime logs enabled (--logs). Use --no-logs to hide internal logs.[/dim]"
            )
        if log_file:
            console.print(f"[dim]Logging to {log_file}[/dim]")
        restart_notice = consume_restart_notice_from_env()
        if restart_notice and should_show_cli_restart_notice(restart_notice, session_id):
            cli_terminal._print_agent_response(
                format_restart_completed_message(restart_notice.started_at_raw),
                render_markdown=False,
            )

    async def _close_runtime() -> None:
        await composition.aclose()

    # Shared reference for progress callbacks
    _thinking: ThinkingSpinner | None = None
    runtime: Any | None = None

    def _make_progress(
        renderer: Any | None = None,
    ) -> Callable[..., Awaitable[None]]:
        reasoning_buffer = cli_terminal._ReasoningBuffer()

        async def _cli_progress(
            content: str,
            *,
            tool_hint: bool = False,
            reasoning: bool = False,
            **_kwargs: Any,
        ) -> None:
            assert runtime is not None
            ch = runtime.agent.channels_config

            if _kwargs.get("reasoning_end"):
                if ch and not ch.show_reasoning:
                    reasoning_buffer.clear()
                else:
                    cli_terminal._flush_cli_reasoning(reasoning_buffer, _thinking, renderer)
                return

            if reasoning:
                if ch and not ch.show_reasoning:
                    reasoning_buffer.clear()
                    return
                text = reasoning_buffer.add(content)
                if text:
                    cli_terminal._print_cli_reasoning(text, _thinking, renderer)
                return
            if ch and tool_hint and not ch.send_tool_hints:
                return
            if ch and not tool_hint and not ch.send_progress:
                return
            cli_terminal._print_cli_progress_line(content, _thinking, renderer)

        return _cli_progress

    if message is not None:
        # Single message mode — direct call, no bus needed
        async def run_once() -> None:
            nonlocal runtime
            try:
                runtime = await composition.acquire()
                _prepare_acquired_runtime()
                await runtime.mcp.connect()
                renderer = stream_renderer_class(
                    render_markdown=markdown,
                    bot_name=runtime_config.agents.defaults.bot_name,
                    bot_icon=runtime_config.agents.defaults.bot_icon,
                )
                response = await runtime.agent.process_direct(
                    message,
                    session_id,
                    on_progress=_make_progress(renderer),
                    on_stream=renderer.on_delta,
                    on_stream_end=renderer.on_end,
                )
                if not renderer.streamed:
                    await renderer.close()
                    print_kwargs: dict[str, Any] = {}
                    if renderer.header_printed:
                        print_kwargs["show_header"] = False
                    cli_terminal._print_agent_response(
                        response.content if response else "",
                        render_markdown=markdown,
                        metadata=response.metadata if response else None,
                        **print_kwargs,
                    )
            finally:
                await _close_runtime()

        try:
            asyncio.run(run_once())
        finally:
            clear_compatibility_log_file()
    else:
        # Interactive mode — route through bus like other channels
        from nanobot.bus.events import InboundMessage

        if ":" in session_id:
            cli_channel, cli_chat_id = session_id.split(":", 1)
        else:
            cli_channel, cli_chat_id = "cli", session_id

        def _handle_signal(signum: int, _frame: FrameType | None) -> None:
            sig_name = signal.Signals(signum).name
            cli_terminal._restore_terminal()
            console.print(f"\nReceived {sig_name}, goodbye!")
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        # SIGHUP is not available on Windows
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _handle_signal)
        # Ignore SIGPIPE to prevent silent process termination when writing to closed pipes
        # SIGPIPE is not available on Windows
        if hasattr(signal, "SIGPIPE"):
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)

        async def run_interactive() -> None:
            nonlocal runtime
            runtime = await composition.acquire()
            acquired_runtime = runtime
            _prepare_acquired_runtime()
            cli_terminal._init_prompt_session()
            _model, _preset_tag = _model_display(runtime_config)
            _icon = runtime_config.agents.defaults.bot_icon or __logo__
            console.print(
                f"{_icon} Interactive mode [bold blue]({_model})[/bold blue]{_preset_tag} "
                "— type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit\n"
            )
            await runtime.mcp.connect()
            bus_task = asyncio.create_task(runtime.agent.run())
            turn_done = asyncio.Event()
            turn_done.set()
            turn_response: list[Any] = []
            renderer: Any | None = None
            reasoning_buffer = cli_terminal._ReasoningBuffer()

            async def _consume_outbound() -> None:
                while True:
                    try:
                        msg = await asyncio.wait_for(
                            acquired_runtime.bus.consume_outbound(), timeout=1.0,
                        )
                        event = msg.event

                        if isinstance(event, StreamDeltaEvent):
                            if renderer:
                                await renderer.on_delta(msg.content)
                            continue
                        if isinstance(event, StreamEndEvent):
                            if renderer:
                                await renderer.on_end(
                                    resuming=event.resuming,
                                )
                            continue
                        if isinstance(event, StreamedResponseEvent):
                            if msg.content and renderer and not renderer.streamed:
                                await renderer.close()
                                print_kwargs: dict[str, Any] = {}
                                if renderer.header_printed:
                                    print_kwargs["show_header"] = False
                                cli_terminal._print_agent_response(
                                    msg.content,
                                    render_markdown=markdown,
                                    metadata=msg.metadata,
                                    **print_kwargs,
                                )
                            turn_done.set()
                            continue

                        if await cli_terminal._maybe_print_interactive_progress(
                            msg,
                            None,
                            acquired_runtime.agent.channels_config,
                            renderer,
                            reasoning_buffer,
                        ):
                            continue

                        if not turn_done.is_set():
                            if msg.content:
                                turn_response.append(msg)
                            turn_done.set()
                        elif msg.content:
                            await cli_terminal._print_interactive_response(
                                msg.content,
                                render_markdown=markdown,
                                metadata=msg.metadata,
                            )

                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            outbound_task = asyncio.create_task(_consume_outbound())

            try:
                while True:
                    try:
                        cli_terminal._flush_pending_tty_input()
                        # Stop spinner before user input to avoid prompt_toolkit conflicts
                        if renderer:
                            renderer.stop_for_input()
                        user_input = _sanitize_surrogates(
                            await cli_terminal._read_interactive_input_async()
                        )
                        command = user_input.strip()
                        if not command:
                            continue

                        if cli_terminal._is_exit_command(command):
                            cli_terminal._restore_terminal()
                            console.print("\nGoodbye!")
                            break

                        turn_done.clear()
                        turn_response.clear()
                        reasoning_buffer.clear()
                        renderer = stream_renderer_class(
                            render_markdown=markdown,
                            bot_name=runtime_config.agents.defaults.bot_name,
                            bot_icon=runtime_config.agents.defaults.bot_icon,
                        )

                        await runtime.bus.publish_inbound(
                            InboundMessage(
                                channel=cli_channel,
                                sender_id="user",
                                chat_id=cli_chat_id,
                                content=user_input,
                                metadata={"_wants_stream": True},
                            )
                        )

                        await turn_done.wait()

                        if turn_response:
                            response_msg = turn_response[0]
                            content = response_msg.content
                            meta = response_msg.metadata
                            if content and not isinstance(
                                response_msg.event,
                                StreamedResponseEvent,
                            ):
                                if renderer:
                                    await renderer.close()
                                print_kwargs: dict[str, Any] = {}
                                if renderer and renderer.header_printed:
                                    print_kwargs["show_header"] = False
                                cli_terminal._print_agent_response(
                                    content,
                                    render_markdown=markdown,
                                    metadata=meta,
                                    **print_kwargs,
                                )
                        elif renderer and not renderer.streamed:
                            await renderer.close()
                    except KeyboardInterrupt:
                        cli_terminal._restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    except EOFError:
                        cli_terminal._restore_terminal()
                        console.print("\nGoodbye!")
                        break
            finally:
                runtime.agent.stop()
                outbound_task.cancel()
                await asyncio.gather(bus_task, outbound_task, return_exceptions=True)
                await _close_runtime()

        try:
            asyncio.run(run_interactive())
        finally:
            clear_compatibility_log_file()
