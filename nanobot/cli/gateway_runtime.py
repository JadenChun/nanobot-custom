"""Foreground gateway runtime with one loop-owned Stage 1 composition."""

from __future__ import annotations

import asyncio
import json
import signal
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
from loguru import logger
from rich.console import Console

from nanobot import __logo__, __version__
from nanobot.agent.hook import AgentHook, AgentRunHookContext
from nanobot.agent.memory_recovery_startup_admission import (
    admitted_startup_mutation,
    require_startup_receipt_admission,
)
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.cli.runtime_config import _migrate_cron_store
from nanobot.cli.webui_support import (
    _gateway_health_bind_note,
    _gateway_health_url,
    _host_for_local_browser,
    _launch_browser,
    _prepare_webui_bundle_for_gateway,
    _print_foreground_port_conflict,
    _tcp_endpoint_reachable,
    _webui_browser_url,
    _webui_channel_enabled,
    _webui_display_url,
    _webui_endpoint_reachable,
)
from nanobot.config.paths import get_default_sessions_root, is_default_workspace
from nanobot.config.schema import Config
from nanobot.gateway.runtime import GatewayInstance
from nanobot.security.network import is_loopback_host
from nanobot.utils.helpers import sync_workspace_templates
from nanobot.webui.build import BuildMode
from nanobot.webui.dev import WebUIDevError, WebUIDevServer

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.agent.turn_delivery import TurnDeliveryFactory
    from nanobot.bus.queue import MessageBus
    from nanobot.cron.service import CronService
    from nanobot.runtime_composition import RuntimeComposition, RuntimeHandle
    from nanobot.session.manager import SessionManager
    from nanobot.session.runtime_authority import SessionAuthority
    from nanobot.utils.llm_runtime import LLMRuntime

__all__ = ["_run_gateway"]

console = Console()
_GATEWAY_HEALTH_MAX_CONNECTIONS = 64
_GATEWAY_HEALTH_READ_TIMEOUT_SECONDS = 2.0


def _webui_refresh_notification(refresh: Callable[[], LLMRuntime]) -> Callable[[], None]:
    """Adapt a captured runtime refresh to WebUI's synchronous notification contract."""
    def notify() -> None:
        refresh()

    return notify


class _MCPReadinessHook(AgentHook):
    """Retry application-owned MCP connections before the runner reads tools."""

    def __init__(self, provider: MCPProvider) -> None:
        super().__init__()
        self._provider = provider

    async def before_run(self, context: AgentRunHookContext) -> None:
        await self._provider.connect()


def _http_endpoint_responding(url: str, *, timeout_s: float = 0.25) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout_s):
            return True
    except urllib.error.HTTPError:
        return True
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        return False


async def _watch_webui_dev_server(
    server: WebUIDevServer,
    shutdown_event: asyncio.Event,
    *,
    poll_interval_s: float = 0.2,
) -> None:
    while not shutdown_event.is_set():
        await asyncio.sleep(poll_interval_s)
        if not shutdown_event.is_set():
            server.ensure_running()


def _signal_name(signum: int) -> str:
    with suppress(ValueError):
        return signal.Signals(signum).name
    return f"signal {signum}"


def _install_gateway_shutdown_handlers(
    loop: asyncio.AbstractEventLoop,
    shutdown_event: asyncio.Event,
    tasks: list[asyncio.Task[Any]],
    print_status: Callable[[str], None],
) -> Callable[[], None]:
    loop_signals: list[int] = []
    previous_handlers: list[tuple[int, Any]] = []
    requested = False

    def request_shutdown(signum: int) -> None:
        nonlocal requested
        if requested:
            for task in tasks:
                task.cancel()
            return
        requested = True
        logger.info("Gateway shutdown requested by {}", _signal_name(signum))
        print_status("\nShutting down... Press Ctrl+C again to force.")
        shutdown_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_shutdown, signum)
        except (NotImplementedError, RuntimeError, ValueError):
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, lambda sig, _frame: request_shutdown(sig))
            except (RuntimeError, ValueError):
                continue
            previous_handlers.append((signum, previous))
        else:
            loop_signals.append(signum)

    def restore() -> None:
        for signum in loop_signals:
            with suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(signum)
        for signum, handler in previous_handlers:
            with suppress(RuntimeError, ValueError):
                signal.signal(signum, handler)

    return restore


def _print_gateway_health_endpoint(host: str, port: int) -> None:
    console.print(
        f"[green]✓[/green] Health endpoint: {_gateway_health_url(host, port)}"
        f"{_gateway_health_bind_note(host)}"
    )
    if not is_loopback_host(host):
        console.print(
            "[yellow]Warning: the unauthenticated health endpoint is listening beyond "
            f"loopback. Keep port {port} private or protect it.[/yellow]"
        )


def _gateway_readiness_payload(channels: Any) -> tuple[bool, dict[str, object]]:
    raw: object = channels.get_status() if callable(getattr(channels, "get_status", None)) else {}
    statuses = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
    websocket = statuses.get("websocket")
    required = websocket is not None or "websocket" in channels.enabled_channels
    if not required:
        ready, state = True, "disabled"
    elif isinstance(websocket, dict):
        ready = cast(dict[object, object], websocket).get("running") is True
        state = str(cast(dict[object, object], websocket).get("state") or "unavailable")
    else:
        ready, state = False, "unavailable"
    return ready, {
        "status": "ok" if ready else "degraded",
        "process": "alive",
        "ready": ready,
        "websocket": state,
    }


async def _health_server(host: str, port: int, channels: Any) -> None:
    slots = asyncio.Semaphore(_GATEWAY_HEALTH_MAX_CONNECTIONS)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if slots.locked():
            writer.close()
            return
        async with slots:
            try:
                data = await asyncio.wait_for(
                    reader.read(4096), timeout=_GATEWAY_HEALTH_READ_TIMEOUT_SECONDS,
                )
                parts = data.split(b"\r\n", 1)[0].decode(errors="replace").split(" ")
                if len(parts) >= 2 and parts[:2] == ["GET", "/health"]:
                    ready, payload = _gateway_readiness_payload(channels)
                    body = json.dumps(payload)
                    status = "200 OK" if ready else "503 Service Unavailable"
                    content_type = "application/json"
                else:
                    body, status, content_type = "Not Found", "404 Not Found", "text/plain"
                writer.write(
                    (
                        f"HTTP/1.0 {status}\r\nContent-Type: {content_type}\r\n"
                        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n{body}"
                    ).encode()
                )
                await writer.drain()
            except (asyncio.TimeoutError, ConnectionError):
                pass
            finally:
                writer.close()

    server = await asyncio.start_server(handle, host, port)
    _print_gateway_health_endpoint(host, port)
    async with server:
        await server.serve_forever()


async def _close_gateway_runtime(
    handle: RuntimeHandle,
    composition: RuntimeComposition,
    tasks: Sequence[asyncio.Task[None]],
    *,
    disconnect_webui: Callable[[], None],
    restore: Callable[[], None],
) -> None:
    """Join gateway tasks before closing their single owning composition."""
    handle.agent.stop()
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    disconnect_webui()
    await composition.aclose()
    await handle.bus.drain()
    restore()


def _run_gateway(
    config: Config,
    *,
    port: int | None = None,
    open_browser_url: str | None = None,
    open_browser_ready_url: str | None = None,
    webui_static_dist: bool = True,
    webui_bundle_mode: BuildMode = "warn",
    webui_runtime_surface: str = "browser",
    webui_runtime_capabilities: dict[str, Any] | None = None,
    health_server_enabled: bool = True,
    unconfigured_provider_error: str | None = None,
    webui_dev_server: WebUIDevServer | None = None,
    gateway_instance: GatewayInstance | None = None,
    log_file: str | None = None,
    session_authority: SessionAuthority | None = None,
) -> None:
    """Run one acquired gateway graph with explicitly enabled edge runtimes."""
    sessions_root = (
        config.runtime_data_dir / "sessions"
        if config.runtime_data_dir is not None
        else get_default_sessions_root()
    )
    require_startup_receipt_admission(config.workspace_path, sessions_root=sessions_root)
    port = port if port is not None else config.gateway.port

    from nanobot.agent.hooks import create_file_edit_activity_hook
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.model_presets import load_model_preset_catalog
    from nanobot.channels.manager import ChannelManager
    from nanobot.config.loader import get_config_path
    from nanobot.config.watcher import watch_config_file
    from nanobot.gateway.runtime import (
        GatewayClientLease,
        GatewayRuntime,
        monitor_gateway_clients,
    )
    from nanobot.llm_usage import record_llm_call
    from nanobot.providers.factory import (
        ProviderSnapshot,
        build_provider_snapshot,
        build_unconfigured_provider_snapshot,
        load_provider_snapshot,
    )
    from nanobot.providers.fallback_provider import FallbackProvider
    from nanobot.providers.image_generation import image_gen_provider_configs
    from nanobot.runtime_composition import (
        ChannelBuildOptions,
        RuntimeComposition,
        RuntimeFactories,
    )
    from nanobot.session.recovery import RecoveryCoordinator
    from nanobot.session.webui_turns import (
        WebuiTurnCoordinator,
        WebuiTurnRoutePolicy,
        build_webui_fallback_model_observer,
    )

    defaults = RuntimeFactories.defaults()
    config_path = get_config_path()
    gateway_agent: AgentLoop | None = None
    gateway_mcp: MCPProvider | None = None
    recovery: RecoveryCoordinator | None = None

    def build_channels(config: Config, bus: MessageBus, *,
                       session_manager: SessionManager, cron_service: CronService) -> ChannelManager:
        if gateway_agent is None or gateway_mcp is None or recovery is None:
            raise RuntimeError("gateway foreground coordinators are unavailable")
        agent = gateway_agent

        def update_skill_state(disabled_skills: set[str]) -> None:
            config.agents.defaults.disabled_skills = sorted(disabled_skills)
            agent.context.skills.disabled_skills = set(disabled_skills)
            agent.subagents.disabled_skills = set(disabled_skills)

        return ChannelManager(
            config,
            bus,
            session_manager=session_manager,
            cron_service=cron_service,
            webui_runtime_model_name=lambda: agent.model.strip() or None,
            webui_refresh_runtime_config=_webui_refresh_notification(gateway_agent.refresh_runtime_config),
            webui_cron_pending_job_ids=gateway_agent.pending_cron_job_ids_for_session,
            webui_static_dist=webui_static_dist,
            webui_runtime_surface=webui_runtime_surface,
            webui_runtime_capabilities=webui_runtime_capabilities,
            webui_mcp_runtime_status=gateway_mcp.runtime_status,
            webui_mcp_reload=gateway_mcp.reload,
            webui_skill_state_action=update_skill_state,
            webui_recovery_action=recovery.handle_action,
            config_path=Path(config_path),
        )

    def build_agent(config: Config, bus: MessageBus, sessions: SessionManager,
                    tools: ToolRegistry, mcp: MCPProvider, cron: CronService,
                    delivery: TurnDeliveryFactory) -> AgentLoop:
        nonlocal gateway_agent, gateway_mcp, recovery
        fallback_model_observer = build_webui_fallback_model_observer(bus)

        def observe_provider(snapshot: ProviderSnapshot) -> ProviderSnapshot:
            snapshot.provider.set_llm_call_observer(record_llm_call)
            if isinstance(snapshot.provider, FallbackProvider):
                snapshot.provider.set_fallback_model_observer(fallback_model_observer)
            return snapshot

        def load_gateway_provider_snapshot(*args: Any, **kwargs: Any) -> ProviderSnapshot:
            try:
                return observe_provider(load_provider_snapshot(*args, **kwargs))
            except ValueError as exc:
                if unconfigured_provider_error is None:
                    raise
                return observe_provider(
                    build_unconfigured_provider_snapshot(config, str(exc))
                )

        snapshot = (
            build_unconfigured_provider_snapshot(config, unconfigured_provider_error)
            if unconfigured_provider_error is not None
            else build_provider_snapshot(config)
        )
        snapshot = observe_provider(snapshot)
        recovery = RecoveryCoordinator(
            sessions=sessions,
            bus=bus,
            unified_session=config.agents.defaults.unified_session,
        )
        value = AgentLoop.from_config(
            config,
            bus,
            tool_registry=tools,
            provider=snapshot.provider,
            model=snapshot.model,
            context_window_tokens=snapshot.context_window_tokens,
            provider_signature=snapshot.signature,
            session_manager=sessions,
            cron_service=cron,
            turn_delivery_factory=delivery,
            image_generation_provider_configs=image_gen_provider_configs(config),
            provider_snapshot_loader=load_gateway_provider_snapshot,
            preset_catalog_loader=load_model_preset_catalog,
            enable_legacy_auto_compact=False,
            hooks=[_MCPReadinessHook(mcp)],
            hook_factories=[create_file_edit_activity_hook],
            recovery_admission=recovery,
        )
        value.bus = bus
        value.sessions = sessions
        value.turn_delivery_factory = delivery
        gateway_agent = value
        gateway_mcp = mcp
        return value

    composition = RuntimeComposition.define(
        config,
        sessions_root=sessions_root,
        session_authority=session_authority,
        factories=RuntimeFactories(
            defaults.bus,
            defaults.tools,
            defaults.mcp,
            defaults.cron,
            build_agent,
            route_policy=WebuiTurnRoutePolicy,
        ),
        channel_options=ChannelBuildOptions(factory=build_channels),
        startup_receipt_admitted=True,
    )

    async def run() -> bool:
        handle = await composition.acquire()
        agent = handle.agent
        channels = handle.channels
        assert channels is not None
        tasks: list[asyncio.Task[Any]] = []
        shutdown = asyncio.Event()
        processes = None
        cron_gateway = None
        telegram_busy = None

        def disconnect_webui() -> None:
            pass

        def restore() -> None:
            pass

        try:
            restore = _install_gateway_shutdown_handlers(
                asyncio.get_running_loop(), shutdown, tasks, console.print,
            )
            webui_url = _webui_browser_url(config)
            browser_host = _host_for_local_browser(config.gateway.host)
            if health_server_enabled and _tcp_endpoint_reachable(browser_host, port):
                _print_foreground_port_conflict(
                    webui_url=webui_url,
                    gateway_host=config.gateway.host,
                    gateway_port=port,
                )
                raise typer.Exit(1)
            if _webui_channel_enabled(config) and _webui_endpoint_reachable(webui_url):
                _print_foreground_port_conflict(
                    webui_url=webui_url,
                    gateway_host=config.gateway.host,
                    gateway_port=port,
                )
                raise typer.Exit(1)
            instance = gateway_instance or GatewayInstance.resolve(config_path=config_path)
            gateway_runtime = GatewayRuntime(paths=instance.paths)
            with admitted_startup_mutation(config.workspace_path, sessions_root=sessions_root):
                sync_workspace_templates(config.workspace_path)
                if is_default_workspace(config.workspace_path):
                    _migrate_cron_store(config)
            _prepare_webui_bundle_for_gateway(
                config, mode=webui_bundle_mode, webui_static_dist=webui_static_dist,
            )
            await handle.mcp.connect()
            assert recovery is not None
            await recovery.scan()
            if config.gateway.telegram_busy.enabled:
                from nanobot.channels.telegram.gateway_busy import GatewayTelegramBusy
                from nanobot.channels.telegram.runtime import TelegramChannel

                telegram = channels.get_channel("telegram")
                if not isinstance(telegram, TelegramChannel):
                    raise ValueError("Telegram busy input requires the enabled Telegram channel")
                telegram_busy = GatewayTelegramBusy(config, agent, telegram)
            if config.gateway.cron.enabled:
                from nanobot.cron.gateway import GatewayCron

                root = composition.preflight.runtime_data_root
                if root is None:
                    raise RuntimeError("scheduled execution requires an instance runtime data root")
                cron_gateway = GatewayCron(config.gateway.cron, handle.cron, agent, handle.sessions,
                    channels, config.workspace_path, root,
                    is_codex=lambda: agent.provider.provider_name == "openai_codex")
            if config.gateway.process_continuations:
                import hashlib

                from nanobot.process_completion import ProcessCompletionRuntime

                root = composition.preflight.runtime_data_root
                if root is None:
                    raise RuntimeError("process continuations require an instance runtime data root")
                identity = hashlib.sha256(str(config.workspace_path).encode()).hexdigest()[:24]
                processes = ProcessCompletionRuntime(root / "process-completions" / identity, agent)
                agent.install_process_completion_sink(processes)

            def schedule_webui(awaitable: Awaitable[None]) -> None:
                agent.schedule_background(cast(Coroutine[Any, Any, None], awaitable))

            webui_turns = WebuiTurnCoordinator(
                bus=handle.bus,
                sessions=handle.sessions,
                schedule_background=schedule_webui,
                recovery=recovery,
            )
            disconnect_webui = webui_turns.subscribe()
            console.print(
                f"{__logo__} Starting nanobot gateway version {__version__} on port {port}..."
            )
            console.print("[yellow]○[/yellow] Dream: inactive (P5 Stage 1)")
            console.print("[yellow]○[/yellow] Heartbeat: inactive (P5 Stage 1)")
            console.print("[yellow]○[/yellow] Cron: waiting for channels" if cron_gateway is not None
                          else "[yellow]○[/yellow] Cron: definitions only; runtime inactive")
            options = instance.start_options(port=port, log_file=log_file)

            async def monitor_clients() -> None:
                orphaned = await monitor_gateway_clients(
                    GatewayClientLease(gateway_runtime, kind="gateway-monitor"),
                    shutdown,
                )
                if orphaned:
                    logger.info("Last local client disappeared; stopping on-demand gateway")

            with gateway_runtime.foreground_instance(options):
                tasks.extend([
                    asyncio.create_task(
                        watch_config_file(
                            Path(config_path),
                            agent.invalidate_runtime_config,
                        )
                    ),
                    asyncio.create_task(agent.run()),
                    asyncio.create_task(channels.start_all()),
                    asyncio.create_task(monitor_clients()),
                ])
                if processes is not None:
                    processes.start()
                if cron_gateway is not None:
                    tasks.append(asyncio.create_task(cron_gateway.run()))
                if telegram_busy is not None:
                    tasks.append(asyncio.create_task(telegram_busy.run()))
                if health_server_enabled:
                    tasks.append(asyncio.create_task(_health_server(config.gateway.host, port, channels)))
                if webui_dev_server is not None:
                    tasks.append(asyncio.create_task(_watch_webui_dev_server(webui_dev_server, shutdown)))
                if open_browser_url:
                    if open_browser_ready_url:
                        await asyncio.to_thread(_http_endpoint_responding, open_browser_ready_url)
                    _launch_browser(open_browser_url)
                    console.print(f"[green]✓[/green] Opened browser at {_webui_display_url(open_browser_url)}")
                wait_shutdown = asyncio.create_task(shutdown.wait())
                done, _pending = await asyncio.wait(
                    [*tasks, wait_shutdown], return_when=asyncio.FIRST_COMPLETED,
                )
                if wait_shutdown not in done:
                    for task in done:
                        task.result()
                wait_shutdown.cancel()
                with suppress(asyncio.CancelledError):
                    await wait_shutdown
            return False
        except WebUIDevError:
            logger.exception("WebUI development server failed")
            return True
        finally:
            if telegram_busy is not None:
                await telegram_busy.aclose()
            if cron_gateway is not None:
                await cron_gateway.aclose()
            if processes is not None:
                await processes.aclose()
            await _close_gateway_runtime(
                handle, composition, tasks,
                disconnect_webui=disconnect_webui, restore=restore,
            )

    unexpected_failure = asyncio.run(run())
    if unexpected_failure:
        raise typer.Exit(1)
