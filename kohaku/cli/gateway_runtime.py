"""Foreground gateway runtime with one loop-owned Stage 1 composition."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import typer
from loguru import logger
from rich.console import Console

from kohaku import __logo__, __version__
from kohaku.agent.hook import AgentHook, AgentRunHookContext
from kohaku.agent.tools.mcp import MCPProvider
from kohaku.cli.runtime_config import _migrate_cron_store
from kohaku.cli.webui_support import (
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
from kohaku.config.paths import get_default_sessions_root, is_default_workspace
from kohaku.config.schema import Config
from kohaku.gateway.runtime import GatewayInstance
from kohaku.runtime_errors import has_owned_work_error
from kohaku.security.network import is_loopback_host
from kohaku.utils.helpers import sync_workspace_templates
from kohaku.webui.build import BuildMode
from kohaku.webui.dev import WebUIDevError, WebUIDevServer

if TYPE_CHECKING:
    from kohaku.agent.tools.registry import ToolRegistry
    from kohaku.agent.turn_delivery import TurnDeliveryFactory
    from kohaku.bus.queue import MessageBus
    from kohaku.cron.service import CronService
    from kohaku.runtime_composition import RuntimeComposition, RuntimeHandle
    from kohaku.session.manager import SessionManager
    from kohaku.session.runtime_authority import SessionAuthority
    from kohaku.utils.llm_runtime import LLMRuntime

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
    *,
    on_first_sigint: Callable[[], None] | None = None,
    on_force: Callable[[], None] | None = None,
    on_sigterm: Callable[[], None] | None = None,
) -> Callable[[], None]:
    loop_signals: list[int] = []
    previous_handlers: list[tuple[int, Any]] = []
    requested = False

    def request_shutdown(signum: int) -> None:
        nonlocal requested
        if requested:
            if on_force is not None:
                on_force()
            for task in tasks:
                task.cancel()
            return
        requested = True
        if signum == signal.SIGINT:
            if on_first_sigint is not None:
                on_first_sigint()
        elif on_sigterm is not None:
            on_sigterm()
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


def _gateway_failure_group(message: str, errors: Sequence[BaseException]) -> BaseException:
    if len(errors) == 1:
        return errors[0]
    if all(isinstance(error, Exception) for error in errors):
        return ExceptionGroup(
            message,
            [error for error in errors if isinstance(error, Exception)],
        )
    return BaseExceptionGroup(message, list(errors))


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


async def _health_server(
    host: str,
    port: int,
    channels: Any,
    on_bound: Callable[[int], None] | None = None,
) -> None:
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
    if on_bound is not None:
        actual_ports: set[int] = set()
        for listener in server.sockets or ():
            address = cast(object, listener.getsockname())
            if isinstance(address, tuple):
                tuple_address = cast(tuple[object, ...], address)
                if len(tuple_address) < 2:
                    continue
                actual_port = tuple_address[1]
                if type(actual_port) is int and 1 <= actual_port <= 65535:
                    actual_ports.add(actual_port)
        if len(actual_ports) == 1:
            on_bound(next(iter(actual_ports)))
    _print_gateway_health_endpoint(host, port)
    async with server:
        await server.serve_forever()


async def _publish_gateway_identity_when_ready(
    channels: Any,
    shutdown: asyncio.Event,
    publish: Callable[[], None],
    *,
    poll_interval_s: float = 0.025,
) -> None:
    """Publish desktop gateway identity only once required channels are ready.

    Publishing owner ports while the WebSocket listener is still starting clears
    the provisional starting state prematurely, causing the desktop adapter's
    status poll to observe ``ready=False`` and flash a transient
    ``gateway_unavailable`` error right before opening chat.
    """
    while not shutdown.is_set():
        ready, _payload = _gateway_readiness_payload(channels)
        if ready:
            publish()
            return
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=poll_interval_s)
        except asyncio.TimeoutError:
            pass


async def _close_gateway_runtime(
    handle: RuntimeHandle,
    composition: RuntimeComposition,
    tasks: Sequence[asyncio.Task[None]],
    *,
    disconnect_webui: Callable[[], None],
    restore: Callable[[], None],
) -> None:
    """Join gateway tasks before closing their single owning composition."""
    errors: list[BaseException] = []
    handle.agent.stop()
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    try:
        disconnect_webui()
    except BaseException as exc:
        errors.append(exc)
    try:
        await composition.aclose()
    except BaseException as exc:
        errors.append(exc)
        if has_owned_work_error(exc):
            try:
                restore()
            except BaseException as restore_error:
                errors.append(restore_error)
            raise _gateway_failure_group("gateway shutdown failed", errors)
    try:
        await handle.bus.drain()
    except BaseException as exc:
        errors.append(exc)
    try:
        restore()
    except BaseException as exc:
        errors.append(exc)
    if errors:
        raise _gateway_failure_group("gateway shutdown failed", errors)


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
    desktop_managed: bool = False,
) -> None:
    """Run one acquired gateway graph with explicitly enabled edge runtimes."""
    sessions_root = (
        config.runtime_data_dir / "sessions"
        if config.runtime_data_dir is not None
        else get_default_sessions_root()
    )
    port = port if port is not None else config.gateway.port

    from kohaku.agent.hooks import create_file_edit_activity_hook
    from kohaku.agent.loop import AgentLoop
    from kohaku.agent.model_presets import load_model_preset_catalog
    from kohaku.channels.manager import ChannelManager
    from kohaku.config.loader import get_config_path
    from kohaku.config.watcher import watch_config_file
    from kohaku.gateway.runtime import (
        GatewayClientLease,
        GatewayRuntime,
        monitor_gateway_clients,
    )
    from kohaku.llm_usage import record_llm_call
    from kohaku.providers.factory import (
        ProviderSnapshot,
        build_provider_snapshot,
        build_unconfigured_provider_snapshot,
        load_provider_snapshot,
    )
    from kohaku.providers.fallback_provider import FallbackProvider
    from kohaku.providers.image_generation import image_gen_provider_configs
    from kohaku.runtime_composition import (
        ChannelBuildOptions,
        RuntimeComposition,
        RuntimeFactories,
    )
    from kohaku.session.recovery import RecoveryCoordinator
    from kohaku.session.webui_turns import (
        WebuiTurnCoordinator,
        WebuiTurnRoutePolicy,
        build_webui_fallback_model_observer,
    )
    from kohaku.triggers.local_store import LocalTriggerStore

    local_triggers = LocalTriggerStore(config.workspace_path)

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
            local_trigger_store=local_triggers,
            webui_runtime_model_name=lambda: agent.model.strip() or None,
            webui_refresh_runtime_config=_webui_refresh_notification(gateway_agent.refresh_runtime_config),
            webui_cron_pending_job_ids=lambda key: (
                agent.pending_cron_job_ids_for_session(key)
                | agent.pending_local_trigger_ids_for_session(key)
            ),
            webui_static_dist=webui_static_dist,
            webui_runtime_surface=webui_runtime_surface,
            webui_runtime_capabilities=webui_runtime_capabilities,
            webui_mcp_runtime_status=gateway_mcp.runtime_status,
            webui_mcp_reload=gateway_mcp.reload,
            webui_skill_state_action=update_skill_state,
            webui_recovery_action=recovery.handle_action,
            webui_child_control=agent.control_delegated_child,
            webui_stop_target_for_key=agent.stop_target_for_key,
            webui_run_state_for_key=agent.run_state_for_key,
            webui_run_stop_target=agent.stop_run_target,
            webui_run_steer_target=agent.steer_run_target,
            webui_resolve_run_approval=agent.resolve_run_approval,
            config_path=Path(config_path),
        )

    def build_agent(config: Config, bus: MessageBus, sessions: SessionManager,
                    tools: ToolRegistry, mcp: MCPProvider, cron: CronService,
                    delivery: TurnDeliveryFactory) -> AgentLoop:
        nonlocal gateway_agent, gateway_mcp, recovery
        fallback_model_observer = build_webui_fallback_model_observer(bus)

        def observe_provider(snapshot: ProviderSnapshot) -> ProviderSnapshot:
            snapshot.provider.set_llm_call_observer(
                lambda call: record_llm_call(call, data_dir=config.runtime_data_dir)
            )
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
            local_trigger_store=local_triggers,
            turn_delivery_factory=delivery,
            image_generation_provider_configs=image_gen_provider_configs(config),
            provider_snapshot_loader=load_gateway_provider_snapshot,
            preset_catalog_loader=load_model_preset_catalog,
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
    )

    async def run() -> bool:
        handle = await composition.acquire()
        agent = handle.agent
        idle_compaction = config.agents.defaults.idle_compaction
        if idle_compaction.enabled:
            from kohaku.context.compaction import run_idle_compaction

            agent.schedule_background(run_idle_compaction(
                agent.sessions, idle_compaction, agent.compact_idle_source,
            ))
        channels = handle.channels
        assert channels is not None
        tasks: list[asyncio.Task[Any]] = []
        local_trigger_task: asyncio.Task[None] | None = None
        identity_publish_task: asyncio.Task[None] | None = None
        token_warmup_task: asyncio.Task[None] | None = None
        shutdown = asyncio.Event()
        processes = None
        cron_gateway = None
        cron_shutdown_mode: Literal["graceful", "bounded", "forced"] | None = None
        primary_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []

        def disconnect_webui() -> None:
            pass

        def restore() -> None:
            pass

        def request_cron_shutdown(
            mode: Literal["graceful", "bounded", "forced"],
        ) -> None:
            nonlocal cron_shutdown_mode
            if cron_shutdown_mode == "forced" and mode != "forced":
                return
            cron_shutdown_mode = mode
            if cron_gateway is None:
                return
            if mode == "graceful":
                cron_gateway.begin_graceful_shutdown()
            elif mode == "bounded":
                cron_gateway.request_bounded_shutdown()
            else:
                cron_gateway.force_shutdown()

        def record_cleanup_error(error: BaseException) -> None:
            cleanup_errors.append(error)
            if has_owned_work_error(error):
                errors = ([primary_error] if primary_error is not None else []) + cleanup_errors
                try:
                    restore()
                except BaseException as restore_error:
                    errors.append(restore_error)
                raise _gateway_failure_group("gateway shutdown retained owned work", errors)

        try:
            restore = _install_gateway_shutdown_handlers(
                asyncio.get_running_loop(),
                shutdown,
                tasks,
                console.print,
                on_first_sigint=lambda: request_cron_shutdown("graceful"),
                on_force=lambda: request_cron_shutdown("forced"),
                on_sigterm=lambda: request_cron_shutdown("bounded"),
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
            if not desktop_managed:
                from kohaku.utils.workspace_lock import open_workspace_memory_lock

                config.workspace_path.mkdir(parents=True, exist_ok=True)
                with open_workspace_memory_lock(config.workspace_path):
                    sync_workspace_templates(config.workspace_path)
                    if is_default_workspace(config.workspace_path):
                        _migrate_cron_store(config)
            _prepare_webui_bundle_for_gateway(
                config,
                mode="skip" if desktop_managed else webui_bundle_mode,
                webui_static_dist=webui_static_dist,
            )
            await handle.mcp.connect()
            assert recovery is not None
            await recovery.scan()
            if config.gateway.cron.enabled:
                from kohaku.cron.gateway import GatewayCron

                root = composition.preflight.runtime_data_root
                if root is None:
                    raise RuntimeError("scheduled execution requires an instance runtime data root")
                cron_gateway = GatewayCron(config.gateway.cron, handle.cron, agent, handle.sessions,
                    channels, config.workspace_path, root,
                    is_codex=lambda: agent.provider.provider_name == "openai_codex")
                if cron_shutdown_mode is not None:
                    request_cron_shutdown(cron_shutdown_mode)
            if config.gateway.process_continuations:
                import hashlib

                from kohaku.process_completion import ProcessCompletionRuntime

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
                f"{__logo__} Starting kohaku gateway version {__version__} on port {port}..."
            )
            console.print("[yellow]○[/yellow] Dream: periodic execution unavailable; use /dream")
            console.print("[yellow]○[/yellow] Heartbeat: periodic execution unavailable")
            console.print("[yellow]○[/yellow] Cron: waiting for channels" if cron_gateway is not None
                          else "[yellow]○[/yellow] Cron: definitions only; runtime inactive")
            options = instance.start_options(
                port=port,
                log_file=log_file,
                desktop_managed=desktop_managed,
            )

            async def monitor_clients() -> None:
                orphaned = await monitor_gateway_clients(
                    GatewayClientLease(gateway_runtime, kind="gateway-monitor"),
                    shutdown,
                )
                if orphaned:
                    logger.info("Last local client disappeared; stopping on-demand gateway")

            with gateway_runtime.foreground_instance(options):
                if desktop_managed:
                    gateway_runtime.publish_health_host(config.gateway.host)
                from kohaku.channels.websocket.runtime import WebSocketChannel

                websocket = channels.get_channel("websocket")
                gateway_services = getattr(websocket, "gateway", None)
                token_store = getattr(gateway_services, "tokens", None)
                gateway_id = getattr(token_store, "instance_id", None)
                websocket_ports: tuple[int, ...] = ()
                if isinstance(websocket, WebSocketChannel):
                    websocket_port = websocket.config.port
                    if (
                        not websocket.config.unix_socket_path
                        and type(websocket_port) is int
                        and 1 <= websocket_port <= 65535
                    ):
                        websocket_ports = (websocket_port,)
                def publish_gateway_identity(actual_port: int) -> None:
                    nonlocal identity_publish_task
                    if not (isinstance(gateway_id, str) and gateway_id and websocket_ports):
                        return
                    resolved_gateway_id = gateway_id

                    def _publish() -> None:
                        gateway_runtime.publish_desktop_gateway_identity(
                            resolved_gateway_id,
                            __version__,
                            "managed-gateway-v1",
                            sys.executable,
                            actual_port,
                            websocket_ports,
                        )

                    ready, _payload = _gateway_readiness_payload(channels)
                    if ready:
                        _publish()
                        return
                    if identity_publish_task is None or identity_publish_task.done():
                        identity_publish_task = asyncio.create_task(
                            _publish_gateway_identity_when_ready(
                                channels,
                                shutdown,
                                _publish,
                            )
                        )
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
                from kohaku.utils.helpers import warm_token_encodings

                token_warmup_task = asyncio.create_task(
                    asyncio.to_thread(warm_token_encodings)
                )
                if processes is not None:
                    await processes.astart()
                if cron_gateway is not None:
                    tasks.append(asyncio.create_task(cron_gateway.run()))
                if config.gateway.local_triggers:
                    from kohaku.triggers.local_runner import run_gateway_local_triggers

                    local_trigger_task = asyncio.create_task(run_gateway_local_triggers(
                        store=local_triggers,
                        submit_turn=agent.submit_local_trigger_turn,
                        is_channel_enabled=lambda name: channels.get_channel(name) is not None,
                    ))
                    tasks.append(local_trigger_task)
                if health_server_enabled:
                    tasks.append(asyncio.create_task(
                        _health_server(
                            config.gateway.host,
                            port,
                            channels,
                            publish_gateway_identity,
                        )
                    ))
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
                for task in done:
                    if task is not wait_shutdown and not task.cancelled():
                        task.result()
                wait_shutdown.cancel()
                with suppress(asyncio.CancelledError):
                    await wait_shutdown
            return False
        except WebUIDevError as exc:
            primary_error = exc
            logger.exception("WebUI development server failed")
            return True
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if token_warmup_task is not None and not token_warmup_task.done():
                token_warmup_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await token_warmup_task
            if identity_publish_task is not None and not identity_publish_task.done():
                identity_publish_task.cancel()
                try:
                    await identity_publish_task
                except asyncio.CancelledError:
                    pass
                except BaseException as exc:
                    record_cleanup_error(exc)
            if local_trigger_task is not None and not local_trigger_task.done():
                # Stop queue admission before waiting for other runtime owners.
                local_trigger_task.cancel()
                try:
                    await local_trigger_task
                except asyncio.CancelledError:
                    pass
                except BaseException as exc:
                    record_cleanup_error(exc)
            if cron_gateway is not None:
                try:
                    await cron_gateway.aclose()
                except BaseException as exc:
                    record_cleanup_error(exc)
            if processes is not None:
                try:
                    await processes.aclose()
                except BaseException as exc:
                    record_cleanup_error(exc)
            try:
                await _close_gateway_runtime(
                    handle, composition, tasks,
                    disconnect_webui=disconnect_webui, restore=restore,
                )
            except BaseException as exc:
                record_cleanup_error(exc)
            if cleanup_errors:
                errors = ([primary_error] if primary_error is not None else []) + cleanup_errors
                raise _gateway_failure_group("gateway shutdown failed", errors)

    unexpected_failure = asyncio.run(run())
    if unexpected_failure:
        raise typer.Exit(1)
