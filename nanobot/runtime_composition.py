"""Typed, two-phase ownership boundary for foreground Nanobot runtimes."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from nanobot.runtime_errors import RuntimeOwnedWorkError
from nanobot.session.runtime_authority import (
    DurableSessionAuthority,
    SessionAuthority,
    SessionAuthorityLease,
    SessionAuthorityUnavailableError,
    Stage1UnavailableSessionAuthority,
)

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.tools.mcp import MCPProvider
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.agent.turn_delivery import TurnDeliveryFactory, TurnRoutePolicy
    from nanobot.bus.queue import MessageBus
    from nanobot.channels.manager import ChannelManager
    from nanobot.config.schema import Config
    from nanobot.cron.runtime import CronRuntime, CronRuntimeInjection
    from nanobot.cron.service import CronService
    from nanobot.session.manager import SessionManager


class RuntimePreflightError(ValueError):
    """Configuration is incompatible with the closed Stage 1 runtime."""


class RuntimeLifecycleError(RuntimeError):
    """The requested operation is invalid for the composition lifecycle."""


class RuntimeLoopError(RuntimeLifecycleError):
    """The composition was accessed from a foreign event loop or thread."""


class RuntimeNotReadyError(RuntimeLifecycleError):
    """No published runtime handle is available."""


class RuntimeState(Enum):
    NEW = auto()
    BUILDING = auto()
    READY = auto()
    CLOSING = auto()
    CLOSED = auto()
    FAILED = auto()


@dataclass(frozen=True, slots=True)
class SessionModePreflight:
    state: str
    legacy_sessions: int = 0
    canonical_sessions: int = 0


@dataclass(frozen=True, slots=True)
class RuntimePreflight:
    workspace: Path
    sessions_root: Path
    sessions_root_exists: bool
    legacy_idle_ttl_minutes: int
    idle_runtime_enabled: bool
    heartbeat_configured_enabled: bool
    dream_configured_enabled: bool
    session_modes: SessionModePreflight
    runtime_data_root: Path | None = None


def _session_header(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimePreflightError("session evidence must be a regular file")
    try:
        import json

        with path.open(encoding="utf-8") as handle:
            line = next((candidate for candidate in handle if candidate.strip()), "")
        header = json.loads(line)
        if not isinstance(header, dict) or cast(dict[object, object], header).get("_type") != "metadata":
            raise ValueError
        key = cast(dict[object, object], header).get("key")
        metadata = cast(dict[object, object], header).get("metadata", {})
        if not isinstance(key, str) or not key or not isinstance(metadata, dict):
            raise ValueError
        version = cast(dict[object, object], metadata).get("source_schema_version")
        if version is None:
            return key, "legacy"
        if type(version) is not int or version != 1:
            raise RuntimePreflightError("unsupported session source schema version")
        from nanobot.session.source_store import validate_source_session_file

        validate_source_session_file(path, expected_key=key)
        return key, "canonical"
    except RuntimePreflightError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        raise RuntimePreflightError("malformed session mode evidence") from None


def _matching_namespace(workspace: Path, root: Path) -> Path | None:
    if root.is_symlink():
        raise RuntimePreflightError("session root must not be a symlink")
    marker = workspace / ".nanobot" / "workspace-id"
    candidates: list[Path] = []
    if marker.exists() or marker.is_symlink():
        if marker.is_symlink() or not marker.is_file():
            raise RuntimePreflightError("malformed workspace session identity")
        try:
            identity = marker.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            raise RuntimePreflightError("malformed workspace session identity") from None
        if len(identity) != 32 or any(ch not in "0123456789abcdef" for ch in identity):
            raise RuntimePreflightError("malformed workspace session identity")
        candidates.append(root / identity)
    if root.is_dir():
        for entry in root.iterdir():
            claim = entry / ".workspace"
            if entry.is_symlink():
                raise RuntimePreflightError("session namespace must not be a symlink")
            if not entry.is_dir():
                continue
            if claim.is_symlink():
                raise RuntimePreflightError("session namespace claim must not be a symlink")
            if not claim.is_file():
                continue
            try:
                claimed = Path(claim.read_text(encoding="utf-8").strip()).expanduser()
                if claimed.resolve(strict=False) == workspace:
                    candidates.append(entry)
            except (OSError, UnicodeError, ValueError):
                raise RuntimePreflightError("malformed session namespace claim") from None
    unique = {candidate.resolve(strict=False) for candidate in candidates if candidate.exists()}
    if len(unique) > 1:
        raise RuntimePreflightError("conflicting session namespaces claim this workspace")
    return next(iter(unique), None)


def inspect_session_modes(workspace: Path, root: Path) -> SessionModePreflight:
    """Read existing headers only; never create, select, repair, or convert state."""
    locations: list[Path] = []
    namespace = _matching_namespace(workspace, root)
    if namespace is not None:
        locations.append(namespace)
    legacy_location = workspace / "sessions"
    if legacy_location.is_symlink():
        raise RuntimePreflightError("legacy session evidence must not be a symlink")
    if legacy_location.is_dir():
        locations.append(legacy_location)
    observed: dict[str, str] = {}
    legacy = canonical = 0
    for location in locations:
        for path in sorted(location.glob("*.jsonl")):
            key, mode = _session_header(path)
            if key in observed:
                raise RuntimePreflightError("conflicting dual session-mode evidence")
            observed[key] = mode
            if mode == "legacy":
                legacy += 1
            else:
                canonical += 1
    if legacy and canonical:
        state = "mixed"
    elif legacy:
        state = "legacy"
    elif canonical:
        state = "canonical"
    else:
        state = "absent"
    return SessionModePreflight(state, legacy, canonical)


def runtime_preflight(
    config: Config,
    *,
    sessions_root: Path | None = None,
    startup_receipt_admitted: bool = False,
) -> RuntimePreflight:
    from nanobot.agent.memory_recovery_startup_admission import require_startup_receipt_admission
    from nanobot.config.paths import get_default_sessions_root

    workspace = config.workspace_path.expanduser().resolve(strict=False)
    root = (
        sessions_root
        or (config.runtime_data_dir / "sessions" if config.runtime_data_dir else None)
        or get_default_sessions_root()
    ).expanduser().resolve(strict=False)
    ttl = config.agents.defaults.session_ttl_minutes
    idle_enabled = config.agents.defaults.idle_compaction.enabled
    if ttl:
        raise RuntimePreflightError(
            "legacy idleCompactAfterMinutes/sessionTtlMinutes must be disabled before Stage 1 runtime construction"
        )
    if idle_enabled:
        raise RuntimePreflightError("idle compaction runtime is not activated in P5 Stage 1")
    if not startup_receipt_admitted:
        require_startup_receipt_admission(workspace, sessions_root=root)
    modes = inspect_session_modes(workspace, root)
    return RuntimePreflight(
        workspace=workspace,
        sessions_root=root,
        sessions_root_exists=root.exists(),
        legacy_idle_ttl_minutes=ttl,
        idle_runtime_enabled=idle_enabled,
        heartbeat_configured_enabled=config.gateway.heartbeat.enabled,
        dream_configured_enabled=config.agents.defaults.dream.enabled,
        session_modes=modes,
        runtime_data_root=(config.runtime_data_dir or root.parent).absolute(),
    )


class BusFactory(Protocol):
    def __call__(self) -> MessageBus: ...


class ToolFactory(Protocol):
    def __call__(self) -> ToolRegistry: ...


class MCPFactory(Protocol):
    def __call__(self, config: Config, tools: ToolRegistry) -> MCPProvider: ...


class CronFactory(Protocol):
    def __call__(self, preflight: RuntimePreflight) -> CronService: ...


class AgentFactory(Protocol):
    def __call__(
        self,
        config: Config,
        bus: MessageBus,
        sessions: SessionManager,
        tools: ToolRegistry,
        mcp: MCPProvider,
        cron: CronService,
        delivery: TurnDeliveryFactory,
    ) -> AgentLoop: ...


class ChannelFactory(Protocol):
    def __call__(
        self,
        config: Config,
        bus: MessageBus,
        *,
        session_manager: SessionManager,
        cron_service: CronService,
    ) -> ChannelManager: ...


@dataclass(frozen=True, slots=True)
class RuntimeFactories:
    bus: BusFactory
    tools: ToolFactory
    mcp: MCPFactory
    cron: CronFactory
    agent: AgentFactory
    route_policy: Callable[[SessionManager], TurnRoutePolicy] | None = None

    @classmethod
    def defaults(cls) -> RuntimeFactories:
        from nanobot.agent.hooks import create_file_edit_activity_hook
        from nanobot.agent.loop import AgentLoop
        from nanobot.agent.tools.mcp import MCPProvider
        from nanobot.agent.tools.registry import ToolRegistry
        from nanobot.bus.queue import MessageBus
        from nanobot.cron.service import CronService
        from nanobot.providers.image_generation import image_gen_provider_configs

        def build_agent(
            config: Config,
            bus: MessageBus,
            sessions: SessionManager,
            tools: ToolRegistry,
            mcp: MCPProvider,
            cron: CronService,
            delivery: TurnDeliveryFactory,
        ) -> AgentLoop:
            agent = AgentLoop.from_config(
                config,
                bus,
                tool_registry=tools,
                session_manager=sessions,
                cron_service=cron,
                turn_delivery_factory=delivery,
                enable_legacy_auto_compact=False,
                image_generation_provider_configs=image_gen_provider_configs(config),
                hook_factories=[create_file_edit_activity_hook],
            )
            agent.bus = bus
            agent.sessions = sessions
            agent.turn_delivery_factory = delivery
            return agent

        return cls(
            bus=MessageBus,
            tools=ToolRegistry,
            mcp=lambda config, tools: MCPProvider.from_config(config, tools),
            cron=lambda preflight: CronService(preflight.workspace / "cron" / "jobs.json"),
            agent=build_agent,
        )


@dataclass(frozen=True, slots=True)
class ChannelBuildOptions:
    factory: ChannelFactory | None = None


@dataclass(frozen=True, slots=True)
class RuntimeHandle:
    bus: MessageBus
    sessions: SessionManager
    tools: ToolRegistry
    mcp: MCPProvider
    cron: CronService
    delivery: TurnDeliveryFactory
    agent: AgentLoop
    channels: ChannelManager | None = None
    cron_runtime: CronRuntime | None = None


CloseCallback = Callable[[], Awaitable[None] | None]


async def _close_reverse(callbacks: list[CloseCallback]) -> list[BaseException]:
    errors: list[BaseException] = []
    for close in reversed(callbacks):
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except BaseException as exc:
            errors.append(exc)
            if isinstance(exc, RuntimeOwnedWorkError):
                break  # Resistant effects still own shared channels/sessions.
    return errors


def _combined_error(message: str, primary: BaseException, cleanup: list[BaseException]) -> BaseException:
    values = [primary, *cleanup]
    if all(isinstance(item, Exception) for item in values):
        return ExceptionGroup(message, values)  # type: ignore[arg-type]
    return BaseExceptionGroup(message, values)


class RuntimeComposition:
    """Read-only definition which acquires one concrete graph on its owner loop."""

    def __init__(
        self,
        *,
        config: Config,
        preflight: RuntimePreflight,
        authority: SessionAuthority,
        factories: RuntimeFactories,
        channel_options: ChannelBuildOptions | None,
        cron_injection: CronRuntimeInjection | None = None,
    ) -> None:
        self.config = config
        self.preflight = preflight
        self._authority = authority
        self._factories = factories
        self._channel_options = channel_options
        self._cron_injection = cron_injection
        self._state = RuntimeState.NEW
        self._state_lock = threading.Lock()
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._owner_thread: int | None = None
        self._build_waiter: asyncio.Future[RuntimeHandle] | None = None
        self._close_waiter: asyncio.Future[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._handle: RuntimeHandle | None = None
        self._closers: list[CloseCallback] = []
        self._failure: BaseException | None = None
        self._default_authority = isinstance(authority, Stage1UnavailableSessionAuthority)

    @classmethod
    def define(
        cls,
        config: Config,
        *,
        sessions_root: Path | None = None,
        session_authority: SessionAuthority | None = None,
        factories: RuntimeFactories | None = None,
        channel_options: ChannelBuildOptions | None = None,
        startup_receipt_admitted: bool = False,
        cron_injection: CronRuntimeInjection | None = None,
    ) -> RuntimeComposition:
        """Perform read-only preflight and retain definitions only."""
        report = runtime_preflight(
            config,
            sessions_root=sessions_root,
            startup_receipt_admitted=startup_receipt_admitted,
        )
        return cls(
            config=config,
            preflight=report,
            authority=(
                session_authority
                if session_authority is not None
                else DurableSessionAuthority(
                    report.workspace, sessions_root=report.sessions_root,
                )
            ),
            factories=factories or RuntimeFactories.defaults(),
            channel_options=channel_options,
            cron_injection=cron_injection,
        )

    @property
    def state(self) -> RuntimeState:
        with self._state_lock:
            return self._state

    def _require_owner(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._owner_loop is not loop or self._owner_thread != threading.get_ident():
            raise RuntimeLoopError("runtime composition belongs to another event loop")

    async def acquire(
        self,
        *,
        channel_options: ChannelBuildOptions | None = None,
    ) -> RuntimeHandle:
        loop = asyncio.get_running_loop()
        leader = False
        with self._state_lock:
            if self._state is RuntimeState.NEW:
                self._state = RuntimeState.BUILDING
                self._owner_loop = loop
                self._owner_thread = threading.get_ident()
                self._build_waiter = loop.create_future()
                self._build_waiter.add_done_callback(
                    lambda future: future.exception() if not future.cancelled() else None
                )
                waiter = self._build_waiter
                leader = True
            elif self._state is RuntimeState.BUILDING:
                self._require_owner(loop)
                assert self._build_waiter is not None
                waiter = self._build_waiter
            elif self._state is RuntimeState.READY:
                self._require_owner(loop)
                assert self._handle is not None
                return self._handle
            elif self._state is RuntimeState.FAILED:
                self._require_owner(loop)
                assert self._failure is not None
                raise self._failure
            else:
                if self._owner_loop is not None:
                    self._require_owner(loop)
                raise RuntimeLifecycleError(f"cannot acquire runtime in {self._state.name} state")
        if not leader:
            return await asyncio.shield(waiter)
        try:
            handle, closers = await self._acquire_graph(channel_options or self._channel_options)
        except BaseException as exc:
            with self._state_lock:
                self._state = RuntimeState.FAILED
                self._failure = exc
                if not waiter.done():
                    waiter.set_exception(exc)
            raise
        with self._state_lock:
            self._handle = handle
            self._closers = closers
            self._state = RuntimeState.READY
            waiter.set_result(handle)
        return handle

    async def _acquire_graph(
        self,
        channel_options: ChannelBuildOptions | None,
    ) -> tuple[RuntimeHandle, list[CloseCallback]]:
        from nanobot.agent.turn_delivery import TurnDeliveryFactory
        from nanobot.channels.manager import ChannelManager

        closers: list[CloseCallback] = []
        try:
            lease = await self._authority.acquire()
            if not isinstance(cast(object, lease), SessionAuthorityLease):
                raise TypeError("session authority must return SessionAuthorityLease")
            sessions = lease.manager
            if lease.close is not None:
                closers.append(lease.close)
            bus = self._factories.bus()
            tools = self._factories.tools()
            mcp = self._factories.mcp(self.config, tools)
            closers.append(mcp.aclose)
            cron = self._factories.cron(self.preflight)
            route_policy = (
                self._factories.route_policy(sessions)
                if self._factories.route_policy is not None
                else None
            )
            delivery = TurnDeliveryFactory(bus, route_policy=route_policy)
            agent = self._factories.agent(
                self.config, bus, sessions, tools, mcp, cron, delivery,
            )
            closers.append(agent.aclose)
            if agent.bus is not bus or agent.sessions is not sessions:
                raise ValueError("agent must share composition bus and sessions")
            if agent.turn_delivery_factory is not delivery:
                raise ValueError("agent must share composition delivery factory")
            channels = None
            if channel_options is not None:
                manager_factory = channel_options.factory or ChannelManager
                channels = manager_factory(
                    self.config,
                    bus,
                    session_manager=sessions,
                    cron_service=cron,
                )
                await channels.acquire()
                closers.append(channels.stop_all)
            cron_runtime = None
            if self._cron_injection is not None:
                import hashlib

                from nanobot.cron.runtime import CronRuntime, CronRuntimeInjection

                if not isinstance(cast(object, self._cron_injection), CronRuntimeInjection) or channels is None:
                    raise RuntimePreflightError("dormant cron requires explicit injection and acquired channels")
                root = self.preflight.runtime_data_root
                if root is None:
                    raise RuntimePreflightError("captured runtime data root unavailable")
                identity = hashlib.sha256(b"nanobot-cron-runtime-root-v1\0" + str(self.preflight.workspace).encode() + b"\0" + cron.action_store.namespace.encode() + b"\0" + str(root).encode()).hexdigest()
                cron_runtime = CronRuntime(cron, agent, sessions, channels,
                    root / "cron-runtime" / identity, identity, self._cron_injection)
                # Register before lease acquisition so all partial ownership is
                # unwound in reverse, without an unowned background task.
                closers.append(cron_runtime.aclose)
                cron_runtime.acquire()
            return RuntimeHandle(bus, sessions, tools, mcp, cron, delivery, agent, channels, cron_runtime), closers
        except BaseException as exc:
            cleanup = await _close_reverse(closers)
            if cleanup:
                raise _combined_error("runtime acquisition and rollback failed", exc, cleanup)
            raise

    def require_ready(self) -> RuntimeHandle:
        with self._state_lock:
            if self._state is RuntimeState.READY and self._handle is not None:
                return self._handle
            if self._default_authority and self._state in {
                RuntimeState.NEW,
                RuntimeState.BUILDING,
                RuntimeState.FAILED,
            }:
                raise SessionAuthorityUnavailableError(
                    "session authority is unavailable during P5 Stage 1"
                )
            raise RuntimeNotReadyError(f"runtime is not ready ({self._state.name})")

    async def aclose(self) -> None:
        loop = asyncio.get_running_loop()
        leader = False
        with self._state_lock:
            if self._state is RuntimeState.READY:
                self._require_owner(loop)
                self._state = RuntimeState.CLOSING
                self._close_waiter = loop.create_future()
                self._close_waiter.add_done_callback(
                    lambda future: future.exception() if not future.cancelled() else None
                )
                waiter = self._close_waiter
                leader = True
            elif self._state is RuntimeState.CLOSING:
                self._require_owner(loop)
                assert self._close_waiter is not None
                waiter = self._close_waiter
            elif self._state is RuntimeState.CLOSED:
                if self._owner_loop is not None:
                    self._require_owner(loop)
                return
            elif self._state is RuntimeState.NEW:
                self._state = RuntimeState.CLOSED
                return
            elif self._state is RuntimeState.FAILED:
                self._require_owner(loop)
                if self._closers and self._failure is not None:
                    def owned_failure(error: BaseException) -> bool:
                        return isinstance(error, RuntimeOwnedWorkError) or (
                            isinstance(error, BaseExceptionGroup)
                            and any(owned_failure(item) for item in cast(BaseExceptionGroup[BaseException], error).exceptions)
                        )

                    if not owned_failure(self._failure):
                        raise self._failure
                    self._state = RuntimeState.CLOSING
                    self._close_waiter = loop.create_future()
                    self._close_waiter.add_done_callback(lambda future: future.exception() if not future.cancelled() else None)
                    waiter, leader = self._close_waiter, True
                else:
                    return
            else:
                raise RuntimeLifecycleError("cannot close runtime while acquisition is building")
        if not leader:
            await asyncio.shield(waiter)
            return
        self._close_task = asyncio.create_task(self._finish_close(waiter))
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancelled = True
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _finish_close(self, waiter: asyncio.Future[None]) -> None:
        errors = await _close_reverse(self._closers)
        with self._state_lock:
            self._state = (
                RuntimeState.FAILED if any(isinstance(error, RuntimeOwnedWorkError) for error in errors)
                else RuntimeState.CLOSED
            )
            if errors:
                failure = _combined_error("runtime composition close failed", errors[0], errors[1:])
                self._failure = failure
                waiter.set_exception(failure)
            else:
                waiter.set_result(None)
        if errors:
            raise _combined_error("runtime composition close failed", errors[0], errors[1:])


__all__ = [
    "ChannelBuildOptions",
    "RuntimeComposition",
    "RuntimeFactories",
    "RuntimeHandle",
    "RuntimeLifecycleError",
    "RuntimeLoopError",
    "RuntimeNotReadyError",
    "RuntimePreflight",
    "RuntimePreflightError",
    "RuntimeState",
    "SessionModePreflight",
    "inspect_session_modes",
    "runtime_preflight",
]
