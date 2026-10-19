"""Explicit positive-only, effect-fenced graphs for historical model tests.

Constructor callbacks keep model options separate from infrastructure config.
Borrowed SDK import graphs are caller-owned; all other graphs own durable leases.
"""
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn_delivery import TurnDeliveryFactory
from nanobot.bus.queue import MessageBus
from nanobot.config import loader
from nanobot.config.schema import Config
from nanobot.cron.service import CronService
from nanobot.runtime_composition import RuntimeComposition, RuntimeFactories, RuntimeState
from nanobot.security import network
from nanobot.session.manager import Session, SessionManager
from nanobot.session.runtime_authority import (
    BorrowedSessionAuthority,
    DurableSessionAuthority,
    OfflineSessionOperator,
)


@dataclass(frozen=True)
class AgentInputs:
    config: Config
    bus: MessageBus
    sessions: SessionManager
    tools: ToolRegistry
    mcp: MCPProvider
    cron: CronService
    delivery: TurnDeliveryFactory


AgentBuilder = Callable[[AgentInputs], AgentLoop]


def _denied(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("admitted fixture external effect denied")


class AdmittedGraphs:
    def __init__(self, tmp_path: Path, patches: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.patches = patches
        self.graphs: list[tuple[RuntimeComposition, bool]] = []
        patches.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
        # Legacy fallback must never inspect a real or an outer runner home.
        patches.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "absent-home"))
        patches.setattr(loader, "_current_config_path", loader._current_config_path)
        patches.setattr(network, "_allowed_networks", network._allowed_networks)
        for name in ("connect", "connect_ex", "bind", "listen", "sendto"):
            patches.setattr(socket.socket, name, _denied)
        for name in ("getaddrinfo", "create_connection"):
            patches.setattr(socket, name, _denied)
        patches.setattr(subprocess, "Popen", _denied)
        for name in ("system", "fork", "posix_spawn", "posix_spawnp", "execv", "execve"):
            if hasattr(os, name):
                patches.setattr(os, name, _denied)
        patches.setattr(asyncio, "create_subprocess_exec", _denied)
        patches.setattr(asyncio, "create_subprocess_shell", _denied)
        patches.setattr("nanobot.channels.manager.ChannelManager.acquire", _denied)
        patches.setattr(MCPProvider, "connect", _denied)
        patches.setattr(CronService, "start", _denied)
        patches.setattr(AgentLoop, "run", _denied)
        patches.setattr("nanobot.providers.factory.make_provider", _denied)

    async def acquire(
        self, root: Path, build: AgentBuilder, *, config: Config | None = None,
        borrowed: bool = False, keys: tuple[str, ...] = (),
    ) -> AgentLoop:
        assert root == self.tmp_path
        graph_root = root / f"admitted-{len(self.graphs)}"
        workspace = graph_root / "workspace"
        workspace.mkdir(parents=True)
        path = graph_root / "instance" / "config.json"
        infrastructure = config.model_copy(deep=True) if config is not None else Config()
        infrastructure.agents.defaults.workspace = str(workspace)
        infrastructure.agents.defaults.session_ttl_minutes = 0
        infrastructure.agents.defaults.idle_compaction.enabled = False
        infrastructure.agents.defaults.dream.enabled = False
        infrastructure.gateway.heartbeat.enabled = False
        infrastructure.tools.agent_browser.enabled = False
        infrastructure.tools.agent_device.enabled = False
        infrastructure.tools.desktop_use.enabled = False
        infrastructure.tools.restrict_to_workspace = True
        infrastructure.tools.mcp_servers = {}
        infrastructure.artifacts.enabled = False
        infrastructure.channels = type(infrastructure.channels)()
        loader.save_config(infrastructure, path)
        infrastructure = loader.load_config(path)
        assert infrastructure._source_path == path
        assert infrastructure.runtime_data_dir == path.parent
        loader.set_config_path(path)
        sessions_root = path.parent / "sessions"
        authority = None
        if borrowed:
            authority = BorrowedSessionAuthority(SessionManager(workspace, sessions_root=sessions_root))
            assert not authority.manager.has_managed_authority
            assert (await authority.acquire()).close is None
        elif keys:
            unmanaged = SessionManager(workspace, sessions_root=sessions_root)
            for key in keys:
                unmanaged.save(Session(key), fsync=True)
            lease = await OfflineSessionOperator(
                workspace, copied_sessions_root=sessions_root,
            ).acquire()
            try:
                lease.manager.bootstrap_existing_sessions(offline_copy=True)
            finally:
                assert lease.close is not None
                await lease.close()

        def agent(
            config: Config, bus: MessageBus, sessions: SessionManager, tools: ToolRegistry,
            mcp: MCPProvider, cron: CronService, delivery: TurnDeliveryFactory,
        ) -> AgentLoop:
            return build(AgentInputs(config, bus, sessions, tools, mcp, cron, delivery))

        graph = RuntimeComposition.define(
            infrastructure, session_authority=authority,
            factories=RuntimeFactories(
                MessageBus, ToolRegistry, MCPProvider.from_config,
                lambda report: CronService(report.workspace / "cron" / "jobs.json"), agent,
            ),
        )
        # Register before acquisition: exceptions still retain an owner for rollback verification.
        self.graphs.append((graph, borrowed))
        handle = await graph.acquire()
        loop = handle.agent
        assert loop.bus is handle.bus
        assert loop.sessions is handle.sessions
        assert loop.tools is handle.tools
        assert loop.cron_service is handle.cron
        assert loop.turn_delivery_factory is handle.delivery
        assert handle.channels is None and handle.cron_runtime is None
        assert not handle.mcp.configured_server_names
        assert not handle.mcp.connected_server_names
        assert not handle.cron._running
        if authority is not None:
            assert handle.sessions is authority.manager
        else:
            assert handle.sessions.has_managed_authority
            for key in keys:
                mode = handle.sessions.resolve_session_mode(key)
                assert mode.mode == "legacy" and mode.binding is not None
        return loop

    async def close(self) -> None:
        for graph, borrowed in reversed(self.graphs):
            handle = graph.require_ready() if graph.state is RuntimeState.READY else None
            if handle is not None:
                await handle.bus.drain()
            await graph.aclose()
            assert graph._owner_loop is asyncio.get_running_loop()
            assert graph.state in (RuntimeState.CLOSED, RuntimeState.FAILED)
            if handle is not None:
                assert graph.state is RuntimeState.CLOSED
                assert handle.agent._closed
                assert handle.mcp._closing
                assert not handle.mcp.connected_server_names
                if borrowed:
                    assert not handle.sessions.has_managed_authority
                assert not handle.agent._background_tasks
                assert not handle.bus._pending
                assert all(task.done() for tasks in handle.agent._active_tasks.values() for task in tasks)
            if not borrowed:
                lease = await DurableSessionAuthority(
                    graph.preflight.workspace, sessions_root=graph.preflight.sessions_root,
                ).acquire()
                assert lease.close is not None
                await lease.close()


class AsyncAdmittedFactory(Protocol):
    def __call__(
        self, root: Path, build: AgentBuilder, *, config: Config | None = None,
        borrowed: bool = False, keys: tuple[str, ...] = (),
    ) -> Awaitable[AgentLoop]: ...


class SyncAdmittedFactory:
    def __init__(self, graphs: AdmittedGraphs, runner: asyncio.Runner) -> None:
        self.graphs = graphs
        self.runner = runner

    def __call__(
        self, root: Path, build: AgentBuilder, *, config: Config | None = None,
        borrowed: bool = False, keys: tuple[str, ...] = (),
    ) -> AgentLoop:
        return self.runner.run(self.graphs.acquire(root, build, config=config, borrowed=borrowed, keys=keys))


@pytest.fixture
async def admitted_async(tmp_path: Path) -> AsyncIterator[AsyncAdmittedFactory]:
    # The explicit fixture is requested only by positive tests, never autouse.
    with pytest.MonkeyPatch.context() as patches:
        graphs = AdmittedGraphs(tmp_path, patches)
        try:
            yield graphs.acquire
        finally:
            await graphs.close()


@pytest.fixture
def admitted_sync(tmp_path: Path) -> Iterator[SyncAdmittedFactory]:
    with pytest.MonkeyPatch.context() as patches:
        graphs = AdmittedGraphs(tmp_path, patches)
        runner = asyncio.Runner()
        owner_loop = runner.get_loop()
        try:
            yield SyncAdmittedFactory(graphs, runner)
        finally:
            try:
                runner.run(graphs.close())
            finally:
                runner.close()
            assert owner_loop.is_closed()
