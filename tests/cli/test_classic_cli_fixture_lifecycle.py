"""Qualify test-owned CLI composition without enabling the candidate runtime."""

from __future__ import annotations

import asyncio
import importlib
import os
import socket
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn_delivery import TurnDeliveryFactory
from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.cli.commands import app
from nanobot.config.schema import Config, MCPServerConfig
from nanobot.cron.runtime import CronRuntimeInjection
from nanobot.cron.service import CronService
from nanobot.runtime_composition import (
    ChannelBuildOptions,
    RuntimeComposition,
    RuntimeFactories,
    RuntimeState,
)
from nanobot.session.manager import SessionManager
from tests.cli import p6_classic_cli_fixtures

classic_cli_runtime = p6_classic_cli_fixtures.classic_cli_runtime


def _config(tmp_path: Path) -> Config:
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    config.bind_source_path(tmp_path / "instance" / "config.json")
    return config


class _ScriptedAgent:
    """Only the agent is scripted; session/MCP ownership remains native."""

    def __init__(
        self, bus: MessageBus, sessions: SessionManager,
        tools: ToolRegistry, delivery: TurnDeliveryFactory,
    ) -> None:
        self.bus = bus
        self.sessions = sessions
        self.tools = tools
        self.turn_delivery_factory = delivery
        self.owner_loop = asyncio.get_running_loop()
        self.close_loops: list[asyncio.AbstractEventLoop] = []

    async def aclose(self) -> None:
        loop = asyncio.get_running_loop()
        assert loop is self.owner_loop
        self.close_loops.append(loop)


def test_actual_composition_shares_native_graph_and_releases_writer(
    tmp_path, classic_cli_runtime,
):
    config = _config(tmp_path)
    built = []

    def build(config, bus, sessions, tools, mcp, cron, delivery):
        assert isinstance(mcp, MCPProvider)
        assert isinstance(cron, CronService)
        agent = _ScriptedAgent(bus, sessions, tools, delivery)
        built.append(agent)
        return agent

    assert not config.workspace_path.exists()
    graph = RuntimeComposition.define(
        config, factories=replace(RuntimeFactories.defaults(), agent=build),
    )
    assert graph.state is RuntimeState.NEW
    assert config.workspace_path.is_dir()
    assert classic_cli_runtime.compositions == [graph]

    async def exercise():
        handle = await graph.acquire()
        try:
            assert graph.state is RuntimeState.READY
            assert graph.require_ready() is handle
            assert await graph.acquire() is handle
            assert isinstance(handle.bus, MessageBus)
            assert isinstance(handle.sessions, SessionManager)
            assert handle.sessions.has_managed_authority
            assert isinstance(handle.tools, ToolRegistry)
            assert handle.agent is built[0]
            assert handle.agent.bus is handle.bus
            assert handle.agent.sessions is handle.sessions
            assert handle.agent.tools is handle.tools
            assert handle.agent.turn_delivery_factory is handle.delivery
            assert handle.mcp._registry is handle.tools
            assert handle.channels is None and handle.cron_runtime is None
            assert handle.mcp.configured_server_names == set()
            await handle.mcp.connect()
            assert handle.mcp.connected_server_names == set()
            assert not handle.mcp._closing
        finally:
            await graph.aclose()
        assert handle.mcp._closing
        assert handle.mcp.connected_server_names == set()
        assert handle.agent.close_loops == [asyncio.get_running_loop()]
        await graph.aclose()
        assert len(handle.agent.close_loops) == 1

    asyncio.run(exercise())
    assert graph.state is RuntimeState.CLOSED
    assert built[0].owner_loop.is_closed()
    # A different authority on a new loop must reacquire the actual writer lease.
    classic_cli_runtime.verify_closed()


def test_agent_construction_failure_rolls_back_native_mcp_and_writer(
    tmp_path, classic_cli_runtime,
):
    config = _config(tmp_path)
    failure = RuntimeError("scripted construction failure")
    resources = []

    def fail(config, bus, sessions, tools, mcp, cron, delivery):
        assert sessions.has_managed_authority
        assert isinstance(bus, MessageBus)
        resources.append(mcp)
        raise failure

    graph = RuntimeComposition.define(
        config, factories=replace(RuntimeFactories.defaults(), agent=fail),
    )

    async def exercise():
        with pytest.raises(RuntimeError) as caught:
            await graph.acquire()
        assert caught.value is failure
        assert graph.state is RuntimeState.FAILED
        with pytest.raises(RuntimeError) as repeated:
            await graph.acquire()
        assert repeated.value is failure
        await graph.aclose()

    asyncio.run(exercise())
    assert len(resources) == 1
    assert isinstance(resources[0], MCPProvider)
    assert resources[0]._closing
    classic_cli_runtime.verify_closed()


def test_one_shot_body_failure_closes_actual_graph_and_preserves_exception(
    tmp_path, monkeypatch, classic_cli_runtime,
):
    agent_cli = importlib.import_module("nanobot.cli.agent")
    config = _config(tmp_path)
    # Reach the owned graph through real validation, without constructing a client.
    config.agents.defaults.provider = "custom"
    config.providers.custom.api_base = "https://provider.invalid/v1"
    config.providers.custom.api_key = "synthetic-test-key"
    failure = RuntimeError("scripted command-body failure")
    provider = object()
    built = []
    processed = []

    class CommandAgent(_ScriptedAgent):
        @classmethod
        def from_config(cls, received, bus, **options):
            assert received is config
            assert options["provider"] is provider
            assert options["enable_legacy_auto_compact"] is False
            agent = cls(
                bus, options["session_manager"], options["tool_registry"],
                options["turn_delivery_factory"],
            )
            built.append(agent)
            return agent

        async def process_direct(self, message, session_id, **options):
            assert classic_cli_runtime.compositions[0].state is RuntimeState.READY
            assert not classic_cli_runtime.compositions[0].require_ready().mcp._closing
            processed.append((message, session_id))
            raise failure

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)
    monkeypatch.setattr(agent_cli, "_load_runtime_config", lambda *_: config)
    monkeypatch.setattr(agent_cli, "AgentLoop", CommandAgent)
    monkeypatch.setattr(agent_cli, "sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr(agent_cli, "configure_agent_runtime_logs", lambda **_options: None)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)
    result = CliRunner().invoke(app, ["agent", "-m", "hello", "-s", "cli:fixture"])
    assert result.exit_code == 1
    assert result.exception is failure
    assert processed == [("hello", "cli:fixture")]
    assert len(classic_cli_runtime.compositions) == 1
    graph = classic_cli_runtime.compositions[0]
    assert graph.state is RuntimeState.CLOSED
    assert len(built[0].close_loops) == 1
    assert built[0].close_loops[0].is_closed()
    classic_cli_runtime.verify_closed()


@pytest.mark.parametrize("kind", ["foreign", "relative", "parent-traversal", "symlink"])
def test_non_owned_workspace_refused_before_creation_or_definition(
    tmp_path, classic_cli_runtime, kind,
):
    config = _config(tmp_path)
    foreign = tmp_path.parent / (tmp_path.name + "-foreign")
    if kind == "foreign":
        workspace = foreign
    elif kind == "relative":
        workspace = Path("relative-cli-fixture-workspace")
    elif kind == "parent-traversal":
        workspace = tmp_path / ".." / foreign.name
    else:
        (tmp_path / "link").symlink_to(foreign, target_is_directory=True)
        workspace = tmp_path / "link" / "workspace"
    config.agents.defaults.workspace = str(workspace)
    with pytest.raises(AssertionError):
        RuntimeComposition.define(config)
    assert classic_cli_runtime.compositions == []
    assert not foreign.exists()
    assert not (tmp_path / "relative-cli-fixture-workspace").exists()


class _UnusedClock:
    def wall_time_ms(self) -> int:
        raise AssertionError("cron clock must not run")

    def monotonic_time(self) -> float:
        raise AssertionError("cron clock must not run")

    async def sleep(self, seconds: float) -> None:
        raise AssertionError("cron clock must not run")


@pytest.mark.parametrize("configured", ["mcp", "channels", "cron"])
def test_external_configuration_refused_before_workspace_creation(
    tmp_path, classic_cli_runtime, configured,
):
    config = _config(tmp_path)
    channel_options = None
    cron_injection = None
    if configured == "mcp":
        config.tools.mcp_servers["blocked"] = MCPServerConfig(command="must-not-launch")
    elif configured == "channels":
        channel_options = ChannelBuildOptions()
    else:
        cron_injection = CronRuntimeInjection(
            _UnusedClock(), manual_authorizer=lambda *_: False,
        )
    with pytest.raises(AssertionError):
        RuntimeComposition.define(
            config, channel_options=channel_options, cron_injection=cron_injection,
        )
    assert not config.workspace_path.exists()
    assert classic_cli_runtime.compositions == []


_EFFECTS = [
    (socket.socket, name) for name in ("connect", "connect_ex", "bind", "listen", "sendto")
] + [
    (socket, "getaddrinfo"), (socket, "create_connection"), (subprocess, "Popen"),
    (asyncio, "create_subprocess_exec"), (asyncio, "create_subprocess_shell"),
    (ChannelManager, "acquire"), (CronService, "start"), (AgentLoop, "run"),
] + [
    (os, name) for name in ("system", "fork", "posix_spawn", "posix_spawnp", "execv", "execve")
    if hasattr(os, name)
]


@pytest.mark.parametrize(
    "owner,attribute", _EFFECTS, ids=[owner.__name__ + "." + name for owner, name in _EFFECTS],
)
def test_each_external_effect_trap_fails_without_underlying_io(
    classic_cli_runtime, owner, attribute,
):
    with pytest.raises(AssertionError, match="classic CLI fixture external effect denied"):
        getattr(owner, attribute)()
    assert classic_cli_runtime.compositions == []


def test_classic_cli_preflight_rejects_enabled_idle_compaction_without_effects(
    tmp_path, monkeypatch, classic_cli_runtime,
):
    agent_cli = importlib.import_module("nanobot.cli.agent")
    config = _config(tmp_path)
    config.agents.defaults.idle_compaction.enabled = True
    loaded = []

    def load(*_args):
        loaded.append(True)
        return config

    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)
    monkeypatch.setattr(agent_cli, "_load_runtime_config", load)
    # This test intentionally leaves the config-aware foreground preflight real.
    result = CliRunner().invoke(app, ["agent", "-m", "hello"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.exception.code == 1
    assert "Agent cannot start" in result.output
    assert "idle" in result.output.lower()
    assert loaded == [True]
    assert classic_cli_runtime.compositions == []
    assert not config.workspace_path.exists()
