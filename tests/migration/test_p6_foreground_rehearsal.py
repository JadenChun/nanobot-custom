"""TEST-LOCAL offline foreground qualification, never a public activation seam."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from nanobot import migration_guard
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import RunStatus, TurnCallbacks, TurnRequest, TurnResult
from nanobot.bus.queue import MessageBus
from nanobot.config.errors import ConfigLoadError
from nanobot.config.loader import get_config_path, load_config, set_config_path
from nanobot.config.schema import Config
from nanobot.cron.service import CronService
from nanobot.providers.base import LLMProvider, LLMResponse, LLMUsage
from nanobot.runtime_composition import (
    RuntimeComposition,
    RuntimeFactories,
    RuntimeState,
    runtime_preflight,
)


@dataclass
class _Effects:
    attempts: list[str] = field(default_factory=list)
    provider_calls: int = 0
    sent: list[str] = field(default_factory=list)
    mcp_closes: int = 0

    async def send(self, result: TurnResult) -> None:
        assert result.status is RunStatus.COMPLETED
        self.sent.append(result.content)


class _ScriptedProvider(LLMProvider):
    def __init__(self, effects: _Effects) -> None:
        super().__init__(provider_name="p6-test-scripted")
        self.effects = effects

    async def chat(self, messages, tools=None, model=None, **kwargs) -> LLMResponse:
        self.effects.provider_calls += 1
        assert self.effects.provider_calls == 1, "the rehearsal permits exactly one call"
        assert not tools
        assert model == "p6-test-scripted"
        assert sum(row.get("role") == "user" for row in messages) == 1
        return LLMResponse(content="synthetic accepted answer", usage=LLMUsage.reported(
            input_tokens=128, output_tokens=4,
        ))

    def estimate_prompt_tokens(self, messages, tools=None, model=None):
        # The supported provider-counter seam needs no downloadable tokenizer.
        return len(json.dumps(messages).encode()) + 4 * len(messages), "scripted-byte-budget"

    def get_default_model(self) -> str:
        return "p6-test-scripted"


class _NoMCP:
    def __init__(self, effects: _Effects) -> None:
        self.effects = effects

    async def connect(self) -> None:
        return None

    async def aclose(self) -> None:
        self.effects.mcp_closes += 1


@pytest.fixture
def isolated_effects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    effects = _Effects()

    def denial(label):
        def denied(*_args, **_kwargs):
            effects.attempts.append(label)
            raise AssertionError(f"P6 prohibited effect: {label}")
        return denied

    for name in ("connect", "connect_ex", "bind", "listen", "sendto"):
        monkeypatch.setattr(socket.socket, name, denial(f"network.{name}"))
    for name in ("getaddrinfo", "create_connection"):
        monkeypatch.setattr(socket, name, denial(f"network.{name}"))
    monkeypatch.setattr(subprocess, "Popen", denial("process.Popen"))
    for name in ("system", "fork", "posix_spawn", "posix_spawnp", "execv", "execve"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, denial(f"process.{name}"))
    for name in ("create_subprocess_exec", "create_subprocess_shell", "start_server"):
        monkeypatch.setattr(asyncio, name, denial(f"asyncio.{name}"))
    monkeypatch.setattr(httpx.Client, "send", denial("httpx.send"))
    monkeypatch.setattr(httpx.AsyncClient, "send", denial("httpx.async_send"))
    # Workspace classification computes Path.home even for an explicit workspace.
    # Bind that comparison to an absent synthetic home, never the real user root.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "synthetic-home")
    # The real CLI binds the active path separately from loading its Config.
    # Restore that global after this test; no default/user config is loaded.
    monkeypatch.setattr("nanobot.config.loader._current_config_path", None)
    for target, label in (
        ("nanobot.providers.factory.make_provider", "provider.factory"),
        ("nanobot.channels.manager.ChannelManager.__init__", "channel.construct"),
        ("nanobot.cron.service.CronService.start", "cron.start"),
        ("nanobot.agent.loop.AgentLoop.run", "agent.background_loop"),
        ("nanobot.agent.loop.AgentLoop.schedule_background", "agent.background_task"),
        ("nanobot.agent.tools.mcp.MCPProvider.connect", "mcp.connect"),
    ):
        monkeypatch.setattr(target, denial(label))
    # Exercise the real Settings/loader with no environment source or expansion.
    # The outer OS-isolated runner already supplies a sanitized environment.
    with patch.dict(os.environ, {}, clear=True):
        yield effects


def _reference_config(root: Path) -> tuple[Config, Path]:
    workspace = root / "workspace"
    runtime = root / "reference-instance"
    workspace.mkdir()
    runtime.mkdir()
    path = runtime / "config.json"
    path.write_text(json.dumps({
        "agents": {"defaults": {
            "workspace": str(workspace), "model": "p6-test-scripted",
            "maxToolIterations": 1, "idleCompactAfterMinutes": 0,
            "idleCompaction": {"enabled": False}, "dream": {"enabled": False},
        }},
        "gateway": {"heartbeat": {"enabled": False}},
        "tools": {
            "restrictToWorkspace": True, "mcpServers": {},
            "agentBrowser": {"enabled": False}, "agentDevice": {"enabled": False},
            "desktopUse": {"enabled": False},
        },
        "artifacts": {"enabled": False}, "channels": {},
    }))
    assert path.is_file()  # Missing-file/default fallback is not rehearsal evidence.
    set_config_path(path)
    assert get_config_path() == path
    config = load_config(path)
    assert type(config) is Config  # No alternative Settings subclass/source policy.
    assert config.runtime_data_dir == runtime
    assert config.workspace_path == workspace
    assert not config.gateway.heartbeat.enabled
    assert not config.agents.defaults.dream.enabled
    assert not config.agents.defaults.idle_compaction.enabled
    assert config.agents.defaults.session_ttl_minutes == 0
    assert not config.artifacts.enabled
    return config, path


def _graph(config: Config, effects: _Effects) -> RuntimeComposition:
    provider = _ScriptedProvider(effects)

    def agent(config, bus, sessions, tools, mcp, cron, delivery):
        return AgentLoop.from_config(
            config, bus, tool_registry=tools, provider=provider,
            session_manager=sessions, cron_service=cron, turn_delivery_factory=delivery,
            enable_legacy_auto_compact=False,
        )

    return RuntimeComposition.define(config, factories=RuntimeFactories(
        MessageBus, ToolRegistry, lambda *_: _NoMCP(effects),
        lambda report: CronService(report.workspace / "cron" / "jobs.json"), agent,
    ))


def _tree(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


async def test_positive_foreground_close_restart_retains_one_accepted_turn(
    tmp_path: Path, isolated_effects: _Effects,
) -> None:
    effects = isolated_effects
    config, path = _reference_config(tmp_path)
    config_bytes = path.read_bytes()
    guard_path = Path(migration_guard.__file__)
    guard_hash = hashlib.sha256(guard_path.read_bytes()).hexdigest()
    original_guard = migration_guard.require_runtime_ready
    original_loader = ToolLoader.load
    loader_calls: list[ToolRegistry] = []

    def empty_tools(_loader, _context, registry):
        assert registry.tool_names == []
        loader_calls.append(registry)
        return []

    baseline = _tree(tmp_path)
    tasks = asyncio.all_tasks()
    preflight = runtime_preflight(config)
    assert preflight.session_modes.state == "absent"
    assert _tree(tmp_path) == baseline
    assert effects.provider_calls == 0

    # Only this dedicated test patches the existing guard; no production flag,
    # CLI helper or runtime source can reach this injection.
    with pytest.MonkeyPatch.context() as local:
        local.setattr(migration_guard, "require_runtime_ready", lambda: None)
        local.setattr(ToolLoader, "load", empty_tools)
        first = _graph(config, effects)
        assert first.state is RuntimeState.NEW
        assert _tree(tmp_path) == baseline
        handle = await first.acquire()
        try:
            assert first.state is RuntimeState.READY
            assert type(handle.agent) is AgentLoop
            assert handle.channels is None and handle.cron_runtime is None
            assert handle.tools.tool_names == []
            assert loader_calls == [handle.tools]
            assert not handle.cron._running and handle.cron._timer_task is None
            assert handle.agent._background_tasks == set()
            assert handle.agent.auto_compact is None
            assert handle.agent._loop_task is None and not handle.agent._running
            assert handle.agent._idle_integration is None
            assert handle.agent.local_trigger_store is None
            assert effects.provider_calls == 0  # Clean startup spends no tokens.
            result = await handle.agent.execute_turn(TurnRequest(
                content="synthetic foreground request", session_key="cli:p6-rehearsal",
                callbacks=TurnCallbacks(on_complete=effects.send),
            ))
            assert result.status is RunStatus.COMPLETED
            assert result.content == "synthetic accepted answer"
            assert result.tools_used == []
            assert effects.provider_calls == 1
            assert effects.sent == ["synthetic accepted answer"]
            accepted = handle.sessions.get_or_create("cli:p6-rehearsal").messages
            assert [(row["role"], row["content"]) for row in accepted] == [
                ("user", "synthetic foreground request"),
                ("assistant", "synthetic accepted answer"),
            ]
            backend = handle.sessions.source_backend("cli:p6-rehearsal")
            assert backend is not None  # New accepted history is canonical source.
            source_path = backend.get_session_path("cli:p6-rehearsal")
            persisted = source_path.read_bytes()
            assert persisted
        finally:
            await first.aclose()
        assert first.state is RuntimeState.CLOSED
        assert handle.agent._closed
        assert handle.agent._background_tasks == set()
        assert effects.mcp_closes == 1
        assert not handle.cron._running and handle.cron._timer_task is None
        await first.aclose()
        assert effects.mcp_closes == 1

        second = _graph(load_config(path), effects)
        reopened = await second.acquire()
        try:
            assert second.state is RuntimeState.READY
            assert reopened.sessions is not handle.sessions
            assert reopened.sessions.get_or_create("cli:p6-rehearsal").messages == accepted
            assert source_path.read_bytes() == persisted
            assert effects.provider_calls == 1  # Restart never replays the turn.
            assert reopened.tools.tool_names == []
            assert loader_calls == [handle.tools, reopened.tools]
            assert reopened.channels is None and reopened.cron_runtime is None
        finally:
            await second.aclose()
        assert second.state is RuntimeState.CLOSED
        assert effects.mcp_closes == 2

    assert migration_guard.require_runtime_ready is original_guard
    assert ToolLoader.load is original_loader
    assert hashlib.sha256(guard_path.read_bytes()).hexdigest() == guard_hash
    after = _tree(tmp_path)
    preflight = runtime_preflight(config)
    assert preflight.session_modes.legacy_sessions + preflight.session_modes.canonical_sessions == 1
    assert _tree(tmp_path) == after
    assert path.read_bytes() == config_bytes
    assert effects.attempts == []
    assert effects.provider_calls == 1
    assert asyncio.all_tasks() <= tasks


@pytest.mark.parametrize("effect", [
    "network", "listener", "http", "process", "provider", "channel", "background", "cron", "mcp",
])
def test_prohibited_effects_fail_closed(isolated_effects: _Effects, effect: str) -> None:
    actions = {
        "network": lambda: socket.create_connection(("example.invalid", 443)),
        "listener": lambda: socket.socket.bind(None, ("127.0.0.1", 0)),
        "http": lambda: httpx.AsyncClient.send(None, None),
        "process": lambda: subprocess.Popen(["never-executed"]),
        "provider": lambda: __import__("nanobot.providers.factory", fromlist=["make_provider"]).make_provider(None),
        "channel": lambda: __import__("nanobot.channels.manager", fromlist=["ChannelManager"]).ChannelManager(None, None),
        "background": lambda: AgentLoop.schedule_background(None, None),
        "cron": lambda: CronService.start(None),
        "mcp": lambda: MCPProvider.connect(None),
    }
    with pytest.raises(AssertionError, match="P6 prohibited effect"):
        actions[effect]()
    assert len(isolated_effects.attempts) == 1


def test_legacy_fence_remains_closed_and_foreground_preflight_accepts_clean_config(
    tmp_path: Path, isolated_effects: _Effects,
) -> None:
    config, _path = _reference_config(tmp_path)
    baseline = _tree(tmp_path)
    with pytest.raises(migration_guard.MigrationIncompleteError):
        migration_guard.require_runtime_ready()
    preflight = runtime_preflight(config)
    assert preflight.session_modes.state == "absent"
    assert _tree(tmp_path) == baseline
    assert isolated_effects.provider_calls == 0
    assert isolated_effects.attempts == []


def test_real_reference_loader_rejects_malformed_config_without_default_fallback(
    tmp_path: Path, isolated_effects: _Effects,
) -> None:
    path = tmp_path / "config.json"
    path.write_text("{invalid")
    with pytest.raises(ConfigLoadError) as caught:
        load_config(path)
    assert caught.value.kind == "invalid_json"
    assert isolated_effects.attempts == []
