"""Positive-only admission helper lifecycle, with real offline runtime services."""

import asyncio
import inspect
import socket
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot import migration_guard
from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import Consolidator
from nanobot.agent.subagent import SubagentManager
from nanobot.config import loader
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.runtime_composition import RuntimeState
from nanobot.security import network
from nanobot.session.manager import SessionManager
from nanobot.session.runtime_authority import DurableSessionAuthority
from tests.agent.p6_admitted_fixtures import (
    AdmittedGraphs,
    AgentInputs,
    SyncAdmittedFactory,
)


def _build(inputs: AgentInputs) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.supports_tools = True
    provider.generation = GenerationSettings(max_tokens=100)
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done"))
    loop = AgentLoop(
        bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
        context_window_tokens=32_000, session_manager=inputs.sessions,
        tool_registry=inputs.tools, cron_service=inputs.cron,
        turn_delivery_factory=inputs.delivery, tools_config=inputs.config.tools,
        restrict_to_workspace=True, enable_legacy_auto_compact=False,
    )
    assert loop.bus is inputs.bus
    assert loop.sessions is inputs.sessions
    assert loop.tools is inputs.tools
    assert loop.cron_service is inputs.cron
    assert loop.turn_delivery_factory is inputs.delivery
    assert isinstance(loop.sessions, SessionManager)
    assert isinstance(loop.subagents, SubagentManager)
    assert isinstance(loop.consolidator, Consolidator)
    assert loop.consolidator.store is loop.context.memory
    assert loop.context.memory.workspace == inputs.config.workspace_path
    assert loop.sessions.has_managed_authority
    assert not inputs.mcp.configured_server_names
    assert not inputs.mcp.connected_server_names
    assert not inputs.cron._running
    assert inputs.config._source_path is not None
    assert inputs.config.runtime_data_dir == inputs.config._source_path.parent
    return loop


async def _reacquire(graphs: AdmittedGraphs) -> None:
    for graph, borrowed in graphs.graphs:
        assert not borrowed
        lease = await DurableSessionAuthority(
            graph.preflight.workspace, sessions_root=graph.preflight.sessions_root,
        ).acquire()
        assert lease.close is not None
        await lease.close()


def _globals() -> tuple[object, ...]:
    return (
        migration_guard.require_runtime_ready, inspect.getattr_static(Path, "home"),
        socket.socket.connect, socket.getaddrinfo, subprocess.Popen,
        loader._current_config_path, network._allowed_networks,
    )


@pytest.mark.asyncio
async def test_async_real_graph_turn_reverse_close_and_scope_restoration(tmp_path):
    before = _globals()
    owner = asyncio.get_running_loop()
    closed: list[tuple[str, asyncio.AbstractEventLoop]] = []
    with pytest.MonkeyPatch.context() as patches:
        graphs = AdmittedGraphs(tmp_path, patches)
        try:
            first = await graphs.acquire(tmp_path, _build, keys=("dream:test",))
            second = await graphs.acquire(tmp_path, _build, keys=("cli:normal",))
            for loop, key in ((first, "dream:test"), (second, "cli:normal")):
                mode = loop.sessions.resolve_session_mode(key)
                assert mode.mode == "legacy" and mode.binding is not None
                response = await loop.process_direct("test", session_key=key)
                assert response is not None and response.content == "done"
                loop.provider.chat_with_retry.assert_awaited_once()
                # Composition captured the actual bound close at acquisition.
                # Observe its real close through the instance's subagent owner.
                sub_close = loop.subagents.close

                async def close_subagents(sub_close=sub_close, key=key):
                    closed.append((key, asyncio.get_running_loop()))
                    await sub_close()

                patches.setattr(loop.subagents, "close", close_subagents)
            with pytest.raises(AssertionError, match="external effect denied"):
                socket.getaddrinfo("example.invalid", 443)
        finally:
            await graphs.close()
        assert closed == [("cli:normal", owner), ("dream:test", owner)]
        assert all(graph._owner_loop is owner for graph, _ in graphs.graphs)
        assert all(graph.state is RuntimeState.CLOSED for graph, _ in graphs.graphs)
        assert first._closed and second._closed
        await _reacquire(graphs)
    assert _globals() == before
    with pytest.raises(migration_guard.MigrationIncompleteError):
        migration_guard.require_runtime_ready()


def test_sync_retains_one_runner_through_acquire_and_close(tmp_path):
    before = _globals()
    with pytest.MonkeyPatch.context() as patches:
        graphs = AdmittedGraphs(tmp_path, patches)
        runner = asyncio.Runner()
        owner = runner.get_loop()
        factory = SyncAdmittedFactory(graphs, runner)
        try:
            loop = factory(tmp_path, _build, keys=("dream:test",))
            assert factory.runner is runner
            assert graphs.graphs[0][0]._owner_loop is owner
            assert not owner.is_running()
            assert loop.model == "test-model"
            response = runner.run(loop.process_direct("test", session_key="dream:test"))
            assert response is not None and response.content == "done"
        finally:
            try:
                runner.run(graphs.close())
                runner.run(_reacquire(graphs))
            finally:
                runner.close()
        assert owner.is_closed()
        assert loop._closed
        assert graphs.graphs[0][0].state is RuntimeState.CLOSED
    assert _globals() == before
    with pytest.raises(migration_guard.MigrationIncompleteError):
        migration_guard.require_runtime_ready()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["builder", "identity"])
async def test_acquisition_failure_real_rollback_releases_writer_and_restores(tmp_path, failure):
    before = _globals()
    inputs_seen: list[AgentInputs] = []
    loops: list[AgentLoop] = []
    with pytest.MonkeyPatch.context() as patches:
        graphs = AdmittedGraphs(tmp_path, patches)

        def fail(inputs: AgentInputs) -> AgentLoop:
            inputs_seen.append(inputs)
            if failure == "builder":
                raise ValueError("builder refused")
            loop = _build(inputs)
            loops.append(loop)
            # Fail the real composition identity check after registering agent close.
            loop.turn_delivery_factory = None
            return loop

        try:
            with pytest.raises(ValueError, match="builder refused|share composition delivery"):
                await graphs.acquire(tmp_path, fail, keys=("dream:test",))
            graph, borrowed = graphs.graphs[0]
            assert not borrowed
            assert graph.state is RuntimeState.FAILED
            assert graph._owner_loop is asyncio.get_running_loop()
            assert inputs_seen[0].mcp._closing
            assert all(loop._closed for loop in loops)
            await _reacquire(graphs)
        finally:
            await graphs.close()
    assert _globals() == before
    with pytest.raises(migration_guard.MigrationIncompleteError):
        migration_guard.require_runtime_ready()
