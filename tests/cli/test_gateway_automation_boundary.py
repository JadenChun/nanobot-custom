"""Real gateway composition keeps legacy automation inert without activation."""

import asyncio
from contextlib import contextmanager

import pytest
import typer

from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.channels.manager import ChannelManager
from nanobot.cli import gateway_runtime
from nanobot.config.schema import Config
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.providers.factory import ProviderSnapshot
from nanobot.providers.unconfigured_provider import UnconfiguredProvider
from nanobot.runtime_composition import RuntimeComposition, RuntimeState
from nanobot.triggers.local_store import LocalTriggerStore
from nanobot.webui.dev import WebUIDevError


def _run_foreground_gateway(
    tmp_path, monkeypatch, *, content, dream, heartbeat, cursor,
    pending=None, setup_error=None, termination="complete",
    process_continuations=False,
    scheduled_execution=False,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = MemoryStore(workspace)
    memory.append_history("first retained entry")
    memory.append_history("second retained entry")
    memory.set_last_dream_cursor(cursor)
    if pending == "trigger":
        triggers = LocalTriggerStore(workspace)
        trigger = triggers.create(
            name="Retained trigger", channel="telegram", chat_id="-100123",
            session_key="telegram:-100123:topic:42", origin_metadata={"message_thread_id": 42},
        )
        triggers.enqueue(trigger.id, "Pending work must not be claimed by the default gateway")
    elif pending is not None:
        cron = CronService(workspace / "cron" / "jobs.json")
        binding = {} if pending == "unbound" else {
            "session_key": "telegram:-100123:topic:42",
            "origin_channel": "telegram", "origin_chat_id": "-100123",
            "origin_metadata": {"message_thread_id": 42},
        }
        cron.add_job(
            "Retained reminder", CronSchedule(kind="every", every_ms=1),
            "Pending work must not execute", deliver=True, channel="telegram", to="-100123",
            **binding,
        )
    # Definition authority may initialize an empty journal during acquisition.
    # Preserve the seeded store exactly; runtime absence is asserted separately.
    automation_folders = () if pending is None else (("triggers",) if pending == "trigger" else ("cron",))
    automation_before = {
        p.relative_to(workspace): p.read_bytes()
        for folder in automation_folders
        for p in (workspace / folder).rglob("*") if p.is_file()
    }
    (workspace / "HEARTBEAT.md").write_text(content)
    before = {p.relative_to(workspace): p.read_bytes()
              for p in (workspace / "memory").rglob("*") if p.is_file()}
    config = Config.model_validate({
        "agents": {"defaults": {
            "workspace": str(workspace), "sessionTtlMinutes": 0,
            "idleCompaction": {"enabled": False}, "dream": {"enabled": dream},
        }},
        "gateway": {"heartbeat": {"enabled": heartbeat}, "processContinuations": process_continuations,
                    "cron": {"enabled": scheduled_execution}},
        "channels": {"websocket": {"enabled": False}},
        "tools": {
            "exec": {"enable": False}, "agentBrowser": {"enabled": False},
            "agentDevice": {"enabled": False}, "desktopUse": {"enabled": False},
        },
    })
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir()
    config.bind_source_path(config_file)
    config_file.write_text(config.model_dump_json(by_alias=True))
    provider = UnconfiguredProvider("test/model")
    snapshot = ProviderSnapshot(provider, "test/model", 4096, ("test",))
    compositions = []
    observed = []
    violations = []
    lifecycle = []
    runtime_tasks = []
    shutdown_events = []
    process_runtimes = []
    install_process_sink = AgentLoop.install_process_completion_sink
    define = RuntimeComposition.define
    agent_run = AgentLoop.run
    agent_close = AgentLoop.aclose
    agent_close_operation = AgentLoop._aclose_operation
    mcp_connect = MCPProvider.connect
    mcp_close = MCPProvider.aclose
    channels_close = ChannelManager.stop_all

    async def run_agent(agent):
        lifecycle.append("agent-start")
        try:
            await agent_run(agent)
        finally:
            lifecycle.append("agent-joined")

    async def close_agent(agent):
        await agent_close(agent)
        lifecycle.append("agent-close")

    async def close_agent_operation(agent):
        await agent_close_operation(agent)
        lifecycle.append("agent-cleanup")

    async def connect_mcp(mcp):
        await mcp_connect(mcp)
        lifecycle.append("mcp-connect")

    async def close_mcp(mcp):
        await mcp_close(mcp)
        lifecycle.append("mcp-close")

    async def close_channels(channels):
        await channels_close(channels)
        lifecycle.append("channels-close")

    def install_handlers(_loop, shutdown, tasks, _print_status):
        runtime_tasks.append(tasks)
        shutdown_events.append(shutdown)

        def restore():
            assert all(task.done() for task in tasks)
            assert compositions[0].state is RuntimeState.CLOSED
            lifecycle.append("restore")

        return restore

    def capture_definition(*args, **kwargs):
        value = define(*args, **kwargs)
        compositions.append(value)
        return value

    def capture_process_sink(agent, sink):
        process_runtimes.append(sink)
        install_process_sink(agent, sink)

    def forbidden(*_args, **_kwargs):
        violations.append("forbidden effect attempted")
        raise AssertionError("foreground gateway must not start legacy automation or external effects")

    async def idle(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            lifecycle.append("foreground-task-joined")

    async def bounded_health(*_args):
        for _ in range(3):
            await asyncio.sleep(0)
        graph = compositions[0]
        handle = graph.require_ready()
        observed.append(graph.state)
        assert handle.cron_runtime is None
        assert handle.cron.on_job is None
        assert not handle.cron._running
        assert (handle.cron.occurrence_store is not None) is scheduled_execution
        assert handle.cron.store_path == workspace / "cron" / "jobs.json"
        assert len(process_runtimes) == int(process_continuations)
        if process_runtimes:
            assert process_runtimes[0]._started and not process_runtimes[0]._closed
        if termination == "failure":
            raise WebUIDevError("injected post-start failure")
        if termination == "shutdown":
            shutdown_events[0].set()
            await asyncio.Event().wait()

    @contextmanager
    def foreground(*_args, **_kwargs):
        yield

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.config.loader.get_config_path", lambda: config_file)
    monkeypatch.setattr(RuntimeComposition, "define", capture_definition)
    monkeypatch.setattr("nanobot.providers.factory.build_provider_snapshot", lambda *_: snapshot)
    monkeypatch.setattr(UnconfiguredProvider, "chat", forbidden)
    monkeypatch.setattr(AgentLoop, "run", run_agent)
    monkeypatch.setattr(AgentLoop, "install_process_completion_sink", capture_process_sink)
    monkeypatch.setattr(AgentLoop, "aclose", close_agent)
    monkeypatch.setattr(AgentLoop, "_aclose_operation", close_agent_operation)
    monkeypatch.setattr(MCPProvider, "connect", connect_mcp)
    monkeypatch.setattr(MCPProvider, "aclose", close_mcp)
    monkeypatch.setattr(ChannelManager, "stop_all", close_channels)
    monkeypatch.setattr(CronService, "start", forbidden)
    monkeypatch.setattr(CronService, "status", forbidden)
    monkeypatch.setattr(CronService, "register_system_job", forbidden)
    monkeypatch.setattr(LocalTriggerStore, "claim_deliveries", forbidden)
    monkeypatch.setattr(LocalTriggerStore, "recover_processing_deliveries", forbidden)
    monkeypatch.setattr(MemoryStore, "set_last_dream_cursor", forbidden)
    monkeypatch.setattr(type(memory.git), "auto_commit", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.bind", forbidden)
    monkeypatch.setattr("subprocess.Popen.__init__", forbidden)
    monkeypatch.setattr(ChannelManager, "start_all", idle)
    monkeypatch.setattr("nanobot.config.watcher.watch_config_file", idle)
    monkeypatch.setattr("nanobot.gateway.runtime.monitor_gateway_clients", idle)
    monkeypatch.setattr("nanobot.gateway.runtime.GatewayRuntime.foreground_instance", foreground)
    monkeypatch.setattr(gateway_runtime, "_install_gateway_shutdown_handlers", install_handlers)
    monkeypatch.setattr(gateway_runtime, "_tcp_endpoint_reachable", lambda *_: False)
    monkeypatch.setattr(gateway_runtime, "_webui_endpoint_reachable", lambda *_: False)
    monkeypatch.setattr(gateway_runtime, "_prepare_webui_bundle_for_gateway", lambda *_a, **_kw: None)
    monkeypatch.setattr(gateway_runtime, "sync_workspace_templates", lambda *_: None)
    monkeypatch.setattr(gateway_runtime, "_health_server", bounded_health)

    if termination == "failure":
        with pytest.raises(typer.Exit) as result:
            gateway_runtime._run_gateway(config, unconfigured_provider_error=setup_error)
        assert result.value.exit_code == 1
    else:
        gateway_runtime._run_gateway(config, unconfigured_provider_error=setup_error)

    assert len(compositions) == 1
    assert violations == []
    assert observed == [RuntimeState.READY]
    assert compositions[0].state is RuntimeState.CLOSED
    assert len(runtime_tasks) == 1 and len(runtime_tasks[0]) == 5 + int(scheduled_execution)
    assert all(task.done() for task in runtime_tasks[0])
    assert lifecycle.count("foreground-task-joined") == 3
    assert lifecycle.count("agent-joined") == 1
    # run() joins its own teardown, then composition close joins that same
    # idempotent operation. Two waiters must still perform cleanup only once.
    assert lifecycle.count("agent-cleanup") == 1
    assert lifecycle.index("agent-cleanup") < lifecycle.index("agent-joined")
    assert lifecycle.index("mcp-connect") < lifecycle.index("agent-start")
    assert lifecycle.index("agent-joined") < lifecycle.index("channels-close")
    assert max(i for i, event in enumerate(lifecycle) if event == "foreground-task-joined") < lifecycle.index("channels-close")
    assert [event for event in lifecycle if event.endswith("-close") or event == "restore"] == [
        "agent-close", "channels-close", "agent-close", "mcp-close", "restore",
    ]
    assert compositions[0].preflight.dream_configured_enabled is dream
    assert compositions[0].preflight.heartbeat_configured_enabled is heartbeat
    assert memory.get_last_dream_cursor() == cursor
    assert (workspace / "HEARTBEAT.md").read_text() == content
    after = {p.relative_to(workspace): p.read_bytes()
             for p in (workspace / "memory").rglob("*") if p.is_file()}
    assert after == before
    assert (config.runtime_data_dir / "cron-runtime").exists() is scheduled_execution
    automation_after = {
        p.relative_to(workspace): p.read_bytes()
        for folder in automation_folders
        for p in (workspace / folder).rglob("*") if p.is_file()
    }
    assert automation_after == automation_before
    if pending is not None:
        assert automation_before
    if process_continuations:
        assert process_runtimes[0]._closed
        assert not process_runtimes[0]._tasks
        assert process_runtimes[0].path.is_relative_to(config.runtime_data_dir)


def test_gateway_owns_explicit_process_completion_runtime(tmp_path, monkeypatch):
    _run_foreground_gateway(tmp_path, monkeypatch, content="", dream=False, heartbeat=False,
                            cursor=0, process_continuations=True)


def test_gateway_owns_explicit_scheduled_execution_runtime(tmp_path, monkeypatch):
    # croniter probes platform architecture once at import. Load dependencies
    # before this fixture denies all subprocess effects during gateway work.
    import nanobot.cron.gateway  # noqa: F401

    _run_foreground_gateway(tmp_path, monkeypatch, content="", dream=False, heartbeat=False,
                            cursor=0, scheduled_execution=True)


@pytest.mark.parametrize("dream,heartbeat", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("cursor", [0, 10])
def test_foreground_gateway_does_not_advance_dream_or_activate_configured_automation(
    tmp_path, monkeypatch, dream, heartbeat, cursor,
):
    _run_foreground_gateway(
        tmp_path, monkeypatch, content="## Active Tasks\n\n- Retained task\n",
        dream=dream, heartbeat=heartbeat, cursor=cursor,
    )


@pytest.mark.parametrize("content", [
    "",
    "# Title\n\n## Active Tasks\n",
    "<!--\nmulti-line\ncomment\n-->\n",
    "<!-- single line -->\n",
    "## Active Tasks\n\n- water the plants\n",
    "## Active Tasks\n\n### Garden\n\n- water the plants\n",
    "## Notes\n\nsome random note\n",
    "stray text before any heading\n## Active Tasks\n\n- task\n",
    "stray text before any heading\n",
])
def test_legacy_heartbeat_file_does_not_activate_gateway_execution(tmp_path, monkeypatch, content):
    _run_foreground_gateway(
        tmp_path, monkeypatch, content=content, dream=True, heartbeat=True, cursor=0,
    )


def test_bundled_heartbeat_template_does_not_activate_gateway_execution(tmp_path, monkeypatch):
    from nanobot.utils.helpers import load_bundled_template

    _run_foreground_gateway(
        tmp_path, monkeypatch, content=load_bundled_template("HEARTBEAT.md"),
        dream=True, heartbeat=True, cursor=0,
    )


@pytest.mark.parametrize("pending", ["unbound", "bound", "trigger"])
@pytest.mark.parametrize("setup_error", [None, "No API key configured"])
def test_default_gateway_preserves_pending_automation_without_execution(
    tmp_path, monkeypatch, pending, setup_error,
):
    _run_foreground_gateway(
        tmp_path, monkeypatch, content="", dream=True, heartbeat=True, cursor=0,
        pending=pending, setup_error=setup_error,
    )


@pytest.mark.parametrize("termination", ["failure", "shutdown"])
def test_gateway_joins_tasks_and_closes_owned_composition_before_exit(
    tmp_path, monkeypatch, termination,
):
    _run_foreground_gateway(
        tmp_path, monkeypatch, content="", dream=False, heartbeat=False, cursor=0,
        termination=termination,
    )
