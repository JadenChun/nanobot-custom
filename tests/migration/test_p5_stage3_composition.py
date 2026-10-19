from types import SimpleNamespace

import pytest

from nanobot.agent.turn import TurnResult
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.cron.service import CronService
from nanobot.runtime_composition import ChannelBuildOptions, RuntimeComposition, RuntimeFactories
from tests.cron.stage3_fixtures import (
    FakeChannels,
    FakeClock,
    definitions,
    offline_denial,  # noqa: F401
)


def composition(tmp_path, *, injection=None, events=None, compute=None):
    from nanobot.cron.runtime import CronRuntimeInjection

    events = events if events is not None else []
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    config = Config.model_validate({"agents": {"defaults": {"workspace": str(workspace)}}})
    config.bind_source_path(tmp_path / "runtime-data" / "config.json")
    clock = FakeClock()
    store, item, clock = definitions(tmp_path, clock)
    channels = FakeChannels()

    async def acquire():
        events.append("channels-acquire")

    async def channel_close():
        events.append("channels-close")

    channels.acquire, channels.stop_all = acquire, channel_close

    async def model(request):
        events.append("compute")
        return TurnResult(run_id=request.run_id, content="private result")

    async def mcp_close():
        events.append("mcp-close")

    async def agent_close():
        events.append("agent-close")

    def agent(_config, bus, sessions, _tools, _mcp, cron, delivery):
        return SimpleNamespace(bus=bus, sessions=sessions, cron_service=cron,
            turn_delivery_factory=delivery, aclose=agent_close,
            execute_silent_cron_turn=compute or model)

    factories = RuntimeFactories(MessageBus, lambda: object(),
        lambda *_: SimpleNamespace(aclose=mcp_close),
        lambda _: CronService(tmp_path / "definitions" / "jobs.json", definition_store=store), agent)
    selected = CronRuntimeInjection(clock, lambda *_: True, 0.01) if injection is True else injection
    value = RuntimeComposition.define(config, factories=factories,
        channel_options=ChannelBuildOptions(factory=lambda *_args, **_kwargs: channels),
        startup_receipt_admitted=True, cron_injection=selected)
    return value, item, clock, events, channels


async def test_default_acquire_close_has_no_optional_graph(tmp_path):
    value, item, clock, events, channels = composition(tmp_path)
    assert not (tmp_path / "runtime-data" / "cron-runtime").exists()
    handle = await value.acquire()
    assert handle.cron_runtime is None
    assert handle.cron.occurrence_store is None and handle.cron.on_job is None
    assert not (tmp_path / "runtime-data" / "cron-runtime").exists()
    await value.aclose()
    assert events == ["channels-acquire", "channels-close", "agent-close", "mcp-close"]
    assert channels.calls == []


async def test_injected_graph_exact_shared_identities_managed_absent_visible_route(tmp_path):
    value, item, clock, events, channels = composition(tmp_path, injection=True)
    handle = await value.acquire()
    graph = handle.cron_runtime
    assert graph.service is handle.cron and graph.agent is handle.agent
    assert graph.sessions is handle.sessions and graph.channels is handle.channels
    assert graph._timer_task is None
    assert events == ["channels-acquire"] and channels.calls == []
    clock.now = 1100
    await graph.scheduler.wake_once()
    assert channels.calls == ["A", "B"] and events == ["channels-acquire", "compute"]
    visible = handle.sessions.get_or_create("telegram:A")
    assert visible.metadata["source_schema_version"] == 1
    assert len(visible.recent_scheduled_runs()) == 1 and visible.messages == []
    assert not any(path.name == "callbacks" for path in tmp_path.rglob("*"))
    await value.aclose()
    assert graph.scheduler._lease is None


async def test_managed_existing_visible_link_preserves_transcript_and_merges_ring(tmp_path, monkeypatch):
    from nanobot.cron.visible_link import link_completed_text_result
    from nanobot.session.manager import Session

    value, item, clock, events, channels = composition(tmp_path, injection=True)
    handle = await value.acquire()
    manager = handle.sessions
    manager.upsert_session_metadata("telegram:A", {"marker": "unchanged"}, create_if_absent=True, fsync=True)
    from nanobot.session.source import new_source_event

    source = manager.source_backend("telegram:A").source
    captured = source.load_source_snapshot("telegram:A")
    source.append_source_events("telegram:A", captured.state,
        [new_source_event({"role": "user", "content": "retained transcript"}, run_id="old")])
    transcript_bytes = manager._get_session_path("telegram:A").read_bytes().splitlines()[1:]
    original = manager.read_session_snapshot("telegram:A")
    # Deterministic intervening metadata writer before the linker acquires its
    # serialization scope; the linker must reread rather than publish stale ring.
    prior = Session("telegram:A")
    prior.add_scheduled_run(job_id="other", run_id="other", instruction="prior", result="prior", detail_session_key="cron:other")
    manager.upsert_session_metadata("telegram:A", {Session.SCHEDULED_RING_KEY: prior.recent_scheduled_runs()}, fsync=True)
    clock.now = 1100
    await handle.cron_runtime.scheduler.wake_once()
    record = handle.cron.occurrence_store.inspect_records()[0]
    link_completed_text_result(handle.cron, manager, record.occurrence_id)
    current = manager.read_session_snapshot("telegram:A")
    assert current.messages == original.messages
    assert manager._get_session_path("telegram:A").read_bytes().splitlines()[1:] == transcript_bytes
    ring = manager.get_or_create("telegram:A").recent_scheduled_runs()
    assert [entry["run_id"] for entry in ring] == ["other", f"cron-occurrence-{record.occurrence_id}-1"]
    before = manager.get_or_create("telegram:A").metadata.copy()
    monkeypatch.setattr(manager, "upsert_session_metadata", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("failed metadata")))
    with pytest.raises(OSError):
        link_completed_text_result(handle.cron, manager, record.occurrence_id)
    assert manager.get_or_create("telegram:A").metadata == before
    await value.aclose()


async def test_real_agent_runner_provider_once_in_shared_managed_graph(tmp_path, monkeypatch):
    from dataclasses import replace
    from unittest.mock import AsyncMock

    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.providers.base import LLMResponse
    from tests.cron.test_canonical_execution import ScriptedProvider

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    value, item, clock, events, channels = composition(tmp_path, injection=True)
    provider = ScriptedProvider([LLMResponse(content="exact report")])
    value.config.agents.defaults.model = "synthetic-model"
    value.config.agents.defaults.model_preset = None

    def agent(config, bus, sessions, tools, _mcp, cron, delivery):
        return AgentLoop.from_config(config, bus, tool_registry=tools, provider=provider,
            session_manager=sessions, cron_service=cron, turn_delivery_factory=delivery,
            enable_legacy_auto_compact=False)

    value._factories = replace(value._factories, agent=agent, tools=ToolRegistry)
    handle = await value.acquire()
    handle.bus.publish_outbound = AsyncMock(side_effect=AssertionError("ordinary outbound"))
    handle.bus.publish_event = AsyncMock(side_effect=AssertionError("ordinary delivery"))
    capability = handle.cron_runtime.authorize_manual("explicit-one", item.job_id, item.job_instance_id, item.definition_revision)
    result = await handle.cron_runtime.execute_now(capability)
    assert len(provider.calls) == 1 and channels.calls == ["A", "B"]
    assert result.turn.sent_messages == ()
    assert handle.cron_runtime.agent is handle.agent and handle.cron_runtime.sessions is handle.sessions
    detail = handle.sessions.get_or_create("cron:job")
    assert [message["content"] for message in detail.get_run_messages(result.turn.run_id)] == ["private compute", "exact report"]
    assert detail.provider_state is None
    assert len(handle.sessions.get_or_create("telegram:A").recent_scheduled_runs()) == 1
    handle.bus.publish_outbound.assert_not_awaited()
    handle.bus.publish_event.assert_not_awaited()
    assert (await handle.agent.approval_admission.snapshot("cron:job", "private compute")).pending is None
    await value.aclose()
