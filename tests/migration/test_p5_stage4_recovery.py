"""Joined real-engine session and whole-runtime cron crash/nonreplay drills."""

import asyncio
from copy import deepcopy

import pytest

from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.runtime_authority import OfflineSessionOperator
from nanobot.session.session_export import export_canonical_session
from nanobot.session.session_mode import SessionExportTransaction
from tests.cron.stage3_fixtures import definitions
from tests.migration.p5_stage4_fixtures import (
    Effects,
    Provider,
    config_at,
    define_graph,
    file_bytes,
    joined,
    offline_denial,  # noqa: F401
    seed_modes,
)


@pytest.mark.parametrize("conversion_phase", ["target_installed", "mode_record_published"])
@pytest.mark.parametrize("export_phase", ["after_install", "after_published"])
async def test_tool_turn_close_convert_export_crash_fresh_operator_runtime_no_replay(
    tmp_path, monkeypatch, conversion_phase, export_phase,
):
    config = config_at(tmp_path)
    key = "cli:restart"
    await seed_modes(config, {key: "legacy"})
    effects = Effects()
    provider = Provider(effects, [LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="effect-1", name="count_effect", arguments={})]),
        LLMResponse(content="tool completed")])
    first = define_graph(config, provider, effects)
    handle = await first.acquire()
    result = await joined(handle.agent.process_direct("count once", session_key=key))
    assert result.content == "tool completed" and effects.tools == 1 and effects.provider == 2
    await joined(first.aclose())
    root = config.runtime_data_dir / "sessions"
    operator = await OfflineSessionOperator(config.workspace_path, copied_sessions_root=root).acquire()
    original = operator.manager._get_session_path(key).read_bytes()
    with pytest.raises(RuntimeError, match="injected"):
        operator.manager._mode_resolver.convert(key, offline_copy=True, fault_after=conversion_phase)
    await operator.close()
    fresh = await OfflineSessionOperator(config.workspace_path, copied_sessions_root=root).acquire()
    recovered = fresh.manager.recover_session_mode(key)
    if conversion_phase == "target_installed":
        assert recovered.mode == "legacy" and fresh.manager._get_session_path(key).read_bytes() == original
        fresh.manager.convert_legacy_session(key, offline_copy=True)
    else:
        assert recovered.mode == "canonical"
    canonical = fresh.manager._get_session_path(key).read_bytes()
    destination = tmp_path / "export-bundle"
    real = SessionExportTransaction.mark_published

    def crash(transaction):
        if export_phase == "after_install":
            real(transaction, fault_after_install=True)
        else:
            real(transaction)
            raise RuntimeError("injected export crash")

    with monkeypatch.context() as injection:
        injection.setattr(SessionExportTransaction, "mark_published", crash)
        with pytest.raises(RuntimeError, match="injected"):
            export_canonical_session(fresh.manager, key, destination)
    await fresh.close()
    final_operator = await OfflineSessionOperator(config.workspace_path, copied_sessions_root=root).acquire()
    assert final_operator.manager.recover_session_mode(key).mode == "canonical"
    assert final_operator.manager._get_session_path(key).read_bytes() == canonical
    assert (destination / "canonical-source.jsonl").read_bytes() == canonical
    await final_operator.close()
    assert effects.tools == 1 and effects.provider == 2
    second_provider = Provider(effects, [LLMResponse(content="next admitted answer")])
    second = define_graph(config, second_provider, effects)
    resumed = await second.acquire()
    try:
        response = await joined(resumed.agent.process_direct("next request", session_key=key))
        assert response.content == "next admitted answer"
        assert resumed.sessions.resolve_session_mode(key).mode == "canonical"
        assert effects.tools == 1 and effects.provider == 3
        assert "fake tool evidence" in str(second_provider.messages[0])
    finally:
        await joined(second.aclose())


async def test_joined_scheduler_missing_observation_bridge_fences_startup_before_effects(tmp_path, monkeypatch):
    from nanobot.cron.action_store import CronActionStore
    from nanobot.cron.definition_store import CronDefinitionStore
    from nanobot.cron.runtime_inspection import reconcile_occurrence
    from nanobot.cron.scheduler_store import SchedulerConflictError, SchedulerStore

    config = config_at(tmp_path)
    store, item, clock = definitions(tmp_path)
    effects = Effects()
    first = define_graph(config, Provider(effects), effects, cron=True, clock=clock, definition_store=store)
    handle = await first.acquire()
    graph = handle.cron_runtime

    async def prebegin(*_args):
        raise OSError("injected before begin")

    clock.now = 1100
    with monkeypatch.context() as injection:
        injection.setattr(graph.scheduler, "_dispatch", prebegin)
        with pytest.raises(OSError):
            await joined(graph.scheduler.wake_once())
    record = handle.cron.occurrence_store.inspect_records()[0]
    occurrence_root, scheduler_root = handle.cron.occurrence_store.root, graph.store.root
    namespace, identity = graph.store.namespace, graph.store.runtime_identity
    await joined(first.aclose())
    with monkeypatch.context() as injection:
        injection.setattr(SchedulerStore, "compare_and_swap",
            lambda *_args: (_ for _ in ()).throw(OSError("between durable authorities")))
        with pytest.raises(OSError):
            reconcile_occurrence(occurrence_root, namespace=namespace,
                occurrence_id=record.occurrence_id, expected_owner=record.owner,
                expected_generation=record.generation, resolution="confirmed_not_started",
                external_evidence="EXTERNAL-EVIDENCE-VERIFIED", successor_owner="explicit-successor",
                scheduler_root=scheduler_root, runtime_identity=identity)
    clock.now = 1200
    new_store = CronDefinitionStore(CronActionStore(tmp_path / "definitions", namespace="stage3"),
                                    commit_clock_ms=clock.wall_time_ms)
    second = define_graph(config, Provider(effects), effects, cron=True, clock=clock, definition_store=new_store)
    before = set(asyncio.all_tasks())
    with pytest.raises(SchedulerConflictError):
        await joined(second.acquire())
    await joined(second.aclose())
    assert effects.provider == effects.tools == effects.callbacks == 0 and effects.sends == []
    assert effects.lifecycle == ["channels-acquire", "channels-close", "mcp-close"] * 2
    assert not [task for task in asyncio.all_tasks() - before if not task.done()]


async def test_real_runner_resistant_effect_keeps_resources_until_join_and_explicit_retry(tmp_path, monkeypatch):
    from nanobot.runtime_errors import RuntimeOwnedWorkError
    from nanobot.session.mode_storage import LifetimeWriterLease

    config = config_at(tmp_path)
    store, item, clock = definitions(tmp_path)
    effects = Effects()
    provider = Provider(effects)
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    real = provider.chat

    async def resistant(*args, **kwargs):
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
        return await real(*args, **kwargs)

    monkeypatch.setattr(provider, "chat", resistant)
    value = define_graph(config, provider, effects, cron=True, clock=clock, definition_store=store)
    handle = await value.acquire()
    graph = handle.cron_runtime
    clock.now = 1100
    wake = asyncio.create_task(graph.scheduler.wake_once())
    try:
        await joined(entered.wait())
        with pytest.raises(ExceptionGroup) as error:
            await joined(value.aclose())
        await joined(cancelled.wait())
        assert isinstance(error.value.exceptions[0], RuntimeOwnedWorkError)
        assert graph.scheduler._lease is not None and "channels-close" not in effects.lifecycle
        assert "mcp-close" not in effects.lifecycle and not graph._accepting
        other = LifetimeWriterLease(config.workspace_path, config.runtime_data_dir / "sessions")
        with pytest.raises(RuntimeError):
            other.acquire()
        release.set()
        await joined(wake)
        await joined(value.aclose())
        assert graph.scheduler._lease is None and effects.provider == 1
        assert effects.sends == ["A", "B"] and not graph._tasks
        assert effects.lifecycle[-2:] == ["channels-close", "mcp-close"]
    finally:
        release.set()
        await joined(asyncio.gather(wake, return_exceptions=True))
        await joined(value.aclose())


@pytest.mark.parametrize("boundary", ["prebegin", "started", "uncertain", "settled"])
async def test_whole_cron_runtime_rebuild_preserves_old_claim_outbox_and_effects(
    tmp_path, monkeypatch, boundary,
):
    from nanobot.cron.action_store import CronActionStore
    from nanobot.cron.callback_store import CronCallbackStore
    from nanobot.cron.composition import compose_claimed_text_occurrence
    from nanobot.cron.definition_store import CronDefinitionStore

    config = config_at(tmp_path)
    store, item, clock = definitions(tmp_path)
    effects = Effects()
    provider = Provider(effects)
    first = define_graph(config, provider, effects, cron=True, clock=clock, definition_store=store)
    handle = await first.acquire()
    graph = handle.cron_runtime
    assert graph.agent is handle.agent and graph.sessions is handle.sessions and graph.channels is handle.channels

    async def observe(_observation):
        effects.callbacks += 1

    callback = CronCallbackStore(tmp_path / "injected-callbacks", observe)

    async def callback_composition(*args, **kwargs):
        return await compose_claimed_text_occurrence(*args, callback_authority=callback, **kwargs)

    clock.now = 1100
    with monkeypatch.context() as fault:
        fault.setattr("nanobot.cron.runtime.compose_claimed_text_occurrence", callback_composition)
        if boundary == "prebegin":
            async def prebegin(*_args):
                raise OSError("injected before begin")
            fault.setattr(graph.scheduler, "_dispatch", prebegin)
        elif boundary == "started":
            def started(*_args, **_kwargs):
                raise OSError("injected after actual computation before completion")
            fault.setattr(handle.cron, "complete_occurrence", started)
        elif boundary == "uncertain":
            handle.channels.fail_send = True
        if boundary in {"prebegin", "started"}:
            with pytest.raises(OSError):
                await joined(graph.scheduler.wake_once())
        else:
            await joined(graph.scheduler.wake_once())
    old = deepcopy(handle.cron.occurrence_store.inspect_records())
    outboxes = deepcopy(handle.cron.list_deliveries())
    counts = (effects.provider, effects.tools, tuple(effects.sends), effects.callbacks)
    old_occurrence_bytes = file_bytes(handle.cron.occurrence_store.root)
    old_outbox_bytes = file_bytes(handle.cron.delivery_runtime.store.root)
    old_callback_bytes = file_bytes(tmp_path / "injected-callbacks")
    assert counts == {
        "prebegin": (0, 0, (), 0), "started": (1, 0, (), 0),
        "uncertain": (1, 0, ("A", "B"), 1), "settled": (1, 0, ("A", "B"), 1),
    }[boundary]
    if boundary == "uncertain":
        assert all(destination.status == "uncertain" for destination in outboxes[0].destinations)
    elif boundary == "settled":
        assert all(destination.status == "delivered" for destination in outboxes[0].destinations)
    if boundary == "started":
        with pytest.raises(ExceptionGroup):
            await joined(first.aclose())
    else:
        await joined(first.aclose())
    assert graph.scheduler._lease is None and not graph._tasks
    clock.now = 1200
    rebuilt_store = CronDefinitionStore(CronActionStore(tmp_path / "definitions", namespace="stage3"),
                                        commit_clock_ms=clock.wall_time_ms)
    second = define_graph(config, Provider(effects), effects, cron=True, clock=clock,
                          definition_store=rebuilt_store)
    restarted = await second.acquire()
    try:
        current = restarted.cron_runtime
        assert current is not graph and current.agent is not handle.agent
        assert restarted.cron.occurrence_store.inspect_records() == old
        assert restarted.cron.list_deliveries() == outboxes
        assert file_bytes(restarted.cron.occurrence_store.root) == old_occurrence_bytes
        assert file_bytes(restarted.cron.delivery_runtime.store.root) == old_outbox_bytes
        assert file_bytes(tmp_path / "injected-callbacks") == old_callback_bytes
        await joined(current.scheduler.wake_once())
        assert (effects.provider, effects.tools, tuple(effects.sends), effects.callbacks) == counts
        capability = current.authorize_manual("new-explicit", item.job_id, item.job_instance_id, item.definition_revision)
        await joined(current.execute_now(capability))
        assert effects.provider == counts[0] + 1 and effects.sends[-2:] == ["A", "B"]
        assert effects.callbacks == counts[3]  # No production callback authority.
        retained = restarted.cron.occurrence_store.inspect_records()
        assert len(retained) == 2 and any(record == old[0] for record in retained)
        assert len({record.occurrence_id for record in retained}) == 2
    finally:
        await joined(second.aclose())
