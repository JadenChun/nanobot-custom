"""Revisiting the same settings must not reopen an old execution revision."""

from types import SimpleNamespace

import pytest

from kohaku.agent.turn import TurnResult
from kohaku.cron.definition_store import (
    CronDefinition,
    CronDefinitionAction,
    DefinitionStoreConflictError,
    DefinitionStoreValidationError,
)
from kohaku.cron.runtime import CronRuntime, GatewayCronOptions
from kohaku.cron.service import CronService
from kohaku.session.manager import SessionManager
from tests.cron.stage3_fixtures import (
    FakeChannels,
    definition,
    definitions,
    offline_denial,  # noqa: F401
)


def _tokens(job):
    return {
        "job_instance_id": job.job_instance_id,
        "definition_revision": job.definition_revision,
    }


def _runtime(tmp_path, item):
    store, _, clock = definitions(tmp_path, item=item)
    service = CronService(tmp_path / "definitions" / "jobs.json", definition_store=store)
    calls = []

    async def compute(request):
        calls.append(request.run_id)
        return TurnResult(run_id=request.run_id, content="fixture")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    graph = CronRuntime(
        service, SimpleNamespace(execute_silent_cron_turn=compute),
        SessionManager(workspace, sessions_root=tmp_path / "sessions"),
        FakeChannels(), tmp_path / "runtime", "test", GatewayCronOptions(clock),
    )
    graph.acquire()
    return graph, service, clock, calls


async def test_resume_has_fresh_cursor_and_does_not_replay_paused_fires(tmp_path):
    item = definition(destinations=())
    graph, service, clock, calls = _runtime(tmp_path, item)
    try:
        current = service.get_job(item.job_id)
        assert current is not None
        clock.now = 1020
        paused = service.enable_job(item.job_id, False, **_tokens(current))
        assert paused is not None and not paused.enabled
        clock.now = 1500
        assert await graph.scheduler.wake_once() is None

        resumed = service.enable_job(item.job_id, True, **_tokens(paused))
        assert resumed is not None and resumed.enabled
        assert resumed.schedule == current.schedule
        assert resumed.job_instance_id == current.job_instance_id
        assert resumed.definition_revision != current.definition_revision
        assert await graph.scheduler.wake_once() is None
        key = f"{resumed.job_instance_id}:{resumed.definition_revision}"
        cursor = next(c for c in graph.store.read()["cursors"] if c["key"] == key)
        assert (cursor["boundary"], cursor["next_fire"]) == (1500, 1600)
        assert not calls

        with pytest.raises(DefinitionStoreConflictError, match="revision"):
            service.enable_job(item.job_id, False, **_tokens(current))

        clock.now = 1600
        assert await graph.scheduler.wake_once() is not None
        assert len(calls) == 1
        records = service.occurrence_store.inspect_records()
        assert [record.identity.scheduled_fire_time_ms for record in records] == [1600]
        assert CronDefinition.decode_wire(records[0].job_snapshot).definition_revision == resumed.definition_revision
    finally:
        await graph.aclose()


@pytest.mark.parametrize("reacquire", [False, True])
@pytest.mark.parametrize("resume_at", [1050, 1300])
async def test_resumed_one_shot_only_fires_if_still_future(tmp_path, reacquire, resume_at):
    schedule = {"kind": "at", "atMs": 1100, "everyMs": None, "expr": None, "tz": None}
    item = definition(schedule=schedule, destinations=())
    graph, service, clock, calls = _runtime(tmp_path, item)
    try:
        current = service.get_job(item.job_id)
        assert current is not None
        paused = service.enable_job(item.job_id, False, **_tokens(current))
        assert paused is not None
        clock.now = resume_at
        resumed = service.enable_job(item.job_id, True, **_tokens(paused))
        assert resumed is not None and resumed.enabled
        if reacquire:
            graph.stop_claiming()
            graph.scheduler.release_lease()
            graph.acquire()
        assert await graph.scheduler.wake_once() is None
        assert calls == []
        key = f"{resumed.job_instance_id}:{resumed.definition_revision}"
        cursor = next(c for c in graph.store.read()["cursors"] if c["key"] == key)
        expected_fire = 1100 if resume_at < 1100 else None
        assert (cursor["boundary"], cursor["next_fire"]) == (resume_at, expected_fire)
        if expected_fire is not None:
            clock.now = expected_fire
            assert await graph.scheduler.wake_once() is not None
            assert len(calls) == 1
        assert await graph.scheduler.wake_once() is None
    finally:
        await graph.aclose()


def test_reverting_an_edit_keeps_old_tokens_stale_and_noop_does_not_reset(tmp_path):
    store, item, _ = definitions(tmp_path)
    service = CronService(tmp_path / "definitions" / "jobs.json", definition_store=store)
    before = service.get_job(item.job_id)
    assert before is not None
    changed = service.update_job(item.job_id, name="changed", **_tokens(before))
    reverted = service.update_job(item.job_id, name=before.name, **_tokens(changed))
    assert reverted.name == before.name
    assert len({before.definition_revision, changed.definition_revision, reverted.definition_revision}) == 3
    with pytest.raises(DefinitionStoreConflictError, match="revision"):
        service.update_job(item.job_id, name="stale", **_tokens(before))
    generation = store.replay().generation
    assert service.update_job(item.job_id, name=before.name, **_tokens(reverted)) == reverted
    assert service.enable_job(item.job_id, True, **_tokens(reverted)) == reverted
    assert store.replay().generation == generation


def test_legacy_wire_unchanged_and_new_revision_binds_its_parent(tmp_path):
    store, item, _ = definitions(tmp_path)
    legacy = item.to_wire()
    assert "revisionParent" not in legacy
    assert CronDefinition.decode_wire(legacy).to_wire() == legacy

    service = CronService(tmp_path / "definitions" / "jobs.json", definition_store=store)
    service.enable_job(item.job_id, False, **_tokens(item))
    current, = store.replay().candidate.definitions
    wire = current.to_wire()
    assert wire["revisionParent"] == item.definition_revision
    assert CronDefinition.decode_wire(wire) == current
    with pytest.raises(DefinitionStoreValidationError, match="definitionRevision"):
        CronDefinition.decode_wire({**wire, "revisionParent": "f" * 64})
    with pytest.raises(DefinitionStoreValidationError, match="parent"):
        CronDefinitionAction.update(
            current, expected_job_instance_id=item.job_instance_id,
            expected_definition_revision="f" * 64,
        )
