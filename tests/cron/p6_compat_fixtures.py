"""Positive current-authority counterparts for the legacy cron regressions.

Only copied private stores, fake clocks, fake computation and receipt adapters
are used. Historical test node names remain mapped to these explicit invariants.
"""

import asyncio
import hashlib
import json
import os
import stat
import threading
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from nanobot.agent.turn import TurnResult
from nanobot.cron.action_store import (
    ActionStoreCorruptError,
    ActionStoreLegacyConversionError,
    ActionStorePersistenceError,
    ActionStoreStaleGenerationError,
    CronActionStore,
)
from nanobot.cron.completion import CronCompletionProjector
from nanobot.cron.definition_store import (
    CronDefinition,
    CronDefinitionAction,
    DefinitionStoreConflictError,
)
from nanobot.cron.occurrence_store import OccurrenceExecutionStartedError
from nanobot.cron.runtime import CronWorkFailedError
from nanobot.cron.schedule import next_fire
from nanobot.cron.scheduler_store import SchedulerConflictError
from nanobot.cron.service import (
    CronDefinitionServiceError as CronDefinitionServiceError,
 )
from nanobot.cron.service import (
    CronDefinitionTokenRequiredError,
    CronJobSkippedError,
    CronService,
)
from nanobot.cron.types import CronDestination, CronSchedule
from nanobot.session.manager import SessionManager
from tests.cron.stage3_fixtures import definition, runtime
from tests.cron.test_action_store_conversion import _job, _legacy_root


def tokens(job):
    return {"job_instance_id": job.job_instance_id,
            "definition_revision": job.definition_revision}


def candidate(service):
    return deepcopy(service.action_store.replay())


def record_run_id(record):
    return f"cron-occurrence-{record.occurrence_id}-{record.generation}"


def legacy_job():
    job = _job(job_id="legacy", name="legacy")
    job["schedule"] = {"kind": "every", "everyMs": 60000}
    return job


def convert(tmp_path, job=None, *, actions=None, raw=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    job = legacy_job() if job is None else job
    raw = json.dumps({"version": 1, "jobs": [job]}).encode() if raw is None else raw
    root = _legacy_root(tmp_path, jobs=raw, actions=actions)
    result = CronActionStore.convert_legacy(root, namespace="p6-compat")
    assert (root / "jobs.json").read_bytes() == raw
    digest = hashlib.sha256(raw).hexdigest()
    assert result.jobs_backup_digest == digest
    assert (root / "legacy-backups" / f"{digest}.bin").read_bytes() == raw
    store = CronActionStore(root, namespace="p6-compat")
    return root, result, store


def converted_view(root, store):
    service = CronService(root / "jobs.json", action_store=store)
    jobs = service.list_jobs(include_disabled=True)
    assert all(job.state.run_history == [] and job.state.next_run_at_ms is None
               and job.created_at_ms == job.updated_at_ms == 0 for job in jobs)
    return service, jobs


def strict_scalar(tmp_path, field, value):
    job = legacy_job()
    target = job
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    target[parts[-1]] = value
    root, result, _store = convert(tmp_path / "snapshot", job)
    assert result.snapshot_status == "unavailable" and not result.snapshot_converted
    assert (root / "jobs.json").exists()
    raw = (json.dumps({"action": "add", "params": job}) + "\n").encode()
    (tmp_path / "action").mkdir(parents=True)
    action_root = _legacy_root(tmp_path / "action", actions=raw)
    with pytest.raises(ActionStoreLegacyConversionError):
        CronActionStore.convert_legacy(action_root, namespace="p6-compat")
    assert (action_root / "action.jsonl").read_bytes() == raw


def route_roundtrip(tmp_path, *, update=False, legacy=False, anonymous=False):
    job = legacy_job()
    job["payload"]["additionalDestinations"] = [{"channel": "telegram", "to": "secondary"}]
    if anonymous:
        job["state"]["runHistory"][0].pop("runId")
    if legacy:
        root, result, store = convert(tmp_path, job)
        assert result.snapshot_converted
        evidence = store.replay().candidate["jobs"][0]
        assert evidence["state"]["runHistory"][0]["durationMs"] == 12
        if anonymous:
            assert evidence["state"]["runHistory"][0]["runId"] is None
        service, jobs = converted_view(root, store)
        loaded = jobs[0]
        assert loaded.payload.channel_meta == job["payload"]["channelMeta"]
        assert loaded.payload.origin_metadata == job["payload"]["originMetadata"]
    else:
        path = tmp_path / "definitions" / "jobs.json"
        service = CronService(path)
        loaded = service.add_job("routes", CronSchedule(kind="every", every_ms=60000), "hello",
            deliver=True, channel="telegram", to="primary",
            additional_destinations=[CronDestination("telegram", "secondary")],
            channel_meta={"topic": 42}, session_key="telegram:primary",
            origin_channel="telegram", origin_chat_id="primary", origin_metadata={"thread": {"id": 42}})
        expected = deepcopy(loaded)
        if update:
            before = candidate(service)
            changes = {"channel": "discord", "to": "updated-primary"}
            with pytest.raises(CronDefinitionTokenRequiredError):
                service.update_job(loaded.id, **changes)
            assert candidate(service) == before
            previous_revision = loaded.definition_revision
            expected.payload.channel = "discord"
            expected.payload.to = "updated-primary"
            loaded = service.update_job(loaded.id, **changes, **tokens(loaded))
            assert loaded.definition_revision != previous_revision
            expected.definition_revision = loaded.definition_revision
        assert asdict(loaded) == asdict(expected)
        reopened = CronService(path).get_job(loaded.id)
        assert asdict(reopened) == asdict(expected)
        loaded.payload.origin_metadata["thread"]["id"] = 0
        assert CronService(path).get_job(loaded.id).payload.origin_metadata == {"thread": {"id": 42}}
        assert not path.exists() and not (path.parent / "action.jsonl").exists()
    destinations = loaded.payload.delivery_destinations()
    if legacy:
        assert [item.to for item in destinations] == ["42", "secondary"]
    else:
        assert destinations == [
            CronDestination("discord", "updated-primary") if update else CronDestination("telegram", "primary"),
            CronDestination("telegram", "secondary"),
        ]
    return service, loaded


def add_current(graph, *, name="second", interval=100):
    body = definition(name=name, interval=interval).definition
    body = deepcopy(body)
    body["id"] = name
    body["payload"]["sessionKey"] = f"cron:{name}"
    item = CronDefinition.from_definition(body, job_instance_id=hashlib.sha256(name.encode()).hexdigest())
    graph.service.definition_store.append(action_id=f"create-{name}", action=CronDefinitionAction.create(item))
    graph.service.definition_store.checkpoint(graph.service.definition_store.replay())
    return item


def reacquire(graph):
    graph.stop_claiming()
    graph.scheduler.release_lease()
    graph.acquire()


async def finish(graph):
    await asyncio.wait_for(graph.aclose(), 2)


async def completion(tmp_path, *, status="ok", error=None, forge=False):
    if status != "ok":
        graph, item, _clock, calls, channels = runtime(tmp_path)
        try:
            claim = graph.service.claim_scheduled_occurrence(item.job_id, 1100,
                expected_job_instance_id=item.job_instance_id,
                expected_definition_revision=item.definition_revision)
            body = {"status": status, "error": error, "duration_ms": 12, "run_at_ms": 1100}
            with graph.service.occurrence_store.acquire_execution_fence(claim) as fence:
                graph.service.occurrence_store.begin_execution(claim, fence)
                projection = graph.service.complete_occurrence(claim, body, fence=fence)
            assert projection.status == status
            assert {key: projection.completion[key] for key in body} == body
            assert projection.completion['run_id'] == claim.run_id
            assert projection.completion['occurrence_id'] == claim.occurrence_id
            assert projection.completion['job_revision'] == claim.identity.job_revision
            assert projection.completion['job_id'] == claim.identity.job_id
            assert projection.completion['generation'] == claim.generation
            assert calls == [] and channels.calls == []
            assert graph.service.project_occurrence_completion(claim.occurrence_id) == projection
            assert graph.service.get_job(item.job_id).state.run_history == []
        finally:
            await finish(graph)
        return
    calls = []
    async def compute(request):
        calls.append(request.run_id)
        return TurnResult(run_id="forged" if forge else request.run_id, content="done")
    graph, item, _clock, _calls, channels = runtime(tmp_path,
        agent=SimpleNamespace(execute_silent_cron_turn=compute))
    capability = graph.authorize_manual("manual", item.job_id, item.job_instance_id, item.definition_revision)
    before = candidate(graph.service)
    try:
        if forge:
            with pytest.raises(ValueError, match="run ID"):
                await asyncio.wait_for(graph.execute_now(capability), 2)
            record = graph.service.occurrence_store.inspect_records()[0]
            assert record.state == "claimed" and record.completion is None and channels.calls == []
        else:
            result = await asyncio.wait_for(graph.execute_now(capability), 2)
            record = graph.service.occurrence_store.inspect_records()[0]
            assert record.state == "completed"
            assert result.turn.run_id == result.completion.run_id == record_run_id(record) == calls[0]
            assert result.completion.status == "ok" and not result.completion.completion.get("error")
            assert result.outbox.run_id == record_run_id(record) and channels.calls == ["A", "B"]
            path = graph.service.completion_projector.projection_path(record_run_id(record))
            assert path.is_relative_to(tmp_path / "runtime" / "runs") and path.exists()
            fresh = CronCompletionProjector(graph.service.occurrence_store, path.parent)
            assert fresh.project_completed(record.occurrence_id) == result.completion
            assert graph.service.get_job(item.job_id).state.run_history == []
        assert candidate(graph.service) == before and len(calls) == 1
    finally:
        if forge:
            with pytest.raises(CronWorkFailedError):
                await finish(graph)
        else:
            await finish(graph)


async def compute_failure(tmp_path, *, cancel=False):
    async def compute(_request):
        if cancel:
            raise asyncio.CancelledError("turn cancelled")
        raise RuntimeError("boom")
    graph, item, clock, _calls, channels = runtime(tmp_path,
        agent=SimpleNamespace(execute_silent_cron_turn=compute))
    capability = graph.authorize_manual("failure", item.job_id, item.job_instance_id, item.definition_revision)
    try:
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await asyncio.wait_for(graph.execute_now(capability), 2)
        record = graph.service.occurrence_store.inspect_records()[0]
        assert record.state == "claimed" and record.completion is None and channels.calls == []
        with pytest.raises(OccurrenceExecutionStartedError):
            await asyncio.wait_for(graph.execute_now(capability), 2)
        assert graph.store.read()["cursors"][0]["next_fire"] == clock.now + 100
    finally:
        with pytest.raises(CronWorkFailedError):
            await finish(graph)


async def reentrant(tmp_path, *, scheduled=False, two=False, add=False):
    calls = []
    graph = None
    async def compute(request):
        calls.append(request.run_id)
        assert all(job.state.run_history == [] for job in graph.service.list_jobs(include_disabled=True))
        if add and len(calls) == 1:
            add_current(graph, name="added-during-compute")
        return TurnResult(run_id=request.run_id, content="done")
    graph, item, clock, _calls, channels = runtime(tmp_path,
        agent=SimpleNamespace(execute_silent_cron_turn=compute))
    try:
        if two:
            add_current(graph)
            reacquire(graph)
        before = candidate(graph.service)
        if scheduled:
            clock.now = 1100
            for _ in range(2 if two else 1):
                await asyncio.wait_for(graph.scheduler.wake_once(), 2)
            assert await graph.scheduler.wake_once() is None
            assert {cursor["next_fire"] for cursor in graph.store.read()["cursors"]} == {1200}
        else:
            capability = graph.authorize_manual("read", item.job_id, item.job_instance_id, item.definition_revision)
            await asyncio.wait_for(graph.execute_now(capability), 2)
        records = graph.service.occurrence_store.inspect_records()
        assert len(records) == len(calls) == (2 if two else 1)
        assert len({record.occurrence_id for record in records}) == len(records)
        assert all(record.state == "completed" for record in records)
        if add:
            assert {job.id for job in graph.service.list_jobs()} == {"job", "added-during-compute"}
            assert records[0].job_snapshot["definitionRevision"] == item.definition_revision
            reacquire(graph)
            clock.now = 1100
            await asyncio.wait_for(graph.scheduler.wake_once(), 2)
            assert len(calls) == 2
        else:
            assert candidate(graph.service) == before
        for record in records:
            assert graph.service.project_occurrence_completion(record.occurrence_id).run_id == record_run_id(record)
        assert len(channels.calls) == len(calls) * 2
    finally:
        await finish(graph)


async def overlap(tmp_path, *, closing=False):
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    calls = []
    async def compute(request):
        index = len(calls)
        calls.append(request.run_id)
        entered[index].set()
        await asyncio.wait_for(release[index].wait(), 2)
        return TurnResult(run_id=request.run_id, content="done")
    graph, item, _clock, _calls, channels = runtime(tmp_path,
        agent=SimpleNamespace(execute_silent_cron_turn=compute), timeout=1)
    tasks, close = [], None
    try:
        for index in range(1 if closing else 2):
            capability = graph.authorize_manual(str(index), item.job_id, item.job_instance_id, item.definition_revision)
            tasks.append(asyncio.create_task(graph.execute_now(capability)))
            await asyncio.wait_for(entered[index].wait(), 1)
            if tasks[-1].done():
                await tasks[-1]
        if closing:
            graph.stop_claiming()
            close = asyncio.create_task(graph.aclose())
            with pytest.raises(PermissionError):
                graph.authorize_manual("late", item.job_id, item.job_instance_id, item.definition_revision)
            assert graph.scheduler._lease is not None and not graph._accepting
        release[0].set()
        await asyncio.wait_for(tasks[0], 2)
        if not closing:
            assert not tasks[1].done()
            release[1].set()
            await asyncio.wait_for(tasks[1], 2)
        if close is not None:
            await asyncio.wait_for(close, 2)
            assert graph.scheduler._lease is None
        records = graph.service.occurrence_store.inspect_records()
        assert len(records) == len(calls) == (1 if closing else 2)
        assert len({record_run_id(record) for record in records}) == len(records)
        assert all(record.state == "completed" for record in records)
        assert graph.service._running is False and graph._timer_task is None
        assert channels.calls == ["A", "B"] * len(calls)
    finally:
        for event in release:
            event.set()
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
        if close is not None:
            await asyncio.wait_for(asyncio.gather(close, return_exceptions=True), 2)
        await finish(graph)


async def external_scheduler(tmp_path, *, disable=False):
    graph, item, clock, calls, channels = runtime(tmp_path)
    try:
        second = add_current(graph)
        # New commits have no cursor: lease possession never invents one.
        with pytest.raises(SchedulerConflictError, match="reacquisition"):
            await graph.scheduler.wake_once()
        reacquire(graph)
        external = CronService(graph.service.store_path, definition_store=graph.service.definition_store)
        if disable:
            old = external.get_job(item.job_id)
            disabled = external.enable_job(item.job_id, False, **tokens(old))
            assert disabled.enabled is False
            assert {job.id for job in graph.service.list_jobs()} == {second.job_id}
            assert len(graph.service.list_jobs(include_disabled=True)) == 2
            with pytest.raises(SchedulerConflictError):
                graph.authorize_manual("disabled", item.job_id, item.job_instance_id, item.definition_revision)
        clock.now = 1100
        for _ in range(1 if disable else 2):
            await asyncio.wait_for(graph.scheduler.wake_once(), 2)
        assert await graph.scheduler.wake_once() is None
        records = graph.service.occurrence_store.inspect_records()
        assert {record.identity.job_id for record in records} == ({second.job_id} if disable else {item.job_id, second.job_id})
        assert {record.identity.scheduled_fire_time_ms for record in records} == {1100}
        assert {cursor["next_fire"] for cursor in graph.store.read()["cursors"] if cursor["job_id"] != item.job_id or not disable} == {1200}
        assert len(calls) == len(records) and len(channels.calls) == 2 * len(calls)
    finally:
        await finish(graph)


async def history_mutation(tmp_path):
    graph, item, _clock, calls, _channels = runtime(tmp_path)
    try:
        capability = graph.authorize_manual("history", item.job_id, item.job_instance_id, item.definition_revision)
        result = await asyncio.wait_for(graph.execute_now(capability), 2)
        record = graph.service.occurrence_store.inspect_records()[0]
        path = graph.service.completion_projector.projection_path(record_run_id(record))
        before = path.read_bytes()
        external = CronService(graph.service.store_path, definition_store=graph.service.definition_store)
        old = external.get_job(item.job_id)
        changed = external.update_job(old.id, name="renamed", **tokens(old))
        disabled = external.enable_job(old.id, False, **tokens(changed))
        assert disabled.definition_revision != item.definition_revision
        assert disabled.job_instance_id == item.job_instance_id and disabled.state.run_history == []
        assert graph.service.occurrence_store.get(record.occurrence_id) == record
        assert record.job_snapshot["definitionRevision"] == item.definition_revision
        assert graph.service.project_occurrence_completion(record.occurrence_id) == result.completion
        assert path.read_bytes() == before and len(calls) == 1
        with pytest.raises(SchedulerConflictError):
            graph.authorize_manual("stale", item.job_id, item.job_instance_id, item.definition_revision)
        fresh = CronCompletionProjector(graph.service.occurrence_store, path.parent)
        assert fresh.project_completed(record.occurrence_id) == result.completion
    finally:
        await finish(graph)


async def ring(tmp_path):
    graph, item, _clock, calls, _channels = runtime(tmp_path)
    visible = graph.sessions.get_or_create("telegram:A")
    visible.metadata["unrelated"] = {"keep": True}
    visible.add_message("user", "historical input")
    graph.sessions.save(visible)
    try:
        count = visible.SCHEDULED_RING_LIMIT + 2
        for index in range(count):
            capability = graph.authorize_manual(f"ring-{index}", item.job_id, item.job_instance_id, item.definition_revision)
            await asyncio.wait_for(graph.execute_now(capability), 2)
        reopened = SessionManager(graph.sessions.workspace, sessions_root=graph.sessions.sessions_root)
        view = reopened.get_or_create("telegram:A")
        recent = view.recent_scheduled_runs()
        assert len(recent) == visible.SCHEDULED_RING_LIMIT
        assert len({entry["run_id"] for entry in recent}) == len(recent)
        assert {entry["run_id"] for entry in recent} == set(calls[-visible.SCHEDULED_RING_LIMIT:])
        assert view.metadata["unrelated"] == {"keep": True}
        assert [message["content"] for message in view.messages] == ["historical input"]
        records = graph.service.occurrence_store.inspect_records()
        assert len(records) == count
        assert all(graph.service.project_occurrence_completion(record.occurrence_id).run_id == record_run_id(record) for record in records)
    finally:
        await finish(graph)


def changed_definition(tmp_path, field):
    service = CronService(tmp_path / "jobs.json")
    old = service.add_job("old", CronSchedule(kind="every", every_ms=60000), "hello",
        deliver=True, channel="telegram", to="primary",
        additional_destinations=[CronDestination("telegram", "secondary")],
        channel_meta={"topic": 42}, session_key="telegram:primary",
        origin_channel="telegram", origin_chat_id="primary", origin_metadata={"thread": {"id": 42}})
    changes = {
        "name": {"name": "new"},
        "message": {"message": "changed"},
        "schedule": {"schedule": CronSchedule(kind="every", every_ms=120000)},
        "cron": {"schedule": CronSchedule(kind="cron", expr="0 18 * * *", tz="UTC")},
    }[field]
    expected = deepcopy(old)
    if field == "name":
        expected.name = "new"
    elif field == "message":
        expected.payload.message = "changed"
    else:
        expected.schedule = deepcopy(changes["schedule"])
    before = candidate(service)
    with pytest.raises(CronDefinitionTokenRequiredError):
        service.update_job(old.id, **changes)
    assert candidate(service) == before
    new = service.update_job(old.id, **changes, **tokens(old))
    assert new.job_instance_id == old.job_instance_id and new.definition_revision != old.definition_revision
    expected.definition_revision = new.definition_revision
    assert asdict(new) == asdict(expected)
    assert asdict(CronService(service.store_path).get_job(old.id)) == asdict(expected)
    assert new.state.next_run_at_ms is None
    commits = service.definition_store.replay().candidate.scheduling_commits
    commit = next(item for item in commits if item["definition_revision"] == new.definition_revision)
    assert commit["schedule"] == expected.to_store_dict()["schedule"]
    assert next_fire(commit["schedule"], commit["commit_clock_ms"], commit["commit_clock_ms"]) > commit["commit_clock_ms"]
    assert len(candidate(service).receipts) == len(before.receipts) + 1
    assert not service.store_path.exists()
    return service, old, new


async def definition_fence(tmp_path, *, running=False):
    called = []
    async def callback(job):
        called.append(job.id)
    service = CronService(tmp_path / "jobs.json", on_job=callback)
    job = service.add_job("definition", CronSchedule(kind="every", every_ms=100), "hello")
    before = candidate(service)
    if running:
        await service.start()
    try:
        assert await service.run_job(job.id, force=True) is False
        await service._on_timer()
        with pytest.raises(CronJobSkippedError, match="adapter is not installed"):
            await service._execute_job(job)
        with pytest.raises(CronJobSkippedError):
            service.write_run_record("unbound", {"status": "ok"})
        assert service._running is running and service._timer_task is None
        assert service._get_next_wake_ms() is None and called == []
        assert candidate(service) == before and not service._run_records_dir.exists()
    finally:
        service.stop()


def typed_roundtrip(tmp_path):
    service = CronService(tmp_path / "jobs.json")
    job = service.add_job("Daily Loving Message", CronSchedule(kind="cron", expr="0 10 * * *", tz="Asia/Kuwait"), "hello",
        deliver=True, channel="telegram", to="42", session_key="telegram:42",
        origin_channel="telegram", origin_chat_id="42", origin_metadata={"nested": {"keep": True}})
    expected = asdict(job)
    generation = candidate(service).generation
    for _ in range(3):
        service._save_store()
        service = CronService(service.store_path)
        assert [asdict(item) for item in service.list_jobs(include_disabled=True)] == [expected]
        assert asdict(service.get_job(job.id)) == expected
        assert candidate(service).generation == generation
    assert not service.store_path.exists() and not (service.store_path.parent / "action.jsonl").exists()
    assert not list(service.store_path.parent.glob("*.tmp"))
    assert set(candidate(service).candidate["definitions"][0]).isdisjoint({"state", "createdAtMs", "updatedAtMs"})


def failed_checkpoint(tmp_path, monkeypatch, *, directory=False):
    service = CronService(tmp_path / "jobs.json")
    job = service.add_job("old", CronSchedule(kind="every", every_ms=100), "hello")
    store = service.action_store
    snapshot = store.root / "snapshot.json"
    before = snapshot.read_bytes()
    prior = store.replay()
    if directory:
        real = os.fsync
        def fail(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(22, "directory fsync rejected")
            real(fd)
        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", fail)
            with pytest.raises(ActionStorePersistenceError):
                store.checkpoint(expected_generation=prior.generation, candidate=prior.candidate)
        recovered = store.checkpoint(expected_generation=prior.generation, candidate=prior.candidate)
        assert recovered.candidate == prior.candidate and recovered.receipts == prior.receipts
    else:
        def fail(*_args, **_kwargs):
            raise ActionStorePersistenceError("snapshot publication rejected")
        with monkeypatch.context() as patch:
            patch.setattr(store, "_replace_file", fail)
            with pytest.raises(ActionStorePersistenceError):
                service.update_job(job.id, name="new", **tokens(job))
        assert snapshot.read_bytes() == before
        pending = store.replay()
        assert len(pending.pending_actions) == 1 and len(pending.receipts) == len(prior.receipts) + 1
        receipt = pending.receipts[-1]
        fresh = CronService(service.store_path)
        assert fresh.get_job(job.id).name == "new"
        assert fresh.action_store.replay().receipts[-1] == receipt
        assert fresh.action_store.replay().pending_actions == ()
    assert not service.store_path.exists()


def corrupt_authority(tmp_path, *, converted=False):
    if converted:
        root, _result, store = convert(tmp_path)
        service, jobs = converted_view(root, store)
        assert len(jobs) == 1
        path = root / "jobs.json"
        original = path.read_bytes()
        backup = root / "legacy-backups" / f"{hashlib.sha256(original).hexdigest()}.bin"
    else:
        service = CronService(tmp_path / "jobs.json")
        service.add_job("acknowledged", CronSchedule(kind="every", every_ms=100), "hello")
        path = service.action_store.root / "snapshot.json"
        original = path.read_bytes()
        backup = None
    path.write_bytes(b"{not valid json")
    with pytest.raises(ActionStoreCorruptError):
        service.list_jobs(include_disabled=True)
    assert path.read_bytes() == b"{not valid json"
    if backup is not None:
        assert backup.read_bytes() == original
    assert not list(path.parent.glob("*.corrupt-*"))


def scalar_siblings(tmp_path, *, null=False):
    fields = ["state.runHistory.0.runAtMs", "state.runHistory.0.durationMs", "createdAtMs", "updatedAtMs"] if null else ["schedule.everyMs", "state.nextRunAtMs", "state.lastRunAtMs"]
    for index, field in enumerate(fields):
        strict_scalar(tmp_path / str(index), field, None if null else "60000")
    job = legacy_job()
    job["state"]["nextRunAtMs"] = job["state"]["lastRunAtMs"] = None
    _root, result, store = convert(tmp_path / "valid", job)
    assert result.snapshot_converted
    assert store.replay().candidate["jobs"][0]["state"]["runHistory"][0]["durationMs"] == 12


def alias_conversion(tmp_path):
    job = _job(snake_case=True)
    job["schedule"] = {"kind": "every", "every_ms": 60000}
    job["payload"]["session_key"] = "websocket:chat-1"
    job["state"]["run_history"][0]["run_at_ms"] = 1000
    root, result, store = convert(tmp_path, job)
    assert result.snapshot_converted
    legacy = store.replay().candidate["jobs"][0]
    assert legacy["schedule"]["everyMs"] == 60000
    assert legacy["payload"]["sessionKey"] == "websocket:chat-1"
    assert legacy["state"]["runHistory"][0]["runAtMs"] == 1000
    assert legacy["state"]["runHistory"][0]["durationMs"] == 12
    service, jobs = converted_view(root, store)
    assert jobs[0].job_instance_id and jobs[0].definition_revision
    with pytest.raises(RuntimeError, match='legacy jobs.json is not a cron definition authority'):
        service._load_jobs()


def definition_display(tool, *, field):
    job = tool._cron.add_job('Definition', CronSchedule(kind='cron', expr='0 9 * * *', tz='UTC'), 'test', session_key='websocket:chat-1', origin_channel='websocket', origin_chat_id='chat-1')
    before = candidate(tool._cron)
    job.state.last_run_at_ms = 1773673200000
    job.state.last_status = 'error' if field == 'error' else 'ok'
    job.state.last_error = 'forged-timeout' if field == 'error' else None
    job.state.next_run_at_ms = 1773684000000
    tool._cron._save_store()
    result = tool._list_jobs()
    assert 'cron: 0 9 * * * (UTC)' in result
    assert 'Last run:' not in result and 'Next run:' not in result and 'forged-timeout' not in result
    assert candidate(tool._cron) == before
    loaded = tool._cron.get_job(job.id)
    assert loaded.state.last_status is None and loaded.state.next_run_at_ms is None
    assert loaded.state.last_run_at_ms is None and loaded.state.last_error is None
    assert not tool._cron.store_path.exists()


async def unsupported_execution(tmp_path):
    from unittest.mock import AsyncMock

    from nanobot.cron.canonical_execution import execute_claimed_fresh_text
    from tests.cron.test_canonical_execution import _service_claim
    service, claim = _service_claim(tmp_path, legacy=False)
    agent = AsyncMock()
    with pytest.raises(ValueError, match='legacy_fresh'):
        await asyncio.wait_for(execute_claimed_fresh_text(service, agent, claim), 2)
    agent.execute_silent_cron_turn.assert_not_awaited()
    record = service.occurrence_store.get(claim.occurrence_id)
    assert record.state == 'claimed' and record.completion is None


def unbound_conversion(tmp_path, *, malformed=False):
    job = legacy_job()
    job["payload"] = {"kind": "agent_turn", "message": "hello", "deliver": malformed}
    if malformed:
        # Structurally invalid scalar, unlike an unresolved but valid binding.
        job["payload"]["message"] = {"not": "text"}
    root, result, store = convert(tmp_path, job)
    if malformed:
        assert result.snapshot_status == "unavailable"
    else:
        assert result.snapshot_converted
        service, jobs = converted_view(root, store)
        assert jobs[0].enabled and service.admission_diagnostic(jobs[0])
        assert store.replay().candidate["definitions"][0]["enabled"]


def history_elements(tmp_path):
    job = legacy_job()
    job["state"]["runHistory"].append(None)
    _root, result, _store = convert(tmp_path / "invalid", job)
    assert result.snapshot_status == "unavailable"
    job["state"]["runHistory"] = [legacy_job()["state"]["runHistory"][0],
        {"runAtMs": 100, "durationMs": 2, "status": "ok"}]
    _root, result, store = convert(tmp_path / "valid", job)
    assert result.snapshot_converted
    assert [record["runAtMs"] for record in store.replay().candidate["jobs"][0]["state"]["runHistory"]] == [90, 100]


def stale_remove(tmp_path):
    one = CronService(tmp_path / "jobs.json")
    first = one.add_job("first", CronSchedule(kind="every", every_ms=100), "first")
    two = CronService(one.store_path)
    stale = two.get_job(first.id)
    second = one.add_job("second", CronSchedule(kind="every", every_ms=100), "second")
    assert two.remove_job(first.id, **tokens(stale)) == "removed"
    assert {job.id for job in CronService(one.store_path).list_jobs()} == {second.id}
    changed = one.update_job(second.id, name="changed", **tokens(second))
    before = candidate(one)
    with pytest.raises(DefinitionStoreConflictError):
        two.remove_job(second.id, **tokens(second))
    assert candidate(one) == before and two.get_job(second.id).definition_revision == changed.definition_revision


def concurrent_cas(tmp_path):
    service = CronService(tmp_path / "jobs.json")
    removed = service.add_job("remove-me", CronSchedule(kind="every", every_ms=100), "remove")
    keep = service.add_job("keep-me", CronSchedule(kind="every", every_ms=100), "keep")
    prior = service.action_store.replay()
    done, errors = threading.Event(), []
    def mutate():
        try:
            external = CronService(service.store_path)
            external.remove_job(removed.id, **tokens(removed))
            external.add_job("added-by-cli", CronSchedule(kind="every", every_ms=100), "added")
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()
    thread = threading.Thread(target=mutate)
    thread.start()
    try:
        assert done.wait(2)
        with pytest.raises(ActionStoreStaleGenerationError):
            service.action_store.checkpoint(expected_generation=prior.generation, candidate=prior.candidate)
    finally:
        thread.join(2)
    assert not thread.is_alive() and not errors
    fresh = CronService(service.store_path)
    assert {job.name for job in fresh.list_jobs()} == {keep.name, "added-by-cli"}
    assert all(commit["definition_revision"] in {job.definition_revision for job in fresh.list_jobs()} for commit in fresh.definition_store.replay().candidate.scheduling_commits if commit["job_instance_id"] in {job.job_instance_id for job in fresh.list_jobs()})


async def projection_retry(tmp_path, monkeypatch):
    graph, item, _clock, calls, channels = runtime(tmp_path)
    capability = graph.authorize_manual("retry", item.job_id, item.job_instance_id, item.definition_revision)
    def fail(*_args, **_kwargs):
        raise OSError("projection publication rejected")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(graph.service.completion_projector, "_replace_projection", fail)
            with pytest.raises(OSError, match="projection"):
                await asyncio.wait_for(graph.execute_now(capability), 2)
        record = graph.service.occurrence_store.inspect_records()[0]
        assert record.state == "completed" and len(calls) == 1 and channels.calls == []
        projection = graph.service.project_occurrence_completion(record.occurrence_id)
        assert projection.run_id == record_run_id(record)
        graph.service.prepare_completed_delivery(record.occurrence_id)
        from nanobot.cron.occurrence_store import OccurrenceAlreadyCompletedError
        with pytest.raises(OccurrenceAlreadyCompletedError):
            await asyncio.wait_for(graph.execute_now(capability), 2)
        assert len(calls) == 1
    finally:
        with pytest.raises(CronWorkFailedError):
            await finish(graph)
