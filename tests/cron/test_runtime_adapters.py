import pytest

from nanobot.cron.delivery_store import DeliveryOutboxCorruptError, _validate_occurrence_identity
from nanobot.cron.occurrence_store import (
    CronOccurrenceStore,
    OccurrenceAlreadyCompletedError,
    OccurrenceCorruptError,
    ScheduledOccurrenceIdentity,
    TriggeredOccurrenceIdentity,
    _occurrence_id,
)
from tests.cron.stage3_fixtures import offline_denial, runtime  # noqa: F401


async def test_unauthorized_allocates_nothing_and_equal_clock_requests_unique(tmp_path):
    graph, item, clock, calls, channels = runtime(tmp_path, authorizer=lambda *_: False)
    with pytest.raises(PermissionError):
        graph.authorize_manual("request", item.job_id, item.job_instance_id, item.definition_revision)
    assert graph.store.read()["requests"] == []
    assert graph.service.occurrence_store.inspect_records() == []
    await graph.aclose()

    graph, item, clock, calls, channels = runtime(tmp_path / "authorized")
    one = graph.authorize_manual("one", item.job_id, item.job_instance_id, item.definition_revision)
    two = graph.authorize_manual("two", item.job_id, item.job_instance_id, item.definition_revision)
    await graph.execute_now(one)
    await graph.execute_now(two)
    records = graph.service.occurrence_store.inspect_records()
    assert len({record.occurrence_id for record in records}) == 2
    assert {record.identity.scheduled_fire_time_ms for record in records} == {1000}
    assert {record.identity.sequence for record in records} == {1, 2}
    with pytest.raises(OccurrenceAlreadyCompletedError):
        await graph.execute_now(one)
    assert len(calls) == 2 and channels.calls == ["A", "B", "A", "B"]
    for record in records:
        outbox = graph.service.prepare_completed_delivery(record.occurrence_id)
        assert outbox.occurrence_identity == record.identity.as_dict()
    await graph.aclose()


async def test_named_system_controller_separate_from_manual_and_timer(tmp_path):
    graph, item, clock, calls, channels = runtime(tmp_path)
    manual = graph.authorize_manual("one", item.job_id, item.job_instance_id, item.definition_revision)
    system = graph.construct_internal_controller("explicit-test").request("one", item.job_id, item.job_instance_id, item.definition_revision)
    await graph.execute_now(manual)
    await graph.execute_now(system)
    identities = [record.identity for record in graph.service.occurrence_store.inspect_records()]
    assert {identity.source for identity in identities} == {"manual", "system"}
    timer = ScheduledOccurrenceIdentity("stage3", item.job_id, item.definition_revision, 1000)
    assert _occurrence_id(timer) not in {_occurrence_id(identity) for identity in identities}
    with pytest.raises(ValueError):
        graph.construct_internal_controller("Dream")
    await graph.aclose()


@pytest.mark.parametrize("field,value", [("source", "timer"), ("sequence", True), ("sequence", 0), ("trigger_version", 1), ("controller", "forged")])
def test_malformed_trigger_rejected_by_ledger_and_e2(field, value):
    identity = TriggeredOccurrenceIdentity("stage3", "job", "b" * 64, 1000, "manual", 1)
    wire = {**identity.as_dict(), field: value}
    with pytest.raises(OccurrenceCorruptError):
        CronOccurrenceStore._identity_from_wire(wire)
    with pytest.raises(DeliveryOutboxCorruptError):
        _validate_occurrence_identity(wire, namespace="stage3", error=DeliveryOutboxCorruptError)


def test_timer_hash_and_four_field_wire_are_unchanged():
    import hashlib
    import json

    identity = ScheduledOccurrenceIdentity("stage3", "job", "rev", 1000)
    assert len(identity.as_dict()) == 4
    expected = hashlib.sha256(b"nanobot-cron-occurrence-v1\0" + json.dumps(identity.as_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert _occurrence_id(identity) == expected


async def test_changed_copied_foreign_controller_allocates_nothing(tmp_path):
    from copy import copy

    graph, item, clock, calls, channels = runtime(tmp_path)
    controller = graph.construct_internal_controller("named")
    clone = copy(controller)
    with pytest.raises(PermissionError):
        clone.request("one", item.job_id, item.job_instance_id, item.definition_revision)
    controller._name = "changed"
    with pytest.raises(PermissionError):
        controller.request("one", item.job_id, item.job_instance_id, item.definition_revision)
    assert graph.store.read()["requests"] == []
    assert graph.service.occurrence_store.inspect_records() == []
    await graph.aclose()


async def test_allocated_crash_request_is_not_execution_authority_after_restart(tmp_path):
    from nanobot.cron.scheduler_store import SchedulerConflictError, SchedulerStore

    graph, item, clock, calls, channels = runtime(tmp_path)
    graph.store.allocate_request(source="manual", controller=None, request_id="crashed",
        inputs={"job_id": item.job_id, "job_instance_id": item.job_instance_id,
                "definition_revision": item.definition_revision}, clock_ms=clock.now)
    reopened = SchedulerStore(graph.store.root, namespace="stage3", runtime_identity="copied-root")
    graph.store = reopened
    capability = graph.authorize_manual("crashed", item.job_id, item.job_instance_id, item.definition_revision)
    with pytest.raises(SchedulerConflictError, match="pre-effect"):
        await graph.execute_now(capability)
    assert calls == [] and graph.service.occurrence_store.inspect_records() == []
    await graph.aclose()


@pytest.mark.parametrize("source", ["manual", "system"])
async def test_allocated_request_clock_cannot_be_reauthorized_at_new_time(tmp_path, monkeypatch, source):
    from nanobot.cron.scheduler_store import SchedulerConflictError

    graph, item, clock, calls, channels = runtime(tmp_path)
    controller = graph.construct_internal_controller("clock-test")
    issue = graph.authorize_manual if source == "manual" else controller.request
    original = issue("lost", item.job_id, item.job_instance_id, item.definition_revision)
    real = graph.service.occurrence_store.claim_triggered
    with monkeypatch.context() as crash:
        crash.setattr(graph.service.occurrence_store, "claim_triggered", lambda *a, **k: (_ for _ in ()).throw(OSError("before claim")))
        with pytest.raises(OSError):
            await graph.execute_now(original)
    allocation = graph.store.read()["requests"][0]
    clock.now = 2000
    changed = issue("lost", item.job_id, item.job_instance_id, item.definition_revision)
    with pytest.raises(SchedulerConflictError, match="inputs changed"):
        await graph.execute_now(changed)
    assert graph.store.read()["requests"] == [allocation]
    assert graph.service.occurrence_store.inspect_records() == []
    assert calls == [] and channels.calls == []
    assert graph.service.occurrence_store.claim_triggered == real
    await graph.execute_now(original)
    record = graph.service.occurrence_store.inspect_records()[0]
    assert record.identity.scheduled_fire_time_ms == 1000 and record.identity.sequence == 1
    with pytest.raises(OccurrenceAlreadyCompletedError):
        await graph.execute_now(original)
    assert len(calls) == 1 and channels.calls == ["A", "B"]
    await graph.aclose()
