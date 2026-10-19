"""Typed owning ingress operations retain the original dormant cron guards."""
from copy import copy
from dataclasses import asdict, replace

import pytest

import nanobot.cron.occurrence as occurrence
import nanobot.cron.runtime as runtime_module
import nanobot.cron.service as service
from nanobot.cron.action_store import CronActionStore
from nanobot.cron.definition_store import (
    CronDefinition,
    CronDefinitionAction,
    CronDefinitionCandidate,
    CronDefinitionStore,
    DefinitionStoreConflictError,
    DefinitionStoreValidationError,
)
from nanobot.cron.occurrence import CronOccurrenceDefinitionConflictError
from nanobot.cron.scheduler_store import SchedulerConflictError, SchedulerStore
from nanobot.cron.types import CronJob, CronSchedule
from tests.cron.stage3_fixtures import definition, runtime


def test_definition_validation_returns_the_original_detached_result():
    item = definition()
    checked = item.validated()
    assert checked == item._validated() == item
    assert checked is not item and checked.definition is not item.definition
    invalid = replace(item, definition_revision="0" * 64)
    with pytest.raises(DefinitionStoreValidationError) as original:
        invalid._validated()
    with pytest.raises(DefinitionStoreValidationError) as owned:
        invalid.validated()
    assert type(owned.value) is type(original.value)
    assert str(owned.value) == str(original.value)


@pytest.mark.parametrize("fields", [
    {"definitions": (None,)}, {"applied_action_ids": ("bad/id",)},
    {"applied_action_ids": ("same", "same")}, {"scheduling_version": 3},
])
def test_candidate_validation_retains_original_error_order(fields):
    candidate = CronDefinitionCandidate(scheduling_commits=(None,), **fields)
    with pytest.raises(DefinitionStoreValidationError) as original:
        candidate._validated()
    with pytest.raises(DefinitionStoreValidationError) as owned:
        candidate.validated()
    assert type(owned.value) is type(original.value)
    assert str(owned.value) == str(original.value)


def test_candidate_empty_retains_subclass_factory_and_validator_fault_seams(monkeypatch):
    class Derived(CronDefinitionCandidate):
        pass

    assert type(Derived.empty()) is Derived
    assert Derived.empty().to_wire() == Derived._empty().to_wire()
    expected = CronDefinitionCandidate((definition(),))
    monkeypatch.setattr(CronDefinitionCandidate, "_empty", classmethod(lambda cls: expected))
    assert Derived.empty() is expected
    failure = RuntimeError("original candidate validator")

    def fail(_self):
        raise failure

    monkeypatch.setattr(CronDefinitionCandidate, "_validated", fail)
    with pytest.raises(RuntimeError) as caught:
        expected.validated()
    assert caught.value is failure


def test_replay_identity_query_returns_no_owner_and_checkpoint_stays_bound(tmp_path):
    first = CronDefinitionStore(CronActionStore(tmp_path / "first", namespace="ingress"))
    second = CronDefinitionStore(CronActionStore(tmp_path / "second", namespace="ingress"))
    replay = first.checkpoint(first.replay())
    assert replay.matches_owner(first._owner) is True
    assert replay.matches_owner(second._owner) is False
    assert replay.matches_owner(object()) is False
    with pytest.raises(DefinitionStoreConflictError):
        second.checkpoint(replay)
    assert first.checkpoint(replay).candidate == replay.candidate


def test_destination_policy_retains_detachment_and_original_fault_seam(monkeypatch):
    item = definition()
    before = item.to_wire()
    policy = occurrence.destination_policy(item)
    assert policy == occurrence._destination_policy(item)
    policy["additionalDestinations"].clear()
    policy["originMetadata"]["changed"] = True
    assert item.to_wire() == before
    failure = RuntimeError("original policy seam")

    def fail(_item):
        raise failure

    monkeypatch.setattr(occurrence, "_destination_policy", fail)
    with pytest.raises(RuntimeError) as caught:
        occurrence.destination_policy(item)
    assert caught.value is failure


async def test_scheduler_requires_real_current_lease_and_observation_is_not_proof(tmp_path):
    graph, _, _, calls, channels = runtime(tmp_path)
    try:
        lease = graph.scheduler._lease
        assert lease.is_active is True
        assert graph.scheduler.require_lifetime_lease() is None
        graph.scheduler.release_lease()
        assert lease.is_active is False
        assert graph.scheduler._lease is None
        with pytest.raises(SchedulerConflictError, match="lifetime lease unavailable"):
            graph.scheduler.require_lifetime_lease()
        assert calls == [] and channels.calls == []
    finally:
        await graph.aclose()


def test_active_lease_observation_does_not_admit_a_replaced_root(tmp_path):
    store = SchedulerStore(tmp_path / "scheduler", namespace="ingress", runtime_identity="test")
    lease = store.acquire_lease()
    original = store.root
    retained = tmp_path / "retained"
    original.rename(retained)
    original.mkdir(mode=0o700)
    try:
        assert lease.is_active is True
        with pytest.raises(RuntimeError):
            lease.reprove()
    finally:
        original.rmdir()
        retained.rename(original)
        lease.release()
    assert lease.is_active is False


async def test_controller_owner_operation_issues_only_through_original_guard(tmp_path):
    graph, item, _, calls, channels = runtime(tmp_path)
    try:
        controller = graph.construct_internal_controller("named")
        capability = graph.request_from_controller(controller, "named", "request", item.job_id,
                                                  item.job_instance_id, item.definition_revision)
        assert capability.runtime is graph and capability.source == "system"
        assert capability.controller == "named" and capability.request_id == "request"
        assert graph.store.read()["requests"] == []
        assert graph.service.occurrence_store.inspect_records() == []
        assert calls == [] and channels.calls == []
    finally:
        await graph.aclose()


@pytest.mark.parametrize("attack", ["copy", "foreign", "name", "pid"])
async def test_controller_owner_operation_rejects_changed_bindings_before_allocation(tmp_path, monkeypatch, attack):
    graph, item, _, calls, channels = runtime(tmp_path)
    other = None
    try:
        controller = graph.construct_internal_controller("named")
        supplied, name = controller, "named"
        if attack == "copy":
            supplied = copy(controller)
        if attack == "foreign":
            foreign_path = tmp_path / "foreign"
            foreign_path.mkdir()
            other, _, _, _, _ = runtime(foreign_path)
            supplied = other.construct_internal_controller("named")
        if attack == "name":
            name = "changed"
        with monkeypatch.context() as patch:
            if attack == "pid":
                original_pid = runtime_module.os.getpid()
                patch.setattr(runtime_module.os, "getpid", lambda: original_pid + 1)
            with pytest.raises(PermissionError, match="foreign, copied or changed"):
                graph.request_from_controller(supplied, name, "request", item.job_id,
                                              item.job_instance_id, item.definition_revision)
        assert graph.store.read()["requests"] == []
        assert graph.service.occurrence_store.inspect_records() == []
        assert calls == [] and channels.calls == []
    finally:
        if other is not None:
            await other.aclose()
        await graph.aclose()


async def test_public_definition_read_preserves_pending_checkpoint_fencing(tmp_path):
    graph, item, _, _, _ = runtime(tmp_path)
    try:
        authority = graph.service.occurrence_runtime
        assert authority.committed_definition(item.job_id) == authority._committed_definition(item.job_id)
        store = graph.service.definition_store
        changed = CronDefinition.from_definition(
            {**item.definition, "enabled": False}, job_instance_id=item.job_instance_id,
        )
        store.append(action_id="pending-disable", action=CronDefinitionAction.update(
            changed, expected_job_instance_id=item.job_instance_id,
            expected_definition_revision=item.definition_revision,
        ))
        pending = store.replay()
        assert len(pending.pending_actions) == 1
        with pytest.raises(CronOccurrenceDefinitionConflictError, match="awaiting checkpoint") as original:
            authority._committed_definition(item.job_id)
        with pytest.raises(CronOccurrenceDefinitionConflictError, match="awaiting checkpoint") as owned:
            authority.committed_definition(item.job_id)
        assert str(owned.value) == str(original.value)
        store.checkpoint(pending)
        assert authority.committed_definition(item.job_id).definition["enabled"] is False
    finally:
        await graph.aclose()


def test_legacy_public_aliases_preserve_function_identity_and_nonmutation(monkeypatch):
    assert service.legacy_now_ms is service._now_ms
    assert service.legacy_compute_next_run is service._compute_next_run
    assert service.legacy_normalize_agent_turn_job is service._normalize_agent_turn_job
    assert service.legacy_now_ms.__name__ == "_now_ms"
    monkeypatch.setattr(service.time, "time", lambda: 1.234)
    assert service.legacy_now_ms() == 1234
    assert service.legacy_compute_next_run(CronSchedule(kind="every", every_ms=100), 1000) == 1100
    assert service.legacy_compute_next_run(CronSchedule(kind="at", at_ms=999), 1000) is None
    assert service.legacy_compute_next_run(CronSchedule(kind="cron", expr="invalid"), 1000) is None
    job = CronJob("job", "unchanged")
    before = asdict(job)
    assert service.legacy_normalize_agent_turn_job(job) is False
    assert asdict(job) == before
