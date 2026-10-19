from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from nanobot.cron.action_store import CronActionStore
from nanobot.cron.definition_store import (
    CronDefinitionAction,
    CronDefinitionStore,
    DefinitionStoreConflictError,
    DefinitionStoreValidationError,
)
from tests.cron.stage3_fixtures import (  # noqa: F401
    FakeClock,
    definition,
    definitions,
    offline_denial,
)


def test_atomic_receipt_generation_and_pending_non_authority(tmp_path):
    clock = FakeClock(1234)
    store = CronDefinitionStore(CronActionStore(tmp_path / "definitions", namespace="stage3"), commit_clock_ms=clock.wall_time_ms)
    item = definition()
    receipt = store.append(action_id="create", action=CronDefinitionAction.create(item))
    pending = store.replay()
    assert pending.candidate.scheduling_commits == ()
    settled = store.checkpoint(pending)
    commit = settled.candidate.scheduling_commits[0]
    assert commit["commit_clock_ms"] == 1234
    assert commit["mutation_receipt"] == asdict(receipt)
    assert commit["predecessor"] == asdict(pending.generation)
    assert commit["target"] == asdict(settled.generation)
    assert set(item.to_wire()) == set(settled.candidate.definitions[0].to_wire())
    assert "commit_clock_ms" not in item.to_wire()


def test_noop_and_a_b_a_preserve_first_commit(tmp_path):
    store, original, clock = definitions(tmp_path)
    first = deepcopy(store.replay().candidate.scheduling_commits[0])
    store.checkpoint(store.replay())
    clock.now = 5000
    changed = definition("changed")
    for name, old, new in (("to-b", original, changed), ("to-a", changed, original)):
        store.append(action_id=name, action=CronDefinitionAction.update(new,
            expected_job_instance_id=old.job_instance_id, expected_definition_revision=old.definition_revision))
        store.checkpoint(store.replay())
    commits = store.replay().candidate.scheduling_commits
    assert len(commits) == 2 and commits[0] == first
    store.append(action_id="delete", action=CronDefinitionAction.delete(original.job_id,
        expected_job_instance_id=original.job_instance_id, expected_definition_revision=original.definition_revision))
    assert store.checkpoint(store.replay()).candidate.scheduling_commits == commits


def test_lost_ack_reuses_identical_prepared_commit(tmp_path, monkeypatch):
    store, item, clock = definitions(tmp_path)
    changed = definition("changed")
    store.append(action_id="update", action=CronDefinitionAction.update(changed,
        expected_job_instance_id=item.job_instance_id, expected_definition_revision=item.definition_revision))
    pending = store.replay()
    real = store.action_store.checkpoint
    calls = []

    def lost_ack(**kwargs):
        calls.append(deepcopy(kwargs["candidate"]))
        result = real(**kwargs)
        if len(calls) == 1:
            raise OSError("lost ack")
        return result

    monkeypatch.setattr(store.action_store, "checkpoint", lost_ack)
    with pytest.raises(OSError):
        store.checkpoint(pending)
    clock.now += 10000
    store.checkpoint(pending)
    assert calls[0] == calls[1]


def test_replaced_replay_candidate_or_receipt_cannot_commit(tmp_path):
    store, item, clock = definitions(tmp_path)
    replay = store.replay()
    changed = replace(replay.candidate, definitions=(definition("forged"),))
    with pytest.raises(DefinitionStoreConflictError):
        store.checkpoint(replace(replay, candidate=changed))
    commit = deepcopy(replay.candidate.scheduling_commits[0])
    commit["commit_clock_ms"] += 1
    with pytest.raises(DefinitionStoreConflictError):
        store.checkpoint(replace(replay, candidate=replace(replay.candidate, scheduling_commits=(commit,))))
    commit["target"]["snapshot_generation"] = True
    with pytest.raises(DefinitionStoreValidationError):
        replace(replay.candidate, scheduling_commits=(commit,)).to_wire()


def test_explicit_normalization_is_labelled_and_stopped_writer_only(tmp_path):
    from nanobot.cron.occurrence_store import CronOccurrenceStore
    from nanobot.cron.scheduler_store import SchedulerConflictError, SchedulerStore

    store, item, clock = definitions(tmp_path)
    raw = store.action_store.replay()
    wire = {key: value for key, value in raw.candidate.items() if key != "scheduling_commits"}
    wire["kind"], wire["schema_version"] = "cron_definition_candidate_v1", 1
    store.action_store.checkpoint(expected_generation=raw.generation, candidate=wire)
    assert store.replay().candidate.scheduling_commits == ()
    scheduling = SchedulerStore(tmp_path / "scheduler", namespace="stage3", runtime_identity="copied")
    occurrences = CronOccurrenceStore(tmp_path / "occurrences", namespace="stage3")
    arguments = {"job_id": item.job_id, "expected_job_instance_id": item.job_instance_id,
        "expected_definition_revision": item.definition_revision, "scheduler_store": scheduling,
        "occurrence_store": occurrences}
    lease = scheduling.acquire_lease()
    with pytest.raises(SchedulerConflictError):
        store.normalize_missing_evidence(**arguments)
    lease.release()
    clock.now = 5000
    settled = store.normalize_missing_evidence(**arguments)
    commit = settled.candidate.scheduling_commits[0]
    assert commit["commit_kind"] == "normalization" and commit["mutation_receipt"] is None
    assert commit["commit_clock_ms"] == 5000
    clock.now = 10000
    assert store.normalize_missing_evidence(**arguments).candidate.scheduling_commits == (commit,)
