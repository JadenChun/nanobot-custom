"""C2 typed cron-definition adapter coverage over the durable action store."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

from nanobot.cron.action_store import CronActionStore
from nanobot.cron.definition_store import (
    CronDefinition,
    CronDefinitionAction,
    CronDefinitionCandidate,
    CronDefinitionStore,
    DefinitionStoreConflictError,
    DefinitionStoreValidationError,
)

NAMESPACE = "cron:definition-store-tests"


@pytest.mark.parametrize("fields,message", [
    ({"definitions": (None,)}, "cron definition action requires a CronDefinition"),
    ({"applied_action_ids": ("invalid/id",)}, "cron definition candidate action id is invalid"),
    ({"applied_action_ids": ("duplicate", "duplicate")}, "cron definition candidate contains duplicate action ids"),
    ({"legacy_conversion_evidence": []}, "legacy conversion evidence must be a JSON object"),
    ({"scheduling_version": 3}, "invalid scheduling version"),
])
def test_candidate_early_field_errors_precede_invalid_commits(fields, message):
    candidate = CronDefinitionCandidate(scheduling_commits=(None,), **fields)
    with pytest.raises(DefinitionStoreValidationError, match="^" + message + "$"):
        candidate._validated()


def test_wire_duplicate_action_ids_precede_invalid_commits():
    wire = {"schema_version": 2, "kind": "cron_definition_candidate_v2",
            "definitions": [], "applied_action_ids": ["duplicate", "duplicate"],
            "scheduling_commits": [None]}
    with pytest.raises(DefinitionStoreValidationError, match="^cron definition candidate contains duplicate action ids$"):
        CronDefinitionCandidate._from_wire(wire)


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _definition_data(job_id: str, *, name: str) -> dict[str, object]:
    return {
        "id": job_id,
        "name": name,
        "enabled": True,
        "schedule": {
            "kind": "every",
            "atMs": None,
            "everyMs": 60_000,
            "expr": None,
            "tz": None,
        },
        "payload": {
            "kind": "agent_turn",
            "message": "send report",
            "deliver": False,
            "channel": None,
            "to": None,
            "additionalDestinations": [],
            "channelMeta": {},
            "sessionKey": None,
            "executionMode": "session_bound",
            "originChannel": None,
            "originChatId": None,
            "originMetadata": {},
        },
        "deleteAfterRun": False,
    }


def _definition(job_id: str, *, name: str) -> CronDefinition:
    return CronDefinition.from_definition(
        _definition_data(job_id, name=name),
        job_instance_id=_digest(f"instance:{job_id}"),
    )


@pytest.mark.parametrize("record_type", [CronDefinition, CronDefinitionAction, CronDefinitionCandidate])
def test_owned_wire_decoder_matches_validated_parser_and_detached_output(record_type):
    definition = _definition("decode", name="owner operation")
    expected = {
        CronDefinition: definition,
        CronDefinitionAction: CronDefinitionAction.create(definition),
        CronDefinitionCandidate: CronDefinitionCandidate((definition,)),
    }[record_type]
    wire = expected.to_wire()

    decoded = record_type.decode_wire(wire)

    assert decoded == record_type._from_wire(wire) == expected
    assert decoded is not expected and decoded.to_wire() == wire
    wire.clear()
    assert decoded.to_wire() == expected.to_wire()


@pytest.mark.parametrize("record_type", [CronDefinition, CronDefinitionAction, CronDefinitionCandidate])
@pytest.mark.parametrize("wire", [None, [], {}, {7: "non-string-key"}])
def test_owned_wire_decoder_preserves_invalid_wire_error_type_and_text(record_type, wire):
    options = {"label": "captured definition"} if record_type is CronDefinition else {}
    with pytest.raises(DefinitionStoreValidationError) as original:
        record_type._from_wire(wire, **options)

    with pytest.raises(DefinitionStoreValidationError) as owned:
        record_type.decode_wire(wire, **options)

    assert type(owned.value) is type(original.value)
    assert str(owned.value) == str(original.value)


@pytest.mark.parametrize("record_type", [CronDefinition, CronDefinitionAction, CronDefinitionCandidate])
def test_owned_wire_decoder_preserves_subclass_private_seam_and_exception_instance(record_type, monkeypatch):
    class Derived(record_type):
        pass

    wire = object()
    calls = []
    expected = record_type._from_wire({
        CronDefinition: _definition("seam", name="private parser").to_wire(),
        CronDefinitionAction: CronDefinitionAction.create(_definition("seam", name="private parser")).to_wire(),
        CronDefinitionCandidate: CronDefinitionCandidate().to_wire(),
    }[record_type])
    options = {"label": "checked seam"} if record_type is CronDefinition else {}

    def decode(owner, value, **kwargs):
        calls.append((owner, value, kwargs))
        return expected

    monkeypatch.setattr(record_type, "_from_wire", classmethod(decode))
    assert Derived.decode_wire(wire, **options) is expected
    assert calls == [(Derived, wire, options)]
    failure = RuntimeError("private parser failure")

    def fail(owner, value, **kwargs):
        calls.append((owner, value, kwargs))
        raise failure

    monkeypatch.setattr(record_type, "_from_wire", classmethod(fail))
    with pytest.raises(RuntimeError) as error:
        Derived.decode_wire(wire, **options)
    assert error.value is failure
    assert calls == [(Derived, wire, options), (Derived, wire, options)]


def _store(tmp_path: Path) -> CronDefinitionStore:
    root = tmp_path / "cron-actions"
    root.mkdir(mode=0o700, parents=True)
    return CronDefinitionStore(CronActionStore(root, namespace=NAMESPACE))


def _legacy_job(job_id: str, *, name: str) -> dict[str, object]:
    job = _definition_data(job_id, name=name)
    job.update(
        {
            "state": {
                "nextRunAtMs": 100,
                "lastRunAtMs": 90,
                "lastStatus": "ok",
                "lastError": None,
                "lastRunStatus": "succeeded",
                "runHistory": [
                    {
                        "runAtMs": 90,
                        "status": "ok",
                        "durationMs": 4,
                        "error": None,
                        "runId": "run-1",
                        "runStatus": "succeeded",
                        "occurrenceId": "occurrence-1",
                    }
                ],
            },
            "createdAtMs": 1,
            "updatedAtMs": 2,
            "jobInstanceId": None,
            "definitionRevision": None,
        }
    )
    return job


def _write_private(path: Path, value: bytes) -> None:
    path.write_bytes(value)
    os.chmod(path, 0o600)


def test_definition_action_candidate_contains_no_runtime_state_and_keeps_stable_identity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    created = _definition("daily", name="Daily v1")
    store.append(action_id="create-daily", action=CronDefinitionAction.create(created))

    replay = store.replay()

    assert replay.candidate.definitions == (created,)
    assert replay.candidate.applied_action_ids == ("create-daily",)
    wire = replay.candidate.to_wire()
    assert "state" not in json.dumps(wire, sort_keys=True)
    assert wire["definitions"][0]["jobInstanceId"] == created.job_instance_id
    assert wire["definitions"][0]["definitionRevision"] == created.definition_revision

    settled = store.checkpoint(replay)
    assert settled.pending_actions == ()
    assert replay.candidate.scheduling_commits == ()
    assert settled.candidate.definitions == replay.candidate.definitions
    assert settled.candidate.applied_action_ids == replay.candidate.applied_action_ids
    assert settled.candidate.legacy_conversion_evidence == replay.candidate.legacy_conversion_evidence
    assert len(settled.candidate.scheduling_commits) == 1
    assert settled.candidate.scheduling_commits[0]["schedule"] == created.definition["schedule"]


def test_update_and_delete_require_exact_instance_and_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original = _definition("daily", name="Daily v1")
    store.append(action_id="create-daily", action=CronDefinitionAction.create(original))
    store.checkpoint(store.replay())

    changed = _definition("daily", name="Daily v2")
    store.append(
        action_id="bad-update",
        action=CronDefinitionAction.update(
            changed,
            expected_job_instance_id=original.job_instance_id,
            expected_definition_revision=_digest("wrong-revision"),
        ),
    )
    with pytest.raises(DefinitionStoreConflictError, match="revision"):
        store.replay()

    delete_store = _store(tmp_path / "delete")
    delete_store.append(action_id="create-daily", action=CronDefinitionAction.create(original))
    delete_store.checkpoint(delete_store.replay())
    delete_store.append(
        action_id="bad-delete",
        action=CronDefinitionAction.delete(
            "daily",
            expected_job_instance_id=_digest("wrong-instance"),
            expected_definition_revision=original.definition_revision,
        ),
    )
    with pytest.raises(DefinitionStoreConflictError, match="instance"):
        delete_store.replay()


def test_exact_update_and_delete_are_checkpointed_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original = _definition("daily", name="Daily v1")
    changed = _definition("daily", name="Daily v2")
    store.append(action_id="create-daily", action=CronDefinitionAction.create(original))
    store.checkpoint(store.replay())
    store.append(
        action_id="update-daily",
        action=CronDefinitionAction.update(
            changed,
            expected_job_instance_id=original.job_instance_id,
            expected_definition_revision=original.definition_revision,
        ),
    )

    updated = store.checkpoint(store.replay())
    assert updated.candidate.definitions == (changed,)
    assert updated.candidate.applied_action_ids == ("create-daily", "update-daily")

    store.append(
        action_id="delete-daily",
        action=CronDefinitionAction.delete(
            "daily",
            expected_job_instance_id=changed.job_instance_id,
            expected_definition_revision=changed.definition_revision,
        ),
    )
    deleted = store.checkpoint(store.replay())
    assert deleted.candidate.definitions == ()
    assert deleted.candidate.applied_action_ids == (
        "create-daily",
        "update-daily",
        "delete-daily",
    )
    assert store.replay() == deleted


def test_definition_actions_reject_runtime_fields(tmp_path: Path) -> None:
    data = _definition_data("daily", name="Daily")
    data["state"] = {"runHistory": ["must not enter C2"]}

    with pytest.raises(DefinitionStoreValidationError, match="unknown fields"):
        CronDefinition.from_definition(data, job_instance_id=_digest("instance:daily"))


def test_replay_rejects_a_bypassed_typed_wire_action_with_runtime_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    request = CronDefinitionAction.create(_definition("daily", name="Daily")).to_wire()
    request["definition"]["state"] = {"runHistory": []}
    store.action_store.append(action_id="malformed-create", request=request)

    with pytest.raises(DefinitionStoreValidationError, match="unknown fields"):
        store.replay()


def test_converted_embedded_operations_are_materialized_once_without_runtime_state(tmp_path: Path) -> None:
    root = tmp_path / "legacy-cron"
    root.mkdir(mode=0o700)
    initial = _legacy_job("daily", name="Daily snapshot")
    obsolete = _legacy_job("obsolete", name="Obsolete")
    added = _legacy_job("daily", name="Daily v1")
    updated = _legacy_job("daily", name="Daily v2")
    actions = b"".join(
        json.dumps(action, separators=(",", ":")).encode("utf-8") + b"\n"
        for action in (
            {"action": "add", "params": added},
            {"action": "update", "params": updated},
            {"action": "del", "params": {"job_id": "obsolete"}},
        )
    )
    _write_private(root / "action.jsonl", actions)
    _write_private(root / "jobs.json", json.dumps({"version": 1, "jobs": [initial, obsolete]}).encode("utf-8"))
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    store = CronDefinitionStore(CronActionStore(root, namespace=NAMESPACE))
    raw = store.action_store.replay()
    assert raw.candidate is not None
    original_evidence = raw.candidate["legacy_conversion_evidence"]
    first = store.replay()
    second = store.replay()

    assert first == second
    assert [definition.definition["name"] for definition in first.candidate.definitions] == ["Daily v2"]
    assert first.candidate.legacy_conversion_evidence == original_evidence
    assert first.candidate.applied_action_ids == tuple(
        operation["action_id"] for operation in raw.candidate["operations"]
    )
    assert "state" not in json.dumps(first.candidate.to_wire(), sort_keys=True)

    settled = store.checkpoint(first)
    assert settled.pending_actions == ()
    assert settled.candidate.definitions == first.candidate.definitions
    assert settled.candidate.applied_action_ids == first.candidate.applied_action_ids
    assert settled.candidate.legacy_conversion_evidence == first.candidate.legacy_conversion_evidence
    assert first.candidate.scheduling_commits == settled.candidate.scheduling_commits == ()
    assert store.action_store.replay().candidate["legacy_conversion_evidence"] == original_evidence
