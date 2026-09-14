"""Stopped-writer conversion coverage for the version-2 cron action store."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import nanobot.cron.action_store as action_store
from nanobot.cron.action_store import (
    ActionStoreConversionRequiredError,
    ActionStoreCorruptError,
    ActionStoreLegacyConversionError,
    ActionStorePersistenceError,
    CronActionStore,
)

NAMESPACE = "cron-conversion:tests"


def _write_private(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    os.chmod(path, 0o600)


def _legacy_root(tmp_path: Path, *, actions: bytes | None = None, jobs: bytes | None = None) -> Path:
    root = tmp_path / "legacy-cron"
    root.mkdir(mode=0o700)
    if actions is not None:
        _write_private(root / "action.jsonl", actions)
    if jobs is not None:
        _write_private(root / "jobs.json", jobs)
    return root


def _job(*, job_id: str = "daily", name: str = "Daily", snake_case: bool = False) -> dict[str, Any]:
    schedule: dict[str, Any] = {"kind": "cron", "expr": "0 9 * * *", "tz": "Asia/Kuala_Lumpur"}
    payload: dict[str, Any] = {
        "kind": "agent_turn",
        "message": "send the daily report",
        "deliver": True,
        "channel": "telegram",
        "to": "42",
        "additionalDestinations": [{"channel": "slack", "to": "C1"}],
        "channelMeta": {"topic": "reports"},
        "sessionKey": "telegram:42",
        "executionMode": "session_bound",
        "originChannel": "telegram",
        "originChatId": "42",
        "originMetadata": {"from": "legacy"},
    }
    state: dict[str, Any] = {
        "nextRunAtMs": 100,
        "lastRunAtMs": 90,
        "lastStatus": "ok",
        "lastError": None,
        "lastRunStatus": "succeeded",
        "runHistory": [
            {
                "runAtMs": 90,
                "status": "ok",
                "durationMs": 12,
                "error": None,
                "runId": "run-1",
                "runStatus": "succeeded",
                "occurrenceId": "occurrence-1",
            }
        ],
    }
    job: dict[str, Any] = {
        "id": job_id,
        "name": name,
        "enabled": True,
        "schedule": schedule,
        "payload": payload,
        "state": state,
        "createdAtMs": 10,
        "updatedAtMs": 20,
        "deleteAfterRun": False,
    }
    if snake_case:
        schedule = {"kind": "cron", "expr": "0 9 * * *", "tz": "Asia/Kuala_Lumpur"}
        payload = {
            "kind": "agent_turn",
            "message": "send the daily report",
            "deliver": True,
            "channel": "telegram",
            "to": "42",
            "additional_destinations": [{"channel": "slack", "to": "C1"}],
            "channel_meta": {"topic": "reports"},
            "session_key": "telegram:42",
            "execution_mode": "session_bound",
            "origin_channel": "telegram",
            "origin_chat_id": "42",
            "origin_metadata": {"from": "legacy"},
        }
        state = {
            "next_run_at_ms": 100,
            "last_run_at_ms": 90,
            "last_status": "ok",
            "last_error": None,
            "last_run_status": "succeeded",
            "run_history": [
                {
                    "run_at_ms": 90,
                    "status": "ok",
                    "duration_ms": 12,
                    "error": None,
                    "run_id": "run-1",
                    "run_status": "succeeded",
                    "occurrence_id": "occurrence-1",
                }
            ],
        }
        job.update(
            {
                "schedule": schedule,
                "payload": payload,
                "state": state,
                "created_at_ms": 10,
                "updated_at_ms": 20,
                "delete_after_run": False,
            }
        )
        job.pop("createdAtMs")
        job.pop("updatedAtMs")
        job.pop("deleteAfterRun")
    return job


def _action(operation: str, params: dict[str, Any]) -> bytes:
    return json.dumps({"action": operation, "params": params}, separators=(",", ":")).encode("utf-8") + b"\n"


def _manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "conversion.json").read_text(encoding="utf-8"))


def test_conversion_preserves_legacy_content_and_maps_actions_deterministically(tmp_path: Path) -> None:
    added = _job(job_id="daily", name="Daily v1")
    updated = _job(job_id="daily", name="Daily v2")
    action_bytes = _action("add", added) + _action("add", added) + _action("update", updated) + _action(
        "del", {"job_id": "obsolete"}
    )
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job(snake_case=True)]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)

    result = CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    assert result.journal_converted is True
    assert result.snapshot_converted is True
    assert result.snapshot_status == "converted"
    assert result.action_count == 4
    assert not (root / "action.jsonl").exists()
    assert (root / "jobs.json").read_bytes() == jobs_bytes

    manifest = _manifest(root)
    assert manifest["action"]["status"] == "converted"
    assert manifest["jobs"]["status"] == "converted"
    assert result.action_backup_digest == hashlib.sha256(action_bytes).hexdigest()
    assert result.jobs_backup_digest == hashlib.sha256(jobs_bytes).hexdigest()
    backup_dir = root / "legacy-backups"
    assert backup_dir.stat().st_mode & 0o777 == 0o700
    assert (backup_dir / f"{result.action_backup_digest}.bin").read_bytes() == action_bytes
    assert (backup_dir / f"{result.jobs_backup_digest}.bin").read_bytes() == jobs_bytes
    assert (backup_dir / f"{result.action_backup_digest}.bin").stat().st_mode & 0o777 == 0o600
    assert (root / "conversion.json").stat().st_mode & 0o777 == 0o600

    replay = CronActionStore(root, namespace=NAMESPACE).replay()
    assert replay.pending_actions == ()
    candidate = replay.candidate
    assert candidate is not None
    assert candidate["runtime_state_policy"] == "snapshot_wins"
    evidence = candidate["legacy_conversion_evidence"]
    assert evidence["schema_version"] == 1
    assert evidence["conversion_identity"] == result.conversion_identity
    assert evidence["action_backup_digest"] == result.action_backup_digest
    assert evidence["jobs_backup_digest"] == result.jobs_backup_digest
    canonical_job = candidate["jobs"][0]
    assert canonical_job["schedule"]["tz"] == "Asia/Kuala_Lumpur"
    assert canonical_job["payload"]["additionalDestinations"] == [{"channel": "slack", "to": "C1"}]
    assert canonical_job["state"]["runHistory"][0]["occurrenceId"] == "occurrence-1"
    assert canonical_job["jobInstanceId"]
    assert canonical_job["definitionRevision"]

    operations = candidate["operations"]
    assert [operation["request"]["operation"] for operation in operations] == ["upsert", "upsert", "upsert", "delete"]
    assert [operation["request"]["legacy_operation"] for operation in operations] == ["add", "add", "update", "del"]
    assert operations[0]["action_id"] != operations[1]["action_id"]
    assert operations[0]["request"]["runtime_state_policy"] == "snapshot_wins"
    assert operations[0]["request"]["job"]["jobInstanceId"] == operations[1]["request"]["job"]["jobInstanceId"]
    assert operations[0]["request"]["job"]["definitionRevision"] != operations[2]["request"]["job"]["definitionRevision"]
    assert [receipt.action_id for receipt in replay.receipts] == [operation["action_id"] for operation in operations]


def test_corrupt_jobs_still_converts_actions_and_a_restored_snapshot_can_resume(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    corrupt_jobs = b"{not valid json"
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=corrupt_jobs)

    unavailable = CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    assert unavailable.journal_converted is True
    assert unavailable.snapshot_converted is False
    assert unavailable.snapshot_status == "unavailable"
    assert not (root / "action.jsonl").exists()
    assert (root / "jobs.json").read_bytes() == corrupt_jobs
    assert (root / "legacy-backups" / f"{unavailable.jobs_backup_digest}.bin").read_bytes() == corrupt_jobs
    assert CronActionStore(root, namespace=NAMESPACE).replay().candidate is None

    restored_jobs = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    _write_private(root / "jobs.json", restored_jobs)
    resumed = CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    assert resumed.snapshot_status == "converted"
    assert resumed.snapshot_converted is True
    assert CronActionStore(root, namespace=NAMESPACE).replay().candidate is not None


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("action_reappears", "reappeared"),
        ("jobs_changes", "changed"),
        ("jobs_disappears", "disappeared"),
    ],
)
def test_cached_replay_has_no_authority_after_completed_conversion_source_changes(
    tmp_path: Path, mutation: str, match: str
) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    store = CronActionStore(root, namespace=NAMESPACE)
    cached = store.replay()
    assert cached.candidate is not None

    if mutation == "action_reappears":
        _write_private(root / "action.jsonl", action_bytes)
    elif mutation == "jobs_changes":
        _write_private(root / "jobs.json", b'{"version":1,"jobs":[]}')
    else:
        (root / "jobs.json").unlink()

    with pytest.raises(ActionStoreCorruptError, match=match):
        store.replay()


@pytest.mark.parametrize("operation", ["replay", "append", "checkpoint"])
def test_completed_conversion_evidence_is_checked_by_every_public_operation(
    tmp_path: Path, operation: str
) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    store = CronActionStore(root, namespace=NAMESPACE)
    state = store.replay()
    _write_private(root / "jobs.json", b'{"version":1,"jobs":[]}')

    with pytest.raises(ActionStoreCorruptError, match="changed"):
        if operation == "replay":
            store.replay()
        elif operation == "append":
            store.append(action_id="new-action", request={"opaque": operation})
        else:
            store.checkpoint(expected_generation=state.generation, candidate={"opaque": operation})


@pytest.mark.parametrize("evidence_object", ["backup", "manifest"])
def test_completed_conversion_rechecks_retained_backup_and_manifest_evidence(
    tmp_path: Path, evidence_object: str
) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    result = CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    store = CronActionStore(root, namespace=NAMESPACE)
    assert store.replay().candidate is not None

    if evidence_object == "backup":
        assert result.action_backup_digest is not None
        _write_private(root / "legacy-backups" / f"{result.action_backup_digest}.bin", b"tampered")
        match = "backup"
    else:
        (root / "conversion.json").unlink()
        match = "manifest"

    with pytest.raises(ActionStoreCorruptError, match=match):
        store.append(action_id="new-action", request={"opaque": evidence_object})


def test_completed_conversion_accepts_a_transformed_candidate_that_carries_its_evidence(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    store = CronActionStore(root, namespace=NAMESPACE)
    current = store.replay()
    assert current.candidate is not None
    evidence = current.candidate["legacy_conversion_evidence"]

    transformed = {
        "schema_version": 9,
        "kind": "canonical_cron_candidate",
        "legacy_conversion_evidence": evidence,
        "adapter_owned_state": {"preserved": True},
    }
    store.checkpoint(expected_generation=current.generation, candidate=transformed)

    assert store.replay().candidate == transformed


def test_completed_conversion_checkpoint_rejects_a_candidate_that_drops_its_evidence(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    store = CronActionStore(root, namespace=NAMESPACE)
    current = store.replay()

    with pytest.raises(ActionStoreCorruptError, match="candidate lacks matching evidence"):
        store.checkpoint(
            expected_generation=current.generation,
            candidate={"schema_version": 9, "kind": "canonical_cron_candidate"},
        )


def test_completed_conversion_reads_the_pre_evidence_raw_candidate_compatibly(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    snapshot = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    snapshot["candidate"].pop("legacy_conversion_evidence")
    _write_private(root / "snapshot.json", json.dumps(snapshot, separators=(",", ":")).encode())

    candidate = CronActionStore(root, namespace=NAMESPACE).replay().candidate
    assert candidate is not None
    assert candidate["kind"] == "legacy_cron_snapshot_candidate"


def test_completed_conversion_rejects_candidate_evidence_that_no_longer_matches_the_manifest(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    store = CronActionStore(root, namespace=NAMESPACE)
    assert store.replay().candidate is not None

    snapshot = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    snapshot["candidate"]["legacy_conversion_evidence"]["conversion_identity"] = "0" * 64
    _write_private(root / "snapshot.json", json.dumps(snapshot, separators=(",", ":")).encode())

    with pytest.raises(ActionStoreCorruptError, match="candidate evidence"):
        store.replay()


@pytest.mark.parametrize(("target", "status"), [("action", "pending"), ("action", "removing"), ("jobs", "pending")])
@pytest.mark.parametrize("operation", ["replay", "append", "exact_retry", "checkpoint"])
def test_public_operations_fence_incomplete_conversion_states(
    tmp_path: Path, target: str, status: str, operation: str
) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    store = CronActionStore(root, namespace=NAMESPACE)
    state = store.replay()
    assert state.candidate is not None
    retry = state.candidate["operations"][0]
    manifest = _manifest(root)
    manifest[target]["status"] = status
    _write_private(root / "conversion.json", json.dumps(manifest, separators=(",", ":")).encode())

    with pytest.raises(ActionStoreCorruptError, match="incomplete"):
        if operation == "replay":
            store.replay()
        elif operation == "append":
            store.append(action_id="new-action", request={"opaque": operation})
        elif operation == "exact_retry":
            store.append(action_id=retry["action_id"], request=retry["request"])
        else:
            store.checkpoint(expected_generation=state.generation, candidate=state.candidate)


def test_completed_conversion_requires_an_untampered_completion_provenance_marker(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    store = CronActionStore(root, namespace=NAMESPACE)
    assert store.replay().candidate is not None

    marker = json.loads((root / "conversion-complete.json").read_text(encoding="utf-8"))
    marker["conversion_evidence"]["conversion_identity"] = "0" * 64
    _write_private(root / "conversion-complete.json", json.dumps(marker, separators=(",", ":")).encode())

    with pytest.raises(ActionStoreCorruptError, match="completion provenance"):
        store.replay()


@pytest.mark.parametrize(("target", "match"), [("marker", "completion provenance"), ("manifest", "incomplete")])
def test_completed_conversion_fences_safe_file_substitution_of_its_provenance(
    tmp_path: Path, target: str, match: str
) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    store = CronActionStore(root, namespace=NAMESPACE)
    assert store.replay().candidate is not None

    path = root / ("conversion-complete.json" if target == "marker" else "conversion.json")
    replacement = path.with_name(path.name + ".replacement")
    value = json.loads(path.read_text(encoding="utf-8"))
    if target == "marker":
        value["store_namespace"] = "other-store"
    else:
        value["jobs"]["status"] = "pending"
    _write_private(replacement, json.dumps(value, separators=(",", ":")).encode())
    os.replace(replacement, path)

    with pytest.raises(ActionStoreCorruptError, match=match):
        store.replay()


@pytest.mark.parametrize("mode", ["absent", "unavailable"])
def test_marker_and_backup_evidence_fence_deleted_manifest_and_recreated_legacy_action(
    tmp_path: Path, mode: str
) -> None:
    action_bytes = _action("add", _job())
    if mode == "absent":
        root = _legacy_root(tmp_path)
    else:
        root = _legacy_root(tmp_path, actions=action_bytes, jobs=b"{broken jobs")
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    store = CronActionStore(root, namespace=NAMESPACE)
    assert store.replay().candidate is None
    (root / "conversion.json").unlink()

    with pytest.raises(ActionStoreCorruptError, match="provenance is missing its manifest"):
        store.replay()

    _write_private(root / "action.jsonl", action_bytes)

    with pytest.raises(ActionStoreCorruptError, match="reappeared"):
        store.replay()
    with pytest.raises(ActionStoreCorruptError, match="reappeared"):
        CronActionStore(root, namespace=NAMESPACE)

    (root / "conversion-complete.json").unlink()
    if mode == "unavailable":
        with pytest.raises(ActionStoreCorruptError, match="provenance|reappeared"):
            store.append(action_id="new-action", request={"opaque": mode})


def test_converter_resumes_a_crash_after_terminal_manifest_before_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    store = CronActionStore(root, namespace=NAMESPACE, _allow_legacy_source=True)

    def fail_completion_marker(*args: object, **kwargs: object) -> None:
        raise ActionStorePersistenceError("injected completion marker failure")

    monkeypatch.setattr(store, "_write_conversion_completion", fail_completion_marker)
    with pytest.raises(ActionStorePersistenceError, match="completion marker"):
        store.convert_legacy_sources()
    assert _manifest(root)["jobs"]["status"] == "converted"
    assert not (root / "conversion-complete.json").exists()
    with pytest.raises(ActionStoreCorruptError, match="completion provenance"):
        CronActionStore(root, namespace=NAMESPACE).replay()

    resumed = CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    assert resumed.snapshot_status == "converted"
    assert (root / "conversion-complete.json").exists()
    assert CronActionStore(root, namespace=NAMESPACE).replay().candidate is not None


@pytest.mark.parametrize(
    "action_bytes",
    [
        b'{"action":"unknown","params":{}}\n',
        b'{"action":"del","params":{"job_id":"x","extra":true}}\n',
        b'{"action":"add","params":{}}',
        b'{"action":"add","params":{}}\n\n',
    ],
)
def test_malformed_or_unknown_actions_are_rejected_without_mutating_source(
    tmp_path: Path, action_bytes: bytes
) -> None:
    root = _legacy_root(tmp_path, actions=action_bytes)

    with pytest.raises(ActionStoreLegacyConversionError):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    assert (root / "action.jsonl").read_bytes() == action_bytes
    assert not (root / "conversion.json").exists()
    assert not (root / "legacy-backups").exists()


def test_malformed_jobs_entries_fail_closed_without_dropping_them(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job(), "not-a-job"]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)

    result = CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    assert result.snapshot_status == "unavailable"
    assert (root / "jobs.json").read_bytes() == jobs_bytes
    assert _manifest(root)["jobs"]["status"] == "unavailable"


def test_repeated_conversion_is_idempotent_and_fences_legacy_writer_reappearance(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)

    first = CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    replay = CronActionStore(root, namespace=NAMESPACE).replay()
    second = CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    assert second == first
    assert CronActionStore(root, namespace=NAMESPACE).replay().generation == replay.generation

    _write_private(root / "action.jsonl", action_bytes)
    with pytest.raises(ActionStoreCorruptError, match="reappeared"):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    with pytest.raises(ActionStoreCorruptError, match="reappeared"):
        CronActionStore(root, namespace=NAMESPACE)


def test_converted_jobs_digest_change_and_journal_version_downgrade_are_fenced(tmp_path: Path) -> None:
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, jobs=jobs_bytes)
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    _write_private(root / "jobs.json", json.dumps({"version": 1, "jobs": []}).encode())
    with pytest.raises(ActionStoreCorruptError, match="changed"):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    journal = root / "journal.jsonl"
    header = json.loads(journal.read_text(encoding="utf-8"))
    header["schema_version"] = 1
    _write_private(journal, json.dumps(header).encode() + b"\n")
    with pytest.raises(ActionStoreConversionRequiredError):
        CronActionStore(root, namespace=NAMESPACE)


def test_tampered_conversion_backup_is_fenced(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    root = _legacy_root(tmp_path, actions=action_bytes)
    result = CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    assert result.action_backup_digest is not None
    _write_private(root / "legacy-backups" / f"{result.action_backup_digest}.bin", b"tampered")

    with pytest.raises(ActionStoreCorruptError, match="backup"):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)


def test_resume_rederives_operations_from_backup_and_rejects_tampered_manifest(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    manifest = _manifest(root)
    manifest["action"]["operations"] = []
    _write_private(root / "conversion.json", json.dumps(manifest, separators=(",", ":")).encode())

    with pytest.raises(ActionStoreCorruptError, match="operations"):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)


def test_pending_to_removing_tamper_cannot_retire_source_without_receipts(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    root = _legacy_root(tmp_path, actions=action_bytes)
    store = CronActionStore(root, namespace=NAMESPACE, _allow_legacy_source=True)
    with store._held_lock() as held:
        source_digest = store._legacy_digest(action_bytes)
        identity = store._conversion_identity(source_digest, None)
        operations = store._legacy_operations(
            store._parse_legacy_actions(action_bytes),
            conversion_identity=identity,
            source_digest=source_digest,
        )
        manifest = {
            "schema_version": 1,
            "kind": "legacy_cron_conversion",
            "store_namespace": NAMESPACE,
            "conversion_identity": identity,
            "identity_action_digest": source_digest,
            "identity_jobs_digest": None,
            "action": {
                "status": "pending",
                "source_digest": source_digest,
                "backup_digest": store._backup_legacy_bytes(held, action_bytes),
                "operations": operations,
            },
            "jobs": {"status": "absent", "source_digest": None, "backup_digest": None},
        }
        # This simulates a tamper between the durable pending manifest and
        # action append. The manifest must not authorize source retirement.
        manifest["action"]["status"] = "removing"
        store._write_conversion_manifest(held, manifest)

    with pytest.raises(ActionStoreCorruptError, match="receipt"):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    assert (root / "action.jsonl").read_bytes() == action_bytes


def test_conflicting_reconstructed_receipt_cannot_retire_source(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    root = _legacy_root(tmp_path, actions=action_bytes)
    store = CronActionStore(root, namespace=NAMESPACE, _allow_legacy_source=True)
    with store._held_lock() as held:
        source_digest = store._legacy_digest(action_bytes)
        identity = store._conversion_identity(source_digest, None)
        operations = store._legacy_operations(
            store._parse_legacy_actions(action_bytes),
            conversion_identity=identity,
            source_digest=source_digest,
        )
        wrong_request = {"kind": "tampered"}
        receipt = action_store.ActionReceipt(
            NAMESPACE,
            operations[0]["action_id"],
            action_store._digest_request(wrong_request),
            1,
            store._read_state(held).journal.generation,
        )
        store._append_journal(held, action_store.ActionRequest(receipt, wrong_request))
        manifest = {
            "schema_version": 1,
            "kind": "legacy_cron_conversion",
            "store_namespace": NAMESPACE,
            "conversion_identity": identity,
            "identity_action_digest": source_digest,
            "identity_jobs_digest": None,
            "action": {
                "status": "removing",
                "source_digest": source_digest,
                "backup_digest": store._backup_legacy_bytes(held, action_bytes),
                "operations": operations,
            },
            "jobs": {"status": "absent", "source_digest": None, "backup_digest": None},
        }
        store._write_conversion_manifest(held, manifest)

    with pytest.raises(ActionStoreCorruptError, match="receipt conflicts"):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    assert (root / "action.jsonl").read_bytes() == action_bytes


def test_conversion_identity_tamper_is_rejected_before_manifest_reconstruction(tmp_path: Path) -> None:
    action_bytes = _action("add", _job())
    root = _legacy_root(tmp_path, actions=action_bytes)
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    manifest = _manifest(root)
    manifest["conversion_identity"] = "0" * 64
    _write_private(root / "conversion.json", json.dumps(manifest, separators=(",", ":")).encode())

    with pytest.raises(ActionStoreCorruptError, match="conversion identity"):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)


def test_backup_directory_substitution_is_rejected_before_source_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    action_bytes = _action("add", _job())
    root = _legacy_root(tmp_path, actions=action_bytes)
    original_fsync = os.fsync
    switched = False

    def replace_backup_dir(fd: int) -> None:
        nonlocal switched
        original_fsync(fd)
        backup_dir = root / "legacy-backups"
        if not switched and backup_dir.exists() and os.fstat(fd).st_ino == backup_dir.stat().st_ino:
            backup_dir.rename(root / "legacy-backups-substituted")
            backup_dir.mkdir(mode=0o700)
            switched = True

    monkeypatch.setattr(action_store.os, "fsync", replace_backup_dir)
    with pytest.raises(ActionStoreCorruptError, match="backup directory binding"):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    assert switched
    assert (root / "action.jsonl").read_bytes() == action_bytes


def test_backup_file_substitution_during_fsync_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    action_bytes = _action("add", _job())
    root = _legacy_root(tmp_path, actions=action_bytes)
    original_fsync = os.fsync
    switched = False

    def replace_backup_file(fd: int) -> None:
        nonlocal switched
        original_fsync(fd)
        backup_dir = root / "legacy-backups"
        if switched or not backup_dir.exists() or os.fstat(fd).st_ino != backup_dir.stat().st_ino:
            return
        for backup in backup_dir.glob("*.bin"):
            replacement = backup.with_name(backup.name + ".replacement")
            _write_private(replacement, backup.read_bytes())
            os.replace(replacement, backup)
            switched = True
            return

    monkeypatch.setattr(action_store.os, "fsync", replace_backup_file)
    with pytest.raises(ActionStoreCorruptError, match="private|binding"):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    assert switched
    assert (root / "action.jsonl").read_bytes() == action_bytes


def test_partial_backup_write_leaves_only_an_ignored_temp_and_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    action_bytes = _action("add", _job())
    root = _legacy_root(tmp_path, actions=action_bytes)
    store = CronActionStore(root, namespace=NAMESPACE, _allow_legacy_source=True)
    original_write_all = store._write_all
    failed = False

    def partial_backup_write(fd: int, data: bytes) -> None:
        nonlocal failed
        if not failed and data == action_bytes:
            failed = True
            assert os.write(fd, data[:5]) == 5
            raise OSError("injected partial backup write")
        original_write_all(fd, data)

    monkeypatch.setattr(store, "_write_all", partial_backup_write)
    with pytest.raises(ActionStorePersistenceError):
        store.convert_legacy_sources()
    monkeypatch.setattr(store, "_write_all", original_write_all)

    assert (root / "action.jsonl").read_bytes() == action_bytes
    assert list((root / "legacy-backups").glob(".legacy-backup.*.tmp"))
    result = CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    assert result.journal_converted is True
    assert (root / "legacy-backups" / f"{result.action_backup_digest}.bin").read_bytes() == action_bytes


@pytest.mark.parametrize("history", [{}, False, 0, ""])
def test_conversion_rejects_falsey_or_nonlist_run_history_without_retiring_action(
    tmp_path: Path, history: object
) -> None:
    job = _job()
    job["state"]["runHistory"] = history
    action_bytes = _action("add", job)
    root = _legacy_root(tmp_path, actions=action_bytes)

    with pytest.raises(ActionStoreLegacyConversionError, match="run history"):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    assert (root / "action.jsonl").read_bytes() == action_bytes


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", False),
        ("name", 0),
        ("name", ""),
        ("enabled", "true"),
        ("enabled", 1),
        ("schedule", {}),
        ("schedule", False),
        ("schedule", 0),
        ("schedule", "cron"),
    ],
)
def test_conversion_rejects_invalid_job_scalars_without_retiring_action(
    tmp_path: Path, field: str, value: object
) -> None:
    job = _job()
    job[field] = value
    action_bytes = _action("add", job)
    root = _legacy_root(tmp_path, actions=action_bytes)

    with pytest.raises(ActionStoreLegacyConversionError):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    assert (root / "action.jsonl").read_bytes() == action_bytes


@pytest.mark.parametrize(
    ("mutate", "value"),
    [
        ("schedule_kind", "unknown"),
        ("last_status", False),
        ("last_status", 0),
        ("last_status", "unknown"),
        ("record_status", False),
        ("record_status", "unknown"),
        ("destination", {"channel": "telegram"}),
        ("destination", {"channel": "", "to": "42"}),
    ],
)
def test_conversion_rejects_invalid_schedule_status_and_destination(
    tmp_path: Path, mutate: str, value: object
) -> None:
    job = _job()
    if mutate == "schedule_kind":
        job["schedule"]["kind"] = value
    elif mutate == "last_status":
        job["state"]["lastStatus"] = value
    elif mutate == "record_status":
        job["state"]["runHistory"][0]["status"] = value
    else:
        job["payload"]["additionalDestinations"] = [value]
    action_bytes = _action("add", job)
    root = _legacy_root(tmp_path, actions=action_bytes)

    with pytest.raises(ActionStoreLegacyConversionError):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)

    assert (root / "action.jsonl").read_bytes() == action_bytes


@pytest.mark.parametrize("fail_at", range(1, 56))
def test_conversion_fsync_interruptions_leave_legacy_bytes_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    original_fsync = os.fsync
    calls = 0

    def fail_once(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("injected fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(action_store.os, "fsync", fail_once)
    with pytest.raises(ActionStorePersistenceError):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    monkeypatch.setattr(action_store.os, "fsync", original_fsync)

    result = CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    assert result.snapshot_status == "converted"
    assert result.action_backup_digest is not None
    assert (root / "legacy-backups" / f"{result.action_backup_digest}.bin").read_bytes() == action_bytes
    assert result.jobs_backup_digest is not None
    assert (root / "legacy-backups" / f"{result.jobs_backup_digest}.bin").read_bytes() == jobs_bytes


@pytest.mark.parametrize("fail_at", range(1, 10))
def test_conversion_rename_interruptions_are_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    action_bytes = _action("add", _job())
    jobs_bytes = json.dumps({"version": 1, "jobs": [_job()]}, separators=(",", ":")).encode()
    root = _legacy_root(tmp_path, actions=action_bytes, jobs=jobs_bytes)
    store = CronActionStore(root, namespace=NAMESPACE, _allow_legacy_source=True)
    original_rename = os.rename
    calls = 0

    def fail_after_rename(*args: object, **kwargs: object) -> None:
        nonlocal calls
        original_rename(*args, **kwargs)  # type: ignore[arg-type]
        calls += 1
        if calls == fail_at:
            raise OSError("injected post-rename failure")

    monkeypatch.setattr(action_store.os, "rename", fail_after_rename)
    with pytest.raises(ActionStorePersistenceError):
        store.convert_legacy_sources()
    monkeypatch.setattr(action_store.os, "rename", original_rename)

    result = CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    assert result.snapshot_status == "converted"
    assert (root / "legacy-backups" / f"{result.action_backup_digest}.bin").read_bytes() == action_bytes
