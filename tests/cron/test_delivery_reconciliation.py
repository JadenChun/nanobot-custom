"""Focused tests for the offline cron delivery reconciliation plane."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

import nanobot.cron.delivery_reconciliation as reconciliation_module
from nanobot.channels.delivery_contract import ChannelTextPartReceipt, ChannelTextReceipt
from nanobot.cron.action_store import CronActionStore
from nanobot.cron.completion import CronCompletionProjector
from nanobot.cron.definition_store import CronDefinitionStore
from nanobot.cron.delivery_reconciliation import (
    ReconciliationConflictError,
    ReconciliationUnsafeError,
    apply_plan,
    create_plan,
    import_evidence,
    inspect_reconciliation,
    show_plan,
)
from nanobot.cron.delivery_store import CronDeliveryOutboxStore, DeliverySendBusyError
from nanobot.cron.occurrence_store import CronOccurrenceStore
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule

NAMESPACE = "cron:offline-reconciliation-tests"


def test_artifact_directory_stat_failure_closes_open_descriptor(tmp_path: Path, monkeypatch) -> None:
    directory = tmp_path / "private-artifacts"
    directory.mkdir(mode=0o700)
    opened: list[int] = []
    real_open = os.open

    def track_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def fail_stat(*args, **kwargs):
        raise OSError("injected post-open stat failure")

    with monkeypatch.context() as patch:
        patch.setattr(reconciliation_module.os, "open", track_open)
        patch.setattr(reconciliation_module.os, "stat", fail_stat)
        with pytest.raises(ReconciliationUnsafeError):
            reconciliation_module._ArtifactStore._open_dir(directory)
    assert len(opened) == 1
    try:
        os.fstat(opened[0])
    except OSError:
        closed = True
    else:
        closed = False
        os.close(opened[0])  # Keep the red regression run leak-free too.
    assert closed, "post-open failure leaked the artifact directory descriptor"


def _claimed(tmp_path: Path):
    cron_root = tmp_path / "cron"
    definitions = CronDefinitionStore(CronActionStore(cron_root, namespace=NAMESPACE))
    occurrences = CronOccurrenceStore(cron_root / "occurrences", namespace=NAMESPACE)
    service = CronService(
        cron_root / "jobs.json",
        definition_store=definitions,
        occurrence_store=occurrences,
        completion_projector=CronCompletionProjector(occurrences, cron_root / "runs"),
    )
    job = service.add_job(
        "job", CronSchedule(kind="every", every_ms=60_000), "prompt",
        deliver=True, channel="telegram", to="raw-target-secret",
    )
    occurrence = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )

    async def complete():
        with occurrences.acquire_execution_fence(occurrence) as fence:
            occurrences.begin_execution(occurrence, fence)
            return service.complete_occurrence(
                occurrence,
                {
                    "status": "ok",
                    "delivery": {
                        "version": 1,
                        "deliver": True,
                        "messages": [{"content": "secret body", "media": []}],
                    },
                },
                fence=fence,
            )

    asyncio.run(complete())
    outbox = CronDeliveryOutboxStore(cron_root / "outbox", namespace=NAMESPACE)
    prepared = outbox.prepare_completed(occurrences, occurrence.occurrence_id)
    destination = prepared.destinations[0]
    claim = outbox.claim_destination(
        prepared.outbox_id, destination.destination_id, prepared.record_revision, owner="sender"
    )
    current = outbox.get(prepared.outbox_id)
    return outbox, current, claim


def _mark_sending(outbox: CronDeliveryOutboxStore, claim) -> None:
    async def run() -> None:
        with outbox.acquire_send_fence(claim) as fence:
            outbox.mark_sending(claim, fence)

    asyncio.run(run())


def test_absent_existing_only_inspection_is_zero_write(tmp_path: Path) -> None:
    outbox = tmp_path / "missing" / "outbox"
    artifacts = tmp_path / "missing" / "artifacts"
    before = set(tmp_path.rglob("*"))
    result = inspect_reconciliation(outbox, namespace=NAMESPACE, artifacts_root=artifacts)
    assert result.clean is True
    assert result.entries == ()
    assert set(tmp_path.rglob("*")) == before


def test_inspection_is_redacted_and_existing_only(tmp_path: Path) -> None:
    outbox, current, _claim = _claimed(tmp_path)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    result = inspect_reconciliation(
        outbox.root, namespace=NAMESPACE, artifacts_root=tmp_path / "artifacts"
    )
    assert result.clean is False
    assert result.entries[0].status == "claimed"
    assert result.entries[0].target_redacted is True
    assert "alpha" not in repr(result)
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before
    assert result.entries[0].record_revision == current.record_revision


def test_claimed_plan_requires_confirmation_and_reapplies_idempotently(tmp_path: Path) -> None:
    outbox, current, _claim = _claimed(tmp_path)
    destination = current.destinations[0]
    artifacts = tmp_path / "artifacts"
    evidence = import_evidence(
        outbox.root,
        namespace=NAMESPACE,
        artifacts_root=artifacts,
        outbox_id=current.outbox_id,
        destination_id=destination.destination_id,
        outcome="confirmed_not_sent",
    )
    plan_artifact = create_plan(
        outbox.root,
        namespace=NAMESPACE,
        artifacts_root=artifacts,
        evidence_ref=evidence.reference,
    )
    assert plan_artifact.semantic_id is not None
    shown = show_plan(
        outbox.root,
        namespace=NAMESPACE,
        artifacts_root=artifacts,
        plan_id=plan_artifact.semantic_id,
    )
    assert shown["outcome"] == "confirmed_not_sent"
    assert "secret body" not in json.dumps(shown)

    before = outbox.get(current.outbox_id)
    with pytest.raises(ReconciliationConflictError):
        apply_plan(
            outbox.root,
            namespace=NAMESPACE,
            artifacts_root=artifacts,
            plan_id=plan_artifact.semantic_id,
            confirm_plan="0" * 64,
            confirmation="CONFIRM-NOT-SENT",
        )
    assert outbox.get(current.outbox_id) == before
    assert not list((artifacts / "applications").iterdir())

    first = apply_plan(
        outbox.root,
        namespace=NAMESPACE,
        artifacts_root=artifacts,
        plan_id=plan_artifact.semantic_id,
        confirm_plan=plan_artifact.semantic_id,
        confirmation="CONFIRM-NOT-SENT",
    )
    revision = outbox.get(current.outbox_id).record_revision
    second = apply_plan(
        outbox.root,
        namespace=NAMESPACE,
        artifacts_root=artifacts,
        plan_id=plan_artifact.semantic_id,
        confirm_plan=plan_artifact.semantic_id,
        confirmation="CONFIRM-NOT-SENT",
    )
    assert first["application_ref"] == second["application_ref"]
    assert outbox.get(current.outbox_id).record_revision == revision


def test_unsafe_artifact_binding_and_invalid_outcome_fail_closed(tmp_path: Path) -> None:
    outbox, current, _claim = _claimed(tmp_path)
    destination = current.destinations[0]
    artifacts = tmp_path / "artifacts"
    with pytest.raises(ReconciliationConflictError):
        import_evidence(
            outbox.root,
            namespace=NAMESPACE,
            artifacts_root=artifacts,
            outbox_id=current.outbox_id,
            destination_id=destination.destination_id,
            outcome="abandon",
        )
    artifacts.mkdir(mode=0o700)
    (artifacts / "evidence").symlink_to(tmp_path)
    with pytest.raises(ReconciliationUnsafeError):
        import_evidence(
            outbox.root,
            namespace=NAMESPACE,
            artifacts_root=artifacts,
            outbox_id=current.outbox_id,
            destination_id=destination.destination_id,
            outcome="confirmed_not_sent",
        )


def test_existing_inspection_rejects_temporary_residue(tmp_path: Path) -> None:
    outbox, _current, _claim = _claimed(tmp_path)
    residue = outbox.root / f".{('a' * 64)}.{('b' * 32)}.tmp"
    residue.write_bytes(b"partial")
    os.chmod(residue, 0o600)
    with pytest.raises(ReconciliationUnsafeError):
        inspect_reconciliation(
            outbox.root, namespace=NAMESPACE, artifacts_root=tmp_path / "artifacts"
        )


def test_confirmed_delivery_artifacts_redact_target_and_store_exact_receipt(tmp_path: Path) -> None:
    outbox, claimed, claim = _claimed(tmp_path)
    _mark_sending(outbox, claim)
    current = outbox.get(claimed.outbox_id)
    destination = current.destinations[0]
    attempt = destination.attempts[-1]
    receipt = ChannelTextReceipt(
        channel="telegram",
        target=destination.to,
        action_id=attempt.action_id,
        request_digest=attempt.request_digest,
        parts=(ChannelTextPartReceipt(0, 0, 0, 71, 999),),
    ).to_json()
    artifacts = tmp_path / "artifacts"
    evidence = import_evidence(
        outbox.root,
        namespace=NAMESPACE,
        artifacts_root=artifacts,
        outbox_id=current.outbox_id,
        destination_id=destination.destination_id,
        outcome="confirmed_delivered",
        receipt=receipt,
    )
    plan = create_plan(
        outbox.root,
        namespace=NAMESPACE,
        artifacts_root=artifacts,
        evidence_ref=evidence.reference,
    )
    assert plan.semantic_id is not None
    serialized_artifacts = b"".join(path.read_bytes() for path in artifacts.rglob("*.json"))
    assert b"raw-target-secret" not in serialized_artifacts
    assert b"secret body" not in serialized_artifacts
    apply_plan(
        outbox.root,
        namespace=NAMESPACE,
        artifacts_root=artifacts,
        plan_id=plan.semantic_id,
        confirm_plan=plan.semantic_id,
        confirmation="CONFIRM-DELIVERED",
    )
    resolved = outbox.get(current.outbox_id)
    assert resolved.destinations[0].status == "delivered"
    assert resolved.destinations[0].attempts[-1].receipt == receipt


def test_application_receipt_failure_retries_without_second_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outbox, current, _claim = _claimed(tmp_path)
    destination = current.destinations[0]
    artifacts = tmp_path / "artifacts"
    evidence = import_evidence(
        outbox.root,
        namespace=NAMESPACE,
        artifacts_root=artifacts,
        outbox_id=current.outbox_id,
        destination_id=destination.destination_id,
        outcome="confirmed_not_sent",
    )
    plan = create_plan(
        outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
        evidence_ref=evidence.reference,
    )
    assert plan.semantic_id is not None
    original_publish = reconciliation_module._ArtifactStore.publish
    failed = False

    def fail_once(store, kind, value):
        nonlocal failed
        if kind == "applications" and not failed:
            failed = True
            raise ReconciliationUnsafeError("simulated lost publication")
        return original_publish(store, kind, value)

    monkeypatch.setattr(reconciliation_module._ArtifactStore, "publish", fail_once)
    with pytest.raises(ReconciliationUnsafeError):
        apply_plan(
            outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
            plan_id=plan.semantic_id, confirm_plan=plan.semantic_id,
            confirmation="CONFIRM-NOT-SENT",
        )
    revision = outbox.get(current.outbox_id).record_revision
    result = apply_plan(
        outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
        plan_id=plan.semantic_id, confirm_plan=plan.semantic_id,
        confirmation="CONFIRM-NOT-SENT",
    )
    assert result["application_ref"].startswith("sha256:")
    assert outbox.get(current.outbox_id).record_revision == revision


def test_sending_abandon_and_live_send_fence_are_conservative(tmp_path: Path) -> None:
    outbox, claimed, claim = _claimed(tmp_path)
    _mark_sending(outbox, claim)
    current = outbox.get(claimed.outbox_id)
    destination = current.destinations[0]
    artifacts = tmp_path / "artifacts"
    evidence = import_evidence(
        outbox.root,
        namespace=NAMESPACE,
        artifacts_root=artifacts,
        outbox_id=current.outbox_id,
        destination_id=destination.destination_id,
        outcome="abandon",
    )
    plan = create_plan(
        outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
        evidence_ref=evidence.reference,
    )
    assert plan.semantic_id is not None

    async def contend() -> None:
        with outbox.acquire_send_fence(claim) as fence:
            outbox.mark_sending(claim, fence)
            revision = outbox.get(current.outbox_id).record_revision
            with pytest.raises(DeliverySendBusyError):
                apply_plan(
                    outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
                    plan_id=plan.semantic_id, confirm_plan=plan.semantic_id,
                    confirmation="CONFIRM-ABANDON",
                )
            assert outbox.get(current.outbox_id).record_revision == revision
            assert not list((artifacts / "applications").iterdir())

    asyncio.run(contend())
    apply_plan(
        outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
        plan_id=plan.semantic_id, confirm_plan=plan.semantic_id,
        confirmation="CONFIRM-ABANDON",
    )
    assert outbox.get(current.outbox_id).destinations[0].status == "failed"


@pytest.mark.parametrize("failure_point", ["file", "directory"])
def test_artifact_complete_residue_retry_reproves_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    root = tmp_path / "artifacts"
    with reconciliation_module._ArtifactStore(root, create=True) as store:
        original_fsync = reconciliation_module.os.fsync
        failed = False

        def fail_once(fd: int) -> None:
            nonlocal failed
            is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
            if not failed and is_directory == (failure_point == "directory"):
                failed = True
                raise OSError("simulated durability acknowledgement loss")
            original_fsync(fd)

        monkeypatch.setattr(reconciliation_module.os, "fsync", fail_once)
        with pytest.raises(ReconciliationUnsafeError):
            store.publish("evidence", {"schema": "test", "value": 1})
        assert failed
    monkeypatch.setattr(reconciliation_module.os, "fsync", original_fsync)
    with reconciliation_module._ArtifactStore(root, create=False) as store:
        replay = store.publish("evidence", {"schema": "test", "value": 1})
        assert replay.reference.startswith("sha256:")


def test_artifact_identical_path_replacement_before_directory_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    with reconciliation_module._ArtifactStore(root, create=True) as store:
        original_fsync = reconciliation_module.os.fsync
        replaced = False

        def replace_before_directory_fsync(fd: int) -> None:
            nonlocal replaced
            if not replaced and stat.S_ISDIR(os.fstat(fd).st_mode):
                files = list((root / "evidence").glob("*.json"))
                if files:
                    replaced = True
                    path = files[0]
                    data = path.read_bytes()
                    moved = tmp_path / "detached-original"
                    path.rename(moved)
                    path.write_bytes(data)
                    path.chmod(0o600)
            original_fsync(fd)

        monkeypatch.setattr(
            reconciliation_module.os, "fsync", replace_before_directory_fsync
        )
        with pytest.raises(ReconciliationUnsafeError):
            store.publish("evidence", {"schema": "test", "value": 1})
        assert replaced


def test_apply_missing_record_and_replaced_root_create_nothing(tmp_path: Path) -> None:
    outbox, current, _claim = _claimed(tmp_path)
    destination = current.destinations[0]
    artifacts = tmp_path / "artifacts"
    evidence = import_evidence(
        outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
        outbox_id=current.outbox_id, destination_id=destination.destination_id,
        outcome="confirmed_not_sent",
    )
    plan = create_plan(
        outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
        evidence_ref=evidence.reference,
    )
    assert plan.semantic_id is not None
    record_path = outbox.root / f"{current.outbox_id}.json"
    record_path.unlink()
    with pytest.raises(ReconciliationConflictError):
        apply_plan(
            outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
            plan_id=plan.semantic_id, confirm_plan=plan.semantic_id,
            confirmation="CONFIRM-NOT-SENT",
        )
    assert not record_path.exists()
    assert not list((artifacts / "applications").iterdir())

    moved = tmp_path / "moved-outbox"
    outbox.root.rename(moved)
    with pytest.raises(ReconciliationConflictError):
        apply_plan(
            outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
            plan_id=plan.semantic_id, confirm_plan=plan.semantic_id,
            confirmation="CONFIRM-NOT-SENT",
        )
    assert not outbox.root.exists()


def test_artifact_json_must_be_canonical_bounded_and_semantically_exact(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    with reconciliation_module._ArtifactStore(root, create=True):
        pass
    noncanonical = b'{"schema": "test"}'
    digest = hashlib.sha256(reconciliation_module._ARTIFACT_DOMAIN + noncanonical).hexdigest()
    path = root / "plans" / f"{digest}.json"
    path.write_bytes(noncanonical)
    path.chmod(0o600)
    with pytest.raises(ReconciliationUnsafeError):
        reconciliation_module._ArtifactStore(root, create=False)
    path.unlink()

    deep: object = "leaf"
    for _ in range(40):
        deep = {"nested": deep}
    data = reconciliation_module._canonical({"schema": "test", "value": deep})
    digest = hashlib.sha256(reconciliation_module._ARTIFACT_DOMAIN + data).hexdigest()
    path = root / "plans" / f"{digest}.json"
    path.write_bytes(data)
    path.chmod(0o600)
    with pytest.raises(ReconciliationUnsafeError):
        reconciliation_module._ArtifactStore(root, create=False)


def test_exact_reapply_uses_same_application_after_successor_revision(tmp_path: Path) -> None:
    outbox, current, _claim = _claimed(tmp_path)
    destination = current.destinations[0]
    artifacts = tmp_path / "artifacts"
    evidence = import_evidence(
        outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
        outbox_id=current.outbox_id, destination_id=destination.destination_id,
        outcome="confirmed_not_sent",
    )
    plan = create_plan(
        outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
        evidence_ref=evidence.reference,
    )
    assert plan.semantic_id is not None
    first = apply_plan(
        outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
        plan_id=plan.semantic_id, confirm_plan=plan.semantic_id,
        confirmation="CONFIRM-NOT-SENT",
    )
    after_apply = outbox.get(current.outbox_id)
    outbox.claim_destination(
        current.outbox_id, destination.destination_id, after_apply.record_revision,
        owner="successor",
    )
    advanced_revision = outbox.get(current.outbox_id).record_revision
    second = apply_plan(
        outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
        plan_id=plan.semantic_id, confirm_plan=plan.semantic_id,
        confirmation="CONFIRM-NOT-SENT",
    )
    assert second["application_ref"] == first["application_ref"]
    assert outbox.get(current.outbox_id).record_revision == advanced_revision


def test_plan_show_rejects_semantically_inconsistent_receipt(tmp_path: Path) -> None:
    outbox, current, _claim = _claimed(tmp_path)
    destination = current.destinations[0]
    artifacts = tmp_path / "artifacts"
    evidence = import_evidence(
        outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
        outbox_id=current.outbox_id, destination_id=destination.destination_id,
        outcome="confirmed_not_sent",
    )
    created = create_plan(
        outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
        evidence_ref=evidence.reference,
    )
    with reconciliation_module._ArtifactStore(artifacts, create=False) as store:
        name = next(iter(os.listdir(store.kind_fds["plans"])))
        malformed = store.read("plans", name[:-5])
        malformed["receipt"] = {"unexpected": True}
        body = {key: value for key, value in malformed.items() if key != "plan_id"}
        malformed["plan_id"] = hashlib.sha256(
            reconciliation_module._PLAN_DOMAIN + reconciliation_module._canonical(body)
        ).hexdigest()
        store.publish("plans", malformed)
    assert created.semantic_id is not None
    with pytest.raises(ReconciliationUnsafeError):
        show_plan(
            outbox.root, namespace=NAMESPACE, artifacts_root=artifacts,
            plan_id=malformed["plan_id"],
        )
