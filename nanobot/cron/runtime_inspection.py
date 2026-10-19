"""Explicit-root existing-only content-free offline cron operator plane."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from nanobot.cron.occurrence_store import CronOccurrenceStore
from nanobot.cron.scheduler_store import (
    ReconciliationBridge,
    SchedulerConflictError,
    SchedulerStore,
    bridge_reconciliation,
    immutable_occurrence_digest,
    matches_bridge_reconciliation,
    prove_trigger_request,
    verify_occurrence_evidence,
)

MAX_INSPECTION_RECORDS = 4096


def inspect_scheduler(root: Path, *, namespace: str, runtime_identity: str) -> dict[str, object]:
    return SchedulerStore.inspect_existing_evidence(Path(root), namespace=namespace, runtime_identity=runtime_identity)


def inspect_occurrences(root: Path, *, namespace: str) -> dict[str, object]:
    return CronOccurrenceStore.inspect_existing_evidence(Path(root), namespace=namespace,
        max_records=lambda: MAX_INSPECTION_RECORDS)


def reconcile_occurrence(root: Path, *, namespace: str, occurrence_id: str,
                         expected_owner: str, expected_generation: int,
                         resolution: str, external_evidence: str, successor_owner: str,
                         scheduler_root: Path, runtime_identity: str) -> dict[str, object]:
    """Explicit external attestation plus the existing internally fenced CAS.

    A free execution lock is never the evidence. This operation returns only
    content-free successor evidence; it does not compute or dispatch anything.
    """
    supported_resolution = CronOccurrenceStore.validate_reconciliation_request(
        external_evidence=external_evidence, resolution=resolution,
        expected_generation=expected_generation, expected_owner=expected_owner,
        successor_owner=successor_owner)
    inspect_occurrences(root, namespace=namespace)  # Existing-only before writer construction.
    inspect_scheduler(scheduler_root, namespace=namespace, runtime_identity=runtime_identity)
    scheduler = SchedulerStore(scheduler_root, namespace=namespace, runtime_identity=runtime_identity)
    with scheduler.stopped_writer() as lease:
        state = scheduler.read_for_acquisition(lease)
        store = CronOccurrenceStore(root, namespace=namespace)
        records = store.inspect_records()
        record = next((item for item in records if item.occurrence_id == occurrence_id), None)
        if record is None:
            raise SchedulerConflictError("reconciliation occurrence absent")
        digest = immutable_occurrence_digest(record)
        prove_trigger_request(state, record)
        bridge: ReconciliationBridge = {"occurrence_id": occurrence_id,
            "identity": record.identity.as_dict(), "immutable_digest": digest,
            "predecessor_owner": expected_owner, "predecessor_generation": expected_generation,
            "successor_owner": successor_owner, "successor_generation": expected_generation + 1,
            "resolution": supported_resolution, "external_evidence": external_evidence,
            "publication_generation": state["generation"] + 1}
        retry = (record.owner, record.generation) == (successor_owner, expected_generation + 1)
        if record.state != "claimed" or (retry and not matches_bridge_reconciliation(record, bridge)):
            raise SchedulerConflictError("reconciliation exact successor conflict")
        if not retry and (record.owner, record.generation) != (expected_owner, expected_generation):
            raise SchedulerConflictError("reconciliation predecessor stale")
        retained = next((item for item in state.get("bridges", []) if item["occurrence_id"] == occurrence_id
                         and item["predecessor_generation"] == expected_generation), None)
        if retained is not None:
            bridge["publication_generation"] = retained["publication_generation"]
            if not retry or retained != bridge:
                raise SchedulerConflictError("reconciliation retained bridge conflict")
            successor = state
        else:
            successor = scheduler.prepare_bridge(state, bridge)  # Bounds BEFORE occurrence mutation.
        if not retry:
            # A previous transition must be durably retained before its last
            # occurrence reconciliation attestation can be overwritten.
            if record.reconciliation is not None and not any(item["occurrence_id"] == occurrence_id
                    and item["successor_generation"] == record.generation for item in state.get("bridges", [])):
                raise SchedulerConflictError("repair prior reconciliation bridge first")
            verify_occurrence_evidence(state, records)
            anticipated = replace(record, owner=successor_owner, generation=expected_generation + 1,
                                  reconciliation=bridge_reconciliation(bridge))
            verify_occurrence_evidence(successor, [anticipated if item is record else item for item in records])
            lease.reprove()
            # No short scheduler lock spans this unchanged internally fenced CAS.
            store.reclaim_after_reconciliation(occurrence_id, owner=successor_owner,
                expected_owner=expected_owner, expected_generation=expected_generation, resolution=supported_resolution)
        records = store.inspect_records()  # Mandatory durable successor reproof, even after a lost ack.
        verified = next(item for item in records if item.occurrence_id == occurrence_id)
        if (verified.state != "claimed" or verified.identity != record.identity
                or verified.owner != successor_owner or verified.generation != expected_generation + 1
                or not matches_bridge_reconciliation(verified, bridge)
                or immutable_occurrence_digest(verified) != digest):
            raise SchedulerConflictError("reconciliation successor changed")
        verify_occurrence_evidence(successor, records)
        lease.reprove()
        scheduler.compare_and_swap(state, successor)
        published = scheduler.read()
        verify_occurrence_evidence(published, store.inspect_records())
        lease.reprove()
    return {"classification": "reconciled_claim", "occurrence_id": occurrence_id,
            "owner": successor_owner, "generation": expected_generation + 1}
