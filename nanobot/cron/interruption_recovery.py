"""Qualified, non-replaying recovery for interrupted legacy-fresh cron sessions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

from nanobot.cron.definition_store import CronDefinition, DefinitionStoreValidationError
from nanobot.cron.occurrence import destination_policy
from nanobot.cron.occurrence_store import (
    CronOccurrenceStore,
    OccurrenceExecutionBusyError,
    OccurrenceOwnershipError,
    OccurrenceRecord,
    ScheduledOccurrenceIdentity,
)
from nanobot.cron.service import CronService
from nanobot.cron.types import CronPayload
from nanobot.session.execution_owner import (
    ExecutionBusyError,
    ExecutionOwnershipError,
    acquire_execution,
)
from nanobot.session.manager import SessionManager, SessionStorageError
from nanobot.session.persistence_metadata import (
    BUSY_INBOX_METADATA_KEY,
    LEGACY_RECOVERY_KEYS,
    SOURCE_PENDING_FOLLOWUPS_METADATA_KEY,
)
from nanobot.session.session_mode import (
    SessionInterruptionRecoveryRefusedError,
    SessionModeRecordError,
)
from nanobot.session.source import SourceValidationError, validate_execution_links
from nanobot.session.source_store import SourceSnapshot

_RECOVERY_RUN_PREFIX = "cron-occurrence-"
_RECOVERY_FENCE_REASON = "execution_recovery_required"
_RESET_CLEARED_METADATA = frozenset({
    "context_checkpoint",
    "artifact_manifest",
    "_last_summary",
    "user_activity_revision",
    "last_user_activity_at",
    SOURCE_PENDING_FOLLOWUPS_METADATA_KEY,
    "workspace_access_request",
    *LEGACY_RECOVERY_KEYS,
})


class CronInterruptionRecoveryRefusedError(RuntimeError):
    """The session does not meet the exact cron interruption recovery contract."""


def _refuse() -> CronInterruptionRecoveryRefusedError:
    return CronInterruptionRecoveryRefusedError(_RECOVERY_FENCE_REASON)


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _qualified_occurrence(
    record: OccurrenceRecord,
    *,
    job_id: str,
    session_key: str,
    namespace: str,
) -> bool:
    if (
        type(record.identity) is not ScheduledOccurrenceIdentity
        or record.identity.store_namespace != namespace
        or record.identity.job_id != job_id
        or record.state != "claimed"
        or record.completion is not None
        or record.reconciliation is not None
    ):
        return False
    try:
        definition = CronDefinition.decode_wire(record.job_snapshot, label="cron occurrence")
        payload = CronPayload.from_store_dict(definition.definition["payload"])
    except (DefinitionStoreValidationError, KeyError, TypeError, ValueError):
        return False
    return (
        definition.job_id == job_id
        and definition.definition_revision == record.identity.job_revision
        and definition.definition["enabled"] is True
        and payload.kind == "agent_turn"
        and payload.execution_mode == "legacy_fresh"
        and payload.session_key == session_key == f"cron:{job_id}"
        and payload.origin_channel == (payload.channel or "cli")
        and payload.origin_chat_id == str(payload.to or "direct")
        and bool(payload.message.strip())
        and record.destination_policy == destination_policy(definition)
    )


def _interrupted_tail(snapshot: SourceSnapshot) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    events = cast(list[dict[str, object]], snapshot.events)
    unresolved = validate_execution_links(events)
    if len(unresolved) != 1 or not events:
        raise _refuse()
    unresolved_id = next(iter(unresolved))
    intent_indices = [
        index
        for index, event in enumerate(events)
        if isinstance(event.get("_execution_ids"), list)
        and unresolved_id in cast(list[object], event["_execution_ids"])
    ]
    if intent_indices != [len(events) - 1]:
        raise _refuse()
    run_id = events[-1].get("_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise _refuse()
    start = next(
        (index for index, event in enumerate(events) if event.get("_run_id") == run_id),
        len(events),
    )
    prefix = events[:start]
    tail = events[start:]
    if (
        not tail
        or tail[0].get("role") != "user"
        or tail[-1].get("role") != "assistant"
        or any(event.get("_run_id") != run_id for event in tail)
        or tail[-1].get("_run_id") != run_id
        or validate_execution_links(prefix)
        or (prefix and prefix[-1].get("role") != "assistant")
    ):
        raise _refuse()
    return run_id, prefix, tail


def _validate_source_snapshot(
    snapshot: SourceSnapshot,
    *,
    key: str,
    run_id: str,
    checkpoint_path: Path,
) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    metadata = snapshot.header.get("metadata")
    if not isinstance(metadata, dict) or not all(
        isinstance(name, str) for name in cast(dict[object, object], metadata)
    ):
        raise _refuse()
    metadata = cast(dict[str, object], metadata)
    if type(metadata.get("source_schema_version")) is not int or metadata.get("source_schema_version") != 1:
        raise _refuse()
    if snapshot.state.session_key != key:
        raise _refuse()
    if snapshot.checkpoint is not None or "context_checkpoint" in metadata:
        raise _refuse()
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        raise _refuse()
    if _RESET_CLEARED_METADATA.intersection(metadata):
        raise _refuse()
    if BUSY_INBOX_METADATA_KEY in metadata:
        raise _refuse()
    if "runtime_checkpoint" in metadata or "pending_user_turn" in metadata or "pending_user_followups" in metadata:
        raise _refuse()

    marker = metadata.get("execution_recovery_required")
    if "execution_recovery_required" in metadata and marker is not True:
        raise _refuse()
    lease = metadata.get("execution_lease")
    if "execution_lease" in metadata:
        lease_fields = {
            "schema_version", "token", "run_id", "incarnation", "generation",
            "acquired_at", "recovery_required",
        }
        if (
            not isinstance(lease, dict)
            or set(cast(dict[object, object], lease)) != lease_fields
            or type(cast(dict[object, object], lease).get("schema_version")) is not int
            or cast(dict[object, object], lease).get("schema_version") != 1
            or not isinstance(cast(dict[object, object], lease).get("token"), str)
            or not cast(str, cast(dict[object, object], lease)["token"])
            or cast(dict[object, object], lease).get("run_id") != run_id
            or cast(dict[object, object], lease).get("incarnation") != snapshot.state.incarnation
            or type(cast(dict[object, object], lease).get("generation")) is not int
            or cast(dict[object, object], lease).get("generation") != snapshot.state.generation
            or not _valid_timestamp(cast(dict[object, object], lease).get("acquired_at"))
            or type(cast(dict[object, object], lease).get("recovery_required")) is not bool
        ):
            raise _refuse()
    elif marker is not True:
        raise _refuse()

    run_id_from_history, prefix, tail = _interrupted_tail(snapshot)
    if run_id_from_history != run_id:
        raise _refuse()
    return run_id_from_history, prefix, tail


async def recover_interrupted_cron_sources(
    sessions: SessionManager,
    service: CronService,
    occurrences: CronOccurrenceStore,
) -> dict[str, str]:
    """Recover only exact abandoned timer runs; return per-job admission fences."""

    records = occurrences.inspect_records()
    records_by_id = {record.occurrence_id: record for record in records}
    fences: dict[str, str] = {}
    for job in service.list_jobs(include_disabled=True):
        key = job.payload.session_key
        if not isinstance(key, str) or key != f"cron:{job.id}":
            continue
        if sessions.has_runtime_checkpoint(key):
            fences[job.id] = _RECOVERY_FENCE_REASON
            continue
        try:
            pending_reference = sessions.peek_cron_interruption_reset_occurrence(key)
        except (SessionInterruptionRecoveryRefusedError, SessionModeRecordError) as exc:
            if isinstance(exc, SessionModeRecordError) and exc.session_key != key:
                raise
            fences[job.id] = _RECOVERY_FENCE_REASON
            continue

        occurrence: OccurrenceRecord | None = None
        if pending_reference is not None:
            occurrence_id, pending_run_id = pending_reference
            candidate_occurrence = records_by_id.get(occurrence_id)
            if (
                candidate_occurrence is None
                or (
                    f"{_RECOVERY_RUN_PREFIX}{candidate_occurrence.occurrence_id}-"
                    f"{candidate_occurrence.generation}"
                ) != pending_run_id
                or not _qualified_occurrence(
                    candidate_occurrence,
                    job_id=job.id,
                    session_key=key,
                    namespace=occurrences.namespace,
                )
            ):
                fences[job.id] = _RECOVERY_FENCE_REASON
                continue
            occurrence = candidate_occurrence
        else:
            try:
                candidate = sessions.read_existing_canonical_source(key)
            except (SessionStorageError, SourceValidationError, SessionModeRecordError) as exc:
                if isinstance(exc, SessionModeRecordError) and exc.session_key != key:
                    raise
                fences[job.id] = _RECOVERY_FENCE_REASON
                continue
            if candidate is None:
                continue
            _source_backend, snapshot = candidate
            metadata = snapshot.header.get("metadata")
            if not isinstance(metadata, dict):
                fences[job.id] = _RECOVERY_FENCE_REASON
                continue
            try:
                unresolved = validate_execution_links(snapshot.events)
            except SourceValidationError:
                fences[job.id] = _RECOVERY_FENCE_REASON
                continue
            if (
                "execution_recovery_required" not in metadata
                and "execution_lease" not in metadata
                and not unresolved
            ):
                continue
            try:
                run_id, _prefix, _tail = _interrupted_tail(snapshot)
            except (SourceValidationError, CronInterruptionRecoveryRefusedError):
                fences[job.id] = _RECOVERY_FENCE_REASON
                continue
            matches = [
                record
                for record in records
                if _qualified_occurrence(
                    record,
                    job_id=job.id,
                    session_key=key,
                    namespace=occurrences.namespace,
                )
                and run_id == f"{_RECOVERY_RUN_PREFIX}{record.occurrence_id}-{record.generation}"
            ]
            if len(matches) != 1:
                fences[job.id] = _RECOVERY_FENCE_REASON
                continue
            occurrence = matches[0]

        try:
            with occurrences.hold_claimed_for_recovery(occurrence):
                current_pending = sessions.peek_cron_interruption_reset_occurrence(key)
                expected_run_id = (
                    f"{_RECOVERY_RUN_PREFIX}{occurrence.occurrence_id}-{occurrence.generation}"
                )
                expected_pending = (occurrence.occurrence_id, expected_run_id)
                if current_pending is not None and current_pending != expected_pending:
                    raise _refuse()
                reset_result = sessions.recover_cron_interruption_reset(
                    key,
                    expected_occurrence_id=occurrence.occurrence_id,
                )
                if reset_result == "completed":
                    continue

                try:
                    candidate = sessions.read_existing_canonical_source(key)
                except (SessionStorageError, SourceValidationError, SessionModeRecordError) as exc:
                    if isinstance(exc, SessionModeRecordError) and exc.session_key != key:
                        raise
                    raise _refuse() from None
                if candidate is None:
                    raise _refuse()
                source_backend, snapshot = candidate
                metadata = snapshot.header.get("metadata")
                if not isinstance(metadata, dict):
                    raise _refuse()
                try:
                    unresolved = validate_execution_links(snapshot.events)
                except SourceValidationError:
                    raise _refuse() from None
                if (
                    "execution_recovery_required" not in metadata
                    and "execution_lease" not in metadata
                    and not unresolved
                ):
                    if reset_result == "aborted":
                        raise _refuse()
                    continue
                run_id, prefix, tail = _interrupted_tail(snapshot)
                if run_id != expected_run_id:
                    raise _refuse()
                checkpoint_path = source_backend.get_runtime_checkpoint_path(key)
                _validate_source_snapshot(
                    snapshot,
                    key=key,
                    run_id=run_id,
                    checkpoint_path=checkpoint_path,
                )

                source_key: str = key

                def validate_pre_acquisition(current: SourceSnapshot) -> None:
                    if (
                        current.state != snapshot.state
                        or current.header != snapshot.header
                        or current.events != snapshot.events
                    ):
                        raise _refuse()
                    _validate_source_snapshot(
                        current,
                        key=source_key,
                        run_id=run_id,
                        checkpoint_path=checkpoint_path,
                    )

                async with acquire_execution(
                    source_backend,
                    key,
                    run_id=run_id,
                    wait_for_lock=False,
                    validate_snapshot=validate_pre_acquisition,
                ) as owner:
                    current = owner.read_snapshot()
                    if (
                        current.state.store_identity != snapshot.state.store_identity
                        or current.state.incarnation != snapshot.state.incarnation
                        or current.state.generation != snapshot.state.generation
                        or current.events != snapshot.events
                        or not owner.recovery_required
                    ):
                        raise _refuse()
                    identity, digest = sessions.publish_cron_interruption_archive(
                        key,
                        owner=owner,
                        occurrence_id=occurrence.occurrence_id,
                        run_id=run_id,
                        prefix_events=prefix,
                        interrupted_events=tail,
                        unresolved_execution_id=next(iter(unresolved)),
                    )
                    await owner.quiesce()
                    sessions.reset_session(
                        key,
                        owner=owner,
                        replacement_events=prefix,
                        cron_interruption_archive_identity=identity,
                        cron_interruption_archive_digest=digest,
                    )
        except (
            ExecutionBusyError,
            ExecutionOwnershipError,
            OccurrenceExecutionBusyError,
            OccurrenceOwnershipError,
            SessionInterruptionRecoveryRefusedError,
            SessionModeRecordError,
            CronInterruptionRecoveryRefusedError,
            SourceValidationError,
            SessionStorageError,
        ) as exc:
            if isinstance(exc, SessionModeRecordError) and exc.session_key != key:
                raise
            fences[job.id] = _RECOVERY_FENCE_REASON
    return fences


__all__ = ["CronInterruptionRecoveryRefusedError", "recover_interrupted_cron_sources"]
