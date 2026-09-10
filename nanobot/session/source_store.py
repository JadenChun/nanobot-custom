"""Strict source transactions over the existing private, locked JSONL store.

Explicitly converted sessions use this API with a fenced execution owner.
Storage CAS prevents stale commits; dispatch authority belongs to that owner.
"""

from __future__ import annotations

import hashlib
import json
import stat
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.session.context_checkpoint import validate_context_checkpoint
from nanobot.session.persistence_metadata import LEGACY_RECOVERY_KEYS, accepted_metadata_updates
from nanobot.session.source import (
    SourceValidationError,
    canonical_events,
    merge_source_events,
    validate_event_ids,
    validate_execution_links,
)

if TYPE_CHECKING:
    from nanobot.session.artifact import ArtifactDescriptor
    from nanobot.session.artifact_store import ArtifactBlobTransaction, PrivateArtifactStore
    from nanobot.session.execution_owner import ExecutionOwner
    from nanobot.session.manager import JsonlSessionStore, Session


@dataclass(frozen=True)
class SourceState:
    store_identity: str
    session_key: str
    incarnation: str
    revision: int
    generation: int
    checkpoint_id: str | None
    checkpoint_fingerprint: str


@dataclass(frozen=True)
class SourceSnapshot:
    """Detached committed data. Mutating its dictionaries cannot change storage."""

    state: SourceState
    header: dict[str, Any]
    events: list[dict[str, Any]]
    checkpoint: dict[str, Any] | None

    def to_session(self) -> Session:
        from nanobot.session.manager import Session

        session = Session(
            key=self.state.session_key,
            messages=deepcopy(self.events),
            metadata=deepcopy(self.header["metadata"]),
            revision=self.state.revision,
            generation=self.state.generation,
            incarnation=self.state.incarnation,
            created_at=datetime.fromisoformat(self.header["created_at"]),
            updated_at=datetime.fromisoformat(self.header["updated_at"]),
            last_consolidated=self.header.get("last_archived", self.header.get("last_consolidated", 0)),
            provider_state=None,
        )
        session._remember_committed_base()
        return session


def _counter(value: object) -> int:
    if type(value) is not int or value < 0:
        raise SourceValidationError("Invalid source state counter")
    return value


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceValidationError("Duplicate source object key")
        result[key] = value
    return result


def _state(header: dict[str, Any], store_identity: str) -> SourceState:
    raw = header["metadata"].get("context_checkpoint")
    identity = raw.get("checkpoint_id") if isinstance(raw, dict) else None
    return SourceState(
        store_identity=store_identity,
        session_key=header["key"], incarnation=header.get("incarnation", "legacy"),
        revision=_counter(header.get("revision", 0)),
        generation=_counter(header.get("generation", 0)),
        checkpoint_id=identity if isinstance(identity, str) and identity else None,
        checkpoint_fingerprint=hashlib.sha256(canonical_events([{"checkpoint": raw}])).hexdigest(),
    )


def _upgrade_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign missing identities from structured evidence, never from text."""
    upgraded = deepcopy(events)
    pending: dict[str, list[str]] = {}
    for event in upgraded:
        if "_event_id" not in event:
            event["_event_id"] = uuid.uuid4().hex
        if "tool_calls" in event:
            calls = event["tool_calls"]
            if event.get("role") != "assistant" or not isinstance(calls, list):
                raise SourceValidationError("Invalid legacy execution linkage")
            identities = event.get("_execution_ids")
            if identities is None:
                identities = [uuid.uuid4().hex for _ in calls]
                event["_execution_ids"] = identities
            if not isinstance(identities, list) or len(identities) != len(calls):
                raise SourceValidationError("Invalid legacy execution linkage")
            for call, identity in zip(calls, identities):
                if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
                    raise SourceValidationError("Invalid legacy provider call")
                if pending.get(call["id"]):
                    raise SourceValidationError("Ambiguous legacy execution linkage")
                pending[call["id"]] = [identity]
        if event.get("role") == "tool":
            provider_id = event.get("tool_call_id")
            if not isinstance(provider_id, str) or len(pending.get(provider_id, [])) != 1:
                raise SourceValidationError("Unmatched legacy execution result")
            identity = pending.pop(provider_id)[0]
            if "_execution_id" in event and event["_execution_id"] != identity:
                raise SourceValidationError("Conflicting legacy execution linkage")
            event["_execution_id"] = identity
    validate_event_ids(upgraded)
    validate_execution_links(upgraded)
    return upgraded


class SourceSessionStore:
    """Mandatory strict operations; never fall back to ordinary Session.save."""

    def __init__(self, backend: JsonlSessionStore, *, artifact_store: PrivateArtifactStore | None = None):
        self._backend = backend
        if artifact_store is not None and (
            artifact_store.workspace_id != backend.sessions_dir.name
            or artifact_store.root == backend.workspace
            or artifact_store.root.is_relative_to(backend.workspace)
        ):
            raise SourceValidationError('Invalid configured artifact store binding')
        self._artifact_store = artifact_store
        self._store_identity = hashlib.sha256(str(backend.sessions_dir.resolve()).encode()).hexdigest()

    def _state(self, header: dict[str, Any]) -> SourceState:
        return _state(header, self._store_identity)

    def _read_unlocked(self, key: str, *, require_source: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        from nanobot.session.manager import SessionStorageError

        path = self._backend.get_session_path(key)
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                raise SourceValidationError("Non-regular source file")
            with path.open(encoding="utf-8") as handle:
                records = [json.loads(line, object_pairs_hook=_object_pairs) for line in handle if line.strip()]
            if not records or not isinstance(records[0], dict):
                raise SourceValidationError("Invalid source header")
            header = records[0]
            canonical_events(records)
            if header.get("_type") != "metadata" or header.get("key") != key:
                raise SourceValidationError("Invalid source identity")
            metadata = header.get("metadata", {})
            if not isinstance(metadata, dict):
                raise SourceValidationError("Invalid source metadata")
            header["metadata"] = metadata
            schema = metadata.get("source_schema_version")
            if (require_source or "source_schema_version" in metadata) and (type(schema) is not int or schema != 1):
                raise SourceValidationError("Unsupported source schema")
            if require_source or "source_schema_version" in metadata:
                if not {"revision", "generation", "incarnation", "last_archived", "last_consolidated"}.issubset(header):
                    raise SourceValidationError("Missing source state")
                if header["last_archived"] != header["last_consolidated"]:
                    raise SourceValidationError("Inconsistent source archive cursor")
                if "runtime_checkpoint" in metadata:
                    raise SourceValidationError("Legacy runtime checkpoint in source session")
            state = self._state(header)
            if not isinstance(state.incarnation, str) or not state.incarnation:
                raise SourceValidationError("Invalid source incarnation")
            for name in ("created_at", "updated_at"):
                if name in header:
                    datetime.fromisoformat(header[name])
                elif require_source:
                    raise SourceValidationError("Missing source timestamp")
            events = []
            for record in records[1:]:
                if not isinstance(record, dict):
                    raise SourceValidationError("Invalid source record")
                if record.get("_type") == "provider_state" and not require_source and schema is None:
                    continue  # Handles are explicitly invalidated by the upgrade transaction.
                if (
                    "_type" in record or not isinstance(record.get("role"), str)
                    or record["role"] not in {"user", "assistant", "tool", "system", "developer"}
                ):
                    raise SourceValidationError("Invalid source event")
                events.append(record)
            for name in ("last_archived", "last_consolidated"):
                if name in header and _counter(header[name]) > len(events):
                    raise SourceValidationError("Invalid source archive boundary")
            if require_source or "source_schema_version" in metadata:
                validate_event_ids(events)
                validate_execution_links(events)
            return header, events
        except FileNotFoundError:
            raise
        except (ValueError, TypeError, AttributeError, UnicodeError, RecursionError):
            raise SessionStorageError() from None

    def _snapshot(self, header: dict[str, Any], events: list[dict[str, Any]]) -> SourceSnapshot:
        state = self._state(header)
        candidate = header["metadata"].get("context_checkpoint")
        checkpoint = None
        if candidate is not None:
            try:
                if not isinstance(candidate, dict):
                    raise SourceValidationError("Invalid stored checkpoint")
                source_revision = _counter(candidate.get("source_revision"))
                parent = candidate.get("parent_checkpoint_id")
                if source_revision >= state.revision or (
                    parent is not None and (not isinstance(parent, str) or not parent)
                ):
                    raise SourceValidationError("Invalid stored checkpoint provenance")
                checkpoint = validate_context_checkpoint(
                    candidate, events, generation=state.generation,
                    source_revision=source_revision, parent_checkpoint_id=parent,
                )
            except SourceValidationError:
                logger.warning("Ignoring invalid context checkpoint; retained source remains authoritative")
        return SourceSnapshot(state, deepcopy(header), deepcopy(events), checkpoint)

    def load_source_snapshot(self, key: str) -> SourceSnapshot:
        with self._backend.locked_session_files():
            return self._snapshot(*self._read_unlocked(key))

    def _check(self, header: dict[str, Any], expected: SourceState) -> None:
        from nanobot.session.manager import SessionWriteConflict

        if (
            type(expected) is not SourceState
            or type(expected.revision) is not int or type(expected.generation) is not int
            or self._state(header) != expected
        ):
            raise SessionWriteConflict()
        if self._backend.get_runtime_checkpoint_path(header["key"]).exists():
            raise SessionWriteConflict()

    def _current_unlocked(self, key: str, expected: SourceState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        from nanobot.session.manager import SessionWriteConflict

        try:
            header, events = self._read_unlocked(key)
        except FileNotFoundError:
            raise SessionWriteConflict() from None
        self._check(header, expected)
        return header, events

    def _write(self, header: dict[str, Any], events: list[dict[str, Any]]) -> SourceSnapshot:
        header, events = deepcopy(header), deepcopy(events)
        # Validate and prepare the detached result before the irreversible replace.
        validate_event_ids(events)
        validate_execution_links(events)
        for event in events:
            if (
                "_type" in event or not isinstance(event.get("role"), str)
                or event["role"] not in {"user", "assistant", "tool", "system", "developer"}
            ):
                raise SourceValidationError("Invalid source event")
        records = [header, *events]
        canonical_events(records)
        result = self._snapshot(header, events)
        text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records)
        self._backend._write_text_atomic(self._backend.get_session_path(header["key"]), text)
        return result

    def upgrade_source_session(self, key: str, *, expected_revision: int, expected_generation: int, expected_incarnation: str) -> SourceSnapshot:
        from nanobot.session.execution_owner import execution_transaction
        from nanobot.session.manager import SessionWriteConflict

        with execution_transaction(self._backend, key, allow_legacy=True):
            header, events = self._read_unlocked(key, require_source=False)
            state = self._state(header)
            if (
                type(expected_revision) is not int or type(expected_generation) is not int
                or (state.revision, state.generation, state.incarnation) != (
                    expected_revision, expected_generation, expected_incarnation,
                )
                or "source_schema_version" in header["metadata"]
                or "runtime_checkpoint" in header["metadata"]
                or header["metadata"].get("pending_user_turn")
                or header["metadata"].get("pending_user_followups")
                or (
                    header["metadata"].get("webui_recovery") is not None
                    and (
                        not isinstance(header["metadata"]["webui_recovery"], dict)
                        or header["metadata"]["webui_recovery"].get("status") != "recovered"
                    )
                )
                or self._backend.get_runtime_checkpoint_path(key).exists()
            ):
                raise SessionWriteConflict()
            upgraded = _upgrade_events(events)
            header["metadata"]["source_schema_version"] = 1
            # Upstream model-index summary markers remain source rows, but never
            # become authoritative source checkpoints or projection boundaries.
            header["revision"] = state.revision + 1
            header["generation"] = state.generation
            header["incarnation"] = state.incarnation
            cursor = header.get("last_archived", header.get("last_consolidated", 0))
            header["last_archived"] = header["last_consolidated"] = cursor
            now = datetime.now().isoformat()
            header.setdefault("created_at", now)
            header.setdefault("updated_at", now)
            return self._write(header, upgraded)

    def append_source_events(self, key: str, expected: SourceState, events: list[dict[str, Any]], *, owner: ExecutionOwner | None = None) -> SourceSnapshot:
        from nanobot.session.execution_owner import execution_transaction

        with execution_transaction(self._backend, key, owner):
            header, committed = self._current_unlocked(key, expected)
            merged = merge_source_events(committed, events)
            if merged == committed:
                return self._snapshot(header, committed)
            header["revision"] += 1
            header["updated_at"] = datetime.now().isoformat()
            return self._write(header, merged)

    def commit_context_checkpoint(self, key: str, expected: SourceState, candidate: dict[str, Any], *, owner: ExecutionOwner | None = None) -> SourceSnapshot:
        from nanobot.session.execution_owner import execution_transaction

        with execution_transaction(self._backend, key, owner):
            header, events = self._current_unlocked(key, expected)
            validated = validate_context_checkpoint(
                candidate, events, generation=expected.generation,
                source_revision=expected.revision, parent_checkpoint_id=expected.checkpoint_id,
            )
            header["metadata"]["context_checkpoint"] = validated
            header["revision"] += 1
            return self._write(header, events)

    def commit_artifact_descriptor(self, key: str, expected: SourceState, descriptor: ArtifactDescriptor, *,
                                   owner: ExecutionOwner, blobs: ArtifactBlobTransaction) -> SourceSnapshot:
        """Trusted internal publication; this does not attest origin or grant reads.

        Runtime producers must authenticate source_access before calling. No
        retrieval path may treat this dormant primitive's manifest membership as
        origin proof. The caller retains the configured artifact transaction,
        ordered before owner/session locks, throughout publication.
        """
        from nanobot.session.artifact import (
            bound_artifact_text,
            parse_artifact_descriptor,
            published_artifacts,
        )
        from nanobot.session.artifact_store import ArtifactBlobTransaction
        from nanobot.session.execution_owner import ExecutionOwnershipError, execution_transaction

        if owner is None:
            raise ExecutionOwnershipError()
        if self._artifact_store is None or not isinstance(blobs, ArtifactBlobTransaction):
            raise SourceValidationError('Artifact publication requires a configured store transaction')
        descriptor = parse_artifact_descriptor(descriptor.to_dict())
        blobs.verify_durable(descriptor, store=self._artifact_store)
        owner.validate_dispatch()
        with execution_transaction(self._backend, key, owner):
            header, events = self._current_unlocked(key, expected)
            owner.validate_dispatch()
            snapshot = self._snapshot(header, events)
            bound_artifact_text(descriptor, snapshot, workspace_id=self._backend.sessions_dir.name)
            existing = published_artifacts(snapshot, workspace_id=self._backend.sessions_dir.name)
            for item in existing:
                if item.artifact_id == descriptor.artifact_id or item.idempotency_key == descriptor.idempotency_key:
                    if item != descriptor:
                        raise SourceValidationError('Conflicting immutable artifact publication')
                    blobs.verify_durable(descriptor, store=self._artifact_store)
                    return snapshot
            header['metadata']['artifact_manifest'] = {'schema_version': 1, 'descriptors': {
                item.artifact_id: item.to_dict() for item in (*existing, descriptor)
            }}
            header['revision'] += 1
            blobs.assert_reachable()
            return self._write(header, events)

    def reset_source_session(self, key: str, expected: SourceState, replacement_events: list[dict[str, Any]], *, owner: ExecutionOwner | None = None) -> SourceSnapshot:
        from nanobot.session.execution_owner import execution_transaction

        with execution_transaction(self._backend, key, owner, destructive=True):
            header, _ = self._current_unlocked(key, expected)
            header["revision"] += 1
            header["generation"] += 1
            header["last_archived"] = header["last_consolidated"] = 0
            header["updated_at"] = datetime.now().isoformat()
            for name in {"context_checkpoint", "artifact_manifest", "_last_summary", "execution_lease", "execution_recovery_required", *LEGACY_RECOVERY_KEYS}:
                header["metadata"].pop(name, None)
            result = self._write(header, replacement_events)
            if owner is not None:
                owner._invalidate_after_destructive_write()
            return result

    def update_metadata(self, key: str, updates: dict[str, Any]) -> SourceSnapshot:
        with self._backend.locked_session_files():
            header, events = self._read_unlocked(key)
            self._check(header, self._state(header))
            header["metadata"].update(accepted_metadata_updates(updates, source=True))
            header["revision"] += 1
            return self._write(header, events)

    def advance_consolidated(self, key: str, expected: SourceState, end_offset: int, *, owner: ExecutionOwner | None = None) -> SourceSnapshot:
        from nanobot.session.execution_owner import execution_transaction

        with execution_transaction(self._backend, key, owner):
            header, events = self._current_unlocked(key, expected)
            if not header.get("last_consolidated", 0) <= _counter(end_offset) <= len(events):
                raise SourceValidationError("Invalid consolidation boundary")
            if end_offset == header.get("last_consolidated", 0):
                return self._snapshot(header, events)
            header["last_archived"] = header["last_consolidated"] = end_offset
            header["revision"] += 1
            return self._write(header, events)


    def delete_source_session(self, key: str, expected: SourceState, *, owner: ExecutionOwner | None = None) -> None:
        from nanobot.session.execution_owner import execution_transaction

        with execution_transaction(self._backend, key, owner, destructive=True):
            self._delete_unlocked(key, expected, owner=owner)

    def _delete_unlocked(self, key: str, expected: SourceState, *, owner: ExecutionOwner | None = None) -> None:
        self._current_unlocked(key, expected)
        self._backend.get_session_path(key).unlink()
        if owner is not None:
            owner._invalidate_after_destructive_write()
        self._backend._fsync_directory(self._backend.sessions_dir)
