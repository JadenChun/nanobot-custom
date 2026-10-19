"""Strict source transactions over the existing private, locked JSONL store.

Explicitly converted sessions use this API with a fenced execution owner.
Storage CAS prevents stale commits; dispatch authority belongs to that owner.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Protocol, cast

from loguru import logger

from nanobot.session.context_checkpoint import validate_context_checkpoint
from nanobot.session.idle_compaction import (
    LAST_USER_ACTIVITY_AT_KEY,
    USER_ACTIVITY_REVISION_KEY,
    IdleCaptureConflictError,
    IdleCaptureToken,
    IdleCheckpointAdmission,
    capture_idle_token,
    checkpoint_candidate_fingerprint,
    read_user_activity,
    validate_idle_capture,
)
from nanobot.session.pending_followups import (
    PendingFollowupRecord,
    parse_pending_followup_records,
)
from nanobot.session.persistence_metadata import (
    BUSY_INBOX_METADATA_KEY,
    LEGACY_RECOVERY_KEYS,
    SOURCE_PENDING_FOLLOWUPS_METADATA_KEY,
    accepted_metadata_updates,
)
from nanobot.session.source import (
    SourceValidationError,
    canonical_events,
    merge_source_events,
    validate_event_ids,
    validate_execution_links,
)

if TYPE_CHECKING:
    from nanobot.session.artifact import ArtifactDescriptor
    from nanobot.session.artifact_reader import ArtifactReader
    from nanobot.session.artifact_store import ArtifactBlobTransaction, PrivateArtifactStore
    from nanobot.session.checkpoint_dependencies import SummaryDependencyProof
    from nanobot.session.execution_owner import ExecutionOwner
    from nanobot.session.idle_checkpoint import (
        IdleCaptureIssuer,
        IdleCheckpointCapture,
        IdlePolicyBinding,
    )
    from nanobot.session.manager import JsonlSessionStore, Session
    from nanobot.session.source_discovery import SourceDiscoveryCursor, SourceDiscoveryLimits
    from nanobot.session.source_root import SourceRootAuthority


class ArtifactSourceLocation(Protocol):
    """Read-only location metadata; no storage or invocation authority."""

    @property
    def workspace(self) -> Path: ...

    @property
    def sessions_dir(self) -> Path: ...

    @property
    def workspace_id(self) -> str: ...


@dataclass(frozen=True)
class _ArtifactSourceLocation:
    _backend: JsonlSessionStore

    @property
    def workspace(self) -> Path:
        return self._backend.workspace

    @property
    def sessions_dir(self) -> Path:
        return self._backend.sessions_dir

    @property
    def workspace_id(self) -> str:
        return self._backend.sessions_dir.name


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
        session.record_committed_base()
        return session


@dataclass(frozen=True, slots=True)
class PreparedSourcePersistence:
    """Detached serialization data, not storage or admission authority."""

    snapshot: SourceSnapshot
    text: str
    read_key: Callable[[], str]


@dataclass(frozen=True)
class SourceResetPlan:
    """Frozen deterministic target for one canonical reset transition."""

    expected: SourceState
    updated_at: str
    target_bytes: bytes
    target_state: SourceState


@dataclass(frozen=True)
class ResetSourceRead:
    """Parsed reset data and lazy read operations; not storage or execution authority."""

    events: list[dict[str, object]]
    has_runtime_checkpoint: Callable[[], bool]
    snapshot: Callable[[SourceSessionStore], SourceSnapshot]


def _source_bytes(header: dict[str, Any], events: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in (header, *events)
    ).encode("utf-8")


def _counter(value: object) -> int:
    if type(value) is not int or value < 0:
        raise SourceValidationError("Invalid source state counter")
    return value


def reject_duplicate_source_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Apply the existing duplicate-key policy to decoded source object pairs."""
    return _object_pairs(pairs)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceValidationError("Duplicate source object key")
        result[key] = value
    return result


def _validate_source_records(
    records: list[object],
    *,
    expected_key: str | None,
    require_source: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate decoded JSONL records through the canonical source grammar."""
    if not records or not isinstance(records[0], dict):
        raise SourceValidationError("Invalid source header")
    canonical_events(records)
    # Canonical JSON validation above rejects non-string keys recursively.
    header = cast(dict[str, Any], records[0])
    key = header.get("key")
    if (
        header.get("_type") != "metadata"
        or not isinstance(key, str)
        or not key
        or (expected_key is not None and key != expected_key)
    ):
        raise SourceValidationError("Invalid source identity")
    metadata = header.get("metadata", {})
    if not isinstance(metadata, dict):
        raise SourceValidationError("Invalid source metadata")
    metadata = cast(dict[str, Any], metadata)
    header["metadata"] = metadata
    schema = metadata.get("source_schema_version")
    source = require_source or "source_schema_version" in metadata
    if source and (type(schema) is not int or schema != 1):
        raise SourceValidationError("Unsupported source schema")
    if source:
        required = {"revision", "generation", "incarnation", "last_archived", "last_consolidated"}
        if not required.issubset(header):
            raise SourceValidationError("Missing source state")
        if header["last_archived"] != header["last_consolidated"]:
            raise SourceValidationError("Inconsistent source archive cursor")
        if "runtime_checkpoint" in metadata:
            raise SourceValidationError("Legacy runtime checkpoint in source session")
    _counter(header.get("revision", 0))
    _counter(header.get("generation", 0))
    incarnation = header.get("incarnation", "legacy")
    if not isinstance(incarnation, str) or not incarnation:
        raise SourceValidationError("Invalid source incarnation")
    for name in ("created_at", "updated_at"):
        if name in header:
            if not isinstance(header[name], str):
                raise SourceValidationError("Invalid source timestamp")
            datetime.fromisoformat(header[name])
        elif require_source:
            raise SourceValidationError("Missing source timestamp")
    events: list[dict[str, Any]] = []
    for raw_record in records[1:]:
        if not isinstance(raw_record, dict):
            raise SourceValidationError("Invalid source record")
        record = cast(dict[str, Any], raw_record)
        if record.get("_type") == "provider_state" and not require_source and schema is None:
            continue
        if (
            "_type" in record
            or not isinstance(record.get("role"), str)
            or record["role"] not in {"user", "assistant", "tool", "system", "developer"}
        ):
            raise SourceValidationError("Invalid source event")
        events.append(record)
    for name in ("last_archived", "last_consolidated"):
        if name in header and _counter(header[name]) > len(events):
            raise SourceValidationError("Invalid source archive boundary")
    if source:
        validate_event_ids(events)
        validate_execution_links(events)
    return header, events


def validate_source_session_file(
    path: Path,
    *,
    expected_key: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and validate one canonical v1 source file without mutating storage."""
    if not stat.S_ISREG(path.lstat().st_mode):
        raise SourceValidationError("Non-regular source file")
    with path.open(encoding="utf-8") as handle:
        records = [
            json.loads(line, object_pairs_hook=_object_pairs)
            for line in handle
            if line.strip()
        ]
    return _validate_source_records(
        records,
        expected_key=expected_key,
        require_source=True,
    )


def validate_source_session_bytes(
    data: bytes,
    *,
    expected_key: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Strict side-effect-free canonical-v1 byte parser."""

    try:
        text = data.decode("utf-8")
        if text.encode("utf-8") != data:
            raise SourceValidationError("Invalid source encoding")
        records = [
            json.loads(line, object_pairs_hook=_object_pairs)
            for line in text.splitlines()
            if line.strip()
        ]
        header, events = _validate_source_records(
            records,
            expected_key=expected_key,
            require_source=True,
        )
        if _source_bytes(header, events) != data:
            raise SourceValidationError("Invalid source bytes")
        return header, events
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        raise SourceValidationError("Invalid source bytes") from None


def _state(header: dict[str, Any], store_identity: str) -> SourceState:
    raw = header["metadata"].get("context_checkpoint")
    identity: object = None
    if isinstance(raw, dict):
        raw = cast(dict[object, object], raw)
        identity = raw.get("checkpoint_id")
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
    pending: dict[str, list[object]] = {}
    for event in upgraded:
        # Legacy metadata cannot retroactively authenticate a producing read.
        event.pop("_source_origin", None)
        event.pop("_artifact_read", None)
        if "_event_id" not in event:
            event["_event_id"] = uuid.uuid4().hex
        if "tool_calls" in event:
            calls = event["tool_calls"]
            if event.get("role") != "assistant" or not isinstance(calls, list):
                raise SourceValidationError("Invalid legacy execution linkage")
            calls = cast(list[object], calls)
            identities = event.get("_execution_ids")
            if identities is None:
                identities = [uuid.uuid4().hex for _ in calls]
                event["_execution_ids"] = identities
            if not isinstance(identities, list) or len(cast(list[object], identities)) != len(calls):
                raise SourceValidationError("Invalid legacy execution linkage")
            for call, identity in zip(calls, cast(list[object], identities)):
                if not isinstance(call, dict):
                    raise SourceValidationError("Invalid legacy provider call")
                provider_id = cast(dict[object, object], call).get("id")
                if not isinstance(provider_id, str) or not provider_id:
                    raise SourceValidationError("Invalid legacy provider call")
                if pending.get(provider_id):
                    raise SourceValidationError("Ambiguous legacy execution linkage")
                pending[provider_id] = [identity]
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


def select_legacy_source_event_upgrader() -> Callable[
    [list[dict[str, object]]], list[dict[str, object]]
]:
    """Select the current stateless converter without granting storage authority."""
    return _upgrade_events


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

    @staticmethod
    def publish_execution_lease(
        backend: JsonlSessionStore,
        snapshot: SourceSnapshot,
        metadata: dict[str, object],
        *,
        token: str,
        run_id: str,
        recovery_required: bool,
        select_acquired_at: Callable[[], str],
    ) -> SourceSnapshot:
        """Publish fixed lease fields under acquisition's existing execution/files locks.

        Keep the previously selected metadata and original late clock/write hooks.
        This operation neither issues an owner nor admits a namespace or transaction.
        """
        metadata["execution_lease"] = {
            "schema_version": 1, "token": token, "run_id": run_id,
            "incarnation": snapshot.state.incarnation, "generation": snapshot.state.generation,
            "acquired_at": select_acquired_at(),
            "recovery_required": recovery_required,
        }
        snapshot.header["revision"] += 1
        return backend.source._write(snapshot.header, snapshot.events)

    @staticmethod
    def delete_unowned_source(backend: JsonlSessionStore, key: str) -> None:
        """Delete an unowned source under backend.delete's existing guard/files locks."""
        from nanobot.session.execution_owner import ExecutionOwnershipError

        snapshot = backend.source.load_source_snapshot(key)
        if any(name in snapshot.header["metadata"] for name in ("execution_lease", "execution_recovery_required")):
            raise ExecutionOwnershipError()
        backend.source._delete_unlocked(key, snapshot.state)

    def execution_transaction(
        self,
        key: str,
        owner: ExecutionOwner | None = None,
        *,
        destructive: bool = False,
        allow_legacy: bool = False,
        root_authority: SourceRootAuthority | None = None,
    ) -> AbstractContextManager[None]:
        """Return the existing fenced transaction without acquiring its locks early."""
        from nanobot.session.execution_owner import execution_transaction

        return execution_transaction(
            self._backend, key, owner, destructive=destructive,
            allow_legacy=allow_legacy, root_authority=root_authority,
        )

    def with_artifact_store(self, artifact_store: PrivateArtifactStore) -> SourceSessionStore:
        """Create a separately validated artifact binding over this exact source store."""
        return SourceSessionStore(self._backend, artifact_store=artifact_store)

    def matches_backend(self, backend: JsonlSessionStore) -> bool:
        """Observe exact backend identity only; this does not admit an operation."""
        return backend is self._backend

    def select_artifact_location_metadata(self) -> ArtifactSourceLocation:
        """Retain this backend selection while observing only its live location values."""
        return _ArtifactSourceLocation(self._backend)

    def store_identity_differs(self, identity: str) -> bool:
        """Retain the original live store-identity inequality without returning it."""
        return self._store_identity != identity

    @property
    def artifact_workspace_id(self) -> str:
        """Observe live artifact binding metadata, without resolving or exposing storage."""
        return self._backend.sessions_dir.name

    def source_snapshot_identity_differs(self, snapshot: SourceSnapshot) -> bool:
        """Read the owned identity before the snapshot value, returning no authority."""
        return self._store_identity != snapshot.state.store_identity

    def resolved_runtime_source_root(self) -> str:
        """Return existing resolved-root metadata, never backend or root authority."""
        return str(self._backend.sessions_dir.resolve(strict=False))

    def require_current_root_authority(self, authority: SourceRootAuthority) -> None:
        """Reprove the actual authority against the current owned backend."""
        authority.require_current(self._backend)

    def matches_configured_artifact_store(self, store: PrivateArtifactStore | None) -> bool:
        """Observe exact store identity only; this does not admit an operation."""
        return store is self._artifact_store

    def require_idle_artifact_binding(
        self,
        expected_store: PrivateArtifactStore | None,
        expected_binding: tuple[str, str, str, int] | None,
    ) -> None:
        """Validate the current configured store and sealed metadata, returning no authority."""

        from nanobot.session.idle_checkpoint import IdleCheckpointError

        store = self._artifact_store
        store_binding = (
            None
            if store is None
            else (store.workspace_id, store.session_key, store.incarnation, store.generation)
        )
        if expected_store is not store or expected_binding != store_binding:
            raise IdleCheckpointError("Idle capture artifact store binding changed")

    def _state(self, header: dict[str, Any]) -> SourceState:
        return _state(header, self._store_identity)

    def _read_unlocked(
        self,
        key: str,
        *,
        require_source: bool = True,
        root_authority: SourceRootAuthority | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        from nanobot.session.manager import SessionStorageError

        path = self._backend.get_session_path(key)
        descriptor: int | None = None
        try:
            if root_authority is None:
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise SourceValidationError("Non-regular source file")
                handle = path.open(encoding="utf-8")
            else:
                root_authority.require_current(self._backend)
                descriptor = root_authority.open_regular(
                    f"{self._backend.storage_key(key)}.jsonl",
                    flags=os.O_RDONLY | getattr(os, "O_NONBLOCK", 0),
                )
                handle = os.fdopen(descriptor, "r", encoding="utf-8")
                descriptor = None
            with handle:
                records = [json.loads(line, object_pairs_hook=_object_pairs) for line in handle if line.strip()]
            return _validate_source_records(
                records,
                expected_key=key,
                require_source=require_source,
            )
        except FileNotFoundError:
            raise
        except (ValueError, TypeError, AttributeError, UnicodeError, RecursionError):
            raise SessionStorageError() from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _snapshot(self, header: dict[str, Any], events: list[dict[str, Any]]) -> SourceSnapshot:
        state = self._state(header)
        candidate = header["metadata"].get("context_checkpoint")
        checkpoint = None
        if candidate is not None:
            try:
                if not isinstance(candidate, dict):
                    raise SourceValidationError("Invalid stored checkpoint")
                # _read_unlocked validated the complete header as canonical JSON.
                candidate = cast(dict[str, Any], candidate)
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

    def assert_no_execution_metadata(
        self,
        key: str,
        *,
        allow_legacy: bool = False,
        root_authority: SourceRootAuthority | None = None,
    ) -> None:
        """Read-only admission check; caller retains execution and storage locks."""
        from nanobot.session.execution_owner import ExecutionOwnershipError
        from nanobot.session.manager import SessionWriteConflict

        if root_authority is not None:
            try:
                header, _ = self._read_unlocked(key, root_authority=root_authority)
            except FileNotFoundError:
                raise SessionWriteConflict() from None
            metadata = header["metadata"]
            if "execution_lease" in metadata or "execution_recovery_required" in metadata:
                raise ExecutionOwnershipError()
            return
        try:
            header, _ = self._read_unlocked(key, require_source=not allow_legacy)
        except FileNotFoundError:
            raise SessionWriteConflict() from None
        if "execution_lease" in header["metadata"] or "execution_recovery_required" in header["metadata"]:
            raise ExecutionOwnershipError()

    def load_source_snapshot(
        self,
        key: str,
        *,
        root_authority: SourceRootAuthority | None = None,
    ) -> SourceSnapshot:
        if root_authority is None:
            with self._backend.locked_session_files():
                return self._snapshot(*self._read_unlocked(key))
        with root_authority.operation(self._backend):
            with root_authority.lock_file(".session-files.lock", blocking=True):
                return self._snapshot(
                    *self._read_unlocked(key, root_authority=root_authority)
                )

    def read_reset_input(self, key: str) -> ResetSourceRead:
        """Hand off only data; the caller retains its existing execution/files locks."""
        header, events = self._read_unlocked(key)
        return ResetSourceRead(
            events=events,
            has_runtime_checkpoint=lambda: "runtime_checkpoint" in header["metadata"],
            snapshot=lambda source: source._snapshot(header, events),
        )

    def create_discovery_cursor(self, *, limits: SourceDiscoveryLimits) -> SourceDiscoveryCursor:
        """Create a bounded, read-only cursor over this exact source directory.

        The cursor constructor pins only directory identity; it performs no
        traversal.  Keeping this factory on the source owner prevents an idle
        integration from accidentally discovering through the preview/listing
        or legacy-repair paths.
        """

        from nanobot.session.source_discovery import SourceDiscoveryCursor
        authority = self.create_source_root_authority()
        try:
            return SourceDiscoveryCursor(
                self._backend,
                limits=limits,
                root_authority=authority,
                own_root_authority=True,
            )
        except BaseException:
            authority.close()
            raise

    def create_discovery_cursor_with_root(
        self,
        *,
        limits: SourceDiscoveryLimits,
        root_authority: SourceRootAuthority,
    ) -> SourceDiscoveryCursor:
        """Create the existing read-only cursor while borrowing its caller-owned root."""
        from nanobot.session.source_discovery import SourceDiscoveryCursor

        return SourceDiscoveryCursor(
            self._backend,
            limits=limits,
            root_authority=root_authority,
        )

    discovery_cursor = create_discovery_cursor

    def create_source_root_authority(self) -> SourceRootAuthority:
        from nanobot.session.source_root import SourceRootAuthority

        return SourceRootAuthority.issue(self._backend)

    source_root_authority = create_source_root_authority

    def _runtime_checkpoint_present(
        self,
        key: str,
        *,
        root_authority: SourceRootAuthority | None = None,
    ) -> bool:
        path = self._backend.get_runtime_checkpoint_path(key)
        if root_authority is not None:
            root_authority.require_current(self._backend)
            value = root_authority.stat_relative(path.name)
            return value is not None
        return path.exists() or path.is_symlink()

    def capture_idle_compaction(self, key: str) -> IdleCaptureToken | None:
        """Capture dormant idle state without taking an execution lock.

        Callers compute any summary after this method returns.  A later
        integration-owned publisher must revalidate the returned token while
        retaining its existing checkpoint transaction.
        """

        with self._backend.locked_session_files():
            header, events = self._read_unlocked(key)
            return capture_idle_token(
                self._snapshot(header, events),
                runtime_checkpoint_present=self._runtime_checkpoint_present(key),
            )

    def require_idle_root_authority(
        self,
        root_authority: SourceRootAuthority | None,
    ) -> None:
        from nanobot.session import idle_checkpoint as idle_checkpoint_module

        if root_authority is None:
            return
        if type(root_authority) is not idle_checkpoint_module.SourceRootAuthority:
            raise idle_checkpoint_module.IdleCheckpointError("Invalid idle source-root authority")
        try:
            root_authority.require_current(self._backend)
        except idle_checkpoint_module.SourceRootAuthorityError:
            raise idle_checkpoint_module.IdleCheckpointConflict("Idle source-root authority changed") from None

    def idle_root_descriptors(
        self,
        snapshot: SourceSnapshot,
    ) -> tuple[ArtifactDescriptor, ...]:
        """Return only current manifest descriptors from root-history events."""

        from nanobot.session import idle_checkpoint as idle_checkpoint_module

        store = self._artifact_store
        if store is None:
            return ()
        if (
            store.workspace_id != self._backend.sessions_dir.name
            or store.session_key != snapshot.state.session_key
            or store.incarnation != snapshot.state.incarnation
            or store.generation != snapshot.state.generation
            or snapshot.state.store_identity != self._store_identity
        ):
            raise idle_checkpoint_module.ArtifactValidationError("Idle artifact store binding changed")
        descriptors = idle_checkpoint_module.published_artifacts(snapshot, workspace_id=store.workspace_id)
        root: list[ArtifactDescriptor] = []
        for descriptor in descriptors:
            event = next((row for row in snapshot.events if row.get("_event_id") == descriptor.event_id), None)
            if (
                event is not None
                and event.get("_parent_stream_id") is None
                and descriptor.stream_id == event.get("_stream_id")
            ):
                root.append(idle_checkpoint_module.parse_artifact_descriptor(descriptor.to_dict()))
        return tuple(root)

    def issue_idle_capture_artifacts(
        self, issuer: IdleCaptureIssuer, snapshot: SourceSnapshot, token: IdleCaptureToken,
        detached_snapshot: SourceSnapshot, nonce: object, *, runtime_fingerprint: str,
        policy_fingerprint: str, policy: IdlePolicyBinding | None,
        root_authority: SourceRootAuthority | None,
    ) -> IdleCheckpointCapture:
        """Keep configured-store reads here while the sealed issuer owns construction."""
        from nanobot.session import idle_checkpoint as idle_checkpoint_module

        factory = idle_checkpoint_module.IdleCaptureIssuer
        factory.require_source(issuer, self)
        dependencies = None
        if self._artifact_store is not None and policy is not None:
            dependency_snapshot = idle_checkpoint_module.copy.deepcopy(detached_snapshot)
            descriptors = factory.root_descriptors(self, dependency_snapshot)
            dependencies = factory.dependency_constructor(issuer, self, nonce, token)(
                self._artifact_store,
                self._artifact_store.workspace_id,
                snapshot.state.session_key,
                snapshot.state.incarnation,
                snapshot.state.generation,
                policy.fingerprint,
                dependency_snapshot,
                factory.snapshot_digest(dependency_snapshot),
                descriptors,
            )
        capture = factory.make_capture(
            issuer, self, token, detached_snapshot, runtime_fingerprint, policy_fingerprint,
            nonce, dependencies, root_authority,
        )
        factory.attestation_constructor(issuer, self, capture, nonce, token)(
            self._artifact_store,
            (
                None
                if self._artifact_store is None
                else (
                    self._artifact_store.workspace_id,
                    self._artifact_store.session_key,
                    self._artifact_store.incarnation,
                    self._artifact_store.generation,
                )
            ),
            detached_snapshot, dependencies,
        )
        return capture

    def captured_idle_root_descriptors(
        self, capture: IdleCheckpointCapture,
    ) -> tuple[ArtifactDescriptor, ...]:
        """Inspect captured descriptor data without lending the configured artifact store."""
        from nanobot.session import idle_checkpoint as idle_checkpoint_module

        store = self._artifact_store
        if store is None:
            metadata = capture.snapshot.header.get("metadata")
            if not isinstance(metadata, dict) or "artifact_manifest" not in metadata:
                return ()
            manifest = cast(dict[object, object], metadata)["artifact_manifest"]
            if not isinstance(manifest, dict):
                raise idle_checkpoint_module.ArtifactValidationError("Invalid artifact manifest schema")
            checked_manifest = cast(dict[object, object], manifest)
            if (
                set(checked_manifest) != {"schema_version", "descriptors"}
                or type(checked_manifest["schema_version"]) is not int
                or checked_manifest["schema_version"] != 1
                or not isinstance(checked_manifest["descriptors"], dict)
            ):
                raise idle_checkpoint_module.ArtifactValidationError("Invalid artifact manifest schema")
            if checked_manifest["descriptors"]:
                raise idle_checkpoint_module.ArtifactValidationError("Idle artifact store binding is unavailable")
            return ()
        return idle_checkpoint_module.select_idle_root_descriptor_reader()(self, capture.snapshot)

    def idle_capture_identity_differs(self, token: IdleCaptureToken) -> bool:
        """Compare the original token-then-source identity operands; not admission."""
        return token.store_identity != self._store_identity

    def capture_idle_checkpoint(
        self,
        key: str,
        *,
        runtime_fingerprint: str,
        policy_fingerprint: str,
        policy: IdlePolicyBinding | None = None,
        root_authority: SourceRootAuthority | None = None,
    ) -> IdleCheckpointCapture | None:
        """Capture the dormant integration-owned idle checkpoint substrate.

        This is an explicit, unwired seam. The integration module owns the
        opaque capability and detached snapshot; the ordinary foreground store
        path continues to use :meth:`capture_idle_compaction` unchanged.
        """

        from nanobot.session import idle_checkpoint as idle_checkpoint_module

        runtime_fingerprint, policy_fingerprint = (
            idle_checkpoint_module.IdleCheckpointCapture.admit_source_capture(
                self,
                runtime_fingerprint=runtime_fingerprint,
                policy_fingerprint=policy_fingerprint,
                policy=policy,
                root_authority=root_authority,
            )
        )
        backend = self._backend
        if root_authority is None:
            with backend.locked_session_files():
                header, events = self._read_unlocked(key)
                snapshot = self._snapshot(header, events)
                return idle_checkpoint_module.IdleCheckpointCapture.issue_from_snapshot(
                    self,
                    key,
                    snapshot,
                    runtime_fingerprint=runtime_fingerprint,
                    policy_fingerprint=policy_fingerprint,
                    policy=policy,
                    runtime_checkpoint_present=self._runtime_checkpoint_present(key),
                    root_authority=None,
                )
        with root_authority.operation(backend):
            with root_authority.lock_file(".session-files.lock", blocking=True):
                header, events = self._read_unlocked(key, root_authority=root_authority)
                snapshot = self._snapshot(header, events)
                return idle_checkpoint_module.IdleCheckpointCapture.issue_from_snapshot(
                    self,
                    key,
                    snapshot,
                    runtime_fingerprint=runtime_fingerprint,
                    policy_fingerprint=policy_fingerprint,
                    policy=policy,
                    runtime_checkpoint_present=self._runtime_checkpoint_present(
                        key, root_authority=root_authority
                    ),
                    root_authority=root_authority,
                )

    def publish_idle_checkpoint(
        self,
        key: str,
        capture: IdleCheckpointCapture,
        candidate: dict[str, Any],
        *,
        runtime_fingerprint: str,
        policy_fingerprint: str,
        policy: IdlePolicyBinding | None = None,
        proof: SummaryDependencyProof | None = None,
        root_authority: SourceRootAuthority | None = None,
    ) -> SourceSnapshot:
        """Attempt one dormant idle checkpoint publication without a foreground owner.

        This source owner acquires artifact storage before the nonblocking
        execution/session transaction and performs exact revalidation. Capture
        and dependency owners retain their sealed evidence operations. No
        foreground ExecutionOwner is created.
        """

        from nanobot.session import idle_checkpoint as idle_checkpoint_module

        validated, references, detached_proof = (
            idle_checkpoint_module.IdleCheckpointCapture.prepare_source_publication(
                self,
                key,
                capture,
                candidate,
                runtime_fingerprint=runtime_fingerprint,
                policy_fingerprint=policy_fingerprint,
                policy=policy,
                proof=proof,
                root_authority=root_authority,
            )
        )
        if references:
            access = capture.dependencies
            if (
                access is None
                or policy is None
                or not idle_checkpoint_module.IdleDependencyAccess.matches_configured_store(
                    access, self._artifact_store
                )
            ):
                raise idle_checkpoint_module.IdleCheckpointError("Idle dependency access is unavailable")
            try:
                authority_context = (
                    root_authority.operation(self._backend)
                    if root_authority is not None
                    else idle_checkpoint_module.nullcontext()
                )
                with authority_context:
                    return idle_checkpoint_module.IdleDependencyAccess.publish_in_artifact_transaction(
                        access,
                        self,
                        key,
                        capture,
                        validated,
                        references,
                        detached_proof,
                        policy=policy,
                        root_authority=root_authority,
                    )
            except idle_checkpoint_module.ArtifactStorageBusyError:
                raise idle_checkpoint_module.ExecutionBusyError() from None
        authority_context = (
            root_authority.operation(self._backend)
            if root_authority is not None
            else idle_checkpoint_module.nullcontext()
        )
        with authority_context:
            return SourceSessionStore.publish_idle_checkpoint_in_execution_transaction(
                self,
                key,
                capture,
                validated,
                references,
                detached_proof,
                policy=policy,
                blobs=None,
                root_authority=root_authority,
            )

    def publish_idle_checkpoint_in_execution_transaction(
        self,
        key: str,
        capture: IdleCheckpointCapture,
        validated: dict[str, object],
        references: tuple[str, ...],
        proof: SummaryDependencyProof | None,
        *,
        policy: IdlePolicyBinding | None,
        blobs: ArtifactBlobTransaction | None,
        root_authority: SourceRootAuthority | None,
    ) -> SourceSnapshot:
        """Run one fixed idle commit under the nonblocking execution/source transaction."""

        from nanobot.session import idle_checkpoint as idle_checkpoint_module

        with idle_checkpoint_module.execution_transaction(
            self._backend, key, root_authority=root_authority
        ):
            return SourceSessionStore.commit_idle_checkpoint_unlocked(
                self,
                key,
                capture,
                validated,
                references,
                proof,
                policy=policy,
                blobs=blobs,
                root_authority=root_authority,
            )

    def commit_idle_checkpoint_unlocked(
        self,
        key: str,
        capture: IdleCheckpointCapture,
        validated: dict[str, object],
        references: tuple[str, ...],
        proof: SummaryDependencyProof | None,
        *,
        policy: IdlePolicyBinding | None,
        blobs: ArtifactBlobTransaction | None,
        root_authority: SourceRootAuthority | None,
    ) -> SourceSnapshot:
        """Revalidate and write once while the caller retains the required transactions."""

        from nanobot.session import idle_checkpoint as idle_checkpoint_module

        header, events = self._read_unlocked(key, root_authority=root_authority)
        snapshot = self._snapshot(header, events)
        try:
            idle_checkpoint_module.validate_idle_capture(
                capture.token,
                snapshot,
                runtime_checkpoint_present=self._runtime_checkpoint_present(
                    key, root_authority=root_authority
                ),
            )
        except idle_checkpoint_module.IdleCaptureConflictError:
            raise idle_checkpoint_module.IdleCheckpointConflict("Idle source state changed during publication") from None
        # Re-run exact structural validation against the lock-held source rather
        # than trusting the detached copy used while the model computed.
        validated_current = idle_checkpoint_module.validate_context_checkpoint(
            validated,
            events,
            generation=snapshot.state.generation,
            source_revision=snapshot.state.revision,
            parent_checkpoint_id=snapshot.state.checkpoint_id,
        )
        if validated_current != validated:
            raise idle_checkpoint_module.IdleCheckpointConflict("Idle checkpoint proposal changed during publication")
        if proof is not None:
            if proof.candidate.snapshot != snapshot:
                raise idle_checkpoint_module.IdleCheckpointConflict("Idle dependency proof source changed during publication")
            if proof.candidate.stream_id is not None:
                raise idle_checkpoint_module.IdleCheckpointError("Idle checkpoints require root-history proof")
            if (
                not proof.chunks
                or proof.chunks[-1].evidence.boundary != validated_current["boundary"]
                or proof.chunks[-1].summary != validated_current["summary"]
            ):
                raise idle_checkpoint_module.IdleCheckpointConflict("Idle dependency proof changed during publication")
            try:
                locked_references = idle_checkpoint_module.derive_summary_references(proof, snapshot)
            except idle_checkpoint_module.SourceValidationError:
                raise idle_checkpoint_module.IdleCheckpointConflict("Idle dependency proof changed during publication") from None
            if locked_references != references:
                raise idle_checkpoint_module.IdleCheckpointConflict("Idle dependency proof references changed during publication")
        if references:
            idle_checkpoint_module.IdleDependencyAccess.verify_captured_references(
                capture, references, snapshot, proof, blobs, policy
            )
        header["metadata"]["context_checkpoint"] = validated_current
        header["revision"] += 1
        return self._write(header, events, root_authority=root_authority)

    def record_user_activity(self, key: str, expected: SourceState, *,
                             accepted_foreground_user: bool,
                             owner: ExecutionOwner | None = None) -> SourceSnapshot:
        """Atomically record one already accepted foreground user admission.

        The integration owner must invoke this immediately after it admits a
        foreground user turn and before it begins that turn's execution.  When
        a turn already owns the session, pass that exact ``owner`` so this
        update joins its existing execution transaction rather than opening a
        second ownerless transaction.  This dormant store primitive
        deliberately does not infer admission from a source event, so
        assistant/tool persistence cannot advance activity.
        """

        from nanobot.session.execution_owner import execution_transaction

        if accepted_foreground_user is not True:
            raise SourceValidationError("Foreground user admission is required")
        if owner is not None:
            owner.validate_dispatch()
        with execution_transaction(self._backend, key, owner):
            header, events = self._current_unlocked(key, expected)
            activity = read_user_activity(header["metadata"])
            header["metadata"][USER_ACTIVITY_REVISION_KEY] = (
                (activity.revision if activity is not None else 0) + 1
            )
            header["metadata"][LAST_USER_ACTIVITY_AT_KEY] = datetime.now(timezone.utc).isoformat()
            header["revision"] += 1
            return self._write(header, events)

    def validate_idle_checkpoint_admission(self, key: str, token: IdleCaptureToken,
                                           candidate: dict[str, Any]) -> IdleCheckpointAdmission:
        """Validate a prospective idle checkpoint without mutating storage.

        Dependency-bearing schema-v2 checkpoints intentionally fail closed.
        Their existing proof publisher requires an active execution owner,
        while idle admission requires none.  The integration owner must expose
        a publisher that reuses that proof path and revalidates this token in
        the same source transaction before idle checkpoint writes can exist.
        """

        from nanobot.session.manager import SessionWriteConflict

        with self._backend.locked_session_files():
            try:
                header, events = self._read_unlocked(key)
                snapshot = self._snapshot(header, events)
                validate_idle_capture(
                    token,
                    snapshot,
                    runtime_checkpoint_present=self._runtime_checkpoint_present(key),
                )
            except IdleCaptureConflictError:
                raise SessionWriteConflict() from None
            validated = validate_context_checkpoint(
                candidate,
                events,
                generation=snapshot.state.generation,
                source_revision=snapshot.state.revision,
                parent_checkpoint_id=snapshot.state.checkpoint_id,
            )
            if validated["artifact_references"]:
                raise SourceValidationError(
                    "Idle checkpoint requires the integration-owned dependency publisher"
                )
            return IdleCheckpointAdmission(
                capture=token,
                candidate_fingerprint=checkpoint_candidate_fingerprint(validated),
            )

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

    @contextmanager
    def rematerialization_snapshot(
        self,
        key: str,
        expected: SourceState,
        *,
        owner: ExecutionOwner,
    ) -> Generator[SourceSnapshot, None, None]:
        """Yield one detached exact snapshot while execution/source locks remain held.

        This dormant read-only seam deliberately requires the caller's captured
        state.  It never refreshes that state and has no write path.
        """
        from nanobot.session.execution_owner import execution_transaction

        with execution_transaction(self._backend, key, owner):
            header, events = self._current_unlocked(key, expected)
            yield self._snapshot(header, events)

    def prepare_source_persistence(
        self,
        header: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> PreparedSourcePersistence:
        """Run the existing source validation and serialization without selecting a writer."""
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
        text = _source_bytes(header, events).decode("utf-8")
        return PreparedSourcePersistence(result, text, lambda: header["key"])

    def _write(
        self,
        header: dict[str, object],
        events: list[dict[str, object]],
        *,
        root_authority: SourceRootAuthority | None = None,
    ) -> SourceSnapshot:
        from nanobot.session.manager import JsonlSessionStore

        return JsonlSessionStore.persist_source_records(
            self, header, events, root_authority=root_authority,
            observe_backend=lambda: self._backend,
        )

    @staticmethod
    def settle_execution_close(backend: JsonlSessionStore, owner: ExecutionOwner) -> None:
        """Settle one real drained owner; never accept arbitrary source records."""
        from nanobot.session import execution_owner

        with execution_owner.ExecutionOwner.closing_transaction(owner, backend) as snapshot:
            header = snapshot.header
            retained_lease = execution_owner.ExecutionOwner.copy_execution_lease(
                header["metadata"]["execution_lease"]
            )
            if owner.recovery_required or execution_owner.validate_execution_links(snapshot.events):
                header["metadata"]["execution_recovery_required"] = True
            header["metadata"].pop("execution_lease", None)
            header["revision"] += 1
            try:
                backend.source._write(header, snapshot.events)
            except BaseException as failure:
                # A replacement may have happened before fsync failed. Keep
                # abandoned-owner evidence even in that complete-new-file case.
                try:
                    current = backend.source.load_source_snapshot(owner.key)
                    if execution_owner.ExecutionOwner.matches_current_execution_lifetime(owner, current):
                        retained_lease["recovery_required"] = True
                        current.header["metadata"]["execution_lease"] = retained_lease
                        current.header["metadata"]["execution_recovery_required"] = True
                        current.header["revision"] += 1
                        backend.source._write(current.header, current.events)
                except BaseException:
                    failure.add_note("Persisting execution recovery metadata also failed.")
                raise

    def upgrade_source_session(self, key: str, *, expected_revision: int, expected_generation: int, expected_incarnation: str) -> SourceSnapshot:
        from nanobot.session.execution_owner import execution_transaction
        from nanobot.session.manager import SessionWriteConflict

        self._backend.require_unmanaged_namespace()
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

        # Nonempty dependencies need a separate attested publication path.
        # Structural version-2 loading alone must not enable generic writes.
        if not isinstance(cast(object, candidate), dict) or candidate.get('artifact_references') != []:
            raise SourceValidationError('Checkpoint artifact publication requires dependency validation')
        with execution_transaction(self._backend, key, owner):
            header, events = self._current_unlocked(key, expected)
            validated = validate_context_checkpoint(
                candidate, events, generation=expected.generation,
                source_revision=expected.revision, parent_checkpoint_id=expected.checkpoint_id,
            )
            header["metadata"]["context_checkpoint"] = validated
            header["revision"] += 1
            return self._write(header, events)

    def commit_dependent_context_checkpoint(self, key: str, expected: SourceState,
                                            candidate: dict[str, Any], proof: SummaryDependencyProof, *,
                                            owner: ExecutionOwner, reader: ArtifactReader,
                                            blobs: ArtifactBlobTransaction) -> SourceSnapshot:
        """Atomically publish one fully attested, nonempty schema-v2 checkpoint.

        Callers must already retain the exact configured artifact transaction.
        Taking the execution/session transaction only after that transaction
        preserves the global blob-before-source lock order and intentionally
        leaves CAS conflicts to the caller without reloading or retrying.
        """
        from nanobot.session.artifact_reader import ArtifactReader
        from nanobot.session.artifact_store import ArtifactBlobTransaction
        from nanobot.session.checkpoint_dependencies import (
            select_summary_dependency_metadata_validator,
            validate_summary_dependencies,
        )
        validate_proof_metadata = select_summary_dependency_metadata_validator()
        from nanobot.session.execution_owner import (
            ExecutionOwner,
            ExecutionOwnershipError,
            execution_transaction,
        )

        writer_store = reader.writer.store
        if (type(owner) is not ExecutionOwner or type(reader) is not ArtifactReader
                or type(blobs) is not ArtifactBlobTransaction
                or writer_store._backend is not self._backend
                or writer_store._store_identity != self._store_identity
                or reader.writer.owner is not owner or self._artifact_store is None
                or reader.store is not self._artifact_store or not blobs.matches_store(self._artifact_store)
                or owner.key != key):
            raise ExecutionOwnershipError()
        # Reject before taking a source lock when the caller did not retain an
        # active configured artifact transaction. `assert_reachable` also
        # checks its private lock and namespace identities.
        blobs.assert_reachable()
        owner.validate_dispatch()
        with execution_transaction(self._backend, key, owner):
            header, events = self._current_unlocked(key, expected)
            snapshot = self._snapshot(header, events)
            verified = validate_proof_metadata(proof, snapshot)
            if (not isinstance(cast(object, candidate), dict) or candidate.get('schema_version') != 2
                    or not verified.references or candidate.get('artifact_references') != list(verified.references)
                    or candidate.get('boundary') != verified.boundary
                    or candidate.get('summary') != verified.summary):
                raise SourceValidationError('Dependent checkpoint does not exactly match its proof')
            validated = validate_context_checkpoint(
                candidate, events, generation=expected.generation,
                source_revision=expected.revision, parent_checkpoint_id=expected.checkpoint_id,
            )
            references = validate_summary_dependencies(proof, snapshot, reader, blobs)
            if tuple(validated['artifact_references']) != references:
                raise SourceValidationError('Dependent checkpoint references changed during validation')
            blobs.assert_reachable()
            owner.validate_dispatch()
            header['metadata']['context_checkpoint'] = validated
            header['revision'] += 1
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

        if cast(object, owner) is None:
            raise ExecutionOwnershipError()
        if self._artifact_store is None or not isinstance(cast(object, blobs), ArtifactBlobTransaction):
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

    def plan_source_reset(
        self,
        snapshot: SourceSnapshot,
        replacement_events: list[dict[str, Any]],
        *,
        updated_at: str,
    ) -> SourceResetPlan:
        """Build and validate the exact reset output without writing storage."""

        from nanobot.session.busy_inbox import require_busy_inbox_disposition

        require_busy_inbox_disposition(snapshot.header["metadata"])
        try:
            parsed_time = datetime.fromisoformat(updated_at)
        except (TypeError, ValueError):
            raise SourceValidationError("Invalid reset timestamp") from None
        if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
            raise SourceValidationError("Invalid reset timestamp")
        header = deepcopy(snapshot.header)
        events = deepcopy(replacement_events)
        header["revision"] = snapshot.state.revision + 1
        header["generation"] = snapshot.state.generation + 1
        header["last_archived"] = header["last_consolidated"] = 0
        header["updated_at"] = updated_at
        for name in {
            "context_checkpoint", "artifact_manifest", "_last_summary", "execution_lease",
            "execution_recovery_required", USER_ACTIVITY_REVISION_KEY,
            LAST_USER_ACTIVITY_AT_KEY, SOURCE_PENDING_FOLLOWUPS_METADATA_KEY,
            *LEGACY_RECOVERY_KEYS,
        }:
            header["metadata"].pop(name, None)
        records: list[object] = [header, *events]
        validated_header, validated_events = _validate_source_records(
            records,
            expected_key=snapshot.state.session_key,
            require_source=True,
        )
        target = self._snapshot(validated_header, validated_events)
        return SourceResetPlan(
            expected=snapshot.state,
            updated_at=updated_at,
            target_bytes=_source_bytes(validated_header, validated_events),
            target_state=target.state,
        )

    def reset_source_session(
        self,
        key: str,
        expected: SourceState,
        replacement_events: list[dict[str, Any]],
        *,
        owner: ExecutionOwner | None = None,
        plan: SourceResetPlan | None = None,
    ) -> SourceSnapshot:
        from nanobot.session.execution_owner import execution_transaction

        if plan is not None:
            if not isinstance(cast(object, plan), SourceResetPlan) or plan.expected != expected:
                raise SourceValidationError("Invalid reset plan")
            try:
                planned_header, planned_events = validate_source_session_bytes(
                    plan.target_bytes,
                    expected_key=key,
                )
            except SourceValidationError:
                raise SourceValidationError("Invalid reset plan") from None
            planned_snapshot = self._snapshot(planned_header, planned_events)
            if (
                planned_snapshot.state != plan.target_state
                or planned_events != replacement_events
                or planned_header.get("updated_at") != plan.updated_at
            ):
                raise SourceValidationError("Invalid reset plan")
        with execution_transaction(self._backend, key, owner, destructive=True):
            return self._reset_source_session_unlocked(
                key, expected, replacement_events, owner=owner, plan=plan
            )

    @staticmethod
    def install_prepared_mode_reset(
        backend: JsonlSessionStore,
        key: str,
        replacement_events: list[dict[str, object]],
        *,
        plan: SourceResetPlan,
        owner: ExecutionOwner | None,
    ) -> SourceSnapshot:
        """Install the planned mode transition under the caller's existing locks."""
        return backend.source._reset_source_session_unlocked(
            key, plan.expected, replacement_events, owner=owner, plan=plan,
        )

    def _reset_source_session_unlocked(
        self,
        key: str,
        expected: SourceState,
        replacement_events: list[dict[str, Any]],
        *,
        owner: ExecutionOwner | None = None,
        plan: SourceResetPlan | None = None,
    ) -> SourceSnapshot:
        """Apply an exact reset while the caller retains execution and files locks."""

        header, events = self._current_unlocked(key, expected)
        snapshot = self._snapshot(header, events)
        frozen = plan or self.plan_source_reset(
            snapshot, replacement_events, updated_at=datetime.now(timezone.utc).isoformat()
        )
        exact = self.plan_source_reset(
            snapshot,
            replacement_events,
            updated_at=frozen.updated_at,
        )
        if frozen != exact:
            raise SourceValidationError("Invalid reset plan")
        target_header, target_events = validate_source_session_bytes(
            frozen.target_bytes,
            expected_key=key,
        )
        if self._snapshot(target_header, target_events).state != frozen.target_state:
            raise SourceValidationError("Invalid reset plan")
        result = self._write(target_header, target_events)
        if owner is not None:
            from nanobot.session.execution_owner import ExecutionOwner

            ExecutionOwner.retire_after_destructive_write(owner)
        return result

    def commit_busy_inbox_metadata(
        self, key: str, expected: object, replacement: dict[str, object],
    ) -> SourceSnapshot:
        """Commit only inbox metadata, preserving the active execution/source."""
        with self._backend.locked_session_files():
            header, events = self._read_unlocked(key)
            self._check(header, self._state(header))
            if header["metadata"].get(BUSY_INBOX_METADATA_KEY) != expected:
                raise SourceValidationError("busy inbox changed before commit")
            header["metadata"][BUSY_INBOX_METADATA_KEY] = deepcopy(replacement)
            header["revision"] += 1
            return self._write(header, events)

    def admit_pending_followup(
        self,
        key: str,
        owner: ExecutionOwner,
        record: PendingFollowupRecord,
    ) -> bool:
        """Durably admit one server-identified follow-up under its live owner."""
        from nanobot.session.execution_owner import execution_transaction

        if type(record) is not PendingFollowupRecord:
            raise SourceValidationError("Invalid pending follow-up record")
        with execution_transaction(self._backend, key, owner):
            header, events = self._read_unlocked(key)
            records = parse_pending_followup_records(
                header["metadata"].get(SOURCE_PENDING_FOLLOWUPS_METADATA_KEY, [])
            )
            existing = next((item for item in records if item.identifier == record.identifier), None)
            if existing is not None:
                if existing != record:
                    raise SourceValidationError("Conflicting pending follow-up retry")
                if any(event.get("_pending_followup_id") == record.identifier for event in events):
                    raise SourceValidationError("Pending follow-up journal conflicts with source event")
                return True
            committed = [
                event for event in events
                if event.get("_pending_followup_id") == record.identifier
            ]
            if committed:
                if len(committed) != 1 or not self._pending_followup_event_matches(committed[0], record):
                    raise SourceValidationError("Conflicting consumed follow-up retry")
                return False
            records.append(record)
            header["metadata"][SOURCE_PENDING_FOLLOWUPS_METADATA_KEY] = [
                item.to_dict() for item in records
            ]
            header["revision"] += 1
            header["updated_at"] = datetime.now().isoformat()
            self._write(header, events)
            return True

    def consume_pending_followup(
        self,
        key: str,
        owner: ExecutionOwner,
        record: PendingFollowupRecord,
        event: dict[str, object],
    ) -> tuple[SourceSnapshot, dict[str, object]]:
        """Append the original input and remove its journal row in one replacement."""
        from nanobot.session.execution_owner import execution_transaction

        if (
            type(record) is not PendingFollowupRecord
            or not self._pending_followup_event_matches(event, record)
            or not isinstance(event.get("_event_id"), str)
        ):
            raise SourceValidationError("Invalid pending follow-up source event")
        with execution_transaction(self._backend, key, owner):
            header, events = self._read_unlocked(key)
            records = parse_pending_followup_records(
                header["metadata"].get(SOURCE_PENDING_FOLLOWUPS_METADATA_KEY, [])
            )
            stored = next((item for item in records if item.identifier == record.identifier), None)
            committed = [
                item for item in events
                if item.get("_pending_followup_id") == record.identifier
            ]
            if committed:
                if (
                    len(committed) != 1
                    or not self._pending_followup_event_matches(committed[0], record)
                    or stored is not None
                ):
                    raise SourceValidationError("Conflicting consumed follow-up source event")
                return self._snapshot(header, events), dict(committed[0])
            if stored != record:
                raise SourceValidationError("Pending follow-up admission is missing")
            event_id = event.get("_event_id")
            if any(item.get("_event_id") == event_id for item in events):
                raise SourceValidationError("Duplicate pending follow-up source event")
            remaining = [item for item in records if item.identifier != record.identifier]
            metadata = header["metadata"]
            if remaining:
                metadata[SOURCE_PENDING_FOLLOWUPS_METADATA_KEY] = [
                    item.to_dict() for item in remaining
                ]
            else:
                metadata.pop(SOURCE_PENDING_FOLLOWUPS_METADATA_KEY, None)
            header["revision"] += 1
            header["updated_at"] = datetime.now().isoformat()
            snapshot = self._write(header, [*events, event])
            return snapshot, dict(event)

    @staticmethod
    def _pending_followup_event_matches(
        event: dict[str, object],
        record: PendingFollowupRecord,
    ) -> bool:
        return (
            event.get("_pending_followup_id") == record.identifier
            and event.get("role") == "user"
            and event.get("content") == record.content
            and event.get("media", []) == list(record.media)
        )

    def discard_pending_followups(
        self,
        key: str,
        owner: ExecutionOwner,
    ) -> tuple[str, ...]:
        """Explicitly discard pending rows before `/stop` releases the owner."""
        from nanobot.session.execution_owner import execution_transaction

        with execution_transaction(self._backend, key, owner):
            header, events = self._read_unlocked(key)
            records = parse_pending_followup_records(
                header["metadata"].get(SOURCE_PENDING_FOLLOWUPS_METADATA_KEY, [])
            )
            if not records:
                return ()
            del header["metadata"][SOURCE_PENDING_FOLLOWUPS_METADATA_KEY]
            header["revision"] += 1
            header["updated_at"] = datetime.now().isoformat()
            self._write(header, events)
            return tuple(record.identifier for record in records)

    def update_metadata(
        self,
        key: str,
        updates: dict[str, Any],
        *,
        removals: tuple[str, ...] = (),
    ) -> SourceSnapshot:
        if isinstance(removals, (str, bytes)) or any(not isinstance(cast(object, name), str) for name in removals):
            raise SourceValidationError("Invalid metadata removals")
        with self._backend.locked_session_files():
            header, events = self._read_unlocked(key)
            self._check(header, self._state(header))
            header["metadata"].update(accepted_metadata_updates(updates, source=True))
            reserved = set(LEGACY_RECOVERY_KEYS) | {
                BUSY_INBOX_METADATA_KEY, SOURCE_PENDING_FOLLOWUPS_METADATA_KEY,
                "context_checkpoint", "source_schema_version", "execution_lease",
                "execution_recovery_required", "revision", "generation", "incarnation",
                "artifact_manifest", USER_ACTIVITY_REVISION_KEY, LAST_USER_ACTIVITY_AT_KEY,
            }
            for name in removals:
                if name not in reserved:
                    header["metadata"].pop(name, None)
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
        from nanobot.session.busy_inbox import require_busy_inbox_disposition

        header, _events = self._current_unlocked(key, expected)
        require_busy_inbox_disposition(header["metadata"])
        self._backend.get_session_path(key).unlink()
        from nanobot.session.manager import JsonlSessionStore

        JsonlSessionStore.settle_source_deletion(owner, lambda: self._backend)
