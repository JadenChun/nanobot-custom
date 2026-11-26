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

from kohaku.security.workspace_access import WORKSPACE_SCOPE_METADATA_KEY
from kohaku.session.canonical_event_index import (
    delete_event_index,
    update_event_index_header,
)
from kohaku.session.canonical_event_index import (
    read_canonical_events_by_ids as _read_canonical_events_by_ids,
)
from kohaku.session.context_checkpoint import validate_context_checkpoint
from kohaku.session.harness_binding import validate_harness_update
from kohaku.session.input_attempt import (
    InputAttempt,
    InputAttemptConflictError,
    InputAttemptSource,
    find_input_attempt,
    fingerprint,
)
from kohaku.session.next_turn_preferences import (
    NextTurnPreferences,
    consume_preferences,
    read_preferences,
    remove_preferences,
    update_preferences,
)
from kohaku.session.pending_followups import (
    PendingFollowupRecord,
    parse_pending_followup_records,
)
from kohaku.session.persistence_metadata import (
    DURABLE_METADATA_KEYS,
    LEGACY_RECOVERY_KEYS,
    SOURCE_PENDING_FOLLOWUPS_METADATA_KEY,
    accepted_metadata_updates,
)
from kohaku.session.source import (
    SourceValidationError,
    canonical_events,
    merge_source_events,
    validate_event_ids,
    validate_execution_links,
)
from kohaku.workspace_access_recovery import (
    WorkspaceAccessRecord,
    WorkspaceAccessSourceBinding,
    WorkspaceScopeMetadata,
    parse_workspace_access_record,
    parse_workspace_access_source_binding,
    parse_workspace_scope_metadata,
)

MAX_SOURCE_SERIALIZED_BYTES = 67_108_864

if TYPE_CHECKING:
    from kohaku.session.artifact import ArtifactDescriptor
    from kohaku.session.artifact_reader import ArtifactReader
    from kohaku.session.artifact_store import ArtifactBlobTransaction, PrivateArtifactStore
    from kohaku.session.checkpoint_dependencies import SummaryDependencyProof
    from kohaku.session.execution_owner import ExecutionOwner
    from kohaku.session.manager import JsonlSessionStore, Session
    from kohaku.session.mode_storage import PrivateFileVersion
    from kohaku.session.source_discovery import SourceDiscoveryCursor, SourceDiscoveryLimits
    from kohaku.session.source_root import SourceRootAuthority


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

    def same_origin(self, other: SourceState) -> bool:
        """Revisions/checkpoints advance within one source lifetime."""
        return (self.store_identity, self.session_key, self.incarnation, self.generation) == (
            other.store_identity, other.session_key, other.incarnation, other.generation,
        )


@dataclass(frozen=True, slots=True)
class SavedSourceMetadata:
    """Detached browse facts. These neither authorize execution nor contain events."""

    state: SourceState
    webui: bool
    workspace_scope: object
    workspace_scope_present: bool


@dataclass(frozen=True, slots=True)
class SavedSourceProof:
    version: PrivateFileVersion
    metadata: SavedSourceMetadata
    digest: str | None = None
    header: dict[str, Any] | None = None
    event_count: int = 0
    body_offset: int = 0


class SourceOriginMismatchError(RuntimeError):
    """Delayed work no longer targets its original canonical source."""

    def __init__(self) -> None:
        super().__init__("The original session source is no longer available")


@dataclass(frozen=True)
class SourceSnapshot:
    """Detached committed data. Mutating its dictionaries cannot change storage."""

    state: SourceState
    header: dict[str, Any]
    events: list[dict[str, Any]]
    checkpoint: dict[str, Any] | None

    def to_session(self) -> Session:
        from kohaku.session.manager import Session

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


def _invalidate_workspace_access_request(metadata: dict[str, Any]) -> bool:
    """Consume a pending decision when its task focus or input changes."""
    raw = metadata.get("workspace_access_request")
    if raw is None:
        return False
    try:
        record = parse_workspace_access_record(raw)
    except (TypeError, ValueError):
        return False
    if record["status"] not in {"pending", "ready", "continuing"}:
        return False
    metadata["workspace_access_request"] = parse_workspace_access_record({
        **record,
        "revision": record["revision"] + 1,
        "status": "invalidated",
    })
    return True


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


def _validate_source_header(
    header: dict[str, Any],
    *,
    expected_key: str | None,
    require_source: bool,
) -> bool:
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
    return source


def _validate_archive_boundary(header: dict[str, Any], event_count: int) -> None:
    for name in ("last_archived", "last_consolidated"):
        if name in header and _counter(header[name]) > event_count:
            raise SourceValidationError("Invalid source archive boundary")


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
    source = _validate_source_header(header, expected_key=expected_key, require_source=require_source)
    schema = header["metadata"].get("source_schema_version")
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
    _validate_archive_boundary(header, len(events))
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
        # The writer delimits records with LF. Other Unicode line separators
        # are legal message content and must not split a JSON record.
        records = [
            json.loads(line, object_pairs_hook=_object_pairs)
            for line in text.split("\n")
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
        event.pop("_input_attempt", None)
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
        from kohaku.session.execution_owner import ExecutionOwnershipError

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
        from kohaku.session.execution_owner import execution_transaction

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

    def shares_store_with(self, other: SourceSessionStore) -> bool:
        """Observe one exact backend behind two validated store bindings."""
        return type(other) is SourceSessionStore and other._backend is self._backend

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

    def _state(self, header: dict[str, Any]) -> SourceState:
        return _state(header, self._store_identity)

    def describe_saved_source(
        self, header: dict[str, Any], *, event_count: int, expected_key: str,
    ) -> SavedSourceMetadata:
        """Project only the validated header facts needed by saved-history readers."""
        _validate_source_header(header, expected_key=expected_key, require_source=True)
        _validate_archive_boundary(header, event_count)
        metadata = header["metadata"]
        return SavedSourceMetadata(
            self._state(header), metadata.get("webui") is True,
            deepcopy(cast(object, metadata.get(WORKSPACE_SCOPE_METADATA_KEY))),
            WORKSPACE_SCOPE_METADATA_KEY in metadata,
        )

    def _read_unlocked(
        self,
        key: str,
        *,
        require_source: bool = True,
        root_authority: SourceRootAuthority | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        from kohaku.session.manager import SessionStorageError

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
        from kohaku.session.execution_owner import ExecutionOwnershipError
        from kohaku.session.manager import SessionWriteConflict

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
            from kohaku.session.mode_storage import private_file_version

            with self._backend.locked_session_files():
                path = self._backend.get_session_path(key)
                has_checkpoint = self._backend.get_runtime_checkpoint_path(key).exists()
                current_version: Any = None
                if not has_checkpoint:
                    try:
                        current_version = private_file_version(path.lstat())
                    except FileNotFoundError:
                        current_version = None
                    if current_version is not None:
                        cached = self._backend.cached_source_snapshot(key, current_version)
                        if cached is not None:
                            return cached
                snapshot = self._snapshot(*self._read_unlocked(key))
                if not has_checkpoint and current_version is not None:
                    self._backend.remember_source_snapshot(key, current_version, snapshot)
                return snapshot
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

        from kohaku.session.source_discovery import SourceDiscoveryCursor
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
        from kohaku.session.source_discovery import SourceDiscoveryCursor

        return SourceDiscoveryCursor(
            self._backend,
            limits=limits,
            root_authority=root_authority,
        )

    discovery_cursor = create_discovery_cursor

    def create_source_root_authority(self) -> SourceRootAuthority:
        from kohaku.session.source_root import SourceRootAuthority

        return SourceRootAuthority.issue(self._backend)

    source_root_authority = create_source_root_authority

    def _check(self, header: dict[str, Any], expected: SourceState) -> None:
        from kohaku.session.manager import SessionWriteConflict

        if (
            type(expected) is not SourceState
            or type(expected.revision) is not int or type(expected.generation) is not int
            or self._state(header) != expected
        ):
            raise SessionWriteConflict()
        if self._backend.get_runtime_checkpoint_path(header["key"]).exists():
            raise SessionWriteConflict()

    def _current_unlocked(self, key: str, expected: SourceState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        from kohaku.session.manager import SessionWriteConflict

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
        from kohaku.session.execution_owner import execution_transaction

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
        raw = _source_bytes(header, events)
        if len(raw) > MAX_SOURCE_SERIALIZED_BYTES:
            raise SourceValidationError(
                "source exceeds the maximum readable serialized size"
            )
        text = raw.decode("utf-8")
        return PreparedSourcePersistence(result, text, lambda: header["key"])

    def _write(
        self,
        header: dict[str, object],
        events: list[dict[str, object]],
        *,
        root_authority: SourceRootAuthority | None = None,
    ) -> SourceSnapshot:
        from kohaku.session.manager import JsonlSessionStore

        return JsonlSessionStore.persist_source_records(
            self, header, events, root_authority=root_authority,
            observe_backend=lambda: self._backend,
        )

    @staticmethod
    def settle_execution_close(backend: JsonlSessionStore, owner: ExecutionOwner) -> None:
        """Settle one real drained owner; never accept arbitrary source records."""
        from kohaku.session import execution_owner

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
        from kohaku.session.execution_owner import execution_transaction
        from kohaku.session.manager import SessionWriteConflict

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
        from kohaku.session.execution_owner import execution_transaction

        with execution_transaction(self._backend, key, owner):
            header, committed = self._current_unlocked(key, expected)
            merged = merge_source_events(committed, events)
            # Validate the entire batch inside the atomic file/CAS transaction.
            for event in events:
                if "_input_attempt" not in event:
                    continue
                attempt = InputAttempt.parse_evidence(event["_input_attempt"])
                content = event.get("content")
                if (event.get("role") != "user" or attempt.source != InputAttemptSource.from_state(expected)
                        or not isinstance(content, str) or fingerprint(content) != attempt.fingerprint):
                    raise InputAttemptConflictError("Input attempt source or payload changed")
                prior = find_input_attempt(merged, attempt)
                if prior is not None and (
                    prior.event_id != event.get("_event_id") or prior.run_id != event.get("_run_id")
                ):
                    raise InputAttemptConflictError("Input attempt already committed")
            if merged == committed:
                return self._snapshot(header, committed)
            if any(event.get("role") == "user" for event in events):
                _invalidate_workspace_access_request(header["metadata"])
            # The result and its recovery fence are one atomic source replacement.
            # A matching call/result link does not settle an unresolved effect.
            if any("_filesystem_effects" in event for event in events) and validate_execution_links(merged):
                header["metadata"]["execution_recovery_required"] = True
            header["revision"] += 1
            header["updated_at"] = datetime.now().isoformat()
            return self._write(header, merged)

    def invalidate_workspace_access_request(self, key: str) -> bool:
        """Consume any pending decision when canonical user input is accepted."""
        with self._backend.locked_session_files():
            header_state = self.read_source_header_state(key)
            if header_state is not None:
                _, check_header = header_state
                if not _invalidate_workspace_access_request(deepcopy(check_header["metadata"])):
                    return False
            header, events = self._read_unlocked(key)
            self._check(header, self._state(header))
            if not _invalidate_workspace_access_request(header["metadata"]):
                return False
            header["revision"] += 1
            header["updated_at"] = datetime.now().isoformat()
            self._write(header, events)
            return True

    def commit_context_checkpoint(self, key: str, expected: SourceState, candidate: dict[str, Any], *, owner: ExecutionOwner | None = None) -> SourceSnapshot:
        from kohaku.session.execution_owner import execution_transaction

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
        from kohaku.session.artifact_reader import ArtifactReader
        from kohaku.session.artifact_store import ArtifactBlobTransaction
        from kohaku.session.checkpoint_dependencies import (
            select_summary_dependency_metadata_validator,
            validate_summary_dependencies,
        )
        validate_proof_metadata = select_summary_dependency_metadata_validator()
        from kohaku.session.execution_owner import (
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
        from kohaku.session.artifact import (
            bound_artifact_text,
            parse_artifact_descriptor,
            published_artifacts,
        )
        from kohaku.session.artifact_store import ArtifactBlobTransaction
        from kohaku.session.execution_owner import ExecutionOwnershipError, execution_transaction

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

        from kohaku.session.legacy_h1 import require_legacy_h1_disposition

        require_legacy_h1_disposition(snapshot.header["metadata"])
        try:
            parsed_time = datetime.fromisoformat(updated_at)
        except (TypeError, ValueError):
            raise SourceValidationError("Invalid reset timestamp") from None
        if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
            raise SourceValidationError("Invalid reset timestamp")
        header = deepcopy(snapshot.header)
        events = deepcopy(replacement_events)
        for event in events:
            event.pop("_input_attempt", None)
        header["revision"] = snapshot.state.revision + 1
        header["generation"] = snapshot.state.generation + 1
        header["last_archived"] = header["last_consolidated"] = 0
        header["updated_at"] = updated_at
        for name in {
            "context_checkpoint", "artifact_manifest", "_last_summary", "execution_lease",
            "execution_recovery_required", "user_activity_revision",
            "last_user_activity_at", SOURCE_PENDING_FOLLOWUPS_METADATA_KEY,
            "workspace_access_request", *LEGACY_RECOVERY_KEYS,
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
        from kohaku.session.execution_owner import execution_transaction

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
                or planned_events != [{key: value for key, value in event.items()
                                      if key != "_input_attempt"} for event in replacement_events]
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
        remove_preferences(self._backend, key)
        if owner is not None:
            from kohaku.session.execution_owner import ExecutionOwner

            ExecutionOwner.retire_after_destructive_write(owner)
        return result

    def commit_workspace_access_request(
        self,
        key: str,
        *,
        expected_binding: WorkspaceAccessSourceBinding,
        expected_record: WorkspaceAccessRecord | None,
        replacement_record: WorkspaceAccessRecord | None,
        expected_latest_input_event_id: str | None = None,
        expected_workspace_scope: WorkspaceScopeMetadata | None = None,
        replacement_workspace_scope: WorkspaceScopeMetadata | None = None,
    ) -> SourceSnapshot:
        """CAS one reserved recovery field without touching transcript events."""
        from kohaku.session.manager import SessionWriteConflict

        try:
            expected_binding = parse_workspace_access_source_binding(expected_binding)
        except (TypeError, ValueError):
            raise SourceValidationError("Invalid workspace access source binding") from None
        if expected_record is not None:
            expected_record = parse_workspace_access_record(expected_record)
        if replacement_record is not None:
            replacement_record = parse_workspace_access_record(replacement_record)
        try:
            if expected_workspace_scope is not None:
                expected_workspace_scope = parse_workspace_scope_metadata(expected_workspace_scope)
            if replacement_workspace_scope is not None:
                replacement_workspace_scope = parse_workspace_scope_metadata(
                    replacement_workspace_scope
                )
        except (TypeError, ValueError):
            raise SourceValidationError("Invalid workspace access focus") from None
        if (expected_workspace_scope is None) != (replacement_workspace_scope is None):
            # None/record pairs are meaningful only when an explicit replacement
            # exists; deleting focus is never part of request resolution.
            if replacement_workspace_scope is None:
                raise SourceValidationError("Workspace access focus cannot be removed")

        with self._backend.locked_session_files():
            header, events = self._read_unlocked(key)
            self._check(header, self._state(header))
            actual = self._state(header)
            binding = {
                "store_identity": actual.store_identity,
                "incarnation": actual.incarnation,
                "generation": actual.generation,
            }
            if binding != expected_binding:
                raise SessionWriteConflict()
            if expected_latest_input_event_id is not None:
                latest_user = next(
                    (event for event in reversed(events) if event.get("role") == "user"),
                    None,
                )
                if (
                    latest_user is None
                    or latest_user.get("_event_id") != expected_latest_input_event_id
                ):
                    raise SessionWriteConflict()
            raw_current = header["metadata"].get("workspace_access_request")
            if raw_current is None:
                current: WorkspaceAccessRecord | None = None
            else:
                try:
                    current = parse_workspace_access_record(raw_current)
                except (TypeError, ValueError):
                    raise SourceValidationError("Invalid workspace access record") from None
            if current != expected_record:
                raise SessionWriteConflict()
            if replacement_workspace_scope is not None:
                if header["metadata"].get("workspace_scope") != expected_workspace_scope:
                    raise SessionWriteConflict()
                header["metadata"]["workspace_scope"] = replacement_workspace_scope
            if replacement_record is None:
                header["metadata"].pop("workspace_access_request", None)
            else:
                header["metadata"]["workspace_access_request"] = replacement_record
            header["revision"] += 1
            return self._write(header, events)

    def admit_pending_followup(
        self,
        key: str,
        owner: ExecutionOwner,
        record: PendingFollowupRecord,
    ) -> bool:
        """Durably admit one server-identified follow-up under its live owner."""
        from kohaku.session.execution_owner import execution_transaction

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
            _invalidate_workspace_access_request(header["metadata"])
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
        from kohaku.session.execution_owner import execution_transaction

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
            and event.get("_child_followup") == (
                record.child_followup.to_dict() if record.child_followup is not None else None
            )
            and event.get("_background_result") == (
                record.background_result.to_dict()
                if record.background_result is not None
                else None
            )
        )

    def discard_pending_followups(
        self,
        key: str,
        owner: ExecutionOwner,
        *,
        expected_records: tuple[PendingFollowupRecord, ...] | None = None,
    ) -> tuple[str, ...]:
        """Discard all rows or exact observed rows before the owner is released."""
        from kohaku.session.execution_owner import execution_transaction

        with execution_transaction(self._backend, key, owner):
            header, events = self._read_unlocked(key)
            records = parse_pending_followup_records(
                header["metadata"].get(SOURCE_PENDING_FOLLOWUPS_METADATA_KEY, [])
            )
            if expected_records is None:
                selected_ids = {record.identifier for record in records}
            else:
                expected_by_id = {record.identifier: record for record in expected_records}
                if len(expected_by_id) != len(expected_records):
                    raise SourceValidationError("Duplicate pending follow-up discard identity")
                stored_by_id = {record.identifier: record for record in records}
                if any(stored_by_id.get(identifier) != expected
                       for identifier, expected in expected_by_id.items()):
                    raise SourceValidationError("Pending follow-up changed before discard")
                selected_ids = set(expected_by_id)
            discarded = tuple(
                record.identifier for record in records if record.identifier in selected_ids
            )
            if not discarded:
                return ()
            remaining = [record for record in records if record.identifier not in selected_ids]
            if remaining:
                header["metadata"][SOURCE_PENDING_FOLLOWUPS_METADATA_KEY] = [
                    record.to_dict() for record in remaining
                ]
            else:
                del header["metadata"][SOURCE_PENDING_FOLLOWUPS_METADATA_KEY]
            header["revision"] += 1
            header["updated_at"] = datetime.now().isoformat()
            self._write(header, events)
            return discarded

    def read_source_header_state(
        self, key: str,
    ) -> tuple[SourceState, dict[str, Any]] | None:
        """Read validated canonical state and header without scanning settled events."""
        from kohaku.session.mode_storage import private_file_version

        with self._backend.locked_session_files():
            path = self._backend.get_session_path(key)
            proof = self._backend.saved_source_proof(key)
            if (
                proof is not None
                and proof.header is not None
                and not self._backend.get_runtime_checkpoint_path(key).exists()
            ):
                try:
                    current_version = private_file_version(path.lstat())
                except FileNotFoundError:
                    return None
                if current_version is not None and current_version == proof.version:
                    return proof.metadata.state, deepcopy(proof.header)
            try:
                snapshot = self.load_source_snapshot(key)
            except FileNotFoundError:
                return None
            return snapshot.state, deepcopy(snapshot.header)

    def _try_update_metadata_header_unlocked(
        self,
        key: str,
        updates: dict[str, Any],
        *,
        removals: tuple[str, ...] = (),
    ) -> SourceSnapshot | None:
        """Update canonical header metadata in O(1) when proof of the file is current."""
        from kohaku.session.manager import SessionStorageError
        from kohaku.session.mode_storage import MODE_JSON_LIMIT, private_file_version

        if (
            type(self) is not SourceSessionStore
            or "_write" in self.__dict__
            or "_write_text_atomic" in self._backend.__dict__
        ):
            return None
        proof = self._backend.saved_source_proof(key)
        if (
            proof is None
            or proof.header is None
            or proof.body_offset <= 0
            or self._backend.get_runtime_checkpoint_path(key).exists()
        ):
            return None
        path = self._backend.get_session_path(key)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            src_fd = os.open(path, flags)
        except OSError:
            return None
        try:
            current_version = private_file_version(os.fstat(src_fd))
            if current_version is None or current_version != proof.version:
                return None
            header = deepcopy(proof.header)
            self._check(header, self._state(header))
            accepted = accepted_metadata_updates(updates, source=True)
            metadata = header["metadata"]
            validate_harness_update(metadata, accepted, removals)
            previous_scope = metadata.get("workspace_scope")
            scope_removed = "workspace_scope" in removals and "workspace_scope" in metadata
            scope_changed = (
                "workspace_scope" in accepted
                and accepted["workspace_scope"] != previous_scope
            ) or scope_removed
            metadata.update(accepted)
            reserved = set(DURABLE_METADATA_KEYS) | set(LEGACY_RECOVERY_KEYS)
            for name in removals:
                if name not in reserved:
                    metadata.pop(name, None)
            if scope_changed:
                _invalidate_workspace_access_request(metadata)
            header["revision"] += 1
            _validate_source_header(header, expected_key=key, require_source=True)
            _validate_archive_boundary(header, proof.event_count)
            canonical_events([header])
            try:
                header_line = (
                    json.dumps(header, ensure_ascii=False, allow_nan=False) + "\n"
                ).encode("utf-8")
            except (TypeError, ValueError, UnicodeError):
                raise SessionStorageError() from None
            body_size = proof.version.size - proof.body_offset
            if body_size < 0 or len(header_line) + body_size > MAX_SOURCE_SERIALIZED_BYTES:
                raise SourceValidationError(
                    "source exceeds the maximum readable serialized size"
                )
            installed = self._backend.replace_header_atomic(
                path,
                header_line,
                src_fd=src_fd,
                expected_version=proof.version,
                body_offset=proof.body_offset,
            )
        finally:
            os.close(src_fd)
        state = self._state(header)
        if installed is None or len(header_line) - 1 > MODE_JSON_LIMIT:
            self._backend.forget_saved_source(key)
        else:
            try:
                saved_meta = self.describe_saved_source(
                    header, event_count=proof.event_count, expected_key=key,
                )
            except ValueError:
                self._backend.forget_saved_source(key)
            else:
                self._backend.remember_saved_source(
                    key,
                    SavedSourceProof(
                        installed,
                        saved_meta,
                        header=deepcopy(header),
                        event_count=proof.event_count,
                        body_offset=len(header_line),
                    ),
                )
                prev_cached = self._backend.cached_source_snapshot(key, proof.version)
                if prev_cached is not None:
                    raw_cp = header["metadata"].get("context_checkpoint")
                    cp: dict[str, Any] | None = (
                        deepcopy(cast(dict[str, Any], raw_cp))
                        if isinstance(raw_cp, dict)
                        else None
                    )
                    self._backend.remember_source_snapshot(
                        key,
                        installed,
                        SourceSnapshot(state, deepcopy(header), prev_cached.events, cp),
                    )
                update_event_index_header(
                    self._backend,
                    key,
                    state=state,
                    old_version=proof.version,
                    new_version=installed,
                    new_body_offset=len(header_line),
                )
        raw_checkpoint = header["metadata"].get("context_checkpoint")
        checkpoint: dict[str, Any] | None = (
            deepcopy(cast(dict[str, Any], raw_checkpoint))
            if isinstance(raw_checkpoint, dict)
            else None
        )
        return SourceSnapshot(state, deepcopy(header), [], checkpoint)

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
            fast = self._try_update_metadata_header_unlocked(key, updates, removals=removals)
            if fast is not None:
                return fast
            header, events = self._read_unlocked(key)
            self._check(header, self._state(header))
            accepted = accepted_metadata_updates(updates, source=True)
            metadata = header["metadata"]
            validate_harness_update(metadata, accepted, removals)
            previous_scope = metadata.get("workspace_scope")
            scope_removed = "workspace_scope" in removals and "workspace_scope" in metadata
            scope_changed = (
                "workspace_scope" in accepted
                and accepted["workspace_scope"] != previous_scope
            ) or scope_removed
            metadata.update(accepted)
            reserved = set(DURABLE_METADATA_KEYS) | set(LEGACY_RECOVERY_KEYS)
            for name in removals:
                if name not in reserved:
                    metadata.pop(name, None)
            if scope_changed:
                _invalidate_workspace_access_request(metadata)
            header["revision"] += 1
            return self._write(header, events)

    def read_canonical_events_by_ids(
        self,
        key: str,
        event_ids: list[str] | tuple[str, ...],
        *,
        expected_source: SourceState,
    ) -> dict[str, dict[str, Any]]:
        """Bounded ``os.pread`` lookup of specific canonical events; 0 full-body loads."""
        with self._backend.locked_session_files():
            return _read_canonical_events_by_ids(
                self._backend,
                key,
                event_ids,
                expected_source=expected_source,
            )

    def read_next_turn_preferences(self, key: str) -> NextTurnPreferences | None:
        """Bounded read of the next-turn record; no event or body access."""
        with self._backend.locked_session_files():
            return read_preferences(self._backend, key)

    def update_next_turn_preferences(
        self,
        key: str,
        updates: dict[str, Any],
        *,
        removals: tuple[str, ...] = (),
        expected_revision: int | None = None,
    ) -> NextTurnPreferences | None:
        """Atomic, fsynced O(1) write; never rewrites header, events or revision."""
        with self._backend.locked_session_files():
            return update_preferences(
                self._backend, key, updates, removals=removals,
                expected_revision=expected_revision,
            )

    def consume_next_turn_preferences(
        self, key: str, consumed: dict[str, Any],
    ) -> NextTurnPreferences | None:
        with self._backend.locked_session_files():
            return consume_preferences(self._backend, key, consumed)

    def advance_consolidated(self, key: str, expected: SourceState, end_offset: int, *, owner: ExecutionOwner | None = None) -> SourceSnapshot:
        from kohaku.session.execution_owner import execution_transaction

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
        from kohaku.session.execution_owner import execution_transaction

        with execution_transaction(self._backend, key, owner, destructive=True):
            self._delete_unlocked(key, expected, owner=owner)

    def _delete_unlocked(self, key: str, expected: SourceState, *, owner: ExecutionOwner | None = None) -> None:
        from kohaku.session.legacy_h1 import require_legacy_h1_disposition
        from kohaku.session.next_turn_preferences import preference_path

        header, _events = self._current_unlocked(key, expected)
        require_legacy_h1_disposition(header["metadata"])
        pref = preference_path(self._backend, key)
        self._backend.get_session_path(key).unlink()
        from kohaku.session.manager import JsonlSessionStore

        JsonlSessionStore.settle_source_deletion(owner, lambda: self._backend)
        pref.unlink(missing_ok=True)
        delete_event_index(self._backend, key)
