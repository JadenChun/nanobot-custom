"""Sealed invocation-local grants for direct artifact stream handoff.

Artifacts remain source-bound evidence, not capabilities.  This module keeps the
one narrow capability needed by foreground delegation out of source metadata and
out of serializable tool inputs.  Every use is revalidated by ``ArtifactReader``
while it holds the private-blob transaction followed by the execution/source
transaction.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, NoReturn, SupportsIndex, cast

from nanobot.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
    parse_artifact_reference,
)

if TYPE_CHECKING:
    from nanobot.agent.child_sessions import ChildAdmission
    from nanobot.agent.run_store import RunStore
    from nanobot.session.artifact_reader import ArtifactReader
    from nanobot.session.artifact_store import ArtifactBlobTransaction
    from nanobot.session.source_store import SourceSessionStore, SourceSnapshot, SourceState
    from nanobot.session.turn_writer import SourceTurnWriter

_MAX_ARTIFACT_REFERENCES = 16
_CAPABILITY = object()
_CHILD_RELATION_VOLATILE_FIELDS = frozenset({
    "state",
    "state_sequence",
    "holds_capacity",
    "pending_approval_id",
    "pending_effect_id",
    "pending_summary",
    "blocked_run_id",
    "outcome",
    "transcript",
    "updated_at",
})


def _child_relation_signature(relation: Mapping[str, object]) -> str:
    """Bind stable child identity fields while allowing the terminal transition."""
    stable_fields = {
        key: value
        for key, value in relation.items()
        if key not in _CHILD_RELATION_VOLATILE_FIELDS
    }
    try:
        return json.dumps(
            cast(object, stable_fields),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise ArtifactValidationError("Artifact child relation is invalid") from None


def _parse_grant_references(references: object) -> tuple[str, ...]:
    if not isinstance(references, list) or len(cast(list[object], references)) > _MAX_ARTIFACT_REFERENCES:
        raise ArtifactValidationError("Invalid artifact reference grant set")
    parsed = tuple(parse_artifact_reference(value) for value in cast(list[object], references))
    if len(set(parsed)) != len(parsed):
        raise ArtifactValidationError("Artifact references must be unique")
    return parsed


@dataclass(frozen=True, slots=True)
class CanonicalChildArtifactInput:
    """Trusted, in-memory artifact input for one awaited canonical child turn."""

    admission: ChildAdmission = field(repr=False, compare=False)
    parent_reader: ArtifactReader = field(repr=False, compare=False)
    references: tuple[str, ...]

    def __post_init__(self) -> None:
        from nanobot.agent.child_sessions import ChildAdmission as RuntimeChildAdmission
        from nanobot.session.artifact_reader import ArtifactReader as RuntimeArtifactReader

        if (
            type(self.admission) is not RuntimeChildAdmission
            or type(self.parent_reader) is not RuntimeArtifactReader
            or self.admission.policy_facts.get("background") is not False
        ):
            raise ArtifactValidationError("Canonical child artifact input requires awaited execution")
        raw_references = cast(object, self.references)
        if not isinstance(raw_references, tuple):
            raise ArtifactValidationError("Invalid canonical child artifact references")
        references = _parse_grant_references(list(cast(tuple[object, ...], raw_references)))
        if references != raw_references:
            raise ArtifactValidationError("Invalid canonical child artifact references")

    def __reduce__(self) -> NoReturn:
        raise TypeError("Canonical child artifact input cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        _ = protocol
        raise TypeError("Canonical child artifact input cannot be serialized")


@dataclass(frozen=True, slots=True)
class ArtifactDirectChildBinding:
    """Private relation between one admitted parent invocation and canonical child run."""

    _parent_reader: ArtifactReader
    _child_writer: SourceTurnWriter
    _parent_state: SourceState
    _child_state: SourceState
    _child_id: str
    _parent_run_id: str
    _parent_stream_id: str
    _child_run_id: str
    _child_stream_id: str
    _child_relation_signature: str
    _nonce: bytes

    def __init__(
        self,
        capability: object,
        parent_reader: ArtifactReader,
        child_writer: SourceTurnWriter,
        parent_state: SourceState,
        child_state: SourceState,
        child_id: str,
        parent_run_id: str,
        parent_stream_id: str,
        child_run_id: str,
        child_stream_id: str,
        child_relation_signature: str,
        nonce: bytes,
    ) -> None:
        if capability is not _CAPABILITY:
            raise TypeError("Artifact child bindings are runtime-owned")
        object.__setattr__(self, "_parent_reader", parent_reader)
        object.__setattr__(self, "_child_writer", child_writer)
        object.__setattr__(self, "_parent_state", parent_state)
        object.__setattr__(self, "_child_state", child_state)
        object.__setattr__(self, "_child_id", child_id)
        object.__setattr__(self, "_parent_run_id", parent_run_id)
        object.__setattr__(self, "_parent_stream_id", parent_stream_id)
        object.__setattr__(self, "_child_run_id", child_run_id)
        object.__setattr__(self, "_child_stream_id", child_stream_id)
        object.__setattr__(self, "_child_relation_signature", child_relation_signature)
        object.__setattr__(self, "_nonce", nonce)

    @property
    def parent_reader(self) -> ArtifactReader:
        return self._parent_reader

    @property
    def child_writer(self) -> SourceTurnWriter:
        return self._child_writer

    @property
    def parent_source_state(self) -> SourceState:
        return self._parent_state

    @property
    def child_source_state(self) -> SourceState:
        return self._child_state

    @property
    def child_id(self) -> str:
        return self._child_id

    @property
    def parent_run_id(self) -> str:
        return self._parent_run_id

    @property
    def parent_stream_id(self) -> str:
        return self._parent_stream_id

    @property
    def child_run_id(self) -> str:
        return self._child_run_id

    @property
    def child_stream_id(self) -> str:
        return self._child_stream_id

    def matches_direct_child_nonce(self, nonce: bytes) -> bool:
        """Compare the runtime token without exposing the private invocation nonce."""
        return nonce is self._nonce

    def nonce_for_grant(self, capability: object) -> bytes:
        """Provide the private nonce only to this module's grant issuer."""
        if capability is not _CAPABILITY:
            raise ArtifactValidationError("Artifact child invocation binding changed")
        return self._nonce

    def matches_child_relation(self, relation: Mapping[str, object]) -> bool:
        """Check the captured durable child identity after allowed state changes."""
        try:
            return _child_relation_signature(relation) == self._child_relation_signature
        except ArtifactValidationError:
            return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("Artifact child bindings cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        _ = protocol
        raise TypeError("Artifact child bindings cannot be serialized")

    @classmethod
    def capture(
        cls,
        admission: ChildAdmission,
        parent_reader: ArtifactReader,
        child_writer: SourceTurnWriter,
    ) -> ArtifactDirectChildBinding:
        """Capture a durable direct-child admission against its live source owners."""
        from nanobot.agent.child_sessions import DELEGATION_CHILD_METADATA_KEY
        from nanobot.agent.child_sessions import ChildAdmission as RuntimeChildAdmission
        from nanobot.session.runtime_authority import parse_runtime_session_metadata

        if type(admission) is not RuntimeChildAdmission:
            raise ArtifactValidationError("Artifact grants require a canonical child admission")
        parent_writer = parent_reader.writer
        if (
            parent_writer.owner is child_writer.owner
            or child_writer.parent_stream_id is not None
            or not admission.live
            or admission.state != "active"
            or not admission.holds_capacity
            or admission.policy_facts.get("background") is not False
            or admission.session_key != child_writer.owner.key
            or admission.parent_key != parent_writer.owner.key
            or admission.current_run_id != child_writer.run_id
            or not admission.child_id
            or parent_writer.failed
            or child_writer.failed
        ):
            raise ArtifactValidationError("Artifact grant is not bound to a live direct child")

        parent_writer.owner.validate_dispatch()
        child_writer.owner.validate_dispatch()
        parent_snapshot = parent_writer.owner.read_snapshot()
        child_snapshot = child_writer.owner.read_snapshot()
        parent_state = parent_snapshot.state
        child_state = child_snapshot.state
        admitted_parent = admission.parent_source_state
        admitted_child = admission.source_state
        if (
            admitted_parent is None
            or admitted_child is None
            or admission.parent_incarnation != parent_state.incarnation
            or not admitted_parent.same_origin(parent_state)
            or not admitted_child.same_origin(child_state)
            or parent_state.session_key != parent_writer.owner.key
            or child_state.session_key != child_writer.owner.key
            or parent_state.store_identity != child_state.store_identity
            or not _writer_lease_matches(parent_writer, parent_snapshot)
            or not _writer_lease_matches(child_writer, child_snapshot)
        ):
            raise ArtifactValidationError("Artifact child source admission changed")

        metadata = child_snapshot.header.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ArtifactValidationError("Artifact child relation is unavailable")
        raw_metadata = cast(Mapping[object, object], metadata)
        if any(not isinstance(key, str) for key in raw_metadata):
            raise ArtifactValidationError("Artifact child relation is unavailable")
        metadata_record = cast(Mapping[str, object], metadata)
        child_relation = metadata_record.get(DELEGATION_CHILD_METADATA_KEY)
        if not isinstance(child_relation, Mapping):
            raise ArtifactValidationError("Artifact child relation is unavailable")
        raw_child_relation = cast(Mapping[object, object], child_relation)
        if any(not isinstance(key, str) for key in raw_child_relation):
            raise ArtifactValidationError("Artifact child relation is unavailable")
        child_relation_record = cast(Mapping[str, object], child_relation)
        if dict(child_relation_record) != admission.metadata():
            raise ArtifactValidationError("Artifact child relation is stale")
        runtime = parse_runtime_session_metadata(
            metadata_record,
            parent_incarnation=parent_state.incarnation,
        )
        if (
            not runtime.present
            or not runtime.executable
            or runtime.metadata.parent_key != admission.parent_key
            or runtime.metadata.parent_incarnation != parent_state.incarnation
            or runtime.metadata.profile != admission.profile
        ):
            raise ArtifactValidationError("Artifact child relation is stale")

        return cls(
            _CAPABILITY,
            parent_reader,
            child_writer,
            parent_state,
            child_state,
            admission.child_id,
            parent_writer.run_id,
            parent_writer.stream_id,
            child_writer.run_id,
            child_writer.stream_id,
            _child_relation_signature(child_relation_record),
            secrets.token_bytes(32),
        )

    def binds(
        self,
        parent_reader: ArtifactReader,
        child_writer: SourceTurnWriter,
    ) -> bool:
        return (
            self._parent_reader is parent_reader
            and self._child_writer is child_writer
            and parent_reader.writer.run_id == self._parent_run_id
            and parent_reader.writer.stream_id == self._parent_stream_id
            and child_writer.run_id == self._child_run_id
            and child_writer.stream_id == self._child_stream_id
        )

    def validate_live(self) -> tuple[SourceSnapshot, SourceSnapshot]:
        parent_writer = self._parent_reader.writer
        child_writer = self._child_writer
        parent_writer.owner.validate_dispatch()
        child_writer.owner.validate_dispatch()
        parent_snapshot = parent_writer.owner.read_snapshot()
        child_snapshot = child_writer.owner.read_snapshot()
        if (
            parent_writer.failed
            or child_writer.failed
            or not parent_snapshot.state.same_origin(self._parent_state)
            or not child_snapshot.state.same_origin(self._child_state)
            or not _writer_lease_matches(parent_writer, parent_snapshot)
            or not _writer_lease_matches(child_writer, child_snapshot)
            or not _stored_child_relation_matches(child_snapshot, self)
        ):
            raise ArtifactValidationError("Artifact child source admission changed")
        return parent_snapshot, child_snapshot


def _writer_lease_matches(writer: SourceTurnWriter, snapshot: SourceSnapshot) -> bool:
    metadata = snapshot.header.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    raw_metadata = cast(Mapping[object, object], metadata)
    if any(not isinstance(key, str) for key in raw_metadata):
        return False
    metadata_record = cast(Mapping[str, object], metadata)
    lease = metadata_record.get("execution_lease")
    if not isinstance(lease, Mapping):
        return False
    raw_lease = cast(Mapping[object, object], lease)
    if any(not isinstance(key, str) for key in raw_lease):
        return False
    lease_record = cast(Mapping[str, object], lease)
    return lease_record.get("run_id") == writer.run_id


def _stored_child_relation_matches(
    snapshot: SourceSnapshot,
    binding: ArtifactDirectChildBinding,
) -> bool:
    from nanobot.agent.child_sessions import DELEGATION_CHILD_METADATA_KEY
    from nanobot.session.runtime_authority import parse_runtime_session_metadata

    metadata = snapshot.header.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    raw_metadata = cast(Mapping[object, object], metadata)
    if any(not isinstance(key, str) for key in raw_metadata):
        return False
    metadata_record = cast(Mapping[str, object], metadata)
    child_relation = metadata_record.get(DELEGATION_CHILD_METADATA_KEY)
    if not isinstance(child_relation, Mapping):
        return False
    raw_child_relation = cast(Mapping[object, object], child_relation)
    if any(not isinstance(key, str) for key in raw_child_relation):
        return False
    relation = cast(Mapping[str, object], child_relation)
    if not binding.matches_child_relation(relation):
        return False
    runtime = parse_runtime_session_metadata(
        metadata_record,
        parent_incarnation=binding.parent_source_state.incarnation,
    )
    return (
        relation.get("child_id") == binding.child_id
        and relation.get("session_key") == binding.child_source_state.session_key
        and relation.get("parent_key") == binding.parent_source_state.session_key
        and relation.get("parent_incarnation") == binding.parent_source_state.incarnation
        and relation.get("state") in {"active", "completed"}
        and relation.get("current_run_id") == binding.child_run_id
        and runtime.present
        and runtime.executable
        and runtime.metadata.parent_key == binding.parent_source_state.session_key
        and runtime.metadata.parent_incarnation == binding.parent_source_state.incarnation
        and runtime.metadata.profile == relation.get("profile")
    )


@dataclass(frozen=True, slots=True)
class _Grant:
    """An exact descriptor binding between two directly related streams."""

    owner: ArtifactGrantSet
    direction: str
    source_invocation_stream: str
    target_invocation_stream: str
    descriptor: ArtifactDescriptor
    nonce: bytes
    origin_reader: ArtifactReader
    origin_state: SourceState
    target_writer: SourceTurnWriter
    target_state: SourceState
    child_binding: ArtifactDirectChildBinding | None = None
    run_store: RunStore | None = None
    parent_result_event_id: str | None = None
    handoff_tool_events: tuple[str, ...] = ()
    direct_child_nonce: bytes | None = None


class ArtifactGrantSet:
    """Nonserializable, sealed artifact grants for one admitted writer stream.

    A set is bound to one owner, store namespace, and invocation stream.  It
    carries no durable authority and never mutates source metadata.  New grants
    are minted only from native source evidence (never a previously received
    grant), preventing transitive regranting.
    """

    __slots__ = (
        "_owner",
        "_writer",
        "_store",
        "_invocation_nonce",
        "_grants",
        "_child_binding",
        "_reader",
        "_closed",
    )

    def __init__(self, capability: object, writer: SourceTurnWriter) -> None:
        if capability is not _CAPABILITY:
            raise TypeError("ArtifactGrantSet is runtime-owned")
        self._owner = writer.owner
        self._writer = writer
        self._store = None
        self._invocation_nonce = secrets.token_bytes(32)
        self._grants: tuple[_Grant, ...] = ()
        self._child_binding: ArtifactDirectChildBinding | None = None
        self._reader: ArtifactReader | None = None
        self._closed = False

    def __reduce__(self) -> NoReturn:
        raise TypeError("ArtifactGrantSet cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        _ = protocol
        raise TypeError("ArtifactGrantSet cannot be serialized")

    @classmethod
    def for_reader(cls, reader: ArtifactReader) -> ArtifactGrantSet:
        """Create a set for ``reader``; ``ArtifactReader`` seals the binding."""
        return cls(_CAPABILITY, reader.writer)

    @property
    def writer(self) -> SourceTurnWriter:
        return self._writer

    @property
    def references(self) -> tuple[str, ...]:
        """Return runtime-only notices; this never writes source state."""
        return tuple("artifact:" + grant.descriptor.artifact_id for grant in self._grants)

    def bind_reader(self, reader: ArtifactReader) -> None:
        if self._reader is not None and self._reader is not reader:
            raise ArtifactValidationError("Artifact grants are bound to another invocation")
        if reader.writer is not self._writer or reader.writer.owner is not self._owner:
            raise ArtifactValidationError("Artifact grants do not match execution owner")
        if (
            self._child_binding is not None
            and not self._child_binding.binds(self._child_binding.parent_reader, reader.writer)
        ):
            raise ArtifactValidationError("Artifact grants do not match child execution")
        if self._store is not None and self._store is not reader.store:
            raise ArtifactValidationError("Artifact grants do not match private store")
        self._reader = reader
        self._store = reader.store

    def close(self) -> None:
        self._closed = True
        self._grants = ()
        self._child_binding = None

    def _validate_reader(self, reader: ArtifactReader) -> None:
        if self._closed or self._reader is not reader or self._store is not reader.store:
            raise ArtifactValidationError("Artifact grants are unavailable")
        if reader.writer is not self._writer or reader.writer.owner is not self._owner:
            raise ArtifactValidationError("Artifact grants do not match invocation")
        self._owner.validate_dispatch()
        if reader.writer.failed:
            raise ArtifactValidationError("Artifact grants were revoked")

    @staticmethod
    def _references(references: object) -> tuple[str, ...]:
        # Retain the existing class hook and late-bound subclass overrides.
        return _parse_grant_references(references)

    @classmethod
    def issue_downward(
        cls,
        parent: ArtifactReader,
        child_writer: SourceTurnWriter,
        references: object,
        *,
        child_binding: ArtifactDirectChildBinding | None = None,
    ) -> ArtifactGrantSet:
        """Mint exact parent-to-direct-child grants without any source write."""
        identifiers = cls._references(references)
        same_owner_child_stream = (
            child_writer.owner is parent.writer.owner
            and child_writer.parent_stream_id == parent.writer.stream_id
        )
        if child_binding is None:
            if not same_owner_child_stream:
                raise ArtifactValidationError("Artifact grants require an admitted direct child")
        elif (
            same_owner_child_stream
            or type(child_binding) is not ArtifactDirectChildBinding
            or not child_binding.binds(parent, child_writer)
        ):
            raise ArtifactValidationError("Artifact grants require an admitted direct child")
        child = cls(_CAPABILITY, child_writer)
        child._child_binding = child_binding
        if not identifiers:
            if child_binding is not None:
                child_binding.validate_live()
            return child
        parent.require_grant_issuer()
        grants: list[_Grant] = []
        with parent.store.transaction() as blobs:
            with parent.writer.store.execution_transaction(
                parent.writer.owner.key, parent.writer.owner
            ):
                if child_binding is None:
                    parent.writer.owner.validate_dispatch()
                    snapshot = parent.writer.owner.read_snapshot()
                    target_snapshot = child_writer.owner.read_snapshot()
                    for identifier in identifiers:
                        # Only native evidence is eligible. A received grant cannot
                        # be forwarded as child-produced evidence.
                        descriptor = parent.verify_native_artifact(identifier, snapshot, blobs)
                        grants.append(_Grant(
                            child,
                            "downward",
                            parent.writer.stream_id,
                            child_writer.stream_id,
                            descriptor,
                            child._invocation_nonce,
                            parent,
                            snapshot.state,
                            child_writer,
                            target_snapshot.state,
                        ))
                    blobs.assert_reachable()
                    parent.writer.owner.validate_dispatch()
                else:
                    with child_writer.store.execution_transaction(
                        child_writer.owner.key, child_writer.owner
                    ):
                        parent_snapshot, target_snapshot = child_binding.validate_live()
                        for identifier in identifiers:
                            descriptor = parent.verify_native_artifact(
                                identifier, parent_snapshot, blobs
                            )
                            grants.append(_Grant(
                                child,
                                "downward",
                                parent.writer.stream_id,
                                child_writer.stream_id,
                                descriptor,
                                child._invocation_nonce,
                                parent,
                                parent_snapshot.state,
                                child_writer,
                                target_snapshot.state,
                                child_binding,
                            direct_child_nonce=child_binding.nonce_for_grant(_CAPABILITY),
                            ))
                        blobs.assert_reachable()
                        child_binding.validate_live()
        child._grants = tuple(grants)
        return child

    def find_separate_owner_grant(
        self,
        reader: ArtifactReader,
        identifier: str,
    ) -> _Grant | None:
        """Select an exact sealed cross-owner descriptor for this child/parent reader."""
        self._validate_reader(reader)
        candidates = [
            grant for grant in self._grants
            if grant.owner is self
            and grant.nonce is self._invocation_nonce
            and grant.descriptor.artifact_id == identifier
            and grant.target_writer is reader.writer
            and (
                grant.child_binding is None
                or (
                    grant.direct_child_nonce is not None
                    and grant.child_binding.matches_direct_child_nonce(grant.direct_child_nonce)
                )
            )
            and (
                grant.child_binding is not None
                or grant.origin_reader.writer.owner is not reader.writer.owner
            )
        ]
        if not candidates:
            return None
        if len(candidates) != 1:
            raise ArtifactValidationError("Artifact grant is ambiguous")
        grant = candidates[0]
        if (
            grant.target_writer.run_id != reader.writer.run_id
            or grant.target_invocation_stream != reader.writer.stream_id
        ):
            raise ArtifactValidationError("Artifact grant targets another invocation")
        return grant

    def read_separate_owner_grant(
        self,
        reader: ArtifactReader,
        grant: _Grant,
        *,
        reference: str,
        offset: int,
        max_bytes: int,
    ) -> tuple[ArtifactDescriptor, bytes]:
        """Read only the sealed origin namespace while both source lifetimes are current."""
        if grant.owner is not self or grant.nonce is not self._invocation_nonce:
            raise ArtifactValidationError("Artifact grant is unavailable")
        origin_reader = grant.origin_reader
        origin_writer = origin_reader.writer
        origin_store = origin_reader.store
        target_writer = grant.target_writer
        if parse_artifact_reference(reference) != grant.descriptor.artifact_id:
            raise ArtifactValidationError("Artifact grant is unavailable")
        binding = grant.child_binding
        if binding is not None and (
            grant.direct_child_nonce is None
            or not binding.matches_direct_child_nonce(grant.direct_child_nonce)
        ):
            raise ArtifactValidationError("Artifact child invocation binding changed")
        if binding is not None and (
            grant.direction != "downward"
            and binding.child_writer is not origin_writer
        ):
            raise ArtifactValidationError("Artifact handoff binding changed")

        with origin_store.transaction() as blobs:
            if grant.direction == "downward":
                with origin_writer.store.execution_transaction(
                    origin_writer.owner.key, origin_writer.owner
                ):
                    with target_writer.store.execution_transaction(
                        target_writer.owner.key, target_writer.owner
                    ):
                        origin_snapshot = origin_writer.owner.read_snapshot()
                        target_snapshot = target_writer.owner.read_snapshot()
                        if (
                            not origin_snapshot.state.same_origin(grant.origin_state)
                            or not target_snapshot.state.same_origin(grant.target_state)
                            or target_writer.failed
                            or (binding is not None and not binding.binds(origin_reader, target_writer))
                            or (
                                binding is not None
                                and not _stored_child_relation_matches(target_snapshot, binding)
                            )
                        ):
                            raise ArtifactValidationError("Artifact source admission changed")
                        origin_view = reader.origin_view(origin_reader)
                        origin_view.verify_published(grant.descriptor, origin_snapshot, blobs)
                        if binding is not None:
                            binding.validate_live()
                        raw = blobs.read_range(grant.descriptor, offset=offset, limit=max_bytes)
                        origin_writer.owner.validate_dispatch()
                        target_writer.owner.validate_dispatch()
            elif grant.direction == "upward":
                if grant.run_store is None or grant.parent_result_event_id is None:
                    raise ArtifactValidationError("Artifact handoff settlement is unavailable")
                with origin_writer.store.execution_transaction(origin_writer.owner.key):
                    origin_snapshot = origin_writer.store.load_source_snapshot(origin_writer.owner.key)
                    self._validate_settled_origin(grant, origin_snapshot)
                    with target_writer.store.execution_transaction(
                        target_writer.owner.key, target_writer.owner
                    ):
                        target_snapshot = target_writer.owner.read_snapshot()
                        if (
                            not target_snapshot.state.same_origin(grant.target_state)
                            or target_writer.failed
                        ):
                            raise ArtifactValidationError("Artifact source admission changed")
                        parent_result = next(
                            (event for event in target_snapshot.events
                             if event.get("_event_id") == grant.parent_result_event_id),
                            None,
                        )
                        if not self._is_parent_result_event(parent_result, target_writer):
                            raise ArtifactValidationError("Artifact handoff result changed")
                        origin_view = reader.origin_view(origin_reader)
                        origin_view.verify_settled_published(
                            grant.descriptor,
                            origin_snapshot,
                            blobs,
                            expected_state=grant.origin_state,
                            expected_run_id=origin_writer.run_id,
                            expected_stream_id=(
                                binding.child_stream_id if binding is not None
                                else grant.descriptor.stream_id
                            ),
                        )
                        raw = blobs.read_range(grant.descriptor, offset=offset, limit=max_bytes)
                        target_writer.owner.validate_dispatch()
            else:
                raise ArtifactValidationError("Artifact grant direction is invalid")
            blobs.assert_reachable()
            return grant.descriptor, raw

    @staticmethod
    def _is_parent_result_event(
        event: Mapping[str, object] | None,
        writer: SourceTurnWriter,
    ) -> bool:
        return (
            event is not None
            and event.get("role") == "tool"
            and event.get("_stream_id") == writer.stream_id
            and event.get("_run_id") == writer.run_id
        )

    @staticmethod
    def _validate_settled_origin(grant: _Grant, snapshot: SourceSnapshot) -> None:
        if (
            not snapshot.state.same_origin(grant.origin_state)
        ):
            raise ArtifactValidationError("Artifact child settlement changed")
        metadata = snapshot.header.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or any(key in metadata for key in ("execution_lease", "execution_recovery_required"))
        ):
            raise ArtifactValidationError("Artifact child settlement is unresolved")
        if grant.child_binding is not None and not _stored_child_relation_matches(
            snapshot, grant.child_binding,
        ):
            raise ArtifactValidationError("Artifact child relation changed")
        from nanobot.session.source import validate_execution_links

        if validate_execution_links(snapshot.events):
            raise ArtifactValidationError("Artifact child settlement is unresolved")
        retained = {
            event.get("_event_id"): event
            for event in snapshot.events
            if event.get("_stream_id") == grant.origin_reader.writer.stream_id
        }
        for event_id in grant.handoff_tool_events:
            event = retained.get(event_id)
            if (
                event is None
                or event.get("role") != "tool"
                or event.get("name") != "handoff_artifact"
                or event.get("_run_id") != grant.origin_reader.writer.run_id
                or event.get("_outcome_status") != "ok"
                or event.get("_dispatched") is not True
            ):
                raise ArtifactValidationError("Artifact handoff evidence changed")
        run_store = grant.run_store
        if run_store is None:
            raise ArtifactValidationError("Artifact child settlement is unavailable")
        from nanobot.agent.turn import RunStatus

        origin_run_id = grant.origin_reader.writer.run_id
        record = run_store.load(origin_run_id)
        if (
            record is None
            or record.status is not RunStatus.COMPLETED
            or record.session_ref is None
            or record.session_ref.session_key != snapshot.state.session_key
            or record.run_id != origin_run_id
            or record.stop_reason != "completed"
            or record.completed_at is None
        ):
            raise ArtifactValidationError("Artifact child run did not settle successfully")

    def authorize(
        self,
        reader: ArtifactReader,
        descriptor: ArtifactDescriptor,
        snapshot: SourceSnapshot,
        blobs: ArtifactBlobTransaction,
    ) -> None:
        """Revalidate exactly one received grant immediately before a read."""
        self._validate_reader(reader)
        candidates = [
            grant
            for grant in self._grants
            if grant.owner is self
            and grant.nonce is self._invocation_nonce
            and grant.descriptor == descriptor
            and grant.target_invocation_stream == reader.writer.stream_id
        ]
        if len(candidates) != 1:
            raise ArtifactValidationError("Artifact is outside the admitted stream")
        grant = candidates[0]
        if grant.direction == "downward":
            direct = reader.writer.parent_stream_id == grant.source_invocation_stream
        elif grant.direction == "upward":
            event = next(
                (row for row in snapshot.events if row.get("_event_id") == descriptor.event_id),
                None,
            )
            direct = (
                event is not None
                and event.get("_stream_id") == grant.source_invocation_stream
                and event.get("_parent_stream_id") == reader.writer.stream_id
            )
        else:
            direct = False
        if not direct:
            raise ArtifactValidationError("Artifact grant stream relation changed")
        reader.verify_received_descriptor(descriptor, snapshot, blobs)

    def accept_handoff(
        self,
        receipt: ArtifactHandoffReceipt,
        *,
        parent_result_event_id: str,
    ) -> tuple[str, ...]:
        """Mint child-to-parent grants after the parent's tool result is durable."""
        reader = self._reader
        if reader is None:
            raise ArtifactValidationError("Artifact handoff has no admitted reader")
        self._validate_reader(reader)
        if not isinstance(cast(object, receipt), ArtifactHandoffReceipt):
            raise ArtifactValidationError("Invalid artifact handoff receipt")
        if not isinstance(cast(object, parent_result_event_id), str) or not parent_result_event_id:
            raise ArtifactValidationError("Artifact handoff has no durable parent result")
        descriptors = receipt.preflight_for_parent(_CAPABILITY, reader)
        current_ids = {grant.descriptor.artifact_id for grant in self._grants}
        ids = [descriptor.artifact_id for descriptor in descriptors]
        if len(descriptors) > _MAX_ARTIFACT_REFERENCES or len(set(ids)) != len(ids):
            raise ArtifactValidationError("Invalid artifact handoff set")
        if current_ids.intersection(ids) or len(current_ids) + len(ids) > _MAX_ARTIFACT_REFERENCES:
            raise ArtifactValidationError("Artifact handoff exceeds invocation limit")
        if receipt.child_binding is not None:
            return self._accept_separate_owner_handoff(
                reader,
                receipt,
                descriptors,
                parent_result_event_id=parent_result_event_id,
            )
        minted: list[_Grant] = []
        with reader.store.transaction() as blobs:
            with reader.writer.store.execution_transaction(reader.writer.owner.key, reader.writer.owner):
                reader.writer.owner.validate_dispatch()
                snapshot = reader.writer.owner.read_snapshot()
                parent_result = next(
                    (event for event in snapshot.events if event.get("_event_id") == parent_result_event_id),
                    None,
                )
                if (
                    parent_result is None
                    or parent_result.get("role") != "tool"
                    or parent_result.get("_stream_id") != reader.writer.stream_id
                ):
                    raise ArtifactValidationError("Artifact handoff has no durable parent result")
                for descriptor in descriptors:
                    reader.verify_received_descriptor(descriptor, snapshot, blobs)
                    event = next(event for event in snapshot.events if event.get("_event_id") == descriptor.event_id)
                    if event.get("_parent_stream_id") != reader.writer.stream_id:
                        raise ArtifactValidationError("Artifact handoff is not from a direct child")
                    minted.append(_Grant(
                        self,
                        "upward",
                        descriptor.stream_id,
                        reader.writer.stream_id,
                        descriptor,
                        self._invocation_nonce,
                        receipt.collector.reader,
                        reader.writer.owner.read_snapshot().state,
                        reader.writer,
                        reader.writer.owner.read_snapshot().state,
                    ))
                blobs.assert_reachable()
                reader.writer.owner.validate_dispatch()
                previous_grants = self._grants
                new_grants = (*self._grants, *minted)
                self._grants = new_grants
                try:
                    receipt.consume_after_validation(_CAPABILITY)
                except BaseException:
                    self._grants = previous_grants
                    raise
        return tuple("artifact:" + descriptor.artifact_id for descriptor in descriptors)

    def _accept_separate_owner_handoff(
        self,
        reader: ArtifactReader,
        receipt: ArtifactHandoffReceipt,
        descriptors: tuple[ArtifactDescriptor, ...],
        *,
        parent_result_event_id: str,
    ) -> tuple[str, ...]:
        """Accept a settled canonical child receipt under origin-blob then source locks."""
        binding = receipt.child_binding
        source_store = receipt.source_store
        run_store = receipt.run_store
        child_reader = receipt.collector.reader
        child_writer = child_reader.writer
        if (
            binding is None
            or source_store is None
            or source_store is not child_writer.store
            or run_store is None
            or binding.parent_reader is not reader
            or binding.child_writer is not child_writer
        ):
            raise ArtifactValidationError("Artifact handoff settlement is unavailable")
        minted: list[_Grant] = []
        with child_reader.store.transaction() as blobs:
            with source_store.execution_transaction(child_writer.owner.key):
                child_snapshot = source_store.load_source_snapshot(child_writer.owner.key)
                self.validate_settled_child(
                    receipt,
                    child_snapshot,
                    blobs,
                    reader,
                    run_store=run_store,
                )
                with reader.writer.store.execution_transaction(
                    reader.writer.owner.key, reader.writer.owner
                ):
                    reader.writer.owner.validate_dispatch()
                    parent_snapshot = reader.writer.owner.read_snapshot()
                    if not parent_snapshot.state.same_origin(binding.parent_source_state):
                        raise ArtifactValidationError("Artifact parent source admission changed")
                    parent_result = next(
                        (event for event in parent_snapshot.events
                         if event.get("_event_id") == parent_result_event_id),
                        None,
                    )
                    if not self._is_parent_result_event(parent_result, reader.writer):
                        raise ArtifactValidationError("Artifact handoff has no durable parent result")
                    for descriptor in descriptors:
                        minted.append(_Grant(
                            self,
                            "upward",
                            descriptor.stream_id,
                            reader.writer.stream_id,
                            descriptor,
                            self._invocation_nonce,
                            child_reader,
                            child_snapshot.state,
                            reader.writer,
                            parent_snapshot.state,
                            binding,
                            run_store,
                            parent_result_event_id,
                            receipt.tool_events,
                            binding.nonce_for_grant(_CAPABILITY),
                        ))
                    blobs.assert_reachable()
                    reader.writer.owner.validate_dispatch()
                    previous_grants = self._grants
                    self._grants = (*self._grants, *minted)
                    try:
                        receipt.consume_after_validation(_CAPABILITY)
                    except BaseException:
                        self._grants = previous_grants
                        raise
        return tuple("artifact:" + descriptor.artifact_id for descriptor in descriptors)

    @staticmethod
    def validate_settled_child(
        receipt: ArtifactHandoffReceipt,
        snapshot: SourceSnapshot,
        blobs: ArtifactBlobTransaction,
        parent_reader: ArtifactReader,
        *,
        run_store: RunStore,
    ) -> None:
        binding = receipt.child_binding
        child_reader = receipt.collector.reader
        prepared_state = receipt.prepared_state
        if binding is None or prepared_state is None:
            raise ArtifactValidationError("Artifact handoff settlement is unavailable")
        if (
            not snapshot.state.same_origin(prepared_state)
            or not snapshot.state.same_origin(binding.child_source_state)
        ):
            raise ArtifactValidationError("Artifact child source admission changed")
        if not _stored_child_relation_matches(snapshot, binding):
            raise ArtifactValidationError("Artifact child relation changed")
        pending = snapshot.header.get("metadata")
        if not isinstance(pending, Mapping) or any(
            key in pending for key in ("execution_lease", "execution_recovery_required")
        ):
            raise ArtifactValidationError("Artifact child settlement is unresolved")
        from nanobot.session.source import validate_execution_links

        if validate_execution_links(snapshot.events):
            raise ArtifactValidationError("Artifact child settlement is unresolved")
        event_ids = {event.get("_event_id") for event in snapshot.events}
        if (
            not receipt.tool_events
            or not set(receipt.tool_events) <= event_ids
        ):
            raise ArtifactValidationError("Artifact handoff result was not retained")
        stream = [
            event for event in snapshot.events
            if event.get("_stream_id") == binding.child_stream_id
        ]
        if (
            not stream
            or stream[-1].get("role") != "assistant"
            or stream[-1].get("tool_calls")
            or stream[-1].get("_run_id") != binding.child_run_id
        ):
            raise ArtifactValidationError("Artifact handoff has no successful child terminal event")
        for event_id in receipt.tool_events:
            event = next(
                (row for row in stream if row.get("_event_id") == event_id),
                None,
            )
            if (
                event is None
                or event.get("role") != "tool"
                or event.get("name") != "handoff_artifact"
                or event.get("_run_id") != binding.child_run_id
                or event.get("_outcome_status") != "ok"
                or event.get("_dispatched") is not True
            ):
                raise ArtifactValidationError("Artifact handoff result was not successful")
        record = run_store.load(binding.child_run_id)
        from nanobot.agent.turn import RunStatus

        if (
            record is None
            or record.status is not RunStatus.COMPLETED
            or record.run_id != binding.child_run_id
            or record.session_ref is None
            or record.session_ref.session_key != snapshot.state.session_key
            or record.stop_reason != "completed"
            or record.completed_at is None
        ):
            raise ArtifactValidationError("Artifact child run did not settle successfully")
        origin_view = parent_reader.origin_view(child_reader)
        for descriptor in receipt.descriptors:
            origin_view.verify_settled_published(
                descriptor,
                snapshot,
                blobs,
                expected_state=prepared_state,
                expected_run_id=binding.child_run_id,
                expected_stream_id=binding.child_stream_id,
            )


class ArtifactHandoffReceipt:
    """A sealed child selection that becomes usable only after child completion."""

    __slots__ = (
        "_collector",
        "_descriptors",
        "_tool_events",
        "_sealed",
        "_consumed",
        "_child_binding",
        "_prepared_state",
        "_source_store",
        "_run_store",
    )

    def __init__(
        self,
        capability: object,
        collector: ArtifactHandoffCollector,
        descriptors: tuple[ArtifactDescriptor, ...],
        *,
        child_binding: ArtifactDirectChildBinding | None = None,
        prepared_state: SourceState | None = None,
    ) -> None:
        if capability is not _CAPABILITY:
            raise TypeError("ArtifactHandoffReceipt is runtime-owned")
        self._collector = collector
        self._descriptors = descriptors
        self._tool_events: list[str] = []
        self._sealed = False
        self._consumed = False
        self._child_binding = child_binding
        self._prepared_state = prepared_state
        self._source_store: SourceSessionStore | None = None
        self._run_store: RunStore | None = None

    @property
    def collector(self) -> ArtifactHandoffCollector:
        return self._collector

    @property
    def descriptors(self) -> tuple[ArtifactDescriptor, ...]:
        return self._descriptors

    @property
    def tool_events(self) -> tuple[str, ...]:
        return tuple(self._tool_events)

    @property
    def child_binding(self) -> ArtifactDirectChildBinding | None:
        return self._child_binding

    @property
    def prepared_state(self) -> SourceState | None:
        return self._prepared_state

    @property
    def source_store(self) -> SourceSessionStore | None:
        return self._source_store

    @property
    def run_store(self) -> RunStore | None:
        return self._run_store

    def __reduce__(self) -> NoReturn:
        raise TypeError("ArtifactHandoffReceipt cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        _ = protocol
        raise TypeError("ArtifactHandoffReceipt cannot be serialized")

    def verify_child_evidence(
        self,
        reader: ArtifactReader,
        snapshot: SourceSnapshot,
        blobs: ArtifactBlobTransaction,
        *,
        event_ids: set[object],
    ) -> tuple[ArtifactDescriptor, ...]:
        """Verify retained native evidence without sealing or consuming this receipt."""
        if not self._tool_events or not set(self._tool_events) <= event_ids:
            raise ArtifactValidationError("Artifact handoff result was not retained")
        verified: list[ArtifactDescriptor] = []
        for descriptor in self._descriptors:
            reader.verify_published(descriptor, snapshot, blobs)
            verified.append(descriptor)
        return tuple(verified)

    def _record_tool_event(self, capability: object, writer: SourceTurnWriter, event_id: str) -> None:
        if capability is not _CAPABILITY or writer is not self._collector.reader.writer:
            raise ArtifactValidationError("Artifact handoff result is not durable")
        if not isinstance(cast(object, event_id), str) or not event_id:
            raise ArtifactValidationError("Artifact handoff result identity is invalid")
        self._tool_events.append(event_id)

    def record_tool_event(self, capability: object, writer: SourceTurnWriter, event_id: str) -> None:
        """Retain the guarded durable-result hook, including late-bound overrides."""
        self._record_tool_event(capability, writer, event_id)

    def copy_tool_events_from(
        self, capability: object, receipts: list[ArtifactHandoffReceipt],
    ) -> None:
        """Copy ordered merge evidence under the constructor's runtime capability."""
        if capability is not _CAPABILITY:
            raise ArtifactValidationError("Artifact handoff is unavailable")
        self._tool_events = [event for receipt in receipts for event in receipt._tool_events]

    def belongs_to(self, collector: ArtifactHandoffCollector) -> bool:
        """Return whether this unconsumed receipt came from ``collector``."""
        return not self._consumed and self._collector is collector

    def targets_parent_writer(self, writer: SourceTurnWriter) -> bool:
        """Check the direct parent stream before a trusted outer tool forwards it."""
        if self._child_binding is not None:
            return (
                not self._consumed
                and self._child_binding.parent_reader.writer is writer
                and writer.run_id == self._child_binding.parent_run_id
                and writer.stream_id == self._child_binding.parent_stream_id
            )
        child = self._collector.reader.writer
        return (
            not self._consumed
            and child.owner is writer.owner
            and child.parent_stream_id == writer.stream_id
        )

    def _seal(self, capability: object) -> None:
        if capability is not _CAPABILITY or not self._tool_events:
            raise ArtifactValidationError("Artifact handoff was not durably recorded")
        self._sealed = True

    def seal_after_completion(self, capability: object) -> None:
        """Use the original seal hook after successful durable-child verification."""
        self._seal(capability)

    def seal_after_settlement(
        self,
        *,
        run_store: RunStore,
        source_store: SourceSessionStore,
    ) -> None:
        """Seal a canonical-child candidate only after its registered run and source close."""
        if self._child_binding is None:
            raise ArtifactValidationError("Artifact handoff settlement is unavailable")
        writer = self._collector.reader.writer
        if (
            self._sealed
            or self._consumed
            or self._prepared_state is None
            or source_store is not writer.store
        ):
            raise ArtifactValidationError("Artifact handoff settlement is unavailable")
        from nanobot.agent.run_store import RunStore as RuntimeRunStore

        if type(run_store) is not RuntimeRunStore:
            raise ArtifactValidationError("Artifact handoff requires the canonical run store")
        with self._collector.reader.store.transaction() as blobs:
            with source_store.execution_transaction(writer.owner.key):
                snapshot = source_store.load_source_snapshot(writer.owner.key)
                ArtifactGrantSet.validate_settled_child(
                    self,
                    snapshot,
                    blobs,
                    self._collector.reader,
                    run_store=run_store,
                )
                blobs.assert_reachable()
        self._source_store = source_store
        self._run_store = run_store
        self._sealed = True

    def _preflight_for_parent(
        self,
        capability: object,
        parent: ArtifactReader,
    ) -> tuple[ArtifactDescriptor, ...]:
        if capability is not _CAPABILITY or not self._sealed or self._consumed:
            raise ArtifactValidationError("Artifact handoff is unavailable")
        child = self._collector.reader.writer
        if self._child_binding is not None:
            if (
                not self._child_binding.binds(parent, child)
                or self._child_binding.parent_reader is not parent
            ):
                raise ArtifactValidationError("Artifact handoff is not a direct child result")
        elif (
            child.owner is not parent.writer.owner
            or child.parent_stream_id != parent.writer.stream_id
        ):
            raise ArtifactValidationError("Artifact handoff is not a direct child result")
        return self._descriptors

    def _consume_after_validation(self, capability: object) -> None:
        """Commit a handoff only after the caller has finished every check."""
        if capability is not _CAPABILITY or not self._sealed or self._consumed:
            raise ArtifactValidationError("Artifact handoff is unavailable")
        self._consumed = True

    def preflight_for_parent(
        self, capability: object, parent: ArtifactReader,
    ) -> tuple[ArtifactDescriptor, ...]:
        """Use the guarded, one-shot direct-parent preflight without minting a grant."""
        return self._preflight_for_parent(capability, parent)

    def consume_after_validation(self, capability: object) -> None:
        """Retain the original consumption fault seam and caller rollback order."""
        self._consume_after_validation(capability)

    def _invalidate(self, capability: object) -> None:
        if capability is not _CAPABILITY:
            raise ArtifactValidationError("Artifact handoff is unavailable")
        self._consumed = True

    def invalidate_after_merge(self, capability: object) -> None:
        """Retain the original invalidation hook after a merged receipt is sealed."""
        self._invalidate(capability)


class ArtifactHandoffCollector:
    """Collect child-only selections and seal them after a successful terminal event."""

    __slots__ = ("reader", "_receipts", "_selected", "_closed", "_child_binding")

    def __init__(
        self,
        reader: ArtifactReader,
        *,
        child_binding: ArtifactDirectChildBinding | None = None,
    ) -> None:
        if child_binding is None and reader.writer.parent_stream_id is None:
            raise ArtifactValidationError("Artifact handoff requires a child stream")
        if child_binding is not None:
            if not child_binding.binds(child_binding.parent_reader, reader.writer):
                raise ArtifactValidationError("Artifact handoff requires an admitted direct child")
            child_binding.validate_live()
        self.reader = reader
        self._receipts: list[ArtifactHandoffReceipt] = []
        self._selected: set[str] = set()
        self._closed = False
        self._child_binding = child_binding

    def select(self, references: object) -> ArtifactHandoffReceipt:
        if self._closed:
            raise ArtifactValidationError("Artifact handoff is unavailable")
        identifiers = _parse_grant_references(references)
        if not identifiers:
            raise ArtifactValidationError("Artifact handoff requires at least one reference")
        if self._selected.intersection(identifiers) or len(self._selected) + len(identifiers) > _MAX_ARTIFACT_REFERENCES:
            raise ArtifactValidationError("Artifact handoff references must be unique and bounded")
        descriptors: list[ArtifactDescriptor] = []
        with self.reader.store.transaction() as blobs:
            with self.reader.writer.store.execution_transaction(self.reader.writer.owner.key,
                                       self.reader.writer.owner):
                self.reader.writer.owner.validate_dispatch()
                snapshot = self.reader.writer.owner.read_snapshot()
                for identifier in identifiers:
                    # Handoff never accepts an inbound parent grant.
                    descriptor = self.reader.verify_native_artifact(identifier, snapshot, blobs)
                    if (
                        descriptor.stream_id != self.reader.writer.stream_id
                        or descriptor.run_id != self.reader.writer.run_id
                    ):
                        raise ArtifactValidationError("Artifact handoff requires child-produced evidence")
                    descriptors.append(descriptor)
                blobs.assert_reachable()
                self.reader.writer.owner.validate_dispatch()
        self._selected.update(identifiers)
        receipt = ArtifactHandoffReceipt(_CAPABILITY, self, tuple(descriptors))
        self._receipts.append(receipt)
        return receipt

    def record(self, receipt: ArtifactHandoffReceipt, event_id: str) -> None:
        if receipt not in self._receipts or self._closed:
            raise ArtifactValidationError("Artifact handoff is unavailable")
        receipt.record_tool_event(_CAPABILITY, self.reader.writer, event_id)

    def finalize(self, *, stop_reason: str) -> ArtifactHandoffReceipt | None:
        if self._child_binding is not None:
            return self.prepare_after_completion(stop_reason=stop_reason)
        if self._closed:
            return None
        self._closed = True
        if stop_reason != "completed" or not self._receipts:
            return None
        writer = self.reader.writer
        with self.reader.store.transaction() as blobs:
            with writer.store.execution_transaction(writer.owner.key, writer.owner):
                writer.owner.validate_dispatch()
                snapshot = writer.owner.read_snapshot()
                stream = [event for event in snapshot.events if event.get("_stream_id") == writer.stream_id]
                if not stream or stream[-1].get("role") != "assistant" or stream[-1].get("tool_calls"):
                    raise ArtifactValidationError("Artifact handoff has no successful child terminal event")
                event_ids: set[object] = {event.get("_event_id") for event in stream}
                descriptors: list[ArtifactDescriptor] = []
                for receipt in self._receipts:
                    descriptors.extend(receipt.verify_child_evidence(
                        self.reader, snapshot, blobs, event_ids=event_ids,
                    ))
                blobs.assert_reachable()
                writer.owner.validate_dispatch()
        # Merge receipts into one one-shot authority for the parent tool result.
        merged = ArtifactHandoffReceipt(_CAPABILITY, self, tuple(descriptors))
        merged.copy_tool_events_from(_CAPABILITY, self._receipts)
        merged.seal_after_completion(_CAPABILITY)
        for receipt in self._receipts:
            receipt.invalidate_after_merge(_CAPABILITY)
        return merged

    def prepare_after_completion(self, *, stop_reason: str) -> ArtifactHandoffReceipt | None:
        """Prepare a canonical child receipt before owner close; settlement seals it later."""
        if self._child_binding is None:
            return self.finalize(stop_reason=stop_reason)
        if self._closed:
            return None
        self._closed = True
        if stop_reason != "completed" or not self._receipts:
            return None
        writer = self.reader.writer
        binding = self._child_binding
        with self.reader.store.transaction() as blobs:
            with writer.store.execution_transaction(writer.owner.key, writer.owner):
                parent_snapshot, child_snapshot = binding.validate_live()
                _ = parent_snapshot
                snapshot = writer.owner.read_snapshot()
                if not snapshot.state.same_origin(binding.child_source_state):
                    raise ArtifactValidationError("Artifact child source admission changed")
                if not _stored_child_relation_matches(snapshot, binding):
                    raise ArtifactValidationError("Artifact child relation changed")
                stream = [
                    event for event in snapshot.events
                    if event.get("_stream_id") == binding.child_stream_id
                ]
                if (
                    not stream
                    or stream[-1].get("role") != "assistant"
                    or stream[-1].get("tool_calls")
                    or stream[-1].get("_run_id") != binding.child_run_id
                ):
                    raise ArtifactValidationError(
                        "Artifact handoff has no successful child terminal event"
                    )
                event_ids: set[object] = {event.get("_event_id") for event in stream}
                for receipt in self._receipts:
                    if not receipt.tool_events or not set(receipt.tool_events) <= event_ids:
                        raise ArtifactValidationError("Artifact handoff result was not retained")
                    for event_id in receipt.tool_events:
                        event = next(
                            (row for row in stream if row.get("_event_id") == event_id),
                            None,
                        )
                        if (
                            event is None
                            or event.get("role") != "tool"
                            or event.get("name") != "handoff_artifact"
                            or event.get("_run_id") != binding.child_run_id
                            or event.get("_outcome_status") != "ok"
                            or event.get("_dispatched") is not True
                        ):
                            raise ArtifactValidationError(
                                "Artifact handoff result was not successful"
                            )
                    receipt.verify_child_evidence(
                        self.reader,
                        snapshot,
                        blobs,
                        event_ids=event_ids,
                    )
                blobs.assert_reachable()
                writer.owner.validate_dispatch()
                child_state = child_snapshot.state
        descriptors = tuple(
            descriptor
            for receipt in self._receipts
            for descriptor in receipt.descriptors
        )
        merged = ArtifactHandoffReceipt(
            _CAPABILITY,
            self,
            descriptors,
            child_binding=binding,
            prepared_state=child_state,
        )
        merged.copy_tool_events_from(_CAPABILITY, self._receipts)
        for receipt in self._receipts:
            receipt.invalidate_after_merge(_CAPABILITY)
        return merged


__all__ = [
    "CanonicalChildArtifactInput",
    "ArtifactDirectChildBinding",
    "ArtifactGrantSet",
    "ArtifactHandoffCollector",
    "ArtifactHandoffReceipt",
    "_MAX_ARTIFACT_REFERENCES",
]
