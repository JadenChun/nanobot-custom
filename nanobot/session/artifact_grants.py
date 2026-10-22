"""Sealed invocation-local grants for direct artifact stream handoff.

Artifacts remain source-bound evidence, not capabilities.  This module keeps the
one narrow capability needed by foreground delegation out of source metadata and
out of serializable tool inputs.  Every use is revalidated by ``ArtifactReader``
while it holds the private-blob transaction followed by the execution/source
transaction.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, SupportsIndex, cast

from nanobot.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
    parse_artifact_reference,
)

if TYPE_CHECKING:
    from nanobot.session.artifact_reader import ArtifactReader
    from nanobot.session.artifact_store import ArtifactBlobTransaction
    from nanobot.session.source_store import SourceSnapshot
    from nanobot.session.turn_writer import SourceTurnWriter

_MAX_ARTIFACT_REFERENCES = 16
_CAPABILITY = object()


def _parse_grant_references(references: object) -> tuple[str, ...]:
    if not isinstance(references, list) or len(cast(list[object], references)) > _MAX_ARTIFACT_REFERENCES:
        raise ArtifactValidationError("Invalid artifact reference grant set")
    parsed = tuple(parse_artifact_reference(value) for value in cast(list[object], references))
    if len(set(parsed)) != len(parsed):
        raise ArtifactValidationError("Artifact references must be unique")
    return parsed


@dataclass(frozen=True, slots=True)
class _Grant:
    """An exact descriptor binding between two directly related streams."""

    owner: ArtifactGrantSet
    direction: str
    source_invocation_stream: str
    target_invocation_stream: str
    descriptor: ArtifactDescriptor
    nonce: bytes


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
        if self._store is not None and self._store is not reader.store:
            raise ArtifactValidationError("Artifact grants do not match private store")
        self._reader = reader
        self._store = reader.store

    def close(self) -> None:
        self._closed = True
        self._grants = ()

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
    ) -> ArtifactGrantSet:
        """Mint exact parent-to-direct-child grants without any source write."""
        identifiers = cls._references(references)
        if child_writer.owner is not parent.writer.owner or child_writer.parent_stream_id != parent.writer.stream_id:
            raise ArtifactValidationError("Artifact grants require a direct child stream")
        child = cls(_CAPABILITY, child_writer)
        if not identifiers:
            return child
        parent.require_grant_issuer()
        grants: list[_Grant] = []
        with parent.store.transaction() as blobs:
            with parent.writer.store.execution_transaction(parent.writer.owner.key, parent.writer.owner):
                parent.writer.owner.validate_dispatch()
                snapshot = parent.writer.owner.read_snapshot()
                for identifier in identifiers:
                    # Only native evidence is eligible.  A child cannot pass a
                    # received grant on to another child.
                    descriptor = parent.verify_native_artifact(identifier, snapshot, blobs)
                    grants.append(_Grant(
                        child,
                        "downward",
                        parent.writer.stream_id,
                        child_writer.stream_id,
                        descriptor,
                        child._invocation_nonce,
                    ))
                blobs.assert_reachable()
                parent.writer.owner.validate_dispatch()
        child._grants = tuple(grants)
        return child

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


class ArtifactHandoffReceipt:
    """A sealed child selection that becomes usable only after child completion."""

    __slots__ = ("_collector", "_descriptors", "_tool_events", "_sealed", "_consumed")

    def __init__(
        self,
        capability: object,
        collector: ArtifactHandoffCollector,
        descriptors: tuple[ArtifactDescriptor, ...],
    ) -> None:
        if capability is not _CAPABILITY:
            raise TypeError("ArtifactHandoffReceipt is runtime-owned")
        self._collector = collector
        self._descriptors = descriptors
        self._tool_events: list[str] = []
        self._sealed = False
        self._consumed = False

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

    def _preflight_for_parent(
        self,
        capability: object,
        parent: ArtifactReader,
    ) -> tuple[ArtifactDescriptor, ...]:
        if capability is not _CAPABILITY or not self._sealed or self._consumed:
            raise ArtifactValidationError("Artifact handoff is unavailable")
        child = self._collector.reader.writer
        if (child.owner is not parent.writer.owner
                or child.parent_stream_id != parent.writer.stream_id):
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

    __slots__ = ("reader", "_receipts", "_selected", "_closed")

    def __init__(self, reader: ArtifactReader) -> None:
        if reader.writer.parent_stream_id is None:
            raise ArtifactValidationError("Artifact handoff requires a child stream")
        self.reader = reader
        self._receipts: list[ArtifactHandoffReceipt] = []
        self._selected: set[str] = set()
        self._closed = False

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
                    if descriptor.stream_id != self.reader.writer.stream_id:
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


__all__ = [
    "ArtifactGrantSet",
    "ArtifactHandoffCollector",
    "ArtifactHandoffReceipt",
    "_MAX_ARTIFACT_REFERENCES",
]
