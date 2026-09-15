"""Dormant lease-free idle checkpoint capture and publication.

The module is an integration seam only. It has no scanner, provider, model,
summarizer, or factory and therefore cannot activate idle work by itself. A
caller captures one detached source snapshot, computes a proposal outside all
storage locks, and later asks this module to perform one exact, non-rebased
publication attempt.

Idle publication deliberately does not construct an ``ExecutionOwner``. It
first holds the configured private artifact transaction (when dependencies are
present), then takes the nonblocking execution lock and source-file lock. The
source and idle token are re-read under those locks immediately before the one
atomic JSONL replacement.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from nanobot.agent.tools.read_provenance import FilesystemReadReceipt
from nanobot.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
    parse_artifact_descriptor,
    published_artifacts,
)
from nanobot.session.artifact_store import (
    ArtifactBlobTransaction,
    ArtifactStorageBusyError,
    PrivateArtifactStore,
)
from nanobot.session.artifact_verification import verify_published_artifact
from nanobot.session.checkpoint_dependencies import (
    SummaryDependencyProof,
    derive_summary_references,
)
from nanobot.session.context_checkpoint import validate_context_checkpoint
from nanobot.session.execution_owner import ExecutionBusyError, execution_transaction
from nanobot.session.idle_compaction import (
    IdleCaptureConflictError,
    IdleCaptureToken,
    capture_idle_token,
    validate_idle_capture,
)
from nanobot.session.source import SourceValidationError

if TYPE_CHECKING:
    from nanobot.context_repo.policy import ResourceAccessPolicy
    from nanobot.session.source_store import SourceSessionStore, SourceSnapshot


_CAPABILITY = object()


class IdleCheckpointError(SourceValidationError):
    """Base error for a rejected idle checkpoint capture or publication."""


class IdleCheckpointConflictError(IdleCheckpointError):
    """The detached idle proposal no longer matches durable source state."""


IdleCheckpointConflict = IdleCheckpointConflictError


@dataclass(frozen=True, slots=True)
class IdlePolicyBinding:
    """Current policy inputs used only to revalidate recorded read receipts.

    ``workspace`` is the effective project/workspace root used by the source
    receipt. ``resolve_read`` is the already-admitted resolver from the owning
    runtime; this object never exposes a filesystem reader or tool.
    """

    fingerprint: str
    workspace: Path | None
    resource_policy: ResourceAccessPolicy | None
    resolve_read: Callable[[str], Path]

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, str) or not self.fingerprint.strip():
            raise IdleCheckpointError("Invalid idle policy fingerprint")
        if not callable(self.resolve_read):
            raise IdleCheckpointError("Invalid idle policy resolver")
        if self.workspace is not None and not isinstance(self.workspace, Path):
            raise IdleCheckpointError("Invalid idle policy workspace")


@dataclass(frozen=True, slots=True)
class IdleDependencyAccess:
    """Opaque, nonserializable access to already-published root dependencies.

    The constructor is capability-sealed. It intentionally exposes no store,
    filesystem reader, grant set, or byte/range read method. The publisher in
    this module is the only consumer and can verify only references captured
    from the exact root-history manifest.
    """

    _capability: object
    _source: SourceSessionStore
    _capture_nonce: object
    _token: IdleCaptureToken
    _store: PrivateArtifactStore
    _workspace_id: str
    _session_key: str
    _incarnation: str
    _generation: int
    _policy_fingerprint: str
    _descriptors: tuple[ArtifactDescriptor, ...]

    def __post_init__(self) -> None:
        if self._capability is not _CAPABILITY:
            raise IdleCheckpointError("Invalid idle dependency capability")
        if type(self._token) is not IdleCaptureToken:
            raise IdleCheckpointError("Invalid idle dependency capture")
        if not self._descriptors:
            # An empty root-history manifest is a valid capability. It still
            # must never be widened after capture.
            return
        if any(type(item) is not ArtifactDescriptor for item in self._descriptors):
            raise IdleCheckpointError("Invalid idle dependency descriptor")

    def __reduce__(self) -> object:
        raise TypeError("Idle dependency access cannot be serialized")

    def __deepcopy__(self, memo: dict[int, object]) -> IdleDependencyAccess:
        raise TypeError("Idle dependency access cannot be copied")

    def _matches_capture(self, source: SourceSessionStore, capture: IdleCheckpointCapture) -> bool:
        return (
            self._source is source
            and self._capture_nonce is capture._nonce
            and self._token is capture.token
            and self._store is source._artifact_store
            and self._store.workspace_id == self._workspace_id
            and self._store.session_key == self._session_key
            and self._store.incarnation == self._incarnation
            and self._store.generation == self._generation
        )

    def _verify(
        self,
        reference: str,
        snapshot: SourceSnapshot,
        blobs: ArtifactBlobTransaction,
        policy: IdlePolicyBinding,
    ) -> FilesystemReadReceipt:
        if policy.fingerprint != self._policy_fingerprint:
            raise IdleCheckpointConflict("Idle policy fingerprint changed")
        descriptor = next((item for item in self._descriptors if item.reference == reference), None)
        if descriptor is None:
            raise ArtifactValidationError("Idle dependency is outside captured root history")
        return verify_published_artifact(
            descriptor,
            snapshot,
            blobs,
            store=self._store,
            workspace=policy.workspace,
            policy=policy.resource_policy,
            resolve_read=policy.resolve_read,
            root_history=True,
        )


@dataclass(frozen=True, slots=True)
class IdleCheckpointCapture:
    """Detached idle admission evidence passed from capture to publication."""

    token: IdleCaptureToken
    snapshot: SourceSnapshot
    runtime_fingerprint: str
    policy_fingerprint: str
    _source: SourceSessionStore = field(repr=False, compare=False)
    _nonce: object = field(repr=False, compare=False)
    dependencies: IdleDependencyAccess | None = None

    def __post_init__(self) -> None:
        if type(self.token) is not IdleCaptureToken or not isinstance(self.snapshot, object):
            raise IdleCheckpointError("Invalid idle checkpoint capture")
        for name, value in (
            ("runtime", self.runtime_fingerprint),
            ("policy", self.policy_fingerprint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise IdleCheckpointError(f"Invalid idle {name} fingerprint")
        if self.dependencies is not None and type(self.dependencies) is not IdleDependencyAccess:
            raise IdleCheckpointError("Invalid idle dependency capability")

    def __reduce__(self) -> object:
        raise TypeError("Idle checkpoint capture cannot be serialized")

    def __deepcopy__(self, memo: dict[int, object]) -> IdleCheckpointCapture:
        raise TypeError("Idle checkpoint capture cannot be copied")


def _validate_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IdleCheckpointError(f"Invalid idle {label} fingerprint")
    return value


def _root_descriptors(
    source: SourceSessionStore,
    snapshot: SourceSnapshot,
) -> tuple[ArtifactDescriptor, ...]:
    """Return only current manifest descriptors from root-history events."""

    store = source._artifact_store
    if store is None:
        return ()
    if (
        store.workspace_id != source._backend.sessions_dir.name
        or store.session_key != snapshot.state.session_key
        or store.incarnation != snapshot.state.incarnation
        or store.generation != snapshot.state.generation
        or snapshot.state.store_identity != source._store_identity
    ):
        raise ArtifactValidationError("Idle artifact store binding changed")
    descriptors = published_artifacts(snapshot, workspace_id=store.workspace_id)
    root: list[ArtifactDescriptor] = []
    for descriptor in descriptors:
        event = next((row for row in snapshot.events if row.get("_event_id") == descriptor.event_id), None)
        if (
            event is not None
            and event.get("_parent_stream_id") is None
            and descriptor.stream_id == event.get("_stream_id")
        ):
            root.append(parse_artifact_descriptor(descriptor.to_dict()))
    return tuple(root)


def _capture_with_snapshot(
    source: SourceSessionStore,
    key: str,
    snapshot: SourceSnapshot,
    *,
    runtime_fingerprint: str,
    policy_fingerprint: str,
    policy: IdlePolicyBinding | None,
    runtime_checkpoint_present: bool,
) -> IdleCheckpointCapture | None:
    token = capture_idle_token(snapshot, runtime_checkpoint_present=runtime_checkpoint_present)
    if token is None:
        return None
    nonce = object()
    dependencies = None
    if source._artifact_store is not None and policy is not None:
        dependencies = IdleDependencyAccess(
            _CAPABILITY,
            source,
            nonce,
            token,
            source._artifact_store,
            source._artifact_store.workspace_id,
            snapshot.state.session_key,
            snapshot.state.incarnation,
            snapshot.state.generation,
            policy.fingerprint,
            _root_descriptors(source, snapshot),
        )
    return IdleCheckpointCapture(
        token,
        copy.deepcopy(snapshot),
        _validate_fingerprint(runtime_fingerprint, "runtime"),
        _validate_fingerprint(policy_fingerprint, "policy"),
        source,
        nonce,
        dependencies,
    )


def capture_idle_checkpoint(
    source: SourceSessionStore,
    key: str,
    *,
    runtime_fingerprint: str,
    policy_fingerprint: str,
    policy: IdlePolicyBinding | None = None,
) -> IdleCheckpointCapture | None:
    """Capture idle source and dependency state without holding a lock for compute."""

    if type(source) is not _source_store_type():
        raise IdleCheckpointError("Invalid idle source store")
    runtime_fingerprint = _validate_fingerprint(runtime_fingerprint, "runtime")
    policy_fingerprint = _validate_fingerprint(policy_fingerprint, "policy")
    if policy is not None and policy.fingerprint != policy_fingerprint:
        raise IdleCheckpointError("Idle policy fingerprint does not match binding")
    backend = source._backend
    with backend.locked_session_files():
        header, events = source._read_unlocked(key)
        snapshot = source._snapshot(header, events)
        return _capture_with_snapshot(
            source,
            key,
            snapshot,
            runtime_fingerprint=runtime_fingerprint,
            policy_fingerprint=policy_fingerprint,
            policy=policy,
            runtime_checkpoint_present=source._runtime_checkpoint_present(key),
        )


def _source_store_type() -> type[SourceSessionStore]:
    # Import lazily to avoid source_store -> idle_checkpoint import cycles.
    from nanobot.session.source_store import SourceSessionStore

    return SourceSessionStore


def _candidate_and_proof(
    capture: IdleCheckpointCapture,
    candidate: object,
    proof: SummaryDependencyProof | None,
) -> tuple[dict[str, object], tuple[str, ...], SummaryDependencyProof | None]:
    if not isinstance(candidate, dict):
        raise IdleCheckpointError("Invalid idle checkpoint proposal")
    try:
        detached_candidate = copy.deepcopy(candidate)
        detached_proof = copy.deepcopy(proof) if proof is not None else None
    except (TypeError, ValueError, RecursionError):
        raise IdleCheckpointError("Invalid idle checkpoint proposal") from None
    validated = validate_context_checkpoint(
        detached_candidate,
        capture.snapshot.events,
        generation=capture.token.generation,
        source_revision=capture.token.source_revision,
        parent_checkpoint_id=capture.token.checkpoint_id,
    )
    references = tuple(validated["artifact_references"])
    if detached_proof is not None:
        if type(detached_proof) is not SummaryDependencyProof:
            raise IdleCheckpointError("Invalid idle dependency proof")
        if detached_proof.candidate.snapshot != capture.snapshot:
            raise IdleCheckpointConflict("Idle dependency proof source changed")
        if detached_proof.candidate.stream_id is not None:
            raise IdleCheckpointError("Idle checkpoints require root-history proof")
        if (
            not detached_proof.chunks
            or detached_proof.chunks[-1].evidence.boundary != validated["boundary"]
            or detached_proof.chunks[-1].summary != validated["summary"]
        ):
            raise IdleCheckpointError("Idle dependency proof does not match proposal")
        derived = derive_summary_references(detached_proof, capture.snapshot)
        if derived != references:
            raise IdleCheckpointError("Idle dependency proof references changed")
    elif references:
        raise IdleCheckpointError("Dependent idle checkpoint requires a complete proof")
    if references and validated["schema_version"] != 2:
        raise IdleCheckpointError("Dependent idle checkpoint requires schema version 2")
    if references and (
        validated.get("projection_scope", "root") != "root"
        or validated.get("projection_version", 1) != 1
    ):
        raise IdleCheckpointError("Dependent idle checkpoint requires root projection")
    return validated, references, detached_proof


def _validate_capture_for_source(
    source: SourceSessionStore,
    key: str,
    capture: IdleCheckpointCapture,
    *,
    runtime_fingerprint: str,
    policy_fingerprint: str,
) -> None:
    if (
        type(capture) is not IdleCheckpointCapture
        or capture._source is not source
        or capture.token.store_identity != source._store_identity
        or capture.token.session_key != key
        or capture.runtime_fingerprint != runtime_fingerprint
        or capture.policy_fingerprint != policy_fingerprint
    ):
        raise IdleCheckpointConflict("Idle checkpoint capture binding changed")
    if capture.dependencies is not None and not capture.dependencies._matches_capture(source, capture):
        raise IdleCheckpointConflict("Idle dependency capability binding changed")


def _publish_unlocked(
    source: SourceSessionStore,
    key: str,
    capture: IdleCheckpointCapture,
    validated: dict[str, object],
    references: tuple[str, ...],
    proof: SummaryDependencyProof | None,
    *,
    policy: IdlePolicyBinding | None,
    blobs: ArtifactBlobTransaction | None,
) -> SourceSnapshot:
    header, events = source._read_unlocked(key)
    snapshot = source._snapshot(header, events)
    try:
        validate_idle_capture(
            capture.token,
            snapshot,
            runtime_checkpoint_present=source._runtime_checkpoint_present(key),
        )
    except IdleCaptureConflictError:
        raise IdleCheckpointConflict("Idle source state changed during publication") from None
    # Re-run exact structural validation against the lock-held source rather
    # than trusting the detached copy used while the model computed.
    validated_current = validate_context_checkpoint(
        validated,
        events,
        generation=snapshot.state.generation,
        source_revision=snapshot.state.revision,
        parent_checkpoint_id=snapshot.state.checkpoint_id,
    )
    if validated_current != validated:
        raise IdleCheckpointConflict("Idle checkpoint proposal changed during publication")
    if proof is not None:
        if proof.candidate.snapshot != snapshot:
            raise IdleCheckpointConflict("Idle dependency proof source changed during publication")
        if proof.candidate.stream_id is not None:
            raise IdleCheckpointError("Idle checkpoints require root-history proof")
        if (
            not proof.chunks
            or proof.chunks[-1].evidence.boundary != validated_current["boundary"]
            or proof.chunks[-1].summary != validated_current["summary"]
        ):
            raise IdleCheckpointConflict("Idle dependency proof changed during publication")
        try:
            locked_references = derive_summary_references(proof, snapshot)
        except SourceValidationError:
            raise IdleCheckpointConflict("Idle dependency proof changed during publication") from None
        if locked_references != references:
            raise IdleCheckpointConflict("Idle dependency proof references changed during publication")
    if references:
        if proof is None or blobs is None or capture.dependencies is None or policy is None:
            raise IdleCheckpointError("Idle dependency access is unavailable")
        for reference in references:
            capture.dependencies._verify(reference, snapshot, blobs, policy)
        blobs.assert_reachable()
    header["metadata"]["context_checkpoint"] = validated_current
    header["revision"] += 1
    return source._write(header, events)


def publish_idle_checkpoint(
    source: SourceSessionStore,
    key: str,
    capture: IdleCheckpointCapture,
    candidate: dict[str, object],
    *,
    runtime_fingerprint: str,
    policy_fingerprint: str,
    policy: IdlePolicyBinding | None = None,
    proof: SummaryDependencyProof | None = None,
) -> SourceSnapshot:
    """Attempt exactly one lease-free idle checkpoint publication.

    The method never waits for or fabricates a foreground owner and never
    retries a stale CAS. Dependency-bearing proposals hold the blob transaction
    before the nonblocking execution/source transaction. All policy, source,
    token, proof, namespace, and durable-byte checks run immediately before the
    single checkpoint write.
    """

    if type(source) is not _source_store_type():
        raise IdleCheckpointError("Invalid idle source store")
    runtime_fingerprint = _validate_fingerprint(runtime_fingerprint, "runtime")
    policy_fingerprint = _validate_fingerprint(policy_fingerprint, "policy")
    if policy is not None and policy.fingerprint != policy_fingerprint:
        raise IdleCheckpointError("Idle policy fingerprint does not match binding")
    _validate_capture_for_source(
        source,
        key,
        capture,
        runtime_fingerprint=runtime_fingerprint,
        policy_fingerprint=policy_fingerprint,
    )
    validated, references, detached_proof = _candidate_and_proof(capture, candidate, proof)
    if references:
        access = capture.dependencies
        if access is None or policy is None or source._artifact_store is not access._store:
            raise IdleCheckpointError("Idle dependency access is unavailable")
        try:
            with access._store.transaction(blocking=False) as blobs:
                blobs.assert_reachable()
                with execution_transaction(source._backend, key):
                    return _publish_unlocked(
                        source,
                        key,
                        capture,
                        validated,
                        references,
                        detached_proof,
                        policy=policy,
                        blobs=blobs,
                    )
        except ArtifactStorageBusyError:
            raise ExecutionBusyError() from None
    with execution_transaction(source._backend, key):
        return _publish_unlocked(
            source,
            key,
            capture,
            validated,
            references,
            detached_proof,
            policy=policy,
            blobs=None,
        )


__all__ = [
    "IdleCheckpointCapture",
    "IdleCheckpointConflict",
    "IdleCheckpointConflictError",
    "IdleCheckpointError",
    "IdleDependencyAccess",
    "IdlePolicyBinding",
    "capture_idle_checkpoint",
    "publish_idle_checkpoint",
]
