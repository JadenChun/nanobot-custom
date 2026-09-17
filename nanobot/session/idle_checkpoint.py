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
import hashlib
import json
from contextlib import nullcontext
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
from nanobot.session.source_root import SourceRootAuthority, SourceRootAuthorityError

if TYPE_CHECKING:
    from nanobot.context_repo.policy import ResourceAccessPolicy
    from nanobot.session.source_store import SourceSessionStore, SourceSnapshot


_CAPABILITY = object()
_UNBOUND_ATTESTATION = object()


@dataclass(frozen=True, slots=True)
class _IdleCaptureAttestation:
    """Factory-issued identity and immutable digest for one idle capture."""

    _capability: object = field(repr=False)
    _capture_id: int = field(repr=False)
    _nonce: object = field(repr=False)
    _token: IdleCaptureToken = field(repr=False)
    _source: SourceSessionStore = field(repr=False)
    _store: PrivateArtifactStore | None = field(repr=False)
    _store_binding: tuple[str, str, str, int] | None
    _runtime_fingerprint: str
    _policy_fingerprint: str
    _snapshot_digest: str
    _dependency_present: bool
    _dependency_id: int | None = field(repr=False)
    _descriptor_digest: str | None
    _root_authority: SourceRootAuthority | None = field(repr=False)


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
    _snapshot: SourceSnapshot
    _snapshot_digest: str
    _descriptors: tuple[ArtifactDescriptor, ...]

    def __post_init__(self) -> None:
        if self._capability is not _CAPABILITY:
            raise IdleCheckpointError("Invalid idle dependency capability")
        if type(self._token) is not IdleCaptureToken:
            raise IdleCheckpointError("Invalid idle dependency capture")
        if type(self._snapshot) is not _source_snapshot_type():
            raise IdleCheckpointError("Invalid idle dependency snapshot")
        if self._snapshot_digest != _snapshot_digest(self._snapshot):
            raise IdleCheckpointError("Invalid idle dependency snapshot seal")
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
            and self._policy_fingerprint == capture.policy_fingerprint
            and self._snapshot is not capture.snapshot
            and self._snapshot_digest == _snapshot_digest(self._snapshot)
            and self._snapshot_digest == _snapshot_digest(capture.snapshot)
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
    root_authority: SourceRootAuthority | None = field(default=None, repr=False, compare=False)
    _attestation: _IdleCaptureAttestation | object = field(
        default=_UNBOUND_ATTESTATION, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.token) is not IdleCaptureToken or type(self.snapshot) is not _source_snapshot_type():
            raise IdleCheckpointError("Invalid idle checkpoint capture")
        for name, value in (
            ("runtime", self.runtime_fingerprint),
            ("policy", self.policy_fingerprint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise IdleCheckpointError(f"Invalid idle {name} fingerprint")
        if self.dependencies is not None and type(self.dependencies) is not IdleDependencyAccess:
            raise IdleCheckpointError("Invalid idle dependency capability")
        if (
            self._attestation is not _UNBOUND_ATTESTATION
            and type(self._attestation) is not _IdleCaptureAttestation
        ):
            raise IdleCheckpointError("Invalid idle capture attestation")

    def __reduce__(self) -> object:
        raise TypeError("Idle checkpoint capture cannot be serialized")

    def __deepcopy__(self, memo: dict[int, object]) -> IdleCheckpointCapture:
        raise TypeError("Idle checkpoint capture cannot be copied")


def _validate_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IdleCheckpointError(f"Invalid idle {label} fingerprint")
    return value


def _json_digest(value: object, label: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise IdleCheckpointError(f"Invalid idle {label} seal") from None
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_digest(snapshot: SourceSnapshot) -> str:
    if type(snapshot) is not _source_snapshot_type():
        raise IdleCheckpointError("Invalid idle snapshot seal")
    try:
        state = snapshot.state
        payload = {
            "state": {
                "store_identity": state.store_identity,
                "session_key": state.session_key,
                "incarnation": state.incarnation,
                "revision": state.revision,
                "generation": state.generation,
                "checkpoint_id": state.checkpoint_id,
                "checkpoint_fingerprint": state.checkpoint_fingerprint,
            },
            "header": copy.deepcopy(snapshot.header),
            "events": copy.deepcopy(snapshot.events),
            "checkpoint": copy.deepcopy(snapshot.checkpoint),
        }
    except (AttributeError, TypeError, ValueError, RecursionError):
        raise IdleCheckpointError("Invalid idle snapshot seal") from None
    return _json_digest(payload, "snapshot")


def _descriptor_digest(descriptors: tuple[ArtifactDescriptor, ...]) -> str:
    if type(descriptors) is not tuple or any(
        type(item) is not ArtifactDescriptor for item in descriptors
    ):
        raise IdleCheckpointError("Invalid idle dependency descriptor seal")
    return _json_digest([item.to_dict() for item in descriptors], "dependency descriptor")


def _validate_capture_attestation(capture: IdleCheckpointCapture) -> None:
    """Require the factory-issued, immutable identity for one exact capture."""

    if type(capture) is not IdleCheckpointCapture:
        raise IdleCheckpointError("Invalid idle checkpoint capture")
    attestation = capture._attestation
    if (
        type(attestation) is not _IdleCaptureAttestation
        or attestation._capability is not _CAPABILITY
        or attestation._capture_id != id(capture)
        or attestation._nonce is not capture._nonce
        or attestation._token is not capture.token
        or attestation._source is not capture._source
        or attestation._runtime_fingerprint != capture.runtime_fingerprint
        or attestation._policy_fingerprint != capture.policy_fingerprint
        or attestation._root_authority is not capture.root_authority
    ):
        raise IdleCheckpointError("Idle capture attestation changed")
    if type(capture._source) is not _source_store_type():
        raise IdleCheckpointError("Idle capture source binding changed")
    store = capture._source._artifact_store
    store_binding = (
        None
        if store is None
        else (store.workspace_id, store.session_key, store.incarnation, store.generation)
    )
    if attestation._store is not store or attestation._store_binding != store_binding:
        raise IdleCheckpointError("Idle capture artifact store binding changed")
    snapshot_digest = _snapshot_digest(capture.snapshot)
    if snapshot_digest != attestation._snapshot_digest:
        raise IdleCheckpointError("Idle capture snapshot changed")
    dependencies = capture.dependencies
    if (dependencies is not None) != attestation._dependency_present:
        raise IdleCheckpointError("Idle capture dependency authority changed")
    if dependencies is None:
        if attestation._dependency_id is not None or attestation._descriptor_digest is not None:
            raise IdleCheckpointError("Idle capture dependency authority changed")
        return
    if (
        type(dependencies) is not IdleDependencyAccess
        or attestation._dependency_id != id(dependencies)
        or dependencies._source is not capture._source
        or dependencies._capture_nonce is not capture._nonce
        or dependencies._token is not capture.token
        or dependencies._store is not capture._source._artifact_store
        or dependencies._policy_fingerprint != capture.policy_fingerprint
        or dependencies._snapshot_digest != snapshot_digest
        or _snapshot_digest(dependencies._snapshot) != dependencies._snapshot_digest
        or attestation._descriptor_digest != _descriptor_digest(dependencies._descriptors)
    ):
        raise IdleCheckpointError("Idle capture dependency authority changed")


def _validate_root_authority(
    source: SourceSessionStore,
    root_authority: SourceRootAuthority | None,
) -> None:
    if root_authority is None:
        return
    if type(root_authority) is not SourceRootAuthority:
        raise IdleCheckpointError("Invalid idle source-root authority")
    try:
        root_authority.require_current(source._backend)
    except SourceRootAuthorityError:
        raise IdleCheckpointConflict("Idle source-root authority changed") from None


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


def _captured_root_descriptors(
    capture: IdleCheckpointCapture,
) -> tuple[ArtifactDescriptor, ...]:
    """Validate and return only root descriptors from the detached capture."""

    source = capture._source
    store = source._artifact_store
    if store is None:
        metadata = capture.snapshot.header.get("metadata")
        if not isinstance(metadata, dict) or "artifact_manifest" not in metadata:
            return ()
        manifest = metadata["artifact_manifest"]
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"schema_version", "descriptors"}
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or not isinstance(manifest["descriptors"], dict)
        ):
            raise ArtifactValidationError("Invalid artifact manifest schema")
        if manifest["descriptors"]:
            raise ArtifactValidationError("Idle artifact store binding is unavailable")
        return ()
    return _root_descriptors(source, capture.snapshot)


def project_idle_root_nodes(
    capture: IdleCheckpointCapture,
    nodes: tuple[object, ...],
) -> tuple[object, ...]:
    """Project captured root producer nodes through sealed artifact descriptors.

    This is the I2-to-I3c projection boundary.  It reads only the detached
    snapshot, canonical root nodes, and descriptor allowlist sealed at capture;
    it never reads source or artifact storage and returns detached source nodes
    without descriptors, stores, grants, or policy objects.
    """

    if type(capture) is not IdleCheckpointCapture:
        raise IdleCheckpointError("Invalid idle checkpoint capture")
    if type(nodes) is not tuple:
        raise IdleCheckpointError("Invalid idle root projection")
    _validate_capture_attestation(capture)

    from nanobot.session.artifact_projection import (
        render_artifact_projection,
        validate_artifact_projection,
    )
    from nanobot.session.source_projection import SourceNode, source_nodes

    if any(type(node) is not SourceNode for node in nodes):
        raise IdleCheckpointError("Invalid idle root projection")
    canonical = source_nodes(capture.snapshot.events)
    if nodes != canonical:
        raise IdleCheckpointError("Idle root projection is not canonical")

    descriptors = _captured_root_descriptors(capture)
    access = capture.dependencies
    if access is None:
        if descriptors:
            raise IdleCheckpointError("Idle dependency capability is unavailable")
        return tuple(copy.deepcopy(node) for node in canonical)
    if type(access) is not IdleDependencyAccess or not access._matches_capture(
        capture._source, capture
    ):
        raise IdleCheckpointError("Idle dependency capability binding changed")
    if tuple(access._descriptors) != descriptors:
        raise IdleCheckpointError("Idle dependency descriptor allowlist changed")

    by_event: dict[str, ArtifactDescriptor] = {}
    for descriptor in descriptors:
        if descriptor.block_index is not None or descriptor.event_id in by_event:
            raise ArtifactValidationError("Idle root artifact descriptor is ambiguous")
        by_event[descriptor.event_id] = descriptor

    projected: list[SourceNode] = []
    for canonical_node in canonical:
        descriptor = by_event.get(canonical_node.event_id)
        if descriptor is None:
            projected.append(copy.deepcopy(canonical_node))
            continue
        try:
            candidate = render_artifact_projection(canonical_node, descriptor)
            validate_artifact_projection(candidate, canonical_node, capture.snapshot)
        except (ArtifactValidationError, KeyError, TypeError, ValueError):
            raise ArtifactValidationError("Idle artifact projection is invalid") from None
        projected.append(copy.deepcopy(candidate))
    return tuple(projected)


def _capture_with_snapshot(
    source: SourceSessionStore,
    key: str,
    snapshot: SourceSnapshot,
    *,
    runtime_fingerprint: str,
    policy_fingerprint: str,
    policy: IdlePolicyBinding | None,
    runtime_checkpoint_present: bool,
    root_authority: SourceRootAuthority | None,
) -> IdleCheckpointCapture | None:
    token = capture_idle_token(snapshot, runtime_checkpoint_present=runtime_checkpoint_present)
    if token is None:
        return None
    detached_snapshot = copy.deepcopy(snapshot)
    nonce = object()
    dependencies = None
    if source._artifact_store is not None and policy is not None:
        dependency_snapshot = copy.deepcopy(detached_snapshot)
        descriptors = _root_descriptors(source, dependency_snapshot)
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
            dependency_snapshot,
            _snapshot_digest(dependency_snapshot),
            descriptors,
        )
    capture = IdleCheckpointCapture(
        token,
        detached_snapshot,
        _validate_fingerprint(runtime_fingerprint, "runtime"),
        _validate_fingerprint(policy_fingerprint, "policy"),
        source,
        nonce,
        dependencies,
        root_authority,
        _UNBOUND_ATTESTATION,
    )
    object.__setattr__(
        capture,
        "_attestation",
        _IdleCaptureAttestation(
            _CAPABILITY,
            id(capture),
            nonce,
            token,
            source,
            source._artifact_store,
            (
                None
                if source._artifact_store is None
                else (
                    source._artifact_store.workspace_id,
                    source._artifact_store.session_key,
                    source._artifact_store.incarnation,
                    source._artifact_store.generation,
                )
            ),
            capture.runtime_fingerprint,
            capture.policy_fingerprint,
            _snapshot_digest(detached_snapshot),
            dependencies is not None,
            id(dependencies) if dependencies is not None else None,
            _descriptor_digest(dependencies._descriptors) if dependencies is not None else None,
            capture.root_authority,
        ),
    )
    return capture


def capture_idle_checkpoint(
    source: SourceSessionStore,
    key: str,
    *,
    runtime_fingerprint: str,
    policy_fingerprint: str,
    policy: IdlePolicyBinding | None = None,
    root_authority: SourceRootAuthority | None = None,
) -> IdleCheckpointCapture | None:
    """Capture idle source and dependency state without holding a lock for compute."""

    if type(source) is not _source_store_type():
        raise IdleCheckpointError("Invalid idle source store")
    runtime_fingerprint = _validate_fingerprint(runtime_fingerprint, "runtime")
    policy_fingerprint = _validate_fingerprint(policy_fingerprint, "policy")
    if policy is not None and policy.fingerprint != policy_fingerprint:
        raise IdleCheckpointError("Idle policy fingerprint does not match binding")
    _validate_root_authority(source, root_authority)
    backend = source._backend
    if root_authority is None:
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
                root_authority=None,
            )
    with root_authority.operation(backend):
        with root_authority.lock_file(".session-files.lock", blocking=True):
            header, events = source._read_unlocked(key, root_authority=root_authority)
            snapshot = source._snapshot(header, events)
            return _capture_with_snapshot(
                source,
                key,
                snapshot,
                runtime_fingerprint=runtime_fingerprint,
                policy_fingerprint=policy_fingerprint,
                policy=policy,
                runtime_checkpoint_present=source._runtime_checkpoint_present(
                    key, root_authority=root_authority
                ),
                root_authority=root_authority,
            )


def _source_store_type() -> type[SourceSessionStore]:
    # Import lazily to avoid source_store -> idle_checkpoint import cycles.
    from nanobot.session.source_store import SourceSessionStore

    return SourceSessionStore


def _source_snapshot_type() -> type[SourceSnapshot]:
    # Import lazily to avoid source_store -> idle_checkpoint import cycles.
    from nanobot.session.source_store import SourceSnapshot

    return SourceSnapshot


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
    root_authority: SourceRootAuthority | None,
) -> None:
    try:
        _validate_capture_attestation(capture)
    except IdleCheckpointError:
        raise IdleCheckpointConflict("Idle checkpoint capture attestation changed") from None
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
    if capture.root_authority is not root_authority:
        raise IdleCheckpointConflict("Idle source-root authority changed")
    if root_authority is not None:
        _validate_root_authority(source, root_authority)


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
    root_authority: SourceRootAuthority | None,
) -> SourceSnapshot:
    header, events = source._read_unlocked(key, root_authority=root_authority)
    snapshot = source._snapshot(header, events)
    try:
        validate_idle_capture(
            capture.token,
            snapshot,
            runtime_checkpoint_present=source._runtime_checkpoint_present(
                key, root_authority=root_authority
            ),
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
    return source._write(header, events, root_authority=root_authority)


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
    root_authority: SourceRootAuthority | None = None,
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
        root_authority=root_authority,
    )
    validated, references, detached_proof = _candidate_and_proof(capture, candidate, proof)
    if references:
        access = capture.dependencies
        if access is None or policy is None or source._artifact_store is not access._store:
            raise IdleCheckpointError("Idle dependency access is unavailable")
        try:
            authority_context = (
                root_authority.operation(source._backend)
                if root_authority is not None
                else nullcontext()
            )
            with authority_context:
                with access._store.transaction(blocking=False) as blobs:
                    blobs.assert_reachable()
                    with execution_transaction(
                        source._backend,
                        key,
                        root_authority=root_authority,
                    ):
                        return _publish_unlocked(
                            source,
                            key,
                            capture,
                            validated,
                            references,
                            detached_proof,
                            policy=policy,
                            blobs=blobs,
                            root_authority=root_authority,
                        )
        except ArtifactStorageBusyError:
            raise ExecutionBusyError() from None
    authority_context = (
        root_authority.operation(source._backend)
        if root_authority is not None
        else nullcontext()
    )
    with authority_context:
        with execution_transaction(source._backend, key, root_authority=root_authority):
            return _publish_unlocked(
                source,
                key,
                capture,
                validated,
                references,
                detached_proof,
                policy=policy,
                blobs=None,
                root_authority=root_authority,
            )


__all__ = [
    "IdleCheckpointCapture",
    "IdleCheckpointConflict",
    "IdleCheckpointConflictError",
    "IdleCheckpointError",
    "IdleDependencyAccess",
    "IdlePolicyBinding",
    "capture_idle_checkpoint",
    "project_idle_root_nodes",
    "publish_idle_checkpoint",
]
