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
from contextlib import nullcontext as nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NoReturn, Protocol, cast

from nanobot.agent.tools.read_provenance import FilesystemReadReceipt
from nanobot.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
)
from nanobot.session.artifact import (
    parse_artifact_descriptor as parse_artifact_descriptor,
)
from nanobot.session.artifact import (
    published_artifacts as published_artifacts,
)
from nanobot.session.artifact_store import ArtifactBlobTransaction, PrivateArtifactStore
from nanobot.session.artifact_store import ArtifactStorageBusyError as ArtifactStorageBusyError
from nanobot.session.artifact_verification import verify_published_artifact
from nanobot.session.checkpoint_dependencies import (
    SummaryDependencyProof,
    derive_summary_references,
)
from nanobot.session.context_checkpoint import validate_context_checkpoint
from nanobot.session.execution_owner import ExecutionBusyError as ExecutionBusyError
from nanobot.session.execution_owner import execution_transaction as execution_transaction
from nanobot.session.idle_compaction import IdleCaptureConflictError as IdleCaptureConflictError
from nanobot.session.idle_compaction import IdleCaptureToken, capture_idle_token
from nanobot.session.idle_compaction import validate_idle_capture as validate_idle_capture
from nanobot.session.source import SourceValidationError
from nanobot.session.source_root import SourceRootAuthority
from nanobot.session.source_root import (
    SourceRootAuthorityError as SourceRootAuthorityError,
)

if TYPE_CHECKING:
    from nanobot.context_repo.policy import ResourceAccessPolicy
    from nanobot.session.source_projection import SourceNode
    from nanobot.session.source_store import SourceSessionStore, SourceSnapshot


_CAPABILITY = object()
_UNBOUND_ATTESTATION = object()


class _IdleSourceBinding(Protocol):
    @property
    def source_store(self) -> object: ...


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class IdleSourceIdentityObservation:
    """One opaque identity operand, not an admission or storage capability.

    The observation retains the source read at its caller's original argument
    boundary. It can only compare identity; all attestation, live transport and
    current-binding checks remain mandatory at their existing boundaries.
    """

    _source: object = field(repr=False)

    def __repr__(self) -> str:
        return "<idle-source-identity observation>"

    def matches_source(self, source: object) -> bool:
        return self._source is source

    def __reduce__(self) -> NoReturn:
        raise TypeError("Idle source identity observation cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("Idle source identity observation cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("Idle source identity observation cannot be copied")


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

    def require_capture(
        self,
        capture: IdleCheckpointCapture,
    ) -> None:
        """Validate sealed fields through the exact capture's original live reads."""

        if (
            self._capability is not _CAPABILITY
            or self._capture_id != id(capture)
            or not IdleCheckpointCapture.matches_nonce(capture, self._nonce)
            or self._token is not capture.token
            or not IdleCheckpointCapture.matches_source(capture, self._source)
            or self._runtime_fingerprint != capture.runtime_fingerprint
            or self._policy_fingerprint != capture.policy_fingerprint
            or self._root_authority is not capture.root_authority
        ):
            raise IdleCheckpointError("Idle capture attestation changed")
        IdleCheckpointCapture.require_source_kind(capture)
        IdleCheckpointCapture.require_artifact_binding(capture, self._store, self._store_binding)
        snapshot_digest = _snapshot_digest(capture.snapshot)
        if snapshot_digest != self._snapshot_digest:
            raise IdleCheckpointError("Idle capture snapshot changed")
        dependencies = capture.dependencies
        if (dependencies is not None) != self._dependency_present:
            raise IdleCheckpointError("Idle capture dependency authority changed")
        if dependencies is None:
            if self._dependency_id is not None or self._descriptor_digest is not None:
                raise IdleCheckpointError("Idle capture dependency authority changed")
            return
        if type(dependencies) is not IdleDependencyAccess or self._dependency_id != id(dependencies):
            raise IdleCheckpointError("Idle capture dependency authority changed")
        IdleDependencyAccess.require_sealed_binding(
            dependencies, capture, snapshot_digest=snapshot_digest,
            descriptor_digest=self._descriptor_digest,
        )


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
        if not isinstance(cast(object, self.fingerprint), str) or not self.fingerprint.strip():
            raise IdleCheckpointError("Invalid idle policy fingerprint")
        if not callable(self.resolve_read):
            raise IdleCheckpointError("Invalid idle policy resolver")
        if self.workspace is not None and not isinstance(cast(object, self.workspace), Path):
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

    def __reduce__(self) -> NoReturn:
        raise TypeError("Idle dependency access cannot be serialized")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("Idle dependency access cannot be copied")

    def _matches_capture(self, source: SourceSessionStore, capture: IdleCheckpointCapture) -> bool:
        from nanobot.session.source_store import SourceSessionStore

        return (
            self._source is source
            and IdleCheckpointCapture.matches_nonce(capture, self._capture_nonce)
            and self._token is capture.token
            and SourceSessionStore.matches_configured_artifact_store(source, self._store)
            and self._store.workspace_id == self._workspace_id
            and self._store.session_key == self._session_key
            and self._store.incarnation == self._incarnation
            and self._store.generation == self._generation
            and self._policy_fingerprint == capture.policy_fingerprint
            and self._snapshot is not capture.snapshot
            and self._snapshot_digest == _snapshot_digest(self._snapshot)
            and self._snapshot_digest == _snapshot_digest(capture.snapshot)
        )

    def matches_capture(self, source: SourceSessionStore, capture: IdleCheckpointCapture) -> bool:
        """Retain the live capture matcher without exposing dependency authority."""
        return self._matches_capture(source, capture)

    def differs_from_summary_binding(
        self, binding: _IdleSourceBinding, policy: IdlePolicyBinding,
    ) -> bool:
        """Compare source then policy through their original live operand reads."""
        return (
            self._source is not binding.source_store
            or self._policy_fingerprint != policy.fingerprint
        )

    def descriptor_digest(self) -> str:
        """Hash the sealed descriptors through the current helper, returning no grants."""
        return _descriptor_digest(self._descriptors)

    def descriptors_differ(self, descriptors: tuple[ArtifactDescriptor, ...]) -> bool:
        """Retain tuple conversion and comparison without returning the allowlist."""
        return tuple(self._descriptors) != descriptors

    def require_sealed_binding(
        self,
        capture: IdleCheckpointCapture,
        *,
        snapshot_digest: str,
        descriptor_digest: str | None,
    ) -> None:
        """Validate dependency fields after the capture's exact-type/id checks."""

        if (
            not IdleCheckpointCapture.matches_source(capture, self._source)
            or not IdleCheckpointCapture.matches_nonce(capture, self._capture_nonce)
            or self._token is not capture.token
            or not IdleCheckpointCapture.matches_artifact_store(capture, self._store)
            or self._policy_fingerprint != capture.policy_fingerprint
            or self._snapshot_digest != snapshot_digest
            or _snapshot_digest(self._snapshot) != self._snapshot_digest
            or descriptor_digest != _descriptor_digest(self._descriptors)
        ):
            raise IdleCheckpointError("Idle capture dependency authority changed")

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

    def matches_configured_store(
        self,
        store: PrivateArtifactStore | None,
    ) -> bool:
        """Compare the exact configured store without exposing sealed authority."""

        return store is self._store

    def publish_in_artifact_transaction(
        self,
        source: SourceSessionStore,
        key: str,
        capture: IdleCheckpointCapture,
        validated: dict[str, object],
        references: tuple[str, ...],
        proof: SummaryDependencyProof | None,
        *,
        policy: IdlePolicyBinding,
        root_authority: SourceRootAuthority | None,
    ) -> SourceSnapshot:
        """Publish through one held artifact transaction and fixed source operation."""

        from nanobot.session.source_store import SourceSessionStore

        with self._store.transaction(blocking=False) as blobs:
            blobs.assert_reachable()
            return SourceSessionStore.publish_idle_checkpoint_in_execution_transaction(
                source,
                key,
                capture,
                validated,
                references,
                proof,
                policy=policy,
                blobs=blobs,
                root_authority=root_authority,
            )

    @staticmethod
    def verify_captured_references(
        capture: IdleCheckpointCapture,
        references: tuple[str, ...],
        snapshot: SourceSnapshot,
        proof: SummaryDependencyProof | None,
        blobs: ArtifactBlobTransaction | None,
        policy: IdlePolicyBinding | None,
    ) -> None:
        """Verify captured references under the existing artifact and source locks."""

        if proof is None or blobs is None or capture.dependencies is None or policy is None:
            raise IdleCheckpointError("Idle dependency access is unavailable")
        for reference in references:
            capture.dependencies._verify(reference, snapshot, blobs, policy)
        blobs.assert_reachable()


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
            if not isinstance(cast(object, value), str) or not value.strip():
                raise IdleCheckpointError(f"Invalid idle {name} fingerprint")
        if self.dependencies is not None and type(self.dependencies) is not IdleDependencyAccess:
            raise IdleCheckpointError("Invalid idle dependency capability")
        if (
            self._attestation is not _UNBOUND_ATTESTATION
            and type(self._attestation) is not _IdleCaptureAttestation
        ):
            raise IdleCheckpointError("Invalid idle capture attestation")

    def __reduce__(self) -> NoReturn:
        raise TypeError("Idle checkpoint capture cannot be serialized")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("Idle checkpoint capture cannot be copied")

    def require_attestation(self) -> None:
        """Require the factory-issued, immutable identity for one exact capture."""

        if type(self) is not IdleCheckpointCapture:
            raise IdleCheckpointError("Invalid idle checkpoint capture")
        attestation = self._attestation
        if type(attestation) is not _IdleCaptureAttestation:
            raise IdleCheckpointError("Idle capture attestation changed")
        _IdleCaptureAttestation.require_capture(attestation, self)

    def matches_nonce(self, nonce: object) -> bool:
        """Observe exact nonce identity without exposing the sealed nonce."""
        return self._nonce is nonce

    def matches_source(self, source: SourceSessionStore) -> bool:
        """Observe exact source identity without returning its authority."""
        return self._source is source

    def matches_binding(self, binding: _IdleSourceBinding) -> bool:
        """Retain the binding-before-capture source identity comparison."""
        return binding.source_store is self._source

    def observe_source_identity(self) -> IdleSourceIdentityObservation:
        """Read one live source operand without returning usable source authority."""
        return IdleSourceIdentityObservation(self._source)

    def matches_dependencies(self, dependencies: IdleDependencyAccess) -> bool:
        """Compare dependency authority against this capture's current source."""
        return dependencies.matches_capture(self._source, self)

    def require_source_kind(self) -> None:
        """Retain the original source-class rejection and live helper lookup."""
        if type(self._source) is not _source_store_type():
            raise IdleCheckpointError("Idle capture source binding changed")

    def require_artifact_binding(
        self,
        expected_store: PrivateArtifactStore | None,
        expected_binding: tuple[str, str, str, int] | None,
    ) -> None:
        """Check the original live source's configured binding without returning it."""
        from nanobot.session.source_store import SourceSessionStore

        SourceSessionStore.require_idle_artifact_binding(self._source, expected_store, expected_binding)

    def matches_artifact_store(self, store: PrivateArtifactStore | None) -> bool:
        """Observe the original live source's store identity, not admission."""
        from nanobot.session.source_store import SourceSessionStore

        return SourceSessionStore.matches_configured_artifact_store(self._source, store)

    def captured_root_descriptors(self) -> tuple[ArtifactDescriptor, ...]:
        """Read captured root descriptors through the selected source, not its store."""
        source = self._source
        from nanobot.session.source_store import SourceSessionStore

        return SourceSessionStore.captured_idle_root_descriptors(source, self)

    @staticmethod
    def admit_source_capture(
        source: SourceSessionStore,
        *,
        runtime_fingerprint: str,
        policy_fingerprint: str,
        policy: IdlePolicyBinding | None,
        root_authority: SourceRootAuthority | None,
    ) -> tuple[str, str]:
        """Validate the existing source capture admission before storage access."""

        if type(source) is not _source_store_type():
            raise IdleCheckpointError("Invalid idle source store")
        runtime_fingerprint = _validate_fingerprint(runtime_fingerprint, "runtime")
        policy_fingerprint = _validate_fingerprint(policy_fingerprint, "policy")
        if policy is not None and policy.fingerprint != policy_fingerprint:
            raise IdleCheckpointError("Idle policy fingerprint does not match binding")
        _validate_root_authority(source, root_authority)
        return runtime_fingerprint, policy_fingerprint

    @staticmethod
    def issue_from_snapshot(
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
        """Issue one sealed idle capture from a lock-held source snapshot."""

        token = capture_idle_token(snapshot, runtime_checkpoint_present=runtime_checkpoint_present)
        if token is None:
            return None
        detached_snapshot = copy.deepcopy(snapshot)
        nonce = object()
        from nanobot.session.source_store import SourceSessionStore

        return SourceSessionStore.issue_idle_capture_artifacts(
            source, IdleCaptureIssuer(_CAPABILITY, source), snapshot, token, detached_snapshot, nonce,
            runtime_fingerprint=runtime_fingerprint,
            policy_fingerprint=policy_fingerprint,
            policy=policy,
            root_authority=root_authority,
        )

    @staticmethod
    def prepare_source_publication(
        source: SourceSessionStore,
        key: str,
        capture: IdleCheckpointCapture,
        candidate: dict[str, object],
        *,
        runtime_fingerprint: str,
        policy_fingerprint: str,
        policy: IdlePolicyBinding | None,
        proof: SummaryDependencyProof | None,
        root_authority: SourceRootAuthority | None,
    ) -> tuple[dict[str, object], tuple[str, ...], SummaryDependencyProof | None]:
        """Detach and validate the existing publication proposal before transactions."""

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
        return validated, references, detached_proof


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class IdleCaptureIssuer:
    """Source-bound construction service, never a publication or storage proof."""

    _capability: object = field(repr=False)
    _source: SourceSessionStore = field(repr=False)

    def __post_init__(self) -> None:
        if type(self) is not IdleCaptureIssuer or self._capability is not _CAPABILITY:
            raise IdleCheckpointError("Invalid idle capture issuer")

    def __repr__(self) -> str:
        return "<idle capture issuer>"

    def __reduce__(self) -> NoReturn:
        raise TypeError("Idle capture issuer cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("Idle capture issuer cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("Idle capture issuer cannot be copied")

    def require_source(self, source: SourceSessionStore) -> None:
        if (
            type(self) is not IdleCaptureIssuer
            or self._capability is not _CAPABILITY
            or self._source is not source
        ):
            raise IdleCheckpointError("Invalid idle capture issuer")

    @staticmethod
    def root_descriptors(source: SourceSessionStore, snapshot: SourceSnapshot) -> tuple[ArtifactDescriptor, ...]:
        return _root_descriptors(source, snapshot)

    @staticmethod
    def snapshot_digest(snapshot: SourceSnapshot) -> str:
        return _snapshot_digest(snapshot)

    def dependency_constructor(
        self, source: SourceSessionStore, nonce: object, token: IdleCaptureToken,
    ) -> Callable[[PrivateArtifactStore, str, str, str, int, str, SourceSnapshot, str, tuple[ArtifactDescriptor, ...]], IdleDependencyAccess]:
        """Select the real constructor and prefix before its live store operands."""
        IdleCaptureIssuer.require_source(self, source)
        constructor, capability = IdleDependencyAccess, _CAPABILITY

        def construct(
            store: PrivateArtifactStore, workspace_id: str, session_key: str,
            incarnation: str, generation: int, policy_fingerprint: str,
            snapshot: SourceSnapshot, snapshot_digest: str, descriptors: tuple[ArtifactDescriptor, ...],
        ) -> IdleDependencyAccess:
            return constructor(
                capability, source, nonce, token, store, workspace_id, session_key,
                incarnation, generation, policy_fingerprint, snapshot, snapshot_digest, descriptors,
            )

        return construct

    def make_capture(
        self, source: SourceSessionStore, token: IdleCaptureToken, detached_snapshot: SourceSnapshot,
        runtime_fingerprint: str, policy_fingerprint: str, nonce: object,
        dependencies: IdleDependencyAccess | None, root_authority: SourceRootAuthority | None,
    ) -> IdleCheckpointCapture:
        IdleCaptureIssuer.require_source(self, source)
        return IdleCheckpointCapture(
            token, detached_snapshot,
            _validate_fingerprint(runtime_fingerprint, "runtime"),
            _validate_fingerprint(policy_fingerprint, "policy"),
            source, nonce, dependencies, root_authority, _UNBOUND_ATTESTATION,
        )

    def attestation_constructor(
        self, source: SourceSessionStore, capture: IdleCheckpointCapture, nonce: object,
        token: IdleCaptureToken,
    ) -> Callable[[PrivateArtifactStore | None, tuple[str, str, str, int] | None, SourceSnapshot, IdleDependencyAccess | None], None]:
        """Keep the attestation constructor selection before configured-store reads."""
        IdleCaptureIssuer.require_source(self, source)
        bind_attestation = object.__setattr__
        constructor, capability, capture_id = _IdleCaptureAttestation, _CAPABILITY, id(capture)

        def construct(
            store: PrivateArtifactStore | None, store_binding: tuple[str, str, str, int] | None,
            detached_snapshot: SourceSnapshot, dependencies: IdleDependencyAccess | None,
        ) -> None:
            bind_attestation(
                capture, "_attestation",
                constructor(
                    capability, capture_id, nonce, token, source, store, store_binding,
                    capture.runtime_fingerprint, capture.policy_fingerprint,
                    _snapshot_digest(detached_snapshot), dependencies is not None,
                    id(dependencies) if dependencies is not None else None,
                    dependencies.descriptor_digest() if dependencies is not None else None,
                    capture.root_authority,
                ),
            )

        return construct


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

    return IdleCheckpointCapture.require_attestation(capture)


def _validate_root_authority(
    source: SourceSessionStore,
    root_authority: SourceRootAuthority | None,
) -> None:
    from nanobot.session.source_store import SourceSessionStore

    return SourceSessionStore.require_idle_root_authority(source, root_authority)


def _root_descriptors(
    source: SourceSessionStore,
    snapshot: SourceSnapshot,
) -> tuple[ArtifactDescriptor, ...]:
    """Return only current manifest descriptors from root-history events."""

    from nanobot.session.source_store import SourceSessionStore

    return SourceSessionStore.idle_root_descriptors(source, snapshot)


def _captured_root_descriptors(
    capture: IdleCheckpointCapture,
) -> tuple[ArtifactDescriptor, ...]:
    """Validate and return only root descriptors from the detached capture."""

    return IdleCheckpointCapture.captured_root_descriptors(capture)


def select_idle_root_descriptor_reader() -> Callable[[SourceSessionStore, SourceSnapshot], tuple[ArtifactDescriptor, ...]]:
    """Select the original late descriptor reader, not storage or admission authority."""
    return _root_descriptors


def project_idle_root_nodes(
    capture: IdleCheckpointCapture,
    nodes: tuple[object, ...],
) -> tuple[SourceNode, ...]:
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
    if type(access) is not IdleDependencyAccess or not IdleCheckpointCapture.matches_dependencies(
        capture, access
    ):
        raise IdleCheckpointError("Idle dependency capability binding changed")
    if access.descriptors_differ(descriptors):
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
    return IdleCheckpointCapture.issue_from_snapshot(
        source,
        key,
        snapshot,
        runtime_fingerprint=runtime_fingerprint,
        policy_fingerprint=policy_fingerprint,
        policy=policy,
        runtime_checkpoint_present=runtime_checkpoint_present,
        root_authority=root_authority,
    )


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

    from nanobot.session.source_store import SourceSessionStore

    return SourceSessionStore.capture_idle_checkpoint(
        source,
        key,
        runtime_fingerprint=runtime_fingerprint,
        policy_fingerprint=policy_fingerprint,
        policy=policy,
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
        detached_candidate = copy.deepcopy(cast(dict[object, object], candidate))
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
    from nanobot.session.source_store import SourceSessionStore

    if (
        type(capture) is not IdleCheckpointCapture
        or not IdleCheckpointCapture.matches_source(capture, source)
        or SourceSessionStore.idle_capture_identity_differs(source, capture.token)
        or capture.token.session_key != key
        or capture.runtime_fingerprint != runtime_fingerprint
        or capture.policy_fingerprint != policy_fingerprint
    ):
        raise IdleCheckpointConflict("Idle checkpoint capture binding changed")
    if capture.dependencies is not None and not capture.dependencies.matches_capture(source, capture):
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
    from nanobot.session.source_store import SourceSessionStore

    return SourceSessionStore.commit_idle_checkpoint_unlocked(
        source,
        key,
        capture,
        validated,
        references,
        proof,
        policy=policy,
        blobs=blobs,
        root_authority=root_authority,
    )


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

    from nanobot.session.source_store import SourceSessionStore

    return SourceSessionStore.publish_idle_checkpoint(
        source,
        key,
        capture,
        candidate,
        runtime_fingerprint=runtime_fingerprint,
        policy_fingerprint=policy_fingerprint,
        policy=policy,
        proof=proof,
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
    # Legacy direct-call compatibility; callers retain all required capture
    # locks or publication transactions and applicable root scope.
    "_capture_with_snapshot",
    "_publish_unlocked",
]
