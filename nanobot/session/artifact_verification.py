"""Shared, owner-independent verification of already published artifacts.

This module only verifies an artifact descriptor, its canonical source receipt,
current filesystem policy, and its durable private blob.  It does not grant a
caller a reader or a tool.  Invocation readers retain their own execution-owner
and stream checks before calling this helper; dormant idle publication uses a
sealed allowlist in :mod:`nanobot.session.idle_checkpoint`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from nanobot.agent.tools.read_provenance import FilesystemReadReceipt, revalidate_read_receipt
from nanobot.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
    attested_artifact_origin,
    parse_artifact_descriptor,
    published_artifacts,
)
from nanobot.session.artifact_store import ArtifactBlobTransaction, PrivateArtifactStore

if TYPE_CHECKING:
    from nanobot.context_repo.policy import ResourceAccessPolicy
    from nanobot.session.source_store import SourceSnapshot


def verify_published_artifact(
    descriptor: ArtifactDescriptor,
    snapshot: SourceSnapshot,
    blobs: ArtifactBlobTransaction,
    *,
    store: PrivateArtifactStore,
    workspace: Path | None,
    policy: ResourceAccessPolicy | None,
    resolve_read: Callable[[str], Path],
    root_history: bool = False,
) -> FilesystemReadReceipt:
    """Verify one currently published descriptor under a held blob transaction.

    The helper deliberately has no execution-owner parameter and performs no
    reads through a filesystem tool.  ``workspace``, ``policy``, and
    ``resolve_read`` are the current runtime policy inputs used only to
    revalidate the receipt recorded by the canonical producer.  A caller must
    supply a configured private store transaction; the descriptor and source
    snapshot cannot select one.
    """

    if not isinstance(descriptor, ArtifactDescriptor):
        raise ArtifactValidationError("Invalid artifact verification input")
    if type(blobs) is not ArtifactBlobTransaction or blobs._store is not store:
        raise ArtifactValidationError("Artifact transaction does not match configured store")
    if not callable(resolve_read):
        raise ArtifactValidationError("Artifact verification lacks a current path resolver")

    descriptor = parse_artifact_descriptor(descriptor.to_dict())
    state = snapshot.state
    if (
        descriptor.workspace_id != store.workspace_id
        or descriptor.session_key != store.session_key
        or descriptor.incarnation != store.incarnation
        or descriptor.generation != store.generation
        or descriptor.session_key != state.session_key
        or descriptor.incarnation != state.incarnation
        or descriptor.generation != state.generation
    ):
        raise ArtifactValidationError("Artifact source admission changed")

    event = next((row for row in snapshot.events if row.get("_event_id") == descriptor.event_id), None)
    if event is None or (root_history and event.get("_parent_stream_id") is not None):
        raise ArtifactValidationError("Artifact is outside the admitted history")

    receipt = attested_artifact_origin(
        descriptor, snapshot, workspace_id=store.workspace_id
    )
    revalidate_read_receipt(
        receipt,
        workspace=workspace,
        policy=policy,
        resolve_read=resolve_read,
    )
    if descriptor not in published_artifacts(snapshot, workspace_id=store.workspace_id):
        raise ArtifactValidationError("Artifact is not published")
    blobs.verify_durable(descriptor, store=store)
    blobs.assert_reachable()
    return receipt


__all__ = ["verify_published_artifact"]
