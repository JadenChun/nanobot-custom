"""Dormant, detached idle-compaction admission state.

This module deliberately does not schedule work, select a model, or publish a
checkpoint.  It records the exact durable state an eventual idle publisher must
revalidate under its checkpoint transaction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from nanobot.session.source import SourceValidationError, canonical_events, validate_execution_links

if TYPE_CHECKING:
    from nanobot.session.source_store import SourceSnapshot


USER_ACTIVITY_REVISION_KEY = "user_activity_revision"
LAST_USER_ACTIVITY_AT_KEY = "last_user_activity_at"


class IdleCaptureConflictError(SourceValidationError):
    """An idle capture no longer describes the durable source state."""


@dataclass(frozen=True)
class UserActivity:
    """Strictly typed foreground-user activity metadata."""

    revision: int
    last_user_activity_at: str


@dataclass(frozen=True)
class IdlePendingState:
    """Content-free exact binding for state that prevents idle work."""

    execution_lease_present: bool
    execution_recovery_required: bool
    runtime_checkpoint_present: bool
    pending_user_turn_present: bool
    pending_user_followups_present: bool
    webui_recovery_present: bool
    unresolved_execution_ids: tuple[str, ...]
    fingerprint: str

    @property
    def idle_eligible(self) -> bool:
        return not (
            self.execution_lease_present
            or self.execution_recovery_required
            or self.runtime_checkpoint_present
            or self.pending_user_turn_present
            or self.pending_user_followups_present
            or self.webui_recovery_present
            or self.unresolved_execution_ids
        )


@dataclass(frozen=True)
class IdleCaptureToken:
    """A detached snapshot binding for one prospective idle computation."""

    store_identity: str
    session_key: str
    incarnation: str
    source_revision: int
    generation: int
    source_fingerprint: str
    checkpoint_id: str | None
    checkpoint_revision: int | None
    checkpoint_fingerprint: str
    checkpoint_valid: bool
    user_activity_revision: int
    last_user_activity_at: str
    pending_state: IdlePendingState


@dataclass(frozen=True)
class IdleCheckpointAdmission:
    """Non-mutating validation evidence for a future integration-owned commit."""

    capture: IdleCaptureToken
    candidate_fingerprint: str


def _counter(value: object, *, message: str) -> int:
    if type(value) is not int or value < 0:
        raise SourceValidationError(message)
    return value


def read_user_activity(metadata: object) -> UserActivity | None:
    """Read activity fields without filling in legacy source sessions.

    A source session written before this dormant substrate has neither field.
    It remains ineligible until the foreground admission path records activity.
    Partial or malformed new metadata is rejected rather than overwritten.
    """

    if not isinstance(metadata, dict):
        raise SourceValidationError("Invalid idle activity metadata")
    revision_present = USER_ACTIVITY_REVISION_KEY in metadata
    timestamp_present = LAST_USER_ACTIVITY_AT_KEY in metadata
    if not revision_present and not timestamp_present:
        return None
    if not revision_present or not timestamp_present:
        raise SourceValidationError("Invalid idle activity metadata")
    revision = _counter(metadata[USER_ACTIVITY_REVISION_KEY], message="Invalid idle activity metadata")
    timestamp = metadata[LAST_USER_ACTIVITY_AT_KEY]
    if not isinstance(timestamp, str) or not timestamp:
        raise SourceValidationError("Invalid idle activity metadata")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        raise SourceValidationError("Invalid idle activity metadata") from None
    if parsed.tzinfo is None:
        raise SourceValidationError("Invalid idle activity metadata")
    return UserActivity(revision, timestamp)


def _canonical_fingerprint(value: object) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise SourceValidationError("Invalid idle durable state") from None
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_binding(snapshot: SourceSnapshot) -> tuple[str | None, int | None, bool]:
    metadata = snapshot.header.get("metadata")
    raw = metadata.get("context_checkpoint") if isinstance(metadata, dict) else None
    if raw is None:
        return None, None, True
    if snapshot.checkpoint is None:
        return snapshot.state.checkpoint_id, None, False
    revision = snapshot.checkpoint.get("source_revision")
    if type(revision) is not int or revision < 0:
        # SourceSnapshot normally validates this already.  Keep the idle
        # substrate conservative if a future reader changes that contract.
        return snapshot.state.checkpoint_id, None, False
    return snapshot.state.checkpoint_id, revision, True


def durable_idle_state(snapshot: SourceSnapshot, *, runtime_checkpoint_present: bool) -> IdlePendingState:
    """Detach every durable flag that makes an idle checkpoint unsafe."""

    metadata = snapshot.header.get("metadata")
    if not isinstance(metadata, dict):
        raise SourceValidationError("Invalid idle durable state")
    unresolved = tuple(sorted(validate_execution_links(snapshot.events)))
    values = {
        "execution_lease": metadata.get("execution_lease") if "execution_lease" in metadata else None,
        "execution_recovery_required": (
            metadata.get("execution_recovery_required")
            if "execution_recovery_required" in metadata
            else None
        ),
        "pending_user_turn": metadata.get("pending_user_turn") if "pending_user_turn" in metadata else None,
        "pending_user_followups": (
            metadata.get("pending_user_followups") if "pending_user_followups" in metadata else None
        ),
        "webui_recovery": metadata.get("webui_recovery") if "webui_recovery" in metadata else None,
        "runtime_checkpoint_present": runtime_checkpoint_present,
        "unresolved_execution_ids": unresolved,
    }
    return IdlePendingState(
        execution_lease_present="execution_lease" in metadata,
        execution_recovery_required="execution_recovery_required" in metadata,
        runtime_checkpoint_present=runtime_checkpoint_present,
        pending_user_turn_present="pending_user_turn" in metadata,
        pending_user_followups_present="pending_user_followups" in metadata,
        webui_recovery_present="webui_recovery" in metadata,
        unresolved_execution_ids=unresolved,
        fingerprint=_canonical_fingerprint(values),
    )


def capture_idle_token(
    snapshot: SourceSnapshot, *, runtime_checkpoint_present: bool
) -> IdleCaptureToken | None:
    """Create a detached token; legacy sessions with no activity stay ineligible."""

    if not isinstance(runtime_checkpoint_present, bool):
        raise SourceValidationError("Invalid idle durable state")
    activity = read_user_activity(snapshot.header.get("metadata"))
    if activity is None:
        return None
    checkpoint_id, checkpoint_revision, checkpoint_valid = _checkpoint_binding(snapshot)
    return IdleCaptureToken(
        store_identity=snapshot.state.store_identity,
        session_key=snapshot.state.session_key,
        incarnation=snapshot.state.incarnation,
        source_revision=snapshot.state.revision,
        generation=snapshot.state.generation,
        source_fingerprint=hashlib.sha256(canonical_events(snapshot.events)).hexdigest(),
        checkpoint_id=checkpoint_id,
        checkpoint_revision=checkpoint_revision,
        checkpoint_fingerprint=snapshot.state.checkpoint_fingerprint,
        checkpoint_valid=checkpoint_valid,
        user_activity_revision=activity.revision,
        last_user_activity_at=activity.last_user_activity_at,
        pending_state=durable_idle_state(
            snapshot, runtime_checkpoint_present=runtime_checkpoint_present
        ),
    )


def validate_idle_capture(
    token: IdleCaptureToken, snapshot: SourceSnapshot, *, runtime_checkpoint_present: bool
) -> None:
    """Require an exact, still-idle durable state before a checkpoint commit."""

    if type(token) is not IdleCaptureToken:
        raise SourceValidationError("Invalid idle capture token")
    current = capture_idle_token(
        snapshot, runtime_checkpoint_present=runtime_checkpoint_present
    )
    if current is None or current != token:
        raise IdleCaptureConflictError("Idle capture token no longer matches source state")
    if not current.checkpoint_valid or not current.pending_state.idle_eligible:
        raise IdleCaptureConflictError("Idle source state is not eligible for compaction")


def checkpoint_candidate_fingerprint(candidate: object) -> str:
    """Bind an admission result to the exact proposal, without publishing it."""

    if not isinstance(candidate, dict):
        raise SourceValidationError("Invalid idle checkpoint candidate")
    return hashlib.sha256(canonical_events([candidate])).hexdigest()
