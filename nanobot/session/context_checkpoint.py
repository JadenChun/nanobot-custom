"""Validation for durable context checkpoints; no projection or storage effects."""

from copy import deepcopy
from datetime import datetime
from typing import Any

from nanobot.session.source import (
    SourceValidationError,
    canonical_events,
    prefix_digest,
    validate_closed_boundary,
    validate_event_ids,
)

MAX_SUMMARY_CHARS = 64_000


def validate_context_checkpoint(
    candidate: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    generation: int,
    source_revision: int,
    parent_checkpoint_id: str | None,
) -> dict[str, Any]:
    """Validate a candidate against the exact locked source snapshot.

    The storage owner must still compare revision/generation/parent with disk
    and atomically commit. This function grants no artifact authorization.
    Invalid persisted checkpoints may be ignored by the reader; invalid source
    evidence must propagate as a storage failure instead.
    """
    if not isinstance(candidate, dict):
        raise SourceValidationError("Invalid context checkpoint schema")
    canonical_events([candidate])
    validate_event_ids(events)
    required = {
        "schema_version", "digest_version", "summary_format_version", "session_generation",
        "source_revision", "checkpoint_id", "parent_checkpoint_id", "producing_model",
        "producing_provider", "digest_algorithm", "created_at", "summary",
        "artifact_references", "boundary", "boundary_event_id", "prefix_digest", "auxiliary_usage",
    }
    if not required.issubset(candidate):
        raise SourceValidationError("Invalid context checkpoint schema")
    # None explicitly denotes unavailable usage, without fabricating zero cost.
    usage = candidate["auxiliary_usage"]
    if usage is not None and (
        not isinstance(usage, dict)
        or not usage
        or any(type(value) is not int or value < 0 for value in usage.values())
    ):
        raise SourceValidationError("Invalid context checkpoint auxiliary usage")
    version = candidate.get('schema_version')
    if type(version) is not int or version not in (1, 2):
        raise SourceValidationError('Invalid context checkpoint version')
    for name, expected in (
        ("schema_version", version), ("digest_version", 1), ("summary_format_version", 1),
        ("session_generation", generation), ("source_revision", source_revision),
    ):
        value = candidate.get(name)
        if type(value) is not int or value < 0 or value != expected:
            raise SourceValidationError("Invalid context checkpoint version or state")
    for name in ("checkpoint_id", "producing_model", "producing_provider"):
        value = candidate.get(name)
        if not isinstance(value, str) or not value.strip():
            raise SourceValidationError("Invalid context checkpoint identity")
    if (
        candidate.get("parent_checkpoint_id") != parent_checkpoint_id
        or candidate["checkpoint_id"] == parent_checkpoint_id
        or candidate.get("digest_algorithm") != "sha256"
    ):
        raise SourceValidationError("Invalid context checkpoint provenance")
    created_at = candidate.get("created_at")
    try:
        if not isinstance(created_at, str) or datetime.fromisoformat(created_at).tzinfo is None:
            raise ValueError
    except ValueError:
        raise SourceValidationError("Invalid context checkpoint timestamp") from None
    summary = candidate.get("summary")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > MAX_SUMMARY_CHARS:
        raise SourceValidationError("Invalid context checkpoint summary")
    references = candidate.get('artifact_references')
    if version == 1 and references != []:
        raise SourceValidationError('Version-1 context checkpoint cannot carry artifacts')
    if not isinstance(references, list) or any(not isinstance(value, str) for value in references):
        raise SourceValidationError('Invalid context checkpoint artifact references')
    if references != sorted(set(references)):
        raise SourceValidationError('Context checkpoint references must be sorted and unique')
    if references:
        from nanobot.session.artifact import parse_artifact_reference

        for reference in references:
            parse_artifact_reference(reference)
    boundary = candidate.get("boundary")
    digest = prefix_digest(events, boundary)
    if (
        candidate.get("prefix_digest") != digest
        or candidate.get("boundary_event_id") != (
            events[boundary - 1]["_event_id"] if boundary else None
        )
    ):
        raise SourceValidationError("Invalid context checkpoint source prefix")
    validate_closed_boundary(events, boundary)
    # Also reject non-JSON auxiliary fields, NaN, and invalid Unicode before a write.
    canonical_events([candidate])
    return deepcopy(candidate)
