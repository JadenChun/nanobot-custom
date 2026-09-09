"""Exact-source validation; committing a checkpoint remains a store responsibility."""

from copy import deepcopy

import pytest

from nanobot.session.context_checkpoint import validate_context_checkpoint
from nanobot.session.source import SourceValidationError, new_source_event, prefix_digest


def fixture():
    events = [new_source_event({"role": "user", "content": "original"}, run_id="run")]
    candidate = {
        "schema_version": 1, "digest_version": 1, "summary_format_version": 1,
        "session_generation": 2, "source_revision": 3,
        "checkpoint_id": "checkpoint", "parent_checkpoint_id": None,
        "producing_model": "model", "producing_provider": "provider",
        "digest_algorithm": "sha256", "created_at": "2026-09-09T00:00:00+00:00",
        "summary": "Retained facts", "artifact_references": [], "boundary": 1,
        "boundary_event_id": events[0]["_event_id"], "prefix_digest": prefix_digest(events, 1),
        "auxiliary_usage": {"tokens": 20},
    }
    return events, candidate


def validate(events, candidate):
    return validate_context_checkpoint(
        candidate, events, generation=2, source_revision=3, parent_checkpoint_id=None
    )


def test_validation_is_detached_and_never_edits_source():
    events, candidate = fixture()
    before = deepcopy(events)
    result = validate(events, candidate)
    result["auxiliary_usage"]["tokens"] = 100
    assert candidate["auxiliary_usage"]["tokens"] == 20
    assert events == before


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("digest_version", 2), ("summary_format_version", 0),
    ("session_generation", 3), ("source_revision", 4), ("parent_checkpoint_id", "other"),
    ("checkpoint_id", ""), ("producing_model", ""), ("producing_provider", None),
    ("digest_algorithm", "unknown"), ("created_at", "2026-09-09"),
    ("created_at", "bad"), ("summary", " "), ("summary", "x" * 64001),
    ("artifact_references", ["untrusted"]), ("boundary", True), ("boundary", 2),
    ("boundary_event_id", "other"), ("prefix_digest", "bad"),
    ("auxiliary_usage", {"tokens": float("nan")}),
])
def test_invalid_checkpoint_rejected(field, value):
    events, candidate = fixture()
    candidate[field] = value
    with pytest.raises(SourceValidationError):
        validate(events, candidate)


def test_internal_source_mutation_invalidates_checkpoint():
    events, candidate = fixture()
    events[0]["latency"] = 10
    with pytest.raises(SourceValidationError):
        validate(events, candidate)


@pytest.mark.parametrize("usage", [{"tokens": -1}, {"tokens": True}, {"tokens": "2"}, {}])
def test_invalid_usage_rejected(usage):
    events, candidate = fixture()
    candidate["auxiliary_usage"] = usage
    with pytest.raises(SourceValidationError):
        validate(events, candidate)


def test_required_fields_and_explicit_unknown_usage():
    events, candidate = fixture()
    candidate["auxiliary_usage"] = None
    assert validate(events, candidate)["auxiliary_usage"] is None
    for field in candidate:
        partial = dict(candidate)
        del partial[field]
        with pytest.raises(SourceValidationError):
            validate(events, partial)


@pytest.mark.parametrize("candidate", [None, [], "text", 1])
def test_malformed_candidate_root_is_content_free(candidate):
    events, _ = fixture()
    with pytest.raises(SourceValidationError, match="^Invalid context checkpoint schema$"):
        validate(events, candidate)
