"""Source identity and exact-prefix evidence for dormant P4 persistence."""

from copy import deepcopy

import pytest

from nanobot.session.source import (
    SourceValidationError,
    canonical_events,
    merge_source_events,
    new_source_event,
    prefix_digest,
    validate_closed_boundary,
    validate_execution_links,
)


def event(text="same"):
    return new_source_event({"role": "user", "content": text}, run_id="run")


def test_trusted_identity_detaches_and_strips_provider_attribution():
    payload = {"content": ["text"], "_event_id": "fake", "_run_id": "fake",
               "_execution_id": "fake", "_execution_ids": ["fake"]}
    result = new_source_event(payload, run_id="trusted")
    payload["content"].append("later")
    assert result["content"] == ["text"]
    assert result["_run_id"] == "trusted"
    assert result["_event_id"] != "fake"
    assert "_execution_id" not in result and "_execution_ids" not in result


def test_equal_content_is_distinct_but_same_identity_retry_is_idempotent():
    first, second = event(), event()
    assert first["_event_id"] != second["_event_id"]
    result = merge_source_events([first], [first, second])
    assert result == [first, second]
    result[0]["content"] = "changed"
    assert first["content"] == "same"
    assert merge_source_events([first, second], [first, second]) == [first, second]


def test_identity_reuse_cannot_edit_committed_evidence():
    first = event()
    changed = dict(first, content="changed")
    with pytest.raises(SourceValidationError, match="Conflicting"):
        merge_source_events([first], [changed])


def test_retry_cannot_reorder_or_duplicate_evidence():
    first, second, third = event(), event(), event()
    for incoming in ([second, first], [first, third], [third, second], [second, second]):
        with pytest.raises(SourceValidationError):
            merge_source_events([first, second], incoming)


def test_digest_exact_internal_fields_and_unicode_without_normalization():
    first = {"content": "é", "_event_id": "one"}
    assert canonical_events([first]) == '[{"_event_id":"one","content":"é"}]'.encode()
    assert prefix_digest([first], 1) == prefix_digest([dict(reversed(list(first.items())))], 1)
    for changed in (dict(first, content="e\u0301"), dict(first, latency=1)):
        assert prefix_digest([first], 1) != prefix_digest([changed], 1)
    with pytest.raises(SourceValidationError):
        canonical_events([{"content": float("nan")}])
    with pytest.raises(SourceValidationError):
        prefix_digest([first], True)


def test_execution_links_allow_repeated_provider_ids_but_not_split_groups():
    rows = []
    for execution in ("exec-one", "exec-two"):
        rows.extend([
            {"role": "assistant", "tool_calls": [{"id": "same-provider-id"}],
             "_execution_ids": [execution]},
            {"role": "tool", "tool_call_id": "same-provider-id", "_execution_id": execution},
        ])
    validate_closed_boundary(rows, 4)
    validate_closed_boundary(rows, 2)
    with pytest.raises(SourceValidationError, match="Unresolved"):
        validate_closed_boundary(rows, 3)
    broken = deepcopy(rows)
    broken[3]["_execution_id"] = "exec-one"
    with pytest.raises(SourceValidationError):
        validate_closed_boundary(broken, 4)
    del broken[0]["_execution_ids"]
    with pytest.raises(SourceValidationError):
        validate_closed_boundary(broken, 2)


@pytest.mark.parametrize("value", [{2: "a", 10: "b"}, ("tuple",), {"nested": {1: "bad"}}])
def test_canonical_representation_rejects_non_json_types(value):
    with pytest.raises(SourceValidationError):
        canonical_events([{"content": value}])


@pytest.mark.parametrize("calls", [None, False, 0, "", {}, [None]])
def test_malformed_tool_calls_are_rejected_even_when_falsy(calls):
    with pytest.raises(SourceValidationError):
        validate_closed_boundary([
            {"role": "assistant", "tool_calls": calls, "_execution_ids": []},
        ], 1)


def test_result_must_match_provider_call_as_well_as_execution_identity():
    with pytest.raises(SourceValidationError, match="provenance"):
        validate_closed_boundary([
            {"role": "assistant", "tool_calls": [{"id": "a"}], "_execution_ids": ["exec"]},
            {"role": "tool", "tool_call_id": "b", "_execution_id": "exec"},
        ], 2)


@pytest.mark.parametrize("payload", [None, [], "text", 1])
def test_malformed_payload_root_is_content_free(payload):
    with pytest.raises(SourceValidationError, match="^Invalid source event representation$"):
        new_source_event(payload, run_id="run")


def test_recursive_payload_is_content_free():
    payload = {}
    payload["nested"] = payload
    with pytest.raises(SourceValidationError, match="^Invalid source representation$"):
        new_source_event(payload, run_id="run")


@pytest.mark.parametrize("call", [None, [], {}, {"id": None}, {"id": 1}, {"id": ""}, {"id": " "}])
def test_execution_call_identity_is_validated_before_use(call):
    with pytest.raises(SourceValidationError, match="^Invalid source provider call identity$"):
        validate_execution_links([
            {"role": "assistant", "tool_calls": [call], "_execution_ids": ["execution"]},
        ])


@pytest.mark.parametrize("identities", [None, {}, [], [None], [False], [""], [" "]])
def test_execution_identity_container_and_values_are_checked(identities):
    with pytest.raises(SourceValidationError):
        validate_execution_links([
            {"role": "assistant", "tool_calls": [{"id": "provider"}], "_execution_ids": identities},
        ])
