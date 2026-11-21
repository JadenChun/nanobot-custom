"""Bounded strict Stop wire parsing and content-free pending inventory digests."""

from copy import deepcopy
from dataclasses import replace

import pytest

from nanobot.agent.stop_target import (
    MAX_STOP_TARGET_PENDING_RECORDS,
    StopSourceOrigin,
    StopTargetUnavailableError,
    parse_stop_expectation,
    pending_work_digest,
)
from nanobot.channels.websocket.runtime import _parse_envelope
from nanobot.session.pending_followups import PendingChildFollowup, PendingFollowupRecord


def target_value():
    return {
        "version": 1,
        "source": {"store_identity": "store", "session_key": "websocket:chat", "incarnation": "life", "generation": 0},
        "control_epoch": 0,
        "active_root_present": True,
        "active_turn_id": None,
        "active_root_run_ids": ["root-run"],
        "children": [],
        "pending_work_digest": "0" * 64,
    }


@pytest.mark.parametrize("field,value", [
    ("version", True), ("version", 2), ("control_epoch", True),
    ("control_epoch", -1), ("control_epoch", 2 ** 53), ("active_root_present", 1),
    ("active_turn_id", ""), ("active_turn_id", "a\x00b"),
    ("active_root_run_ids", []), ("active_root_run_ids", ["same", "same"]),
    ("active_root_run_ids", ["z", "a"]), ("active_root_run_ids", [True]),
    ("active_root_run_ids", ["bad/id"]), ("active_root_run_ids", ["x"] * 65),
    ("children", {}), ("children", [None]),
    ("pending_work_digest", "A" * 64), ("pending_work_digest", "0" * 63),
])
def test_expectation_refuses_invalid_types_shapes_and_sets(field, value):
    raw = target_value()
    raw[field] = value
    with pytest.raises(ValueError):
        parse_stop_expectation(raw)


@pytest.mark.parametrize("field,value", [
    ("generation", True), ("generation", -1), ("incarnation", ""),
    ("store_identity", "x" * 129), ("session_key", "../escape"),
])
def test_expectation_refuses_invalid_source(field, value):
    raw = target_value()
    raw["source"][field] = value
    with pytest.raises(ValueError):
        parse_stop_expectation(raw)


@pytest.mark.parametrize("change", ["missing", "unknown", "absent-runs", "absent-turn", "child-duplicate", "child-unsorted", "child-incomplete", "child-bool"])
def test_expectation_refuses_incomplete_or_ambiguous_identity(change):
    raw = target_value()
    child = {"child_id": "a", "source": deepcopy(raw["source"]), "child_run_id": None, "run_sequence": 0, "control_epoch": 0}
    if change == "missing":
        del raw["source"]["incarnation"]
    elif change == "unknown":
        raw["extra"] = None
    elif change.startswith("absent"):
        raw["active_root_present"] = False
        if change == "absent-turn":
            raw["active_root_run_ids"] = []
            raw["active_turn_id"] = "turn"
    else:
        raw["children"] = [child]
        if change == "child-duplicate":
            raw["children"].append(deepcopy(child))
        elif change == "child-unsorted":
            raw["children"].insert(0, {**child, "child_id": "z"})
        elif change == "child-incomplete":
            del child["child_run_id"]
        else:
            child["run_sequence"] = True
    with pytest.raises(ValueError):
        parse_stop_expectation(raw)


def test_nested_duplicate_json_fields_are_not_erased_before_validation():
    import json

    payload = json.dumps(target_value()).replace('"generation": 0', '"generation": 0, "generation": 0')
    envelope = _parse_envelope('{"type":"message","stop_expectation":' + payload + '}')
    assert envelope is not None
    with pytest.raises(ValueError, match="duplicate"):
        parse_stop_expectation(envelope["stop_expectation"])


def test_root_presence_shapes_round_trip_and_do_not_retain_input_aliases():
    raw = target_value()
    expected = parse_stop_expectation(raw)
    raw["active_root_run_ids"].clear()
    assert expected.active_root_run_ids == ("root-run",)
    absent = {**target_value(), "active_root_present": False, "active_root_run_ids": []}
    assert parse_stop_expectation(absent).to_dict() == absent


def test_pending_digest_is_sorted_exact_content_free_and_bounded():
    source = StopSourceOrigin("store", "websocket:chat", "life", 0)
    one = PendingFollowupRecord("1" * 32, "user", "chat", "private content", (), {})
    two = replace(one, identifier="2" * 32)
    digest = pending_work_digest([(source, (one, two))])
    assert digest == pending_work_digest([(source, (two, one))])
    assert digest == pending_work_digest([(source, (replace(one, content="changed"), two))])
    assert digest != pending_work_digest([(replace(source, generation=1), (one, two))])
    assert digest != pending_work_digest([(source, (replace(one, identifier="3" * 32), two))])
    with pytest.raises(StopTargetUnavailableError):
        pending_work_digest([(source, (one,) * (MAX_STOP_TARGET_PENDING_RECORDS + 1))])


def test_pending_digest_binds_typed_child_intent_and_stop_epoch():
    source = StopSourceOrigin("store", "delegated:child", "child-life", 0)
    intent = PendingChildFollowup(
        child_id="1" * 32, parent_key="websocket:parent", parent_store_identity="store",
        parent_incarnation="parent-life", parent_generation=0, parent_run_id="parent-run",
        parent_tool_event_id="tool-event", parent_execution_id="execution",
        parent_dispatch_ordinal=1, child_key="delegated:child", child_store_identity="store",
        child_incarnation="child-life", child_generation=0, child_run_id="child-run",
        child_state_sequence=1, child_control_epoch=0,
    )
    record = PendingFollowupRecord("2" * 32, "user", "parent", "private input", (), {}, intent)
    digest = pending_work_digest([(source, (record,))])
    assert digest == pending_work_digest([(source, (replace(record, content="other content"),))])
    for changed in (
        replace(intent, child_control_epoch=1),
        replace(intent, child_run_id="new-run"),
        replace(intent, parent_tool_event_id="new-intent"),
    ):
        assert digest != pending_work_digest([(source, (replace(record, child_followup=changed),))])
