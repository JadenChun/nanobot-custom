"""Regression tests for replaying persisted delegated-child transcript events."""

from __future__ import annotations

from typing import Any

from nanobot.webui.transcript import (
    replay_transcript_to_ui_messages,
    validate_retained_transcript_record,
)


def _field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def _delegated_rows(messages: list[Any]) -> list[Any]:
    return [message for message in messages if _field(message, "kind") == "delegated_child"]


def test_replay_collapses_child_lifecycle_to_one_row() -> None:
    lines = [
        {
            "event": "user",
            "text": "do the task",
            "turn_id": "turn-1",
            "turn_phase": "user",
            "turn_seq": 1,
            "created_at_ms": 100,
        },
        {
            "event": "delegated_child",
            "child_id": "child-a",
            "child_session_key": "delegated:child-a",
            "profile": "delegated",
            "state": "active",
            "turn_id": "turn-1",
            "runtime_generation": 0,
            "created_at_ms": 110,
        },
        {
            "event": "delegated_child",
            "child_id": "child-a",
            "child_session_key": "delegated:child-a",
            "profile": "delegated",
            "state": "completed",
            "outcome": "completed",
            "turn_id": "turn-1",
            "runtime_generation": 0,
            "created_at_ms": 120,
        },
        {
            "event": "stream_end",
            "text": "done",
            "turn_id": "turn-1",
            "turn_phase": "answer",
            "turn_seq": 2,
            "created_at_ms": 130,
        },
    ]

    messages = replay_transcript_to_ui_messages(lines)

    rows = _delegated_rows(messages)
    assert len(rows) == 1, f"expected exactly one delegated_child row, got {len(rows)}"

    row = rows[0]
    assert _field(row, "delegatedChild") == {
        "child_session_key": "delegated:child-a",
        "child_id": "child-a",
        "profile": "delegated",
        "state": "completed",
        "outcome": "completed",
        "runtime_generation": 0,
        "turn_id": "turn-1",
    }
    assert _field(row, "turnId") == "turn-1"
    assert _field(row, "id") == "delegated-child-child-a"

    assert any(_field(message, "content") == "done" for message in messages)
    assert _field(messages[-1], "content") == "done"


def test_replay_preserves_durable_child_state() -> None:
    lines = [
        {
            "event": "delegated_child",
            "child_id": "child-a",
            "child_session_key": "delegated:child-a",
            "profile": "delegated",
            "state": "active",
            "runtime_generation": 1,
            "parent_run_id": "parent-run",
            "child_run_id": "child-run",
            "parent_store_identity": "store",
            "parent_incarnation": "parent-inc",
            "parent_generation": 2,
            "root_key": "websocket:parent",
            "root_incarnation": "root-inc",
            "root_generation": 2,
            "child_incarnation": "child-inc",
            "child_generation": 3,
            "run_sequence": 2,
            "state_sequence": 5,
            "control_epoch": 1,
            "holds_capacity": True,
            "can_stop": True,
            "owned_work": True,
            "pending_approval_id": "approval",
            "blocked_run_id": "child-run",
            "result_disposition": "pending",
            "turn_id": "turn-1",
            "created_at_ms": 110,
        },
    ]

    messages = replay_transcript_to_ui_messages(lines)
    rows = _delegated_rows(messages)
    assert len(rows) == 1
    child = _field(rows[0], "delegatedChild")
    assert child["parent_run_id"] == "parent-run"
    assert child["child_run_id"] == "child-run"
    assert child["parent_incarnation"] == "parent-inc"
    assert child["root_key"] == "websocket:parent"
    assert child["child_incarnation"] == "child-inc"
    assert child["run_sequence"] == 2
    assert child["state_sequence"] == 5
    assert child["control_epoch"] == 1
    assert child["holds_capacity"] is True
    assert child["can_stop"] is True
    assert child["owned_work"] is True
    assert child["pending_approval_id"] == "approval"
    assert child["blocked_run_id"] == "child-run"
    assert child["result_disposition"] == "pending"


def test_retained_delegated_row_accepts_durable_fields() -> None:
    validate_retained_transcript_record({
        "event": "delegated_child",
        "child_id": "child-a",
        "child_session_key": "delegated:child-a",
        "profile": "delegated",
        "state": "active",
        "runtime_generation": 1,
        "parent_run_id": "parent-run",
        "child_run_id": "child-run",
        "parent_incarnation": "parent-inc",
        "root_key": "websocket:parent",
        "child_incarnation": "child-inc",
        "run_sequence": 2,
        "state_sequence": 5,
        "control_epoch": 1,
        "holds_capacity": True,
        "can_stop": True,
        "owned_work": True,
        "pending_approval_id": "approval",
        "blocked_run_id": "child-run",
        "result_disposition": "pending",
    })


def test_replay_keeps_distinct_rows_for_two_children() -> None:
    lines = [
        {
            "event": "user",
            "text": "do the task",
            "turn_id": "turn-2",
            "turn_phase": "user",
            "turn_seq": 1,
            "created_at_ms": 200,
        },
        {
            "event": "delegated_child",
            "child_id": "child-a",
            "child_session_key": "delegated:child-a",
            "profile": "delegated",
            "state": "active",
            "turn_id": "turn-2",
            "runtime_generation": 0,
            "created_at_ms": 210,
        },
        {
            "event": "delegated_child",
            "child_id": "child-b",
            "child_session_key": "delegated:child-b",
            "profile": "delegated",
            "state": "active",
            "turn_id": "turn-2",
            "runtime_generation": 0,
            "created_at_ms": 220,
        },
        {
            "event": "stream_end",
            "text": "done",
            "turn_id": "turn-2",
            "turn_phase": "answer",
            "turn_seq": 2,
            "created_at_ms": 230,
        },
    ]

    messages = replay_transcript_to_ui_messages(lines)

    rows = _delegated_rows(messages)
    assert len(rows) == 2, f"expected one delegated_child row per child, got {len(rows)}"

    by_child_id = {_field(row, "delegatedChild")["child_id"]: row for row in rows}
    assert set(by_child_id) == {"child-a", "child-b"}

    for child_id in ("child-a", "child-b"):
        row = by_child_id[child_id]
        assert _field(row, "id") == f"delegated-child-{child_id}"
        assert _field(row, "turnId") == "turn-2"
        assert _field(row, "delegatedChild")["child_session_key"] == f"delegated:{child_id}"
        assert _field(row, "delegatedChild")["profile"] == "delegated"
        assert _field(row, "delegatedChild")["state"] == "active"

    assert len({_field(row, "id") for row in rows}) == 2
    assert _field(messages[-1], "content") == "done"
