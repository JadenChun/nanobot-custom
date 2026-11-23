"""Current-context completion metadata does not change canonical answer replay."""
import pytest

from kohaku.webui.transcript import replay_transcript_to_ui_messages


@pytest.mark.parametrize("text", ["firstsecondfinal", "# Rewritten\n![plot](/signed/image.png)\nFinal answer"])
def test_cumulative_replay_keeps_text_once_and_completion_snapshot_on_notice(text) -> None:
    records = [
        {"event": "user", "text": "hello", "turn_id": "t", "turn_seq": 0},
        {"event": "stream_end", "text": "first", "stream_id": "s", "resuming": True, "merge_next": True, "turn_id": "t", "turn_seq": 1},
        {"event": "context_compaction", "compaction_id": "c", "phase": "succeeded", "turn_id": "t", "turn_seq": 2},
        {"event": "stream_end", "text": text, "stream_id": "s", "turn_id": "t", "turn_seq": 3},
        {"event": "turn_end", "turn_id": "t", "turn_seq": 4, "usage": {"prompt_tokens": 12000, "context_tokens": 2000},
         "context_usage": {"context_tokens": 2000, "context_window_tokens": 12000}},
    ]
    rows = replay_transcript_to_ui_messages(records)
    answers = [m["content"] for m in rows if m["role"] == "assistant" and m.get("kind") != "compaction"]
    assert answers == [text]
    carrier = next(m for m in rows if m.get("contextUsage"))
    assert carrier["turnId"] == "t"
    assert carrier["contextUsage"] == {"contextTokens": 2000, "contextWindowTokens": 12000, "completionTurnSeq": 4}


def test_context_only_completion_uses_exact_user_not_prior_assistant() -> None:
    records = [
        {"event": "message", "text": "old", "turn_id": "old", "turn_seq": 1},
        {"event": "user", "text": "new", "turn_id": "new", "turn_seq": 0},
        {"event": "turn_end", "turn_id": "new", "turn_seq": 2, "latency_ms": 100,
         "usage": {"context_tokens": 9000}, "context_usage": {"context_tokens": None, "context_window_tokens": None}},
    ]
    rows = replay_transcript_to_ui_messages(records)
    assert "contextUsage" not in rows[0]
    assert "usage" not in rows[0]
    assert rows[-1]["role"] == "user"
    assert rows[-1]["contextUsage"] == {"contextTokens": None, "contextWindowTokens": None, "completionTurnSeq": 2}
    assert "usage" not in rows[-1]
    assert "latencyMs" not in rows[-1]


@pytest.mark.parametrize("value", [None, {}, {"context_tokens": True, "context_window_tokens": 12000}])
def test_malformed_present_context_is_explicit_unavailable_not_aggregate(value) -> None:
    rows = replay_transcript_to_ui_messages([
        {"event": "user", "text": "hello", "turn_id": "t"},
        {"event": "message", "text": "answer", "turn_id": "t"},
        {"event": "turn_end", "turn_id": "t", "turn_seq": 3, "usage": {"context_tokens": 9000}, "context_usage": value},
    ])
    assert rows[-1]["usage"]["context_tokens"] == 9000
    assert rows[-1]["contextUsage"] == {"contextTokens": None, "contextWindowTokens": None, "completionTurnSeq": 3}


@pytest.mark.parametrize("turn_id", [None, "other"])
def test_snapshot_cannot_cross_or_restamp_a_different_user_boundary(turn_id) -> None:
    rows = replay_transcript_to_ui_messages([
        {"event": "message", "text": "old", "turn_id": "old"},
        {"event": "user", "text": "new", "turn_id": "new"},
        {"event": "turn_end", "turn_id": turn_id, "turn_seq": 3,
         "context_usage": {"context_tokens": 2000, "context_window_tokens": 12000}},
    ])
    assert all("contextUsage" not in row for row in rows)


def test_no_slot_is_omitted_and_retained_snapshots_validate() -> None:
    from kohaku.bus.outbound_events import TurnEndEvent
    from kohaku.providers.base import ContextUsageSnapshot
    from kohaku.webui.outbound_wire import encode_turn_end
    from kohaku.webui.transcript import (
        validate_retained_transcript_record,
        validate_retained_webui_thread,
    )

    assert "context_usage" not in encode_turn_end("t", TurnEndEvent())
    wire = dict(encode_turn_end("t", TurnEndEvent(context_usage=ContextUsageSnapshot(None, None))))
    assert wire["context_usage"] == {"context_tokens": None, "context_window_tokens": None}
    validate_retained_transcript_record(wire)
    rows = replay_transcript_to_ui_messages([
        {"event": "user", "text": "hello", "turn_id": "t"}, {**wire, "turn_id": "t", "turn_seq": 3},
    ])
    validate_retained_webui_thread({"schemaVersion": 3, "messages": rows})
    with pytest.raises(ValueError, match="context usage"):
        validate_retained_transcript_record({**wire, "context_usage": {}})


@pytest.mark.parametrize("sequence", [True, -1, 1.5, "3", float("inf")])
def test_invalid_boundary_sequence_is_not_repaired_into_provenance(sequence) -> None:
    rows = replay_transcript_to_ui_messages([
        {"event": "context_compaction", "compaction_id": "c", "phase": "succeeded", "turn_id": "t", "turn_seq": sequence},
        {"event": "turn_end", "turn_id": "t", "turn_seq": sequence,
         "context_usage": {"context_tokens": 2000, "context_window_tokens": 12000}},
    ])
    assert "turnSeq" not in rows[0]
    assert "completionTurnSeq" not in rows[0]["contextUsage"]
