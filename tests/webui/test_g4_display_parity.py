"""Gate G4: display-fact provenance, live-consumer parity, and reversible rollback.

This suite compares the canonical session source with the WebUI display
transcript for every required Gate G4 category. It does not change production
behavior. It records which display facts are reconstructible from the canonical
source alone and which are distinct display-only facts that Package 5 must keep.

Every case uses ``tmp_path`` isolation and the real record writers, readers, and
index. No fixture copies an implementation branch.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest

from kohaku.bus.events import CommittedUserInput
from kohaku.session.execution_owner import acquire_execution
from kohaku.session.input_attempt import find_input_attempt
from kohaku.session.manager import JsonlSessionStore, Session
from kohaku.session.source import new_source_event
from kohaku.session.source_store import SourceState
from kohaku.session.turn_writer import SourceTurnWriter
from kohaku.webui import transcript as tr
from kohaku.webui.metadata import WEBUI_DISPLAY_SOURCE_KEY, WEBUI_TURN_METADATA_KEY
from kohaku.webui.transcript import (
    WEBUI_FORK_MARKER_EVENT,
    WebUITranscriptRecorder,
    _bounded_tool_event,
    _delegated_child_lifetime,
    _filter_covered_file_edit_tool_events,
    _media_from_signed_urls,
    _prompt_id_from_record,
    _record_origin,
    _recover_incomplete_turns,
    _repair_unfinished_tail,
    _sanitize_turn_usage,
    _session_backfill_turns,
    active_transcript_path,
    append_transcript_object,
    build_webui_thread_response,
    completed_turn_ids,
    delete_webui_transcript_result,
    fork_boundary_message_count,
    has_pending_tool_calls,
    has_unfinished_transcript_tail,
    read_transcript_lines,
    replay_transcript_to_ui_messages,
    rewrite_local_markdown_images,
    validate_retained_transcript_record,
    validate_retained_webui_thread,
    webui_transcript_segments_dir,
)
from kohaku.webui.transcript_index import (
    delete_transcript_index,
    read_history_window,
    read_preindex_latest_tail,
    read_prompt_summary_range,
    rebuild_transcript_index,
    transcript_index_dir,
)
from tests.agent.test_input_attempt_reconciliation import isolate_projection as isolate_projection
from tests.agent.test_migration_engine import engine_config as engine_config

KEY = "websocket:g4-parity"
CHAT_ID = "g4-parity"


# ---------------------------------------------------------------------------
# Isolation and small builders
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_webui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import kohaku.webui.thread_disk  # noqa: F401 - bind the real dir before patching

    webui_dir = tmp_path / "webui"
    webui_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("kohaku.config.paths.get_webui_dir", lambda: webui_dir)
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("kohaku.webui.transcript.get_webui_dir", lambda: webui_dir)
    return webui_dir


def _state(
    key: str = KEY,
    *,
    store_identity: str = "store-g4",
    incarnation: str = "inc-g4",
    revision: int = 1,
    generation: int = 0,
) -> SourceState:
    return SourceState(
        store_identity=store_identity,
        session_key=key,
        incarnation=incarnation,
        revision=revision,
        generation=generation,
        checkpoint_id=None,
        checkpoint_fingerprint="",
    )


def _append(session_key: str, source: SourceState, **row: Any) -> None:
    """Append one display row bound to *source* with deterministic timestamps."""
    record: dict[str, Any] = dict(row)
    record.setdefault("chat_id", CHAT_ID)
    record.setdefault("created_at_ms", 1_700_000_000_000)
    record["_webui_source_origin"] = _record_origin(source)
    append_transcript_object(session_key, record)


def _texts(messages: list[dict[str, Any]]) -> list[str]:
    return [str(m.get("content") or "") for m in messages]


def _replay(session_key: str, state: SourceState | None = None) -> list[dict[str, Any]]:
    return replay_transcript_to_ui_messages(
        read_transcript_lines(session_key, expected_source=state)
    )


async def _canonical(
    tmp_path: Path,
    session_key: str = KEY,
    *,
    run_id: str = "g4-run",
) -> tuple[JsonlSessionStore, SourceState]:
    """Create a canonical session owned by the real JSONL store."""
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    session = Session(session_key)
    backend.save(session)
    snapshot = backend.source.upgrade_source_session(
        session_key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    async with acquire_execution(backend, session_key, run_id=run_id) as owner:
        owner.validate_dispatch()
    return backend, snapshot.state


# ---------------------------------------------------------------------------
# 1. Prompt, answer, turn identity, and admission
# ---------------------------------------------------------------------------


async def test_prompt_answer_and_turn_identity_match_canonical_admission(
    isolated_webui: Path,
    tmp_path: Path,
) -> None:
    backend, state = await _canonical(tmp_path)
    recorder = WebUITranscriptRecorder(log=None)
    metadata = {
        WEBUI_DISPLAY_SOURCE_KEY: state,
        WEBUI_TURN_METADATA_KEY: "turn-g4-1",
    }

    prompt = "Summarize the release notes"
    answer = "The release adds bounded history reads."

    async with acquire_execution(backend, KEY, run_id="g4-turn") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="g4-turn")
        user_event = writer.append_message({"role": "user", "content": prompt})
        writer.append_message({"role": "assistant", "content": answer})
        committed = owner.read_snapshot()

    assert recorder.append_committed_user_message(
        CHAT_ID,
        CommittedUserInput(
            content=prompt,
            media_paths=(),
            attachments=(),
            created_at_ms=1_700_000_000_000,
            event_id=str(user_event["_event_id"]),
        ),
        metadata,
    )
    assert recorder.prepare_and_append(
        CHAT_ID,
        {"event": "stream_end", "chat_id": CHAT_ID, "text": answer},
        metadata=metadata,
        phase="answer",
    )
    assert recorder.prepare_and_append(
        CHAT_ID,
        {"event": "turn_end", "chat_id": CHAT_ID, "turn_id": "turn-g4-1"},
        metadata=metadata,
        phase="complete",
    )

    canonical_user = next(row for row in committed.events if row.get("role") == "user")
    canonical_answer = next(row for row in committed.events if row.get("role") == "assistant")

    rows = read_transcript_lines(KEY, expected_source=state)
    display_user = next(row for row in rows if row.get("event") == "user")
    display_answer = next(row for row in rows if row.get("event") == "stream_end")

    # The display prompt identity is the canonical admitted event id, and the
    # displayed content is the canonical content, not a second decision.
    assert display_user["_webui_source_prompt_id"] == canonical_user["_event_id"]
    assert display_user["text"] == canonical_user["content"] == prompt
    assert display_answer["text"] == canonical_answer["content"] == answer

    page = build_webui_thread_response(KEY, expected_source=state, limit=10, direction="latest")
    assert page is not None
    assert [m["role"] for m in page["messages"]] == ["user", "assistant"]
    assert page["messages"][0]["promptId"] == canonical_user["_event_id"]
    assert page["messages"][0]["turnId"] == "turn-g4-1"
    assert page["messages"][1]["turnId"] == "turn-g4-1"
    assert page["completed_turn_ids"] == ["turn-g4-1"]


async def test_display_row_is_not_proof_of_canonical_admission(
    isolated_webui: Path,
    tmp_path: Path,
) -> None:
    backend, state = await _canonical(tmp_path)
    canonical = backend.source.load_source_snapshot(KEY)
    assert canonical.events == []

    # A display publisher can write a prompt row before any canonical input.
    _append(
        KEY,
        state,
        event="user",
        text="never admitted",
        turn_id="turn-unadmitted",
        _webui_source_prompt_id="forged-event-id",
    )

    rows = read_transcript_lines(KEY, expected_source=state)
    assert [row["text"] for row in rows if row.get("event") == "user"] == ["never admitted"]
    assert _replay(KEY, state)[0]["promptId"] == "forged-event-id"

    # Canonical evidence does not corroborate the display identity.
    assert canonical.events == []
    attempt_evidence = {
        "version": 1,
        "turn_id": "turn-unadmitted",
        "source": {
            "store_identity": state.store_identity,
            "session_key": state.session_key,
            "incarnation": state.incarnation,
            "generation": state.generation,
        },
        "fingerprint": "0" * 64,
        "principal": "0" * 64,
        "intent": "new",
    }
    from kohaku.session.input_attempt import InputAttempt

    assert find_input_attempt(canonical.events, InputAttempt.parse_evidence(attempt_evidence)) is None
    # Nothing in the canonical projection claims this prompt.
    assert all(row.get("content") != "never admitted" for row in canonical.events)


# ---------------------------------------------------------------------------
# 2. Permitted reasoning detail and redaction
# ---------------------------------------------------------------------------


async def test_permitted_reasoning_replays_while_private_reasoning_is_never_persisted(
    isolated_webui: Path,
    tmp_path: Path,
) -> None:
    backend, state = await _canonical(tmp_path)

    async with acquire_execution(backend, KEY, run_id="g4-reasoning") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="g4-reasoning")
        writer.append_message({"role": "user", "content": "explain"})
        writer.append_message({
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "raw private chain"},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "reasoning_content", "text": "raw provider reasoning"},
                {"type": "text", "text": "Permitted answer."},
            ],
            "reasoning_content": "raw provider reasoning",
            "thinking_blocks": [{"text": "raw private chain"}],
        })
        committed = owner.read_snapshot()

    persisted = next(e for e in committed.events if e.get("role") == "assistant")
    serialized = json.dumps(persisted, ensure_ascii=False)
    for private in ("raw private chain", "raw provider reasoning", "opaque"):
        assert private not in serialized
    assert "Permitted answer." in json.dumps(persisted["content"], ensure_ascii=False)
    assert "reasoning_content" not in persisted
    assert "thinking_blocks" not in persisted

    # Only a permitted, display-supplied reasoning summary survives replay.
    _append(KEY, state, event="user", text="explain", _webui_source_prompt_id="evt-permitted")
    _append(KEY, state, event="reasoning_end", text="Checked the two constraints.")
    _append(KEY, state, event="stream_end", text="Permitted answer.")
    _append(KEY, state, event="turn_end", turn_id="turn-permitted")

    messages = _replay(KEY, state)
    assert messages[1]["reasoning"] == "Checked the two constraints."
    assert "raw provider reasoning" not in json.dumps(messages, ensure_ascii=False)

    page = build_webui_thread_response(KEY, expected_source=state, limit=10, direction="latest")
    assert page is not None
    retained = {key: value for key, value in page.items() if key != "page"}
    validate_retained_webui_thread(cast(dict[str, object], retained))
    for row in read_transcript_lines(KEY, expected_source=state):
        validate_retained_transcript_record(cast(dict[str, object], row))
    replayed_rows = cast(
        list[dict[str, object]], cast(dict[str, object], retained)["messages"]
    )
    assert any(row.get("reasoning") == "Checked the two constraints." for row in replayed_rows)
    assert all("raw private chain" not in json.dumps(row) for row in replayed_rows)


# ---------------------------------------------------------------------------
# 3. Bounded tool traces, approvals, and file-edit diffs
# ---------------------------------------------------------------------------


def test_bounded_tool_events_redact_inline_payloads_and_oversized_results() -> None:
    data_url = "data:image/png;base64," + "A" * 9000
    oversized = "R" * 12000
    event = {
        "version": 1,
        "phase": "end",
        "call_id": "call-1",
        "name": "read_file",
        "arguments": {"path": "report.txt"},
        "result": [oversized, {"type": "image_url", "image_url": {"url": data_url}}],
        "error": None,
    }

    bounded = _bounded_tool_event(dict(event))

    assert bounded.get("result_truncated") is True
    dumped = json.dumps(bounded, ensure_ascii=False)
    assert data_url not in dumped
    assert "<omitted" in dumped
    assert len(dumped) <= 16 * 1024 + 1024
    # Start-phase events without a large field stay identical.
    start = {"phase": "start", "call_id": "call-1", "name": "read_file", "arguments": {"path": "a"}}
    assert _bounded_tool_event(dict(start)) == start


def test_file_edit_display_suppresses_duplicate_tool_trace_events() -> None:
    events = [
        {"phase": "start", "call_id": "edit-1", "name": "edit_file", "arguments": {"path": "a.py"}},
        {"phase": "end", "call_id": "edit-1", "name": "edit_file", "result": "ok"},
        {"phase": "end", "call_id": "exec-1", "name": "exec", "result": "pytest"},
    ]
    messages = [{
        "role": "tool",
        "kind": "trace",
        "fileEdits": [{"call_id": "edit-1", "tool": "edit_file", "path": "a.py"}],
    }]

    remaining = _filter_covered_file_edit_tool_events(messages, events)

    # The structured file edit owns the display; only the unrelated tool stays.
    assert [event["call_id"] for event in remaining] == ["exec-1"]


def test_approval_outcome_and_tool_trace_survive_replay_and_retention(
    isolated_webui: Path,
) -> None:
    state = _state()
    approval = {
        "event": "delegated_child",
        "child_id": "child-1",
        "child_session_key": "delegated:child-1",
        "profile": "delegated",
        "state": "waiting_approval",
        "parent_session_key": KEY,
        "parent_store_identity": state.store_identity,
        "parent_incarnation": state.incarnation,
        "parent_generation": state.generation,
        "root_key": KEY,
        "root_store_identity": state.store_identity,
        "root_incarnation": state.incarnation,
        "root_generation": state.generation,
        "child_store_identity": state.store_identity,
        "child_incarnation": "child-inc",
        "child_generation": 0,
        "run_sequence": 1,
        "state_sequence": 1,
        "runtime_generation": 0,
        "pending_approval_id": "approval-7",
        "can_stop": True,
        "owned_work": True,
        "task_label": "run tests",
    }
    _append(KEY, state, **approval)
    _append(KEY, state, event="user", text="run tests", _webui_source_prompt_id="evt-approval")
    _append(
        KEY,
        state,
        event="message",
        kind="tool_hint",
        text="exec(pytest)",
        tool_events=[{
            "version": 1,
            "phase": "start",
            "call_id": "exec-1",
            "name": "exec",
            "arguments": {"command": "pytest"},
            "result": None,
            "error": None,
        }],
    )
    _append(
        KEY,
        state,
        event="turn_end",
        turn_id="turn-approval",
        outcome="waiting_approval",
        approval_outcome="approved_once",
    )

    messages = _replay(KEY, state)
    child = next(m for m in messages if m.get("kind") == "delegated_child")
    assert child["delegatedChild"]["state"] == "waiting_approval"
    assert child["delegatedChild"]["pending_approval_id"] == "approval-7"
    assert child["delegatedChild"]["can_stop"] is True
    trace = next(m for m in messages if m.get("kind") == "trace")
    assert trace["toolEvents"][0]["name"] == "exec"
    assert trace["traces"] == ["exec({\"command\": \"pytest\"})"]

    page = build_webui_thread_response(KEY, expected_source=state, limit=10, direction="latest")
    assert page is not None
    retained = cast(dict[str, object], {k: v for k, v in page.items() if k != "page"})
    validate_retained_webui_thread(retained)
    for row in read_transcript_lines(KEY, expected_source=state):
        validate_retained_transcript_record(cast(dict[str, object], row))

    # The turn-end outcome and approval label are durable display facts; replay
    # does not invent a settled approval row from them.
    turn_end = next(
        row for row in read_transcript_lines(KEY, expected_source=state)
        if row.get("event") == "turn_end"
    )
    assert turn_end["outcome"] == "waiting_approval"
    assert turn_end["approval_outcome"] == "approved_once"
    assert not any(m.get("approvalOutcome") for m in page["messages"])


# ---------------------------------------------------------------------------
# 4. Context compaction markers and turn usage
# ---------------------------------------------------------------------------


def test_compaction_and_turn_usage_survive_replay_rotation_and_index_rebuild(
    isolated_webui: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    monkeypatch.setattr(tr, "_ACTIVE_TRANSCRIPT_ROTATE_BYTES", 900)
    monkeypatch.setattr(tr, "_TARGET_ACTIVE_TRANSCRIPT_BYTES", 400)

    for ordinal in range(4):
        _append(KEY, state, event="user", text=f"Prompt {ordinal}", turn_id=f"turn-{ordinal}",
                _webui_source_prompt_id=f"evt-{ordinal}")
        if ordinal == 1:
            _append(KEY, state, event="context_compaction", compaction_id="cmp-1",
                    phase="started")
            _append(KEY, state, event="context_compaction", compaction_id="cmp-1",
                    phase="succeeded")
        _append(KEY, state, event="stream_end", text=f"Reply {ordinal}",
                turn_id=f"turn-{ordinal}")
        _append(
            KEY,
            state,
            event="turn_end",
            turn_id=f"turn-{ordinal}",
            usage={"prompt_tokens": 10 + ordinal, "completion_tokens": 2, "bogus": "drop"},
            round_usages=[{"input_tokens": 4, "fraction": 1.5}],
            context_window_tokens=128000,
            context_usage={"context_tokens": 1024, "context_window_tokens": 128000},
        )

    assert _sanitize_turn_usage({"prompt_tokens": 3, "x": "y", "n": -1, "b": True}) == {
        "prompt_tokens": 3
    }
    # Real rotation moves committed turns into immutable segments.
    segments = list(webui_transcript_segments_dir(KEY).glob("*.jsonl"))
    assert segments

    def project() -> list[dict[str, Any]]:
        return _replay(KEY, state)

    before = project()
    compaction = next(m for m in before if m.get("kind") == "compaction")
    assert compaction["compaction"] == {"id": "cmp-1", "phase": "succeeded"}
    completed = [m for m in before if m.get("usage")]
    assert [m["usage"] for m in completed] == [
        {"prompt_tokens": 10 + ordinal, "completion_tokens": 2} for ordinal in range(4)
    ]
    assert all(m.get("contextWindowTokens") == 128000 for m in completed)
    assert all("bogus" not in m["usage"] for m in completed)
    assert completed[0]["roundUsages"] == [{"input_tokens": 4}]

    assert delete_transcript_index(KEY)
    head = rebuild_transcript_index(KEY, expected_source=state)
    assert head is not None
    assert head.total_turns == 4
    assert project() == before


# ---------------------------------------------------------------------------
# 5. Fork and parent/child delegated lineage
# ---------------------------------------------------------------------------


def test_fork_boundary_is_stable_and_stale_parent_lifetime_is_ignored(
    isolated_webui: Path,
) -> None:
    state = _state()
    _append(KEY, state, event="user", text="parent prompt", _webui_source_prompt_id="evt-p1")
    _append(KEY, state, event="stream_end", text="parent answer")
    _append(KEY, state, event="turn_end", turn_id="turn-p1")
    assert fork_boundary_message_count(read_transcript_lines(KEY, expected_source=state)) is None
    _append(KEY, state, event=WEBUI_FORK_MARKER_EVENT)
    _append(KEY, state, event="user", text="fork prompt", _webui_source_prompt_id="evt-f1")

    rows = read_transcript_lines(KEY, expected_source=state)
    assert fork_boundary_message_count(rows) == 2
    page = build_webui_thread_response(KEY, expected_source=state, limit=10, direction="latest")
    assert page is not None
    assert page["fork_boundary_message_count"] == 2

    child_lifetime = {
        "parent_session_key": KEY,
        "parent_store_identity": state.store_identity,
        "parent_incarnation": state.incarnation,
        "parent_generation": state.generation,
        "root_key": KEY,
        "root_store_identity": state.store_identity,
        "root_incarnation": state.incarnation,
        "root_generation": state.generation,
        "child_session_key": "delegated:child",
        "child_store_identity": state.store_identity,
        "child_incarnation": "child-inc",
        "child_generation": 0,
    }
    assert _delegated_child_lifetime(child_lifetime) is not None
    stale = dict(child_lifetime)
    stale["parent_incarnation"] = "retired-incarnation"
    assert _delegated_child_lifetime(stale) != _delegated_child_lifetime(child_lifetime)
    missing = dict(child_lifetime)
    missing.pop("root_generation")
    assert _delegated_child_lifetime(missing) is None


def test_delegated_child_replay_keeps_one_row_and_rejects_cross_lifetime_updates(
    isolated_webui: Path,
) -> None:
    state = _state()
    base: dict[str, Any] = {
        "event": "delegated_child",
        "child_id": "child-1",
        "child_session_key": "delegated:child-1",
        "profile": "delegated",
        "parent_session_key": KEY,
        "parent_store_identity": state.store_identity,
        "parent_incarnation": state.incarnation,
        "parent_generation": state.generation,
        "root_key": KEY,
        "root_store_identity": state.store_identity,
        "root_incarnation": state.incarnation,
        "root_generation": state.generation,
        "child_store_identity": state.store_identity,
        "child_incarnation": "child-inc",
        "child_generation": 0,
        "run_sequence": 1,
        "state_sequence": 1,
        "runtime_generation": 0,
    }
    _append(KEY, state, **base)
    _append(KEY, state, **{**base, "state": "completed", "state_sequence": 2,
                           "outcome": "completed"})
    # A stale parent lifetime and an older sequence must not replace the row.
    _append(KEY, state, **{**base, "state": "active", "state_sequence": 2,
                           "root_incarnation": "retired"})
    _append(KEY, state, **{**base, "state": "active", "state_sequence": 1})

    children = [m for m in _replay(KEY, state) if m.get("kind") == "delegated_child"]
    assert len(children) == 1
    assert children[0]["delegatedChild"]["state"] == "completed"


# ---------------------------------------------------------------------------
# 6. Incomplete tails, interruption, and crash/restart recoverability
# ---------------------------------------------------------------------------


def test_unfinished_tail_and_incomplete_turn_recover_from_canonical_history(
    isolated_webui: Path,
) -> None:
    state = _state()
    _append(KEY, state, event="user", text="first", _webui_source_prompt_id="evt-1")
    _append(KEY, state, event="stream_end", text="first answer")
    _append(KEY, state, event="turn_end", turn_id="turn-1")
    # An interrupted turn with no terminal record.
    _append(KEY, state, event="user", text="second", turn_id="turn-2",
            _webui_source_prompt_id="evt-2")
    _append(KEY, state, event="stream_end", text="partial second", turn_id="turn-2")

    assert has_unfinished_transcript_tail(KEY) is True
    lines = read_transcript_lines(KEY, expected_source=state)
    # An unterminated stream_end tail is an unresolved pending tail.
    assert has_pending_tool_calls(lines) is True

    session_messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "durable second answer"},
    ]
    turns = _session_backfill_turns(KEY, session_messages)
    repaired = _repair_unfinished_tail(lines, turns)

    assert [row["event"] for row in repaired[-2:]] == ["message", "turn_end"]
    assert repaired[-2]["text"] == "durable second answer"
    assert has_pending_tool_calls(repaired) is False
    assert completed_turn_ids(repaired)[-1] == "turn-2"

    # The gateway response applies the same repair and reports a settled tail.
    page = build_webui_thread_response(
        KEY,
        expected_source=state,
        session_messages=session_messages,
        limit=10,
        direction="latest",
    )
    assert page is not None
    assert _texts(page["messages"]) == [
        "first", "first answer", "second", "durable second answer",
    ]
    assert page["has_pending_tool_calls"] is False


def test_incomplete_marker_recovery_replaces_partial_answer_and_clears_marker(
    isolated_webui: Path,
) -> None:
    state = _state()
    _append(KEY, state, event="user", text="question", turn_id="turn-1",
            _webui_source_prompt_id="evt-q")
    _append(KEY, state, event="stream_end", text="partial", turn_id="turn-1")
    _append(KEY, state, event="turn_end", turn_id="turn-1", transcript_incomplete=True)

    lines = read_transcript_lines(KEY, expected_source=state)
    assert has_pending_tool_calls(lines) is True

    turns = _session_backfill_turns(KEY, [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "complete answer"},
    ])
    recovered = _recover_incomplete_turns(lines, turns)

    assert has_pending_tool_calls(recovered) is False
    assert not any(row.get("transcript_incomplete") for row in recovered)
    assert next(row for row in recovered if row.get("event") == "message")["text"] == (
        "complete answer"
    )


def test_interrupted_rotation_intent_is_rolled_back_and_read_once(
    isolated_webui: Path,
) -> None:
    state = _state()
    for ordinal in range(3):
        _append(KEY, state, event="user", text=f"Prompt {ordinal}",
                _webui_source_prompt_id=f"evt-{ordinal}")
        _append(KEY, state, event="stream_end", text=f"Reply {ordinal}")
        _append(KEY, state, event="turn_end", turn_id=f"turn-{ordinal}")

    committed = tr._read_segment_manifest_entries(KEY)
    tr._write_transcript_rotation_intent(
        KEY,
        source_active=active_transcript_path(KEY).name,
        target_active="abandoned.active.jsonl",
        committed_segments=[str(entry["id"]) for entry in committed],
    )
    intent = webui_transcript_segments_dir(KEY) / "rotation.json"
    assert intent.is_file()

    rows = read_transcript_lines(KEY, expected_source=state)
    assert not intent.is_file()
    assert _texts(replay_transcript_to_ui_messages(rows)) == [
        "Prompt 0", "Reply 0", "Prompt 1", "Reply 1", "Prompt 2", "Reply 2",
    ]
    assert not (isolated_webui / "abandoned.active.jsonl").exists()


# ---------------------------------------------------------------------------
# 7. Media, attachments, signed URLs, and generated assets
# ---------------------------------------------------------------------------


def test_media_references_are_rewritten_and_raw_host_paths_never_leak(
    isolated_webui: Path,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "media-workspace"
    workspace.mkdir()
    (workspace / "chart.png").write_bytes(b"png")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")

    signed: list[str] = []

    def sign(path: Path) -> dict[str, str] | None:
        signed.append(str(path))
        return {"url": f"/api/media/{path.name}", "name": path.name}

    rewritten = rewrite_local_markdown_images(
        "![chart](chart.png) ![escape](../outside.png) ![abs](/etc/passwd.png)",
        workspace_path=workspace,
        sign_path=sign,
    )
    assert "![chart](/api/media/chart.png)" in rewritten
    # Outside-workspace and absolute host paths are refused, not signed.
    assert "../outside.png" in rewritten
    assert "/etc/passwd.png" in rewritten
    assert all("outside" not in path and "passwd" not in path for path in signed)

    display_media = _media_from_signed_urls([
        {"url": "/api/media/a.png", "name": "a.png"},
        {"url": "/api/media/clip.mp4", "name": "clip.mp4"},
        {"name": "missing-url.png"},
        {"url": "", "name": "empty.png"},
    ])
    assert display_media == [
        {"kind": "image", "url": "/api/media/a.png", "name": "a.png"},
        {"kind": "video", "url": "/api/media/clip.mp4", "name": "clip.mp4"},
    ]

    state = _state()
    _append(KEY, state, event="user", text="see chart", media_paths=["chart.png"],
            _webui_source_prompt_id="evt-media")
    _append(KEY, state, event="stream_end", text="here", media=["/host/secret/answer.png"])
    _append(KEY, state, event="turn_end", turn_id="turn-media")

    page = build_webui_thread_response(
        KEY,
        expected_source=state,
        augment_user_media=lambda paths: [
            {"kind": "image", "url": f"/api/media/{Path(p).name}", "name": Path(p).name}
            for p in paths
        ],
        augment_assistant_media=lambda paths: [],
        limit=10,
        direction="latest",
    )
    assert page is not None
    assert page["messages"][0]["media"] == [
        {"kind": "image", "url": "/api/media/chart.png", "name": "chart.png"}
    ]
    dumped = json.dumps(page["messages"])
    assert "/host/secret/answer.png" not in dumped


# ---------------------------------------------------------------------------
# 8. Lifecycle cleanup and incarnation replacement
# ---------------------------------------------------------------------------


def test_delete_removes_segments_manifest_and_index(
    isolated_webui: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    monkeypatch.setattr(tr, "_ACTIVE_TRANSCRIPT_ROTATE_BYTES", 700)
    monkeypatch.setattr(tr, "_TARGET_ACTIVE_TRANSCRIPT_BYTES", 300)
    for ordinal in range(6):
        _append(KEY, state, event="user", text=f"Prompt {ordinal}",
                _webui_source_prompt_id=f"evt-{ordinal}")
        _append(KEY, state, event="stream_end", text=f"Reply {ordinal}")
        _append(KEY, state, event="turn_end", turn_id=f"turn-{ordinal}")

    segments_dir = webui_transcript_segments_dir(KEY)
    assert segments_dir.is_dir()
    assert transcript_index_dir(KEY).is_dir()
    assert (segments_dir / "manifest.json").is_file()

    result = delete_webui_transcript_result(KEY)

    assert result.complete is True and result.remaining == ()
    assert not segments_dir.exists()
    assert not transcript_index_dir(KEY).exists()
    assert read_transcript_lines(KEY) == []
    assert read_history_window(KEY, expected_source=state, limit=4,
                               direction="latest").status == "empty"


def test_incarnation_bump_rejects_prior_lifetime_rows_and_cursors(
    isolated_webui: Path,
) -> None:
    original = _state()
    for ordinal in range(5):
        _append(KEY, original, event="user", text=f"Old prompt {ordinal}",
                _webui_source_prompt_id=f"old-{ordinal}")
        _append(KEY, original, event="stream_end", text=f"Old reply {ordinal}")
        _append(KEY, original, event="turn_end", turn_id=f"old-{ordinal}")

    latest = read_history_window(KEY, expected_source=original, limit=2, direction="latest")
    assert latest.status == "ok"
    summaries = read_prompt_summary_range(KEY, expected_source=original, limit=5)
    assert summaries.status == "ok"
    old_locator = next(e.locator for e in summaries.entries if e.prompt_id == "old-0")

    replacement = _state(incarnation="inc-g4-replacement", generation=1)
    assert read_history_window(
        KEY, expected_source=replacement, limit=3, direction="latest"
    ).status == "empty"
    assert read_history_window(
        KEY, expected_source=replacement, limit=2, direction="before",
        cursor=latest.before_cursor,
    ).status == "stale_cursor"
    assert read_history_window(
        KEY, expected_source=replacement, limit=4, direction="around",
        prompt_locator=old_locator,
    ).status == "stale_cursor"
    assert read_transcript_lines(KEY, expected_source=replacement) == []
    assert build_webui_thread_response(
        KEY, expected_source=replacement, limit=4, direction="latest"
    ) is None


# ---------------------------------------------------------------------------
# 9. Reversible read compatibility and rollback proof
# ---------------------------------------------------------------------------


def _projection_snapshot(session_key: str, state: SourceState) -> dict[str, Any]:
    latest = read_history_window(session_key, expected_source=state, limit=4, direction="latest")
    assert latest.status == "ok"
    before = read_history_window(
        session_key, expected_source=state, limit=3, direction="before",
        cursor=latest.before_cursor,
    )
    assert before.status == "ok"
    after = read_history_window(
        session_key, expected_source=state, limit=3, direction="after",
        cursor=before.after_cursor,
    )
    assert after.status == "ok"
    around = read_history_window(
        session_key, expected_source=state, limit=6, direction="around", prompt_id="evt-steer-2",
    )
    assert around.status == "ok"
    summaries = read_prompt_summary_range(session_key, expected_source=state, limit=50)
    assert summaries.status == "ok"
    turns = tr.session_turn_index(session_key, limit=100, expected_source=state)
    assert turns is not None
    return {
        "latest": replay_transcript_to_ui_messages(latest.lines),
        "early": before.lines,
        "later": after.lines,
        "around": replay_transcript_to_ui_messages(around.lines),
        "cursors": {
            "latest_before": latest.before_cursor,
            "latest_after": latest.after_cursor,
            "before_after": before.after_cursor,
            "around_before": around.before_cursor,
            "around_after": around.after_cursor,
        },
        "target": around.target_prompt_id,
        "prompts": [
            (e.prompt_id, e.turn_ordinal, e.prompt_ordinal, e.label, e.created_at_ms)
            for e in summaries.entries
        ],
        "turns": turns,
    }


def test_index_delete_and_rebuild_reproduces_rows_prompt_ids_and_cursors(
    isolated_webui: Path,
) -> None:
    state = _state()
    for ordinal in range(12):
        _append(KEY, state, event="user", text=f"Prompt {ordinal}",
                _webui_source_prompt_id=f"evt-{ordinal}")
        if ordinal == 2:
            _append(KEY, state, event="user", text="Steer 2",
                    _webui_source_prompt_id="evt-steer-2")
            _append(KEY, state, event="stream_end", text="After steer 2")
        _append(KEY, state, event="stream_end", text=f"Reply {ordinal}")
        _append(KEY, state, event="turn_end", turn_id=f"turn-{ordinal}")

    before = _projection_snapshot(KEY, state)
    assert transcript_index_dir(KEY).is_dir()

    assert delete_transcript_index(KEY)
    head = rebuild_transcript_index(KEY, expected_source=state)
    assert head is not None

    after = _projection_snapshot(KEY, state)
    assert after == before
    # The Steer turn keeps both of its user prompts in the revealed window.
    revealed = [m["promptId"] for m in after["around"] if m["role"] == "user"]
    assert "evt-2" in revealed and "evt-steer-2" in revealed
    assert revealed.index("evt-2") < revealed.index("evt-steer-2")


def test_preindex_history_is_served_without_bulk_migration_or_index_creation(
    isolated_webui: Path,
) -> None:
    state = _state()
    for ordinal in range(6):
        _append(KEY, state, event="user", text=f"Legacy prompt {ordinal}",
                _webui_source_prompt_id=f"evt-{ordinal}")
        _append(KEY, state, event="stream_end", text=f"Legacy reply {ordinal}")
        _append(KEY, state, event="turn_end", turn_id=f"turn-{ordinal}")

    assert delete_transcript_index(KEY)
    tail = read_preindex_latest_tail(KEY, limit=4, expected_source=state)
    assert tail.status == "ok"
    assert tail.coverage == "tail_only"
    assert tail.total_turn_count is None
    assert _texts(replay_transcript_to_ui_messages(tail.lines))[-4:] == [
        "Legacy prompt 4", "Legacy reply 4", "Legacy prompt 5", "Legacy reply 5",
    ]
    # The bounded tail read never creates the read index.
    assert not transcript_index_dir(KEY).exists()

    page = build_webui_thread_response(KEY, expected_source=state, limit=4, direction="latest")
    assert page is not None
    assert page["page"]["coverage"] == "tail_only"
    assert _texts(page["messages"]) == [
        "Legacy prompt 4", "Legacy reply 4", "Legacy prompt 5", "Legacy reply 5",
    ]
    assert not transcript_index_dir(KEY).exists()


def test_rollback_without_index_reads_committed_jsonl_manifest_and_segments(
    isolated_webui: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    monkeypatch.setattr(tr, "_ACTIVE_TRANSCRIPT_ROTATE_BYTES", 800)
    monkeypatch.setattr(tr, "_TARGET_ACTIVE_TRANSCRIPT_BYTES", 320)
    for ordinal in range(7):
        _append(KEY, state, event="user", text=f"Prompt {ordinal}",
                _webui_source_prompt_id=f"evt-{ordinal}")
        _append(KEY, state, event="stream_end", text=f"Reply {ordinal}")
        _append(KEY, state, event="turn_end", turn_id=f"turn-{ordinal}")

    indexed = build_webui_thread_response(KEY, expected_source=state, limit=100,
                                          direction="latest")
    assert indexed is not None
    segments_dir = webui_transcript_segments_dir(KEY)
    manifest = json.loads((segments_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["segments"], "rotated fixture must keep committed segments"

    # Rollback: drop the rebuildable index only. Durable JSONL, segments, and
    # the manifest must still replay identical content directly.
    assert delete_transcript_index(KEY)
    rows = read_transcript_lines(KEY, expected_source=state)
    replayed = replay_transcript_to_ui_messages(rows)
    assert [(m["role"], m.get("content"), m.get("promptId")) for m in replayed] == [
        (m["role"], m.get("content"), m.get("promptId")) for m in indexed["messages"]
    ]

    # Without the index, the reader deliberately serves only the bounded tail.
    # It never silently claims the whole chat was read.
    tail_only = build_webui_thread_response(
        KEY, expected_source=state, limit=4, direction="latest"
    )
    assert tail_only is not None
    assert tail_only["page"]["coverage"] == "tail_only"
    assert tail_only["page"].get("total_turn_count") is None
    assert _texts(tail_only["messages"])[-2:] == ["Prompt 6", "Reply 6"]

    # Rebuilding the index restores the identical full projection.
    head = rebuild_transcript_index(KEY, expected_source=state)
    assert head is not None
    after = build_webui_thread_response(KEY, expected_source=state, limit=100,
                                        direction="latest")
    assert after is not None
    assert after["messages"] == indexed["messages"]
    assert after["completed_turn_ids"] == indexed["completed_turn_ids"]
    assert after["page"]["total_turn_count"] == indexed["page"]["total_turn_count"]


def test_prompt_id_derivation_matches_index_locator_contract() -> None:
    committed = {"event": "user", "text": "x", "_webui_source_prompt_id": "evt-committed"}
    assert _prompt_id_from_record(committed) == "evt-committed"
    legacy = {"event": "user", "text": "y", "_webui_replay_identity": "turn:3:record:1"}
    assert _prompt_id_from_record(legacy) == "legacy:3:1"
    assert _prompt_id_from_record({"event": "user", "text": "z"}) is None


async def test_canonical_only_session_has_no_display_fact_to_lose(
    isolated_webui: Path,
    tmp_path: Path,
) -> None:
    """Rollback proof: a canonical-only turn remains readable without display rows."""
    backend, state = await _canonical(tmp_path, "websocket:g4-canonical-only")
    key = "websocket:g4-canonical-only"

    async with acquire_execution(backend, key, run_id="canonical-only") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="canonical-only")
        writer.append_message({"role": "user", "content": "canonical question"})
        writer.append_message({"role": "assistant", "content": "canonical answer"})

    assert read_transcript_lines(key, expected_source=state) == []
    page = build_webui_thread_response(
        key,
        expected_source=state,
        session_messages=[
            {"role": "user", "content": "canonical question"},
            {"role": "assistant", "content": "canonical answer"},
        ],
        limit=10,
        direction="latest",
    )
    # No display rows means no display projection; the canonical body is not a
    # substitute for the display store, so this is an explicit empty read.
    assert page is None

    source_events = backend.source.load_source_snapshot(key).events
    assert [row.get("content") for row in source_events] == [
        "canonical question", "canonical answer",
    ]
    provenance = {
        "prompt_text": source_events[0]["content"],
        "answer_text": source_events[1]["content"],
        "turn_identity": None,
        "prompt_identity": source_events[0]["_event_id"],
    }
    assert provenance["prompt_text"] == "canonical question"
    assert provenance["answer_text"] == "canonical answer"
    # The canonical source has no display-only turn marker, compaction,
    # reasoning, tool, media, or usage fact to replay into the page.
    assert not any(
        row.get("event") in {"turn_end", "context_compaction", "reasoning_end", "file_edit"}
        for row in source_events
    )
    assert asdict(state)["session_key"] == key


def test_new_source_event_rejects_noncanonical_payload() -> None:
    """Canonical admission refuses payloads that cannot be persisted safely."""
    from kohaku.session.source import SourceValidationError

    with pytest.raises(SourceValidationError):
        new_source_event({"role": "user", "content": object()}, run_id="run")


# ---------------------------------------------------------------------------
# Live gateway parity: canonical admission is the display authority
# ---------------------------------------------------------------------------


async def test_live_turn_display_matches_canonical_then_survives_restart(
    engine_config: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One real managed turn: admission, display identity, restart, and rollback.

    The display prompt row must not exist before the canonical append, must
    carry the committed canonical event id afterwards, and must replay the same
    prompt/answer text after a restart and an index delete and rebuild.
    """
    from tests.webui.test_managed_display_origin import CHAT_ID as _LIVE_CHAT_ID
    from tests.webui.test_managed_display_origin import _gateway, _submit

    async with _gateway(engine_config, tmp_path, "Canonical answer") as (loop, channel, conn, _):
        from kohaku.session.source_store import SourceSessionStore

        chat_key = f"websocket:{_LIVE_CHAT_ID}"
        await _submit(channel, conn, "Canonical prompt", turn_id="g4-live-turn")
        assert loop.bus.inbound_size == 1
        # Nothing display-visible exists before the canonical commit runs.
        assert not any(
            row.get("event") == "user" for row in read_transcript_lines(chat_key)
        )

        await loop._dispatch(await loop.bus.consume_inbound())
        while loop.bus.outbound_size:
            from kohaku.channels.manager import ChannelManager

            await ChannelManager._send_once(channel, await loop.bus.consume_outbound())

        saved = loop.sessions.read_saved_source_metadata(chat_key)
        assert saved is not None
        canonical = loop.sessions.read_existing_canonical_source(chat_key)
        assert canonical is not None
        committed_user = next(row for row in canonical[1].events if row.get("role") == "user")
        committed_answer = next(
            row for row in canonical[1].events if row.get("role") == "assistant"
        )

        rows = read_transcript_lines(chat_key, expected_source=saved.state)
        display_user = next(row for row in rows if row.get("event") == "user")
        assert display_user["_webui_source_prompt_id"] == committed_user["_event_id"]
        assert display_user["_webui_text_ref"] == committed_user["_event_id"]
        assert "text" not in display_user

        def _load_canonical(event_ids: list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
            return loop.sessions.read_canonical_events_by_ids(
                chat_key,
                list(event_ids),
                expected_source=saved.state,
            )

        live = build_webui_thread_response(
            chat_key,
            expected_source=saved.state,
            canonical_events_loader=_load_canonical,
            limit=10,
            direction="latest",
        )
        assert live is not None
        assert [m["role"] for m in live["messages"]] == ["user", "assistant"]
        assert live["messages"][0]["content"] == committed_user["content"] == "Canonical prompt"
        assert live["messages"][0]["promptId"] == committed_user["_event_id"]
        assert live["messages"][1]["content"] == committed_answer["content"] == "Canonical answer"

        # Simulate a restart: rebuild the reader from durable rows only.
        assert delete_transcript_index(chat_key)
        restarted = build_webui_thread_response(
            chat_key,
            expected_source=saved.state,
            canonical_events_loader=_load_canonical,
            limit=10,
            direction="latest",
        )
        assert restarted is not None
        assert restarted["messages"] == live["messages"]
        head = rebuild_transcript_index(chat_key, expected_source=saved.state)
        assert head is not None
        rebuilt = build_webui_thread_response(
            chat_key,
            expected_source=saved.state,
            canonical_events_loader=_load_canonical,
            limit=10,
            direction="latest",
        )
        assert rebuilt is not None
        assert rebuilt["messages"] == live["messages"]
        assert rebuilt["completed_turn_ids"] == live["completed_turn_ids"]

        # A late canonical append failure must never create a display row.
        with monkeypatch.context() as patch_ctx:
            def refuse(*args: object, **kwargs: object) -> None:
                raise OSError("canonical append failed")

            patch_ctx.setattr(SourceSessionStore, "append_source_events", refuse)
            await _submit(channel, conn, "Rejected prompt", turn_id="g4-rejected-turn")
            await loop._dispatch(await loop.bus.consume_inbound())
            assert not any(
                row.get("event") == "user" and row.get("text") == "Rejected prompt"
                for row in read_transcript_lines(chat_key)
            )
