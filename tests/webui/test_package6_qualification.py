"""Package 6: Linux-executable qualification for the chat page.

This suite records the automated qualification evidence that a Linux host can
prove. It covers large-history bounded reads, cold-start settled recovery, an
end-to-end durability drill, and the active/next-turn isolation journey.

Hardware-only checks stay out of scope: the macOS Codex Desktop visual/native
comparison, the paired Android device smoke, and the Go mobile broker
allowlist. The work plan records them as remaining external checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from kohaku.session.source_store import SourceSessionStore, SourceState
from kohaku.webui import transcript as tr
from kohaku.webui.transcript import (
    _record_origin,
    active_transcript_path,
    append_transcript_object,
    build_webui_thread_response,
    delete_webui_transcript_result,
    read_transcript_lines,
    replay_transcript_to_ui_messages,
)
from kohaku.webui.transcript_index import (
    delete_transcript_index,
    read_history_window,
    rebuild_transcript_index,
    transcript_index_dir,
)

KEY = "websocket:pkg6-qualification"
CHAT_ID = "pkg6-qualification"
LARGE_TURNS = 220


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
    incarnation: str = "pkg6-inc",
    generation: int = 0,
) -> SourceState:
    return SourceState(
        store_identity="pkg6-store",
        session_key=key,
        incarnation=incarnation,
        revision=1,
        generation=generation,
        checkpoint_id=None,
        checkpoint_fingerprint="",
    )


def _append(session_key: str, source: SourceState, **row: Any) -> None:
    record: dict[str, Any] = dict(row)
    record.setdefault("chat_id", CHAT_ID)
    record.setdefault("created_at_ms", 1_700_000_000_000)
    record["_webui_source_origin"] = _record_origin(source)
    append_transcript_object(session_key, record)


def _write_turns(
    session_key: str,
    count: int,
    *,
    source: SourceState,
    pad_bytes: int = 0,
    start_index: int = 0,
) -> None:
    padding = "x" * pad_bytes
    for index in range(start_index, start_index + count):
        _append(
            session_key,
            source,
            event="user",
            text=f"Prompt {index}",
            turn_id=f"turn-{index}",
            _webui_source_prompt_id=f"evt-{index}",
        )
        _append(
            session_key,
            source,
            event="stream_end",
            text=f"Reply {index} {padding}".strip(),
            turn_id=f"turn-{index}",
        )
        _append(
            session_key,
            source,
            event="turn_end",
            turn_id=f"turn-{index}",
            usage={"prompt_tokens": index + 1},
        )


def _count_canonical_body_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    loaded: list[str] = []
    original = SourceSessionStore.load_source_snapshot

    def tracked(self: SourceSessionStore, key: str, *args: object, **kwargs: object):
        loaded.append(key)
        return original(self, key, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(SourceSessionStore, "load_source_snapshot", tracked)
    return loaded


# ---------------------------------------------------------------------------
# 1. Large-history bounded reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotated", [False, True])
def test_large_history_windows_read_no_canonical_body(
    isolated_webui: Path,
    monkeypatch: pytest.MonkeyPatch,
    rotated: bool,
) -> None:
    if rotated:
        monkeypatch.setattr(tr, "_ACTIVE_TRANSCRIPT_ROTATE_BYTES", 6000)
        monkeypatch.setattr(tr, "_TARGET_ACTIVE_TRANSCRIPT_BYTES", 2400)
    state = _state()
    _write_turns(KEY, LARGE_TURNS, source=state, pad_bytes=140 if rotated else 0)

    if rotated:
        segments = list(tr.webui_transcript_segments_dir(KEY).glob("*.jsonl"))
        assert len(segments) >= 20, "rotated fixture must span many segments"

    loads = _count_canonical_body_loads(monkeypatch)

    latest = read_history_window(KEY, expected_source=state, limit=20, direction="latest")
    assert latest.status == "ok"
    assert latest.coverage == "indexed"
    assert latest.total_turn_count == LARGE_TURNS
    assert latest.metrics.records_decoded <= 66
    assert latest.metrics.index_pages_read <= 2
    assert latest.metrics.manifest_entries_decoded == 0

    before = read_history_window(
        KEY, expected_source=state, limit=20, direction="before",
        cursor=latest.before_cursor,
    )
    assert before.status == "ok" and before.metrics.records_decoded <= 66

    after = read_history_window(
        KEY, expected_source=state, limit=20, direction="after",
        cursor=before.after_cursor,
    )
    assert after.status == "ok" and after.metrics.records_decoded <= 66

    around = read_history_window(
        KEY, expected_source=state, limit=20, direction="around", prompt_id="evt-40",
    )
    assert around.status == "ok"
    assert around.target_prompt_id == "evt-40"
    assert around.metrics.records_decoded <= 66

    turns = tr.session_turn_index(KEY, limit=100, expected_source=state)
    assert turns is not None
    assert turns["total_turn_count"] == LARGE_TURNS
    assert len(turns["turns"]) == 100
    assert len(turns["prompts"]) == 100

    assert loads == [], "settled indexed reads must load zero canonical bodies"


def test_large_history_http_windows_are_bounded_and_paginate_without_gaps(
    isolated_webui: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from kohaku.session.runtime_authority import DurableSessionAuthority
    from tests.webui.test_child_thread_route import _make_channel

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tr, "_ACTIVE_TRANSCRIPT_ROTATE_BYTES", 7000)
    monkeypatch.setattr(tr, "_TARGET_ACTIVE_TRANSCRIPT_BYTES", 2800)

    import asyncio

    async def run() -> None:
        lease = await DurableSessionAuthority(
            workspace, sessions_root=tmp_path / "sessions",
        ).acquire()
        try:
            manager = lease.manager
            assert manager.upsert_session_metadata(KEY, {"webui": True}, create_if_absent=True)
            saved = manager.read_saved_source_metadata(KEY)
            assert saved is not None
            _write_turns(KEY, LARGE_TURNS, source=saved.state, pad_bytes=140)

            gateway = _make_channel(tmp_path, manager).gateway
            token = gateway.tokens.issue_api_token(300)
            headers = Headers([("Host", "127.0.0.1:8765"), ("Authorization", f"Bearer {token}")])
            connection = SimpleNamespace(
                remote_address=("127.0.0.1", 1234),
                request=SimpleNamespace(headers=headers),
            )
            encoded = quote(KEY, safe="")
            loads = _count_canonical_body_loads(monkeypatch)

            latest = await gateway.http.dispatch(
                connection,
                Request(
                    f"/api/sessions/{encoded}/webui-thread?view=history&limit=20&direction=latest",
                    headers,
                ),
            )
            assert latest is not None and latest.status_code == 200
            latest_body = json.loads(latest.body)
            assert latest_body["page"]["coverage"] == "indexed"
            assert latest_body["page"]["total_turn_count"] == LARGE_TURNS
            assert [m["content"] for m in latest_body["messages"][-2:]] == [
                f"Prompt {LARGE_TURNS - 1}",
                f"Reply {LARGE_TURNS - 1} {'x' * 140}".strip(),
            ]

            turns = await gateway.http.dispatch(
                connection,
                Request(f"/api/sessions/{encoded}/turns?limit=100", headers),
            )
            assert turns is not None and turns.status_code == 200
            turns_body = json.loads(turns.body)
            assert turns_body["total_turn_count"] == LARGE_TURNS
            assert len(turns_body["prompts"]) == 100

            # Page backward to the start without gaps or duplicates.
            cursor = latest_body["page"]["before_cursor"]
            expected_newest = latest_body["page"]["first_turn_ordinal"] - 1
            seen: list[int] = [
                int(m["content"].split()[1])
                for m in latest_body["messages"]
                if m["role"] == "user"
            ]
            assert seen[-1] == LARGE_TURNS - 1
            assert expected_newest == seen[0] - 1
            while cursor is not None:
                response = await gateway.http.dispatch(
                    connection,
                    Request(
                        f"/api/sessions/{encoded}/webui-thread?view=history"
                        f"&direction=before&before={quote(cursor, safe='')}&limit=20",
                        headers,
                    ),
                )
                assert response is not None and response.status_code == 200
                body = json.loads(response.body)
                prompts = [
                    int(m["content"].split()[1])
                    for m in body["messages"]
                    if m["role"] == "user"
                ]
                # The older window ends exactly at the previous edge; no gap and
                # no duplicate prompt across the boundary.
                assert prompts
                assert prompts[-1] == expected_newest
                assert prompts == list(
                    range(expected_newest - len(prompts) + 1, expected_newest + 1)
                )
                seen = prompts + seen
                expected_newest -= len(prompts)
                cursor = body["page"]["before_cursor"]
            assert expected_newest == -1
            assert seen == list(range(LARGE_TURNS))
            assert loads == [], "HTTP history windows must load zero canonical bodies"
        finally:
            await lease.close()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# 2. Cold-start settled recovery
# ---------------------------------------------------------------------------


async def test_cold_start_recovers_only_the_unsettled_session(
    isolated_webui: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kohaku.bus.queue import MessageBus
    from kohaku.session.pending_followups import (
        mint_pending_followup_id,
        record_pending_followup,
    )
    from kohaku.session.persistence_metadata import (
        SOURCE_PENDING_FOLLOWUPS_METADATA_KEY,
    )
    from kohaku.session.recovery import RecoveryCoordinator
    from kohaku.session.runtime_authority import DurableSessionAuthority

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(
        workspace, sessions_root=tmp_path / "sessions",
    ).acquire()
    manager = lease.manager
    try:
        settled_keys = [f"websocket:pkg6-settled-{idx}" for idx in range(3)]
        unsettled_key = "websocket:pkg6-unsettled"
        for key in [*settled_keys, unsettled_key]:
            assert manager.upsert_session_metadata(
                key, {"webui": True}, create_if_absent=True,
            )
        backend = manager._jsonl_store

        # Large settled histories; the display store keeps them bounded.
        for key in settled_keys:
            saved = manager.read_saved_source_metadata(key)
            assert saved is not None
            _write_turns(key, 40, source=saved.state)

        from kohaku.bus.events import InboundMessage
        from kohaku.session.execution_owner import acquire_execution

        message = InboundMessage(
            channel="websocket",
            sender_id="user",
            chat_id="pkg6-unsettled",
            content="finish this turn",
            metadata={"webui": True},
        )
        mint_pending_followup_id(message)
        record = record_pending_followup(message)
        unsettled_backend = manager.source_backend(unsettled_key)
        assert unsettled_backend is not None
        async with acquire_execution(
            unsettled_backend, unsettled_key, run_id="pkg6-startup",
        ) as owner:
            assert unsettled_backend.source.admit_pending_followup(
                unsettled_key, owner, record,
            )
        header = unsettled_backend.source.load_source_snapshot(unsettled_key).header
        assert SOURCE_PENDING_FOLLOWUPS_METADATA_KEY in header["metadata"]
        assert backend is not None

        loads = _count_canonical_body_loads(monkeypatch)
        bus = MessageBus()
        await RecoveryCoordinator(manager, bus).scan()

        # Settled sessions are classified from bounded headers; only the
        # journaled session pays for one owner-validated body load.
        assert [key for key in settled_keys if key in loads] == []
        assert loads.count(unsettled_key) >= 1
        restored = bus.inbound.get_nowait()
        assert restored.content == "finish this turn"
    finally:
        await lease.close()


# ---------------------------------------------------------------------------
# 3. End-to-end durability drill
# ---------------------------------------------------------------------------


def test_durability_drill_rotation_recovery_rebuild_and_restart(
    isolated_webui: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tr, "_ACTIVE_TRANSCRIPT_ROTATE_BYTES", 3000)
    monkeypatch.setattr(tr, "_TARGET_ACTIVE_TRANSCRIPT_BYTES", 1200)
    state = _state()
    _write_turns(KEY, 60, source=state, pad_bytes=70)

    # Collect the whole chat through bounded windows from the tail.
    def full_projection() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        window = read_history_window(KEY, expected_source=state, limit=100, direction="latest")
        assert window.status == "ok"
        collected = list(window.lines)
        cursors = {
            "latest_before": window.before_cursor,
            "total_turn_count": window.total_turn_count,
        }
        while window.before_cursor is not None:
            window = read_history_window(
                KEY, expected_source=state, limit=100, direction="before",
                cursor=window.before_cursor,
            )
            assert window.status == "ok"
            collected = list(window.lines) + collected
        return replay_transcript_to_ui_messages(collected), cursors

    messages_before, cursors_before = full_projection()
    prompts_before = [m["promptId"] for m in messages_before if m["role"] == "user"]
    assert prompts_before == [f"evt-{index}" for index in range(60)]
    assert cursors_before["total_turn_count"] == 60
    first_page = build_webui_thread_response(
        KEY, expected_source=state, limit=100, direction="latest"
    )
    assert first_page is not None
    # The 100-message window budget caps the number of loaded turns, not the
    # total, and reports the bounded older cursor honestly.
    assert first_page["page"]["total_turn_count"] == 60
    assert first_page["page"]["has_more_before"] is True

    # Interrupt a rotation: leave the intent file on disk before recovery.
    committed = tr._read_segment_manifest_entries(KEY)
    tr._write_transcript_rotation_intent(
        KEY,
        source_active=active_transcript_path(KEY).name,
        target_active="abandoned.active.jsonl",
        committed_segments=[str(entry["id"]) for entry in committed],
    )
    intent = tr.webui_transcript_segments_dir(KEY) / "rotation.json"
    assert intent.is_file()
    assert read_transcript_lines(KEY, expected_source=state)
    assert not intent.is_file(), "read must recover the interrupted rotation"

    # Delete and rebuild the index: identical rows, prompt ids, and cursors.
    assert delete_transcript_index(KEY)
    head = rebuild_transcript_index(KEY, expected_source=state)
    assert head is not None and head.total_turns == 60

    messages_after, cursors_after = full_projection()
    assert messages_after == messages_before
    assert cursors_after == cursors_before

    restarted = build_webui_thread_response(KEY, expected_source=state, limit=100,
                                            direction="latest")
    assert restarted is not None
    assert restarted["messages"] == first_page["messages"]
    assert restarted["page"] == first_page["page"]
    assert restarted["completed_turn_ids"] == first_page["completed_turn_ids"]

    # Deletion clears every display artifact and the index.
    result = delete_webui_transcript_result(KEY)
    assert result.complete is True
    assert not transcript_index_dir(KEY).exists()
    assert read_transcript_lines(KEY) == []


# ---------------------------------------------------------------------------
# 4. Active/next-turn isolation
# ---------------------------------------------------------------------------


async def test_active_and_next_turn_isolation_end_to_end(
    isolated_webui: Path,
    tmp_path: Path,
) -> None:
    """Turn A keeps folder A; the next turn B is admitted under B; Once retires.

    This uses the real public session owner (`SessionManager` next-turn
    preferences) rather than re-implementing the authority drill, which
    `tests/session/test_next_turn_preferences.py` already proves in full.
    """
    from kohaku.session.runtime_authority import DurableSessionAuthority

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(
        workspace, sessions_root=tmp_path / "sessions",
    ).acquire()
    manager = lease.manager
    key = "websocket:pkg6-isolation"
    try:
        assert manager.upsert_session_metadata(key, {"webui": True}, create_if_absent=True)
        saved = manager.read_saved_source_metadata(key)
        assert saved is not None
        state = saved.state

        folder_a = tmp_path / "folder-a"
        folder_b = tmp_path / "folder-b"
        folder_a.mkdir()
        folder_b.mkdir()

        # Turn A is admitted under folder A. The next-turn choice is folder B.
        active_a = {"project_path": str(folder_a), "access_mode": "restricted"}
        next_b = {"project_path": str(folder_b), "access_mode": "restricted"}
        assert manager.upsert_session_metadata(key, {"workspace_scope": active_a})
        written = manager.update_next_turn_preferences(
            key,
            {
                "workspace_scope": next_b,
                "model_preset": "deep",
                "reasoning_effort": "high",
            },
        )
        assert written is not None

        # The active scope is unchanged while the next-turn choice holds B.
        canonical_scope = manager.read_session_metadata(key)["metadata"]["workspace_scope"]
        assert canonical_scope["project_path"] == str(folder_a)
        pending = manager.read_next_turn_preferences(key)
        assert pending is not None
        assert pending.get("workspace_scope")["project_path"] == str(folder_b)
        assert pending.get("model_preset") == "deep"

        # Turn A completion cannot overwrite the newer B choice.
        assert manager.upsert_session_metadata(key, {"title": "completion touched the header"})
        still_pending = manager.read_next_turn_preferences(key)
        assert still_pending is not None
        assert still_pending.get("workspace_scope")["project_path"] == str(folder_b)

        # A new turn consumes exactly the admitted choice; it is never reapplied.
        # Consuming every retained value removes the record, so the call returns
        # ``None`` and a later read is empty.
        manager.consume_next_turn_preferences(
            key,
            {"workspace_scope": next_b, "model_preset": "deep", "reasoning_effort": "high"},
        )
        assert manager.read_next_turn_preferences(key) is None

        # Reset invalidates any retained choice; the replacement origin starts
        # clean. A reset keeps the incarnation but bumps the generation.
        manager.update_next_turn_preferences(key, {"workspace_scope": next_b})
        assert manager.reset_session(key) is not None
        replaced = manager.read_saved_source_metadata(key)
        assert replaced is not None
        assert replaced.state.generation == state.generation + 1
        assert replaced.state.incarnation == state.incarnation
        assert manager.read_next_turn_preferences(key) is None
    finally:
        await lease.close()
