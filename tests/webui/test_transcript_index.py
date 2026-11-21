"""Gate G2 proofs for the seekable display read index and bounded pre-index tail."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nanobot.session.source_store import SourceState
from nanobot.webui import transcript as tr
from nanobot.webui.transcript import (
    _PROMPT_SOURCE_EVENT,
    _RECORD_ORIGIN,
    _record_origin,
    active_transcript_path,
    append_transcript_object,
    replay_transcript_to_ui_messages,
)
from nanobot.webui.transcript_index import (
    INDEX_PAGE_SIZE,
    delete_transcript_index,
    read_history_window,
    read_preindex_latest_tail,
    read_prompt_summary_range,
    rebuild_transcript_index,
    transcript_index_dir,
)

KEY = "websocket:g2-index-proof"
CHAT_ID = "g2-index-proof"


def _source(
    *,
    key: str = KEY,
    store_identity: str = "store-a",
    incarnation: str = "inc-1",
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


def _write_turns(
    session_key: str,
    count: int,
    *,
    source: SourceState,
    start_index: int = 0,
    pad_bytes: int = 0,
    multi_prompt_at: int | None = None,
    literal_turn_end_at: int | None = None,
) -> None:
    origin = _record_origin(source)
    padding = "x" * pad_bytes
    for idx in range(start_index, start_index + count):
        append_transcript_object(session_key, {
            "event": "user",
            "chat_id": CHAT_ID,
            "text": f"Prompt {idx}",
            "created_at_ms": 1_700_000_000_000 + idx * 1000,
            _RECORD_ORIGIN: origin,
            _PROMPT_SOURCE_EVENT: f"evt-u-{idx}",
        })
        answer_text = (
            f"Reply {idx} mentions turn_end literally {padding}".strip()
            if idx == literal_turn_end_at
            else f"Reply {idx} {padding}".strip()
        )
        append_transcript_object(session_key, {
            "event": "message",
            "chat_id": CHAT_ID,
            "text": answer_text,
            "created_at_ms": 1_700_000_000_100 + idx * 1000,
            _RECORD_ORIGIN: origin,
        })
        if idx == multi_prompt_at:
            append_transcript_object(session_key, {
                "event": "user",
                "chat_id": CHAT_ID,
                "text": f"Steer {idx}",
                "created_at_ms": 1_700_000_000_200 + idx * 1000,
                _RECORD_ORIGIN: origin,
                _PROMPT_SOURCE_EVENT: f"evt-steer-{idx}",
            })
            append_transcript_object(session_key, {
                "event": "message",
                "chat_id": CHAT_ID,
                "text": f"After steer {idx}",
                "created_at_ms": 1_700_000_000_300 + idx * 1000,
                _RECORD_ORIGIN: origin,
            })
        append_transcript_object(session_key, {
            "event": "turn_end",
            "chat_id": CHAT_ID,
            "turn_id": f"turn-{idx}",
            "created_at_ms": 1_700_000_000_400 + idx * 1000,
            _RECORD_ORIGIN: origin,
        })


@pytest.fixture
def isolated_webui_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import nanobot.webui.thread_disk  # noqa: F401 - bind real get_webui_dir before patching

    webui_dir = tmp_path / "webui"
    webui_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("nanobot.config.paths.get_webui_dir", lambda: webui_dir)
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.webui.transcript.get_webui_dir", lambda: webui_dir)
    return webui_dir


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("indexed", [True, False])
def test_g2_bounded_latest_read_constant_work_across_24_and_240_turns(
    isolated_webui_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    rotated: bool,
    indexed: bool,
) -> None:
    if rotated:
        # Force real rotation roughly every 4 turns (~5 segments for 24, ~59 for 240).
        monkeypatch.setattr(tr, "_ACTIVE_TRANSCRIPT_ROTATE_BYTES", 2400)
        monkeypatch.setattr(tr, "_TARGET_ACTIVE_TRANSCRIPT_BYTES", 1000)

    metrics_by_size: dict[int, tuple[list[str], object]] = {}
    for total_turns in (24, 240):
        key = f"websocket:g2-{total_turns}-{int(rotated)}-{int(indexed)}"
        source = _source(key=key)
        _write_turns(key, total_turns, source=source, pad_bytes=120 if rotated else 0)
        if rotated:
            seg_dir = tr.webui_transcript_segments_dir(key)
            seg_files = list(seg_dir.glob("*.jsonl"))
            assert len(seg_files) >= (3 if total_turns == 24 else 30)
        if not indexed:
            assert delete_transcript_index(key)

        window = read_history_window(key, expected_source=source, limit=4, direction="latest")
        assert window.status == "ok"
        texts = [m["content"] for m in replay_transcript_to_ui_messages(window.lines)]
        expected_tail = [
            f"Prompt {total_turns - 2}",
            f"Reply {total_turns - 2}" + (f" {'x' * 120}" if rotated else ""),
            f"Prompt {total_turns - 1}",
            f"Reply {total_turns - 1}" + (f" {'x' * 120}" if rotated else ""),
        ]
        assert texts == expected_tail
        assert window.has_more_before is True
        assert window.has_more_after is False
        assert window.metrics.manifest_entries_decoded == 0
        if indexed:
            assert window.coverage == "indexed"
            assert window.total_turn_count == total_turns
            assert window.metrics.index_pages_read == 1
            assert window.metrics.chunks_opened == 1
            assert window.metrics.records_decoded == 6
        else:
            assert window.coverage == "tail_only"
            assert window.total_turn_count is None
            assert window.metrics.chunks_opened == 1
        metrics_by_size[total_turns] = (texts, window.metrics)

    short_metrics = metrics_by_size[24][1]
    long_metrics = metrics_by_size[240][1]
    # 10x history must NOT scale decoded records or bytes read on latest open.
    assert long_metrics.records_decoded <= short_metrics.records_decoded + 6  # type: ignore[attr-defined]
    assert long_metrics.bytes_read <= short_metrics.bytes_read + 4096  # type: ignore[attr-defined]


def test_g2_before_after_and_around_exact_prompt_reveal_without_gaps(
    isolated_webui_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tr, "_ACTIVE_TRANSCRIPT_ROTATE_BYTES", 4000)
    monkeypatch.setattr(tr, "_TARGET_ACTIVE_TRANSCRIPT_BYTES", 1600)
    source = _source()
    total_turns = INDEX_PAGE_SIZE * 2 + 12  # 140 turns across 3 index pages + multiple segments
    _write_turns(
        KEY,
        total_turns,
        source=source,
        pad_bytes=80,
        multi_prompt_at=15,
        literal_turn_end_at=14,
    )

    # 1. Prompt summaries read only directory pages (0 JSONL chunks opened).
    first_page = read_prompt_summary_range(KEY, expected_source=source, limit=25)
    assert first_page.status == "ok"
    assert first_page.total_turn_count == total_turns
    assert first_page.total_prompt_count == total_turns + 1
    assert first_page.metrics.chunks_opened == 0
    assert first_page.has_more_before is True

    # Walk backward through prompt summaries until we reach prompt 15 and its Steer prompt.
    collected = list(first_page.entries)
    cursor = first_page.before_cursor
    while cursor is not None:
        older_page = read_prompt_summary_range(KEY, expected_source=source, limit=40, before=cursor)
        assert older_page.status == "ok"
        assert older_page.metrics.chunks_opened == 0
        collected = [*older_page.entries, *collected]
        cursor = older_page.before_cursor
    assert len(collected) == total_turns + 1

    steer_entry = next(item for item in collected if item.prompt_id == "evt-steer-15")
    assert steer_entry.turn_ordinal == 15
    assert steer_entry.label == "Steer 15"
    literal_entry = next(item for item in collected if item.prompt_id == "evt-u-14")
    assert literal_entry.turn_ordinal == 14

    # 2. Exact around(prompt_locator) reveal on the unloaded Steer prompt in turn 15.
    around = read_history_window(
        KEY,
        expected_source=source,
        limit=10,
        direction="around",
        prompt_locator=steer_entry.locator,
    )
    assert around.status == "ok"
    assert around.target_prompt_id == "evt-steer-15"
    assert around.has_more_before is True
    assert around.has_more_after is True
    assert around.metrics.index_pages_read <= 3
    around_msgs = replay_transcript_to_ui_messages(around.lines)
    assert any(m["content"] == "Steer 15" for m in around_msgs)
    assert any("mentions turn_end literally" in m["content"] for m in around_msgs)
    assert around.first_turn_ordinal is not None and around.last_turn_ordinal is not None

    # 3. Page older and newer from the revealed window with ZERO gaps or duplicates.
    older = read_history_window(
        KEY,
        expected_source=source,
        limit=6,
        direction="before",
        cursor=around.before_cursor,
    )
    assert older.status == "ok"
    assert older.last_turn_ordinal == around.first_turn_ordinal - 1

    newer = read_history_window(
        KEY,
        expected_source=source,
        limit=6,
        direction="after",
        cursor=around.after_cursor,
    )
    assert newer.status == "ok"
    assert newer.first_turn_ordinal == around.last_turn_ordinal + 1


def test_g2_origin_invalidation_and_title_edit_preservation(
    isolated_webui_dir: Path,
) -> None:
    v1 = _source(revision=1, generation=0)
    _write_turns(KEY, 10, source=v1)

    latest_v1 = read_history_window(KEY, expected_source=v1, limit=4, direction="latest")
    assert latest_v1.status == "ok"
    assert latest_v1.before_cursor is not None
    summaries_v1 = read_prompt_summary_range(KEY, expected_source=v1, limit=5)
    assert summaries_v1.status == "ok"
    old_locator = summaries_v1.entries[0].locator

    # Title-only canonical metadata edit bumps revision only -> origin unchanged!
    v1_title_edit = _source(revision=2, generation=0)
    assert v1_title_edit.same_origin(v1)
    before_after_title = read_history_window(
        KEY,
        expected_source=v1_title_edit,
        limit=4,
        direction="before",
        cursor=latest_v1.before_cursor,
    )
    assert before_after_title.status == "ok"
    around_after_title = read_history_window(
        KEY,
        expected_source=v1_title_edit,
        limit=4,
        direction="around",
        prompt_locator=old_locator,
    )
    assert around_after_title.status == "ok"

    # Reset or delete+recreate bumps generation/incarnation while leaving old file on disk.
    v2 = _source(revision=1, generation=1)
    assert not v2.same_origin(v1)
    assert read_history_window(
        KEY, expected_source=v2, limit=4, direction="latest",
    ).status == "empty"
    assert read_history_window(
        KEY, expected_source=v2, limit=4, direction="before", cursor=latest_v1.before_cursor,
    ).status == "stale_cursor"
    assert read_history_window(
        KEY, expected_source=v2, limit=4, direction="around", prompt_locator=old_locator,
    ).status == "stale_cursor"

    # Appending 1 new turn under v2 into the retained file returns ONLY v2's turn.
    _write_turns(KEY, 1, source=v2, start_index=100)
    delete_transcript_index(KEY)
    preindex_v2 = read_history_window(KEY, expected_source=v2, limit=4, direction="latest")
    assert preindex_v2.status == "ok"
    assert preindex_v2.has_more_before is False
    assert [m["content"] for m in replay_transcript_to_ui_messages(preindex_v2.lines)] == [
        "Prompt 100", "Reply 100",
    ]
    rebuild_transcript_index(KEY, expected_source=v2)
    indexed_v2 = read_history_window(KEY, expected_source=v2, limit=4, direction="latest")
    assert indexed_v2.status == "ok"
    assert indexed_v2.total_turn_count == 1
    assert [m["content"] for m in replay_transcript_to_ui_messages(indexed_v2.lines)] == [
        "Prompt 100", "Reply 100",
    ]


def test_g2_interrupted_index_update_catches_up_only_delta(
    isolated_webui_dir: Path,
) -> None:
    source = _source()
    _write_turns(KEY, 80, source=source)

    # Simulate an interrupted index update: append 2 committed turns straight to
    # the active file without advancing head.json.
    active = active_transcript_path(KEY)
    origin = _record_origin(source)
    extra_lines: list[dict[str, object]] = []
    for idx in (80, 81):
        extra_lines.extend([
            {
                "event": "user", "chat_id": CHAT_ID, "text": f"Prompt {idx}",
                "created_at_ms": 1_700_000_000_000 + idx * 1000,
                _RECORD_ORIGIN: origin, _PROMPT_SOURCE_EVENT: f"evt-u-{idx}",
            },
            {
                "event": "message", "chat_id": CHAT_ID, "text": f"Reply {idx}",
                "created_at_ms": 1_700_000_000_100 + idx * 1000,
                _RECORD_ORIGIN: origin,
            },
            {
                "event": "turn_end", "chat_id": CHAT_ID, "turn_id": f"turn-{idx}",
                "created_at_ms": 1_700_000_000_200 + idx * 1000,
                _RECORD_ORIGIN: origin,
            },
        ])
    with open(active, "ab") as handle:
        for row in extra_lines:
            handle.write((json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))

    window = read_history_window(KEY, expected_source=source, limit=4, direction="latest")
    assert window.status == "ok"
    assert window.coverage == "indexed"
    assert window.total_turn_count == 82
    assert [m["content"] for m in replay_transcript_to_ui_messages(window.lines)] == [
        "Prompt 80", "Reply 80", "Prompt 81", "Reply 81",
    ]
    # Decoded ONLY the 6 delta rows + 6 latest-window rows, never the 240 earlier rows.
    assert window.metrics.records_decoded == 12


def test_g2_stale_byte_offsets_and_replaced_file_fail_closed(
    isolated_webui_dir: Path,
) -> None:
    source = _source()
    _write_turns(KEY, 6, source=source)

    # 1. Corrupting a directory page's start_byte to mid-line fails closed.
    page_file = transcript_index_dir(KEY) / "pages" / "000000.json"
    page_data = json.loads(page_file.read_text(encoding="utf-8"))
    page_data["turns"][-1]["start_byte"] += 5
    page_file.write_text(json.dumps(page_data), encoding="utf-8")
    corrupt = read_history_window(
        KEY, expected_source=source, limit=2, direction="latest", allow_rebuild=False,
    )
    assert corrupt.status == "unavailable"
    assert corrupt.lines == []

    # 2. Replacing the active file in place (new inode) rejects the stale head.
    rebuild_transcript_index(KEY, expected_source=source)
    active = active_transcript_path(KEY)
    tmp = active.with_suffix(".replaced")
    tmp.write_bytes(active.read_bytes())
    os.replace(tmp, active)
    # Without rebuild, latest falls back to bounded pre-index tail read (never stale offsets).
    after_replace = read_history_window(
        KEY, expected_source=source, limit=2, direction="latest", allow_rebuild=False,
    )
    assert after_replace.status == "ok"
    assert after_replace.coverage == "tail_only"


def test_g2_oversized_recent_record_is_not_an_empty_chat(
    isolated_webui_dir: Path,
) -> None:
    source = _source()
    origin = _record_origin(source)
    huge_text = "A" * (300 * 1024)
    append_transcript_object(KEY, {
        "event": "user", "chat_id": CHAT_ID, "text": "Huge prompt",
        _RECORD_ORIGIN: origin, _PROMPT_SOURCE_EVENT: "evt-huge",
    })
    append_transcript_object(KEY, {
        "event": "message", "chat_id": CHAT_ID, "text": huge_text,
        _RECORD_ORIGIN: origin,
    })
    append_transcript_object(KEY, {
        "event": "turn_end", "chat_id": CHAT_ID, "turn_id": "turn-huge",
        _RECORD_ORIGIN: origin,
    })
    assert delete_transcript_index(KEY)

    tail = read_preindex_latest_tail(KEY, limit=4, expected_source=source)
    assert tail.status == "oversized_tail"
    assert tail.has_more_before is True

    # Rebuilding the index still recovers the oversized turn completely.
    head = rebuild_transcript_index(KEY, expected_source=source)
    assert head is not None and head.total_turns == 1
    indexed = read_history_window(KEY, expected_source=source, limit=4, direction="latest")
    assert indexed.status == "ok"
    assert replay_transcript_to_ui_messages(indexed.lines)[1]["content"] == huge_text


def test_g2_delete_and_rebuild_index_matches_incremental_index(
    isolated_webui_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tr, "_ACTIVE_TRANSCRIPT_ROTATE_BYTES", 3000)
    monkeypatch.setattr(tr, "_TARGET_ACTIVE_TRANSCRIPT_BYTES", 1200)
    source = _source()
    _write_turns(KEY, 90, source=source, pad_bytes=70, multi_prompt_at=20)

    before_latest = read_history_window(KEY, expected_source=source, limit=6, direction="latest")
    before_summaries = read_prompt_summary_range(KEY, expected_source=source, limit=30)
    assert before_latest.status == "ok" and before_summaries.status == "ok"

    assert delete_transcript_index(KEY)
    rebuilt_head = rebuild_transcript_index(KEY, expected_source=source)
    assert rebuilt_head is not None
    assert rebuilt_head.total_turns == 90
    assert rebuilt_head.total_prompts == 91

    after_latest = read_history_window(KEY, expected_source=source, limit=6, direction="latest")
    after_summaries = read_prompt_summary_range(KEY, expected_source=source, limit=30)
    assert replay_transcript_to_ui_messages(after_latest.lines) == (
        replay_transcript_to_ui_messages(before_latest.lines)
    )
    assert after_latest.before_cursor == before_latest.before_cursor
    assert after_summaries.entries == before_summaries.entries


async def test_g2_managed_browse_proof_plus_index_reads_zero_canonical_body_bytes(
    isolated_webui_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end G2 proof: managed browse metadata + display index reads 0 canonical body bytes."""
    from nanobot.session.runtime_authority import DurableSessionAuthority
    from nanobot.session.source_store import SourceSessionStore

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    lease = await DurableSessionAuthority(
        workspace,
        sessions_root=tmp_path / "sessions",
    ).acquire()
    try:
        sessions = lease.manager
        sessions.upsert_session_metadata(KEY, {"webui": True, "title": "Long chat"}, create_if_absent=True)
        saved = sessions.read_saved_source_metadata(KEY)
        assert saved is not None and saved.webui is True
        _write_turns(KEY, 240, source=saved.state)

        # After acquisition/save, browsing saved metadata + reading the latest
        # display window must NOT load the canonical snapshot body at all.
        canonical_loads = 0
        orig_load = SourceSessionStore.load_source_snapshot

        def count_load(self: SourceSessionStore, key: str, *args: object, **kwargs: object):
            nonlocal canonical_loads
            canonical_loads += 1
            return orig_load(self, key, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(SourceSessionStore, "load_source_snapshot", count_load)

        browsed = sessions.read_saved_source_metadata(KEY)
        assert browsed is not None and browsed.state.same_origin(saved.state)
        window = read_history_window(KEY, expected_source=browsed.state, limit=4, direction="latest")
        assert window.status == "ok"
        assert window.total_turn_count == 240
        assert window.metrics.records_decoded == 6
        assert window.metrics.index_pages_read == 1
        assert canonical_loads == 0

        # Deleting and recreating the key with the retained display file on disk
        # must NEVER return the old 240 turns.
        assert sessions.delete_session(KEY)
        assert sessions.upsert_session_metadata(KEY, {"webui": True}, create_if_absent=True)
        recreated = sessions.read_saved_source_metadata(KEY)
        assert recreated is not None and not recreated.state.same_origin(saved.state)
        recreated_window = read_history_window(
            KEY, expected_source=recreated.state, limit=4, direction="latest",
        )
        assert recreated_window.status == "empty"
        assert recreated_window.lines == []
        assert canonical_loads == 0
    finally:
        await lease.close()


async def test_gateway_http_history_and_turns_serve_bounded_windows_with_zero_canonical_loads(
    isolated_webui_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package 2 HTTP proof: latest, before, after, around, /turns, and reset 409."""
    from types import SimpleNamespace
    from urllib.parse import quote

    from websockets.datastructures import Headers
    from websockets.http11 import Request

    from nanobot.session.runtime_authority import DurableSessionAuthority
    from nanobot.session.source_store import SourceSessionStore
    from tests.webui.test_child_thread_route import _make_channel

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    lease = await DurableSessionAuthority(
        workspace,
        sessions_root=tmp_path / "sessions",
    ).acquire()
    try:
        manager = lease.manager
        assert manager.upsert_session_metadata(KEY, {"webui": True}, create_if_absent=True)
        saved = manager.read_saved_source_metadata(KEY)
        assert saved is not None
        _write_turns(KEY, 130, source=saved.state, multi_prompt_at=15)

        gateway = _make_channel(tmp_path, manager).gateway
        token = gateway.tokens.issue_api_token(300)
        headers = Headers([("Host", "127.0.0.1:8765"), ("Authorization", f"Bearer {token}")])
        connection = SimpleNamespace(
            remote_address=("127.0.0.1", 1234),
            request=SimpleNamespace(headers=headers),
        )
        encoded_key = quote(KEY, safe="")

        canonical_loads = 0
        orig_load = SourceSessionStore.load_source_snapshot

        def count_load(self: SourceSessionStore, key: str, *args: object, **kwargs: object):
            nonlocal canonical_loads
            canonical_loads += 1
            return orig_load(self, key, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(SourceSessionStore, "load_source_snapshot", count_load)

        # 1. Latest history window with 0 canonical body loads.
        latest_resp = await gateway.http.dispatch(
            connection,
            Request(f"/api/sessions/{encoded_key}/webui-thread?view=history&limit=4&direction=latest", headers),
        )
        assert latest_resp is not None and latest_resp.status_code == 200
        latest = json.loads(latest_resp.body)
        assert latest["view"] == "history"
        assert [m["content"] for m in latest["messages"]] == [
            "Prompt 128", "Reply 128", "Prompt 129", "Reply 129",
        ]
        assert latest["page"]["total_turn_count"] == 130
        assert latest["page"]["total_prompt_count"] == 131
        assert latest["page"]["coverage"] == "indexed"
        assert latest["page"]["has_more_before"] is True
        assert latest["page"]["has_more_after"] is False
        assert canonical_loads == 0

        # 2. Prompt summaries via /turns with 0 canonical body loads.
        turns_resp = await gateway.http.dispatch(
            connection,
            Request(f"/api/sessions/{encoded_key}/turns?limit=200", headers),
        )
        assert turns_resp is not None and turns_resp.status_code == 200
        turns_data = json.loads(turns_resp.body)
        assert turns_data["total_turn_count"] == 130
        assert turns_data["total_prompt_count"] == 131
        steer_prompt = next(p for p in turns_data["prompts"] if p["prompt_id"] == "evt-steer-15")
        assert steer_prompt["label"] == "Steer 15"
        assert canonical_loads == 0

        # 3. Exact around(locator), then before and after without gaps.
        around_resp = await gateway.http.dispatch(
            connection,
            Request(
                f"/api/sessions/{encoded_key}/webui-thread?view=history&direction=around"
                f"&around={quote(steer_prompt['locator'], safe='')}&limit=8",
                headers,
            ),
        )
        assert around_resp is not None and around_resp.status_code == 200
        around = json.loads(around_resp.body)
        assert around["page"]["target_prompt_id"] == "evt-steer-15"
        assert any(
            m["content"] == "Steer 15" and m.get("promptId") == "evt-steer-15"
            for m in around["messages"]
        )
        assert any(
            m["content"] == "Prompt 15" and m.get("promptId") == "evt-u-15"
            for m in around["messages"]
        )
        assert around["page"]["has_more_before"] is True
        assert around["page"]["has_more_after"] is True

        before_resp = await gateway.http.dispatch(
            connection,
            Request(
                f"/api/sessions/{encoded_key}/webui-thread?view=history&direction=before"
                f"&before={quote(around['page']['before_cursor'], safe='')}&limit=4",
                headers,
            ),
        )
        assert before_resp is not None and before_resp.status_code == 200
        older = json.loads(before_resp.body)
        assert older["page"]["last_turn_ordinal"] == around["page"]["first_turn_ordinal"] - 1

        after_resp = await gateway.http.dispatch(
            connection,
            Request(
                f"/api/sessions/{encoded_key}/webui-thread?view=history&direction=after"
                f"&after={quote(around['page']['after_cursor'], safe='')}&limit=4",
                headers,
            ),
        )
        assert after_resp is not None and after_resp.status_code == 200
        newer = json.loads(after_resp.body)
        assert newer["page"]["first_turn_ordinal"] == around["page"]["last_turn_ordinal"] + 1
        assert canonical_loads == 0

        # 4. Resetting the session invalidates old cursors (409) and hides old rows.
        assert manager.reset_session(KEY) is not None
        canonical_loads = 0
        reset_latest_resp = await gateway.http.dispatch(
            connection,
            Request(f"/api/sessions/{encoded_key}/webui-thread?view=history&limit=4&direction=latest", headers),
        )
        assert reset_latest_resp is not None and reset_latest_resp.status_code == 200
        reset_latest = json.loads(reset_latest_resp.body)
        assert reset_latest["messages"] == []
        assert canonical_loads == 0

        stale_around = await gateway.http.dispatch(
            connection,
            Request(
                f"/api/sessions/{encoded_key}/webui-thread?view=history&direction=around"
                f"&around={quote(steer_prompt['locator'], safe='')}&limit=8",
                headers,
            ),
        )
        assert stale_around is not None and stale_around.status_code == 409

        stale_before = await gateway.http.dispatch(
            connection,
            Request(
                f"/api/sessions/{encoded_key}/webui-thread?view=history&direction=before"
                f"&before={quote(latest['page']['before_cursor'], safe='')}&limit=4",
                headers,
            ),
        )
        assert stale_before is not None and stale_before.status_code == 409
    finally:
        await lease.close()


def test_preindex_tail_only_before_cursor_rebuilds_and_pages_to_start(
    isolated_webui_dir: Path,
) -> None:
    """Scrolling up from a pre-index tail_only window must page back cleanly."""
    state = _source(generation=1)
    _write_turns(KEY, 10, source=state)
    assert delete_transcript_index(KEY)

    tail = read_history_window(KEY, expected_source=state, limit=4, direction="latest")
    assert tail.status == "ok"
    assert tail.coverage == "tail_only"
    assert tail.has_more_before is True
    assert tail.before_cursor is not None
    assert not transcript_index_dir(KEY).exists()

    # Stale origin is still rejected fail-closed on a tail cursor.
    stale = read_history_window(
        KEY,
        expected_source=_source(generation=2),
        limit=4,
        direction="before",
        cursor=tail.before_cursor,
    )
    assert stale.status in {"stale_cursor", "unavailable"}

    # Paging before the tail cursor builds the index once and returns turns 6..7, etc.
    seen = [r["text"] for r in tail.lines if r.get("event") == "user"]
    cursor = tail.before_cursor
    while cursor is not None:
        older = read_history_window(
            KEY, expected_source=state, limit=4, direction="before", cursor=cursor,
        )
        assert older.status == "ok"
        assert older.coverage == "indexed"
        batch = [r["text"] for r in older.lines if r.get("event") == "user"]
        assert batch
        seen = batch + seen
        cursor = older.before_cursor
    assert seen == [f"Prompt {i}" for i in range(10)]
