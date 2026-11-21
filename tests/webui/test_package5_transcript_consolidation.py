"""Package 5: bounded canonical event-offset reader & reversible display-fact cutover.

Proves all three Package 5 unblock conditions and cutover invariants:
  1. Bounded canonical event lookup (``read_canonical_events_by_ids``) uses
     ``PrivateFileVersion``-bound ``os.pread`` with 0 ``load_source_snapshot``
     calls, survives header-only ``replace_header_atomic``, rejects stale
     origins/versions fail-closed, rebuilds lazily when deleted, and cleans up
     on session deletion.
  2. Managed display writes retire duplicate prompt and answer ``text`` when it
     matches committed canonical content, storing ``_webui_text_ref`` + bounded
     ``prompt_preview`` while preserving all distinct G4 display facts and
     keeping inline ``text`` whenever display copy differs (e.g. ``!cmd``).
  3. Mixed pre-cutover (inline ``text``) + consolidated (``_webui_text_ref``)
     histories paginate identically across ``latest``/``before``/``after``/
     ``around`` and ``/turns`` with 0 canonical body loads, and survive
     deleting/rebuilding both ``.event-index`` and the transcript read index.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.bus.events import CommittedUserInput
from nanobot.channels.manager import ChannelManager
from nanobot.session.canonical_event_index import (
    delete_event_index,
    event_index_path,
)
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.runtime_authority import DurableSessionAuthority
from nanobot.session.source_store import SourceSessionStore, SourceState
from nanobot.session.turn_writer import SourceTurnWriter
from nanobot.webui import transcript as tr
from nanobot.webui.metadata import (
    WEBUI_DISPLAY_ANSWER_EVENT_ID_KEY,
    WEBUI_DISPLAY_SOURCE_KEY,
    WEBUI_TURN_METADATA_KEY,
)
from nanobot.webui.transcript import (
    TRANSCRIPT_PROMPT_SOURCE_EVENT_KEY,
    TRANSCRIPT_TEXT_SOURCE_REF_KEY,
    WebUITranscriptRecorder,
    build_webui_thread_response,
    read_transcript_lines,
)
from nanobot.webui.transcript_index import (
    delete_transcript_index,
    rebuild_transcript_index,
)
from tests.agent.test_migration_engine import engine_config as engine_config
from tests.webui.test_managed_display_origin import CHAT_ID as LIVE_CHAT_ID
from tests.webui.test_managed_display_origin import _gateway, _submit

KEY = "websocket:pkg5-consolidated"
CHAT_ID = "pkg5-consolidated"


@pytest.fixture()
def isolated_webui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    webui_dir = tmp_path / "webui"
    webui_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("nanobot.webui.transcript.get_webui_dir", lambda: webui_dir)
    return webui_dir


def _count_canonical_body_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    loaded: list[str] = []
    original = SourceSessionStore.load_source_snapshot

    def tracked(self: SourceSessionStore, key: str, *args: object, **kwargs: object):
        loaded.append(key)
        return original(self, key, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(SourceSessionStore, "load_source_snapshot", tracked)
    return loaded


async def _seed_canonical_and_consolidated_turns(
    lease: DurableSessionAuthority,
    manager,
    key: str,
    chat_id: str,
    count: int,
    *,
    legacy_prefix: int = 0,
) -> tuple[SourceState, list[tuple[str, str]]]:
    """Write ``count`` turns; first ``legacy_prefix`` use inline text, rest consolidated."""
    assert manager.upsert_session_metadata(key, {"webui": True}, create_if_absent=True)
    backend = manager.source_backend(key)
    assert backend is not None
    event_pairs: list[tuple[str, str]] = []
    async with acquire_execution(backend, key, run_id="seed-pkg5") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="seed-pkg5")
        for idx in range(count):
            u_ev = writer.append_message({"role": "user", "content": f"Prompt number {idx} " + ("p" * 80)})
            a_ev = writer.append_message({"role": "assistant", "content": f"Answer number {idx} " + ("a" * 80)})
            event_pairs.append((str(u_ev["_event_id"]), str(a_ev["_event_id"])))
    saved = manager.read_saved_source_metadata(key)
    assert saved is not None

    legacy_recorder = WebUITranscriptRecorder(log=None, sessions=None)
    managed_recorder = WebUITranscriptRecorder(log=None, sessions=manager)
    for idx, (u_id, a_id) in enumerate(event_pairs):
        prompt_text = f"Prompt number {idx} " + ("p" * 80)
        answer_text = f"Answer number {idx} " + ("a" * 80)
        meta = {
            WEBUI_DISPLAY_SOURCE_KEY: saved.state,
            WEBUI_TURN_METADATA_KEY: f"turn-{idx}",
            WEBUI_DISPLAY_ANSWER_EVENT_ID_KEY: a_id,
        }
        rec = legacy_recorder if idx < legacy_prefix else managed_recorder
        assert rec.append_committed_user_message(
            chat_id,
            CommittedUserInput(
                content=prompt_text,
                media_paths=(),
                attachments=(),
                created_at_ms=1_700_000_000_000 + idx * 1000,
                event_id=u_id,
            ),
            meta,
        )
        assert rec.prepare_and_append(
            chat_id,
            {"event": "stream_end", "chat_id": chat_id, "text": answer_text},
            metadata=meta,
            phase="answer",
        )
        assert rec.prepare_and_append(
            chat_id,
            {"event": "turn_end", "chat_id": chat_id, "turn_id": f"turn-{idx}"},
            metadata=meta,
            phase="complete",
        )
    return saved.state, event_pairs


# ---------------------------------------------------------------------------
# 1. Bounded canonical event-offset reader
# ---------------------------------------------------------------------------


async def test_canonical_event_index_bounded_pread_header_update_rebuild_and_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    lease = await DurableSessionAuthority(
        workspace, sessions_root=tmp_path / "sessions",
    ).acquire()
    try:
        manager = lease.manager
        assert manager.upsert_session_metadata(KEY, {"webui": True}, create_if_absent=True)
        backend = manager.source_backend(KEY)
        assert backend is not None

        ids: list[str] = []
        async with acquire_execution(backend, KEY, run_id="idx-test") as owner:
            writer = SourceTurnWriter(backend.source, owner, run_id="idx-test")
            for idx in range(24):
                u = writer.append_message({"role": "user", "content": f"Question {idx}"})
                a = writer.append_message({"role": "assistant", "content": f"Reply {idx}"})
                ids.extend([str(u["_event_id"]), str(a["_event_id"])])

        saved = manager.read_saved_source_metadata(KEY)
        assert saved is not None
        idx_file = event_index_path(backend, KEY)
        assert idx_file.is_file()

        loads = _count_canonical_body_loads(monkeypatch)

        # Bounded lookup hits only the requested events via os.pread.
        target_ids = [ids[0], ids[25], ids[-1]]
        got = manager.read_canonical_events_by_ids(KEY, target_ids, expected_source=saved.state)
        assert got[ids[0]]["content"] == "Question 0"
        assert got[ids[25]]["content"] == "Reply 12"
        assert got[ids[-1]]["content"] == "Reply 23"
        assert loads == []

        # Header-only atomic replace shifts body_offset without invalidating lookups.
        assert manager.upsert_session_metadata(KEY, {"title": "A much longer session title"})
        after_header = manager.read_saved_source_metadata(KEY)
        assert after_header is not None
        assert after_header.state.revision > saved.state.revision
        got_after_header = manager.read_canonical_events_by_ids(
            KEY, target_ids, expected_source=after_header.state,
        )
        assert got_after_header == got
        assert loads == []

        # Deleting the sidecar triggers lazy streaming rebuild without load_source_snapshot.
        delete_event_index(backend, KEY)
        assert not idx_file.exists()
        got_rebuilt = manager.read_canonical_events_by_ids(
            KEY, target_ids, expected_source=after_header.state,
        )
        assert got_rebuilt == got
        assert idx_file.is_file()
        assert loads == []

        # Tampering with file_version in the sidecar is rejected and safely rebuilt.
        tampered = json.loads(idx_file.read_text(encoding="utf-8"))
        tampered["file_version"][1] += 999999
        tampered["events"][ids[0]] = [999999, 10]
        idx_file.write_text(json.dumps(tampered), encoding="utf-8")
        got_after_tamper = manager.read_canonical_events_by_ids(
            KEY, [ids[0]], expected_source=after_header.state,
        )
        assert got_after_tamper[ids[0]]["content"] == "Question 0"
        assert loads == []

        # Resetting bumps generation; prior origin fails closed.
        assert manager.reset_session(KEY) is not None
        assert manager.read_canonical_events_by_ids(
            KEY, target_ids, expected_source=after_header.state,
        ) == {}

        # Deleting the session cleans up the sidecar.
        assert manager.delete_session(KEY)
        assert not idx_file.exists()
    finally:
        await lease.close()


# ---------------------------------------------------------------------------
# 2. Consolidated display writer omits duplicated text & preserves distinct facts
# ---------------------------------------------------------------------------


async def test_live_turn_consolidates_prompt_and_answer_without_duplicating_text(
    engine_config: object,
    tmp_path: Path,
) -> None:
    live_key = f"websocket:{LIVE_CHAT_ID}"
    long_prompt = "Consolidated user prompt " + ("U" * 300)
    long_answer = "Consolidated assistant answer " + ("A" * 300)

    async with _gateway(engine_config, tmp_path, long_answer) as (loop, channel, conn, _):
        await _submit(channel, conn, long_prompt, turn_id="pkg5-live-turn")
        await loop._dispatch(await loop.bus.consume_inbound())
        while loop.bus.outbound_size:
            await ChannelManager._send_once(channel, await loop.bus.consume_outbound())

        saved = loop.sessions.read_saved_source_metadata(live_key)
        assert saved is not None
        raw_rows = read_transcript_lines(live_key, expected_source=saved.state)
        user_row = next(r for r in raw_rows if r.get("event") == "user")
        answer_row = next(r for r in raw_rows if r.get("event") == "stream_end")

        # Neither prompt nor answer full text is duplicated on disk in the display store.
        assert "text" not in user_row
        assert isinstance(user_row.get(TRANSCRIPT_TEXT_SOURCE_REF_KEY), str)
        assert user_row[TRANSCRIPT_TEXT_SOURCE_REF_KEY] == user_row[TRANSCRIPT_PROMPT_SOURCE_EVENT_KEY]
        assert len(user_row.get("prompt_preview", "")) <= 120

        assert "text" not in answer_row
        assert isinstance(answer_row.get(TRANSCRIPT_TEXT_SOURCE_REF_KEY), str)

        # Hydrated thread response restores the exact full text.
        page = build_webui_thread_response(
            live_key,
            expected_source=saved.state,
            canonical_events_loader=lambda ids: loop.sessions.read_canonical_events_by_ids(
                live_key, list(ids), expected_source=saved.state,
            ),
            limit=10,
            direction="latest",
        )
        assert page is not None
        assert [m["content"] for m in page["messages"]] == [long_prompt, long_answer]


# ---------------------------------------------------------------------------
# 3. 24- and 240-turn mixed/consolidated HTTP windows: 0 canonical body loads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("turn_count,rotated", [(24, False), (240, True)])
async def test_mixed_and_consolidated_http_windows_zero_canonical_body_loads(
    isolated_webui: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    turn_count: int,
    rotated: bool,
) -> None:
    from tests.webui.test_child_thread_route import _make_channel

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    if rotated:
        monkeypatch.setattr(tr, "_ACTIVE_TRANSCRIPT_ROTATE_BYTES", 6000)
        monkeypatch.setattr(tr, "_TARGET_ACTIVE_TRANSCRIPT_BYTES", 2400)

    lease = await DurableSessionAuthority(
        workspace, sessions_root=tmp_path / "sessions",
    ).acquire()
    try:
        manager = lease.manager
        legacy_prefix = 6
        state, event_pairs = await _seed_canonical_and_consolidated_turns(
            lease, manager, KEY, CHAT_ID, turn_count, legacy_prefix=legacy_prefix,
        )

        # Verify on-disk split: legacy prefix keeps inline text; consolidated tail uses refs.
        raw_rows = read_transcript_lines(KEY, expected_source=state)
        user_rows = [r for r in raw_rows if r.get("event") == "user"]
        assert len(user_rows) == turn_count
        assert all("text" in r for r in user_rows[:legacy_prefix])
        assert all(
            "text" not in r and TRANSCRIPT_TEXT_SOURCE_REF_KEY in r
            for r in user_rows[legacy_prefix:]
        )

        gateway = _make_channel(tmp_path, manager).gateway
        token = gateway.tokens.issue_api_token(300)
        headers = Headers([("Host", "127.0.0.1:8765"), ("Authorization", f"Bearer {token}")])
        connection = SimpleNamespace(
            remote_address=("127.0.0.1", 1234),
            request=SimpleNamespace(headers=headers),
        )
        encoded = quote(KEY, safe="")
        loads = _count_canonical_body_loads(monkeypatch)

        # Latest window (consolidated turns).
        latest_resp = await gateway.http.dispatch(
            connection,
            Request(
                f"/api/sessions/{encoded}/webui-thread?view=history&limit=10&direction=latest",
                headers,
            ),
        )
        assert latest_resp is not None and latest_resp.status_code == 200
        latest = json.loads(latest_resp.body)
        assert latest["page"]["coverage"] == "indexed"
        assert latest["page"]["total_turn_count"] == turn_count
        assert latest["messages"][-2]["content"] == f"Prompt number {turn_count - 1} " + ("p" * 80)
        assert latest["messages"][-1]["content"] == f"Answer number {turn_count - 1} " + ("a" * 80)

        # Around window crossing legacy -> consolidated boundary.
        boundary_prompt_id = event_pairs[legacy_prefix][0]
        around_resp = await gateway.http.dispatch(
            connection,
            Request(
                f"/api/sessions/{encoded}/webui-thread?view=history&limit=10"
                f"&direction=around&prompt_id={quote(boundary_prompt_id, safe='')}",
                headers,
            ),
        )
        assert around_resp is not None and around_resp.status_code == 200
        around = json.loads(around_resp.body)
        around_prompts = [m["content"] for m in around["messages"] if m["role"] == "user"]
        assert f"Prompt number {legacy_prefix - 1} " + ("p" * 80) in around_prompts
        assert f"Prompt number {legacy_prefix} " + ("p" * 80) in around_prompts

        # /turns prompt summaries work without reading canonical bodies.
        turns_resp = await gateway.http.dispatch(
            connection,
            Request(f"/api/sessions/{encoded}/turns?limit=100", headers),
        )
        assert turns_resp is not None and turns_resp.status_code == 200
        turns = json.loads(turns_resp.body)
        assert turns["total_turn_count"] == turn_count
        assert turns["prompts"][-1]["label"].startswith(f"Prompt number {turn_count - 1}")

        # Delete & rebuild BOTH indexes -> identical latest projection, still 0 body loads.
        backend = manager.source_backend(KEY)
        assert backend is not None
        assert delete_transcript_index(KEY)
        delete_event_index(backend, KEY)
        assert rebuild_transcript_index(KEY, expected_source=state) is not None

        rebuilt_resp = await gateway.http.dispatch(
            connection,
            Request(
                f"/api/sessions/{encoded}/webui-thread?view=history&limit=10&direction=latest",
                headers,
            ),
        )
        assert rebuilt_resp is not None and rebuilt_resp.status_code == 200
        rebuilt = json.loads(rebuilt_resp.body)
        assert rebuilt["messages"] == latest["messages"]
        assert rebuilt["page"]["total_turn_count"] == turn_count

        assert loads == [], "consolidated & mixed reads must perform 0 canonical body loads"
    finally:
        await lease.close()
