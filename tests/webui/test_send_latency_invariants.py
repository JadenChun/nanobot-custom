"""Regression tests for native Desktop send-latency fixes."""

from __future__ import annotations

import asyncio
import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from filelock import Timeout
from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest

from kohaku.agent.skills import SkillsLoader
from kohaku.bus.queue import MessageBus
from kohaku.channels.websocket.runtime import WebSocketChannel, WebSocketConfig
from kohaku.session.history_visibility import HIDDEN_HISTORY_META
from kohaku.session.manager import Session, SessionManager
from kohaku.utils import helpers
from kohaku.utils.helpers import estimate_prompt_tokens
from kohaku.webui import session_list_index
from kohaku.webui.gateway_services import build_gateway_services
from kohaku.webui.session_list_index import list_webui_sessions


@pytest.fixture(autouse=True)
def _isolate_runtime_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)


def _make_gateway_channel(tmp_path: Path) -> tuple[WebSocketChannel, SessionManager]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    session_manager = SessionManager(workspace, sessions_root=tmp_path / "sessions")
    config = WebSocketConfig.model_validate({
        "enabled": True,
        "allowFrom": ["*"],
        "host": "127.0.0.1",
        "port": 0,
        "path": "/",
        "websocketRequiresToken": False,
    })
    bus = MessageBus()
    gateway = build_gateway_services(
        config=config,
        bus=bus,
        session_manager=session_manager,
        static_dist_path=None,
        workspace_path=workspace,
        default_restrict_to_workspace=False,
        runtime_model_name=None,
        runtime_surface="desktop",
        runtime_capabilities_overrides=None,
    )
    channel = WebSocketChannel(config, bus, gateway=gateway)
    return channel, session_manager


async def test_desktop_webui_thread_and_file_preview_offload_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel, session_manager = _make_gateway_channel(tmp_path)
    key = "websocket:desktop-offload"
    session = Session(key=key, metadata={"webui": True})
    session.add_message("user", "hello desktop")
    session.add_message("assistant", "hi there")
    session_manager.save(session)
    session_manager._jsonl_store.source.upgrade_source_session(  # pyright: ignore[reportPrivateUsage]
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )

    sample_file = tmp_path / "workspace" / "note.txt"
    sample_file.write_text("preview content", encoding="utf-8")

    loop_thread_id = threading.get_ident()
    build_thread_ids: list[int] = []
    stop_target_thread_ids: list[int] = []
    preview_thread_ids: list[int] = []

    original_build = channel.gateway.http._build_webui_thread_payload
    original_preview = channel.gateway.http._handle_file_preview

    def observed_build(*args: Any, **kwargs: Any) -> Any:
        build_thread_ids.append(threading.get_ident())
        return original_build(*args, **kwargs)

    def observed_stop_target(_key: str, *, source_snapshot: Any = None) -> dict[str, Any] | None:
        stop_target_thread_ids.append(threading.get_ident())
        return {"kind": "run", "run_id": "run-1"}

    def observed_preview(*args: Any, **kwargs: Any) -> Any:
        preview_thread_ids.append(threading.get_ident())
        return original_preview(*args, **kwargs)

    monkeypatch.setattr(channel.gateway.http, "_build_webui_thread_payload", observed_build)
    monkeypatch.setattr(channel.gateway.http, "stop_target_for_key", observed_stop_target)
    monkeypatch.setattr(channel.gateway.http, "_handle_file_preview", observed_preview)

    encoded_key = quote(key, safe="")
    auth_header = ("Authorization", f"Bearer {channel.gateway.http.tokens.issue_api_token(60)}")
    thread_req = WsRequest(
        f"/api/sessions/{encoded_key}/webui-thread",
        Headers([auth_header, ("Accept-Encoding", "gzip")]),
    )
    thread_resp = await channel.gateway.http._dispatch_session_routes(
        None,
        thread_req,
        f"/api/sessions/{encoded_key}/webui-thread",
    )
    assert thread_resp is not None
    assert thread_resp.status_code == 200
    assert build_thread_ids and build_thread_ids[0] != loop_thread_id
    # stop_target reads live loop state; it must be resolved back on the event loop thread.
    assert stop_target_thread_ids == [loop_thread_id]
    raw_body = (
        gzip.decompress(thread_resp.body)
        if thread_resp.headers.get("Content-Encoding") == "gzip"
        else thread_resp.body
    )
    payload = json.loads(raw_body)
    assert payload["sessionKey"] == key
    assert payload["stop_target"] == {"kind": "run", "run_id": "run-1"}

    preview_path = quote(str(sample_file), safe="")
    preview_req = WsRequest(
        f"/api/sessions/{encoded_key}/file-preview?path={preview_path}",
        Headers([auth_header]),
    )
    preview_resp = await channel.gateway.http._dispatch_session_routes(
        None,
        preview_req,
        f"/api/sessions/{encoded_key}/file-preview",
    )
    assert preview_resp is not None
    assert preview_resp.status_code == 200
    assert preview_thread_ids and preview_thread_ids[0] != loop_thread_id


async def test_message_accepted_emitted_before_persist_scope_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel, _session_manager = _make_gateway_channel(tmp_path)
    sent_frames: list[dict[str, Any]] = []
    persist_entered = threading.Event()
    release_persist = threading.Event()
    original_persist_scope = channel.gateway.workspaces.persist_scope
    persist_calls: list[str] = []

    class _FakeConnection:
        remote_address = ("127.0.0.1", 55555)
        request = WsRequest("/", Headers())

        async def send(self, raw: str) -> None:
            sent_frames.append(json.loads(raw))

    def slow_persist_scope(chat_id: str, *args: Any, **kwargs: Any) -> Any:
        persist_calls.append(chat_id)
        persist_entered.set()
        assert release_persist.wait(timeout=5.0)
        return original_persist_scope(chat_id, *args, **kwargs)

    monkeypatch.setattr(channel.gateway.workspaces, "persist_scope", slow_persist_scope)
    conn = _FakeConnection()

    envelope = {
        "type": "message",
        "chat_id": "instant-ack-chat",
        "content": "ping",
        "turn_id": "turn-ack-1",
        "webui": True,
    }
    task = asyncio.create_task(channel._dispatch_envelope(conn, "desktop-client", envelope))  # type: ignore[arg-type]
    try:
        # Wait until persist_scope has started in the worker thread.
        await asyncio.to_thread(persist_entered.wait, 5.0)
        assert persist_entered.is_set()
        accepted_events = [f for f in sent_frames if f.get("event") == "message_accepted"]
        assert len(accepted_events) == 1
        assert accepted_events[0]["turn_id"] == "turn-ack-1"
    finally:
        release_persist.set()
    await task
    assert persist_calls == ["instant-ack-chat"]


def test_session_files_lock_wakes_contending_thread_without_poll_delay(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    lock = manager._jsonl_store._session_files_lock  # pyright: ignore[reportPrivateUsage]

    # Re-entrancy and timeout=0 on same thread
    with lock:
        assert lock.is_locked
        with lock.acquire(timeout=0):
            assert lock.is_locked
    assert not lock.is_locked

    # Contending thread: timeout=0 fails immediately; blocking acquire wakes promptly on release
    holder_ready = threading.Event()
    release_holder = threading.Event()
    waiter_started = threading.Event()
    released_at: list[float] = []
    acquired_at: list[float] = []

    def hold_lock() -> None:
        with lock:
            holder_ready.set()
            assert release_holder.wait(timeout=5.0)
            released_at.append(time.perf_counter())

    def contend_lock() -> None:
        assert holder_ready.wait(timeout=5.0)
        with pytest.raises(Timeout):
            lock.acquire(timeout=0)
        waiter_started.set()
        with lock:
            acquired_at.append(time.perf_counter())

    with ThreadPoolExecutor(max_workers=2) as pool:
        holder_fut = pool.submit(hold_lock)
        waiter_fut = pool.submit(contend_lock)
        assert waiter_started.wait(timeout=5.0)
        time.sleep(0.01)
        release_holder.set()
        holder_fut.result(timeout=5.0)
        waiter_fut.result(timeout=5.0)

    assert not lock.is_locked
    assert acquired_at and released_at
    # filelock's default 50ms poll sleep would average ~25-50ms; futex RLock wakes in <20ms.
    assert (acquired_at[0] - released_at[0]) < 0.03


def test_session_list_index_tail_seek_and_unlocked_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    key = "websocket:large-tail-session"
    session_path = manager._get_session_path(key)  # pyright: ignore[reportPrivateUsage]

    header = {
        "_type": "metadata",
        "key": key,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "metadata": {"webui": True},
    }
    first_user = {
        "role": "user",
        "content": "first preview question",
        "timestamp": "2026-01-01T00:01:00",
    }
    filler_tool = {
        "role": "tool",
        "content": "x" * 16_384,
        "timestamp": "2026-01-01T00:02:00",
    }
    latest_assistant = {
        "role": "assistant",
        "content": "latest visible reply",
        "timestamp": "2026-01-01T01:23:45",
    }
    trailing_hidden = {
        "role": "user",
        "content": "hidden",
        "timestamp": "2026-01-01T02:00:00",
        HIDDEN_HISTORY_META: True,
    }
    lines = [
        json.dumps(header),
        json.dumps(first_user),
        *[json.dumps(filler_tool) for _ in range(12)],
        json.dumps(latest_assistant),
        json.dumps(trailing_hidden),
    ]
    session_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert session_path.stat().st_size > session_list_index._TAIL_SEEK_THRESHOLD_BYTES  # pyright: ignore[reportPrivateUsage]

    write_lock_states: list[bool] = []
    original_write = session_list_index._write_index_rows  # pyright: ignore[reportPrivateUsage]

    def observed_write(sessions_dir: Path, rows: list[dict[str, Any]]) -> None:
        write_lock_states.append(manager._jsonl_store.session_files_locked)  # pyright: ignore[reportPrivateUsage]
        original_write(sessions_dir, rows)

    monkeypatch.setattr(session_list_index, "_write_index_rows", observed_write)

    rows = list_webui_sessions(manager)
    assert len(rows) == 1
    assert rows[0]["key"] == key
    assert rows[0]["preview"] == "first preview question"
    assert rows[0]["updated_at"] == "2026-01-01T01:23:45"
    assert write_lock_states == [False]

    # Second call should reuse the cached index without rewriting
    rows_again = list_webui_sessions(manager)
    assert rows_again == rows
    assert len(write_lock_states) == 1


def test_estimate_prompt_tokens_caches_deepcopied_tool_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers._get_token_encoding.cache_clear()  # pyright: ignore[reportPrivateUsage]
    helpers._TOOLS_TOKEN_CACHE.clear()  # pyright: ignore[reportPrivateUsage]

    class FakeEncoding:
        name = "fake-encoding"

        def __init__(self) -> None:
            self.encoded: list[str] = []

        def encode(self, text: str) -> list[int]:
            self.encoded.append(text)
            return list(range(max(1, len(text) // 4)))

    fake_encoding = FakeEncoding()
    monkeypatch.setattr(helpers.tiktoken, "get_encoding", lambda _name: fake_encoding)

    tools = [{"type": "function", "function": {"name": "demo", "description": "cached"}}]
    messages = [{"role": "user", "content": "hello"}]

    first = estimate_prompt_tokens(messages, deepcopy(tools))
    second = estimate_prompt_tokens(messages, deepcopy(tools))

    assert first == second
    rendered_tools = "\n" + json.dumps(tools, ensure_ascii=False)
    assert fake_encoding.encoded.count(rendered_tools) == 1

    helpers._get_token_encoding.cache_clear()  # pyright: ignore[reportPrivateUsage]
    helpers._TOOLS_TOKEN_CACHE.clear()  # pyright: ignore[reportPrivateUsage]


def test_skills_loader_avoids_quadratic_directory_walks_and_invalidates_on_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    skills_dir = workspace / "skills"
    for i in range(10):
        skill_dir = skills_dir / f"skill-{i}"
        skill_dir.mkdir(parents=True)
        always_line = "always: true\n" if i == 0 else ""
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: skill-{i}\ndescription: Desc {i}\n{always_line}---\nBody {i}\n",
            encoding="utf-8",
        )

    loader = SkillsLoader(workspace, builtin_skills_dir=tmp_path / "empty-builtin")
    walk_calls = 0
    original_entries = loader._skill_entries_from_dir  # pyright: ignore[reportPrivateUsage]

    def counted_entries(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
        nonlocal walk_calls
        walk_calls += 1
        return original_entries(*args, **kwargs)

    monkeypatch.setattr(loader, "_skill_entries_from_dir", counted_entries)

    listed = loader.list_skills(filter_unavailable=True)
    assert len(listed) == 10
    assert walk_calls == 1

    walk_calls = 0
    always = loader.get_always_skills()
    assert always == ["skill-0"]
    assert walk_calls == 1

    walk_calls = 0
    summary = loader.build_skills_summary()
    assert "Desc 0" in summary
    assert walk_calls == 1

    # Editing a skill invalidates the frontmatter cache via (mtime_ns, size)
    target = skills_dir / "skill-0" / "SKILL.md"
    target.write_text(
        "---\nname: skill-0\ndescription: Updated Description 0\nalways: true\n---\nBody 0\n",
        encoding="utf-8",
    )
    updated_summary = loader.build_skills_summary()
    assert "Updated Description 0" in updated_summary
