"""Regression tests for the authorized child transcript read route (A3-5)."""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote, urlencode

import pytest

from nanobot.agent.tools.execution import KnownToolResult
from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.channels.websocket.tests.ws_test_client import (
    InProcessHttpChannel,
    http_get,
)
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import Session, SessionManager
from nanobot.session.runtime_authority import RuntimeSessionMetadata
from nanobot.session.turn_writer import SourceTurnWriter
from nanobot.webui.gateway_services import build_gateway_services
from nanobot.webui.ws_http import _decode_child_cursor

_PARENT_KEY = "websocket:chat-a"
_CHILD_KEY = "delegated:a3-5-child"
_TRANSCRIPT = [
    {"role": "user", "content": "assignment"},
    {"role": "assistant", "content": "result"},
]
_IN_PROCESS_PORTS = itertools.count(41_100)


@pytest.fixture(autouse=True)
def _isolate_runtime_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)


def _next_port() -> int:
    """Allocate a unique in-process route key, not a native listening socket."""
    return next(_IN_PROCESS_PORTS)


def _make_channel(tmp_path: Path, session_manager: SessionManager) -> InProcessHttpChannel:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config = WebSocketConfig.model_validate({
        "enabled": True,
        "allowFrom": ["*"],
        "host": "127.0.0.1",
        "port": _next_port(),
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
        runtime_surface="browser",
        runtime_capabilities_overrides=None,
    )
    return InProcessHttpChannel(config, bus, gateway=gateway)


def _make_session_manager(tmp_path: Path) -> SessionManager:
    return SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "sessions")


def _seed_child_session(
    session_manager: SessionManager,
    *,
    child_key: str = _CHILD_KEY,
    parent_key: str = _PARENT_KEY,
    transcript: list[dict[str, str]] | None = None,
) -> None:
    session_manager.save(Session(key=parent_key))
    parent = session_manager.read_session_snapshot(parent_key)
    assert parent is not None
    session_manager.upsert_session_metadata(
        child_key,
        {
            "delegation_child": {
                "parent_key": parent_key,
                "session_key": child_key,
                "parent_incarnation": parent.incarnation,
                "transcript": [
                    dict(message) for message in (transcript or _TRANSCRIPT)
                ],
            }
        },
        create_if_absent=True,
    )


async def _seed_canonical_child(
    session_manager: SessionManager,
    *,
    child_key: str,
    parent_key: str,
    messages: list[dict[str, str]],
) -> None:
    parent = Session(key=parent_key)
    session_manager.save(parent)
    child = session_manager.get_or_create(child_key)
    session_manager.save(child)
    backend = session_manager._jsonl_store
    backend.source.upgrade_source_session(
        parent_key,
        expected_revision=parent.revision,
        expected_generation=parent.generation,
        expected_incarnation=parent.incarnation,
    )
    session_manager.invalidate(parent_key)
    child = session_manager.get_or_create(child_key)
    backend.source.upgrade_source_session(
        child_key,
        expected_revision=child.revision,
        expected_generation=child.generation,
        expected_incarnation=child.incarnation,
    )
    session_manager.invalidate(child_key)
    child_backend = session_manager.source_backend(child_key)
    assert child_backend is not None
    async with acquire_execution(child_backend, child_key, run_id="child-run") as owner:
        writer = SourceTurnWriter(child_backend.source, owner, run_id="child-run")
        for message in messages:
            writer.append_message(message)
        child_state = owner.read_snapshot().state
    parent_state = backend.source.load_source_snapshot(parent_key).state
    session_manager.upsert_session_metadata(
        child_key,
        {
            "delegation_child": {
                "child_id": child_key.removeprefix("delegated:"),
                "session_key": child_key,
                "parent_key": parent_key,
                "parent_incarnation": parent_state.incarnation,
                "profile": "delegated",
                "state": "completed",
                "runtime_generation": 0,
                "source_state": asdict(child_state),
                "parent_source_state": asdict(parent_state),
            }
        },
        create_if_absent=False,
    )


def _thread_url(channel: InProcessHttpChannel, child_key: str) -> str:
    encoded = quote(child_key, safe="")
    return (
        f"http://127.0.0.1:{channel.config.port}"
        f"/api/sessions/{encoded}/webui-thread"
    )


def _auth_headers(channel: InProcessHttpChannel) -> dict[str, str]:
    token = channel.gateway.tokens.issue_api_token(300)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize(
    "cursor_payload",
    [
        {"v": True, "kind": "child_history", "anchor": "event-1"},
        {"v": 1, "kind": "child_history", "anchor": True},
    ],
)
def test_child_cursor_rejects_invalid_json_field_types(cursor_payload: dict[str, object]) -> None:
    encoded = base64.urlsafe_b64encode(
        json.dumps(cursor_payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")

    assert _decode_child_cursor(encoded, kind="child_history") is None


@pytest.mark.asyncio
async def test_child_thread_returns_bounded_transcript(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    _seed_child_session(session_manager)
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        response = await http_get(
            _thread_url(channel, _CHILD_KEY),
            headers=_auth_headers(channel),
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert [
            (message["role"], message["content"]) for message in body["messages"]
        ] == [
            ("user", "assignment"),
            ("assistant", "result"),
        ]
        assert body["read_only"] is True
        assert body["history_kind"] == "summary"
        assert body["source"] is None
        assert all(row["createdAt"] == 0 and row["timestampAvailable"] is False for row in body["messages"])
        assert body["page"]["loaded_message_count"] == 2
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
async def test_child_thread_reads_canonical_full_history(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = "delegated:a3-8-canonical-child"
    messages = [
        {"role": "user", "content": "assignment one"},
        {"role": "assistant", "content": "result one"},
        {"role": "user", "content": "assignment two"},
        {"role": "assistant", "content": "result two"},
    ]
    await _seed_canonical_child(
        session_manager,
        child_key=child_key,
        parent_key=_PARENT_KEY,
        messages=messages,
    )
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        response = await http_get(
            _thread_url(channel, child_key),
            headers=_auth_headers(channel),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [
            (message["role"], message["content"]) for message in body["messages"]
        ] == [
            ("user", "assignment one"),
            ("assistant", "result one"),
            ("user", "assignment two"),
            ("assistant", "result two"),
        ]
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
async def test_child_thread_rejects_stale_canonical_source(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = "delegated:a3-8-stale-source"
    await _seed_canonical_child(
        session_manager,
        child_key=child_key,
        parent_key=_PARENT_KEY,
        messages=[
            {"role": "user", "content": "private assignment"},
            {"role": "assistant", "content": "private result"},
        ],
    )
    raw = session_manager.read_session_metadata(child_key)
    metadata = raw.get("metadata") if isinstance(raw, dict) else None
    payload = (
        metadata.get("delegation_child") if isinstance(metadata, dict) else None
    )
    assert isinstance(payload, dict)
    source_state = payload.get("source_state")
    assert isinstance(source_state, dict)
    source_state["generation"] = int(source_state["generation"]) + 1
    session_manager.upsert_session_metadata(
        child_key,
        {"delegation_child": payload},
        create_if_absent=False,
    )
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        response = await http_get(
            _thread_url(channel, child_key),
            headers=_auth_headers(channel),
        )
        assert response.status_code == 404
        assert "private assignment" not in response.text
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
async def test_child_thread_does_not_fall_back_from_malformed_canonical_reference(
    tmp_path: Path,
) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = "delegated:a3-malformed-source-ref"
    _seed_child_session(
        session_manager,
        child_key=child_key,
        transcript=[
            {"role": "user", "content": "private summary fallback"},
            {"role": "assistant", "content": "private result"},
        ],
    )
    session_manager.upsert_session_metadata(
        child_key,
        {
            "delegation_child": {
                "parent_key": _PARENT_KEY,
                "session_key": child_key,
                "parent_incarnation": session_manager.read_session_snapshot(
                    _PARENT_KEY
                ).incarnation,
                "profile": "delegated",
                "state": "completed",
                "runtime_generation": 0,
                "source_state": {"store_identity": "malformed"},
                "transcript": [
                    {"role": "user", "content": "private summary fallback"},
                    {"role": "assistant", "content": "private result"},
                ],
            }
        },
        create_if_absent=False,
    )
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        response = await http_get(
            _thread_url(channel, child_key),
            headers=_auth_headers(channel),
        )
        assert response.status_code == 404
        assert "private summary fallback" not in response.text
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
@pytest.mark.parametrize("source_name", ["parent_source_state", "root_source_state"])
async def test_child_thread_rejects_replaced_ancestry_source(
    tmp_path: Path,
    source_name: str,
) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = f"delegated:a3-replaced-{source_name}"
    await _seed_canonical_child(
        session_manager,
        child_key=child_key,
        parent_key=_PARENT_KEY,
        messages=[
            {"role": "user", "content": "private canonical history"},
            {"role": "assistant", "content": "private result"},
        ],
    )
    metadata = session_manager.read_session_metadata(child_key)
    metadata_value = metadata.get("metadata") if isinstance(metadata, dict) else None
    payload = metadata_value.get("delegation_child") if isinstance(metadata_value, dict) else None
    assert isinstance(payload, dict)
    expected = payload.get("parent_source_state")
    assert isinstance(expected, dict)
    replacement = dict(expected)
    replacement["generation"] = int(replacement["generation"]) + 1
    payload[source_name] = replacement
    payload["transcript"] = [
        {"role": "user", "content": "must not be used as fallback"},
    ]
    session_manager.upsert_session_metadata(
        child_key,
        {"delegation_child": payload},
        create_if_absent=False,
    )
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        response = await http_get(
            _thread_url(channel, child_key),
            headers=_auth_headers(channel),
        )
        assert response.status_code == 404
        assert "private canonical history" not in response.text
        assert "must not be used as fallback" not in response.text
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
async def test_delegated_children_hydration_returns_current_children(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    parent = Session(key=_PARENT_KEY)
    session_manager.save(parent)
    current = session_manager.read_session_snapshot(_PARENT_KEY)
    assert current is not None
    session_manager.upsert_session_metadata(
        "delegated:a3-8-current",
        {
            "delegation_child": {
                "child_id": "a3-8-current",
                "session_key": "delegated:a3-8-current",
                "parent_key": _PARENT_KEY,
                "parent_incarnation": current.incarnation,
                "profile": "delegated",
                "state": "active",
                "runtime_generation": 2,
                "run_sequence": 1,
                "state_sequence": 3,
                "control_epoch": 4,
                "attempt_id": "parent-attempt",
                "task_label": "read and summarize",
                "task_label_truncated": False,
                "result_summary": "completed successfully",
                "result_summary_truncated": False,
                "holds_capacity": True,
                "current_run_id": "child-run",
                "updated_at": 100.0,
            },
            "_runtime_session": RuntimeSessionMetadata(
                profile="delegated",
                execution_kind="native",
                delegation_depth=1,
                parent_key=_PARENT_KEY,
                parent_incarnation=current.incarnation,
                backend_binding="native-runner",
            ).payload(),
        },
        create_if_absent=True,
    )
    session_manager.upsert_session_metadata(
        "delegated:a3-8-current-newer",
        {
            "delegation_child": {
                "child_id": "a3-8-current-newer",
                "session_key": "delegated:a3-8-current-newer",
                "parent_key": _PARENT_KEY,
                "parent_incarnation": current.incarnation,
                "profile": "delegated",
                "state": "completed",
                "runtime_generation": 1,
                "holds_capacity": False,
                "updated_at": 200.0,
            },
            "_runtime_session": RuntimeSessionMetadata(
                profile="delegated",
                execution_kind="native",
                delegation_depth=1,
                parent_key=_PARENT_KEY,
                parent_incarnation=current.incarnation,
                backend_binding="native-runner",
            ).payload(),
        },
        create_if_absent=True,
    )
    session_manager.upsert_session_metadata(
        "delegated:a3-8-stale-parent-source",
        {
            "delegation_child": {
                "child_id": "a3-8-stale-parent-source",
                "session_key": "delegated:a3-8-stale-parent-source",
                "parent_key": _PARENT_KEY,
                "parent_incarnation": current.incarnation,
                "parent_source_state": {
                    "store_identity": "wrong-store",
                    "session_key": _PARENT_KEY,
                    "incarnation": current.incarnation,
                    "generation": 1,
                },
                "profile": "delegated",
                "state": "active",
                "runtime_generation": 0,
                "updated_at": 300.0,
            }
        },
        create_if_absent=True,
    )
    session_manager.upsert_session_metadata(
        "delegated:a3-8-stale",
        {
            "delegation_child": {
                "child_id": "a3-8-stale",
                "session_key": "delegated:a3-8-stale",
                "parent_key": _PARENT_KEY,
                "parent_incarnation": "old-incarnation",
                "profile": "delegated",
                "state": "active",
                "runtime_generation": 0,
            }
        },
        create_if_absent=True,
    )
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        encoded = quote(_PARENT_KEY, safe="")
        response = await http_get(
            f"http://127.0.0.1:{channel.config.port}"
            f"/api/sessions/{encoded}/delegated-children",
            headers=_auth_headers(channel),
        )
        assert response.status_code == 200, response.text
        children = response.json()["children"]
        assert [child["child_id"] for child in children] == [
            "a3-8-current-newer",
            "a3-8-current",
        ]
        child = next(item for item in children if item["child_id"] == "a3-8-current")
        assert child["history_kind"] == "summary"
        assert child["state"] == "active"
        assert child["run_sequence"] == 1
        assert child["state_sequence"] == 3
        assert child["control_epoch"] == 4
        assert child["holds_capacity"] is True
        assert child["owned_work"] is True
        assert child["can_stop"] is False
        assert child["parent_session_key"] == _PARENT_KEY
        assert child["parent_run_id"] == "parent-attempt"
        assert child["child_run_id"] == "child-run"
        assert child["root_key"] == _PARENT_KEY
        assert child["task_label"] == "read and summarize"
        assert child["result_summary"] == "completed successfully"
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
async def test_hydration_marks_only_genuine_summaries_and_reads_fresh_stop_target(tmp_path):
    from tests.agent.test_stop_expectation import target_value

    sessions = _make_session_manager(tmp_path)
    await _seed_canonical_child(sessions, child_key=_CHILD_KEY, parent_key=_PARENT_KEY,
                                messages=_TRANSCRIPT)
    parent = sessions.read_session_snapshot(_PARENT_KEY)
    assert parent is not None
    for suffix, source in (("summary", None), ("corrupt", {"store_identity": "broken"})):
        payload = {"child_id": suffix, "session_key": f"delegated:{suffix}",
                   "parent_key": _PARENT_KEY, "parent_incarnation": parent.incarnation,
                   "profile": "delegated", "state": "completed", "runtime_generation": 0,
                   "source_state": source}
        sessions.upsert_session_metadata(f"delegated:{suffix}", {"delegation_child": payload},
                                        create_if_absent=True)
    channel = _make_channel(tmp_path, sessions)
    target = target_value()
    target["source"]["session_key"] = _PARENT_KEY
    targets = iter((target, None))
    channel.gateway.http.stop_target_for_key = lambda key: next(targets)
    url = (f"http://127.0.0.1:{channel.config.port}/api/sessions/"
           f"{quote(_PARENT_KEY, safe='')}/delegated-children")
    try:
        response = await http_get(url, headers=_auth_headers(channel))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["stop_target"] == target
        children = {row["child_session_key"]: row for row in body["children"]}
        assert set(children) == {_CHILD_KEY, "delegated:summary"}
        assert children[_CHILD_KEY]["history_kind"] == "canonical"
        summary = children["delegated:summary"]
        assert summary["history_kind"] == "summary"
        assert summary["task_label"] is None and summary["child_store_identity"] is None
        assert summary["can_stop"] is False
        refreshed = await http_get(url, headers=_auth_headers(channel))
        assert refreshed.status_code == 200
        assert refreshed.json()["stop_target"] is None
    finally:
        await channel.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing", "null", "malformed", "mismatched", "unreadable"])
async def test_canonical_child_cannot_downgrade_to_summary_in_hydration_or_detail(
    tmp_path, monkeypatch, fault,
):
    sessions = _make_session_manager(tmp_path)
    await _seed_canonical_child(sessions, child_key=_CHILD_KEY, parent_key=_PARENT_KEY,
                                messages=_TRANSCRIPT)
    payload = sessions.read_session_metadata(_CHILD_KEY)["metadata"]["delegation_child"]
    if fault in {"missing", "unreadable"}:
        payload.pop("source_state")
    elif fault == "null":
        payload["source_state"] = None
    elif fault == "malformed":
        payload["source_state"] = {"store_identity": "broken"}
    else:
        payload["source_state"]["generation"] += 1
    payload["transcript"] = [{"role": "assistant", "content": "must never downgrade"}]
    sessions.upsert_session_metadata(_CHILD_KEY, {"delegation_child": payload})
    if fault == "unreadable":
        source = sessions.source_backend(_CHILD_KEY).source
        original_load = source.load_source_snapshot

        def load(key):
            if key == _CHILD_KEY:
                raise OSError("canonical child history is unavailable")
            return original_load(key)

        monkeypatch.setattr(source, "load_source_snapshot", load)
    channel = _make_channel(tmp_path, sessions)
    try:
        response = await http_get(
            f"http://127.0.0.1:{channel.config.port}/api/sessions/{quote(_PARENT_KEY, safe='')}/delegated-children",
            headers=_auth_headers(channel),
        )
        assert response.status_code == 200
        assert response.json()["children"] == []
        detail = await http_get(_thread_url(channel, _CHILD_KEY), headers=_auth_headers(channel))
        assert detail.status_code == 404
        assert "must never downgrade" not in detail.text
    finally:
        await channel.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("canonical", [False, True])
async def test_child_history_uses_only_persisted_event_timestamps(tmp_path, canonical):
    sessions = _make_session_manager(tmp_path)
    messages = [
        {"role": "user", "content": "unknown time"},
        {"role": "assistant", "content": "epoch", "created_at_ms": 0},
        {"role": "user", "content": "known", "timestamp": "2026-01-01T00:00:00+00:00"},
    ]
    if canonical:
        await _seed_canonical_child(sessions, child_key=_CHILD_KEY, parent_key=_PARENT_KEY,
                                    messages=messages)
    else:
        _seed_child_session(sessions, transcript=messages)
    channel = _make_channel(tmp_path, sessions)
    try:
        response = await http_get(_thread_url(channel, _CHILD_KEY), headers=_auth_headers(channel))
        assert response.status_code == 200, response.text
        rows = response.json()["messages"]
        by_content = {row["content"]: row for row in rows}
        if canonical:
            assert by_content["unknown time"]["createdAt"] == 0
            assert by_content["unknown time"]["timestampAvailable"] is False
        assert by_content["epoch"]["createdAt"] == 0
        assert by_content["epoch"]["timestampAvailable"] is True
        assert by_content["known"]["createdAt"] == 1767225600000
        assert by_content["known"]["timestampAvailable"] is True
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_delegated_children_hydrates_nested_descendants_in_pages(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    root = Session(key=_PARENT_KEY)
    session_manager.save(root)
    current_root = session_manager.read_session_snapshot(_PARENT_KEY)
    assert current_root is not None

    parent_key = "delegated:hydration-parent"
    session_manager.save(Session(key=parent_key))
    current_parent = session_manager.read_session_snapshot(parent_key)
    assert current_parent is not None
    session_manager.upsert_session_metadata(
        parent_key,
        {
            "delegation_child": {
                "child_id": "hydration-parent",
                "session_key": parent_key,
                "parent_key": _PARENT_KEY,
                "parent_incarnation": current_root.incarnation,
                "root_key": _PARENT_KEY,
                "profile": "delegated",
                "state": "completed",
                "runtime_generation": 0,
            }
        },
        create_if_absent=False,
    )
    session_manager.upsert_session_metadata(
        "delegated:hydration-nested",
        {
            "delegation_child": {
                "child_id": "hydration-nested",
                "session_key": "delegated:hydration-nested",
                "parent_key": parent_key,
                "parent_incarnation": current_parent.incarnation,
                "root_key": _PARENT_KEY,
                "profile": "delegated",
                "state": "completed",
                "runtime_generation": 0,
            }
        },
        create_if_absent=True,
    )
    for index in range(4):
        child_key = f"delegated:hydration-sibling-{index}"
        session_manager.upsert_session_metadata(
            child_key,
            {
                "delegation_child": {
                    "child_id": f"hydration-sibling-{index}",
                    "session_key": child_key,
                    "parent_key": _PARENT_KEY,
                    "parent_incarnation": current_root.incarnation,
                    "root_key": _PARENT_KEY,
                    "profile": "delegated",
                    "state": "completed",
                    "runtime_generation": 0,
                }
            },
            create_if_absent=True,
        )

    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        encoded = quote(_PARENT_KEY, safe="")
        base_url = (
            f"http://127.0.0.1:{channel.config.port}"
            f"/api/sessions/{encoded}/delegated-children"
        )
        headers = _auth_headers(channel)
        first = await http_get(f"{base_url}?limit=2", headers=headers)
        assert first.status_code == 200, first.text
        first_body = first.json()
        assert len(first_body["children"]) == 2
        assert first_body["page"]["loaded_message_count"] == 2
        assert first_body["page"]["has_more_before"] is True
        cursor = first_body["page"]["before_cursor"]
        assert isinstance(cursor, str) and cursor
        page_bodies = [first_body]
        while cursor:
            next_page = await http_get(
                f"{base_url}?{urlencode({'limit': 2, 'before': cursor})}",
                headers=headers,
            )
            assert next_page.status_code == 200, next_page.text
            page_body = next_page.json()
            page_bodies.append(page_body)
            cursor = page_body["page"]["before_cursor"]
        combined_ids = [
            child["child_id"]
            for body in page_bodies
            for child in body["children"]
        ]
        assert set(combined_ids) == {
            "hydration-parent",
            "hydration-nested",
            "hydration-sibling-0",
            "hydration-sibling-1",
            "hydration-sibling-2",
            "hydration-sibling-3",
        }
        nested = next(
            child for body in page_bodies for child in body["children"]
            if child["child_id"] == "hydration-nested"
        )
        assert nested["parent_session_key"] == parent_key
        assert nested["parent_run_id"] is None
        assert nested["child_run_id"] is None
        assert nested["root_key"] == _PARENT_KEY
        assert nested["history_kind"] == "summary"
        assert nested["run_sequence"] is None
        assert nested["state_sequence"] is None
        session_manager.reset_session(_PARENT_KEY)
        stale_page = await http_get(
            f"{base_url}?{urlencode({'limit': 2, 'before': first_body['page']['before_cursor']})}",
            headers=headers,
        )
        assert stale_page.status_code == 409
        assert "delegated_children_cursor_stale" in stale_page.text
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
async def test_child_thread_pages_full_canonical_messages_after_append(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = "delegated:a3-canonical-pages"
    messages: list[dict[str, object]] = []
    for index in range(220):
        if index == 219:
            content: object = [{"type": "text", "text": "structured tail"}]
        elif index == 218:
            content = "x" * 12_000
        else:
            content = f"message-{index}"
        messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": content})
    await _seed_canonical_child(
        session_manager,
        child_key=child_key,
        parent_key=_PARENT_KEY,
        messages=messages,  # type: ignore[arg-type]
    )
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        headers = _auth_headers(channel)
        url = f"{_thread_url(channel, child_key)}?limit=50&direction=latest"
        first = await http_get(url, headers=headers)
        assert first.status_code == 200, first.text
        body = first.json()
        rows = body["messages"]
        assert len(rows) == 50
        assert body["read_only"] is True
        assert body["history_kind"] == "canonical"
        assert body["parent_session_key"] == _PARENT_KEY
        assert body["root_key"] == _PARENT_KEY
        assert body["source"]["session_key"] == child_key
        assert body["page"]["loaded_message_count"] == 50
        assert body["page"]["has_more_before"] is True
        assert rows[-1]["content"] == "structured tail"
        assert len(rows[-2]["content"]) == 12_000
        assert rows[-1]["id"]
        assert rows[-1]["turnId"] == "child-run"
        cursor = body["page"]["before_cursor"]
        assert isinstance(cursor, str) and cursor

        child_backend = session_manager.source_backend(child_key)
        assert child_backend is not None
        child = session_manager.read_session_snapshot(child_key)
        assert child is not None
        async with acquire_execution(child_backend, child_key, run_id="append-run") as owner:
            writer = SourceTurnWriter(child_backend.source, owner, run_id="append-run")
            writer.append_message({"role": "assistant", "content": "appended later"})

        older = await http_get(
            f"{_thread_url(channel, child_key)}?{urlencode({'limit': 50, 'direction': 'latest', 'before': cursor})}",
            headers=headers,
        )
        assert older.status_code == 200, older.text
        older_rows = older.json()["messages"]
        assert len(older_rows) == 50
        assert older_rows[-1]["content"] == "message-169"
        assert "appended later" not in [row["content"] for row in older_rows]
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["child", "root"])
async def test_child_history_cursor_is_invalidated_by_source_reset(
    tmp_path: Path,
    replacement: str,
) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = f"delegated:a3-cursor-replacement-{replacement}"
    await _seed_canonical_child(
        session_manager,
        child_key=child_key,
        parent_key=_PARENT_KEY,
        messages=[
            {"role": "user", "content": f"message-{index}"}
            for index in range(12)
        ],  # type: ignore[list-item]
    )
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        headers = _auth_headers(channel)
        first = await http_get(
            f"{_thread_url(channel, child_key)}?limit=3&direction=latest",
            headers=headers,
        )
        assert first.status_code == 200, first.text
        cursor = first.json()["page"]["before_cursor"]
        assert isinstance(cursor, str) and cursor

        target_key = child_key if replacement == "child" else _PARENT_KEY
        target_backend = session_manager.source_backend(target_key)
        assert target_backend is not None
        old_snapshot = target_backend.source.load_source_snapshot(target_key)
        replacement_snapshot = target_backend.source.reset_source_session(
            target_key,
            old_snapshot.state,
            [],
        )
        session_manager.invalidate(target_key)

        metadata = session_manager.read_session_metadata(child_key)
        metadata_value = metadata.get("metadata") if isinstance(metadata, dict) else None
        payload = (
            metadata_value.get("delegation_child")
            if isinstance(metadata_value, dict) else None
        )
        assert isinstance(payload, dict)
        if replacement == "child":
            payload["source_state"] = asdict(replacement_snapshot.state)
        else:
            payload["parent_incarnation"] = replacement_snapshot.state.incarnation
            payload["root_key"] = _PARENT_KEY
            payload["parent_source_state"] = asdict(replacement_snapshot.state)
            payload["root_source_state"] = asdict(replacement_snapshot.state)
        payload["transcript"] = [
            {"role": "user", "content": "stale summary must not be used"},
        ]
        session_manager.upsert_session_metadata(
            child_key,
            {"delegation_child": payload},
            create_if_absent=False,
        )

        older = await http_get(
            f"{_thread_url(channel, child_key)}?{urlencode({'limit': 3, 'direction': 'latest', 'before': cursor})}",
            headers=headers,
        )
        assert older.status_code == 409, older.text
        assert "stale summary must not be used" not in older.text
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
async def test_child_thread_projects_tool_calls_and_outcomes(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = "delegated:a3-canonical-tools"
    await _seed_canonical_child(
        session_manager,
        child_key=child_key,
        parent_key=_PARENT_KEY,
        messages=[{"role": "user", "content": "run the tool"}],  # type: ignore[list-item]
    )
    child_backend = session_manager.source_backend(child_key)
    assert child_backend is not None
    async with acquire_execution(child_backend, child_key, run_id="tool-run") as owner:
        writer = SourceTurnWriter(child_backend.source, owner, run_id="tool-run")
        writer.begin_tool_batch({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "stable-tool-call",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\": \"notes.txt\"}"},
            }],
        })
        await writer.record_tool_result_event(KnownToolResult(
            0,
            "stable-tool-call",
            "read_file",
            "tool output",
            "tool output",
            {"name": "read_file", "status": "ok"},
            True,
        ))
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        response = await http_get(
            _thread_url(channel, child_key) + "?limit=10&direction=latest",
            headers=_auth_headers(channel),
        )
        assert response.status_code == 200, response.text
        rows = response.json()["messages"]
        call_row = next(row for row in rows if row.get("toolEvents", [{}])[0].get("phase") == "start")
        result_row = next(row for row in rows if row.get("role") == "tool")
        assert call_row["toolEvents"][0]["call_id"] == "stable-tool-call"
        assert call_row["toolEvents"][0]["arguments"] == {"path": "notes.txt"}
        assert result_row["id"]
        assert result_row["turnId"] == "tool-run"
        assert result_row["toolEvents"][0]["phase"] == "end"
        assert result_row["toolEvents"][0]["result"] == "tool output"
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
async def test_child_thread_requires_known_websocket_parent(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    orphan_key = "delegated:a3-5-no-parent"
    non_websocket_key = "delegated:a3-5-non-websocket-parent"
    unknown_parent_key = "delegated:a3-5-unknown-parent"

    session_manager.save(Session(key=orphan_key))
    session_manager.save(Session(key="telegram:chat-a"))
    session_manager.upsert_session_metadata(
        non_websocket_key,
        {
            "delegation_child": {
                "parent_key": "telegram:chat-a",
                "session_key": non_websocket_key,
                "transcript": [dict(message) for message in _TRANSCRIPT],
            }
        },
        create_if_absent=True,
    )
    session_manager.upsert_session_metadata(
        unknown_parent_key,
        {
            "delegation_child": {
                "parent_key": "websocket:missing-parent",
                "session_key": unknown_parent_key,
                "transcript": [dict(message) for message in _TRANSCRIPT],
            }
        },
        create_if_absent=True,
    )
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        headers = _auth_headers(channel)
        for child_key in (orphan_key, non_websocket_key, unknown_parent_key):
            response = await http_get(_thread_url(channel, child_key), headers=headers)
            assert response.status_code == 404, (child_key, response.text)
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
async def test_child_thread_requires_token(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    _seed_child_session(session_manager)
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        url = _thread_url(channel, _CHILD_KEY)
        missing = await http_get(url)
        assert missing.status_code == 401, missing.text

        incorrect = await http_get(
            url,
            headers={"Authorization": "Bearer nbwt_not-a-valid-token"},
        )
        assert incorrect.status_code == 401, incorrect.text
    finally:
        await channel.stop()
        await server_task
