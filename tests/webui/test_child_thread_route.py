"""Regression tests for the authorized child transcript read route (A3-5)."""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import pytest

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
    session_manager.upsert_session_metadata(
        child_key,
        {
            "delegation_child": {
                "parent_key": parent_key,
                "session_key": child_key,
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
                "holds_capacity": True,
                "current_run_id": "child-run",
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
        assert [child["child_id"] for child in children] == ["a3-8-current"]
        child = children[0]
        assert child["state"] == "active"
        assert child["run_sequence"] == 1
        assert child["state_sequence"] == 3
        assert child["control_epoch"] == 4
        assert child["holds_capacity"] is True
        assert child["owned_work"] is True
        assert child["can_stop"] is True
        assert child["child_run_id"] == "child-run"
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
