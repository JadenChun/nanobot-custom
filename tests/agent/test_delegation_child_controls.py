"""Regression tests for delegated child controls (A3-5 AC5.4)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.session.manager import SessionManager
from nanobot.webui.inbound_commands import WebUICommandRouter, WebUIRequestResult


def _manager(tmp_path):
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return ForegroundAgentManager(
        provider=provider,
        workspace=tmp_path,
        sessions=SessionManager(
            tmp_path / "workspace",
            sessions_root=tmp_path / "sessions",
        ),
    )


def _parent(manager, key="websocket:parent"):
    parent = manager._delegation_tree.sessions.get_or_create(key)
    manager._delegation_tree.sessions.save(parent)
    return parent


async def _live_child(manager, parent, *, assignment, attempt_id):
    admission = await manager._delegation_tree.reserve(
        parent=parent,
        parent_key=parent.key,
        assignment=assignment,
        attempt_id=attempt_id,
    )
    assert admission.child_id
    await manager._delegation_tree.activate(admission.child_id)
    return admission


def _router(chats):
    router = object.__new__(WebUICommandRouter)
    router._transport = SimpleNamespace(
        webui_connection_chats=lambda _conn: set(chats),
    )
    router.gateway = SimpleNamespace(
        child_control=AsyncMock(return_value={"ok": True, "state": "active"}),
    )
    return router


def _stop_payload():
    return {
        "parent_key": "websocket:chat-a",
        "child_id": "child-a",
        "action": "stop",
    }


@pytest.mark.asyncio
async def test_manager_control_stops_child_and_reports_native_steer_unavailable(
    tmp_path,
):
    manager = _manager(tmp_path)
    parent = _parent(manager)

    stoppable = await _live_child(
        manager,
        parent,
        assignment="do work",
        attempt_id="turn-1",
    )

    stopped = await manager.control_child(
        parent_key="websocket:parent",
        child_id=stoppable.child_id,
        action="stop",
    )
    assert stopped["ok"] is True
    assert stopped["state"] == "stopped"

    steerable = await _live_child(
        manager,
        parent,
        assignment="steer this work",
        attempt_id="turn-2",
    )
    steered = await manager.control_child(
        parent_key="websocket:parent",
        child_id=steerable.child_id,
        action="steer",
        content="focus",
    )
    assert steered["ok"] is False
    assert steered["reason"]


@pytest.mark.asyncio
async def test_manager_control_wait_observes_attached_task(tmp_path):
    manager = _manager(tmp_path)
    parent = _parent(manager)

    admission = await _live_child(
        manager,
        parent,
        assignment="wait for this work",
        attempt_id="turn-wait",
    )

    task = asyncio.create_task(asyncio.sleep(0.01))
    await manager._delegation_tree.attach_task(admission.child_id, task)

    waited = await manager.control_child(
        parent_key="websocket:parent",
        child_id=admission.child_id,
        action="wait",
    )
    assert waited["ok"] is True
    assert "state" in waited
    assert task.done()
    assert not task.cancelled()


@pytest.mark.asyncio
async def test_router_authorizes_only_attached_parent_chat():
    connection = object()

    allowed_router = _router({"chat-a"})
    allowed = await allowed_router._control_delegated_child(connection, _stop_payload())
    assert isinstance(allowed, WebUIRequestResult)
    assert allowed.outcome == "applied"
    assert allowed.result["state"] == "active"
    allowed_router.gateway.child_control.assert_awaited_once()

    denied_router = _router({"chat-b"})
    denied = await denied_router._control_delegated_child(connection, _stop_payload())
    assert isinstance(denied, WebUIRequestResult)
    assert denied.outcome == "not_applied"
    assert denied.status == 403
    denied_router.gateway.child_control.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_rejects_unknown_child_action():
    router = _router({"chat-a"})
    payload = _stop_payload()
    payload["action"] = "delete"

    rejected = await router._control_delegated_child(object(), payload)
    assert isinstance(rejected, WebUIRequestResult)
    assert rejected.outcome == "not_applied"
    assert rejected.status == 400
    router.gateway.child_control.assert_not_awaited()
