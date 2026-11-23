"""Regression coverage for WebUI child-control generation validation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from nanobot.webui.inbound_commands import WebUICommandRouter, WebUIRequestResult


def _router(callback):
    router = object.__new__(WebUICommandRouter)
    router._transport = SimpleNamespace(webui_connection_chats=lambda _conn: {"chat-a"})
    router.gateway = SimpleNamespace(child_control=callback, session_manager=object())
    return router


def _payload(**overrides):
    payload = {
        "parent_key": "websocket:chat-a",
        "child_id": "child-a",
        "runtime_generation": 0,
        "action": "stop",
    }
    payload.update(overrides)
    return payload


async def test_matching_generation_reaches_the_owner():
    callback = AsyncMock(return_value={"ok": True, "state": "stopped"})

    result = await _router(callback)._control_delegated_child(object(), _payload())

    assert isinstance(result, WebUIRequestResult)
    assert result.outcome == "applied"
    callback.assert_awaited_once()
    assert callback.await_args.args == ("websocket:chat-a", "child-a", "stop", None, 0)


async def test_missing_or_malformed_generation_is_rejected():
    for overrides in (
        {"runtime_generation": None},
        {"runtime_generation": "0"},
        {"runtime_generation": -1},
    ):
        callback = AsyncMock(return_value={"ok": True, "state": "stopped"})
        result = await _router(callback)._control_delegated_child(
            object(), _payload(**overrides)
        )

        assert isinstance(result, WebUIRequestResult)
        assert result.status == 400
        assert result.outcome == "not_applied"
        callback.assert_not_awaited()


async def test_stale_generation_is_not_applied():
    callback = AsyncMock(return_value={"ok": False, "reason": "stale_generation"})

    result = await _router(callback)._control_delegated_child(object(), _payload())

    assert isinstance(result, WebUIRequestResult)
    assert result.status == 409
    assert result.outcome == "not_applied"
    callback.assert_awaited_once()
