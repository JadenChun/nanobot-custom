"""Mutation outcomes are distinct from response acknowledgement (A3-4 AC4.4)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.webui.inbound_commands import WebUICommandRouter, WebUIRequestResult
from nanobot.webui.replay_store import DurableWebUIReplay


def _bare_router(dispatch) -> WebUICommandRouter:
    router = object.__new__(WebUICommandRouter)
    router.request_locks = {}
    router.request_tasks = {}
    router._http_router = SimpleNamespace(dispatch_webui_mutation=dispatch)
    return router


@pytest.mark.asyncio
async def test_execute_outcomes_distinguish_applied_partial_refused_timeout_and_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def dispatch_ok(*_args, **_kwargs) -> SimpleNamespace:
        return SimpleNamespace(status_code=200, body=b"{}", reason_phrase="OK")

    applied = await _bare_router(dispatch_ok).execute_webui_request(
        object(), "settings.agent.update", {}
    )
    assert applied.outcome == "applied"

    async def dispatch_partial(*_args, **_kwargs) -> SimpleNamespace:
        return SimpleNamespace(
            status_code=200,
            body=b'{"outcome": "partial"}',
            reason_phrase="OK",
        )

    partial = await _bare_router(dispatch_partial).execute_webui_request(
        object(), "settings.agent.update", {}
    )
    assert partial.outcome == "partial"

    async def dispatch_refused(*_args, **_kwargs) -> SimpleNamespace:
        return SimpleNamespace(
            status_code=409,
            body=b'{"error": {"message": "refused"}}',
            reason_phrase="Conflict",
        )

    refused = await _bare_router(dispatch_refused).execute_webui_request(
        object(), "settings.agent.update", {}
    )
    assert refused.status == 409
    assert refused.outcome == "not_applied"

    from nanobot.remote import operations as operations_module

    spec = SimpleNamespace(timeout_s=0.01, max_request_bytes=1024)
    monkeypatch.setattr(
        operations_module,
        "load_registry",
        lambda: SimpleNamespace(mutations={"workspace.access.resolve": spec}),
    )

    async def slow_dispatch(*_args, **_kwargs) -> SimpleNamespace:
        await asyncio.sleep(0.05)
        return SimpleNamespace(status_code=200, body=b"{}", reason_phrase="OK")

    timed_out = await _bare_router(slow_dispatch).execute_webui_request(
        object(), "workspace.access.resolve", {}
    )
    assert timed_out.status == 504
    assert timed_out.outcome == "uncertain"

    async def raising_dispatch(*_args, **_kwargs) -> SimpleNamespace:
        raise RuntimeError("dispatch exploded")

    unknown = await _bare_router(raising_dispatch).execute_webui_request(
        object(), "settings.agent.update", {}
    )
    assert unknown.status == 500
    assert unknown.outcome == "uncertain"


@pytest.mark.asyncio
async def test_send_webui_response_carries_outcome_in_success_and_error() -> None:
    connection = object()
    events: list[dict[str, object]] = []

    async def send(_connection, event, **payload) -> None:
        events.append({"event": event, **payload})

    router = object.__new__(WebUICommandRouter)
    router.request_locks = {}
    router.request_tasks = {}
    router._transport = SimpleNamespace(webui_send_event=send)
    router._webui_connections = {connection}

    await router.send_webui_response(
        connection,
        "req-applied",
        result={"saved": True},
        outcome="applied",
    )
    await router.send_webui_response(
        connection,
        "req-uncertain",
        status=504,
        message="timeout",
        outcome="uncertain",
    )

    assert len(events) == 2
    applied_event = events[0]
    assert applied_event["event"] == "webui_response"
    assert applied_event["request_id"] == "req-applied"
    assert applied_event["ok"] is True
    assert applied_event["result"] == {"saved": True}
    assert applied_event["outcome"] == "applied"

    uncertain_event = events[1]
    assert uncertain_event["event"] == "webui_response"
    assert uncertain_event["request_id"] == "req-uncertain"
    assert uncertain_event["ok"] is False
    error = uncertain_event["error"]
    assert isinstance(error, dict)
    assert error["status"] == 504
    assert error["message"] == "timeout"
    assert error["outcome"] == "uncertain"


@pytest.mark.asyncio
async def test_applied_result_still_invalidates_when_delivery_fails() -> None:
    connection = object()
    router = object.__new__(WebUICommandRouter)
    router.request_locks = {}
    router.request_tasks = {}
    router._webui_connections = {connection}
    router._http_router = SimpleNamespace(
        authorize_webui_mutation=lambda *_args: None
    )
    router._transport = SimpleNamespace(
        webui_send_event=AsyncMock(side_effect=RuntimeError("transport down"))
    )
    router.notify_state_changed = AsyncMock()

    async def resolve() -> WebUIRequestResult:
        return WebUIRequestResult(result={"saved": True}, outcome="applied")

    operation_task = asyncio.create_task(resolve())

    with pytest.raises(RuntimeError):
        await router.deliver_webui_request(
            connection,
            "req-delivery-fail",
            operation_task,
            action="settings.agent.update",
            payload={"agent": {"name": "test"}},
        )

    router.notify_state_changed.assert_awaited_once()
    assert router.notify_state_changed.await_args.args[0] == "settings.agent.update"


@pytest.mark.asyncio
async def test_projection_failure_is_reported_separately() -> None:
    connection = object()
    events: list[dict[str, object]] = []

    async def send(_connection, event, **payload) -> None:
        events.append({"event": event, **payload})

    router = object.__new__(WebUICommandRouter)
    router.request_locks = {}
    router.request_tasks = {}
    router._webui_connections = {connection}
    router._http_router = SimpleNamespace(
        authorize_webui_mutation=lambda *_args: None
    )
    router._transport = SimpleNamespace(webui_send_event=send)
    router.notify_state_changed = AsyncMock(
        side_effect=RuntimeError("projection refresh failed")
    )

    async def resolve() -> WebUIRequestResult:
        return WebUIRequestResult(result={"saved": True}, outcome="applied")

    operation_task = asyncio.create_task(resolve())

    await router.deliver_webui_request(
        connection,
        "req-projection-fail",
        operation_task,
        action="settings.agent.update",
        payload={"agent": {"name": "test"}},
    )

    projection_errors = [
        event for event in events if event.get("event") == "projection_error"
    ]
    assert len(projection_errors) == 1
    assert projection_errors[0]["reason"] == "projection_refresh_failed"

    responses = [event for event in events if event.get("event") == "webui_response"]
    assert len(responses) == 1
    assert responses[0]["ok"] is True
    assert responses[0]["outcome"] == "applied"


@pytest.mark.asyncio
async def test_replay_store_round_trips_outcome_and_keeps_lost_reply_applied(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "replay.json"
    store = DurableWebUIReplay(store_path)
    assert store.begin(
        "req-lost-reply",
        "settings.agent.update",
        "digest-1",
        None,
    )
    store.complete(
        "req-lost-reply",
        WebUIRequestResult(result={"ok": True}, outcome="applied"),
    )

    reloaded = DurableWebUIReplay(store_path)
    state, entry = reloaded.lookup(
        "req-lost-reply",
        "settings.agent.update",
        "digest-1",
        None,
    )
    assert state == "completed"
    assert entry is not None
    stored = entry["result"]
    assert stored["outcome"] == "applied"
    assert stored["result"] == {"ok": True}
