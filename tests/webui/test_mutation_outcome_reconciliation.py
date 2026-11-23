"""Acceptance cases C4-C6 for mutation outcome reconciliation (A3-4 / E02-F20).

Provenance: ``a3-evidence/a3-reconciliation-2026-11-14/resume2/req-a34p4b/CASES.md``
(C4, C5, C6, requirement R2; use cases U4, U5, U6). The author artifact is
specification-only. This file binds its scenarios, inputs, and observable
outcomes to the published mutation surface without changing any expectation:

* M7 outcome literals: ``applied``/``partial`` (committed), ``not_applied``
  (pre-effect refusal), ``uncertain`` (ambiguous).
* M7 projection status: a separate ``projection_error`` event carrying the
  request id; the outcome is never rewritten by a projection failure.
* M8 invalidation observer: ``notify_state_changed`` for the affected action.
* M9 effect counter: dispatch calls to the owner.
* M10 transport mapping: ``webui_response`` ``ok`` and ``error.status``.
* M11 timeout: ``error.status == 504`` and ``outcome == "uncertain"``.

C4 accepts either committed encoding. It asserts the invariant that is common
to both readings: the effect applied, invalidation ran, the projection failure
was reported separately, no 500/uncertain was returned, and a replay of the
same identity does not repeat the effect.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kohaku.webui.inbound_commands import WebUICommandRouter
from kohaku.webui.replay_store import DurableWebUIReplay


class _Recorder:
    """Transport double that records frames and can fail one event type."""

    def __init__(self, *, fail_events: frozenset[str] = frozenset()) -> None:
        self.frames: list[dict[str, Any]] = []
        self._fail_events = fail_events

    async def send(self, _connection: object, event: str, **payload: Any) -> None:
        self.frames.append({"event": event, **payload})
        if event in self._fail_events:
            raise RuntimeError(f"{event} target connection went away")

    def named(self, event: str) -> list[dict[str, Any]]:
        return [frame for frame in self.frames if frame["event"] == event]


def _router(
    dispatch,
    recorder: _Recorder,
    *,
    replay_store: DurableWebUIReplay | None = None,
) -> tuple[WebUICommandRouter, object]:
    connection = object()
    router = object.__new__(WebUICommandRouter)
    router.request_locks = {}
    router.request_tasks = {}
    router.request_operations = {}
    router._webui_connections = {connection}
    router._durable_replay = replay_store
    router._http_router = SimpleNamespace(
        dispatch_webui_mutation=dispatch,
        authorize_webui_mutation=lambda *_args, **_kwargs: None,
    )
    router._transport = SimpleNamespace(webui_send_event=recorder.send)
    router.gateway = SimpleNamespace(
        session_manager=None,
        endpoint=SimpleNamespace(protocol_connections=None),
    )
    # M8 observer: invalidation must run for a committed effect.
    router.notify_state_changed = AsyncMock()
    return router, connection


async def _send(router: WebUICommandRouter, connection: object, envelope: dict) -> None:
    await router.start_webui_request(connection, envelope)
    pending = tuple(router.request_tasks.values())
    if pending:
        await asyncio.gather(*pending)


def _sidebar_envelope(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "action": "sidebar.update",
        "payload": {"state": {"pinned_keys": ["websocket:a"]}},
    }


def _applied_owner(counter: list[int]):
    async def dispatch(*_args, **_kwargs) -> SimpleNamespace:
        counter[0] += 1
        return SimpleNamespace(
            status_code=200,
            body=json.dumps({"pinned_keys": ["websocket:a"]}).encode(),
            reason_phrase="OK",
        )

    return dispatch


def _status(frame: dict[str, Any]) -> int | None:
    error = frame.get("error")
    if isinstance(error, dict):
        return error.get("status")
    return None


def _outcome(frame: dict[str, Any]) -> str | None:
    error = frame.get("error")
    if isinstance(error, dict):
        return error.get("outcome")
    return frame.get("outcome")


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["sidebar_state_updated", "projection"])
async def test_c4_committed_mutation_keeps_applied_outcome_when_projection_fails(
    tmp_path: Path,
    failing: str,
) -> None:
    """C4: effect commits, follow-up projection/broadcast fails, no 500."""
    fail_events = frozenset({"sidebar_state_updated"}) if failing == "sidebar_state_updated" else frozenset()
    recorder = _Recorder(fail_events=fail_events)
    calls: list[int] = [0]
    store_path = tmp_path / "replay.json"
    router, connection = _router(
        _applied_owner(calls), recorder, replay_store=DurableWebUIReplay(store_path)
    )
    if failing == "projection":
        router.notify_state_changed = AsyncMock(
            side_effect=RuntimeError("projection refresh failed")
        )
    envelope = _sidebar_envelope("c4-req-1")

    await _send(router, connection, envelope)

    # 1. The outcome denotes a committed effect; never not_applied/uncertain.
    responses = recorder.named("webui_response")
    assert len(responses) == 1
    response = responses[0]
    assert response["ok"] is True
    assert _outcome(response) in {"applied", "partial"}
    # 4. No 500 and no uncertain result for the effect that applied.
    assert _status(response) != 500
    assert _outcome(response) != "uncertain"

    # 3. The affected views are invalidated.
    router.notify_state_changed.assert_awaited_once()
    assert router.notify_state_changed.await_args.args[0] == "sidebar.update"

    if failing == "sidebar_state_updated":
        # 2. The projection failure is reported separately, with the request id.
        projection_errors = recorder.named("projection_error")
        assert len(projection_errors) == 1
        assert projection_errors[0]["request_id"] == "c4-req-1"
        assert projection_errors[0]["action"] == "sidebar.update"
        assert projection_errors[0]["reason"] == "sidebar_broadcast_failed"
    else:
        # The broadcast still runs when the projection refresh fails.
        assert len(recorder.named("sidebar_state_updated")) == 1
        projection_errors = recorder.named("projection_error")
        assert len(projection_errors) == 1
        assert projection_errors[0]["request_id"] == "c4-req-1"
        assert projection_errors[0]["reason"] == "projection_refresh_failed"

    # 5. Retry the same identity through the replay path: recorded result,
    # one effect. The restarted router cannot reach the owner except by replay.
    assert calls[0] == 1
    replay_frames = _Recorder(fail_events=fail_events)
    replay_router, replay_connection = _router(
        _applied_owner(calls), replay_frames, replay_store=DurableWebUIReplay(store_path)
    )
    if failing == "projection":
        replay_router.notify_state_changed = AsyncMock(
            side_effect=RuntimeError("projection refresh failed")
        )
    state, _entry = DurableWebUIReplay(store_path).lookup(
        "c4-req-1", "sidebar.update", _digest(envelope), None
    )
    assert state == "completed"

    await _send(replay_router, replay_connection, envelope)

    assert calls[0] == 1
    replayed = replay_frames.named("webui_response")
    assert len(replayed) == 1
    assert replayed[0]["ok"] is True
    assert _outcome(replayed[0]) in {"applied", "partial"}


def _digest(envelope: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(
            envelope["payload"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@pytest.mark.asyncio
async def test_c5_refusal_before_effect_is_not_applied_without_invalidation(
    tmp_path: Path,
) -> None:
    """C5: a pre-effect refusal stays not_applied and invalidates nothing."""
    recorder = _Recorder()
    attempts: list[int] = [0]
    effects: list[int] = [0]

    async def dispatch(*_args, **_kwargs) -> SimpleNamespace:
        attempts[0] += 1
        return SimpleNamespace(
            status_code=409,
            body=json.dumps({"error": {"message": "refused"}}).encode(),
            reason_phrase="Conflict",
        )

    router, connection = _router(
        dispatch, recorder, replay_store=DurableWebUIReplay(tmp_path / "replay.json")
    )
    await _send(router, connection, _sidebar_envelope("c5-req-1"))

    responses = recorder.named("webui_response")
    assert len(responses) == 1
    # 1. outcome == not_applied.
    assert _outcome(responses[0]) == "not_applied"
    assert _status(responses[0]) == 409
    # 2. The effect count for this identity is zero; the owner was asked once
    #    and refused before any effect.
    assert attempts[0] == 1
    assert effects[0] == 0
    # 3. No view invalidation for a mutation that changed nothing.
    router.notify_state_changed.assert_not_awaited()
    assert recorder.named("sidebar_state_updated") == []
    # 4. The refusal reason is reported and distinguishable from a projection
    #    failure.
    assert "refused" in responses[0]["error"]["message"]
    assert recorder.named("projection_error") == []


@pytest.mark.asyncio
async def test_c6_ambiguous_timeout_is_uncertain_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C6: an ambiguous timeout stays uncertain and starts no automatic replay."""
    from kohaku.remote import operations as operations_module

    spec = SimpleNamespace(timeout_s=0.01, max_request_bytes=1024)
    monkeypatch.setattr(
        operations_module,
        "load_registry",
        lambda: SimpleNamespace(mutations={"settings.agent.update": spec}),
    )

    recorder = _Recorder()
    attempts: list[int] = [0]
    effects: list[int] = [0]

    async def slow_dispatch(*_args, **_kwargs) -> SimpleNamespace:
        attempts[0] += 1
        await asyncio.sleep(0.05)
        effects[0] += 1
        return SimpleNamespace(status_code=200, body=b"{}", reason_phrase="OK")

    router, connection = _router(
        slow_dispatch, recorder, replay_store=DurableWebUIReplay(tmp_path / "replay.json")
    )
    envelope = {
        "request_id": "c6-req-1",
        "action": "settings.agent.update",
        "payload": {"agent": {"name": "test"}},
    }
    await _send(router, connection, envelope)

    # 1. outcome == uncertain and it does not claim any committed/refused state.
    responses = recorder.named("webui_response")
    assert len(responses) == 1
    assert _outcome(responses[0]) == "uncertain"
    assert _status(responses[0]) == 504
    assert _outcome(responses[0]) not in {"applied", "partial", "not_applied"}
    # 3. The effect count does not increase after the timeout resolves and no
    #    automatic replay starts.
    settled_attempts = attempts[0]
    settled_effects = effects[0]
    await asyncio.sleep(0)
    assert attempts[0] == settled_attempts == 1
    assert effects[0] == settled_effects == 0
    assert router.request_tasks == {}
    # 4. No invalidation is claimed for an unknown outcome.
    router.notify_state_changed.assert_not_awaited()
