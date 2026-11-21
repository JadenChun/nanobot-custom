"""Requirements-only cases C1-C5 for remote-access settings mutations (A3 P5).

Source cases: a3-evidence/a3-reconciliation-2026-11-14/req-p5/CASES.md,
authored by the REQUIREMENTS_ONLY_TEST_AUTHOR session. The cases are exercised
through the real WebSocket settings-mutation boundary: mutation ledger,
authorization, route classification, settings dispatch, and the owner-backed
renewal domain.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest

from nanobot.config.loader import save_config
from nanobot.config.schema import Config
from nanobot.remote.management import RemoteAccessManager
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.webui.http_utils import http_error, http_json_response, parse_query
from nanobot.webui.inbound_commands import WebUICommandRouter
from nanobot.webui.replay_store import DurableWebUIReplay
from nanobot.webui.settings_routes import WebUISettingsRouter
from nanobot.webui.settings_services import WebUISettingsServices
from nanobot.webui.ws_http import GatewayHTTPHandler

_ENABLE_ACTION = "settings.remote_access.enable"
_RENEW_ACTION = "settings.remote_access.pairing.renew"


class _Connection:
    def __init__(self) -> None:
        self.remote_address = ("127.0.0.1", 45678)
        self.request = SimpleNamespace(
            headers=Headers([("Host", "127.0.0.1:8765")])
        )


class _SettingsRouteHTTPHandler(GatewayHTTPHandler):
    """Minimal GatewayHTTPHandler that resolves only settings routes."""

    async def _dispatch_resolved(
        self,
        connection: Any,
        request: WsRequest,
        got: str,
    ) -> Any | None:
        return await self.settings_routes.dispatch(connection, request, got)


@dataclass
class _Harness:
    router: WebUICommandRouter
    connection: _Connection
    frames: list[dict[str, Any]]
    replay: DurableWebUIReplay
    settings: WebUISettingsServices
    owner: RemoteAccessOwner

    async def mutate(
        self,
        action: str,
        payload: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        before = len(self.frames)
        await self.router.start_webui_request(
            self.connection,
            {"request_id": request_id, "action": action, "payload": payload},
        )
        pending = tuple(self.router.request_tasks.values())
        if pending:
            await asyncio.gather(*pending)
        assert len(self.frames) == before + 1
        return self.frames[before]


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[_Harness]:
    owner = RemoteAccessOwner(
        tmp_path / "remote-access",
        serve_verifier=lambda _origin: True,
    )
    owner.start()
    config_path = tmp_path / "config.json"
    save_config(
        Config.model_validate(
            {
                "remoteAccess": {
                    "enabled": True,
                    "stateDir": str(owner.store.root),
                    "socketPath": str(owner.socket_path),
                }
            }
        ),
        config_path,
    )
    settings = WebUISettingsServices.create(config_path)
    settings_routes = WebUISettingsRouter(
        settings=settings,
        bus=SimpleNamespace(),
        logger=SimpleNamespace(exception=lambda *_args: None),
        check_api_token=lambda _request: True,
        parse_query=parse_query,
        json_response=http_json_response,
        error_response=http_error,
        runtime_surface="browser",
        runtime_capabilities={},
    )
    handler = object.__new__(_SettingsRouteHTTPHandler)
    handler.config = SimpleNamespace(host="127.0.0.1", token_issue_path=None)
    handler.settings_routes = settings_routes
    handler.owner_audit_client = None
    handler.mobile_revalidator = None
    handler._log = SimpleNamespace(exception=lambda *_args: None)
    connection = _Connection()
    frames: list[dict[str, Any]] = []

    async def send(target: object, event: str, **payload: Any) -> None:
        frames.append({"event": event, **payload})

    replay = DurableWebUIReplay(tmp_path / "webui" / "mutation-replay.json")
    router = WebUICommandRouter(
        SimpleNamespace(
            webui_send_event=send,
            revalidate_mobile_connection=lambda _connection: None,
        ),
        SimpleNamespace(
            http=handler,
            endpoint=SimpleNamespace(webui_connections={connection}),
            media=None,
            ingress=None,
            transcripts=None,
            workspaces=None,
            temporary_chats=None,
            session_projection=None,
            session_manager=None,
        ),
        replay_store=replay,
    )
    try:
        yield _Harness(
            router=router,
            connection=connection,
            frames=frames,
            replay=replay,
            settings=settings,
            owner=owner,
        )
    finally:
        owner.stop()


def _set_enabled(harness: _Harness, enabled: bool) -> None:
    harness.settings.config.update(
        lambda config: setattr(config.remote_access, "enabled", enabled)
    )
    harness.owner.store.set_enabled(enabled)


def _read_enabled(harness: _Harness) -> bool:
    return harness.settings.config.load().remote_access.enabled


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@pytest.mark.asyncio
async def test_c1_real_boolean_accepted(harness: _Harness) -> None:
    _set_enabled(harness, True)
    assert _read_enabled(harness) is True

    frame = await harness.mutate(_ENABLE_ACTION, {"enabled": False}, "req-c1")

    assert frame["event"] == "webui_response"
    assert frame["ok"] is True
    assert isinstance(frame.get("result"), dict)
    assert _read_enabled(harness) is False


@pytest.mark.asyncio
async def test_c2_string_boolean_rejected(harness: _Harness) -> None:
    _set_enabled(harness, True)
    recorded = _read_enabled(harness)
    assert recorded is True

    frame = await harness.mutate(_ENABLE_ACTION, {"enabled": "false"}, "req-c2")

    assert frame["event"] == "webui_response"
    assert frame["ok"] is False
    error = frame["error"]
    assert isinstance(error, dict)
    assert 400 <= error["status"] < 500
    assert _read_enabled(harness) is recorded
    assert harness.owner.store.snapshot()["enabled"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_id", "body"),
    [
        ("req-c3a", {"enabled": 1}),
        ("req-c3b", {"enabled": None}),
        ("req-c3c", {}),
    ],
)
async def test_c3_non_boolean_and_missing_values_rejected(
    harness: _Harness,
    request_id: str,
    body: dict[str, Any],
) -> None:
    _set_enabled(harness, True)
    assert _read_enabled(harness) is True

    frame = await harness.mutate(_ENABLE_ACTION, body, request_id)

    assert frame["event"] == "webui_response"
    assert frame["ok"] is False
    error = frame["error"]
    assert isinstance(error, dict)
    assert 400 <= error["status"] < 500
    assert _read_enabled(harness) is True


@pytest.mark.asyncio
async def test_c4_pairing_renew_routed_to_domain(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = RemoteAccessManager.renew_pairing

    def counting_renew(self: RemoteAccessManager, pairing_id: str) -> dict[str, Any]:
        calls.append(pairing_id)
        return original(self, pairing_id)

    monkeypatch.setattr(RemoteAccessManager, "renew_pairing", counting_renew)
    _set_enabled(harness, True)
    offer = harness.owner.pairing.create_challenge(
        origin="https://fixture.example:8443",
        label="phone",
        chat_consent=True,
        media_consent=False,
    )

    payload = {"pairingId": offer.pairing_id}
    frame = await harness.mutate(_RENEW_ACTION, payload, "req-c4")

    assert frame["event"] == "webui_response"
    assert frame["ok"] is True
    result = frame["result"]
    assert isinstance(result, dict)
    assert result["pairing_id"] == offer.pairing_id
    assert calls == [offer.pairing_id]
    assert "req-c4" in harness.router.request_operations
    state, _entry = harness.replay.lookup(
        "req-c4",
        _RENEW_ACTION,
        _payload_digest(payload),
        None,
    )
    assert state == "completed"


@pytest.mark.asyncio
async def test_c5_lost_reply_replays_recorded_outcome(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = RemoteAccessManager.renew_pairing

    def counting_renew(self: RemoteAccessManager, pairing_id: str) -> dict[str, Any]:
        calls.append(pairing_id)
        return original(self, pairing_id)

    monkeypatch.setattr(RemoteAccessManager, "renew_pairing", counting_renew)
    _set_enabled(harness, True)
    offer = harness.owner.pairing.create_challenge(
        origin="https://fixture.example:8443",
        label="phone",
        chat_consent=True,
        media_consent=False,
    )
    payload = {"pairingId": offer.pairing_id}
    request_id = "req-c5"

    # The first reply is lost after the domain ran, so it is not observed.
    await harness.mutate(_RENEW_ACTION, payload, request_id)
    assert calls == [offer.pairing_id]
    recorded = harness.frames[-1]

    first_state, _entry = harness.replay.lookup(
        request_id,
        _RENEW_ACTION,
        _payload_digest(payload),
        None,
    )
    assert first_state == "completed"

    replay_frame = await harness.mutate(_RENEW_ACTION, payload, request_id)

    assert replay_frame["event"] == "webui_response"
    assert replay_frame["ok"] is True
    assert replay_frame["result"] == recorded["result"]
    assert calls == [offer.pairing_id]
    assert list(harness.router.request_operations) == [request_id]
    replay_state, _entry = harness.replay.lookup(
        request_id,
        _RENEW_ACTION,
        _payload_digest(payload),
        None,
    )
    assert replay_state == "completed"
