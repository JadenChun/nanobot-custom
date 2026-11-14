"""Author cases U4, U5 and U6 for A3-7 AC7.2: desktop notification admission.

Source cases: a3-evidence/a3-reconciliation-2026-11-14/req-a37/CASES.md (U4-U6),
authored by the requirements-only session.

Contract decision: the listener front end accepts GET handshakes only, so the
notification side effect is an authenticated WebSocket mutation
(``desktop.notify``) rather than an HTTP POST. A direct HTTP request is refused
with 405, an unauthenticated handshake with 401, and a foreign Origin with 403.
No refusal path builds or spawns the notification command.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest

from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.config.loader import save_config
from nanobot.config.schema import Config
from nanobot.webui.gateway_services import build_gateway_services


class _Connection:
    def __init__(self, headers: list[tuple[str, str]] | None = None) -> None:
        self.remote_address = ("127.0.0.1", 45678)
        base = [("Host", "127.0.0.1:8765")]
        self.request = SimpleNamespace(headers=Headers(base + (headers or [])))

    def respond(self, status: int, text: str) -> SimpleNamespace:
        return SimpleNamespace(status_code=status, body=text.encode())


class _NotifySpy:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **_kwargs: Any) -> SimpleNamespace:
        self.commands.append(list(command))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    spy = _NotifySpy()
    monkeypatch.setattr("nanobot.webui.ws_http.shutil.which", lambda _name: "/usr/bin/notify-send")
    monkeypatch.setattr("nanobot.webui.ws_http.subprocess.run", spy)
    services = build_gateway_services(
        config=WebSocketConfig(host="127.0.0.1", port=8765),
        bus=MessageBus(),
        session_manager=None,
        workspace_path=tmp_path / "workspace",
        default_restrict_to_workspace=False,
        config_path=config_path,
        runtime_model_name=lambda: None,
        runtime_surface="browser",
        runtime_capabilities_overrides=None,
    )
    return services, spy


def test_u4_notification_without_authentication_fails_closed(gateway) -> None:
    services, spy = gateway

    # A direct request is not an authenticated mutation.
    direct = asyncio.run(
        services.http._handle_desktop_notify(
            _Connection([("Origin", "http://127.0.0.1:8765")]),
            WsRequest(
                "/api/desktop/notify?title=t&body=hello",
                Headers([
                    ("Host", "127.0.0.1:8765"),
                    ("Origin", "http://127.0.0.1:8765"),
                ]),
            ),
        )
    )
    assert direct.status_code in {401, 403, 405}
    assert spy.commands == []

    # The same request through the HTTP router is refused before the handler.
    routed = asyncio.run(
        services.http.dispatch(
            _Connection(),
            WsRequest(
                "/api/desktop/notify?title=t&body=hello",
                Headers([
                    ("Host", "127.0.0.1:8765"),
                    ("Origin", "http://127.0.0.1:8765"),
                ]),
            ),
        )
    )
    assert routed.status_code == 405
    assert spy.commands == []

    # No unauthenticated socket can carry the mutation.
    handshake = services.endpoint.authorize_websocket_handshake(
        _Connection(),
        {},
        Headers([("Host", "127.0.0.1:8765")]),
    )
    assert handshake.status_code == 401
    assert spy.commands == []


def test_u5_cross_origin_notification_fails_closed(gateway) -> None:
    services, spy = gateway
    response = asyncio.run(
        services.http.dispatch_webui_mutation(
            _Connection([("Origin", "https://evil.example")]),
            "desktop.notify",
            {"title": "t", "body": "cross origin"},
        )
    )
    assert 400 <= response.status_code < 500
    assert spy.commands == []


def test_u6_same_origin_authenticated_notification_notifies_once(gateway) -> None:
    services, spy = gateway
    response = asyncio.run(
        services.http.dispatch_webui_mutation(
            _Connection([("Origin", "http://127.0.0.1:8765")]),
            "desktop.notify",
            {"title": "Task finished", "body": "one notification", "tag": "turn-1"},
        )
    )
    assert response.status_code == 200
    assert json.loads(response.body) == {"ok": True}
    assert len(spy.commands) == 1
    assert "one notification" in spy.commands[0]
    assert "Task finished" in spy.commands[0]
