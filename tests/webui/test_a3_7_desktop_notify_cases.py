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
from pathlib import Path
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
        return SimpleNamespace(returncode=0, stdout=b"42", stderr=b"")


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


def test_u6_same_origin_notification_passes_bundled_icon_as_one_argv_value(
    gateway, tmp_path: Path
) -> None:
    services, spy = gateway
    static_dist_path = tmp_path / "web ui dist with spaces"
    icon_path = static_dist_path / "brand" / "nanobot_icon_192.png"
    icon_path.parent.mkdir(parents=True)
    icon_path.write_bytes(b"bundled icon")
    services.http.static_dist_path = static_dist_path

    response = asyncio.run(
        services.http.dispatch_webui_mutation(
            _Connection([("Origin", "http://127.0.0.1:8765")]),
            "desktop.notify",
            {"title": "Task finished", "body": "one notification"},
        )
    )

    assert response.status_code == 200
    assert len(spy.commands) == 1
    command = spy.commands[0]
    assert command[command.index("--icon") + 1] == str(icon_path)
    assert command[-2:] == ["Task finished", "one notification"]
    assert command.count(str(icon_path)) == 1


def test_u6_notification_without_bundled_icon_still_notifies(gateway, tmp_path: Path) -> None:
    services, spy = gateway
    services.http.static_dist_path = tmp_path / "missing web ui dist"

    response = asyncio.run(
        services.http.dispatch_webui_mutation(
            _Connection([("Origin", "http://127.0.0.1:8765")]),
            "desktop.notify",
            {"title": "Task finished", "body": "one notification"},
        )
    )

    assert response.status_code == 200
    assert len(spy.commands) == 1
    assert "--icon" not in spy.commands[0]


def test_u6_remote_notification_fails_closed_without_side_effects(gateway) -> None:
    services, spy = gateway
    connection = _Connection([("Origin", "http://127.0.0.1:8765")])
    connection.remote_address = ("192.0.2.1", 45678)

    response = asyncio.run(
        services.http.dispatch_webui_mutation(
            connection,
            "desktop.notify",
            {"title": "Task finished", "body": "one notification"},
        )
    )

    assert response.status_code == 403
    assert spy.commands == []


def test_u6_missing_notify_send_keeps_existing_unavailable_response(gateway, monkeypatch) -> None:
    services, spy = gateway
    monkeypatch.setattr("nanobot.webui.ws_http.shutil.which", lambda _name: None)

    response = asyncio.run(
        services.http.dispatch_webui_mutation(
            _Connection([("Origin", "http://127.0.0.1:8765")]),
            "desktop.notify",
            {"title": "Task finished", "body": "one notification"},
        )
    )

    assert response.status_code == 503
    assert spy.commands == []


def test_u6_notify_send_failure_keeps_existing_error_response(gateway, monkeypatch) -> None:
    services, _spy = gateway
    calls: list[list[str]] = []

    def fail(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(list(command))
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"failed")

    monkeypatch.setattr("nanobot.webui.ws_http.subprocess.run", fail)
    response = asyncio.run(
        services.http.dispatch_webui_mutation(
            _Connection([("Origin", "http://127.0.0.1:8765")]),
            "desktop.notify",
            {"title": "Task finished", "body": "one notification"},
        )
    )

    assert response.status_code == 500
    assert len(calls) == 1

def test_thread_notifications_replace_by_tag_and_honor_sound(gateway) -> None:
    services, spy = gateway
    for sound in [True, False]:
        response = asyncio.run(services.http.dispatch_webui_mutation(
            _Connection([("Origin", "http://127.0.0.1:8765")]),
            "desktop.notify", {"title": "Kohaku", "body": "Done", "tag": "chat-17", "sound": sound},
        ))
        assert response.status_code == 200
    first, second = spy.commands
    assert "--hint=string:sound-name:complete" in first
    assert "--hint=boolean:suppress-sound:true" in second
    assert "--hint=string:x-canonical-private-synchronous:chat-17" in second
    assert "--replace-id=42" in second


def test_legacy_notify_send_delivers_tagged_notification(gateway, monkeypatch) -> None:
    services, spy = gateway

    def legacy_notify(command: list[str], **kwargs: Any) -> SimpleNamespace:
        if "--print-id" in command or any(arg.startswith("--replace-id=") for arg in command):
            return SimpleNamespace(returncode=1, stderr=b"Unknown option", stdout=b"")
        return spy(command, **kwargs)

    monkeypatch.setattr("nanobot.webui.ws_http.subprocess.run", legacy_notify)
    response = asyncio.run(services.http.dispatch_webui_mutation(
        _Connection([("Origin", "http://127.0.0.1:8765")]),
        "desktop.notify", {"title": "Kohaku", "body": "Done", "tag": "chat-17", "sound": False},
    ))
    assert response.status_code == 200
    delivered, = spy.commands
    assert delivered[-2:] == ["Kohaku", "Done"]
    assert "--hint=boolean:suppress-sound:true" in delivered
    assert "--hint=string:x-canonical-private-synchronous:chat-17" in delivered


def test_darwin_fallback_preserves_notification_text(gateway, monkeypatch) -> None:
    services, spy = gateway
    monkeypatch.setattr("nanobot.webui.ws_http.sys.platform", "darwin")
    monkeypatch.setattr("nanobot.webui.ws_http.shutil.which", lambda name: "/usr/bin/osascript" if name == "osascript" else None)
    body = 'Done — "release"; $(echo test)\nReady'
    response = asyncio.run(services.http.dispatch_webui_mutation(
        _Connection([("Origin", "http://127.0.0.1:8765")]),
        "desktop.notify", {"title": "Release review", "body": body},
    ))
    assert response.status_code == 200
    command, = spy.commands
    assert command[0] == "/usr/bin/osascript"
    assert command[-2:] == ["Release review", body]
    assert body not in command[2]
