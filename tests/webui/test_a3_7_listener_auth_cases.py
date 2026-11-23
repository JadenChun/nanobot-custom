"""Author cases U1.1, U1.2 and U2 for A3-7 AC7.2 on the real listener boundary.

Source cases: a3-evidence/a3-reconciliation-2026-11-14/req-a37/CASES.md, authored
by the requirements-only session. The scenarios, inputs and expected outcomes are
preserved. Contract decision: a tokenless handshake on any non-loopback listener is
rejected even when ``websocket_requires_token`` is disabled, and token issuance with
no configured secret returns 403 and produces no token.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic_core import ValidationError
from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest

from kohaku.bus.queue import MessageBus
from kohaku.channels.websocket.runtime import WebSocketConfig
from kohaku.config.loader import save_config
from kohaku.config.schema import Config
from kohaku.webui.gateway_endpoint import WebUIGatewayEndpoint
from kohaku.webui.gateway_services import build_gateway_services
from kohaku.webui.gateway_tokens import GatewayTokenStore
from kohaku.webui.ws_http import GatewayHTTPHandler


class _Connection:
    def __init__(self, peer: tuple[str, int] = ("127.0.0.1", 45678)) -> None:
        self.remote_address = peer
        self.request = SimpleNamespace(headers=Headers([("Host", "127.0.0.1:8765")]))

    def respond(self, status: int, text: str) -> SimpleNamespace:
        return SimpleNamespace(status_code=status, body=text.encode())


def _endpoint(config: WebSocketConfig) -> WebUIGatewayEndpoint:
    http = SimpleNamespace(dispatch=None, mobile_revalidator=None)
    return WebUIGatewayEndpoint(config=config, http=http, tokens=GatewayTokenStore())


def _handshake(
    endpoint: WebUIGatewayEndpoint,
    *,
    peer: tuple[str, int],
    host: str,
    token: str | None = None,
) -> SimpleNamespace | None:
    query = {"token": [token]} if token is not None else {}
    return endpoint.authorize_websocket_handshake(
        _Connection(peer),
        query,
        Headers([("Host", host)]),
    )


def test_u1_1_non_loopback_listener_without_credential_fails_closed() -> None:
    startup_refused = False
    try:
        WebSocketConfig(
            host="192.168.1.10",
            token="",
            token_issue_secret="",
            trusted_proxy_auth=None,
            websocket_requires_token=False,
        )
    except ValidationError:
        startup_refused = True

    # A config that skipped validation must still reject every tokenless handshake.
    unvalidated = WebSocketConfig(host="127.0.0.1", websocket_requires_token=False)
    unvalidated.host = "192.168.1.10"
    response = _handshake(
        _endpoint(unvalidated),
        peer=("192.168.1.50", 4567),
        host="192.168.1.10:8765",
    )
    handshake_rejected = getattr(response, "status_code", None) == 401

    assert startup_refused or handshake_rejected
    # A routable listener with a wildcard bind also refuses a tokenless handshake.
    wildcard = WebSocketConfig(host="0.0.0.0", token_issue_secret="s3cret")
    assert _handshake(
        _endpoint(wildcard),
        peer=("198.51.100.7", 4567),
        host="0.0.0.0:8765",
    ).status_code == 401


def test_u1_1_non_loopback_listener_accepts_only_an_issued_token() -> None:
    config = WebSocketConfig(host="192.168.1.10", token_issue_secret="s3cret")
    endpoint = _endpoint(config)
    token = endpoint._tokens.issue_token(300)
    assert _handshake(
        endpoint,
        peer=("192.168.1.50", 4567),
        host="192.168.1.10:8765",
        token=token,
    ) is None


@pytest.mark.parametrize("secret", ["", None])
def test_u1_2_issuance_without_secret_is_refused(secret: str | None) -> None:
    handler = object.__new__(GatewayHTTPHandler)
    handler.config = WebSocketConfig(
        host="127.0.0.1",
        token_issue_path="/token",
        token_issue_secret=secret or "",
        token="",
    )
    handler.tokens = GatewayTokenStore()
    handler._log = SimpleNamespace(error=lambda *_args, **_kwargs: None)

    response = GatewayHTTPHandler._handle_token_issue(
        handler,
        _Connection(),
        WsRequest("/token", Headers([("Host", "127.0.0.1:8765")])),
    )

    assert response.status_code == 403
    body = json.loads(response.body)
    assert "token" not in body
    assert handler.tokens.issued_tokens == {}


def test_u2_loopback_listener_keeps_local_behavior(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    config = WebSocketConfig(host="127.0.0.1", port=8765, websocket_requires_token=False)
    gateway = build_gateway_services(
        config=config,
        bus=MessageBus(),
        session_manager=None,
        workspace_path=tmp_path / "workspace",
        default_restrict_to_workspace=False,
        config_path=config_path,
        runtime_model_name=lambda: None,
        runtime_surface="browser",
        runtime_capabilities_overrides=None,
    )

    endpoint = gateway.endpoint
    assert _handshake(
        endpoint,
        peer=("127.0.0.1", 4567),
        host="127.0.0.1:8765",
    ) is None

    bootstrap = gateway.http._handle_bootstrap(
        _Connection(),
        WsRequest(
            "/webui/bootstrap",
            Headers([("Host", "127.0.0.1:8765")]),
        ),
    )
    assert bootstrap.status_code == 200
    assert json.loads(bootstrap.body)["ws_path"]
