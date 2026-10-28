"""Exact local WebUI recovery action routing and payload validation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.webui.ws_http import WEBSOCKET_MUTATION_ROUTES, GatewayHTTPHandler


@pytest.mark.parametrize(("action", "path"), [
    ("workspace.access.resolve", "/api/workspaces/access/resolve"),
    ("workspace.access.continue", "/api/workspaces/access/continue"),
    ("workspace.access.cancel", "/api/workspaces/access/cancel"),
])
def test_recovery_actions_have_frozen_local_mutation_routes(action: str, path: str) -> None:
    assert WEBSOCKET_MUTATION_ROUTES[action] == path


@pytest.mark.parametrize("path", [
    "/api/workspaces/access/resolve",
    "/api/workspaces/access/continue",
    "/api/workspaces/access/cancel",
])
@pytest.mark.asyncio
async def test_recovery_routes_reject_direct_http_writes(path: str) -> None:
    handler = object.__new__(GatewayHTTPHandler)
    handler.config = SimpleNamespace(trusted_proxy_auth=None)
    handler.settings_routes = SimpleNamespace(is_mutation_path=lambda _path: False)
    handler._log = SimpleNamespace()
    handler._log_slow_http = lambda *_args: None
    connection = SimpleNamespace(remote_address=("127.0.0.1", 1234))
    response = await handler.dispatch(
        connection,
        Request(path, Headers([("Host", "127.0.0.1:8765")])),
    )
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_recovery_action_requires_local_browser_connection() -> None:
    result = {
        "workspace_access_request": {"status": "ready"},
        "workspace_scope": {"access_mode": "restricted"},
    }

    class RecoveryOwner:
        calls = 0

        def resolve(self, _payload):
            self.calls += 1
            return result

        async def publish(self, *_args):
            return None

    owner = RecoveryOwner()
    handler = object.__new__(GatewayHTTPHandler)
    handler.config = SimpleNamespace(host="127.0.0.1")
    handler.workspace_access_recovery = owner
    handler.check_api_token = lambda _request: True
    handler._log = SimpleNamespace(exception=lambda *_args: None)
    request = Request(
        "/api/workspaces/access/resolve",
        Headers([("Host", "127.0.0.1:8765")]),
    )
    setattr(request, "_nanobot_webui_mutation_request", True)
    setattr(request, "_nanobot_webui_mutation_payload", {"chat_id": "chat-a"})

    response = await handler._handle_workspace_access_action(
        SimpleNamespace(remote_address=("203.0.113.1", 1234)),
        request,
        "/api/workspaces/access/resolve",
    )
    assert response.status_code == 403
    assert response.body == b"local_ui_required"
    assert owner.calls == 0


@pytest.mark.asyncio
async def test_recovery_action_commits_before_publishing_and_returns_exact_result() -> None:
    events: list[tuple[str, dict]] = []
    result = {
        "workspace_access_request": {"status": "ready"},
        "workspace_scope": {"access_mode": "restricted"},
    }

    class RecoveryOwner:
        def resolve(self, _payload):
            events.append(("commit", {}))
            return result

        async def publish(self, chat_id, published):
            events.append(("publish", {"chat_id": chat_id, "result": published}))

    handler = object.__new__(GatewayHTTPHandler)
    handler.config = SimpleNamespace(host="127.0.0.1")
    handler.workspace_access_recovery = RecoveryOwner()
    handler.check_api_token = lambda _request: True
    handler._log = SimpleNamespace(exception=lambda *_args: None)
    request = Request(
        "/api/workspaces/access/resolve",
        Headers([("Host", "127.0.0.1:8765")]),
    )
    setattr(request, "_nanobot_webui_mutation_request", True)
    setattr(
        request,
        "_nanobot_webui_mutation_payload",
        {
            "chat_id": "chat-a",
            "access_request_id": "0" * 32,
            "expected_revision": 1,
            "path": "/tmp/project",
            "expected_grants_revision": 0,
        },
    )

    response = await handler._handle_workspace_access_action(
        SimpleNamespace(
            remote_address=("127.0.0.1", 1234),
            request=SimpleNamespace(headers={"Host": "127.0.0.1:8765"}),
        ),
        request,
        "/api/workspaces/access/resolve",
    )
    assert response.status_code == 200
    assert json.loads(response.body) == result
    assert [kind for kind, _payload in events] == ["commit", "publish"]
