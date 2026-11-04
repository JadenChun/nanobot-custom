from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace

import pytest

from nanobot.remote.edge import AccessIdentity
from nanobot.webui.inbound_commands import _mobile_mutation_allowed
from nanobot.webui.ws_http import GatewayHTTPHandler
from tests.webui.mobile_mutation_helper import mobile_mutation_payload


def mobile_connection(identity: AccessIdentity | None):
    return SimpleNamespace(mobile_identity=identity)


def test_legacy_mobile_keeps_only_shared_organization_actions() -> None:
    connection = mobile_connection(
        AccessIdentity("install-legacy", ("chat.read", "chat.write"), 1e12, 1)
    )
    assert _mobile_mutation_allowed(connection, "sidebar.update")
    assert _mobile_mutation_allowed(connection, "recovery.continue")
    assert not _mobile_mutation_allowed(connection, "settings.agent.update")
    assert not _mobile_mutation_allowed(connection, "automation.enable")
    assert not _mobile_mutation_allowed(connection, "workspace.grant.upsert")


def test_owner_mobile_allows_enabled_registry_actions_only() -> None:
    connection = mobile_connection(
        AccessIdentity("install-owner", ("chat.read", "chat.write", "owner"), 1e12, 1)
    )
    assert _mobile_mutation_allowed(connection, "settings.agent.update")
    assert _mobile_mutation_allowed(connection, "settings.provider.create")
    assert _mobile_mutation_allowed(connection, "automation.enable")
    assert _mobile_mutation_allowed(connection, "skill.install")
    assert _mobile_mutation_allowed(connection, "workspace.grant.upsert")
    assert _mobile_mutation_allowed(connection, "workspace.grant.revoke")
    assert not _mobile_mutation_allowed(connection, "settings.remote_access.enable")
    assert not _mobile_mutation_allowed(connection, "unregistered.action")


def test_webui_connection_without_mobile_identity_is_not_mobile_allowed() -> None:
    assert not _mobile_mutation_allowed(SimpleNamespace(), "sidebar.update")


@pytest.mark.asyncio
async def test_mobile_mutation_payload_denies_non_owner_and_local_only() -> None:
    stub = SimpleNamespace()
    legacy = AccessIdentity("install-legacy", ("chat.read", "chat.write"), 1e12, 1)
    status, payload = await mobile_mutation_payload(stub, legacy, "settings.agent.update", {}
    )
    assert status == 403
    assert payload["error"] == "owner_required"

    owner = AccessIdentity("install-owner", ("chat.read", "chat.write", "owner"), 1e12, 1)
    status, payload = await mobile_mutation_payload(stub, owner, "settings.remote_access.enable", {}
    )
    assert status == 403
    assert payload["error"] == "operation_not_allowed"

    status, payload = await mobile_mutation_payload(stub, owner, "unregistered.action", {}
    )
    assert status == 403
    assert payload["error"] == "operation_not_allowed"


@pytest.mark.asyncio
async def test_owner_mutation_payload_enforces_request_bound() -> None:
    stub = SimpleNamespace()
    owner = AccessIdentity("install-owner", ("chat.read", "chat.write", "owner"), 1e12, 1)
    status, payload = await mobile_mutation_payload(stub,
        owner,
        "settings.agent.update",
        {"blob": "x" * (1024 * 1024 + 1)},
    )
    assert status == 413
    assert payload["error"] == "request_too_large"


def test_remote_access_settings_actions_map_to_mutation_paths() -> None:
    assert GatewayHTTPHandler._webui_mutation_path(
        "settings.remote_access.revoke", {}
    ) == "/api/settings/remote-access/revoke"
    assert GatewayHTTPHandler._webui_mutation_path(
        "settings.remote_access.enable", {}
    ) == "/api/settings/remote-access/enable"
    assert GatewayHTTPHandler._webui_mutation_path(
        "settings.remote_access.pairing.start", {}
    ) == "/api/settings/remote-access/pairing/start"
    assert GatewayHTTPHandler._webui_mutation_path(
        "settings.remote_access.pairing.approve", {}
    ) == "/api/settings/remote-access/pairing/approve"


@pytest.mark.asyncio
async def test_owner_standard_mutation_writes_actor_audit() -> None:
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    from nanobot.webui.http_utils import http_json_response

    audits: list[dict[str, object]] = []

    class _Audit:
        def audit_event(self, **kwargs):
            audits.append(kwargs)
            return kwargs

    async def dispatch(_connection, _request, _path):
        return http_json_response({"ok": True})

    stub = SimpleNamespace(
        owner_audit_client=_Audit(),
        authorize_webui_mutation=lambda *_args: None,
        _webui_mutation_path=GatewayHTTPHandler._webui_mutation_path,
        _connection_headers=lambda _connection: Headers(),
        _dispatch_resolved=dispatch,
    )
    owner = AccessIdentity(
        "install-owner",
        ("chat.read", "chat.write", "owner"),
        1e12,
        1,
    )
    connection = SimpleNamespace(mobile_identity=owner)
    response = await GatewayHTTPHandler.dispatch_webui_mutation(
        stub,
        connection,
        "sidebar.update",
        {"state": {}},
    )
    assert isinstance(response, Response)
    assert response.status_code == 200
    assert audits == [
        {
            "installation_id": "install-owner",
            "operation": "sidebar.update",
            "decision": "attempted",
            "actor_installation_id": "install-owner",
        },
        {
            "installation_id": "install-owner",
            "operation": "sidebar.update",
            "decision": "applied",
            "actor_installation_id": "install-owner",
        },
    ]


@pytest.mark.asyncio
async def test_owner_standard_mutation_fails_closed_when_audit_unavailable() -> None:
    from websockets.datastructures import Headers

    from nanobot.webui.http_utils import http_json_response

    class _Audit:
        def audit_event(self, **_kwargs):
            raise RuntimeError("audit unavailable")

    dispatched: list[bool] = []

    async def dispatch(_connection, _request, _path):
        dispatched.append(True)
        return http_json_response({"ok": True})

    stub = SimpleNamespace(
        owner_audit_client=_Audit(),
        authorize_webui_mutation=lambda *_args: None,
        _webui_mutation_path=GatewayHTTPHandler._webui_mutation_path,
        _connection_headers=lambda _connection: Headers(),
        _dispatch_resolved=dispatch,
        _log=SimpleNamespace(exception=lambda *_args, **_kwargs: None),
    )
    owner = AccessIdentity(
        "install-owner",
        ("chat.read", "chat.write", "owner"),
        1e12,
        1,
    )
    connection = SimpleNamespace(mobile_identity=owner)
    response = await GatewayHTTPHandler.dispatch_webui_mutation(
        stub,
        connection,
        "sidebar.update",
        {"state": {}},
    )
    assert response.status_code == 503
    assert dispatched == []


def test_workspace_owner_actions_have_delegation_receipts() -> None:
    from nanobot.remote.permissions import ACTION_OPERATIONS

    for action in (
        "workspace.access.add_folder",
        "workspace.access.allow_once",
        "workspace.access.cancel",
        "workspace.access.continue",
        "workspace.access.resolve",
        "workspace.folder.edit",
        "workspace.grant.revoke",
        "repository.discover",
        "repository.forget",
        "repository.remember",
    ):
        assert action in ACTION_OPERATIONS


@pytest.mark.asyncio
async def test_workspace_owner_action_calls_permission_adapter() -> None:
    from websockets.datastructures import Headers

    from nanobot.webui.http_utils import http_json_response

    calls: list[dict[str, object]] = []

    class _Adapter:
        def authorize(self, **kwargs):
            calls.append(kwargs)
            return True

    async def dispatch(_connection, _request, _path):
        return http_json_response({"ok": True})

    stub = SimpleNamespace(
        owner_permission_adapter=_Adapter(),
        authorize_webui_mutation=lambda *_args: None,
        _webui_mutation_path=GatewayHTTPHandler._webui_mutation_path,
        _connection_headers=lambda _connection: Headers(),
        _dispatch_resolved=dispatch,
        _log=SimpleNamespace(exception=lambda *_args, **_kwargs: None),
    )
    owner = AccessIdentity(
        "install-owner",
        ("chat.read", "chat.write", "owner"),
        1e12,
        1,
    )
    connection = SimpleNamespace(
        mobile_identity=owner,
        mobile_access_token="access-token",
    )
    stub._authorize_owner_permission = MethodType(
        GatewayHTTPHandler._authorize_owner_permission, stub
    )
    response = await GatewayHTTPHandler.dispatch_webui_mutation(
        stub,
        connection,
        "workspace.access.resolve",
        {"chat_id": "chat-a"},
    )
    assert response.status_code == 200
    assert calls
    assert calls[0]["action"] == "workspace.access.resolve"
    assert calls[0]["access_token"] == "access-token"


@pytest.mark.asyncio
async def test_owner_standard_mutation_records_failed_audit() -> None:
    from websockets.datastructures import Headers

    audits: list[dict[str, object]] = []

    class _Audit:
        def audit_event(self, **kwargs):
            audits.append(kwargs)
            return kwargs

    async def dispatch(_connection, _request, _path):
        raise RuntimeError("effect failed")

    stub = SimpleNamespace(
        owner_audit_client=_Audit(),
        authorize_webui_mutation=lambda *_args: None,
        _webui_mutation_path=GatewayHTTPHandler._webui_mutation_path,
        _connection_headers=lambda _connection: Headers(),
        _dispatch_resolved=dispatch,
        _log=SimpleNamespace(exception=lambda *_args, **_kwargs: None, debug=lambda *_args, **_kwargs: None),
    )
    owner = AccessIdentity(
        "install-owner",
        ("chat.read", "chat.write", "owner"),
        1e12,
        1,
    )
    connection = SimpleNamespace(mobile_identity=owner)
    response = await GatewayHTTPHandler.dispatch_webui_mutation(
        stub,
        connection,
        "sidebar.update",
        {"state": {}},
    )
    assert response.status_code == 500
    assert [entry["decision"] for entry in audits] == ["attempted", "failed"]


@pytest.mark.asyncio
async def test_owner_standard_mutation_records_cancelled_audit() -> None:
    from websockets.datastructures import Headers

    audits: list[dict[str, object]] = []

    class _Audit:
        def audit_event(self, **kwargs):
            audits.append(kwargs)
            return kwargs

    async def dispatch(_connection, _request, _path):
        raise asyncio.CancelledError()

    stub = SimpleNamespace(
        owner_audit_client=_Audit(),
        authorize_webui_mutation=lambda *_args: None,
        _webui_mutation_path=GatewayHTTPHandler._webui_mutation_path,
        _connection_headers=lambda _connection: Headers(),
        _dispatch_resolved=dispatch,
        _log=SimpleNamespace(exception=lambda *_args, **_kwargs: None, debug=lambda *_args, **_kwargs: None),
    )
    owner = AccessIdentity(
        "install-owner",
        ("chat.read", "chat.write", "owner"),
        1e12,
        1,
    )
    connection = SimpleNamespace(mobile_identity=owner)
    with pytest.raises(asyncio.CancelledError):
        await GatewayHTTPHandler.dispatch_webui_mutation(
            stub,
            connection,
            "sidebar.update",
            {"state": {}},
        )
    assert [entry["decision"] for entry in audits] == ["attempted", "failed"]
