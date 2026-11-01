from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest
from websockets.http11 import Response

from nanobot.remote.client import RemoteAccessClient
from nanobot.remote.edge import (
    AccessIdentity,
    EdgeAuthError,
    MediaResult,
    MobileGatewayEdge,
    OwnerAccessValidator,
)
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.remote.protocol import MutationEnvelope, MutationMode, stable_operation_digest
from nanobot.remote.replay import OperationReplayCache


def token_headers(token: str) -> Headers:
    return Headers([("Authorization", f"Bearer {token}")])


def request(path: str, token: str | None = None, extra: list[tuple[str, str]] | None = None) -> WsRequest:
    headers = Headers(extra or [])
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return WsRequest(path, headers)


def body(response: Response) -> dict[str, Any]:
    value = json.loads(response.body)
    assert isinstance(value, dict)
    return value


@pytest.fixture
def edge_setup(tmp_path: Path):
    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    pair = owner.tokens.issue(
        "install-0001",
        ("chat.read", "chat.write", "media.read", "attachment.write", "device.self.logout"),
    )
    validator = OwnerAccessValidator(
        RemoteAccessClient(owner.socket_path, timeout=2.0),
        cache_ttl=0.0,
    )
    edge = MobileGatewayEdge(
        validator,
        generation_provider=lambda: int(owner.store.snapshot()["revision"]),
        socket_epoch=7,
        read_handler=lambda identity, query: _read(identity, query),
        media_handler=lambda identity, sig, payload, range_header: _media(identity, sig, payload, range_header),
        attachment_handler=lambda identity, data: _attachment(identity, data),
        logout_handler=lambda identity: _logout(owner, identity),
        model_settings_handler=lambda identity: _model_settings(identity),
        workspaces_handler=lambda identity: _workspaces(identity),
        thread_handler=lambda identity, key, query: _thread(identity, key, query),
        api_handler=lambda identity, path: _api(identity, path),
        feature_handler=lambda identity, path: _feature(identity, path),
    )
    try:
        yield owner, edge, pair
    finally:
        owner.stop()


async def _read(identity: AccessIdentity, query: dict[str, list[str]]) -> dict[str, Any]:
    return {"items": [identity.installation_id], "cursor": query.get("cursor", [None])[0]}


async def _media(identity: AccessIdentity, sig: str, payload: str, range_header: str | None) -> MediaResult:
    assert identity.installation_id == "install-0001"
    assert sig == "sig" and payload == "payload"
    return MediaResult(
        status=206,
        body=b"PNG!",
        media_type="image/png",
        headers={"Content-Range": "bytes 0-3/4", "X-Range": range_header or "none"},
    )


async def _attachment(identity: AccessIdentity, data: bytes) -> dict[str, Any]:
    return {"installation_id": identity.installation_id, "bytes": len(data)}


async def _logout(owner: RemoteAccessOwner, identity: AccessIdentity) -> dict[str, Any]:
    owner.tokens.revoke(identity.installation_id)
    return {"installation_id": identity.installation_id, "erased": True}


async def _model_settings(identity: AccessIdentity) -> dict[str, Any]:
    return {"agent": {"model_preset": "default"}, "providers": [], "viewer": identity.installation_id}


async def _workspaces(identity: AccessIdentity) -> dict[str, Any]:
    return {"controls": {"permission_v2": True}, "viewer": identity.installation_id}


async def _thread(
    identity: AccessIdentity,
    key: str,
    query: dict[str, list[str]],
) -> tuple[int, dict[str, Any]]:
    if key == "missing":
        return 404, {"error": "thread_not_found"}
    return 200, {"key": key, "limit": query.get("limit", [None])[0], "viewer": identity.installation_id}


async def _api(identity: AccessIdentity, path: str) -> tuple[int, dict[str, Any]]:
    return 200, {"path": path, "viewer": identity.installation_id}


async def _feature(identity: AccessIdentity, path: str) -> tuple[int, dict[str, Any]]:
    if path == "/api/settings":
        return 200, {"owner": True, "viewer": identity.installation_id}
    return 404, {"error": "feature_not_found"}


@pytest.mark.asyncio
async def test_edge_handshake_read_media_scopes_and_unknown_routes(edge_setup) -> None:
    _owner, edge, pair = edge_setup
    response = await edge.handle(None, request("/mobile/v1/handshake", pair.access_token))
    assert response is not None
    assert response.status_code == 200
    payload = body(response)
    assert payload["mobile_protocol"] == 1
    assert payload["bridge_version"] == 1
    assert payload["principal"] == "mobile:install-0001"
    assert payload["socket_epoch"] == 7
    assert payload["capabilities"]["chat"] is True
    assert payload["limits"]["max_attachment_bytes"] == 256 * 1024

    read = await edge.handle(None, request("/mobile/v1/read?cursor=next", pair.access_token))
    assert read is not None and read.status_code == 200
    assert body(read)["cursor"] == "next"
    assert body(read)["items"] == ["install-0001"]

    media = await edge.handle(
        None,
        request("/mobile/v1/media/sig/payload", pair.access_token, [("Range", "bytes=0-3")]),
    )
    assert media is not None and media.status_code == 206
    assert media.body == b"PNG!"
    assert media.headers["Content-Range"] == "bytes 0-3/4"
    assert media.headers["X-Range"] == "bytes=0-3"

    assert (await edge.handle(None, request("/mobile/v1/read"))).status_code == 401
    assert (await edge.handle(None, request("/mobile/v1/read", "wrong"))).status_code == 401
    assert (await edge.handle(None, request("/mobile/v1/nope", pair.access_token))).status_code == 404
    assert (await edge.handle(None, request("/api/bootstrap", pair.access_token))) is None


@pytest.mark.asyncio
async def test_edge_mobile_ui_reads_and_thread(edge_setup) -> None:
    owner, edge, pair = edge_setup
    model = await edge.handle(None, request("/mobile/v1/model-settings", pair.access_token))
    assert model is not None and model.status_code == 200
    assert body(model)["agent"]["model_preset"] == "default"

    workspaces = await edge.handle(None, request("/mobile/v1/workspaces", pair.access_token))
    assert workspaces is not None and workspaces.status_code == 200
    assert body(workspaces)["controls"]["permission_v2"] is True

    key = quote("websocket:chat-1", safe="")
    thread = await edge.handle(
        None,
        request(f"/mobile/v1/sessions/{key}/thread?limit=2", pair.access_token),
    )
    assert thread is not None and thread.status_code == 200
    assert body(thread)["key"] == "websocket:chat-1"
    assert body(thread)["limit"] == "2"

    missing = await edge.handle(
        None,
        request(f"/mobile/v1/sessions/{quote('missing', safe='')}/thread", pair.access_token),
    )
    assert missing is not None and missing.status_code == 404

    write_only = owner.tokens.issue("install-0003", ("chat.write",))
    for path in ("/mobile/v1/model-settings", "/mobile/v1/workspaces", f"/mobile/v1/sessions/{key}/thread"):
        denied = await edge.handle(None, request(path, write_only.access_token))
        assert denied is not None and denied.status_code == 403


@pytest.mark.asyncio
async def test_edge_allowlisted_chat_api_proxy(edge_setup) -> None:
    _owner, edge, pair = edge_setup
    encoded = quote("/api/commands", safe="")
    allowed = await edge.handle(None, request(f"/mobile/v1/api?path={encoded}", pair.access_token))
    assert allowed is not None and allowed.status_code == 200
    assert body(allowed)["path"] == "/api/commands"

    preview = quote(
        "/api/sessions/websocket%3Achat-1/file-preview?path=notes%2Fready.md&probe=1",
        safe="",
    )
    allowed_preview = await edge.handle(
        None,
        request(f"/mobile/v1/api?path={preview}", pair.access_token),
    )
    assert allowed_preview is not None and allowed_preview.status_code == 200
    assert "file-preview" in body(allowed_preview)["path"]

    for raw in (
        "/api/settings",
        "/api/webui/skills",
        "/api/sessions/websocket%3Achat-1/file-preview?path=notes&unexpected=1",
        "https://evil.test/api/commands",
    ):
        denied = await edge.handle(
            None,
            request(f"/mobile/v1/api?path={quote(raw, safe='')}", pair.access_token),
        )
        assert denied is not None and denied.status_code == 400


@pytest.mark.asyncio
async def test_edge_scope_denial_and_ws_frame_epoch(edge_setup) -> None:
    owner, edge, _pair = edge_setup
    limited = owner.tokens.issue("install-0002", ("chat.read",))
    response = await edge.handle(None, request("/mobile/v1/handshake", limited.access_token))
    assert response is not None and response.status_code == 200
    media = await edge.handle(None, request("/mobile/v1/media/sig/payload", limited.access_token))
    assert media is not None and media.status_code == 403

    identity = AccessIdentity("install-0001", ("chat.read", "attachment.write"), 0, 2)
    with pytest.raises(EdgeAuthError) as mismatch:
        edge.authorize_ws_frame(identity, 8, "chat.read")
    assert mismatch.value.status == 409
    with pytest.raises(EdgeAuthError) as denied:
        edge.authorize_ws_frame(identity, 7, "media.read")
    assert denied.value.status == 403

    attachment = await edge.handle_ws_frame(
        identity,
        7,
        {"type": "attachment", "data_base64": base64.b64encode(b"hello").decode()},
    )
    assert attachment["ok"] is True
    assert attachment["result"]["bytes"] == 5
    unknown = await edge.handle_ws_frame(identity, 7, {"type": "unknown"})
    assert unknown["ok"] is False


@pytest.mark.asyncio
async def test_edge_revocation_owner_loss_and_logout(edge_setup) -> None:
    owner, edge, pair = edge_setup
    assert (await edge.handle(None, request("/mobile/v1/handshake", pair.access_token))).status_code == 200
    owner.tokens.revoke("install-0001")
    await edge.validator.invalidate("install-0001")
    revoked = await edge.handle(None, request("/mobile/v1/handshake", pair.access_token))
    assert revoked is not None and revoked.status_code == 401

    second = owner.tokens.issue("install-0003", ("chat.read", "device.self.logout"))
    logout = await edge.handle(None, request("/mobile/v1/logout", second.access_token))
    assert logout is not None and logout.status_code == 404
    frame = await edge.handle_ws_frame(
        AccessIdentity("install-0003", ("device.self.logout",), 0, 3),
        7,
        {"type": "logout"},
    )
    assert frame["ok"] is True
    after = await edge.handle(None, request("/mobile/v1/handshake", second.access_token))
    assert after is not None and after.status_code == 401

    owner.stop()
    unavailable = await edge.handle(None, request("/mobile/v1/handshake", pair.access_token))
    assert unavailable is not None and unavailable.status_code == 503


@pytest.mark.asyncio
async def test_edge_mutation_frame_uses_replay_cache(edge_setup) -> None:
    owner, original, _pair = edge_setup
    cache = OperationReplayCache()
    effects = 0

    async def mutate(_identity, _envelope):
        nonlocal effects
        effects += 1
        return {"effect": effects}

    edge = MobileGatewayEdge(
        original.validator,
        generation_provider=lambda: int(owner.store.snapshot()["revision"]),
        socket_epoch=7,
        replay_cache=cache,
        mutation_handler=mutate,
    )
    identity = AccessIdentity(
        "install-0001", ("chat.read", "chat.write", "media.read", "attachment.write", "device.self.logout"), 0, 2
    )
    digest = stable_operation_digest("device-0001", "sidebar.update", {"state": {}})
    envelope = MutationEnvelope.model_validate(
        {
            "request_id": "req-edge-1",
            "installation_id": "install-0001",
            "principal": "device-0001",
            "action": "sidebar.update",
            "digest": digest,
            "gateway_generation": int(owner.store.snapshot()["revision"]),
            "epoch": 7,
            "mode": "submit",
            "payload": {"state": {}},
        }
    )
    first = await edge.handle_ws_frame(
        identity, 7, {"type": "mutation", "envelope": envelope.model_dump(mode="json")}
    )
    assert first["ok"] is True
    assert first["result"]["decision"] == "applied"
    replay_envelope = envelope.model_copy(update={"mode": MutationMode.REPLAY_ONLY})
    second = await edge.handle_ws_frame(
        identity, 7, {"type": "mutation", "envelope": replay_envelope.model_dump(mode="json")}
    )
    assert second["ok"] is True
    assert second["result"]["decision"] == "replayed"
    assert effects == 1


@pytest.mark.asyncio
async def test_mobile_websocket_authorization_binds_identity(edge_setup) -> None:
    _owner, edge, pair = edge_setup
    connection = type("Connection", (), {})()
    response = await edge.authorize_websocket(connection, request("/mobile/v1/ws", pair.access_token))
    assert response is None
    assert connection.mobile_identity.installation_id == "install-0001"
    denied = await edge.authorize_websocket(connection, request("/mobile/v1/ws"))
    assert denied is not None and denied.status_code == 401


@pytest.mark.asyncio
async def test_edge_pairing_collect_approve_poll(edge_setup) -> None:
    owner, edge, _pair = edge_setup
    offer = owner.pairing.create_challenge(
        origin="https://fedora.taila1e707.ts.net",
        label="phone",
        chat_consent=True,
        media_consent=True,
    )
    pairing_headers = [
        ("X-Nanobot-Pairing-Id", offer.pairing_id),
        ("X-Nanobot-Pairing-Nonce", offer.nonce),
        ("X-Nanobot-Pairing-Secret", offer.collector_secret),
    ]
    collect = await edge.handle(
        None,
        request(
            "/mobile/v1/pair/collect",
            None,
            [*pairing_headers, ("X-Nanobot-Installation", "install-pair-1"), ("X-Nanobot-Label", "phone")],
        ),
    )
    assert collect is not None and collect.status_code == 200
    pending = body(collect)
    assert pending["pairing_id"] == offer.pairing_id
    assert pending["comparison_code"] == offer.comparison_code

    waiting = await edge.handle(None, request("/mobile/v1/pair/poll", None, pairing_headers))
    assert waiting is not None and waiting.status_code == 200
    assert body(waiting)["status"] == "pending"

    owner.pairing.approve(pairing_id=offer.pairing_id, comparison_code=offer.comparison_code)
    approved = await edge.handle(None, request("/mobile/v1/pair/poll", None, pairing_headers))
    assert approved is not None and approved.status_code == 200
    result = body(approved)
    assert result["status"] == "approved"
    assert result["installation_id"] == "install-pair-1"
    assert result["access_token"].startswith("nbm_at_")
    assert result["refresh_token"].startswith("nbm_rt_")

    consumed = await edge.handle(None, request("/mobile/v1/pair/poll", None, pairing_headers))
    assert consumed is not None and consumed.status_code == 410
    assert body(consumed)["error"] == "pairing_consumed"

    missing = await edge.handle(None, request("/mobile/v1/pair/collect", None))
    assert missing is not None and missing.status_code == 400
    wrong = await edge.handle(
        None,
        request(
            "/mobile/v1/pair/collect",
            None,
            [
                ("X-Nanobot-Pairing-Id", offer.pairing_id),
                ("X-Nanobot-Pairing-Nonce", offer.nonce),
                ("X-Nanobot-Pairing-Secret", "wrong-secret"),
                ("X-Nanobot-Installation", "install-pair-1"),
            ],
        ),
    )
    assert wrong is not None and wrong.status_code == 403


@pytest.mark.asyncio
async def test_edge_refresh_rotates_access_without_bearer(edge_setup) -> None:
    owner, edge, pair = edge_setup
    headers = [
        ("X-Nanobot-Installation", "install-0001"),
        ("X-Nanobot-Refresh-Token", pair.refresh_token),
    ]
    response = await edge.handle(None, request("/mobile/v1/refresh", None, headers))
    assert response is not None and response.status_code == 200
    payload = json.loads(response.body)
    assert payload["access_token"].startswith("nbm_at_")
    assert payload["refresh_token"].startswith("nbm_rt_")
    assert payload["expires_in"] > 0
    replay = await edge.handle(None, request("/mobile/v1/refresh", None, headers))
    assert replay is not None and replay.status_code == 401
    assert body(replay)["error"] == "invalid_refresh"
    missing = await edge.handle(None, request("/mobile/v1/refresh", None))
    assert missing is not None and missing.status_code == 400

    owner.tokens.revoke("install-0001")
    revoked = await edge.handle(
        None,
        request(
            "/mobile/v1/refresh",
            None,
            [
                ("X-Nanobot-Installation", "install-0001"),
                ("X-Nanobot-Refresh-Token", payload["refresh_token"]),
            ],
        ),
    )
    assert revoked is not None and revoked.status_code == 401
    assert body(revoked)["error"] == "device_revoked"


@pytest.mark.asyncio
async def test_owner_feature_reads_use_registry_and_deny_legacy(edge_setup) -> None:
    owner, edge, pair = edge_setup
    owner_pair = owner.tokens.issue("install-owner", ("chat.read", "chat.write", "owner"))

    owner_response = await edge.handle(
        None,
        request("/mobile/v1/feature?path=%2Fapi%2Fsettings", owner_pair.access_token),
    )
    assert owner_response is not None and owner_response.status_code == 200
    assert body(owner_response)["owner"] is True

    legacy = await edge.handle(
        None,
        request("/mobile/v1/feature?path=%2Fapi%2Fsettings", pair.access_token),
    )
    assert legacy is not None and legacy.status_code == 403
    assert body(legacy)["error"] == "owner_required"

    for path in (
        "%2Fapi%2F..%2Fsecret",
        "https%3A%2F%2Fevil.example%2Fapi%2Fsessions",
        "%2Fmobile%2Fv1%2Fread",
        "%2Fapi%2Funknown",
        "%2Fapi%2Fworkspace-directories%2Fmissing",
    ):
        denied = await edge.handle(
            None,
            request(f"/mobile/v1/feature?path={path}", owner_pair.access_token),
        )
        assert denied is not None and denied.status_code == 400


@pytest.mark.asyncio
async def test_owner_ws_mutation_registry_and_legacy_ceiling(edge_setup) -> None:
    owner, _edge, pair = edge_setup
    owner.tokens.issue("install-owner", ("chat.read", "chat.write", "owner"))
    effects: list[str] = []

    async def mutation(identity: AccessIdentity, envelope: MutationEnvelope) -> dict[str, Any]:
        effects.append(envelope.action)
        return {"ok": True}

    edge = MobileGatewayEdge(
        _edge.validator,
        generation_provider=lambda: int(owner.store.snapshot()["revision"]),
        socket_epoch=7,
        replay_cache=OperationReplayCache(),
        mutation_handler=mutation,
    )

    def envelope(installation_id: str, action: str, request_id: str) -> dict[str, Any]:
        payload = {"request_id": request_id, "installation_id": installation_id}
        return {
            "request_id": request_id,
            "installation_id": installation_id,
            "principal": f"mobile:{installation_id}",
            "action": action,
            "digest": stable_operation_digest(f"mobile:{installation_id}", action, payload),
            "gateway_generation": int(owner.store.snapshot()["revision"]),
            "epoch": 7,
            "mode": "submit",
            "payload": payload,
        }

    owner_frame = await edge.handle_ws_frame(
        AccessIdentity("install-owner", ("chat.read", "chat.write", "owner"), 1e12, 1),
        7,
        {"type": "mutation", "envelope": envelope("install-owner", "sidebar.update", "req-owner")},
    )
    assert owner_frame["ok"] is True
    assert effects == ["sidebar.update"]

    legacy_settings = await edge.handle_ws_frame(
        AccessIdentity("install-0001", ("chat.read", "chat.write"), 1e12, 1),
        7,
        {"type": "mutation", "envelope": envelope("install-0001", "settings.agent.update", "req-legacy")},
    )
    assert legacy_settings["ok"] is False
    assert legacy_settings["error"] == "owner_required"

    workspace_action = await edge.handle_ws_frame(
        AccessIdentity("install-owner", ("chat.read", "chat.write", "owner"), 1e12, 1),
        7,
        {"type": "mutation", "envelope": envelope("install-owner", "workspace.grant.revoke", "req-workspace")},
    )
    assert workspace_action["ok"] is True
    assert effects == ["sidebar.update", "workspace.grant.revoke"]


class _RevalidatingValidator:
    def __init__(self, identity: AccessIdentity | Exception) -> None:
        self.identity = identity
        self.calls = 0

    async def validate(self, _token: str) -> AccessIdentity:
        self.calls += 1
        if isinstance(self.identity, Exception):
            raise self.identity
        return self.identity


@pytest.mark.asyncio
async def test_established_mobile_socket_revalidates_expiry_and_identity() -> None:
    import time

    current = AccessIdentity("install-owner", ("chat.read", "owner"), time.time() + 60, 1)
    validator = _RevalidatingValidator(current)
    edge = MobileGatewayEdge(validator, generation_provider=lambda: 1)
    refreshed = await edge.revalidate_mobile_access(current, "token")
    assert refreshed.installation_id == "install-owner"
    assert validator.calls == 1

    expired = AccessIdentity("install-owner", ("chat.read", "owner"), time.time() - 1, 1)
    with pytest.raises(EdgeAuthError) as exc:
        await edge.revalidate_mobile_access(expired, "token")
    assert exc.value.code == "access_expired"

    other = AccessIdentity("install-other", ("chat.read", "owner"), time.time() + 60, 1)
    validator.identity = other
    with pytest.raises(EdgeAuthError) as exc:
        await edge.revalidate_mobile_access(current, "token")
    assert exc.value.code == "installation_mismatch"


@pytest.mark.asyncio
async def test_revoked_owner_socket_revalidation_denies(edge_setup) -> None:
    owner, edge, _pair = edge_setup
    owner_pair = owner.tokens.issue("install-owner", ("chat.read", "chat.write", "owner"))
    from nanobot.remote.edge import OwnerAccessValidator

    validator = OwnerAccessValidator(
        RemoteAccessClient(owner.socket_path, timeout=2.0),
        cache_ttl=0.0,
    )
    connected = MobileGatewayEdge(validator, generation_provider=lambda: 1)
    connection = type("Connection", (), {})()
    assert await connected.authorize_websocket(
        connection, request("/mobile/v1/ws", owner_pair.access_token)
    ) is None
    assert connection.mobile_access_token == owner_pair.access_token
    identity = connection.mobile_identity

    owner.tokens.revoke("install-owner")
    with pytest.raises(EdgeAuthError):
        await connected.revalidate_mobile_access(identity, connection.mobile_access_token)
