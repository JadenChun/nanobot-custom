from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest
from websockets.http11 import Response

from nanobot.remote import edge as edge_module
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
from tests.remote.proof_helpers import encode_proof_base64url, new_device_key, sign_claims

ORIGIN = "https://fedora.taila1e707.ts.net"
ORIGIN_HOST = "fedora.taila1e707.ts.net"


def token_headers(token: str) -> Headers:
    return Headers([("Authorization", f"Bearer {token}")])


def request(path: str, token: str | None = None, extra: list[tuple[str, str]] | None = None) -> WsRequest:
    headers = Headers(extra or [])
    if headers.get("Host") is None:
        headers["Host"] = ORIGIN_HOST
    if headers.get("Origin") is None:
        headers["Origin"] = ORIGIN
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
    pair = owner.tokens.issue_device(
        "install-0001",
        ("chat.read", "chat.write", "media.read", "attachment.write", "device.self.logout"),
        device_fingerprint="a" * 64,
        device_generation=1,
        origin=ORIGIN,
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
        delegated_children_handler=lambda identity, key, query: _delegated_children(identity, key, query),
        api_handler=lambda identity, path: _api(identity, path),
        feature_handler=lambda identity, path: _feature(identity, path),
    )
    try:
        yield owner, edge, pair
    finally:
        owner.stop()




def _proof_headers(owner: RemoteAccessOwner, pairing_id: str, key, public_der: bytes) -> list[tuple[str, str]]:
    from nanobot.remote.device_proof import DeviceProofClaims

    payload = owner.pairing.proof_challenge_claims(pairing_id)
    assert payload is not None
    claims = DeviceProofClaims.from_payload(payload)
    return [
        ("X-Nanobot-Pairing-Id", pairing_id),
        (
            "X-Nanobot-Device-Signature",
            encode_proof_base64url(sign_claims(key, claims)),
        ),
        (
            "X-Nanobot-Proof-Claims",
            encode_proof_base64url(
                json.dumps(
                    claims.payload(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
        ),
    ]

async def _read(identity: AccessIdentity, query: dict[str, list[str]]) -> dict[str, Any]:
    return {"items": [identity.installation_id], "cursor": query.get("cursor", [None])[0]}


async def _delegated_children(
    identity: AccessIdentity,
    key: str,
    query: dict[str, list[str]],
) -> tuple[int, dict[str, Any]]:
    return 200, {
        "key": key,
        "installation_id": identity.installation_id,
        "limit": query.get("limit", [None])[0],
        "before": query.get("before", [None])[0],
    }


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
    assert payload["limits"]["max_attachment_bytes"] == 24 * 1024 * 1024

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

    delegated = await edge.handle(
        None,
        request(
            f"/mobile/v1/sessions/{key}/delegated-children?limit=20&before=cursor",
            pair.access_token,
        ),
    )
    assert delegated is not None and delegated.status_code == 200
    assert body(delegated) == {
        "key": "websocket:chat-1",
        "installation_id": "install-0001",
        "limit": "20",
        "before": "cursor",
    }

    invalid_query = await edge.handle(
        None,
        request(
            f"/mobile/v1/sessions/{key}/delegated-children?path=%2Fetc%2Fpasswd",
            pair.access_token,
        ),
    )
    assert invalid_query is not None and invalid_query.status_code == 400

    missing = await edge.handle(
        None,
        request(f"/mobile/v1/sessions/{quote('missing', safe='')}/thread", pair.access_token),
    )
    assert missing is not None and missing.status_code == 404

    write_only = owner.tokens.issue_device(
        "install-0003", ("chat.write",), device_fingerprint="b" * 64,
        device_generation=1, origin=ORIGIN,
    )
    for path in ("/mobile/v1/model-settings", "/mobile/v1/workspaces", f"/mobile/v1/sessions/{key}/thread"):
        denied = await edge.handle(None, request(path, write_only.access_token))
        assert denied is not None and denied.status_code == 403
    denied_children = await edge.handle(
        None,
        request(f"/mobile/v1/sessions/{key}/delegated-children", write_only.access_token),
    )
    assert denied_children is not None and denied_children.status_code == 403

    wrong_origin = await edge.handle(
        None,
        request(
            f"/mobile/v1/sessions/{key}/delegated-children",
            pair.access_token,
            [("Host", ORIGIN_HOST), ("Origin", "https://other.example.test")],
        ),
    )
    assert wrong_origin is not None and wrong_origin.status_code == 403


@pytest.mark.asyncio
async def test_delegated_children_revalidates_access_before_delivery(edge_setup) -> None:
    owner, edge, pair = edge_setup

    async def revoke_during_read(identity, key, query):
        owner.tokens.revoke(identity.installation_id)
        return 200, {"private": "must not be delivered", "key": key, "query": query}

    guarded = MobileGatewayEdge(
        edge.validator,
        generation_provider=lambda: int(owner.store.snapshot()["revision"]),
        delegated_children_handler=revoke_during_read,
    )
    key = quote("websocket:chat-1", safe="")
    response = await guarded.handle(
        None,
        request(
            f"/mobile/v1/sessions/{key}/delegated-children?limit=5",
            pair.access_token,
        ),
    )
    assert response is not None and response.status_code == 401
    assert body(response)["error"] == "invalid_access"
    assert "private" not in response.body.decode()


@pytest.mark.asyncio
async def test_delegated_children_uses_its_registered_response_bound(edge_setup) -> None:
    _owner, edge, pair = edge_setup
    payload = {"children": "x" * 1_600_000}
    assert edge._fits(payload, "sessions.thread")
    assert not edge._fits(payload, "sessions.delegated_children")

    async def oversized_children(identity, key, query):
        return 200, payload

    edge.delegated_children_handler = oversized_children
    key = quote("websocket:chat-1", safe="")
    response = await edge.handle(
        None,
        request(f"/mobile/v1/sessions/{key}/delegated-children?limit=100", pair.access_token),
    )
    assert response is not None and response.status_code == 413
    assert body(response)["error"] == "response_too_large"


@pytest.mark.asyncio
async def test_chat_read_cannot_send_stop_or_other_mutation_frames(edge_setup) -> None:
    _owner, edge, _pair = edge_setup
    identity = AccessIdentity("install-readonly", ("chat.read",), 1e12, 1, ORIGIN)
    stop = await edge.handle_ws_frame(identity, 7, {"type": "stop"})
    assert stop == {"ok": False, "error": "unknown_frame"}

    from nanobot.remote.operations import mobile_frame_allowed

    assert not mobile_frame_allowed(
        identity.scopes,
        {"type": "webui_request", "action": "session.delete"},
    )
    assert not mobile_frame_allowed(identity.scopes, {"type": "message", "content": "stop"})


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

    assets = quote("/api/sessions/websocket%3Achat-1/generated-assets?limit=50", safe="")
    allowed_assets = await edge.handle(
        None,
        request(f"/mobile/v1/api?path={assets}", pair.access_token),
    )
    assert allowed_assets is not None and allowed_assets.status_code == 200
    assert "generated-assets" in body(allowed_assets)["path"]

    library = quote("/api/generated-assets?limit=50&cursor=1", safe="")
    allowed_library = await edge.handle(
        None,
        request(f"/mobile/v1/api?path={library}", pair.access_token),
    )
    assert allowed_library is not None and allowed_library.status_code == 200
    assert body(allowed_library)["path"] == "/api/generated-assets?limit=50&cursor=1"

    for raw in (
        "/api/settings",
        "/api/webui/skills",
        "/api/sessions/websocket%3Achat-1/file-preview?path=notes&unexpected=1",
        "/api/sessions/websocket%3Achat-1/generated-assets?unexpected=1",
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
    limited = owner.tokens.issue_device(
        "install-0002", ("chat.read",), device_fingerprint="c" * 64,
        device_generation=1, origin=ORIGIN,
    )
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
    large = b"x" * (3 * 1024 * 1024)
    large_attachment = await edge.handle_ws_frame(
        identity,
        7,
        {"type": "attachment", "data_base64": base64.b64encode(large).decode()},
    )
    assert large_attachment["ok"] is True
    assert large_attachment["result"]["bytes"] == len(large)
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

    second = owner.tokens.issue_device(
        "install-0003", ("chat.read", "device.self.logout"),
        device_fingerprint="d" * 64, device_generation=1, origin=ORIGIN,
    )
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
        origin=ORIGIN,
        label="phone",
        chat_consent=True,
        media_consent=True,
    )
    key, public_der = new_device_key()
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
            [
                *pairing_headers,
                ("X-Nanobot-Installation", "install-pair-1"),
                ("X-Nanobot-Label", "phone"),
                ("X-Nanobot-Device-Key", encode_proof_base64url(public_der)),
            ],
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
    prove = await edge.handle(
        None,
        request(
            "/mobile/v1/pair/prove",
            None,
            _proof_headers(owner, offer.pairing_id, key, public_der),
        ),
    )
    assert prove is not None and prove.status_code == 200
    approved = await edge.handle(None, request("/mobile/v1/pair/poll", None, pairing_headers))
    assert approved is not None and approved.status_code == 200
    result = body(approved)
    assert result["status"] == "approved"
    assert result["installation_id"] == "install-pair-1"
    assert result["access_token"].startswith("nbm_at_")
    assert result["refresh_token"] == ""

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
async def test_edge_manual_pairing_code_collects_without_nonce(edge_setup) -> None:
    owner, edge, _pair = edge_setup
    offer = owner.pairing.create_challenge(
        origin=ORIGIN,
        label="phone",
        chat_consent=True,
        media_consent=True,
        full_control_consent=True,
        owner_only=True,
    )

    wrong_code = f"{(int(offer.pairing_code) + 1) % 100_000_000:08d}"
    wrong = await edge.handle(
        None,
        request(
            "/mobile/v1/pair/code",
            None,
            [
                ("X-Nanobot-Origin", offer.origin),
                ("X-Nanobot-Pairing-Code", wrong_code),
                ("X-Nanobot-Installation", "install-code-1"),
            ],
        ),
    )
    assert wrong is not None and wrong.status_code == 403
    assert body(wrong)["error"] == "pairing_code_invalid"

    key, public_der = new_device_key()
    collected = await edge.handle(
        None,
        request(
            "/mobile/v1/pair/code",
            None,
            [
                ("Host", "fedora.taila1e707.ts.net"),
                ("X-Nanobot-Pairing-Code", offer.pairing_code),
                ("X-Nanobot-Installation", "install-code-1"),
                ("X-Nanobot-Label", "Pixel 10"),
                ("X-Nanobot-Platform", "Android 16"),
                ("X-Nanobot-Model", "Pixel 10"),
                ("X-Nanobot-Device-Key", encode_proof_base64url(public_der)),
            ],
        ),
    )
    assert collected is not None and collected.status_code == 200
    pending = body(collected)
    assert pending["state"] == "pending_approval"
    assert pending["device"]["platform"] == "Android 16"
    assert pending["transport"] == "tailscale"
    assert pending["nonce"] == offer.nonce
    assert pending["collector_secret"] == offer.collector_secret
    assert pending["challenge"] == f"{offer.pairing_id}.{offer.nonce}.{offer.collector_secret}"

    owner.pairing.approve(
        pairing_id=offer.pairing_id,
        pairing_code=offer.pairing_code,
    )
    prove = await edge.handle(
        None,
        request(
            "/mobile/v1/pair/prove",
            None,
            _proof_headers(owner, offer.pairing_id, key, public_der),
        ),
    )
    assert prove is not None and prove.status_code == 200
    approved = await edge.handle(
        None,
        request(
            "/mobile/v1/pair/poll",
            None,
            [
                ("X-Nanobot-Pairing-Id", offer.pairing_id),
                ("X-Nanobot-Pairing-Nonce", offer.nonce),
                ("X-Nanobot-Pairing-Secret", offer.collector_secret),
            ],
        ),
    )
    assert approved is not None and approved.status_code == 200
    assert body(approved)["status"] == "approved"
    assert body(approved)["refresh_expires_in"] == 0


@pytest.mark.asyncio
async def test_edge_refresh_rejects_legacy_bearer(edge_setup) -> None:
    owner, edge, _pair = edge_setup
    legacy = owner.tokens.issue("install-legacy", ("chat.read", "chat.write"))
    owner.store.revoke_legacy_bearer_devices()
    headers = [
        ("X-Nanobot-Installation", "install-legacy"),
        ("X-Nanobot-Refresh-Token", legacy.refresh_token),
    ]
    response = await edge.handle(None, request("/mobile/v1/refresh", None, headers))
    assert response is not None and response.status_code == 426
    assert body(response)["error"] == "upgrade_required"
    missing = await edge.handle(None, request("/mobile/v1/refresh", None))
    assert missing is not None and missing.status_code == 400


@pytest.mark.asyncio
async def test_owner_feature_reads_use_registry_and_deny_legacy(edge_setup) -> None:
    owner, edge, pair = edge_setup
    owner_pair = owner.tokens.issue_device(
        "install-owner", ("chat.read", "chat.write", "owner"),
        device_fingerprint="e" * 64, device_generation=1, origin=ORIGIN,
    )

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
    owner.tokens.issue_device(
        "install-owner", ("chat.read", "chat.write", "owner"),
        device_fingerprint="f" * 64, device_generation=1, origin=ORIGIN,
    )
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

    async def validate_uncached(self, _token: str) -> AccessIdentity:
        self.calls += 1
        if isinstance(self.identity, Exception):
            raise self.identity
        return self.identity


@pytest.mark.asyncio
async def test_established_mobile_socket_revalidates_expiry_and_identity() -> None:
    import time

    current = AccessIdentity(
        "install-owner", ("chat.read", "owner"), time.time() + 60, 1, ORIGIN,
    )
    validator = _RevalidatingValidator(current)
    edge = MobileGatewayEdge(validator, generation_provider=lambda: 1)
    refreshed = await edge.revalidate_mobile_access(current, "token")
    assert refreshed.installation_id == "install-owner"
    assert validator.calls == 1

    expired = AccessIdentity(
        "install-owner", ("chat.read", "owner"), time.time() - 1, 1, ORIGIN,
    )
    with pytest.raises(EdgeAuthError) as exc:
        await edge.revalidate_mobile_access(expired, "token")
    assert exc.value.code == "access_expired"

    other = AccessIdentity(
        "install-other", ("chat.read", "owner"), time.time() + 60, 1, ORIGIN,
    )
    validator.identity = other
    with pytest.raises(EdgeAuthError) as exc:
        await edge.revalidate_mobile_access(current, "token")
    assert exc.value.code == "installation_mismatch"


@pytest.mark.asyncio
async def test_revoked_owner_socket_revalidation_denies(edge_setup) -> None:
    owner, edge, _pair = edge_setup
    owner_pair = owner.tokens.issue_device(
        "install-owner", ("chat.read", "chat.write", "owner"),
        device_fingerprint="1" * 64, device_generation=1, origin=ORIGIN,
    )
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


@pytest.mark.asyncio
async def test_edge_api_timeout_and_response_bound(edge_setup, monkeypatch) -> None:
    _owner, edge, pair = edge_setup
    encoded = quote("/api/commands", safe="")

    async def slow_api(identity: AccessIdentity, path: str):
        await asyncio.sleep(0.05)
        return 200, {"path": path}

    slow_edge = MobileGatewayEdge(
        edge.validator,
        generation_provider=lambda: 1,
        socket_epoch=7,
        api_handler=slow_api,
    )
    monkeypatch.setattr(edge_module, "MOBILE_API_TIMEOUT_S", 0.01)
    response = await slow_edge.handle(
        None, request(f"/mobile/v1/api?path={encoded}", pair.access_token)
    )
    assert response is not None and response.status_code == 504

    async def large_api(identity: AccessIdentity, path: str):
        return 200, {"blob": "x" * 100}

    large_edge = MobileGatewayEdge(
        edge.validator,
        generation_provider=lambda: 1,
        socket_epoch=7,
        api_handler=large_api,
    )
    monkeypatch.setattr(edge_module, "MOBILE_API_MAX_RESPONSE_BYTES", 16)
    response = await large_edge.handle(
        None, request(f"/mobile/v1/api?path={encoded}", pair.access_token)
    )
    assert response is not None and response.status_code == 413


@pytest.mark.asyncio
async def test_edge_mutation_effect_timeout(edge_setup, monkeypatch) -> None:
    owner, edge, _pair = edge_setup
    spec = SimpleNamespace(
        enabled=True,
        timeout_s=0.01,
        max_request_bytes=1024,
        max_response_bytes=1024,
        scope="owner",
        kind="mutation",
        replay="submit",
    )
    monkeypatch.setattr(
        edge_module,
        "load_registry",
        lambda: SimpleNamespace(mutations={"sidebar.update": spec}),
    )

    async def slow_mutation(identity: AccessIdentity, item: MutationEnvelope):
        await asyncio.sleep(0.05)
        return {"action": item.action}

    timed_edge = MobileGatewayEdge(
        edge.validator,
        generation_provider=lambda: int(owner.store.snapshot()["revision"]),
        socket_epoch=7,
        replay_cache=OperationReplayCache(),
        mutation_handler=slow_mutation,
    )
    identity = AccessIdentity("install-0001", ("chat.read", "chat.write"), 1e12, 1)
    digest = stable_operation_digest("install-0001", "sidebar.update", {"state": {}})
    envelope = MutationEnvelope.model_validate(
        {
            "request_id": "req-timeout-1",
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
    response = await timed_edge.handle_ws_frame(
        identity, 7, {"type": "mutation", "envelope": envelope.model_dump(mode="json")}
    )
    assert response["ok"] is False
    assert response["error"] == "timeout"


@pytest.mark.asyncio
async def test_revocation_before_private_delivery_denies_warm_cached_request(edge_setup) -> None:
    owner, _edge, pair = edge_setup
    entered = asyncio.Event()
    release = asyncio.Event()
    blocking = False
    marker = "PRIVATE-MARKER-DO-NOT-DELIVER"

    async def private_read(identity: AccessIdentity, query: dict[str, list[str]]) -> dict[str, Any]:
        if blocking:
            entered.set()
            await release.wait()
        return {"secret": marker, "viewer": identity.installation_id}

    validator = OwnerAccessValidator(
        RemoteAccessClient(owner.socket_path, timeout=2.0),
        cache_ttl=30.0,
    )
    edge = MobileGatewayEdge(
        validator,
        generation_provider=lambda: 1,
        read_handler=private_read,
    )
    warm = await edge.handle(None, request("/mobile/v1/read", pair.access_token))
    assert warm.status_code == 200
    assert marker.encode() in warm.body

    blocking = True
    pending = asyncio.create_task(
        edge.handle(None, request("/mobile/v1/read", pair.access_token)),
    )
    await entered.wait()
    owner.tokens.revoke("install-0001")
    release.set()
    response = await pending
    assert response.status_code == 401
    assert marker.encode() not in response.body


@pytest.mark.asyncio
async def test_owner_loss_before_private_delivery_fails_closed(edge_setup) -> None:
    owner, _edge, pair = edge_setup
    entered = asyncio.Event()
    release = asyncio.Event()
    blocking = False
    marker = "PRIVATE-MARKER-DO-NOT-DELIVER"

    async def private_read(identity: AccessIdentity, query: dict[str, list[str]]) -> dict[str, Any]:
        if blocking:
            entered.set()
            await release.wait()
        return {"secret": marker, "viewer": identity.installation_id}

    validator = OwnerAccessValidator(
        RemoteAccessClient(owner.socket_path, timeout=2.0),
        cache_ttl=30.0,
    )
    edge = MobileGatewayEdge(
        validator,
        generation_provider=lambda: 1,
        read_handler=private_read,
    )
    warm = await edge.handle(None, request("/mobile/v1/read", pair.access_token))
    assert warm.status_code == 200

    blocking = True
    pending = asyncio.create_task(
        edge.handle(None, request("/mobile/v1/read", pair.access_token)),
    )
    await entered.wait()
    owner.stop()
    release.set()
    response = await pending
    assert response.status_code == 503
    assert marker.encode() not in response.body


@pytest.mark.asyncio
async def test_standard_dispatch_is_owner_only_and_forwards_the_exact_path(edge_setup) -> None:
    owner, _edge, pair = edge_setup
    owner_pair = owner.tokens.issue_device(
        "install-owner",
        ("chat.read", "chat.write", "owner"),
        device_fingerprint="2" * 64,
        device_generation=1,
        origin=ORIGIN,
    )
    seen: list[tuple[str, str, str]] = []

    async def standard(
        identity: AccessIdentity,
        method: str,
        path: str,
    ) -> tuple[int, dict[str, Any]]:
        seen.append((identity.installation_id, method, path))
        return 200, {"path": path}

    validator = OwnerAccessValidator(
        RemoteAccessClient(owner.socket_path, timeout=2.0),
        cache_ttl=0.0,
    )
    edge = MobileGatewayEdge(
        validator,
        generation_provider=lambda: 1,
        standard_handler=standard,
    )
    denied = await edge.handle(
        None,
        request(
            "/mobile/v1/dispatch?path=" + quote("/api/settings", safe=""),
            pair.access_token,
        ),
    )
    assert denied is not None and denied.status_code == 403
    assert body(denied)["error"] == "owner_required"
    assert seen == []

    allowed = await edge.handle(
        None,
        request(
            "/mobile/v1/dispatch?path=" + quote("/api/settings?section=models", safe=""),
            owner_pair.access_token,
        ),
    )
    assert allowed is not None and allowed.status_code == 200
    assert body(allowed) == {"path": "/api/settings?section=models"}
    assert seen == [("install-owner", "GET", "/api/settings?section=models")]
