"""Production owner -> edge -> shared command/HTTP authorization regressions."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.remote.auth import TokenService
from nanobot.remote.edge import MobileGatewayEdge, OwnerAccessValidator
from nanobot.remote.ipc import OwnerIPCServer
from nanobot.remote.store import RemoteAccessStore
from nanobot.webui.gateway_endpoint import WebUIGatewayEndpoint
from nanobot.webui.inbound_commands import WebUICommandRouter, _mobile_mutation_allowed
from nanobot.webui.ws_http import GatewayHTTPHandler

ORIGIN = "https://admitted.example"

class Connection:
    remote_address = ("203.0.113.9", 1234)
    async def close(self, *args):
        pass

@pytest.fixture
def boundary(tmp_path):
    store = RemoteAccessStore(tmp_path / "state")
    store.load_or_create()
    store.set_enabled(True)
    tokens = TokenService(store)
    ipc = OwnerIPCServer(store, tokens=tokens)
    class OwnerTransport:
        def validate_access(self, token):
            response = ipc._dispatch({"method": "validate_access", "role": "gateway", "params": {"access_token": token}})
            assert response["ok"], response
            return response["result"]
    edge = MobileGatewayEdge(OwnerAccessValidator(OwnerTransport()), generation_provider=lambda: 1)
    handler = object.__new__(GatewayHTTPHandler)
    handler.config = WebSocketConfig(host="127.0.0.1")
    endpoint = WebUIGatewayEndpoint(config=handler.config, http=handler, tokens=None, mobile_edge=edge)
    return store, tokens, handler, endpoint

async def admit(boundary, scopes, *, host="admitted.example", origin=ORIGIN):
    store, tokens, handler, endpoint = boundary
    pair = tokens.issue_device("install-boundary", scopes, device_fingerprint="a" * 64, device_generation=1, origin=ORIGIN)
    connection = Connection()
    headers = Headers({"Upgrade": "websocket", "Connection": "Upgrade", "Authorization": "Bearer " + pair.access_token, "Host": host})
    if origin is not None:
        headers["Origin"] = origin
    connection.request = request = Request("/mobile/v1/ws", headers)
    result = await endpoint.process_request(connection, request, is_allowed=lambda _: True)
    return connection, result

@pytest.mark.parametrize("host,origin", [("other.example", ORIGIN), ("admitted.example", "https://wrong.example"), ("admitted.example", "null")])
async def test_rejects_host_origin_mismatch(boundary, host, origin):
    _, result = await admit(boundary, ("chat.read",), host=host, origin=origin)
    assert result is not None and result.status_code == 403

async def test_native_no_origin_requires_bound_host(boundary):
    _, result = await admit(boundary, ("chat.read",), origin=None)
    assert result is None

async def test_media_only_cannot_upgrade(boundary):
    _, result = await admit(boundary, ("media.read",))
    assert result is not None and result.status_code == 403

async def test_current_scope_reduction_blocks_attach_and_all_mutations(boundary):
    store, _, handler, endpoint = boundary
    connection, result = await admit(boundary, ("chat.read", "chat.write", "attachment.write", "owner"))
    assert result is None
    store.update_scopes("install-boundary", ("media.read",))
    frames, hydrated = [], []
    async def send(_connection, event, **payload):
        frames.append((event, payload))
    async def hydrate(chat_id):
        hydrated.append(chat_id)
    transport = SimpleNamespace(revalidate_mobile_connection=endpoint.revalidate_mobile_connection, webui_send_event=send, webui_attach=lambda *_: None, webui_hydrate=hydrate)
    gateway = SimpleNamespace(http=handler, media=None, ingress=None, transcripts=None, workspaces=None, temporary_chats=SimpleNamespace(validate_attach=lambda _: None), session_projection=SimpleNamespace(attach_fields=lambda _: {}), endpoint=endpoint, session_manager=None)
    router = WebUICommandRouter(transport, gateway)
    await router.dispatch(connection, "client", {"type": "attach", "chat_id": "audit-chat"})
    assert not hydrated
    assert frames[-1][1]["detail"] in {"access_expired", "scope_denied"}
    for action in ("sidebar.update", "session.delete", "settings.agent.update", "unknown"):
        assert not _mobile_mutation_allowed(connection, action)
        assert handler.authorize_webui_mutation(connection, action, {}) is not None

@pytest.mark.parametrize("envelope", [{"type": "attachment"}, {"type": "message", "content": "/stop"}, {"type": "new_chat"}, {"type": "webui_request", "action": "sidebar.update"}])
async def test_read_only_cannot_dispatch_effects(boundary, envelope):
    _, _, handler, endpoint = boundary
    connection, result = await admit(boundary, ("chat.read",))
    assert result is None
    frames = []
    async def send(_connection, event, **payload):
        frames.append((event, payload))
    transport = SimpleNamespace(revalidate_mobile_connection=endpoint.revalidate_mobile_connection, webui_send_event=send)
    gateway = SimpleNamespace(http=handler, media=None, ingress=None, transcripts=None, workspaces=None, temporary_chats=None, session_projection=None, endpoint=endpoint, session_manager=None)
    router = WebUICommandRouter(transport, gateway)
    await router.dispatch(connection, "client", envelope)
    assert frames[-1][1]["detail"] == "scope_denied"
    assert handler.authorize_webui_mutation(connection, "sidebar.update", {}) is not None

async def test_owner_still_requires_action_scope_and_registry(boundary):
    connection, result = await admit(boundary, ("chat.read", "owner"))
    assert result is None
    assert not _mobile_mutation_allowed(connection, "sidebar.update")
    assert not _mobile_mutation_allowed(connection, "unknown")
    assert _mobile_mutation_allowed(connection, "settings.agent.update")


@pytest.mark.parametrize("write", [False, True])
async def test_typed_stop_keeps_message_write_scope_and_live_revocation(boundary, write):
    from tests.agent.test_stop_expectation import target_value

    store, _, handler, endpoint = boundary
    connection, result = await admit(boundary, ("chat.read", "chat.write") if write else ("chat.read",))
    assert result is None
    frames = []

    async def send(_connection, event, **payload):
        frames.append((event, payload))

    transport = SimpleNamespace(
        revalidate_mobile_connection=endpoint.revalidate_mobile_connection, webui_send_event=send,
    )
    gateway = SimpleNamespace(
        http=handler, media=None, ingress=None, transcripts=None, workspaces=None,
        temporary_chats=None, session_projection=None, endpoint=endpoint, session_manager=None,
    )
    router = WebUICommandRouter(transport, gateway)
    dispatch = AsyncMock()
    router._dispatch_message = dispatch
    request = {
        "type": "message", "chat_id": "exact-stop", "content": "/stop",
        "stop_expectation": target_value(),
    }
    await router.dispatch(connection, "client", request)
    if not write:
        dispatch.assert_not_awaited()
        assert frames[-1][1]["detail"] == "scope_denied"
        return
    dispatch.assert_awaited_once_with(connection, "client", request)
    store.update_scopes("install-boundary", ("chat.read",))
    await router.dispatch(connection, "client", request)
    assert dispatch.await_count == 1
    assert frames[-1][1]["detail"] in {"scope_denied", "access_expired"}
