"""Integration tests for the WebSocket channel using WsTestClient.

Complements the unit/lightweight tests in test_websocket_channel.py by covering
multi-client scenarios, edge cases, and realistic usage patterns.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import websockets

from nanobot.bus.events import OutboundMessage
from nanobot.bus.outbound_events import ProgressEvent
from nanobot.channels.websocket.runtime import WebSocketChannel, WebSocketConfig
from nanobot.config.schema import RemoteAccessConfig
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.webui.gateway_services import build_gateway_services

from .ws_test_client import WsTestClient, issue_token, issue_token_ok


def _ch(
    bus: Any,
    port: int,
    *,
    remote_access: RemoteAccessConfig | None = None,
    **kw: Any,
) -> WebSocketChannel:
    cfg: dict[str, Any] = {
        "enabled": True,
        "allowFrom": ["*"],
        "host": "127.0.0.1",
        "port": port,
        "path": "/",
        "websocketRequiresToken": False,
    }
    cfg.update(kw)
    parsed = WebSocketConfig.model_validate(cfg)
    gateway = build_gateway_services(
        config=parsed,
        bus=bus,
        session_manager=None,
        static_dist_path=None,
        workspace_path=Path.cwd(),
        default_restrict_to_workspace=False,
        runtime_model_name=None,
        runtime_surface="browser",
        runtime_capabilities_overrides=None,
        remote_access=remote_access,
    )
    return WebSocketChannel(cfg, bus, gateway=gateway)


@pytest.fixture()
def bus() -> MagicMock:
    b = MagicMock()
    b.publish_inbound = AsyncMock()
    return b


# -- Connection basics ----------------------------------------------------


@pytest.mark.asyncio
async def test_ready_event_fields(bus: MagicMock) -> None:
    ch = _ch(bus, 29901)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29901/", client_id="c1") as c:
            r = await c.recv_ready()
            assert r.event == "ready"
            assert len(r.chat_id) == 36
            assert r.client_id == "c1"
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_anonymous_client_gets_generated_id(bus: MagicMock) -> None:
    ch = _ch(bus, 29902)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29902/", client_id="") as c:
            r = await c.recv_ready()
            assert r.client_id.startswith("anon-")
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_each_connection_unique_chat_id(bus: MagicMock) -> None:
    ch = _ch(bus, 29903)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29903/", client_id="a") as c1:
            async with WsTestClient("ws://127.0.0.1:29903/", client_id="b") as c2:
                assert (await c1.recv_ready()).chat_id != (await c2.recv_ready()).chat_id
    finally:
        await ch.stop()
        await t


# -- Inbound messages (client -> server) ----------------------------------


@pytest.mark.asyncio
async def test_plain_text(bus: MagicMock) -> None:
    ch = _ch(bus, 29904)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29904/", client_id="p") as c:
            await c.recv_ready()
            await c.send_text("hello world")
            await asyncio.sleep(0.1)
            inbound = bus.publish_inbound.call_args[0][0]
            assert inbound.content == "hello world"
            assert inbound.sender_id == "p"
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_json_content_field(bus: MagicMock) -> None:
    ch = _ch(bus, 29905)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29905/", client_id="j") as c:
            await c.recv_ready()
            await c.send_json({"content": "structured"})
            await asyncio.sleep(0.1)
            assert bus.publish_inbound.call_args[0][0].content == "structured"
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_json_text_and_message_fields(bus: MagicMock) -> None:
    ch = _ch(bus, 29906)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29906/", client_id="x") as c:
            await c.recv_ready()
            await c.send_json({"text": "via text"})
            await asyncio.sleep(0.1)
            assert bus.publish_inbound.call_args[0][0].content == "via text"
            await c.send_json({"message": "via message"})
            await asyncio.sleep(0.1)
            assert bus.publish_inbound.call_args[0][0].content == "via message"
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_empty_payload_ignored(bus: MagicMock) -> None:
    ch = _ch(bus, 29907)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29907/", client_id="e") as c:
            await c.recv_ready()
            await c.send_text("   ")
            await c.send_json({})
            await asyncio.sleep(0.1)
            bus.publish_inbound.assert_not_awaited()
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_messages_preserve_order(bus: MagicMock) -> None:
    ch = _ch(bus, 29908)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29908/", client_id="o") as c:
            await c.recv_ready()
            for i in range(5):
                await c.send_text(f"msg-{i}")
            await asyncio.sleep(0.2)
            contents = [call[0][0].content for call in bus.publish_inbound.call_args_list]
            assert contents == [f"msg-{i}" for i in range(5)]
    finally:
        await ch.stop()
        await t


# -- Outbound messages (server -> client) ---------------------------------


@pytest.mark.asyncio
async def test_server_send_message(bus: MagicMock) -> None:
    ch = _ch(bus, 29909)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29909/", client_id="r") as c:
            ready = await c.recv_ready()
            await ch.send(OutboundMessage(
                channel="websocket", chat_id=ready.chat_id, content="reply",
            ))
            msg = await c.recv_message()
            assert msg.text == "reply"
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_revoked_mobile_client_does_not_receive_queued_private_message(
    bus: MagicMock,
    tmp_path: Path,
) -> None:
    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    channel = _ch(
        bus,
        port=0,
        allowFrom=["desktop"],
        token="desktop-secret",
        websocketRequiresToken=True,
        remote_access=RemoteAccessConfig(
            enabled=True,
            state_dir=str(owner.store.root),
            socket_path=str(owner.socket_path),
        ),
    )
    server_task = asyncio.create_task(channel.start())
    mobile = None
    desktop = None
    release_mobile_send = asyncio.Event()
    mobile_send_held = asyncio.Event()
    try:
        for _ in range(500):
            if channel._server is not None:
                break
            if server_task.done():
                await server_task
            await asyncio.sleep(0.01)
        assert channel._server is not None
        port = channel._server.sockets[0].getsockname()[1]
        origin = f"https://127.0.0.1:{port}"
        pair = owner.tokens.issue_device(
            "install-mobile",
            ("chat.read",),
            device_fingerprint="a" * 64,
            device_generation=1,
            origin=origin,
        )

        mobile = await websockets.connect(
            f"ws://127.0.0.1:{port}/mobile/v1/ws",
            origin=origin,
            additional_headers={"Authorization": f"Bearer {pair.access_token}"},
        )
        mobile_ready = json.loads(await asyncio.wait_for(mobile.recv(), timeout=2))
        desktop = await websockets.connect(
            f"ws://127.0.0.1:{port}/?client_id=desktop&token=desktop-secret"
        )
        await asyncio.wait_for(desktop.recv(), timeout=2)
        await desktop.send(json.dumps({
            "type": "attach",
            "chat_id": mobile_ready["chat_id"],
        }))
        while True:
            attached = json.loads(await asyncio.wait_for(desktop.recv(), timeout=2))
            if (
                attached.get("event") == "attached"
                and attached.get("chat_id") == mobile_ready["chat_id"]
            ):
                break

        original_revalidate = channel.revalidate_mobile_connection

        async def hold_mobile_send(connection: Any) -> bool:
            if getattr(connection, "mobile_identity", None) is not None:
                mobile_send_held.set()
                await release_mobile_send.wait()
            return await original_revalidate(connection)

        channel.revalidate_mobile_connection = hold_mobile_send
        message = "queued-mobile-revocation-7d27b7"
        await channel.send(OutboundMessage(
            channel="websocket",
            chat_id=mobile_ready["chat_id"],
            content=message,
        ))
        await asyncio.wait_for(mobile_send_held.wait(), timeout=2)

        while True:
            delivered = json.loads(await asyncio.wait_for(desktop.recv(), timeout=2))
            if delivered.get("text") == message:
                break

        owner.tokens.revoke("install-mobile")
        release_mobile_send.set()
        await asyncio.wait_for(mobile.wait_closed(), timeout=3)
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await asyncio.wait_for(mobile.recv(), timeout=1)
    finally:
        release_mobile_send.set()
        if mobile is not None:
            await mobile.close()
        if desktop is not None:
            await desktop.close()
        await channel.stop()
        await server_task
        owner.stop()


@pytest.mark.asyncio
async def test_malformed_mobile_identity_blocks_raw_message_but_desktop_still_works(
    bus: MagicMock,
    tmp_path: Path,
) -> None:
    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    channel = _ch(
        bus,
        port=0,
        allowFrom=["desktop"],
        token="desktop-secret",
        websocketRequiresToken=True,
        remote_access=RemoteAccessConfig(
            enabled=True,
            state_dir=str(owner.store.root),
            socket_path=str(owner.socket_path),
        ),
    )
    server_task = asyncio.create_task(channel.start())
    mobile = None
    desktop = None
    try:
        for _ in range(500):
            if channel._server is not None:
                break
            if server_task.done():
                await server_task
            await asyncio.sleep(0.01)
        assert channel._server is not None
        port = channel._server.sockets[0].getsockname()[1]
        origin = f"https://127.0.0.1:{port}"
        pair = owner.tokens.issue_device(
            "install-mobile",
            ("chat.read", "chat.write"),
            device_fingerprint="b" * 64,
            device_generation=1,
            origin=origin,
        )
        mobile = await websockets.connect(
            f"ws://127.0.0.1:{port}/mobile/v1/ws",
            origin=origin,
            additional_headers={"Authorization": f"Bearer {pair.access_token}"},
        )
        await asyncio.wait_for(mobile.recv(), timeout=2)
        mobile_connection = next(
            connection
            for connection in channel.gateway.endpoint.protocol_connections
            if getattr(connection, "mobile_access_token", None) == pair.access_token
        )
        mobile_connection.mobile_identity = object()

        desktop = await websockets.connect(
            f"ws://127.0.0.1:{port}/?client_id=desktop&token=desktop-secret"
        )
        await asyncio.wait_for(desktop.recv(), timeout=2)

        received: list[str] = []
        desktop_processed = asyncio.Event()

        async def record_inbound(message: Any) -> None:
            received.append(message.content)
            if message.content == "desktop-raw-message-6f5f8b":
                desktop_processed.set()

        bus.publish_inbound.side_effect = record_inbound
        await mobile.send("mobile-raw-message-1135dd")
        await asyncio.wait_for(mobile.wait_closed(), timeout=3)

        await desktop.send("desktop-raw-message-6f5f8b")
        await asyncio.wait_for(desktop_processed.wait(), timeout=3)
        assert received == ["desktop-raw-message-6f5f8b"]
    finally:
        if mobile is not None:
            await mobile.close()
        if desktop is not None:
            await desktop.close()
        await channel.stop()
        await server_task
        owner.stop()


@pytest.mark.asyncio
async def test_server_send_tags_tool_hint_with_kind(bus: MagicMock) -> None:
    """Tool-hint progress events surface as ``kind: "tool_hint"``."""
    ch = _ch(bus, 29919)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29919/", client_id="h") as c:
            ready = await c.recv_ready()
            # Plain reply: no "kind" field.
            await ch.send(OutboundMessage(
                channel="websocket", chat_id=ready.chat_id, content="hi",
            ))
            plain = await c.recv_message()
            assert plain.raw.get("kind") is None

            # Tool-hint breadcrumb: kind == "tool_hint".
            await ch.send(OutboundMessage(
                channel="websocket", chat_id=ready.chat_id,
                content='weather("get")',
                event=ProgressEvent(content='weather("get")', tool_hint=True),
            ))
            hint = await c.recv_message()
            assert hint.raw.get("kind") == "tool_hint"
            assert hint.text == 'weather("get")'

            # Generic progress (non-tool-hint) gets the softer "progress" label.
            await ch.send(OutboundMessage(
                channel="websocket", chat_id=ready.chat_id,
                content="thinking…",
                event=ProgressEvent(content="thinking…"),
            ))
            prog = await c.recv_message()
            assert prog.raw.get("kind") == "progress"
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_server_send_with_media_and_reply(bus: MagicMock) -> None:
    ch = _ch(bus, 29910)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29910/", client_id="m") as c:
            ready = await c.recv_ready()
            await ch.send(OutboundMessage(
                channel="websocket", chat_id=ready.chat_id, content="img",
                media=["/tmp/a.png"], reply_to="m1",
            ))
            msg = await c.recv_message()
            assert msg.text == "img"
            assert msg.media == ["/tmp/a.png"]
            assert msg.reply_to == "m1"
    finally:
        await ch.stop()
        await t


# -- Streaming ------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_deltas_and_end(bus: MagicMock) -> None:
    ch = _ch(bus, 29911, streaming=True)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29911/", client_id="s") as c:
            cid = (await c.recv_ready()).chat_id
            for part in ("Hello", " ", "world", "!"):
                await ch.send_delta(cid, part, stream_id="s1")
            await ch.send_delta(cid, "", stream_id="s1", stream_end=True)

            msgs = await c.collect_stream()
            deltas = [m for m in msgs if m.event == "delta"]
            assert "".join(d.text for d in deltas) == "Hello world!"
            ends = [m for m in msgs if m.event == "stream_end"]
            assert len(ends) == 1
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_interleaved_streams(bus: MagicMock) -> None:
    ch = _ch(bus, 29912, streaming=True)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29912/", client_id="i") as c:
            cid = (await c.recv_ready()).chat_id
            await ch.send_delta(cid, "A1", stream_id="sa")
            await ch.send_delta(cid, "B1", stream_id="sb")
            await ch.send_delta(cid, "A2", stream_id="sa")
            await ch.send_delta(cid, "", stream_id="sa", stream_end=True)
            await ch.send_delta(cid, "B2", stream_id="sb")
            await ch.send_delta(cid, "", stream_id="sb", stream_end=True)

            msgs = await c.recv_n(6)
            sa = "".join(m.text for m in msgs if m.event == "delta" and m.stream_id == "sa")
            sb = "".join(m.text for m in msgs if m.event == "delta" and m.stream_id == "sb")
            assert sa == "A1A2"
            assert sb == "B1B2"
    finally:
        await ch.stop()
        await t


# -- Multi-client ---------------------------------------------------------


@pytest.mark.asyncio
async def test_independent_sessions(bus: MagicMock) -> None:
    ch = _ch(bus, 29913)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29913/", client_id="u1") as c1:
            async with WsTestClient("ws://127.0.0.1:29913/", client_id="u2") as c2:
                r1, r2 = await c1.recv_ready(), await c2.recv_ready()
                await ch.send(OutboundMessage(
                    channel="websocket", chat_id=r1.chat_id, content="for-u1",
                ))
                assert (await c1.recv_message()).text == "for-u1"
                await ch.send(OutboundMessage(
                    channel="websocket", chat_id=r2.chat_id, content="for-u2",
                ))
                assert (await c2.recv_message()).text == "for-u2"
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_same_session_projects_one_turn_to_both_clients(bus: MagicMock) -> None:
    ch = _ch(bus, 29925)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29925/", client_id="terminal-a") as a:
            async with WsTestClient("ws://127.0.0.1:29925/", client_id="terminal-b") as b:
                chat_id = (await a.recv_ready()).chat_id
                await b.recv_ready()
                await b.send_json({"type": "attach", "chat_id": chat_id})
                attached = await b.recv()
                assert attached.event == "attached"
                assert attached.chat_id == chat_id

                await a.send_json(
                    {
                        "type": "message",
                        "chat_id": chat_id,
                        "content": "hello from terminal A",
                        "webui": True,
                        "turn_id": "turn-a",
                    }
                )

                accepted = await a.recv()
                projected = await b.recv()
                assert accepted.event == "message_accepted"
                assert accepted.raw["turn_id"] == "turn-a"
                assert accepted.raw["starts_turn"] is True
                assert accepted.raw["active_turn_id"] == "turn-a"
                assert projected.raw == {
                    "event": "user_message",
                    "chat_id": chat_id,
                    "text": "hello from terminal A",
                    "starts_turn": True,
                    "turn_id": "turn-a",
                    "active_turn_id": "turn-a",
                    "started_at": projected.raw["started_at"],
                }

                await ch.send_delta(chat_id, "shared reply", stream_id="stream-a")
                assert (await a.recv_delta()).text == "shared reply"
                assert (await b.recv_delta()).text == "shared reply"

                assert bus.publish_inbound.await_count == 1
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_disconnected_client_cleanup(bus: MagicMock) -> None:
    ch = _ch(bus, 29914)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29914/", client_id="tmp") as c:
            chat_id = (await c.recv_ready()).chat_id
        # disconnected
        await asyncio.sleep(0.1)
        await ch.send(OutboundMessage(
            channel="websocket", chat_id=chat_id, content="orphan",
        ))
        assert chat_id not in ch._subs
    finally:
        await ch.stop()
        await t


# -- Authentication -------------------------------------------------------


@pytest.mark.asyncio
async def test_static_token_accepted(bus: MagicMock) -> None:
    ch = _ch(bus, 29915, token="secret")
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29915/", client_id="a", token="secret") as c:
            assert (await c.recv_ready()).client_id == "a"
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_static_token_rejected(bus: MagicMock) -> None:
    ch = _ch(bus, 29916, token="correct")
    t = asyncio.create_task(ch.start())
    try:
        with pytest.raises(websockets.exceptions.InvalidStatus) as exc:
            async with WsTestClient("ws://127.0.0.1:29916/", client_id="b", token="wrong"):
                pass
        assert exc.value.response.status_code == 401
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_token_issue_full_flow(bus: MagicMock) -> None:
    ch = _ch(bus, 29917, path="/ws",
             tokenIssuePath="/auth/token", tokenIssueSecret="s",
             websocketRequiresToken=True)
    t = asyncio.create_task(ch.start())
    try:
        # no secret -> 401
        _, status = await issue_token(port=29917, issue_path="/auth/token")
        assert status == 401

        # with secret -> token
        token = await issue_token_ok(port=29917, issue_path="/auth/token", secret="s")

        # no token -> 401
        with pytest.raises(websockets.exceptions.InvalidStatus) as exc:
            async with WsTestClient("ws://127.0.0.1:29917/ws", client_id="x"):
                pass
        assert exc.value.response.status_code == 401

        # valid token -> ok
        async with WsTestClient("ws://127.0.0.1:29917/ws", client_id="ok", token=token) as c:
            assert (await c.recv_ready()).client_id == "ok"

        # reuse -> 401
        with pytest.raises(websockets.exceptions.InvalidStatus) as exc:
            async with WsTestClient("ws://127.0.0.1:29917/ws", client_id="r", token=token):
                pass
        assert exc.value.response.status_code == 401
    finally:
        await ch.stop()
        await t


# -- Path routing ---------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_path(bus: MagicMock) -> None:
    ch = _ch(bus, 29918, path="/my-chat")
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29918/my-chat", client_id="p") as c:
            assert (await c.recv_ready()).event == "ready"
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_wrong_path_404(bus: MagicMock) -> None:
    ch = _ch(bus, 29919, path="/ws")
    t = asyncio.create_task(ch.start())
    try:
        with pytest.raises(websockets.exceptions.InvalidStatus) as exc:
            async with WsTestClient("ws://127.0.0.1:29919/wrong", client_id="x"):
                pass
        assert exc.value.response.status_code == 404
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_trailing_slash_normalized(bus: MagicMock) -> None:
    ch = _ch(bus, 29920, path="/ws")
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29920/ws/", client_id="s") as c:
            assert (await c.recv_ready()).event == "ready"
    finally:
        await ch.stop()
        await t


# -- Edge cases -----------------------------------------------------------


@pytest.mark.asyncio
async def test_large_message(bus: MagicMock) -> None:
    ch = _ch(bus, 29921)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29921/", client_id="big") as c:
            await c.recv_ready()
            big = "x" * 100_000
            await c.send_text(big)
            await asyncio.sleep(0.2)
            assert bus.publish_inbound.call_args[0][0].content == big
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_unicode_roundtrip(bus: MagicMock) -> None:
    ch = _ch(bus, 29922)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29922/", client_id="u") as c:
            ready = await c.recv_ready()
            text = "你好世界 🌍 日本語テスト"
            await c.send_text(text)
            await asyncio.sleep(0.1)
            assert bus.publish_inbound.call_args[0][0].content == text
            await ch.send(OutboundMessage(
                channel="websocket", chat_id=ready.chat_id, content=text,
            ))
            assert (await c.recv_message()).text == text
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_rapid_fire(bus: MagicMock) -> None:
    ch = _ch(bus, 29923)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29923/", client_id="r") as c:
            ready = await c.recv_ready()
            for i in range(50):
                await c.send_text(f"in-{i}")
            await asyncio.sleep(0.5)
            assert bus.publish_inbound.await_count == 50
            for i in range(50):
                await ch.send(OutboundMessage(
                    channel="websocket", chat_id=ready.chat_id, content=f"out-{i}",
                ))
            received = [(await c.recv_message()).text for _ in range(50)]
            assert received == [f"out-{i}" for i in range(50)]
    finally:
        await ch.stop()
        await t


@pytest.mark.asyncio
async def test_invalid_json_as_plain_text(bus: MagicMock) -> None:
    ch = _ch(bus, 29924)
    t = asyncio.create_task(ch.start())
    try:
        async with WsTestClient("ws://127.0.0.1:29924/", client_id="j") as c:
            await c.recv_ready()
            await c.send_text("{broken json")
            await asyncio.sleep(0.1)
            assert bus.publish_inbound.call_args[0][0].content == "{broken json"
    finally:
        await ch.stop()
        await t
