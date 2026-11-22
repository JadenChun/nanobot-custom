import os
from pathlib import Path

import pytest

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import OutboundMessage
from nanobot.config.paths import get_workspace_path
from nanobot.context_repo.policy import ResourceAccessPolicy
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)


@pytest.mark.asyncio
async def test_message_tool_returns_error_when_no_target_context() -> None:
    tool = MessageTool()
    result = await tool.execute(content="test")
    assert result == "Error: No target channel/chat specified"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "not a list",
        [["ok"], "row-not-a-list"],
        [["ok", 42]],
        [[None]],
    ],
)
async def test_message_tool_rejects_malformed_buttons(bad) -> None:
    """``buttons`` must be ``list[list[str]]``; the tool validates the shape
    up front so a malformed LLM payload errors visibly instead of slipping
    into the channel layer where Telegram would silently reject the frame."""
    tool = MessageTool()
    result = await tool.execute(
        content="hi",
        channel="telegram",
        chat_id="1",
        buttons=bad,
    )
    assert result == "Error: buttons must be a list of list of strings"


@pytest.mark.asyncio
async def test_message_tool_suppresses_delivery_when_active() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)

    token = tool.set_suppress_delivery(True)
    try:
        result = await tool.execute(content="all clear", channel="telegram", chat_id="1")
    finally:
        tool.reset_suppress_delivery(token)
    assert sent == []
    assert "not delivered" in result

    await tool.execute(content="real", channel="telegram", chat_id="1")
    assert len(sent) == 1
    assert sent[0].content == "real"


@pytest.mark.asyncio
async def test_message_tool_records_media_deliveries() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)

    await tool.execute(
        content="image",
        channel="websocket",
        chat_id="chat-1",
        media=["/tmp/generated.png"],
    )

    assert sent[0].metadata == {"_record_channel_delivery": True}


@pytest.mark.asyncio
async def test_message_tool_inherits_metadata_for_same_target() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    slack_meta = {"slack": {"thread_ts": "111.222", "channel_type": "channel"}}

    with request_context(RequestContext(channel="slack", chat_id="C123", metadata=slack_meta)):
        await tool.execute(content="thread reply")

    assert sent[0].metadata == slack_meta


@pytest.mark.asyncio
async def test_message_tool_clears_metadata_when_context_has_none() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    rich_context = RequestContext(
        channel="slack",
        chat_id="C123",
        metadata={"slack": {"thread_ts": "111.222", "channel_type": "channel"}},
    )
    with request_context(rich_context):
        await tool.execute(content="thread reply")
    sent.clear()

    with request_context(
        RequestContext(
            channel="slack",
            chat_id="C123",
            metadata={},
        ),
    ):
        await tool.execute(content="plain reply")

    assert sent[0].metadata == {}


@pytest.mark.asyncio
async def test_message_tool_does_not_inherit_metadata_for_cross_target() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    with request_context(
        RequestContext(
            channel="slack",
            chat_id="C123",
            metadata={"slack": {"thread_ts": "111.222", "channel_type": "channel"}},
        ),
    ):
        await tool.execute(content="channel reply", channel="slack", chat_id="C999")

    assert sent[0].metadata == {}


@pytest.mark.asyncio
async def test_message_tool_resolves_relative_media_paths() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=["output/image.png"],
    )

    expected = str(get_workspace_path() / "output/image.png")
    assert sent[0].media == [expected]


@pytest.mark.asyncio
async def test_message_tool_resolves_relative_media_paths_from_active_workspace(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    workspace = tmp_path / "workspace"
    tool = MessageTool(send_callback=_send, workspace=workspace)

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=["output/image.png"],
    )

    assert sent[0].media == [str(workspace / "output/image.png")]


@pytest.mark.asyncio
async def test_message_tool_rejects_outside_workspace_absolute_media_when_restricted(
    tmp_path,
) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    tool = MessageTool(send_callback=_send, workspace=workspace, restrict_to_workspace=True)

    result = await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[str(outside)],
    )

    assert result.startswith("Error: media path is not allowed:")
    assert "outside allowed directory" in result
    assert sent == []


@pytest.mark.asyncio
async def test_message_tool_allows_workspace_absolute_media_when_restricted(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = workspace / "image.png"
    image.write_text("image", encoding="utf-8")
    tool = MessageTool(send_callback=_send, workspace=workspace, restrict_to_workspace=True)

    result = await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[str(image)],
    )

    assert result == "Message sent to telegram:1 with 1 attachments"
    assert sent[0].media == [str(image.resolve())]


def _folder_grant_setup(tmp_path: Path) -> tuple[Path, Path, Path, FolderGrantStore, ResourceAccessPolicy]:
    home = tmp_path / "home"
    granted = tmp_path / "granted"
    image = granted / "photo.png"
    home.mkdir()
    granted.mkdir()
    image.write_bytes(b"png")
    store = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    store.upsert(
        {"path": str(granted), "file_access": "read_write", "commands": True},
        expected_revision=0,
    )
    policy = ResourceAccessPolicy(workspace=home, folder_grants=store.view)
    return home, granted, image, store, policy


@pytest.mark.asyncio
async def test_message_tool_allows_folder_granted_media_in_managed_mode(tmp_path: Path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    home, _granted, image, _store, policy = _folder_grant_setup(tmp_path)
    tool = MessageTool(
        send_callback=_send,
        workspace=home,
        restrict_to_workspace=True,
        resource_policy=policy,
    )
    token = bind_workspace_scope(
        build_workspace_scope(home, "restricted", source_channel="websocket")
    )
    try:
        result = await tool.execute(
            content="see attached",
            channel="websocket",
            chat_id="granted",
            media=[str(image)],
        )
    finally:
        reset_workspace_scope(token)

    assert result == "Message sent to websocket:granted with 1 attachments"
    assert len(sent) == 1
    assert len(sent[0].media) == 1
    assert Path(sent[0].media[0]).samefile(image)


@pytest.mark.asyncio
async def test_message_tool_fails_closed_when_folder_grants_are_damaged(
    tmp_path: Path,
) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    home, _granted, image, store, policy = _folder_grant_setup(tmp_path)
    store.path.write_text(
        '{"version": 1, "enabled": true, "revision": 99, "grants": []}',
        encoding="utf-8",
    )
    tool = MessageTool(
        send_callback=_send,
        workspace=home,
        restrict_to_workspace=True,
        resource_policy=policy,
    )
    token = bind_workspace_scope(
        build_workspace_scope(home, "restricted", source_channel="websocket")
    )
    try:
        result = await tool.execute(
            content="see attached",
            channel="websocket",
            chat_id="damaged",
            media=[str(image)],
        )
    finally:
        reset_workspace_scope(token)

    assert result.startswith("Error: media path is not allowed:")
    assert sent == []


@pytest.mark.asyncio
async def test_message_tool_passes_through_absolute_media_paths() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)

    abs_path = os.path.abspath(os.path.join(os.sep, "tmp", "abs_image.png"))

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[abs_path],
    )

    assert sent[0].media == [abs_path]


@pytest.mark.asyncio
async def test_message_tool_passes_through_url_media_paths() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)

    url = "https://example.com/image.png"

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[url],
    )

    assert sent[0].media == [url]


@pytest.mark.asyncio
async def test_message_tool_resolves_mixed_media_paths() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)

    abs_path = os.path.abspath(os.path.join(os.sep, "tmp", "absolute.png"))

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[
            "output/relative.png",
            abs_path,
            "https://example.com/url.png",
            "http://example.com/http.png",
        ],
    )

    expected_relative = str(get_workspace_path() / "output/relative.png")
    assert sent[0].media == [
        expected_relative,
        abs_path,
        "https://example.com/url.png",
        "http://example.com/http.png",
    ]


@pytest.mark.asyncio
async def test_message_tool_rejects_wrong_explicit_ws_chat_id(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    conv = "550e8400-e29b-41d4-a716-446655440000"
    f = tmp_path / "doc.md"
    f.write_text("hello", encoding="utf-8")
    with request_context(RequestContext(channel="websocket", chat_id=conv, metadata={})):
        result = await tool.execute(
            content="see file",
            channel="websocket",
            chat_id="anon-deadbeefcafe",
            media=[str(f)],
        )
    assert result.startswith("Error: chat_id does not match")
    assert sent == []


@pytest.mark.asyncio
async def test_message_tool_allows_ws_explicit_when_matches_context(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    conv = "550e8400-e29b-41d4-a716-446655440000"
    f = tmp_path / "doc.md"
    f.write_text("hello", encoding="utf-8")
    with request_context(RequestContext(channel="websocket", chat_id=conv, metadata={})):
        result = await tool.execute(
            content="see file",
            channel="websocket",
            chat_id=conv,
            media=[str(f)],
        )
    assert result.startswith("Message sent")
    assert sent[0].chat_id == conv


@pytest.mark.asyncio
async def test_message_tool_cli_context_may_target_other_ws_chat(tmp_path) -> None:
    """Cron / CLI handlers keep non-websocket defaults; explicit websocket + uuid remains valid."""
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    target = "550e8400-e29b-41d4-a716-446655440000"
    f = tmp_path / "doc.md"
    f.write_text("hello", encoding="utf-8")
    with request_context(RequestContext(channel="cli", chat_id="direct", metadata={})):
        result = await tool.execute(
            content="ping",
            channel="websocket",
            chat_id=target,
            media=[str(f)],
        )
    assert result.startswith("Message sent")
    assert sent[0].channel == "websocket"
    assert sent[0].chat_id == target


@pytest.mark.asyncio
async def test_message_tool_attaches_webui_display_source_for_same_websocket_session(
    tmp_path: Path,
) -> None:
    """Managed WebUI turns must carry source authority so media messages persist to transcript."""
    from nanobot.session.source_store import SourceState
    from nanobot.webui.metadata import WEBUI_DISPLAY_SOURCE_KEY

    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    conv = "550e8400-e29b-41d4-a716-446655440000"
    img = tmp_path / "icon.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    source = SourceState(
        store_identity="store-1",
        session_key=f"websocket:{conv}",
        incarnation="inc-1",
        revision=3,
        generation=0,
        checkpoint_id=None,
        checkpoint_fingerprint="fp-1",
    )
    stale_source = SourceState(
        store_identity="store-1",
        session_key="websocket:other-chat",
        incarnation="inc-stale",
        revision=0,
        generation=0,
        checkpoint_id=None,
        checkpoint_fingerprint="fp-stale",
    )
    with request_context(
        RequestContext(
            channel="websocket",
            chat_id=conv,
            session_key=f"websocket:{conv}",
            metadata={WEBUI_DISPLAY_SOURCE_KEY: stale_source, "webui_turn_id": "turn-1"},
            source_state=source,
        )
    ):
        result = await tool.execute(
            content="The four monochrome masters.",
            media=[str(img)],
        )

    assert result == f"Message sent to websocket:{conv} with 1 attachments"
    assert len(sent) == 1
    assert sent[0].metadata[WEBUI_DISPLAY_SOURCE_KEY] == source
    assert sent[0].metadata["webui_turn_id"] == "turn-1"

    # Cross-channel sends must never inherit WebUI display source state.
    sent.clear()
    with request_context(
        RequestContext(
            channel="websocket",
            chat_id=conv,
            session_key=f"websocket:{conv}",
            metadata={"webui_turn_id": "turn-1"},
            source_state=source,
        )
    ):
        await tool.execute(
            content="cross channel",
            channel="telegram",
            chat_id="tg-1",
        )
    assert len(sent) == 1
    assert WEBUI_DISPLAY_SOURCE_KEY not in sent[0].metadata


@pytest.mark.asyncio
async def test_message_tool_media_persists_in_managed_webui_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outbound media sent via MessageTool in a managed session survives transcript replay."""
    from nanobot.session.execution_owner import acquire_execution
    from nanobot.session.runtime_authority import DurableSessionAuthority
    from nanobot.session.turn_writer import SourceTurnWriter
    from nanobot.webui.metadata import WEBUI_TURN_METADATA_KEY
    from nanobot.webui.transcript import (
        WebUITranscriptRecorder,
        build_webui_thread_response,
    )

    webui_dir = tmp_path / "webui"
    webui_dir.mkdir()
    monkeypatch.setattr("nanobot.webui.transcript.get_webui_dir", lambda: webui_dir)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(
        workspace,
        sessions_root=tmp_path / "sessions",
    ).acquire()
    try:
        sessions = lease.manager
        conv = "bde81382-a01f-467b-a271-999462600861"
        session_key = f"websocket:{conv}"
        assert sessions.upsert_session_metadata(session_key, {"webui": True}, create_if_absent=True)
        backend = sessions.source_backend(session_key)
        assert backend is not None
        async with acquire_execution(backend, session_key, run_id="seed-media") as owner:
            writer = SourceTurnWriter(backend.source, owner, run_id="seed-media")
            writer.append_message({"role": "user", "content": "show me the icon"})
            source_state = owner.read_snapshot().state

        recorder = WebUITranscriptRecorder(log=None, sessions=sessions)
        sent: list[OutboundMessage] = []

        async def _send(msg: OutboundMessage) -> None:
            sent.append(msg)

        img = workspace / "final-1024-mono-black.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        tool = MessageTool(send_callback=_send, workspace=workspace)

        with request_context(
            RequestContext(
                channel="websocket",
                chat_id=conv,
                session_key=session_key,
                metadata={WEBUI_TURN_METADATA_KEY: "turn-icons", "webui": True},
                source_state=source_state,
            )
        ):
            await tool.execute(content="Monochrome masters.", media=[str(img)])

        assert len(sent) == 1
        from loguru import logger as test_logger

        from nanobot.webui.media_gateway import WebUIMediaGateway

        media_gateway = WebUIMediaGateway(
            secret=b"test-secret",
            workspace_path=workspace,
            logger=test_logger,
        )
        signed = media_gateway.sign_or_stage_media_path(Path(sent[0].media[0]))
        assert signed is not None
        payload = {
            "event": "message",
            "chat_id": conv,
            "text": sent[0].content,
            "media": sent[0].media,
            "media_urls": [signed],
        }
        assert recorder.prepare_and_append(
            conv,
            payload,
            metadata=sent[0].metadata,
            phase="answer",
            include_source=True,
        )

        response = build_webui_thread_response(session_key, expected_source=source_state)
        assert response is not None
        replayed = response["messages"]
        assert len(replayed) == 1
        assert replayed[0]["content"] == "Monochrome masters."
        assert replayed[0]["media"][0]["name"] == "final-1024-mono-black.png"
        assert replayed[0]["media"][0]["url"] == signed["url"]
    finally:
        await lease.close()
