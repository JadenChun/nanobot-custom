"""Test message tool suppress logic for final replies."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context import TranscriptInput
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.turn import DeliveryState, DeliveryTarget, RunPolicyContext, TurnContext
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.utils.progress_events import output_events
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async
from tests.tools.p6_message_fixtures import execute_message, make_loop


class TestMessageToolSuppressLogic:
    """Final reply suppressed only when message tool sends to the same target."""

    @pytest.mark.asyncio
    async def test_suppress_when_sent_to_same_target(self, tmp_path: Path, admitted_async) -> None:
        loop = await make_loop(tmp_path, admitted_async)
        tool_call = ToolCallRequest(
            id="call1", name="message",
            arguments={"content": "Hello", "channel": "feishu", "chat_id": "chat123"},
        )
        calls = iter([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="Hello", tool_calls=[]),
        ])
        loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])

        sent: list[OutboundMessage] = []
        mt = loop.tools.get("message")
        if isinstance(mt, MessageTool):
            mt.set_send_callback(AsyncMock(side_effect=lambda m: sent.append(m)))

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Send")
        turn = await execute_message(loop, msg)
        result = turn.outbound

        assert len(sent) == 1
        assert isinstance(mt, MessageTool)
        assert turn.sent_messages == tuple(sent)
        assert result is None  # suppressed

    @pytest.mark.asyncio
    async def test_not_suppress_when_primary_message_is_auxiliary_text(self, tmp_path: Path, admitted_async) -> None:
        loop = await make_loop(tmp_path, admitted_async)
        tool_call = ToolCallRequest(
            id="call1", name="message",
            arguments={"content": "Attachment ready", "channel": "feishu", "chat_id": "chat123"},
        )
        calls = iter([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="The final answer", tool_calls=[]),
        ])
        loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])

        sent: list[OutboundMessage] = []
        mt = loop.tools.get("message")
        if isinstance(mt, MessageTool):
            mt.set_send_callback(AsyncMock(side_effect=lambda m: sent.append(m)))

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Send")
        result = (await execute_message(loop, msg)).outbound

        assert len(sent) == 1
        assert result is not None
        assert result.content == "The final answer"

    @pytest.mark.asyncio
    async def test_not_suppress_when_primary_send_only_has_attachment(self, tmp_path: Path, admitted_async) -> None:
        loop = await make_loop(tmp_path, admitted_async)
        attachment = loop.workspace / "report.pdf"
        attachment.write_bytes(b"synthetic attachment")
        tool_call = ToolCallRequest(
            id="call1", name="message",
            arguments={
                "content": "",
                "channel": "feishu",
                "chat_id": "chat123",
                "media": ["report.pdf"],
            },
        )
        calls = iter([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="The report is attached", tool_calls=[]),
        ])
        loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])

        sent: list[OutboundMessage] = []
        mt = loop.tools.get("message")
        if isinstance(mt, MessageTool):
            mt.set_send_callback(AsyncMock(side_effect=lambda m: sent.append(m)))

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Send")
        result = (await execute_message(loop, msg)).outbound

        assert len(sent) == 1
        assert sent[0].content == ""
        assert sent[0].media == [str(attachment)]
        assert result is not None
        assert result.content == "The report is attached"

    @pytest.mark.asyncio
    async def test_not_suppress_when_sent_to_different_target(self, tmp_path: Path, admitted_async) -> None:
        loop = await make_loop(tmp_path, admitted_async)
        tool_call = ToolCallRequest(
            id="call1", name="message",
            arguments={"content": "Email content", "channel": "email", "chat_id": "user@example.com"},
        )
        calls = iter([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="I've sent the email.", tool_calls=[]),
        ])
        loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])

        sent: list[OutboundMessage] = []
        mt = loop.tools.get("message")
        if isinstance(mt, MessageTool):
            mt.set_send_callback(AsyncMock(side_effect=lambda m: sent.append(m)))

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Send email")
        result = (await execute_message(loop, msg)).outbound

        assert len(sent) == 1
        assert sent[0].channel == "email"
        assert result is not None  # not suppressed
        assert result.channel == "feishu"

    @pytest.mark.asyncio
    async def test_cron_run_suppresses_final_result_after_explicit_message(self, tmp_path: Path, admitted_async) -> None:
        loop = await make_loop(tmp_path, admitted_async)
        tool_call = ToolCallRequest(
            id="call1",
            name="message",
            arguments={"content": "Attachment ready", "channel": "telegram", "chat_id": "123"},
        )
        calls = iter([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="Attachment ready", tool_calls=[]),
        ])
        loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])

        sent: list[OutboundMessage] = []
        mt = loop.tools.get("message")
        if isinstance(mt, MessageTool):
            mt.set_send_callback(AsyncMock(side_effect=lambda message: sent.append(message)))

        msg = InboundMessage(channel="telegram", sender_id="cron", chat_id="123", content="Run")
        result = (await execute_message(loop, msg, session_key="cron:daily-content")).outbound

        assert len(sent) == 1
        assert result is None  # explicit Telegram delivery already completed the turn

    @pytest.mark.asyncio
    async def test_not_suppress_when_no_message_tool_used(self, tmp_path: Path, admitted_async) -> None:
        loop = await make_loop(tmp_path, admitted_async)
        loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="Hello!", tool_calls=[]))
        loop.tools.get_definitions = MagicMock(return_value=[])

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Hi")
        result = (await execute_message(loop, msg)).outbound

        assert result is not None
        assert "Hello" in result.content

    async def test_progress_hides_internal_reasoning(self, tmp_path: Path, admitted_async) -> None:
        loop = await make_loop(tmp_path, admitted_async)
        (loop.workspace / "foo.txt").write_text("ok", encoding="utf-8")
        tool_call = ToolCallRequest(id="call1", name="read_file", arguments={"path": "foo.txt"})
        calls = iter([
            LLMResponse(
                content="Visible<think>hidden</think>",
                tool_calls=[tool_call],
                reasoning_content="secret reasoning",
                thinking_blocks=[{"signature": "sig", "thought": "secret thought"}],
            ),
            LLMResponse(content="Done", tool_calls=[]),
        ])
        loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])

        progress: list[tuple[str, bool]] = []

        async def on_progress(content: str, *, tool_hint: bool = False) -> None:
            progress.append((content, tool_hint))

        result = await loop._run_agent_loop(
            TranscriptInput(history=[], current_message=None),
            runtime=loop.llm_runtime(), events=output_events(on_progress=on_progress),
            run_context=TurnContext(policy=RunPolicyContext(
                workspace=loop.workspace,
                context_manager=loop._fork_dependencies.resource_policy.context_manager,
            )),
        )

        assert result.final_content == "Done"
        assert progress == [
            ("Visible", False),
            ('read foo.txt', True),
        ]


class TestMessageToolTurnTracking:

    async def test_sent_in_turn_tracks_same_target(self, tmp_path: Path) -> None:
        sent: list[OutboundMessage] = []
        tool = MessageTool(send_callback=AsyncMock(side_effect=sent.append), workspace=tmp_path)
        context = TurnContext(delivery=DeliveryState(primary=DeliveryTarget("feishu", "chat1")))
        assert not context.delivery.delivered
        await tool.execute_with_context(context, content="done")
        assert context.delivery.delivered
        assert context.delivery.sent_messages == sent
        assert [(msg.channel, msg.chat_id, msg.content) for msg in sent] == [
            ("feishu", "chat1", "done"),
        ]

    async def test_start_turn_resets(self, tmp_path: Path) -> None:
        tool = MessageTool(send_callback=AsyncMock(), workspace=tmp_path)
        first = TurnContext(delivery=DeliveryState(primary=DeliveryTarget("feishu", "chat1")))
        await tool.execute_with_context(first, content="done")
        second = TurnContext(delivery=DeliveryState(primary=DeliveryTarget("feishu", "chat1")))
        assert not second.delivery.delivered
        assert second.delivery.sent_messages == []
        assert first.delivery.delivered
        assert len(first.delivery.sent_messages) == 1
        await tool.execute_with_context(second, content="next")
        assert [message.content for message in first.delivery.sent_messages] == ["done"]
        assert [message.content for message in second.delivery.sent_messages] == ["next"]
