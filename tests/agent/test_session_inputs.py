import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.session_messages import SendSessionMessageTool
from nanobot.bus.events import InboundMessage
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.runtime_context import public_history_message
from nanobot.session.session_messages import SESSION_MESSAGE_METADATA_KEY
from tests.agent import p6_admitted_fixtures
from tests.agent.test_loop_runner_integration import _execute_inbound

admitted_async = p6_admitted_fixtures.admitted_async
_run_agent = AgentLoop.run


async def _loop(tmp_path: Path, admitted_async, *, session_tool=False) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="Reviewed", tool_calls=[], usage=None)
    )
    def build(inputs):
        return AgentLoop(
            bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            model="test-model", session_manager=inputs.sessions,
            tool_registry=inputs.tools, cron_service=inputs.cron,
            turn_delivery_factory=inputs.delivery, enable_legacy_auto_compact=False,
        )

    loop = await admitted_async(tmp_path, build, keys=("telegram:target",))
    assert not loop.tools.has("send_session_message")
    if session_tool:
        # The foreground manifest does not install this optional tool. Its
        # runtime-context contribution must be tested with the real tool present.
        loop.tools.register(SendSessionMessageTool(sessions=loop.sessions, bus=loop.bus))
    return loop


def _message(content: str = "Please review") -> InboundMessage:
    envelope = {
        "message_id": "message-1",
        "created_at_ms": 1,
        "expect_reply": True,
        "source_handle": "luma",
        "source_session_key": "websocket:source",
        "target_session_key": "telegram:target",
    }
    return InboundMessage(
        channel="system",
        sender_id="session",
        chat_id="telegram:target",
        content=content,
        metadata={SESSION_MESSAGE_METADATA_KEY: envelope},
        session_key_override="telegram:target",
        input_role="user",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("session_tool", [False, True])
async def test_session_message_runs_as_user_input_and_replies_on_target_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admitted_async,
    session_tool,
) -> None:
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path / "state")
    loop = await _loop(tmp_path, admitted_async, session_tool=session_tool)
    loop.sessions.save(loop.sessions.get_or_create("telegram:target"))
    msg = _message()

    response = await _execute_inbound(loop, msg)

    assert response is not None
    assert (response.channel, response.chat_id, response.content) == (
        "telegram",
        "target",
        "Reviewed",
    )
    provider_messages = loop.provider.chat_with_retry.await_args.kwargs["messages"]
    provider_input = next(
        row for row in reversed(provider_messages) if row.get("role") == "user"
    )
    assert provider_input["content"].startswith("Please review")
    if session_tool:
        assert "Message from @luma." in provider_input["content"]
        assert "Reply with send_session_message." in provider_input["content"]
    else:
        assert provider_input["content"] == "Please review"

    stored = loop.sessions.get_or_create("telegram:target").messages
    user_row = next(row for row in stored if row.get("role") == "user")
    assert public_history_message(user_row)["content"] == "Please review"
    assert SESSION_MESSAGE_METADATA_KEY not in user_row


@pytest.mark.asyncio
@pytest.mark.parametrize("session_tool", [False, True])
async def test_session_message_text_is_not_dispatched_as_a_slash_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admitted_async,
    session_tool,
) -> None:
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path / "state")
    loop = await _loop(tmp_path, admitted_async, session_tool=session_tool)
    loop.sessions.save(loop.sessions.get_or_create("telegram:target"))
    # Invoke only this real consumer while the fixture's external-effect fences remain.
    task = asyncio.create_task(_run_agent(loop))
    try:
        await loop.bus.publish_inbound(_message("/stop"))
        response = await asyncio.wait_for(loop.bus.consume_outbound(), timeout=2)

        assert response.content == "Reviewed"
        loop.provider.chat_with_retry.assert_awaited_once()
    finally:
        loop.stop()
        await asyncio.wait_for(task, timeout=2)
