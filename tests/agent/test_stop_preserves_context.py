"""Real cancellation preserves completed work; gateway shutdown retains recovery.

Regression coverage for https://github.com/HKUDS/nanobot/issues/2966.
The real runner records a completed read before cancellation, rather than
injecting a checkpoint into mocked storage or a removed execution method.
"""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.session.recovery import RUNTIME_CHECKPOINT_KEY, restore_runtime_checkpoint
from tests.agent import p6_admitted_fixtures

admitted_async = p6_admitted_fixtures.admitted_async


async def _cancel_after_completed_read(tmp_path, admitted_async, *, shutdown):
    blocked = asyncio.Event()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False
    provider.estimate_prompt_tokens.return_value = (100, "fixture")
    calls = 0

    async def chat(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return LLMResponse(content="Let me read.", tool_calls=[ToolCallRequest(
                id="tc_1", name="read_file", arguments={"path": "evidence.txt"},
            )])
        assert calls == 2
        tool_rows = [row for row in kwargs["messages"] if row.get("role") == "tool"]
        assert len(tool_rows) == 1 and "Search hit." in tool_rows[0]["content"]
        blocked.set()
        await asyncio.Event().wait()
        raise AssertionError("blocked provider unexpectedly resumed")

    provider.chat_with_retry = AsyncMock(side_effect=chat)

    def build(inputs):
        return AgentLoop(
            bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            model="test-model", session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    loop = await admitted_async(tmp_path, build, keys=("test:c1",))
    (loop.workspace / "evidence.txt").write_text("Search hit.", encoding="utf-8")
    if shutdown:
        loop.preserve_inflight_turns_on_shutdown()
    task = asyncio.create_task(loop._dispatch(InboundMessage(
        channel="test", sender_id="u1", chat_id="c1", content="Read the evidence",
    )))
    try:
        await asyncio.wait_for(blocked.wait(), timeout=2)
        before = loop.sessions.read_session_snapshot("test:c1")
        assert before is not None
        checkpoint = deepcopy(before.metadata[RUNTIME_CHECKPOINT_KEY])
        assert checkpoint["phase"] == "tools_completed"
        assert checkpoint["assistant_message"]["tool_calls"][0]["id"] == "tc_1"
        assert len(checkpoint["completed_tool_results"]) == 1
        assert "Search hit." in checkpoint["completed_tool_results"][0]["content"]
        assert checkpoint["pending_tool_calls"] == []
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
    after = loop.sessions.read_session_snapshot("test:c1")
    assert after is not None
    provider.chat_with_retry.assert_awaited()
    assert provider.chat_with_retry.await_count == 2
    assert not loop._pending_queues
    return loop, before, after, checkpoint


@pytest.mark.asyncio
async def test_dispatch_cancellation_restores_checkpoint(tmp_path, admitted_async):
    loop, before, after, _ = await _cancel_after_completed_read(
        tmp_path, admitted_async, shutdown=False,
    )
    assert [row["role"] for row in before.messages] == ["user"]
    assert [row["role"] for row in after.messages] == ["user", "assistant", "tool"]
    assert after.messages[1]["content"] == "Let me read."
    assert after.messages[2]["tool_call_id"] == "tc_1"
    assert "Search hit." in after.messages[2]["content"]
    assert RUNTIME_CHECKPOINT_KEY not in after.metadata
    assert loop.sessions.get_cached("test:c1").messages == after.messages
    # The detached disk snapshot proves persistence and restoration is idempotent.
    unchanged = deepcopy(after.messages)
    assert not restore_runtime_checkpoint(after)
    assert after.messages == unchanged


@pytest.mark.asyncio
async def test_dispatch_cancellation_keeps_checkpoint_for_gateway_shutdown(
    tmp_path, admitted_async,
):
    loop, before, after, checkpoint = await _cancel_after_completed_read(
        tmp_path, admitted_async, shutdown=True,
    )
    assert after.metadata[RUNTIME_CHECKPOINT_KEY] == checkpoint
    assert after.messages == before.messages
    assert [row["role"] for row in after.messages] == ["user"]
    await loop.aclose()
    reloaded = loop.sessions.read_session_snapshot("test:c1")
    assert reloaded is not None
    assert reloaded.metadata[RUNTIME_CHECKPOINT_KEY] == checkpoint
    assert reloaded.messages == before.messages
    assert loop.provider.chat_with_retry.await_count == 2
