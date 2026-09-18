"""Acceptance checks for the canonical per-turn execution boundary."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.foreground import build_foreground_dependencies
from nanobot.agent.policy import RiskyActionPolicy
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.turn import (
    DelegationBudget,
    DeliveryTarget,
    RunStatus,
    ToolOutcome,
    TurnCallbacks,
    TurnContext,
    TurnRequest,
    TurnSource,
)
from nanobot.agent.write_guard import FileLockRegistry
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ToolsConfig
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.utils.llm_runtime import LLMRuntime


@pytest.fixture(autouse=True)
def _allow_guarded_constructor_for_this_suite(monkeypatch):
    """Open only this suite's constructor seam; restore the guard per test."""
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)


class _StoppingTool(Tool):
    @property
    def name(self) -> str:
        return "stopping_tool"

    @property
    def description(self) -> str:
        return "test tool that requests a terminal turn stop"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **_: object) -> str:
        return "legacy result"

    async def execute_with_context(self, context, **_: object) -> ToolOutcome:
        return ToolOutcome(
            content="stop now",
            stop_reason=RunStatus.POLICY_BLOCKED.value,
            policy_metadata={"reason": "test stop"},
        )


class _ToolErrorTool(_StoppingTool):
    @property
    def name(self) -> str:
        return "tool_error"

    async def execute_with_context(self, context, **_: object) -> ToolOutcome:
        return ToolOutcome(
            content="tool failed",
            stop_reason=RunStatus.TOOL_ERROR.value,
            policy_metadata={"reason": "test error"},
        )


@pytest.mark.asyncio
async def test_execute_turn_preserves_system_compat_route_and_source(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="system result", tool_calls=[], usage=None)
    )
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)

    result = await loop.execute_turn(TurnRequest(
        content="run the task",
        source=TurnSource.SYSTEM_COMPAT,
        session_key="telegram:room:thread",
        route=DeliveryTarget(channel="telegram", chat_id="room:thread"),
    ))

    assert result.content == "system result"
    assert result.outbound is not None
    assert (result.outbound.channel, result.outbound.chat_id) == ("telegram", "room:thread")
    assert result.record is not None
    assert result.record.source is TurnSource.SYSTEM_COMPAT


@pytest.mark.asyncio
async def test_contextual_tool_stop_does_not_make_an_extra_model_call(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(
            id="stop-1",
            name="stopping_tool",
            arguments={},
        )],
        usage=None,
    ))
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.tools.register(_StoppingTool())

    result = await loop.execute_turn(TurnRequest(
        content="stop after tool",
        session_key="cli:stop",
        tool_names=("stopping_tool",),
    ))

    assert provider.chat_with_retry.await_count == 1
    assert result.status is RunStatus.POLICY_BLOCKED
    assert result.stop_reason == RunStatus.POLICY_BLOCKED.value
    assert result.content == "stop now"


@pytest.mark.asyncio
async def test_contextual_tool_error_is_terminal_without_an_extra_model_call(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="error-1", name="tool_error", arguments={})],
        usage=None,
    ))
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.tools.register(_ToolErrorTool())

    result = await loop.execute_turn(TurnRequest(
        content="stop on error",
        session_key="cli:error",
        tool_names=("tool_error",),
    ))

    assert provider.chat_with_retry.await_count == 1
    assert result.status is RunStatus.TOOL_ERROR
    assert result.stop_reason == RunStatus.TOOL_ERROR.value
    assert result.content.startswith("tool failed")


@pytest.mark.asyncio
async def test_execute_turn_finalizes_unexpected_error_and_notifies_callback(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop._process_turn = AsyncMock(side_effect=RuntimeError("boom"))
    on_error = AsyncMock()

    result = await loop.execute_turn(TurnRequest(
        content="fail",
        session_key="cli:fail",
        callbacks=TurnCallbacks(on_error=on_error),
    ))

    assert result.status is RunStatus.ERROR
    assert result.error == "boom"
    on_error.assert_awaited_once_with(result)
    stored = loop.run_store.load(result.run_id or "")
    assert stored is not None
    assert stored.status is RunStatus.ERROR


@pytest.mark.asyncio
async def test_message_context_uses_route_local_delivery_ledger() -> None:
    sent = []
    tool = MessageTool(send_callback=AsyncMock(side_effect=sent.append))
    first = TurnContext(
        request=TurnRequest(
            content="A",
            session_key="telegram:A",
            route=DeliveryTarget(channel="telegram", chat_id="A"),
        ),
        run_id="run-a",
    )
    second = TurnContext(
        request=TurnRequest(
            content="B",
            session_key="telegram:B",
            route=DeliveryTarget(channel="telegram", chat_id="B"),
        ),
        run_id="run-b",
    )

    await tool.execute_with_context(first, content="for A")
    await tool.execute_with_context(second, content="for B")

    assert [message.chat_id for message in sent] == ["A", "B"]
    assert [message.chat_id for message in first.delivery.sent_messages] == ["A"]
    assert [message.chat_id for message in second.delivery.sent_messages] == ["B"]


@pytest.mark.asyncio
async def test_auxiliary_message_does_not_mark_primary_delivery_complete() -> None:
    tool = MessageTool(send_callback=AsyncMock())
    context = TurnContext(
        request=TurnRequest(
            content="send elsewhere",
            session_key="telegram:primary",
            route=DeliveryTarget(channel="telegram", chat_id="primary"),
        ),
        run_id="run-aux",
    )

    await tool.execute_with_context(
        context,
        content="auxiliary",
        channel="discord",
        chat_id="other",
    )

    assert context.delivery.sent_messages[0].channel == "discord"
    assert context.delivery.delivered is False


@pytest.mark.asyncio
async def test_cron_context_uses_route_and_rejects_nested_scheduling(tmp_path) -> None:
    service = MagicMock()
    service.add_job.return_value = SimpleNamespace(name="job", id="job-1")
    tool = CronTool(service)
    context = TurnContext(
        request=TurnRequest(
            content="schedule",
            source=TurnSource.DIRECT,
            session_key="telegram:A",
            route=DeliveryTarget(channel="telegram", chat_id="A"),
        ),
        run_id="run-a",
    )

    result = await tool.execute_with_context(
        context,
        action="add",
        message="remind me",
        every_seconds=60,
        additional_destinations=[{"channel": "discord", "to": "B"}],
    )
    assert "Created job" in result.content
    assert service.add_job.call_args.kwargs["channel"] == "telegram"
    assert service.add_job.call_args.kwargs["to"] == "A"
    assert service.add_job.call_args.kwargs["additional_destinations"][0].channel == "discord"
    assert service.add_job.call_args.kwargs["additional_destinations"][0].to == "B"

    cron_context = TurnContext(
        request=TurnRequest(
            content="nested",
            source=TurnSource.CRON,
            session_key="cron:job-1",
            route=DeliveryTarget(channel="telegram", chat_id="A"),
        ),
        run_id="run-cron",
    )
    nested = await tool.execute_with_context(
        cron_context,
        action="add",
        message="not allowed",
        every_seconds=60,
    )
    assert "cannot schedule" in nested.content
    assert service.add_job.call_count == 1


@pytest.mark.asyncio
async def test_filesystem_context_uses_lock_owner(tmp_path) -> None:
    target = tmp_path / "draft.md"
    target.write_text("hello world", encoding="utf-8")
    locks = FileLockRegistry()
    await locks.acquire(target, "owner-a")
    tool = EditFileTool(workspace=tmp_path, lock_registry=locks)
    context = TurnContext(
        request=TurnRequest(content="edit", session_key="cli:B"),
        run_id="run-b",
        lock_owner="owner-b",
    )
    try:
        result = await tool.execute_with_context(
            context,
            path="draft.md",
            old_text="world",
            new_text="earth",
        )
    finally:
        await locks.release(target, "owner-a")
    assert "owner-a" in result
    assert target.read_text(encoding="utf-8") == "hello world"


def test_foreground_worker_capability_and_context_budget_use_shared_dependencies(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=4096, temperature=0.1)
    runtime = LLMRuntime.capture(provider, "test-model", context_window_tokens=120000)
    dependencies = build_foreground_dependencies(
        workspace=tmp_path,
        runtime=runtime,
        tools_config=ToolsConfig(),
        restrict_to_workspace=True,
    )
    manager = dependencies.foreground_manager
    scopes = manager._normalize_scopes(tmp_path, ["src/"])
    worker_tools = manager._worker_tools("call-1", scopes)
    assert set(worker_tools.tool_names) == {"read_file", "list_dir", "write_file", "edit_file"}

    first = TurnContext(
        request=TurnRequest(content="one", session_key="cli:one"),
        run_id="one",
        delegation_budget=DelegationBudget(max_calls=1, max_worker_corrections=1),
    )
    second = TurnContext(
        request=TurnRequest(content="two", session_key="cli:two"),
        run_id="two",
        delegation_budget=DelegationBudget(max_calls=1, max_worker_corrections=1),
    )
    assert manager._consume_call("planner", first) is None
    assert "limit reached" in (manager._consume_call("reviewer", first) or "")
    assert manager._consume_call("reviewer", second) is None


@pytest.mark.asyncio
async def test_browser_policy_allows_navigation_and_requires_approval_for_mutation(tmp_path) -> None:
    from nanobot.providers.base import ToolCallRequest

    policy = RiskyActionPolicy(workspace=tmp_path)
    read = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(id="read", name="agent_browser", arguments={"args": ["open", "https://example.com"]})],
    )
    mutate = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(id="mutate", name="agent_browser", arguments={"args": ["click", "@e1"]})],
    )
    unknown = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(id="unknown", name="agent_browser", arguments={"args": ["future-command"]})],
    )
    assert read.action == "allow"
    assert mutate.stop_reason == "approval_required"
    assert unknown.stop_reason == "approval_required"
