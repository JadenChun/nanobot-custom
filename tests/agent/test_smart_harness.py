from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import _PlanDecision, _VerificationResult
from nanobot.agent.policy import RiskyActionPolicy, ToolPolicy, ToolPolicyDecision
from nanobot.agent.runner import AgentRunResult, AgentRunSpec, AgentRunner
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path, *, planning_mode: str = "agent"):
    from nanobot.agent.loop import AgentLoop

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        planning_mode=planning_mode,
    )
    return loop, provider


@pytest.mark.asyncio
async def test_runner_tool_policy_can_stop_before_execution():
    class StopPolicy(ToolPolicy):
        async def evaluate(self, *, messages, tool_calls):
            return ToolPolicyDecision(
                action="respond",
                stop_reason="approval_required",
                response="Need approval first.",
                metadata={"summary": "dangerous action"},
            )

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="exec", arguments={"command": "git push"})],
        usage={},
    ))

    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="should not run")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
        tool_policy=StopPolicy(),
    ))

    assert result.stop_reason == "approval_required"
    assert result.final_content == (
        "Need approval first.\n\n"
        "Proposed approach before approval:\nworking"
    )
    assert result.policy_metadata == {"summary": "dangerous action"}
    tools.execute.assert_not_awaited()


def test_risky_action_policy_detects_large_overwrite(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("\n".join(f"line {i}" for i in range(300)), encoding="utf-8")

    policy = RiskyActionPolicy(workspace=tmp_path)
    reason = policy._risky_write_reason(
        "app.py",
        "\n".join(f"new {i}" for i in range(320)),
    )

    assert "overwrite a large portion" in reason


@pytest.mark.asyncio
async def test_process_direct_requires_approval_then_resumes(tmp_path):
    loop, provider = _make_loop(tmp_path, planning_mode="off")

    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="working",
            tool_calls=[ToolCallRequest(id="call_1", name="exec", arguments={"command": "git push origin main"})],
            usage={},
        ),
        LLMResponse(content="Push complete.", tool_calls=[], usage={}),
    ])

    first = await loop.process_direct("push the current branch")
    assert first is not None
    assert "Reply `yes` to continue" in first.content

    second = await loop.process_direct("yes")
    assert second is not None
    assert second.content == "Push complete."


@pytest.mark.asyncio
async def test_process_direct_approval_prompt_stays_visible_when_streaming(tmp_path):
    loop, provider = _make_loop(tmp_path, planning_mode="off")

    provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="I’ll update the file after approval.",
            tool_calls=[ToolCallRequest(id="call_1", name="edit_file", arguments={
                "path": "script.md",
                "old_text": "old",
                "new_text": "\n".join(f"new line {i}" for i in range(300)),
                "replace_all": False,
            })],
            usage={},
        ),
    ])

    progress_updates: list[str] = []
    stream_updates: list[str] = []

    response = await loop.process_direct(
        "update the script",
        on_progress=progress_updates.append,
        on_stream=stream_updates.append,
        on_stream_end=AsyncMock(),
    )

    assert response is not None
    assert response.metadata.get("_streamed") is not True
    assert "Reply `yes` to continue" in response.content
    assert "Proposed approach before approval" in response.content


@pytest.mark.asyncio
async def test_run_main_task_retries_after_verifier_failure(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="agent")

    initial_result = AgentRunResult(
        final_content="Initial answer",
        messages=[{"role": "assistant", "content": "Initial answer"}],
        tools_used=["edit_file"],
    )
    revised_result = AgentRunResult(
        final_content="Revised answer",
        messages=[{"role": "assistant", "content": "Revised answer"}],
        tools_used=["edit_file"],
    )

    loop._run_agent = AsyncMock(side_effect=[initial_result, revised_result])  # type: ignore[method-assign]
    loop._run_internal_verifier = AsyncMock(side_effect=[
        _VerificationResult(verdict="FAIL", issues=["Missing test"], feedback="Add the missing test."),
        _VerificationResult(verdict="PASS", issues=[], feedback=""),
    ])  # type: ignore[method-assign]

    result = await loop._run_main_task(
        [{"role": "user", "content": "fix the bug"}],
        task_text="fix the bug",
        channel="cli",
        chat_id="direct",
        message_id=None,
        approval_granted=False,
        planned=_PlanDecision(decision="execute", summary="Fix the bug safely", needs_verification=True),
    )

    assert result.final_content == "Revised answer"
    assert loop._run_agent.await_count == 2  # type: ignore[attr-defined]
    assert loop._run_internal_verifier.await_count == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_run_main_task_verifies_without_plan_object(tmp_path):
    loop, _provider = _make_loop(tmp_path, planning_mode="agent")

    initial_result = AgentRunResult(
        final_content="Done",
        messages=[{"role": "assistant", "content": "Done"}],
        tools_used=["edit_file"],
    )

    loop._run_agent = AsyncMock(return_value=initial_result)  # type: ignore[method-assign]
    loop._run_internal_verifier = AsyncMock(return_value=  # type: ignore[method-assign]
        _VerificationResult(verdict="PASS", issues=[], feedback="")
    )

    result = await loop._run_main_task(
        [{"role": "user", "content": "ok"}],
        task_text="ok",
        channel="cli",
        chat_id="direct",
        message_id=None,
        approval_granted=False,
        planned=None,
    )

    assert result.final_content == "Done"
    loop._run_internal_verifier.assert_awaited_once_with(  # type: ignore[attr-defined]
        [{"role": "assistant", "content": "Done"}],
        goal="ok",
        channel="cli",
        chat_id="direct",
    )
