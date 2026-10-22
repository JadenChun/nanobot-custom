"""A real canonical continuation retains A while chat remains selected on B."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.exec_session import ExecCompletion
from nanobot.agent.turn import DeliveryTarget, TurnRequest
from nanobot.process_completion import ProcessCompletionRuntime
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.security.workspace_access import build_workspace_scope
from tests.agent import p6_admitted_fixtures

admitted_async = p6_admitted_fixtures.admitted_async


@pytest.mark.asyncio
async def test_process_continuation_queues_behind_foreground_and_keeps_its_own_project(tmp_path, admitted_async):
    entered, release = asyncio.Event(), asyncio.Event()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False
    provider.estimate_prompt_tokens.return_value = (100, "fixture")
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs["messages"])
        if len(calls) == 1:
            assert "Project B instruction" in str(kwargs["messages"][0])
            entered.set()
            await release.wait()
            return LLMResponse(content="B finished")
        assert len(calls) == 2
        assert "Project A instruction" in str(kwargs["messages"][0])
        assert "Project B instruction" not in str(kwargs["messages"][0])
        assert "original-a-turn" in str(kwargs["messages"])
        return LLMResponse(content="A continuation finished")

    provider.chat_with_retry = AsyncMock(side_effect=chat)

    def build(inputs):
        return AgentLoop(bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
                         model="test-model", session_manager=inputs.sessions, tool_registry=inputs.tools,
                         cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
                         tools_config=inputs.config.tools, restrict_to_workspace=True,
                         enable_legacy_auto_compact=False)

    loop = await admitted_async(tmp_path, build, keys=("websocket:c1",))
    project_a, project_b = loop.workspace / "a", loop.workspace / "b"
    for path, label in ((project_a, "A"), (project_b, "B")):
        path.mkdir()
        (path / "AGENTS.md").write_text(f"Project {label} instruction")
    selected_b = build_workspace_scope(project_b, "restricted").metadata()
    assert loop.sessions.update_session_metadata("websocket:c1", {"workspace_scope": selected_b})
    process = ProcessCompletionRuntime(tmp_path / "process-runtime", loop)
    process.start()
    process.started("job-a", "run A's check", str(project_a), RequestContext(
        channel="websocket", chat_id="c1", session_key="websocket:c1",
        turn_id="original-a-turn", original_user_text="Work on A", workspace=project_a,
        workspace_scope=build_workspace_scope(project_a, "restricted"),
    ))
    foreground = asyncio.create_task(loop.execute_turn(TurnRequest(
        content="Work on B", session_key="websocket:c1", route=DeliveryTarget("websocket", "c1"),
        metadata={"workspace_scope": selected_b},
    )))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        process.finished("job-a", ExecCompletion("A's check passed", 0))
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(calls) == 1
        release.set()
        await asyncio.wait_for(foreground, 3)
        async with asyncio.timeout(3):
            while process.get("job-a").state != "done":
                await asyncio.sleep(.01)
        assert len(calls) == 2
        assert loop.sessions.read_session_metadata("websocket:c1")["metadata"]["workspace_scope"] == selected_b
    finally:
        release.set()
        await process.aclose()
        if not foreground.done():
            foreground.cancel()
        await asyncio.gather(foreground, return_exceptions=True)
