"""Focused tests for MyTool runtime sync side effects."""

from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.runtime_control import AgentRuntimeControl
from nanobot.agent.tools.self import MyTool
from tests.agent import p6_admitted_fixtures

admitted_async = p6_admitted_fixtures.admitted_async


@pytest.mark.asyncio
async def test_my_tool_max_iterations_syncs_subagent_limit(tmp_path, admitted_async) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    def build(inputs: p6_admitted_fixtures.AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            max_iterations=40, session_manager=inputs.sessions,
            tool_registry=inputs.tools, cron_service=inputs.cron,
            turn_delivery_factory=inputs.delivery, enable_legacy_auto_compact=False,
        )

    loop = await admitted_async(tmp_path, build)
    assert loop.tools.get("my") is None
    tool = MyTool(runtime_control=AgentRuntimeControl(loop))

    result = await tool.execute(action="set", key="max_iterations", value=80)

    assert "Set max_iterations = 80" in result
    assert loop.max_iterations == 80
    assert loop.subagents.max_iterations == 80
