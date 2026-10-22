from pathlib import Path
from unittest.mock import MagicMock

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.cron import CronTool
from nanobot.providers.base import GenerationSettings
from tests.agent import p6_admitted_fixtures

admitted_sync = p6_admitted_fixtures.admitted_sync


def test_agent_loop_registers_cron_tool_with_configured_timezone(tmp_path: Path, admitted_sync) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()

    def build(inputs):
        return AgentLoop(
            bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            model="test-model", session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            timezone="Asia/Shanghai",
        )

    loop = admitted_sync(tmp_path, build)

    cron_tool = loop.tools.get("cron")

    assert isinstance(cron_tool, CronTool)
    assert cron_tool._default_timezone == "Asia/Shanghai"
