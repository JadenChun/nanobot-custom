from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.config.schema import AgentDeviceConfig


def _make_loop(*, agent_device_config=None):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as mock_subagents:
        mock_subagents.return_value.cancel_by_session = AsyncMock(return_value=0)
        return AgentLoop(
            bus=bus,
            provider=provider,
            workspace=MagicMock(),
            agent_device_config=agent_device_config,
        )


def test_loop_registers_agent_device_by_default() -> None:
    loop = _make_loop()
    assert loop.tools.get("agent_device") is not None


def test_loop_skips_agent_device_when_disabled() -> None:
    loop = _make_loop(agent_device_config=AgentDeviceConfig(enabled=False))
    assert loop.tools.get("agent_device") is None


def test_subagent_generation_tools_skip_agent_device_when_disabled(tmp_path) -> None:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=bus,
        agent_device_config=AgentDeviceConfig(enabled=False),
    )

    tools = mgr._build_generation_tools()
    assert tools.get("agent_device") is None


def test_subagent_generation_tools_register_agent_device_by_default(tmp_path) -> None:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)

    tools = mgr._build_generation_tools()
    assert tools.get("agent_device") is not None
