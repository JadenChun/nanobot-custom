"""Dormant subagent adapter's timeout contract, separate from runtime admission."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.agent.subagent import SubagentManager, SubagentStatus
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings
from nanobot.utils.llm_runtime import LLMRuntime


@pytest.mark.asyncio
async def test_subagent_forwards_resolver_to_agent_run_spec(tmp_path: Path, monkeypatch) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "m"
    provider.generation = GenerationSettings()
    mgr = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=64,
        llm_wall_timeout_for_session=lambda sk: 0.0 if sk == "cli:direct" else None,
    )

    mgr.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="ok", messages=[], stop_reason="completed")
    )
    mgr._announce_result = AsyncMock()
    tools = ToolRegistry()
    # Isolate adapter argument forwarding, not the unsupported tool factory.
    monkeypatch.setattr(mgr, "_build_tools", lambda **kwargs: tools)
    monkeypatch.setattr(mgr, "_build_subagent_prompt", lambda **kwargs: "isolated adapter test")

    status = SubagentStatus(
        task_id="t1",
        label="lbl",
        task_description="task",
        started_at=0.0,
    )
    await mgr._run_subagent(
        "t1",
        "task",
        "lbl",
        {"channel": "cli", "chat_id": "direct", "session_key": "cli:direct"},
        status,
        LLMRuntime.capture(provider, "m", context_window_tokens=128_000),
    )
    mgr.runner.run.assert_called_once()
    spec = mgr.runner.run.call_args[0][0]
    assert spec.session_key == "cli:direct"
    assert spec.llm_timeout_s == 0.0
    assert spec.tools is tools
    await mgr.close()


def test_legacy_subagent_tool_factory_rejects_unsupported_scope(tmp_path):
    mgr = SubagentManager(workspace=tmp_path, bus=MessageBus(), max_tool_result_chars=64)
    with pytest.raises(ValueError, match="Unsupported runtime tool scope"):
        mgr._build_tools()
