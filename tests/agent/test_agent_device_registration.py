from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.config.schema import AgentDeviceConfig, ImageConfig


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


def test_loop_passes_image_config_to_subagents() -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    image_config = ImageConfig(provider="openrouter", api_key="test-key")

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as mock_subagents:
        mock_subagents.return_value.cancel_by_session = AsyncMock(return_value=0)
        AgentLoop(
            bus=bus,
            provider=provider,
            workspace=MagicMock(),
            image_config=image_config,
        )

    assert mock_subagents.call_args.kwargs["image_config"] is image_config


def test_loop_passes_context_paths_to_subagents(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = tmp_path / "workspace"
    context_repo = tmp_path / "context-repo"
    workspace.mkdir()
    context_repo.mkdir()

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as mock_subagents:
        mock_subagents.return_value.cancel_by_session = AsyncMock(return_value=0)
        AgentLoop(
            bus=bus,
            provider=provider,
            workspace=workspace,
            context_paths=[context_repo],
        )

    assert mock_subagents.call_args.kwargs["context_paths"] == [context_repo]


def test_subagent_prompt_includes_context_repo_skills(tmp_path) -> None:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

    workspace = tmp_path / "workspace"
    context_repo = tmp_path / "context-repo"
    skill_dir = context_repo / "skills" / "seo-review"
    workspace.mkdir()
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: seo-review\n"
        "description: Context SEO workflow skill\n"
        "---\n\n"
        "# SEO Review\n",
        encoding="utf-8",
    )

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(
        provider=provider,
        workspace=workspace,
        bus=bus,
        context_paths=[context_repo],
    )

    prompt = mgr._build_subagent_prompt()

    assert str(context_repo.resolve()) in prompt
    assert "<name>seo-review</name>" in prompt
    assert "Context SEO workflow skill" in prompt


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


def test_subagent_generation_tools_register_image_tool_by_default(tmp_path) -> None:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)

    tools = mgr._build_generation_tools()
    assert tools.get("generate_image") is not None
