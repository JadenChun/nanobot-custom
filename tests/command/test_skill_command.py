"""Tests for the /skill built-in command."""

from pathlib import Path
from shutil import copytree
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.skills import SkillsLoader
from nanobot.bus.events import InboundMessage
from nanobot.command.builtin import cmd_skill, register_builtin_commands
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.config.schema import ModelPresetConfig
from tests.agent.p6_admitted_fixtures import AgentInputs, AsyncAdmittedFactory
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async


def _provider(default_model: str = "test-model") -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = default_model
    provider.generation = MagicMock()
    provider.generation.max_tokens = 4096
    provider.generation.temperature = 0.1
    provider.generation.reasoning_effort = None
    return provider


async def _make_loop(tmp_path: Path, admitted_async: AsyncAdmittedFactory) -> AgentLoop:
    def build(inputs: AgentInputs) -> AgentLoop:
        prepared_skills = tmp_path / "skills"
        if prepared_skills.exists():
            copytree(prepared_skills, inputs.config.workspace_path / "skills", dirs_exist_ok=True)
        return AgentLoop(
            bus=inputs.bus,
            provider=_provider(),
            workspace=inputs.config.workspace_path,
            model="test-model",
            context_window_tokens=8000,
            model_presets={
                "default": ModelPresetConfig(
                    model="test-model",
                    max_tokens=4096,
                    context_window_tokens=8000,
                ),
            },
            session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    return await admitted_async(tmp_path, build)


def _ctx(loop: AgentLoop, raw: str = "/skill") -> CommandContext:
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content=raw)
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, args="", loop=loop)


def _write_skill(base: Path, name: str, *, description: str = "", body: str = "# Skill\n") -> None:
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = f"---\nname: {name}\n"
    if description:
        frontmatter += f"description: {description}\n"
    frontmatter += "---\n\n"
    (skill_dir / "SKILL.md").write_text(frontmatter + body, encoding="utf-8")


async def _loop_with_skills(tmp_path: Path, admitted_async: AsyncAdmittedFactory) -> AgentLoop:
    """Create a loop with an empty builtin dir so only workspace skills appear."""
    loop = await _make_loop(tmp_path, admitted_async)
    empty_builtin = tmp_path / "empty_builtin"
    empty_builtin.mkdir()
    loop.context.skills = SkillsLoader(loop.workspace, builtin_skills_dir=empty_builtin)
    return loop


@pytest.mark.asyncio
async def test_skill_command_no_skills(tmp_path: Path, admitted_async) -> None:
    loop = await _loop_with_skills(tmp_path, admitted_async)
    out = await cmd_skill(_ctx(loop))
    assert out.content == "No skills available."


@pytest.mark.asyncio
async def test_skill_command_lists_names_and_descriptions(tmp_path: Path, admitted_async) -> None:
    ws_skills = tmp_path / "skills"
    ws_skills.mkdir()
    _write_skill(ws_skills, "weather", description="Get current weather and forecasts")
    _write_skill(ws_skills, "cron", description="Schedule recurring tasks")

    loop = await _loop_with_skills(tmp_path, admitted_async)
    out = await cmd_skill(_ctx(loop))

    assert "Available skills (2):" in out.content
    assert "**weather** — Get current weather and forecasts" in out.content
    assert "**cron** — Schedule recurring tasks" in out.content
    # Must NOT contain file paths
    assert ".md" not in out.content
    assert "/skills/" not in out.content


@pytest.mark.asyncio
async def test_skill_command_excludes_disabled(tmp_path: Path, admitted_async) -> None:
    ws_skills = tmp_path / "skills"
    ws_skills.mkdir()
    _write_skill(ws_skills, "alpha", description="Alpha skill")
    _write_skill(ws_skills, "beta", description="Beta skill")

    loop = await _make_loop(tmp_path, admitted_async)
    loop.context.skills.disabled_skills = {"alpha"}

    out = await cmd_skill(_ctx(loop))

    assert "alpha" not in out.content
    assert "**beta** — Beta skill" in out.content


@pytest.mark.asyncio
async def test_skill_command_fallback_description(tmp_path: Path, admitted_async) -> None:
    ws_skills = tmp_path / "skills"
    ws_skills.mkdir()
    _write_skill(ws_skills, "plain", description="", body="# Plain Skill\n")

    loop = await _loop_with_skills(tmp_path, admitted_async)
    out = await cmd_skill(_ctx(loop))

    assert "**plain** — plain" in out.content


@pytest.mark.asyncio
async def test_skill_command_no_render_as_text(tmp_path: Path, admitted_async) -> None:
    """Output is markdown; CLI should render it (not forced as plain text)."""
    loop = await _make_loop(tmp_path, admitted_async)
    out = await cmd_skill(_ctx(loop))
    assert out.metadata.get("render_as") != "text"


@pytest.mark.asyncio
async def test_skill_command_does_not_list_goal_runtime_protocol(tmp_path: Path, admitted_async) -> None:
    loop = await _make_loop(tmp_path, admitted_async)
    out = await cmd_skill(_ctx(loop))

    assert "long-goal" not in out.content


@pytest.mark.asyncio
async def test_skill_command_registered_on_router(tmp_path: Path, admitted_async) -> None:
    router = CommandRouter()
    register_builtin_commands(router)
    loop = await _loop_with_skills(tmp_path, admitted_async)

    out = await router.dispatch(_ctx(loop, "/skill"))

    assert out is not None
    assert "No skills available." in out.content


def test_skill_command_unadmitted_loop_refuses_before_effects(tmp_path, monkeypatch) -> None:
    from nanobot.agent.memory_recovery_startup_admission import StartupReceiptAdmissionError
    from nanobot.bus.queue import MessageBus
    from nanobot.session.manager import SessionManager

    provider = _provider()
    bus = MessageBus()
    workspace = tmp_path / "unadmitted"
    pending = workspace / "memory" / ".consolidation.pending.json"
    pending.parent.mkdir(parents=True)
    pending.write_text("{broken", encoding="utf-8")

    def denied(*_args, **_kwargs):
        pytest.fail("unadmitted construction reached a session or directory effect")

    monkeypatch.setattr(SessionManager, "__init__", denied)
    monkeypatch.setattr(Path, "mkdir", denied)
    before = tuple(tmp_path.iterdir())
    with pytest.raises(StartupReceiptAdmissionError):
        AgentLoop(
            bus=bus, provider=provider, workspace=workspace,
            sessions_root=tmp_path / "sessions",
        )
    assert tuple(tmp_path.iterdir()) == before
    provider.chat.assert_not_called()
    provider.chat_with_retry.assert_not_called()
    assert not bus._pending


@pytest.mark.asyncio
async def test_skill_command_owned_workspace_listing_has_no_optional_activity(
    tmp_path, admitted_async,
) -> None:
    from nanobot.config.loader import load_config

    prepared = tmp_path / "skills"
    _write_skill(prepared, "owned", description="Owned fixture skill")
    original = (prepared / "owned" / "SKILL.md").read_bytes()
    loop = await _loop_with_skills(tmp_path, admitted_async)
    copied = loop.workspace / "skills" / "owned" / "SKILL.md"
    config = load_config()
    assert config.workspace_path == loop.workspace
    assert loop.context.skills.workspace == loop.workspace
    assert loop.context.skills.workspace_skills == loop.workspace / "skills"
    assert copied.read_bytes() == original
    assert loop.sessions.has_managed_authority
    assert loop.turn_delivery_factory.bus is loop.bus
    assert loop.tools_config.restrict_to_workspace is True
    assert loop.auto_compact is None
    assert not loop._background_tasks
    assert not loop.cron_service._running

    router = CommandRouter()
    register_builtin_commands(router)
    out = await router.dispatch(_ctx(loop))

    assert out is not None
    assert out.content == "Available skills (1):\n\n- **owned** — Owned fixture skill"
    assert str(loop.workspace) not in out.content
    assert "/skills/" not in out.content
    assert copied.read_bytes() == original
    assert (prepared / "owned" / "SKILL.md").read_bytes() == original
    loop.provider.chat.assert_not_called()
    loop.provider.chat_with_retry.assert_not_called()
    assert not loop._background_tasks
    assert not loop.cron_service._running
