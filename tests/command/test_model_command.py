from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.goal_permission import goal_mutation_allowed
from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.command.builtin import (
    build_help_text,
    builtin_command_palette,
    cmd_goal,
    cmd_model,
    register_builtin_commands,
)
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.config.schema import ModelPresetConfig
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.session.model_selection import (
    SESSION_MODEL_PRESET_METADATA_KEY,
    model_preset_from_metadata,
)
from tests.agent.p6_admitted_fixtures import AgentInputs
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async


def _provider(default_model: str, max_tokens: int = 123) -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = default_model
    provider.generation = GenerationSettings(
        max_tokens=max_tokens,
        temperature=0.1,
        reasoning_effort=None,
    )
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False
    return provider


async def _make_loop(tmp_path, admitted_async, *, preset_snapshot_loader=None,
                     model_presets=None, keys=(), context_window_tokens=1000) -> AgentLoop:
    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus,
            provider=_provider("base-model", max_tokens=123),
            workspace=inputs.config.workspace_path,
            model="base-model",
            context_window_tokens=context_window_tokens,
            model_presets=model_presets or {
                "default": ModelPresetConfig(
                    model="base-model",
                    max_tokens=123,
                    context_window_tokens=context_window_tokens,
                ),
                "fast": ModelPresetConfig(
                    model="openai/gpt-4.1",
                    max_tokens=4096,
                    context_window_tokens=32_768,
                ),
            },
            preset_snapshot_loader=preset_snapshot_loader,
            session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    return await admitted_async(tmp_path, build, keys=keys)


def _ctx(loop: AgentLoop, raw: str, args: str = "") -> CommandContext:
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content=raw)
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, args=args, loop=loop)


def _ctx_session(loop: AgentLoop, raw: str, args: str = "") -> CommandContext:
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content=raw)
    return CommandContext(
        msg=msg, session=MagicMock(), key=msg.session_key, raw=raw, args=args, loop=loop,
        is_user_turn=True,
    )


def _saved_model_preset(loop: AgentLoop, session_key: str = "cli:direct") -> str | None:
    session = loop.sessions.get_or_create(session_key)
    return model_preset_from_metadata(session.metadata)


@pytest.mark.asyncio
async def test_model_command_lists_current_and_available_presets(tmp_path, admitted_async) -> None:
    loop = await _make_loop(tmp_path, admitted_async)

    out = await cmd_model(_ctx(loop, "/model"))

    assert "Current model: `base-model`" in out.content
    assert "Current preset: `default`" in out.content
    assert "Available presets: `default`, `fast`" in out.content
    assert "`fast`" in out.content
    assert out.metadata == {"render_as": "text"}


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_model_command_switches_preset(tmp_path, admitted_async, legacy) -> None:
    loop = await _make_loop(tmp_path, admitted_async, keys=("cli:direct",) if legacy else ())

    out = await cmd_model(_ctx(loop, "/model fast", args="fast"))

    assert "Switched model preset to `fast`." in out.content
    assert "Scope: current session" in out.content
    assert "Model: `openai/gpt-4.1`" in out.content
    assert _saved_model_preset(loop) == "fast"
    assert loop.model_preset is None
    assert loop.model == "base-model"

    await loop.process_direct("/new", session_key="cli:direct")
    assert _saved_model_preset(loop) == "fast"
    status = await loop.process_direct("/status", session_key="cli:direct")
    assert status is not None and "openai/gpt-4.1" in status.content


@pytest.mark.asyncio
async def test_model_command_accepts_canonical_names_with_spaces(tmp_path, admitted_async) -> None:
    loop = await _make_loop(
        tmp_path, admitted_async,
        model_presets={
            "default": ModelPresetConfig(model="base-model"),
            "Deep Research": ModelPresetConfig(model="deep-model"),
        },
    )

    out = await cmd_model(
        _ctx(loop, "/model deep research", args="deep research"),
    )

    assert "Switched model preset to `Deep Research`." in out.content
    assert _saved_model_preset(loop) == "Deep Research"


@pytest.mark.asyncio
async def test_model_command_switches_back_to_default(tmp_path, admitted_async) -> None:
    loop = await _make_loop(tmp_path, admitted_async)
    await cmd_model(_ctx(loop, "/model fast", args="fast"))

    out = await cmd_model(_ctx(loop, "/model default", args="default"))

    assert "Switched model preset to `default`." in out.content
    assert _saved_model_preset(loop) == "default"
    assert loop.model_preset is None
    assert loop.model == "base-model"
    assert loop.context_window_tokens == 1000


@pytest.mark.asyncio
async def test_model_command_unknown_preset_keeps_old_state(tmp_path, admitted_async) -> None:
    loop = await _make_loop(tmp_path, admitted_async)

    out = await cmd_model(_ctx(loop, "/model missing", args="missing"))

    assert "Could not switch model preset" in out.content
    assert "\"model_preset" not in out.content
    assert "Available presets: `default`, `fast`" in out.content
    assert loop.model_preset is None
    assert loop.model == "base-model"


@pytest.mark.asyncio
async def test_model_command_reports_provider_configuration_errors(tmp_path, admitted_async) -> None:
    def fail_preset(_name: str):
        raise ValueError("No API key configured for provider 'openai'.")

    loop = await _make_loop(tmp_path, admitted_async, preset_snapshot_loader=fail_preset)

    switched = await cmd_model(_ctx(loop, "/model fast", args="fast"))
    session = loop.sessions.get_or_create("cli:direct")
    session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = "fast"
    status = await cmd_model(_ctx(loop, "/model"))

    assert "Could not switch model preset" in switched.content
    assert "No API key configured for provider 'openai'." in switched.content
    assert "Current selection error" in status.content
    assert "No API key configured for provider 'openai'." in status.content


@pytest.mark.asyncio
async def test_model_command_does_not_depend_on_my_allow_set(tmp_path, admitted_async) -> None:
    loop = await _make_loop(tmp_path, admitted_async)
    assert loop.tools_config.my.allow_set is False

    await cmd_model(_ctx(loop, "/model fast", args="fast"))

    assert _saved_model_preset(loop) == "fast"


@pytest.mark.asyncio
async def test_model_command_registered_as_exact_and_prefix(tmp_path, admitted_async) -> None:
    router = CommandRouter()
    register_builtin_commands(router)
    loop = await _make_loop(tmp_path, admitted_async)

    out = await router.dispatch(_ctx(loop, "/model fast"))

    assert out is not None
    assert out.channel == "cli"
    assert out.chat_id == "direct"
    assert out.metadata == {"render_as": "text"}
    assert out.content == "\n".join([
        "Switched model preset to `fast`.",
        "- Scope: current session",
        "- Model: `openai/gpt-4.1`",
        "- Context window: 32768",
        "- Max output tokens: 4096",
    ])
    assert _saved_model_preset(loop) == "fast"


@pytest.mark.asyncio
async def test_model_command_does_not_change_another_session(tmp_path, admitted_async) -> None:
    loop = await _make_loop(tmp_path, admitted_async)

    await cmd_model(_ctx(loop, "/model fast", args="fast"))
    other = InboundMessage(channel="cli", sender_id="user", chat_id="other", content="/model")
    out = await cmd_model(
        CommandContext(msg=other, session=None, key=other.session_key, raw="/model", loop=loop)
    )

    assert "Current preset: `default`" in out.content
    assert _saved_model_preset(loop) == "fast"


@pytest.mark.asyncio
async def test_model_command_reports_and_recovers_removed_session_preset(tmp_path, admitted_async) -> None:
    loop = await _make_loop(tmp_path, admitted_async)
    assert loop.sessions.upsert_session_metadata(
        "cli:direct", {SESSION_MODEL_PRESET_METADATA_KEY: "removed"}, create_if_absent=True,
    )

    status = await loop.process_direct("/model", session_key="cli:direct")
    switched = await loop.process_direct("/model default", session_key="cli:direct")

    assert status is not None
    assert "model_preset 'removed' not found" in status.content
    assert "Available presets: `default`, `fast`" in status.content
    assert "Switch with `/model <preset>`" in status.content
    assert switched is not None
    assert "Switched model preset to `default`." in switched.content
    assert _saved_model_preset(loop) == "default"


def test_model_command_in_help_and_palette() -> None:
    palette = builtin_command_palette()

    model = next(item for item in palette if item["command"] == "/model")
    assert model["arg_hint"] == "[preset]"
    assert model["lifecycle"] == "side_channel"
    assert model["accepts_args"] is True
    assert "/model [preset]" in build_help_text()


@pytest.mark.asyncio
async def test_goal_command_shows_usage_without_args(tmp_path, admitted_async) -> None:
    loop = await _make_loop(tmp_path, admitted_async)
    out = await cmd_goal(_ctx(loop, "/goal"))
    assert out is not None
    assert out.channel == "cli"
    assert out.chat_id == "direct"
    assert out.metadata == {"render_as": "text"}
    assert out.content == "Usage: /goal <long-running task description>"


@pytest.mark.asyncio
async def test_goal_command_rejects_mid_turn_without_session(tmp_path, admitted_async) -> None:
    loop = await _make_loop(tmp_path, admitted_async)
    out = await cmd_goal(_ctx(loop, "/goal do work", args="do work"))
    assert out is not None
    assert out.channel == "cli"
    assert out.chat_id == "direct"
    assert out.metadata == {"render_as": "text"}
    assert out.content == (
        "A task is already running for this chat. "
        "Use `/stop` first, then send `/goal <long-running task description>` again."
    )


@pytest.mark.asyncio
async def test_goal_command_marks_turn_and_preserves_explicit_request(tmp_path, admitted_async) -> None:
    loop = await _make_loop(tmp_path, admitted_async)
    ctx = _ctx_session(loop, "/goal audit the repo", args="audit the repo")
    out = await cmd_goal(ctx)
    assert out is None
    assert ctx.msg.content == "/goal audit the repo"
    assert ctx.msg.metadata.get("original_command") == "/goal"
    assert ctx.msg.metadata.get("original_content") == "/goal audit the repo"
    assert ctx.msg.metadata.get("goal_requested") is True
    assert isinstance(ctx.msg.metadata.get("goal_started_at"), int | float)
    assert len(ctx.turn_scopes) == 1
    with ctx.turn_scopes[0]:
        assert goal_mutation_allowed() is True
    assert goal_mutation_allowed() is False


@pytest.mark.asyncio
async def test_goal_command_registered_on_router(tmp_path, admitted_async) -> None:
    router = CommandRouter()
    register_builtin_commands(router)
    loop = await _make_loop(tmp_path, admitted_async)
    ctx = _ctx_session(loop, "/goal ship it", args="ship it")
    out = await router.dispatch(ctx)
    assert out is None
    assert "ship it" in ctx.msg.content
    assert len(ctx.turn_scopes) == 1
    with ctx.turn_scopes[0]:
        assert goal_mutation_allowed() is True
    assert goal_mutation_allowed() is False


@pytest.mark.asyncio
async def test_goal_command_does_not_allow_internal_turn(tmp_path, admitted_async) -> None:
    loop = await _make_loop(tmp_path, admitted_async)
    ctx = CommandContext(
        msg=InboundMessage(
            channel="cli",
            sender_id="system",
            chat_id="direct",
            content="/goal internal work",
        ),
        session=MagicMock(),
        key="cli:direct",
        raw="/goal internal work",
        args="internal work",
        loop=loop,
        is_user_turn=False,
    )

    out = await cmd_goal(ctx)

    assert out is not None
    assert "only be started by a user" in out.content
    assert ctx.turn_scopes == []


def test_goal_command_in_help_and_palette() -> None:
    palette = builtin_command_palette()
    goal = next(item for item in palette if item["command"] == "/goal")
    assert goal["arg_hint"] == "<goal>"
    assert goal["lifecycle"] == "agent_turn_with_args"
    assert goal["accepts_args"] is True
    assert "/goal <goal>" in build_help_text()


def test_model_command_rejects_pending_startup_receipt_before_effects(tmp_path) -> None:
    from nanobot.agent.memory_recovery_startup_admission import StartupReceiptAdmissionError

    pending = tmp_path / "memory" / ".consolidation.pending.json"
    pending.parent.mkdir(parents=True)
    pending.write_text("{broken", encoding="utf-8")

    with pytest.raises(StartupReceiptAdmissionError):
        AgentLoop(
            bus=MagicMock(), provider=_provider("base-model"), workspace=tmp_path,
            sessions_root=tmp_path / "sessions",
            model="base-model",
        )
    assert pending.read_text(encoding="utf-8") == "{broken"
    assert goal_mutation_allowed() is False


@pytest.mark.asyncio
async def test_model_command_managed_selection_persists_without_optional_activity(
    tmp_path, admitted_async,
) -> None:
    loop = await _make_loop(tmp_path, admitted_async)
    assert loop.sessions.has_managed_authority
    assert loop.turn_delivery_factory.bus is loop.bus
    assert loop.auto_compact is None
    assert not loop._background_tasks
    assert goal_mutation_allowed() is False

    out = await cmd_model(_ctx(loop, "/model fast", args="fast"))
    assert "Switched model preset to `fast`." in out.content
    assert loop.sessions.resolve_session_mode("cli:direct").mode == "canonical"
    loop.sessions.invalidate("cli:direct")
    assert _saved_model_preset(loop) == "fast"
    assert _saved_model_preset(loop, "cli:other") is None
    assert loop.model == "base-model"
    assert loop.model_preset is None
    loop.provider.chat_with_retry.assert_not_called()
    assert not loop._background_tasks
    assert goal_mutation_allowed() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_removed_model_preset_recovery_persists_metadata_only(tmp_path, admitted_async, legacy):
    key = "cli:direct"
    # Unlike command-only fixtures, this case runs a real turn to seed history.
    loop = await _make_loop(
        tmp_path, admitted_async, keys=(key,) if legacy else (), context_window_tokens=32768,
    )
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="saved reply"))
    reply = await loop.process_direct("keep this history", session_key=key)
    assert reply is not None and reply.content == "saved reply"
    assert loop.sessions.upsert_session_metadata(
        key, {SESSION_MODEL_PRESET_METADATA_KEY: "removed", "marker": "keep"},
        create_if_absent=True,
    )
    before = loop.sessions.read_session_snapshot(key)
    session = loop.sessions.get_or_create(key)
    runtime = loop.runtime_for_session(session)
    assert runtime.model == "base-model"
    assert SESSION_MODEL_PRESET_METADATA_KEY not in session.metadata
    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert SESSION_MODEL_PRESET_METADATA_KEY not in reloaded.metadata
    assert reloaded.metadata["marker"] == "keep"
    after = loop.sessions.read_session_snapshot(key)
    assert before is not None and after is not None
    assert before.messages
    assert before.messages == after.messages


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
async def test_model_selection_write_failure_preserves_cached_and_durable_value(
    tmp_path, admitted_async, monkeypatch, recover,
):
    key = "cli:direct"
    loop = await _make_loop(tmp_path, admitted_async)
    old = "removed" if recover else "default"
    assert loop.sessions.upsert_session_metadata(
        key, {SESSION_MODEL_PRESET_METADATA_KEY: old}, create_if_absent=True,
    )
    session = loop.sessions.get_or_create(key)
    backend = loop.sessions.source_backend(key)
    assert backend is not None

    def fail_write(*_args, **_kwargs):
        raise OSError("metadata commit failed")

    with monkeypatch.context() as patcher:
        # source is a property returning a fresh adapter on each access.
        patcher.setattr(type(backend.source), "update_metadata", fail_write)
        with pytest.raises(OSError, match="metadata commit failed"):
            if recover:
                loop.runtime_for_session(session)
            else:
                loop.set_session_model_preset(key, "fast")
    assert session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] == old
    assert loop.sessions.get_cached(key) is session
    loop.sessions.invalidate(key)
    assert _saved_model_preset(loop, key) == old


@pytest.mark.asyncio
async def test_temporary_model_selection_never_creates_durable_session(tmp_path, admitted_async):
    loop = await _make_loop(tmp_path, admitted_async)
    key = "cli:temporary"
    session = loop.sessions.get_or_create_transient(key, retain=True)
    runtime = loop.set_session_model_preset(key, "fast")
    assert runtime.model == "openai/gpt-4.1"
    assert session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] == "fast"
    session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = "removed"
    assert loop.runtime_for_session(session).model == "base-model"
    assert SESSION_MODEL_PRESET_METADATA_KEY not in session.metadata
    assert loop.sessions.resolve_session_mode(key).state == "absent"
    assert loop.sessions.read_session_file(key) is None
    assert loop.sessions.get_cached(key) is session
