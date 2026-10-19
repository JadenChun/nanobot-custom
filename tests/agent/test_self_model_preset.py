from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.runtime_control import AgentRuntimeControl
from nanobot.agent.tools.self import MyTool
from nanobot.config.schema import ModelPresetConfig
from nanobot.providers.factory import ProviderSnapshot
from nanobot.session.model_selection import model_preset_from_metadata
from tests.agent.p6_admitted_fixtures import (
    AgentInputs,
)
from tests.agent.p6_admitted_fixtures import (
    admitted_sync as admitted_sync,
)


def _provider(default_model: str, max_tokens: int = 123) -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = default_model
    provider.generation = SimpleNamespace(
        max_tokens=max_tokens, temperature=0.1, reasoning_effort=None
    )
    return provider


def _make_loop(admitted_sync, tmp_path, presets=None, active_preset=None):
    provider = _provider("base-model")
    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus,
            provider=provider,
            workspace=inputs.config.workspace_path,
            model="base-model",
            context_window_tokens=1000,
            model_presets=presets or {},
            model_preset=active_preset,
            session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    return admitted_sync(tmp_path, build, keys=("cli:one", "cli:two"))


def _my_tool(loop: AgentLoop) -> MyTool:
    return MyTool(
        runtime_control=AgentRuntimeControl(loop),
        modify_allowed=True,
    )


def test_model_preset_getter_none_when_not_set(admitted_sync, tmp_path) -> None:
    loop = _make_loop(admitted_sync, tmp_path)
    assert loop.model_preset is None


def test_model_preset_setter_updates_state(admitted_sync, tmp_path) -> None:
    presets = {
        "fast": ModelPresetConfig(
            model="openai/gpt-4.1",
            provider="openai",
            max_tokens=4096,
            context_window_tokens=32_768,
            temperature=0.5,
            reasoning_effort="low",
        )
    }
    loop = _make_loop(admitted_sync, tmp_path, presets=presets)
    loop.model_preset = "fast"

    assert loop.model_preset == "fast"
    assert loop.model == "openai/gpt-4.1"
    assert loop.context_window_tokens == 32_768
    runtime = loop.llm_runtime()
    assert runtime.generation.temperature == 0.5
    assert runtime.generation.max_tokens == 4096
    assert runtime.generation.reasoning_effort == "low"
    assert loop.llm_runtime().model == "openai/gpt-4.1"
    assert loop.llm_runtime().context_window_tokens == 32_768
    assert loop.llm_runtime().generation.max_tokens == 4096


def test_model_preset_setter_calls_runtime_model_publisher(admitted_sync, tmp_path) -> None:
    published: list[tuple[str, str | None]] = []
    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus,
            provider=_provider("base-model", max_tokens=123),
            workspace=inputs.config.workspace_path,
            model="base-model",
            context_window_tokens=1000,
            model_presets={"fast": ModelPresetConfig(model="openai/gpt-4.1")},
            runtime_model_publisher=lambda model, preset: published.append((model, preset)),
            session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    loop = admitted_sync(tmp_path, build)

    loop.set_model_preset("fast")

    assert published == [("openai/gpt-4.1", "fast")]


def test_model_preset_setter_replaces_provider_from_snapshot(admitted_sync, tmp_path) -> None:
    old_provider = _provider("base-model", max_tokens=123)
    new_provider = _provider("anthropic/claude-opus-4-5", max_tokens=2048)
    preset = ModelPresetConfig(
        model="anthropic/claude-opus-4-5",
        provider="anthropic",
        max_tokens=2048,
        context_window_tokens=200_000,
    )
    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus,
            provider=old_provider,
            workspace=inputs.config.workspace_path,
            model="base-model",
            context_window_tokens=1000,
            model_presets={"deep": preset},
            preset_snapshot_loader=lambda name: ProviderSnapshot(
                provider=new_provider,
                model=preset.model,
                context_window_tokens=preset.context_window_tokens,
                signature=(name, preset.model),
            ),
            session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    loop = admitted_sync(tmp_path, build)

    loop.set_model_preset("deep")

    assert loop.provider is new_provider
    assert loop.model == "anthropic/claude-opus-4-5"
    assert loop.context_window_tokens == 200_000
    assert loop.llm_runtime().generation.max_tokens == 2048


def test_model_preset_setter_failure_leaves_old_state(admitted_sync, tmp_path) -> None:
    preset = ModelPresetConfig(model="openai/gpt-4.1", max_tokens=4096)
    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus,
            provider=_provider("base-model", max_tokens=123),
            workspace=inputs.config.workspace_path,
            model="base-model",
            context_window_tokens=1000,
            model_presets={"fast": preset},
            preset_snapshot_loader=lambda _name: (_ for _ in ()).throw(
                RuntimeError("provider unavailable")
            ),
            session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    loop = admitted_sync(tmp_path, build)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        loop.set_model_preset("fast")

    assert loop.model_preset is None
    assert loop.model == "base-model"
    assert loop.context_window_tokens == 1000
    assert loop.llm_runtime().generation.max_tokens == 123


def test_active_model_preset_survives_unchanged_config_refresh(admitted_sync, tmp_path) -> None:
    base_provider = _provider("base-model", max_tokens=123)
    fast_provider = _provider("openai/gpt-4.1", max_tokens=4096)
    default_snapshot = ProviderSnapshot(
        provider=base_provider,
        model="base-model",
        context_window_tokens=1000,
        signature=("base-model", "auto", "openai", "sk-old"),
    )
    fast_snapshot = ProviderSnapshot(
        provider=fast_provider,
        model="openai/gpt-4.1",
        context_window_tokens=32_768,
        signature=("openai/gpt-4.1", "auto", "openai", "sk-old"),
    )
    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus,
            provider=base_provider,
            workspace=inputs.config.workspace_path,
            model="base-model",
            context_window_tokens=1000,
            provider_signature=default_snapshot.signature,
            model_presets={"fast": ModelPresetConfig(model="openai/gpt-4.1")},
            provider_snapshot_loader=lambda: default_snapshot,
            preset_snapshot_loader=lambda _name: fast_snapshot,
            session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    loop = admitted_sync(tmp_path, build)

    loop.set_model_preset("fast")
    loop.runtime_resolver.invalidate()
    loop.llm_runtime()

    assert loop.model_preset == "fast"
    assert loop.provider is fast_provider
    assert loop.model == "openai/gpt-4.1"


def test_config_model_refresh_clears_active_model_preset(admitted_sync, tmp_path) -> None:
    base_provider = _provider("base-model", max_tokens=123)
    fast_provider = _provider("openai/gpt-4.1", max_tokens=4096)
    webui_provider = _provider("anthropic/claude-opus-4-5", max_tokens=2048)
    webui_snapshot = ProviderSnapshot(
        provider=webui_provider,
        model="anthropic/claude-opus-4-5",
        context_window_tokens=200_000,
        signature=("anthropic/claude-opus-4-5", "anthropic", "anthropic", "sk-old"),
    )
    fast_snapshot = ProviderSnapshot(
        provider=fast_provider,
        model="openai/gpt-4.1",
        context_window_tokens=32_768,
        signature=("openai/gpt-4.1", "auto", "openai", "sk-old"),
    )
    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus,
            provider=base_provider,
            workspace=inputs.config.workspace_path,
            model="base-model",
            context_window_tokens=1000,
            provider_snapshot_loader=lambda: webui_snapshot,
            provider_signature=("base-model", "auto", "openai", "sk-old"),
            model_presets={"fast": ModelPresetConfig(model="openai/gpt-4.1")},
            preset_snapshot_loader=lambda _name: fast_snapshot,
            session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    loop = admitted_sync(tmp_path, build)

    loop.set_model_preset("fast")
    loop.runtime_resolver.invalidate()
    loop.llm_runtime()

    assert loop.model_preset is None
    assert loop.provider is webui_provider
    assert loop.model == "anthropic/claude-opus-4-5"
    assert loop.context_window_tokens == 200_000


def test_model_preset_setter_raises_on_unknown(admitted_sync, tmp_path) -> None:
    loop = _make_loop(admitted_sync, tmp_path)
    with pytest.raises(KeyError, match="model_preset 'missing' not found"):
        loop.model_preset = "missing"


def test_model_preset_setter_raises_on_empty_string(admitted_sync, tmp_path) -> None:
    loop = _make_loop(admitted_sync, tmp_path)
    with pytest.raises(ValueError, match="model_preset must be a non-empty string"):
        loop.model_preset = ""


def test_self_tool_inspect_shows_model_preset(admitted_sync, tmp_path) -> None:
    presets = {
        "fast": ModelPresetConfig(model="openai/gpt-4.1"),
    }
    loop = _make_loop(admitted_sync, tmp_path, presets=presets, active_preset="fast")
    tool = _my_tool(loop)
    output = tool._inspect_all()
    assert "model_preset: 'fast'" in output


def test_self_tool_set_model_preset_via_modify(admitted_sync, tmp_path) -> None:
    presets = {
        "fast": ModelPresetConfig(model="openai/gpt-4.1"),
    }
    loop = _make_loop(admitted_sync, tmp_path, presets=presets)
    tool = _my_tool(loop)
    result = tool._modify("model_preset", "fast")
    assert "Error" not in result
    assert loop.model_preset == "fast"
    assert loop.model == "openai/gpt-4.1"


def test_self_tool_set_model_preset_switches_back_to_default(admitted_sync, tmp_path) -> None:
    presets = {
        "default": ModelPresetConfig(model="base-model", context_window_tokens=1000),
        "fast": ModelPresetConfig(model="openai/gpt-4.1", context_window_tokens=32_768),
    }
    loop = _make_loop(admitted_sync, tmp_path, presets=presets, active_preset="fast")
    tool = _my_tool(loop)

    result = tool._modify("model_preset", "default")

    assert "Error" not in result
    assert "model is now 'base-model'" in result
    assert loop.model_preset == "default"
    assert loop.model == "base-model"
    assert loop.context_window_tokens == 1000


def test_self_tool_set_model_preset_unknown_lists_available(admitted_sync, tmp_path) -> None:
    presets = {
        "default": ModelPresetConfig(model="base-model"),
        "fast": ModelPresetConfig(model="openai/gpt-4.1"),
    }
    loop = _make_loop(admitted_sync, tmp_path, presets=presets)
    tool = _my_tool(loop)

    result = tool._modify("model_preset", "missing")

    assert result == "Error: model_preset 'missing' not found. Available: default, fast."
    assert loop.model_preset is None
    assert loop.model == "base-model"


def test_self_tool_sets_model_preset_for_current_session(admitted_sync, tmp_path) -> None:
    presets = {
        "default": ModelPresetConfig(model="base-model"),
        "fast": ModelPresetConfig(model="openai/gpt-4.1"),
    }
    loop = _make_loop(admitted_sync, tmp_path, presets=presets)
    tool = _my_tool(loop)

    with request_context(RequestContext(
        channel="cli",
        chat_id="one",
        session_key="cli:one",
        metadata={"source": "self-tool"},
    )):
        result = tool._modify("model_preset", "fast")

    assert "for the next turn" in result
    assert model_preset_from_metadata(
        loop.sessions.get_or_create("cli:one").metadata
    ) == "fast"
    assert loop.model_preset is None
    assert loop.model == "base-model"


def test_self_tool_reports_session_preset_provider_configuration_error(admitted_sync, tmp_path) -> None:
    loop = _make_loop(admitted_sync, tmp_path)
    loop.set_session_model_preset = MagicMock(
        side_effect=ValueError("No API key configured for provider 'openai'.")
    )
    tool = _my_tool(loop)

    with request_context(RequestContext(
        channel="cli",
        chat_id="one",
        session_key="cli:one",
    )):
        result = tool._modify("model_preset", "broken")

    assert result == "Error: No API key configured for provider 'openai'."


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model", "other-model"),
        ("context_window_tokens", 8_192),
    ],
)
def test_self_tool_rejects_instance_runtime_changes_in_session(admitted_sync,
    tmp_path,
    key: str,
    value: object,
) -> None:
    loop = _make_loop(admitted_sync, tmp_path)
    tool = _my_tool(loop)
    session = loop.sessions.get_or_create("cli:one")

    with request_context(RequestContext(
        channel="cli",
        chat_id="one",
        session_key=session.key,
        runtime=loop.runtime_for_session(session),
    )):
        result = tool._modify(key, value)

    other_runtime = loop.runtime_for_session(loop.sessions.get_or_create("cli:two"))
    assert "instance-wide and disabled" in result
    assert "model_preset" in result
    assert other_runtime.model == "base-model"
    assert other_runtime.context_window_tokens == 1000


def test_self_tool_set_model_clears_active_preset(admitted_sync, tmp_path) -> None:
    presets = {
        "fast": ModelPresetConfig(model="openai/gpt-4.1"),
    }
    loop = _make_loop(admitted_sync, tmp_path, presets=presets, active_preset="fast")
    tool = _my_tool(loop)
    result = tool._modify("model", "anthropic/claude-opus-4-5")
    assert "Error" not in result
    assert loop.model_preset is None
    assert loop.model == "anthropic/claude-opus-4-5"


def test_from_config_injects_default_preset(admitted_sync, tmp_path) -> None:
    from unittest.mock import patch

    from nanobot.config.schema import Config
    config = Config.model_validate({
        "agents": {"defaults": {"model": "openai/gpt-4.1", "workspace": str(tmp_path)}},
    })
    fake_provider = _provider("openai/gpt-4.1")
    with patch("nanobot.providers.factory.make_provider", return_value=fake_provider):
        def build(inputs: AgentInputs) -> AgentLoop:
            return AgentLoop.from_config(
                inputs.config, inputs.bus, tool_registry=inputs.tools,
                session_manager=inputs.sessions, cron_service=inputs.cron,
                turn_delivery_factory=inputs.delivery, enable_legacy_auto_compact=False,
            )

        loop = admitted_sync(tmp_path, build, config=config)
    assert loop.model == "openai/gpt-4.1"
    assert loop.model_preset is None
    assert "default" in loop.model_presets
    assert loop.model_presets["default"].model == "openai/gpt-4.1"


def test_from_config_static_preset_loader_does_not_enable_hot_reload(admitted_sync, tmp_path) -> None:
    from unittest.mock import patch

    from nanobot.config.schema import Config
    config = Config.model_validate({
        "agents": {"defaults": {"model": "openai/gpt-4.1", "workspace": str(tmp_path)}},
        "model_presets": {"fast": {"model": "openai/gpt-4.1-mini"}},
    })
    fake_provider = _provider("openai/gpt-4.1")
    with patch("nanobot.providers.factory.make_provider", return_value=fake_provider):
        def build(inputs: AgentInputs) -> AgentLoop:
            return AgentLoop.from_config(
                inputs.config, inputs.bus, tool_registry=inputs.tools,
                session_manager=inputs.sessions, cron_service=inputs.cron,
                turn_delivery_factory=inputs.delivery, enable_legacy_auto_compact=False,
            )

        loop = admitted_sync(tmp_path, build, config=config)
        default_runtime = loop.runtime_resolver.runtime
        resolved = loop.runtime_resolver.resolve_preset("fast")
    assert resolved.model == "openai/gpt-4.1-mini"
    assert loop.runtime_resolver.runtime is default_runtime


def test_admitted_fixture_failed_acquire_releases_writer_and_restores_patches(tmp_path) -> None:
    import asyncio
    from pathlib import Path

    from nanobot.config import loader
    from nanobot.migration_guard import require_runtime_ready
    from nanobot.runtime_composition import RuntimeState
    from tests.agent.p6_admitted_fixtures import AdmittedGraphs

    original_guard = require_runtime_ready
    original_home = Path.home()
    original_config_path = loader._current_config_path
    captured: list[AgentInputs] = []

    def failed_build(inputs: AgentInputs) -> AgentLoop:
        captured.append(inputs)
        raise RuntimeError("intentional admitted builder failure")

    with pytest.MonkeyPatch.context() as patches:
        graphs = AdmittedGraphs(tmp_path, patches)
        with asyncio.Runner() as runner:
            owner_loop = runner.get_loop()
            with pytest.raises(RuntimeError, match="intentional admitted builder failure"):
                runner.run(graphs.acquire(tmp_path, failed_build, keys=("cli:one",)))
            graph = graphs.graphs[0][0]
            assert graph.state is RuntimeState.FAILED
            assert graph._owner_loop is owner_loop
            assert captured[0].mcp._closing
            assert captured[0].mcp.connected_server_names == set()
            # Includes a fresh durable authority acquire/close, proving rollback released the writer.
            runner.run(graphs.close())
        assert owner_loop.is_closed()
    assert loader._current_config_path == original_config_path
    assert Path.home() == original_home
    from nanobot import migration_guard
    assert migration_guard.require_runtime_ready is original_guard


def test_admitted_fixture_sync_acquire_close_retains_owner_loop(admitted_sync, tmp_path) -> None:
    import asyncio

    from nanobot.runtime_composition import RuntimeState

    loop = _make_loop(admitted_sync, tmp_path, presets={"fast": ModelPresetConfig(model="fast-model")})
    graph = admitted_sync.graphs.graphs[0][0]
    owner_loop = admitted_sync.runner.get_loop()
    assert graph._owner_loop is owner_loop
    # Setters and synchronous publisher callbacks must run with no active event loop.
    with pytest.raises(RuntimeError, match="no running event loop"):
        asyncio.get_running_loop()
    loop.set_model_preset("fast")
    assert not loop.bus._pending
    admitted_sync.runner.run(admitted_sync.graphs.close())
    assert graph.state is RuntimeState.CLOSED
    assert loop._closed
    assert graph._owner_loop is owner_loop
