import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.bus.runtime_events import RuntimeModelChanged
from nanobot.config.errors import ConfigLoadError
from nanobot.config.loader import save_config
from nanobot.config.schema import Config, ModelPresetConfig
from nanobot.providers.base import GenerationSettings
from nanobot.providers.factory import ProviderSnapshot, load_provider_snapshot
from nanobot.session.model_selection import (
    SESSION_MODEL_PRESET_METADATA_KEY,
    model_preset_from_metadata,
)
from nanobot.webui.settings_api import update_agent_settings
from tests.agent.p6_admitted_fixtures import (
    AgentInputs,
    AsyncAdmittedFactory,
    SyncAdmittedFactory,
)
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async
from tests.agent.p6_admitted_fixtures import admitted_sync as admitted_sync


def _provider(default_model: str, max_tokens: int = 123) -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = default_model
    provider.generation = SimpleNamespace(max_tokens=max_tokens)
    return provider


def _build_loop(inputs: AgentInputs, **options: object) -> AgentLoop:
    return AgentLoop(
        bus=inputs.bus, workspace=inputs.config.workspace_path,
        session_manager=inputs.sessions, tool_registry=inputs.tools,
        cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
        tools_config=inputs.config.tools, restrict_to_workspace=True,
        enable_legacy_auto_compact=False, **options,
    )


def _make_loop(
    tmp_path: Path, admitted: SyncAdmittedFactory, **options: object,
) -> AgentLoop:
    return admitted(tmp_path, lambda inputs: _build_loop(inputs, **options))


async def _make_async_loop(
    tmp_path: Path, admitted: AsyncAdmittedFactory, **options: object,
) -> AgentLoop:
    return await admitted(tmp_path, lambda inputs: _build_loop(inputs, **options))


def _synthetic_config_provider(
    config: Config, *, preset: ModelPresetConfig | None = None,
) -> MagicMock:
    resolved = preset if preset is not None else config.resolve_preset()
    provider = _provider(resolved.model, resolved.max_tokens)
    provider.generation = resolved.to_generation_settings()
    return provider


def _make_config_loop(
    tmp_path: Path, admitted: SyncAdmittedFactory, config: Config, snapshot_loader,
) -> tuple[AgentLoop, Path]:
    acquired_path = None

    def build(inputs: AgentInputs) -> AgentLoop:
        nonlocal acquired_path
        acquired_path = inputs.config._source_path
        return AgentLoop.from_config(
            inputs.config, bus=inputs.bus, tool_registry=inputs.tools,
            provider=_synthetic_config_provider(inputs.config),
            session_manager=inputs.sessions, cron_service=inputs.cron,
            turn_delivery_factory=inputs.delivery,
            provider_snapshot_loader=snapshot_loader,
            enable_legacy_auto_compact=False,
        )

    loop = admitted(tmp_path, build, config=config)
    assert isinstance(acquired_path, Path)
    return loop, acquired_path


def test_provider_refresh_updates_runtime_resolver(tmp_path: Path, admitted_sync) -> None:
    old_provider = _provider("old-model")
    new_provider = _provider("new-model", max_tokens=456)
    loop = _make_loop(
        tmp_path, admitted_sync,
        provider=old_provider,
        model="old-model",
        context_window_tokens=1000,
        provider_snapshot_loader=lambda: ProviderSnapshot(
            provider=new_provider,
            model="new-model",
            context_window_tokens=2000,
            signature=("new-model",),
        ),
    )
    loop.runtime_resolver.invalidate()

    runtime = loop.llm_runtime()

    assert runtime is loop.runtime_resolver.runtime
    assert loop.provider is new_provider
    assert loop.model == "new-model"
    assert loop.context_window_tokens == 2000


def test_llm_runtime_refreshes_provider_snapshot(tmp_path: Path, admitted_sync) -> None:
    old_provider = _provider("old-model")
    new_provider = _provider("new-model", max_tokens=456)
    loop = _make_loop(
        tmp_path, admitted_sync,
        provider=old_provider,
        model="old-model",
        context_window_tokens=1000,
        provider_snapshot_loader=lambda: ProviderSnapshot(
            provider=new_provider,
            model="new-model",
            context_window_tokens=2000,
            signature=("new-model",),
        ),
    )
    loop.runtime_resolver.invalidate()

    runtime = loop.llm_runtime()

    assert runtime.provider is new_provider
    assert runtime.model == "new-model"
    assert loop.provider is new_provider


def test_llm_runtime_surfaces_invalidated_config_errors(tmp_path: Path, admitted_sync) -> None:
    def fail_refresh() -> ProviderSnapshot:
        raise ValueError("invalid config")

    loop = _make_loop(
        tmp_path, admitted_sync,
        provider=_provider("old-model"),
        model="old-model",
        context_window_tokens=1000,
        provider_snapshot_loader=fail_refresh,
    )
    loop.runtime_resolver.invalidate()

    with pytest.raises(ValueError, match="invalid config"):
        loop.llm_runtime()


def test_provider_snapshot_missing_env_reports_explicit_config_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    name = "NANOBOT_TEST_REFRESH_MISSING_KEY"
    monkeypatch.delenv(name, raising=False)
    config_path = tmp_path / "custom.json"
    config_path.write_text(
        json.dumps({"providers": {"openrouter": {"apiKey": f"${{{name}}}"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError) as exc_info:
        load_provider_snapshot(config_path)

    assert exc_info.value.path == config_path


def test_same_snapshot_default_clears_preset_and_publishes_update(tmp_path: Path, admitted_sync) -> None:
    base_provider = _provider("base-model")
    fast_provider = _provider("fast-model")
    fast_snapshot = ProviderSnapshot(
        provider=fast_provider,
        model="fast-model",
        context_window_tokens=2000,
        signature=("fast-model", "auto", "same-runtime"),
    )
    published: list[tuple[str, str | None]] = []
    loop = _make_loop(
        tmp_path, admitted_sync,
        provider=base_provider,
        model="base-model",
        context_window_tokens=1000,
        provider_signature=("base-model", "auto", "initial"),
        provider_snapshot_loader=lambda: fast_snapshot,
        model_presets={"fast": ModelPresetConfig(model="fast-model")},
        model_preset="fast",
        preset_snapshot_loader=lambda _name: fast_snapshot,
        runtime_model_publisher=lambda model, preset: published.append((model, preset)),
    )
    loop.runtime_resolver.invalidate()

    runtime = loop.llm_runtime()

    assert runtime.model_preset is None
    assert loop.model_preset is None
    assert published == [("fast-model", None)]


def test_named_default_refresh_is_used_by_sessions_without_override(tmp_path: Path, admitted_sync) -> None:
    provider = _provider("shared-model")
    shared_signature = ("shared-model", "auto", "same-settings")
    snapshots = {
        "fast": ProviderSnapshot(
            provider=provider,
            model="shared-model",
            context_window_tokens=16_000,
            signature=shared_signature,
            model_preset="fast",
        ),
        "deep": ProviderSnapshot(
            provider=provider,
            model="shared-model",
            context_window_tokens=16_000,
            signature=shared_signature,
            model_preset="deep",
        ),
    }
    default_snapshot = snapshots["deep"]
    loop = _make_loop(
        tmp_path, admitted_sync,
        provider=provider,
        model="shared-model",
        context_window_tokens=16_000,
        provider_signature=shared_signature,
        provider_snapshot_loader=lambda: default_snapshot,
        model_presets={
            name: ModelPresetConfig(model=snapshot.model)
            for name, snapshot in snapshots.items()
        },
        model_preset="fast",
        preset_snapshot_loader=snapshots.__getitem__,
    )

    loop.runtime_resolver.invalidate()
    runtime = loop.llm_runtime()
    session = loop.sessions.get_or_create("sdk:new-after-refresh")

    assert runtime.model_preset == "deep"
    assert loop.runtime_for_session(session).model_preset == "deep"
    assert model_preset_from_metadata(session.metadata) is None


@pytest.mark.asyncio
async def test_config_invalidation_defers_canonical_notification_until_default_refresh(
    tmp_path: Path,
    admitted_async,
) -> None:
    provider = _provider("model-a")
    catalog = {"fast": ModelPresetConfig(model="model-a")}
    current_model = "model-a"
    published: list[RuntimeModelChanged] = []

    def load_preset(_name: str) -> ProviderSnapshot:
        return ProviderSnapshot(
            provider=provider,
            model=current_model,
            context_window_tokens=16_000,
            signature=(current_model, "auto"),
            model_preset="fast",
        )

    loop = await _make_async_loop(
        tmp_path, admitted_async,
        provider=provider,
        model="model-a",
        context_window_tokens=16_000,
        provider_signature=("model-a", "auto"),
        provider_snapshot_loader=lambda: load_preset("fast"),
        model_presets=catalog,
        preset_catalog_loader=lambda: catalog,
        model_preset="fast",
        preset_snapshot_loader=load_preset,
    )
    loop.bus.subscribe(published.append, RuntimeModelChanged)
    session = loop.sessions.get_or_create("websocket:chat")
    session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = "fast"
    current_model = "model-b"
    catalog["fast"] = ModelPresetConfig(model="model-b")

    loop.invalidate_runtime_config()
    runtime = loop.runtime_for_session(session)
    await asyncio.sleep(0)

    assert published == []
    assert runtime.model == "model-b"
    assert loop.model_presets["fast"].model == "model-b"

    assert loop.llm_runtime().model == "model-b"
    await asyncio.sleep(0)

    assert [(event.model, event.model_preset) for event in published] == [
        ("model-b", "fast"),
    ]


@pytest.mark.asyncio
async def test_config_refresh_publishes_renamed_canonical_preset(tmp_path: Path, admitted_async) -> None:
    provider = _provider("model-a")
    catalog = {"fast": ModelPresetConfig(model="model-a")}
    default_name = "fast"
    published: list[RuntimeModelChanged] = []

    def load_preset(name: str) -> ProviderSnapshot:
        return ProviderSnapshot(
            provider=provider,
            model=catalog[name].model,
            context_window_tokens=16_000,
            signature=(name, catalog[name].model),
            model_preset=name,
        )

    loop = await _make_async_loop(
        tmp_path, admitted_async,
        provider=provider,
        model="model-a",
        context_window_tokens=16_000,
        provider_signature=("fast", "model-a"),
        provider_snapshot_loader=lambda: load_preset(default_name),
        model_presets=catalog,
        preset_catalog_loader=lambda: catalog,
        model_preset="fast",
        preset_snapshot_loader=load_preset,
    )
    loop.bus.subscribe(published.append, RuntimeModelChanged)
    catalog["Codex"] = catalog.pop("fast")
    default_name = "Codex"

    runtime = loop.refresh_runtime_config()
    await asyncio.sleep(0)

    assert (runtime.model, runtime.model_preset) == ("model-a", "Codex")
    assert [(event.model, event.model_preset) for event in published] == [
        ("model-a", "Codex"),
    ]


def test_next_turn_captures_generation_changed_after_previous_admission(
    tmp_path: Path,
    admitted_sync,
) -> None:
    provider = _provider("test-model")
    provider.generation = GenerationSettings(temperature=0.2, max_tokens=1024)
    loop = _make_loop(
        tmp_path, admitted_sync,
        provider=provider,
        model="test-model",
        context_window_tokens=16_384,
    )

    first = loop.llm_runtime()
    provider.generation = GenerationSettings(temperature=0.8, max_tokens=512)
    second = loop.llm_runtime()

    assert first.generation.temperature == 0.2
    assert first.generation.max_tokens == 1024
    assert second.generation.temperature == 0.8
    assert second.generation.max_tokens == 512


def test_settings_context_window_refreshes_runtime_state(
    tmp_path: Path,
    monkeypatch,
    admitted_sync,
) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    config.agents.defaults.model = "openai/gpt-4o"
    config.agents.defaults.provider = "openai"
    config.agents.defaults.context_window_tokens = 65_536
    config.providers.openai.api_key = "sk-test"
    save_config(config, config_path)
    monkeypatch.setattr("nanobot.config.loader._current_config_path", config_path)

    def loader(*, preset_name: str | None = None) -> ProviderSnapshot:
        with monkeypatch.context() as factory_patch:
            factory_patch.setattr("nanobot.providers.factory.make_provider", _synthetic_config_provider)
            return load_provider_snapshot(config_path, preset_name=preset_name)

    loop, config_path = _make_config_loop(
        tmp_path, admitted_sync, config, loader,
    )

    payload = update_agent_settings({"context_window_tokens": ["262144"]})
    loop.runtime_resolver.invalidate()
    loop.llm_runtime()

    assert payload["requires_restart"] is False
    assert loop.context_window_tokens == 262_144
    assert loop.llm_runtime().context_window_tokens == 262_144


def test_runtime_refresh_unadmitted_constructor_refuses_before_effects(tmp_path, monkeypatch) -> None:
    from nanobot.agent.memory_recovery_startup_admission import StartupReceiptAdmissionError
    from nanobot.session.manager import SessionManager

    provider = _provider("unadmitted-model")
    bus = MessageBus()
    workspace = tmp_path / "unadmitted"
    pending = workspace / "memory" / ".consolidation.pending.json"
    pending.parent.mkdir(parents=True)
    pending.write_text("{broken", encoding="utf-8")

    def denied(*_args, **_kwargs):
        pytest.fail("unadmitted runtime refresh reached a session or directory effect")

    monkeypatch.setattr(SessionManager, "__init__", denied)
    monkeypatch.setattr(Path, "mkdir", denied)
    before = tuple(tmp_path.iterdir())
    with pytest.raises(StartupReceiptAdmissionError):
        AgentLoop(
            bus=bus, provider=provider, workspace=workspace,
            sessions_root=tmp_path / "sessions",
        )
    assert tuple(tmp_path.iterdir()) == before
    assert not bus._pending
    provider.chat.assert_not_called()
    provider.chat_with_retry.assert_not_called()


def test_runtime_refresh_settings_use_acquired_config_without_optional_effects(
    tmp_path, monkeypatch, admitted_sync,
) -> None:
    from nanobot.config.loader import load_config
    from nanobot.providers import factory

    config = Config()
    config.agents.defaults.model = "openai/gpt-4o"
    config.agents.defaults.provider = "openai"
    config.agents.defaults.context_window_tokens = 8192
    config.providers.openai.api_key = "synthetic-only"
    config_path = tmp_path / "unused-config.json"
    constructed = []
    denied_factory = factory.make_provider

    def make_synthetic(current: Config, *, preset=None):
        provider = _synthetic_config_provider(current, preset=preset)
        constructed.append((current._source_path, provider))
        return provider

    def loader(*, preset_name=None):
        with monkeypatch.context() as factory_patch:
            factory_patch.setattr(factory, "make_provider", make_synthetic)
            return load_provider_snapshot(config_path, preset_name=preset_name)

    loop, config_path = _make_config_loop(tmp_path, admitted_sync, config, loader)
    initial = loop.llm_runtime()
    assert initial.context_window_tokens == 8192
    assert load_config()._source_path == config_path
    assert load_config().workspace_path == loop.workspace
    assert config_path.parent == loop.sessions.sessions_root.parent
    assert not (tmp_path / "unused-config.json").exists()
    assert loop.sessions.has_managed_authority
    assert loop.turn_delivery_factory.bus is loop.bus
    assert loop.tools_config.restrict_to_workspace is True
    assert loop.auto_compact is None
    assert not loop._background_tasks
    assert not loop.cron_service._running

    payload = update_agent_settings({"context_window_tokens": ["65536"]})
    loop.invalidate_runtime_config()
    refreshed = loop.llm_runtime()

    assert payload["requires_restart"] is False
    assert refreshed.context_window_tokens == 65536
    assert initial.context_window_tokens == 8192
    assert load_config(config_path).agents.defaults.context_window_tokens == 65536
    assert len(constructed) == 1
    assert constructed[0][0] == config_path
    assert refreshed.provider is constructed[0][1]
    assert factory.make_provider is denied_factory
    assert not loop._background_tasks
    assert not loop.cron_service._running
    loop.provider.chat.assert_not_called()
    loop.provider.chat_with_retry.assert_not_called()
