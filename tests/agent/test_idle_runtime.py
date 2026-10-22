"""Focused contract tests for the dormant I3b runtime binding substrate."""

from __future__ import annotations

import asyncio
import copy
import os
import pickle
import threading
from pathlib import Path

import pytest

from nanobot.agent.idle_compaction import IdleCompactionPolicy
from nanobot.agent.idle_runtime import (
    IdleRuntimeBindingError,
    IdleRuntimeBindingRegistry,
    IdleRuntimeClosedError,
    IdleRuntimeCrossContextError,
    IdleRuntimeCrossProcessError,
    IdleRuntimeDormantError,
    IdleRuntimeLimits,
    IdleRuntimeResolutionError,
    IdleSummaryProjectionConfig,
    IdleWorkspaceBinding,
)
from nanobot.agent.model_runtime import ModelRuntimeResolver
from nanobot.config.schema import Config, ModelPresetConfig
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.providers.factory import ProviderSnapshot, provider_signature
from nanobot.session.idle_checkpoint import IdlePolicyBinding
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source_store import SourceSessionStore
from nanobot.utils.llm_runtime import LLMRuntime


class _Provider(LLMProvider):
    def __init__(self, *, secret: str, model: str = "foreground") -> None:
        super().__init__(provider_name="fixture")
        self.api_key = secret
        self.backend = "fixture-backend"
        self.account = "fixture-account"
        self.endpoint = "https://fixture.invalid/v1"
        self.proxy = "http://proxy.invalid"
        self.extra_headers = {"X-Test": "header"}
        self.extra_body = {"body": "value"}
        self.extra_query = {"query": "value"}
        self.generation = GenerationSettings(max_tokens=128, temperature=0.1)
        self.default_model = model

    async def chat(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("I3b must never call a provider")

    def get_default_model(self) -> str:
        return self.default_model


def _runtime(
    provider: _Provider,
    *,
    model: str,
    preset: str | None = None,
    signature: tuple[object, ...] | None = None,
) -> LLMRuntime:
    return LLMRuntime(
        provider=provider,
        model=model,
        generation=GenerationSettings(max_tokens=128, temperature=0.1),
        context_window_tokens=4096,
        model_preset=preset,
        snapshot_signature=signature or ("fixture", model, preset),
    )


def _source(tmp_path: Path, *, key: str = "cli:i3b", preset: str | None = "summary") -> SourceSessionStore:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    session = Session(key)
    backend.save(session)
    source = backend.source
    source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    if preset is not None:
        source.update_metadata(key, {"_nanobot_model_preset": preset})
    return source


def _policy(tmp_path: Path) -> IdlePolicyBinding:
    workspace = tmp_path / "policy-workspace"
    return IdlePolicyBinding(
        fingerprint="policy-1",
        workspace=workspace,
        resource_policy=None,
        resolve_read=lambda value: workspace / value,
    )


def _registry(
    tmp_path: Path,
    *,
    source: SourceSessionStore | None = None,
    coordinator: object | None = None,
    preset_runtimes: dict[str, LLMRuntime | ProviderSnapshot] | None = None,
    foreground: LLMRuntime | None = None,
    auxiliary: LLMRuntime | None = None,
) -> IdleRuntimeBindingRegistry:
    source = source or _source(tmp_path, preset=None)
    foreground = foreground or _runtime(_Provider(secret="foreground-secret"), model="foreground")
    auxiliary = auxiliary or _runtime(_Provider(secret="aux-secret"), model="auxiliary")
    return IdleRuntimeBindingRegistry(
        source_store=source,
        foreground_runtime=foreground,
        auxiliary_runtime=auxiliary,
        preset_runtimes=preset_runtimes,
        summary_projection=IdleSummaryProjectionConfig(
            summary_version=2,
            projection_version=1,
            max_input_tokens=100,
            max_output_tokens=50,
            max_chunks=3,
        ),
        limits=IdleRuntimeLimits(
            enabled=False,
            idle_ttl_seconds=30.0,
            queue_capacity=2,
            max_concurrent_jobs=1,
            max_auxiliary_calls=1,
            max_auxiliary_tokens=100,
            max_input_tokens=100,
            max_output_tokens=50,
            max_chunks=3,
            max_auxiliary_wall_seconds=10.0,
            initial_backoff_seconds=1.0,
            max_backoff_seconds=4.0,
        ),
        workspace_binding=IdleWorkspaceBinding(tmp_path / "workspace-output", "artifact-ns-1"),
        policy_binding=_policy(tmp_path),
        coordinator=coordinator,  # type: ignore[arg-type]
    )


async def test_construction_is_inert_and_start_is_dormant(tmp_path: Path) -> None:
    source = _source(tmp_path)
    foreground = _runtime(_Provider(secret="foreground"), model="foreground")
    auxiliary = _runtime(_Provider(secret="auxiliary"), model="auxiliary")
    before = {task for task in asyncio.all_tasks() if not task.done()}
    registry = IdleRuntimeBindingRegistry(
        source_store=source,
        foreground_runtime=foreground,
        auxiliary_runtime=auxiliary,
        policy_binding=_policy(tmp_path),
    )
    after = {task for task in asyncio.all_tasks() if not task.done()}
    assert after == before
    assert registry.issued_count == 0
    assert registry.observability()["binding_count"] == 0
    with pytest.raises(IdleRuntimeDormantError):
        registry.start()


def test_issue_loads_authoritative_store_and_selects_preset(tmp_path: Path) -> None:
    source = _source(tmp_path)
    foreground = _runtime(_Provider(secret="foreground"), model="foreground")
    selected = _runtime(_Provider(secret="selected"), model="summary", preset="summary")
    registry = IdleRuntimeBindingRegistry(
        source_store=source,
        foreground_runtime=foreground,
        auxiliary_runtime=selected,
        preset_runtimes={"summary": selected},
        policy_binding=_policy(tmp_path),
    )
    binding = registry.issue("cli:i3b")
    current = source.load_source_snapshot("cli:i3b")
    assert binding.session_key == "cli:i3b"
    assert binding.incarnation == current.state.incarnation
    assert binding.revision == current.state.revision
    assert binding.generation == current.state.generation
    assert binding.selected_preset == "summary"
    assert binding.foreground_runtime is selected
    assert binding.auxiliary_runtime is selected


def test_unchanged_issue_reuses_current_and_revalidates_true(tmp_path: Path) -> None:
    registry = _registry(tmp_path, preset_runtimes={})
    first = registry.issue("cli:i3b")
    second = registry.issue("cli:i3b")
    assert second is first
    assert registry.revalidate(first, "cli:i3b") is True
    assert registry.issued_count == 1
    assert registry.binding_count == 1


def test_issue_rejects_per_binding_authority_overrides(tmp_path: Path) -> None:
    registry = _registry(tmp_path, preset_runtimes={})
    alternate = _runtime(_Provider(secret="alternate"), model="alternate")
    overrides = [
        {"foreground_runtime": alternate},
        {"auxiliary_runtime": alternate},
        {"summary_projection": IdleSummaryProjectionConfig(summary_version=9)},
        {"limits": IdleRuntimeLimits(idle_ttl_seconds=1.0)},
        {"policy_binding": registry.policy_binding},
    ]
    for override in overrides:
        with pytest.raises(TypeError):
            registry.issue("cli:i3b", **override)  # type: ignore[call-overload]

    first = registry.issue("cli:i3b")
    second = registry.issue("cli:i3b")
    assert second is first
    assert registry.issued_count == 1


def test_missing_or_removed_authoritative_preset_fails_closed(tmp_path: Path) -> None:
    registry = _registry(
        tmp_path,
        source=_source(tmp_path / "removed"),
        preset_runtimes={},
    )
    with pytest.raises(IdleRuntimeResolutionError):
        registry.issue("cli:i3b")

    no_preset_source = _source(tmp_path / "none", preset=None)
    no_preset = _registry(tmp_path / "none-registry", source=no_preset_source, preset_runtimes={})
    binding = no_preset.issue("cli:i3b")
    assert binding.selected_preset is None


def test_detached_or_tampered_snapshot_cannot_be_authority(tmp_path: Path) -> None:
    source = _source(tmp_path, preset=None)
    registry = _registry(tmp_path, source=source, preset_runtimes={})
    detached = source.load_source_snapshot("cli:i3b")
    detached.header["metadata"]["_nanobot_model_preset"] = "removed"
    with pytest.raises(TypeError):
        registry.issue(detached)  # type: ignore[arg-type]
    binding = registry.issue("cli:i3b")
    with pytest.raises(TypeError):
        registry.revalidate(binding, detached)  # type: ignore[arg-type]


def test_model_runtime_resolver_read_only_path_uses_existing_provider_without_refresh(
    tmp_path: Path,
) -> None:
    provider = _Provider(secret="foreground")
    foreground = _runtime(provider, model="foreground")
    resolver = ModelRuntimeResolver(
        foreground,
        model_presets={
            "summary": ModelPresetConfig(
                model="summary",
                provider="fixture",
                max_tokens=64,
                context_window_tokens=2048,
            )
        },
    )
    qualified_snapshot = ProviderSnapshot(
        provider=provider,
        model="summary",
        context_window_tokens=2048,
        signature=("summary", "summary"),
    )
    pure_runtime = resolver.resolve_preset_read_only("summary", snapshot=qualified_snapshot)
    assert pure_runtime.provider is provider
    assert resolver._resolved_presets == {}
    selected = _runtime(
        provider,
        model="summary",
        preset="summary",
        signature=qualified_snapshot.signature,
    )
    registry = _registry(
        tmp_path,
        source=_source(tmp_path / "resolver-source", preset="summary"),
        foreground=foreground,
        auxiliary=foreground,
        preset_runtimes={"summary": qualified_snapshot},
    )
    binding = registry.issue("cli:i3b")
    assert binding.foreground_runtime.model == "summary"
    assert binding.foreground_runtime.provider is provider
    assert binding.foreground_runtime.snapshot_signature == selected.snapshot_signature
    assert resolver._resolved_presets == {}


def test_source_revision_mutation_invalidates_and_replacement_is_bounded(tmp_path: Path) -> None:
    source = _source(tmp_path, preset=None)
    registry = _registry(tmp_path, source=source, preset_runtimes={})
    binding = registry.issue("cli:i3b")
    current = source.load_source_snapshot("cli:i3b")
    source.update_metadata("cli:i3b", {"test_marker": "changed"})
    assert registry.revalidate(binding, "cli:i3b") is False
    replacement = registry.issue("cli:i3b")
    assert replacement is not binding
    assert registry.binding_count == 1
    assert replacement.revision > current.state.revision


def test_provider_internal_mutation_is_irrelevant_after_capture(tmp_path: Path) -> None:
    provider = _Provider(secret="secret-before")
    runtime = _runtime(provider, model="foreground")
    registry = _registry(tmp_path, foreground=runtime, auxiliary=runtime)
    binding = registry.issue("cli:i3b")
    provider.api_key = "secret-after"
    provider.extra_body["nested"] = {"changed": True}
    provider.extra_headers["X-Test"] = "changed"
    provider.extra_query["q"] = "changed"
    provider.proxy = "http://changed.invalid"
    assert registry.revalidate(binding, "cli:i3b") is True


def _provider_config(*, api_key: str = "secret-before", extra_body: object | None = None,
                     extra_headers: object | None = None, extra_query: object | None = None,
                     proxy: str | None = "http://proxy.invalid") -> Config:
    return Config.model_validate(
        {
            "agents": {"defaults": {"modelPreset": "fast"}},
            "modelPresets": {
                "fast": {"model": "custom/test-model", "provider": "custom"},
            },
            "providers": {
                "custom": {
                    "apiKey": api_key,
                    "extraBody": extra_body or {"nested": {"value": "before"}},
                    "extraHeaders": extra_headers or {"X-Test": "before"},
                    "extraQuery": extra_query or {"q": "before"},
                    "proxy": proxy,
                },
            },
        }
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"api_key": "secret-after"},
        {"extra_body": {"nested": {"value": "after"}}},
        {"extra_headers": {"X-Test": "after"}},
        {"extra_query": {"q": "after"}},
        {"proxy": "http://changed.invalid"},
    ],
)
def test_provider_signature_refresh_mutation_invalidates(
    tmp_path: Path,
    changed: dict[str, object],
) -> None:
    before = _provider_config()
    before_runtime = _runtime(
        _Provider(secret="secret-before"),
        model="custom/test-model",
        signature=provider_signature(before),
    )
    registry = _registry(tmp_path, foreground=before_runtime, auxiliary=before_runtime)
    old_binding = registry.issue("cli:i3b")

    values = {
        "api_key": "secret-before",
        "extra_body": {"nested": {"value": "before"}},
        "extra_headers": {"X-Test": "before"},
        "extra_query": {"q": "before"},
        "proxy": "http://proxy.invalid",
    }
    values.update(changed)
    after = _provider_config(**values)
    after_runtime = _runtime(
        _Provider(secret="secret-after"),
        model="custom/test-model",
        signature=provider_signature(after),
    )
    registry.replace_runtimes(
        foreground_runtime=after_runtime,
        auxiliary_runtime=after_runtime,
    )
    assert registry.revalidate(old_binding, "cli:i3b") is False
    new_binding = registry.issue("cli:i3b")
    assert new_binding.runtime_fingerprint != old_binding.runtime_fingerprint
    rendered = repr(new_binding) + repr(registry.observability())
    assert "secret-before" not in rendered
    assert "secret-after" not in rendered


def test_missing_signature_is_rejected_without_provider_traversal(tmp_path: Path) -> None:
    provider = _Provider(secret="secret")
    runtime = _runtime(provider, model="foreground", signature=None)
    runtime = LLMRuntime(
        provider=runtime.provider,
        model=runtime.model,
        generation=runtime.generation,
        context_window_tokens=runtime.context_window_tokens,
        model_preset=runtime.model_preset,
        snapshot_signature=None,
    )
    registry = _registry(tmp_path, foreground=runtime, auxiliary=runtime)
    with pytest.raises(IdleRuntimeBindingError, match="captured provider signature"):
        registry.issue("cli:i3b")


def test_signature_encoder_rejects_cycles_and_unsupported_objects(tmp_path: Path) -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    cyclic_runtime = _runtime(
        _Provider(secret="cycle"),
        model="foreground",
        signature=("cycle", cyclic),
    )
    cyclic_registry = _registry(tmp_path / "cycle", foreground=cyclic_runtime, auxiliary=cyclic_runtime)
    with pytest.raises(IdleRuntimeBindingError, match="unsupported fingerprint material"):
        cyclic_registry.issue("cli:i3b")

    unsupported_runtime = _runtime(
        _Provider(secret="unsupported"),
        model="foreground",
        signature=("unsupported", object()),
    )
    unsupported_registry = _registry(
        tmp_path / "unsupported",
        foreground=unsupported_runtime,
        auxiliary=unsupported_runtime,
    )
    with pytest.raises(IdleRuntimeBindingError, match="unsupported fingerprint material"):
        unsupported_registry.issue("cli:i3b")


@pytest.mark.parametrize(
    "oversized",
    [
        "secret-string-" + "x" * (64 * 1024),
        b"secret-bytes-" + b"x" * (64 * 1024),
        ("x" * 60_000, "y" * 60_000, "z" * 60_000, "w" * 60_000, "v" * 60_000),
    ],
)
def test_signature_encoder_rejects_oversized_secret_material(
    tmp_path: Path,
    oversized: object,
) -> None:
    runtime = _runtime(
        _Provider(secret="provider-secret"),
        model="foreground",
        signature=("oversized-secret-marker", oversized),
    )
    registry = _registry(tmp_path, foreground=runtime, auxiliary=runtime)
    with pytest.raises(IdleRuntimeBindingError, match="unsupported fingerprint material") as caught:
        registry.issue("cli:i3b")
    assert "oversized-secret-marker" not in str(caught.value)


def test_signature_encoder_rejects_oversized_config_material_without_exposure(
    tmp_path: Path,
) -> None:
    marker = "oversized-config-marker-" + "x" * (64 * 1024)
    registry = _registry(tmp_path, preset_runtimes={})
    registry.replace_config(
        summary_projection=IdleSummaryProjectionConfig(projection_scope=marker),
    )
    with pytest.raises(IdleRuntimeBindingError, match="unsupported fingerprint material") as caught:
        registry.issue("cli:i3b")
    assert "oversized-config-marker" not in str(caught.value)


class _OrdinaryResolver:
    def resolve_preset(self, name: str, **kwargs: object) -> LLMRuntime:
        raise AssertionError("ordinary resolver must never be called")


class _QualifiedResolver:
    def __init__(self, runtime: LLMRuntime) -> None:
        self.runtime = runtime
        self.calls: list[tuple[str, ProviderSnapshot]] = []

    def resolve_qualified_preset(self, name: str, snapshot: ProviderSnapshot) -> LLMRuntime:
        self.calls.append((name, snapshot))
        return self.runtime


class _MutatingQualifiedResolver:
    def __init__(self, registry: IdleRuntimeBindingRegistry, runtime: LLMRuntime) -> None:
        self.registry = registry
        self.runtime = runtime

    def resolve_qualified_preset(self, name: str, snapshot: ProviderSnapshot) -> LLMRuntime:
        self.registry.replace_runtimes(
            foreground_runtime=self.runtime,
            auxiliary_runtime=self.runtime,
        )
        return self.runtime


def test_qualified_resolver_cannot_reenter_authority_mutation(tmp_path: Path) -> None:
    source = _source(tmp_path)
    foreground = _runtime(_Provider(secret="foreground"), model="foreground")
    selected = _runtime(_Provider(secret="selected"), model="summary", preset="summary")
    snapshot = ProviderSnapshot(
        provider=foreground.provider,
        model="summary",
        context_window_tokens=4096,
        signature=selected.snapshot_signature,
    )
    registry = IdleRuntimeBindingRegistry(
        source_store=source,
        foreground_runtime=foreground,
        auxiliary_runtime=foreground,
        preset_resolver=None,
        preset_snapshots={"summary": snapshot},
        policy_binding=_policy(tmp_path),
    )
    registry._preset_resolver = _MutatingQualifiedResolver(registry, selected)  # type: ignore[assignment]
    epoch = registry.runtime_epoch
    with pytest.raises(IdleRuntimeCrossContextError, match="nested synchronous mutation"):
        registry.issue("cli:i3b")
    assert registry.runtime_epoch == epoch
    assert registry.binding_count == 0


def test_ordinary_resolver_is_rejected_and_qualified_adapter_is_snapshot_only(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    foreground = _runtime(_Provider(secret="foreground"), model="foreground")
    qualified = ProviderSnapshot(
        provider=foreground.provider,
        model="summary",
        context_window_tokens=4096,
        signature=("qualified", "summary"),
    )
    with pytest.raises(TypeError):
        IdleRuntimeBindingRegistry(
            source_store=source,
            foreground_runtime=foreground,
            auxiliary_runtime=foreground,
            preset_resolver=_OrdinaryResolver(),  # type: ignore[arg-type]
            policy_binding=_policy(tmp_path),
        )
    adapter_runtime = _runtime(
        _Provider(secret="qualified"),
        model="summary",
        preset="summary",
        signature=qualified.signature,
    )
    adapter = _QualifiedResolver(adapter_runtime)
    registry = IdleRuntimeBindingRegistry(
        source_store=source,
        foreground_runtime=foreground,
        auxiliary_runtime=foreground,
        preset_resolver=adapter,
        preset_snapshots={"summary": qualified},
        policy_binding=_policy(tmp_path),
    )
    binding = registry.issue("cli:i3b")
    assert binding.foreground_runtime is adapter_runtime
    assert adapter.calls == [("summary", qualified)]


class _SecretResolver:
    def resolve_qualified_preset(self, name: str, snapshot: ProviderSnapshot) -> LLMRuntime:
        raise RuntimeError("api-key=resolver-secret")


def test_resolver_failure_is_sanitized_without_exception_chain(tmp_path: Path) -> None:
    source = _source(tmp_path)
    foreground = _runtime(_Provider(secret="foreground"), model="foreground")
    snapshot = ProviderSnapshot(
        provider=foreground.provider,
        model="summary",
        context_window_tokens=4096,
        signature=("summary",),
    )
    registry = IdleRuntimeBindingRegistry(
        source_store=source,
        foreground_runtime=foreground,
        auxiliary_runtime=foreground,
        preset_resolver=_SecretResolver(),
        preset_snapshots={"summary": snapshot},
        policy_binding=_policy(tmp_path),
    )
    with pytest.raises(IdleRuntimeResolutionError) as caught:
        registry.issue("cli:i3b")
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "resolver-secret" not in str(caught.value)


def test_config_and_runtime_epochs_invalidate(tmp_path: Path) -> None:
    registry = _registry(tmp_path, preset_runtimes={})
    binding = registry.issue("cli:i3b")
    old_runtime_epoch = registry.runtime_epoch
    old_config_epoch = registry.config_epoch
    registry.replace_runtimes(
        foreground_runtime=binding.foreground_runtime,
        auxiliary_runtime=binding.auxiliary_runtime,
    )
    assert registry.runtime_epoch == old_runtime_epoch + 1
    assert registry.revalidate(binding, "cli:i3b") is False
    replacement = registry.issue("cli:i3b")
    registry.replace_config(summary_projection=IdleSummaryProjectionConfig(summary_version=3))
    assert registry.config_epoch == old_config_epoch + 1
    assert registry.revalidate(replacement, "cli:i3b") is False


def test_binding_config_epoch_is_captured_across_config_replacement(tmp_path: Path) -> None:
    registry = _registry(tmp_path, preset_runtimes={})
    old_binding = registry.issue("cli:i3b")
    old_epoch = old_binding.config_epoch

    assert old_epoch == old_binding.epoch
    registry.replace_config(summary_projection=IdleSummaryProjectionConfig(summary_version=3))

    assert old_binding.config_epoch == old_epoch
    assert registry.revalidate(old_binding, "cli:i3b") is False

    new_binding = registry.issue("cli:i3b")
    assert new_binding.config_epoch == old_epoch + 1
    assert new_binding.config_epoch == new_binding.epoch


def test_copy_pickle_and_foreign_registry_are_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path, preset_runtimes={})
    binding = registry.issue("cli:i3b")
    with pytest.raises(TypeError):
        copy.copy(binding)
    with pytest.raises(TypeError):
        copy.deepcopy(binding)
    with pytest.raises(TypeError):
        pickle.dumps(binding)

    other = _registry(tmp_path / "other", preset_runtimes={})
    with pytest.raises(IdleRuntimeBindingError):
        other.revalidate(binding, "cli:i3b")


async def test_publish_if_current_holds_gate_against_update(tmp_path: Path) -> None:
    registry = _registry(tmp_path, preset_runtimes={})
    binding = registry.issue("cli:i3b")
    callback_state: list[object] = []
    update_task: asyncio.Task[object] | None = None

    async def update() -> None:
        registry.replace_config(summary_projection=IdleSummaryProjectionConfig(summary_version=2))
        callback_state.append("updated")

    def callback(current: object) -> str:
        nonlocal update_task
        callback_state.append(("callback", current))
        update_task = asyncio.create_task(update())
        assert not update_task.done()
        return "published"

    assert await registry.publish_if_current(binding, callback) == "published"
    assert callback_state[0][0] == "callback"  # type: ignore[index]
    assert update_task is not None
    await update_task
    assert callback_state[-1] == "updated"


async def test_publication_gate_rejects_concurrent_sync_update(tmp_path: Path) -> None:
    registry = _registry(tmp_path, preset_runtimes={})

    async def update() -> None:
        registry.replace_config(summary_projection=IdleSummaryProjectionConfig(summary_version=2))

    async with registry.publication_gate():
        update_task = asyncio.create_task(update())
        await asyncio.sleep(0)
        with pytest.raises(IdleRuntimeCrossContextError):
            await update_task


async def test_publish_rejects_awaitable_callback(tmp_path: Path) -> None:
    registry = _registry(tmp_path, preset_runtimes={})
    binding = registry.issue("cli:i3b")

    def callback(current: IdleRuntimeBindingRegistry) -> object:
        return asyncio.Future()

    with pytest.raises(IdleRuntimeBindingError, match="without awaiting"):
        await registry.publish_if_current(binding, callback)


class _Coordinator:
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.foreground: list[str] = []
        self.close_calls = 0

    async def foreground_admitted(self, key: str) -> None:
        self.foreground.append(key)

    async def aclose(self) -> None:
        self.close_calls += 1
        await asyncio.sleep(0)
        self.closed.set()


async def test_foreground_and_repeated_cancelled_close_join_coordinator(tmp_path: Path) -> None:
    coordinator = _Coordinator()
    registry = _registry(tmp_path, coordinator=coordinator, preset_runtimes={})
    await registry.foreground_admitted("cli:i3b")
    assert coordinator.foreground == ["cli:i3b"]
    first = asyncio.create_task(registry.aclose())
    second = asyncio.create_task(registry.aclose())
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert coordinator.closed.is_set()
    assert coordinator.close_calls == 1
    await registry.aclose()


async def test_cancelled_close_queued_behind_publication_gate_still_joins(
    tmp_path: Path,
) -> None:
    coordinator = _Coordinator()
    registry = _registry(tmp_path, coordinator=coordinator, preset_runtimes={})
    registry.issue("cli:i3b")
    async with registry.publication_gate():
        first = asyncio.create_task(registry.aclose())
        second = asyncio.create_task(registry.aclose())
        await asyncio.sleep(0)
        assert registry._close_task is not None
        with pytest.raises(IdleRuntimeCrossContextError):
            registry.issue("cli:i3b")
        first.cancel()
        second.cancel()
        await asyncio.sleep(0)
        first.cancel()
        second.cancel()
        await asyncio.sleep(0)
        assert not coordinator.closed.is_set()
        assert not first.done()
        assert not second.done()

    with pytest.raises(asyncio.CancelledError):
        await first
    with pytest.raises(asyncio.CancelledError):
        await second
    assert registry.closed is True
    assert coordinator.closed.is_set()
    assert coordinator.close_calls == 1
    with pytest.raises(IdleRuntimeClosedError):
        registry.issue("cli:i3b")


def test_policy_and_limits_are_captured_immutably(tmp_path: Path) -> None:
    policy = IdleCompactionPolicy(
        idle_ttl_seconds=30.0,
        queue_capacity=2,
        max_concurrent_jobs=1,
        max_auxiliary_calls=1,
        max_auxiliary_tokens=100,
        max_auxiliary_wall_seconds=10.0,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=4.0,
    )
    registry = IdleRuntimeBindingRegistry(
        source_store=_source(tmp_path, preset=None),
        foreground_runtime=_runtime(_Provider(secret="s"), model="foreground"),
        auxiliary_runtime=_runtime(_Provider(secret="aux"), model="auxiliary"),
        limits=policy,
        policy_binding=_policy(tmp_path),
    )
    binding = registry.issue("cli:i3b")
    assert binding.limits.queue_capacity == 2
    assert binding.summary_projection.projection_version == 1
    assert binding.policy_binding.workspace == tmp_path / "policy-workspace"


def test_calls_from_other_thread_are_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path, preset_runtimes={})
    errors: list[BaseException] = []

    def call() -> None:
        try:
            registry.issue("cli:i3b")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=call)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], IdleRuntimeCrossContextError)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable")
def test_forked_process_cannot_use_parent_authority(tmp_path: Path) -> None:
    registry = _registry(tmp_path, preset_runtimes={})
    binding = registry.issue("cli:i3b")
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            registry.revalidate(binding, "cli:i3b")
        except IdleRuntimeCrossProcessError:
            os.write(write_fd, b"rejected")
        finally:
            os.close(write_fd)
            os._exit(0)
    os.close(write_fd)
    try:
        assert os.read(read_fd, 32) == b"rejected"
    finally:
        os.close(read_fd)
        os.waitpid(pid, 0)
