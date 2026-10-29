import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.agent.foreground import build_foreground_dependencies, validate_foreground_dependencies
from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.web import WebFetchTool
from nanobot.agent.write_guard import FileLockRegistry
from nanobot.providers.factory import ProviderSnapshot
from nanobot.utils.llm_runtime import LLMRuntime

from .test_migration_engine import ScriptedProvider

pytest_plugins = ("tests.agent.test_migration_engine",)


def build(config):
    provider = ScriptedProvider([])
    runtime = LLMRuntime.capture(provider, 'baseline', context_window_tokens=9000)
    return build_foreground_dependencies(workspace=config.workspace_path, runtime=runtime,
        tools_config=config.tools, restrict_to_workspace=True, config=config)


def test_shared_identity_and_disabled_crawler_does_not_construct_provider(engine_config, monkeypatch):
    monkeypatch.setattr('nanobot.extensions.crawler.build_provider_snapshot', Mock(side_effect=AssertionError('provider constructed')))
    deps = build(engine_config)
    manager = deps.foreground_manager
    assert deps.resource_policy is manager.resource_policy
    assert deps.file_locks is manager.file_lock_registry
    validate_foreground_dependencies(deps, workspace=engine_config.workspace_path, restrict_to_workspace=True)
    with pytest.raises(ValueError, match='Inconsistent'):
        validate_foreground_dependencies(replace(deps, file_locks=FileLockRegistry()),
                                          workspace=engine_config.workspace_path, restrict_to_workspace=True)


async def test_owned_manager_closed_once_and_cleanup_continues_after_failure(engine_config):
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([]))
    close = AsyncMock(side_effect=RuntimeError('foreground cleanup failed'))
    loop._fork_dependencies.foreground_manager.close = close
    loop.subagents.close = AsyncMock()
    loop._exec_session_manager.close_all = AsyncMock()
    with pytest.raises(RuntimeError, match='foreground cleanup failed'):
        await loop.aclose()
    loop.subagents.close.assert_awaited_once()
    loop._exec_session_manager.close_all.assert_awaited_once()
    with pytest.raises(RuntimeError, match='foreground cleanup failed'):
        await loop.aclose()
    close.assert_awaited_once()


async def test_injected_bundle_is_shared_and_not_owned(engine_config, monkeypatch):
    deps = build(engine_config)
    monkeypatch.setattr('nanobot.agent.loop.build_foreground_dependencies', Mock(side_effect=AssertionError('replacement constructed')))
    close = AsyncMock()
    deps.foreground_manager.close = close
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([]), fork_dependencies=deps)
    try:
        assert loop.context.context_manager is deps.resource_policy.context_manager
        assert loop._fork_dependencies is deps
    finally:
        await loop.aclose()
    close.assert_not_awaited()


async def test_runtime_inheritance_is_immutable_and_crawler_is_independent(engine_config):
    deps = build(engine_config)
    manager = deps.foreground_manager
    observed = []
    async def record(spec):
        await asyncio.sleep(0)
        observed.append(spec.runtime)
        return AgentRunResult(final_content='result', messages=[])
    manager.runner = SimpleNamespace(run=record)
    one, two = ScriptedProvider([]), ScriptedProvider([])
    runtime_one = LLMRuntime.capture(one, 'one', context_window_tokens=8000, model_preset='one', snapshot_signature=('one',))
    runtime_two = LLMRuntime.capture(two, 'two', context_window_tokens=16000, model_preset='two', snapshot_signature=('two',))
    one.generation = replace(one.generation, max_tokens=999)
    two.generation = replace(two.generation, max_tokens=888)
    async def call(runtime):
        with request_context(RequestContext(channel='cli', chat_id='test', runtime=runtime)):
            await manager._run_role(role='planner', prompt='plan', assignment='task', tools=ToolRegistry(), max_iterations=1)
    await asyncio.gather(call(runtime_one), call(runtime_two))
    assert runtime_one in observed and runtime_two in observed
    assert all(runtime.generation.max_tokens == 256 for runtime in observed)
    manager.provider.generation = replace(manager.provider.generation, max_tokens=777)
    await manager._run_role(role='planner', prompt='plan', assignment='task', tools=ToolRegistry(), max_iterations=1)
    assert observed[-1].generation.max_tokens == 256 and observed[-1].context_window_tokens == 9000
    with request_context(RequestContext(channel='cli', chat_id='test', runtime=runtime_two)):
        await manager._run_role(role='crawler', prompt='crawl', assignment='task', tools=ToolRegistry(), max_iterations=1,
                                runtime_override=manager._crawler_runtime)
    assert observed[-1] is manager._crawler_runtime


@pytest.mark.parametrize('enabled_by_environment', [False, True])
def test_crawler_uses_explicit_snapshot_without_main_fallbacks(engine_config, monkeypatch, enabled_by_environment):
    config = engine_config
    config.agents.crawler.enabled = True
    config.agents.crawler.provider = 'openrouter'
    config.agents.crawler.model = 'crawler-model'
    config.agents.defaults.fallback_models = ['unresolved-main-fallback']
    crawler_provider = ScriptedProvider([])
    seen = []
    def snapshot(copied, *, preset):
        seen.append((copied, preset))
        assert copied is not config
        assert copied.agents.defaults.fallback_models == [] and copied.agents.defaults.fallback == []
        return ProviderSnapshot(provider=crawler_provider, model=preset.model,
            context_window_tokens=preset.context_window_tokens, signature=('crawler',), generation=preset.to_generation_settings())
    monkeypatch.setattr('nanobot.extensions.crawler.crawl_tools_enabled', lambda: enabled_by_environment)
    monkeypatch.setattr('nanobot.extensions.crawler.build_provider_snapshot', snapshot)
    deps = build(config)
    assert deps.crawler_enabled is enabled_by_environment
    assert len(seen) == int(enabled_by_environment)
    assert config.agents.defaults.fallback_models == ['unresolved-main-fallback']
    if enabled_by_environment:
        runtime = deps.foreground_manager._crawler_runtime
        assert runtime.provider is crawler_provider and runtime.model == 'crawler-model'
        assert runtime.generation.max_tokens == config.agents.crawler.max_tokens.output
        assert seen[0][1].provider == 'openrouter'


async def test_configured_context_repositories_reach_prompt_and_tool_policy(engine_config, tmp_path):
    repo = tmp_path / 'context'
    repo.mkdir()
    (repo / 'CONTEXT.md').write_text('shared repository instructions')
    engine_config.agents.defaults.context_paths = [str(repo)]
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([]))
    try:
        deps = loop._fork_dependencies
        assert deps.resource_policy.context_manager.paths == [repo]
        assert 'shared repository instructions' in loop.context.build_system_prompt(include_memory=False)
        assert loop.tools.get('read_file')._resource_policy is deps.resource_policy
    finally:
        await loop.aclose()


def test_configured_crawler_failure_does_not_fall_back_to_main(engine_config, monkeypatch):
    engine_config.agents.crawler.enabled = True
    engine_config.agents.crawler.provider = 'missing-provider'
    monkeypatch.setattr('nanobot.extensions.crawler.crawl_tools_enabled', lambda: True)
    monkeypatch.setattr('nanobot.extensions.crawler.build_provider_snapshot', Mock(side_effect=ValueError('crawler configuration invalid')))
    with pytest.raises(ValueError, match='crawler configuration invalid'):
        build(engine_config)


@pytest.mark.parametrize('field,value', [('workspace', 'different'), ('restrict_to_workspace', False)])
def test_injected_manager_scope_must_match_its_policy(engine_config, field, value):
    deps = build(engine_config)
    if field == 'workspace':
        value = engine_config.workspace_path / value
    setattr(deps.foreground_manager, field, value)
    with pytest.raises(ValueError, match='Inconsistent'):
        validate_foreground_dependencies(deps, workspace=engine_config.workspace_path, restrict_to_workspace=True)


async def test_direct_constructor_builds_coherent_owned_defaults(engine_config):
    from nanobot.bus.queue import MessageBus
    from nanobot.session.manager import SessionManager

    config = engine_config
    sessions = SessionManager(config.workspace_path, sessions_root=config.runtime_data_dir / 'sessions')
    loop = AgentLoop(bus=MessageBus(), provider=ScriptedProvider([]), workspace=config.workspace_path,
                     tools_config=config.tools, session_manager=sessions, restrict_to_workspace=True)
    try:
        deps = loop._fork_dependencies
        validate_foreground_dependencies(deps, workspace=config.workspace_path, restrict_to_workspace=True)
        assert loop._owned_foreground_manager is deps.foreground_manager
        assert loop.context.context_manager is deps.resource_policy.context_manager
    finally:
        await loop.aclose()


async def test_real_foreground_builder_preserves_web_fetch_policy(engine_config, monkeypatch):
    config = engine_config
    config.tools.web.search.provider = "brave"
    config.tools.web.search.api_key = "brave-search-sentinel"
    config.tools.web.fetch.provider = "local"
    config.tools.web.fetch.use_jina_reader = False
    deps = build(config)
    delegated = deps.foreground_manager.tools_for_role("planner").get("web_fetch")
    assert isinstance(delegated, WebFetchTool)
    assert delegated.config.provider == "local"
    assert delegated.config.use_jina_reader is False
    assert delegated.config.blocked_hosts == []
    assert delegated._tinyfish_fetch_key() == ""

    async def readable(self, url, extract_mode, max_chars):
        return "local result"

    tinyfish = AsyncMock(return_value="tinyfish result")
    jina = AsyncMock(return_value="jina result")
    monkeypatch.setattr(WebFetchTool, "_fetch_readability", readable)
    monkeypatch.setattr(WebFetchTool, "_fetch_tinyfish", tinyfish)
    monkeypatch.setattr(WebFetchTool, "_fetch_jina", jina)
    monkeypatch.setattr(
        "nanobot.agent.tools.web._stream_with_safe_redirects",
        AsyncMock(return_value=(SimpleNamespace(headers={"content-type": "text/html"}), None, None, False)),
    )
    monkeypatch.setattr("nanobot.agent.tools.web._validate_url_safe", lambda url: (True, ""))

    assert await delegated.execute("https://allowed.example/page") == "local result"
    blocked_config = config.model_copy(deep=True)
    blocked_config.tools.web.fetch.provider = "tinyfish"
    blocked_config.tools.web.fetch.api_key = "fetch-sentinel"
    blocked_config.tools.web.fetch.blocked_hosts = ["blocked.example"]
    blocked_deps = build(blocked_config)
    blocked_tool = blocked_deps.foreground_manager.tools_for_role("planner").get("web_fetch")
    assert isinstance(blocked_tool, WebFetchTool)
    assert blocked_tool.config.blocked_hosts == ["blocked.example"]
    blocked = str(await blocked_tool.execute("https://blocked.example/page"))

    assert "disabled" in blocked.lower()
    tinyfish.assert_not_awaited()
    jina.assert_not_awaited()


async def test_foreground_web_fetch_refresh_matches_core_refresh_intent(engine_config):
    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=provider,
        provider_snapshot_loader=Mock(),
    )
    try:
        core_fetch = loop.tools.get("web_fetch")
        delegated = loop._fork_dependencies.foreground_manager.tools_for_role("planner").get("web_fetch")
        assert isinstance(core_fetch, WebFetchTool)
        assert isinstance(delegated, WebFetchTool)
        assert core_fetch._config_loader is not None
        assert delegated._config_loader is not None
    finally:
        await loop.aclose()
