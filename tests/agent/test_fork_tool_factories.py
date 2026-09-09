"""The real loader must expose only explicitly selected foreground capabilities."""
from unittest.mock import MagicMock

import pytest

from nanobot.agent.tools.context import ForkToolDependencies, ToolContext
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.write_guard import FileLockRegistry
from nanobot.config.schema import ToolsConfig
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.migration_guard import MigrationIncompleteError


@pytest.fixture
def context(tmp_path):
    policy = ResourceAccessPolicy(workspace=tmp_path, context_manager=ContextRepoManager())
    return ToolContext(config=ToolsConfig(), workspace=str(tmp_path), fork_dependencies=ForkToolDependencies(
        foreground_manager=MagicMock(), resource_policy=policy, file_locks=FileLockRegistry()))


def load(context, monkeypatch):
    monkeypatch.setattr('nanobot.migration_guard.require_runtime_ready', lambda: None)
    loader = ToolLoader()
    loader.discover = MagicMock(side_effect=AssertionError('discovery must not run'))
    loader._discover_plugins = MagicMock(side_effect=AssertionError('plugins must not run'))
    registry = ToolRegistry()
    names = loader.load(context, registry)
    return names, registry


def test_unpatched_loader_remains_guarded(context):
    with pytest.raises(MigrationIncompleteError):
        ToolLoader().load(context, ToolRegistry())


def test_exact_default_main_tools_and_dependency_handoff(context, monkeypatch):
    monkeypatch.delenv('NANOBOT_TREND_VPN_TOOLS_ENABLED', raising=False)
    monkeypatch.setattr('nanobot.agent.tools.fork_factories.vpn_tools_enabled', lambda: False)
    names, registry = load(context, monkeypatch)
    assert set(names) == {'read_file', 'write_file', 'edit_file', 'list_dir', 'exec', 'web_search',
                          'web_fetch', 'message', 'plan_task', 'delegate_task', 'review_work',
                          'explore', 'agent_browser', 'agent_device'}
    assert registry.get('write_file')._resource_policy is context.fork_dependencies.resource_policy
    assert registry.get('write_file')._lock_registry is context.fork_dependencies.file_locks
    assert registry.get('agent_browser').working_dir == context.workspace
    assert registry.get('plan_task')._manager is context.fork_dependencies.foreground_manager


def test_missing_dependencies_and_collisions_leave_registry_unchanged(context, monkeypatch):
    monkeypatch.setattr('nanobot.migration_guard.require_runtime_ready', lambda: None)
    registry = ToolRegistry()
    bare = ToolContext(config=context.config, workspace=context.workspace)
    with pytest.raises(ValueError, match='foreground_manager'):
        ToolLoader().load(bare, registry)
    assert registry.tool_names == []
    _, populated = load(context, monkeypatch)
    existing = populated.get('read_file')
    registry.register(existing)
    with pytest.raises(ValueError, match='Duplicate'):
        ToolLoader().load(context, registry)
    assert registry.tool_names == ['read_file']
    assert registry.get('read_file') is existing


@pytest.mark.parametrize('enabled,flag', [(False, False), (False, True), (True, False), (True, True)])
def test_crawler_requires_both_switches(context, monkeypatch, enabled, flag):
    from dataclasses import replace

    context.fork_dependencies = replace(context.fork_dependencies, crawler_enabled=enabled)
    monkeypatch.setattr('nanobot.agent.tools.fork_factories.crawl_tools_enabled', lambda: flag)
    names, _ = load(context, monkeypatch)
    assert ('crawl_research' in names) == (enabled and flag)
    assert 'social_crawl' not in names


def test_optional_constructor_settings_and_one_image(context, monkeypatch):
    from nanobot.agent.tools.image_generation import ImageGenerationTool

    context.config.agent_browser.package = 'configured-browser'
    context.config.agent_device.enabled = False
    context.config.desktop_use.enabled = True
    context.config.desktop_use.humanize_typing = False
    context.config.image_generation.enabled = True
    names, registry = load(context, monkeypatch)
    assert registry.get('agent_browser').package == 'configured-browser'
    assert 'agent_device' not in names
    assert registry.get('desktop_use')._backend is None
    assert registry.get('desktop_use').humanize_typing is False
    assert names.count('generate_image') == 1
    assert isinstance(registry.get('generate_image'), ImageGenerationTool)


@pytest.mark.parametrize('scope', ['subagent', 'worker', 'crawler', 'unknown'])
def test_generic_scope_cannot_add_background_capabilities(context, monkeypatch, scope):
    monkeypatch.setattr('nanobot.migration_guard.require_runtime_ready', lambda: None)
    with pytest.raises(ValueError, match='scope'):
        ToolLoader().load(context, ToolRegistry(), scope=scope)


@pytest.mark.parametrize('flag,enabled', [(None, False), ('false', False), ('1', False),
                                         ('true', True), (' TrUe ', True)])
def test_vpn_preserves_exact_existing_flag(context, monkeypatch, flag, enabled):
    if flag is None:
        monkeypatch.delenv('AGENT_VPN_BROKER_ENABLED', raising=False)
    else:
        monkeypatch.setenv('AGENT_VPN_BROKER_ENABLED', flag)
    names, _ = load(context, monkeypatch)
    vpn = {name for name in names if name.startswith('trend_vpn_')}
    assert vpn == ({'trend_vpn_session_start', 'trend_vpn_fetch', 'trend_vpn_browser_fetch',
                    'trend_vpn_session_close'} if enabled else set())
