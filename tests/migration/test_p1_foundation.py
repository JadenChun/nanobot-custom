"""P1 foundation checks; runtime parity remains a P2-P5 acceptance gate."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nanobot.config.loader import load_config, save_config
from nanobot.config.schema import Config
from nanobot.migration_guard import MigrationIncompleteError


@pytest.mark.parametrize('module', [
    'nanobot.agent.loop', 'nanobot.agent.runner', 'nanobot.agent.turn',
    'nanobot.agent.run_store', 'nanobot.config.schema', 'nanobot.cli.commands',
    'nanobot.api.server', 'nanobot.context_repo.manager', 'nanobot.crawler.worker',
])
def test_foundation_imports(module):
    importlib.import_module(module)


def test_legacy_config_roundtrip_and_no_implicit_activation(tmp_path):
    source = {
        'agents': {'defaults': {'maxTokens': {'input': 48000, 'output': 3000},
            'contextRepos': [{'path': '/synthetic/context', 'autoSync': False}],
            'fallback': [{'provider': 'openai', 'model': 'fallback'}]},
            'crawler': {'enabled': True, 'maxTokens': {'input': 20000, 'output': 2000}}},
        'providers': {'openai': {'apiKeys': ['synthetic-a', 'synthetic-b'], 'rateLimit': 8, 'timeout': 33}},
        'channels': {'telegram': {'enabled': True, 'allowChats': ['-123'], 'allowFrom': ['456']},
                     'privatePlugin': {'setting': 'retained'}},
        'tools': {'agentBrowser': {'timeout': 42}, 'desktopUse': {'enabled': False}},
    }
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(source), encoding='utf-8-sig')
    before = path.read_bytes()
    cfg = load_config(path)
    assert path.read_bytes() == before
    assert cfg.agents.defaults.max_tokens == 3000
    assert cfg.agents.defaults.context_window_tokens == 48000
    assert cfg.providers.openai.effective_keys == ['synthetic-a', 'synthetic-b']
    assert cfg.providers.openai.rate_limit == 8
    assert cfg.providers.openai.timeout == 33
    assert cfg.agents.defaults.context_repos[0].auto_sync is False
    assert cfg.tools.agent_browser.timeout == 42
    assert cfg.agents.crawler.max_tokens.input == 20000
    assert cfg.channels.websocket['enabled'] is False
    assert cfg.agents.defaults.dream.enabled is False
    assert cfg.agents.defaults.session_ttl_minutes == 0
    target = tmp_path / 'saved.json'
    save_config(cfg, target)
    reloaded = load_config(target)
    assert reloaded.model_dump() == cfg.model_dump()
    assert reloaded.channels.model_extra['privatePlugin'] == {'setting': 'retained'}
    assert reloaded.channels.model_extra['telegram']['allowChats'] == ['-123']


def test_direct_construction_and_scalar_tokens():
    cfg = Config(agents={'defaults': {'maxTokens': {'input': 24000, 'output': 2000}}})
    assert cfg.agents.defaults.max_tokens == 2000
    assert cfg.agents.defaults.context_window_tokens == 24000
    scalar = Config(agents={'defaults': {'maxTokens': 2000, 'contextWindowTokens': 24000}})
    assert scalar.agents.defaults.max_tokens == 2000
    assert scalar.agents.defaults.context_window_tokens == 24000
    assert Config().channels.websocket == {'enabled': False}


def test_legacy_capacity_precedence_is_preserved():
    cfg = Config(agents={"defaults": {"maxTokens": {"input": 24000, "output": 2000}, "contextWindowTokens": 12000}})
    assert cfg.agents.defaults.context_window_tokens == 24000
    partial = Config(agents={"defaults": {"maxTokens": {"output": 2000}, "contextWindowTokens": 24000}})
    assert partial.agents.defaults.context_window_tokens == 24000


def test_runtime_blocked_before_workspace_creation(tmp_path):
    from nanobot.agent.loop import AgentLoop
    workspace = tmp_path / 'must-not-exist'
    with pytest.raises(MigrationIncompleteError):
        AgentLoop(bus=object(), provider=object(), workspace=workspace)
    assert not workspace.exists()


def test_loader_blocks_before_discovery(monkeypatch):
    from nanobot.agent.tools.loader import ToolLoader
    from nanobot.agent.tools.registry import ToolRegistry
    loader = ToolLoader()
    def forbidden():
        pytest.fail('unadapted tool/plugin discovery executed')
    monkeypatch.setattr(loader, 'discover', forbidden)
    monkeypatch.setattr(loader, '_discover_plugins', forbidden)
    with pytest.raises(MigrationIncompleteError):
        loader.load(object(), ToolRegistry())


def test_registry_rejects_duplicate_names():
    from types import SimpleNamespace

    from nanobot.agent.tools.registry import ToolRegistry
    registry = ToolRegistry()
    first = SimpleNamespace(name='read_file')
    registry.register(first)
    with pytest.raises(ValueError, match='Duplicate'):
        registry.register(SimpleNamespace(name='read_file'))
    assert registry.get('read_file') is first


def _cli(tmp_path, *args):
    home = tmp_path / 'home'
    home.mkdir(exist_ok=True)
    env = {'HOME': str(home), 'PATH': str(Path(sys.executable).parent) + ':/usr/bin:/bin',
           'PYTHONPATH': str(Path(__file__).resolve().parents[2]),
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'NO_COLOR': '1', 'TERM': 'dumb'}
    return subprocess.run([sys.executable, '-m', 'nanobot', *args], env=env,
                          text=True, capture_output=True, timeout=30)


def test_bare_cli_displays_help_without_state(tmp_path):
    result = _cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'Usage:' in result.stdout
    assert list((tmp_path / 'home').iterdir()) == []


def test_gateway_service_arguments_parse_before_block(tmp_path):
    result = _cli(tmp_path, 'gateway', '--config', str(tmp_path / 'absent.json'),
                  '--log-file', str(tmp_path / 'gateway.log'))
    assert result.returncode != 0
    assert 'Migration candidate is not runnable' in result.stderr
    assert 'No such option' not in result.stderr
    assert not (tmp_path / 'gateway.log').exists()
    assert list((tmp_path / 'home').iterdir()) == []


def test_classic_default():
    import inspect

    from nanobot.cli.agent import agent
    assert inspect.signature(agent).parameters['classic'].default.default is True


def test_logfile_preserves_rotation_and_output(tmp_path, monkeypatch):
    from loguru import logger

    from nanobot.cli.log_control import add_compatibility_log_file
    path = tmp_path / 'runtime.log'
    original = logger.add
    settings = {}
    def capture(*args, **kwargs):
        settings.update(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(logger, 'add', capture)
    handler = add_compatibility_log_file(str(path))
    try:
        logger.info('synthetic-log-check')
        logger.complete()
    finally:
        logger.remove(handler)
    assert settings['rotation'] == '20 MB'
    assert settings['retention'] == '7 days'
    assert settings['compression'] == 'gz'
    assert 'synthetic-log-check' in path.read_text()


def test_partial_websocket_config_remains_disabled():
    cfg = Config(channels={"websocket": {"host": "127.0.0.1"}, "sendProgress": False})
    assert cfg.channels.websocket["enabled"] is False
    assert cfg.channels.task_update_mode == "result"
    assert cfg.agents.defaults.timezone == "UTC"


@pytest.mark.asyncio
async def test_http_stream_rejected_before_execution():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nanobot.api.server import handle_chat_completions
    loop = SimpleNamespace(process_direct=AsyncMock())
    request = SimpleNamespace(content_type="application/json", app={"agent_loop": loop},
                              json=AsyncMock(return_value={"stream": True, "messages": []}))
    response = await handle_chat_completions(request)
    assert response.status == 400
    assert json.loads(response.body)["error"]["code"] == 400
    loop.process_direct.assert_not_awaited()


def test_mcp_missing_only_timeout_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOBOT_AGENTS__DEFAULTS__MAX_TOKENS", "1234")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tools": {"mcpServers": {"ltx-desktop": {"command": "synthetic"}}}}))
    cfg = load_config(path)
    assert cfg.agents.defaults.max_tokens == 1234
    assert cfg.tools.mcp_servers["ltx-desktop"].tool_timeout == 120
    path.write_text(json.dumps({"tools": {"mcpServers": {"ltx-desktop": {"command": "synthetic", "toolTimeout": 9}}}}))
    assert load_config(path).tools.mcp_servers["ltx-desktop"].tool_timeout == 9


@pytest.mark.parametrize("args", [
    ("gateway", "restart"), ("gateway", "install-service"), ("webui",), ("onboard",), ("serve",),
])
def test_mutating_entrypoints_block_before_effects(tmp_path, args):
    result = _cli(tmp_path, *args)
    assert result.returncode != 0
    assert "Migration candidate is not runnable" in result.stderr
    assert list((tmp_path / "home").iterdir()) == []


def test_gateway_help_retains_log_file_option(tmp_path):
    result = _cli(tmp_path, "gateway", "--help")
    assert result.returncode == 0
    assert "--log-file" in result.stdout


def test_legacy_context_and_heartbeat_fields_survive():
    cfg = Config(agents={"defaults": {"contextPath": "/synthetic/context"}},
                 gateway={"heartbeat": {"keepRecentMessages": 13}})
    assert cfg.agents.defaults.context_paths == ["/synthetic/context"]
    assert cfg.gateway.heartbeat.keep_recent_messages == 13


def test_legacy_context_path_appends():
    cfg = Config(agents={"defaults": {"context_paths": ["/one"], "context_path": "/two"}})
    assert cfg.agents.defaults.context_paths == ["/one", "/two"]


def test_sdk_factory_blocks_before_config_loading(monkeypatch):
    import nanobot.config.loader as loader
    from nanobot.nanobot import Nanobot
    def forbidden(*args, **kwargs):
        pytest.fail("config loading happened before admission")
    monkeypatch.setattr(loader, "load_config", forbidden)
    with pytest.raises(MigrationIncompleteError):
        Nanobot.from_config()


def test_desktop_entry_blocks_before_discovery(monkeypatch):
    import nanobot.cli.desktop_tui as desktop
    def forbidden(*args, **kwargs):
        pytest.fail("desktop discovery happened before admission")
    monkeypatch.setattr(desktop, "discover_desktop_target", forbidden)
    with pytest.raises(MigrationIncompleteError):
        desktop.main()
