"""Fresh interpreters must expose complete config types in every import order."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('prefix', [
    '',
    'from nanobot.agent.tools.execution import KnownToolResult',
    'import nanobot.agent.delegation',
    'import nanobot.agent.tools.shell',
    'import nanobot.agent.tools.web',
    'from nanobot.agent import AgentLoop, AgentHook',
])
@pytest.mark.parametrize('first_use', [
    'Config.model_validate({})',
    'Config.model_validate_json("{}")',
    'ToolsConfig.model_validate({})',
    'Config.model_json_schema()',
    'from nanobot.config.schema import ExecToolConfig, WebToolsConfig',
])
def test_fresh_config_validation_after_module_import(prefix, first_use):
    code = prefix + '\nfrom nanobot.config.schema import Config, ToolsConfig\n' + first_use + '\n' + '''
from nanobot.config.schema import Config, ToolsConfig, ExecToolConfig, WebToolsConfig
from nanobot.agent.tools.shell import ExecToolConfig as CanonicalExec
from nanobot.agent.tools.web import WebToolsConfig as CanonicalWeb
assert ExecToolConfig is CanonicalExec
assert WebToolsConfig is CanonicalWeb
for config in (Config(), Config.model_validate({}), Config.model_validate_json("{}")):
    assert isinstance(config.tools.exec, CanonicalExec)
    assert isinstance(config.tools.web, CanonicalWeb)
assert isinstance(ToolsConfig.model_validate({}).web, CanonicalWeb)
assert Config.model_json_schema()
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=30)
    assert result.returncode == 0, result.stderr


def test_tool_import_does_not_initialize_loop_and_public_exports_preserve_identity():
    code = '''
import sys
import nanobot.config.paths
assert 'nanobot.config.schema' not in sys.modules
import nanobot.agent.tools.execution
assert 'nanobot.agent.loop' not in sys.modules
import nanobot.agent as agent
from importlib import import_module
for name in agent.__all__:
    value = getattr(agent, name)
    assert value is getattr(import_module(agent._LAZY_EXPORTS[name], agent.__name__), name)
import nanobot.config as config
for name in config.__all__:
    assert getattr(config, name) is getattr(import_module(config._LAZY_EXPORTS[name], config.__name__), name)
try:
    agent.not_an_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown export accepted')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=30)
    assert result.returncode == 0, result.stderr
