import ast
import subprocess
import sys
from pathlib import Path


def test_config_base_import_does_not_load_config_schema():
    code = """
import sys
from nanobot.config_base import Base
print("nanobot.config.schema" in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_builtin_tool_configs_do_not_depend_on_config_schema_base():
    repo = Path(__file__).resolve().parents[2]
    tool_paths = sorted((repo / "nanobot/agent/tools").glob("*.py"))

    violations = []
    for path in tool_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "nanobot.config.schema":
                continue
            if any(alias.name == "Base" for alias in node.names):
                violations.append(str(path.relative_to(repo)))

    assert violations == []


def test_tool_first_import_preserves_schema_rebuild_and_historical_aliases():
    code = """
from nanobot.agent.tools.web import WebToolsConfig
from nanobot.config.schema import Config, ToolsConfig, WebToolsConfig as HistoricalWebToolsConfig
assert HistoricalWebToolsConfig is WebToolsConfig
assert ToolsConfig.model_rebuild(force=True) is True
assert Config.model_rebuild(force=True) is True
config = Config()
assert type(config.tools.web) is WebToolsConfig
assert config.channels.websocket == {"enabled": False}
assert config.artifacts.enabled is False
assert config.agents.defaults.idle_compaction.enabled is False
assert config.tools.model_dump(by_alias=True)["imageGeneration"]["configurationSource"] == "legacy"
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "ok"
