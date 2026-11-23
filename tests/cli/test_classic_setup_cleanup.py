"""Requirements test: classic agent setup failure releases the runtime graph once.

If MCP connect or terminal prompt-session setup fails, the acquired runtime graph
must be released exactly once and the command must exit non-zero. The model loop
must never start.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from kohaku.agent.tools.mcp import MCPProvider
from kohaku.cli.commands import app
from kohaku.config.schema import Config
from kohaku.runtime_composition import RuntimeComposition
from kohaku.session.runtime_authority import SessionAuthorityLease

started: list[str] = []
closes: list[str] = []


class _FakeAgentLoop:
    def __init__(self, bus):
        self.bus = bus
        self.channels_config = None

    @classmethod
    def from_config(cls, config, bus, **kwargs):
        instance = cls(bus)
        instance.session_manager = kwargs["session_manager"]
        instance.turn_delivery_factory = kwargs["turn_delivery_factory"]
        return instance

    async def run(self):
        started.append("agent-run")

    def stop(self):
        pass

    async def aclose(self):
        closes.append("agent-aclose")


class _Authority:
    async def acquire(self, *args, **kwargs):
        return SessionAuthorityLease(object())


def _build_config(tmp_path):
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    config.agents.defaults.provider = "custom"
    config.providers.custom.api_base = "https://provider.invalid/v1"
    config.providers.custom.api_key = "synthetic-test-key"
    return config


def _patch_common(monkeypatch, config):
    original_define = RuntimeComposition.define

    def define(config_arg, **kwargs):
        return original_define(config_arg, session_authority=_Authority(), **kwargs)

    monkeypatch.setattr(RuntimeComposition, "define", staticmethod(define))
    for target in (
        "kohaku.cli.agent.AgentLoop",
        "kohaku.agent.loop.AgentLoop",
        "kohaku.cli.commands.AgentLoop",
    ):
        monkeypatch.setattr(target, _FakeAgentLoop, raising=False)
    monkeypatch.setattr("kohaku.cli.agent._load_runtime_config", lambda *a, **k: config)
    monkeypatch.setattr("kohaku.cli.agent.sync_workspace_templates", lambda *a, **k: None)
    monkeypatch.setattr("kohaku.cli.agent.is_default_workspace", lambda *a, **k: False)
    monkeypatch.setattr("kohaku.cli.agent.configure_agent_runtime_logs", lambda *a, **k: None)
    monkeypatch.setattr("kohaku.migration_guard.require_runtime_ready", lambda *a, **k: None)
    monkeypatch.setattr("kohaku.cli.commands.set_cli_process_identity", lambda *a, **k: None)
    monkeypatch.setattr("kohaku.cli.agent._model_display", lambda *a, **k: ("test-model", ""))
    monkeypatch.setattr(
        "kohaku.cli.agent.consume_restart_notice_from_env", lambda *a, **k: None
    )
    monkeypatch.setattr("kohaku.providers.factory.make_provider", lambda *a, **k: object())
    monkeypatch.setattr(
        "kohaku.providers.image_generation.image_gen_provider_configs", lambda *a, **k: []
    )
    monkeypatch.setattr("kohaku.cron.service.CronService", lambda *a, **k: object())
    monkeypatch.setattr("kohaku.cli.agent.signal.signal", lambda *a, **k: None)
    monkeypatch.setattr("kohaku.cli.terminal._flush_pending_tty_input", lambda *a, **k: None)
    monkeypatch.setattr("kohaku.cli.terminal._restore_terminal", lambda *a, **k: None)


@pytest.mark.parametrize("failure", ["mcp_connect", "prompt_init"])
def test_classic_setup_failure_releases_graph_once(monkeypatch, tmp_path, failure):
    started.clear()
    closes.clear()
    config = _build_config(tmp_path)
    _patch_common(monkeypatch, config)

    async def connect_ok(self):
        return None

    async def connect_fail(self):
        raise OSError("mcp connect failed")

    def prompt_ok(*args, **kwargs):
        return None

    def prompt_fail(*args, **kwargs):
        raise OSError("prompt setup failed")

    async def no_prompt(*args, **kwargs):
        raise AssertionError("must not prompt")

    monkeypatch.setattr(
        MCPProvider, "connect", connect_fail if failure == "mcp_connect" else connect_ok
    )
    monkeypatch.setattr(
        "kohaku.cli.terminal._init_prompt_session",
        prompt_fail if failure == "prompt_init" else prompt_ok,
    )
    monkeypatch.setattr("kohaku.cli.terminal._read_interactive_input_async", no_prompt)

    result = CliRunner().invoke(app, ["agent", "--classic", "--session", "cli:probe"])

    assert result.exit_code != 0
    assert closes == ["agent-aclose"]
    assert started == []
