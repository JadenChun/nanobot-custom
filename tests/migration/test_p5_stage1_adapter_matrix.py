from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from typer.testing import CliRunner

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.api.server import handle_chat_completions
from nanobot.bus.events import InboundMessage
from nanobot.config.schema import Config
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse
from nanobot.session.runtime_authority import (
    SessionAuthorityUnavailableError,
    Stage1UnavailableSessionAuthority,
)


def _tree(root: Path):
    if not root.exists():
        return ()
    return tuple(
        (
            str(path.relative_to(root)),
            path.lstat().st_ino,
            path.lstat().st_mtime_ns,
            path.read_bytes() if path.is_file() else None,
        )
        for path in sorted(root.rglob("*"))
    )


def _install_default_cutoff_sentinels(monkeypatch):
    from nanobot.session.manager import JsonlSessionStore, SessionManager

    def unreachable(*_args, **_kwargs):
        raise AssertionError("persistent session construction is unreachable")

    monkeypatch.setattr(SessionManager, "__init__", unreachable)
    monkeypatch.setattr(JsonlSessionStore, "__init__", unreachable)
    monkeypatch.setattr(JsonlSessionStore, "_migrate_from_workspace", unreachable)
    monkeypatch.setattr(asyncio, "create_task", unreachable)


def _inject_stage1_cutoff(monkeypatch: pytest.MonkeyPatch) -> None:
    from nanobot.runtime_composition import RuntimeComposition

    define = RuntimeComposition.define.__func__

    def with_cutoff(cls, config, **kwargs):
        if kwargs.get("session_authority") is None:
            kwargs["session_authority"] = Stage1UnavailableSessionAuthority()
        return define(cls, config, **kwargs)

    monkeypatch.setattr(RuntimeComposition, "define", classmethod(with_cutoff))


class _ScriptedProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__(provider_name="synthetic")
        self.responses = deque(responses)
        self.generation = GenerationSettings(max_tokens=256, temperature=0.1)

    async def chat(self, messages, tools=None, model=None, **kwargs):
        assert self.responses, "unexpected extra provider request"
        return self.responses.popleft()

    def get_default_model(self):
        return "synthetic-model"


@pytest.fixture
def engine_config(tmp_path, monkeypatch):
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr(
        "subprocess.Popen",
        Mock(side_effect=AssertionError("process execution")),
    )
    for target in (
        "socket.socket.connect",
        "socket.socket.connect_ex",
        "socket.create_connection",
        "socket.getaddrinfo",
    ):
        monkeypatch.setattr(target, Mock(side_effect=AssertionError("outbound network access")))
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        AsyncMock(side_effect=AssertionError("process execution")),
    )
    monkeypatch.setattr(
        "asyncio.create_subprocess_shell",
        AsyncMock(side_effect=AssertionError("process execution")),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config()
    config.bind_source_path(tmp_path / "runtime" / "config.json")
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = "synthetic-model"
    config.agents.defaults.model_preset = None
    config.tools.agent_browser.enabled = False
    config.tools.agent_device.enabled = False
    config.tools.desktop_use.enabled = False
    return config


class _ApiRequest:
    content_type = "application/json"

    def __init__(self, loop: AgentLoop) -> None:
        self.app = {
            "agent_loop": loop,
            "model_name": "synthetic-model",
            "request_timeout": 1.0,
        }

    async def json(self):
        return {
            "messages": [{"role": "user", "content": "one"}],
            "session_id": "adapter-matrix",
        }


@pytest.mark.parametrize(
    "surface",
    [
        "sdk-run",
        "sdk-stream",
        "bus",
        "api",
    ],
)
async def test_direct_adapter_matrix_delegates_exactly_one_execute_turn(
    engine_config,
    surface,
):
    """Every foreground adapter has one canonical turn-execution boundary."""
    provider = _ScriptedProvider([LLMResponse(content=f"{surface} result")])
    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=provider,
    )
    original = loop.execute_turn
    loop.execute_turn = AsyncMock(wraps=original)  # type: ignore[method-assign]
    try:
        if surface == "sdk-run":
            from nanobot.nanobot import Nanobot

            await Nanobot(loop).run("one", session_key="sdk:adapter-matrix")
        elif surface == "sdk-stream":
            from nanobot.nanobot import Nanobot

            streamed = await Nanobot(loop).run_streamed(
                "one",
                session_key="sdk:adapter-matrix-stream",
            )
            await streamed.wait()
        elif surface == "api":
            response = await handle_chat_completions(_ApiRequest(loop))
            assert response.status == 200
            assert json.loads(response.body)["choices"][0]["message"]["content"] == (
                "api result"
            )
        else:
            await loop._dispatch(
                InboundMessage(
                    channel="discord",
                    sender_id="user",
                    chat_id="adapter-matrix",
                    content="one",
                )
            )
        loop.execute_turn.assert_awaited_once()  # type: ignore[attr-defined]
    finally:
        await loop.aclose()


def test_classic_cli_orchestration_delegates_exactly_one_execute_turn(
    engine_config,
    monkeypatch,
):
    from nanobot.cli import agent as cli_agent

    # The CLI validates settings before entering the scripted composition.
    engine_config.agents.defaults.provider = "custom"
    engine_config.providers.custom.api_base = "https://provider.invalid/v1"
    engine_config.providers.custom.api_key = "synthetic-test-key"
    provider = _ScriptedProvider([LLMResponse(content="classic result")])
    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=provider,
    )
    original = loop.execute_turn
    loop.execute_turn = AsyncMock(wraps=original)  # type: ignore[method-assign]

    class MCP:
        async def connect(self):
            pass

    class Composition:
        async def acquire(self):
            return SimpleNamespace(agent=loop, mcp=MCP())

        async def aclose(self):
            await loop.aclose()

    class Renderer:
        streamed = False
        header_printed = False

        def __init__(self, **_kwargs):
            pass

        async def on_delta(self, _content):
            pass

        async def on_end(self, **_kwargs):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(cli_agent, "_load_runtime_config", lambda *_args: engine_config)
    monkeypatch.setattr(cli_agent, "StreamRenderer", Renderer)
    monkeypatch.setattr(cli_agent, "sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr(cli_agent, "is_default_workspace", lambda _path: False)
    monkeypatch.setattr(cli_agent, "consume_restart_notice_from_env", lambda: None)
    monkeypatch.setattr(
        "nanobot.runtime_composition.RuntimeComposition.define",
        lambda *_args, **_kwargs: Composition(),
    )
    cli_agent.agent(
        message="one",
        session_id="cli:adapter-matrix",
        workspace=None,
        config=None,
        markdown=False,
        logs=False,
        log_file=None,
        classic=True,
        theme="auto",
    )
    loop.execute_turn.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.parametrize("surface", ["classic-cli", "gateway", "api", "sdk"])
def test_foreground_adapters_run_real_session_preflight_before_acquisition(
    monkeypatch, tmp_path, surface,
):
    workspace = tmp_path / "workspace"
    config_data = {
        "agents": {"defaults": {
            "workspace": str(workspace),
            "model": "ollama/llama3.2",
            "provider": "ollama",
            "sessionTtlMinutes": 15,
        }},
        "providers": {"ollama": {"apiBase": "http://localhost:11434/v1"}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    config_bytes = config_path.read_bytes()
    config = Config.model_validate(config_data)
    config.bind_source_path(config_path)
    before = _tree(tmp_path)
    from nanobot.runtime_composition import RuntimePreflightError

    if surface == "classic-cli":
        from nanobot.cli.commands import app

        result = CliRunner().invoke(
            app, ["agent", "-m", "offline", "--config", str(config_path)],
        )
        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit)
        assert result.exception.code == 1
        assert "idle" in result.output.lower()
    elif surface == "gateway":
        from nanobot.cli.gateway_runtime import _run_gateway

        with pytest.raises(RuntimePreflightError, match="idle"):
            _run_gateway(config)
    elif surface == "api":
        from nanobot.cli.commands import app

        result = CliRunner().invoke(
            app, ["serve", "--config", str(config_path)],
        )
        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit)
        assert result.exception.code == 1
        assert "idle" in result.output.lower()
    else:
        from nanobot.config import loader
        from nanobot.nanobot import Nanobot

        monkeypatch.setattr(loader, "load_config", lambda *_args, **_kwargs: config)
        with pytest.raises(RuntimePreflightError, match="idle"):
            Nanobot.from_config()

    assert _tree(tmp_path) == before
    assert config_path.read_bytes() == config_bytes
    assert not workspace.exists()


async def test_sdk_from_config_is_definition_only_and_default_fails_closed(
    monkeypatch,
    tmp_path,
):
    from nanobot.nanobot import Nanobot

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": str(tmp_path / "workspace")}},
    })
    config.bind_source_path(tmp_path / "runtime" / "config.json")
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path: config)
    _install_default_cutoff_sentinels(monkeypatch)
    _inject_stage1_cutoff(monkeypatch)
    before = _tree(tmp_path)
    bot = Nanobot.from_config()
    assert bot._loop is None
    with pytest.raises(SessionAuthorityUnavailableError):
        await bot.run("closed")
    assert _tree(tmp_path) == before
    await bot.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await bot.run("closed again")


def test_gateway_default_authority_precedes_provider_plugin_and_listener(
    monkeypatch,
    tmp_path,
):
    from nanobot.cli.gateway_runtime import _run_gateway

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr(
        "nanobot.providers.factory.build_provider_snapshot",
        lambda *_args, **_kwargs: pytest.fail("provider must remain unreachable"),
    )
    monkeypatch.setattr(
        "nanobot.channels.manager.ChannelManager.__init__",
        lambda *_args, **_kwargs: pytest.fail("channels must remain unreachable"),
    )
    monkeypatch.setattr(
        "asyncio.start_server",
        AsyncMock(side_effect=AssertionError("listener must remain unreachable")),
    )
    monkeypatch.setattr(
        "nanobot.cli.gateway_runtime.sync_workspace_templates",
        lambda *_args: pytest.fail("workspace mutation must remain unreachable"),
    )
    _install_default_cutoff_sentinels(monkeypatch)
    _inject_stage1_cutoff(monkeypatch)
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": str(tmp_path / "workspace")}},
    })
    config.bind_source_path(tmp_path / "runtime" / "config.json")
    before = _tree(tmp_path)
    with pytest.raises(SessionAuthorityUnavailableError):
        _run_gateway(config)
    assert _tree(tmp_path) == before


def test_classic_cli_default_cutoff_precedes_mutation_and_construction(
    monkeypatch,
    tmp_path,
):
    from nanobot.cli import agent as cli_agent

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": str(tmp_path / "workspace")}},
    })
    # Valid settings let this test reach the session-authority cutoff itself.
    config.agents.defaults.provider = "custom"
    config.providers.custom.api_base = "https://provider.invalid/v1"
    config.providers.custom.api_key = "synthetic-test-key"
    config.bind_source_path(tmp_path / "runtime" / "config.json")
    monkeypatch.setattr(cli_agent, "_load_runtime_config", lambda *_args: config)
    monkeypatch.setattr(
        cli_agent,
        "sync_workspace_templates",
        lambda *_args: pytest.fail("workspace mutation must remain unreachable"),
    )
    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider",
        lambda *_args, **_kwargs: pytest.fail("provider must remain unreachable"),
    )
    monkeypatch.setattr(
        AgentLoop,
        "from_config",
        lambda *_args, **_kwargs: pytest.fail("agent must remain unreachable"),
    )
    _install_default_cutoff_sentinels(monkeypatch)
    _inject_stage1_cutoff(monkeypatch)
    before = _tree(tmp_path)
    with pytest.raises(SessionAuthorityUnavailableError):
        cli_agent.agent(
            message="closed",
            session_id=None,
            workspace=None,
            config=None,
            markdown=False,
            logs=False,
            log_file=None,
            classic=True,
            theme="auto",
        )
    assert _tree(tmp_path) == before


def test_api_default_cutoff_precedes_mutation_provider_and_listener(
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("aiohttp")
    from aiohttp import web

    from nanobot.cli import commands

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": str(tmp_path / "workspace")}},
    })
    config.bind_source_path(tmp_path / "runtime" / "config.json")
    monkeypatch.setattr(commands, "_load_runtime_config", lambda *_args: config)
    monkeypatch.setattr(
        commands,
        "sync_workspace_templates",
        lambda *_args: pytest.fail("workspace mutation must remain unreachable"),
    )
    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider",
        lambda *_args, **_kwargs: pytest.fail("provider must remain unreachable"),
    )
    _install_default_cutoff_sentinels(monkeypatch)
    _inject_stage1_cutoff(monkeypatch)

    def run_app(app, **_kwargs):
        async def setup():
            runner = web.AppRunner(app)
            try:
                await runner.setup()
            finally:
                await runner.cleanup()

        asyncio.run(setup())

    monkeypatch.setattr(web, "run_app", run_app)
    before = _tree(tmp_path)
    with pytest.raises(SessionAuthorityUnavailableError):
        commands.serve(
            port=None,
            host=None,
            timeout=None,
            verbose=False,
            workspace=None,
            config=None,
        )
    assert _tree(tmp_path) == before


async def test_api_cleanup_context_closes_on_later_startup_failure():
    pytest.importorskip("aiohttp")
    from aiohttp import web

    from nanobot.api.server import create_app

    events: list[str] = []
    mcp = SimpleNamespace(connect=AsyncMock(side_effect=lambda: events.append("connect")))

    class Composition:
        async def acquire(self):
            events.append("acquire")
            return SimpleNamespace(agent=object(), mcp=mcp)

        async def aclose(self):
            events.append("close")

    app = create_app(composition=Composition())

    async def fail(_app):
        events.append("later-startup")
        raise RuntimeError("later startup failed")

    app.on_startup.append(fail)
    runner = web.AppRunner(app)
    with pytest.raises(RuntimeError, match="later startup failed"):
        await runner.setup()
    await runner.cleanup()
    assert events == ["acquire", "connect", "later-startup", "close"]


def test_injected_create_app_agent_compatibility_is_preserved():
    pytest.importorskip("aiohttp")
    from nanobot.api.server import _AGENT_LOOP_KEY, create_app

    agent = object()
    app = create_app(agent)
    assert app[_AGENT_LOOP_KEY] is agent


def test_gateway_post_acquire_setup_failure_closes_before_tasks_or_listener(
    monkeypatch,
    tmp_path,
):
    from nanobot.cli.gateway_runtime import _run_gateway

    events: list[str] = []

    class Agent:
        async def run(self):
            events.append("agent-task")

        def stop(self):
            events.append("agent-stop")

    class Bus:
        async def drain(self):
            events.append("drain")

    class Composition:
        async def acquire(self):
            events.append("acquire")
            return SimpleNamespace(
                agent=Agent(), channels=SimpleNamespace(), bus=Bus(),
                mcp=SimpleNamespace(connect=AsyncMock()),
            )

        async def aclose(self):
            events.append("close")

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr(
        "nanobot.runtime_composition.RuntimeComposition.define",
        lambda *_args, **_kwargs: Composition(),
    )
    monkeypatch.setattr(
        "nanobot.cli.gateway_runtime._install_gateway_shutdown_handlers",
        lambda *_args, **_kwargs: lambda: events.append("restore"),
    )
    monkeypatch.setattr(
        "nanobot.cli.gateway_runtime._tcp_endpoint_reachable",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LookupError("setup failed")),
    )
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": str(tmp_path / "workspace")}},
    })
    config.bind_source_path(tmp_path / "runtime" / "config.json")
    with pytest.raises(LookupError, match="setup failed"):
        _run_gateway(config)
    assert events == ["acquire", "agent-stop", "close", "drain", "restore"]
