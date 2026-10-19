"""One copied-state runtime graph, real loop/runner, independently retained effects."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.cron.service import CronService
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse
from nanobot.runtime_composition import ChannelBuildOptions, RuntimeComposition, RuntimeFactories
from nanobot.session.manager import Session, SessionManager
from nanobot.session.runtime_authority import OfflineSessionOperator


@pytest.fixture(autouse=True)
def offline_denial(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("external effects denied")

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    for name in ("connect", "connect_ex", "bind", "sendto"):
        monkeypatch.setattr(socket.socket, name, denied)
    for name in ("getaddrinfo", "create_connection"):
        monkeypatch.setattr(socket, name, denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    for name in ("system", "fork", "posix_spawn", "posix_spawnp", "execv", "execve"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, denied)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", denied)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", denied)


@dataclass
class Effects:
    provider: int = 0
    tools: int = 0
    sends: list[str] = field(default_factory=list)
    callbacks: int = 0
    lifecycle: list[str] = field(default_factory=list)


class Provider(LLMProvider):
    def __init__(self, effects: Effects, responses=()):
        super().__init__(provider_name="synthetic")
        self.effects = effects
        self.responses = deque(responses)
        self.messages = []
        self.generation = GenerationSettings(max_tokens=256, temperature=0.1)
        self.entered: dict[str, asyncio.Event] = {}
        self.release: dict[str, asyncio.Event] = {}

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.effects.provider += 1
        self.messages.append(deepcopy(messages))
        # Tests name their user requests; real context/runner construction stays intact.
        text = next((row.get("content", "") for row in reversed(messages)
                     if row.get("role") == "user"), "")
        for label, event in self.entered.items():
            if label in str(text):
                event.set()
                await self.release[label].wait()
                break
        return self.responses.popleft() if self.responses else LLMResponse(content="fake answer")

    def get_default_model(self):
        return "synthetic-model"


class CounterTool(Tool):
    def __init__(self, effects):
        self.effects = effects

    @property
    def name(self):
        return "count_effect"

    @property
    def description(self):
        return "Count one synthetic effect."

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        self.effects.tools += 1
        return "fake tool evidence"


class MCP:
    def __init__(self, effects):
        self.effects = effects

    async def connect(self):
        return None

    async def aclose(self):
        self.effects.lifecycle.append("mcp-close")


class Channels:
    """Exact fake Telegram receipt authority plus real authorized inbound adapter."""
    def __init__(self, config, bus, effects):
        from nanobot.channels.telegram.runtime import TelegramChannel, TelegramConfig

        self.telegram = TelegramChannel(TelegramConfig.model_validate(config.channels.telegram), bus)
        self.effects = effects
        self.fail_send = False

    async def acquire(self):
        self.effects.lifecycle.append("channels-acquire")

    async def stop_all(self):
        await self.telegram.stop()
        self.effects.lifecycle.append("channels-close")

    def get_channel(self, name):
        return self if name == "telegram" else None

    async def send_text_with_receipt(self, target, messages, *, action_id, request_digest):
        from nanobot.channels.delivery_contract import ChannelTextPartReceipt, ChannelTextReceipt

        self.effects.sends.append(target)
        if self.fail_send:
            raise OSError("fake transport lost acknowledgement")
        return ChannelTextReceipt(channel="telegram", target=target, action_id=action_id,
            request_digest=request_digest,
            parts=(ChannelTextPartReceipt(0, 0, 0, len(self.effects.sends), 1),))


def config_at(root: Path) -> Config:
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config = Config()
    config.bind_source_path(root / "instance" / "config.json")
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = "synthetic-model"
    config.agents.defaults.model_preset = None
    config.agents.defaults.max_tool_iterations = 4
    config.tools.agent_browser.enabled = False
    config.tools.agent_device.enabled = False
    config.tools.desktop_use.enabled = False
    config.tools.restrict_to_workspace = True
    config.channels.telegram = {"allowFrom": ["42"], "token": "synthetic-token", "enabled": True}
    return config


async def seed_modes(config: Config, modes: dict[str, str]):
    root = config.runtime_data_dir / "sessions"
    unmanaged = SessionManager(config.workspace_path, sessions_root=root)
    for key in modes:
        unmanaged.save(Session(key, messages=[{"role": "user", "content": "retained fixture"},
                                             {"role": "assistant", "content": "retained answer"}]), fsync=True)
    operator = await OfflineSessionOperator(config.workspace_path, copied_sessions_root=root).acquire()
    try:
        operator.manager.bootstrap_existing_sessions(offline_copy=True)
        for key, mode in modes.items():
            if mode == "canonical":
                operator.manager.convert_legacy_session(key, offline_copy=True)
    finally:
        await operator.close()


def define_graph(config: Config, provider: Provider, effects: Effects, *, cron=False, clock=None,
                 definition_store=None, channels=True):
    def tools():
        registry = ToolRegistry()
        registry.register(CounterTool(effects))
        return registry

    def agent(config, bus, sessions, tools, mcp, service, delivery):
        return AgentLoop.from_config(config, bus, tool_registry=tools, provider=provider,
            session_manager=sessions, cron_service=service, turn_delivery_factory=delivery,
            enable_legacy_auto_compact=False)

    factories = RuntimeFactories(MessageBus, tools, lambda *_: MCP(effects),
        lambda report: CronService(report.workspace / "cron" / "jobs.json",
                                   definition_store=definition_store), agent)
    injection = None
    if cron:
        from nanobot.cron.runtime import CronRuntimeInjection

        injection = CronRuntimeInjection(clock, lambda *_: True, 0.05)
    return RuntimeComposition.define(config, factories=factories,
        channel_options=ChannelBuildOptions(factory=lambda cfg, bus, **_: Channels(cfg, bus, effects))
        if channels else None, cron_injection=injection)


async def joined(awaitable):
    return await asyncio.wait_for(awaitable, timeout=5)


def file_bytes(root: Path):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}
