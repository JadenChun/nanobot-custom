"""Positive-only SDK fixtures: real acquisition, caller-owned legacy sessions, no effects."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from collections import deque
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.hooks import create_file_edit_activity_hook
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.turn import TurnResult
from nanobot.nanobot import Nanobot
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse
from nanobot.providers.image_generation import image_gen_provider_configs
from nanobot.runtime_composition import RuntimeComposition, RuntimeFactories
from nanobot.session.manager import SessionManager
from nanobot.session.runtime_authority import BorrowedSessionAuthority


class Provider(LLMProvider):
    def __init__(self, responses=()):
        super().__init__(provider_name="synthetic")
        self.responses = deque(responses)
        self.calls = []
        self.generation = GenerationSettings(max_tokens=8192, temperature=0.1)

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append((messages, tools, model))
        return self.responses.popleft() if self.responses else LLMResponse(content="done")

    def get_default_model(self):
        return "synthetic-model"


class RecordingTool(Tool):
    def __init__(self, name, effects):
        self._name, self.effects = name, effects

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "Record one local no-effect fixture call."

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        self.effects.append(self.name)
        return "safe result"


def turn(content="", *, outbound=None, **kwargs):
    return TurnResult(content=content, final_content=content, outbound=outbound, **kwargs)


@pytest.fixture
def offline_sdk_positive(monkeypatch):
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
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config, **_kwargs: Provider())


@pytest.fixture
async def sdk_bot_factory(monkeypatch, offline_sdk_positive):
    bots = []
    original_define = RuntimeComposition.define

    async def acquired_bot(config_path, *, workspace, provider=None, hooks=(), managed=False,
                           model=None, model_preset=None):
        if provider is not None:
            monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config, **_kwargs: provider)

        def define(config, **kwargs):
            defaults = RuntimeFactories.defaults()

            def agent(config, bus, sessions, tools, mcp, cron, delivery):
                loop = AgentLoop.from_config(
                    config, bus, tool_registry=tools, session_manager=sessions,
                    cron_service=cron, turn_delivery_factory=delivery,
                    enable_legacy_auto_compact=False, hooks=list(hooks),
                    image_generation_provider_configs=image_gen_provider_configs(config),
                    hook_factories=[create_file_edit_activity_hook],
                )
                loop.aclose = AsyncMock(wraps=loop.aclose)
                return loop

            def mcp(config, tools):
                value = defaults.mcp(config, tools)
                value.aclose = AsyncMock(wraps=value.aclose)
                return value

            return original_define(config, factories=replace(defaults, agent=agent, mcp=mcp),
                                   **kwargs)

        monkeypatch.setattr(RuntimeComposition, "define", define)
        # A borrowed manager is explicitly caller-owned, never materialized by define().
        authority = None
        if not managed:
            manager = SessionManager(Path(workspace),
                                     sessions_root=Path(config_path).parent / "sessions")
            authority = BorrowedSessionAuthority(manager)
        bot = Nanobot.from_config(config_path, workspace=workspace,
                                  model=model, model_preset=model_preset,
                                  session_authority=authority)
        assert bot._loop is None and bot._mcp_provider is None
        bots.append(bot)
        await bot._ensure_runtime()
        return bot

    yield acquired_bot
    for bot in reversed(bots):
        await bot.aclose()
