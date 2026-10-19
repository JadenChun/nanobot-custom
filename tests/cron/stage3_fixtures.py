"""Copied-state Stage3 fixtures; all network and external process effects denied."""

import asyncio
import os
import socket
import subprocess
from types import SimpleNamespace

import pytest

from nanobot.agent.turn import TurnResult
from nanobot.channels.delivery_contract import ChannelTextPartReceipt, ChannelTextReceipt
from nanobot.cron.action_store import CronActionStore
from nanobot.cron.definition_store import CronDefinition, CronDefinitionAction, CronDefinitionStore
from nanobot.cron.runtime import CronRuntime, CronRuntimeInjection
from nanobot.cron.service import CronService
from nanobot.session.manager import SessionManager

POSIX_FORK = getattr(os, "fork", None)


@pytest.fixture(autouse=True)
def offline_denial(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("external effects denied")

    for function in ("connect", "connect_ex", "bind", "sendto"):
        monkeypatch.setattr(socket.socket, function, denied)
    for function in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, function, denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    for function in ("system", "fork", "posix_spawn", "posix_spawnp", "execv", "execve"):
        if hasattr(os, function):
            monkeypatch.setattr(os, function, denied)


class FakeClock:
    def __init__(self, now=1000):
        self.now, self.monotonic = now, 0.0

    def wall_time_ms(self):
        return self.now

    def monotonic_time(self):
        return self.monotonic

    async def sleep(self, seconds):
        self.monotonic += seconds
        await asyncio.sleep(0)


def definition(name="job", *, interval=100, schedule=None, destinations=("A", "B")):
    return CronDefinition.from_definition({"id": "job", "name": name, "enabled": True,
        "schedule": schedule or {"kind": "every", "atMs": None, "everyMs": interval, "expr": None, "tz": None},
        "payload": {"kind": "agent_turn", "message": "private compute", "deliver": bool(destinations),
            "channel": "telegram" if destinations else None, "to": destinations[0] if destinations else None,
            "additionalDestinations": [{"channel": "telegram", "to": target} for target in destinations[1:]],
            "channelMeta": {}, "sessionKey": "cron:job", "executionMode": "legacy_fresh",
            "originChannel": "telegram" if destinations else "cli", "originChatId": destinations[0] if destinations else "direct",
            "originMetadata": {}}, "deleteAfterRun": False}, job_instance_id="a" * 64)


def definitions(tmp_path, clock=None, *, item=None):
    clock = clock or FakeClock()
    store = CronDefinitionStore(CronActionStore(tmp_path / "definitions", namespace="stage3"),
                                commit_clock_ms=clock.wall_time_ms)
    item = item or definition()
    store.append(action_id="create", action=CronDefinitionAction.create(item))
    store.checkpoint(store.replay())
    return store, item, clock


class FakeChannels:
    def __init__(self):
        self.calls = []

    def get_channel(self, name):
        return self if name == "telegram" else None

    async def send_text_with_receipt(self, target, messages, *, action_id, request_digest):
        self.calls.append(target)
        return ChannelTextReceipt(channel="telegram", target=target, action_id=action_id,
            request_digest=request_digest,
            parts=(ChannelTextPartReceipt(0, 0, 0, len(self.calls), 1),))


def runtime(tmp_path, *, authorizer=lambda *_: True, clock=None, agent=None, sessions=None, timeout=0.1):
    store, item, clock = definitions(tmp_path, clock)
    service = CronService(tmp_path / "definitions" / "jobs.json", definition_store=store)
    calls = []

    async def compute(request):
        calls.append(request.run_id)
        return TurnResult(run_id=request.run_id, content="private result")

    agent = agent or SimpleNamespace(execute_silent_cron_turn=compute)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    sessions = sessions or SessionManager(workspace, sessions_root=tmp_path / "sessions")
    channels = FakeChannels()
    graph = CronRuntime(service, agent, sessions, channels, tmp_path / "runtime", "copied-root",
                        CronRuntimeInjection(clock, authorizer, timeout))
    graph.acquire()
    return graph, item, clock, calls, channels
