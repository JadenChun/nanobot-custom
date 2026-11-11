"""Opt-in normal gateway local delivery, including recovery and a busy chat."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.channels.telegram.runtime import TelegramChannel
from nanobot.cli import gateway_runtime
from nanobot.config.schema import Config
from nanobot.providers.base import LLMResponse
from nanobot.providers.factory import ProviderSnapshot
from nanobot.providers.unconfigured_provider import UnconfiguredProvider
from nanobot.triggers.local_store import LocalTriggerStore


@pytest.mark.parametrize("scenario", ["queued", "recovered", "busy", "shutdown"])
def test_gateway_owns_local_trigger_delivery(tmp_path, monkeypatch, scenario):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": str(workspace), "sessionTtlMinutes": 0}},
        "gateway": {"localTriggers": True, "cron": {"enabled": scenario == "shutdown"}},
        "channels": {"websocket": {"enabled": False}, "telegram": {
            "enabled": True, "token": "local-fixture", "allowFrom": ["42"], "allowChats": ["42"]}},
        "tools": {"exec": {"enable": False}, "agentBrowser": {"enabled": False},
                  "agentDevice": {"enabled": False}, "desktopUse": {"enabled": False}},
    })
    path = tmp_path / "instance" / "config.json"
    path.parent.mkdir()
    config.bind_source_path(path)
    path.write_text(config.model_dump_json(by_alias=True))
    store = LocalTriggerStore(workspace)
    trigger = store.create(name="Local result", channel="telegram", chat_id="42", session_key="telegram:42")
    if scenario != "busy":
        store.enqueue(trigger.id, "Review local result")
    if scenario == "recovered":
        assert len(store.claim_deliveries()) == 1
    provider = UnconfiguredProvider("test/task")
    agents, calls, delivered = [], [], []
    release = asyncio.Event()
    foreground_started = asyncio.Event()
    original_run = AgentLoop.run

    async def run(agent):
        agents.append(agent)
        await original_run(agent)

    async def chat(**kwargs):
        calls.append(kwargs)
        if scenario == "shutdown":
            foreground_started.set()
            await asyncio.Event().wait()
        if scenario == "busy" and len(calls) == 1:
            foreground_started.set()
            await release.wait()
        return LLMResponse(content="Local verified response")

    provider.chat = chat

    def snapshot(_config, **_kwargs):
        return ProviderSnapshot(provider, provider.get_default_model(), 20000, ("fixture",))

    async def start(channel):
        channel._running = True
        channel._bot_user_id = 999
        channel._app = SimpleNamespace()
        channel._app_ready.set()
        await asyncio.Event().wait()

    async def stop(channel):
        channel._running = False
        channel._app_ready.clear()
        channel._app = None

    async def send(_channel, message):
        delivered.append(message)

    async def health(*_args):
        async with asyncio.timeout(10):
            while not agents or not agents[0].local_trigger_store.delivery_active:
                await asyncio.sleep(0.005)
            if scenario == "shutdown":
                await foreground_started.wait()
                return
            if scenario == "busy":
                await agents[0].bus.publish_inbound(InboundMessage(
                    channel="telegram", sender_id="42", chat_id="42", content="Foreground work",
                ))
                await foreground_started.wait()
                store.enqueue(trigger.id, "Review local result")
                while trigger.id not in agents[0].pending_local_trigger_ids_for_session("telegram:42"):
                    await asyncio.sleep(0.005)
                assert len(calls) == 1
                release.set()
            while store.get(trigger.id).last_status != "ok":
                await asyncio.sleep(0.005)
            while len(delivered) < (2 if scenario == "busy" else 1):
                await asyncio.sleep(0.005)

    async def idle(*_args, **_kwargs):
        await asyncio.Event().wait()

    @contextmanager
    def foreground(*_args, **_kwargs):
        yield

    monkeypatch.setattr("nanobot.config.loader.get_config_path", lambda: path)
    monkeypatch.setattr("nanobot.providers.factory.build_provider_snapshot", snapshot)
    monkeypatch.setattr(AgentLoop, "run", run)
    monkeypatch.setattr(TelegramChannel, "start", start)
    monkeypatch.setattr(TelegramChannel, "stop", stop)
    monkeypatch.setattr(TelegramChannel, "send", send)
    monkeypatch.setattr("nanobot.gateway.runtime.monitor_gateway_clients", idle)
    monkeypatch.setattr("nanobot.gateway.runtime.GatewayRuntime.foreground_instance", foreground)
    monkeypatch.setattr(gateway_runtime, "_install_gateway_shutdown_handlers", lambda *_a, **_kw: lambda: None)
    monkeypatch.setattr(gateway_runtime, "_tcp_endpoint_reachable", lambda *_: False)
    monkeypatch.setattr(gateway_runtime, "_webui_endpoint_reachable", lambda *_: False)
    monkeypatch.setattr(gateway_runtime, "_prepare_webui_bundle_for_gateway", lambda *_a, **_kw: None)
    monkeypatch.setattr(gateway_runtime, "sync_workspace_templates", lambda *_: None)
    monkeypatch.setattr(gateway_runtime, "_health_server", health)
    if scenario == "shutdown":
        from nanobot.cron.gateway import GatewayCron

        close_cron = GatewayCron.aclose

        async def close_after_trigger_stops(owner):
            assert not agents[0].local_trigger_store.delivery_active
            assert not list(store.processing_dir.glob("*.json"))
            assert len(list(store.inbox_dir.glob("*.json"))) == 1
            await close_cron(owner)

        monkeypatch.setattr(GatewayCron, "aclose", close_after_trigger_stops)
    gateway_runtime._run_gateway(config)
    assert len(calls) == (2 if scenario == "busy" else 1)
    if scenario == "shutdown":
        assert len(list(store.inbox_dir.glob("*.json"))) == 1
        assert not delivered
    else:
        assert store.get(trigger.id).last_status == "ok"
        assert not list(store.inbox_dir.glob("*.json"))
    assert not list(store.processing_dir.glob("*.json"))
    assert agents[0].local_trigger_store.delivery_active is False
