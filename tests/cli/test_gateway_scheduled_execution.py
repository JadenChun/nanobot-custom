"""A real gateway-owned timer reaches the canonical runner and a local receipt adapter."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from nanobot.channels.delivery_contract import ChannelTextPartReceipt, ChannelTextReceipt
from nanobot.channels.telegram.runtime import TelegramChannel
from nanobot.cli import gateway_runtime
from nanobot.config.schema import Config
from nanobot.cron.gateway import GatewayCron, validate_gateway_job
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.providers.base import LLMResponse
from nanobot.providers.factory import ProviderSnapshot
from nanobot.providers.unconfigured_provider import UnconfiguredProvider


def test_normal_gateway_runs_due_job_and_owns_busy_registration(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": str(workspace), "sessionTtlMinutes": 0}},
        "gateway": {"cron": {"enabled": True, "pollIntervalS": 0.05},
                    "telegramBusy": {"enabled": True, "sessionKey": "telegram:42",
                                     "ownerSenderIds": ["42"], "ownerChatIds": ["42"],
                                     "controlProvider": "openai", "controlModel": "test/control"}},
        "channels": {"websocket": {"enabled": False}, "telegram": {
            "enabled": True, "token": "local-fixture", "allowFrom": ["42"], "allowChats": ["42"]}},
        "tools": {"exec": {"enable": False}, "agentBrowser": {"enabled": False},
                  "agentDevice": {"enabled": False}, "desktopUse": {"enabled": False}},
    })
    path = tmp_path / "instance" / "config.json"
    path.parent.mkdir()
    config.bind_source_path(path)
    path.write_text(config.model_dump_json(by_alias=True))
    service = CronService(workspace / "cron" / "jobs.json")
    service.configure_gateway_execution(validate_gateway_job, lambda: {})
    service.add_job("controlled due job", CronSchedule(kind="every", every_ms=100),
                    "Return the local test result", deliver=True, channel="telegram", to="42")
    provider = UnconfiguredProvider("test/task")
    provider.chat = AsyncMock(return_value=LLMResponse(content="local verified result"))
    control = UnconfiguredProvider("test/control")
    control.chat = AsyncMock()
    graphs, deliveries, authenticated = [], [], []
    original_init = GatewayCron.__init__

    def capture(graph, *args, **kwargs):
        original_init(graph, *args, **kwargs)
        graphs.append(graph)

    def snapshot(_config, *, preset=None, **_kwargs):
        selected = provider if preset is None else control
        return ProviderSnapshot(selected, selected.get_default_model(), 20000, ("fixture",))

    async def start(channel):
        channel._running = True
        channel._bot_user_id = 999
        await channel._authenticated_startup(channel)
        assert channel._busy_inbox is not None
        authenticated.append(channel._busy_inbox)
        channel._app = SimpleNamespace()
        channel._app_ready.set()
        await asyncio.Event().wait()

    async def stop(channel):
        channel._running = False
        channel._app_ready.clear()
        channel._app = None

    async def receipt(_channel, target, messages, *, action_id, request_digest):
        deliveries.append((target, tuple(messages)))
        return ChannelTextReceipt(channel="telegram", target=target, action_id=action_id,
            request_digest=request_digest, parts=(ChannelTextPartReceipt(0, 0, 0, 1, 42),))

    async def health(*_args):
        async with asyncio.timeout(10):
            while not deliveries:
                await asyncio.sleep(0.005)
        assert graphs[0].status()["enabled"] is True

    async def idle(*_args, **_kwargs):
        await asyncio.Event().wait()

    @contextmanager
    def foreground(*_args, **_kwargs):
        yield

    monkeypatch.setattr("nanobot.config.loader.get_config_path", lambda: path)
    monkeypatch.setattr("nanobot.providers.factory.build_provider_snapshot", snapshot)
    monkeypatch.setattr("nanobot.channels.telegram.gateway_busy.build_provider_snapshot", snapshot)
    monkeypatch.setattr(GatewayCron, "__init__", capture)
    monkeypatch.setattr(TelegramChannel, "start", start)
    monkeypatch.setattr(TelegramChannel, "stop", stop)
    monkeypatch.setattr(TelegramChannel, "send_text_with_receipt", receipt)
    monkeypatch.setattr("nanobot.gateway.runtime.monitor_gateway_clients", idle)
    monkeypatch.setattr("nanobot.gateway.runtime.GatewayRuntime.foreground_instance", foreground)
    monkeypatch.setattr(gateway_runtime, "_install_gateway_shutdown_handlers", lambda *_: lambda: None)
    monkeypatch.setattr(gateway_runtime, "_tcp_endpoint_reachable", lambda *_: False)
    monkeypatch.setattr(gateway_runtime, "_webui_endpoint_reachable", lambda *_: False)
    monkeypatch.setattr(gateway_runtime, "_prepare_webui_bundle_for_gateway", lambda *_a, **_kw: None)
    monkeypatch.setattr(gateway_runtime, "sync_workspace_templates", lambda *_: None)
    monkeypatch.setattr(gateway_runtime, "_health_server", health)
    gateway_runtime._run_gateway(config)
    assert authenticated and provider.chat.await_count == 1
    assert deliveries == [("42", ("local verified result",))]
    assert graphs[0].status()["state"] == "closed"
    assert graphs[0].service.list_deliveries()[0].status == "delivered"
    assert len(graphs[0].service.occurrence_store.inspect_records()) == 1
