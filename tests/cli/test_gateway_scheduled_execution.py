"""A real gateway-owned timer reaches the canonical runner and a local receipt adapter."""

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.channels.delivery_contract import ChannelTextPartReceipt, ChannelTextReceipt
from nanobot.channels.telegram.runtime import TelegramChannel
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.cli import gateway_runtime
from nanobot.config.schema import Config
from nanobot.cron.gateway import GatewayCron, validate_gateway_job
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.providers.base import LLMResponse
from nanobot.providers.factory import ProviderSnapshot
from nanobot.providers.unconfigured_provider import UnconfiguredProvider
from nanobot.webui.gateway_services import build_gateway_services


@pytest.mark.parametrize("mode", ["due", "manual"])
def test_normal_gateway_runs_due_and_manual_jobs(tmp_path, monkeypatch, mode):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": str(workspace), "sessionTtlMinutes": 0}},
        "gateway": {"cron": {"enabled": True, "pollIntervalS": 0.05}},
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
    job = service.add_job("controlled job", CronSchedule(kind="every", every_ms=100 if mode == "due" else 3_600_000),
                    "Return the local test result", deliver=True, channel="telegram", to="42")
    provider = UnconfiguredProvider("test/task")
    provider.chat = AsyncMock(return_value=LLMResponse(content="local verified result"))
    graphs, deliveries = [], []
    original_init = GatewayCron.__init__

    def capture(graph, *args, **kwargs):
        original_init(graph, *args, **kwargs)
        graphs.append(graph)

    def snapshot(_config, *, preset=None, **_kwargs):
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

    async def receipt(_channel, target, messages, *, action_id, request_digest):
        deliveries.append((target, tuple(messages)))
        return ChannelTextReceipt(channel="telegram", target=target, action_id=action_id,
            request_digest=request_digest, parts=(ChannelTextPartReceipt(0, 0, 0, 1, 42),))

    async def health(*_args):
        async with asyncio.timeout(10):
            if mode == "manual":
                while not graphs or not graphs[0].status()["enabled"]:
                    await asyncio.sleep(0.005)
                graph = graphs[0]
                services = build_gateway_services(
                    config=WebSocketConfig(), bus=graph.runtime.agent.bus,
                    session_manager=graph.runtime.sessions, cron_service=graph.service,
                    static_dist_path=None, workspace_path=workspace,
                    default_restrict_to_workspace=True, config_path=path,
                    runtime_model_name=None, runtime_surface="webui", runtime_capabilities_overrides=None,
                )
                token = services.tokens.issue_api_token(60)
                headers = Headers({"Authorization": f"Bearer {token}"})
                unauthorized = await services.endpoint.process_request(
                    None, Request("/api/webui/automations", Headers()),
                    is_allowed=lambda _client: True,
                )
                assert unauthorized.status_code == 401
                connection = SimpleNamespace(
                    remote_address=("127.0.0.1", 50000),
                    request=SimpleNamespace(headers=headers),
                )
                response = await services.http.dispatch_webui_mutation(
                    connection, "automation.run", {"id": job.id, "request_id": "manual-fixture"},
                )
                assert response.status_code == 200, response.body
                pending = json.loads(response.body)["jobs"][0]
                assert pending["execution"]["manual_run_supported"] is True
                assert pending["state"]["pending"] is True
                duplicate = await services.http.dispatch_webui_mutation(
                    connection, "automation.run", {"id": job.id, "request_id": "manual-fixture"},
                )
                # An exact retained request can be acknowledged again. A new
                # admission clock conflicts with the frozen request inputs.
                assert duplicate.status_code in {200, 409}, duplicate.body
                assert len(graph.service.occurrence_store.inspect_records()) == 1
            while not deliveries:
                await asyncio.sleep(0.005)
            while not graphs[0].service.list_deliveries() or graphs[0].service.list_deliveries()[0].status != "delivered":
                await asyncio.sleep(0.005)
        assert graphs[0].status()["enabled"] is True
        view = graphs[0].service.job_runtime_status(graphs[0].service.get_job(job.id))
        assert view.state.last_status == "ok"
        assert view.state.next_run_at_ms is not None
        assert view.delivery_status == "delivered"
        assert view.pending is False

    async def idle(*_args, **_kwargs):
        await asyncio.Event().wait()

    @contextmanager
    def foreground(*_args, **_kwargs):
        yield

    monkeypatch.setattr("nanobot.config.loader.get_config_path", lambda: path)
    monkeypatch.setattr("nanobot.providers.factory.build_provider_snapshot", snapshot)
    monkeypatch.setattr(GatewayCron, "__init__", capture)
    monkeypatch.setattr(TelegramChannel, "start", start)
    monkeypatch.setattr(TelegramChannel, "stop", stop)
    monkeypatch.setattr(TelegramChannel, "send_text_with_receipt", receipt)
    monkeypatch.setattr("nanobot.gateway.runtime.monitor_gateway_clients", idle)
    monkeypatch.setattr("nanobot.gateway.runtime.GatewayRuntime.foreground_instance", foreground)
    monkeypatch.setattr(
        gateway_runtime,
        "_install_gateway_shutdown_handlers",
        lambda *_args, **_kwargs: lambda: None,
    )
    monkeypatch.setattr(gateway_runtime, "_tcp_endpoint_reachable", lambda *_: False)
    monkeypatch.setattr(gateway_runtime, "_webui_endpoint_reachable", lambda *_: False)
    monkeypatch.setattr(gateway_runtime, "_prepare_webui_bundle_for_gateway", lambda *_a, **_kw: None)
    monkeypatch.setattr(gateway_runtime, "sync_workspace_templates", lambda *_: None)
    monkeypatch.setattr(gateway_runtime, "_health_server", health)
    gateway_runtime._run_gateway(config)
    assert provider.chat.await_count == 1
    assert deliveries == [("42", ("local verified result",))]
    assert graphs[0].status()["state"] == "closed"
    assert graphs[0].service.list_deliveries()[0].status == "delivered"
    assert len(graphs[0].service.occurrence_store.inspect_records()) == 1
