"""Follow real delegation references from a gateway chat."""

import asyncio
from collections import deque
from urllib.parse import quote

from nanobot.channels.websocket.runtime import WebSocketChannel, WebSocketConfig
from nanobot.channels.websocket.tests.ws_test_client import WsTestClient, http_get
from nanobot.config import loader
from nanobot.config.schema import Config
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.runtime_composition import RuntimeComposition
from nanobot.session.webui_turns import WebuiTurnCoordinator
from nanobot.webui.gateway_services import build_gateway_services


class DelegatingProvider(LLMProvider):
    def __init__(self):
        super().__init__(provider_name="delegation-test")
        self.responses = deque([
            LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id="delegate",
                name="delegate_task",
                arguments={
                    "contract": "Inspect the fixture with a delegate and report the result.",
                    "write_scope": ["out.txt"],
                },
            )]),
            LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id="explore",
                name="explore",
                arguments={"task": "Read the fixture", "thoroughness": "quick"},
            )]),
            LLMResponse(content=(
                "---EXPLORE---\n"
                '{"summary":"grandchild result","findings":["fixture"],"references":[],'
                '"open_questions":[],"searched_areas":[],"partial":false}\n'
                "---END---"
            )),
            LLMResponse(content="child report"),
            LLMResponse(content="parent report"),
        ])

    async def chat(self, messages, tools=None, model=None, **kwargs):
        return self.responses.popleft()

    def estimate_prompt_tokens(self, messages, tools=None, model=None):
        return len(messages) + 1, "delegation-fixture"

    def get_default_model(self):
        return "delegation-fixture"


async def test_a3_child_references_e2e(tmp_path, monkeypatch, unused_tcp_port):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    config_path = runtime_dir / "config.json"
    config = Config()
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = "delegation-fixture"
    config.agents.defaults.dream.enabled = False
    config.gateway.heartbeat.enabled = False
    config.artifacts.enabled = False
    config.tools.restrict_to_workspace = True
    config.bind_source_path(config_path)
    monkeypatch.setattr(loader, "_current_config_path", config_path)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: DelegatingProvider())

    composition = RuntimeComposition.define(
        config, sessions_root=runtime_dir / "sessions", startup_receipt_admitted=True,
    )
    handle = await composition.acquire()
    ws_config = WebSocketConfig(enabled=True, allow_from=["*"], port=unused_tcp_port)
    gateway = build_gateway_services(
        config=ws_config,
        bus=handle.bus,
        session_manager=handle.sessions,
        static_dist_path=None,
        workspace_path=workspace,
        default_restrict_to_workspace=True,
        config_path=config_path,
        runtime_model_name=None,
        runtime_surface="browser",
        runtime_capabilities_overrides=None,
    )
    channel = WebSocketChannel(ws_config, handle.bus, gateway=gateway)
    coordinator = WebuiTurnCoordinator(
        bus=handle.bus, sessions=handle.sessions, schedule_background=handle.agent.schedule_background,
    )
    unsubscribe = coordinator.subscribe()

    async def deliver_outbound():
        while True:
            await channel.send(await handle.bus.consume_outbound())

    server = asyncio.create_task(channel.start())
    agent = asyncio.create_task(handle.agent.run())
    outbound = asyncio.create_task(deliver_outbound())
    try:
        ws_token = gateway.tokens.issue_token(60, audience="webui")
        api_token = gateway.tokens.issue_api_token(60)
        async with asyncio.timeout(10), WsTestClient(
            f"ws://127.0.0.1:{unused_tcp_port}{ws_config.path}", token=ws_token,
        ) as client:
            await client.recv_ready()
            await client.send_json({"type": "new_chat"})
            while (attached := await client.recv_raw()).get("event") != "attached":
                pass
            chat = attached["chat_id"]
            await client.send_json({"type": "message", "chat_id": chat, "content": "Delegate this task."})
            references = {}
            while True:
                event = await client.recv_raw()
                if event.get("event") == "delegated_child":
                    references[event["parent_session_key"]] = event["child_session_key"]
                if event.get("event") == "turn_end":
                    break

        child_key = references[f"websocket:{chat}"]
        grandchild_key = references[child_key]
        for key, expected in ((child_key, "child report"), (grandchild_key, "grandchild result")):
            response = await http_get(
                f"http://127.0.0.1:{unused_tcp_port}/api/sessions/{quote(key, safe='')}/webui-thread",
                headers={"Authorization": f"Bearer {api_token}"},
            )
            assert response.status_code == 200, response.text
            assert any(expected in str(message.get("content")) for message in response.json()["messages"])
    finally:
        unsubscribe()
        for task in (agent, outbound):
            task.cancel()
        await asyncio.gather(agent, outbound, return_exceptions=True)
        await channel.stop()
        await asyncio.gather(server, return_exceptions=True)
        await composition.aclose()
