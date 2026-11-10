"""End-to-end regression test for A3-5 child references.

Run a real nested delegation through the normal manager entry points, then
read the child and grandchild sessions through the authorized child-thread
HTTP route. Each response must contain that session's own work.
"""

import asyncio
import itertools
from unittest.mock import MagicMock
from urllib.parse import quote

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.turn import TurnContext
from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.channels.websocket.tests.ws_test_client import InProcessHttpChannel, http_get
from nanobot.session.manager import Session, SessionManager
from nanobot.webui.gateway_services import build_gateway_services

_PORT = itertools.count(42_000)
EXPLORE_OK = (
    "---EXPLORE---\n"
    '{"summary":"grandchild summary","findings":["g"],"references":[],'
    '"open_questions":[],"searched_areas":[],"partial":false}\n'
    "---END---"
)


class NestedRunner:
    def __init__(self, manager):
        self.manager = manager
        self.calls = 0

    async def run(self, spec):
        self.calls += 1
        if self.calls == 1:
            nested = await self.manager.run_explore(
                task="grandchild read-only task",
                thoroughness="quick",
                max_iterations=2,
                turn_context=spec.turn_context,
            )
            return AgentRunResult(
                final_content=f"child result: {nested['summary']}",
                messages=[],
            )
        return AgentRunResult(final_content=EXPLORE_OK, messages=[])


async def test_a3_child_references_e2e(tmp_path):
    workspace = tmp_path / "workspace"
    sessions_root = tmp_path / "sessions"
    workspace.mkdir()
    sessions_root.mkdir()

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    sessions = SessionManager(workspace, sessions_root=sessions_root)
    manager = ForegroundAgentManager(
        provider=provider, workspace=workspace, sessions=sessions
    )
    parent_key = "websocket:parent"
    sessions.save(Session(key=parent_key))

    nested_runner = NestedRunner(manager)
    manager.runner = nested_runner

    await manager.run_worker(
        contract="Do the bounded work with acceptance evidence and validation.",
        write_scope=["out.txt"],
        turn_context=TurnContext(session_key=parent_key, run_id="turn-1"),
    )

    children = manager._delegation_tree.children_of(parent_key)
    assert len(children) == 1
    child = children[0]
    grandchildren = manager._delegation_tree.children_of(child.session_key)
    assert len(grandchildren) == 1
    grandchild = grandchildren[0]

    config = WebSocketConfig.model_validate(
        {
            "enabled": True,
            "allowFrom": ["*"],
            "host": "127.0.0.1",
            "port": next(_PORT),
            "path": "/",
            "websocketRequiresToken": False,
        }
    )
    bus = MessageBus()
    gateway = build_gateway_services(
        config=config,
        bus=bus,
        session_manager=sessions,
        static_dist_path=None,
        workspace_path=workspace,
        default_restrict_to_workspace=False,
        runtime_model_name=None,
        runtime_surface="browser",
        runtime_capabilities_overrides=None,
    )
    channel = InProcessHttpChannel(config, bus, gateway=gateway)
    server_task = asyncio.create_task(channel.start())
    try:
        token = channel.gateway.tokens.issue_api_token(300)
        headers = {"Authorization": f"Bearer {token}"}

        async def read(key: str):
            encoded = quote(key, safe="")
            url = (
                f"http://127.0.0.1:{config.port}"
                f"/api/sessions/{encoded}/webui-thread"
            )
            response = await http_get(url, headers=headers)
            assert response.status_code == 200, response.text
            return response.json()

        child_body = await read(child.session_key)
        grandchild_body = await read(grandchild.session_key)

        child_text = "\n".join(
            str(message.get("content") or "")
            for message in child_body["messages"]
        )
        grandchild_text = "\n".join(
            str(message.get("content") or "")
            for message in grandchild_body["messages"]
        )

        assert "child result" in child_text or "bounded work" in child_text
        assert "grandchild" in grandchild_text
        assert child_text != grandchild_text
    finally:
        await channel.stop()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
