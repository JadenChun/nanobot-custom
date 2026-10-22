"""Canonical command routing retains trusted shell policy and durable history."""

from unittest.mock import AsyncMock

import pytest

from nanobot.agent.tools.shell import ExecTool
from nanobot.bus.events import INBOUND_META_USER_SHELL, InboundMessage
from nanobot.command.builtin import USER_SHELL_COMMAND
from tests.agent.p6_admitted_fixtures import AdmittedGraphs
from tests.tools.p6_message_fixtures import execute_message, make_loop


@pytest.fixture
async def admitted_async(tmp_path):
    with pytest.MonkeyPatch.context() as patches:
        graphs = AdmittedGraphs(tmp_path, patches)
        try:
            yield graphs.acquire
        finally:
            await graphs.close()


@pytest.mark.parametrize("trusted", [False, True])
async def test_source_shell_uses_trusted_handler_and_persists_before_effect(
    tmp_path, admitted_async, monkeypatch, trusted,
):
    loop = await make_loop(tmp_path, admitted_async)
    key = "websocket:shell-test"
    content = f"{USER_SHELL_COMMAND} printf shell-ok"

    async def execute(**kwargs):
        assert kwargs["command"] == "printf shell-ok"
        assert kwargs["working_dir"] == str(loop.workspace)
        snapshot = loop.sessions.read_session_snapshot(key)
        assert snapshot is not None
        assert any(row.get("role") == "user" and row.get("content") == content
                   for row in snapshot.messages)
        return "shell-ok\n\nExit code: 0"

    tool = loop.tools.get("exec")
    assert isinstance(tool, ExecTool)
    effect = AsyncMock(side_effect=execute)
    monkeypatch.setattr(tool, "execute", effect)
    message = InboundMessage(
        channel="websocket", sender_id="test-user", chat_id="shell-test",
        content=content, metadata={"webui": True, INBOUND_META_USER_SHELL: trusted},
    )
    result = await execute_message(loop, message)
    assert loop.sessions.resolve_session_mode(key).mode == "canonical"
    if trusted:
        assert result.content == "shell-ok\n\nExit code: 0"
        assert result.outbound is not None
        assert result.outbound.metadata["render_as"] == "text"
        effect.assert_awaited_once()
    else:
        assert "trusted local client" in result.content
        effect.assert_not_awaited()
    snapshot = loop.sessions.read_session_snapshot(key)
    assert snapshot is not None
    assert sum(row.get("role") == "user" and row.get("content") == content
               for row in snapshot.messages) == 1
    assert any(row.get("role") == "assistant" and row.get("content") == result.content
               for row in snapshot.messages)
