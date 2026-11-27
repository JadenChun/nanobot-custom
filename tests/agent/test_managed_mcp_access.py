"""Capability truth and bounded recovery for unavailable managed tools."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kohaku.agent.runner import AgentRunner
from kohaku.agent.tools.base import (
    Tool,
    ToolPreparationError,
    ToolPreparationErrorKind,
)
from kohaku.agent.tools.context import RequestContext, request_context
from kohaku.agent.tools.mcp import MCPToolWrapper
from kohaku.agent.tools.registry import (
    ToolDispatchGuard,
    ToolRegistry,
)
from kohaku.config.schema import AgentDefaults
from kohaku.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from kohaku.security.folder_grants import FolderGrantStore
from tests.agent.runner_helpers import make_run_spec

_MAX_TOOL_RESULT_CHARS = AgentDefaults().max_tool_result_chars


class _RecordingTool(Tool):
    def __init__(self, tool_name: str, calls: list[str]) -> None:
        self._name = tool_name
        self._calls = calls

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A permitted fixture tool."

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: object) -> str:
        _ = kwargs
        self._calls.append(self._name)
        return "safe result"


class _DenyGuard(ToolDispatchGuard):
    """A live guard (like Dream's operation fence) that withdraws one tool."""

    def __init__(self, denied: str) -> None:
        self._denied = denied

    def error(self, name: str) -> str | None:
        if name != self._denied:
            return None
        return f"Tool '{name}' is not available in the current run."


def _managed_store(tmp_path: Path) -> tuple[Path, FolderGrantStore]:
    home = tmp_path / "agent-home"
    grant = tmp_path / "granted"
    home.mkdir()
    grant.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    store.upsert(
        {"path": str(grant), "file_access": "read_only", "commands": True},
        expected_revision=0,
    )
    return home, store


def _mcp_tool(server: str, raw_name: str, session: object) -> MCPToolWrapper:
    return MCPToolWrapper(
        session,
        server,
        SimpleNamespace(
            name=raw_name,
            description=f"{raw_name} fixture",
            inputSchema={"type": "object", "properties": {}},
        ),
        tool_timeout=5,
        transport_type="stdio",
        server_command="fixture",
    )


@pytest.mark.asyncio
async def test_managed_chat_has_no_tool_name_allowlist(tmp_path: Path) -> None:
    # Tool availability in a managed chat is the policy layer's decision; the
    # dispatch guard never withholds a tool by name.
    home, store = _managed_store(tmp_path)
    calls: list[str] = []
    registry = ToolRegistry()
    for name in ("agent_browser", "create_goal", "my", "list_dir"):
        registry.register(_RecordingTool(name, calls))
    request = RequestContext(
        channel="websocket",
        chat_id="fixture-chat",
        session_key="websocket:fixture-chat",
        workspace=home,
        persistent_session=True,
    )

    with request_context(request):
        assert store.view.is_managed
        assert all(registry.dispatch_error(name) is None for name in registry.tool_names)
        result = await registry.execute("agent_browser", {})

    assert result == "safe result"
    assert calls == ["agent_browser"]


@pytest.mark.asyncio
async def test_managed_chat_calls_configured_mcp_tool(tmp_path: Path) -> None:
    home, store = _managed_store(tmp_path)
    session = SimpleNamespace(
        call_tool=AsyncMock(return_value=SimpleNamespace(content=[], isError=False))
    )
    search = _mcp_tool("comfy", "search", session)
    registry = ToolRegistry()
    registry.register(search)
    request = RequestContext(
        channel="websocket",
        chat_id="fixture-chat",
        session_key="websocket:fixture-chat",
        workspace=home,
        persistent_session=True,
    )

    with request_context(request):
        result = await registry.execute(search.name, {})

    assert str(result) == "(no output)"
    session.call_tool.assert_awaited_once()
    assert session.call_tool.await_args.args[0] == "search"


@pytest.mark.asyncio
async def test_managed_typo_remains_an_unknown_tool(tmp_path: Path) -> None:
    home, store = _managed_store(tmp_path)
    registry = ToolRegistry()
    registry.register(_RecordingTool("list_dir", []))
    registry.set_dispatch_guard(ToolDispatchGuard())
    request = RequestContext(
        channel="websocket",
        chat_id="fixture-chat",
        session_key="websocket:fixture-chat",
        workspace=home,
        persistent_session=True,
    )

    with request_context(request):
        _tool, _params, error = registry.prepare_call("read_fiel", {})

    assert isinstance(error, ToolPreparationError)
    assert error.kind is ToolPreparationErrorKind.UNKNOWN_TOOL
    assert "not available in the current run" not in str(error)


@pytest.mark.asyncio
async def test_admission_race_denial_gets_one_safe_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, store = _managed_store(tmp_path)
    registry = ToolRegistry()
    registry.register(_RecordingTool("list_dir", []))
    registry.set_dispatch_guard(ToolDispatchGuard())

    @contextmanager
    def denied(_name: str):
        yield "Tool 'list_dir' is not available in the current run."

    monkeypatch.setattr(registry, "dispatch_admission", denied)
    provider = MagicMock(spec=LLMProvider)
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="denied", name="list_dir", arguments={})],
            usage=None,
        ),
        LLMResponse(content="Finished without retrying.", tool_calls=[], usage=None),
    ])
    request = RequestContext(
        channel="websocket",
        chat_id="fixture-chat",
        session_key="websocket:fixture-chat",
        workspace=home,
        persistent_session=True,
    )

    with request_context(request):
        result = await AgentRunner().run(make_run_spec(
            provider,
            initial_messages=[{"role": "user", "content": "do work"}],
            tools=registry,
            model="test-model",
            max_iterations=4,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            fail_on_tool_error=True,
            allow_pre_dispatch_recovery=True,
        ))

    assert provider.chat_with_retry.await_count == 2
    assert result.stop_reason == "completed"
    assert result.final_content == "Finished without retrying."


@pytest.mark.asyncio
async def test_managed_unavailable_tool_gets_one_safe_recovery(tmp_path: Path) -> None:
    home, store = _managed_store(tmp_path)
    _ = store
    calls: list[str] = []
    safe = _RecordingTool("list_dir", calls)
    stale_host_tool = _RecordingTool("agent_browser", calls)
    full_registry = ToolRegistry()
    full_registry.register(safe)
    full_registry.register(stale_host_tool)
    registry = full_registry.filtered(["list_dir"])
    registry.set_dispatch_guard(_DenyGuard("agent_browser"))
    provider = MagicMock(spec=LLMProvider)
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(
                id="unavailable",
                name="agent_browser",
                arguments={},
            )],
            usage=None,
        ),
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="safe", name="list_dir", arguments={})],
            usage=None,
        ),
        LLMResponse(content="Used a permitted tool.", tool_calls=[], usage=None),
    ])
    request = RequestContext(
        channel="websocket",
        chat_id="fixture-chat",
        session_key="websocket:fixture-chat",
        workspace=home,
        persistent_session=True,
    )

    with request_context(request):
        result = await AgentRunner().run(make_run_spec(
            provider,
            initial_messages=[{"role": "user", "content": "check the local status"}],
            tools=registry,
            model="test-model",
            max_iterations=4,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            fail_on_tool_error=True,
            allow_pre_dispatch_recovery=True,
        ))

    assert provider.chat_with_retry.await_count == 3
    assert result.stop_reason == "completed"
    assert result.final_content == "Used a permitted tool."
    assert calls == ["list_dir"]
    recovery_request = provider.chat_with_retry.await_args_list[1].kwargs["messages"]
    tool_message = next(
        message for message in recovery_request
        if message.get("tool_call_id") == "unavailable"
    )
    assert "not available in the current run" in str(tool_message["content"])
    guidance = [
        message["content"] for message in result.messages
        if message.get("role") == "user"
        and str(message.get("content", "")).startswith("[Internal recovery note]")
    ]
    assert len(guidance) == 1
    assert "not available in the current run" in str(guidance[0])
