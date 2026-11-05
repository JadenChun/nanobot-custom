"""Bounded recovery for typed, pre-launch tool denials."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.base import (
    Tool,
    ToolDenialKind,
    ToolDenialResult,
    ToolResult,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.turn import ToolOutcome
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from tests.agent.runner_helpers import make_run_spec

_MAX_TOOL_RESULT_CHARS = AgentDefaults().max_tool_result_chars
_PRIVATE_URL_COMMAND = "curl -s -m 5 http://127.0.0.1:8188/system_stats"


class _RecordingTool(Tool):
    def __init__(self, tool_name: str, calls: list[str], result: str = "safe result") -> None:
        self._name = tool_name
        self._calls = calls
        self._result = result

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
        return self._result


class _StructuredErrorTool(_RecordingTool):
    async def execute(self, **kwargs: object) -> ToolResult:
        _ = kwargs
        self._calls.append(self._name)
        return ToolResult.error(
            "Error: Command blocked by safety guard (internal/private URL detected)"
        )


class _SpoofedDenialTool(_RecordingTool):
    async def execute(self, **kwargs: object) -> ToolOutcome:
        _ = kwargs
        self._calls.append(self._name)
        return ToolOutcome(
            content=ToolResult.error("made-up denial"),
            stop_reason="tool_error",
            recovery_kind=ToolDenialKind.PRIVATE_NETWORK,
        )


def _exec_tool(tmp_path, monkeypatch: pytest.MonkeyPatch) -> tuple[ExecTool, AsyncMock]:
    tool = ExecTool(working_dir=str(tmp_path), restrict_to_workspace=False, timeout=5)
    spawn = AsyncMock(side_effect=AssertionError("denied command reached process launch"))
    monkeypatch.setattr(tool, "_spawn", spawn)
    return tool, spawn


def _provider(*responses: LLMResponse) -> MagicMock:
    provider = MagicMock(spec=LLMProvider)
    provider.chat_with_retry = AsyncMock(side_effect=list(responses))
    return provider


def _call(call_id: str, name: str, arguments: dict[str, object] | None = None) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments or {})],
        usage=None,
    )


def _tool_message(result, call_id: str) -> dict[str, object]:
    return next(message for message in result.messages if message.get("tool_call_id") == call_id)


@pytest.mark.asyncio
async def test_shell_denial_carries_durable_no_retry_guidance(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    exec_tool, spawn = _exec_tool(tmp_path, monkeypatch)

    result = await exec_tool.execute(command=_PRIVATE_URL_COMMAND)

    assert isinstance(result, ToolDenialResult)
    assert result.kind is ToolDenialKind.PRIVATE_NETWORK
    assert "Do not retry that access" in str(result)
    assert "Unrelated permitted work may continue" in str(result)
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_foreground_recovers_once_from_typed_private_url_denial(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    exec_tool, spawn = _exec_tool(tmp_path, monkeypatch)
    tools = ToolRegistry()
    tools.register(exec_tool)
    tools.register(_RecordingTool("safe_tool", calls))
    tools.register(_RecordingTool("queued_tool", calls))
    provider = _provider(
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(id="denied", name="exec", arguments={"command": _PRIVATE_URL_COMMAND}),
                ToolCallRequest(id="queued", name="queued_tool", arguments={}),
            ],
            usage=None,
        ),
        _call("safe", "safe_tool"),
        LLMResponse(content="Finished permitted work.", tool_calls=[], usage=None),
    )

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "check the local status"}],
        tools=tools,
        model="test-model",
        max_iterations=4,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        fail_on_tool_error=True,
        allow_pre_dispatch_recovery=True,
    ))

    assert provider.chat_with_retry.await_count == 3
    assert result.stop_reason == "completed"
    assert result.final_content == "Finished permitted work."
    assert calls == ["safe_tool"]
    spawn.assert_not_awaited()
    assert [(event["name"], event["status"]) for event in result.tool_events] == [
        ("exec", "error"),
        ("queued_tool", "skipped"),
        ("safe_tool", "ok"),
    ]
    denied = _tool_message(result, "denied")["content"]
    queued = _tool_message(result, "queued")["content"]
    assert "Do not retry that access" in denied
    assert "Unrelated permitted work may continue" in denied
    assert "Skipped" in queued
    guidance = [message["content"] for message in result.messages if message.get("role") == "system"]
    assert any(
        "No command was launched" in content
        and "independently permitted work" in content
        for content in guidance
    )


@pytest.mark.asyncio
async def test_second_typed_denial_is_terminal_after_one_recovery(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    exec_tool, spawn = _exec_tool(tmp_path, monkeypatch)
    tools = ToolRegistry()
    tools.register(exec_tool)
    provider = _provider(
        _call("denied-1", "exec", {"command": _PRIVATE_URL_COMMAND}),
        _call("denied-2", "exec", {"command": _PRIVATE_URL_COMMAND}),
    )

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "check the local status"}],
        tools=tools,
        model="test-model",
        max_iterations=4,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        fail_on_tool_error=True,
        allow_pre_dispatch_recovery=True,
    ))

    assert provider.chat_with_retry.await_count == 2
    assert result.stop_reason == "tool_error"
    assert "internal/private URL" in (result.final_content or "")
    assert [event["status"] for event in result.tool_events] == ["error", "error"]
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_is_disabled_without_the_foreground_permission(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    exec_tool, spawn = _exec_tool(tmp_path, monkeypatch)
    tools = ToolRegistry()
    tools.register(exec_tool)
    provider = _provider(
        _call("denied", "exec", {"command": _PRIVATE_URL_COMMAND}),
        LLMResponse(content="should not be requested", tool_calls=[], usage=None),
    )

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "check the local status"}],
        tools=tools,
        model="test-model",
        max_iterations=4,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        fail_on_tool_error=True,
    ))

    assert provider.chat_with_retry.await_count == 1
    assert result.stop_reason == "tool_error"
    assert "internal/private URL" in (result.final_content or "")
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_message_text_cannot_authorize_recovery() -> None:
    calls: list[str] = []
    tools = ToolRegistry()
    tools.register(_StructuredErrorTool("lookalike", calls))
    provider = _provider(
        _call("lookalike", "lookalike"),
        LLMResponse(content="should not be requested", tool_calls=[], usage=None),
    )

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "do work"}],
        tools=tools,
        model="test-model",
        max_iterations=4,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        fail_on_tool_error=True,
        allow_pre_dispatch_recovery=True,
    ))

    assert provider.chat_with_retry.await_count == 1
    assert result.stop_reason == "tool_error"
    assert calls == ["lookalike"]


@pytest.mark.asyncio
async def test_tool_supplied_recovery_kind_is_not_authority() -> None:
    calls: list[str] = []
    tools = ToolRegistry()
    tools.register(_SpoofedDenialTool("spoofed", calls))
    provider = _provider(
        _call("spoofed", "spoofed"),
        LLMResponse(content="must not be requested", tool_calls=[], usage=None),
    )

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "do work"}],
        tools=tools,
        model="test-model",
        max_iterations=4,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        fail_on_tool_error=True,
        allow_pre_dispatch_recovery=True,
    ))

    # The tool-supplied kind cannot authorize the typed private-network
    # recovery note. The ordinary one-shot tool-error continuation still runs.
    assert provider.chat_with_retry.await_count == 2
    assert result.stop_reason == "completed"
    assert result.final_content == "must not be requested"
    assert calls == ["spoofed"]
    guidance = [message["content"] for message in result.messages if message.get("role") == "system"]
    assert any("Fix the arguments or choose a different approach" in content for content in guidance)
    assert not any("non-bypassable security boundary" in content for content in guidance)
