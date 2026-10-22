"""The SDK result facade preserves exact calls and canonical execution behavior."""

from __future__ import annotations

import asyncio
import inspect
import json
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.hook import AgentHook
from nanobot.agent.loop import AgentLoop
from nanobot.agent.turn import TraceMode, TurnResult, TurnSource
from nanobot.sdk.runtime import build_process_direct_kwargs
from tests.migration import p5_sdk_compat_fixtures as compat

offline_sdk_positive = compat.offline_sdk_positive
sdk_bot_factory = compat.sdk_bot_factory
pytestmark = pytest.mark.usefixtures("offline_sdk_positive")


@pytest.fixture
async def bot(tmp_path, sdk_bot_factory):
    config_path = tmp_path / "instance" / "config.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({
        "providers": {"openrouter": {"apiKey": "sk-test-key"}},
        "agents": {"defaults": {"model": "openai/gpt-4.1"}},
        "tools": {"agentBrowser": {"enabled": False}, "agentDevice": {"enabled": False},
                  "desktopUse": {"enabled": False}},
    }))
    return await sdk_bot_factory(config_path, workspace=tmp_path / "workspace")


def test_optional_schema_matches_existing_direct_implementation():
    from nanobot.agent.direct_execution import DirectExecutionOptions, DirectRequestOptions

    implementation_keys = set(inspect.signature(AgentLoop._execute_direct).parameters) - {
        "self", "content",
    }
    assert set(DirectExecutionOptions.__annotations__) == implementation_keys
    assert DirectExecutionOptions.__required_keys__ == frozenset()
    assert DirectExecutionOptions.__optional_keys__ == implementation_keys
    assert set(DirectRequestOptions.__annotations__) == implementation_keys - {
        "hooks", "source", "trace_mode",
    }


@pytest.mark.parametrize("mode", ["omitted", "explicit-none", "all-options"])
async def test_public_facade_keeps_exact_kwargs_and_late_bound_result(bot, mode):
    loop = bot._loop

    async def callback(*_args, **_kwargs):
        pass

    kwargs = {}
    if mode == "explicit-none":
        kwargs = {"media": None, "attributes": None, "runtime": None, "hooks": None}
    elif mode == "all-options":
        kwargs = {
            "session_key": "sdk:options", "channel": "sdk", "chat_id": "chat",
            "sender_id": "alice", "media": ["input.png"], "on_progress": callback,
            "on_stream": callback, "on_stream_end": callback, "ephemeral": True,
            "_run_extra_hooks_for_ephemeral": True, "hooks": [AgentHook()],
            "hook_factories": None, "tools": loop.tools, "persist_user_message": False,
            "runtime": loop.llm_runtime(), "on_runtime_admitted": callback,
            "attributes": {"tenant": "caller"}, "source": TurnSource.SDK,
            "trace_mode": TraceMode.SANITIZED,
        }
    first = TurnResult(content="first", final_content="first", run_id="first")
    second = TurnResult(content="second", final_content="second", run_id="second")
    first_call = AsyncMock(return_value=first)
    loop._execute_direct = first_call
    assert await loop.process_direct_result("input", **kwargs) is first
    first_call.assert_awaited_once_with("input", **kwargs)
    second_call = AsyncMock(return_value=second)
    loop._execute_direct = second_call
    assert await loop.process_direct_result("input", **kwargs) is second
    second_call.assert_awaited_once_with("input", **kwargs)
    first_call.assert_awaited_once()


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError, asyncio.CancelledError])
async def test_public_facade_propagates_fault_or_cancellation_identity(bot, failure_type):
    failure = failure_type("direct-fault")
    call = AsyncMock(side_effect=failure)
    bot._loop._execute_direct = call
    with pytest.raises(failure_type) as captured:
        await bot._loop.process_direct_result("input", session_key="sdk:fault")
    assert captured.value is failure
    call.assert_awaited_once_with("input", session_key="sdk:fault")


async def test_real_public_result_preserves_canonical_content_and_history(bot):
    result = await bot._loop.process_direct_result(
        "input", session_key="sdk:result", channel="sdk", source=TurnSource.SDK,
    )
    assert isinstance(result, TurnResult)
    assert result.content == result.final_content == "done"
    assert result.run_id
    assert result.tools_used == []
    assert result.outbound is not None and result.outbound.content == "done"
    snapshot = bot.sessions.export("sdk:result")
    assert snapshot is not None
    assert [message["role"] for message in snapshot.messages] == ["user", "assistant"]


async def test_real_public_result_keeps_reserved_channel_refusal(bot):
    with pytest.raises(ValueError, match="reserved"):
        await bot._loop.process_direct_result("input", channel="system")
    assert bot._loop.provider.calls == []


async def test_sdk_run_uses_public_result_boundary_without_private_bypass(bot, monkeypatch):
    call = AsyncMock(return_value=TurnResult(content="public", final_content="public"))
    private = AsyncMock(side_effect=AssertionError("private bypass"))
    monkeypatch.setattr(bot._loop, "process_direct_result", call)
    monkeypatch.setattr(bot._loop, "_execute_direct", private)
    result = await bot.run("input")
    assert result.content == "public"
    call.assert_awaited_once_with(
        "input", session_key="sdk:default", channel="sdk", chat_id="sdk:default",
        hooks=[], source=TurnSource.SDK, trace_mode=TraceMode.NONE,
    )
    private.assert_not_awaited()


async def test_sdk_stream_uses_public_result_boundary_without_private_bypass(bot, monkeypatch):
    call = AsyncMock(return_value=TurnResult(content="public", final_content="public"))
    private = AsyncMock(side_effect=AssertionError("private bypass"))
    monkeypatch.setattr(bot._loop, "process_direct_result", call)
    monkeypatch.setattr(bot._loop, "_execute_direct", private)
    run = await bot.run_streamed("input", include_messages=True)
    events = [event async for event in run.stream_events()]
    result = await run.wait()
    assert result.content == "public"
    assert [event.type for event in events] == ["run.started", "run.completed"]
    call.assert_awaited_once()
    kwargs = call.await_args.kwargs
    assert kwargs["source"] is TurnSource.SDK and kwargs["trace_mode"] is TraceMode.SANITIZED
    assert kwargs["session_key"] == kwargs["chat_id"] == "sdk:default"
    assert len(kwargs["hooks"]) == 1
    assert callable(kwargs["on_stream"]) and callable(kwargs["on_stream_end"])
    assert callable(kwargs["on_runtime_admitted"])
    private.assert_not_awaited()


def test_kwargs_builder_keeps_omissions_and_detaches_attributes():
    attributes = {"tenant": "original"}
    options = build_process_direct_kwargs(
        session_key="sdk:options", channel="cli", chat_id="direct", sender_id="user",
        media=None, ephemeral=False, attributes=attributes,
    )
    assert options == {"session_key": "sdk:options", "attributes": {"tenant": "original"}}
    assert options["attributes"] is not attributes
    attributes["tenant"] = "changed"
    assert options["attributes"] == {"tenant": "original"}
