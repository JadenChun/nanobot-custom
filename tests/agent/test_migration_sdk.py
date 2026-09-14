import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from agent import test_migration_engine as engine_fixtures
from agent.test_migration_engine import ScriptedProvider
from nanobot.agent.hook import AgentHook, AgentRunHookContext
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.base import Tool
from nanobot.agent.turn import RunStatus, TraceMode, TurnResult, TurnSource
from nanobot.bus.events import OutboundMessage
from nanobot.nanobot import Nanobot
from nanobot.providers.base import LLMResponse, LLMUsage, ToolCallRequest
from nanobot.sdk.types import result_from_turn

from .test_migration_canonical_admission import host

engine_config = engine_fixtures.engine_config


def bot_host(tmp_path):
    loop = host(tmp_path)
    loop.runtime_resolver = SimpleNamespace(resolve_override=Mock(return_value=None))
    loop.runtime_event_publisher = SimpleNamespace(run_status_changed=AsyncMock(), clear_turn=Mock())
    bot = object.__new__(Nanobot)
    bot._loop, bot._config, bot._mcp_provider = loop, None, None
    return bot, loop


async def test_sdk_uses_canonical_outcome_even_when_outbound_suppressed(tmp_path):
    bot, loop = bot_host(tmp_path)
    usage = LLMUsage.reported(input_tokens=12, output_tokens=3, cache_read_tokens=2)
    loop._process_turn.return_value.run_result = AgentRunResult(
        final_content='answer', messages=[], usage=usage, tools_used=['read_file'], stop_reason='policy_blocked')
    result = await bot.run('task')
    request = loop._process_turn.call_args.kwargs['run_context'].request
    assert request.source is TurnSource.SDK
    assert request.route.channel == 'sdk' and request.route.chat_id == 'sdk:default'
    assert request.hooks == () and request.trace_mode is TraceMode.NONE
    assert result.run_id and result.status is RunStatus.POLICY_BLOCKED
    assert result.content == 'answer' and result.tools_used == ['read_file']
    assert result.messages == []
    assert result.usage == usage and result.usage is not usage
    loop.runtime_event_publisher.run_status_changed.assert_awaited_once()
    loop.runtime_event_publisher.clear_turn.assert_called_once()


async def test_sdk_options_survive_shared_direct_bridge(tmp_path):
    bot, loop = bot_host(tmp_path)
    runtime, hook = object(), object()
    loop.runtime_resolver.resolve_override.return_value = runtime
    bot._mcp_provider = SimpleNamespace(connect=AsyncMock())
    result = await bot.run('task', session_key='sdk:other', channel='cli', chat_id='direct',
        sender_id='caller', media=['image.png'], ephemeral=True, attributes={'tag': ['value']},
        hooks=[hook], include_messages=True, model='override')
    call = loop._process_turn.call_args
    request = call.kwargs['run_context'].request
    assert request.source is TurnSource.SDK and request.session_key == 'sdk:other'
    assert request.route.channel == 'cli' and request.route.chat_id == 'direct'
    assert request.media == ('image.png',) and request.sender_id == 'caller'
    assert request.hooks == (hook,) and request.trace_mode is TraceMode.SANITIZED
    assert call.kwargs['runtime'] is runtime and call.kwargs['ephemeral']
    assert call.kwargs['run_extra_hooks_for_ephemeral']
    assert call.kwargs['attributes'] == {'tag': ['value']}
    assert result.messages == []
    bot._mcp_provider.connect.assert_awaited_once()


def test_sdk_converter_keeps_trace_opt_in_detached_and_sanitized():
    raw = [{'role': 'system', 'content': 'private'}, {'role': 'assistant', 'content': 'answer',
        '_run_id': 'internal', 'tool_calls': [{'id': 'call', 'function': {'name': 'read_file', 'arguments': 'private'}}]}]
    outbound = OutboundMessage(channel='sdk', chat_id='test', content='answer', metadata={'nested': ['one']})
    response = TurnResult(run_id='run', content='answer', messages=raw, outbound=outbound)
    assert result_from_turn(response).messages == []
    result = result_from_turn(response, include_messages=True)
    assert result.messages == [{'role': 'assistant', 'content': 'answer', 'tool_calls': [{'id': 'call', 'name': 'read_file'}]}]
    result.messages[0]['tool_calls'][0]['name'] = 'changed'
    result.metadata['nested'].append('two')
    assert raw[1]['tool_calls'][0]['function']['name'] == 'read_file'
    assert outbound.metadata == {'nested': ['one']}


async def test_concurrent_sdk_runs_isolate_hooks_usage_and_trace(tmp_path):
    bot, loop = bot_host(tmp_path)
    async def execute(request, **kwargs):
        await asyncio.sleep(0)
        return TurnResult(run_id=request.content, content=request.content,
            tools_used=[request.content], messages=[{'role': 'user', 'content': request.content}],
            llm_usage=LLMUsage.reported(input_tokens=len(request.content), output_tokens=1))
    loop.execute_turn = AsyncMock(side_effect=execute)
    a, b = object(), object()
    results = await asyncio.gather(bot.run('a', hooks=[a], include_messages=True), bot.run('bb', hooks=[b]))
    assert [result.run_id for result in results] == ['a', 'bb']
    assert results[0].messages == [{'role': 'user', 'content': 'a'}]
    assert results[1].messages == []
    assert [result.usage.input_tokens for result in results] == [1, 2]
    assert [call.args[0].hooks for call in loop.execute_turn.await_args_list] == [(a,), (b,)]


@pytest.mark.parametrize('status,error,event', [
    (RunStatus.COMPLETED, None, 'run.completed'),
    (RunStatus.APPROVAL_REQUIRED, None, 'run.completed'),
    (RunStatus.ERROR, 'provider failed', 'run.failed'),
    (RunStatus.TOOL_ERROR, None, 'run.failed'),
    (RunStatus.CANCELLED, None, 'run.failed'),
])
async def test_stream_final_result_uses_canonical_conversion_once(tmp_path, status, error, event):
    bot, loop = bot_host(tmp_path)
    async def execute(request, **kwargs):
        assert request.source is TurnSource.SDK
        await request.callbacks.on_stream('answer')
        return TurnResult(run_id='stream', status=status, content='answer', error=error,
                          messages=[{'role': 'system', 'content': 'private'}, {'role': 'assistant', 'content': 'answer'}])
    loop.execute_turn = AsyncMock(side_effect=execute)
    run = await bot.run_streamed('task', include_messages=True)
    events = [item async for item in run.stream_events()]
    result = await run.wait()
    assert result.run_id == 'stream' and result.status is status
    assert result.messages == [{'role': 'assistant', 'content': 'answer'}]
    terminal = [item for item in events if item.type in ('run.completed', 'run.failed')]
    assert len(terminal) == 1 and terminal[0].type == event
    assert terminal[0].result is result
    loop.execute_turn.assert_awaited_once()


async def test_sdk_cancelled_lock_waiter_does_not_publish_idle(tmp_path):
    bot, loop = bot_host(tmp_path)
    lock = loop._get_session_lock('sdk:default')
    await lock.acquire()
    run = await bot.run_streamed('task')
    await asyncio.sleep(0)
    await run.cancel()
    loop.runtime_event_publisher.run_status_changed.assert_not_awaited()
    loop._process_turn.assert_not_awaited()
    lock.release()


def test_top_level_sdk_aliases_still_match():
    from nanobot import Nanobot as PublicNanobot
    from nanobot import RunResult as PublicResult
    from nanobot.nanobot import RunResult
    assert PublicNanobot is Nanobot and PublicResult is RunResult


@pytest.mark.parametrize('streamed', [False, True])
async def test_model_selector_conflict_rejected_before_admission(tmp_path, streamed):
    from nanobot.agent.model_runtime import ModelRuntimeResolver
    bot, loop = bot_host(tmp_path)
    loop.runtime_resolver = object.__new__(ModelRuntimeResolver)
    call = bot.run_streamed if streamed else bot.run
    with pytest.raises(ValueError, match='mutually exclusive'):
        await call('task', model='one', model_preset='two')
    loop._process_turn.assert_not_awaited()


async def test_stream_exception_emits_one_failure_without_replay(tmp_path):
    bot, loop = bot_host(tmp_path)
    loop._process_turn.side_effect = RuntimeError('stage failed')
    run = await bot.run_streamed('task')
    events = [event async for event in run.stream_events()]
    with pytest.raises(RuntimeError, match='stage failed'):
        await run.wait()
    assert [event.type for event in events if event.type.startswith('run.')] == ['run.started', 'run.failed']
    loop._process_turn.assert_awaited_once()


async def test_stream_iterator_forwards_trace_option_and_explicit_route(tmp_path):
    bot, loop = bot_host(tmp_path)
    loop.execute_turn = AsyncMock(return_value=TurnResult(content='answer', messages=[{'role': 'assistant', 'content': 'answer'}]))
    events = [event async for event in bot.stream('task', channel='telegram', chat_id='42', include_messages=True)]
    request = loop.execute_turn.call_args.args[0]
    assert request.route.channel == 'telegram' and request.route.chat_id == '42'
    assert request.trace_mode is TraceMode.SANITIZED
    assert events[-1].result.messages == [{'role': 'assistant', 'content': 'answer'}]


class _SdkTraceTool(Tool):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "sdk_trace_tool"

    @property
    def description(self) -> str:
        return "Return a safe fixture result while accepting an untrusted token."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"token": {"type": "string"}},
            "required": ["token"],
        }

    async def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return "safe tool output"


class _SdkSecretHook(AgentHook):
    def __init__(self, secret: str) -> None:
        super().__init__()
        self.secret = secret

    async def before_run(self, context: AgentRunHookContext) -> None:
        context.messages.append({"role": "system", "content": self.secret})


async def test_from_config_real_runner_keeps_default_trace_private_and_sanitizes_streamed_opt_in(
    engine_config,
    monkeypatch,
) -> None:
    """Exercise the public constructor, factory, runner, and trace projection together."""
    private_token = "sdk-private-tool-argument"
    hook_secret = "sdk-private-hook-secret"
    provider = ScriptedProvider([
        LLMResponse(content="default answer"),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="sdk-trace-call",
            name="sdk_trace_tool",
            arguments={"token": private_token},
        )]),
        LLMResponse(content="streamed answer"),
    ])
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path: engine_config)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)

    bot = Nanobot.from_config(workspace=engine_config.workspace_path)
    tool = _SdkTraceTool()
    bot._loop.tools.register(tool)
    try:
        default_result = await bot.run("default trace remains private")
        streamed = await bot.run_streamed(
            "compose a safe trace",
            include_messages=True,
            hooks=[_SdkSecretHook(hook_secret)],
        )
        trace_result = await streamed.wait()

        assert default_result.content == "default answer"
        assert default_result.messages == []
        assert trace_result.content == "streamed answer"
        assert tool.calls == 1
        assert trace_result.messages == [
            {"role": "user", "content": "compose a safe trace"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "sdk-trace-call", "name": "sdk_trace_tool"},
            ]},
            {"role": "tool", "content": "safe tool output", "tool_call_id": "sdk-trace-call", "name": "sdk_trace_tool"},
            {"role": "assistant", "content": "streamed answer"},
        ]
        rendered = repr(trace_result.messages)
        assert private_token not in rendered
        assert hook_secret not in rendered
        assert "arguments" not in rendered
        assert "system" not in rendered
        assert "SDKCaptureHook" not in rendered
    finally:
        await bot.aclose()
