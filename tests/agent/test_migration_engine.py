"""Offline integration proofs using the real stage pipeline and runner."""
from collections import deque
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TraceMode, TurnRequest
from nanobot.bus.events import InboundMessage
from nanobot.config.schema import Config
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest


class ScriptedProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__(provider_name='synthetic')
        self.responses = deque(responses)
        self.calls = []
        self.generation = GenerationSettings(max_tokens=256, temperature=0.1)

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append((messages, tools, model))
        assert self.responses, 'unexpected extra provider request'
        return self.responses.popleft()

    def get_default_model(self):
        return 'synthetic-model'


@pytest.mark.parametrize("adapter", ["classic", "sdk", "bus"])
async def test_representative_adapters_delegate_exactly_one_execute_turn(
    engine_config, adapter,
):
    provider = ScriptedProvider([LLMResponse(content=f"{adapter} result")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    original = loop.execute_turn
    loop.execute_turn = AsyncMock(wraps=original)  # type: ignore[method-assign]
    try:
        if adapter == "classic":
            response = await loop.process_direct("one", session_key="cli:one")
            assert response is not None and response.content == "classic result"
        elif adapter == "sdk":
            from nanobot.nanobot import Nanobot

            result = await Nanobot(loop).run("one", session_key="sdk:one")
            assert result.content == "sdk result"
        else:
            await loop._dispatch(InboundMessage(
                channel="telegram",
                sender_id="user",
                chat_id="one",
                content="one",
            ))
        loop.execute_turn.assert_awaited_once()  # type: ignore[attr-defined]
    finally:
        await loop.aclose()


@pytest.fixture
def engine_config(tmp_path, monkeypatch):
    # This fixture alone opens guarded construction with a deterministic provider.
    # Production guard code and all activation gates remain unchanged.
    monkeypatch.setattr('nanobot.migration_guard.require_runtime_ready', lambda: None)
    monkeypatch.setattr('subprocess.Popen', Mock(side_effect=AssertionError('process execution')))
    for target in ('socket.socket.connect', 'socket.socket.connect_ex', 'socket.create_connection', 'socket.getaddrinfo'):
        monkeypatch.setattr(target, Mock(side_effect=AssertionError('outbound network access')))
    monkeypatch.setattr('asyncio.create_subprocess_exec', AsyncMock(side_effect=AssertionError('process execution')))
    monkeypatch.setattr('asyncio.create_subprocess_shell', AsyncMock(side_effect=AssertionError('shell execution')))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    config = Config()
    config.bind_source_path(tmp_path / 'instance' / 'config.json')
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = 'synthetic-model'
    config.agents.defaults.model_preset = None
    config.agents.defaults.max_tool_iterations = 4
    config.tools.agent_browser.enabled = False
    config.tools.agent_device.enabled = False
    config.tools.desktop_use.enabled = False
    config.tools.restrict_to_workspace = True
    return config


async def test_real_factory_stages_runner_and_contextual_read(engine_config):
    config = engine_config
    (config.workspace_path / 'sample.txt').write_text('fixture contents')
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='read-1', name='read_file', arguments={'path': 'sample.txt'})]),
        LLMResponse(content='done'),
    ])
    loop = AgentLoop.from_config(config, tool_registry=ToolRegistry(), provider=provider)
    try:
        result = await loop.execute_turn(TurnRequest(content='read the fixture', session_key='cli:test', trace_mode=TraceMode.SANITIZED))
        assert result.content == 'done' and result.stop_reason == 'completed'
        assert result.tools_used == ['read_file']
        assert len(provider.calls) == 2
        assert any(message.get('role') == 'tool' and 'fixture contents' in str(message.get('content'))
                   for message in provider.calls[1][0])
        details = loop.run_store.load_trace(result.run_id)
        assert [message['role'] for message in details] == ['user', 'assistant', 'tool', 'assistant']
        assert result.messages[0] == {'role': 'user', 'content': 'read the fixture'}
        assert loop.context.context_manager is loop._fork_dependencies.resource_policy.context_manager
    finally:
        await loop.aclose()


async def test_real_main_runner_delegates_to_real_planner_runner(engine_config):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='plan-1', name='plan_task', arguments={'objective': 'Plan fixture work'})]),
        LLMResponse(content='A bounded plan'),
        LLMResponse(content='planned'),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        result = await loop.execute_turn(TurnRequest(content='plan fixture work', session_key='cli:test'))
        assert result.content == 'planned' and result.tools_used == ['plan_task']
        assert len(provider.calls) == 3
        assert 'Planning Role' in provider.calls[1][0][0]['content']
        assert 'A bounded plan' in str(provider.calls[2][0])
        planner_tools = {tool['function']['name'] for tool in provider.calls[1][1]}
        assert 'write_file' not in planner_tools and 'exec' not in planner_tools
    finally:
        await loop.aclose()


async def test_real_mixed_risky_batch_executes_no_tools(engine_config):
    provider = ScriptedProvider([LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id='send', name='message', arguments={'content': 'must not send', 'channel': 'telegram', 'chat_id': '42'}),
        ToolCallRequest(id='danger', name='exec', arguments={'command': 'rm -rf .'}),
    ])])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        result = await loop.execute_turn(TurnRequest(content='fixture batch', session_key='cli:test'))
        assert result.stop_reason == 'approval_required'
        assert len(provider.calls) == 1 and result.sent_messages == ()
        assert result.tools_used == []
        assert all(message.content != 'must not send' for message in list(loop.bus.outbound._queue))
    finally:
        await loop.aclose()


async def test_real_message_submission_uses_shared_canonical_ledger_once(engine_config):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='send', name='message', arguments={
            'content': 'proactive fixture', 'channel': 'telegram', 'chat_id': '42'})]),
        LLMResponse(content='done'),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        result = await loop.execute_turn(TurnRequest(content='send fixture', session_key='cli:test'))
        assert result.content == 'done'
        assert len(result.sent_messages) == 1
        assert result.sent_messages[0].content == 'proactive fixture'
        outbound = []
        while loop.bus.outbound_size:
            outbound.append(await loop.bus.consume_outbound())
        assert len([message for message in outbound if message.content == 'proactive fixture']) == 1
    finally:
        await loop.aclose()


async def test_concurrent_real_delegation_keeps_admitted_runtime(engine_config):
    import asyncio
    from dataclasses import replace

    from nanobot.utils.llm_runtime import LLMRuntime

    class MutatingProvider(ScriptedProvider):
        def __init__(self, name, limit):
            super().__init__([
                LLMResponse(content=None, tool_calls=[ToolCallRequest(id=name, name='plan_task', arguments={'objective': 'Plan fixture work'})]),
                LLMResponse(content=f'{name} plan'), LLMResponse(content=f'{name} final'),
            ])
            self.generation = replace(self.generation, max_tokens=limit)
            self.kwargs = []
        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.kwargs.append(kwargs)
            self.generation = replace(self.generation, max_tokens=9999)
            await asyncio.sleep(0)
            return await super().chat(messages, tools, model, **kwargs)

    one, two = MutatingProvider('one', 300), MutatingProvider('two', 400)
    runtimes = [LLMRuntime.capture(provider, name, context_window_tokens=120000, model_preset=name, snapshot_signature=(name,))
                for provider, name in ((one, 'one'), (two, 'two'))]
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([]))
    try:
        results = await asyncio.gather(*[
            loop._execute_direct('plan fixture', session_key=f'cli:{runtime.model}', runtime=runtime)
            for runtime in runtimes
        ])
        assert [result.content for result in results] == ['one final', 'two final']
        for provider, name, limit in ((one, 'one', 300), (two, 'two', 400)):
            assert len(provider.calls) == 3
            assert all(call[2] == name for call in provider.calls)
            assert all(kwargs['max_tokens'] == limit for kwargs in provider.kwargs)
    finally:
        await loop.aclose()


async def test_real_nested_read_only_block_stops_main_without_approval(engine_config):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='plan', name='plan_task', arguments={'objective': 'Inspect fixture'})]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='write', name='write_file', arguments={'path': 'forbidden.txt', 'content': 'must not write'})]),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        result = await loop.execute_turn(TurnRequest(content='inspect fixture', session_key='cli:test'))
        assert result.stop_reason == 'policy_blocked'
        assert len(provider.calls) == 2
        assert not (engine_config.workspace_path / 'forbidden.txt').exists()
        assert (await loop.approval_admission.snapshot('cli:test', 'yes')).pending is None
    finally:
        await loop.aclose()


async def test_real_worker_writes_only_its_explicit_scope(engine_config):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='delegate', name='delegate_task', arguments={
            'contract': 'Write the fixture file with exactly the requested content.', 'write_scope': ['allowed.txt']})]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='write', name='write_file', arguments={'path': 'allowed.txt', 'content': 'worker output'})]),
        LLMResponse(content='worker complete'), LLMResponse(content='main complete'),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        result = await loop.execute_turn(TurnRequest(content='write a fixture file', session_key='cli:test'))
        assert result.content == 'main complete'
        assert (engine_config.workspace_path / 'allowed.txt').read_text() == 'worker output'
        assert len(provider.calls) == 4
    finally:
        await loop.aclose()


def test_engine_fixture_blocks_network_and_process_launch(engine_config):
    import socket
    import subprocess
    with pytest.raises(AssertionError, match='outbound network'):
        socket.create_connection(('example.com', 443))
    with socket.socket() as sock:
        with pytest.raises(AssertionError, match='outbound network'):
            sock.connect(('127.0.0.1', 1))
    with pytest.raises(AssertionError, match='process execution'):
        subprocess.Popen(['should-not-launch'])
